"""Real HuggingFace runtime for MedLatent-H same-backbone training."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .hf_data import IGNORE_INDEX, MedLatentDiagnosisDataset, collate_medlatent
from .losses import diagnosis_cross_entropy
from .modules import BoundaryEmbeddings, LatentDistiller


def _build_position_ids(prefix_mask: torch.Tensor, current_len: int) -> torch.Tensor:
    starts = prefix_mask.sum(dim=1, keepdim=True).long()
    offsets = torch.arange(current_len, device=prefix_mask.device, dtype=torch.long).unsqueeze(0)
    return starts + offsets


def _append_embedding(model, embedding: torch.Tensor, prefix_mask: torch.Tensor, past_key_values):
    batch_size = prefix_mask.shape[0]
    inputs = embedding.to(device=prefix_mask.device, dtype=model.get_input_embeddings().weight.dtype)
    inputs = inputs.view(1, 1, -1).expand(batch_size, 1, -1)
    attention_mask = torch.cat([prefix_mask, prefix_mask.new_ones((batch_size, 1))], dim=1)
    outputs = model(
        inputs_embeds=inputs,
        attention_mask=attention_mask,
        position_ids=_build_position_ids(prefix_mask, 1),
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )
    return outputs.past_key_values, attention_mask


def _legacy_cache(past_key_values):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


def _slice_last_positions(past_key_values, num_positions: int):
    return tuple((k[:, :, -num_positions:, :], v[:, :, -num_positions:, :]) for k, v in _legacy_cache(past_key_values))


def _to_dynamic_cache(legacy_cache):
    cache = DynamicCache()
    for layer_idx, (key, value) in enumerate(legacy_cache):
        cache.update(key, value, layer_idx)
    return cache


def _assemble_blocks(blocks_by_hospital: dict[int, tuple], hospital_order: list[int]):
    layers = len(next(iter(blocks_by_hospital.values())))
    combined = []
    for layer_idx in range(layers):
        keys = [blocks_by_hospital[h][layer_idx][0] for h in hospital_order]
        values = [blocks_by_hospital[h][layer_idx][1] for h in hospital_order]
        combined.append((torch.cat(keys, dim=2), torch.cat(values, dim=2)))
    length = combined[0][0].shape[2]
    mask = combined[0][0].new_ones((combined[0][0].shape[0], length), dtype=torch.long)
    return tuple(combined), mask


def forward_medlatent_h_batch(
    *,
    model,
    distiller: LatentDistiller,
    boundary: BoundaryEmbeddings,
    batch: dict,
    num_latents: int,
    hospital_order: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    hospital_blocks: dict[int, tuple] = {}
    block_len = int(num_latents) + 2
    for hospital_id in hospital_order:
        input_ids = batch["hospital_ids_all"][hospital_id].to(device)
        attention_mask = batch["hospital_mask_all"][hospital_id].to(device)
        with torch.no_grad():
            prompt_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        past_key_values = prompt_out.past_key_values
        lengths = attention_mask.sum(dim=1) - 1
        hidden = prompt_out.hidden_states[-1][torch.arange(input_ids.shape[0], device=device), lengths, :]
        prefix_mask = attention_mask

        past_key_values, prefix_mask = _append_embedding(model, boundary.begin, prefix_mask, past_key_values)
        for step in range(int(num_latents)):
            if step == 0:
                step_input = distiller.begin_embedding(dtype=model.get_input_embeddings().weight.dtype, device=device)
                step_input = step_input.expand(input_ids.shape[0], 1, -1)
            else:
                step_input = distiller(hidden).unsqueeze(1)
            attention = torch.cat([prefix_mask, prefix_mask.new_ones((input_ids.shape[0], 1))], dim=1)
            latent_out = model(
                inputs_embeds=step_input,
                attention_mask=attention,
                position_ids=_build_position_ids(prefix_mask, 1),
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past_key_values = latent_out.past_key_values
            hidden = latent_out.hidden_states[-1][:, -1, :]
            prefix_mask = attention
        past_key_values, prefix_mask = _append_embedding(model, boundary.end, prefix_mask, past_key_values)
        hospital_blocks[hospital_id] = _slice_last_positions(past_key_values, block_len)

    legacy_cache, latent_mask = _assemble_blocks(hospital_blocks, hospital_order)
    host_ids = batch["host_question_ids"].to(device)
    host_mask = batch["host_question_mask"].to(device)
    host_attention = torch.cat([latent_mask.to(device), host_mask], dim=1)
    host_out = model(
        input_ids=host_ids,
        attention_mask=host_attention,
        position_ids=_build_position_ids(latent_mask.to(device), host_ids.shape[1]),
        past_key_values=_to_dynamic_cache(legacy_cache),
        use_cache=True,
        return_dict=True,
    )

    target_ids = batch["target_ids"].to(device)
    target_labels = batch["target_labels"].to(device)
    target_attention = torch.cat([latent_mask.to(device), host_mask, torch.ones_like(target_ids)], dim=1)
    host_prefix = torch.cat([latent_mask.to(device), host_mask], dim=1)
    target_out = model(
        input_ids=target_ids,
        attention_mask=target_attention,
        position_ids=_build_position_ids(host_prefix, target_ids.shape[1]),
        past_key_values=host_out.past_key_values,
        use_cache=False,
        return_dict=True,
    )
    loss = diagnosis_cross_entropy(host_out.logits[:, -1, :], target_out.logits, target_labels, ignore_index=IGNORE_INDEX)
    first_pred = host_out.logits[:, -1, :].argmax(dim=-1)
    first_gold = target_labels[:, 0]
    first_mask = first_gold != IGNORE_INDEX
    first_acc = (first_pred[first_mask] == first_gold[first_mask]).float().mean().item() if first_mask.any() else 0.0
    return loss, {"loss_ce": float(loss.detach().cpu()), "first_token_acc": float(first_acc)}


def train_medlatent_h_real(
    *,
    model_name: str,
    train_file: str,
    val_file: str | None = None,
    hospital_dir: str,
    output_dir: str,
    num_hospitals: int = 3,
    num_latents: int = 32,
    max_prompt_length: int = 320,
    max_target_length: int = 64,
    batch_size: int = 1,
    effective_batch_size: int = 8,
    epochs: int = 5,
    warmup_steps: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    max_steps: int = 0,
    seed: int = 42,
    device: str = "cuda",
    dtype: str = "bfloat16",
    local_files_only: bool = False,
    hpo_embeddings_file: str | None = None,
    hpo_ic_file: str | None = None,
) -> dict[str, float]:
    torch.manual_seed(seed)
    resolved_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16 if dtype == "float16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    ).to(resolved_device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    hidden_size = int(model.config.hidden_size)
    distiller = LatentDistiller(hidden_size).to(device=resolved_device, dtype=torch_dtype)
    boundary = BoundaryEmbeddings(hidden_size).to(device=resolved_device, dtype=torch_dtype)
    train_dataset = MedLatentDiagnosisDataset(
        data_file=train_file,
        hospital_dir=hospital_dir,
        tokenizer=tokenizer,
        num_hospitals=num_hospitals,
        max_prompt_length=max_prompt_length,
        max_target_length=max_target_length,
        limit=-1,
        hpo_embeddings_file=hpo_embeddings_file,
        hpo_ic_file=hpo_ic_file,
    )
    val_dataset = MedLatentDiagnosisDataset(
        data_file=val_file or train_file,
        hospital_dir=hospital_dir,
        tokenizer=tokenizer,
        num_hospitals=num_hospitals,
        max_prompt_length=max_prompt_length,
        max_target_length=max_target_length,
        limit=-1 if val_file else min(len(train_dataset), 128),
        hpo_embeddings_file=hpo_embeddings_file,
        hpo_ic_file=hpo_ic_file,
    )
    optimizer = torch.optim.AdamW(
        list(distiller.parameters()) + list(boundary.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    micro_batch_size = max(1, int(batch_size))
    accumulation = max(1, int(effective_batch_size) // micro_batch_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_medlatent(rows, pad_token_id=tokenizer.pad_token_id),
    )
    updates_per_epoch = (len(train_loader) + accumulation - 1) // accumulation
    total_updates = updates_per_epoch * max(1, int(epochs))

    def lr_multiplier(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    hospital_order = list(range(num_hospitals))
    best_val = float("inf")
    global_update = 0
    last_stats: dict[str, float] = {}

    def validation_loss() -> float:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda rows: collate_medlatent(rows, pad_token_id=tokenizer.pad_token_id),
        )
        total = 0.0
        count = 0
        distiller.eval()
        boundary.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                loss, _ = forward_medlatent_h_batch(
                    model=model,
                    distiller=distiller,
                    boundary=boundary,
                    batch=val_batch,
                    num_latents=num_latents,
                    hospital_order=hospital_order,
                    device=resolved_device,
                )
                total += float(loss.detach().cpu())
                count += 1
        return total / max(1, count)

    train_generator = torch.Generator().manual_seed(seed)
    for epoch in range(max(1, int(epochs))):
        distiller.train()
        boundary.train()
        indices = torch.randperm(len(train_dataset), generator=train_generator).tolist()
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        update_count = 0
        for index in indices:
            batch = collate_medlatent([train_dataset[index]], pad_token_id=tokenizer.pad_token_id)
            loss, stats = forward_medlatent_h_batch(
                model=model,
                distiller=distiller,
                boundary=boundary,
                batch=batch,
                num_latents=num_latents,
                hospital_order=hospital_order,
                device=resolved_device,
            )
            loss.backward()
            update_loss += float(loss.detach().cpu())
            update_count += 1
            is_last = index == indices[-1]
            if update_count >= accumulation or is_last:
                divisor = update_count
                for parameter in list(distiller.parameters()) + list(boundary.parameters()):
                    if parameter.grad is not None:
                        parameter.grad.div_(divisor)
                torch.nn.utils.clip_grad_norm_(list(distiller.parameters()) + list(boundary.parameters()), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
                last_stats = {**stats, "train_ce": update_loss / divisor, "learning_rate": scheduler.get_last_lr()[0]}
                print(json.dumps({"event": "train_update", "epoch": epoch + 1, "update": global_update, **last_stats}))
                update_loss = 0.0
                update_count = 0
                if max_steps > 0 and global_update >= max_steps:
                    break
        current_val = validation_loss()
        print(json.dumps({"event": "validation", "epoch": epoch + 1, "val_ce": current_val}))
        if current_val < best_val:
            best_val = current_val
            output = Path(output_dir)
            output.mkdir(parents=True, exist_ok=True)
            distiller.save(output / "distiller_final.pt")
            boundary.save(output / "boundary_final.pt")
        if max_steps > 0 and global_update >= max_steps:
            break

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "distiller_final.pt").exists():
        distiller.save(output / "distiller_final.pt")
        boundary.save(output / "boundary_final.pt")
    (output / "training_summary.json").write_text(json.dumps({
        "epochs": epoch + 1,
        "updates": global_update,
        "best_val_ce": best_val,
        "effective_batch_size": effective_batch_size,
        "micro_batch_size": micro_batch_size,
        **last_stats,
    }, indent=2))
    return {"steps": float(global_update), "best_val_ce": float(best_val), **last_stats}


