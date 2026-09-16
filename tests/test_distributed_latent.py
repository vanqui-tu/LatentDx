import torch
import pytest

from medlatent.distributed.latent import (
    DistributedLatentProtocol, accumulation_steps_for, batched_pilot_loss, load_distributed_latent_checkpoint, merge_kv_blocks,
    batched_relay_aggregate, batched_rollout, ring_two_hop_branches, save_distributed_latent_checkpoint,
    slice_kv_block, tiny_overfit, two_hop_ring_blocks,
)
from medlatent.modules import BoundaryEmbeddings, LatentDistiller


class _Output:
    def __init__(self, past_key_values, last_hidden_state, logits=None):
        self.past_key_values = past_key_values
        self.last_hidden_state = last_hidden_state
        self.logits = logits


class _Decoder(torch.nn.Module):
    def __init__(self, hidden=4, vocab=9):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, hidden)
        self.update = torch.nn.Linear(hidden, hidden, bias=False)
        self.output = torch.nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, **kwargs):
        values = inputs_embeds if inputs_embeds is not None else self.embedding(input_ids)
        hidden = self.update(values)
        if past_key_values is not None:
            hidden = hidden + past_key_values[0][0].mean(dim=(1, 2)).unsqueeze(1)
        key = hidden.unsqueeze(1)
        value = key + 0.1
        if past_key_values is not None:
            key = torch.cat([past_key_values[0][0], key], dim=2)
            value = torch.cat([past_key_values[0][1], value], dim=2)
        return _Output(((key, value),), hidden)


class _Model(_Decoder):
    base_model = property(lambda self: self)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        return _Output(output.past_key_values, output.last_hidden_state, self.output(output.last_hidden_state))


def test_two_hop_rollout_reencodes_and_keeps_gradients():
    model = _Model()
    protocol = DistributedLatentProtocol(model, LatentDistiller(4), BoundaryEmbeddings(4), num_latents=2)
    ids = torch.tensor([[1, 2]])
    mask = torch.ones_like(ids)

    first = protocol.rollout(ids, mask)
    second = protocol.rollout(ids, mask, [first])
    assert first[0][0].shape[2] == 4  # BEGIN + m positions + END
    assert second[0][0].shape[2] == 4
    assert second[0][0].data_ptr() != first[0][0].data_ptr()
    assert merge_kv_blocks([first, first])[0][0].shape[2] == 8

    protocol.source_logits(ids, mask, [second]).sum().backward()
    assert protocol.distiller.projection[1].weight.grad.abs().sum() > 0
    assert protocol.distiller.latent_begin.grad.abs().sum() > 0
    assert protocol.boundary.begin.grad.abs().sum() > 0
    assert protocol.boundary.end.grad.abs().sum() > 0


def test_source_teacher_forcing_loss_uses_batched_branches():
    model = _Model()
    protocol = DistributedLatentProtocol(model, LatentDistiller(4), BoundaryEmbeddings(4), num_latents=2)
    ids = torch.tensor([[1, 2], [3, 4]])
    mask = torch.ones_like(ids)
    branches = two_hop_ring_blocks(protocol, leaf_ids=ids, leaf_mask=mask, relay_ids=ids, relay_mask=mask)
    source_ids = torch.tensor([[1, 2]])
    source_mask = torch.ones_like(source_ids)
    target_ids = torch.tensor([[1, 2, 3]])
    labels = target_ids.clone()
    loss = protocol.source_loss(source_ids, source_mask, target_ids, labels, branches)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_sliced_block_copies_storage_without_detaching():
    key = torch.randn(1, 1, 12, 4, requires_grad=True)
    value = torch.randn(1, 1, 12, 4, requires_grad=True)

    block = slice_kv_block(((key, value),), 4)

    assert block[0][0].untyped_storage().data_ptr() != key.untyped_storage().data_ptr()
    block[0][0].sum().backward()
    assert key.grad[:, :, -4:, :].abs().sum() > 0


def test_batched_local_and_relay_operator_detaches_children():
    model = _Model()
    protocol = DistributedLatentProtocol(model, LatentDistiller(4), BoundaryEmbeddings(4), num_latents=2)
    ids = torch.tensor([[1, 2], [3, 4]])
    mask = torch.ones_like(ids)
    local = batched_rollout(protocol, ids, mask, detach_children=False)
    relay = batched_relay_aggregate(protocol, ids, mask, [local])
    assert local[0][0].shape[:3] == relay[0][0].shape[:3] == (2, 1, 4)
    assert torch.autograd.grad(relay[0][0].sum(), local[0][0], allow_unused=True) == (None,)


def _pilot_tensors():
    source_ids = torch.tensor([[1, 2]])
    leaf_ids = torch.tensor([[1, 2], [3, 4]])
    return dict(
        source_ids=source_ids, source_mask=torch.ones_like(source_ids),
        target_ids=torch.tensor([[3, 4]]), target_labels=torch.tensor([[3, 4]]),
        leaf_ids=leaf_ids, leaf_mask=torch.ones_like(leaf_ids),
        relay_ids=leaf_ids, relay_mask=torch.ones_like(leaf_ids),
    )


def test_effective_batch_size_matches_medlatent_h_accumulation_semantics():
    assert accumulation_steps_for(batch_size=1, effective_batch_size=8) == 8
    assert accumulation_steps_for(batch_size=2, effective_batch_size=8) == 4
    assert accumulation_steps_for(batch_size=4, effective_batch_size=8) == 2
    with pytest.raises(ValueError, match="divisible"):
        accumulation_steps_for(batch_size=3, effective_batch_size=8)


def test_pilot_loss_only_updates_active_interface_parameters():
    protocol = DistributedLatentProtocol(_Model(), LatentDistiller(4), BoundaryEmbeddings(4), num_latents=2)
    first = batched_pilot_loss(protocol, use_relay=True, **_pilot_tensors())
    second = batched_pilot_loss(protocol, use_relay=True, **_pilot_tensors())
    assert torch.allclose(first, second)
    first.backward()

    assert protocol.distiller.projection[1].weight.grad.abs().sum() > 0
    assert protocol.boundary.begin.grad.abs().sum() > 0
    assert protocol.boundary.end.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in protocol.model.parameters())


def test_tiny_overfit_reduces_local_and_relay_losses_deterministically():
    torch.manual_seed(0)
    protocol = DistributedLatentProtocol(_Model(), LatentDistiller(4), BoundaryEmbeddings(4), num_latents=2)
    optimizer = torch.optim.AdamW([*protocol.distiller.parameters(), *protocol.boundary.parameters()], lr=0.05)
    result = tiny_overfit(protocol, optimizer, steps=20, **_pilot_tensors())

    assert result["local_final_loss"] < result["local_initial_loss"]
    assert result["relay_final_loss"] < result["relay_initial_loss"]


def test_fixed_route_and_checkpoint_metadata(tmp_path):
    assert ring_two_hop_branches(0, 10) == ((2, 1), (8, 9))
    model = _Model()
    protocol = DistributedLatentProtocol(model, LatentDistiller(4), BoundaryEmbeddings(4), num_latents=2)
    optimizer = torch.optim.AdamW(list(protocol.distiller.parameters()) + list(protocol.boundary.parameters()))
    path = tmp_path / "distributed_latent.pt"
    save_distributed_latent_checkpoint(
        path, protocol=protocol, model_name="fake/model",
        route={"num_agents": 5, "R": 4, "k": 2, "m": 2, "pilot_hospital_ids": [0, 1, 2, 3, 4],
               "graph_edges": [[0, 1]], "split_paths": {"train": "train.json"}},
        optimizer=optimizer, training={"updates": 1, "tiny_overfit": {"local_final_loss": 1.0}},
    )
    payload = load_distributed_latent_checkpoint(path)
    assert payload["module_type"] == "DistributedLatentKV"
    assert payload["num_latents"] == 2
    assert payload["route"]["num_agents"] == 5
    assert payload["route"]["pilot_hospital_ids"] == [0, 1, 2, 3, 4]
    assert payload["training"]["tiny_overfit"]["local_final_loss"] == 1.0
