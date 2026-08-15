#!/usr/bin/env python
"""Train MedLatent-H same-backbone latent distillation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_common_args, dry_run_or_raise, read_config_text
from medlatent.hf_medlatent_h import train_medlatent_h_real
from medlatent.toy_runtime import train_toy_medlatent_h


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--toy", action="store_true", help="Run a small CPU toy training job for smoke testing.")
    parser.add_argument("--output_dir", default="artifacts/toy_medlatent_h")
    parser.add_argument("--steps", type=int, default=0, help="Optional optimizer-update cap; 0 runs the configured epochs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", default="", help="HF model name/path for real MedLatent-H training.")
    parser.add_argument("--train_file", default="", help="Training JSON file.")
    parser.add_argument("--hospital_dir", default="", help="Directory containing hospital_0.json ... hospital_N.json.")
    parser.add_argument("--num_hospitals", type=int, default=3)
    parser.add_argument("--num_latents", type=int, default=32)
    parser.add_argument("--max_prompt_length", type=int, default=320)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--batch_hospitals", action="store_true", help="Run all hospitals together at each latent step; uses more VRAM.")
    parser.add_argument("--effective_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--val_file", default="", help="Validation JSON file for checkpoint selection.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--hpo_embeddings_file", default="", help="Optional HPO embedding JSON for IC-weighted cosine retrieval.")
    parser.add_argument("--hpo_ic_file", default="", help="Optional HPO information-content JSON for IC-weighted cosine retrieval.")
    args = parser.parse_args()
    if args.toy:
        metrics = train_toy_medlatent_h(args.output_dir, steps=args.steps, seed=args.seed)
        print(f"toy training complete: output_dir={args.output_dir} final_loss={metrics['final_loss']:.4f}")
        return
    if args.model_name or args.train_file or args.hospital_dir:
        missing = [name for name in ("model_name", "train_file", "hospital_dir") if not getattr(args, name)]
        if missing:
            raise SystemExit(f"Missing required real-training args: {', '.join('--' + name for name in missing)}")
        summary = train_medlatent_h_real(
            model_name=args.model_name,
            train_file=args.train_file,
            val_file=args.val_file or None,
            hospital_dir=args.hospital_dir,
            output_dir=args.output_dir,
            num_hospitals=args.num_hospitals,
            num_latents=args.num_latents,
            max_prompt_length=args.max_prompt_length,
            max_target_length=args.max_target_length,
            batch_size=args.batch_size,
            batch_hospitals=args.batch_hospitals,
            effective_batch_size=args.effective_batch_size,
            epochs=args.epochs,
            warmup_steps=args.warmup_steps,
            max_steps=args.steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
            dtype=args.dtype,
            local_files_only=args.local_files_only,
            hpo_embeddings_file=args.hpo_embeddings_file or None,
            hpo_ic_file=args.hpo_ic_file or None,
        )
        print(f"real MedLatent-H training complete: output_dir={args.output_dir} steps={int(summary['steps'])}")
        return
    dry_run_or_raise(name="train_medlatent_h", config_text=read_config_text(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
