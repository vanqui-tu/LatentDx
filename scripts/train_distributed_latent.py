#!/usr/bin/env python
"""Train the fixed-route distributed latent KV baseline (M3/L3.3)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medlatent.distributed import train_distributed_latent_real


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--train_file", required=True, type=Path)
    parser.add_argument("--hospital_dir", required=True, type=Path)
    parser.add_argument("--hpo_embeddings_file", required=True, type=Path)
    parser.add_argument("--hpo_ic_file", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--num_latents", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=320)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()
    summary = train_distributed_latent_real(
        model_name=args.model_name, train_file=args.train_file, hospital_dir=args.hospital_dir,
        output_dir=args.output_dir, hpo_embeddings_file=args.hpo_embeddings_file,
        hpo_ic_file=args.hpo_ic_file, num_latents=args.num_latents,
        max_prompt_length=args.max_prompt_length, max_target_length=args.max_target_length,
        epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, seed=args.seed, device=args.device, dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    print(f"distributed latent training complete: output_dir={args.output_dir} "
          f"updates={int(summary['updates'])} loss={summary['last_loss']:.4f}")


if __name__ == "__main__":
    main()
