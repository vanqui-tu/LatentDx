#!/usr/bin/env python
"""Evaluate MedLatent diagnosis utility."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import add_common_args, dry_run_or_raise, read_config_text
from medlatent.hf_eval import evaluate_medlatent_real
from medlatent.toy_runtime import evaluate_toy_diagnosis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--method", choices=["medlatent_h", "medlatent_x"], default="medlatent_h")
    parser.add_argument("--toy", action="store_true", help="Evaluate a toy MedLatent-H checkpoint.")
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model_name", default="", help="Frozen HuggingFace model path or id.")
    parser.add_argument("--data_file", default="", help="Diagnosis JSON file.")
    parser.add_argument("--hospital_dir", default="", help="Directory containing hospital_*.json shards.")
    parser.add_argument("--cache_dir", default="", help="Required for medlatent_x.")
    parser.add_argument("--num_hospitals", type=int, default=3)
    parser.add_argument("--num_latents", type=int, default=32)
    parser.add_argument("--max_prompt_length", type=int, default=320)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--hpo_embeddings_file", default="", help="HPO embedding JSON required for IC-weighted cosine retrieval.")
    parser.add_argument("--hpo_ic_file", default="", help="HPO information-content JSON required for IC-weighted cosine retrieval.")
    args = parser.parse_args()
    if args.toy:
        if not args.checkpoint_dir:
            raise SystemExit("--checkpoint_dir is required with --toy")
        metrics = evaluate_toy_diagnosis(args.checkpoint_dir, output=args.output or None, seed=args.seed)
        print(f"toy diagnosis evaluation complete: accuracy={metrics['accuracy']:.4f}")
        return
    real_args = [args.model_name, args.checkpoint_dir, args.data_file, args.hospital_dir, args.output]
    if any(real_args):
        missing = [name for name, value in {
            "model_name": args.model_name,
            "checkpoint_dir": args.checkpoint_dir,
            "data_file": args.data_file,
            "hospital_dir": args.hospital_dir,
            "output": args.output,
        }.items() if not value]
        if missing:
            parser.error(f"real evaluation requires: {', '.join(missing)}")
        metrics = evaluate_medlatent_real(
            method=args.method,
            model_name=args.model_name,
            checkpoint_dir=args.checkpoint_dir,
            data_file=args.data_file,
            hospital_dir=args.hospital_dir,
            output=args.output,
            cache_dir=args.cache_dir or None,
            num_hospitals=args.num_hospitals,
            num_latents=args.num_latents,
            max_prompt_length=args.max_prompt_length,
            max_target_length=args.max_target_length,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
            device=args.device,
            dtype_name=args.dtype,
            local_files_only=args.local_files_only,
            hpo_embeddings_file=args.hpo_embeddings_file or None,
            hpo_ic_file=args.hpo_ic_file or None,
        )
        print(
            "real diagnosis evaluation complete: "
            f"accuracy={metrics['accuracy']:.4f} token_f1={metrics['token_f1']:.4f} samples={int(metrics['num_samples'])}"
        )
        return
    dry_run_or_raise(name="evaluate_diagnosis", config_text=read_config_text(args.config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
