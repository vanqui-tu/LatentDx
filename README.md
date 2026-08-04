# LatentDx

LatentDx is an artifact implementation of **compact latent KV communication for cross-hospital rare-disease diagnosis**.

The repository is intentionally focused on the paper's main method. It does not include text-sharing baselines, external multi-agent baselines, model checkpoints, latent caches, or experiment logs.

## What This Repository Contains

MedLatent has two method variants:

| Paper name | Code name | Purpose |
|---|---|---|
| LatentDx-H | `LatentDistiller` | Same-backbone latent KV distillation. Each hospital compresses its private retrieved prompt into a fixed-length latent KV block. |
| LatentDx-X | `LatentProjector` | Cross-family latent alignment. Foreign-family latent blocks are projected into the host-family embedding space. |

All LLM backbones are frozen. The trainable components are limited to:

- `LatentDistiller`: LayerNorm + bias-free Linear, same hidden size.
- `LatentProjector`: LayerNorm + bias-free Linear, encoder dimension to host dimension.
- `BoundaryEmbeddings`: learned begin/end markers around one hospital block.

## Repository Layout

```text
medlatent/
  modules.py          Trainable latent interface modules.
  retrieval.py        HPO IC-weighted cosine retrieval.
  prompts.py          Hospital, host, answer, and leakage prompts.
  cache.py            Family latent cache loader for MedLatent-X.
  latent_blocks.py    Compact latent block helpers.
  forward.py          Paper-facing MedLatent-H/X forward interfaces.
  losses.py           Diagnosis cross-entropy including the first answer token.
  leakage.py          Passive-observer reconstruction helpers.
  metrics.py          Lightweight exact-match and token-F1 metrics.

scripts/
  train_medlatent_h.py     Same-backbone training entry point.
  build_latent_cache.py    Cross-family cache construction entry point.
  train_medlatent_x.py     Cross-family projector training entry point.
  evaluate_diagnosis.py    Diagnosis utility evaluation entry point.
  evaluate_leakage.py      Reconstruction leakage evaluation entry point.

configs/
  models.yaml
  medlatent_h.yaml
  medlatent_x.yaml
  eval.yaml

tests/
  Unit tests and an end-to-end toy training/evaluation smoke test.
```

## Install

From this directory:

```bash
python -m pip install -e .
```

For a quick code review, installation is optional because the scripts bootstrap the local package path.

## Verified Toy Smoke Test

The repository includes a tiny CPU-only toy runtime. It does not require real LLM weights or private data. Its purpose is to verify that the trainable modules, checkpointing, and CLI flow execute end to end.

```bash
python scripts/train_medlatent_h.py \
  --toy \
  --output_dir /tmp/medlatent_toy_smoke \
  --steps 4 \
  --seed 7

python scripts/evaluate_diagnosis.py \
  --toy \
  --checkpoint_dir /tmp/medlatent_toy_smoke \
  --output /tmp/medlatent_toy_smoke/metrics.json
```

Expected behavior:

```text
toy training complete: output_dir=/tmp/medlatent_toy_smoke final_loss=...
toy diagnosis evaluation complete: accuracy=...
```

Run tests:

```bash
python -m pytest -q
```

Current smoke-test status:

```text
9 passed
```

## Real Data and Model Inputs

The repository includes the diagnosis JSON files and hospital partitions needed by the main experiments under `data/`. Frozen HuggingFace CausalLM checkpoints are not included.

```bash
export MEDLATENT_DATA_DIR="$PWD/data"
export MEDLATENT_MODEL_DIR=/path/to/hf_models
export MEDLATENT_ARTIFACT_DIR=/path/to/medlatent_outputs
```

Expected data layout:

```text
$MEDLATENT_DATA_DIR/
  train.json
  val.json
  test.json
  hospital_0.json
  hospital_1.json
  hospital_2.json
  hospital_3.json
  hospital_4.json
  hpo_embeddings.json.gz
  hpo_ic.json
```

Each diagnosis row contains HPO codes under `Phenotype`, diseases under `RareDisease`, and optionally disease/phenotype names in the hospital shards.

If `--hpo_embeddings_file` and `--hpo_ic_file` are provided, retrieval uses IC-weighted HPO cosine similarity. If they are omitted, the runtime falls back to HPO-set Jaccard retrieval; this fallback is intended for smoke tests, not paper-number reproduction.

## Real MedLatent-H Training

MedLatent-H trains a same-backbone latent distiller and boundary embeddings while keeping the backbone frozen:

```bash
python scripts/train_medlatent_h.py \
  --model_name "$MEDLATENT_MODEL_DIR/host-family-model" \
  --train_file "$MEDLATENT_DATA_DIR/train.json" \
  --hospital_dir "$MEDLATENT_DATA_DIR" \
  --output_dir "$MEDLATENT_ARTIFACT_DIR/medlatent_h" \
  --num_hospitals 3 \
  --num_latents 32 \
  --max_prompt_length 320 \
  --max_target_length 64 \
  --batch_size 1 \
  --steps 1000 \
  --device cuda \
  --dtype bfloat16 \
  --hpo_embeddings_file "$MEDLATENT_DATA_DIR/hpo_embeddings.json.gz" \
  --hpo_ic_file "$MEDLATENT_DATA_DIR/hpo_ic.json" \
  --local_files_only
```

Outputs:

```text
distiller_final.pt
boundary_final.pt
training_summary.json
```

## Real MedLatent-X Training

MedLatent-X has two real steps.

First, build a frozen encoder-family latent cache from a trained MedLatent-H distiller:

```bash
python scripts/build_latent_cache.py \
  --encoder_model_name "$MEDLATENT_MODEL_DIR/encoder-family-model" \
  --distiller_checkpoint "$MEDLATENT_ARTIFACT_DIR/encoder_medlatent_h/distiller_final.pt" \
  --train_file "$MEDLATENT_DATA_DIR/train.json" \
  --hospital_dir "$MEDLATENT_DATA_DIR" \
  --output_dir "$MEDLATENT_ARTIFACT_DIR/encoder_latent_cache" \
  --split train \
  --num_hospitals 3 \
  --num_latents 32 \
  --max_prompt_length 320 \
  --max_target_length 64 \
  --batch_size 1 \
  --device cuda \
  --dtype bfloat16 \
  --hpo_embeddings_file "$MEDLATENT_DATA_DIR/hpo_embeddings.json.gz" \
  --hpo_ic_file "$MEDLATENT_DATA_DIR/hpo_ic.json" \
  --local_files_only
```

Then train the host-side projector:

```bash
python scripts/train_medlatent_x.py \
  --host_model_name "$MEDLATENT_MODEL_DIR/host-family-model" \
  --cache_dir "$MEDLATENT_ARTIFACT_DIR/encoder_latent_cache" \
  --train_file "$MEDLATENT_DATA_DIR/train.json" \
  --hospital_dir "$MEDLATENT_DATA_DIR" \
  --output_dir "$MEDLATENT_ARTIFACT_DIR/medlatent_x" \
  --num_hospitals 3 \
  --max_prompt_length 320 \
  --max_target_length 64 \
  --batch_size 1 \
  --steps 1000 \
  --device cuda \
  --dtype bfloat16 \
  --hpo_embeddings_file "$MEDLATENT_DATA_DIR/hpo_embeddings.json.gz" \
  --hpo_ic_file "$MEDLATENT_DATA_DIR/hpo_ic.json" \
  --local_files_only
```

Outputs:

```text
projector_final.pt
boundary_final.pt
training_summary.json
```

## Real Diagnosis Inference

MedLatent-H:

```bash
python scripts/evaluate_diagnosis.py \
  --method medlatent_h \
  --model_name "$MEDLATENT_MODEL_DIR/host-family-model" \
  --checkpoint_dir "$MEDLATENT_ARTIFACT_DIR/medlatent_h" \
  --data_file "$MEDLATENT_DATA_DIR/test.json" \
  --hospital_dir "$MEDLATENT_DATA_DIR" \
  --output "$MEDLATENT_ARTIFACT_DIR/medlatent_h_predictions.jsonl" \
  --num_hospitals 3 \
  --num_latents 32 \
  --max_prompt_length 320 \
  --max_target_length 64 \
  --max_new_tokens 64 \
  --max_samples 401 \
  --device cuda \
  --dtype bfloat16 \
  --hpo_embeddings_file "$MEDLATENT_DATA_DIR/hpo_embeddings.json.gz" \
  --hpo_ic_file "$MEDLATENT_DATA_DIR/hpo_ic.json" \
  --local_files_only
```

MedLatent-X:

```bash
python scripts/evaluate_diagnosis.py \
  --method medlatent_x \
  --model_name "$MEDLATENT_MODEL_DIR/host-family-model" \
  --checkpoint_dir "$MEDLATENT_ARTIFACT_DIR/medlatent_x" \
  --cache_dir "$MEDLATENT_ARTIFACT_DIR/encoder_latent_cache" \
  --data_file "$MEDLATENT_DATA_DIR/test.json" \
  --hospital_dir "$MEDLATENT_DATA_DIR" \
  --output "$MEDLATENT_ARTIFACT_DIR/medlatent_x_predictions.jsonl" \
  --num_hospitals 3 \
  --num_latents 32 \
  --max_prompt_length 320 \
  --max_target_length 64 \
  --max_new_tokens 64 \
  --max_samples 401 \
  --device cuda \
  --dtype bfloat16 \
  --hpo_embeddings_file "$MEDLATENT_DATA_DIR/hpo_embeddings.json.gz" \
  --hpo_ic_file "$MEDLATENT_DATA_DIR/hpo_ic.json" \
  --local_files_only
```

Each evaluation writes a JSONL prediction file and a sibling `.metrics.json` file.

## Verified Real Runtime Smoke

The following real paths were smoke-tested on a local HuggingFace CausalLM checkpoint and OMIM-style JSON/hospital shards:

- LatentDx-H training: 1 step forward/backward/checkpoint save.
- LatentDx-X cache build: encoder latent cache construction from a trained distiller.
- LatentDx-X training: 1 step projector training and checkpoint save.
- Diagnosis inference: greedy generation for both MedLatent-H and MedLatent-X.

The smoke uses a tiny model and `num_latents=2` only to validate execution. Paper reproduction should use the defaults in the next section.

## Main Paper Defaults

The paper's main MedLatent runs use:

| Setting | Value |
|---|---|
| Hospitals per query | `N = 3` |
| Latent positions per hospital | `m = 32` |
| Retrieved cases per hospital | `1` |
| Max prompt length | `320` tokens |
| Max target length | `64` tokens |
| Decoding | greedy, temperature `0` |
| Frozen modules | all LLM backbones |

## Privacy and Leakage Evaluation

MedLatent evaluates empirical reconstruction leakage from transmitted latent objects. The attack prompt is implemented in `medlatent/prompts.py` and the token-F1 helper is implemented in `medlatent/leakage.py`.

Text-sharing baselines are excluded from this repository because their leakage is direct text exposure rather than reconstruction from a latent representation.
