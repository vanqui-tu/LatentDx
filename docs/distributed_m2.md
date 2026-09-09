# Distributed M2 Runs

Run these commands from the repository root with a CUDA-compatible PyTorch
installation. The structured baselines do not use the GPU; the text pilot does.

The skewed medical setting needs these copied paths on the instance:

```text
data_original_skewed/hospital_0.json ... hospital_4.json
data_original_skewed/test.json
data/hpo_embeddings.json.gz
data/hpo_ic.json
```

Install the repository and verify the checkout before an experiment:

```bash
python -m pip install -e . --no-deps
python -m pytest -q
git rev-parse HEAD
```

Every run writes `episodes.jsonl`, `summary.json`, and `run.json`. The manifest
records the command, Git commit, graph, seed, input hashes, library versions,
and evaluation rule. Accuracy accepts the canonical disease label or any declared
`disease_aliases` entry after case-folding and punctuation normalization.

## Structured Canonical Run

This is M2L2's primary comparison on all 401 skewed test cases.

```bash
python scripts/run_distributed_medical.py \
  --hospital_dir data_original_skewed \
  --split_file data_original_skewed/test.json \
  --hpo_embeddings_file data/hpo_embeddings.json.gz \
  --hpo_ic_file data/hpo_ic.json \
  --output_dir outputs/distributed_m2/structured_ring_r4_k2 \
  --channel structured \
  --seed 42 \
  --num_agents 5 \
  --topology ring \
  --rounds 4 \
  --max_fanout 2
```

## Structured Ablations

Run exactly these two after the canonical result.

```bash
python scripts/run_distributed_medical.py \
  --hospital_dir data_original_skewed \
  --split_file data_original_skewed/test.json \
  --hpo_embeddings_file data/hpo_embeddings.json.gz \
  --hpo_ic_file data/hpo_ic.json \
  --output_dir outputs/distributed_m2/structured_ring_r2_k2 \
  --channel structured \
  --seed 42 \
  --num_agents 5 \
  --topology ring \
  --rounds 2 \
  --max_fanout 2

python scripts/run_distributed_medical.py \
  --hospital_dir data_original_skewed \
  --split_file data_original_skewed/test.json \
  --hpo_embeddings_file data/hpo_embeddings.json.gz \
  --hpo_ic_file data/hpo_ic.json \
  --output_dir outputs/distributed_m2/structured_complete_r2_k4 \
  --channel structured \
  --seed 42 \
  --num_agents 5 \
  --topology complete \
  --rounds 2 \
  --max_fanout 4
```

## Text Preflight And Pilot

Pin a local Hugging Face model path or an immutable revision before running this.
Greedy decoding and deterministic PyTorch mode are enabled by default.

For a separate GPU server, vLLM is supported through its OpenAI-compatible
`/v1/chat/completions` endpoint. Start the server (the model name must match the
served model ID):

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 --host 0.0.0.0 --port 8000
```

Then pass `--vllm_base_url http://127.0.0.1:8000` to the text runner. This path
does not import or initialize a local Transformers model; structured runs are
unchanged. Thinking is disabled by default so the returned text follows the
same bounded-summary contract as the local adapter.

```bash
python scripts/run_distributed_medical.py \
  --hospital_dir data_original_skewed \
  --split_file data_original_skewed/test.json \
  --hpo_embeddings_file data/hpo_embeddings.json.gz \
  --hpo_ic_file data/hpo_ic.json \
  --output_dir outputs/distributed_m2/text_preflight \
  --channel text \
  --model_name /models/frozen-text-model \
  --local_files_only \
  --max_samples 1 \
  --methods B2 \
  --seed 42 \
  --num_agents 5 \
  --topology ring \
  --rounds 4 \
  --max_fanout 2 \
  --device cuda \
  --dtype bfloat16

python scripts/run_distributed_medical.py \
  --hospital_dir data_original_skewed \
  --split_file data_original_skewed/test.json \
  --hpo_embeddings_file data/hpo_embeddings.json.gz \
  --hpo_ic_file data/hpo_ic.json \
  --output_dir outputs/distributed_m2/text_vllm_b2_first50 \
  --channel text \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --vllm_base_url http://127.0.0.1:8000 \
  --max_samples 50 \
  --methods B2 \
  --seed 42 \
  --num_agents 5 \
  --topology ring \
  --rounds 4 \
  --max_fanout 2

python scripts/run_distributed_medical.py \
  --hospital_dir data_original_skewed \
  --split_file data_original_skewed/test.json \
  --hpo_embeddings_file data/hpo_embeddings.json.gz \
  --hpo_ic_file data/hpo_ic.json \
  --output_dir outputs/distributed_m2/text_b2_first50 \
  --channel text \
  --model_name /models/frozen-text-model \
  --local_files_only \
  --max_samples 50 \
  --methods B2 \
  --seed 42 \
  --num_agents 5 \
  --topology ring \
  --rounds 4 \
  --max_fanout 2 \
  --device cuda \
  --dtype bfloat16
```

For a determinism check, repeat one structured command in a different output
directory and compare only its prediction and summary artifacts:

```bash
cmp outputs/distributed_m2/structured_ring_r4_k2/episodes.jsonl outputs/distributed_m2/structured_ring_r4_k2_repeat/episodes.jsonl
cmp outputs/distributed_m2/structured_ring_r4_k2/summary.json outputs/distributed_m2/structured_ring_r4_k2_repeat/summary.json
```

`run.json` intentionally differs because it contains the output-path command.
