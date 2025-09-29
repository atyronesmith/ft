# End-to-End Fine-Tuning Test Plan (Tiny Model → Ollama)

> Canonical Header
- Version: 0.1.0
- Status: See docs/design/STATUS.md
- Owners: Product Eng; ML Lead; Platform TL
- Last Updated: 2025-09-16
- Linked Commit: 682ba289170b (describe: 682ba28)

## Objective

Validate the full production workflow: model acquisition (with cache), synthetic dataset generation (100 Q/A pairs), fine-tuning with LoRA on Apple Silicon (MLX-first), export/convert to an Ollama-compatible format, serve with Ollama, run 100 inference queries, and summarize training quality.

## Scope

- Minimal viable model capable of fine-tuning (smallest LLM from HuggingFace that supports instruction-style SFT)
- Deterministic, reproducible pipeline with caching and environment capture
- No manual intervention; fully scriptable using existing CLI plus small glue scripts

## Out of Scope / Planned

- Human annotation of quality; we use automatic heuristics (BLEU/ROUGE/LLM-as-judge optional)
- Multi-node/distributed runs
- Very large models (>3B parameters)

## Model Selection

- Candidate: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B, chat-aligned)
  - Rationale: smallest widely-used chat-capable Llama derivative suitable for LoRA SFT
  - Alternatives: `gpt2` (too small for chat coherence), `tiiuae/falcon-rw-1b` (larger RAM), `microsoft/DialoGPT-small` (dialog-only; OK but older architecture)

## Caching Strategy

- Use Hugging Face cache: `HF_HOME` and `TRANSFORMERS_CACHE` set to a shared location, e.g., `~/.cache/huggingface`
- Pre-pull model and tokenizer via CLI step to avoid network during testing window
- Persist dataset and checkpoints under project-local cache dir (e.g., `./output/cache`)

## Environment & Determinism

- Record: chip (M-series), RAM, macOS version, Python, MLX/PyTorch/Transformers versions
- Fix seeds: 42 for Python/NumPy/Torch/MLX where applicable
- Capture git SHA, config checksum, and CLI args into a run manifest (JSON)

## Data Generation (100 Q/A Pairs)

- Goal: coherent, on-topic, reproducible set
- Strategy: generate synthetic Q/A focused on a single domain (e.g., world capitals, basic math word problems, or macOS how-tos)
  - Template fields: instruction, input (optional), output
  - Ensure non-triviality (vary entity slots, add distractors)
  - Validation: no empty fields, length bounds, uniqueness ≥ 95%
- Format: JSONL with fields: `instruction`, `input`, `output`
- Split: train=90, val=10

## Fine-Tuning Configuration

- Backend: auto (MLX preferred, PyTorch fallback)
- LoRA: rank=16, alpha=32, dropout=0.05; target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Training:
  - batch_size: 4
  - grad_accum: 4
  - max_steps: 300 (or 2 epochs, whichever first)
  - lr: 1e-4, warmup_steps: 30, scheduler: cosine
  - precision: fp16/bf16 where supported
- Checkpoints:
  - Save every 50 steps; keep best on val loss
  - Save LoRA adapters and merged checkpoint for export

## Steps Overview

1) Prepare environment
- Set caches: `HF_HOME`, `TRANSFORMERS_CACHE`
- Verify MLX availability; record environment using `scripts/benchmark.py --backend mlx --steps 1 --warmup 0 --output env.json`

2) Download model and tokenizer (cached)
- `ft models pull TinyLlama/TinyLlama-1.1B-Chat-v1.0`

3) Generate dataset (100 Q/A)
- Run a script to emit `data/train.jsonl` and `data/val.jsonl` using a deterministic template generator
- Validate via `ft dataset info data/train.jsonl` and `ft dataset template --format alpaca`

4) Validate configuration
- Author `config/train.yml` with parameters above; run `ft train validate config/train.yml`

5) Fine-tune (LoRA)
- `ft train start TinyLlama/TinyLlama-1.1B-Chat-v1.0 data/train.jsonl --template chatml --epochs 2 --batch-size 4 --lora-rank 16 --profile chat`
- Outputs: checkpoints in `./output/checkpoints/…` and final merged weights

6) Export/Convert for Ollama
- Convert model to GGUF or other Ollama-supported format
  - Preferred: export HF -> GGUF via conversion tool, or MLX → PyTorch → GGUF if needed
  - Produce `model.gguf` and a minimal `Modelfile`

7) Serve with Ollama
- `ollama create tiny-sft -f Modelfile`
- `ollama run tiny-sft`

8) Evaluation (100 questions)
- Send the 100 questions (from the dataset’s instruction/input), capture responses
- Scoring:
  - Exact/approximate match for closed-form tasks
  - Heuristic LLM-as-judge prompt for coherence and helpfulness (optional offline)
  - Aggregate metrics: accuracy (if applicable), ROUGE-L, BLEU, average judge score

9) Reporting
- Emit `reports/e2e_summary.json` with:
  - environment, model id, dataset stats, training metrics (final loss), export format
  - inference metrics on 100 Qs, sample predictions
  - artifact paths (checkpoints, gguf, Modelfile)

## Pass/Fail Criteria

- Pipeline completes without errors in < 60 minutes on M-series with ≥ 48GB RAM
- All artifacts produced: dataset, checkpoints, merged model, GGUF, Modelfile, summary report
- Inference responses produced for all 100 questions
- Quality: ≥ 70% exact/heuristic correctness for domain-selected tasks (tunable)

## Risks & Mitigations

- Download failures → use cached HF and pre-pull step
- Memory limits on smaller Macs → reduce batch size and steps, use gradient accumulation
- Conversion toolchain drift → pin converter versions and add smoke test
- Ollama incompatibilities → test with a tiny GGUF baseline first

## Extensions

- Swap model to `microsoft/DialoGPT-small` for dialog data; compare results
- Add human-in-the-loop scoring subset (10 items)
- Track tokens/sec and step time using the benchmark harness and include in the report
