# Performance and Memory Methodology

> Canonical Header
- Version: 0.1.0
- Status: See docs/design/STATUS.md
- Owners: Platform TL; ML Lead
- Last Updated: 2025-09-16
- Linked Commit: 682ba289170b (describe: 682ba28)

## Purpose

Define a reproducible methodology for benchmarking throughput and characterizing memory usage across MLX and PyTorch backends.

## Environment Capture (Record in every run)

- Hardware: chip (e.g., M4 Max), performance/efficiency core counts, unified memory (GB)
- OS: macOS version
- Python: version (e.g., 3.11.x)
- Packages: MLX, PyTorch, Transformers, PEFT versions
- Backend/device: MLX enabled; PyTorch device = cpu|mps|cuda
- Power state: plugged-in vs battery (if relevant), performance mode

## Workload Definition

- Model: synthetic transformer-like MLP stack (for apples-to-apples without downloads)
- Tokenizer: N/A for synthetic; for real models, record tokenizer version
- Sequence length: L (e.g., 1024)
- Precision: fp32 | bf16 | fp16
- Batch size: micro-batch per step (B)
- Steps: warmup (W), measured steps (N)
- Dataset: synthetic random tensors unless specified; for real datasets, record name/version and preprocessing

## Measurement Protocol

1. Seed RNGs for reproducibility.
2. Run W warmup steps (ignore timings).
3. Time N steps:
   - Training step = forward + loss + backward + optimizer update
   - Force evaluation/sync as needed (MLX: `mx.eval(...)`, Torch: `torch.cuda.synchronize()` if CUDA)
4. Record per-step elapsed seconds and compute:
   - p50 (median), p90 of step time
   - tokens/sec = (B × L) / median_step_time
5. Capture memory:
   - RSS (GB), available (GB), percent via psutil
   - Optional: backend-reported memory if available
6. Report JSON summary with all environment fields and metrics.

## Metrics

- Step time p50, p90 (seconds)
- Tokens/sec (median) = (B × L) / p50
- Memory: RSS GB, available GB, percent
- Optional: throughput per watt if power sampling available

## Memory Model (Training)

Let:
- P = number of model parameters (trainable or frozen as specified)
- bytes(dtype) = 4 for fp32, 2 for fp16/bf16
- For Adam/AdamW, optimizer states ≈ 2× weights (m, v)

### Full Fine-Tuning (all parameters trainable)

- Weights: P × bytes(dtype)
- Gradients: P × bytes(dtype)
- Optimizer states (AdamW): 2 × P × bytes(dtype)
- Activations: k × P × bytes(dtype)  where k depends on batch size, sequence length, and architecture; empirically k ≈ 0.2–1.0
- Total (approx):
  - fp32: \( M_{total} ≈ (1 + 1 + 2 + k) · P · 4 \) bytes = \( (4 + 4k)P \)
  - fp16/bf16: \( M_{total} ≈ (1 + 1 + 2 + k) · P · 2 \) bytes = \( (2 + 2k)P \)

### LoRA-Only Training (base frozen, adapters trainable)

Let r be LoRA rank and f be the fraction of layers/modules adapted. Let P_base be base parameters and P_lora be trainable LoRA parameters. Typically, \( P_{lora} << P_{base} \).

- Weights: P_base × bytes(dtype) + P_lora × bytes(dtype)
- Gradients: P_lora × bytes(dtype) (no grads for frozen base)
- Optimizer states: 2 × P_lora × bytes(dtype)
- Activations: Similar to full model forward; dominated by batch and sequence length (approximately independent of trainable fraction)
- Total (approx):
  - fp16/bf16: \( M_{total} ≈ (P_{base} + P_{lora})·2 + P_{lora}·2 + 2·P_{lora}·2 + k·P_{base}·2 \) bytes
  - Simplify: \( M_{total} ≈ 2·P_{base}·(1 + k) + 6·P_{lora} \) bytes

Notes:
- With gradient checkpointing, effective activation memory reduces by a factor g (compute/memory tradeoff): \( k' ≈ k/g \).
- Optimizer choice changes coefficients: SGD without momentum reduces optimizer states to 0–1× weights.

### Dtype Summary

- fp32: double the weight/grad/opt memory vs fp16/bf16
- bf16 vs fp16: similar memory footprint; stability characteristics differ

## Reporting and Reproducibility

- Always include: chip, memory, OS, Python, package versions, backend, dtype, B, L, W, N
- Attach raw per-step timings and summary JSON emitted by the harness
- Version-lock the harness commit SHA and configuration snapshot

## Tools

- psutil (RSS/available)
- time.perf_counter
- MLX: `mx.eval` to force computation; `mlx.core`, `mlx.nn`
- PyTorch: device synchronization (if CUDA), `torch` tensors and ops

## Running the Benchmark Harness

```bash
# MLX backend
python scripts/benchmark.py --backend mlx --layers 4 --hidden-size 2048 \
  --seq-len 1024 --batch-size 4 --steps 30 --warmup 5 --dtype fp16

# PyTorch (MPS) backend
python scripts/benchmark.py --backend torch --device mps --layers 4 --hidden-size 2048 \
  --seq-len 1024 --batch-size 4 --steps 30 --warmup 5 --dtype fp16
```
