#!/usr/bin/env python3
import argparse
import json
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # Optional

# Optional imports; resolved based on backend
mx = None
nn = None
th = None


@dataclass
class EnvironmentInfo:
    python_version: str
    os: str
    os_version: str
    machine: str
    processor: str
    chip: str
    total_memory_gb: float
    backend: str
    backend_device: str
    mlx_version: str | None
    torch_version: str | None


@dataclass
class BenchmarkConfig:
    layers: int
    hidden_size: int
    seq_len: int
    batch_size: int
    warmup_steps: int
    measure_steps: int
    dtype: str
    seed: int


@dataclass
class BenchmarkResult:
    step_times_s: list[float]
    p50_step_s: float
    p90_step_s: float
    tokens_per_sec: float
    rss_gb: float | None
    available_gb: float | None
    mem_percent: float | None


def read_memory() -> tuple[float | None, float | None, float | None]:
    if psutil is None:
        return None, None, None
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / 1024**3
    vm = psutil.virtual_memory()
    return rss_gb, vm.available / 1024**3, vm.percent


def detect_environment(backend: str, device: str) -> EnvironmentInfo:
    total_mem_gb = None
    if psutil is not None:
        total_mem_gb = psutil.virtual_memory().total / 1024**3
    else:
        total_mem_gb = 0.0

    mlx_version = None
    torch_version = None

    if backend == "mlx":
        try:
            import mlx.core as _mx  # type: ignore

            global mx, nn
            mx = _mx
            import mlx.nn as _nn  # type: ignore

            nn = _nn
            mlx_version = getattr(mx, "__version__", None)
        except Exception:
            pass
    elif backend == "torch":
        try:
            import torch as _th  # type: ignore

            global th
            th = _th
            torch_version = th.__version__
        except Exception:
            pass

    return EnvironmentInfo(
        python_version=sys.version.split(" ")[0],
        os=platform.system(),
        os_version=platform.release(),
        machine=platform.machine(),
        processor=platform.processor(),
        chip=platform.platform(),
        total_memory_gb=float(total_mem_gb),
        backend=backend,
        backend_device=device,
        mlx_version=mlx_version,
        torch_version=torch_version,
    )


def set_seeds(seed: int, backend: str):
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    if backend == "torch" and th is not None:
        th.manual_seed(seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(seed)


# ------------------------ MLX Model ------------------------


def build_mlx_mlp(layers: int, hidden: int, dtype: str):
    d = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}[dtype]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = []
            for _ in range(layers):
                self.layers.append(
                    nn.Sequential(
                        nn.Linear(hidden, hidden),
                        nn.ReLU(),
                        nn.Linear(hidden, hidden),
                    )
                )
            self.to_dtype(d)

        def to_dtype(self, dtype_):
            # MLX modules are dtype-agnostic; we rely on input dtype
            self.dtype_ = dtype_

        def __call__(self, x):
            for block in self.layers:
                x = block(x)
            return x

    return MLP()


def run_mlx(config: BenchmarkConfig) -> BenchmarkResult:
    x = mx.random.uniform(shape=(config.batch_size, config.seq_len, config.hidden_size))
    if config.dtype != "fp32":
        cast = {"fp16": mx.float16, "bf16": mx.bfloat16}[config.dtype]
        x = x.astype(cast)

    model = build_mlx_mlp(config.layers, config.hidden_size, config.dtype)

    def loss_fn(m, batch):
        y = m(batch)
        return mx.mean(y * y)

    opt = nn.optim.Adam(learning_rate=1e-3)

    def train_step():
        loss, grads = mx.value_and_grad(loss_fn)(model, x)
        opt.update(model, grads)
        mx.eval(loss)
        return loss

    # Warmup
    for _ in range(config.warmup_steps):
        train_step()

    # Measure
    step_times = []
    for _ in range(config.measure_steps):
        t0 = time.perf_counter()
        train_step()
        mx.eval(x)
        step_times.append(time.perf_counter() - t0)

    p50 = statistics.median(step_times)
    p90 = statistics.quantiles(step_times, n=10)[8] if len(step_times) >= 2 else p50
    tokens_per_sec = (config.batch_size * config.seq_len) / p50 if p50 > 0 else 0.0
    rss, avail, pct = read_memory()
    return BenchmarkResult(step_times, p50, p90, tokens_per_sec, rss, avail, pct)


# ------------------------ Torch Model ------------------------


def build_torch_mlp(layers: int, hidden: int, dtype: str, device: str):
    d = {"fp32": th.float32, "fp16": th.float16, "bf16": th.bfloat16}[dtype]

    class MLP(th.nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            for _ in range(layers):
                blocks.extend(
                    [
                        th.nn.Linear(hidden, hidden),
                        th.nn.ReLU(),
                        th.nn.Linear(hidden, hidden),
                    ]
                )
            self.net = th.nn.Sequential(*blocks)

        def forward(self, x):
            return self.net(x)

    model = MLP().to(device=device, dtype=d)
    return model, d


def run_torch(config: BenchmarkConfig, device: str) -> BenchmarkResult:
    device_ = th.device(device)
    model, dtype_ = build_torch_mlp(config.layers, config.hidden_size, config.dtype, device_)
    x = th.rand(
        (config.batch_size, config.seq_len, config.hidden_size), device=device_, dtype=dtype_
    )

    opt = th.optim.AdamW(model.parameters(), lr=1e-3)

    def train_step():
        opt.zero_grad(set_to_none=True)
        y = model(x)
        loss = (y * y).mean()
        loss.backward()
        opt.step()
        return loss

    # Warmup
    for _ in range(config.warmup_steps):
        train_step()

    # Measure
    step_times: list[float] = []
    for _ in range(config.measure_steps):
        t0 = time.perf_counter()
        loss = train_step()
        if device_.type == "cuda":
            th.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)

    p50 = statistics.median(step_times)
    p90 = statistics.quantiles(step_times, n=10)[8] if len(step_times) >= 2 else p50
    tokens_per_sec = (config.batch_size * config.seq_len) / p50 if p50 > 0 else 0.0
    rss, avail, pct = read_memory()
    return BenchmarkResult(step_times, p50, p90, tokens_per_sec, rss, avail, pct)


def main():
    parser = argparse.ArgumentParser(description="Synthetic training benchmark for MLX and Torch")
    parser.add_argument("--backend", choices=["mlx", "torch"], required=True)
    parser.add_argument("--device", default="cpu", help="torch: cpu|mps|cuda; mlx: ignored")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="")

    args = parser.parse_args()

    env = detect_environment(args.backend, args.device)
    set_seeds(args.seed, args.backend)

    cfg = BenchmarkConfig(
        layers=args.layers,
        hidden_size=args.hidden_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        warmup_steps=args.warmup,
        measure_steps=args.steps,
        dtype=args.dtype,
        seed=args.seed,
    )

    if args.backend == "mlx":
        if mx is None:
            print("MLX is not available. Install mlx and rerun.", file=sys.stderr)
            sys.exit(1)
        result = run_mlx(cfg)
    else:
        if th is None:
            print("PyTorch is not available. Install torch and rerun.", file=sys.stderr)
            sys.exit(1)
        result = run_torch(cfg, args.device)

    summary = {
        "environment": asdict(env),
        "config": asdict(cfg),
        "metrics": {
            "p50_step_s": result.p50_step_s,
            "p90_step_s": result.p90_step_s,
            "tokens_per_sec": result.tokens_per_sec,
            "rss_gb": result.rss_gb,
            "available_gb": result.available_gb,
            "mem_percent": result.mem_percent,
        },
        "step_times_s": result.step_times_s,
        "harness": {
            "commit": "682ba289170b",
        },
    }

    output = json.dumps(summary, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
