"""
MLX backend implementation for Apple Silicon optimization.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import mlx
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None
    optim = None

if TYPE_CHECKING and MLX_AVAILABLE:
    # For type hints only
    import mlx.core as mx

from finetune.backends.base import Backend


class MLXBackend(Backend):
    """MLX backend for training on Apple Silicon."""

    def __init__(self):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not installed. Please install with: pip install mlx")
        self.device = mx.default_device()

    @property
    def name(self) -> str:
        return "mlx"

    @property
    def is_available(self) -> bool:
        return MLX_AVAILABLE

    def load_model(self, model_name: str, **kwargs) -> Any:
        """Load and convert a model to MLX format."""
        # For now, return a placeholder
        # Full implementation will convert from HuggingFace
        raise NotImplementedError("Model loading will be implemented in the model loader module")

    def create_optimizer(self, params: Any, learning_rate: float, **kwargs) -> Any:
        """Create an MLX optimizer."""
        weight_decay = kwargs.get("weight_decay", 0.01)
        betas = kwargs.get("betas", [0.9, 0.999])

        return optim.AdamW(learning_rate=learning_rate, betas=betas, weight_decay=weight_decay)

    def compute_loss(self, logits: Any, labels: Any) -> Any:
        """Compute cross-entropy loss using MLX."""
        # Reshape for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        # Compute cross-entropy loss
        loss = mx.mean(nn.losses.cross_entropy(logits_flat, labels_flat, reduction="none"))
        return loss

    def backward(self, loss: Any) -> None:
        """MLX handles gradients automatically with value_and_grad."""
        # In MLX, gradients are computed differently
        # This is handled in the training loop
        pass

    def optimizer_step(self, optimizer: Any) -> None:
        """Update parameters with MLX optimizer."""
        # MLX optimizers work differently - they need model and gradients
        # This will be handled in the training loop
        pass

    def zero_grad(self, optimizer: Any) -> None:
        """MLX doesn't require explicit zero_grad."""
        pass

    def to_device(self, tensor: Any) -> Any:
        """Convert to MLX array."""
        if isinstance(tensor, mx.array):
            return tensor
        elif isinstance(tensor, np.ndarray):
            return mx.array(tensor)
        elif hasattr(tensor, "numpy"):  # PyTorch tensor
            return mx.array(tensor.detach().cpu().numpy())
        else:
            return mx.array(tensor)

    def save_checkpoint(self, model: Any, path: Path, **kwargs) -> None:
        """Save MLX model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model weights
        weights = model.parameters() if hasattr(model, "parameters") else model
        mx.save(str(path), weights)

        # Save additional metadata if provided
        metadata = kwargs.get("metadata", {})
        if metadata:
            import json

            metadata_path = path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

    def load_checkpoint(self, path: Path, **kwargs) -> Any:
        """Load MLX model checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        weights = mx.load(str(path))
        return weights

    def get_device_info(self) -> dict[str, Any]:
        """Get MLX device information."""
        import platform
        import subprocess

        info = {
            "backend": "mlx",
            "device": str(self.device),
            "mlx_version": getattr(mlx, "__version__", "unknown") if MLX_AVAILABLE else None,
            "platform": platform.platform(),
            "processor": platform.processor(),
        }

        # Try to get Apple Silicon chip info
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            if result.returncode == 0:
                info["chip"] = result.stdout.strip()
        except:
            pass

        # Get memory info
        try:
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            if result.returncode == 0:
                mem_bytes = int(result.stdout.strip())
                info["total_memory_gb"] = mem_bytes / (1024**3)
        except:
            pass

        return info

    def convert_weights_from_torch(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert PyTorch state dict to MLX arrays."""
        mlx_weights = {}
        for name, param in state_dict.items():
            if hasattr(param, "numpy"):
                # PyTorch tensor
                numpy_array = param.detach().cpu().numpy()
            else:
                numpy_array = np.array(param)

            mlx_weights[name] = mx.array(numpy_array)

        return mlx_weights

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        # MLX handles memory efficiently by default
        # Additional optimizations can be added here
        pass

    def compile_model(self, model: Any) -> Any:
        """Compile model for optimized execution."""
        # MLX compilation is handled automatically
        # We can add JIT compilation here if needed
        return model
