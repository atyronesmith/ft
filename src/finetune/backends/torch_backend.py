"""
PyTorch backend implementation for fallback support.
"""

from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None

from finetune.backends.base import Backend
from finetune.backends.device import DeviceType


class PyTorchBackend(Backend):
    """PyTorch backend for training."""

    def __init__(self, device_type: DeviceType = DeviceType.CPU):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Please install with: pip install torch")

        # Set device based on availability
        if device_type == DeviceType.APPLE_SILICON and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif device_type == DeviceType.CUDA and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    @property
    def name(self) -> str:
        return "pytorch"

    @property
    def is_available(self) -> bool:
        return TORCH_AVAILABLE

    def load_model(self, model_name: str, **kwargs) -> Any:
        """Load a model from HuggingFace."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            **kwargs,
        )

        if self.device.type == "mps":
            model = model.to(self.device)

        return model

    def create_optimizer(self, params: Any, learning_rate: float, **kwargs) -> Any:
        """Create a PyTorch optimizer."""
        weight_decay = kwargs.get("weight_decay", 0.01)
        betas = kwargs.get("betas", (0.9, 0.999))
        eps = kwargs.get("eps", 1e-8)

        return optim.AdamW(
            params, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay
        )

    def compute_loss(self, logits: Any, labels: Any) -> Any:
        """Compute cross-entropy loss."""
        # Shift for language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        loss = loss_fct(shift_logits, shift_labels)
        return loss

    def backward(self, loss: Any) -> None:
        """Compute gradients via backpropagation."""
        loss.backward()

    def optimizer_step(self, optimizer: Any) -> None:
        """Update model parameters."""
        optimizer.step()

    def zero_grad(self, optimizer: Any) -> None:
        """Zero out gradients."""
        optimizer.zero_grad()

    def to_device(self, tensor: Any) -> Any:
        """Move tensor to device."""
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        else:
            return torch.tensor(tensor, device=self.device)

    def save_checkpoint(self, model: Any, path: Path, **kwargs) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": kwargs.get("config", {}),
            "epoch": kwargs.get("epoch", 0),
            "global_step": kwargs.get("global_step", 0),
        }

        if "optimizer" in kwargs:
            checkpoint["optimizer_state_dict"] = kwargs["optimizer"].state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path, **kwargs) -> Any:
        """Load model checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        return checkpoint

    def get_device_info(self) -> dict[str, Any]:
        """Get device information."""
        info = {
            "backend": "pytorch",
            "device": str(self.device),
            "torch_version": torch.__version__ if TORCH_AVAILABLE else None,
        }

        if self.device.type == "cuda":
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        elif self.device.type == "mps":
            info["mps_available"] = True
            # MPS doesn't provide detailed memory info yet

        return info

    def enable_gradient_checkpointing(self, model: Any) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()

    def compile_model(self, model: Any) -> Any:
        """Compile model for optimized execution (PyTorch 2.0+)."""
        if hasattr(torch, "compile"):
            return torch.compile(model)
        return model
