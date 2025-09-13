"""
Abstract base class for training backends.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Backend(ABC):
    """Abstract base class for ML training backends."""

    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> Any:
        """Load a model from HuggingFace or local path."""
        pass

    @abstractmethod
    def create_optimizer(self, params: Any, learning_rate: float, **kwargs) -> Any:
        """Create an optimizer for training."""
        pass

    @abstractmethod
    def compute_loss(self, logits: Any, labels: Any) -> Any:
        """Compute loss from logits and labels."""
        pass

    @abstractmethod
    def backward(self, loss: Any) -> None:
        """Compute gradients via backpropagation."""
        pass

    @abstractmethod
    def optimizer_step(self, optimizer: Any) -> None:
        """Update model parameters."""
        pass

    @abstractmethod
    def zero_grad(self, optimizer: Any) -> None:
        """Zero out gradients."""
        pass

    @abstractmethod
    def to_device(self, tensor: Any) -> Any:
        """Move tensor to appropriate device."""
        pass

    @abstractmethod
    def save_checkpoint(self, model: Any, path: Path, **kwargs) -> None:
        """Save model checkpoint."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: Path, **kwargs) -> Any:
        """Load model checkpoint."""
        pass

    @abstractmethod
    def get_device_info(self) -> dict[str, Any]:
        """Get information about the current device."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the backend."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available on this system."""
        pass
