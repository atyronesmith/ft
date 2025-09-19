"""LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning."""

import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation.

    Attributes:
        r: Rank of the low-rank matrices (default: 8)
        alpha: Scaling factor for LoRA updates (default: 16)
        dropout: Dropout probability for LoRA layers (default: 0.0)
        target_modules: List of module names to apply LoRA to
        use_rslora: Whether to use rank-stabilized LoRA (default: False)
        use_dora: Whether to use weight-decomposed LoRA (default: False)
    """
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: list[str] | None = None
    use_rslora: bool = False
    use_dora: bool = False

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]

        if self.r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.r}")

        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")

        if not 0 <= self.dropout < 1:
            raise ValueError(f"Dropout must be in [0, 1), got {self.dropout}")

    @property
    def scaling(self) -> float:
        """Calculate the LoRA scaling factor."""
        if self.use_rslora:
            return self.alpha / math.sqrt(self.r)
        return self.alpha / self.r


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    This layer adds trainable low-rank matrices A and B to a frozen linear layer,
    computing: output = Wx + (BA)x * scaling
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        bias: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Frozen base layer
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.base.freeze()

        # LoRA parameters with more conservative initialization
        # Use smaller scale to prevent exploding gradients
        init_scale = 0.01 / math.sqrt(config.r)  # Much smaller initialization
        self.lora_a = mx.random.normal(
            shape=(config.r, in_features),
            scale=init_scale
        )
        self.lora_b = mx.zeros((out_features, config.r))

        # Optional dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        # DoRA components (if enabled)
        if config.use_dora:
            self.magnitude = mx.ones((out_features, 1))

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with LoRA adaptation."""
        # Base forward pass
        result = self.base(x)

        # Apply dropout if configured
        if self.dropout is not None and self.training:
            x = self.dropout(x)

        # LoRA forward pass: (BA)x * scaling
        lora_out = x @ self.lora_a.T @ self.lora_b.T
        lora_out = lora_out * self.config.scaling

        # DoRA: apply magnitude vector
        if self.config.use_dora:
            # Normalize and scale by magnitude
            weight_norm = mx.linalg.norm(
                self.base.weight + self.lora_b @ self.lora_a * self.config.scaling,
                axis=1,
                keepdims=True
            )
            lora_out = lora_out * (self.magnitude / weight_norm).T

        return result + lora_out

    def merged_weight(self) -> mx.array:
        """Get the merged weight matrix (base + LoRA)."""
        return self.base.weight + (self.lora_b @ self.lora_a) * self.config.scaling


class LoRALayer:
    """Mixin for adding LoRA functionality to existing layers."""

    def _resolve_path(self, root: nn.Module | list, path: str):
        """Resolve dotted path supporting list indices (e.g., layers.21.attn)."""
        obj = root
        for part in path.split(".") if path else []:
            if isinstance(obj, list):
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    raise AttributeError(f"List index must be numeric, got {part}")
            else:
                obj = getattr(obj, part)
        return obj

    def add_lora(self, config: LoRAConfig) -> None:
        """Add LoRA adapters to compatible submodules."""
        for name, module in self.named_modules():
            if any(target in name for target in config.target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with LoRA version
                    lora_linear = LoRALinear(
                        module.weight.shape[1],
                        module.weight.shape[0],
                        config,
                        bias=module.bias is not None
                    )
                    # Copy weights (create new tensors to ensure they're truly frozen)
                    lora_linear.base.weight = mx.array(module.weight)
                    if module.bias is not None:
                        lora_linear.base.bias = mx.array(module.bias)

                    # Ensure the base weights are truly frozen
                    lora_linear.base.freeze()

                    # Replace module
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = self._resolve_path(self, parent_name)
                    setattr(parent, child_name, lora_linear)

    def get_lora_params(self) -> dict[str, mx.array]:
        """Get only the LoRA parameters for saving/loading."""
        lora_params = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                lora_params[f"{name}.lora_a"] = module.lora_a
                lora_params[f"{name}.lora_b"] = module.lora_b
                if module.config.use_dora:
                    lora_params[f"{name}.magnitude"] = module.magnitude
        return lora_params

    def load_lora_params(self, params: dict[str, mx.array]) -> None:
        """Load LoRA parameters."""
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                if f"{name}.lora_a" in params:
                    module.lora_a = params[f"{name}.lora_a"]
                if f"{name}.lora_b" in params:
                    module.lora_b = params[f"{name}.lora_b"]
                if f"{name}.magnitude" in params and module.config.use_dora:
                    module.magnitude = params[f"{name}.magnitude"]

    def merge_lora(self) -> None:
        """Merge LoRA weights into base weights (for inference)."""
        for _name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                module.base.weight = module.merged_weight()
                # Reset LoRA parameters
                module.lora_a = mx.zeros_like(module.lora_a)
                module.lora_b = mx.zeros_like(module.lora_b)

    def trainable_parameters(self) -> list[mx.array]:
        """Get only trainable (LoRA) parameters."""
        params = []
        for _name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                params.extend([module.lora_a, module.lora_b])
                if module.config.use_dora:
                    params.append(module.magnitude)
        return params


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Apply LoRA adaptation to an existing model.

    Args:
        model: The model to adapt
        config: LoRA configuration

    Returns:
        The model with LoRA adapters added
    """
    # Find and replace target modules
    def _resolve_path(root: nn.Module | list, path: str):
        obj = root
        for part in path.split(".") if path else []:
            if isinstance(obj, list):
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    raise AttributeError(f"List index must be numeric, got {part}")
            else:
                obj = getattr(obj, part)
        return obj

    for name, module in model.named_modules():
        # Check if this module name matches any target
        module_name = name.split(".")[-1] if "." in name else name
        if module_name in config.target_modules:
            if isinstance(module, nn.Linear):
                # Get parent and create LoRA replacement
                parent_name = ".".join(name.split(".")[:-1])

                lora_linear = LoRALinear(
                    module.weight.shape[1],
                    module.weight.shape[0],
                    config,
                    bias=hasattr(module, "bias") and module.bias is not None
                )

                # Copy original weights
                lora_linear.base.weight = module.weight
                if hasattr(module, "bias") and module.bias is not None:
                    lora_linear.base.bias = module.bias

                # Replace in parent
                parent = _resolve_path(model, parent_name)
                setattr(parent, module_name, lora_linear)

    return model


def get_lora_trainable_params(model: nn.Module) -> tuple[list[mx.array], int, int]:
    """Get trainable LoRA parameters from a model.

    Returns:
        Tuple of (parameters, trainable_count, total_count)
    """
    trainable_params = []
    trainable_count = 0
    total_count = 0

    for name, param in model.named_parameters():
        total_count += param.size

        # Check if this is a LoRA parameter
        if "lora_" in name or "magnitude" in name:
            trainable_params.append(param)
            trainable_count += param.size

    return trainable_params, trainable_count, total_count


def save_lora_weights(model: nn.Module, path: str | Path) -> None:
    """Save only LoRA weights to a file."""
    lora_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_weights[f"{name}.lora_a"] = module.lora_a
            lora_weights[f"{name}.lora_b"] = module.lora_b
            if hasattr(module, "magnitude"):
                lora_weights[f"{name}.magnitude"] = module.magnitude
    # Save as npz (dict of arrays)
    mx.savez(str(path), **lora_weights)


def load_lora_weights(model: nn.Module, path: str | Path) -> None:
    """Load LoRA weights into a model."""
    lora_weights = mx.load(str(path))

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if f"{name}.lora_a" in lora_weights:
                module.lora_a = lora_weights[f"{name}.lora_a"]
            if f"{name}.lora_b" in lora_weights:
                module.lora_b = lora_weights[f"{name}.lora_b"]
            if f"{name}.magnitude" in lora_weights:
                module.magnitude = lora_weights[f"{name}.magnitude"]
