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
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

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
    """LoRA Linear layer - EXACT MLX example implementation."""

    @staticmethod
    def from_linear(linear: nn.Linear, rank: int | LoRAConfig = 8):
        # TODO remove when input_dims and output_dims are attributes
        # on linear and quantized linear
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits
        lora_lin = LoRALinear(input_dims, output_dims, rank)
        lora_lin.linear = linear
        # CRITICAL: Freeze the base layer to exclude it from trainable_parameters
        lora_lin.linear.freeze()
        lora_lin.base.freeze()
        return lora_lin

    def to_linear(self):
        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        is_quantized = isinstance(linear, nn.QuantizedLinear)

        # Use the same type as the linear weight if not quantized
        dtype = weight.dtype

        if is_quantized:
            dtype = mx.float16
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                linear.group_size,
                linear.bits,
            )
        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        # Keep LoRA weights in their original dtype (bfloat16) - no conversion
        lora_b = self.scale * self.lora_b.T
        lora_a = self.lora_a.T
        fused_linear.weight = weight + lora_b @ lora_a
        if bias:
            fused_linear.bias = linear.bias

        if is_quantized:
            fused_linear = nn.QuantizedLinear.from_linear(
                fused_linear,
                linear.group_size,
                linear.bits,
            )

        return fused_linear

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        lora_rank: int | LoRAConfig = 8,
        bias: bool = False,
        scale: float = None,
    ):
        super().__init__()

        # Handle both LoRAConfig and int for rank (backward compatibility)
        if isinstance(lora_rank, LoRAConfig):
            rank = lora_rank.r
            # Use LoRAConfig dropout if provided
            self.dropout = lora_rank.dropout if lora_rank.dropout > 0 else None
            # Use proper LoRA scaling from config
            self.scale = scale if scale is not None else lora_rank.scaling
        else:
            rank = lora_rank
            self.dropout = None
            # Default scaling for backward compatibility (alpha=16, r=8 -> scale=2.0)
            self.scale = scale if scale is not None else 16.0 / rank

        # Regular linear layer weights
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        # Add base attribute for test compatibility
        self.base = self.linear

        # Low rank lora weights - initialize in bfloat16 for consistency
        init_scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(input_dims, rank),
            dtype=mx.bfloat16
        )
        self.lora_b = mx.zeros(shape=(rank, output_dims), dtype=mx.bfloat16)

    def __call__(self, x):
        # Use consistent bfloat16 dtype - no conversions needed
        y = self.linear(x)

        # Apply dropout to input if configured (for compatibility)
        lora_input = x
        if self.dropout is not None and self.training:
            # Create dropout on the fly (MLX pattern)
            dropout_layer = nn.Dropout(self.dropout)
            lora_input = dropout_layer(x)

        z = (lora_input @ self.lora_a) @ self.lora_b
        return y + self.scale * z


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
                    # Use the new from_linear factory method
                    lora_linear = LoRALinear.from_linear(module, config)

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
                # Use the new from_linear factory method
                lora_linear = LoRALinear.from_linear(module, config)

                # Replace in parent
                parent_name = ".".join(name.split(".")[:-1])
                parent = _resolve_path(model, parent_name)
                setattr(parent, module_name, lora_linear)

    return model


def get_lora_trainable_params(model: nn.Module) -> tuple[list[mx.array], int, int]:
    """Get trainable LoRA parameters from a model - MLX example pattern.

    Returns:
        Tuple of (parameters, trainable_count, total_count)
    """
    # Use MLX tree_flatten to get all parameters
    from mlx.utils import tree_flatten

    all_params = tree_flatten(model.parameters())
    trainable_params = tree_flatten(model.trainable_parameters())

    # Count total parameters (exact count, not millions)
    total_count = sum(v.size for _, v in all_params)

    # Count trainable parameters (exact count, not millions)
    trainable_count = sum(v.size for _, v in trainable_params)

    # Return the actual trainable parameter arrays
    trainable_param_arrays = [v for _, v in trainable_params]

    return trainable_param_arrays, trainable_count, total_count


def save_lora_weights(model: nn.Module, path: str | Path) -> None:
    """Save only LoRA weights to a file - MLX example pattern."""
    from mlx.utils import tree_flatten

    # Use MLX tree_flatten like the example
    lora_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.savez(str(path), **lora_weights)


def load_lora_weights(model: nn.Module, path: str | Path) -> None:
    """Load LoRA weights into a model - MLX example pattern."""
    # Load weights using MLX pattern
    model.load_weights(str(path), strict=False)
