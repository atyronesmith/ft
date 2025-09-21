# LoRA Training Loop Rewrite Plan

**Created:** 2025-09-21
**Status:** CRITICAL - Training System Broken
**Priority:** P0 - Blocks all fine-tuning functionality

## Executive Summary

The current LoRA training implementation has a fundamental architectural mismatch with MLX's automatic differentiation system. This document outlines a complete rewrite plan based on analysis of STATUS.md and LORA-TRAIN.md to create a native MLX training system.

## Root Cause Analysis

### Current Implementation Problems

1. **PyTorch-Style Parameter Handling**: Current code tries to use PyTorch patterns with MLX
   - Attempts path-based parameter navigation incompatible with MLX parameter trees
   - Uses dict-style parameter access that breaks MLX's value_and_grad function
   - Results in "'dict' object has no attribute 'forward'" errors

2. **MLX value_and_grad Misunderstanding**:
   - Current code passes model objects to value_and_grad incorrectly
   - MLX converts models to parameter trees (dicts), breaking forward() method
   - Improper handling of MLX's functional automatic differentiation

3. **Parameter Tree Navigation Issues**:
   - MLX models use nested structures (lists for layers) incompatible with path-based updates
   - Current implementation expects PyTorch-style flat parameter handling

## Rewrite Strategy: Native MLX Approach

### Option A: Complete Functional Redesign (RECOMMENDED)

**Timeline: 4-6 hours**
**Approach**: Rewrite training loop using MLX's native functional patterns from LORA-TRAIN.md

#### Phase 1: MLX-Native Loss Function (1 hour)

Replace current training_step with canonical MLX pattern:

```python
def loss_fn(model, batch):
    """Pure function for MLX value_and_grad compatibility."""
    input_ids = batch["input_ids"]
    labels = batch.get("labels", input_ids)

    # Forward pass - ensure model is callable, not parameter tree
    logits = model(input_ids)

    # Handle tuple return from model if necessary
    if isinstance(logits, tuple):
        logits = logits[0]

    # Compute cross-entropy loss
    return compute_cross_entropy_loss(logits, labels)

# Create the gradient function - this is the MLX way
loss_and_grad_fn = mx.value_and_grad(loss_fn)
```

#### Phase 2: Proper Parameter Filtering (2 hours)

Implement MLX tree operations for LoRA parameter isolation:

```python
def is_lora_parameter(path: str, param: mx.array) -> bool:
    """Identify LoRA parameters using path inspection."""
    return 'lora_a' in path or 'lora_b' in path

def filter_lora_gradients(grads: dict) -> dict:
    """Filter gradients to only LoRA parameters using tree operations."""
    from mlx.utils import tree_flatten, tree_unflatten

    flat_grads = tree_flatten(grads)
    lora_grads = {k: v for k, v in flat_grads if is_lora_parameter(k, v)}
    return tree_unflatten(lora_grads)

def training_step(model, batch, optimizer):
    """MLX-native training step following canonical pattern."""

    # Step 1: Compute loss and gradients (lazy)
    loss, all_grads = loss_and_grad_fn(model, batch)

    # Step 2: Filter to only LoRA gradients
    lora_grads = filter_lora_gradients(all_grads)

    # Step 3: Update optimizer (lazy)
    optimizer.update(model, lora_grads)

    # Step 4: Execute all lazy operations
    mx.eval(model.parameters(), optimizer.state)

    return {"loss": float(loss)}
```

#### Phase 3: Model Parameter Management (1 hour)

Implement proper MLX model freezing and LoRA injection:

```python
def setup_lora_model(model, lora_config):
    """Set up model for LoRA training using MLX patterns."""

    # Freeze base model using MLX method
    model.freeze()

    # Add LoRA layers (these will be unfrozen by default)
    add_lora_layers_to_model(model, lora_config)

    # Verify only LoRA parameters are trainable
    trainable_params = model.trainable_parameters()
    trainable_count = sum(p.size for p in tree_flatten(trainable_params)[1])

    return model, trainable_count
```

#### Phase 4: Integration and Testing (1-2 hours)

Update trainer.py to use new MLX-native patterns:

```python
class LoRATrainer:
    def __init__(self, model, lora_config, training_config):
        # Set up model with proper MLX LoRA pattern
        self.model, self.trainable_count = setup_lora_model(model, lora_config)

        # Create MLX-native loss and gradient function
        self.loss_and_grad_fn = mx.value_and_grad(self.loss_fn)

        # Initialize optimizer
        self.optimizer = optim.AdamW(learning_rate=training_config.learning_rate)

    def loss_fn(self, model, batch):
        """Pure loss function for MLX compatibility."""
        return self.compute_loss(model, batch)

    def training_step(self, batch):
        """Native MLX training step."""
        # Use the canonical three-step MLX pattern
        loss, all_grads = self.loss_and_grad_fn(self.model, batch)
        lora_grads = filter_lora_gradients(all_grads)
        self.optimizer.update(self.model, lora_grads)
        mx.eval(self.model.parameters(), self.optimizer.state)

        return {"loss": float(loss)}
```

### Option B: Reference Implementation Integration (FALLBACK)

**Timeline: 6-8 hours**
**Approach**: Replace current implementation with proven MLX LoRA code

#### Investigation Phase (2 hours)
- Research mlx-examples LoRA implementations
- Identify working reference code
- Analyze license compatibility

#### Integration Phase (4-6 hours)
- Replace src/finetune/training/lora.py with reference implementation
- Update trainer.py to use reference patterns
- Modify model loading to work with reference approach

## Implementation Checklist

### Critical Requirements

- [ ] **Remove all PyTorch-style parameter handling**
- [ ] **Implement pure loss function for mx.value_and_grad**
- [ ] **Use MLX tree operations for parameter filtering**
- [ ] **Follow canonical MLX training loop pattern**
- [ ] **Ensure proper model.freeze() and LoRA injection**

### Success Criteria

1. **Training Execution**: `make test-e2e-mlx-short` passes without MLX errors
2. **Parameter Updates**: LoRA parameters actually change during training
3. **Loss Convergence**: Training loss decreases properly (8.95 → <2.0)
4. **Model Quality**: Fine-tuned model generates coherent geography answers
5. **No Exceptions**: No "dict has no forward", "too many values", or MLX gradient errors

### Validation Tests

```bash
# Primary test - must pass
env FT_E2E_VERBOSE=1 FT_E2E_TRAINING=short make test-e2e-mlx-short

# Parameter update verification
python -c "
from src.finetune.training.trainer import LoRATrainer
# Verify trainable parameter count and updates
"

# Generation quality test
make test-generation-debug
```

## Risk Mitigation

### Backup Strategy
- Commit current broken implementation to git branch `broken-training`
- Implement new approach on feature branch `mlx-native-training`
- Maintain ability to rollback if new approach fails

### Testing Strategy
- Unit tests for each new component before integration
- Incremental testing: loss function → gradient computation → parameter updates
- Use TinyLlama model for fast iteration cycles

### Resource Management
- Focus on short training sessions (2-5 epochs) for validation
- Use small datasets (30-50 examples) for rapid testing
- Implement proper mx.eval() calls to prevent memory leaks

## Next Steps

1. **Immediate (Today)**:
   - Create git branch for rewrite
   - Implement pure loss function following LORA-TRAIN.md patterns
   - Test gradient computation in isolation

2. **Phase 1 (Next Session)**:
   - Complete MLX-native training step implementation
   - Update LoRATrainer class
   - Run first successful training test

3. **Validation (Following Session)**:
   - Verify model quality improvements
   - Test with different LoRA configurations
   - Update STATUS.md to reflect working system

## References

- **STATUS.md**: Current failure analysis and technical details
- **LORA-TRAIN.md**: Canonical MLX patterns and best practices
- **MLX Documentation**: value_and_grad, tree operations, training patterns
- **mlx-examples**: Reference implementations for validation

---

**Priority**: This rewrite is the critical blocker for all fine-tuning functionality. All other features depend on having a working training system.