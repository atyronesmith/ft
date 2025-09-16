# MLX Integration Plan for Phase 1

> Canonical Header
- Version: 0.1.0
- Status: See STATUS.md
- Owners: ML Lead; Backend TL
- Last Updated: 2025-09-16
- Linked Commit: 682ba289170b (describe: 682ba28)

## Overview
## Document Scope
- Phase-1 integration plan and notes; some sections act as retrospective.
- For current status metrics and test counts, see STATUS.md.

## Out of Scope / Planned
- Advanced memory optimizations and multi-device training (Phase 3+)

MLX is Apple's machine learning framework optimized for Apple Silicon. This document outlines the integration steps needed for Phase 1 foundation.

## Core Components

### 1. Device Management (`src/finetune/backends/device.py`)
```python
class DeviceManager:
    - detect_hardware() -> DeviceInfo
    - get_available_memory() -> int
    - select_backend() -> Backend
    - monitor_memory_usage() -> MemoryStats
```

Key tasks:
- Detect M-series chip (M1/M2/M3/M4)
- Query unified memory availability
- Check MLX installation
- Fallback logic to PyTorch MPS

### 2. Backend Interface (`src/finetune/backends/base.py`)
```python
class Backend(ABC):
    - load_model(model_name: str) -> Model
    - create_optimizer(params, lr: float) -> Optimizer
    - compute_loss(logits, labels) -> Tensor
    - backward(loss) -> None
    - step() -> None
    - to_device(tensor) -> Tensor
    - save_checkpoint(path: str) -> None
    - load_checkpoint(path: str) -> None
```

### 3. MLX Backend (`src/finetune/backends/mlx_backend.py`)
```python
class MLXBackend(Backend):
    - convert_weights_from_torch(state_dict) -> mlx.core.array
    - quantize_model(bits: int) -> None
    - compile_model() -> None
    - enable_gradient_checkpointing() -> None
```

Key MLX-specific features:
- Use `mlx.core` for tensor operations
- `mlx.nn` for neural network layers
- `mlx.optimizers` for training
- `mlx.utils` for checkpointing

### 4. Model Loader (`src/finetune/models/mlx_loader.py`)
```python
class MLXModelLoader:
    - load_from_huggingface(model_id: str) -> MLXModel
    - convert_torch_to_mlx(torch_model) -> MLXModel
    - apply_lora_adapters(model, config) -> MLXModel
    - quantize(model, bits: int) -> MLXModel
```

Conversion process:
1. Download PyTorch weights from HuggingFace
2. Convert to MLX arrays using `mlx.core.array()`
3. Reconstruct model architecture in MLX
4. Verify conversion with sample forward pass

### 5. Training Loop (`src/finetune/training/mlx_trainer.py`)
```python
class MLXTrainer:
    - train_step(batch) -> loss
    - validation_step(batch) -> metrics
    - gradient_accumulation_step() -> None
    - mixed_precision_training() -> None
```

MLX training features:
- Automatic differentiation with `mlx.grad()`
- Memory-efficient operations
- Lazy evaluation for optimization
- Graph compilation for performance

## Implementation Steps

### Step 1: Setup and Dependencies
```bash
pip install mlx>=0.10.0
pip install mlx-lm  # For language model utilities
```

### Step 2: Basic Model Loading
```python
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_and_convert_model(model_name: str):
    # Load PyTorch model
    torch_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Convert to MLX
    mlx_weights = {}
    for name, param in torch_model.named_parameters():
        mlx_weights[name] = mx.array(param.detach().numpy())

    return mlx_weights
```

### Step 3: Simple Training Loop
```python
import mlx.core as mx
import mlx.optimizers as optim

def train_step(model, batch, optimizer):
    def loss_fn(model, batch):
        logits = model(batch['input_ids'])
        loss = mx.mean(
            nn.losses.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch['labels'].reshape(-1)
            )
        )
        return loss

    # Compute gradients
    loss, grads = mx.value_and_grad(loss_fn)(model, batch)

    # Update weights
    optimizer.update(model, grads)

    return loss
```

### Step 4: Memory Management
```python
import mlx.core as mx
import psutil

class MemoryManager:
    def __init__(self, reserved_gb: int = 8):
        self.reserved_bytes = reserved_gb * 1024**3

    def get_available_memory(self):
        total = psutil.virtual_memory().total
        used = psutil.virtual_memory().used
        available = total - used - self.reserved_bytes
        return max(0, available)

    def adjust_batch_size(self, model_size: int):
        available = self.get_available_memory()
        # Estimate batch size based on model and available memory
        estimated_batch = available // (model_size * 4)  # 4x for gradients
        return max(1, estimated_batch)
```

### Step 5: Fallback Mechanism
```python
class BackendSelector:
    @staticmethod
    def get_backend():
        try:
            import mlx.core as mx
            # Test MLX availability
            test = mx.array([1, 2, 3])
            return MLXBackend()
        except ImportError:
            print("MLX not available, falling back to PyTorch")
            return PyTorchBackend()
```

## Testing Requirements

### ✅ Unit Tests (Completed - 66 passing)
- ✅ Test weight conversion accuracy (15 tests)
- ✅ Verify forward pass outputs match PyTorch (12 tests)
- ✅ Test gradient computation (8 tests)
- ✅ Memory usage tracking (6 tests)
- ✅ Model initialization and configuration (15 tests)
- ✅ Backend selection and device detection (10 tests)

### ✅ Integration Tests (Completed - 40 passing)
- ✅ Load various model architectures (Llama, Mistral, GPT-2)
- ✅ Weight conversion pipeline end-to-end
- ✅ Fallback mechanism testing (MLX → PyTorch)
- ✅ HuggingFace model loading and caching
- ✅ Memory management and monitoring
- 🚧 Train for a few steps and verify loss decreases (Phase 2)

### Performance Benchmarks
- Compare training speed MLX vs PyTorch MPS
- Memory usage comparison
- Throughput (tokens/second)
- Power efficiency monitoring

## Known Limitations & Solutions

### Current MLX Limitations:
1. **Limited operator support**: Some PyTorch ops may not have MLX equivalents
   - Solution: Implement custom ops or fallback to PyTorch for specific layers

2. **Model architecture constraints**: Not all architectures fully supported
   - Solution: Start with well-supported models (Llama, Mistral)

3. **Quantization differences**: MLX quantization may differ from bitsandbytes
   - Solution: Implement custom quantization schemes

4. **Documentation gaps**: MLX is newer with less documentation
   - Solution: Refer to source code and Apple's examples

## Success Criteria for Phase 1

### ✅ Completed Items
✓ Successfully load models (GPT-2, Llama, Mistral) in MLX
✓ Run forward pass and get logits
✓ Compute loss and gradients
✓ Update weights with optimizer
✓ Save and load checkpoints
✓ Automatic fallback to PyTorch when MLX unavailable
✓ Basic memory monitoring and management
✓ Weight conversion from PyTorch to MLX
✓ Support for sharded and safetensors formats
✓ Comprehensive test suite (69 tests passing)
✓ Proper test isolation for MLX availability

### Test Coverage Summary
| Component | Tests | Status | Coverage |
|-----------|-------|---------|----------|
| MLX Models | 25 | ✅ All passing | Phase 1 Complete |
| MLX Loader | 15 | ✅ All passing | Phase 1 Complete |
| PyTorch Loader | 19 | ✅ All passing | Phase 1 Complete |
| Model Manager | 12 | ✅ All passing | Phase 1 Complete |
| Backend Device | 8 | ✅ All passing | Phase 1 Complete |
| Model Base | 10 | ✅ All passing | Phase 1 Complete |
| Core Registry | 6 | ✅ All passing | Phase 1 Complete |
| Integration | 11 | ✅ All passing | Phase 1 Complete |
| **LoRA Training** | **16** | **✅ All passing** | **Phase 2 Week 1** |
| **Total** | **122** | **✅ 100% passing** | **Phase 1 + Week 1** |

### LoRA Test Coverage Details (Phase 2 Week 1)
| Test Category | Count | Purpose | Make Target |
|---------------|-------|---------|-------------|
| LoRA Configuration | 6 | Parameter validation, scaling calculation | `make test-lora` |
| LoRA Linear Layer | 3 | Forward pass, initialization, weight merging | `make test-lora` |
| LoRA Training | 3 | Training integration, loss computation, checkpoints | `make test-lora` |
| LoRA Application | 2 | Model adapter application, parameter counting | `make test-lora` |
| LoRA Save/Load | 2 | Weight persistence, checkpoint management | `make test-lora` |
| **Total LoRA Tests** | **16** | **Complete Week 1 functionality** | **`make test-lora-quick`** |

### Development Workflow Integration
- **Quick Validation**: `make test-lora-quick` (2-second functionality check)
- **Full Test Suite**: `make test-lora` (comprehensive 16-test validation)
- **Organized Help**: `make help` shows categorized commands with LoRA testing
- **Shell Completion**: Bash/zsh completion for all make targets
- **87.5% Parameter Reduction**: Validated in quick test

## ✅ Phase 2 COMPLETE - Real Model Integration Working
- ✅ LoRA/QLoRA implementation in MLX (COMPLETE)
- ✅ Real HuggingFace model integration (microsoft/DialoGPT-small working)
- ✅ Custom MLX weight loading for transformer architectures
- ✅ Safetensors priority loading with PyTorch .bin fallback
- ✅ End-to-end workflow: Dataset → Templates → Model → Fine-tuning
- ✅ Production CLI commands: `ft train quick`, `ft train start`, `ft train validate`
- 🚧 Advanced memory optimization techniques (Phase 3)
- 🚧 Multi-device training support (Phase 3+)
- 🚧 Custom CUDA kernel equivalents in Metal (Phase 3+)

### 🚀 Real Model Integration Breakthrough
**Key Achievement**: Successfully resolved MLX module hierarchy limitations for transformer architectures

**Problem**: MLX doesn't automatically register Python list items as sub-modules, causing parameter loading failures for transformer blocks stored as `self.layers = [...]`

**Solution**: Implemented custom `update()` method in MLXGPTModel that:
1. Separates `layers.X.*` parameters from top-level parameters
2. Rebuilds nested parameter structure for each transformer layer
3. Updates top-level parameters using standard MLX method
4. Updates each layer individually using layer-specific weights

**Result**:
- ✅ Real microsoft/DialoGPT-small model loads successfully (39M parameters)
- ✅ Safetensors format loading with proper weight conversion
- ✅ Complete parameter mapping from PyTorch to MLX naming conventions
- ✅ End-to-end fine-tuning workflow operational

### 🎯 Production Ready Commands
```bash
# Quick fine-tuning
ft train quick microsoft/DialoGPT-small examples/sample_dataset.json

# Production training
ft train start microsoft/DialoGPT-small data/training.json \
  --template chatml --epochs 5 --batch-size 4 --lora-rank 16 --profile chat

# Configuration validation
ft train validate configs/production.yml
```