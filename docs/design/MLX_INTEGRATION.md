# MLX Integration Plan for Phase 1

## Overview
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

### Unit Tests
- Test weight conversion accuracy
- Verify forward pass outputs match PyTorch
- Test gradient computation
- Memory usage tracking

### Integration Tests
- Load various model architectures (Llama, Mistral, GPT)
- Train for a few steps and verify loss decreases
- Save and load checkpoints
- Fallback mechanism testing

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

✓ Successfully load a small model (GPT-2) in MLX
✓ Run forward pass and get logits
✓ Compute loss and gradients
✓ Update weights with optimizer
✓ Save and load checkpoints
✓ Automatic fallback to PyTorch when MLX unavailable
✓ Basic memory monitoring and management

## Next Steps (Phase 2)
- LoRA/QLoRA implementation in MLX
- Advanced memory optimization techniques
- Multi-device training support
- Custom CUDA kernel equivalents in Metal