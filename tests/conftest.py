"""
Pytest configuration and shared fixtures.
"""

import gc
import json
import tempfile
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def suppress_resource_warnings():
    """Suppress ResourceWarning for known Python 3.13 mock issues."""
    # Force collection before test
    gc.collect()

    # Close any lingering SQLite connections
    import sqlite3

    sqlite3.enable_callback_tracebacks(False)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ResourceWarning)
        yield

    # Force garbage collection after each test
    gc.collect()
    # Additional collection to ensure cleanup
    gc.collect()
    # Final collection
    gc.collect()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_model_config():
    """Create a sample model configuration."""
    return {
        "model_type": "llama",
        "vocab_size": 32000,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "intermediate_size": 11008,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
    }


@pytest.fixture
def small_model_config():
    """Create a small model config for testing."""
    return {
        "model_type": "gpt2",
        "vocab_size": 1000,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "max_position_embeddings": 512,
        "layer_norm_eps": 1e-5,
    }


@pytest.fixture
def mock_huggingface_model(temp_dir, sample_model_config):
    """Create a mock HuggingFace model directory."""
    model_dir = temp_dir / "mock_model"
    model_dir.mkdir(parents=True)

    # Save config
    with open(model_dir / "config.json", "w") as f:
        json.dump(sample_model_config, f)

    # Create mock weight files
    (model_dir / "pytorch_model.bin").touch()

    # Create tokenizer files
    tokenizer_config = {
        "tokenizer_class": "LlamaTokenizer",
        "model_max_length": 2048,
    }
    with open(model_dir / "tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f)

    return model_dir


@pytest.fixture
def mock_mlx_model(temp_dir, sample_model_config):
    """Create a mock MLX model directory."""
    model_dir = temp_dir / "mlx_model"
    model_dir.mkdir(parents=True)

    # Save config in MLX format
    with open(model_dir / "mlx_config.json", "w") as f:
        json.dump(sample_model_config, f)

    # Create mock MLX weight file
    (model_dir / "mlx_model.safetensors").touch()

    return model_dir


@pytest.fixture
def mock_pytorch_weights():
    """Create mock PyTorch weights."""
    try:
        import torch

        return {
            "model.embed_tokens.weight": torch.randn(1000, 128),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(128, 128),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(128, 128),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(128, 128),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(512, 128),
            "model.layers.0.mlp.up_proj.weight": torch.randn(512, 128),
            "model.layers.0.mlp.down_proj.weight": torch.randn(128, 512),
            "model.norm.weight": torch.ones(128),
            "lm_head.weight": torch.randn(1000, 128),
        }
    except ImportError:
        # Return numpy arrays if PyTorch not available
        return {
            "model.embed_tokens.weight": np.random.randn(1000, 128),
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(128, 128),
            "model.layers.0.self_attn.k_proj.weight": np.random.randn(128, 128),
            "model.layers.0.self_attn.v_proj.weight": np.random.randn(128, 128),
            "model.layers.0.self_attn.o_proj.weight": np.random.randn(128, 128),
            "model.layers.0.mlp.gate_proj.weight": np.random.randn(512, 128),
            "model.layers.0.mlp.up_proj.weight": np.random.randn(512, 128),
            "model.layers.0.mlp.down_proj.weight": np.random.randn(128, 512),
            "model.norm.weight": np.ones(128),
            "lm_head.weight": np.random.randn(1000, 128),
        }


@pytest.fixture
def mock_mlx():
    """Mock MLX module."""
    mlx = MagicMock()
    mlx.core = MagicMock()
    mlx.nn = MagicMock()
    mlx.core.array = lambda x: MagicMock(shape=getattr(x, "shape", (1,)))
    return mlx


@pytest.fixture
def mock_torch():
    """Mock PyTorch module."""
    torch = MagicMock()
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.cuda.is_available.return_value = False
    torch.backends.mps.is_available.return_value = True
    torch.device.return_value = "mps"
    return torch


# Skip markers for optional dependencies
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "requires_mlx: mark test as requiring MLX")
    config.addinivalue_line("markers", "requires_torch: mark test as requiring PyTorch")
    config.addinivalue_line("markers", "requires_transformers: mark test as requiring Transformers")
    config.addinivalue_line("markers", "slow: mark test as slow")


def mlx_available():
    """Check if MLX is available."""
    try:
        import mlx

        return True
    except ImportError:
        return False


def torch_available():
    """Check if PyTorch is available."""
    try:
        import torch

        return True
    except ImportError:
        return False


def transformers_available():
    """Check if Transformers is available."""
    try:
        import transformers

        return True
    except ImportError:
        return False


# Auto-skip tests based on requirements
pytest_plugins = []

requires_mlx = pytest.mark.skipif(not mlx_available(), reason="MLX not available")

requires_torch = pytest.mark.skipif(not torch_available(), reason="PyTorch not available")

requires_transformers = pytest.mark.skipif(
    not transformers_available(), reason="Transformers not available"
)
