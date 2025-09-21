"""
Pytest configuration and shared fixtures.
"""

import gc
import json
import tempfile
import warnings
from pathlib import Path

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
    from tests.utils import ModelConfigFactory

    config = ModelConfigFactory.create_sample_config()
    # Return as dict for backward compatibility
    return {
        "model_type": config.model_type,
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": getattr(config, "rms_norm_eps", 1e-6),
        "rope_theta": getattr(config, "rope_theta", 10000.0),
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", False),
    }


@pytest.fixture
def small_model_config():
    """Create a small model config for testing."""
    from tests.utils import ModelConfigFactory

    config = ModelConfigFactory.create_small_config("gpt2")
    # Return as dict for backward compatibility
    return {
        "model_type": config.model_type,
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "layer_norm_eps": getattr(config, "layer_norm_eps", 1e-5),
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
    from tests.utils import MockFactory

    return MockFactory.create_mock_pytorch_weights()


@pytest.fixture
def mock_mlx():
    """Mock MLX module."""
    from tests.utils import MockFactory

    return MockFactory.create_mock_mlx()


@pytest.fixture
def mock_torch():
    """Mock PyTorch module."""
    from tests.utils import MockFactory

    return MockFactory.create_mock_torch()


# Skip markers for optional dependencies
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "requires_mlx: mark test as requiring MLX")
    config.addinivalue_line("markers", "requires_torch: mark test as requiring PyTorch")
    config.addinivalue_line("markers", "requires_transformers: mark test as requiring Transformers")
    config.addinivalue_line("markers", "slow: mark test as slow")


def mlx_available():
    """Check if MLX is available."""
    from tests.utils import TestEnvironment

    return TestEnvironment.mlx_available()


def torch_available():
    """Check if PyTorch is available."""
    from tests.utils import TestEnvironment

    return TestEnvironment.torch_available()


def transformers_available():
    """Check if Transformers is available."""
    from tests.utils import TestEnvironment

    return TestEnvironment.transformers_available()


# Auto-skip tests based on requirements
pytest_plugins = []

requires_mlx = pytest.mark.skipif(not mlx_available(), reason="MLX not available")

requires_torch = pytest.mark.skipif(not torch_available(), reason="PyTorch not available")

requires_transformers = pytest.mark.skipif(
    not transformers_available(), reason="Transformers not available"
)
