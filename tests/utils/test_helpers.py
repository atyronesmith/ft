"""
Common test utilities and helpers for the FineTune test suite.

This module provides reusable functionality to reduce code duplication
across test files and ensure consistent test setup.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock

import pytest
from finetune.models.base import ModelConfig


class ModelConfigFactory:
    """Factory for creating test model configurations."""

    @staticmethod
    def create_small_config(model_type: str = "llama") -> ModelConfig:
        """Create a small model config for testing."""
        return ModelConfig(
            model_type=model_type,
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
        )

    @staticmethod
    def create_sample_config(model_type: str = "llama") -> ModelConfig:
        """Create a sample model config for testing."""
        return ModelConfig(
            model_type=model_type,
            vocab_size=32000,
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=32,
            intermediate_size=11008,
            max_position_embeddings=2048,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=False,
        )

    @staticmethod
    def create_gpt_config() -> ModelConfig:
        """Create a GPT-style config for testing."""
        return ModelConfig(
            model_type="gpt2",
            vocab_size=50257,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=1024,
            layer_norm_eps=1e-5,
        )


class MockFactory:
    """Factory for creating common mocks used in tests."""

    @staticmethod
    def create_mock_mlx():
        """Create a mock MLX module."""
        mlx = MagicMock()
        mlx.core = MagicMock()
        mlx.nn = MagicMock()
        mlx.core.array = lambda x: MagicMock(shape=getattr(x, "shape", (1,)))
        mlx.metal = MagicMock()
        mlx.metal.is_available.return_value = True
        return mlx

    @staticmethod
    def create_mock_torch():
        """Create a mock PyTorch module."""
        torch = MagicMock()
        torch.float16 = "float16"
        torch.float32 = "float32"
        torch.cuda.is_available.return_value = False
        torch.backends.mps.is_available.return_value = True
        torch.device.return_value = "mps"
        torch.Tensor = MagicMock
        return torch

    @staticmethod
    def create_mock_pytorch_weights():
        """Create mock PyTorch weights for testing."""
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
            import numpy as np
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


class DatasetFactory:
    """Factory for creating test datasets."""

    @staticmethod
    def create_sample_dataset(size: int = 3) -> List[Dict[str, str]]:
        """Create a sample instruction-following dataset."""
        base_data = [
            {
                "instruction": "Explain what machine learning is",
                "output": "Machine learning is a method of data analysis that automates analytical model building.",
            },
            {
                "instruction": "What is Python?",
                "output": "Python is a high-level programming language known for its simplicity and readability.",
            },
            {
                "instruction": "How do neural networks work?",
                "output": "Neural networks are computing systems inspired by biological neural networks.",
            },
            {
                "instruction": "What is the capital of France?",
                "output": "Paris",
            },
            {
                "instruction": "What is 2 + 2?",
                "output": "4",
            },
        ]
        return base_data[:size]

    @staticmethod
    def create_qa_dataset(size: int = 10) -> List[Dict[str, str]]:
        """Create a Q&A dataset with capitals."""
        capitals = [
            ("France", "Paris"),
            ("Germany", "Berlin"),
            ("Italy", "Rome"),
            ("Spain", "Madrid"),
            ("Portugal", "Lisbon"),
            ("United Kingdom", "London"),
            ("Netherlands", "Amsterdam"),
            ("Sweden", "Stockholm"),
            ("Norway", "Oslo"),
            ("Denmark", "Copenhagen"),
        ]

        data = []
        for i, (country, capital) in enumerate(capitals[:size]):
            data.append({
                "instruction": f"What is the capital of {country}?",
                "output": capital
            })
        return data

    @staticmethod
    def create_math_dataset(size: int = 5) -> List[Dict[str, str]]:
        """Create a simple math dataset."""
        problems = [
            ("What is 2 + 2?", "4"),
            ("What is 5 + 3?", "8"),
            ("What is 10 - 7?", "3"),
            ("What is 6 + 4?", "10"),
            ("What is 9 - 5?", "4"),
        ]

        return [
            {"instruction": q, "output": a}
            for q, a in problems[:size]
        ]


class FileHelper:
    """Helper for creating temporary files and directories."""

    @staticmethod
    def create_temp_json_file(data: Any, suffix: str = ".json") -> str:
        """Create a temporary JSON file with the given data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            json.dump(data, f)
            return f.name

    @staticmethod
    def create_temp_jsonl_file(data: List[Dict], suffix: str = ".jsonl") -> str:
        """Create a temporary JSONL file with the given data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            return f.name

    @staticmethod
    def create_mock_model_directory(temp_dir: Path, model_type: str = "llama") -> Path:
        """Create a mock HuggingFace model directory."""
        model_dir = temp_dir / "mock_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create config
        config = {
            "model_type": model_type,
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

        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)

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

    @staticmethod
    def create_mock_mlx_model_directory(temp_dir: Path, model_type: str = "llama") -> Path:
        """Create a mock MLX model directory."""
        model_dir = temp_dir / "mlx_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Create MLX config
        config = {
            "model_type": model_type,
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

        with open(model_dir / "mlx_config.json", "w") as f:
            json.dump(config, f)

        # Create mock MLX weight file
        (model_dir / "mlx_model.safetensors").touch()

        return model_dir


class TestEnvironment:
    """Helper for test environment setup and checks."""

    @staticmethod
    def mlx_available() -> bool:
        """Check if MLX is available."""
        try:
            import mlx
            return True
        except ImportError:
            return False

    @staticmethod
    def torch_available() -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False

    @staticmethod
    def transformers_available() -> bool:
        """Check if Transformers is available."""
        try:
            import transformers
            return True
        except ImportError:
            return False

    @staticmethod
    def skip_if_no_mlx():
        """Skip test if MLX is not available."""
        return pytest.mark.skipif(
            not TestEnvironment.mlx_available(),
            reason="MLX not available"
        )

    @staticmethod
    def skip_if_no_torch():
        """Skip test if PyTorch is not available."""
        return pytest.mark.skipif(
            not TestEnvironment.torch_available(),
            reason="PyTorch not available"
        )

    @staticmethod
    def verbose_print(message: str, enabled: bool = None):
        """Print verbose output if enabled."""
        if enabled is None:
            enabled = os.environ.get("FT_VERBOSE", "0") == "1"
        if enabled:
            print(f"[TEST] {message}")


class AssertionHelpers:
    """Helper methods for common test assertions."""

    @staticmethod
    def assert_model_config_valid(config: ModelConfig):
        """Assert that a model config is valid."""
        assert hasattr(config, "model_type")
        assert hasattr(config, "vocab_size")
        assert hasattr(config, "hidden_size")
        assert hasattr(config, "num_hidden_layers")
        assert hasattr(config, "num_attention_heads")
        assert config.vocab_size > 0
        assert config.hidden_size > 0
        assert config.num_hidden_layers > 0
        assert config.num_attention_heads > 0

    @staticmethod
    def assert_dataset_valid(dataset: List[Dict[str, str]], required_fields: List[str] = None):
        """Assert that a dataset is valid."""
        if required_fields is None:
            required_fields = ["instruction", "output"]

        assert isinstance(dataset, list)
        assert len(dataset) > 0

        for i, item in enumerate(dataset):
            assert isinstance(item, dict), f"Item {i} is not a dict"
            for field in required_fields:
                assert field in item, f"Item {i} missing field '{field}'"
                assert isinstance(item[field], str), f"Item {i} field '{field}' is not a string"
                assert len(item[field].strip()) > 0, f"Item {i} field '{field}' is empty"

    @staticmethod
    def assert_loss_convergence(losses: List[float], min_reduction: float = 0.05):
        """Assert that training loss converges."""
        assert len(losses) >= 2, "Need at least 2 loss values"

        loss_reduction = (losses[0] - losses[-1]) / losses[0]
        assert loss_reduction >= min_reduction, f"Loss reduction {loss_reduction:.2%} < {min_reduction:.2%}"

        # Check for NaN/inf values
        for i, loss in enumerate(losses):
            assert not (loss != loss), f"Loss at step {i} is NaN"  # NaN check
            assert loss != float('inf'), f"Loss at step {i} is infinite"
            assert loss != float('-inf'), f"Loss at step {i} is negative infinite"


# Commonly used fixtures that can be imported
@pytest.fixture
def small_model_config():
    """Fixture for small model config."""
    return ModelConfigFactory.create_small_config()


@pytest.fixture
def sample_model_config():
    """Fixture for sample model config."""
    return ModelConfigFactory.create_sample_config()


@pytest.fixture
def sample_dataset():
    """Fixture for sample dataset."""
    return DatasetFactory.create_sample_dataset()


@pytest.fixture
def qa_dataset():
    """Fixture for Q&A dataset."""
    return DatasetFactory.create_qa_dataset()


@pytest.fixture
def mock_mlx():
    """Fixture for mock MLX module."""
    return MockFactory.create_mock_mlx()


@pytest.fixture
def mock_torch():
    """Fixture for mock PyTorch module."""
    return MockFactory.create_mock_torch()


@pytest.fixture
def mock_pytorch_weights():
    """Fixture for mock PyTorch weights."""
    return MockFactory.create_mock_pytorch_weights()