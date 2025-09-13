"""
FineTune: A modular fine-tuning application for Apple Silicon.
"""

__version__ = "0.1.0"
__author__ = "FineTune Team"

from finetune.core.config import Config
from finetune.core.registry import ModelRegistry

__all__ = ["Config", "ModelRegistry", "__version__"]
