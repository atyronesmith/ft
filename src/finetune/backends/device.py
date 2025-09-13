"""
Device detection and backend selection for optimal performance.
"""

import platform
import subprocess
from enum import Enum
from typing import Any

import psutil


class DeviceType(Enum):
    """Available device types."""

    APPLE_SILICON = "apple_silicon"
    CUDA = "cuda"
    CPU = "cpu"


class DeviceInfo:
    """Information about the current device."""

    def __init__(self):
        self.device_type = self._detect_device_type()
        self.chip_name = self._get_chip_name()
        self.total_memory = psutil.virtual_memory().total
        self.available_memory = psutil.virtual_memory().available
        self.cpu_count = psutil.cpu_count()

    def _detect_device_type(self) -> DeviceType:
        """Detect the type of device."""
        system = platform.system()

        if system == "Darwin":  # macOS
            processor = platform.processor()
            # Check if it's Apple Silicon
            if "arm" in processor.lower() or self._is_apple_silicon():
                return DeviceType.APPLE_SILICON

        # Check for CUDA
        try:
            import torch

            if torch.cuda.is_available():
                return DeviceType.CUDA
        except ImportError:
            pass

        return DeviceType.CPU

    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon."""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.optional.arm64"], capture_output=True, text=True
            )
            return result.returncode == 0 and result.stdout.strip() == "1"
        except:
            return False

    def _get_chip_name(self) -> str | None:
        """Get the chip name (e.g., Apple M1, M2, etc.)."""
        if self.device_type != DeviceType.APPLE_SILICON:
            return None

        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            if result.returncode == 0:
                brand = result.stdout.strip()
                # Extract M1, M2, M3, M4, etc.
                if "Apple" in brand:
                    return brand
        except:
            pass

        return "Apple Silicon"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "device_type": self.device_type.value,
            "chip_name": self.chip_name,
            "total_memory_gb": self.total_memory / (1024**3),
            "available_memory_gb": self.available_memory / (1024**3),
            "cpu_count": self.cpu_count,
        }


class DeviceManager:
    """Manages device detection and backend selection."""

    def __init__(self):
        self.device_info = DeviceInfo()
        self._backend = None

    def get_optimal_backend(self):
        """Get the optimal backend for the current device."""
        if self._backend is not None:
            return self._backend

        device_type = self.device_info.device_type

        if device_type == DeviceType.APPLE_SILICON:
            # Try MLX first
            try:
                from finetune.backends.mlx_backend import MLXBackend

                self._backend = MLXBackend()
                print(f"Using MLX backend on {self.device_info.chip_name}")
                return self._backend
            except ImportError:
                print("MLX not available, falling back to PyTorch")

        # Fallback to PyTorch
        try:
            from finetune.backends.torch_backend import PyTorchBackend

            self._backend = PyTorchBackend(device_type)
            print(f"Using PyTorch backend on {device_type.value}")
            return self._backend
        except ImportError:
            raise RuntimeError("No suitable backend found. Please install MLX or PyTorch.")

    def get_memory_info(self) -> dict[str, float]:
        """Get current memory information."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_gb": mem.used / (1024**3),
            "percent": mem.percent,
        }

    def estimate_batch_size(self, model_size_gb: float, sequence_length: int) -> int:
        """Estimate optimal batch size based on available memory."""
        # Reserve some memory for system and other processes
        reserved_gb = 8
        available_gb = self.device_info.available_memory / (1024**3) - reserved_gb

        if available_gb <= 0:
            return 1

        # Rough estimation: model takes up memory, plus activations and gradients
        # Assume 4x model size for full training memory
        memory_per_sample_gb = (model_size_gb * 4) / 32  # Rough estimate

        # Adjust for sequence length
        memory_per_sample_gb *= sequence_length / 2048  # Normalized to 2048 tokens

        batch_size = int(available_gb / memory_per_sample_gb)
        return max(1, min(batch_size, 32))  # Cap at 32 for stability

    def monitor_memory(self) -> None:
        """Monitor and log memory usage."""
        import time

        from loguru import logger

        while True:
            mem_info = self.get_memory_info()
            logger.debug(
                f"Memory: {mem_info['used_gb']:.1f}/{mem_info['total_gb']:.1f} GB "
                f"({mem_info['percent']:.1f}%)"
            )
            time.sleep(10)  # Check every 10 seconds


# Global device manager instance
device_manager = DeviceManager()
