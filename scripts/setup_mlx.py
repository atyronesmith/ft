#!/usr/bin/env python3
"""
Setup script for MLX on Apple Silicon.

This script helps set up MLX and verifies the installation works correctly.
"""

import subprocess
import sys
import platform
import importlib.util


def check_apple_silicon():
    """Check if we're running on Apple Silicon."""
    return platform.machine() == "arm64" and platform.system() == "Darwin"


def check_mlx_installed():
    """Check if MLX is already installed."""
    spec = importlib.util.find_spec("mlx")
    return spec is not None


def install_mlx():
    """Install MLX using pip."""
    print("Installing MLX...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "mlx"],
                      check=True, capture_output=True, text=True)
        print("✅ MLX installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install MLX: {e}")
        print(f"stderr: {e.stderr}")
        return False


def verify_mlx():
    """Verify MLX installation works."""
    print("Verifying MLX installation...")
    try:
        import mlx.core as mx
        # Test basic functionality
        x = mx.array([1, 2, 3])
        y = mx.array([4, 5, 6])
        z = x + y
        print(f"✅ MLX verification successful: {x} + {y} = {z}")
        print(f"✅ MLX version: {mx.__version__ if hasattr(mx, '__version__') else 'unknown'}")
        return True
    except Exception as e:
        print(f"❌ MLX verification failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 MLX Setup for FineTune")
    print("=" * 40)

    # Check if we're on Apple Silicon
    if not check_apple_silicon():
        print("⚠️  Warning: MLX is optimized for Apple Silicon (M1/M2/M3).")
        print("   You are running on:", platform.machine(), platform.system())
        print("   MLX may not work optimally on this platform.")

        response = input("Continue anyway? (y/N): ").lower().strip()
        if response != 'y':
            print("Exiting...")
            return False
    else:
        print("✅ Apple Silicon detected - MLX is supported!")

    # Check if already installed
    if check_mlx_installed():
        print("✅ MLX is already installed!")
        return verify_mlx()

    # Install MLX
    if not install_mlx():
        return False

    # Verify installation
    return verify_mlx()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)