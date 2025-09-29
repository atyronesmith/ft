#!/usr/bin/env python3
"""
Test script to validate MLX data loader compatibility.

This script tests our MLX data loader against the original MLX examples data
to ensure we can load and process the data exactly like the MLX examples.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from finetune.data.mlx_loader import load_mlx_datasets, iterate_batches_mlx, compute_loss_mlx
from finetune.models.manager import ModelManager


def test_mlx_data_loading():
    """Test loading MLX example data."""
    print("🔍 Testing MLX data loader...")

    # Test loading datasets
    try:
        train, valid, test = load_mlx_datasets("mlx_example_data")
        print(f"✅ Successfully loaded datasets:")
        print(f"   Train: {len(train)} examples")
        print(f"   Valid: {len(valid)} examples")
        print(f"   Test: {len(test)} examples")
    except Exception as e:
        print(f"❌ Failed to load datasets: {e}")
        return False

    # Test data format
    print(f"\n📝 Sample data from training set:")
    for i in range(min(3, len(train))):
        text = train[i]
        print(f"   Example {i}: {text[:100]}...")

    return True


def test_mlx_batch_iteration():
    """Test batch iteration exactly like MLX examples."""
    print("\n🔄 Testing MLX batch iteration...")

    try:
        # Load datasets
        train, valid, test = load_mlx_datasets("mlx_example_data")

        # Load tokenizer
        print("   Loading tokenizer...")
        manager = ModelManager()
        _, tokenizer, _ = manager.load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        # Test batch iteration
        print("   Testing batch iteration...")
        batch_size = 4
        batch_count = 0

        for inputs, targets, lengths in iterate_batches_mlx(train, tokenizer, batch_size, train=False):
            print(f"   Batch {batch_count}: inputs {inputs.shape}, targets {targets.shape}, lengths {lengths.shape}")

            # Verify shapes
            assert inputs.shape[0] == batch_size, f"Expected batch size {batch_size}, got {inputs.shape[0]}"
            assert targets.shape[0] == batch_size, f"Expected batch size {batch_size}, got {targets.shape[0]}"
            assert lengths.shape[0] == batch_size, f"Expected batch size {batch_size}, got {lengths.shape[0]}"

            # Verify that targets are shifted by 1 from inputs
            for i in range(batch_size):
                seq_len = int(lengths[i].item())
                if seq_len > 1:
                    # Check that targets[i, 0] == inputs[i, 1] (shifted by 1)
                    assert targets[i, 0] == inputs[i, 1], f"Targets not properly shifted for sequence {i}"

            batch_count += 1
            if batch_count >= 3:  # Just test a few batches
                break

        print(f"   ✅ Successfully processed {batch_count} batches")
        return True

    except Exception as e:
        print(f"   ❌ Batch iteration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mlx_loss_computation():
    """Test loss computation exactly like MLX examples."""
    print("\n📊 Testing MLX loss computation...")

    try:
        # Load datasets and model
        train, valid, test = load_mlx_datasets("mlx_example_data")

        print("   Loading model...")
        manager = ModelManager()
        model, tokenizer, _ = manager.load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        print("   Computing loss on sample batch...")
        batch_size = 2

        # Get one batch
        for inputs, targets, lengths in iterate_batches_mlx(train, tokenizer, batch_size, train=False):
            # Compute loss using MLX examples approach
            loss, ntoks = compute_loss_mlx(model, inputs, targets, lengths)

            print(f"   ✅ Loss computation successful:")
            print(f"      Loss: {float(loss.item()):.4f}")
            print(f"      Tokens: {int(ntoks.item())}")

            # Verify loss is reasonable (should be positive, finite)
            loss_val = float(loss.item())
            assert loss_val > 0, f"Loss should be positive, got {loss_val}"
            assert not (loss_val != loss_val), f"Loss should not be NaN"  # NaN check
            assert loss_val != float('inf'), f"Loss should not be infinite"

            break  # Only test one batch

        return True

    except Exception as e:
        print(f"   ❌ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_our_format():
    """Compare MLX format with our current chat format."""
    print("\n🔍 Comparing MLX format with our chat format...")

    try:
        # Load MLX data
        train, valid, test = load_mlx_datasets("mlx_example_data")

        print("   MLX format examples:")
        for i in range(min(2, len(train))):
            text = train[i]
            print(f"      Example {i}: {text}")

        # Load our format
        from finetune.data.loaders import JSONLoader
        our_data_path = Path("training_data/train.json")

        if our_data_path.exists():
            loader = JSONLoader()
            our_data = loader.load(our_data_path)

            print(f"\n   Our format examples:")
            for i in range(min(2, len(our_data))):
                conversation = our_data[i]
                print(f"      Example {i}: {conversation}")

            print(f"\n   📊 Data comparison:")
            print(f"      MLX train: {len(train)} examples")
            print(f"      Our train: {len(our_data)} examples")

        else:
            print("   Our training data not found - run 'make generate-data' first")

        return True

    except Exception as e:
        print(f"   ❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 MLX Data Loader Compatibility Test")
    print("=" * 50)

    success = True

    # Test data loading
    success &= test_mlx_data_loading()

    # Test batch iteration
    success &= test_mlx_batch_iteration()

    # Test loss computation
    success &= test_mlx_loss_computation()

    # Compare formats
    success &= compare_with_our_format()

    print("\n" + "=" * 50)
    if success:
        print("✅ All MLX data loader tests passed!")
    else:
        print("❌ Some MLX data loader tests failed!")

    print("🎯 MLX data loader is ready for training comparison")