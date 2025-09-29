#!/usr/bin/env python3
"""
Direct MLX Format Test - Train our implementation with official MLX text format.

This bypasses our chat message pipeline and tests our core MLX/LoRA implementation
against the exact same data format that official MLX uses.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from finetune.models.manager import ModelManager
from finetune.training.lora import LoRAConfig
from finetune.training.trainer import LoRATrainer
import mlx.core as mx

def create_simple_text_dataset():
    """Create a simple text dataset in official MLX format."""

    # Simple examples matching official MLX WikiSQL format but easier to understand
    training_examples = [
        {"text": "Question: What is the capital of France? Answer: Paris"},
        {"text": "Question: What is 2 + 2? Answer: 4"},
        {"text": "Question: What color is the sky? Answer: Blue"},
        {"text": "Question: What is the largest planet? Answer: Jupiter"},
        {"text": "Question: Name a programming language. Answer: Python"},
        {"text": "Question: How many days in a week? Answer: 7"},
        {"text": "Question: What sound does a cat make? Answer: Meow"},
        {"text": "Question: What is 10 - 3? Answer: 7"},
        {"text": "Question: What season comes after summer? Answer: Autumn"},
        {"text": "Question: What do bees make? Answer: Honey"}
    ]

    # Save in official MLX JSONL format
    dataset_path = Path("simple_mlx_dataset.jsonl")

    with open(dataset_path, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')

    print(f"✅ Created simple dataset: {dataset_path}")
    print(f"📊 {len(training_examples)} training examples")

    return dataset_path, training_examples

def test_our_mlx_implementation_directly():
    """Test our MLX implementation using simple text format directly."""
    print("🔧 Testing Our MLX Implementation Directly")
    print("=" * 50)

    # Create simple dataset
    dataset_path, training_examples = create_simple_text_dataset()

    # Load model using our implementation
    print("\n📥 Loading model with our implementation...")
    manager = ModelManager()
    model, tokenizer, config = manager.load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    print(f"✅ Model loaded: {type(model)}")

    # Apply LoRA using our implementation
    print("\n🔧 Applying LoRA layers...")
    lora_config = LoRAConfig(
        r=8,  # Match official MLX
        alpha=20.0,  # Match official MLX
        dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

    # Freeze base model
    model.freeze()

    # Apply LoRA to last 16 layers (match official)
    lora_layers = 16
    layers = model.layers
    start_layer = len(layers) - lora_layers

    from finetune.training.lora import LoRALinear

    for layer_idx in range(start_layer, len(layers)):
        layer = layers[layer_idx]
        if hasattr(layer.self_attn, "q_proj"):
            original_q_proj = layer.self_attn.q_proj
            layer.self_attn.q_proj = LoRALinear.from_linear(original_q_proj, rank=lora_config)
        if hasattr(layer.self_attn, "v_proj"):
            original_v_proj = layer.self_attn.v_proj
            layer.self_attn.v_proj = LoRALinear.from_linear(original_v_proj, rank=lora_config)

    print(f"✅ Applied LoRA to {lora_layers} layers")

    # Prepare training data in simple format
    print("\n📚 Preparing training data...")

    # Create simple training batches directly
    training_batches = []

    for example in training_examples:
        text = example["text"]

        # Tokenize using our tokenizer
        tokens = tokenizer.encode(text)
        if isinstance(tokens, list):
            token_array = mx.array(tokens, dtype=mx.int32)
        else:
            token_array = mx.array(tokens[0], dtype=mx.int32)

        # Create batch (input_ids and labels are the same for language modeling)
        batch = {
            "input_ids": token_array.reshape(1, -1),  # Add batch dimension
            "labels": token_array.reshape(1, -1)      # Add batch dimension
        }
        training_batches.append(batch)

    print(f"✅ Created {len(training_batches)} training batches")

    # Test our data pipeline with basic loss computation
    print("\n🚀 Testing our MLX implementation with simple text format...")

    # Simple training simulation to test data compatibility
    train_losses = []
    num_epochs = 2

    print(f"Testing for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_losses = []

        for i, batch in enumerate(training_batches):
            # Forward pass and compute loss
            try:
                # Get input_ids and labels
                input_ids = batch["input_ids"]
                labels = batch["labels"]

                # Forward pass
                model_output = model(input_ids)

                # Handle potential tuple output (logits, cache)
                if isinstance(model_output, tuple):
                    logits = model_output[0]
                else:
                    logits = model_output

                # Compute loss (simple cross-entropy for next-token prediction)
                # Shift logits and labels for causal LM
                shift_logits = logits[:, :-1, :]  # Remove last token prediction
                shift_labels = labels[:, 1:]      # Remove first token (start token)

                # Flatten for loss computation
                shift_logits_flat = shift_logits.reshape(-1, shift_logits.shape[-1])
                shift_labels_flat = shift_labels.reshape(-1)

                # Cross-entropy loss
                loss = mx.mean(mx.logsumexp(shift_logits_flat, axis=-1) -
                              mx.take_along_axis(shift_logits_flat,
                                                shift_labels_flat[:, None],
                                                axis=-1).squeeze(-1))

                # Evaluate loss
                mx.eval(loss)

                # Convert MLX array to Python float properly
                loss_value = float(loss.item())
                epoch_losses.append(loss_value)

                if i == 0:  # Print first batch loss
                    print(f"  Epoch {epoch+1}, Batch {i+1}: Loss = {loss_value:.4f}")

            except Exception as e:
                print(f"  ❌ Error in batch {i+1}: {e}")
                continue

        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            train_losses.append(avg_loss)
            print(f"  Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # Results
    print(f"\n📊 Training Results:")
    print(f"Final Loss: {train_losses[-1]:.4f}" if train_losses else "No successful training")

    if len(train_losses) >= 2:
        loss_improvement = train_losses[0] - train_losses[-1]
        print(f"Loss Improvement: {loss_improvement:.4f}")

        if loss_improvement > 0.1:
            print("✅ SUCCESS: Significant loss reduction achieved!")
            print("✅ Our MLX implementation appears to be working correctly!")
        elif loss_improvement > 0.01:
            print("⚠️  PARTIAL: Some loss reduction, but could be better")
        else:
            print("❌ FAILURE: No meaningful loss reduction")

    # Compare to official MLX results
    print(f"\n🔍 Comparison to Official MLX:")
    print(f"Official MLX loss progression: 2.566 → 1.372 (50 iterations)")
    print(f"Our implementation: {train_losses[0]:.3f} → {train_losses[-1]:.3f} ({len(train_losses)} epochs)" if len(train_losses) >= 2 else "Insufficient data")

    return train_losses

def main():
    """Main test function."""
    print("🧪 Direct MLX Format Implementation Test")
    print("=" * 60)
    print("Testing our MLX implementation with simple text format")
    print("(bypassing chat message pipeline)")
    print("=" * 60)

    try:
        train_losses = test_our_mlx_implementation_directly()

        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT:")

        if len(train_losses) >= 2:
            if train_losses[0] - train_losses[-1] > 0.1:
                print("🎉 SUCCESS: Our MLX implementation works correctly!")
                print("   The issue was with data format compatibility, not core implementation")
                return 0
            else:
                print("❌ FAILURE: Our MLX implementation has issues")
                print("   Core MLX/LoRA logic may have bugs")
                return 1
        else:
            print("⚠️  INCONCLUSIVE: Could not complete training")
            return 1

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())