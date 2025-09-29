#!/usr/bin/env python3
"""
Debug LoRA fine-tuned generation issues.

This script investigates:
1. LoRA loading issues
2. Early generation stopping
3. Template compatibility problems
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import mlx.core as mx
from finetune.models.manager import ModelManager
from finetune.training.lora import LoRALinear, load_lora_weights
from finetune.inference.generation import MLXTextGenerator

def find_most_recent_training_run():
    """Find the most recent training run."""
    training_dir = Path("training")
    if not training_dir.exists():
        return None

    training_runs = []
    for item in training_dir.iterdir():
        if item.is_dir():
            final_model_path = item / "final_model"
            if final_model_path.exists() and (final_model_path / "lora_weights.npz").exists():
                mtime = final_model_path.stat().st_mtime
                training_runs.append((mtime, final_model_path))

    if not training_runs:
        return None

    training_runs.sort(key=lambda x: x[0], reverse=True)
    return training_runs[0][1]

def find_all_training_runs():
    """Find all available training runs."""
    training_dir = Path("training")
    if not training_dir.exists():
        return []

    training_runs = []
    for item in training_dir.iterdir():
        if item.is_dir():
            final_model_path = item / "final_model"
            if final_model_path.exists() and (final_model_path / "lora_weights.npz").exists():
                mtime = final_model_path.stat().st_mtime
                training_runs.append((mtime, final_model_path, item.name))

    training_runs.sort(key=lambda x: x[0], reverse=True)
    return training_runs

def debug_1_lora_loading():
    """Debug LoRA loading issues."""
    print("🔍 DEBUG 1: LoRA Loading Issues")
    print("=" * 50)

    # Find training run
    most_recent = find_most_recent_training_run()
    if not most_recent:
        print("❌ No training runs found")
        return False

    print(f"📁 Using training run: {most_recent.parent.name}")

    # Load base model
    print("\n📥 Loading base model...")
    manager = ModelManager()
    model, tokenizer, config = manager.load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    print(f"✅ Base model loaded: {type(model)}")

    # Check model before LoRA
    print(f"\n🔍 Model before LoRA application:")
    layer_6_q_proj = model.layers[6].self_attn.q_proj
    print(f"   Layer 6 q_proj type: {type(layer_6_q_proj)}")
    print(f"   Layer 6 q_proj weight shape: {layer_6_q_proj.weight.shape}")

    # Apply LoRA manually (same as test script)
    print("\n🔧 Applying LoRA layers...")
    from finetune.training.lora import LoRAConfig

    lora_config = LoRAConfig(r=8, alpha=16, dropout=0.1, target_modules=["q_proj", "v_proj"])

    # Freeze model
    model.freeze()

    # Apply LoRA to last 16 layers
    lora_layers = 16
    layers = model.layers
    start_layer = len(layers) - lora_layers

    print(f"   Applying LoRA to layers {start_layer} through {len(layers)-1}")

    for layer_idx in range(start_layer, len(layers)):
        layer = layers[layer_idx]

        if hasattr(layer.self_attn, "q_proj"):
            original_q_proj = layer.self_attn.q_proj
            layer.self_attn.q_proj = LoRALinear.from_linear(original_q_proj, rank=lora_config)
            print(f"   ✅ Applied LoRA to layer {layer_idx} q_proj")

        if hasattr(layer.self_attn, "v_proj"):
            original_v_proj = layer.self_attn.v_proj
            layer.self_attn.v_proj = LoRALinear.from_linear(original_v_proj, rank=lora_config)
            print(f"   ✅ Applied LoRA to layer {layer_idx} v_proj")

    # Check model after LoRA
    print(f"\n🔍 Model after LoRA application:")
    layer_6_q_proj_lora = model.layers[6].self_attn.q_proj
    print(f"   Layer 6 q_proj type: {type(layer_6_q_proj_lora)}")
    print(f"   Is LoRALinear: {isinstance(layer_6_q_proj_lora, LoRALinear)}")

    if isinstance(layer_6_q_proj_lora, LoRALinear):
        try:
            print(f"   LoRA rank: {layer_6_q_proj_lora.r}")
            print(f"   LoRA scale: {layer_6_q_proj_lora.scale}")
            print(f"   LoRA A shape: {layer_6_q_proj_lora.lora_a.shape}")
            print(f"   LoRA B shape: {layer_6_q_proj_lora.lora_b.shape}")
        except AttributeError as e:
            print(f"   ⚠️  Could not access LoRA attributes: {e}")
            print(f"   Available attributes: {[attr for attr in dir(layer_6_q_proj_lora) if not attr.startswith('_')]}")

    # Load LoRA weights
    print(f"\n📥 Loading LoRA weights from: {most_recent / 'lora_weights.npz'}")
    try:
        load_lora_weights(model, most_recent / "lora_weights.npz")
        print("✅ LoRA weights loaded successfully")
    except Exception as e:
        print(f"❌ LoRA weights loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check LoRA weights after loading
    print(f"\n🔍 LoRA weights after loading:")
    layer_6_q_proj_loaded = model.layers[6].self_attn.q_proj
    if isinstance(layer_6_q_proj_loaded, LoRALinear):
        try:
            print(f"   LoRA A values range: {mx.min(layer_6_q_proj_loaded.lora_a).item():.6f} to {mx.max(layer_6_q_proj_loaded.lora_a).item():.6f}")
            print(f"   LoRA B values range: {mx.min(layer_6_q_proj_loaded.lora_b).item():.6f} to {mx.max(layer_6_q_proj_loaded.lora_b).item():.6f}")

            # Check if weights are all zeros (indicating loading failure)
            a_sum = mx.sum(mx.abs(layer_6_q_proj_loaded.lora_a)).item()
            b_sum = mx.sum(mx.abs(layer_6_q_proj_loaded.lora_b)).item()
            print(f"   LoRA A sum of absolute values: {a_sum}")
            print(f"   LoRA B sum of absolute values: {b_sum}")

            if a_sum < 1e-6 and b_sum < 1e-6:
                print("   ⚠️  WARNING: LoRA weights appear to be all zeros!")
            else:
                print("   ✅ LoRA weights have non-zero values")
        except AttributeError as e:
            print(f"   ⚠️  Could not access LoRA weights: {e}")

    return model, tokenizer

def debug_2_early_stopping(model, tokenizer):
    """Debug early generation stopping."""
    print("\n🔍 DEBUG 2: Early Generation Stopping")
    print("=" * 50)

    test_prompts = [
        "What is the capital of France?",
        "Question: What is the capital of France?\nAnswer:",
        "### Instruction:\nWhat is the capital of France?\n\n### Response:\n"
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🎯 Test {i}: {repr(prompt[:50])}...")

        # Create generator
        generator = MLXTextGenerator(model, tokenizer)

        # Test with debug generation to see what happens
        try:
            # Encode prompt
            input_ids = tokenizer.encode(prompt)
            if isinstance(input_ids, list):
                tokens = mx.array(input_ids).astype(mx.int32)
            else:
                tokens = mx.array(input_ids[0]).astype(mx.int32)

            print(f"   📝 Input tokens: {len(tokens)} tokens")
            print(f"   📝 Input token IDs: {tokens[:10].tolist()}...")

            # Check EOS token
            print(f"   🔚 EOS token ID: {tokenizer.eos_token_id}")
            print(f"   🔚 BOS token ID: {getattr(tokenizer, 'bos_token_id', 'None')}")

            # Manual generation step by step
            cache = [None] * len(model.layers)
            y = tokens
            generated = []

            print(f"   🚀 Starting generation...")

            for step in range(5):  # Just 5 steps for debugging
                print(f"      Step {step + 1}:")

                # Forward pass
                logits, cache = model(y[None], cache=cache)
                logits = logits[:, -1, :]  # Last token logits

                print(f"         Logits shape: {logits.shape}")
                print(f"         Logits range: {mx.min(logits).item():.3f} to {mx.max(logits).item():.3f}")

                # Sample next token
                next_token = mx.argmax(logits, axis=-1)  # Use argmax for deterministic results
                token_id = int(next_token.item())

                print(f"         Next token ID: {token_id}")

                # Decode token
                try:
                    token_text = tokenizer.decode([token_id])
                    print(f"         Next token text: {repr(token_text)}")
                except:
                    print(f"         Next token text: [DECODE ERROR]")

                # Check for EOS
                if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
                    print(f"         🔚 EOS token encountered! Stopping.")
                    break

                generated.append(token_id)
                y = next_token  # Next iteration uses single token

                print(f"         Generated so far: {generated}")

            # Decode final result
            if generated:
                generated_text = tokenizer.decode(generated)
                print(f"   📤 Final generated text: {repr(generated_text)}")
            else:
                print(f"   📤 No tokens generated!")

        except Exception as e:
            print(f"   ❌ Generation error: {e}")
            import traceback
            traceback.print_exc()

def debug_3_template_compatibility(model, tokenizer):
    """Debug template compatibility problems."""
    print("\n🔍 DEBUG 3: Template Compatibility")
    print("=" * 50)

    # Test different templates that might work better
    templates = {
        "simple": "What is the capital of France?",
        "qa_format": "Question: What is the capital of France?\nAnswer:",
        "chat_format": "<|user|>\nWhat is the capital of France?</s>\n<|assistant|>\n",
        "alpaca_format": "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
        "base_continuation": "The capital of France is",
    }

    for template_name, prompt in templates.items():
        print(f"\n🎯 Testing template: {template_name}")
        print(f"   Prompt: {repr(prompt)}")

        try:
            # Use the simple generator
            generator = MLXTextGenerator(model, tokenizer)

            # Short generation to see if we get anything
            response = generator.generate_simple(prompt, max_tokens=10, temperature=0.1)

            print(f"   Response: {repr(response)}")

            if response.strip():
                print(f"   ✅ Template produces output")
            else:
                print(f"   ❌ Template produces empty output")

        except Exception as e:
            print(f"   ❌ Template error: {e}")

def test_different_checkpoint(checkpoint_path, run_name):
    """Test a specific LoRA checkpoint."""
    print(f"\n🧪 Testing checkpoint: {run_name}")
    print(f"📁 Path: {checkpoint_path}")
    print("-" * 40)

    # Load base model
    print("📥 Loading base model...")
    manager = ModelManager()
    model, tokenizer, config = manager.load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Apply LoRA manually
    from finetune.training.lora import LoRAConfig
    lora_config = LoRAConfig(r=8, alpha=16, dropout=0.1, target_modules=["q_proj", "v_proj"])
    model.freeze()

    # Apply LoRA to last 16 layers
    lora_layers = 16
    layers = model.layers
    start_layer = len(layers) - lora_layers

    for layer_idx in range(start_layer, len(layers)):
        layer = layers[layer_idx]
        if hasattr(layer.self_attn, "q_proj"):
            original_q_proj = layer.self_attn.q_proj
            layer.self_attn.q_proj = LoRALinear.from_linear(original_q_proj, rank=lora_config)
        if hasattr(layer.self_attn, "v_proj"):
            original_v_proj = layer.self_attn.v_proj
            layer.self_attn.v_proj = LoRALinear.from_linear(original_v_proj, rank=lora_config)

    # Load this checkpoint's LoRA weights
    try:
        load_lora_weights(model, checkpoint_path / "lora_weights.npz")
        print("✅ LoRA weights loaded successfully")
    except Exception as e:
        print(f"❌ LoRA weights loading failed: {e}")
        return None

    # Quick test
    test_prompt = "Question: What is the capital of France?\nAnswer:"
    generator = MLXTextGenerator(model, tokenizer)

    try:
        response = generator.generate_simple(test_prompt, max_tokens=20, temperature=0.1)
        print(f"🎯 Test response: {repr(response)}")

        # Check for repetition pattern
        if len(set(response.split()[:5])) <= 2:  # Very few unique words in first 5 tokens
            print("❌ Shows repetition pattern")
            return "REPETITION"
        elif "Paris" in response:
            print("✅ Gives correct answer")
            return "GOOD"
        else:
            print("⚠️  Incorrect but coherent")
            return "COHERENT"
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return "ERROR"

def main():
    """Main debugging function."""
    print("🐛 LoRA Fine-Tuned Generation Debug")
    print("=" * 60)

    # Find all available checkpoints
    all_runs = find_all_training_runs()
    print(f"Found {len(all_runs)} LoRA checkpoints:")

    checkpoint_results = []
    for i, (mtime, path, run_name) in enumerate(all_runs[:3]):  # Test top 3
        result = test_different_checkpoint(path, run_name)
        checkpoint_results.append((run_name, result))

        if i < len(all_runs) - 1:  # Not the last one
            print()

    # Summary
    print("\n" + "=" * 60)
    print("📊 Checkpoint Test Summary:")
    for run_name, result in checkpoint_results:
        status = {"GOOD": "✅", "COHERENT": "⚠️", "REPETITION": "❌", "ERROR": "💥"}.get(result, "❓")
        print(f"   {status} {run_name}: {result}")

    # If any checkpoint works, continue with detailed debug
    good_checkpoints = [name for name, result in checkpoint_results if result == "GOOD"]
    if good_checkpoints:
        print(f"\n🎉 Found working checkpoint: {good_checkpoints[0]}")
        print("Proceeding with detailed debug...")

        # Debug 1: LoRA Loading (using most recent still)
        result = debug_1_lora_loading()
        if not result:
            print("\n❌ LoRA loading debug failed, cannot continue")
            return 1

        model, tokenizer = result

        # Debug 2: Early Stopping
        debug_2_early_stopping(model, tokenizer)

        # Debug 3: Template Compatibility
        debug_3_template_compatibility(model, tokenizer)
    else:
        print("\n❌ All checkpoints show the same degenerate behavior")
        print("The issue is with the training process, not the specific checkpoint")

    print("\n" + "=" * 60)
    print("🏁 Debug completed!")

    return 0

if __name__ == "__main__":
    sys.exit(main())