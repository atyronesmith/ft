#!/usr/bin/env python3
"""
Debug model state and generation logic by examining exactly what the model sees.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import mlx.core as mx
import numpy as np
from finetune.models.manager import ModelManager
from finetune.training.lora import LoRALinear

def debug_model_state():
    """Debug the exact model state and generation."""
    print("🔍 DEBUGGING MODEL STATE AND GENERATION")
    print("=" * 60)

    # Load and setup model exactly like in training
    print("1. Loading and setting up model like training...")
    manager = ModelManager()
    model, tokenizer, config = manager.load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Apply LoRA exactly like training
    model.freeze()
    lora_layers = 16
    layers = model.model.layers if hasattr(model, 'model') else model.layers
    start_layer = len(layers) - lora_layers

    for layer_idx in range(start_layer, len(layers)):
        layer = layers[layer_idx]
        if hasattr(layer, 'self_attn'):
            if hasattr(layer.self_attn, 'q_proj'):
                layer.self_attn.q_proj = LoRALinear.from_linear(layer.self_attn.q_proj, rank=8)
            if hasattr(layer.self_attn, 'v_proj'):
                layer.self_attn.v_proj = LoRALinear.from_linear(layer.self_attn.v_proj, rank=8)

    # Load LoRA weights
    training_dirs = list(Path("training").glob("run-*"))
    latest_run = max(training_dirs, key=lambda p: p.stat().st_mtime)
    lora_weights_path = latest_run / "final_model" / "lora_weights.npz"

    if lora_weights_path.exists():
        model.load_weights(str(lora_weights_path), strict=False)
        mx.eval(model.parameters())
        model.eval()
        print("   ✅ Model setup complete with LoRA weights")
    else:
        print("   ❌ No LoRA weights found")
        return "no_weights"

    # Test the exact prompt format used in training
    print("\n2. Testing exact training prompt format...")

    # This is the exact format we see in the training logs
    training_prompt = ("<|system|>\nYou are a helpful geography assistant who provides accurate, "
                      "concise answers about world capitals.</s>\n<|user|>\nWhat is the capital of France?</s>\n"
                      "<|assistant|>\n")

    print(f"   Prompt: {repr(training_prompt)}")

    # Tokenize exactly like training
    input_ids = tokenizer.encode(training_prompt, add_special_tokens=False)
    print(f"   Tokens: {input_ids}")
    print(f"   Length: {len(input_ids)}")

    # Create input tensor
    input_tensor = mx.array(input_ids).astype(mx.int32)

    print("\n3. Testing model forward pass...")
    try:
        # Test forward pass with full prompt
        result = model.forward(input_tensor[None])  # Add batch dimension
        if isinstance(result, tuple):
            logits = result[0]  # Extract logits from tuple
            print(f"   ✅ Forward pass works: {logits.shape} (from tuple)")
        else:
            logits = result
            print(f"   ✅ Forward pass works: {logits.shape}")

        # Examine the last token's logits (where generation should start)
        last_logits = logits[0, -1, :]  # [vocab_size]
        print(f"   Last token logits shape: {last_logits.shape}")

        # Get top 10 predictions
        top_indices = mx.argsort(-last_logits)[:10]
        print("\n   Top 10 predictions:")
        for i, idx in enumerate(top_indices):
            token_id = int(idx.item())
            token_text = tokenizer.decode([token_id])
            logit_value = float(last_logits[idx].item())
            print(f"     {i+1:2d}. Token {token_id:5d} (logit {logit_value:8.3f}): {repr(token_text)}")

        # Check if newline token (13) is dominating
        newline_logit = float(last_logits[13].item())
        max_logit = float(mx.max(last_logits).item())
        print(f"\n   Newline token (13) logit: {newline_logit:.3f}")
        print(f"   Max logit: {max_logit:.3f}")

        if newline_logit == max_logit:
            print("   ⚠️  Model is predicting newline as most likely!")

    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return "forward_failed"

    print("\n4. Testing with MLX examples generation pattern...")
    try:
        # Use exact MLX examples pattern
        y = input_tensor  # 1D array like MLX examples
        cache = None

        print(f"   Input shape: {y.shape}")

        # MLX examples pattern: model(y[None], cache=cache)
        if hasattr(model, '__call__'):
            result = model(y[None], cache=cache)
            if isinstance(result, tuple):
                logits, new_cache = result
                print(f"   ✅ Cache-based call works: logits {logits.shape}, cache type {type(new_cache)}")
            else:
                logits = result
                print(f"   ✅ Non-cache call works: logits {logits.shape}")
        else:
            print("   ❌ Model doesn't support __call__")
            return "no_call_method"

        # Sample like MLX examples
        def sample(logits_tensor, temp=0.0):
            return (
                mx.argmax(logits_tensor, axis=-1)
                if temp == 0
                else mx.random.categorical(logits_tensor * (1 / temp))
            )

        # Get logits for last position
        next_logits = logits[:, -1, :]  # [1, vocab_size]
        next_token = sample(next_logits)
        token_id = int(next_token.item() if hasattr(next_token, 'item') else next_token)

        print(f"   Sampled token: {token_id} -> {repr(tokenizer.decode([token_id]))}")

        if token_id == 13:  # newline
            print("   ⚠️  Still generating newline with MLX pattern!")

            # Let's check what would happen with temperature
            next_token_temp = sample(next_logits, temp=0.8)
            token_id_temp = int(next_token_temp.item() if hasattr(next_token_temp, 'item') else next_token_temp)
            print(f"   With temp=0.8: {token_id_temp} -> {repr(tokenizer.decode([token_id_temp]))}")

    except Exception as e:
        print(f"   ❌ MLX pattern failed: {e}")
        import traceback
        traceback.print_exc()
        return "mlx_pattern_failed"

    print("\n5. Checking training data consistency...")

    # Check what the model was trained to predict at this position
    training_answer = "Paris</s> \n"
    expected_tokens = tokenizer.encode(training_answer, add_special_tokens=False)
    print(f"   Training answer: {repr(training_answer)}")
    print(f"   Expected tokens: {expected_tokens}")
    print(f"   First expected token: {expected_tokens[0]} -> {repr(tokenizer.decode([expected_tokens[0]]))}")

    if expected_tokens[0] != token_id:
        print("   ❌ Model not generating expected first token!")

        # Check logit for expected token
        expected_logit = float(last_logits[expected_tokens[0]].item())
        print(f"   Expected token logit: {expected_logit:.3f}")
        print(f"   Rank of expected token: {int(mx.sum(last_logits > expected_logit).item())}")

        return "wrong_prediction"
    else:
        print("   ✅ Model generating expected first token!")
        return "correct_prediction"

if __name__ == "__main__":
    result = debug_model_state()
    print(f"\n🎯 Model state debug result: {result}")