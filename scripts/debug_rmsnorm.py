#!/usr/bin/env python3
"""
Debug script to compare MLX vs HuggingFace RMSNorm implementations.

This isolates the RMSNorm computation to find the exact algorithmic difference.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def compare_rmsnorm_implementations():
    """Compare RMSNorm implementations between MLX and HuggingFace."""
    print("🔍 Comparing RMSNorm implementations")
    print("=" * 50)

    try:
        import mlx.core as mx
        from finetune.models.manager import ModelManager
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # Load both models
        print(f"Loading models: {model_id}")
        manager = ModelManager()
        mlx_model, tokenizer, _ = manager.load_model(model_id)
        hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

        # Set both models to eval mode
        mlx_model.eval()
        hf_model.eval()

        # Create identical test input
        test_text = "What is the capital of France?"
        input_ids = tokenizer.encode(test_text, return_tensors="pt")[0]

        # Convert to MLX format
        mlx_input = mx.array(input_ids.numpy()).astype(mx.int32).reshape(1, -1)
        hf_input = input_ids.unsqueeze(0)

        # Get embeddings (we know these match)
        mlx_embeddings = mlx_model.embed_tokens(mlx_input)
        hf_embeddings = hf_model.model.embed_tokens(hf_input)

        print(f"\n📊 Embeddings - MLX: {mlx_embeddings.shape}, HF: {hf_embeddings.shape}")

        # Get the first layer's RMSNorm layers
        mlx_norm = mlx_model.layers[0].input_layernorm
        hf_norm = hf_model.model.layers[0].input_layernorm

        print("\n🔍 RMSNorm configurations:")
        print(f"MLX eps: {mlx_norm.eps}")
        print(f"HF eps: {hf_norm.variance_epsilon}")

        # Compare weights
        mlx_weight = np.array(mlx_norm.weight)
        hf_weight = hf_norm.weight.detach().numpy()

        weights_match = np.allclose(mlx_weight, hf_weight, atol=1e-8)
        print(f"RMSNorm weights match: {weights_match}")
        if not weights_match:
            print(f"Max weight diff: {np.max(np.abs(mlx_weight - hf_weight))}")

        # Apply RMSNorm to the same input
        print("\n🧪 Testing RMSNorm computation:")

        # MLX computation
        mlx_norm_output = mlx_norm(mlx_embeddings)

        # HF computation
        with torch.no_grad():
            hf_norm_output = hf_norm(hf_embeddings)

        # Compare outputs
        mlx_output_np = np.array(mlx_norm_output[0])  # Remove batch dim
        hf_output_np = hf_norm_output[0].detach().numpy()

        outputs_match = np.allclose(mlx_output_np, hf_output_np, atol=1e-5, rtol=1e-4)
        max_diff = np.max(np.abs(mlx_output_np - hf_output_np))
        mean_diff = np.mean(np.abs(mlx_output_np - hf_output_np))

        print(f"✅ RMSNorm outputs match: {outputs_match}")
        print(f"📊 Max difference: {max_diff:.2e}")
        print(f"📊 Mean difference: {mean_diff:.2e}")

        if not outputs_match:
            print("\n🔍 Sample values comparison:")
            print(f"MLX output sample:  {mlx_output_np.flat[:10]}")
            print(f"HF output sample:   {hf_output_np.flat[:10]}")

            # Check if this is a precision issue or algorithmic difference
            relative_diff = max_diff / np.mean(np.abs(hf_output_np))
            print(f"📊 Relative difference: {relative_diff:.2e}")

            if relative_diff > 1e-3:
                print("💥 ALGORITHMIC DIFFERENCE - RMSNorm implementations differ significantly!")
                return "algorithm_difference"
            else:
                print("⚠️  PRECISION DIFFERENCE - Small numerical differences")
                return "precision_difference"
        else:
            print("🎉 RMSNorm implementations match perfectly!")
            return "match"

    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        return "error"


if __name__ == "__main__":
    result = compare_rmsnorm_implementations()
    print(f"\n🎯 RMSNorm debug result: {result}")
