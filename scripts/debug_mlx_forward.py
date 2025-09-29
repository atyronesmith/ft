#!/usr/bin/env python3
"""
Debug script to compare MLX vs HuggingFace forward pass.

This implements Step 2 of the systematic debugging strategy:
Compare intermediate activations in the forward pass to find the divergence point.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def compare_forward_pass():
    """Compare forward pass between MLX and HuggingFace implementations."""
    print("🔍 Step 2: Comparing forward pass intermediate activations")
    print("=" * 60)

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
        print(f"\n🎯 Test input: '{test_text}'")
        print(f"   Token IDs: {input_ids.tolist()}")
        print(f"   Shape: {input_ids.shape}")

        # Convert to MLX format
        mlx_input = mx.array(input_ids.numpy()).astype(mx.int32).reshape(1, -1)
        hf_input = input_ids.unsqueeze(0)  # Add batch dimension

        print(f"\n📊 Input shapes - MLX: {mlx_input.shape}, HF: {hf_input.shape}")

        # Compare embeddings first
        print("\n🔍 Step 2a: Comparing embeddings...")

        # MLX embeddings
        mlx_embeddings = mlx_model.embed_tokens(mlx_input)
        print(f"MLX embeddings shape: {mlx_embeddings.shape}")

        # HF embeddings
        hf_embeddings = hf_model.model.embed_tokens(hf_input)
        print(f"HF embeddings shape: {hf_embeddings.shape}")

        # Compare embeddings
        mlx_emb_np = np.array(mlx_embeddings[0])  # Remove batch dim
        hf_emb_np = hf_embeddings[0].detach().numpy()

        emb_close = np.allclose(mlx_emb_np, hf_emb_np, atol=1e-6, rtol=1e-5)
        emb_max_diff = np.max(np.abs(mlx_emb_np - hf_emb_np))

        status = "✅" if emb_close else "❌"
        print(f"{status} Embeddings match: {emb_close} | Max diff: {emb_max_diff:.2e}")

        if not emb_close:
            print("💥 EMBEDDINGS DIVERGE - Check embedding layer implementation!")
            return "embeddings"

        # Compare first transformer layer
        print("\n🔍 Step 2b: Comparing first transformer layer...")

        # For MLX - manually run through first layer
        # (We'd need to modify the model to expose intermediate outputs)
        # For now, let's check final output

        print("\n🔍 Step 2c: Comparing final model outputs...")

        # Get final outputs
        with torch.no_grad():
            hf_outputs = hf_model(hf_input)
            hf_logits = hf_outputs.logits[0]  # Remove batch dim

        mlx_logits = mlx_model.forward(mlx_input)[0]  # Remove batch dim

        print(f"MLX logits shape: {mlx_logits.shape}")
        print(f"HF logits shape: {hf_logits.shape}")

        # Compare final logits
        mlx_logits_np = np.array(mlx_logits)
        hf_logits_np = hf_logits.detach().numpy()

        logits_close = np.allclose(mlx_logits_np, hf_logits_np, atol=1e-3, rtol=1e-3)
        logits_max_diff = np.max(np.abs(mlx_logits_np - hf_logits_np))

        status = "✅" if logits_close else "❌"
        print(f"{status} Final logits match: {logits_close} | Max diff: {logits_max_diff:.2e}")

        if not logits_close:
            print("\n🔍 Analyzing logit differences...")

            # Check top predictions
            print("\nTop 5 predictions:")
            mlx_top5 = np.argsort(-mlx_logits_np[-1])[:5]  # Last token predictions
            hf_top5 = np.argsort(-hf_logits_np[-1])[:5]

            print("MLX top 5 token IDs:", mlx_top5.tolist())
            print("HF top 5 token IDs: ", hf_top5.tolist())

            for i, (mlx_id, hf_id) in enumerate(zip(mlx_top5, hf_top5, strict=False)):
                mlx_token = tokenizer.decode([mlx_id])
                hf_token = tokenizer.decode([hf_id])
                mlx_prob = float(mlx_logits_np[-1, mlx_id])
                hf_prob = float(hf_logits_np[-1, hf_id])
                print(
                    f"  {i+1}. MLX: '{mlx_token}' ({mlx_prob:.3f}) | HF: '{hf_token}' ({hf_prob:.3f})"
                )

            return "forward_pass"
        else:
            print("🎉 Forward passes match! This is unexpected given generation issues.")
            return "sampling"

    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        return "error"


if __name__ == "__main__":
    result = compare_forward_pass()
    print(f"\n🎯 Debug result: {result}")
