#!/usr/bin/env python3
"""
Debug script to compare MLX vs HuggingFace weight loading.

This implements Step 1 of the systematic debugging strategy:
Verify that weights are loaded identically between our MLX implementation
and the reference HuggingFace PyTorch implementation.
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))


def compare_weight_loading():
    """Compare weight loading between MLX and HuggingFace implementations."""
    print("🔍 Step 1: Verifying weight loading accuracy")
    print("=" * 60)

    try:
        # Load our MLX implementation
        import torch
        from finetune.models.manager import ModelManager
        from transformers import AutoModelForCausalLM

        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"Loading model: {model_id}")

        # Load MLX model
        print("\n📦 Loading MLX model...")
        manager = ModelManager()
        mlx_model, _, _ = manager.load_model(model_id)
        print(f"✅ MLX model loaded: {mlx_model.num_parameters:,} parameters")

        # Load HuggingFace model
        print("\n📦 Loading HuggingFace model...")
        hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        print(f"✅ HF model loaded: {sum(p.numel() for p in hf_model.parameters()):,} parameters")

        # Compare specific layers
        comparisons = [
            # (MLX path, HF path, description)
            ("embed_tokens.weight", "model.embed_tokens.weight", "Embedding layer"),
            (
                "layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.q_proj.weight",
                "First layer Q projection",
            ),
            (
                "layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "First layer K projection",
            ),
            (
                "layers.0.self_attn.v_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "First layer V projection",
            ),
            (
                "layers.0.self_attn.o_proj.weight",
                "model.layers.0.self_attn.o_proj.weight",
                "First layer output projection",
            ),
            (
                "layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "First layer MLP gate",
            ),
            (
                "layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "First layer MLP up",
            ),
            (
                "layers.0.mlp.down_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
                "First layer MLP down",
            ),
            (
                "layers.5.self_attn.q_proj.weight",
                "model.layers.5.self_attn.q_proj.weight",
                "Middle layer Q projection",
            ),
            ("lm_head.weight", "lm_head.weight", "Language model head"),
        ]

        print("\n🔍 Comparing weights:")
        print("-" * 80)

        all_match = True

        for mlx_path, hf_path, description in comparisons:
            try:
                # Get MLX weight
                mlx_weight = mlx_model
                for part in mlx_path.split("."):
                    if part.isdigit():
                        mlx_weight = mlx_weight[int(part)]
                    else:
                        mlx_weight = getattr(mlx_weight, part)

                # Get HuggingFace weight
                hf_weight = hf_model
                for part in hf_path.split("."):
                    if part.isdigit():
                        hf_weight = hf_weight[int(part)]
                    else:
                        hf_weight = getattr(hf_weight, part)

                # Convert to numpy for comparison
                mlx_np = np.array(mlx_weight)
                hf_np = hf_weight.detach().numpy()

                # Check shapes match
                if mlx_np.shape != hf_np.shape:
                    print(
                        f"❌ {description:30} | Shape mismatch: MLX {mlx_np.shape} vs HF {hf_np.shape}"
                    )
                    all_match = False
                    continue

                # Check numerical equivalence
                are_close = np.allclose(mlx_np, hf_np, atol=1e-6, rtol=1e-5)
                max_diff = np.max(np.abs(mlx_np - hf_np))

                status = "✅" if are_close else "❌"
                print(
                    f"{status} {description:30} | Shape: {mlx_np.shape} | Max diff: {max_diff:.2e}"
                )

                if not are_close:
                    all_match = False
                    # Show some sample values for debugging
                    print(f"   MLX sample: {mlx_np.flat[:5]}")
                    print(f"   HF sample:  {hf_np.flat[:5]}")

            except Exception as e:
                print(f"❌ {description:30} | Error: {e}")
                all_match = False

        print("-" * 80)

        if all_match:
            print("🎉 SUCCESS: All weights match between MLX and HuggingFace!")
            print("   → Problem is likely in the forward pass implementation")
            return "forward_pass"
        else:
            print("💥 WEIGHT LOADING ISSUE: Weights don't match!")
            print("   → Problem is in the weight conversion process")
            return "weight_loading"

    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        return "error"


if __name__ == "__main__":
    result = compare_weight_loading()
    print(f"\n🎯 Debug result: {result}")
