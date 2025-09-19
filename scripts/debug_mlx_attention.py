#!/usr/bin/env python3
"""
Debug script to compare MLX vs HuggingFace attention mechanism.

This implements Step 3 of the systematic debugging strategy:
Examine attention mechanism implementation to find the exact divergence point.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def compare_attention_mechanism():
    """Compare attention mechanism between MLX and HuggingFace implementations."""
    print("🔍 Step 3: Examining attention mechanism implementation")
    print("=" * 60)

    try:
        from finetune.models.manager import ModelManager
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import mlx.core as mx

        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # Load both models
        print(f"Loading models: {model_id}")
        manager = ModelManager()
        mlx_model = manager.load_model(model_id)
        hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Set both models to eval mode
        mlx_model.eval()
        hf_model.eval()

        # Create identical test input
        test_text = "What is the capital of France?"
        input_ids = tokenizer.encode(test_text, return_tensors="pt")[0]
        print(f"\n🎯 Test input: '{test_text}'")
        print(f"   Token IDs: {input_ids.tolist()}")

        # Convert to MLX format
        mlx_input = mx.array(input_ids.numpy()).astype(mx.int32).reshape(1, -1)
        hf_input = input_ids.unsqueeze(0)

        print(f"\n📊 Input shapes - MLX: {mlx_input.shape}, HF: {hf_input.shape}")

        # Get embeddings (we know these match from Step 2)
        print("\n🔍 Step 3a: Getting embeddings...")
        mlx_embeddings = mlx_model.embed_tokens(mlx_input)
        hf_embeddings = hf_model.model.embed_tokens(hf_input)
        print(f"✅ Embeddings ready - MLX: {mlx_embeddings.shape}, HF: {hf_embeddings.shape}")

        # Focus on first transformer layer for detailed analysis
        print("\n🔍 Step 3b: Examining first transformer layer...")

        # MLX first layer
        mlx_layer = mlx_model.layers[0]
        print(f"MLX layer type: {type(mlx_layer)}")

        # HF first layer
        hf_layer = hf_model.model.layers[0]
        print(f"HF layer type: {type(hf_layer)}")

        # Compare attention weights first
        print("\n🔍 Step 3c: Comparing attention projection weights...")

        # Q projection weights
        mlx_q_weight = np.array(mlx_layer.self_attn.q_proj.weight)
        hf_q_weight_raw = hf_layer.self_attn.q_proj.weight.detach().numpy()

        print(f"MLX Q weight shape: {mlx_q_weight.shape}")
        print(f"HF Q weight shape (raw): {hf_q_weight_raw.shape}")

        # Both MLX and PyTorch use (out_features, in_features) convention
        # So NO transpose should be needed
        q_weights_match = np.allclose(mlx_q_weight, hf_q_weight_raw, atol=1e-6, rtol=1e-5)
        q_max_diff = np.max(np.abs(mlx_q_weight - hf_q_weight_raw))

        status = "✅" if q_weights_match else "❌"
        print(f"{status} Q projection weights match: {q_weights_match} | Max diff: {q_max_diff:.2e}")

        # K projection weights (note: different dimensions due to GQA)
        mlx_k_weight = np.array(mlx_layer.self_attn.k_proj.weight)
        hf_k_weight_raw = hf_layer.self_attn.k_proj.weight.detach().numpy()

        print(f"MLX K weight shape: {mlx_k_weight.shape}")
        print(f"HF K weight shape (raw): {hf_k_weight_raw.shape}")

        # Both MLX and PyTorch use (out_features, in_features) convention
        # So NO transpose should be needed
        k_weights_match = np.allclose(mlx_k_weight, hf_k_weight_raw, atol=1e-6, rtol=1e-5)
        k_max_diff = np.max(np.abs(mlx_k_weight - hf_k_weight_raw))

        status = "✅" if k_weights_match else "❌"
        print(f"{status} K projection weights match: {k_weights_match} | Max diff: {k_max_diff:.2e}")

        # V projection weights (note: different dimensions due to GQA)
        mlx_v_weight = np.array(mlx_layer.self_attn.v_proj.weight)
        hf_v_weight_raw = hf_layer.self_attn.v_proj.weight.detach().numpy()

        print(f"MLX V weight shape: {mlx_v_weight.shape}")
        print(f"HF V weight shape (raw): {hf_v_weight_raw.shape}")

        v_weights_match = np.allclose(mlx_v_weight, hf_v_weight_raw, atol=1e-6, rtol=1e-5)
        v_max_diff = np.max(np.abs(mlx_v_weight - hf_v_weight_raw))

        status = "✅" if v_weights_match else "❌"
        print(f"{status} V projection weights match: {v_weights_match} | Max diff: {v_max_diff:.2e}")

        # O projection weights
        mlx_o_weight = np.array(mlx_layer.self_attn.o_proj.weight)
        hf_o_weight_raw = hf_layer.self_attn.o_proj.weight.detach().numpy()

        print(f"MLX O weight shape: {mlx_o_weight.shape}")
        print(f"HF O weight shape (raw): {hf_o_weight_raw.shape}")

        o_weights_match = np.allclose(mlx_o_weight, hf_o_weight_raw, atol=1e-6, rtol=1e-5)
        o_max_diff = np.max(np.abs(mlx_o_weight - hf_o_weight_raw))

        status = "✅" if o_weights_match else "❌"
        print(f"{status} O projection weights match: {o_weights_match} | Max diff: {o_max_diff:.2e}")

        if not all([q_weights_match, k_weights_match, v_weights_match, o_weights_match]):
            print("\n💥 ATTENTION WEIGHT MISMATCH - Problem in weight conversion!")
            return "attention_weights"

        # Now test the actual attention computation
        print("\n🔍 Step 3d: Comparing attention computations...")

        # Get the hidden states for the first layer (output from embeddings)
        # For first layer, input is just the embeddings

        # MLX attention forward pass
        # Note: We need to manually call the attention forward to get intermediate values
        print("\n🎯 Testing attention forward pass...")

        # Prepare input for attention (first token only for simplicity)
        seq_len = mlx_embeddings.shape[1]
        print(f"Sequence length: {seq_len}")

        # For MLX model - manually compute attention to get intermediate values
        hidden_states = mlx_embeddings  # [batch_size, seq_len, hidden_size]

        try:
            # Get RMSNorm output (input to attention)
            mlx_norm_input = mlx_layer.input_layernorm(hidden_states)
            hf_norm_input = hf_layer.input_layernorm(hf_embeddings)

            print(f"MLX norm input shape: {mlx_norm_input.shape}")
            print(f"HF norm input shape: {hf_norm_input.shape}")

            # Compare layernorm outputs
            mlx_norm_np = np.array(mlx_norm_input[0])  # Remove batch dim
            hf_norm_np = hf_norm_input[0].detach().numpy()

            norm_match = np.allclose(mlx_norm_np, hf_norm_np, atol=1e-5, rtol=1e-4)
            norm_max_diff = np.max(np.abs(mlx_norm_np - hf_norm_np))

            status = "✅" if norm_match else "❌"
            print(f"{status} LayerNorm outputs match: {norm_match} | Max diff: {norm_max_diff:.2e}")

            if not norm_match:
                print("\n💥 LAYERNORM DIVERGENCE - Problem in normalization!")
                print(f"MLX norm sample: {mlx_norm_np.flat[:5]}")
                print(f"HF norm sample:  {hf_norm_np.flat[:5]}")
                return "layernorm"

            # Compare Q, K, V projections
            print("\n🔍 Step 3e: Comparing Q, K, V projections...")

            # MLX projections
            mlx_q = mlx_layer.self_attn.q_proj(mlx_norm_input)
            mlx_k = mlx_layer.self_attn.k_proj(mlx_norm_input)
            mlx_v = mlx_layer.self_attn.v_proj(mlx_norm_input)

            print(f"MLX Q shape: {mlx_q.shape}")
            print(f"MLX K shape: {mlx_k.shape}")
            print(f"MLX V shape: {mlx_v.shape}")

            # HF projections
            with torch.no_grad():
                hf_q = hf_layer.self_attn.q_proj(hf_norm_input)
                hf_k = hf_layer.self_attn.k_proj(hf_norm_input)
                hf_v = hf_layer.self_attn.v_proj(hf_norm_input)

            print(f"HF Q shape: {hf_q.shape}")
            print(f"HF K shape: {hf_k.shape}")
            print(f"HF V shape: {hf_v.shape}")

            # Compare projections
            for name, mlx_proj, hf_proj in [("Q", mlx_q, hf_q), ("K", mlx_k, hf_k), ("V", mlx_v, hf_v)]:
                mlx_proj_np = np.array(mlx_proj[0])  # Remove batch dim
                hf_proj_np = hf_proj[0].detach().numpy()

                proj_match = np.allclose(mlx_proj_np, hf_proj_np, atol=1e-4, rtol=1e-3)
                proj_max_diff = np.max(np.abs(mlx_proj_np - hf_proj_np))

                status = "✅" if proj_match else "❌"
                print(f"{status} {name} projection match: {proj_match} | Max diff: {proj_max_diff:.2e}")

                if not proj_match:
                    print(f"\n💥 {name} PROJECTION DIVERGENCE!")
                    print(f"MLX {name} sample: {mlx_proj_np.flat[:5]}")
                    print(f"HF {name} sample:  {hf_proj_np.flat[:5]}")
                    return f"{name.lower()}_projection"

            print("\n🎉 All attention projections match!")
            print("   → Problem likely in attention mechanism (RoPE, scaling, softmax)")
            return "attention_mechanism"

        except Exception as e:
            print(f"💥 ERROR in attention computation: {e}")
            import traceback
            traceback.print_exc()
            return "attention_error"

    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return "error"

if __name__ == "__main__":
    result = compare_attention_mechanism()
    print(f"\n🎯 Debug result: {result}")