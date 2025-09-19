#!/usr/bin/env python3
"""
Enhanced debugging script for sub-module level analysis.

This implements the final step of systematic debugging:
Deep dive into transformer layer internals to find the exact divergence point.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def compare_activations(mlx_tensor, hf_tensor, name: str, atol: float = 1e-5, rtol: float = 1e-4) -> bool:
    """Compare MLX and HuggingFace tensor activations."""
    try:
        import mlx.core as mx

        # Handle batch dimension differences
        if len(mlx_tensor.shape) == 3 and len(hf_tensor.shape) == 3:
            mlx_np = np.array(mlx_tensor[0])  # Remove batch dim for MLX
            hf_np = hf_tensor[0].detach().numpy()
        else:
            mlx_np = np.array(mlx_tensor)
            hf_np = hf_tensor.detach().numpy()

        is_close = np.allclose(mlx_np, hf_np, atol=atol, rtol=rtol)
        max_diff = np.max(np.abs(mlx_np - hf_np))
        mean_diff = np.mean(np.abs(mlx_np - hf_np))

        status = "✅" if is_close else "❌"
        print(f"    {status} {name}: Match={is_close} | Max diff: {max_diff:.2e} | Mean diff: {mean_diff:.2e}")

        if not is_close:
            relative_diff = max_diff / (np.mean(np.abs(hf_np)) + 1e-8)
            print(f"        Relative diff: {relative_diff:.2e}")
            if max_diff > 1.0:
                print(f"        🚨 CRITICAL: Very large absolute differences!")
                print(f"        MLX sample: {mlx_np.flat[:5]}")
                print(f"        HF sample:  {hf_np.flat[:5]}")

        return is_close

    except Exception as e:
        print(f"    ❌ {name}: Error during comparison - {e}")
        return False

def debug_attention_internals(mlx_attn, hf_attn, mlx_input, hf_input):
    """Deep dive into attention mechanism internals."""
    print("\n      🔍 Attention Mechanism Deep Dive:")

    try:
        import mlx.core as mx

        # Get dimensions
        B, L, D = mlx_input.shape

        # Compare Q, K, V projections
        mlx_q = mlx_attn.q_proj(mlx_input)
        mlx_k = mlx_attn.k_proj(mlx_input)
        mlx_v = mlx_attn.v_proj(mlx_input)

        with torch.no_grad():
            hf_q = hf_attn.q_proj(hf_input)
            hf_k = hf_attn.k_proj(hf_input)
            hf_v = hf_attn.v_proj(hf_input)

        if not compare_activations(mlx_q, hf_q, "Q projection"):
            print("          🎯 DIVERGENCE: Q projection computation differs")
            return "q_projection"

        if not compare_activations(mlx_k, hf_k, "K projection"):
            print("          🎯 DIVERGENCE: K projection computation differs")
            return "k_projection"

        if not compare_activations(mlx_v, hf_v, "V projection"):
            print("          🎯 DIVERGENCE: V projection computation differs")
            return "v_projection"

        # Reshape for attention
        num_heads = mlx_attn.num_heads
        head_dim = mlx_attn.head_dim

        mlx_q_reshaped = mlx_q.reshape(B, L, num_heads, head_dim).transpose(0, 2, 1, 3)
        mlx_k_reshaped = mlx_k.reshape(B, L, mlx_attn.num_key_value_heads, head_dim).transpose(0, 2, 1, 3)
        mlx_v_reshaped = mlx_v.reshape(B, L, mlx_attn.num_key_value_heads, head_dim).transpose(0, 2, 1, 3)

        with torch.no_grad():
            hf_q_reshaped = hf_q.view(B, L, num_heads, head_dim).transpose(1, 2)
            hf_k_reshaped = hf_k.view(B, L, mlx_attn.num_key_value_heads, head_dim).transpose(1, 2)
            hf_v_reshaped = hf_v.view(B, L, mlx_attn.num_key_value_heads, head_dim).transpose(1, 2)

        if not compare_activations(mlx_q_reshaped, hf_q_reshaped, "Q reshaped"):
            print("          🎯 DIVERGENCE: Q reshaping/transposition differs")
            return "q_reshape"

        # Apply RoPE
        mlx_q_rope = mlx_attn.rope(mlx_q_reshaped)
        mlx_k_rope = mlx_attn.rope(mlx_k_reshaped)

        # HuggingFace RoPE (more complex to replicate exactly)
        print("          ⚠️  RoPE comparison needs HF position embeddings - checking attention scores instead")

        # Compare attention scores (after RoPE)
        mlx_scores = mx.matmul(mlx_q_rope, mlx_k_rope.transpose(0, 1, 3, 2)) / (head_dim ** 0.5)

        # For HF, we'd need to apply their RoPE implementation, which is complex
        # So let's check if the issue is in basic matrix operations
        print("          ℹ️  Skipping detailed RoPE comparison - focus on projections and scores")

        return "attention_scores"

    except Exception as e:
        print(f"          ❌ Error in attention deep dive: {e}")
        import traceback
        traceback.print_exc()
        return "attention_error"

def debug_mlp_internals(mlx_mlp, hf_mlp, mlx_input, hf_input):
    """Deep dive into MLP internals."""
    print("\n      🔍 MLP Deep Dive:")

    try:
        # Compare gate projection
        mlx_gate = mlx_mlp.gate_proj(mlx_input)
        with torch.no_grad():
            hf_gate = hf_mlp.gate_proj(hf_input)

        if not compare_activations(mlx_gate, hf_gate, "Gate projection"):
            print("          🎯 DIVERGENCE: Gate projection computation differs")
            return "gate_proj"

        # Compare up projection
        mlx_up = mlx_mlp.up_proj(mlx_input)
        with torch.no_grad():
            hf_up = hf_mlp.up_proj(hf_input)

        if not compare_activations(mlx_up, hf_up, "Up projection"):
            print("          🎯 DIVERGENCE: Up projection computation differs")
            return "up_proj"

        # Compare activation (SiLU)
        import mlx.nn as nn
        mlx_activated = nn.silu(mlx_gate) * mlx_up
        with torch.no_grad():
            hf_activated = torch.nn.functional.silu(hf_gate) * hf_up

        if not compare_activations(mlx_activated, hf_activated, "SiLU activation"):
            print("          🎯 DIVERGENCE: SiLU activation computation differs")
            return "silu_activation"

        # Compare down projection
        mlx_down = mlx_mlp.down_proj(mlx_activated)
        with torch.no_grad():
            hf_down = hf_mlp.down_proj(hf_activated)

        if not compare_activations(mlx_down, hf_down, "Down projection"):
            print("          🎯 DIVERGENCE: Down projection computation differs")
            return "down_proj"

        print("          ✅ All MLP components match!")
        return "mlp_ok"

    except Exception as e:
        print(f"          ❌ Error in MLP deep dive: {e}")
        return "mlp_error"

def debug_layer_internals(mlx_layer, hf_layer, mlx_input, hf_input, layer_idx: int):
    """
    Performs a deep dive into a single transformer layer to find the
    exact point of divergence.
    """
    print(f"\n🔍 === LAYER {layer_idx} DEEP DIVE ===")

    try:
        import mlx.core as mx

        # 1. Input LayerNorm
        print("\n  📍 Step 1: Input LayerNorm")
        mlx_norm_out = mlx_layer.input_layernorm(mlx_input)
        with torch.no_grad():
            hf_norm_out = hf_layer.input_layernorm(hf_input)

        if not compare_activations(mlx_norm_out, hf_norm_out, "Input LayerNorm"):
            print("    🎯 ROOT CAUSE: Input LayerNorm divergence!")
            return "input_layernorm"

        # 2. Self-Attention Block
        print("\n  📍 Step 2: Self-Attention Block")
        mlx_attn_out, _ = mlx_layer.self_attn(mlx_norm_out)
        with torch.no_grad():
            # HuggingFace requires position_ids for RoPE
            seq_len = hf_norm_out.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
            hf_attn_out = hf_layer.self_attn(hf_norm_out, position_ids=position_ids)[0]

        if not compare_activations(mlx_attn_out, hf_attn_out, "Self-Attention Output"):
            print("    🎯 ROOT CAUSE: Self-Attention mechanism!")
            # Deep dive into attention
            attention_issue = debug_attention_internals(mlx_layer.self_attn, hf_layer.self_attn, mlx_norm_out, hf_norm_out)
            return f"attention_{attention_issue}"

        # 3. First Residual Connection
        print("\n  📍 Step 3: First Residual Connection")
        mlx_residual1 = mlx_input + mlx_attn_out
        with torch.no_grad():
            hf_residual1 = hf_input + hf_attn_out

        if not compare_activations(mlx_residual1, hf_residual1, "First Residual"):
            print("    🎯 ROOT CAUSE: First residual connection!")
            return "first_residual"

        # 4. Post-Attention LayerNorm
        print("\n  📍 Step 4: Post-Attention LayerNorm")
        mlx_post_norm_out = mlx_layer.post_attention_layernorm(mlx_residual1)
        with torch.no_grad():
            hf_post_norm_out = hf_layer.post_attention_layernorm(hf_residual1)

        if not compare_activations(mlx_post_norm_out, hf_post_norm_out, "Post-Attention LayerNorm"):
            print("    🎯 ROOT CAUSE: Post-attention LayerNorm!")
            return "post_attention_layernorm"

        # 5. MLP Block
        print("\n  📍 Step 5: MLP Block")
        mlx_mlp_out = mlx_layer.mlp(mlx_post_norm_out)
        with torch.no_grad():
            hf_mlp_out = hf_layer.mlp(hf_post_norm_out)

        if not compare_activations(mlx_mlp_out, hf_mlp_out, "MLP Output"):
            print("    🎯 ROOT CAUSE: MLP block!")
            # Deep dive into MLP
            mlp_issue = debug_mlp_internals(mlx_layer.mlp, hf_layer.mlp, mlx_post_norm_out, hf_post_norm_out)
            return f"mlp_{mlp_issue}"

        # 6. Second Residual Connection
        print("\n  📍 Step 6: Second Residual Connection")
        mlx_residual2 = mlx_residual1 + mlx_mlp_out
        with torch.no_grad():
            hf_residual2 = hf_residual1 + hf_mlp_out

        if not compare_activations(mlx_residual2, hf_residual2, "Second Residual"):
            print("    🎯 ROOT CAUSE: Second residual connection!")
            return "second_residual"

        print("\n  ✅ All sub-components match - very subtle issue!")
        return "layer_ok"

    except Exception as e:
        print(f"    ❌ Error in layer deep dive: {e}")
        import traceback
        traceback.print_exc()
        return "layer_error"

def run_layer_by_layer_comparison():
    """Main function to run layer-by-layer comparison with deep dive."""
    print("🔍 === LAYER-BY-LAYER DEEP DIVE ANALYSIS ===")
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

        # Create test input
        test_text = "What is the capital of France?"
        input_ids = tokenizer.encode(test_text, return_tensors="pt")[0]

        # Convert to MLX format
        mlx_input = mx.array(input_ids.numpy()).astype(mx.int32).reshape(1, -1)
        hf_input = input_ids.unsqueeze(0)

        print(f"\n🎯 Test input: '{test_text}'")
        print(f"   Input shapes - MLX: {mlx_input.shape}, HF: {hf_input.shape}")

        # Get embeddings
        mlx_hidden = mlx_model.embed_tokens(mlx_input)
        hf_hidden = hf_model.model.embed_tokens(hf_input)

        print(f"\n📊 Starting layer-by-layer analysis...")

        if not compare_activations(mlx_hidden, hf_hidden, "Initial Embeddings"):
            print("💥 CRITICAL: Embeddings diverge!")
            return "embeddings"

        # Process each transformer layer
        num_layers = len(mlx_model.layers)
        print(f"   Processing {num_layers} transformer layers...")

        for i in range(num_layers):
            print(f"\n📍 === Processing Layer {i}/{num_layers-1} ===")

            # Keep reference to layer inputs for deep dive
            mlx_layer_input = mlx_hidden
            hf_layer_input = hf_hidden

            # Forward through layer
            mlx_hidden, _ = mlx_model.layers[i](mlx_hidden)
            with torch.no_grad():
                # HuggingFace requires position_ids for RoPE
                seq_len = hf_hidden.shape[1]
                position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
                hf_hidden = hf_model.model.layers[i](hf_hidden, position_ids=position_ids)[0]

            # Check layer output
            if not compare_activations(mlx_hidden, hf_hidden, f"Layer {i} Output", atol=1e-4):
                print(f"\n💥 DIVERGENCE FOUND IN LAYER {i}!")
                print(f"🔍 Starting deep dive analysis...")

                # Perform deep dive
                issue = debug_layer_internals(
                    mlx_model.layers[i],
                    hf_model.model.layers[i],
                    mlx_layer_input,
                    hf_layer_input,
                    i
                )

                print(f"\n🎯 FINAL DIAGNOSIS: Layer {i} - {issue}")
                return f"layer_{i}_{issue}"

        # Check final components
        print(f"\n📍 === Final Model Components ===")

        # Final LayerNorm
        mlx_final_norm = mlx_model.norm(mlx_hidden)
        with torch.no_grad():
            hf_final_norm = hf_model.model.norm(hf_hidden)

        if not compare_activations(mlx_final_norm, hf_final_norm, "Final LayerNorm"):
            print("💥 DIVERGENCE: Final LayerNorm!")
            return "final_layernorm"

        # Language model head
        mlx_logits = mlx_model.lm_head(mlx_final_norm) if mlx_model.lm_head else mx.matmul(mlx_final_norm, mlx_model.embed_tokens.weight.T)
        with torch.no_grad():
            hf_logits = hf_model.lm_head(hf_final_norm)

        if not compare_activations(mlx_logits, hf_logits, "Final Logits", atol=1e-3):
            print("💥 DIVERGENCE: Language model head!")
            return "lm_head"

        print("\n🎉 ALL COMPONENTS MATCH!")
        return "all_match"

    except Exception as e:
        print(f"💥 CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        return "error"

if __name__ == "__main__":
    result = run_layer_by_layer_comparison()
    print(f"\n🎯 Final diagnosis: {result}")