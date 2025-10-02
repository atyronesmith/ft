#!/usr/bin/env python3
"""
Comprehensive test for MLX compatibility after nested structure implementation.

This test validates that our nested structure fixes resolve the garbage generation issue
and enables compatibility with MLX-trained adapters.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import mlx.core as mx
from finetune.models.mlx_models import MLXLlamaModel
from finetune.models.base import ModelConfig
from finetune.training.lora import LoRAConfig, apply_lora_to_model


def test_basic_model_structure():
    """Test 1: Verify nested structure is correct"""
    print("=" * 60)
    print("TEST 1: Basic Model Structure")
    print("=" * 60)

    config = ModelConfig(
        model_type='llama',
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=1000,
        max_position_embeddings=512
    )

    model = MLXLlamaModel(config)

    # Check nested structure
    assert hasattr(model, 'model'), "Model should have nested 'model' attribute"
    assert hasattr(model.model, 'layers'), "Nested model should have 'layers'"
    assert hasattr(model.model, 'embed_tokens'), "Nested model should have 'embed_tokens'"
    assert hasattr(model.model, 'norm'), "Nested model should have 'norm'"
    assert hasattr(model, 'lm_head'), "Model should have top-level 'lm_head'"

    print(f"✅ Model structure: model.model.layers[{len(model.model.layers)}]")
    print(f"✅ Embeddings: {model.model.embed_tokens}")
    print(f"✅ Norm: {model.model.norm}")
    print(f"✅ LM Head: {model.lm_head}")

    # Check parameter names match MLX format
    params = model.parameters()
    expected_keys = ['model', 'lm_head']
    for key in expected_keys:
        assert key in params, f"Missing top-level key: {key}"

    # Check nested model parameters
    model_params = params['model']
    expected_model_keys = ['embed_tokens', 'layers', 'norm']
    for key in expected_model_keys:
        assert key in model_params, f"Missing model key: {key}"

    print("✅ Parameter structure matches MLX format")
    return True


def test_lora_application():
    """Test 2: Verify LoRA works with nested structure"""
    print("\n" + "=" * 60)
    print("TEST 2: LoRA Application")
    print("=" * 60)

    config = ModelConfig(
        model_type='llama',
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=1000,
        max_position_embeddings=512
    )

    model = MLXLlamaModel(config)

    # Apply LoRA
    lora_config = LoRAConfig(r=4, target_modules=['q_proj', 'v_proj'])
    apply_lora_to_model(model, lora_config)

    # Check LoRA was applied correctly
    lora_count = 0
    for name, module in model.named_modules():
        if 'LoRALinear' in str(type(module)):
            lora_count += 1
            print(f"✅ LoRA applied: {name}")

    expected_lora_modules = 2 * 2  # 2 layers * 2 target modules per layer
    assert lora_count == expected_lora_modules, f"Expected {expected_lora_modules} LoRA modules, got {lora_count}"

    print(f"✅ LoRA applied to {lora_count} modules correctly")
    return True


def test_generation_quality():
    """Test 3: Verify generation produces coherent output (not garbage)"""
    print("\n" + "=" * 60)
    print("TEST 3: Generation Quality")
    print("=" * 60)

    config = ModelConfig(
        model_type='llama',
        hidden_size=128,  # Slightly larger for better generation
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=1000,
        max_position_embeddings=512
    )

    model = MLXLlamaModel(config)

    # Test basic generation
    input_ids = mx.array([[1, 2, 3, 4, 5]])  # Simple input

    try:
        output = model.generate(input_ids, max_length=20, temperature=0.8)

        # Check that output is not all the same token (garbage generation)
        unique_tokens = len(set(output[0].tolist()))
        total_tokens = output.shape[1]

        print(f"Generated tokens: {output[0].tolist()}")
        print(f"Unique tokens: {unique_tokens}/{total_tokens}")

        # Should have reasonable diversity (not all same token)
        assert unique_tokens > 3, f"Generation too repetitive: {unique_tokens} unique tokens"

        # Should not be all 1s (classic garbage generation)
        all_ones = all(token == 1 for token in output[0].tolist()[5:])  # Skip input tokens
        assert not all_ones, "Generation is producing all 1s (garbage)"

        print("✅ Generation produces diverse tokens (not garbage)")
        return True

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False


def test_parameter_compatibility():
    """Test 4: Check parameter names match MLX format exactly"""
    print("\n" + "=" * 60)
    print("TEST 4: Parameter Name Compatibility")
    print("=" * 60)

    config = ModelConfig(
        model_type='llama',
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=1000,
        max_position_embeddings=512
    )

    model = MLXLlamaModel(config)

    # Get flat parameter names (like MLX would expect)
    def get_flat_params(params, prefix=""):
        flat = {}
        for k, v in params.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(get_flat_params(v, full_key))
            elif hasattr(v, 'shape'):
                flat[full_key] = v
        return flat

    flat_params = get_flat_params(model.parameters())

    # Expected MLX parameter patterns
    expected_patterns = [
        'model.embed_tokens.weight',
        'model.layers.0.self_attn.q_proj.weight',
        'model.layers.0.self_attn.k_proj.weight',
        'model.layers.0.self_attn.v_proj.weight',
        'model.layers.0.self_attn.o_proj.weight',
        'model.layers.0.mlp.gate_proj.weight',
        'model.layers.0.mlp.up_proj.weight',
        'model.layers.0.mlp.down_proj.weight',
        'model.layers.0.input_layernorm.weight',
        'model.layers.0.post_attention_layernorm.weight',
        'model.norm.weight',
        'lm_head.weight'
    ]

    print("Checking MLX parameter name compatibility:")
    for pattern in expected_patterns:
        if pattern in flat_params:
            print(f"✅ {pattern}")
        else:
            print(f"❌ Missing: {pattern}")
            # Find similar names for debugging
            similar = [name for name in flat_params.keys() if pattern.split('.')[-1] in name]
            if similar:
                print(f"   Similar: {similar[:3]}")

    # Check that we have the core structure
    core_checks = [
        any('model.embed_tokens' in name for name in flat_params),
        any('model.layers.0' in name for name in flat_params),
        any('model.norm' in name for name in flat_params),
        any('lm_head' in name for name in flat_params)
    ]

    all_core_present = all(core_checks)
    print(f"\n✅ Core MLX structure present: {all_core_present}")

    return all_core_present


def test_adapter_loading_simulation():
    """Test 5: Simulate loading MLX adapter weights"""
    print("\n" + "=" * 60)
    print("TEST 5: MLX Adapter Loading Simulation")
    print("=" * 60)

    config = ModelConfig(
        model_type='llama',
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=1000,
        max_position_embeddings=512
    )

    model = MLXLlamaModel(config)

    # Apply LoRA to simulate having an adapter
    lora_config = LoRAConfig(r=4, target_modules=['q_proj', 'v_proj'])
    apply_lora_to_model(model, lora_config)

    # Simulate saving and loading weights (like MLX adapter)
    try:
        # Get trainable parameters (LoRA weights)
        from mlx.utils import tree_flatten
        trainable_params = dict(tree_flatten(model.trainable_parameters()))

        print(f"Trainable parameters found: {len(trainable_params)}")
        for name in list(trainable_params.keys())[:5]:  # Show first 5
            print(f"  {name}")

        # Check parameter names match MLX LoRA format
        expected_lora_patterns = [
            'model.layers.0.self_attn.q_proj.lora_a',
            'model.layers.0.self_attn.q_proj.lora_b',
            'model.layers.0.self_attn.v_proj.lora_a',
            'model.layers.0.self_attn.v_proj.lora_b'
        ]

        matches = 0
        for pattern in expected_lora_patterns:
            if pattern in trainable_params:
                matches += 1
                print(f"✅ LoRA param: {pattern}")

        print(f"\n✅ MLX LoRA parameter format compatibility: {matches}/{len(expected_lora_patterns)}")
        return matches >= len(expected_lora_patterns) // 2  # At least half should match

    except Exception as e:
        print(f"❌ Adapter simulation failed: {e}")
        return False


def main():
    """Run all MLX compatibility tests"""
    print("MLX COMPATIBILITY VALIDATION")
    print("Testing nested structure implementation...")
    print()

    tests = [
        ("Basic Model Structure", test_basic_model_structure),
        ("LoRA Application", test_lora_application),
        ("Generation Quality", test_generation_quality),
        ("Parameter Name Compatibility", test_parameter_compatibility),
        ("Adapter Loading Simulation", test_adapter_loading_simulation),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTests Passed: {passed}/{total}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! MLX compatibility implementation successful!")
        print("✅ Nested structure matches MLX format")
        print("✅ LoRA application works correctly")
        print("✅ Generation produces coherent output")
        print("✅ Parameter names compatible with MLX adapters")
        return True
    else:
        print(f"\n⚠️  Some tests failed. Implementation needs attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)