#!/usr/bin/env python3
"""
Debug script to check LoRA weight loading and model state.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from finetune.training.workflow import create_quick_workflow
from finetune.training.lora import load_lora_weights
import mlx.core as mx

def debug_lora_loading():
    """Debug LoRA weight loading process."""
    print("🔍 Debugging LoRA Weight Loading")
    print("=" * 50)

    # Find the most recent training run
    repo_root = Path.cwd()
    training_dir = repo_root / "training"

    print(f"Looking for training runs in: {training_dir}")

    if not training_dir.exists():
        print("❌ No training directory found")
        return

    # Find most recent run
    training_runs = []
    for item in training_dir.iterdir():
        if item.is_dir():
            final_model_path = item / "final_model"
            if final_model_path.exists() and (final_model_path / "lora_weights.npz").exists():
                mtime = final_model_path.stat().st_mtime
                training_runs.append((mtime, final_model_path))

    if not training_runs:
        print("❌ No training runs with LoRA weights found")
        return

    # Get most recent
    training_runs.sort(key=lambda x: x[0], reverse=True)
    most_recent = training_runs[0][1]
    lora_path = most_recent / "lora_weights.npz"

    print(f"✅ Found most recent LoRA weights: {lora_path}")
    print(f"   Training run: {most_recent.parent.name}")

    # Load base model first
    print("\n📥 Loading base model...")
    workflow = create_quick_workflow(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        data_file="dummy",
        template="chatml",
        output_dir="/tmp/dummy",
    )

    workflow.model, workflow.tokenizer, _ = workflow.model_manager.load_model(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        load_in_4bit=workflow.config.model.load_in_4bit,
    )

    print("✅ Base model loaded")

    # Check base model parameters before LoRA
    print("\n🔍 Checking base model parameters BEFORE LoRA...")
    base_params = workflow.model.parameters()

    # Sample a few key parameters to check their values
    sample_params = {}
    param_keys = list(base_params.keys())[:5]  # First 5 parameters
    for key in param_keys:
        param = base_params[key]
        if hasattr(param, 'shape') and hasattr(param, 'mean'):
            sample_params[f"base_{key}"] = {
                'shape': param.shape,
                'mean': float(param.mean()),
                'std': float(mx.std(param)),
                'min': float(mx.min(param)),
                'max': float(mx.max(param))
            }
            print(f"  {key}: shape={param.shape}, mean={sample_params[f'base_{key}']['mean']:.6f}")

    # Apply LoRA to model
    print("\n🔧 Adding LoRA layers to model...")
    from finetune.training.lora import LoRAConfig
    lora_config = LoRAConfig(
        r=workflow.config.lora.r,
        alpha=workflow.config.lora.alpha,
        dropout=workflow.config.lora.dropout,
        target_modules=workflow.config.lora.target_modules,
    )
    workflow.model.add_lora(lora_config)
    print("✅ LoRA layers added")

    # Check parameters after LoRA addition but before loading weights
    print("\n🔍 Checking model parameters AFTER LoRA addition (before loading weights)...")
    lora_trainable_params, trainable_count, total_count = workflow.model.get_lora_params()
    print(f"   Trainable parameters: {trainable_count:,} / {total_count:,} ({100*trainable_count/total_count:.2f}%)")

    # Check what's in the LoRA weights file
    print(f"\n📁 Checking LoRA weights file: {lora_path}")
    try:
        weights_dict = mx.load(str(lora_path))
        print(f"   Found {len(weights_dict)} weight entries")

        # Show first few weight keys
        weight_keys = list(weights_dict.keys())
        print("   Weight keys:")
        for i, key in enumerate(weight_keys[:10]):  # First 10 keys
            weight = weights_dict[key]
            print(f"     {i+1}. {key}: shape={weight.shape}, dtype={weight.dtype}")

            # Check for NaN/Inf in LoRA weights
            if mx.any(mx.isnan(weight)):
                print(f"        ❌ WARNING: NaN values found in {key}")
            if mx.any(mx.isinf(weight)):
                print(f"        ❌ WARNING: Inf values found in {key}")

        if len(weight_keys) > 10:
            print(f"     ... and {len(weight_keys) - 10} more")

    except Exception as e:
        print(f"   ❌ Error loading weights file: {e}")
        return

    # Load LoRA weights
    print("\n📥 Loading LoRA weights into model...")
    try:
        load_lora_weights(workflow.model, lora_path)
        print("✅ LoRA weights loaded successfully")
    except Exception as e:
        print(f"❌ Error loading LoRA weights: {e}")
        import traceback
        print(traceback.format_exc())
        return

    # Check parameters after LoRA loading
    print("\n🔍 Checking model parameters AFTER LoRA weight loading...")
    loaded_params = workflow.model.parameters()

    # Compare before/after for same parameters
    for key in param_keys:
        if key in loaded_params:
            param = loaded_params[key]
            if hasattr(param, 'shape') and hasattr(param, 'mean'):
                after_stats = {
                    'mean': float(param.mean()),
                    'std': float(mx.std(param)),
                    'min': float(mx.min(param)),
                    'max': float(mx.max(param))
                }

                base_key = f"base_{key}"
                if base_key in sample_params:
                    before_mean = sample_params[base_key]['mean']
                    after_mean = after_stats['mean']
                    change = abs(after_mean - before_mean)

                    print(f"  {key}:")
                    print(f"    Before: mean={before_mean:.6f}")
                    print(f"    After:  mean={after_mean:.6f}")
                    print(f"    Change: {change:.6f}")

                    if change > 1e-6:
                        print(f"    ✅ Parameter changed (LoRA applied)")
                    else:
                        print(f"    ⚠️  No significant change")

    # Test quick generation to see if gibberish appears
    print("\n🎯 Testing generation with LoRA-loaded model...")
    from finetune.utils.chat import apply_chat_template_for_inference
    from finetune.inference.generation import GenerationConfig, generate_text

    question = "What is the capital of France?"
    prompt = apply_chat_template_for_inference(workflow.tokenizer, question)

    config = GenerationConfig(
        max_tokens=20,
        temperature=0.8,
        top_p=0.9,
        verbose=False,
        stop_on_eos=True,
        stop_on_special_tokens=True
    )

    response = generate_text(workflow.model, workflow.tokenizer, prompt, config)
    print(f"   Question: {question}")
    print(f"   Response: '{response}'")

    # Check for gibberish patterns
    gibberish_patterns = ["AccessorImpl", "Ligações", "Хронологи", "архиви", "]{'"]
    has_gibberish = any(pattern in response for pattern in gibberish_patterns)

    if has_gibberish:
        print("   ❌ GIBBERISH DETECTED after LoRA loading!")
        print("   This confirms the issue is with LoRA weight loading/application")
    else:
        print("   ✅ Response looks reasonable")

if __name__ == "__main__":
    debug_lora_loading()