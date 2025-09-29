#!/usr/bin/env python3
"""
Compare accuracy between base model and LoRA fine-tuned model on the same test questions.

This test will show us exactly how the LoRA fine-tuning affected accuracy
compared to the base model's 80% performance.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from finetune.models.manager import ModelManager
from finetune.training.lora import LoRALinear, load_lora_weights, LoRAConfig
from finetune.inference.generation import MLXTextGenerator

def test_base_model():
    """Test base model accuracy on standard questions."""
    print("🧪 Testing Base Model (No LoRA)")
    print("-" * 40)

    # Load base model
    manager = ModelManager()
    model, tokenizer, config = manager.load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Same test questions as our base model test
    test_cases = [
        {"question": "What is the capital of France?", "expected": "Paris", "category": "Geography"},
        {"question": "What is 2 + 2?", "expected": "4", "category": "Math"},
        {"question": "What color is the sky?", "expected": "blue", "category": "General Knowledge"},
        {"question": "Name a programming language.", "expected": "Python", "category": "Programming"},
        {"question": "What is the largest planet in our solar system?", "expected": "Jupiter", "category": "Science"}
    ]

    results = []
    correct_count = 0

    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        expected = test_case["expected"]
        category = test_case["category"]

        # Use simple prompt format
        prompt = f"Question: {question}\nAnswer:"

        generator = MLXTextGenerator(model, tokenizer)
        response = generator.generate_simple(prompt, max_tokens=20, temperature=0.7)

        # Check if correct
        is_correct = expected.lower() in response.lower()
        if is_correct:
            correct_count += 1

        status = "✅" if is_correct else "❌"
        print(f"{i}. {status} [{category}] {question}")
        print(f"   Expected: {expected}")
        print(f"   Got: {repr(response[:50])}{'...' if len(response) > 50 else ''}")

        results.append({
            "question": question,
            "expected": expected,
            "response": response,
            "correct": is_correct,
            "category": category
        })

    accuracy = (correct_count / len(test_cases)) * 100
    print(f"\n📊 Base Model Accuracy: {accuracy:.1f}% ({correct_count}/{len(test_cases)})")

    return results, accuracy

def test_lora_model(checkpoint_path, checkpoint_name):
    """Test LoRA fine-tuned model accuracy on the same questions."""
    print(f"\n🧪 Testing LoRA Model: {checkpoint_name}")
    print("-" * 40)

    # Load base model
    manager = ModelManager()
    model, tokenizer, config = manager.load_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # Apply LoRA
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

    # Load LoRA weights
    try:
        load_lora_weights(model, checkpoint_path / "lora_weights.npz")
        print("✅ LoRA weights loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load LoRA weights: {e}")
        return [], 0.0

    # Same test questions as base model
    test_cases = [
        {"question": "What is the capital of France?", "expected": "Paris", "category": "Geography"},
        {"question": "What is 2 + 2?", "expected": "4", "category": "Math"},
        {"question": "What color is the sky?", "expected": "blue", "category": "General Knowledge"},
        {"question": "Name a programming language.", "expected": "Python", "category": "Programming"},
        {"question": "What is the largest planet in our solar system?", "expected": "Jupiter", "category": "Science"}
    ]

    results = []
    correct_count = 0

    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        expected = test_case["expected"]
        category = test_case["category"]

        # Use simple prompt format
        prompt = f"Question: {question}\nAnswer:"

        generator = MLXTextGenerator(model, tokenizer)
        response = generator.generate_simple(prompt, max_tokens=20, temperature=0.7)

        # Check if correct
        is_correct = expected.lower() in response.lower()
        if is_correct:
            correct_count += 1

        status = "✅" if is_correct else "❌"
        print(f"{i}. {status} [{category}] {question}")
        print(f"   Expected: {expected}")
        print(f"   Got: {repr(response[:50])}{'...' if len(response) > 50 else ''}")

        # Check for quality issues
        response_words = response.split()
        if len(set(response_words[:5])) <= 2:
            print(f"   ⚠️  Shows repetition pattern")
        elif not response.strip():
            print(f"   ⚠️  Empty response")

        results.append({
            "question": question,
            "expected": expected,
            "response": response,
            "correct": is_correct,
            "category": category
        })

    accuracy = (correct_count / len(test_cases)) * 100
    print(f"\n📊 LoRA Model Accuracy: {accuracy:.1f}% ({correct_count}/{len(test_cases)})")

    return results, accuracy

def find_working_checkpoints():
    """Find the checkpoints that we know work."""
    training_dir = Path("training")
    checkpoints = []

    # We know these work from our previous testing
    working_runs = ["run-29561", "run-29217"]

    for run_name in working_runs:
        run_path = training_dir / run_name / "final_model"
        if run_path.exists() and (run_path / "lora_weights.npz").exists():
            checkpoints.append((run_path, run_name))

    return checkpoints

def main():
    """Main comparison function."""
    print("🎯 Base Model vs LoRA Fine-Tuned Model Accuracy Comparison")
    print("=" * 70)
    print("Testing both models on the same 5 questions to measure fine-tuning impact")
    print("=" * 70)

    # Test base model first
    base_results, base_accuracy = test_base_model()

    # Find working LoRA checkpoints
    checkpoints = find_working_checkpoints()

    if not checkpoints:
        print("\n❌ No working LoRA checkpoints found")
        return 1

    # Test each working LoRA checkpoint
    lora_results = []

    for checkpoint_path, checkpoint_name in checkpoints:
        lora_result, lora_accuracy = test_lora_model(checkpoint_path, checkpoint_name)
        lora_results.append((checkpoint_name, lora_accuracy, lora_result))

    # Summary comparison
    print("\n" + "=" * 70)
    print("📊 ACCURACY COMPARISON SUMMARY")
    print("=" * 70)

    print(f"Base Model (No LoRA):     {base_accuracy:.1f}%")

    for checkpoint_name, lora_accuracy, _ in lora_results:
        print(f"LoRA Model ({checkpoint_name}): {lora_accuracy:.1f}%")

    # Calculate average LoRA performance
    if lora_results:
        avg_lora_accuracy = sum(acc for _, acc, _ in lora_results) / len(lora_results)
        print(f"Average LoRA Performance: {avg_lora_accuracy:.1f}%")

        # Performance change
        change = avg_lora_accuracy - base_accuracy
        if change > 0:
            print(f"🎉 Fine-tuning IMPROVED performance by {change:.1f} percentage points!")
        elif change < -10:
            print(f"❌ Fine-tuning HURT performance by {abs(change):.1f} percentage points")
        else:
            print(f"⚠️  Fine-tuning had minimal impact ({change:+.1f} percentage points)")

    # Detailed analysis
    print(f"\n📋 DETAILED ANALYSIS:")
    print(f"- Base model achieved {base_accuracy:.1f}% accuracy (established baseline)")
    print(f"- LoRA fine-tuning was done on geography dataset (capital cities)")
    print(f"- Test questions include geography but also other domains")

    # Check geography-specific performance
    geography_questions = [q for q in base_results if q["category"] == "Geography"]
    if geography_questions:
        base_geo_correct = sum(1 for q in geography_questions if q["correct"])
        print(f"- Base model geography accuracy: {(base_geo_correct/len(geography_questions))*100:.0f}%")

        # Check LoRA geography performance
        for checkpoint_name, _, lora_result in lora_results:
            lora_geo_questions = [q for q in lora_result if q["category"] == "Geography"]
            lora_geo_correct = sum(1 for q in lora_geo_questions if q["correct"])
            lora_geo_accuracy = (lora_geo_correct/len(lora_geo_questions))*100 if lora_geo_questions else 0
            print(f"- LoRA model ({checkpoint_name}) geography accuracy: {lora_geo_accuracy:.0f}%")

    print("\n🎯 CONCLUSION:")
    if lora_results and avg_lora_accuracy >= base_accuracy - 5:  # Within 5% is considered successful
        print("✅ LoRA fine-tuning maintains comparable performance to base model")
        if any(acc > base_accuracy for _, acc, _ in lora_results):
            print("✅ Some LoRA checkpoints actually improve on base model performance")
    else:
        print("❌ LoRA fine-tuning significantly degraded model performance")
        print("This confirms our diagnosis that the training was problematic")

    return 0

if __name__ == "__main__":
    sys.exit(main())