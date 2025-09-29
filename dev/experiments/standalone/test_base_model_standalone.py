#!/usr/bin/env python3
"""
Standalone test to verify base model generation works without any LoRA training.
This isolates the base model functionality from any fine-tuning complications.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from finetune.training.workflow import create_quick_workflow
from finetune.utils.chat import apply_chat_template_for_inference
from finetune.inference.generation import GenerationConfig, generate_text

def test_base_model_standalone():
    """Test base model generation without any LoRA or fine-tuning."""
    print("🧪 Standalone Base Model Generation Test")
    print("=" * 60)
    print("This test verifies that the base TinyLlama model can generate")
    print("coherent text without any LoRA or fine-tuning applied.")
    print("=" * 60)

    # Create workflow for base model loading only
    print("\n📥 Loading base TinyLlama model...")
    workflow = create_quick_workflow(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        data_file="dummy",
        template="chatml",
        output_dir="/tmp/dummy",
    )

    # Load model WITHOUT any LoRA or fine-tuning
    workflow.model, workflow.tokenizer, _ = workflow.model_manager.load_model(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        load_in_4bit=workflow.config.model.load_in_4bit,
    )

    print(f"✅ Base model loaded successfully")
    print(f"   Vocabulary size: {len(workflow.tokenizer.get_vocab())} tokens")
    print(f"   Model type: {workflow.model.config.model_type}")
    print(f"   Hidden size: {workflow.model.config.hidden_size}")
    print(f"   Num layers: {workflow.model.config.num_hidden_layers}")

    # Test questions - SAME as transformers test for direct comparison
    test_cases = [
        {
            "question": "What is the capital of France?",
            "expected_answer": "Paris",
            "category": "Geography"
        },
        {
            "question": "What is 2 + 2?",
            "expected_answer": "4",
            "category": "Math"
        },
        {
            "question": "What color is the sky?",
            "expected_answer": "blue",
            "category": "General Knowledge"
        },
        {
            "question": "Name a programming language.",
            "expected_answer": "Python",
            "category": "Programming"
        },
        {
            "question": "What is the largest planet in our solar system?",
            "expected_answer": "Jupiter",
            "category": "Science"
        }
    ]

    print(f"\n🎯 Testing Base Model on {len(test_cases)} Questions:")
    print("-" * 60)

    results = []

    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        expected_answer = test_case["expected_answer"]
        category = test_case["category"]

        print(f"\n{i}. [{category}] {question}")
        print(f"   Expected: {expected_answer}")

        try:
            # CRITICAL FIX: Use simple prompt format that works for base models
            # Chat template confuses base models - use direct question format
            prompt = f"Question: {question}\nAnswer:"

            print(f"   Chat template: {repr(prompt[:50])}...")

            # Generate with similar settings to transformers test
            config = GenerationConfig(
                max_tokens=100,  # Same as transformers test
                temperature=0.7,  # Same as transformers test
                top_p=0.95,       # Same as transformers test
                verbose=False,
                stop_on_eos=True,
                stop_on_special_tokens=True
            )

            # Use simple generation that follows MLX examples pattern
            from finetune.inference.generation import MLXTextGenerator
            generator = MLXTextGenerator(workflow.model, workflow.tokenizer)
            response = generator.generate_simple(prompt, max_tokens=config.max_tokens, temperature=config.temperature)

            # Clean up response - extract just the assistant part
            if "<|assistant|>" in response:
                assistant_response = response.split("<|assistant|>", 1)[1]
            else:
                assistant_response = response

            if "</s>" in assistant_response:
                assistant_response = assistant_response.split("</s>", 1)[0]

            assistant_response = assistant_response.strip()

            print(f"   Generated: '{assistant_response}'")

            # Check if correct (same logic as transformers test)
            is_correct = expected_answer.lower() in assistant_response.lower()
            print(f"   Correct: {'✅' if is_correct else '❌'}")

            # Check response quality
            response_lower = assistant_response.lower()

            # Check for gibberish patterns
            gibberish_patterns = [
                "AccessorImpl", "Ligações", "Хронологи", "архиви", "]{'",
                "brázky", "consultato", "homonymes"
            ]
            has_gibberish = any(pattern.lower() in response_lower for pattern in gibberish_patterns)

            # Check if response is reasonable length and contains words
            is_empty = len(assistant_response.strip()) == 0
            is_reasonable_length = 3 <= len(assistant_response.split()) <= 200
            has_english_words = any(word.isalpha() and len(word) > 2 for word in assistant_response.split())

            # Check for template artifacts
            template_artifacts = [
                "<|assistant|>", "<|user|>", "<|system|>",
                "assistant|", "user|", "system|"
            ]
            has_template_artifacts = any(artifact in assistant_response for artifact in template_artifacts)

            # Overall assessment
            if has_gibberish:
                status = "❌ GIBBERISH"
                quality = "FAIL"
            elif is_empty:
                status = "❌ EMPTY"
                quality = "FAIL"
            elif has_template_artifacts:
                status = "⚠️  TEMPLATE ARTIFACTS"
                quality = "PARTIAL"
            elif not has_english_words or not is_reasonable_length:
                status = "⚠️  POOR QUALITY"
                quality = "PARTIAL"
            elif is_correct:
                status = "✅ CORRECT"
                quality = "EXCELLENT"
            else:
                status = "⚠️  INCORRECT BUT COHERENT"
                quality = "GOOD"

            print(f"   Status: {status}")

            results.append({
                "question": question,
                "category": category,
                "expected": expected_answer,
                "generated": assistant_response,
                "is_correct": is_correct,
                "quality": quality,
                "has_gibberish": has_gibberish,
                "has_template_artifacts": has_template_artifacts,
                "status": status
            })

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append({
                "question": question,
                "category": category,
                "expected": expected_answer,
                "generated": f"ERROR: {e}",
                "is_correct": False,
                "quality": "ERROR",
                "status": "❌ ERROR"
            })

    # Calculate statistics (same as transformers test)
    total_questions = len(results)
    correct_answers = sum(1 for r in results if r["is_correct"])
    excellent_responses = sum(1 for r in results if r["quality"] == "EXCELLENT")
    good_responses = sum(1 for r in results if r["quality"] == "GOOD")
    partial_responses = sum(1 for r in results if r["quality"] == "PARTIAL")
    failed_responses = sum(1 for r in results if r["quality"] in ["FAIL", "ERROR"])

    accuracy = (correct_answers / total_questions) * 100
    success_rate = ((excellent_responses + good_responses) / total_questions) * 100

    # Summary (same format as transformers test)
    print(f"\n📊 MLX Base Model Test Results")
    print("=" * 60)
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.1f}%")

    print("\nQuality Distribution:")
    print(f"  ✅ Excellent (correct): {excellent_responses}")
    print(f"  ✅ Good (coherent): {good_responses}")
    print(f"  ⚠️  Partial (issues): {partial_responses}")
    print(f"  ❌ Failed (empty/error): {failed_responses}")
    print()
    print(f"Overall Success Rate: {success_rate:.1f}%")

    # Detailed results (same format as transformers test)
    print(f"\n📋 Detailed Question-Answer Pairs:")
    print("=" * 60)
    for i, result in enumerate(results, 1):
        status = "✅" if result['is_correct'] else "❌"
        print(f"{i}. {status} Q: {result['question']}")
        print(f"   Expected: {result['expected']}")
        print(f"   Got: {result['generated'][:100]}{'...' if len(result['generated']) > 100 else ''}")
        print()

    # Final assessment comparison
    print("🎯 Final Assessment:")
    if accuracy >= 80:
        print("   🎉 EXCELLENT - MLX model matches transformers performance!")
        overall_result = "EXCELLENT"
    elif accuracy >= 60:
        print("   ✅ GOOD - MLX model works well")
        overall_result = "GOOD"
    elif success_rate >= 60:
        print("   ⚠️  FAIR - MLX model generates coherent responses but accuracy is low")
        overall_result = "FAIR"
    elif success_rate >= 40:
        print("   ⚠️  POOR - MLX model has significant issues")
        overall_result = "POOR"
    else:
        print("   ❌ FAILED - MLX model cannot generate useful responses")
        overall_result = "FAILED"

    print(f"\nComparison to Transformers (100% accuracy):")
    print(f"MLX Accuracy: {accuracy:.1f}% vs Transformers: 100.0%")
    if accuracy >= 90:
        print("🎉 MLX performance is EXCELLENT!")
    elif accuracy >= 70:
        print("✅ MLX performance is GOOD")
    elif accuracy >= 50:
        print("⚠️  MLX performance needs improvement")
    else:
        print("❌ MLX performance is significantly behind transformers")

    print("=" * 60)

    return results, accuracy, success_rate, overall_result

if __name__ == "__main__":
    try:
        results, accuracy, success_rate, overall_result = test_base_model_standalone()

        # Exit code based on performance (same logic as transformers test)
        if accuracy >= 60:
            print("🎉 Test PASSED!")
            exit_code = 0
        else:
            print("⚠️  Test needs improvement")
            exit_code = 1

        print(f"Final accuracy: {accuracy:.1f}%")
        sys.exit(exit_code)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)