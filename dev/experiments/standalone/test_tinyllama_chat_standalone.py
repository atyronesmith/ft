#!/usr/bin/env python3
"""
Standalone test using TinyLlama chat model directly with transformers.
This tests the model's native chat capabilities using proper prompt templating.
"""

import torch
from transformers import pipeline
import time
import sys

def test_tinyllama_chat_standalone():
    """Test TinyLlama chat model using transformers pipeline."""
    print("🦜 TinyLlama Chat Model Standalone Test")
    print("=" * 60)
    print("This test uses the TinyLlama-1.1B-Chat-v1.0 model directly")
    print("with transformers pipeline and proper chat templating.")
    print("=" * 60)

    # Load the chat model
    print("\n📥 Loading TinyLlama chat model...")
    try:
        # Check if we're on Apple Silicon for MPS support
        if torch.backends.mps.is_available():
            device = "mps"
            print("   Using Apple Silicon MPS acceleration")
        elif torch.cuda.is_available():
            device = "cuda"
            print("   Using CUDA acceleration")
        else:
            device = "cpu"
            print("   Using CPU")

        generator = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance
            device=device,
            trust_remote_code=True,
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        print(traceback.format_exc())
        return None, 0, 0, "FAILED"

    # Test questions for generation
    test_questions = [
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
            "question": "Name a popular programming language.",
            "expected_answer": "Python",
            "category": "Programming"
        },
        {
            "question": "What is the largest planet in our solar system?",
            "expected_answer": "Jupiter",
            "category": "Science"
        }
    ]

    print(f"\n🎯 Testing {len(test_questions)} Questions:")
    print("-" * 60)

    results = []
    total_time = 0

    for i, test_case in enumerate(test_questions, 1):
        question = test_case["question"]
        expected_answer = test_case["expected_answer"]
        category = test_case["category"]

        print(f"\n{i}. [{category}] {question}")
        print(f"   Expected: {expected_answer}")

        try:
            # Format using TinyLlama's chat template
            prompt = f"""<|user|>
{question}</s>
<|assistant|>
"""

            # Generate response
            start_time = time.time()

            generated_texts = generator(
                prompt,
                max_new_tokens=100,  # Length of new text to generate
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=generator.tokenizer.eos_token_id,
            )

            end_time = time.time()
            generation_time = end_time - start_time
            total_time += generation_time

            # Extract the generated response
            full_response = generated_texts[0]['generated_text']

            # Extract just the assistant's response (after the <|assistant|> token)
            if "<|assistant|>" in full_response:
                assistant_response = full_response.split("<|assistant|>", 1)[1].strip()
            else:
                assistant_response = full_response.strip()

            # Clean up the response
            if "</s>" in assistant_response:
                assistant_response = assistant_response.split("</s>", 1)[0].strip()

            print(f"   Generated: {assistant_response}")
            print(f"   Time: {generation_time:.2f}s")

            # Evaluate the response
            response_lower = assistant_response.lower()
            expected_lower = expected_answer.lower()

            # Check if expected answer is in the response
            is_correct = expected_lower in response_lower

            # Check response quality
            is_empty = len(assistant_response.strip()) == 0
            is_reasonable_length = 3 <= len(assistant_response.split()) <= 200
            has_english_words = any(word.isalpha() and len(word) > 2 for word in assistant_response.split())

            # Check for template artifacts
            template_artifacts = ["<|user|>", "<|assistant|>", "<|system|>", "</s>"]
            has_artifacts = any(artifact in assistant_response for artifact in template_artifacts)

            # Overall assessment
            if is_empty:
                status = "❌ EMPTY"
                quality = "FAIL"
            elif has_artifacts:
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
                "generation_time": generation_time,
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
                "generation_time": 0,
                "status": "❌ ERROR"
            })

    # Calculate statistics
    total_questions = len(results)
    correct_answers = sum(1 for r in results if r["is_correct"])
    excellent_responses = sum(1 for r in results if r["quality"] == "EXCELLENT")
    good_responses = sum(1 for r in results if r["quality"] == "GOOD")
    partial_responses = sum(1 for r in results if r["quality"] == "PARTIAL")
    failed_responses = sum(1 for r in results if r["quality"] in ["FAIL", "ERROR"])

    accuracy = (correct_answers / total_questions) * 100
    success_rate = ((excellent_responses + good_responses) / total_questions) * 100
    avg_time = total_time / total_questions if total_questions > 0 else 0

    # Summary
    print(f"\n📊 TinyLlama Chat Test Results")
    print("=" * 60)
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_answers} ({accuracy:.1f}%)")
    print(f"Average Generation Time: {avg_time:.2f}s")
    print(f"Total Time: {total_time:.2f}s")
    print()
    print("Quality Distribution:")
    print(f"  ✅ Excellent (correct): {excellent_responses}")
    print(f"  ✅ Good (coherent): {good_responses}")
    print(f"  ⚠️  Partial (issues): {partial_responses}")
    print(f"  ❌ Failed (empty/error): {failed_responses}")
    print()
    print(f"Overall Success Rate: {success_rate:.1f}%")

    # Detailed results
    print(f"\n📋 Detailed Question-Answer Pairs:")
    print("=" * 60)
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['category']}] {result['question']}")
        print(f"   Expected: {result['expected']}")
        print(f"   Generated: {result['generated'][:150]}{'...' if len(result['generated']) > 150 else ''}")
        print(f"   Correct: {'✅' if result['is_correct'] else '❌'}")
        print(f"   Quality: {result['quality']}")
        print(f"   Time: {result['generation_time']:.2f}s")
        print()

    # Final assessment
    print("🎯 Final Assessment:")
    if accuracy >= 80:
        print("   🎉 EXCELLENT - Model answers questions very accurately!")
        overall_result = "EXCELLENT"
    elif accuracy >= 60:
        print("   ✅ GOOD - Model answers most questions correctly")
        overall_result = "GOOD"
    elif success_rate >= 60:
        print("   ⚠️  FAIR - Model generates coherent responses but accuracy is low")
        overall_result = "FAIR"
    elif success_rate >= 40:
        print("   ⚠️  POOR - Model has significant issues")
        overall_result = "POOR"
    else:
        print("   ❌ FAILED - Model cannot generate useful responses")
        overall_result = "FAILED"

    print(f"\nOverall Result: {overall_result}")
    print("=" * 60)

    return results, accuracy, success_rate, overall_result

if __name__ == "__main__":
    try:
        print("Starting TinyLlama Chat standalone test...")
        result = test_tinyllama_chat_standalone()

        # Handle case where function returns early due to error
        if result is None or len(result) != 4:
            print("❌ Test failed to complete")
            sys.exit(3)

        results, accuracy, success_rate, overall_result = result

        # Exit code based on performance
        if overall_result in ["EXCELLENT", "GOOD"]:
            exit_code = 0
        elif overall_result == "FAIR":
            exit_code = 1
        else:
            exit_code = 2

        print(f"\nTest completed. Exiting with code {exit_code}")
        sys.exit(exit_code)

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(3)