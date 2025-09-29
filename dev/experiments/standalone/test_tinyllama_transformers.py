#!/usr/bin/env python3
"""
TinyLlama Chat Test using transformers pipeline - following the exact pattern from the guide.
"""

from transformers import pipeline
import torch

def test_tinyllama_questions():
    """Test TinyLlama chat model on Q&A tasks using transformers."""

    # 1. Load the pre-trained text-generation pipeline for TinyLlama
    print("Loading the TinyLlama chat model...")
    generator = pipeline(
        "text-generation",
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dtype=torch.bfloat16,  # Updated parameter name
        device=0 if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    )
    print("Model loaded successfully!")

    # Test questions
    questions = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "What color is the sky?",
        "Name a programming language.",
        "What is the largest planet in our solar system?"
    ]

    expected_answers = [
        "Paris",
        "4",
        "blue",
        "Python",
        "Jupiter"
    ]

    print(f"\n🎯 Testing {len(questions)} Questions:")
    print("="*60)

    results = []

    for i, (question, expected) in enumerate(zip(questions, expected_answers), 1):
        print(f"\n{i}. Question: {question}")
        print(f"   Expected: {expected}")

        # 3. Format the prompt using the model's required chat template
        prompt = f"""<|user|>
{question}</s>
<|assistant|>
"""

        # 4. Use the model to generate a response
        print(f"   Generating response...")
        generated_texts = generator(
            prompt,
            max_new_tokens=100, # Controls the length of the *new* text
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
        )

        # 5. Extract the result
        full_response = generated_texts[0]['generated_text']

        # Extract just the assistant's part
        if "<|assistant|>" in full_response:
            assistant_response = full_response.split("<|assistant|>", 1)[1]
        else:
            assistant_response = full_response

        # Clean up
        if "</s>" in assistant_response:
            assistant_response = assistant_response.split("</s>", 1)[0]

        assistant_response = assistant_response.strip()

        print(f"   Generated: {assistant_response}")

        # Check if correct
        is_correct = expected.lower() in assistant_response.lower()
        print(f"   Correct: {'✅' if is_correct else '❌'}")

        results.append({
            'question': question,
            'expected': expected,
            'generated': assistant_response,
            'correct': is_correct
        })

    # Summary
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = (correct_count / len(results)) * 100

    print(f"\n📊 Results Summary:")
    print("="*60)
    print(f"Total Questions: {len(results)}")
    print(f"Correct Answers: {correct_count}")
    print(f"Accuracy: {accuracy:.1f}%")

    print(f"\n📋 Detailed Results:")
    for i, result in enumerate(results, 1):
        status = "✅" if result['correct'] else "❌"
        print(f"{i}. {status} Q: {result['question']}")
        print(f"   Expected: {result['expected']}")
        print(f"   Got: {result['generated'][:100]}{'...' if len(result['generated']) > 100 else ''}")
        print()

    return results, accuracy

if __name__ == "__main__":
    try:
        results, accuracy = test_tinyllama_questions()

        if accuracy >= 60:
            print("🎉 Test PASSED!")
            exit_code = 0
        else:
            print("⚠️ Test needs improvement")
            exit_code = 1

        print(f"Final accuracy: {accuracy:.1f}%")
        exit(exit_code)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(2)