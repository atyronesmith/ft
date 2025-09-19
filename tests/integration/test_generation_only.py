"""
Test for generation-only functionality without training.

This test focuses on validating model generation capabilities
to isolate generation issues from training complications.
"""

import os
import sys
from pathlib import Path

import mlx.core as mx
import pytest

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from finetune.inference.generation import GenerationConfig, generate_text, load_model_and_tokenizer

VERBOSE = os.environ.get("FT_VERBOSE", "0") == "1"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _vprint(msg: str):
    if VERBOSE:
        print(f"[GEN] {msg}")


@pytest.mark.integration
def test_generation_only():
    """Test model generation without any training."""
    _vprint("🧪 Testing generation-only functionality...")

    # Load model and tokenizer with special tokens properly registered
    _vprint(f"Loading model with special tokens: {MODEL_ID}")
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    model.eval()

    _vprint(f"✅ Model loaded with {model.num_parameters:,} parameters")
    _vprint(f"✅ Tokenizer loaded with vocab size: {len(tokenizer.get_vocab())}")

    # Test questions and expected answers
    test_cases = [
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Germany?", "Berlin"),
        ("What is the capital of Italy?", "Rome"),
        ("What is the capital of Spain?", "Madrid"),
        ("What is the capital of Portugal?", "Lisbon"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What is the capital of China?", "Beijing"),
        ("What is the capital of India?", "New Delhi"),
        ("What is the capital of Australia?", "Canberra"),
        ("What is the capital of Canada?", "Ottawa"),
    ]

    _vprint(f"\n🎯 Testing {len(test_cases)} capital questions...")

    # Test different generation strategies including Ollama defaults
    strategies = [
        ("Greedy (temp=0.0)", GenerationConfig(max_tokens=15, temperature=0.0, verbose=True)),
        ("Ollama Defaults", GenerationConfig.ollama_defaults()),
        ("Ollama for Q&A", GenerationConfig.for_factual_qa()),
        (
            "Conservative (temp=0.3)",
            GenerationConfig(
                max_tokens=15, temperature=0.3, top_p=0.95, top_k=40, repetition_penalty=1.1
            ),
        ),
    ]

    for strategy_name, config in strategies:
        _vprint(f"\n📊 Testing strategy: {strategy_name}")
        _vprint(
            f"Config: temp={config.temperature}, top_p={config.top_p}, top_k={config.top_k}, rep_penalty={config.repetition_penalty}, max_tokens={config.max_tokens}"
        )

        correct_answers = 0
        total_questions = len(test_cases)

        for i, (question, expected) in enumerate(test_cases, 1):
            try:
                # Clear cache between generations for consistency
                mx.eval(model.parameters())
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()

                # Generate answer
                generated = generate_text(
                    model, tokenizer, question, config, debug_fn=_vprint if VERBOSE else None
                )

                # Check if correct answer is contained in generation
                is_correct = expected.lower() in generated.lower()
                status = "✅" if is_correct else "❌"

                if is_correct:
                    correct_answers += 1

                _vprint(f"{status} Q{i}: {question}")
                _vprint(f"   Expected: {expected}")
                _vprint(f"   Generated: {generated}")
                _vprint("")

            except Exception as e:
                _vprint(f"❌ Q{i}: ERROR - {e}")
                _vprint(f"   Question: {question}")
                _vprint("")

        accuracy = (correct_answers / total_questions) * 100
        _vprint(
            f"📈 {strategy_name} Accuracy: {correct_answers}/{total_questions} = {accuracy:.1f}%"
        )

        # Log summary for this strategy
        if accuracy >= 80:
            _vprint(f"🎉 EXCELLENT: {strategy_name} achieved high accuracy!")
        elif accuracy >= 50:
            _vprint(f"✅ GOOD: {strategy_name} achieved decent accuracy")
        elif accuracy >= 20:
            _vprint(f"⚠️  POOR: {strategy_name} needs improvement")
        else:
            _vprint(f"❌ FAILED: {strategy_name} performed very poorly")

    _vprint("\n🏁 Generation-only test completed!")


def test_single_question_detailed():
    """Test a single question with detailed debugging."""
    _vprint("🔍 Detailed single question test...")

    # Load model with special tokens properly registered
    model, tokenizer = load_model_and_tokenizer(MODEL_ID)
    model.eval()

    question = "What is the capital of France?"
    expected = "Paris"

    _vprint(f"Question: {question}")
    _vprint(f"Expected: {expected}")

    # Test with greedy decoding for maximum determinism
    config = GenerationConfig(max_tokens=10, temperature=0.0, verbose=True)  # Pure greedy

    _vprint("\n🎯 Testing with pure greedy decoding (temp=0.0):")

    # Clear cache for clean state
    mx.eval(model.parameters())
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()

    generated = generate_text(model, tokenizer, question, config, debug_fn=_vprint)

    _vprint(f"Final result: '{generated}'")
    contains_answer = expected.lower() in generated.lower()
    _vprint(f"Contains '{expected}': {contains_answer}")

    if contains_answer:
        _vprint("🎉 SUCCESS: Generated text contains the expected answer!")
    else:
        _vprint("❌ FAILURE: Generated text does not contain the expected answer")

    return contains_answer


if __name__ == "__main__":
    # Enable verbose output when run directly
    os.environ["FT_VERBOSE"] = "1"

    print("Running generation-only tests...")
    test_generation_only()
    print("\nRunning detailed single question test...")
    test_single_question_detailed()
