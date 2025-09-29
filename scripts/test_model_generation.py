#!/usr/bin/env python3
"""
Generation test script for the most recent trained model.

This script:
1. Finds the most recent training run
2. Loads the base model + LoRA weights
3. Tests generation on 100 capital questions
4. Reports accuracy and performance metrics

Usage:
    python scripts/test_model_generation.py [base_model_name] [test_data_dir]

Example:
    python scripts/test_model_generation.py microsoft/DialoGPT-small /tmp/test_runs
"""

import json
import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def find_most_recent_training_run(base_dir: Path) -> Path | None:
    """Find the most recent training run with a final_model."""
    if not base_dir.exists():
        return None

    training_runs = []

    # Look for directories containing final_model subdirectories
    for item in base_dir.iterdir():
        if item.is_dir():
            final_model_path = item / "final_model"
            if final_model_path.exists() and (final_model_path / "lora_weights.npz").exists():
                mtime = final_model_path.stat().st_mtime
                training_runs.append((mtime, final_model_path))

    if not training_runs:
        return None

    # Sort by modification time (newest first) and return the most recent
    training_runs.sort(key=lambda x: x[0], reverse=True)
    return training_runs[0][1]


def load_model_with_lora(base_model_name: str, lora_weights_path: Path):
    """Load the base model and apply LoRA weights using the SAME approach as training."""
    from finetune.training.lora import LoRAConfig, load_lora_weights
    from finetune.training.workflow import create_quick_workflow

    print(f"🤖 Loading base model: {base_model_name}")

    # Create a temporary workflow to load the model
    temp_workflow = create_quick_workflow(
        model_name=base_model_name,
        data_file="dummy",  # Won't be used for generation
        template="chatml",  # Use ChatML template to match training
        output_dir="/tmp/dummy",
    )

    # CRITICAL FIX: Use the native tokenizer to match the updated training approach
    # Training now uses native tokenizer to avoid vocabulary expansion issues
    print("🔧 Loading model with native TinyLlama tokenizer (same as training)...")
    from transformers import AutoTokenizer

    # Load model with native tokenizer (SAME AS TRAINING - no vocabulary expansion)
    temp_workflow.model, temp_workflow.tokenizer, _ = temp_workflow.model_manager.load_model(
        base_model_name,
        load_in_4bit=temp_workflow.config.model.load_in_4bit,
    )

    # Set pad token if needed (SAME AS TRAINING)
    if temp_workflow.tokenizer.pad_token_id is None:
        temp_workflow.tokenizer.pad_token = temp_workflow.tokenizer.eos_token
    model = temp_workflow.model
    print(f"✅ Model loaded with native vocabulary: {len(temp_workflow.tokenizer.get_vocab())} tokens (matching training)")

    print("🔧 Adding LoRA layers to model using SAME pattern as training...")
    # CRITICAL FIX: Use the EXACT same LoRA application pattern as training
    # Training applies LoRA manually to specific layers only
    from finetune.training.lora import LoRALinear

    lora_config = LoRAConfig(
        r=temp_workflow.config.lora.r,
        alpha=temp_workflow.config.lora.alpha,
        dropout=temp_workflow.config.lora.dropout,
        target_modules=temp_workflow.config.lora.target_modules,
    )

    # Freeze entire model first (MLX examples pattern)
    model.freeze()

    # Apply LoRA to ONLY the last 16 layers (SAME as training)
    lora_layers = 16
    layers = model.layers  # TinyLlama has model.layers
    start_layer = len(layers) - lora_layers

    print(f"Applying LoRA to layers {start_layer} through {len(layers)-1} (last {lora_layers} layers)")

    for layer_idx in range(start_layer, len(layers)):
        layer = layers[layer_idx]

        # Apply to q_proj and v_proj ONLY (SAME as training)
        if hasattr(layer, "self_attn"):
            if hasattr(layer.self_attn, "q_proj"):
                layer.self_attn.q_proj = LoRALinear.from_linear(
                    layer.self_attn.q_proj, rank=lora_config
                )
                print(f"Applied LoRA to layer {layer_idx} q_proj (scale={lora_config.scaling})")

            if hasattr(layer.self_attn, "v_proj"):
                layer.self_attn.v_proj = LoRALinear.from_linear(
                    layer.self_attn.v_proj, rank=lora_config
                )
                print(f"Applied LoRA to layer {layer_idx} v_proj (scale={lora_config.scaling})")

    print(f"📥 Loading LoRA weights from: {lora_weights_path}")

    # Load the trained LoRA weights
    load_lora_weights(model, lora_weights_path)

    print("✅ Model loaded with LoRA weights applied")

    return model, temp_workflow.tokenizer


def generate_100_capital_questions() -> list[tuple[str, str]]:
    """Generate world capital questions using common utilities for consistency."""
    from finetune.utils.chat import get_geography_questions

    # Use common utility to get questions with test countries prioritized
    return get_geography_questions(max_count=100, prioritize_test_countries=True)


def generate_answer(
    model, tokenizer, question: str, max_tokens: int = 20, debug: bool = True
) -> str:
    """Generate an answer using the fine-tuned model with detailed debugging."""
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError("Generation timed out")

    try:
        # Set a 30-second timeout for generation
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        from finetune.inference.generation import GenerationConfig, generate_text

        if debug:
            print(f"\n{'='*60}")
            print(f"🔍 GENERATION DEBUG for: {question}")
            print(f"{'='*60}")

        # Use more conservative settings to avoid hanging
        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=0.7,  # Higher temperature for more diverse generation
            top_p=0.95,
            verbose=debug,  # Enable verbose for debug mode
            stop_on_eos=True,  # Allow stopping on EOS to prevent hanging
            stop_on_special_tokens=True  # Allow stopping on special tokens
        )

        # CRITICAL FIX: Use common utility to ensure same prompt format as training
        from finetune.utils.chat import apply_chat_template_for_inference

        # Get the exact prompt that will be sent to the model (with system message)
        prompt = apply_chat_template_for_inference(tokenizer, question)

        if debug:
            print(
                f"📝 Prompt preview: {prompt[:100]}..."
                if len(prompt) > 100
                else f"📝 Prompt: {prompt}"
            )
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            print(f"🔢 Input tokens: {len(input_ids)}")

        # Generate the response using the properly formatted prompt
        response = generate_text(model, tokenizer, prompt, config)

        if debug:
            print(f"🤖 Response: '{response}'")

        # Clear the alarm
        signal.alarm(0)
        return response

    except TimeoutError:
        signal.alarm(0)  # Clear the alarm
        error_msg = "[Generation Timeout: 30s]"
        if debug:
            print(f"⏰ GENERATION TIMEOUT: Generation took longer than 30 seconds")
            print(f"{'='*60}\n")
        return error_msg
    except Exception as e:
        signal.alarm(0)  # Clear the alarm
        error_msg = f"[Generation Error: {str(e)[:50]}]"
        if debug:
            print(f"❌ GENERATION ERROR: {e}")
            print(f"{'='*60}\n")
        return error_msg


def evaluate_accuracy(questions: list[tuple[str, str]], answers: list[str]) -> dict:
    """Evaluate the accuracy of generated answers."""
    correct = 0
    partial = 0
    results = []

    for i, ((question, expected), generated) in enumerate(zip(questions, answers, strict=False)):
        # Check for exact match (case insensitive)
        expected_lower = expected.lower()
        generated_lower = generated.lower()

        is_correct = expected_lower in generated_lower
        is_partial = any(
            word in generated_lower for word in expected_lower.split() if len(word) > 2
        )

        if is_correct:
            correct += 1
            status = "✅ CORRECT"
        elif is_partial:
            partial += 1
            status = "🟡 PARTIAL"
        else:
            status = "❌ INCORRECT"

        results.append(
            {
                "question": question,
                "expected": expected,
                "generated": generated,
                "correct": is_correct,
                "partial": is_partial,
                "status": status,
            }
        )

    total = len(questions)
    accuracy = correct / total * 100
    partial_rate = partial / total * 100

    return {
        "total_questions": total,
        "correct": correct,
        "partial": partial,
        "incorrect": total - correct - partial,
        "accuracy_percent": accuracy,
        "partial_percent": partial_rate,
        "results": results,
    }


def save_results(results: dict, output_path: Path):
    """Save detailed results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove detailed results for summary
    summary = {k: v for k, v in results.items() if k != "results"}
    summary["sample_results"] = results["results"][:10]  # Save first 10 for reference

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📊 Detailed results saved to: {output_path}")


def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test model generation on capital questions")
    parser.add_argument(
        "--base-model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    parser.add_argument(
        "--test-dir",
        default="training",
        help="Directory to search for training runs (relative to repo root)",
    )
    parser.add_argument(
        "--output", default="generation_test_results.json", help="Output file for results"
    )
    parser.add_argument("--debug", action="store_true", help="Enable detailed debugging output")
    parser.add_argument(
        "--limit", type=int, default=100, help="Limit number of questions to test (default: 100)"
    )

    args = parser.parse_args()

    # Check for FT_E2E_VERBOSE environment variable
    if not args.debug and os.environ.get("FT_E2E_VERBOSE", "0") == "1":
        args.debug = True
        print("🔧 Debug mode enabled via FT_E2E_VERBOSE environment variable")

    print("🧪 Model Generation Test")
    print("=" * 50)

    # Find the most recent training run
    if Path(args.test_dir).is_absolute():
        test_dir = Path(args.test_dir)
    else:
        # Relative to repo root (where script is run from)
        test_dir = Path.cwd() / args.test_dir
    print(f"🔍 Searching for training runs in: {test_dir}")

    most_recent = find_most_recent_training_run(test_dir)
    if most_recent is None:
        print("❌ No training runs found!")
        print(f"   Searched in: {test_dir}")
        print("   Looking for directories with: final_model/lora_weights.npz")
        return 1

    print(f"📁 Most recent training run: {most_recent.parent.name}")
    print(f"📅 Model path: {most_recent}")

    # Load model with LoRA weights
    try:
        model, tokenizer = load_model_with_lora(args.base_model, most_recent / "lora_weights.npz")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1

    # Generate test questions
    print("📝 Generating capital questions...")
    all_questions = generate_100_capital_questions()
    questions = all_questions[: args.limit]  # Limit questions if specified
    print(f"✅ Generated {len(questions)} questions (limited from {len(all_questions)})")

    if args.debug:
        print("🔧 Debug mode enabled - detailed output for each question")
        if args.limit > 10:
            print(
                f"⚠️  Warning: Debug mode with {args.limit} questions will produce a lot of output!"
            )

    # Run generation test
    print("🚀 Starting generation test...")
    start_time = time.time()

    answers = []
    for i, (question, expected) in enumerate(questions):
        if not args.debug and i % 10 == 0:  # Progress every 10 questions (unless in debug mode)
            print(f"   Progress: {i}/{len(questions)} questions processed...")

        if args.debug:
            print(f"\n🎯 Question {i+1}/{len(questions)}: Testing '{question}'")
            print(f"🎯 Expected answer: '{expected}'")

        answer = generate_answer(model, tokenizer, question, debug=args.debug)
        answers.append(answer)

        if args.debug:
            print(f"🎯 Generated answer: '{answer}'")
            print(
                f"🎯 Match check: {'✅ CORRECT' if expected.lower() in answer.lower() else '❌ INCORRECT'}"
            )
            print(f"{'='*60}")
            if i < len(questions) - 1:  # Not the last question
                print("📝 Continuing to next question...\n")
            else:
                print(f"🏁 All {len(questions)} questions completed!")

    end_time = time.time()
    total_time = end_time - start_time

    print(f"⏱️  Generation completed in {total_time:.2f} seconds")
    print(f"📊 Average time per question: {total_time/len(questions):.3f} seconds")

    # Evaluate results
    print("📈 Evaluating accuracy...")
    evaluation = evaluate_accuracy(questions, answers)

    # Print summary
    print("\n🎯 Generation Test Results")
    print("=" * 30)
    print(f"Total Questions: {evaluation['total_questions']}")
    print(f"Correct Answers: {evaluation['correct']} ({evaluation['accuracy_percent']:.1f}%)")
    print(f"Partial Matches: {evaluation['partial']} ({evaluation['partial_percent']:.1f}%)")
    print(f"Incorrect: {evaluation['incorrect']}")
    print(f"Generation Speed: {len(questions)/total_time:.2f} questions/sec")

    # Show sample results
    if args.debug:
        print("\n📋 All Question-Answer Pairs:")
        print("=" * 80)
        for i, result in enumerate(evaluation["results"]):
            print(f"{i+1:3d}. Q: {result['question']}")
            print(f"     Expected: {result['expected']}")
            print(f"     Generated: {result['generated']}")
            print(f"     Status: {result['status']}")
            print()
        print("=" * 80)
    else:
        print("\n📋 Sample Results (first 5):")
        for i, result in enumerate(evaluation["results"][:5]):
            print(f"{i+1}. Q: {result['question']}")
            print(f"   Expected: {result['expected']}")
            print(f"   Generated: {result['generated']}")
            print(f"   Status: {result['status']}")
            print()

    # Save results
    output_path = Path(args.output)
    save_results(evaluation, output_path)

    # Return exit code based on accuracy
    if evaluation["accuracy_percent"] >= 70:
        print(f"🎉 Test PASSED! Accuracy: {evaluation['accuracy_percent']:.1f}%")
        return 0
    elif evaluation["accuracy_percent"] >= 40:
        print(f"⚠️  Test PARTIAL! Accuracy: {evaluation['accuracy_percent']:.1f}%")
        return 0
    else:
        print(f"❌ Test FAILED! Accuracy: {evaluation['accuracy_percent']:.1f}%")
        return 1


if __name__ == "__main__":
    sys.exit(main())
