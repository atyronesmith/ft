#!/usr/bin/env python3
"""
Generation test script for the most recent trained model.

This script:
1. Finds the most recent training run
2. Loads the base model + LoRA weights
3. Tests generation on the ACTUAL training/validation questions used during training
4. Reports accuracy and performance metrics

Usage:
    python scripts/test_model_generation.py [base_model_name] [test_data_dir]

Example:
    python scripts/test_model_generation.py TinyLlama/TinyLlama-1.1B-Chat-v1.0 /tmp/test_runs
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


def detect_base_model_from_training_log(training_run_dir: Path) -> str | None:
    """Extract the base model name from training_log.json."""
    training_log_path = training_run_dir / "training_log.json"

    if not training_log_path.exists():
        return None

    try:
        with open(training_log_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)

        model_id = log_data.get('model_id')
        if model_id:
            print(f"🔍 Auto-detected base model from training log: {model_id}")
            return model_id

    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠️  Could not parse training log: {e}")
        return None

    return None


def detect_lora_config_from_training(training_run_dir: Path) -> dict | None:
    """Extract the actual LoRA configuration used during training."""
    lora_config_path = training_run_dir / "final_model" / "lora_config.json"

    if not lora_config_path.exists():
        print(f"⚠️  No lora_config.json found at {lora_config_path}")
        return None

    try:
        with open(lora_config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # Calculate scale from alpha and rank
        alpha = config_data.get('alpha', 16)
        rank = config_data.get('r', 8)
        scale = alpha / rank

        print(f"🔍 Auto-detected LoRA config: rank={rank}, alpha={alpha}, scale={scale:.1f}")
        return {
            'rank': rank,
            'alpha': alpha,
            'scale': scale,
            'dropout': config_data.get('dropout', 0.0),
            'target_modules': config_data.get('target_modules', ['q_proj', 'v_proj'])
        }

    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠️  Could not parse LoRA config: {e}")
        return None


def load_model_with_lora(base_model_name: str, adapter_path: Path):
    """Load the base model and apply LoRA weights using WORKING tests/lora approach."""
    # Import the working utilities directly from tests/lora
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "lora"))
    import utils as lora_utils
    from models import LoRALinear
    import mlx.core as mx
    from mlx.utils import tree_flatten

    print(f"🤖 Loading base model using WORKING tests/lora approach: {base_model_name}")

    # Use the EXACT same loading approach as working script
    tokenizer_config = {}
    model, tokenizer, config = lora_utils.load(base_model_name, tokenizer_config)
    print(f"✅ Model loaded with vocabulary: {len(tokenizer.get_vocab())} tokens")

    print("🔧 Adding LoRA layers using EXACT working pattern...")

    # Freeze entire model first (exactly like working script)
    model.freeze()

    # Apply LoRA to last 16 layers (EXACT same as working script)
    lora_layers = 16
    layers = model.model.layers
    start_layer = len(layers) - lora_layers

    print(f"Applying LoRA to layers {start_layer} through {len(layers)-1} (last {lora_layers} layers)")

    # Apply LoRA using EXACT same pattern as working script
    for l in layers[start_layer:]:
        l.self_attn.q_proj = LoRALinear.from_linear(l.self_attn.q_proj)
        l.self_attn.v_proj = LoRALinear.from_linear(l.self_attn.v_proj)
        if hasattr(l, "block_sparse_moe"):
            l.block_sparse_moe.gate = LoRALinear.from_linear(l.block_sparse_moe.gate)

    # Print parameter count (same as working script)
    p = sum(v.size for _, v in tree_flatten(model.parameters())) / 10**6
    print(f"Total parameters {p:.3f}M")
    p = sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 10**6
    print(f"Trainable parameters {p:.3f}M")

    print(f"📥 Loading LoRA weights from: {adapter_path}")

    # Use EXACT same weight loading as working script (simple and direct)
    model.load_weights(str(adapter_path), strict=False)
    print("✅ LoRA weights loaded using EXACT working approach")

    return model, tokenizer


def load_training_data_questions(training_run_path: str = None, use_validation: bool = True, dataset_name: str = "mlx_examples") -> list[tuple[str, str, str]]:
    """
    Load questions from standardized dataset or training run data.

    Args:
        training_run_path: Path to specific training run (legacy support)
        use_validation: Whether to use validation or training split
        dataset_name: Name of standardized dataset to use

    Returns:
        List of (question, expected_answer, table_context) tuples
    """
    # Try to use standardized data loader first
    if training_run_path is None:
        try:
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from finetune.data.loaders import load_valid_data, load_train_data

            print(f"📖 Loading questions from standardized dataset: {dataset_name}")

            if use_validation:
                data = load_valid_data(dataset_name)
                print(f"✅ Loaded {len(data)} validation questions from {dataset_name}")
            else:
                data = load_train_data(dataset_name)
                print(f"✅ Loaded {len(data)} training questions from {dataset_name}")

            questions = []
            for item in data:
                if "messages" in item:
                    # Chat format
                    messages = item["messages"]
                    user_msg = None
                    assistant_msg = None

                    for msg in messages:
                        if msg.get('role') == 'user':
                            user_msg = msg.get('content', '').strip()
                        elif msg.get('role') == 'assistant':
                            assistant_msg = msg.get('content', '').strip()

                    if user_msg and assistant_msg:
                        # Extract expected answer
                        expected = assistant_msg
                        if " is " in expected:
                            parts = expected.split(" is ")
                            if len(parts) >= 2:
                                expected = parts[-1].rstrip('.').strip()
                        questions.append((user_msg, expected, ""))

                elif "text" in item:
                    # MLX text format - extract table context, Q: and A: parts
                    text = item["text"]
                    if "Q: " in text and "A: " in text:
                        # Split on Q: to get table context and question/answer
                        parts = text.split("Q: ")
                        if len(parts) >= 2:
                            table_context = parts[0].strip()  # Everything before "Q:"
                            q_and_a = parts[-1]  # Get the last Q: part
                            if "A: " in q_and_a:
                                q_part, a_part = q_and_a.split("A: ", 1)
                                question = q_part.strip()
                                answer = a_part.strip()
                                questions.append((question, answer, table_context))

            return questions

        except Exception as e:
            print(f"⚠️  Could not load from standardized dataset: {e}")
            if training_run_path is None:
                raise

    # Legacy fallback: load from training run directory
    import json
    from pathlib import Path

    training_path = Path(training_run_path)

    # Use validation data first (smaller, more focused), fall back to training data
    data_file = training_path / "data" / ("val.jsonl" if use_validation else "train.jsonl")

    if not data_file.exists():
        # Try the other file if preferred doesn't exist
        alt_file = training_path / "data" / ("train.jsonl" if use_validation else "val.jsonl")
        if alt_file.exists():
            data_file = alt_file
            print(f"⚠️  Using {'training' if use_validation else 'validation'} data instead")
        else:
            raise FileNotFoundError(f"No training data found in {training_path / 'data'}")

    questions = []

    print(f"📖 Loading questions from: {data_file}")

    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                messages = data.get('messages', [])

                # Extract user question and assistant answer
                user_msg = None
                assistant_msg = None

                for msg in messages:
                    if msg.get('role') == 'user':
                        user_msg = msg.get('content', '').strip()
                    elif msg.get('role') == 'assistant':
                        assistant_msg = msg.get('content', '').strip()

                if user_msg and assistant_msg:
                    # Extract the expected answer (e.g., "The capital of France is Paris." -> "Paris")
                    expected = assistant_msg
                    # Try to extract just the city name if it follows the pattern
                    if " is " in expected:
                        parts = expected.split(" is ")
                        if len(parts) >= 2:
                            # Get the part after "is" and clean it up
                            expected = parts[-1].rstrip('.').strip()

                    questions.append((user_msg, expected, ""))

            except json.JSONDecodeError as e:
                print(f"⚠️  Skipping malformed line {line_num}: {e}")
                continue

    print(f"✅ Loaded {len(questions)} questions from training data")
    return questions


def generate_answer(
    model, tokenizer, question: str, table_context: str = "", max_tokens: int = 50, debug: bool = True
) -> str:
    """Generate an answer using EXACT same approach as working tests/lora script."""
    import signal
    # Use the working lora_utils from tests/lora
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests", "lora"))
    import utils as lora_utils
    import mlx.core as mx

    def timeout_handler(signum, frame):
        raise TimeoutError("Generation timed out")

    try:
        # Set a 30-second timeout for generation
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)

        if debug:
            print(f"\n{'='*60}")
            print(f"🔍 GENERATION DEBUG for: {question}")
            print(f"{'='*60}")

        # Use original raw text format (not ChatML) to match working script
        if table_context:
            prompt = f"{table_context}\nQ: {question}\nA: "
        else:
            # Fallback for chat format or missing context
            prompt = f"table: 1-10015132-16\ncolumns: Player, No., Nationality, Position, Years in Toronto, School/Club Team\nQ: {question}\nA: "

        if debug:
            print(f"📝 Prompt: {prompt}")

        # Use EXACT same generation approach as working script
        print(prompt, end="", flush=True)

        prompt_array = mx.array(tokenizer.encode(prompt))

        tokens = []
        skip = 0

        # Use working lora_utils.generate() with EXACT same parameters (temp=0.8)
        for token, n in zip(
            lora_utils.generate(prompt_array, model, temp=0.8),
            range(max_tokens),
        ):
            if token == tokenizer.eos_token_id:
                break

            tokens.append(token.item())
            s = tokenizer.decode(tokens)
            if len(s) - skip > 1:
                print(s[skip:-1], end="", flush=True)
                skip = len(s) - 1

        # Final decode and print
        final_response = tokenizer.decode(tokens)[skip:]
        print(final_response, flush=True)
        print("=" * 10)

        if len(tokens) == 0:
            print("No tokens generated for this prompt")
            response = ""
        else:
            response = tokenizer.decode(tokens)

        if debug:
            print(f"🤖 Final Response: '{response}'")
            print(f"{'='*60}")

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


def evaluate_accuracy(questions: list[tuple[str, str, str]], answers: list[str]) -> dict:
    """Evaluate the accuracy of generated answers."""
    correct = 0
    partial = 0
    results = []

    for i, ((question, expected, table_context), generated) in enumerate(zip(questions, answers, strict=False)):
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

    parser = argparse.ArgumentParser(description="Test model generation on actual training questions")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model name (if not provided, will auto-detect from most recent training run)",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Path to adapter.npz file (if not provided, will use adapter from most recent training run)",
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
    parser.add_argument(
        "--max-tokens", type=int, default=50, help="Maximum tokens to generate (default: 50, same as working script)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducible generation (default: 0, same as working script)"
    )

    args = parser.parse_args()

    # Set random seed for reproducible generation (same as working script)
    import numpy as np
    np.random.seed(args.seed)
    print(f"🎲 Random seed set to: {args.seed}")

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

    # Auto-detect base model if not provided
    base_model = args.base_model
    if base_model is None:
        training_run_dir = most_recent.parent  # Go from final_model back to run-XXXXX
        base_model = detect_base_model_from_training_log(training_run_dir)
        if base_model is None:
            print("❌ Could not auto-detect base model from training log!")
            print(f"   Please specify --base-model manually")
            return 1

    # Determine adapter path
    adapter_path = args.adapter
    if adapter_path is None:
        # Use default location from most recent training run
        adapter_path = most_recent / "lora_weights.npz"
        print(f"🔧 Using adapter from training run: {adapter_path}")
    else:
        adapter_path = Path(adapter_path)
        print(f"🔧 Using specified adapter: {adapter_path}")

    if not adapter_path.exists():
        print(f"❌ Adapter file not found: {adapter_path}")
        return 1

    # Load model with LoRA weights
    try:
        model, tokenizer = load_model_with_lora(base_model, adapter_path)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1

    # Load questions from standardized dataset first, fall back to training run data
    print("📝 Loading questions from data...")
    try:
        # Try standardized dataset first
        all_questions = load_training_data_questions(training_run_path=None, use_validation=True, dataset_name="mlx_examples")
    except Exception as e:
        print(f"⚠️  Could not load standardized dataset: {e}")
        print("📝 Falling back to training run data...")
        # Fallback to training run data
        training_run_dir = most_recent.parent  # Go from final_model back to run-XXXXX
        all_questions = load_training_data_questions(str(training_run_dir), use_validation=True)
    questions = all_questions[: args.limit]  # Limit questions if specified
    print(f"✅ Loaded {len(questions)} questions from training data (limited from {len(all_questions)})")

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
    for i, (question, expected, table_context) in enumerate(questions):
        if not args.debug and i % 10 == 0:  # Progress every 10 questions (unless in debug mode)
            print(f"   Progress: {i}/{len(questions)} questions processed...")

        if args.debug:
            print(f"\n🎯 Question {i+1}/{len(questions)}: Testing '{question}'")
            print(f"🎯 Expected answer: '{expected}'")

        answer = generate_answer(model, tokenizer, question, table_context, max_tokens=args.max_tokens, debug=args.debug)
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
