#!/usr/bin/env python3
"""
Auto-detecting portable generation test script (like MLX LoRA example).

This script automatically infers LoRA configuration from the .npz weights file,
eliminating the need for separate config files or hardcoded parameters.

Usage:
    python scripts/test_model_generation_auto.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --adapter-file training/run-52537/final_model/lora_weights.npz
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add src to path for minimal imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from transformers import AutoTokenizer, AutoConfig
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    print("❌ MLX not available. This script requires MLX.")
    sys.exit(1)


def infer_lora_config_from_weights(weights_path: str) -> dict:
    """Automatically infer LoRA configuration from weights file (like MLX example)."""
    weights = np.load(weights_path)

    # Infer rank from lora_b matrices
    rank = None
    for key in weights.keys():
        if 'lora_b' in key:
            shape = weights[key].shape
            rank = shape[0]  # First dimension of lora_b is the rank
            break

    # Infer target modules from key patterns
    target_modules = set()
    for key in weights.keys():
        if 'lora_a' in key or 'lora_b' in key:
            parts = key.split('.')
            if len(parts) >= 4:
                module_name = parts[-2]  # e.g., q_proj, v_proj
                target_modules.add(module_name)

    # Infer which layers have LoRA applied
    lora_layers = set()
    for key in weights.keys():
        if 'layers.' in key:
            parts = key.split('.')
            if len(parts) >= 2 and parts[0] == 'layers':
                layer_num = int(parts[1])
                lora_layers.add(layer_num)

    # Use common default for alpha
    alpha = rank * 2.0 if rank else 16.0

    return {
        "rank": rank,
        "alpha": alpha,
        "target_modules": sorted(list(target_modules)),
        "lora_layers": sorted(list(lora_layers)),
        "num_lora_layers": len(lora_layers),
        "layer_range": f"{min(lora_layers)}-{max(lora_layers)}" if lora_layers else "none",
    }


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer directly (like MLX example)."""
    print(f"🤖 Loading model from: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model using our model manager
    from finetune.models.manager import ModelManager
    manager = ModelManager()
    model, _, _ = manager.load_model(model_path)

    print(f"✅ Model loaded: {model.__class__.__name__}")
    print(f"✅ Tokenizer loaded: {len(tokenizer.get_vocab())} tokens")

    return model, tokenizer


def load_adapter_auto(model, adapter_file: str):
    """Auto-load LoRA adapter using common library (no config needed!)."""
    print(f"🤖 Auto-loading LoRA from: {adapter_file}")

    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"Adapter file not found: {adapter_file}")

    from finetune.training.lora import apply_lora_from_weights_auto, get_lora_info_from_weights

    # Get detailed info for display
    lora_info = get_lora_info_from_weights(adapter_file)
    config = lora_info["config"]

    print(f"🔧 Auto-detected LoRA config:")
    print(f"   Rank: {config.r}")
    print(f"   Alpha: {config.alpha}")
    print(f"   Target modules: {config.target_modules}")
    print(f"   Layers: {lora_info['layer_range']} ({lora_info['layer_count']} layers)")

    # Apply LoRA and load weights
    apply_lora_from_weights_auto(model, adapter_file)

    print("✅ LoRA structure applied and weights loaded automatically")

    return config


def get_test_sql_questions():
    """Get SQL test questions (matching our training data format)."""
    return [
        {
            "table": "1-1000181-1",
            "columns": "State/territory, Text/background colour, Format, Current slogan, Current series, Notes",
            "question": "Tell me what the notes are for South Australia",
            "expected": "SELECT Notes FROM 1-1000181-1 WHERE Current slogan = 'SOUTH AUSTRALIA'"
        },
        {
            "table": "1-10007452-3",
            "columns": "Order Year, Manufacturer, Model, Fleet Series (Quantity), Powertrain (Engine/Transmission), Fuel Propulsion",
            "question": "how many times is the fuel propulsion is cng?",
            "expected": "SELECT COUNT Fleet Series (Quantity) FROM 1-10007452-3 WHERE Fuel Propulsion = 'CNG'"
        },
        {
            "table": "1-10015132-1",
            "columns": "Player, No., Nationality, Position, Years in Toronto, School/Club Team",
            "question": "What school did player number 6 come from?",
            "expected": "SELECT School/Club Team FROM 1-10015132-1 WHERE No. = '6'"
        },
        {
            "table": "1-10006830-1",
            "columns": "Aircraft, Description, Max Gross Weight, Total disk area, Max disk Loading",
            "question": "What if the description of a ch-47d chinook?",
            "expected": "SELECT Description FROM 1-10006830-1 WHERE Aircraft = 'CH-47D Chinook'"
        },
        {
            "table": "1-10015132-14",
            "columns": "Player, No., Nationality, Position, Years in Toronto, School/Club Team",
            "question": "Which number was Patrick O'Bryant?",
            "expected": "SELECT No. FROM 1-10015132-14 WHERE Player = 'Patrick O'Bryant'"
        }
    ]


def format_sql_prompt(table: str, columns: str, question: str) -> str:
    """Format prompt exactly like training data."""
    return f"table: {table}\ncolumns: {columns}\nQ: {question}\nA: "


def generate_answer(model, tokenizer, prompt: str, max_tokens: int = 30, temperature: float = 0.1) -> str:
    """Generate answer using MLX (simplified version)."""
    try:
        from finetune.inference.generation import MLXTextGenerator, GenerationConfig

        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False
        )

        generator = MLXTextGenerator(model, tokenizer, config)
        response = generator.generate_simple(prompt, max_tokens=max_tokens, temperature=temperature)

        return response.strip()

    except Exception as e:
        return f"[Error: {str(e)[:50]}]"


def evaluate_sql_quality(generated: str, expected: str) -> dict:
    """Evaluate SQL generation quality."""
    generated_upper = generated.upper()
    expected_upper = expected.upper()

    # Check for SQL keywords
    sql_keywords = ["SELECT", "FROM", "WHERE", "COUNT", "="]
    found_keywords = [kw for kw in sql_keywords if kw in generated_upper]

    # Check structure
    has_select = "SELECT" in generated_upper
    has_from = "FROM" in generated_upper
    has_where = "WHERE" in generated_upper or "=" in generated

    # Check meaningfulness
    is_meaningful = len(generated.strip()) > 5 and not generated.count('1') > len(generated) * 0.7

    # Scoring
    if has_select and has_from and len(found_keywords) >= 3:
        quality = "EXCELLENT"
        score = 3
    elif has_select and (has_from or has_where):
        quality = "GOOD"
        score = 2
    elif has_select and found_keywords:
        quality = "PARTIAL"
        score = 1
    elif is_meaningful and found_keywords:
        quality = "SQL-RELATED"
        score = 1
    elif is_meaningful:
        quality = "MEANINGFUL"
        score = 0
    else:
        quality = "POOR"
        score = 0

    return {
        "quality": quality,
        "score": score,
        "keywords": found_keywords,
        "has_select": has_select,
        "has_from": has_from,
        "has_where": has_where,
        "is_meaningful": is_meaningful
    }


def run_generation_test(model, tokenizer, test_questions: list, debug: bool = False) -> dict:
    """Run generation test on SQL questions."""
    print(f"🚀 Testing generation on {len(test_questions)} SQL questions...")

    results = []
    scores = []
    start_time = time.time()

    for i, question_data in enumerate(test_questions, 1):
        if debug:
            print(f"\n📝 Test {i}/{len(test_questions)}: {question_data['question'][:50]}...")

        # Format prompt
        prompt = format_sql_prompt(
            question_data["table"],
            question_data["columns"],
            question_data["question"]
        )

        # Generate answer
        generated = generate_answer(model, tokenizer, prompt, max_tokens=30, temperature=0.1)

        # Evaluate quality
        evaluation = evaluate_sql_quality(generated, question_data["expected"])
        scores.append(evaluation["score"])

        result = {
            "question": question_data["question"],
            "expected": question_data["expected"],
            "generated": generated,
            "evaluation": evaluation
        }
        results.append(result)

        if debug:
            print(f"🤖 Generated: '{generated}'")
            print(f"📋 Expected:  '{question_data['expected']}'")
            print(f"⭐ Quality: {evaluation['quality']} (score: {evaluation['score']})")
            print(f"🔑 Keywords: {evaluation['keywords']}")

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate summary statistics
    total_score = sum(scores)
    max_score = len(test_questions) * 3  # Max score is 3 per question
    accuracy_percent = (total_score / max_score) * 100

    excellent_count = sum(1 for r in results if r["evaluation"]["quality"] == "EXCELLENT")
    good_count = sum(1 for r in results if r["evaluation"]["quality"] == "GOOD")
    partial_count = sum(1 for r in results if r["evaluation"]["quality"] == "PARTIAL")

    return {
        "total_questions": len(test_questions),
        "total_score": total_score,
        "max_score": max_score,
        "accuracy_percent": accuracy_percent,
        "excellent_count": excellent_count,
        "good_count": good_count,
        "partial_count": partial_count,
        "generation_time": total_time,
        "questions_per_sec": len(test_questions) / total_time,
        "results": results
    }


def print_summary(test_results: dict, lora_config: dict):
    """Print test summary."""
    print("\n🎯 Auto-Detecting SQL Generation Test Results")
    print("=" * 50)
    print(f"LoRA Config: rank={lora_config['rank']}, alpha={lora_config['alpha']}")
    print(f"Target: {lora_config['target_modules']} on layers {lora_config['layer_range']}")
    print(f"Total Questions: {test_results['total_questions']}")
    print(f"Score: {test_results['total_score']}/{test_results['max_score']} ({test_results['accuracy_percent']:.1f}%)")
    print(f"Excellent SQL: {test_results['excellent_count']}")
    print(f"Good SQL: {test_results['good_count']}")
    print(f"Partial SQL: {test_results['partial_count']}")
    print(f"Generation Time: {test_results['generation_time']:.2f}s")
    print(f"Speed: {test_results['questions_per_sec']:.2f} questions/sec")

    # Show sample results
    print("\n📋 Sample Results:")
    for i, result in enumerate(test_results["results"][:3], 1):
        print(f"\n{i}. Q: {result['question']}")
        print(f"   Expected: {result['expected']}")
        print(f"   Generated: {result['generated']}")
        print(f"   Quality: {result['evaluation']['quality']}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Auto-detecting SQL generation test (like MLX LoRA example)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect everything from weights file
  python scripts/test_model_generation_auto.py \\
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
    --adapter-file training/run-52537/final_model/lora_weights.npz

  # With debug output
  python scripts/test_model_generation_auto.py \\
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
    --adapter-file training/run-83378/final_model/lora_weights.npz \\
    --debug
        """
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to model directory or HuggingFace model name"
    )
    parser.add_argument(
        "--adapter-file",
        required=True,
        help="Path to LoRA weights file (*.npz)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug output"
    )
    parser.add_argument(
        "--output",
        default="auto_generation_results.json",
        help="Output file for results"
    )

    args = parser.parse_args()

    if not MLX_AVAILABLE:
        print("❌ MLX not available. Please install MLX.")
        return 1

    print("🧪 Auto-Detecting SQL Generation Test")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Adapter: {args.adapter_file}")
    print("🤖 Auto-detecting LoRA configuration from weights file...")
    print()

    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model)

        # Auto-detect, apply LoRA, and load weights (all in one!)
        lora_config = load_adapter_auto(model, args.adapter_file)

        # Get test questions
        test_questions = get_test_sql_questions()
        print(f"📝 Loaded {len(test_questions)} SQL test questions")

        # Run generation test
        results = run_generation_test(model, tokenizer, test_questions, debug=args.debug)

        # Print summary
        print_summary(results, lora_config)

        # Save results
        output_path = Path(args.output)
        results["lora_config"] = lora_config  # Include auto-detected config
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📊 Detailed results saved to: {output_path}")

        # Determine success
        if results["accuracy_percent"] >= 60:
            print(f"\n🎉 TEST PASSED! Accuracy: {results['accuracy_percent']:.1f}%")
            return 0
        elif results["accuracy_percent"] >= 30:
            print(f"\n⚠️  TEST PARTIAL! Accuracy: {results['accuracy_percent']:.1f}%")
            return 0
        else:
            print(f"\n❌ TEST FAILED! Accuracy: {results['accuracy_percent']:.1f}%")
            return 1

    except Exception as e:
        print(f"❌ Test failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())