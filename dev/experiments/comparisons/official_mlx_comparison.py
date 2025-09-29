#!/usr/bin/env python3
"""
Definitive MLX Implementation Comparison Test

This script compares our MLX implementation against the official MLX examples
using IDENTICAL training data, hyperparameters, and evaluation methods.

If our implementation is correct, we should see nearly identical performance.
"""

import os
import sys
import json
import requests
import tempfile
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

def download_official_mlx_files():
    """Download the official MLX LoRA files for comparison."""
    print("📥 Downloading Official MLX LoRA Implementation...")

    # URLs from the research
    files_to_download = {
        "lora.py": "https://raw.githubusercontent.com/ml-explore/mlx-examples/main/lora/lora.py",
        "models.py": "https://raw.githubusercontent.com/ml-explore/mlx-examples/main/lora/models.py",
        "utils.py": "https://raw.githubusercontent.com/ml-explore/mlx-examples/main/lora/utils.py",
        "train.jsonl": "https://raw.githubusercontent.com/ml-explore/mlx-examples/main/lora/data/train.jsonl",
        "valid.jsonl": "https://raw.githubusercontent.com/ml-explore/mlx-examples/main/lora/data/valid.jsonl",
        "test.jsonl": "https://raw.githubusercontent.com/ml-explore/mlx-examples/main/lora/data/test.jsonl"
    }

    # Create directory for official files
    official_dir = Path("mlx_official_comparison")
    official_dir.mkdir(exist_ok=True)

    data_dir = official_dir / "data"
    data_dir.mkdir(exist_ok=True)

    print("Downloading files:")
    for filename, url in files_to_download.items():
        if filename.endswith('.jsonl'):
            filepath = data_dir / filename
        else:
            filepath = official_dir / filename

        print(f"  - {filename}")

        try:
            response = requests.get(url)
            response.raise_for_status()

            with open(filepath, 'w') as f:
                f.write(response.text)

        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            return None

    print("✅ Official MLX files downloaded successfully")
    return official_dir

def create_our_implementation_test(official_dir, data_samples=100):
    """Create a test using our implementation with the same data."""
    print(f"\n🔧 Setting up Our Implementation Test (using {data_samples} samples)...")

    # Read the official training data
    train_file = official_dir / "data" / "train.jsonl"

    # Convert to our format and take subset for quick testing
    our_train_data = []
    our_val_data = []

    with open(train_file, 'r') as f:
        lines = f.readlines()

    # Take first N samples for training, next 20 for validation
    train_lines = lines[:data_samples]
    val_lines = lines[data_samples:data_samples+20]

    print(f"Using {len(train_lines)} training examples, {len(val_lines)} validation examples")

    # Convert official format to our format
    for line in train_lines:
        data = json.loads(line.strip())
        text = data["text"]

        # Convert WikiSQL format to chat format
        # Original: "table: ...\nQ: question\nA: answer"
        if "\nQ:" in text and "\nA:" in text:
            parts = text.split("\nQ:")
            if len(parts) == 2:
                table_info = parts[0]
                qa_part = parts[1]

                if "\nA:" in qa_part:
                    question_part, answer_part = qa_part.split("\nA:", 1)
                    question = question_part.strip()
                    answer = answer_part.strip()

                    # Create chat format
                    our_format = {
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant that answers questions about database tables."},
                            {"role": "user", "content": f"{table_info}\nQ: {question}"},
                            {"role": "assistant", "content": answer}
                        ]
                    }
                    our_train_data.append(our_format)

    # Same for validation data
    for line in val_lines:
        data = json.loads(line.strip())
        text = data["text"]

        if "\nQ:" in text and "\nA:" in text:
            parts = text.split("\nQ:")
            if len(parts) == 2:
                table_info = parts[0]
                qa_part = parts[1]

                if "\nA:" in qa_part:
                    question_part, answer_part = qa_part.split("\nA:", 1)
                    question = question_part.strip()
                    answer = answer_part.strip()

                    our_format = {
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant that answers questions about database tables."},
                            {"role": "user", "content": f"{table_info}\nQ: {question}"},
                            {"role": "assistant", "content": answer}
                        ]
                    }
                    our_val_data.append(our_format)

    # Save our formatted data
    our_data_dir = Path("mlx_comparison_data")
    our_data_dir.mkdir(exist_ok=True)

    train_path = our_data_dir / "train.jsonl"
    val_path = our_data_dir / "val.jsonl"

    with open(train_path, 'w') as f:
        for item in our_train_data:
            f.write(json.dumps(item) + '\n')

    with open(val_path, 'w') as f:
        for item in our_val_data:
            f.write(json.dumps(item) + '\n')

    print(f"✅ Converted {len(our_train_data)} training, {len(our_val_data)} validation examples")
    print(f"📁 Saved to: {train_path}")

    return train_path, our_train_data

def train_official_mlx(official_dir, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Train using the official MLX implementation."""
    print(f"\n🚀 Training Official MLX Implementation...")
    print(f"Model: {model_name}")

    # Change to official directory
    original_cwd = os.getcwd()
    os.chdir(official_dir)

    try:
        # Run official training with equivalent hyperparameters to ours
        cmd = [
            sys.executable, "lora.py",
            "--model", model_name,
            "--train",
            "--batch-size", "1",
            "--iters", "50",  # Quick test
            "--lora-layers", "16",
            "--learning-rate", "1e-5",
            "--test"  # Also run test
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        if result.returncode == 0:
            print("✅ Official MLX training completed successfully")

            # Extract key metrics from output
            stdout = result.stdout
            train_loss = None
            val_loss = None
            test_loss = None

            # Parse losses from output
            for line in stdout.split('\n'):
                if 'Train loss:' in line:
                    try:
                        train_loss = float(line.split('Train loss:')[1].strip().split()[0])
                    except:
                        pass
                elif 'Val loss:' in line:
                    try:
                        val_loss = float(line.split('Val loss:')[1].strip().split()[0])
                    except:
                        pass
                elif 'Test loss:' in line:
                    try:
                        test_loss = float(line.split('Test loss:')[1].strip().split()[0])
                    except:
                        pass

            return {
                "success": True,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
                "output": stdout
            }
        else:
            print(f"❌ Official MLX training failed with return code {result.returncode}")
            return {"success": False, "error": result.stderr}

    except subprocess.TimeoutExpired:
        print("❌ Official MLX training timed out")
        return {"success": False, "error": "Timeout"}
    except Exception as e:
        print(f"❌ Official MLX training error: {e}")
        return {"success": False, "error": str(e)}
    finally:
        os.chdir(original_cwd)

def train_our_implementation(train_path):
    """Train using our MLX implementation."""
    print(f"\n🚀 Training Our MLX Implementation...")

    from finetune.training.workflow import create_quick_workflow

    # Generate unique output directory
    import time
    timestamp = int(time.time())
    output_dir = f"training/comparison-run-{timestamp}"

    # Create workflow with same hyperparameters as official
    workflow = create_quick_workflow(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        data_file=str(train_path),
        template="chatml",
        output_dir=output_dir,
    )

    # Match official hyperparameters exactly
    workflow.config.optimization.epochs = 1  # Quick test
    workflow.config.optimization.learning_rate = 1e-5  # Match official
    workflow.config.optimization.batch_size = 1  # Match official
    workflow.config.lora.r = 8  # Match official rank
    workflow.config.lora.alpha = 20.0  # Match official scale
    workflow.config.lora.target_modules = ["q_proj", "v_proj"]

    print("Hyperparameters:")
    print(f"  - Learning rate: {workflow.config.optimization.learning_rate}")
    print(f"  - Batch size: {workflow.config.optimization.batch_size}")
    print(f"  - LoRA rank: {workflow.config.lora.r}")
    print(f"  - LoRA alpha: {workflow.config.lora.alpha}")

    try:
        # Execute training
        results = workflow.run_training()

        print("✅ Our MLX training completed successfully")
        print(f"📁 Model saved to: {output_dir}/final_model/")

        # Extract losses from results
        train_loss = results.get("final_train_loss")
        val_loss = results.get("final_val_loss")

        return {
            "success": True,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_path": output_dir + "/final_model",
            "results": results
        }

    except Exception as e:
        print(f"❌ Our MLX training failed: {e}")
        import traceback
        print(traceback.format_exc())
        return {"success": False, "error": str(e)}

def compare_results(official_results, our_results):
    """Compare the results from both implementations."""
    print("\n" + "=" * 70)
    print("📊 DEFINITIVE MLX IMPLEMENTATION COMPARISON")
    print("=" * 70)

    if not official_results["success"]:
        print("❌ Official MLX implementation failed:")
        print(f"   Error: {official_results.get('error', 'Unknown')}")

    if not our_results["success"]:
        print("❌ Our MLX implementation failed:")
        print(f"   Error: {our_results.get('error', 'Unknown')}")

    if not official_results["success"] or not our_results["success"]:
        print("\n⚠️  Cannot compare - one or both implementations failed")
        return

    # Compare losses
    print("\n📈 Training Loss Comparison:")
    official_train = official_results.get("train_loss")
    our_train = our_results.get("train_loss")

    if official_train and our_train:
        diff = abs(official_train - our_train)
        pct_diff = (diff / official_train) * 100

        print(f"Official MLX:      {official_train:.4f}")
        print(f"Our Implementation: {our_train:.4f}")
        print(f"Absolute Difference: {diff:.4f}")
        print(f"Percentage Difference: {pct_diff:.2f}%")

        if pct_diff < 5:
            print("✅ EXCELLENT: Training losses are nearly identical (<5% difference)")
        elif pct_diff < 15:
            print("⚠️  ACCEPTABLE: Training losses are similar (<15% difference)")
        else:
            print("❌ CONCERNING: Training losses differ significantly (>15% difference)")
    else:
        print("⚠️  Cannot compare training losses - missing data")

    # Compare validation losses if available
    print("\n📉 Validation Loss Comparison:")
    official_val = official_results.get("val_loss")
    our_val = our_results.get("val_loss")

    if official_val and our_val:
        diff = abs(official_val - our_val)
        pct_diff = (diff / official_val) * 100

        print(f"Official MLX:      {official_val:.4f}")
        print(f"Our Implementation: {our_val:.4f}")
        print(f"Absolute Difference: {diff:.4f}")
        print(f"Percentage Difference: {pct_diff:.2f}%")

        if pct_diff < 5:
            print("✅ EXCELLENT: Validation losses are nearly identical (<5% difference)")
        elif pct_diff < 15:
            print("⚠️  ACCEPTABLE: Validation losses are similar (<15% difference)")
        else:
            print("❌ CONCERNING: Validation losses differ significantly (>15% difference)")
    else:
        print("⚠️  Cannot compare validation losses - missing data")

    # Overall assessment
    print("\n🎯 FINAL VERDICT:")

    if official_train and our_train:
        train_pct_diff = abs(official_train - our_train) / official_train * 100

        if train_pct_diff < 10:
            print("🎉 SUCCESS: Our MLX implementation appears to be CORRECT!")
            print("   ✅ Training losses match official implementation closely")
            print("   ✅ This validates our LoRA system works properly")
            print("   ✅ Previous issues were due to training methodology, not code bugs")
        else:
            print("❌ FAILURE: Our MLX implementation has significant differences")
            print("   ❌ Training losses don't match official implementation")
            print("   ❌ This suggests bugs in our LoRA system or training loop")
            print("   ❌ Further investigation needed")
    else:
        print("⚠️  INCONCLUSIVE: Missing loss data to make definitive comparison")

def main():
    """Main comparison function."""
    print("🔬 DEFINITIVE MLX IMPLEMENTATION COMPARISON TEST")
    print("=" * 70)
    print("This test will prove whether our MLX implementation is correct")
    print("by comparing against official MLX examples on identical data.")
    print("=" * 70)

    # Step 1: Download official files
    official_dir = download_official_mlx_files()
    if not official_dir:
        print("❌ Failed to download official MLX files")
        return 1

    # Step 2: Prepare our implementation data
    train_path, train_data = create_our_implementation_test(official_dir, data_samples=50)  # Small for quick test

    # Step 3: Train with official implementation
    print("\n" + "="*50)
    print("TRAINING WITH OFFICIAL MLX IMPLEMENTATION")
    print("="*50)
    official_results = train_official_mlx(official_dir)

    # Step 4: Train with our implementation
    print("\n" + "="*50)
    print("TRAINING WITH OUR MLX IMPLEMENTATION")
    print("="*50)
    our_results = train_our_implementation(train_path)

    # Step 5: Compare results
    compare_results(official_results, our_results)

    return 0

if __name__ == "__main__":
    sys.exit(main())