#!/usr/bin/env python3
"""
Quick start example for FineTune workflow.

Demonstrates end-to-end fine-tuning with the integrated pipeline.
"""

import tempfile
import json
from pathlib import Path

from finetune.config import TrainingConfig, ModelConfig, DataConfig, LoRAConfig, OptimizationConfig, ConfigProfile
from finetune.training.workflow import FineTuningWorkflow, create_quick_workflow


def create_sample_data():
    """Create a sample dataset for demonstration."""
    data = [
        {
            "instruction": "What is the capital of France?",
            "output": "The capital of France is Paris."
        },
        {
            "instruction": "Explain photosynthesis briefly",
            "output": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen."
        },
        {
            "instruction": "What is 2 + 2?",
            "output": "2 + 2 equals 4."
        },
        {
            "instruction": "Name three programming languages",
            "output": "Three popular programming languages are Python, JavaScript, and Java."
        },
        {
            "instruction": "What is the largest planet in our solar system?",
            "output": "Jupiter is the largest planet in our solar system."
        }
    ]

    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        return f.name


def demonstrate_quick_workflow():
    """Demonstrate the quick workflow function."""
    print("🚀 FineTune Quick Start Demo")
    print("=" * 40)

    # Create sample dataset
    data_file = create_sample_data()
    print(f"📊 Created sample dataset: {data_file}")

    try:
        # Create quick workflow
        print("\n📋 Creating quick workflow...")
        workflow = create_quick_workflow(
            model_name="microsoft/DialoGPT-small",  # Small model for demo
            data_file=data_file,
            template="alpaca",
            output_dir="./demo_output"
        )

        print("✅ Workflow created successfully!")

        # Display configuration
        print("\n📊 Configuration:")
        print(f"  Model: {workflow.config.model.name}")
        print(f"  Data: {workflow.config.data.train_file}")
        print(f"  Template: {workflow.config.data.template}")
        print(f"  LoRA Rank: {workflow.config.lora.r}")
        print(f"  Epochs: {workflow.config.optimization.epochs}")

        # Prepare dataset (without model loading)
        print("\n📚 Preparing dataset...")
        workflow.prepare_dataset()
        print(f"✅ Loaded {len(workflow.train_dataset)} training examples")

        # Show formatted example
        print("\n📝 Sample formatted example:")
        print("─" * 50)
        print(workflow.train_dataset[0][:200] + "...")
        print("─" * 50)

        print("\n🎉 Demo completed successfully!")
        print("💡 To run actual training, use: ft train start <model> <dataset>")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean up
        Path(data_file).unlink()


def demonstrate_full_config():
    """Demonstrate full configuration workflow."""
    print("\n🔧 Full Configuration Demo")
    print("=" * 40)

    # Create sample dataset
    data_file = create_sample_data()

    try:
        # Create comprehensive configuration
        config = TrainingConfig(
            model=ModelConfig(
                name="microsoft/DialoGPT-small",
                load_in_4bit=False,
            ),
            data=DataConfig(
                train_file=data_file,
                template="chatml",
                validation_split=0.2,
                max_length=512,
            ),
            lora=LoRAConfig(
                r=16,
                alpha=32.0,
                dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            ),
            optimization=OptimizationConfig(
                learning_rate=2e-4,
                batch_size=2,
                epochs=3,
                warmup_steps=50,
                weight_decay=0.01,
            ),
            output_dir="./full_demo_output",
        )

        # Apply chat profile
        chat_config = ConfigProfile.apply_profile(config, "chat")

        print("📋 Configuration created with chat profile:")
        print(f"  Template: {chat_config.data.template}")
        print(f"  LoRA Rank: {chat_config.lora.r}")
        print(f"  Learning Rate: {chat_config.optimization.learning_rate}")

        # Create workflow
        workflow = FineTuningWorkflow(chat_config)

        # Test configuration validation
        from finetune.config import ConfigValidator
        validator = ConfigValidator()
        warnings = validator.validate(chat_config)

        if warnings:
            print("\n⚠️  Configuration warnings:")
            for warning in warnings:
                print(f"  • {warning}")
        else:
            print("\n✅ Configuration validation passed")

        # Memory estimation
        memory_estimate = validator.estimate_memory_usage(chat_config)
        print(f"💾 Estimated memory usage: {memory_estimate:.1f} GB")

        print("\n✅ Full configuration demo completed!")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean up
        Path(data_file).unlink()


def demonstrate_cli_integration():
    """Show CLI command examples."""
    print("\n💻 CLI Integration Examples")
    print("=" * 40)

    print("Quick training:")
    print("  ft train quick microsoft/DialoGPT-small examples/sample_dataset.json")
    print()

    print("Full training with options:")
    print("  ft train start microsoft/DialoGPT-small examples/sample_dataset.json \\")
    print("    --template chatml \\")
    print("    --epochs 3 \\")
    print("    --batch-size 4 \\")
    print("    --lora-rank 16")
    print()

    print("Training with profile:")
    print("  ft train start microsoft/DialoGPT-small examples/sample_dataset.json \\")
    print("    --profile chat")
    print()

    print("Training with config file:")
    print("  ft train start --config configs/chat_training.yml")
    print()

    print("Validate configuration:")
    print("  ft train validate configs/chat_training.yml")
    print()


if __name__ == "__main__":
    try:
        demonstrate_quick_workflow()
        demonstrate_full_config()
        demonstrate_cli_integration()

        print("\n🎉 All demos completed successfully!")
        print("\n📚 Next steps:")
        print("  1. Install a small model: ft models pull microsoft/DialoGPT-small")
        print("  2. Try the CLI: ft train quick microsoft/DialoGPT-small examples/sample_dataset.json")
        print("  3. Check the output: ls -la quick_output/")

    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()