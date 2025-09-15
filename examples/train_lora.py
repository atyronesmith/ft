#!/usr/bin/env python3
"""Example script for fine-tuning a model using LoRA."""

import mlx.core as mx
from pathlib import Path

from finetune.models.manager import ModelManager
from finetune.training import (
    LoRAConfig,
    TrainingConfig,
    LoRATrainer,
    SimpleDataLoader,
)


def create_dummy_dataset(num_samples: int = 100, seq_length: int = 128, vocab_size: int = 50000):
    """Create a dummy dataset for demonstration."""
    data = []
    for _ in range(num_samples):
        # Create random token sequences
        input_ids = mx.random.randint(0, vocab_size, shape=(seq_length,))
        data.append({"input_ids": input_ids})
    return data


def main():
    """Main training function."""
    print("LoRA Fine-tuning Example")
    print("=" * 50)

    # 1. Initialize model manager
    print("\n1. Loading model...")
    manager = ModelManager()

    # For demonstration, we'll create a small model
    # In practice, you'd load a pretrained model:
    # model = manager.load_model("meta-llama/Llama-2-7b-hf")

    from finetune.models.base import ModelConfig
    config = ModelConfig(
        model_type="llama",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        vocab_size=50000,
        max_position_embeddings=2048,
    )

    # Get model (will use MLX if available, otherwise PyTorch)
    model = manager._get_model_from_config(config)
    print(f"  Model type: {type(model).__name__}")
    print(f"  Total parameters: {model.num_parameters:,}")

    # 2. Configure LoRA
    print("\n2. Configuring LoRA...")
    lora_config = LoRAConfig(
        r=8,  # Low rank
        alpha=16,  # Scaling factor
        dropout=0.1,  # Dropout for regularization
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to attention layers
    )

    # Add LoRA adapters to model
    model.add_lora(lora_config)

    # Get trainable parameters info
    from finetune.training.lora import get_lora_trainable_params
    trainable_params, trainable_count, total_count = get_lora_trainable_params(model)
    print(f"  Trainable parameters: {trainable_count:,} / {total_count:,}")
    print(f"  Percentage trainable: {100 * trainable_count / total_count:.2f}%")

    # 3. Prepare datasets
    print("\n3. Preparing datasets...")
    train_data = create_dummy_dataset(num_samples=100)
    eval_data = create_dummy_dataset(num_samples=20)

    train_dataset = SimpleDataLoader(train_data, batch_size=4, shuffle=True)
    eval_dataset = SimpleDataLoader(eval_data, batch_size=4, shuffle=False)
    print(f"  Training samples: {len(train_data)}")
    print(f"  Evaluation samples: {len(eval_data)}")
    print(f"  Batch size: 4")
    print(f"  Training batches: {len(train_dataset)}")

    # 4. Configure training
    print("\n4. Configuring training...")
    training_config = TrainingConfig(
        learning_rate=5e-5,
        num_epochs=3,
        batch_size=4,
        warmup_steps=100,
        max_grad_norm=1.0,
        save_steps=500,
        eval_steps=100,
        logging_steps=10,
        output_dir="./lora_output",
    )
    print(f"  Learning rate: {training_config.learning_rate}")
    print(f"  Epochs: {training_config.num_epochs}")
    print(f"  Output directory: {training_config.output_dir}")

    # 5. Initialize trainer
    print("\n5. Initializing trainer...")
    trainer = LoRATrainer(
        model=model,
        lora_config=lora_config,
        training_config=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # 6. Start training
    print("\n6. Starting training...")
    print("=" * 50)

    # Note: For demonstration, we'll just show the setup
    # In practice, you'd call: trainer.train()
    print("\nTraining setup complete!")
    print("\nTo start actual training, uncomment the line below:")
    print("# trained_model = trainer.train()")

    # 7. Save and load LoRA weights
    print("\n7. Saving LoRA weights...")
    output_path = Path("./lora_output/lora_weights_example.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from finetune.training.lora import save_lora_weights
    save_lora_weights(model, str(output_path))
    print(f"  Saved to: {output_path}")

    # Example of loading LoRA weights
    print("\n8. Loading LoRA weights...")
    from finetune.training.lora import load_lora_weights
    load_lora_weights(model, str(output_path))
    print("  Weights loaded successfully!")

    print("\n" + "=" * 50)
    print("LoRA setup demonstration complete!")
    print("\nKey advantages of LoRA:")
    print("  - Dramatically reduces trainable parameters")
    print("  - Maintains model quality")
    print("  - Fast training and switching between adaptations")
    print("  - Memory efficient")


if __name__ == "__main__":
    main()