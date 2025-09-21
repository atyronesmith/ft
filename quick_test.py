#!/usr/bin/env python3
"""
Quick test to validate the gibberish fix with 30 examples instead of 124.
"""

import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from finetune.inference.generation import GenerationConfig, generate_text, load_model_and_tokenizer
from finetune.training.workflow import create_quick_workflow


def quick_test():
    """Quick test with 30 examples to validate fix."""

    # Test with 30 capitals for faster validation
    capitals_data = [
        ("France", "Paris"),
        ("Germany", "Berlin"),
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
        ("Portugal", "Lisbon"),
        ("Japan", "Tokyo"),
        ("China", "Beijing"),
        ("India", "New Delhi"),
        ("Canada", "Ottawa"),
        ("Australia", "Canberra"),
        ("Brazil", "Brasília"),
        ("Mexico", "Mexico City"),
        ("Russia", "Moscow"),
        ("United Kingdom", "London"),
        ("United States", "Washington, D.C."),
        ("Netherlands", "Amsterdam"),
        ("Sweden", "Stockholm"),
        ("Norway", "Oslo"),
        ("Denmark", "Copenhagen"),
        ("Finland", "Helsinki"),
        ("Poland", "Warsaw"),
        ("Greece", "Athens"),
        ("Turkey", "Ankara"),
        ("Egypt", "Cairo"),
        ("South Africa", "Cape Town"),
        ("Argentina", "Buenos Aires"),
        ("Chile", "Santiago"),
        ("Indonesia", "Jakarta"),
        ("Thailand", "Bangkok"),
        ("South Korea", "Seoul"),
    ]

    test_data = [
        {"instruction": f"What is the capital of {country}?", "output": capital}
        for country, capital in capitals_data
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        train_file = tmp_path / "train.jsonl"

        # Write test data
        with open(train_file, "w") as f:
            import json

            for item in test_data:
                f.write(json.dumps(item) + "\n")

        print("🔍 Quick test with 30 capitals...")

        # Load base model first
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        base_model, base_tokenizer = load_model_and_tokenizer(model_id)

        # Quick test before training
        question = "What is the capital of France?"
        config = GenerationConfig(max_tokens=10, temperature=0.0, verbose=False)
        base_result = generate_text(base_model, base_tokenizer, question, config)
        print(f"Base model: '{base_result}'")

        # Train with proper dataset
        workflow = create_quick_workflow(
            model_name=model_id,
            data_file=str(train_file),
            template="tinyllama",
            output_dir=str(tmp_path / "output"),
        )

        # Test with 4 epochs for better learning (within 1-5 range requested)
        workflow.config.optimization.epochs = 4
        workflow.config.optimization.batch_size = 8

        workflow.prepare_dataset()
        workflow.prepare_model()
        workflow.prepare_trainer()

        # Tokenize properly like the working e2e test
        import mlx.core as mx
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        def tokenize_batch(examples):
            batches = []
            for example in examples:
                # Use centralized common utilities for consistency
                from finetune.utils.chat import apply_chat_template_with_tokenizer

                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals.",
                    },
                    {"role": "user", "content": example["instruction"]},
                    {"role": "assistant", "content": example["output"]},
                ]
                training_text = apply_chat_template_with_tokenizer(tok, messages, for_training=True)

                enc = tok.encode(training_text, return_tensors="np")[0]
                ids = mx.array(enc, dtype=mx.int32)

                if ids.shape[0] > 1:
                    input_seq = ids[:-1]
                    label_seq = ids[1:]
                    mask_seq = mx.ones_like(label_seq)

                    batch_item = {
                        "input_ids": input_seq.reshape(1, -1),
                        "labels": label_seq.reshape(1, -1),
                        "attention_mask": mask_seq.reshape(1, -1),
                    }
                    batches.append(batch_item)

            return batches

        # Replace datasets with tokenized data
        workflow.trainer.train_dataset = tokenize_batch(workflow.train_dataset)
        workflow.trainer.eval_dataset = (
            tokenize_batch(workflow.eval_dataset) if workflow.eval_dataset else None
        )

        print("Training...")
        trained_model = workflow.trainer.train()
        trained_model.eval()

        # Test after training
        result = generate_text(trained_model, base_tokenizer, question, config)
        print(f"Trained model: '{result}'")

        # Check if it's gibberish or reasonable
        if "paris" in result.lower():
            print("✅ SUCCESS: Model generates correct answer!")
        elif any(token in result.lower() for token in ["ass", "|", ">"]):
            print("❌ STILL GIBBERISH: Model still generating template tokens")
        else:
            print(
                f"⚠️  UNCLEAR: Result doesn't contain 'Paris' but isn't obvious gibberish: '{result}'"
            )


if __name__ == "__main__":
    quick_test()
