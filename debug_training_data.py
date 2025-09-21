#!/usr/bin/env python3
"""
Debug script to show how training data flows from JSON to tokenized batches.
Run with: python debug_training_data.py
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    print("🔍 Training Data Flow Debug")
    print("="*80)

    # Step 1: Generate sample training data
    from tests.integration.test_end_to_end_mlx import _generate_dataset

    test_file = Path('/tmp/debug_training.jsonl')
    train_data = _generate_dataset(test_file, n=3)

    print("\n📋 STEP 1: Raw JSON Training Data")
    print("-"*50)
    print(json.dumps(train_data, indent=2, ensure_ascii=False))

    # Step 2: Apply chat template
    print("\n🎨 STEP 2: Chat Template Application")
    print("-"*50)

    from finetune.inference.generation import create_tokenizer_with_special_tokens

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading tokenizer: {model_id}")
    tokenizer = create_tokenizer_with_special_tokens(model_id)
    print(f"Vocabulary size: {len(tokenizer.get_vocab())} tokens")

    for i, example in enumerate(train_data):
        print(f"\n📝 Conversation {i+1}:")
        print(f"  Raw: {example}")

        # Extract messages from the new format
        messages = example["messages"]

        # Use centralized common utilities for consistency
        from finetune.utils.chat import apply_chat_template_with_tokenizer

        training_text = apply_chat_template_with_tokenizer(tokenizer, messages, for_training=True)

        print(f"  Full Conversation Template: {repr(training_text)}")

        # Show individual role messages
        for msg in messages:
            print(f"    {msg['role'].title()}: {repr(msg['content'])}")

        # Tokenize
        token_ids = tokenizer.encode(training_text, add_special_tokens=False)
        print(f"  Token IDs ({len(token_ids)}): {token_ids}")

        # Show special tokens
        print(f"  Special tokens in sequence:")
        for j, token_id in enumerate(token_ids):
            token_text = tokenizer.decode([token_id])
            if token_id in [32000, 32001, 32002]:  # Chat template tokens
                print(f"    Position {j}: {token_id} → '{token_text}' (SPECIAL)")

        # Show input/label shift for language modeling
        if len(token_ids) > 1:
            input_ids = token_ids[:-1]
            label_ids = token_ids[1:]
            print(f"  Language Modeling:")
            print(f"    Input IDs ({len(input_ids)}): {input_ids}")
            print(f"    Label IDs ({len(label_ids)}): {label_ids}")
            print(f"    Input text: {repr(tokenizer.decode(input_ids))}")
            print(f"    Label text: {repr(tokenizer.decode(label_ids))}")

    print("\n✅ Training data flow analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()