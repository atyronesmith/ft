#!/usr/bin/env python3
"""
Debug script to test base model generation without LoRA.
This will help us determine if the issue is with the base model loading
or with the LoRA training/loading process.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from finetune.training.workflow import create_quick_workflow
from finetune.utils.chat import apply_chat_template_for_inference
from finetune.inference.generation import GenerationConfig, generate_text

def test_base_model_generation():
    """Test if the base model can generate reasonable text before LoRA."""
    print("🧪 Testing Base Model Generation (No LoRA)")
    print("=" * 50)

    # Create workflow for base model loading
    workflow = create_quick_workflow(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        data_file="dummy",
        template="chatml",
        output_dir="/tmp/dummy",
    )

    # Load model WITHOUT LoRA
    print("📥 Loading base model without LoRA...")
    workflow.model, workflow.tokenizer, _ = workflow.model_manager.load_model(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        load_in_4bit=workflow.config.model.load_in_4bit,
    )

    print(f"✅ Model loaded with vocabulary: {len(workflow.tokenizer.get_vocab())} tokens")

    # Test questions
    test_questions = [
        "What is the capital of France?",
        "What is 2+2?",
        "Hello, how are you?",
    ]

    print("\n🎯 Testing Base Model Responses:")
    print("-" * 40)

    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Question: {question}")

        try:
            # Create prompt using same method as training
            prompt = apply_chat_template_for_inference(workflow.tokenizer, question)
            print(f"   Prompt preview: {prompt[:100]}...")

            # Generate with conservative settings
            config = GenerationConfig(
                max_tokens=30,
                temperature=0.8,
                top_p=0.9,
                verbose=False,
                stop_on_eos=True,
                stop_on_special_tokens=True
            )

            response = generate_text(workflow.model, workflow.tokenizer, prompt, config)
            print(f"   Response: '{response}'")

            # Check if response contains gibberish patterns
            gibberish_patterns = ["AccessorImpl", "Ligações", "Хронологи", "архиви", "]{'"]
            has_gibberish = any(pattern in response for pattern in gibberish_patterns)

            if has_gibberish:
                print("   ❌ GIBBERISH DETECTED in base model!")
            elif response.strip():
                print("   ✅ REASONABLE response from base model")
            else:
                print("   ⚠️  Empty response from base model")

        except Exception as e:
            print(f"   ❌ Error generating: {e}")

    print("\n" + "=" * 50)
    print("Base model test complete.")

    return workflow

def test_tokenizer_decode():
    """Test if tokenizer decode is working properly."""
    print("\n🔤 Testing Tokenizer Encoding/Decoding")
    print("-" * 40)

    workflow = create_quick_workflow(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        data_file="dummy",
        template="chatml",
        output_dir="/tmp/dummy",
    )

    workflow.model, workflow.tokenizer, _ = workflow.model_manager.load_model(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )

    test_texts = [
        "Hello world",
        "What is the capital of France?",
        "Paris is the capital of France.",
    ]

    for text in test_texts:
        print(f"\nOriginal: '{text}'")

        # Encode
        tokens = workflow.tokenizer.encode(text, add_special_tokens=False)
        print(f"Tokens: {tokens[:10]}..." if len(tokens) > 10 else f"Tokens: {tokens}")

        # Decode
        decoded = workflow.tokenizer.decode(tokens)
        print(f"Decoded: '{decoded}'")

        if decoded.strip() == text.strip():
            print("✅ Perfect round-trip")
        else:
            print("❌ Round-trip failed!")

        # Test individual tokens
        print("Token breakdown:")
        for i, token_id in enumerate(tokens[:5]):
            token_text = workflow.tokenizer.decode([token_id])
            print(f"  {i}: {token_id} -> '{repr(token_text)}'")

if __name__ == "__main__":
    try:
        workflow = test_base_model_generation()
        test_tokenizer_decode()
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        print(traceback.format_exc())