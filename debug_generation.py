#!/usr/bin/env python3
"""
Debug script for post-training generation issues.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from finetune.training.workflow import create_quick_workflow
from finetune.inference.generation import GenerationConfig, generate_text, load_model_and_tokenizer

def debug_generation():
    """Debug the generation issue step by step."""

    # Create comprehensive dataset with all 100 capitals (no repeats)
    capitals_data = [
        ("Afghanistan", "Kabul"), ("Albania", "Tirana"), ("Algeria", "Algiers"), ("Argentina", "Buenos Aires"),
        ("Armenia", "Yerevan"), ("Australia", "Canberra"), ("Austria", "Vienna"), ("Azerbaijan", "Baku"),
        ("Bahrain", "Manama"), ("Bangladesh", "Dhaka"), ("Belarus", "Minsk"), ("Belgium", "Brussels"),
        ("Bolivia", "La Paz"), ("Brazil", "Brasília"), ("Bulgaria", "Sofia"), ("Cambodia", "Phnom Penh"),
        ("Canada", "Ottawa"), ("Chile", "Santiago"), ("China", "Beijing"), ("Colombia", "Bogotá"),
        ("Croatia", "Zagreb"), ("Cuba", "Havana"), ("Cyprus", "Nicosia"), ("Czech Republic", "Prague"),
        ("Denmark", "Copenhagen"), ("Ecuador", "Quito"), ("Egypt", "Cairo"), ("Estonia", "Tallinn"),
        ("Ethiopia", "Addis Ababa"), ("Finland", "Helsinki"), ("France", "Paris"), ("Georgia", "Tbilisi"),
        ("Germany", "Berlin"), ("Ghana", "Accra"), ("Greece", "Athens"), ("Hungary", "Budapest"),
        ("Iceland", "Reykjavik"), ("India", "New Delhi"), ("Indonesia", "Jakarta"), ("Iran", "Tehran"),
        ("Iraq", "Baghdad"), ("Ireland", "Dublin"), ("Israel", "Jerusalem"), ("Italy", "Rome"),
        ("Japan", "Tokyo"), ("Jordan", "Amman"), ("Kazakhstan", "Nur-Sultan"), ("Kenya", "Nairobi"),
        ("Kuwait", "Kuwait City"), ("Latvia", "Riga"), ("Lebanon", "Beirut"), ("Libya", "Tripoli"),
        ("Lithuania", "Vilnius"), ("Luxembourg", "Luxembourg"), ("Malaysia", "Kuala Lumpur"), ("Malta", "Valletta"),
        ("Mexico", "Mexico City"), ("Mongolia", "Ulaanbaatar"), ("Morocco", "Rabat"), ("Netherlands", "Amsterdam"),
        ("New Zealand", "Wellington"), ("Nigeria", "Abuja"), ("North Korea", "Pyongyang"), ("Norway", "Oslo"),
        ("Pakistan", "Islamabad"), ("Peru", "Lima"), ("Philippines", "Manila"), ("Poland", "Warsaw"),
        ("Portugal", "Lisbon"), ("Qatar", "Doha"), ("Romania", "Bucharest"), ("Russia", "Moscow"),
        ("Saudi Arabia", "Riyadh"), ("Serbia", "Belgrade"), ("Singapore", "Singapore"), ("Slovakia", "Bratislava"),
        ("Slovenia", "Ljubljana"), ("South Africa", "Cape Town"), ("South Korea", "Seoul"), ("Spain", "Madrid"),
        ("Sri Lanka", "Colombo"), ("Sweden", "Stockholm"), ("Switzerland", "Bern"), ("Syria", "Damascus"),
        ("Taiwan", "Taipei"), ("Thailand", "Bangkok"), ("Tunisia", "Tunis"), ("Turkey", "Ankara"),
        ("Ukraine", "Kyiv"), ("United Arab Emirates", "Abu Dhabi"), ("United Kingdom", "London"),
        ("United States", "Washington, D.C."), ("Uruguay", "Montevideo"), ("Uzbekistan", "Tashkent"),
        ("Venezuela", "Caracas"), ("Vietnam", "Hanoi"), ("Yemen", "Sana'a"), ("Zimbabwe", "Harare"),
        ("Angola", "Luanda"), ("Benin", "Porto-Novo"), ("Botswana", "Gaborone"), ("Burkina Faso", "Ouagadougou"),
        ("Burundi", "Gitega"), ("Cameroon", "Yaoundé"), ("Chad", "N'Djamena"), ("Republic of the Congo", "Brazzaville"),
        ("Ivory Coast", "Yamoussoukro"), ("Djibouti", "Djibouti"), ("Equatorial Guinea", "Malabo"), ("Eritrea", "Asmara"),
        ("Eswatini", "Mbabane"), ("Gabon", "Libreville"), ("Gambia", "Banjul"), ("Guinea", "Conakry"),
        ("Guinea-Bissau", "Bissau"), ("Lesotho", "Maseru"), ("Liberia", "Monrovia"), ("Madagascar", "Antananarivo"),
        ("Malawi", "Lilongwe"), ("Mali", "Bamako"), ("Mauritania", "Nouakchott"), ("Mauritius", "Port Louis"),
        ("Mozambique", "Maputo"), ("Namibia", "Windhoek")
    ]

    # Convert to training format
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

        print("🔍 Step 1: Testing base model generation BEFORE training...")

        # Load base model first
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        base_model, base_tokenizer = load_model_and_tokenizer(model_id)

        # Test base model generation
        question = "What is the capital of France?"
        config = GenerationConfig(max_tokens=10, temperature=0.0, verbose=True)

        print(f"Base model generation for: {question}")
        base_result = generate_text(base_model, base_tokenizer, question, config)
        print(f"Base result: '{base_result}'")

        print("\n🔍 Step 2: Training model...")

        # Train the model
        workflow = create_quick_workflow(
            model_name=model_id,
            data_file=str(train_file),
            template="tinyllama",
            output_dir=str(tmp_path / "output"),
        )

        # Proper training parameters: use fewer epochs to prevent overfitting
        workflow.config.optimization.epochs = 2  # Start with just 2 epochs for 100 examples
        workflow.config.optimization.batch_size = 8  # Larger batch size for better training

        workflow.prepare_dataset()
        workflow.prepare_model()
        workflow.prepare_trainer()

        # CRITICAL FIX: Tokenize the data before training like the e2e test does
        print("Tokenizing training data...")
        from transformers import AutoTokenizer
        import mlx.core as mx

        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token

        def tokenize_batch(examples):
            batches = []
            for example in examples:
                # Use centralized common utilities for consistency
                from finetune.utils.chat import apply_chat_template_with_tokenizer

                messages = [
                    {"role": "system", "content": "You are a helpful geography assistant who provides accurate, concise answers about world capitals."},
                    {"role": "user", "content": example["instruction"]},
                    {"role": "assistant", "content": example["output"]}
                ]
                training_text = apply_chat_template_with_tokenizer(tok, messages, for_training=True)

                # Tokenize
                enc = tok.encode(training_text, return_tensors="np")[0]
                ids = mx.array(enc, dtype=mx.int32)

                # Create labels by shifting
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

        # Replace the trainer's dataset with properly tokenized data
        workflow.trainer.train_dataset = tokenize_batch(workflow.train_dataset)
        workflow.trainer.eval_dataset = tokenize_batch(workflow.eval_dataset) if workflow.eval_dataset else None

        # Train
        print("Training...")
        trained_model = workflow.trainer.train()
        trained_model.eval()  # Ensure eval mode

        print("\n🔍 Step 3: Testing trained model generation...")

        # Test same question on trained model
        print(f"Trained model generation for: {question}")

        # Check model state
        print(f"Model has LoRA: {hasattr(trained_model, 'get_lora_params')}")
        if hasattr(trained_model, 'get_lora_params'):
            lora_params, _, _ = trained_model.get_lora_params()
            print(f"LoRA params count: {len(lora_params) if lora_params else 0}")

        # Test generation with different configs
        configs_to_test = [
            ("Greedy", GenerationConfig(max_tokens=10, temperature=0.0, verbose=True)),
            ("Low temp", GenerationConfig(max_tokens=10, temperature=0.1, verbose=True)),
            ("Default", GenerationConfig.ollama_defaults()),
        ]

        for name, gen_config in configs_to_test:
            print(f"\n--- Testing {name} config ---")
            try:
                # Use same tokenizer
                result = generate_text(trained_model, base_tokenizer, question, gen_config)
                print(f"{name} result: '{result}'")

                # Check if result looks like gibberish
                if len(result) > 0 and not any(word in result.lower() for word in ["paris", "france", "capital"]):
                    print(f"⚠️  {name}: Possible gibberish detected!")
                else:
                    print(f"✅ {name}: Result looks reasonable")

            except Exception as e:
                print(f"❌ {name}: Error - {e}")

        print("\n🔍 Step 4: Debugging training data format...")

        # Check what the training data looked like
        from finetune.data.templates import TemplateRegistry
        template_registry = TemplateRegistry()
        template = template_registry.get_template("tinyllama")

        example = test_data[0]
        formatted = template.format(example)
        print(f"Training format:\n{formatted}")

        # Use centralized common utilities for consistency
        from finetune.utils.chat import apply_chat_template_for_inference

        gen_prompt = apply_chat_template_for_inference(base_tokenizer, question)
        print(f"\nGeneration format:\n{gen_prompt}")

        # Compare formats (just the user question parts)
        if question in formatted and question in gen_prompt:
            print("✅ Both formats contain the question correctly")
        else:
            print("⚠️  Question format issue detected")

if __name__ == "__main__":
    debug_generation()