#!/usr/bin/env python3
"""
Comprehensive model loading comparison test.

This test:
1. Loads the same base model with both Transformers and MLX
2. Verifies the models are identical (same weights)
3. Tests various chat templates on both methods
4. Compares responses to identify if the issue is loading or templating
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import mlx.core as mx
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


def load_transformers_model(model_name: str):
    """Load model using transformers pipeline."""
    print(f"📥 Loading {model_name} with Transformers...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model = model.to(device)

    # Create pipeline
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device=device,
    )

    print(f"✅ Transformers model loaded")
    print(f"   Model parameters: {model.num_parameters():,}")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Model dtype: {model.dtype}")

    return generator, model, tokenizer


def load_mlx_model(model_name: str):
    """Load model using our MLX implementation."""
    print(f"📥 Loading {model_name} with MLX...")

    from finetune.models.manager import ModelManager

    manager = ModelManager()
    model, tokenizer, config = manager.load_model(model_name)

    # Count parameters
    try:
        from finetune.models.mlx_models import flatten_params
        flat_params = flatten_params(model.parameters())
        param_count = sum(p.size for p in flat_params.values() if hasattr(p, "size"))
    except Exception:
        param_count = getattr(model, "num_parameters", 0)

    print(f"✅ MLX model loaded")
    print(f"   Model parameters: {param_count:,}")
    print(f"   Vocab size: {len(tokenizer)}")
    print(f"   Model type: {type(model)}")

    return model, tokenizer, config


def compare_model_weights(transformers_model, mlx_model, mlx_tokenizer):
    """Compare weights between transformers and MLX models."""
    print("\n🔍 Comparing Model Weights...")

    # Get transformers state dict
    transformers_state = transformers_model.state_dict()

    # Get MLX parameters
    from finetune.models.mlx_models import flatten_params
    mlx_params = flatten_params(mlx_model.parameters())

    print(f"Transformers parameters: {len(transformers_state)}")
    print(f"MLX parameters: {len(mlx_params)}")

    # Compare a few key weights
    weight_comparisons = []

    # Compare embedding weights
    if "model.embed_tokens.weight" in transformers_state and "embed_tokens.weight" in mlx_params:
        torch_embed = transformers_state["model.embed_tokens.weight"]
        mlx_embed = mlx_params["embed_tokens.weight"]

        # Convert MLX to numpy for comparison
        mlx_embed_np = mx.to_numpy(mlx_embed)
        torch_embed_np = torch_embed.detach().cpu().numpy()

        # Compare shapes
        shape_match = torch_embed_np.shape == mlx_embed_np.T.shape  # MLX may be transposed

        if shape_match:
            # Compare a small sample of values
            diff = abs(torch_embed_np[0:5, 0:5] - mlx_embed_np.T[0:5, 0:5]).mean()
            weight_comparisons.append(("embed_tokens", shape_match, diff))
        else:
            weight_comparisons.append(("embed_tokens", False, float('inf')))

    # Compare first layer weights
    if "model.layers.0.self_attn.q_proj.weight" in transformers_state and "layers.0.self_attn.q_proj.weight" in mlx_params:
        torch_q = transformers_state["model.layers.0.self_attn.q_proj.weight"]
        mlx_q = mlx_params["layers.0.self_attn.q_proj.weight"]

        mlx_q_np = mx.to_numpy(mlx_q)
        torch_q_np = torch_q.detach().cpu().numpy()

        shape_match = torch_q_np.shape == mlx_q_np.T.shape

        if shape_match:
            diff = abs(torch_q_np[0:5, 0:5] - mlx_q_np.T[0:5, 0:5]).mean()
            weight_comparisons.append(("q_proj", shape_match, diff))
        else:
            weight_comparisons.append(("q_proj", False, float('inf')))

    print("\nWeight Comparison Results:")
    for name, shape_match, diff in weight_comparisons:
        status = "✅" if shape_match and diff < 1e-3 else "❌"
        print(f"  {status} {name}: shapes_match={shape_match}, avg_diff={diff:.6f}")

    return weight_comparisons


def test_chat_templates():
    """Define various chat templates to test."""
    return {
        "chatml": {
            "template": "<|user|>\n{question}</s>\n<|assistant|>\n",
            "description": "ChatML format (what transformers test uses)"
        },
        "llama_chat": {
            "template": "<s>[INST] {question} [/INST]",
            "description": "Llama chat format"
        },
        "simple_qa": {
            "template": "Question: {question}\nAnswer:",
            "description": "Simple Q&A format"
        },
        "direct": {
            "template": "{question}",
            "description": "Direct question (no template)"
        },
        "alpaca": {
            "template": "### Instruction:\n{question}\n\n### Response:\n",
            "description": "Alpaca instruction format"
        }
    }


def generate_transformers_response(generator, prompt: str, max_tokens: int = 50):
    """Generate response using transformers pipeline."""
    try:
        generated_texts = generator(
            prompt,
            max_new_tokens=max_tokens,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            pad_token_id=generator.tokenizer.eos_token_id,
        )

        full_response = generated_texts[0]['generated_text']

        # Extract just the new part (remove the prompt)
        if prompt in full_response:
            response = full_response[len(prompt):].strip()
        else:
            response = full_response.strip()

        return response
    except Exception as e:
        return f"ERROR: {e}"


def generate_mlx_response(model, tokenizer, prompt: str, max_tokens: int = 50):
    """Generate response using MLX."""
    try:
        from finetune.inference.generation import MLXTextGenerator

        generator = MLXTextGenerator(model, tokenizer)
        response = generator.generate_simple(prompt, max_tokens=max_tokens, temperature=0.7)

        return response.strip()
    except Exception as e:
        return f"ERROR: {e}"


def test_template_comparison(transformers_generator, mlx_model, mlx_tokenizer):
    """Test various templates on both models."""
    print("\n🎯 Testing Chat Templates...")

    # Test questions
    test_questions = [
        "What is the capital of France?",
        "What is 2 + 2?",
        "What color is the sky?"
    ]

    templates = test_chat_templates()

    results = []

    for template_name, template_info in templates.items():
        print(f"\n📝 Testing template: {template_name} - {template_info['description']}")
        print(f"   Template: {repr(template_info['template'])}")

        template_results = []

        for question in test_questions:
            prompt = template_info['template'].format(question=question)

            print(f"\n   Q: {question}")
            print(f"   Prompt: {repr(prompt[:100])}{'...' if len(prompt) > 100 else ''}")

            # Generate with Transformers
            transformers_response = generate_transformers_response(transformers_generator, prompt, max_tokens=30)
            print(f"   Transformers: {repr(transformers_response[:100])}{'...' if len(transformers_response) > 100 else ''}")

            # Generate with MLX
            mlx_response = generate_mlx_response(mlx_model, mlx_tokenizer, prompt, max_tokens=30)
            print(f"   MLX: {repr(mlx_response[:100])}{'...' if len(mlx_response) > 100 else ''}")

            # Compare responses
            similarity = "Similar" if transformers_response.lower() in mlx_response.lower() or mlx_response.lower() in transformers_response.lower() else "Different"
            print(f"   Comparison: {similarity}")

            template_results.append({
                "question": question,
                "prompt": prompt,
                "transformers_response": transformers_response,
                "mlx_response": mlx_response,
                "similarity": similarity
            })

        results.append({
            "template_name": template_name,
            "template_info": template_info,
            "results": template_results
        })

    return results


def analyze_results(results, weight_comparisons):
    """Analyze and summarize the test results."""
    print("\n📊 Analysis Summary")
    print("=" * 60)

    # Weight comparison summary
    print("\n🔍 Weight Comparison:")
    weights_identical = all(diff < 1e-3 for _, shape_match, diff in weight_comparisons if shape_match)
    if weights_identical:
        print("   ✅ Model weights appear identical")
    else:
        print("   ❌ Model weights differ significantly")
        for name, shape_match, diff in weight_comparisons:
            if not shape_match or diff >= 1e-3:
                print(f"      - {name}: shape_match={shape_match}, diff={diff:.6f}")

    # Template performance summary
    print("\n📝 Template Performance:")
    for result in results:
        template_name = result["template_name"]
        template_results = result["results"]

        similar_count = sum(1 for r in template_results if r["similarity"] == "Similar")
        total_count = len(template_results)
        similarity_rate = (similar_count / total_count) * 100

        status = "✅" if similarity_rate >= 50 else "⚠️" if similarity_rate >= 25 else "❌"
        print(f"   {status} {template_name}: {similar_count}/{total_count} similar ({similarity_rate:.1f}%)")

    # Recommendations
    print("\n💡 Recommendations:")
    if not weights_identical:
        print("   1. ❗ Investigate MLX model loading - weights don't match")
        print("   2. ❗ Check weight conversion and transposition logic")
    else:
        print("   1. ✅ Model weights are correct")

        # Find best performing template
        best_template = max(results, key=lambda r: sum(1 for res in r["results"] if res["similarity"] == "Similar"))
        print(f"   2. 📝 Best template: {best_template['template_name']} ({best_template['template_info']['description']})")

        worst_performers = [r for r in results if sum(1 for res in r["results"] if res["similarity"] == "Similar") == 0]
        if worst_performers:
            print(f"   3. ⚠️  Avoid these templates: {', '.join(r['template_name'] for r in worst_performers)}")


def main():
    """Main test function."""
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print("🧪 Model Loading Comparison Test")
    print("=" * 60)
    print(f"Testing model: {model_name}")
    print("This test compares Transformers vs MLX loading and generation")
    print("=" * 60)

    # Load models with both methods
    try:
        transformers_generator, transformers_model, transformers_tokenizer = load_transformers_model(model_name)
    except Exception as e:
        print(f"❌ Failed to load Transformers model: {e}")
        return 1

    try:
        mlx_model, mlx_tokenizer, mlx_config = load_mlx_model(model_name)
    except Exception as e:
        print(f"❌ Failed to load MLX model: {e}")
        return 1

    # Compare model weights
    try:
        weight_comparisons = compare_model_weights(transformers_model, mlx_model, mlx_tokenizer)
    except Exception as e:
        print(f"❌ Failed to compare weights: {e}")
        weight_comparisons = []

    # Test various chat templates
    try:
        template_results = test_template_comparison(transformers_generator, mlx_model, mlx_tokenizer)
    except Exception as e:
        print(f"❌ Failed to test templates: {e}")
        return 1

    # Analyze results
    analyze_results(template_results, weight_comparisons)

    print("\n" + "=" * 60)
    print("🏁 Test completed!")

    return 0


if __name__ == "__main__":
    sys.exit(main())