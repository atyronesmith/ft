"""
MLX-based text generation with professional sampling strategies.

This module provides high-quality text generation for fine-tuned models,
implementing the state-of-the-art approaches outlined in OUTPUT.md.
"""

from collections.abc import Callable
from typing import Any, Optional

from loguru import logger

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None


class GenerationConfig:
    """Configuration for text generation parameters."""

    def __init__(
        self,
        max_tokens: int = 50,
        temperature: float = 0.5,
        top_p: float = 0.95,
        top_k: int = 0,  # 0 means disabled, typical Ollama default is 40
        repetition_penalty: float = 1.05,  # Reduced for better chat template compatibility
        stop_on_eos: bool = True,
        stop_on_special_tokens: bool = True,
        verbose: bool = False,
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.stop_on_eos = stop_on_eos
        self.stop_on_special_tokens = stop_on_special_tokens
        self.verbose = verbose

    @classmethod
    def for_factual_qa(cls) -> "GenerationConfig":
        """Ollama-compatible parameters for factual Q&A tasks."""
        return cls(
            max_tokens=20,
            temperature=0.8,  # Ollama default
            top_p=0.9,  # Ollama default
            top_k=40,  # Ollama default
            repetition_penalty=1.05,  # Reduced for better chat template compatibility
            verbose=False,
        )

    @classmethod
    def ollama_defaults(cls) -> "GenerationConfig":
        """Exact Ollama default parameters."""
        return cls(
            max_tokens=50,
            temperature=0.8,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.05,
            verbose=False,
        )

    @classmethod
    def for_creative_writing(cls) -> "GenerationConfig":
        """Parameters optimized for creative and diverse text."""
        return cls(
            max_tokens=100,
            temperature=0.8,
            top_p=0.9,
            verbose=False,
        )

    @classmethod
    def for_code_generation(cls) -> "GenerationConfig":
        """Parameters optimized for code generation."""
        return cls(
            max_tokens=150,
            temperature=0.2,
            top_p=0.95,
            verbose=False,
        )


class MLXTextGenerator:
    """Professional text generator for MLX models."""

    def __init__(self, model, tokenizer, config: Optional[GenerationConfig] = None):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is not available. Please install MLX.")

        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self._debug_fn: Optional[Callable[[str], None]] = None

    def set_debug_callback(self, debug_fn: Callable[[str], None]) -> None:
        """Set a debug callback function for verbose output."""
        self._debug_fn = debug_fn

    def _debug(self, message: str) -> None:
        """Internal debug logging."""
        if self.config.verbose and self._debug_fn:
            self._debug_fn(message)
        elif self.config.verbose:
            logger.debug(message)

    def _format_prompt(self, text: str) -> tuple[str, list[int]]:
        """Use the provided text as-is (MLX simple text format, no chat template)."""
        # For MLX-style generation, use the text directly without any chat template formatting
        # This matches the official MLX examples approach
        formatted_prompt = text

        # Simple tokenization without special tokens (MLX style)
        input_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        if isinstance(input_ids, list):
            # Already a list
            pass
        else:
            # Handle batched output from tokenizer
            input_ids = input_ids[0].tolist() if hasattr(input_ids[0], 'tolist') else input_ids.tolist()

        self._debug("Using simple text format (MLX style)")
        self._debug(f"Prompt: {repr(formatted_prompt[:100])}...")

        if self.config.verbose:
            print("Input IDs:", input_ids)
            print("Input IDs (as tokens):", [self.tokenizer.decode([tid]) for tid in input_ids])
            print("\n=== EXACT PROMPT SENT TO MODEL ===")
            print(repr(formatted_prompt))
            print("=== END PROMPT ===\n")

        return formatted_prompt, input_ids

    def _sample_next_token(self, logits: Any, temperature: float, top_p: float, top_k: int) -> int:
        """Sample next token with Ollama-compatible sampling strategy.

        Applies: temperature scaling → top-k filtering → top-p filtering → sampling
        """
        # Apply temperature scaling first
        if temperature <= 1e-6:
            argmax_result = mx.argmax(logits)
            if hasattr(argmax_result, "item"):
                return int(argmax_result.item())
            else:
                return int(argmax_result)

        scaled_logits = logits / temperature

        # Apply top-k filtering (limit to k most probable tokens)
        if top_k > 0:
            # Get the k-th largest logit value as threshold
            top_k_logits = mx.topk(scaled_logits, k=min(top_k, scaled_logits.shape[0]))
            threshold = top_k_logits[-1]  # k-th largest value
            # Mask out tokens below threshold
            scaled_logits = mx.where(scaled_logits >= threshold, scaled_logits, -float("inf"))

        # Apply top-p filtering (nucleus sampling) - simplified for MLX compatibility
        if 0.0 < top_p < 1.0:
            # Convert to probabilities
            probs = mx.softmax(scaled_logits, axis=-1)

            # Simple approach: mask out tokens below a threshold
            # This approximates top-p behavior in an MLX-compatible way
            prob_threshold = (1.0 - top_p) * mx.max(probs)
            scaled_logits = mx.where(probs >= prob_threshold, scaled_logits, -float("inf"))

        # Sample from the filtered distribution
        sample = mx.random.categorical(scaled_logits)
        # Handle both integer and array returns from categorical
        if hasattr(sample, "item"):
            return int(sample.item())
        else:
            return int(sample)

    def _should_stop(self, tokens: list[int], step: int, question: str) -> tuple[bool, str]:
        """Determine if generation should stop based on content and conditions."""
        if step >= self.config.max_tokens:
            return True, "max_tokens"

        if len(tokens) < 2:
            return False, ""

        try:
            partial_text = self.tokenizer.decode(tokens, skip_special_tokens=True)

            # For SQL generation: stop when we have a complete SQL query
            if "SELECT" in partial_text and len(partial_text.strip()) >= 10:
                # Check if we have what looks like a complete SQL statement
                sql_text = partial_text.strip()

                # Stop if we have a complete SELECT statement (basic heuristic)
                if (sql_text.count("SELECT") == 1 and
                    ("FROM" in sql_text or "WHERE" in sql_text) and
                    step >= 5):  # Minimum tokens for meaningful SQL
                    return True, "complete_sql_query"

                # Stop if the query ends naturally (e.g., with a semicolon or looks complete)
                if sql_text.endswith(";") or (len(sql_text) >= 20 and step >= 8):
                    return True, "sql_query_end"

            # General stopping patterns for any text generation
            if len(partial_text.strip()) >= 3:
                # Stop on natural completion patterns
                if partial_text.strip().endswith((".", "!", "?", ";")):
                    words = partial_text.strip().split()
                    if len(words) >= 3:  # Minimum meaningful response
                        return True, "natural_completion"

            # Prevent overly long responses (adjusted for SQL which can be longer)
            if len(tokens) >= 25:  # Increased from 15 for SQL queries
                return True, "length_limit"

        except Exception:
            pass

        return False, ""

    def generate_simple(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7) -> str:
        """Simple MLX-style generation following canonical patterns.

        This method follows the official MLX examples approach exactly.
        Based on research from mlx-examples/llms/llama/llama.py and mlx-lm/generate.py

        Args:
            prompt: Input text to generate from
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text string
        """
        # 1. CANONICAL MLX INPUT ENCODING
        # MLX expects simple encoding without special tokens for SQL format
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if isinstance(input_ids, list):
            tokens = mx.array(input_ids, dtype=mx.int32)  # 1D array
        else:
            tokens = mx.array(input_ids[0], dtype=mx.int32)  # Handle batched input

        # 2. CANONICAL MLX CACHE INITIALIZATION
        # Important: cache must be None initially, NOT a list
        cache = None

        # 3. CANONICAL MLX SAMPLING FUNCTION (exact MLX pattern)
        def sample(logits):
            if temperature <= 1e-6:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits / temperature)

        # 4. CANONICAL MLX GENERATION LOOP (exact MLX pattern)
        y = tokens  # Start with full prompt
        generated_tokens = []

        for step in range(max_tokens):
            # KEY: Add batch dimension: y[None] transforms [seq] -> [1, seq]
            logits, cache = self.model(y[None], cache=cache)

            # Get logits for last token (shape: [1, vocab_size])
            logits = logits[:, -1, :]

            # Sample next token
            next_token = sample(logits)

            # Extract token ID safely - MLX arrays need .item() for scalar conversion
            mx.eval(next_token)  # Force evaluation
            token_id = int(next_token.item())

            # Check for EOS token
            if (self.tokenizer.eos_token_id is not None and
                token_id == self.tokenizer.eos_token_id):
                break

            generated_tokens.append(token_id)

            # CRITICAL: For next iteration, use only the new token
            # This is incremental generation - key MLX pattern
            y = mx.array([token_id], dtype=mx.int32)

            # Memory management (every 10 steps)
            if step % 10 == 0:
                mx.eval(y)  # Force evaluation for memory management

        # Decode only the generated tokens (not the input prompt)
        if generated_tokens:
            try:
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            except Exception:
                # Fallback: decode each token individually
                generated_text = "".join([self.tokenizer.decode([t], skip_special_tokens=True)
                                        for t in generated_tokens])
        else:
            generated_text = ""

        return generated_text.strip()

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Generate text from the given prompt.

        Args:
            prompt: Input text to generate from
            config: Optional generation config to override defaults

        Returns:
            Generated text string
        """
        # Use simple generation by default (MLX canonical approach)
        if config is None:
            return self.generate_simple(prompt, max_tokens=self.config.max_tokens, temperature=self.config.temperature)

        # Use simple generation with config parameters for better MLX compatibility
        return self.generate_simple(prompt, max_tokens=config.max_tokens, temperature=config.temperature)

    def generate_advanced(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Advanced generation with all features (renamed from original generate).

        Args:
            prompt: Input text to generate from
            config: Optional generation config to override defaults

        Returns:
            Generated text string
        """
        if config:
            original_config = self.config
            self.config = config

        try:
            return self._generate_internal(prompt)
        finally:
            if config:
                self.config = original_config

    def _apply_repetition_penalty(self, logits: Any, input_ids: list[int], penalty: float) -> Any:
        """Apply repetition penalty to logits to discourage repeated tokens."""
        if penalty == 1.0:
            return logits

        # Start with original logits
        penalty_logits = logits

        # Count token frequencies for more aggressive penalty on frequently repeated tokens
        token_counts = {}
        for token_id in input_ids:
            token_counts[token_id] = token_counts.get(token_id, 0) + 1

        for token_id, count in token_counts.items():
            # Apply stronger penalty for tokens that appear multiple times
            effective_penalty = penalty ** min(
                count, 3
            )  # Cap at penalty^3 for very frequent tokens

            # Always penalize repeated tokens by dividing logits (consistent penalty)
            penalty_logits = mx.where(
                mx.arange(logits.shape[0]) == token_id,
                logits[token_id] / effective_penalty,
                penalty_logits,
            )

        return penalty_logits

    def _generate_internal(self, prompt: str) -> str:
        """Internal generation implementation."""
        try:
            # Format prompt and get input tokens
            formatted_prompt, input_ids = self._format_prompt(prompt)
            input_tensor = mx.array(input_ids).astype(mx.int32).reshape(1, -1)

            self._debug(f"Using prompt: {repr(formatted_prompt[:100])}...")
            self._debug(f"Input token count: {len(input_ids)}")

            generated_tokens = []
            current_ids = input_tensor

            for step in range(self.config.max_tokens):
                # Forward pass - use model() to properly trigger LoRA layers
                logits = self.model(current_ids)[0]
                next_logits = logits[0, -1, :]

                # Apply repetition penalty ONLY to generated tokens, not template tokens
                next_logits = self._apply_repetition_penalty(
                    next_logits, generated_tokens, self.config.repetition_penalty
                )

                # Mask unwanted tokens
                if self.tokenizer.unk_token_id is not None:
                    next_logits = mx.where(
                        mx.arange(next_logits.shape[0]) == self.tokenizer.unk_token_id,
                        -float("inf"),
                        next_logits,
                    )

                # CRITICAL FIX: Mask EOS token for first few steps to prevent immediate termination
                # MLX-native trained models aggressively generate EOS tokens
                if step < 3 and self.tokenizer.eos_token_id is not None:
                    next_logits = mx.where(
                        mx.arange(next_logits.shape[0]) == self.tokenizer.eos_token_id,
                        -float("inf"),
                        next_logits,
                    )
                    self._debug(f"Masked EOS token at step {step}")

                # Sample next token
                next_token_id = self._sample_next_token(
                    next_logits, self.config.temperature, self.config.top_p, self.config.top_k
                )
                generated_tokens.append(next_token_id)

                self._debug(
                    f"Step {step}: token_id={next_token_id}, token='{self.tokenizer.decode([next_token_id])}'"
                )

                # Check stopping conditions - be much more lenient for MLX-native trained models
                if self.config.stop_on_eos and next_token_id == self.tokenizer.eos_token_id:
                    # CRITICAL FIX: MLX-native training produces models that aggressively generate EOS
                    # Don't stop on EOS unless we have substantial meaningful content
                    meaningful_tokens = [t for t in generated_tokens if t not in [13, 29871, 2, 1]]  # Exclude newlines, spaces, EOS, BOS

                    if step < 10 or len(meaningful_tokens) < 3:
                        self._debug(f"Ignoring early EOS at step {step} - only {len(meaningful_tokens)} meaningful tokens")
                        # Continue generation instead of stopping
                    else:
                        self._debug(f"Hit EOS at step {step} with {len(meaningful_tokens)} meaningful tokens")
                        break

                if self.config.stop_on_special_tokens:
                    try:
                        # Simple check for EOS and basic stop tokens
                        current_text = self.tokenizer.decode(generated_tokens)

                        if "</s>" in current_text:
                            self._debug("Hit </s> token")
                            break

                    except Exception:
                        pass

                # Content-based stopping
                should_stop, reason = self._should_stop(generated_tokens, step, prompt)
                if should_stop:
                    self._debug(f"Stopping due to: {reason}")
                    break

                # Add new token and continue
                next_token_tensor = mx.array([[next_token_id]])
                current_ids = mx.concatenate([current_ids, next_token_tensor], axis=1)

            # Decode the generated tokens
            if generated_tokens:
                generated_text = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                ).strip()
                # Clean up any remaining chat template artifacts
                if generated_text.endswith("<|"):
                    generated_text = generated_text[:-2].strip()
            else:
                generated_text = "[No response]"

            self._debug(f"Final generated text: '{generated_text}'")
            return generated_text

        except Exception as e:
            self._debug(f"Generation error: {e}")
            logger.error(f"Generation failed: {e}")
            return f"[Error: {e}]"


def create_tokenizer_with_special_tokens(model_id: str):
    """Create tokenizer with special tokens properly registered."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Add special tokens if they don't exist
    special_tokens = {"additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"]}

    # Only add tokens that aren't already in the tokenizer
    new_tokens = []
    for token in special_tokens["additional_special_tokens"]:
        if token not in tokenizer.get_vocab():
            new_tokens.append(token)

    if new_tokens:
        logger.info(f"Adding {len(new_tokens)} special tokens: {new_tokens}")
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

    return tokenizer


def load_model_with_special_tokens(model_id: str):
    """Load model and tokenizer with special tokens properly configured."""
    from finetune.models.manager import ModelManager

    # Load model and use its tokenizer, then add special tokens
    manager = ModelManager()
    model, tokenizer, _ = manager.load_model(model_id)

    # Add special tokens to the returned tokenizer
    tokenizer = create_tokenizer_with_special_tokens(model_id)

    # CRITICAL DEBUG VALIDATION
    try:
        # For MLX models, use flatten_params to properly count parameters
        from finetune.models.mlx_models import flatten_params

        flat_params = flatten_params(model.parameters())
        param_count = sum(p.size for p in flat_params.values() if hasattr(p, "size"))
    except Exception:
        # Fallback counting method
        param_count = getattr(model, "num_parameters", 0)

    vocab_size = len(tokenizer)

    logger.info("🔍 MODEL DEBUG VALIDATION:")
    logger.info(f"   Model parameters: {param_count:,}")
    logger.info(f"   Tokenizer vocab size: {vocab_size}")
    logger.info(f"   Model config vocab size: {model.config.vocab_size}")

    # Validate critical conditions
    if param_count == 0:
        raise ValueError("CRITICAL: Model loaded with 0 parameters!")

    if vocab_size != model.config.vocab_size:
        logger.warning(f"VOCAB MISMATCH: Tokenizer={vocab_size}, Model={model.config.vocab_size}")

    # Test special tokens
    special_tokens = ["<|system|>", "<|user|>", "<|assistant|>"]
    for token in special_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:
            logger.info(f"   ✅ {token} -> single token {token_ids[0]}")
        else:
            logger.warning(f"   ❌ {token} -> multiple tokens {token_ids}")

    return model, tokenizer, model.config


def load_model_and_tokenizer(model_id: str):
    """Load model and tokenizer, ensuring they are correctly configured."""
    from finetune.models.manager import ModelManager

    # Load the model and use its tokenizer (no special token modifications)
    manager = ModelManager()
    model, tokenizer, _ = manager.load_model(model_id)  # Use original tokenizer

    # DEBUG VALIDATION
    try:
        from finetune.models.mlx_models import flatten_params

        flat_params = flatten_params(model.parameters())
        param_count = sum(p.size for p in flat_params.values() if hasattr(p, "size"))
    except Exception:
        param_count = getattr(model, "num_parameters", 0)

    vocab_size = len(tokenizer)

    logger.info("🔍 MODEL DEBUG VALIDATION:")
    logger.info(f"   Model parameters: {param_count:,}")
    logger.info(f"   Tokenizer vocab size: {vocab_size}")
    logger.info(f"   Model config vocab size: {model.config.vocab_size}")

    # Validate critical conditions
    if param_count == 0:
        raise ValueError("CRITICAL: Model loaded with 0 parameters!")

    if vocab_size != model.config.vocab_size:
        logger.info(
            f"VOCAB INFO: Using original TinyLlama vocab (tokenizer={vocab_size}, model={model.config.vocab_size})"
        )

    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    config: Optional[GenerationConfig] = None,
    debug_fn: Optional[Callable[[str], None]] = None,
) -> str:
    """Convenience function for text generation.

    Args:
        model: MLX model for generation
        tokenizer: Tokenizer for the model
        prompt: Input text to generate from
        config: Generation configuration
        debug_fn: Optional debug callback function

    Returns:
        Generated text string
    """
    # Note: For proper special token support, use load_model_and_tokenizer()
    # to ensure the model and tokenizer are properly synchronized
    generator = MLXTextGenerator(model, tokenizer, config)
    if debug_fn:
        generator.set_debug_callback(debug_fn)
    return generator.generate(prompt)
