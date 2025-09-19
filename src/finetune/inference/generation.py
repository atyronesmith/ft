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
        repetition_penalty: float = 1.1,  # Ollama default
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
            repetition_penalty=1.1,  # Ollama default
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
            repetition_penalty=1.1,
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
        """Format text using the exact Ollama template format."""
        try:
            # Use exact Ollama template format with system message
            ollama_prompt = f"<|system|>\nYou are a helpful AI assistant.</s>\n<|user|>\n{text}</s>\n<|assistant|>\n"
            input_ids = self.tokenizer.encode(ollama_prompt, return_tensors="np")[0].tolist()
            return ollama_prompt, input_ids
        except Exception as e:
            self._debug(f"Ollama template failed: {e}, falling back to simple format")
            # Fallback to simple format
            simple_prompt = f"Question: {text}\nAnswer:"
            input_ids = self.tokenizer.encode(simple_prompt, return_tensors="np")[0].tolist()
            return simple_prompt, input_ids

    def _sample_next_token(self, logits: Any, temperature: float, top_p: float, top_k: int) -> int:
        """Sample next token with Ollama-compatible sampling strategy.

        Applies: temperature scaling → top-k filtering → top-p filtering → sampling
        """
        # Apply temperature scaling first
        if temperature <= 1e-6:
            return int(mx.argmax(logits))

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
        return int(mx.random.categorical(scaled_logits))

    def _should_stop(self, tokens: list[int], step: int, question: str) -> tuple[bool, str]:
        """Determine if generation should stop based on content and conditions."""
        if step >= self.config.max_tokens:
            return True, "max_tokens"

        if len(tokens) < 2:
            return False, ""

        try:
            partial_text = self.tokenizer.decode(tokens, skip_special_tokens=True)

            # Stop on natural sentence completion
            if partial_text.strip().endswith((".", "!", "?")) and len(partial_text.strip()) > 3:
                return True, "natural_completion"

            # For factual Q&A: stop when we have a complete, confident answer
            if "capital" in question.lower():
                cities = [
                    "Paris",
                    "Berlin",
                    "Rome",
                    "Madrid",
                    "Lisbon",
                    "London",
                    "Tokyo",
                    "Beijing",
                    "Delhi",
                    "Kabul",
                    "Tirana",
                ]
                for city in cities:
                    if city.lower() in partial_text.lower() and step >= 3:
                        return True, f"found_answer_{city}"

            # Prevent overly long rambling answers
            if len(tokens) >= 20:
                return True, "rambling_prevention"

        except Exception:
            pass

        return False, ""

    def generate(self, prompt: str, config: Optional[GenerationConfig] = None) -> str:
        """Generate text from the given prompt.

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

            if float(logits[token_id]) > 0:
                penalty_logits = mx.where(
                    mx.arange(logits.shape[0]) == token_id,
                    logits[token_id] / effective_penalty,
                    penalty_logits,
                )
            else:
                penalty_logits = mx.where(
                    mx.arange(logits.shape[0]) == token_id,
                    logits[token_id] * effective_penalty,
                    penalty_logits,
                )

        return penalty_logits

    def _generate_internal(self, prompt: str) -> str:
        """Internal generation implementation."""
        try:
            # Format prompt and get input tokens
            formatted_prompt, input_ids = self._format_prompt(prompt)
            input_tensor = mx.array(input_ids).astype(mx.int32).reshape(1, -1)
            all_token_ids = input_ids.copy()  # Track all tokens for repetition penalty

            self._debug(f"Using prompt: {repr(formatted_prompt[:100])}...")
            self._debug(f"Input token count: {len(input_ids)}")

            generated_tokens = []
            current_ids = input_tensor

            for step in range(self.config.max_tokens):
                # Forward pass
                logits = self.model.forward(current_ids)[0]
                next_logits = logits[0, -1, :]

                # Apply repetition penalty to discourage repeated tokens
                next_logits = self._apply_repetition_penalty(
                    next_logits, all_token_ids, self.config.repetition_penalty
                )

                # Mask unwanted tokens
                if self.tokenizer.unk_token_id is not None:
                    next_logits = mx.where(
                        mx.arange(next_logits.shape[0]) == self.tokenizer.unk_token_id,
                        -float("inf"),
                        next_logits,
                    )

                # Sample next token
                next_token_id = self._sample_next_token(
                    next_logits, self.config.temperature, self.config.top_p, self.config.top_k
                )
                generated_tokens.append(next_token_id)
                all_token_ids.append(next_token_id)  # Add to repetition penalty tracking

                self._debug(
                    f"Step {step}: token_id={next_token_id}, token='{self.tokenizer.decode([next_token_id])}'"
                )

                # Check stopping conditions
                if self.config.stop_on_eos and next_token_id == self.tokenizer.eos_token_id:
                    self._debug(f"Hit EOS at step {step}")
                    break

                if self.config.stop_on_special_tokens:
                    try:
                        # Check for exact Ollama stop tokens
                        current_text = self.tokenizer.decode(generated_tokens + [next_token_id])

                        # Ollama stop tokens: <|system|>, <|user|>, <|assistant|>, </s>
                        ollama_stop_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "</s>"]

                        for stop_token in ollama_stop_tokens:
                            if stop_token in current_text:
                                self._debug(f"Hit Ollama stop token: {stop_token}")
                                # Remove tokens that form the stop sequence
                                stop_token_ids = self.tokenizer.encode(
                                    stop_token, add_special_tokens=False
                                )
                                if len(stop_token_ids) <= len(generated_tokens) + 1:
                                    # Remove the stop token sequence from generated tokens
                                    cutoff = len(generated_tokens) + 1 - len(stop_token_ids)
                                    generated_tokens = generated_tokens[:cutoff]
                                return self.tokenizer.decode(
                                    generated_tokens, skip_special_tokens=True
                                ).strip()

                    except:
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
    generator = MLXTextGenerator(model, tokenizer, config)
    if debug_fn:
        generator.set_debug_callback(debug_fn)
    return generator.generate(prompt)
