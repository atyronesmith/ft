"""Training utilities for fine-tuning with LoRA."""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from loguru import logger

from finetune.models.base import BaseModel
from finetune.training.lora import (
    LoRAConfig,
    get_lora_trainable_params,
    load_lora_weights,
    save_lora_weights,
)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    learning_rate: float = 1e-5  # Lower rate for naturally stable gradients
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 50  # Reduced warmup for faster convergence in tests
    max_grad_norm: float = 1.0  # Standard gradient clipping threshold
    weight_decay: float = 0.01  # Kept for compatibility, but not used with Adam optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-6  # Increased for numerical stability
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    output_dir: str = "./output"
    gradient_checkpointing: bool = False
    mixed_precision: bool = False  # Disabled to prevent float16 overflow causing NaN
    seed: int = 42


class LoRATrainer:
    """Trainer for LoRA fine-tuning."""

    def __init__(
        self,
        model: BaseModel,
        lora_config: LoRAConfig,
        training_config: TrainingConfig,
        train_dataset: Any | None = None,
        eval_dataset: Any | None = None,
        compute_metrics: Callable | None = None,
    ):
        self.model = model
        self.lora_config = lora_config
        self.training_config = training_config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics

        # Freeze entire model first (MLX examples pattern)
        self.model.freeze()

        # Apply LoRA EXACTLY like MLX examples - manual replacement of specific layers
        self._apply_lora_mlx_pattern(lora_config)

        # Get trainable parameters
        self.trainable_params, self.trainable_count, self.total_count = get_lora_trainable_params(
            self.model
        )

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # MLX value_and_grad will be created dynamically in training_step
        # to properly handle the model parameters

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")

        # Create output directory
        self.output_dir = Path(training_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate total steps for learning rate scheduling
        if self.train_dataset:
            if hasattr(self.train_dataset, "__len__"):
                steps_per_epoch = len(self.train_dataset)
            else:
                steps_per_epoch = len(list(self.train_dataset))
            self.total_steps = steps_per_epoch * self.training_config.num_epochs
        else:
            self.total_steps = 1000  # Default fallback

        logger.info(
            f"Initialized LoRA trainer with {self.trainable_count:,} trainable parameters "
            f"out of {self.total_count:,} total ({100 * self.trainable_count / self.total_count:.2f}%)"
        )

    def _apply_lora_mlx_pattern(self, lora_config: LoRAConfig):
        """Apply LoRA exactly like MLX examples - manual replacement pattern."""
        from finetune.training.lora import LoRALinear

        print("Loading pretrained model")

        # Default to 16 layers if not specified
        lora_layers = 16

        # Get the model layers - need to access the actual transformer layers
        # This assumes the model has a structure like model.model.layers or model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        elif hasattr(self.model, "layers"):
            layers = self.model.layers
        else:
            raise ValueError("Cannot find model layers for LoRA application")

        # Apply LoRA to ONLY the last lora_layers layers (MLX example pattern)
        start_layer = len(layers) - lora_layers
        logger.info(
            f"Applying LoRA to layers {start_layer} through {len(layers)-1} (last {lora_layers} layers)"
        )

        for layer_idx in range(start_layer, len(layers)):
            layer = layers[layer_idx]

            # Apply to q_proj and v_proj ONLY (just like MLX example)
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "q_proj"):
                    layer.self_attn.q_proj = LoRALinear.from_linear(
                        layer.self_attn.q_proj, rank=lora_config.r
                    )
                    logger.debug(f"Applied LoRA to layer {layer_idx} q_proj")

                if hasattr(layer.self_attn, "v_proj"):
                    layer.self_attn.v_proj = LoRALinear.from_linear(
                        layer.self_attn.v_proj, rank=lora_config.r
                    )
                    logger.debug(f"Applied LoRA to layer {layer_idx} v_proj")

            # Handle MoE models (like in MLX example)
            if hasattr(layer, "block_sparse_moe"):
                layer.block_sparse_moe.gate = LoRALinear.from_linear(
                    layer.block_sparse_moe.gate, rank=lora_config.r
                )
                logger.debug(f"Applied LoRA to layer {layer_idx} MoE gate")

        logger.info(f"Successfully applied LoRA to {lora_layers} layers using MLX pattern")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for training (following MLX examples pattern)."""
        return optim.Adam(
            learning_rate=self.training_config.learning_rate,
            betas=[self.training_config.adam_beta1, self.training_config.adam_beta2],
            eps=self.training_config.adam_epsilon,
        )

    def _get_learning_rate(self) -> float:
        """Get current learning rate with warmup and cosine schedule."""
        if self.global_step < self.training_config.warmup_steps:
            # Linear warmup
            return (
                self.training_config.learning_rate
                * self.global_step
                / self.training_config.warmup_steps
            )
        else:
            # Cosine decay
            # Add epsilon to prevent division by zero if total_steps == warmup_steps
            denominator = self.total_steps - self.training_config.warmup_steps
            progress = (self.global_step - self.training_config.warmup_steps) / (denominator + 1e-8)
            return self.training_config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    def compute_loss(self, model: BaseModel, batch: dict[str, mx.array]) -> mx.array:
        """Compute loss exactly like MLX examples pattern."""
        inputs = batch["input_ids"].astype(mx.int32)
        targets = batch["labels"].astype(mx.int32)
        lengths = batch.get("lengths")

        # Forward pass - run model on inputs (exactly like MLX example)
        logits = model.forward(inputs)
        if isinstance(logits, tuple):
            logits = logits[0]

        # Convert to float32 for numerical stability (MLX examples pattern)
        logits = logits.astype(mx.float32)

        if lengths is not None:
            # Use length-based masking like MLX examples
            length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

            # Compute cross-entropy loss (no target clipping like MLX)
            ce = nn.losses.cross_entropy(logits, targets) * length_mask
            ntoks = length_mask.sum()
            ce = ce.sum() / ntoks
            return ce
        else:
            # CRITICAL FIX: For chat training with sparse labels (-100 ignore index)
            # We need to use the shifted targets approach but handle -100 properly
            # Fallback to the original approach for backward compatibility
            # For causal LM: predict next token, so shift targets
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            shift_targets = targets[:, 1:].reshape(-1)

            # CRITICAL FIX: Proper handling of -100 ignore index for chat training
            # Only compute loss on assistant response tokens, not system/user prompts
            valid_indices = mx.where(shift_targets != -100)[0]

            if len(valid_indices) == 0:
                return mx.array(0.0, dtype=mx.float32)

            # Only compute loss on valid (non-ignored) tokens
            valid_logits = shift_logits[valid_indices]
            valid_targets = shift_targets[valid_indices]

            # Compute cross-entropy loss only on assistant tokens
            ce_loss = nn.losses.cross_entropy(valid_logits, valid_targets, reduction="mean")

            return ce_loss

    def training_step(self, batch: dict[str, mx.array]) -> dict[str, float]:
        """Perform a single training step using MLX canonical pattern."""

        # MLX canonical training pattern:
        # 1. Create value_and_grad function with model and loss
        # 2. Compute loss and gradients
        # 3. Update optimizer
        # 4. Evaluate everything

        # Step 1: Create value_and_grad function
        loss_value_and_grad = nn.value_and_grad(self.model, self.compute_loss)

        # Step 2: Compute loss and gradients
        loss, grads = loss_value_and_grad(self.model, batch)

        # Step 3: Update optimizer with gradients
        self.optimizer.update(self.model, grads)

        # Step 4: Evaluate all operations
        mx.eval(self.model.parameters(), self.optimizer.state, loss)

        # Update learning rate
        lr = self._get_learning_rate()
        self.optimizer.learning_rate = lr

        return {"loss": loss.item(), "learning_rate": lr}

    def evaluate(self) -> dict[str, float]:
        """Evaluate the model."""
        if self.eval_dataset is None:
            return {}

        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_dataset:
            loss = self.compute_loss(self.model, batch)
            total_loss += float(loss)
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        metrics = {"eval_loss": avg_loss}

        if self.compute_metrics:
            additional_metrics = self.compute_metrics(self.eval_dataset, self.model)
            metrics.update(additional_metrics)

        return metrics

    def save_checkpoint(self, checkpoint_dir: Path | None = None):
        """Save a training checkpoint."""
        if checkpoint_dir is None:
            checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights (arrays) via npz
        save_lora_weights(self.model, checkpoint_dir / "lora_weights.npz")

        # Save training state as JSON (avoid mx.save on dict)
        import json as _json

        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
        }
        (checkpoint_dir / "training_state.json").write_text(_json.dumps(state, indent=2))

        # Save configs as JSON
        (checkpoint_dir / "lora_config.json").write_text(
            _json.dumps(self.lora_config.__dict__, indent=2)
        )

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def cleanup_old_training_runs(self, max_runs: int = 3):
        """Clean up old training runs, keeping only the most recent ones."""
        import shutil
        from datetime import datetime

        if not self.output_dir.exists():
            return

        # Find all final_model directories in parent directory
        parent_dir = self.output_dir.parent
        if not parent_dir.exists():
            return

        # Look for directories that contain final_model subdirectories
        training_runs = []
        for item in parent_dir.iterdir():
            if item.is_dir():
                final_model_path = item / "final_model"
                if final_model_path.exists() and (final_model_path / "lora_weights.npz").exists():
                    # Get modification time of the final_model directory
                    mtime = final_model_path.stat().st_mtime
                    training_runs.append((mtime, item))

        # Sort by modification time (newest first)
        training_runs.sort(key=lambda x: x[0], reverse=True)

        # Keep only the most recent max_runs
        if len(training_runs) > max_runs:
            runs_to_remove = training_runs[max_runs:]
            logger.info(
                f"Cleaning up {len(runs_to_remove)} old training runs (keeping {max_runs} most recent)"
            )

            for mtime, run_dir in runs_to_remove:
                try:
                    timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                    logger.info(f"Removing old training run: {run_dir.name} (from {timestamp})")
                    shutil.rmtree(run_dir)
                except Exception as e:
                    logger.warning(f"Failed to remove {run_dir}: {e}")
        else:
            logger.info(
                f"Found {len(training_runs)} training runs, no cleanup needed (max: {max_runs})"
            )

    def load_checkpoint(self, checkpoint_dir: Path):
        """Load a training checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)

        # Load LoRA weights
        load_lora_weights(self.model, checkpoint_dir / "lora_weights.npz")

        # Load training state from JSON
        import json as _json

        state = _json.loads((checkpoint_dir / "training_state.json").read_text())
        self.global_step = int(state.get("global_step", 0))
        self.epoch = int(state.get("epoch", 0))
        self.best_eval_loss = float(state.get("best_eval_loss", float("inf")))

        logger.info(f"Loaded checkpoint from {checkpoint_dir}")

    def train(self):
        """Main training loop."""
        if self.train_dataset is None:
            raise ValueError("No training dataset provided")

        # Calculate total steps
        # If dataset already consists of tokenized dict batches, treat each as a step
        if (
            isinstance(self.train_dataset, list)
            and self.train_dataset
            and isinstance(self.train_dataset[0], dict)
        ):
            steps_per_epoch = len(self.train_dataset)
        else:
            steps_per_epoch = len(self.train_dataset) // self.training_config.batch_size
        self.total_steps = steps_per_epoch * self.training_config.num_epochs

        logger.info(f"Starting training for {self.training_config.num_epochs} epochs")
        logger.info(f"Total steps: {self.total_steps}")

        # Set up interrupt handling
        import signal

        interrupted = False

        def handle_interrupt(_signum, _frame):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                logger.warning(
                    "🛑 Training interrupted by user (Ctrl+C). Finishing current batch and stopping gracefully..."
                )
                logger.warning("Press Ctrl+C again to force exit immediately.")
            else:
                logger.error("💥 Force exit requested. Stopping immediately.")
                raise KeyboardInterrupt("Force exit")

        # Install signal handler
        original_handler = signal.signal(signal.SIGINT, handle_interrupt)

        try:
            for epoch in range(self.training_config.num_epochs):
                logger.info(f"🔄 Starting epoch {epoch + 1}/{self.training_config.num_epochs}")
                if interrupted:
                    logger.info(f"Training stopped at epoch {epoch + 1} due to user interrupt")
                    break

                self.epoch = epoch
                epoch_loss = 0.0
                num_batches = 0

                start_time = time.time()
                logger.info(
                    f"📊 Epoch {epoch + 1}: Starting batch iteration over {len(self.train_dataset)} batches"
                )

                for _batch_idx, batch in enumerate(self.train_dataset):
                    if interrupted:
                        logger.info(
                            f"Training stopped at batch {_batch_idx + 1} due to user interrupt"
                        )
                        break

                    # Training step
                    try:
                        metrics = self.training_step(batch)
                    except Exception as e:
                        logger.error(f"Training step failed at batch {_batch_idx + 1}: {e}")
                        logger.error(f"Exception type: {type(e).__name__}")
                        import traceback

                        logger.error(f"Traceback: {traceback.format_exc()}")
                        raise
                    epoch_loss += metrics["loss"]
                    num_batches += 1
                    self.global_step += 1

                    # Calculate timing metrics
                    elapsed = time.time() - start_time
                    steps_per_sec = num_batches / elapsed if elapsed > 0 else 0
                    avg_loss = epoch_loss / num_batches

                    logger.info(
                        f"✅ Epoch {epoch + 1}/{self.training_config.num_epochs} | "
                        f"Step {self.global_step}/{self.total_steps} | "
                        f"loss={metrics['loss']:.4f} | "
                        f"avg_loss={avg_loss:.4f} | "
                        f"lr={metrics['learning_rate']:.2e} | "
                        f"steps/s={steps_per_sec:.2f}"
                    )

                    # Verbose progress logging for each batch step
                    import os

                    verbose_mode = os.environ.get("FT_E2E_VERBOSE", "0") == "1"
                    if verbose_mode:
                        current_loss = metrics["loss"]
                        avg_loss = epoch_loss / num_batches
                        elapsed = time.time() - start_time
                        steps_per_sec = num_batches / elapsed if elapsed > 0 else 0

                        logger.info(
                            f"[VERBOSE] Epoch {epoch + 1}/{self.training_config.num_epochs} | "
                            f"Step {self.global_step}/{self.total_steps} | "
                            f"Batch {_batch_idx + 1} | "
                            f"Current Loss: {current_loss:.4f} | "
                            f"Avg Loss: {avg_loss:.4f} | "
                            f"LR: {metrics['learning_rate']:.2e} | "
                            f"Steps/s: {steps_per_sec:.2f}"
                        )

                    # Standard logging
                    if self.global_step % self.training_config.logging_steps == 0:
                        avg_loss = epoch_loss / num_batches
                        elapsed = time.time() - start_time
                        steps_per_sec = num_batches / elapsed

                        logger.info(
                            f"Epoch {epoch + 1}/{self.training_config.num_epochs} | "
                            f"Step {self.global_step}/{self.total_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {metrics['learning_rate']:.2e} | "
                            f"Steps/s: {steps_per_sec:.2f}"
                        )

                    # Evaluation
                    if (
                        self.training_config.eval_steps > 0
                        and self.global_step % self.training_config.eval_steps == 0
                    ):
                        eval_metrics = self.evaluate()
                        logger.info(f"Evaluation metrics: {eval_metrics}")

                        # Save best model
                        if eval_metrics.get("eval_loss", float("inf")) < self.best_eval_loss:
                            self.best_eval_loss = eval_metrics["eval_loss"]
                            self.save_checkpoint(self.output_dir / "best_model")

                    # Save checkpoint
                    if (
                        self.training_config.save_steps > 0
                        and self.global_step % self.training_config.save_steps == 0
                    ):
                        self.save_checkpoint()

                # End of epoch (break out if interrupted)
                if interrupted:
                    break

                avg_epoch_loss = epoch_loss / max(num_batches, 1)
                logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

            # Final save (even if interrupted)
            self.save_checkpoint(self.output_dir / "final_model")

            # Clean up old training runs (keep only 3 most recent)
            self.cleanup_old_training_runs(max_runs=3)

            if interrupted:
                logger.info("Training interrupted but checkpoint saved.")
            else:
                logger.info("Training completed!")

        except KeyboardInterrupt:
            # Handle force exit (second Ctrl+C)
            logger.error("Training force-stopped by user.")
            raise
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)

        return self.model


class SimpleDataLoader:
    """Simple data loader for batching."""

    def __init__(self, data: list, batch_size: int, shuffle: bool = True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = mx.random.permutation(len(self.data))
        else:
            indices = mx.arange(len(self.data))

        for i in range(0, len(self.data), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = [self.data[int(idx)] for idx in batch_indices]

            # Collate batch
            input_ids = mx.stack([item["input_ids"] for item in batch])
            labels = mx.stack([item.get("labels", item["input_ids"]) for item in batch])

            yield {"input_ids": input_ids, "labels": labels}

    def __len__(self):
        return len(self.data) // self.batch_size
