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
    learning_rate: float = 5e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10
    output_dir: str = "./output"
    gradient_checkpointing: bool = False
    mixed_precision: bool = True
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

        # Add LoRA to model
        self.model.add_lora(lora_config)

        # Get trainable parameters
        self.trainable_params, self.trainable_count, self.total_count = get_lora_trainable_params(
            self.model
        )

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")

        # Create output directory
        self.output_dir = Path(training_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized LoRA trainer with {self.trainable_count:,} trainable parameters "
            f"out of {self.total_count:,} total ({100 * self.trainable_count / self.total_count:.2f}%)"
        )

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for training."""
        return optim.AdamW(
            learning_rate=self.training_config.learning_rate,
            betas=[self.training_config.adam_beta1, self.training_config.adam_beta2],
            eps=self.training_config.adam_epsilon,
            weight_decay=self.training_config.weight_decay,
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
            progress = (self.global_step - self.training_config.warmup_steps) / (
                self.total_steps - self.training_config.warmup_steps
            )
            return self.training_config.learning_rate * 0.5 * (1 + math.cos(math.pi * progress))

    def compute_loss(self, batch: dict[str, mx.array]) -> mx.array:
        """Compute loss for a batch."""
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)

        # Forward pass
        logits = self.model.forward(input_ids)

        # Shift for language modeling
        logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
        labels = labels[:, 1:].reshape(-1)

        # Cross-entropy loss
        loss = mx.mean(
            nn.losses.cross_entropy(
                logits,
                labels,
                reduction="none",
            )
        )

        return loss

    def training_step(self, batch: dict[str, mx.array]) -> dict[str, float]:
        """Perform a single training step."""
        # Compute loss and gradients
        loss_and_grad_fn = nn.value_and_grad(self.model, self.compute_loss)
        loss, grads = loss_and_grad_fn(batch)

        # Clip gradients
        if self.training_config.max_grad_norm > 0:
            grads = mx.clip_grad_norm(grads, self.training_config.max_grad_norm)

        # Update parameters
        self.optimizer.update(self.model, grads)

        # Update learning rate
        lr = self._get_learning_rate()
        self.optimizer.learning_rate = lr

        mx.eval(self.model.parameters())

        return {"loss": float(loss), "learning_rate": lr}

    def evaluate(self) -> dict[str, float]:
        """Evaluate the model."""
        if self.eval_dataset is None:
            return {}

        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_dataset:
            loss = self.compute_loss(batch)
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

        # Save LoRA weights
        save_lora_weights(self.model, checkpoint_dir / "lora_weights.npz")

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "optimizer_state": self.optimizer.state,
        }
        mx.save(str(checkpoint_dir / "training_state.npz"), state)

        # Save configs
        self.lora_config.__dict__.update({"_type": "lora_config"})
        mx.save(str(checkpoint_dir / "lora_config.npz"), self.lora_config.__dict__)

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: Path):
        """Load a training checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)

        # Load LoRA weights
        load_lora_weights(self.model, checkpoint_dir / "lora_weights.npz")

        # Load training state
        state = mx.load(str(checkpoint_dir / "training_state.npz"))
        self.global_step = int(state["global_step"])
        self.epoch = int(state["epoch"])
        self.best_eval_loss = float(state["best_eval_loss"])
        self.optimizer.state = state["optimizer_state"]

        logger.info(f"Loaded checkpoint from {checkpoint_dir}")

    def train(self):
        """Main training loop."""
        if self.train_dataset is None:
            raise ValueError("No training dataset provided")

        # Calculate total steps
        steps_per_epoch = len(self.train_dataset) // self.training_config.batch_size
        self.total_steps = steps_per_epoch * self.training_config.num_epochs

        logger.info(f"Starting training for {self.training_config.num_epochs} epochs")
        logger.info(f"Total steps: {self.total_steps}")

        for epoch in range(self.training_config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0

            start_time = time.time()

            for _batch_idx, batch in enumerate(self.train_dataset):
                # Training step
                metrics = self.training_step(batch)
                epoch_loss += metrics["loss"]
                num_batches += 1
                self.global_step += 1

                # Logging
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

            # End of epoch
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Final save
        self.save_checkpoint(self.output_dir / "final_model")
        logger.info("Training completed!")

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
