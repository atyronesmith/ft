"""
MLX Examples compatible data loader.

This module provides data loading functionality that matches the MLX LoRA examples
approach exactly, enabling direct comparison and validation of our training pipeline.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class MLXDataset:
    """
    Light-weight wrapper to hold lines from a jsonl file.

    This exactly matches the Dataset class from MLX LoRA examples for compatibility.
    """

    def __init__(self, path: Path, key: str = "text"):
        """Load JSONL data from file.

        Args:
            path: Path to JSONL file
            key: Key to extract from each JSON object (default "text")
        """
        if not path.exists():
            self._data = None
        else:
            with open(path, "r") as fid:
                self._data = [json.loads(line) for line in fid]
        self._key = key

    def __getitem__(self, idx: int) -> str:
        """Get text at given index."""
        if self._data is None:
            raise IndexError("Dataset is not loaded")
        return self._data[idx][self._key]

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self._data) if self._data is not None else 0


def load_mlx_datasets(data_dir: str | Path) -> tuple[MLXDataset, MLXDataset, MLXDataset]:
    """
    Load train, valid, test datasets from MLX examples format.

    This exactly matches the load() function from MLX LoRA examples.

    Args:
        data_dir: Directory containing {train,valid,test}.jsonl files

    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset)

    Raises:
        FileNotFoundError: If required data files are missing
        ValueError: If datasets are empty when required
    """
    data_path = Path(data_dir)

    def load_and_check(name: str) -> MLXDataset:
        dataset_path = data_path / f"{name}.jsonl"
        try:
            return MLXDataset(dataset_path)
        except Exception as e:
            print(f"Unable to build dataset {dataset_path} ({e})")
            raise

    names = ("train", "valid", "test")
    train, valid, test = (load_and_check(n) for n in names)

    # Validate datasets (same checks as MLX examples)
    if len(train) == 0:
        raise ValueError(
            "Training set not found or empty. Must provide training set for fine-tuning."
        )
    if len(valid) == 0:
        raise ValueError(
            "Validation set not found or empty. Must provide validation set for fine-tuning."
        )
    if len(test) == 0:
        print("Warning: Test set not found or empty.")

    return train, valid, test


def iterate_batches_mlx(dataset: MLXDataset, tokenizer, batch_size: int, train: bool = False):
    """
    Create batches exactly like MLX LoRA examples.

    This exactly matches the iterate_batches() function from MLX LoRA examples
    to ensure identical training behavior.

    Args:
        dataset: MLX dataset to iterate over
        tokenizer: Tokenizer to encode text
        batch_size: Number of examples per batch
        train: Whether this is for training (enables shuffling)

    Yields:
        Tuple of (inputs, targets, lengths) where:
        - inputs: [batch_size, seq_len-1] token IDs for input
        - targets: [batch_size, seq_len-1] token IDs for targets (shifted by 1)
        - lengths: [batch_size] sequence lengths
    """
    # Shuffle indices
    while True:
        indices = np.arange(len(dataset))
        if train:
            indices = np.random.permutation(indices)

        # Collect batches from dataset
        for i in range(0, len(indices) - batch_size + 1, batch_size):
            # Encode batch - exactly like MLX examples
            batch = [tokenizer.encode(dataset[indices[i + j]]) for j in range(batch_size)]
            lengths = [len(x) for x in batch]

            # Check if any sequence is longer than 2048 tokens
            if max(lengths) > 2048:
                print(
                    "[WARNING] Some sequences are longer than 2048 tokens. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to the max length - exactly like MLX examples
            batch_arr = np.zeros((batch_size, max(lengths)), np.int32)

            for j in range(batch_size):
                batch_arr[j, : lengths[j]] = batch[j]
            batch = mx.array(batch_arr)

            # Return inputs (:-1), targets (1:), and lengths - exactly like MLX examples
            yield batch[:, :-1], batch[:, 1:], mx.array(lengths)

        if not train:
            break


def compute_loss_mlx(model, inputs, targets, lengths):
    """
    Compute loss exactly like MLX LoRA examples.

    This exactly matches the loss() function from MLX LoRA examples.

    Args:
        model: The model to compute loss for
        inputs: Input token IDs [batch_size, seq_len]
        targets: Target token IDs [batch_size, seq_len]
        lengths: Sequence lengths [batch_size]

    Returns:
        Tuple of (loss, num_tokens)
    """
    # Run model on inputs - exactly like MLX examples
    logits, _ = model(inputs)
    # Keep logits in bfloat16 for memory efficiency

    # Mask padding tokens - exactly like MLX examples
    length_mask = mx.arange(inputs.shape[1])[None, :] < lengths[:, None]

    # Calculate the loss - exactly like MLX examples
    ce = nn.losses.cross_entropy(logits, targets) * length_mask
    ntoks = length_mask.sum()
    ce = ce.sum() / ntoks
    return ce, ntoks


def evaluate_mlx(model, dataset: MLXDataset, tokenizer, batch_size: int, num_batches: int = -1):
    """
    Evaluate model exactly like MLX LoRA examples.

    Args:
        model: Model to evaluate
        dataset: Dataset to evaluate on
        tokenizer: Tokenizer for encoding
        batch_size: Batch size for evaluation
        num_batches: Number of batches to evaluate (-1 for all)

    Returns:
        Average loss across evaluation batches
    """
    all_losses = []
    ntokens = 0

    # num_batches can be -1 to indicate the entire set
    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for it, batch in zip(
        index_iterator,
        iterate_batches_mlx(dataset, tokenizer, batch_size),
    ):
        losses, toks = compute_loss_mlx(model, *batch)
        all_losses.append((losses * toks).item())
        ntokens += toks.item()

    return np.sum(all_losses) / ntokens


def convert_to_chat_format(mlx_dataset: MLXDataset) -> List[Dict[str, Any]]:
    """
    Convert MLX dataset format to our chat format for comparison.

    Args:
        mlx_dataset: Dataset in MLX format (raw text strings)

    Returns:
        List of conversation dictionaries in our chat format
    """
    conversations = []

    for i in range(len(mlx_dataset)):
        text = mlx_dataset[i]

        # Convert MLX text format to our chat format
        # MLX format is typically: "Question\nAnswer" or structured text
        # We'll create a simple system/user/assistant structure
        conversations.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on structured data."
                },
                {
                    "role": "user",
                    "content": f"Please help with this query: {text[:100]}..."  # Truncate for user message
                },
                {
                    "role": "assistant",
                    "content": text  # Full text as assistant response
                }
            ]
        })

    return conversations