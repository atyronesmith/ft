# Training Data Directory

This directory contains example training datasets for MLX.

## Directory Structure

```
data/
└── mlx_examples/
    ├── test.jsonl
    ├── train.jsonl
    └── valid.jsonl
```

## Data Formats

The data is in the official MLX text format, which is a JSONL file where each line is a JSON object with a single `text` key.

```json
{"text": "table: 1-1000181-1\ncolumns: State/territory, Text/background colour, Format, Current slogan, Current series, Notes\nQ: Tell me what the notes are for South Australia \nA: SELECT Notes FROM 1-1000181-1 WHERE Current slogan = 'SOUTH AUSTRALIA'"}
```

## Available Datasets

### MLX Examples (WikiSQL)
- **Location**: `mlx_examples/`
- **Content**: A subset of the WikiSQL dataset, formatted for MLX. It contains examples of questions and SQL queries for given database tables.
- **Files**:
    - `train.jsonl`: Training set.
    - `valid.jsonl`: Validation set.
    - `test.jsonl`: Test set.
- **Use Case**: Fine-tuning models for SQL generation and question-answering on structured data.
- **Source**: Adapted from the `ml-explore/mlx-examples` repository.

## Usage Examples

To train a model using this dataset, you can run:

```bash
# Use the mlx_examples dataset
ft train quick TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data data/mlx_examples/
```

The trainer will automatically discover `train.jsonl` and `valid.jsonl` within the specified directory.