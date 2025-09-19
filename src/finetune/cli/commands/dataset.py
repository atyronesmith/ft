"""
Dataset management commands for the CLI.
"""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command()
def prepare(
    input_path: Path = typer.Argument(..., help="Path to input dataset"),
    output_path: Path | None = typer.Option(None, "--output", "-o", help="Output path"),
    template: str = typer.Option(
        "alpaca", "--template", "-t", help="Template format (alpaca, chatml, llama)"
    ),
    format: str = typer.Option(
        "auto", "--format", "-f", help="Input format (auto, json, jsonl, csv, parquet)"
    ),
    max_length: int = typer.Option(2048, "--max-length", help="Maximum sequence length"),
):
    """Prepare a dataset for training."""
    console.print("[bold cyan]📊 Preparing Dataset[/bold cyan]")
    console.print(f"Input: {input_path}")
    console.print(f"Template: {template}")

    if not input_path.exists():
        console.print(f"[red]❌ Dataset not found: {input_path}[/red]")
        raise typer.Exit(1)

    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_prepared{input_path.suffix}"

    # Detect format
    if format == "auto":
        if input_path.suffix == ".json":
            format = "json"
        elif input_path.suffix == ".jsonl":
            format = "jsonl"
        elif input_path.suffix == ".csv":
            format = "csv"
        elif input_path.suffix == ".parquet":
            format = "parquet"
        else:
            console.print("[yellow]⚠️  Unknown format, assuming JSON[/yellow]")
            format = "json"

    console.print(f"Format: {format}")
    console.print(f"Max Length: {max_length}")

    # Apply template (simplified implementation)
    templates = {
        "alpaca": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}",
        "chatml": "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>",
        "llama": "[INST] {instruction} [/INST] {output}",
    }

    if template not in templates:
        console.print(f"[red]❌ Unknown template: {template}[/red]")
        console.print(f"Available templates: {', '.join(templates.keys())}")
        raise typer.Exit(1)

    # Basic implementation for JSON format
    if format == "json":
        try:
            with open(input_path) as f:
                data = json.load(f)

            if isinstance(data, list):
                console.print(f"Processing {len(data)} examples...")
                prepared_data = []

                for item in track(data, description="Preparing..."):
                    if "instruction" in item and "output" in item:
                        text = templates[template].format(
                            instruction=item["instruction"], output=item["output"]
                        )
                        prepared_data.append({"text": text[:max_length]})

                # Save prepared data
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(prepared_data, f, indent=2)

                console.print(f"[green]✓ Prepared {len(prepared_data)} examples[/green]")
                console.print(f"[green]✓ Saved to: {output_path}[/green]")
            else:
                console.print("[red]❌ Expected JSON array of examples[/red]")
                raise typer.Exit(1)

        except json.JSONDecodeError as e:
            console.print(f"[red]❌ Invalid JSON: {e}[/red]")
            raise typer.Exit(1)
    else:
        console.print(f"[yellow]⚠️  Format '{format}' preparation coming in Phase 2[/yellow]")


@app.command()
def validate(
    dataset_path: Path = typer.Argument(..., help="Path to dataset"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed validation"),
):
    """Validate dataset format and quality."""
    console.print("[bold cyan]🔍 Validating Dataset[/bold cyan]")
    console.print(f"Path: {dataset_path}\n")

    if not dataset_path.exists():
        console.print(f"[red]❌ Dataset not found: {dataset_path}[/red]")
        raise typer.Exit(1)

    issues = []
    warnings = []

    # Load and validate
    try:
        with open(dataset_path) as f:
            data = json.load(f)

        if not isinstance(data, list):
            issues.append("Dataset must be a JSON array")
        else:
            console.print(f"Found {len(data)} examples")

            # Check each example
            for i, item in enumerate(data[:100]):  # Check first 100
                if not isinstance(item, dict):
                    issues.append(f"Example {i}: Not a dictionary")
                    continue

                # Check for required fields
                if "text" not in item and ("instruction" not in item or "output" not in item):
                    issues.append(
                        f"Example {i}: Missing required fields (text or instruction/output)"
                    )

                # Check lengths
                if "text" in item:
                    text_len = len(item["text"])
                    if text_len == 0:
                        issues.append(f"Example {i}: Empty text")
                    elif text_len > 8192:
                        warnings.append(f"Example {i}: Very long text ({text_len} chars)")

            # Dataset statistics
            if len(data) < 10:
                warnings.append("Very small dataset (< 10 examples)")
            elif len(data) < 100:
                warnings.append("Small dataset (< 100 examples)")

    except json.JSONDecodeError as e:
        issues.append(f"Invalid JSON: {e}")
    except Exception as e:
        issues.append(f"Error reading dataset: {e}")

    # Report results
    if issues:
        console.print("[red]❌ Validation Failed[/red]\n")
        for issue in issues:
            console.print(f"  • {issue}")
    else:
        console.print("[green]✓ Validation Passed[/green]")

    if warnings:
        console.print("\n[yellow]⚠️  Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  • {warning}")

    if verbose and not issues:
        console.print("\n[dim]Dataset appears ready for training.[/dim]")


@app.command()
def split(
    input_path: Path = typer.Argument(..., help="Input dataset path"),
    output_dir: Path = typer.Option(Path("./data"), "--output", "-o", help="Output directory"),
    train_ratio: float = typer.Option(0.8, "--train", help="Training set ratio"),
    val_ratio: float = typer.Option(0.1, "--val", help="Validation set ratio"),
    test_ratio: float = typer.Option(0.1, "--test", help="Test set ratio"),
    seed: int = typer.Option(42, "--seed", help="Random seed for splitting"),
):
    """Split dataset into train/val/test sets."""
    import random

    console.print("[bold cyan]✂️  Splitting Dataset[/bold cyan]")
    console.print(f"Input: {input_path}")
    console.print(
        f"Ratios - Train: {train_ratio:.0%}, Val: {val_ratio:.0%}, Test: {test_ratio:.0%}"
    )

    if not input_path.exists():
        console.print(f"[red]❌ Dataset not found: {input_path}[/red]")
        raise typer.Exit(1)

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        console.print(f"[red]❌ Ratios must sum to 1.0 (got {total_ratio})[/red]")
        raise typer.Exit(1)

    # Load dataset
    try:
        with open(input_path) as f:
            data = json.load(f)

        if not isinstance(data, list):
            console.print("[red]❌ Dataset must be a JSON array[/red]")
            raise typer.Exit(1)

        console.print(f"Total examples: {len(data)}")

        # Shuffle data
        random.seed(seed)
        random.shuffle(data)

        # Calculate split points
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        # Split data
        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        # Save splits
        output_dir.mkdir(parents=True, exist_ok=True)

        splits = {
            "train": train_data,
            "validation": val_data,
            "test": test_data,
        }

        for split_name, split_data in splits.items():
            if len(split_data) > 0:
                output_file = output_dir / f"{split_name}.json"
                with open(output_file, "w") as f:
                    json.dump(split_data, f, indent=2)
                console.print(
                    f"[green]✓ {split_name}: {len(split_data)} examples → {output_file}[/green]"
                )

    except json.JSONDecodeError as e:
        console.print(f"[red]❌ Invalid JSON: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def stats(
    dataset_path: Path = typer.Argument(..., help="Path to dataset"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed statistics"),
):
    """Show dataset statistics."""
    console.print("[bold cyan]📈 Dataset Statistics[/bold cyan]")
    console.print(f"Path: {dataset_path}\n")

    if not dataset_path.exists():
        console.print(f"[red]❌ Dataset not found: {dataset_path}[/red]")
        raise typer.Exit(1)

    try:
        # Get file size
        file_size = dataset_path.stat().st_size / (1024 * 1024)  # MB

        with open(dataset_path) as f:
            data = json.load(f)

        if not isinstance(data, list):
            console.print("[red]❌ Dataset must be a JSON array[/red]")
            raise typer.Exit(1)

        # Calculate statistics
        stats = {
            "Total Examples": len(data),
            "File Size": f"{file_size:.2f} MB",
        }

        if len(data) > 0:
            # Analyze text lengths
            text_lengths = []
            field_counts = {}

            for item in data:
                if isinstance(item, dict):
                    # Count fields
                    for field in item.keys():
                        field_counts[field] = field_counts.get(field, 0) + 1

                    # Get text length
                    if "text" in item:
                        text_lengths.append(len(item["text"]))
                    elif "instruction" in item and "output" in item:
                        combined = str(item.get("instruction", "")) + str(item.get("output", ""))
                        text_lengths.append(len(combined))

            if text_lengths:
                stats["Avg Text Length"] = f"{sum(text_lengths) / len(text_lengths):.0f} chars"
                stats["Min Text Length"] = f"{min(text_lengths)} chars"
                stats["Max Text Length"] = f"{max(text_lengths)} chars"

            # Create table
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Metric", style="dim")
            table.add_column("Value")

            for key, value in stats.items():
                table.add_row(key, str(value))

            console.print(table)

            if detailed and field_counts:
                console.print("\n[bold]Field Distribution:[/bold]")
                field_table = Table(show_header=True, header_style="bold cyan")
                field_table.add_column("Field", style="dim")
                field_table.add_column("Count")
                field_table.add_column("Coverage")

                for field, count in sorted(field_counts.items(), key=lambda x: x[1], reverse=True):
                    coverage = f"{count / len(data) * 100:.1f}%"
                    field_table.add_row(field, str(count), coverage)

                console.print(field_table)
        else:
            console.print("[yellow]Dataset is empty[/yellow]")

    except json.JSONDecodeError as e:
        console.print(f"[red]❌ Invalid JSON: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error analyzing dataset: {e}[/red]")
        raise typer.Exit(1)


@app.command("list")
def list_datasets(
    directory: Path = typer.Option(Path("./data"), "--dir", "-d", help="Directory to search"),
    format: str | None = typer.Option(None, "--format", "-f", help="Filter by format"),
):
    """List available datasets."""
    console.print("[bold cyan]📚 Available Datasets[/bold cyan]")
    console.print(f"Directory: {directory}\n")

    if not directory.exists():
        console.print(f"[yellow]Directory not found: {directory}[/yellow]")
        return

    # Find dataset files
    patterns = ["*.json", "*.jsonl", "*.csv", "*.parquet"] if not format else [f"*.{format}"]
    datasets = []

    for pattern in patterns:
        datasets.extend(directory.glob(pattern))
        datasets.extend(directory.glob(f"**/{pattern}"))  # Recursive

    if not datasets:
        console.print("[yellow]No datasets found[/yellow]")
        return

    # Create table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Name", style="dim")
    table.add_column("Format")
    table.add_column("Size")
    table.add_column("Modified")

    from datetime import datetime

    for dataset_path in sorted(datasets):
        size = dataset_path.stat().st_size
        if size < 1024:
            size_str = f"{size} B"
        elif size < 1024 * 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size / (1024 * 1024):.1f} MB"

        modified = datetime.fromtimestamp(dataset_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        relative_path = (
            dataset_path.relative_to(directory)
            if dataset_path.is_relative_to(directory)
            else dataset_path.name
        )

        table.add_row(str(relative_path), dataset_path.suffix[1:], size_str, modified)

    console.print(table)
    console.print(f"\n[dim]Total: {len(datasets)} datasets[/dim]")
