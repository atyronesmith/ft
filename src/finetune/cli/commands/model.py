"""
Model management commands for the CLI.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command("list")
def list_models(
    source: str | None = typer.Option(None, help="Filter by source (huggingface, local, cache)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
):
    """List available models."""
    from finetune.models.manager import model_manager

    console.print("[bold]Available Models[/bold]\n")

    # Get models from manager
    models = model_manager.list_models(source=source)

    if not models:
        console.print("[yellow]No models found[/yellow]")
        return

    # Create table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Name", style="dim")
    table.add_column("Source")
    table.add_column("Cached")
    table.add_column("Size (GB)")

    if verbose:
        table.add_column("Parameters")
        table.add_column("Path")

    for model in models:
        row = [
            model["name"],
            model.get("source", "unknown"),
            "✓" if model.get("cached") else "✗",
            f"{model.get('size_gb', 0):.1f}" if model.get("size_gb") else "-",
        ]

        if verbose:
            params = model.get("parameters", {})
            if isinstance(params, dict):
                param_count = params.get("parameters", "-")
                if param_count != "-":
                    row.append(f"{param_count:,}")
                else:
                    row.append("-")
            else:
                row.append("-")
            row.append(model.get("path", "-"))

        table.add_row(*row)

    console.print(table)
    console.print(f"\n[dim]Total models: {len(models)}[/dim]")


@app.command()
def pull(
    model_name: str = typer.Argument(..., help="Model name from HuggingFace"),
    quantization: str | None = typer.Option(
        None, "--quantization", "-q", help="Quantization (4bit, 8bit)"
    ),
    revision: str | None = typer.Option(None, "--revision", "-r", help="Model revision/branch"),
):
    """Download a model from HuggingFace."""
    from finetune.models.manager import model_manager

    console.print(f"[bold]Downloading model: {model_name}[/bold]")

    # Parse quantization
    load_in_4bit = quantization == "4bit"
    load_in_8bit = quantization == "8bit"

    if quantization:
        console.print(f"Using {quantization} quantization")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading model...", total=None)

            model = model_manager.load_model(
                model_name,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                revision=revision,
            )

            progress.update(task, completed=True)

        # Show model info
        info = model_manager.loader.get_model_info(model)
        console.print("\n[green]✓[/green] Model downloaded successfully!")
        console.print(f"  Type: {info['type']}")
        console.print(f"  Parameters: {info['parameters']:,}")
        console.print(f"  Memory: {info['memory_footprint_mb']:.1f} MB")

    except Exception as e:
        console.print(f"[red]Error downloading model: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def info(
    model_name: str = typer.Argument(..., help="Model name or path"),
    estimate_memory: bool = typer.Option(False, "--memory", "-m", help="Estimate memory usage"),
):
    """Show information about a model."""
    from finetune.models.manager import model_manager

    console.print(f"[bold]Model Information: {model_name}[/bold]\n")

    # Get model info
    info = model_manager.get_model_info(model_name)

    if not info:
        console.print(f"[red]Model not found: {model_name}[/red]")
        raise typer.Exit(1)

    # Display basic info
    console.print("[cyan]Basic Information:[/cyan]")
    console.print(f"  Name: {info['name']}")
    console.print(f"  Type: {info.get('model_type', 'unknown')}")
    console.print(f"  Path: {info['path']}")
    console.print(f"  Size: {info['size_gb']:.2f} GB")

    # Display architecture
    console.print("\n[cyan]Architecture:[/cyan]")
    console.print(f"  Vocab Size: {info.get('vocab_size', 'N/A')}")
    console.print(f"  Hidden Size: {info.get('hidden_size', 'N/A')}")
    console.print(f"  Layers: {info.get('num_layers', 'N/A')}")
    console.print(f"  Attention Heads: {info.get('num_heads', 'N/A')}")

    # Display files
    console.print("\n[cyan]Files:[/cyan]")
    for file in info.get("files", [])[:5]:  # Show first 5 files
        console.print(f"  - {file}")
    if len(info.get("files", [])) > 5:
        console.print(f"  ... and {len(info['files']) - 5} more files")

    # Estimate memory if requested
    if estimate_memory:
        console.print("\n[cyan]Memory Estimation (Training):[/cyan]")
        mem = model_manager.estimate_memory_usage(model_name, batch_size=4, sequence_length=2048)
        console.print(f"  Model Weights: {mem['model_gb']:.1f} GB")
        console.print(f"  Activations: {mem['activations_gb']:.1f} GB")
        console.print(f"  Gradients: {mem['gradients_gb']:.1f} GB")
        console.print(f"  Optimizer: {mem['optimizer_gb']:.1f} GB")
        console.print(f"  [bold]Total: {mem['total_gb']:.1f} GB[/bold]")


@app.command()
def convert(
    input_path: Path = typer.Argument(..., help="Input model path"),
    output_format: str = typer.Option("gguf", help="Output format (gguf, onnx, coreml)"),
):
    """Convert model to different format."""
    console.print(f"[bold]Converting {input_path} to {output_format}[/bold]")

    # TODO: Implement conversion
    console.print("[yellow]Conversion implementation coming soon![/yellow]")
