"""
Dataset management commands for the CLI.
"""

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def prepare(
    input_path: Path = typer.Argument(..., help="Path to input dataset"),
    output_path: Path | None = typer.Option(None, help="Output path"),
    template: str = typer.Option("alpaca", help="Template format"),
):
    """Prepare a dataset for training."""
    console.print(f"[bold]Preparing dataset: {input_path}[/bold]")
    console.print(f"Using template: {template}")

    if not input_path.exists():
        console.print(f"[red]Dataset not found: {input_path}[/red]")
        raise typer.Exit(1)

    # TODO: Implement dataset preparation
    console.print("[yellow]Preparation implementation coming soon![/yellow]")


@app.command()
def validate(
    dataset_path: Path = typer.Argument(..., help="Path to dataset"),
):
    """Validate dataset format and quality."""
    console.print(f"[bold]Validating dataset: {dataset_path}[/bold]")

    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        raise typer.Exit(1)

    # TODO: Implement validation
    console.print("[yellow]Validation implementation coming soon![/yellow]")


@app.command()
def split(
    input_path: Path = typer.Argument(..., help="Input dataset path"),
    train_ratio: float = typer.Option(0.8, help="Training set ratio"),
    val_ratio: float = typer.Option(0.1, help="Validation set ratio"),
    test_ratio: float = typer.Option(0.1, help="Test set ratio"),
):
    """Split dataset into train/val/test sets."""
    console.print(f"[bold]Splitting dataset: {input_path}[/bold]")
    console.print(f"Ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")

    # TODO: Implement splitting
    console.print("[yellow]Split implementation coming soon![/yellow]")


@app.command()
def stats(
    dataset_path: Path = typer.Argument(..., help="Path to dataset"),
):
    """Show dataset statistics."""
    console.print(f"[bold]Dataset Statistics: {dataset_path}[/bold]")

    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        raise typer.Exit(1)

    # TODO: Show statistics
    console.print("[yellow]Statistics implementation coming soon![/yellow]")
