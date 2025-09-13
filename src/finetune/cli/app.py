"""
Main CLI application for FineTune.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from finetune.backends.device import device_manager
from finetune.cli.commands import dataset, model, train

app = typer.Typer(
    name="finetune", help="Fine-tune language models on Apple Silicon", rich_markup_mode="rich"
)

console = Console()

# Add subcommands
app.add_typer(train.app, name="train", help="Training commands")
app.add_typer(model.app, name="models", help="Model management")
app.add_typer(dataset.app, name="dataset", help="Dataset operations")


@app.command()
def init():
    """Initialize a new fine-tuning project."""
    console.print("[bold green]Initializing FineTune project...[/bold green]")

    # Create necessary directories
    dirs = ["data", "models", "checkpoints", "logs", "configs"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)

    # Create default config if it doesn't exist
    if not Path("train.yml").exists():
        console.print("Creating default train.yml configuration...")
        # Copy from template or create basic config

    console.print("[bold green]✓[/bold green] Project initialized successfully!")


@app.command()
def info():
    """Show system information and available backends."""
    console.print("[bold]System Information[/bold]")

    # Get device info
    device_info = device_manager.device_info.to_dict()
    memory_info = device_manager.get_memory_info()

    # Create table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Property", style="dim")
    table.add_column("Value")

    table.add_row("Device Type", device_info["device_type"])
    if device_info["chip_name"]:
        table.add_row("Chip", device_info["chip_name"])
    table.add_row("Total Memory", f"{device_info['total_memory_gb']:.1f} GB")
    table.add_row("Available Memory", f"{memory_info['available_gb']:.1f} GB")
    table.add_row("CPU Cores", str(device_info["cpu_count"]))

    # Check backend availability
    try:
        backend = device_manager.get_optimal_backend()
        backend_info = backend.get_device_info()
        table.add_row("Backend", backend_info["backend"])
        if "mlx_version" in backend_info:
            table.add_row("MLX Version", backend_info.get("mlx_version", "N/A"))
        if "torch_version" in backend_info:
            table.add_row("PyTorch Version", backend_info.get("torch_version", "N/A"))
    except Exception as e:
        table.add_row("Backend", f"[red]Error: {e}[/red]")

    console.print(table)


@app.command()
def serve(
    port: int = typer.Option(8000, help="Port to run the server on"),
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
):
    """Start the inference API server."""
    console.print(f"[bold]Starting inference server on {host}:{port}...[/bold]")

    # Import here to avoid circular imports
    try:
        import uvicorn

        from finetune.api.app import app as api_app

        uvicorn.run(api_app, host=host, port=port, reload=True)
    except ImportError:
        console.print("[red]FastAPI not installed. Install with: pip install fastapi uvicorn[/red]")
    except Exception as e:
        console.print(f"[red]Error starting server: {e}[/red]")


@app.command()
def ui():
    """Launch the web UI dashboard."""
    console.print("[bold]Starting web UI...[/bold]")

    try:
        import sys
        from pathlib import Path

        import streamlit.cli as stcli

        ui_path = Path(__file__).parent.parent / "ui" / "app.py"
        sys.argv = ["streamlit", "run", str(ui_path)]
        stcli.main()
    except ImportError:
        console.print("[red]Streamlit not installed. Install with: pip install streamlit[/red]")
    except Exception as e:
        console.print(f"[red]Error starting UI: {e}[/red]")


@app.callback()
def callback():
    """
    FineTune - Fine-tuning for Apple Silicon

    Use 'ft --help' to see available commands.
    """
    pass


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
