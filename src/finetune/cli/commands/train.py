"""
Training commands for the CLI.
"""

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def start(
    model: str = typer.Argument(..., help="Model name or path"),
    dataset: Path = typer.Argument(..., help="Path to training dataset"),
    output_dir: Path = typer.Option(
        Path("./checkpoints"), "--output", "-o", help="Output directory for checkpoints"
    ),
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to training configuration file"
    ),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Training batch size"),
    learning_rate: float = typer.Option(2e-4, "--lr", help="Learning rate"),
    method: str = typer.Option("lora", "--method", "-m", help="Training method (lora, qlora, full)"),
    resume: Path | None = typer.Option(None, "--resume", "-r", help="Resume from checkpoint"),
):
    """Start a training run."""
    from finetune.core.config import TrainingConfig
    from finetune.models.manager import model_manager
    
    console.print(f"[bold cyan]🚀 Starting Fine-Tuning[/bold cyan]")
    console.print(f"Model: {model}")
    console.print(f"Dataset: {dataset}")
    console.print(f"Method: {method}")
    console.print(f"Epochs: {epochs}")
    console.print(f"Batch Size: {batch_size}")
    console.print(f"Learning Rate: {learning_rate}")
    
    # Validate inputs
    if not dataset.exists():
        console.print(f"[red]❌ Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration if provided
    if config and config.exists():
        console.print(f"Loading config from: {config}")
        # Would load YAML config here
    
    if resume and resume.exists():
        console.print(f"[yellow]↻ Resuming from checkpoint: {resume}[/yellow]")
    
    # Create training configuration
    training_config = {
        "model": model,
        "dataset": str(dataset),
        "output_dir": str(output_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "method": method,
    }
    
    # Phase 2: Actual training implementation would go here
    console.print("\n[yellow]⚠️  Training loop implementation is part of Phase 2[/yellow]")
    console.print("[dim]Training configuration prepared and validated.[/dim]")


@app.command()
def stop(
    job_id: str | None = typer.Argument(None, help="Job ID to stop (or latest if not specified)")
):
    """Stop a running training job."""
    import signal
    from pathlib import Path
    
    console.print("[bold yellow]⏹ Stopping training...[/bold yellow]")
    
    # Check for running jobs
    pid_file = Path(".finetune.pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text())
            import os
            os.kill(pid, signal.SIGTERM)
            pid_file.unlink()
            console.print(f"[green]✓ Stopped training job (PID: {pid})[/green]")
        except (ValueError, ProcessLookupError) as e:
            console.print(f"[red]❌ No active training job found[/red]")
            pid_file.unlink()
    else:
        console.print("[yellow]No active training jobs[/yellow]")


@app.command()
def status(
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch status in real-time")
):
    """Check status of training jobs."""
    from datetime import datetime
    from pathlib import Path
    from rich.table import Table
    
    console.print("[bold cyan]📊 Training Status[/bold cyan]\n")
    
    # Check for checkpoints
    checkpoint_dir = Path("checkpoints")
    if not checkpoint_dir.exists():
        console.print("[yellow]No checkpoints found. No training jobs have been run.[/yellow]")
        return
    
    # Create status table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Job ID", style="dim")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Epoch")
    table.add_column("Loss")
    table.add_column("Time")
    
    # Check for active job
    pid_file = Path(".finetune.pid")
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text())
            import psutil
            if psutil.pid_exists(pid):
                table.add_row(
                    str(pid),
                    "current",
                    "[green]● Running[/green]",
                    "-",
                    "-",
                    "In Progress"
                )
        except:
            pass
    
    # List recent checkpoints
    checkpoints = sorted(checkpoint_dir.glob("*/checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
    for ckpt in checkpoints:
        model_name = ckpt.parent.name
        epoch = ckpt.name.split("-")[-1] if "-" in ckpt.name else "?"
        mtime = datetime.fromtimestamp(ckpt.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        
        table.add_row(
            ckpt.name[:8],
            model_name,
            "[dim]Completed[/dim]",
            epoch,
            "-",
            mtime
        )
    
    if table.row_count > 0:
        console.print(table)
    else:
        console.print("[yellow]No training history found.[/yellow]")
    
    if watch:
        console.print("\n[dim]Watching for updates... (Press Ctrl+C to stop)[/dim]")
        # Would implement real-time monitoring here
