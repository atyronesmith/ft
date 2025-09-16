"""
Training commands for the CLI.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger

from finetune.config import TrainingConfig, ModelConfig, DataConfig, LoRAConfig, OptimizationConfig, ConfigManager, ConfigProfile
from finetune.training.workflow import FineTuningWorkflow, create_training_workflow_from_config, create_quick_workflow

app = typer.Typer()
console = Console()


@app.command()
def start(
    model: str = typer.Argument(..., help="Model name or path"),
    dataset: Path = typer.Argument(..., help="Path to training dataset"),
    output_dir: Path = typer.Option(
        Path("./output"), "--output", "-o", help="Output directory for checkpoints"
    ),
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to training configuration file"
    ),
    profile: Optional[str] = typer.Option(
        None, "--profile", "-p", help="Configuration profile (chat, instruction, code)"
    ),
    template: str = typer.Option("alpaca", "--template", "-t", help="Prompt template (alpaca, chatml, llama)"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(2, "--batch-size", "-b", help="Training batch size"),
    learning_rate: float = typer.Option(2e-4, "--lr", help="Learning rate"),
    lora_rank: int = typer.Option(8, "--lora-rank", "-r", help="LoRA rank"),
    lora_alpha: float = typer.Option(16.0, "--lora-alpha", "-a", help="LoRA alpha"),
    validation_split: float = typer.Option(0.1, "--val-split", help="Validation split ratio"),
    resume: Optional[Path] = typer.Option(None, "--resume", help="Resume from checkpoint"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate configuration without training"),
):
    """Start a fine-tuning run with LoRA."""

    console.print(f"[bold cyan]🚀 FineTune - Apple Silicon Optimized Training[/bold cyan]")
    console.print()

    # Validate inputs
    if not dataset.exists():
        console.print(f"[red]❌ Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    try:
        # Create or load configuration
        if config:
            console.print(f"[blue]📋 Loading configuration from: {config}[/blue]")
            workflow = create_training_workflow_from_config(str(config))
        else:
            console.print(f"[blue]📋 Creating configuration...[/blue]")

            # Create configuration from command line arguments
            training_config = TrainingConfig(
                model=ModelConfig(name=model),
                data=DataConfig(
                    train_file=str(dataset),
                    template=template,
                    validation_split=validation_split,
                ),
                lora=LoRAConfig(r=lora_rank, alpha=lora_alpha),
                optimization=OptimizationConfig(
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    epochs=epochs,
                ),
                output_dir=str(output_dir),
            )

            # Apply profile if specified
            if profile:
                console.print(f"[blue]🎯 Applying {profile} profile...[/blue]")
                training_config = ConfigProfile.apply_profile(training_config, profile)

            workflow = FineTuningWorkflow(training_config)

        # Display configuration
        _display_config(workflow.config)

        if dry_run:
            console.print("[yellow]✅ Dry run complete - configuration is valid[/yellow]")
            return

        # Run training
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            task = progress.add_task("Preparing for training...", total=None)

            # Execute training workflow
            results = workflow.run_training()

            progress.update(task, description="Training completed!")

        # Display results
        _display_results(results)

        # Save model
        model_path = workflow.save_model()
        console.print(f"[green]✅ Model saved to: {model_path}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Training interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"[red]❌ Training failed: {e}[/red]")
        logger.error(f"Training error: {e}")
        raise typer.Exit(1)


def _display_config(config: TrainingConfig) -> None:
    """Display training configuration."""
    console.print("[bold]📊 Training Configuration:[/bold]")
    console.print(f"  Model: {config.model.name}")
    console.print(f"  Data: {config.data.train_file}")
    console.print(f"  Template: {config.data.template}")
    console.print(f"  LoRA Rank: {config.lora.r} (α={config.lora.alpha})")
    console.print(f"  Learning Rate: {config.optimization.learning_rate}")
    console.print(f"  Batch Size: {config.optimization.batch_size}")
    console.print(f"  Epochs: {config.optimization.epochs}")
    console.print(f"  Output: {config.output_dir}")
    console.print()


def _display_results(results: dict) -> None:
    """Display training results."""
    console.print("[bold]📈 Training Results:[/bold]")
    if results.get("training_loss"):
        final_loss = results["training_loss"][-1]
        console.print(f"  Final Training Loss: {final_loss:.4f}")
    if results.get("eval_loss"):
        final_eval_loss = results["eval_loss"][-1]
        console.print(f"  Final Validation Loss: {final_eval_loss:.4f}")
    console.print(f"  Total Steps: {results.get('global_step', 0)}")
    console.print(f"  Epochs Completed: {results.get('epoch', 0)}")
    console.print()


@app.command()
def quick(
    model: str = typer.Argument(..., help="Model name"),
    dataset: Path = typer.Argument(..., help="Path to training dataset"),
    template: str = typer.Option("alpaca", "--template", "-t", help="Prompt template"),
    output_dir: Path = typer.Option(Path("./quick_output"), "--output", "-o", help="Output directory"),
):
    """Quick training run with minimal configuration."""

    console.print(f"[bold cyan]⚡ Quick Fine-Tuning[/bold cyan]")

    if not dataset.exists():
        console.print(f"[red]❌ Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    try:
        # Create quick workflow
        workflow = create_quick_workflow(
            model_name=model,
            data_file=str(dataset),
            template=template,
            output_dir=str(output_dir),
        )

        console.print(f"[blue]🚀 Starting quick training: {model}[/blue]")
        console.print(f"[blue]📊 Dataset: {dataset}[/blue]")
        console.print(f"[blue]🎨 Template: {template}[/blue]")
        console.print()

        # Run training
        results = workflow.run_training()

        # Save model
        model_path = workflow.save_model()

        console.print(f"[green]✅ Quick training completed![/green]")
        console.print(f"[green]📁 Model saved to: {model_path}[/green]")

    except Exception as e:
        console.print(f"[red]❌ Quick training failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def validate(
    config: Path = typer.Argument(..., help="Path to configuration file"),
):
    """Validate a training configuration."""

    console.print(f"[bold cyan]🔍 Validating Configuration[/bold cyan]")

    if not config.exists():
        console.print(f"[red]❌ Configuration file not found: {config}[/red]")
        raise typer.Exit(1)

    try:
        # Load and validate configuration
        manager = ConfigManager()
        training_config = manager.load_config(config)

        console.print(f"[green]✅ Configuration is valid[/green]")
        _display_config(training_config)

        # Run additional validation
        from finetune.config import ConfigValidator
        validator = ConfigValidator()
        warnings = validator.validate(training_config)

        if warnings:
            console.print("[yellow]⚠️  Configuration Warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  • {warning}")
        else:
            console.print("[green]✅ No configuration warnings[/green]")

    except Exception as e:
        console.print(f"[red]❌ Configuration validation failed: {e}[/red]")
        raise typer.Exit(1)
