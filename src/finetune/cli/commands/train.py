"""
Training commands for the CLI.
"""

from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from finetune.config import (
    ConfigManager,
    ConfigProfile,
    DataConfig,
    LoRAConfig,
    ModelConfig,
    OptimizationConfig,
    TrainingConfig,
)
from finetune.training.workflow import (
    FineTuningWorkflow,
    create_quick_workflow,
    create_training_workflow_from_config,
)

app = typer.Typer()
console = Console()


def _detect_data_format(dataset_path: Path, format_override: Optional[str] = None) -> str:
    """Detect and display data format information."""
    if format_override:
        return f"{format_override} format (override)"

    try:
        import json
        with open(dataset_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                data = json.loads(first_line)
                if "messages" in data:
                    return "chat message format (auto-detected)"
                elif "text" in data:
                    return "MLX text format (auto-detected)"
                else:
                    return "unknown format"
    except Exception:
        pass
    return "unknown format"


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
    template: str = typer.Option(
        "alpaca", "--template", "-t", help="Prompt template (alpaca, chatml, llama)"
    ),
    data_format: Optional[str] = typer.Option(None, "--format", help="Data format override (chat, mlx, auto)"),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(2, "--batch-size", "-b", help="Training batch size"),
    learning_rate: float = typer.Option(2e-4, "--lr", help="Learning rate"),
    lora_rank: int = typer.Option(8, "--lora-rank", "-r", help="LoRA rank"),
    lora_alpha: float = typer.Option(16.0, "--lora-alpha", "-a", help="LoRA alpha"),
    validation_split: float = typer.Option(0.1, "--val-split", help="Validation split ratio"),
    resume: Optional[Path] = typer.Option(None, "--resume", help="Resume from checkpoint"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate configuration without training"
    ),
):
    """Start a fine-tuning run with LoRA."""

    console.print("[bold cyan]🚀 FineTune - Apple Silicon Optimized Training[/bold cyan]")
    console.print()

    # Validate inputs
    if not dataset.exists():
        console.print(f"[red]❌ Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    # Display data format info
    format_info = _detect_data_format(dataset, data_format)
    console.print(f"[blue]📊 Dataset format: {format_info}[/blue]")

    try:
        # Create or load configuration
        if config:
            console.print(f"[blue]📋 Loading configuration from: {config}[/blue]")
            workflow = create_training_workflow_from_config(str(config))
        else:
            console.print("[blue]📋 Creating configuration...[/blue]")

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
    dataset: Optional[Path] = typer.Argument(None, help="Path to training dataset (default: WikiSQL examples)"),
    template: str = typer.Option("alpaca", "--template", "-t", help="Prompt template (alpaca, chatml, llama)"),
    data_format: Optional[str] = typer.Option(None, "--format", help="Data format override (chat, mlx, auto)"),
    output_dir: Path = typer.Option(
        Path("./quick_output"), "--output", "-o", help="Output directory"
    ),
):
    """Quick training run with minimal configuration.

    Uses WikiSQL examples by default for quick testing and validation.
    Supports both chat message format and MLX text format with auto-detection.
    """

    console.print("[bold cyan]⚡ Quick Fine-Tuning[/bold cyan]")

    # Handle default dataset
    if dataset is None:
        # Use WikiSQL as default, choosing format based on template
        script_dir = Path(__file__).parent.parent.parent.parent.parent  # Go to project root
        if template in ["chatml", "llama"]:
            default_dataset = script_dir / "data" / "datasets" / "training" / "wikisql" / "wikisql_chat_format.jsonl"
        else:
            default_dataset = script_dir / "data" / "datasets" / "training" / "wikisql" / "wikisql_mlx_format.jsonl"

        if default_dataset.exists():
            dataset = default_dataset
            console.print("[blue]📊 Using default WikiSQL dataset[/blue]")
        else:
            console.print("[yellow]⚠️  Default WikiSQL dataset not found, please specify dataset path[/yellow]")
            console.print("Available example datasets:")
            console.print("  data/datasets/training/wikisql/wikisql_chat_format.jsonl (for chatml/llama templates)")
            console.print("  data/datasets/training/wikisql/wikisql_mlx_format.jsonl (for alpaca template)")
            console.print("  data/datasets/training/chat/general_conversation.jsonl")
            console.print("  data/datasets/training/instruction/simple_instructions.jsonl")
            raise typer.Exit(1)

    if not dataset.exists():
        console.print(f"[red]❌ Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    # Display data format info
    format_info = _detect_data_format(dataset, data_format)

    try:
        # Create quick workflow
        workflow = create_quick_workflow(
            model_name=model,
            data_file=str(dataset),
            template=template,
            output_dir=str(output_dir),
        )

        console.print(f"[blue]🚀 Starting quick training: {model}[/blue]")
        console.print(f"[blue]📊 Dataset: {dataset} ({format_info})[/blue]")
        console.print(f"[blue]🎨 Template: {template}[/blue]")
        console.print()

        # Run training
        results = workflow.run_training()

        # Save model
        model_path = workflow.save_model()

        console.print("[green]✅ Quick training completed![/green]")
        console.print(f"[green]📁 Model saved to: {model_path}[/green]")

    except Exception as e:
        console.print(f"[red]❌ Quick training failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def validate(
    config: Path = typer.Argument(..., help="Path to configuration file"),
):
    """Validate a training configuration."""

    console.print("[bold cyan]🔍 Validating Configuration[/bold cyan]")

    if not config.exists():
        console.print(f"[red]❌ Configuration file not found: {config}[/red]")
        raise typer.Exit(1)

    try:
        # Load and validate configuration
        manager = ConfigManager()
        training_config = manager.load_config(config)

        console.print("[green]✅ Configuration is valid[/green]")
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


@app.command("list-data")
def list_data():
    """List available example training datasets."""

    console.print("[bold cyan]📊 Available Training Datasets[/bold cyan]")
    console.print()

    script_dir = Path(__file__).parent.parent.parent.parent.parent  # Go to project root
    data_dir = script_dir / "data"

    if not data_dir.exists():
        console.print("[yellow]⚠️  Data directory not found. Create it with example datasets.[/yellow]")
        return

    # WikiSQL datasets
    console.print("[bold]📚 WikiSQL (Database Q&A)[/bold]")
    wikisql_chat = data_dir / "datasets" / "training" / "wikisql" / "wikisql_chat_format.jsonl"
    wikisql_mlx = data_dir / "datasets" / "training" / "wikisql" / "wikisql_mlx_format.jsonl"

    if wikisql_chat.exists():
        console.print(f"  ✅ {wikisql_chat} (chat format - use with chatml/llama templates)")
    else:
        console.print(f"  ❌ {wikisql_chat} (not found)")

    if wikisql_mlx.exists():
        console.print(f"  ✅ {wikisql_mlx} (MLX format - use with alpaca template)")
    else:
        console.print(f"  ❌ {wikisql_mlx} (not found)")

    console.print()

    # Chat datasets
    console.print("[bold]💬 Conversational[/bold]")
    chat_file = data_dir / "chat" / "general_conversation.jsonl"
    if chat_file.exists():
        console.print(f"  ✅ {chat_file} (chat format)")
    else:
        console.print(f"  ❌ {chat_file} (not found)")

    console.print()

    # Instruction datasets
    console.print("[bold]📝 Instruction Following[/bold]")
    instruction_file = data_dir / "instruction" / "simple_instructions.jsonl"
    if instruction_file.exists():
        console.print(f"  ✅ {instruction_file} (MLX text format)")
    else:
        console.print(f"  ❌ {instruction_file} (not found)")

    console.print()

    # Usage examples
    console.print("[bold]🚀 Usage Examples[/bold]")
    console.print("  # Use default WikiSQL:")
    console.print("  ft train quick TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    console.print()
    console.print("  # Use specific dataset:")
    console.print("  ft train quick TinyLlama/TinyLlama-1.1B-Chat-v1.0 data/datasets/training/chat/general_conversation.jsonl")
    console.print()
    console.print("  # Use with specific template:")
    console.print("  ft train quick TinyLlama/TinyLlama-1.1B-Chat-v1.0 --template chatml")
