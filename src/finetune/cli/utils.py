"""
Utility functions and error handling for CLI.
"""

import sys
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

console = Console()


class CLIError(Exception):
    """Base exception for CLI errors."""

    pass


def handle_errors(func: Callable) -> Callable:
    """Decorator to handle errors in CLI commands."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except CLIError as e:
            console.print(f"[red]❌ {e}[/red]")
            raise typer.Exit(1)
        except FileNotFoundError as e:
            console.print(f"[red]❌ File not found: {e}[/red]")
            raise typer.Exit(1)
        except PermissionError as e:
            console.print(f"[red]❌ Permission denied: {e}[/red]")
            raise typer.Exit(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️  Operation cancelled by user[/yellow]")
            raise typer.Exit(130)
        except Exception as e:
            console.print(f"[red]❌ Unexpected error: {e}[/red]")
            if "--debug" in sys.argv:
                console.print_exception()
            else:
                console.print("[dim]Run with --debug for full traceback[/dim]")
            raise typer.Exit(1)

    return wrapper


def validate_path(
    path: Path, must_exist: bool = True, must_be_file: bool = False, must_be_dir: bool = False
) -> Path:
    """Validate a path with various constraints."""
    if must_exist and not path.exists():
        raise CLIError(f"Path does not exist: {path}")

    if must_be_file and not path.is_file():
        raise CLIError(f"Path is not a file: {path}")

    if must_be_dir and not path.is_dir():
        raise CLIError(f"Path is not a directory: {path}")

    return path


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask user for confirmation."""
    try:
        return typer.confirm(message, default=default)
    except KeyboardInterrupt:
        return False


def format_size(size_bytes: int) -> str:
    """Format byte size as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def format_number(num: int) -> str:
    """Format large numbers with commas."""
    return f"{num:,}"


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path.cwd()

    # Look for markers of project root
    markers = ["train.yml", ".finetune", "pyproject.toml", ".git"]

    while current != current.parent:
        for marker in markers:
            if (current / marker).exists():
                return current
        current = current.parent

    # Default to current directory
    return Path.cwd()


def ensure_project_initialized() -> Path:
    """Ensure the project is initialized."""
    root = get_project_root()

    # Check for basic project structure
    required_dirs = ["data", "models", "checkpoints"]
    missing = [d for d in required_dirs if not (root / d).exists()]

    if missing:
        console.print("[yellow]⚠️  Project not fully initialized[/yellow]")
        console.print(f"Missing directories: {', '.join(missing)}")

        if confirm_action("Initialize project now?", default=True):
            for dir_name in missing:
                (root / dir_name).mkdir(parents=True, exist_ok=True)
                console.print(f"[green]✓ Created {dir_name}/[/green]")
        else:
            raise CLIError("Project initialization required. Run 'ft init' first.")

    return root


def load_config(config_path: Path | None = None) -> dict:
    """Load configuration from YAML file."""
    import yaml

    if config_path is None:
        config_path = get_project_root() / "train.yml"

    if not config_path.exists():
        return {}

    try:
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise CLIError(f"Invalid YAML configuration: {e}")


def save_config(config: dict, config_path: Path | None = None) -> None:
    """Save configuration to YAML file."""
    import yaml

    if config_path is None:
        config_path = get_project_root() / "train.yml"

    try:
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise CLIError(f"Failed to save configuration: {e}")
