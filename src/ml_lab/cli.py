"""ML Lab CLI — unified interface for experiment lifecycle management.

Usage:
    ml-lab new gsm8k-grpo
    ml-lab validate 2026-04-gsm8k-grpo
    ml-lab register 2026-04-gsm8k-grpo
    ml-lab leaderboard
    ml-lab sync-wandb
"""

from __future__ import annotations

import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path("experiments")
REGISTRY_FILE = Path("registry/models.jsonl")
SCAFFOLD_DIR = Path.home() / "Documents" / "Project Portfolio" / "ml-experiment-scaffold"


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """ML Lab — ML research control plane."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@click.argument("name")
@click.option("--scaffold-dir", type=click.Path(exists=True, path_type=Path), default=None)
def new(name: str, scaffold_dir: Path | None) -> None:
    """Create a new experiment from the scaffold template."""
    from scripts.new_experiment import create_experiment

    scaffold = scaffold_dir or SCAFFOLD_DIR
    experiment_dir = create_experiment(name, scaffold, EXPERIMENTS_DIR)
    click.echo(f"Experiment created: {experiment_dir}")


@cli.command()
@click.argument("experiment")
def validate(experiment: str) -> None:
    """Validate experiment configs for known issues."""
    from src.ml_lab.config_validator import validate_experiment

    experiment_dir = EXPERIMENTS_DIR / experiment
    if not experiment_dir.exists():
        click.echo(f"Experiment not found: {experiment_dir}", err=True)
        raise SystemExit(1)

    valid = validate_experiment(experiment_dir)
    if not valid:
        raise SystemExit(1)


@cli.command()
@click.argument("experiment")
@click.option("--seed", type=int, default=42)
@click.option("--tags", multiple=True)
def register(experiment: str, seed: int, tags: tuple[str, ...]) -> None:
    """Register a trained model in the registry."""
    from scripts.register_model import register_model

    experiment_dir = EXPERIMENTS_DIR / experiment
    if not experiment_dir.exists():
        click.echo(f"Experiment not found: {experiment_dir}", err=True)
        raise SystemExit(1)

    entry = register_model(experiment_dir, REGISTRY_FILE, seed, list(tags) or None)
    click.echo(f"Registered: {entry['id']}")


@cli.command("leaderboard")
@click.option("--method", type=str, default=None)
@click.option("--sort", type=str, default=None)
def leaderboard_cmd(method: str | None, sort: str | None) -> None:
    """Show cross-experiment model leaderboard."""
    from scripts.cross_compare import filter_entries, generate_leaderboard, load_registry

    entries = load_registry(REGISTRY_FILE)
    entries = filter_entries(entries, method=method)
    click.echo(generate_leaderboard(entries, sort_metric=sort))


@cli.command("sync-wandb")
def sync_wandb_cmd() -> None:
    """Sync offline W&B runs via WSL."""
    from scripts.sync_wandb import sync_all

    synced, failed = sync_all(EXPERIMENTS_DIR)
    if synced or failed:
        click.echo(f"Synced: {synced}, Failed: {failed}")
    else:
        click.echo("No offline runs to sync.")
    if failed:
        raise SystemExit(1)


@cli.command()
def preflight() -> None:
    """Run GPU health check before training."""
    from scripts.preflight import run_preflight

    gpu_diag_dir = Path.home() / "Documents" / "Project Portfolio" / "gpu-server-test-suite"
    healthy = run_preflight(gpu_diag_dir)
    if not healthy:
        raise SystemExit(1)
