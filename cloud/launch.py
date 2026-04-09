"""Cloud training launcher — rsync + SSH orchestrator.

Syncs experiment directory to a remote GPU instance, runs training,
and syncs results back. Same configs work locally and in the cloud.

Usage:
    python cloud/launch.py --experiment experiments/2026-04-gsm8k-grpo --provider runpod
    python cloud/launch.py --experiment experiments/2026-04-gsm8k-grpo --provider runpod --host user@ip
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROVIDERS_FILE = Path(__file__).parent / "providers.yaml"


def load_provider_config(provider: str) -> dict:
    """Load provider configuration from providers.yaml."""
    with open(PROVIDERS_FILE) as f:
        providers = yaml.safe_load(f)

    if provider not in providers:
        available = ", ".join(providers.keys())
        msg = f"Unknown provider: {provider}. Available: {available}"
        raise ValueError(msg)

    return providers[provider]


def _build_rsync_excludes(excludes: list[str]) -> list[str]:
    """Build rsync --exclude flags from a list of patterns."""
    flags = []
    for pattern in excludes:
        flags.extend(["--exclude", pattern])
    return flags


def sync_to_remote(
    experiment_dir: Path,
    host: str,
    remote_dir: str,
    ssh_key: str,
    excludes: list[str],
) -> bool:
    """Rsync experiment directory to remote host."""
    logger.info("Syncing %s → %s:%s", experiment_dir, host, remote_dir)

    cmd = [
        "rsync",
        "-avz",
        "--progress",
        "-e",
        f"ssh -i {ssh_key}",
        *_build_rsync_excludes(excludes),
        f"{experiment_dir}/",
        f"{host}:{remote_dir}/",
    ]

    result = subprocess.run(cmd, text=True)
    return result.returncode == 0


def run_remote_training(
    host: str,
    remote_dir: str,
    ssh_key: str,
    config: str = "configs/grpo.yaml",
) -> bool:
    """Run training on the remote host via SSH."""
    logger.info("Starting remote training on %s", host)

    # Cloud training uses WANDB_MODE=online (no Device Guard on Linux)
    train_cmd = (
        f"cd {remote_dir} && "
        f"export WANDB_MODE=online && "
        f"pip install -e '.[llm]' 2>/dev/null && "
        f"make train CONFIG={config}"
    )

    cmd = [
        "ssh",
        "-i",
        ssh_key,
        host,
        train_cmd,
    ]

    result = subprocess.run(cmd, text=True)
    return result.returncode == 0


def sync_from_remote(
    host: str,
    remote_dir: str,
    experiment_dir: Path,
    ssh_key: str,
) -> bool:
    """Rsync results and checkpoints back from remote."""
    logger.info("Syncing results %s:%s → %s", host, remote_dir, experiment_dir)

    cmd = [
        "rsync",
        "-avz",
        "--progress",
        "-e",
        f"ssh -i {ssh_key}",
        f"{host}:{remote_dir}/checkpoints/",
        f"{experiment_dir}/checkpoints/",
    ]

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        return False

    # Also sync results and wandb
    for subdir in ["results", "wandb"]:
        cmd = [
            "rsync",
            "-avz",
            "-e",
            f"ssh -i {ssh_key}",
            f"{host}:{remote_dir}/{subdir}/",
            f"{experiment_dir}/{subdir}/",
        ]
        subprocess.run(cmd, text=True)

    return True


def launch(
    experiment_dir: Path,
    provider: str,
    host: str | None = None,
    config: str = "configs/grpo.yaml",
) -> bool:
    """Full cloud training lifecycle: sync → train → sync back.

    Args:
        experiment_dir: Local experiment directory
        provider: Cloud provider name (runpod, lambda, vast)
        host: SSH host (user@ip). If None, must be provisioned manually.
        config: Config file path relative to experiment dir

    Returns:
        True if training completed successfully
    """
    provider_config = load_provider_config(provider)

    # Validate API key
    api_key_env = provider_config["api_key_env"]
    if not os.environ.get(api_key_env):
        logger.warning("%s not set — manual instance management only", api_key_env)

    if not host:
        logger.error(
            "No --host specified. Provision an instance and pass --host user@ip.\n"
            "Future: auto-provisioning via provider API."
        )
        return False

    ssh_key = os.path.expanduser(provider_config.get("ssh_key", "~/.ssh/id_rsa"))
    excludes = provider_config.get("sync_exclude", [])
    remote_dir = f"~/experiments/{experiment_dir.name}"

    # Step 1: Sync to remote
    if not sync_to_remote(experiment_dir, host, remote_dir, ssh_key, excludes):
        logger.error("Failed to sync experiment to remote")
        return False

    # Step 2: Run training
    if not run_remote_training(host, remote_dir, ssh_key, config):
        logger.error("Remote training failed")
        # Still try to sync results back
        sync_from_remote(host, remote_dir, experiment_dir, ssh_key)
        return False

    # Step 3: Sync results back
    if not sync_from_remote(host, remote_dir, experiment_dir, ssh_key):
        logger.error("Failed to sync results from remote")
        return False

    logger.info("Cloud training complete: %s", experiment_dir.name)
    return True


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Launch cloud training")
    parser.add_argument("--experiment", type=Path, required=True, help="Experiment directory")
    parser.add_argument("--provider", required=True, choices=["runpod", "lambda", "vast"])
    parser.add_argument("--host", type=str, default=None, help="SSH host (user@ip)")
    parser.add_argument("--config", default="configs/grpo.yaml", help="Config file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    success = launch(args.experiment, args.provider, args.host, args.config)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
