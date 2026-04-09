"""Automated W&B offline run sync via WSL.

Device Guard blocks wandb.exe on Windows. This script syncs offline runs
by invoking wandb through WSL (Ubuntu).

Usage:
    python scripts/sync_wandb.py --experiments-dir experiments/
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SYNCED_RUNS_FILE = Path("registry/synced_runs.txt")


def _get_synced_runs() -> set[str]:
    """Load the set of already-synced run directories."""
    if not SYNCED_RUNS_FILE.exists():
        return set()
    return set(SYNCED_RUNS_FILE.read_text().strip().split("\n"))


def _mark_synced(run_dir: str) -> None:
    """Mark a run directory as synced."""
    SYNCED_RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNCED_RUNS_FILE, "a") as f:
        f.write(run_dir + "\n")


def _windows_to_wsl_path(windows_path: str) -> str:
    """Convert a Windows path to WSL-compatible path."""
    # C:/Users/... -> /mnt/c/Users/...
    path = windows_path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        path = f"/mnt/{drive}{path[2:]}"
    return path


def find_offline_runs(experiments_dir: Path) -> list[Path]:
    """Find all un-synced offline W&B runs across experiments."""
    synced = _get_synced_runs()
    runs = []

    for wandb_dir in experiments_dir.glob("*/wandb"):
        for run_dir in wandb_dir.glob("offline-run-*"):
            if run_dir.is_dir() and str(run_dir) not in synced:
                runs.append(run_dir)

    return sorted(runs)


def sync_run(run_dir: Path, api_key: str) -> bool:
    """Sync a single offline W&B run via WSL.

    Args:
        run_dir: Path to the offline-run-* directory
        api_key: W&B API key

    Returns:
        True if sync succeeded
    """
    wsl_path = _windows_to_wsl_path(str(run_dir.resolve()))

    logger.info("Syncing: %s", run_dir.name)

    try:
        result = subprocess.run(
            [
                "wsl",
                "-d",
                "Ubuntu",
                "--",
                "bash",
                "-c",
                f"WANDB_API_KEY={api_key} wandb sync '{wsl_path}'",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            logger.info("Synced successfully: %s", run_dir.name)
            _mark_synced(str(run_dir))
            return True

        logger.error("Sync failed for %s: %s", run_dir.name, result.stderr.strip())
        return False

    except subprocess.TimeoutExpired:
        logger.error("Sync timed out for %s", run_dir.name)
        return False
    except FileNotFoundError:
        logger.error("WSL not found — cannot sync W&B runs without WSL")
        return False


def sync_all(experiments_dir: Path) -> tuple[int, int]:
    """Sync all un-synced offline W&B runs.

    Returns:
        Tuple of (synced_count, failed_count)
    """
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        logger.error("WANDB_API_KEY not set. Export it before syncing.")
        return 0, 0

    runs = find_offline_runs(experiments_dir)
    if not runs:
        logger.info("No un-synced offline runs found.")
        return 0, 0

    logger.info("Found %d un-synced run(s)", len(runs))

    synced = 0
    failed = 0
    for run_dir in runs:
        if sync_run(run_dir, api_key):
            synced += 1
        else:
            failed += 1

    return synced, failed


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Sync offline W&B runs via WSL")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Path to experiments directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    synced, failed = sync_all(args.experiments_dir)

    if synced or failed:
        print(f"\nW&B sync complete: {synced} synced, {failed} failed")
    else:
        print("\nNo offline runs to sync.")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
