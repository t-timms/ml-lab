"""GPU preflight check — validates hardware health before training.

Shells out to gpu-server-test-suite's diagnostic CLI in preflight mode.
Blocks training if GPU is unhealthy (thermal throttle, ECC errors, VRAM issues).

Usage:
    python scripts/preflight.py
    python scripts/preflight.py --gpu-diag-dir /path/to/gpu-server-test-suite
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_GPU_DIAG_DIR = Path.home() / "Documents" / "Project Portfolio" / "gpu-server-test-suite"


def run_preflight(gpu_diag_dir: Path) -> bool:
    """Run GPU preflight diagnostics.

    Args:
        gpu_diag_dir: Path to gpu-server-test-suite repo

    Returns:
        True if GPU is healthy, False otherwise
    """
    if not gpu_diag_dir.exists():
        logger.warning("gpu-server-test-suite not found at %s — skipping preflight", gpu_diag_dir)
        return True

    main_py = gpu_diag_dir / "src" / "main.py"
    if not main_py.exists():
        logger.warning("gpu-server-test-suite main.py not found — skipping preflight")
        return True

    logger.info("Running GPU preflight diagnostics...")

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.main",
                "diag",
                "--level",
                "quick",
                "--mode",
                "preflight",
            ],
            cwd=str(gpu_diag_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            logger.error("GPU preflight FAILED — do not proceed with training")
            return False

        logger.info("GPU preflight PASSED")
        return True

    except subprocess.TimeoutExpired:
        logger.error("GPU preflight timed out after 120s")
        return False
    except FileNotFoundError:
        logger.warning("Python not found — skipping preflight")
        return True


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="GPU preflight health check")
    parser.add_argument(
        "--gpu-diag-dir",
        type=Path,
        default=DEFAULT_GPU_DIAG_DIR,
        help="Path to gpu-server-test-suite repo",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    healthy = run_preflight(args.gpu_diag_dir)
    if not healthy:
        print("\nGPU preflight FAILED. Fix issues before training.")
        sys.exit(1)
    print("\nGPU preflight PASSED. Ready to train.")


if __name__ == "__main__":
    main()
