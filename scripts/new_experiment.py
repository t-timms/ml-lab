"""Initialize a new experiment from the ml-experiment-scaffold template.

Usage:
    python scripts/new_experiment.py --name gsm8k-grpo --scaffold-dir /path/to/scaffold
"""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Directories and files to copy from the scaffold template
SCAFFOLD_COPY = [
    "configs/",
    "ds_configs/",
    "src/",
    "scripts/",
    "tests/",
    "Makefile",
    "pyproject.toml",
    "environment.yml",
]

# Files to skip (scaffold-specific, not experiment-relevant)
SCAFFOLD_SKIP = {
    "CLAUDE.md",
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "LICENSE",
    ".github",
    ".git",
    ".pre-commit-config.yaml",
    "ml_experiment_scaffold.egg-info",
    "uv.lock",
    "__pycache__",
    ".pytest_cache",
    "notebooks",
}

RESEARCH_LOG_TEMPLATE = """# Research Log — {name}

## Hypothesis

[What you expect to observe. Be specific — not "the model will improve" but
"correctness will increase 5-10% by step 200 as the gradient stabilizes"]

## Run 1 — {date}

### Setup
- Config: configs/[method].yaml
- Hardware: RTX 5070 Ti, 16GB VRAM
- Key hyperparams: [lr, batch_size, epochs, etc.]

### Phase 1 (Steps 0-X): [title describing what happened]

### Unexpected Findings

### Finding Classification
<!-- Classify each observation as one of:
  Expected      — consistent with prior work and theory
  Interesting   — unexpected quantitative detail worth tracking
  Anomalous     — contradicts hypothesis or published results
  Methodological — important for interpreting results correctly
  Breakthrough-Candidate — could be a novel contribution if reproducible
-->

### Comparison to Prior Runs

### Updated Hypotheses

## Cross-Run Comparison

| Run | Config Delta | Peak Correct | Saturation Step | Notes |
|-----|-------------|-------------|----------------|-------|
| Run 1 | baseline | ___% | ___ | |

## Potential Paper Angles
"""

NOTES_TEMPLATE = """# Experiment Notes — {name}

Created: {date}

## Goal

[One paragraph: what are you trying to learn or build?]

## Paper Reference

[If reproducing a paper: title, arxiv link, key claims to verify]

## Key Decisions

[Document non-obvious choices as you make them]

## Results Summary

[Fill in after training completes]
"""

MODEL_CARD_TEMPLATE = """# Model Card — {name}

## Model Details
- **Base model**: [e.g., Qwen/Qwen2.5-3B-Instruct]
- **Method**: [SFT/ORPO/GRPO]
- **Training data**: [dataset name and size]
- **Training steps**: [X]
- **Hardware**: RTX 5070 Ti (16GB VRAM)

## Intended Use
[What is this model for?]

## Training Procedure
[Key hyperparameters, LoRA config, etc.]

## Evaluation Results
| Benchmark | Score |
|-----------|-------|
| arc_easy  |       |
| hellaswag |       |

## Limitations
[Known limitations and failure modes]
"""

PAPER_TEMPLATE = """# Paper Notes — {name}

## Citation

## Key Claims

## Method Summary

## Relevance to This Experiment

## Implementation Notes
"""


def create_experiment(
    name: str,
    scaffold_dir: Path,
    output_dir: Path,
) -> Path:
    """Create a new experiment directory from the scaffold template.

    Args:
        name: Experiment name (e.g., 'gsm8k-grpo')
        scaffold_dir: Path to ml-experiment-scaffold repo
        output_dir: Parent directory for experiments

    Returns:
        Path to the created experiment directory
    """
    date_prefix = datetime.now(tz=UTC).strftime("%Y-%m")
    experiment_name = f"{date_prefix}-{name}"
    experiment_dir = output_dir / experiment_name

    if experiment_dir.exists():
        msg = f"Experiment directory already exists: {experiment_dir}"
        raise FileExistsError(msg)

    experiment_dir.mkdir(parents=True)
    logger.info("Creating experiment: %s", experiment_name)

    # Copy scaffold directories and files
    for item in SCAFFOLD_COPY:
        src = scaffold_dir / item
        if not src.exists():
            logger.warning("Scaffold item not found, skipping: %s", item)
            continue

        dst = experiment_dir / item
        if src.is_dir():
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
            )
        else:
            shutil.copy2(src, dst)
        logger.info("Copied: %s", item)

    # Create experiment-specific directories
    for subdir in ["data", "checkpoints", "results", "logs"]:
        (experiment_dir / subdir).mkdir(exist_ok=True)
        (experiment_dir / subdir / ".gitkeep").touch()

    # Create template files
    date_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    templates = {
        "RESEARCH_LOG.md": RESEARCH_LOG_TEMPLATE.format(name=experiment_name, date=date_str),
        "notes.md": NOTES_TEMPLATE.format(name=experiment_name, date=date_str),
        "model_card.md": MODEL_CARD_TEMPLATE.format(name=experiment_name),
        "paper.md": PAPER_TEMPLATE.format(name=experiment_name),
    }

    for filename, content in templates.items():
        (experiment_dir / filename).write_text(content.strip() + "\n")
        logger.info("Created: %s", filename)

    # Create EXPERIMENT_GUIDE.md placeholder (gitignored)
    (experiment_dir / "EXPERIMENT_GUIDE.md").write_text(
        f"# Experiment Guide — {experiment_name}\n\n"
        "This file is gitignored. Use it for personal notes on the science,\n"
        "architecture decisions, and methodology of this experiment.\n"
    )

    logger.info("Experiment created: %s", experiment_dir)
    return experiment_dir


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Initialize a new ML experiment")
    parser.add_argument("--name", required=True, help="Experiment name (e.g., gsm8k-grpo)")
    parser.add_argument(
        "--scaffold-dir",
        type=Path,
        required=True,
        help="Path to ml-experiment-scaffold repo",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments"),
        help="Parent directory for experiments (default: experiments/)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    experiment_dir = create_experiment(
        name=args.name,
        scaffold_dir=args.scaffold_dir,
        output_dir=args.output_dir,
    )
    print(f"\nExperiment ready: {experiment_dir}")
    print("\nNext steps:")
    print(f"  1. Edit configs in {experiment_dir}/configs/")
    print(f"  2. make validate-config EXP={experiment_dir.name}")
    print(f"  3. make train EXP={experiment_dir.name}")


if __name__ == "__main__":
    main()
