"""Register a trained model in the model registry.

Reads experiment results, computes config hash, and appends an entry
to registry/models.jsonl.

Usage:
    python scripts/register_model.py --experiment-dir experiments/2026-04-gsm8k-grpo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _compute_config_hash(config: dict) -> str:
    """Compute a short hash of the config for reproducibility tracking."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


def _find_checkpoint(experiment_dir: Path) -> str | None:
    """Find the latest checkpoint in the experiment directory."""
    checkpoints_dir = experiment_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return None

    checkpoint_dirs = sorted(
        [d for d in checkpoints_dir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    if checkpoint_dirs:
        return str(checkpoint_dirs[0].relative_to(experiment_dir.parent.parent))
    return None


def _load_eval_scores(experiment_dir: Path) -> dict[str, float]:
    """Load evaluation scores from results directory."""
    results_dir = experiment_dir / "results"
    scores: dict[str, float] = {}

    if not results_dir.exists():
        return scores

    # Try loading from final results JSON
    for results_file in results_dir.glob("*.json"):
        try:
            data = json.loads(results_file.read_text())
            if isinstance(data, dict):
                # lm-eval format
                if "results" in data:
                    for task_name, task_results in data["results"].items():
                        for metric, value in task_results.items():
                            if isinstance(value, (int, float)):
                                scores[f"{task_name}/{metric}"] = round(value, 2)
                else:
                    scores.update(
                        {k: round(v, 2) for k, v in data.items() if isinstance(v, (int, float))}
                    )
        except (json.JSONDecodeError, TypeError):
            continue

    return scores


def register_model(
    experiment_dir: Path,
    registry_path: Path,
    seed: int = 42,
    tags: list[str] | None = None,
) -> dict:
    """Register a model from a completed experiment.

    Args:
        experiment_dir: Path to experiment directory
        registry_path: Path to models.jsonl
        seed: Training seed
        tags: Optional tags for the model

    Returns:
        The registry entry dict
    """
    experiment_name = experiment_dir.name

    # Find the active config (prefer method-specific over base)
    configs_dir = experiment_dir / "configs"
    config = {}
    config_name = "unknown"
    for candidate in ["grpo.yaml", "orpo.yaml", "sft.yaml", "vision.yaml", "tabular.yaml"]:
        config_path = configs_dir / candidate
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
            config_name = candidate.replace(".yaml", "")
            break

    model_config = config.get("model", {})
    training_config = config.get("training", {})
    method = training_config.get("method", config_name)
    base_model = model_config.get("name", "unknown")

    entry = {
        "id": f"{experiment_name}-seed{seed}-{datetime.now(tz=UTC).strftime('%Y%m%d')}",
        "base_model": base_model,
        "method": method,
        "experiment": experiment_name,
        "seed": seed,
        "config_hash": _compute_config_hash(config),
        "eval_scores": _load_eval_scores(experiment_dir),
        "checkpoint_path": _find_checkpoint(experiment_dir),
        "merged_path": None,
        "tags": tags
        or [method, experiment_name.split("-", 2)[-1] if "-" in experiment_name else ""],
        "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "vram_peak_gb": None,
        "training_steps": training_config.get("max_steps")
        or training_config.get("num_train_epochs"),
        "wandb_run": None,
    }

    # Append to registry
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    logger.info("Registered model: %s", entry["id"])
    return entry


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Register a trained model")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=Path("registry/models.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tags", nargs="*", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    entry = register_model(args.experiment_dir, args.registry, args.seed, args.tags)
    print(f"\nRegistered: {entry['id']}")
    print(f"Registry:   {args.registry}")


if __name__ == "__main__":
    main()
