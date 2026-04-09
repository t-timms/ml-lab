"""Cross-experiment leaderboard — compare models across experiments.

Reads registry/models.jsonl and generates a markdown leaderboard table.

Usage:
    python scripts/cross_compare.py --registry registry/models.jsonl
    python scripts/cross_compare.py --method grpo --min-score 70
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_registry(registry_path: Path) -> list[dict]:
    """Load all entries from the model registry."""
    if not registry_path.exists():
        logger.warning("Registry not found: %s", registry_path)
        return []

    entries = []
    for line in registry_path.read_text().strip().split("\n"):
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid registry line: %s", line[:80])
    return entries


def filter_entries(
    entries: list[dict],
    method: str | None = None,
    base_model: str | None = None,
    min_date: str | None = None,
) -> list[dict]:
    """Filter registry entries by method, base model, or date."""
    filtered = entries

    if method:
        filtered = [e for e in filtered if e.get("method") == method]
    if base_model:
        filtered = [e for e in filtered if base_model.lower() in e.get("base_model", "").lower()]
    if min_date:
        filtered = [e for e in filtered if e.get("date", "") >= min_date]

    return filtered


def generate_leaderboard(entries: list[dict], sort_metric: str | None = None) -> str:
    """Generate a markdown leaderboard from registry entries.

    Args:
        entries: List of registry entry dicts
        sort_metric: Metric key to sort by (e.g., 'arc_easy/acc')

    Returns:
        Markdown table string
    """
    if not entries:
        return "No models in registry.\n"

    # Collect all eval metric keys across entries
    all_metrics: set[str] = set()
    for entry in entries:
        all_metrics.update(entry.get("eval_scores", {}).keys())

    metric_cols = sorted(all_metrics)

    # Build table
    lines = ["# Model Leaderboard", ""]

    header = "| ID | Base Model | Method | Seed | " + " | ".join(metric_cols) + " | Date |"
    separator = "|" + "|".join(["---"] * (5 + len(metric_cols))) + "|"
    lines.extend([header, separator])

    # Sort by metric if specified
    if sort_metric and sort_metric in all_metrics:
        entries = sorted(
            entries,
            key=lambda e: e.get("eval_scores", {}).get(sort_metric, -1),
            reverse=True,
        )
    else:
        entries = sorted(entries, key=lambda e: e.get("date", ""), reverse=True)

    for entry in entries:
        scores = entry.get("eval_scores", {})
        metric_vals = " | ".join(str(scores.get(m, "-")) for m in metric_cols)

        base_model = entry.get("base_model", "?")
        # Shorten HF model names for display
        if "/" in base_model:
            base_model = base_model.split("/")[-1]

        row = (
            f"| {entry.get('id', '?')} "
            f"| {base_model} "
            f"| {entry.get('method', '?')} "
            f"| {entry.get('seed', '?')} "
            f"| {metric_vals} "
            f"| {entry.get('date', '?')} |"
        )
        lines.append(row)

    lines.append("")
    lines.append(f"*{len(entries)} model(s) registered*")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Cross-experiment leaderboard")
    parser.add_argument("--registry", type=Path, default=Path("registry/models.jsonl"))
    parser.add_argument("--method", type=str, default=None, help="Filter by method")
    parser.add_argument("--base-model", type=str, default=None, help="Filter by base model")
    parser.add_argument("--sort", type=str, default=None, help="Sort by metric key")
    parser.add_argument("--output", type=Path, default=None, help="Write to file instead of stdout")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    entries = load_registry(args.registry)
    entries = filter_entries(entries, method=args.method, base_model=args.base_model)

    leaderboard = generate_leaderboard(entries, sort_metric=args.sort)

    if args.output:
        args.output.write_text(leaderboard)
        logger.info("Leaderboard written to %s", args.output)
    else:
        print(leaderboard)


if __name__ == "__main__":
    main()
