"""Tests for register_model — model registry operations."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.register_model import register_model


class TestRegisterModel:
    def _create_experiment(self, tmp_path: Path) -> Path:
        """Create a minimal experiment directory."""
        exp_dir = tmp_path / "experiments" / "2026-04-test-grpo"
        exp_dir.mkdir(parents=True)

        configs = exp_dir / "configs"
        configs.mkdir()
        (configs / "grpo.yaml").write_text(
            "model:\n  name: Qwen/Qwen2.5-3B-Instruct\n"
            "training:\n  method: grpo\n  max_steps: 500\n"
        )

        (exp_dir / "results").mkdir()
        (exp_dir / "checkpoints").mkdir()

        return exp_dir

    def test_creates_registry_entry(self, tmp_path: Path) -> None:
        exp_dir = self._create_experiment(tmp_path)
        registry = tmp_path / "registry" / "models.jsonl"

        register_model(exp_dir, registry)

        assert registry.exists()
        lines = registry.read_text().strip().split("\n")
        assert len(lines) == 1

        stored = json.loads(lines[0])
        assert stored["method"] == "grpo"
        assert stored["base_model"] == "Qwen/Qwen2.5-3B-Instruct"

    def test_appends_to_existing_registry(self, tmp_path: Path) -> None:
        exp_dir = self._create_experiment(tmp_path)
        registry = tmp_path / "registry" / "models.jsonl"

        register_model(exp_dir, registry, seed=42)
        register_model(exp_dir, registry, seed=0)

        lines = registry.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_includes_config_hash(self, tmp_path: Path) -> None:
        exp_dir = self._create_experiment(tmp_path)
        registry = tmp_path / "registry" / "models.jsonl"

        entry = register_model(exp_dir, registry)
        assert len(entry["config_hash"]) == 8

    def test_loads_eval_scores(self, tmp_path: Path) -> None:
        exp_dir = self._create_experiment(tmp_path)
        results_dir = exp_dir / "results"
        (results_dir / "eval.json").write_text(json.dumps({"arc_easy": 72.1, "hellaswag": 58.3}))
        registry = tmp_path / "registry" / "models.jsonl"

        entry = register_model(exp_dir, registry)
        assert entry["eval_scores"]["arc_easy"] == 72.1
