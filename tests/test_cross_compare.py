"""Tests for cross_compare — leaderboard generation."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.cross_compare import filter_entries, generate_leaderboard, load_registry


class TestLoadRegistry:
    def test_loads_valid_jsonl(self, tmp_path: Path) -> None:
        registry = tmp_path / "models.jsonl"
        entries = [
            {"id": "test-1", "method": "grpo", "eval_scores": {"arc": 70.0}},
            {"id": "test-2", "method": "sft", "eval_scores": {"arc": 65.0}},
        ]
        registry.write_text("\n".join(json.dumps(e) for e in entries))

        result = load_registry(registry)
        assert len(result) == 2

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        result = load_registry(tmp_path / "missing.jsonl")
        assert result == []

    def test_skips_invalid_lines(self, tmp_path: Path) -> None:
        registry = tmp_path / "models.jsonl"
        registry.write_text('{"id": "good"}\nnot json\n{"id": "also-good"}\n')

        result = load_registry(registry)
        assert len(result) == 2


class TestFilterEntries:
    def test_filter_by_method(self) -> None:
        entries = [
            {"method": "grpo"},
            {"method": "sft"},
            {"method": "grpo"},
        ]
        result = filter_entries(entries, method="grpo")
        assert len(result) == 2

    def test_filter_by_base_model(self) -> None:
        entries = [
            {"base_model": "Qwen/Qwen2.5-3B"},
            {"base_model": "meta-llama/Llama-3-8B"},
        ]
        result = filter_entries(entries, base_model="qwen")
        assert len(result) == 1


class TestGenerateLeaderboard:
    def test_generates_markdown_table(self) -> None:
        entries = [
            {
                "id": "test-1",
                "base_model": "Qwen/Qwen2.5-3B",
                "method": "grpo",
                "seed": 42,
                "eval_scores": {"arc_easy": 72.1},
                "date": "2026-04-09",
            },
        ]
        result = generate_leaderboard(entries)
        assert "# Model Leaderboard" in result
        assert "test-1" in result
        assert "72.1" in result

    def test_empty_entries(self) -> None:
        result = generate_leaderboard([])
        assert "No models" in result
