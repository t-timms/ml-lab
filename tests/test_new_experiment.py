"""Tests for new_experiment.py — experiment initialization from scaffold."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def scaffold_dir(tmp_path: Path) -> Path:
    """Create a minimal scaffold directory for testing."""
    scaffold = tmp_path / "scaffold"
    scaffold.mkdir()

    # Create scaffold structure
    (scaffold / "configs").mkdir()
    (scaffold / "configs" / "base.yaml").write_text("model:\n  name: test\n")
    (scaffold / "configs" / "grpo.yaml").write_text("_base: base.yaml\nmethod: grpo\n")

    (scaffold / "src").mkdir()
    (scaffold / "src" / "__init__.py").touch()
    (scaffold / "src" / "train.py").write_text("# training code\n")

    (scaffold / "tests").mkdir()
    (scaffold / "tests" / "test_config.py").write_text("# tests\n")

    (scaffold / "scripts").mkdir()
    (scaffold / "scripts" / "__init__.py").touch()

    (scaffold / "Makefile").write_text("train:\n\techo training\n")
    (scaffold / "pyproject.toml").write_text("[project]\nname = 'scaffold'\n")

    return scaffold


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create output directory for experiments."""
    out = tmp_path / "experiments"
    out.mkdir()
    return out


class TestCreateExperiment:
    def test_creates_experiment_directory(self, scaffold_dir: Path, output_dir: Path) -> None:
        from scripts.new_experiment import create_experiment

        with patch("scripts.new_experiment.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.side_effect = lambda fmt: (
                "2026-04" if "Y" in fmt and "d" not in fmt else "2026-04-09"
            )

            result = create_experiment("test-grpo", scaffold_dir, output_dir)

        assert result.exists()
        assert "test-grpo" in result.name

    def test_copies_configs(self, scaffold_dir: Path, output_dir: Path) -> None:
        from scripts.new_experiment import create_experiment

        result = create_experiment("test-grpo", scaffold_dir, output_dir)

        assert (result / "configs" / "base.yaml").exists()
        assert (result / "configs" / "grpo.yaml").exists()

    def test_copies_makefile(self, scaffold_dir: Path, output_dir: Path) -> None:
        from scripts.new_experiment import create_experiment

        result = create_experiment("test-grpo", scaffold_dir, output_dir)

        assert (result / "Makefile").exists()

    def test_creates_template_files(self, scaffold_dir: Path, output_dir: Path) -> None:
        from scripts.new_experiment import create_experiment

        result = create_experiment("test-grpo", scaffold_dir, output_dir)

        assert (result / "RESEARCH_LOG.md").exists()
        assert (result / "notes.md").exists()
        assert (result / "model_card.md").exists()
        assert (result / "paper.md").exists()

    def test_creates_subdirectories(self, scaffold_dir: Path, output_dir: Path) -> None:
        from scripts.new_experiment import create_experiment

        result = create_experiment("test-grpo", scaffold_dir, output_dir)

        for subdir in ["data", "checkpoints", "results", "logs"]:
            assert (result / subdir).is_dir()

    def test_raises_on_duplicate(self, scaffold_dir: Path, output_dir: Path) -> None:
        from scripts.new_experiment import create_experiment

        create_experiment("test-grpo", scaffold_dir, output_dir)

        with pytest.raises(FileExistsError):
            create_experiment("test-grpo", scaffold_dir, output_dir)

    def test_creates_experiment_guide(self, scaffold_dir: Path, output_dir: Path) -> None:
        from scripts.new_experiment import create_experiment

        result = create_experiment("test-grpo", scaffold_dir, output_dir)

        assert (result / "EXPERIMENT_GUIDE.md").exists()
