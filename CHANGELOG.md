# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-04-09

### Added
- Experiment lifecycle management (7-stage pipeline)
- `scripts/new_experiment.py` — scaffold template initialization
- `scripts/preflight.py` — GPU health check via gpu-server-test-suite
- `scripts/register_model.py` — model registry with JSONL storage
- `scripts/cross_compare.py` — cross-experiment leaderboard
- `scripts/sync_wandb.py` — automated W&B offline sync via WSL
- `scripts/research_to_wiki.py` — publish findings to llm-wiki
- `src/ml_lab/config_validator.py` — catches impossible training configs
- `src/ml_lab/cli.py` — Click CLI entry point
- Cloud training infrastructure (providers.yaml, launch.py, Dockerfile)
- Top-level Makefile orchestrating full experiment lifecycle
- Test suite with 20+ tests
