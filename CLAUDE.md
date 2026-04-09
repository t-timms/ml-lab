# ML Lab — Claude Code Instructions

## Project Overview

ML research control plane connecting ml-experiment-scaffold (template), gpu-server-test-suite (preflight), and llm-wiki (knowledge) into a unified experiment lifecycle. Manages experiment instances, model registry, cloud training, and W&B sync.

## Architecture

```
Makefile (orchestrator)
├── scripts/new_experiment.py  → copies scaffold into experiments/
├── scripts/preflight.py       → shells out to gpu-server-test-suite
├── src/ml_lab/config_validator.py → validates configs before training
├── scripts/register_model.py  → appends to registry/models.jsonl
├── scripts/cross_compare.py   → generates leaderboard from registry
├── scripts/sync_wandb.py      → WSL-based offline W&B sync
├── scripts/research_to_wiki.py → pushes findings to llm-wiki
└── cloud/launch.py            → rsync + SSH cloud orchestrator
```

## Experiment Lifecycle

1. `make preflight` — GPU health check
2. `make new-experiment NAME=<name>` — init from scaffold
3. Edit configs, run `make validate-config EXP=<name>`
4. `make train EXP=<name>` (local) or `make cloud-train EXP=<name> PROVIDER=runpod`
5. `make eval EXP=<name>` — lm-eval benchmarks
6. `make register EXP=<name>` — add to model registry
7. `make publish EXP=<name>` — sync W&B + push to llm-wiki

## Key Constraints

- Experiments live in `experiments/YYYY-MM-<name>/` — never modify scaffold directly
- Model registry is append-only JSONL — never delete entries
- W&B must use `WANDB_MODE=offline` locally (Device Guard), `online` on cloud
- Config validator catches: fp8 training, bf16=False, seq_length>2048 without DeepSpeed
- Cloud training uses rsync + SSH — same configs as local, different compute

## Commands

```bash
make test                          # Unit tests
make lint                          # Ruff check + format
make new-experiment NAME=test      # Init experiment
make validate-config EXP=<name>    # Validate configs
make train EXP=<name>              # Preflight + train
make register EXP=<name>           # Register model
make leaderboard                   # Cross-experiment comparison
make sync-wandb                    # Sync offline W&B runs
```

## Testing

- Tests in `tests/` — all run without GPU
- `pytest tests/ -v --tb=short`
- Test new scripts by creating a mock experiment in tmp_path

## Code Standards

- `from __future__ import annotations` in every module
- `logging.getLogger(__name__)` — never `print()` in library code
- Type hints on all public functions
- Ruff for lint/format, line length 100
