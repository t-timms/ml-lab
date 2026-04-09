# Model Registry

Append-only JSONL index of trained models at `models.jsonl`.

## Schema

Each line is a JSON object with these fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique ID: `{experiment}-seed{N}-{date}` |
| `base_model` | string | HuggingFace model ID (e.g., `Qwen/Qwen2.5-3B-Instruct`) |
| `method` | string | Training method: `sft`, `orpo`, `grpo`, `vision` |
| `experiment` | string | Experiment directory name |
| `seed` | int | Training seed |
| `config_hash` | string | SHA-256 prefix of the training config (8 chars) |
| `eval_scores` | object | `{metric: score}` from lm-eval |
| `checkpoint_path` | string | Relative path to best checkpoint |
| `merged_path` | string? | Path to merged (non-LoRA) model, if applicable |
| `tags` | string[] | Searchable tags |
| `date` | string | ISO date (YYYY-MM-DD) |
| `vram_peak_gb` | float? | Peak VRAM usage during training |
| `training_steps` | int? | Total training steps |
| `wandb_run` | string? | W&B run path |

## Usage

```bash
# Register a model after training
make register EXP=2026-04-gsm8k-grpo

# View leaderboard
make leaderboard

# Filter by method
python scripts/cross_compare.py --method grpo --sort arc_easy/acc

# Search registry
grep "grpo" registry/models.jsonl | python -m json.tool
```

## Conventions

- **One entry per seed per experiment** — 3-seed runs produce 3 entries
- **Append-only** — never delete entries, track history
- **Config hash** — same hash = same config = reproducible
- **Eval scores** — populated after `make eval`, keys follow `{task}/{metric}` format
