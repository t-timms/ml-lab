# ==============================================================================
# ML Lab — Control Plane Makefile
# Usage: make <target> [EXP=experiment-name] [PROVIDER=runpod]
# ==============================================================================

.DEFAULT_GOAL := help
CONDA_RUN := conda run -n mlenv
SCAFFOLD_DIR := $(HOME)/Documents/Project\ Portfolio/ml-experiment-scaffold
GPU_DIAG_DIR := $(HOME)/Documents/Project\ Portfolio/gpu-server-test-suite
EXP_DIR := experiments
REGISTRY := registry/models.jsonl

.PHONY: help preflight new-experiment validate-config train cloud-train eval \
        register sync-wandb publish leaderboard lint test clean

help:
	@echo ""
	@echo "ML Lab — Control Plane"
	@echo "────────────────────────────────────────────────────────────"
	@echo ""
	@echo "  Lifecycle:"
	@echo "    preflight                GPU health check (gpu-server-test-suite)"
	@echo "    new-experiment NAME=x    Init experiment from scaffold template"
	@echo "    validate-config EXP=x    Validate experiment config"
	@echo "    train EXP=x              Train locally (runs preflight first)"
	@echo "    cloud-train EXP=x        Train on cloud GPU (PROVIDER=runpod|lambda|vast)"
	@echo "    eval EXP=x               Evaluate trained model"
	@echo "    register EXP=x           Register model in registry"
	@echo "    sync-wandb               Sync all offline W&B runs via WSL"
	@echo "    publish EXP=x            Push findings to llm-wiki + update leaderboard"
	@echo ""
	@echo "  Registry:"
	@echo "    leaderboard              Generate cross-experiment leaderboard"
	@echo ""
	@echo "  Quality:"
	@echo "    lint                     Ruff check + format"
	@echo "    test                     Run unit tests"
	@echo ""

# ── Preflight ──────────────────────────────────────────────────────────────
preflight:
	@echo "Running GPU preflight diagnostics..."
	$(CONDA_RUN) python scripts/preflight.py

# ── Experiment Lifecycle ───────────────────────────────────────────────────
new-experiment:
ifndef NAME
	$(error NAME is required. Usage: make new-experiment NAME=gsm8k-grpo)
endif
	$(CONDA_RUN) python scripts/new_experiment.py --name $(NAME) \
		--scaffold-dir "$(SCAFFOLD_DIR)" --output-dir "$(EXP_DIR)"

validate-config:
ifndef EXP
	$(error EXP is required. Usage: make validate-config EXP=2026-04-gsm8k-grpo)
endif
	$(CONDA_RUN) python -m src.ml_lab.config_validator \
		--experiment-dir "$(EXP_DIR)/$(EXP)"

train: preflight
ifndef EXP
	$(error EXP is required. Usage: make train EXP=2026-04-gsm8k-grpo)
endif
	@echo "Training experiment: $(EXP)"
	cd "$(EXP_DIR)/$(EXP)" && $(MAKE) train

cloud-train:
ifndef EXP
	$(error EXP is required. Usage: make cloud-train EXP=2026-04-gsm8k-grpo PROVIDER=runpod)
endif
ifndef PROVIDER
	$(error PROVIDER is required. Usage: make cloud-train EXP=x PROVIDER=runpod|lambda|vast)
endif
	$(CONDA_RUN) python cloud/launch.py --experiment "$(EXP_DIR)/$(EXP)" \
		--provider $(PROVIDER)

eval:
ifndef EXP
	$(error EXP is required. Usage: make eval EXP=2026-04-gsm8k-grpo)
endif
	cd "$(EXP_DIR)/$(EXP)" && $(MAKE) eval

# ── Registry ───────────────────────────────────────────────────────────────
register:
ifndef EXP
	$(error EXP is required. Usage: make register EXP=2026-04-gsm8k-grpo)
endif
	$(CONDA_RUN) python scripts/register_model.py \
		--experiment-dir "$(EXP_DIR)/$(EXP)" --registry $(REGISTRY)

leaderboard:
	$(CONDA_RUN) python scripts/cross_compare.py --registry $(REGISTRY)

# ── W&B Sync ──────────────────────────────────────────────────────────────
sync-wandb:
	$(CONDA_RUN) python scripts/sync_wandb.py --experiments-dir "$(EXP_DIR)"

# ── Knowledge ─────────────────────────────────────────────────────────────
publish:
ifndef EXP
	$(error EXP is required. Usage: make publish EXP=2026-04-gsm8k-grpo)
endif
	$(CONDA_RUN) python scripts/research_to_wiki.py \
		--experiment-dir "$(EXP_DIR)/$(EXP)"
	@$(MAKE) sync-wandb
	@$(MAKE) leaderboard

# ── Quality ────────────────────────────────────────────────────────────────
lint:
	$(CONDA_RUN) ruff check src/ scripts/ tests/ --fix
	$(CONDA_RUN) ruff format src/ scripts/ tests/

test:
	$(CONDA_RUN) pytest tests/ -v --tb=short

clean:
	rm -rf __pycache__ .pytest_cache
