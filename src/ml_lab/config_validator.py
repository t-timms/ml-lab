"""Config validation — catches impossible hyperparameter combos before training.

Validates experiment configs against known hardware constraints and framework
limitations. Run before training to avoid wasting GPU time.

Usage:
    python -m src.ml_lab.config_validator --experiment-dir experiments/2026-04-gsm8k-grpo
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# VRAM budget (GB) for RTX 5070 Ti
LOCAL_VRAM_GB = 16


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML config, resolving _base inheritance."""
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    base_name = config.pop("_base", None)
    if base_name:
        base_path = config_path.parent / base_name
        if base_path.exists():
            base_config = _load_config(base_path)
            base_config = _deep_merge(base_config, config)
            return base_config
        logger.warning("Base config not found: %s", base_path)

    return config


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base. Override wins for leaf values."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_config(config: dict[str, Any], config_name: str = "") -> list[str]:
    """Validate a training config for known issues.

    Args:
        config: Parsed YAML config dict
        config_name: Name for error messages

    Returns:
        List of error messages. Empty = valid.
    """
    errors: list[str] = []
    warnings: list[str] = []

    training = config.get("training", {})
    model = config.get("model", {})
    method = training.get("method", config.get("method", ""))

    # fp8 training not supported in TRL SFTConfig/GRPOConfig
    if training.get("fp8") or config.get("fp8"):
        errors.append(
            "fp8=True in training config is not supported by TRL (v0.29+). "
            "Use bf16=True for training. fp8 is inference-only (Unsloth load_in_fp8)."
        )

    # bf16 must be True on Blackwell
    if training.get("bf16") is False or config.get("bf16") is False:
        errors.append("bf16 must be True on Blackwell (sm_120). Never use fp16.")

    # fp16 explicitly set
    if training.get("fp16") or config.get("fp16"):
        errors.append("fp16=True is dangerous on Blackwell. Use bf16=True instead.")

    # VRAM check: seq_length > 2048 on 16GB without DeepSpeed
    max_seq = config.get("max_seq_length", training.get("max_seq_length", 0))
    has_deepspeed = bool(config.get("deepspeed_config") or training.get("deepspeed_config"))
    if max_seq > 2048 and not has_deepspeed:
        errors.append(
            f"max_seq_length={max_seq} will likely OOM on 16GB VRAM without DeepSpeed. "
            "Use max_seq_length<=2048 or add deepspeed_config."
        )

    # GRPO-specific checks
    if method == "grpo":
        beta = training.get("beta", config.get("beta"))
        if beta is not None and beta > 0:
            warnings.append(
                f"GRPO beta={beta} adds KL penalty. On small models, beta=0.0 "
                "with loss_type='dapo' is more stable (TRL 0.29.1+)."
            )

    # lm-eval batch size
    eval_config = config.get("eval", {})
    if eval_config.get("batch_size") == "auto":
        errors.append("lm-eval --batch_size auto OOMs on generate_until tasks. Use --batch_size 4.")

    # torch.compile before LoRA warning
    if config.get("compile") and model.get("use_lora", True):
        warnings.append(
            "torch.compile must apply AFTER LoRA. Ensure compile happens post-PEFT setup "
            "to avoid _orig_mod. prefix mismatch."
        )

    # Per-device batch size sanity
    batch_size = training.get("per_device_train_batch_size", 0)
    if batch_size > 8 and not has_deepspeed:
        warnings.append(
            f"per_device_train_batch_size={batch_size} is large for 16GB VRAM. "
            "Consider reducing or enabling gradient checkpointing."
        )

    # Print results
    prefix = f"[{config_name}] " if config_name else ""
    for w in warnings:
        logger.warning("%s%s", prefix, w)
    for e in errors:
        logger.error("%s%s", prefix, e)

    return errors


def validate_experiment(experiment_dir: Path) -> bool:
    """Validate all configs in an experiment directory.

    Returns:
        True if all configs are valid, False if any have errors.
    """
    configs_dir = experiment_dir / "configs"
    if not configs_dir.exists():
        logger.error("No configs/ directory found in %s", experiment_dir)
        return False

    all_errors: list[str] = []

    for config_file in sorted(configs_dir.glob("*.yaml")):
        if config_file.name.startswith("_"):
            continue
        logger.info("Validating: %s", config_file.name)
        config = _load_config(config_file)
        errors = validate_config(config, config_name=config_file.name)
        all_errors.extend(errors)

    if all_errors:
        logger.error("Config validation FAILED — %d error(s)", len(all_errors))
        return False

    logger.info("Config validation PASSED")
    return True


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Validate experiment configs")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to experiment directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    valid = validate_experiment(args.experiment_dir)
    if not valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
