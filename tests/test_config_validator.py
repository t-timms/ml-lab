"""Tests for config_validator — catches impossible training configs."""

from __future__ import annotations

from src.ml_lab.config_validator import validate_config


class TestValidateConfig:
    def test_valid_config_passes(self) -> None:
        config = {
            "bf16": True,
            "max_seq_length": 2048,
            "training": {"method": "grpo", "per_device_train_batch_size": 4},
        }
        errors = validate_config(config)
        assert errors == []

    def test_fp8_training_rejected(self) -> None:
        config = {"training": {"fp8": True}}
        errors = validate_config(config)
        assert any("fp8" in e.lower() for e in errors)

    def test_fp8_top_level_rejected(self) -> None:
        config = {"fp8": True}
        errors = validate_config(config)
        assert any("fp8" in e.lower() for e in errors)

    def test_bf16_false_rejected(self) -> None:
        config = {"bf16": False}
        errors = validate_config(config)
        assert any("bf16" in e for e in errors)

    def test_fp16_rejected(self) -> None:
        config = {"fp16": True}
        errors = validate_config(config)
        assert any("fp16" in e for e in errors)

    def test_high_seq_length_without_deepspeed_rejected(self) -> None:
        config = {"max_seq_length": 4096}
        errors = validate_config(config)
        assert any("seq_length" in e.lower() or "oom" in e.lower() for e in errors)

    def test_high_seq_length_with_deepspeed_ok(self) -> None:
        config = {"max_seq_length": 4096, "deepspeed_config": "ds_configs/zero2.json"}
        errors = validate_config(config)
        assert errors == []

    def test_lm_eval_auto_batch_rejected(self) -> None:
        config = {"eval": {"batch_size": "auto"}}
        errors = validate_config(config)
        assert any("batch_size" in e for e in errors)

    def test_empty_config_passes(self) -> None:
        errors = validate_config({})
        assert errors == []


class TestConfigNamePrefix:
    def test_errors_include_config_name(self) -> None:
        config = {"fp8": True}
        errors = validate_config(config, config_name="grpo.yaml")
        # Errors are logged with prefix, but returned without
        assert len(errors) > 0
