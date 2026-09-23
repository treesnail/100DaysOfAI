"""day055 DPO 训练配置测试：构造期校验、步数算术与 TRL 投影（全部离线）.

本文件的重点是**算术**：`step_plan()` 在开训前就把"会做多少次参数更新、
什么时候评估"算准，并且把三种"看似能跑、其实一次更新都不会发生"的配置
在构造/计划期就拦下来。
"""

from __future__ import annotations

import pytest

from smart_research_agent.dpo import (
    DEFAULT_ACCUMULATION_STEPS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_STEPS,
    DEFAULT_LOSS_TYPE,
    DEFAULT_PATIENCE,
    DEFAULT_SAVE_STEPS,
    DEFAULT_WARMUP_RATIO,
    LOSS_TYPES,
    SNAPSHOT,
    DPOError,
    DPOTrainingConfig,
    warmup_floor,
)


class TestDefaults:
    def test_defaults_are_valid_and_documented(self):
        cfg = DPOTrainingConfig()
        assert cfg.loss_type == DEFAULT_LOSS_TYPE
        assert cfg.num_train_epochs == DEFAULT_EPOCHS
        assert cfg.per_device_train_batch_size == DEFAULT_BATCH_SIZE
        assert cfg.gradient_accumulation_steps == DEFAULT_ACCUMULATION_STEPS
        assert cfg.warmup_ratio == DEFAULT_WARMUP_RATIO
        assert cfg.save_steps == DEFAULT_SAVE_STEPS
        assert cfg.eval_steps == DEFAULT_EVAL_STEPS
        assert cfg.early_stopping_patience == DEFAULT_PATIENCE

    def test_snapshot_label_is_day055(self):
        assert SNAPSHOT == "day055"

    def test_batch_and_accumulation_use_pairs_as_unit(self):
        """批大小的单位是"偏好对"，不是样本条数——注释里必须说清，这里钉住。"""
        cfg = DPOTrainingConfig()
        assert cfg.effective_batch_size == 4  # 2 × 2

    def test_complete_epochs_truncates_fraction(self):
        assert DPOTrainingConfig(num_train_epochs=3.9).complete_epochs == 3


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"beta": 0.0}, "beta 必须为正数"),
            ({"beta": -0.1}, "beta 必须为正数"),
            ({"loss_type": "apo_zero"}, "未知的损失函数"),
            ({"learning_rate": 0.0}, "learning_rate 必须为正数"),
            ({"offline_learning_rate": -1.0}, "offline_learning_rate 必须为正数"),
            ({"num_train_epochs": 0.0}, "num_train_epochs 必须为正数"),
            ({"per_device_train_batch_size": 0}, "per_device_train_batch_size 必须为正整数"),
            ({"gradient_accumulation_steps": -1}, "gradient_accumulation_steps 必须为正整数"),
            ({"warmup_ratio": 1.0}, "warmup_ratio 必须落在"),
            ({"warmup_ratio": -0.1}, "warmup_ratio 必须落在"),
            ({"label_smoothing": 1.0}, "label_smoothing 必须落在"),
            ({"max_length": 0}, "max_length 必须为正整数"),
            ({"save_steps": 0}, "save_steps 必须为正整数"),
            ({"eval_steps": -2}, "eval_steps 必须为正整数"),
            ({"logging_steps": 0}, "logging_steps 必须为正整数"),
            ({"early_stopping_patience": 0}, "early_stopping_patience 至少为 1"),
            ({"output_dir": "   "}, "output_dir 不能为空"),
        ],
    )
    def test_invalid_field_rejected_at_construction(self, kwargs, message):
        with pytest.raises(DPOError, match=message):
            DPOTrainingConfig(**kwargs)

    def test_prompt_plus_completion_must_fit_max_length(self):
        """越过 max_length 不会报错，只会让每批被悄悄截断——必须提前拦。"""
        with pytest.raises(DPOError, match="超过"):
            DPOTrainingConfig(max_length=512, max_prompt_length=400, max_completion_length=200)

    def test_exact_fit_is_allowed(self):
        cfg = DPOTrainingConfig(
            max_length=512, max_prompt_length=256, max_completion_length=256
        )
        assert cfg.max_length == 512

    def test_error_is_value_error_subclass(self):
        with pytest.raises(ValueError):
            DPOTrainingConfig(beta=0.0)


class TestStepPlan:
    def test_seven_pairs_default_config(self):
        """7 条对、batch 2、累积 2、3 epoch → 每 epoch 3 个 micro → 1 步/epoch → 3 步。"""
        plan = DPOTrainingConfig().step_plan(7)
        assert plan["micro_batches_per_epoch"] == 3
        assert plan["optimizer_steps_per_epoch"] == 1
        assert plan["complete_epochs"] == 3
        assert plan["total_steps"] == 3
        assert plan["effective_batch_size"] == 4
        assert plan["pairs_dropped_per_epoch"] == 1  # 7 条里有 1 条凑不满一个 micro-batch

    def test_default_warmup_degenerates_to_zero(self):
        """3% 的 warmup 在 9 步以下是 0 步——这正是 ``warmup_floor`` 要回答的问题。"""
        plan = DPOTrainingConfig().step_plan(7)  # 3 步，3% → int(0.09) = 0
        assert plan["total_steps"] == 3
        assert plan["warmup_degenerate"] is True
        assert plan["warmup_steps"] == 0

    def test_warmup_ratio_large_enough_avoids_degenerate(self):
        cfg = DPOTrainingConfig(warmup_ratio=0.5)
        plan = cfg.step_plan(60)  # 45 步，50% → 22 步
        assert plan["warmup_steps"] > 0
        assert plan["warmup_degenerate"] is False

    def test_evaluations_never_below_one(self):
        plan = DPOTrainingConfig().step_plan(7)
        assert plan["evaluations"] >= 1

    def test_pairs_consumed_is_multiple_of_batch(self):
        plan = DPOTrainingConfig().step_plan(7)
        assert plan["pairs_consumed"] % DEFAULT_BATCH_SIZE == 0

    def test_non_positive_pairs_rejected(self):
        with pytest.raises(DPOError, match="train_pairs 必须为正整数"):
            DPOTrainingConfig().step_plan(0)

    def test_fewer_pairs_than_one_batch_rejected(self):
        with pytest.raises(DPOError, match="不足一个 micro-batch"):
            DPOTrainingConfig().step_plan(1)

    def test_accumulation_larger_than_micro_batches_rejected(self):
        with pytest.raises(DPOError, match="凑不满"):
            DPOTrainingConfig().step_plan(3)

    def test_fractional_epochs_produce_no_extra_updates(self):
        cfg = DPOTrainingConfig(num_train_epochs=2.9)
        plan = cfg.step_plan(7)
        assert plan["complete_epochs"] == 2
        assert plan["total_steps"] == 2


class TestWarmupFloor:
    def test_floor_is_reciprocal_of_steps(self):
        assert warmup_floor(9) == pytest.approx(1.0 / 9.0)

    def test_floor_makes_warmup_non_degenerate(self):
        """把 warmup_ratio 设成下限后，warmup 步数至少是 1。"""
        cfg = DPOTrainingConfig()
        plan = cfg.step_plan(7)
        floor = warmup_floor(plan["total_steps"])
        raised = DPOTrainingConfig(warmup_ratio=floor)
        assert raised.step_plan(7)["warmup_steps"] >= 1

    def test_non_positive_steps_rejected(self):
        with pytest.raises(DPOError, match="total_steps 必须为正整数"):
            warmup_floor(0)


class TestProjection:
    def test_to_dpo_config_keys_align_with_trl(self):
        payload = DPOTrainingConfig().to_dpo_config()
        for key in (
            "output_dir",
            "beta",
            "loss_type",
            "learning_rate",
            "num_train_epochs",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "warmup_ratio",
            "max_length",
            "max_prompt_length",
            "max_completion_length",
            "save_steps",
            "eval_steps",
            "logging_steps",
            "label_smoothing",
            "seed",
        ):
            assert key in payload

    def test_fixed_values_are_pinned(self):
        payload = DPOTrainingConfig().to_dpo_config()
        assert payload["gradient_checkpointing"] is True
        assert payload["bf16"] is True
        assert payload["report_to"] == "none"

    def test_to_lora_config_rejected_when_disabled(self):
        with pytest.raises(DPOError, match="use_lora=False"):
            DPOTrainingConfig(use_lora=False).to_lora_config()

    def test_to_lora_config_delegates_to_day051(self):
        payload = DPOTrainingConfig().to_lora_config()
        assert isinstance(payload, dict) and payload

    def test_to_dict_carries_snapshot_and_derived_values(self):
        payload = DPOTrainingConfig().to_dict()
        assert payload["snapshot"] == "day055"
        assert payload["loss_types"] == list(LOSS_TYPES)
        assert payload["effective_batch_size"] == 4
        assert payload["zero_margin_loss"] == pytest.approx(0.693147, abs=1e-6)
        assert "dpo_config" in payload

    def test_plan_bundles_table_and_steps(self):
        plan = DPOTrainingConfig().plan(7)
        assert plan["snapshot"] == "day055"
        assert len(plan["loss_table"]) == len(LOSS_TYPES)
        assert set(plan["zero_margin_losses"]) == set(LOSS_TYPES)
        assert plan["steps"]["total_steps"] == 3

    def test_summary_line_is_human_readable(self):
        line = DPOTrainingConfig().summary_line()
        assert "DPO 配置" in line
        assert "sigmoid" in line
        assert "有效批 4" in line
