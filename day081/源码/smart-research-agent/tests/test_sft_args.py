"""SFT 超参与派生量测试（day050）：步数算术、学习率调度、HF 参数映射.

本文件里最关键的断言是 **day049 预先算出的那六个数字**：
``micro=15 / steps/epoch=3 / total_steps=9 / warmup_steps=0 / effective_batch=8``。
"先算再跑"能不能成立，就看这里。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.sft.args import (
    IMPLEMENTED_SCHEDULERS,
    LR_SOFT_RANGE,
    LR_SOFT_RANGE_PEFT,
    LR_SOFT_RANGE_REFERENCE,
    MIN_MEANINGFUL_STEPS,
    SCHEDULER_TYPES,
    SFTConfigError,
    SFTTrainingArgs,
    TrainingPlan,
    lr_curve,
    plan_training,
)

#: day049 实测的课程数据集规模（37 条 → train 30 / eval 7）
TRAIN_SIZE, EVAL_SIZE = 30, 7


def day049_args(**overrides) -> SFTTrainingArgs:
    """day049 用来算预算的那组超参."""
    base = SFTTrainingArgs(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3.0,
    )
    return base.with_overrides(**overrides) if overrides else base


class TestConstants:
    """常量与区间：它们是"告警阈值必须与模型规模匹配"的代码表达."""

    def test_scheduler_types_cover_implemented(self):
        assert set(IMPLEMENTED_SCHEDULERS) <= set(SCHEDULER_TYPES)

    def test_implemented_schedulers(self):
        assert IMPLEMENTED_SCHEDULERS == ("linear", "cosine", "constant", "constant_with_warmup")

    def test_lr_ranges_are_ordered(self):
        low, high = LR_SOFT_RANGE
        assert low < high
        low_peft, high_peft = LR_SOFT_RANGE_PEFT
        assert low_peft < high_peft

    def test_reference_range_is_four_orders_larger(self):
        """参考模型的合理学习率比 7B 全参大约 4 个数量级——这不是笔误."""
        assert LR_SOFT_RANGE_REFERENCE[0] / LR_SOFT_RANGE[1] > 1000

    def test_min_meaningful_steps(self):
        assert MIN_MEANINGFUL_STEPS == 20


class TestValidate:
    """硬校验：非法超参必须在训练开始前被拒绝."""

    def test_defaults_are_valid(self):
        SFTTrainingArgs().validate()

    @pytest.mark.parametrize(
        "changes, match",
        [
            ({"output_dir": "  "}, "output_dir 不能为空"),
            ({"num_train_epochs": 0}, "num_train_epochs 必须为正数"),
            ({"num_train_epochs": -1.0}, "num_train_epochs 必须为正数"),
            ({"per_device_train_batch_size": 0}, "per_device_train_batch_size 必须为正整数"),
            ({"per_device_eval_batch_size": 0}, "per_device_eval_batch_size 必须为正整数"),
            ({"gradient_accumulation_steps": 0}, "gradient_accumulation_steps 必须为正整数"),
            ({"learning_rate": 0.0}, "learning_rate 必须为正数"),
            ({"weight_decay": -0.1}, "weight_decay 不能为负数"),
            ({"warmup_ratio": 1.0}, r"warmup_ratio 必须落在 \[0, 1\)"),
            ({"warmup_ratio": -0.1}, r"warmup_ratio 必须落在 \[0, 1\)"),
            ({"max_grad_norm": -1.0}, "max_grad_norm 不能为负数"),
            ({"max_length": 1}, "max_length 至少为 2"),
            ({"lr_scheduler_type": "warp"}, "未知的 lr_scheduler_type"),
            ({"eval_strategy": "sometimes"}, "未知的 eval_strategy"),
            ({"save_strategy": "sometimes"}, "未知的 save_strategy"),
            ({"logging_steps": 0}, "logging_steps 必须为正整数"),
            ({"eval_steps": 0}, "eval_steps / save_steps 必须为正整数"),
            ({"save_steps": 0}, "eval_steps / save_steps 必须为正整数"),
            ({"save_total_limit": 0}, "save_total_limit 为 None 或正整数"),
            ({"truncation": "middle"}, "未知的截断策略"),
        ],
    )
    def test_invalid_fields_rejected(self, changes, match):
        with pytest.raises(SFTConfigError, match=match):
            SFTTrainingArgs(**changes).validate()

    def test_bf16_and_fp16_are_mutually_exclusive(self):
        with pytest.raises(SFTConfigError, match="bf16 与 fp16 不能同时启用"):
            SFTTrainingArgs(bf16=True, fp16=True).validate()

    def test_save_total_limit_none_allowed(self):
        SFTTrainingArgs(save_total_limit=None).validate()

    def test_warmup_ratio_zero_allowed(self):
        SFTTrainingArgs(warmup_ratio=0.0).validate()


class TestDerivedArithmetic:
    """派生量算术：与 day049 预先算出的数字逐项对齐."""

    def test_effective_batch_size(self):
        assert day049_args().effective_batch_size == 8

    def test_micro_batches_per_epoch(self):
        assert day049_args().micro_batches_per_epoch(TRAIN_SIZE) == 15

    def test_steps_per_epoch_is_drop_last(self):
        assert day049_args().steps_per_epoch(TRAIN_SIZE) == 3

    def test_total_steps(self):
        assert day049_args().total_steps(TRAIN_SIZE) == 9

    def test_warmup_steps_degenerates_to_zero(self):
        """day049 的关键发现：3% 的 warmup 在 9 步下必然取整为 0."""
        assert day049_args().warmup_steps(TRAIN_SIZE) == 0

    def test_warmup_floor_is_reciprocal_of_total_steps(self):
        args = day049_args()
        total = args.total_steps(TRAIN_SIZE)
        assert total == 9
        assert int(total * (1 / total)) == 1  # 比例 = 1/total 时刚好换回一步
        assert int(total * args.warmup_ratio) == 0

    def test_total_steps_floor_is_one(self):
        """数据少到切不出一个累积窗口时，总步数下界仍是 1（不返回 0）."""
        args = SFTTrainingArgs(per_device_train_batch_size=10, gradient_accumulation_steps=10)
        assert args.total_steps(3) == 1

    def test_drop_last_discards_tail(self):
        """31 条 / batch 2 → 15 个 micro-batch，最后 1 条被丢弃."""
        args = day049_args()
        assert args.micro_batches_per_epoch(31) == 15

    def test_large_dataset_scales_linearly(self):
        args = day049_args()
        assert args.total_steps(3000) == 3 * (3000 // 2 // 4)


class TestLearningRateSchedule:
    """学习率调度：与 HF ``get_*_schedule_with_warmup`` 的分段定义一致."""

    def test_cosine_reaches_full_lr_at_zero_without_warmup(self):
        args = SFTTrainingArgs(lr_scheduler_type="cosine", warmup_ratio=0.0)
        assert args.lr_at(0, 10) == pytest.approx(args.learning_rate)

    def test_cosine_decays_to_zero_at_end(self):
        args = SFTTrainingArgs(lr_scheduler_type="cosine", warmup_ratio=0.0)
        assert args.lr_at(10, 10) == pytest.approx(0.0, abs=1e-12)

    def test_cosine_half_point(self):
        args = SFTTrainingArgs(lr_scheduler_type="cosine", warmup_ratio=0.0)
        expected = args.learning_rate * 0.5 * (1 + math.cos(math.pi * 0.5))
        assert args.lr_at(5, 10) == pytest.approx(expected)

    def test_warmup_is_linear_and_first_step_is_zero(self):
        """warmup 段的第 0 步学习率为 0（``LambdaLR`` 构造时即求值一次）."""
        args = SFTTrainingArgs(lr_scheduler_type="cosine", warmup_ratio=0.2)
        total = 10
        warmup = int(total * 0.2)
        assert warmup == 2
        assert args.lr_at(0, total) == 0.0
        assert args.lr_at(1, total) == pytest.approx(args.learning_rate * 0.5)
        assert args.lr_at(2, total) == pytest.approx(args.learning_rate)

    def test_linear_decays_linearly(self):
        args = SFTTrainingArgs(lr_scheduler_type="linear", warmup_ratio=0.0)
        assert args.lr_at(0, 10) == pytest.approx(args.learning_rate)
        assert args.lr_at(5, 10) == pytest.approx(args.learning_rate * 0.5)
        assert args.lr_at(10, 10) == pytest.approx(0.0)

    def test_constant_holds_plateau(self):
        args = SFTTrainingArgs(lr_scheduler_type="constant", warmup_ratio=0.0)
        assert args.lr_at(0, 10) == pytest.approx(args.learning_rate)
        assert args.lr_at(7, 10) == pytest.approx(args.learning_rate)

    def test_constant_with_warmup_rises_then_holds(self):
        args = SFTTrainingArgs(lr_scheduler_type="constant_with_warmup", warmup_ratio=0.2)
        assert args.lr_at(0, 10) == 0.0
        assert args.lr_at(2, 10) == pytest.approx(args.learning_rate)
        assert args.lr_at(9, 10) == pytest.approx(args.learning_rate)

    def test_step_beyond_total_is_clamped(self):
        args = SFTTrainingArgs(lr_scheduler_type="cosine", warmup_ratio=0.0)
        assert args.lr_at(999, 10) == pytest.approx(args.lr_at(10, 10))

    def test_negative_step_is_clamped_to_zero(self):
        args = SFTTrainingArgs(lr_scheduler_type="cosine", warmup_ratio=0.0)
        assert args.lr_at(-5, 10) == pytest.approx(args.lr_at(0, 10))

    def test_unimplemented_scheduler_rejected(self):
        args = SFTTrainingArgs(lr_scheduler_type="polynomial")
        with pytest.raises(SFTConfigError, match="未覆盖调度器"):
            args.lr_at(1, 10)

    def test_zero_total_steps_rejected(self):
        with pytest.raises(SFTConfigError, match="total_steps 必须为正整数"):
            SFTTrainingArgs().lr_at(0, 0)

    def test_warmup_from_steps_matches_warmup_steps(self):
        args = day049_args()
        total = args.total_steps(TRAIN_SIZE)
        assert args.warmup_from_steps(total) == args.warmup_steps(TRAIN_SIZE)


class TestHuggingFaceMapping:
    """HF 参数映射：字段名必须逐字对齐，否则无法直接展开成构造参数."""

    def test_to_hf_dict_keys(self):
        payload = SFTTrainingArgs().to_hf_dict()
        for key in (
            "output_dir",
            "num_train_epochs",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "warmup_ratio",
            "lr_scheduler_type",
            "max_grad_norm",
            "eval_strategy",
            "save_strategy",
            "bf16",
            "fp16",
            "seed",
            "remove_unused_columns",
        ):
            assert key in payload

    def test_to_hf_dict_uses_eval_strategy_not_deprecated_name(self):
        """``evaluation_strategy`` 已废弃并在 v4.46 移除，必须用新名."""
        payload = SFTTrainingArgs().to_hf_dict()
        assert "eval_strategy" in payload
        assert "evaluation_strategy" not in payload

    def test_to_hf_dict_excludes_sft_specific_fields(self):
        """``max_length`` / ``truncation`` 不属于 TrainingArguments，不能传进去."""
        payload = SFTTrainingArgs().to_hf_dict()
        assert "max_length" not in payload
        assert "truncation" not in payload

    def test_to_trl_dict_adds_sft_switches(self):
        payload = SFTTrainingArgs().to_trl_dict()
        assert payload["max_length"] == 384
        assert payload["packing"] is False
        assert payload["assistant_only_loss"] is True
        assert payload["completion_only_loss"] is True

    def test_to_dict_covers_every_field(self):
        args = SFTTrainingArgs()
        payload = args.to_dict()
        assert payload["max_length"] == args.max_length
        assert payload["truncation"] == args.truncation
        assert len(payload) >= 25

    def test_from_dict_roundtrip(self):
        args = day049_args(max_length=128, bf16=True)
        assert SFTTrainingArgs.from_dict(args.to_dict()) == args

    def test_from_dict_ignores_unknown_keys(self):
        restored = SFTTrainingArgs.from_dict({"max_length": 96, "future_field": 1})
        assert restored.max_length == 96

    def test_with_overrides_does_not_mutate(self):
        args = SFTTrainingArgs()
        changed = args.with_overrides(learning_rate=1.0)
        assert args.learning_rate != 1.0
        assert changed.learning_rate == 1.0


class TestPlanTraining:
    """``plan_training``：派生量 + 把风险写成 warnings."""

    def test_day049_numbers(self):
        plan = plan_training(day049_args(), train_size=TRAIN_SIZE, eval_size=EVAL_SIZE)
        assert plan.micro_batches_per_epoch == 15
        assert plan.steps_per_epoch == 3
        assert plan.total_steps == 9
        assert plan.steps_per_epoch_flushing == 4
        assert plan.total_steps_flushing == 12
        assert plan.warmup_steps == 0
        assert plan.effective_batch_size == 8

    def test_flushing_variant_reported(self):
        """HF Trainer 会 flush 尾部梯度，两种口径必须同时可见."""
        plan = plan_training(day049_args(), train_size=TRAIN_SIZE, eval_size=EVAL_SIZE)
        assert plan.total_steps_flushing > plan.total_steps

    def test_warnings_mention_step_count_and_warmup(self):
        plan = plan_training(day049_args(), train_size=TRAIN_SIZE, eval_size=EVAL_SIZE)
        joined = " ".join(plan.warnings)
        assert "总优化器步数仅 9 步" in joined
        assert "warmup" in joined and "取整为 0" in joined
        assert "训练样本仅 30 条" in joined

    def test_no_eval_set_warning(self):
        plan = plan_training(day049_args(), train_size=TRAIN_SIZE)
        assert any("未提供评估集" in message for message in plan.warnings)

    def test_eval_and_save_beyond_total_warn(self):
        args = day049_args(
            eval_strategy="steps", eval_steps=50, save_strategy="steps", save_steps=50
        )
        plan = plan_training(args, train_size=TRAIN_SIZE, eval_size=EVAL_SIZE)
        joined = " ".join(plan.warnings)
        assert "eval_steps=50 大于 total_steps=9" in joined
        assert "save_steps=50 大于 total_steps=9" in joined

    def test_reachable_eval_and_save_do_not_warn(self):
        args = day049_args(eval_strategy="steps", eval_steps=3, save_strategy="steps", save_steps=3)
        plan = plan_training(args, train_size=TRAIN_SIZE, eval_size=EVAL_SIZE)
        joined = " ".join(plan.warnings)
        assert "eval_steps=3 大于" not in joined
        assert "save_steps=3 大于" not in joined

    def test_zero_steps_per_epoch_warns(self):
        """累积窗口比数据还大：drop_last 口径下一次更新都做不出来."""
        args = SFTTrainingArgs(
            per_device_train_batch_size=2, gradient_accumulation_steps=16, num_train_epochs=1.0
        )
        plan = plan_training(args, train_size=4, eval_size=2)
        assert plan.steps_per_epoch == 0
        assert any("0 次参数更新" in message for message in plan.warnings)

    def test_reference_lr_inside_reference_range_does_not_warn(self):
        plan = plan_training(
            day049_args(learning_rate=8.0), train_size=TRAIN_SIZE, eval_size=EVAL_SIZE,
            lr_range=LR_SOFT_RANGE_REFERENCE,
        )
        assert not any("learning_rate" in message for message in plan.warnings)

    def test_reference_lr_outside_default_range_warns(self):
        """同一组超参在 7B 的区间（缺省）里就越界——区间必须按模型规模选."""
        plan = plan_training(
            day049_args(learning_rate=8.0), train_size=TRAIN_SIZE, eval_size=EVAL_SIZE
        )
        assert any("learning_rate" in message for message in plan.warnings)

    def test_max_length_below_dataset_max_warns(self):
        args = day049_args(max_length=64)
        plan = plan_training(
            args, train_size=TRAIN_SIZE, eval_size=EVAL_SIZE,
            dataset_avg_tokens=250.9, dataset_max_tokens=320.0,
        )
        assert any("小于渲染后最长样本" in message for message in plan.warnings)

    def test_max_length_too_large_warns_about_padding(self):
        args = day049_args(max_length=4096)
        plan = plan_training(
            args, train_size=TRAIN_SIZE, eval_size=EVAL_SIZE,
            dataset_avg_tokens=250.9, dataset_max_tokens=320.0,
        )
        assert any("远大于数据平均长度" in message for message in plan.warnings)

    def test_max_length_matching_data_does_not_warn(self):
        args = day049_args(max_length=320)
        plan = plan_training(
            args, train_size=TRAIN_SIZE, eval_size=EVAL_SIZE,
            dataset_avg_tokens=250.9, dataset_max_tokens=320.0,
        )
        assert not any("max_length" in message for message in plan.warnings)

    def test_logging_and_interval_counts(self):
        args = day049_args(eval_strategy="steps", eval_steps=3, save_strategy="steps", save_steps=3)
        plan = plan_training(args, train_size=TRAIN_SIZE, eval_size=EVAL_SIZE)
        assert plan.epoch_eval_times == 1
        assert plan.epoch_save_times == 1

    def test_large_dataset_avoids_small_data_warnings(self):
        """数据够多时，"先补数据"与"两种口径步数不同"两条告警都不该出现."""
        plan = plan_training(
            day049_args(learning_rate=8.0),
            train_size=3000,
            eval_size=150,
            lr_range=LR_SOFT_RANGE_REFERENCE,
        )
        joined = " ".join(plan.warnings)
        assert "500 条以下" not in joined
        assert "两种口径的步数不同" not in joined
        assert plan.steps_per_epoch == plan.steps_per_epoch_flushing == 375
        assert plan.total_steps == 1125
        assert plan.warmup_steps == 33

    def test_plan_to_dict_is_json_ready(self):
        import json

        plan = plan_training(day049_args(), train_size=TRAIN_SIZE, eval_size=EVAL_SIZE)
        payload = plan.to_dict()
        assert json.loads(json.dumps(payload, ensure_ascii=False))["total_steps"] == 9

    def test_invalid_sizes_rejected(self):
        with pytest.raises(SFTConfigError, match="train_size 必须为正整数"):
            plan_training(SFTTrainingArgs(), train_size=0)
        with pytest.raises(SFTConfigError, match="eval_size 不能为负数"):
            plan_training(SFTTrainingArgs(), train_size=10, eval_size=-1)

    def test_plan_is_a_training_plan(self):
        plan = plan_training(day049_args(), train_size=TRAIN_SIZE, eval_size=EVAL_SIZE)
        assert isinstance(plan, TrainingPlan)


class TestLrCurve:
    """学习率曲线采样：demo 与测试都用它判断"调度器接对了没有"."""

    def test_points_must_be_at_least_two(self):
        with pytest.raises(SFTConfigError, match="points 至少为 2"):
            lr_curve(SFTTrainingArgs(), TRAIN_SIZE, points=1)

    def test_curve_covers_both_ends(self):
        args = SFTTrainingArgs(lr_scheduler_type="cosine", warmup_ratio=0.0)
        curve = lr_curve(args, TRAIN_SIZE, points=6)
        assert curve[0][0] == 0
        assert curve[-1][0] == args.total_steps(TRAIN_SIZE)

    def test_cosine_curve_is_monotonically_decreasing_without_warmup(self):
        args = SFTTrainingArgs(lr_scheduler_type="cosine", warmup_ratio=0.0)
        values = [value for _, value in lr_curve(args, 3000, points=8)]
        assert all(later <= earlier + 1e-12 for earlier, later in zip(values, values[1:]))

    def test_curve_with_warmup_rises_first(self):
        args = SFTTrainingArgs(lr_scheduler_type="cosine", warmup_ratio=0.2)
        curve = lr_curve(args, 3000, points=4)
        assert curve[0][1] == 0.0  # step 0 的学习率是 0
        assert curve[1][1] > curve[0][1]
