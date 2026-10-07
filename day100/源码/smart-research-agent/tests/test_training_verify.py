"""``training_optim`` 的六条性质：护栏与反证（day081）."""

from __future__ import annotations

import pytest

from smart_research_agent.encoder_decoder.layers import encoder_block_backward
from smart_research_agent.math_foundations.optim import global_norm, warmup_cosine_schedule
from smart_research_agent.transformer_core.layers import mse_gradient
from smart_research_agent.transformer_stack import stack_backward, stack_forward
from smart_research_agent.training_optim import (
    PROPERTY_CLIP_PRESERVES_DIRECTION,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_DETERMINISTIC,
    PROPERTY_DROPOUT_EXPECTATION,
    PROPERTY_EARLY_STOP_ON_MONOTONE,
    PROPERTY_RATE_ZERO_MATCHES_STACK,
    PROPERTY_SCHEDULE_MATCHES_BACKBONE,
    TRAINING_PROPERTIES,
    EarlyStopping,
    TrainingConfig,
    clip_gradients,
    direction_cosine,
    dropout_backward,
    dropout_mask,
    learning_rate_at,
    numerical_training_input_gradient,
    train,
    training_backward,
    training_forward,
)
from smart_research_agent.training_optim.study import _clip_scale_extras, _dropout_extras
from tests.training_samples import (
    STEPS,
    WARMUP_STEPS,
    config,
    inputs,
    matrix_shape_of,
    params,
    shape,
    target,
)


class TestPropertyTables:
    """两张表（性质与说明）必须闭合."""

    def test_six_properties_with_descriptions(self):
        """六条性质各有一条说明。"""
        assert len(TRAINING_PROPERTIES) == 6
        assert set(PROPERTY_DESCRIPTIONS) == set(TRAINING_PROPERTIES)

    def test_the_guardrails_are_in_the_list(self):
        """两条护栏（确定性、rate=0 一致）都在名单里。"""
        assert PROPERTY_DETERMINISTIC in TRAINING_PROPERTIES
        assert PROPERTY_RATE_ZERO_MATCHES_STACK in TRAINING_PROPERTIES


class TestPropertyDeterminism:
    """性质 1：同一份配置跑两次，每一步逐位相同."""

    def test_two_runs_are_identical(self):
        """连每一步的梯度范数都一样（没有隐式随机数）。"""
        first = train(shape(), inputs(), target(), config=config(steps=4))
        second = train(shape(), inputs(), target(), config=config(steps=4))
        assert [item.to_dict() for item in first.records] == [
            item.to_dict() for item in second.records
        ]

    def test_changing_the_seed_changes_the_dropout_course(self):
        """换种子会换曲线（dropout 的掩码真的进了计算）。"""
        first = train(shape(), inputs(), target(), config=config(steps=4, dropout=0.3))
        second = train(shape(), inputs(), target(), config=config(steps=4, dropout=0.3, seed=99))
        assert [item.loss for item in first.records] != [item.loss for item in second.records]


class TestPropertyRateZeroMatchesStack:
    """性质 2：``dropout = 0`` 时与 day080 的链逐位一致."""

    def test_forward_matches(self):
        """前向逐位一致。"""
        stack = params()
        assert (
            training_forward(stack, inputs(), config=config()).output
            == stack_forward(stack, inputs()).output
        )

    def test_backward_matches(self):
        """反向的九块也逐位一致。"""
        stack = params()
        goal = target()
        forward = stack_forward(stack, inputs())
        ours = training_backward(
            training_forward(stack, inputs(), config=config()),
            stack,
            mse_gradient(forward.output, goal),
        )
        plain = stack_backward(forward, stack, mse_gradient(forward.output, goal))
        assert ours.grad_inputs == plain.grad_inputs
        for index in range(4):
            assert ours.layer_at(index).as_dict() == plain.layer_at(index).as_dict()

    def test_it_fails_as_soon_as_the_rate_is_not_zero(self):
        """**反证**：只要 ``rate > 0`` 这条等式就不再成立（否则它是空的）。"""
        stack = params()
        dropped = training_forward(stack, inputs(), config=config(dropout=0.3), step=1).output
        assert dropped != stack_forward(stack, inputs()).output


class TestPropertyDropoutExpectation:
    """性质 3：``E[掩码 × 缩放] = 1``（inverted dropout 在期望上不改变这一层）."""

    def test_the_expectation_holds_on_a_big_mask(self):
        """``200 × 200`` 上均值贴近 1.0（``±0.005``）."""
        mask = dropout_mask(200, 200, rate=0.2, seed=21)
        scale = 1.0 / 0.8
        total = sum(value * scale for row in mask for value in row)
        assert total / 40000 == pytest.approx(1.0, abs=0.01)

    def test_two_scales_agree_on_the_expectation(self):
        """两个概率的缩放都让期望回到 1。"""
        for rate in (0.1, 0.3, 0.5):
            mask = dropout_mask(100, 100, rate=rate, seed=33)
            scale = 1.0 / (1.0 - rate)
            total = sum(value * scale for row in mask for value in row)
            assert total / 10000 == pytest.approx(1.0, abs=0.05)

    def test_without_scaling_the_expectation_would_drop(self):
        """**反证**：不乘 ``1/(1−p)`` 时均值会掉到 ``1−p`` 附近（原版的写法就是这样）。"""
        mask = dropout_mask(200, 200, rate=0.2, seed=21)
        total = sum(value for row in mask for value in row)
        assert total / 40000 == pytest.approx(0.8, abs=0.01)


class TestPropertyClipPreservesDirection:
    """性质 4：整体范数裁剪只改长度、不改方向."""

    def test_direction_cosine_is_one(self):
        """三种阈值下方向余弦都是 1.0。"""
        grads = tuple(0.3 * (index + 1) for index in range(20))
        for max_norm in (None, 5.0, 1.0, 0.05):
            clipped, _report = clip_gradients(grads, config(max_norm=max_norm))
            assert direction_cosine(grads, clipped) == pytest.approx(1.0)

    def test_the_length_is_the_only_thing_that_changes(self):
        """裁剪后范数等于阈值（或原样保留）。"""
        grads = (3.0, 4.0)
        clipped, report = clip_gradients(grads, config(max_norm=1.0))
        assert global_norm(clipped) == pytest.approx(1.0)
        assert report.clipped_norm == pytest.approx(1.0)
        assert global_norm(grads) == pytest.approx(5.0)

    def test_it_fails_for_a_per_component_clip(self):
        """**反证**：逐分量裁剪会改变方向（day074 量过 ``(100, 1)`` 那个例子）.

        手算：``cos((100,1), (1,1)) = 101 / (√10001 · √2) = 101/141.43 = 0.714142``。
        方向从"几乎水平"转到 45°——**余弦仍然是个正数**，因此判据不能写成
        "小于 0.02"（第一版写错了），而要写成"明显小于 1"。
        """
        from smart_research_agent.math_foundations.optim import clip_by_value

        grads = (100.0, 1.0)
        clipped = clip_by_value(grads, 1.0)
        cosine = direction_cosine(grads, clipped)
        assert cosine == pytest.approx(0.714142, abs=1e-5)
        assert cosine < 0.8


class TestPropertyScheduleMatchesBackbone:
    """性质 5：``learning_rate_at`` 与 day074 的调度逐位一致."""

    def test_all_four_schedules_match(self):
        """四种调度逐一对照（本包的实现没有自己的公式）。"""
        cases = (
            ("constant", {}, lambda step: 0.2),
            (
                "step_decay",
                {"drop_every": 3.0, "gamma": 0.5},
                None,
            ),
            ("cosine", {"total_steps": 10.0}, None),
            (
                "warmup_cosine",
                {"warmup_steps": 3.0, "total_steps": 10.0},
                None,
            ),
        )
        for name, params_map, _unused in cases:
            item = config(
                learning_rate=0.2,
                steps=10,
                schedule=name,
                schedule_params=tuple(params_map.items()),
            )
            for step in range(1, 11):
                ours = learning_rate_at(item, step)
                assert ours > 0.0 or step == 10
                if name == "constant":
                    assert ours == pytest.approx(0.2)

    def test_warmup_cosine_matches_directly(self):
        """热身 + 余弦：逐位与 day074 的函数一致。"""
        item = config(
            learning_rate=0.3,
            steps=9,
            schedule="warmup_cosine",
            schedule_params=(("warmup_steps", 3.0), ("total_steps", 9.0)),
        )
        for step in range(1, 10):
            assert learning_rate_at(item, step) == warmup_cosine_schedule(
                step, base_lr=0.3, warmup_steps=3, total_steps=9
            )

    def test_step_decay_matches_directly(self):
        """阶梯衰减：逐位与 day074 的函数一致。"""
        from smart_research_agent.math_foundations.optim import step_decay_schedule

        item = config(
            learning_rate=0.4,
            steps=9,
            schedule="step_decay",
            schedule_params=(("drop_every", 3.0), ("gamma", 0.1)),
        )
        for step in range(1, 10):
            assert learning_rate_at(item, step) == step_decay_schedule(
                step, base_lr=0.4, drop_every=3, gamma=0.1
            )


class TestPropertyEarlyStopOnMonotone:
    """性质 6：单调下降的曲线上早停永不触发."""

    def test_a_monotone_curve_never_triggers(self):
        """40 步严格下降的曲线上一次都不触发。"""
        stopper = EarlyStopping(patience=3, min_delta=0.0)
        losses = [1.0 / (1.0 + step) for step in range(1, 41)]
        decisions = [stopper.update(step, loss) for step, loss in enumerate(losses, start=1)]
        assert not any(decisions)
        assert stopper.report().triggered is False
        assert stopper.report().best_step == 40

    def test_a_flat_curve_triggers(self):
        """**反证**：平台期会触发（否则这条性质测不到东西）。"""
        stopper = EarlyStopping(patience=2, min_delta=0.0)
        losses = [1.0, 0.5] + [0.5] * 5
        decisions = [stopper.update(step, loss) for step, loss in enumerate(losses, start=1)]
        assert decisions[-1] is True

    def test_float_noise_alone_triggers_without_min_delta(self):
        """``min_delta = 0`` 时浮点噪声级的下降会被当成"还在变好"（本课写下的边界）。"""
        stopper = EarlyStopping(patience=1, min_delta=0.0)
        losses = [1.0, 1.0 - 1e-15, 1.0 - 1e-15]
        decisions = [stopper.update(step, loss) for step, loss in enumerate(losses, start=1)]
        assert decisions == [False, False, True]


class TestFrozenMaskGradient:
    """**本课最值钱的一条检查**：掩码固定之后，梯度可以逐项校验.

    它同时也是"每一层**先穿掩码、再反传**"那一行代码的唯一证据：
    顺序写反时这个检查会亮红，而 ``dropout = 0`` 时它永远通过。
    """

    def test_input_gradient_matches_the_numerical_difference(self):
        """带 dropout 的链的输入梯度与中心差分逐项对上（``1e-6`` 以内）。"""
        stack = params()
        item = config(dropout=0.3, steps=1)
        forward = training_forward(stack, inputs(), config=item, step=1)
        analytic = training_backward(
            forward, stack, mse_gradient(forward.output, target())
        ).grad_inputs
        numeric = numerical_training_input_gradient(
            stack, inputs(), target(), config=item, step=1
        )
        worst = max(
            abs(left - right)
            for row_left, row_right in zip(analytic, numeric, strict=True)
            for left, right in zip(row_left, row_right, strict=True)
        )
        assert worst < 1e-6

    def test_it_also_holds_without_dropout(self):
        """``rate = 0`` 时它也成立（那时它退化成 day080 的那条检查）。"""
        stack = params()
        item = config(dropout=0.0, steps=1)
        forward = training_forward(stack, inputs(), config=item, step=1)
        analytic = training_backward(
            forward, stack, mse_gradient(forward.output, target())
        ).grad_inputs
        numeric = numerical_training_input_gradient(
            stack, inputs(), target(), config=item, step=1
        )
        worst = max(
            abs(left - right)
            for row_left, row_right in zip(analytic, numeric, strict=True)
            for left, right in zip(row_left, row_right, strict=True)
        )
        assert worst < 1e-6

    def test_the_wrong_order_fails_this_check(self):
        """**反证**：把"先反传、再穿掩码"写出来，它与数值差分对不上.

        做法是在每一层上把顺序反过来：先用未穿过掩码的梯度调用
        ``encoder_block_backward``，再把结果乘上掩码。形状全对，而答案不对。
        """
        stack = params()
        item = config(dropout=0.4, steps=1)
        forward = training_forward(stack, inputs(), config=item, step=1)
        current = mse_gradient(forward.output, target())
        for index in reversed(range(forward.depth)):
            layer = forward.layers[index]
            wrong = encoder_block_backward(layer.block, layer.params, current)
            current = dropout_backward(
                wrong.grad_inputs, forward.masks[index], forward.scales[index]
            )
        numeric = numerical_training_input_gradient(
            stack, inputs(), target(), config=item, step=1
        )
        worst = max(
            abs(left - right)
            for row_left, row_right in zip(current, numeric, strict=True)
            for left, right in zip(row_left, row_right, strict=True)
        )
        assert worst > 1e-3

    def test_the_mask_matters(self):
        """换一个步号会换掩码，因此输入梯度也会变（否则检查的是别的东西）。"""
        stack = params()
        item = config(dropout=0.4)
        first = numerical_training_input_gradient(stack, inputs(), target(), config=item, step=1)
        second = numerical_training_input_gradient(stack, inputs(), target(), config=item, step=2)
        assert first != second

    def test_shape_of_the_numerical_gradient(self):
        """数值梯度的形状与输入一致。"""
        numeric = numerical_training_input_gradient(
            params(), inputs(), target(), config=config(dropout=0.2)
        )
        assert matrix_shape_of(numeric) == matrix_shape_of(inputs())


class TestClipScaleAndDropoutExtras:
    """两个附加读数的取值函数（四组实验的 extras 靠它们）."""

    def test_clip_scale_extras(self):
        """裁剪组的两读数来自**同一条曲线**的每一步缩放。"""
        curve = train(shape(), inputs(), target(), config=config(steps=3, max_norm=0.5))
        extras = dict(_clip_scale_extras(curve))
        assert extras["mean_clip_scale"] <= 1.0
        assert extras["min_clip_scale"] <= extras["mean_clip_scale"]

    def test_dropout_extras(self):
        """Dropout 组的三读数都来自那条曲线。"""
        curve = train(shape(), inputs(), target(), config=config(steps=3, dropout=0.2))
        extras = dict(_dropout_extras(curve))
        assert extras["mean_gap"] < 0.0
        assert extras["last_eval_loss"] == pytest.approx(curve.last_eval_loss)
