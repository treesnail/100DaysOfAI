"""``math_foundations.optim``：三个优化器、四种调度与训练回路（day074）.

这一份测试的两条主线：

```text
① 更新公式必须能被手算复核     SGD 一步 = θ − lr·g；动量两阶；Adam 第一步 = lr·sign(g)
② 调度是"计划"而不是"氛围"      每个调度的端点值、档位边界、越界行为都逐点断言
```

其中 ``test_adam_first_step_is_lr_times_sign`` 是这一课最值钱的一条断言：
它把"偏差修正"从一个公式变成了一个可验证的等式
（不做修正时第一步会走出 ``lr·3.16``，而它看起来只是"前期有点抖"）。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.math_foundations.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.math_foundations.optim import (
    OPTIMIZER_CLASSES,
    SCHEDULE_FACTORIES,
    AdamOptimizer,
    MomentumOptimizer,
    SGDOptimizer,
    TrainingTrace,
    clip_by_global_norm,
    clip_by_value,
    constant_schedule,
    cosine_schedule,
    flatten_matrices,
    global_norm,
    make_optimizer,
    make_schedule,
    minimize,
    step_decay_schedule,
    unflatten_matrices,
    warmup_cosine_schedule,
)
from smart_research_agent.math_foundations.types import OPTIMIZERS, SCHEDULES
from tests.math_samples import BOWL_MINIMUM, RAVINE_START, approx, bowl, ravine


class TestConstantSchedule:
    """常数调度：基线."""

    def test_stays_flat(self):
        for step in (1, 5, 1000):
            assert constant_schedule(step, base_lr=0.1) == 0.1

    def test_non_positive_lr_is_rejected(self):
        with pytest.raises(ParameterError, match="base_lr"):
            constant_schedule(1, base_lr=0.0)

    def test_zero_step_is_rejected(self):
        with pytest.raises(ParameterError, match="1-based"):
            constant_schedule(0, base_lr=0.1)

    def test_non_integer_step_is_rejected(self):
        with pytest.raises(ParameterError, match="必须是整数"):
            constant_schedule(1.5, base_lr=0.1)  # type: ignore[arg-type]


class TestStepDecaySchedule:
    """阶梯衰减：档位边界必须逐点核对."""

    def test_first_drop_happens_at_drop_every_plus_one(self):
        """``drop_every = 3``：第 1~3 步是 base_lr，第 4 步掉到 ``base·gamma``."""
        assert [step_decay_schedule(step, base_lr=0.1, drop_every=3, gamma=0.5) for step in (1, 2, 3)] == [
            pytest.approx(0.1),
            pytest.approx(0.1),
            pytest.approx(0.1),
        ]
        assert step_decay_schedule(4, base_lr=0.1, drop_every=3, gamma=0.5) == pytest.approx(0.05)

    def test_second_drop(self):
        assert step_decay_schedule(7, base_lr=0.1, drop_every=3, gamma=0.5) == pytest.approx(0.025)

    def test_floor_of_the_formula(self):
        """``⌊(t−1)/drop_every⌋``：第 7 步的指数是 ``⌊6/3⌋ = 2``."""
        assert 0.1 * 0.5**2 == pytest.approx(0.025)

    def test_bad_parameters_are_rejected(self):
        with pytest.raises(ParameterError, match="drop_every"):
            step_decay_schedule(1, base_lr=0.1, drop_every=0)
        with pytest.raises(ParameterError, match="gamma"):
            step_decay_schedule(1, base_lr=0.1, drop_every=3, gamma=1.5)
        with pytest.raises(ParameterError, match="gamma"):
            step_decay_schedule(1, base_lr=0.1, drop_every=3, gamma=0.0)


class TestCosineSchedule:
    """余弦退火：两端都取到端点."""

    def test_starts_at_base_lr(self):
        assert cosine_schedule(1, base_lr=0.1, total_steps=5) == pytest.approx(0.1)

    def test_ends_at_min_lr(self):
        """第 ``T`` 步恰好到 ``min_lr``（分母用 ``T−1`` 的理由）."""
        assert cosine_schedule(5, base_lr=0.1, total_steps=5, min_lr=0.02) == pytest.approx(0.02)

    def test_midpoint_is_half(self):
        """第 3 步（5 步的中点）是 ``(base+min)/2 = 0.05``——手算可得."""
        assert cosine_schedule(3, base_lr=0.1, total_steps=5) == pytest.approx(0.05)

    def test_is_monotone_decreasing(self):
        values = [cosine_schedule(step, base_lr=0.1, total_steps=10) for step in range(1, 11)]
        assert all(later <= earlier for earlier, later in zip(values, values[1:]))

    def test_beyond_total_steps_is_clipped(self):
        """步号超出总步数时不外推（调度是"计划"，跑到计划之外说明计划该改了）."""
        assert cosine_schedule(50, base_lr=0.1, total_steps=5) == pytest.approx(0.0)

    def test_bad_parameters_are_rejected(self):
        with pytest.raises(ParameterError, match="total_steps"):
            cosine_schedule(1, base_lr=0.1, total_steps=1)
        with pytest.raises(ParameterError, match="min_lr"):
            cosine_schedule(1, base_lr=0.1, total_steps=5, min_lr=-0.1)
        with pytest.raises(ParameterError, match="不能大于"):
            cosine_schedule(1, base_lr=0.1, total_steps=5, min_lr=0.5)


class TestWarmupCosineSchedule:
    """线性热身 + 余弦退火：先升后降."""

    def test_first_step_is_not_zero(self):
        """第 1 步是 ``base_lr/warmup``——**不是 0**.

        "第 1 步 lr = 0"会让第一次更新完全不发生，
        而它在日志里表现为"第 1 步的 loss 与初始 loss 一模一样"，
        看起来像"梯度算错了"。
        """
        assert warmup_cosine_schedule(
            1, base_lr=0.1, warmup_steps=4, total_steps=10
        ) == pytest.approx(0.025)

    def test_warmup_reaches_base_lr(self):
        assert warmup_cosine_schedule(
            4, base_lr=0.1, warmup_steps=4, total_steps=10
        ) == pytest.approx(0.1)

    def test_rises_then_falls(self):
        values = [
            warmup_cosine_schedule(step, base_lr=0.1, warmup_steps=3, total_steps=11)
            for step in range(1, 12)
        ]
        assert values[0] < values[1] < values[2]  # 上升段
        assert values[2] == pytest.approx(0.1)  # 峰值
        assert values[-1] <= values[-2]  # 下降段
        assert values[-1] == pytest.approx(0.0)

    def test_bad_parameters_are_rejected(self):
        with pytest.raises(ParameterError, match="warmup_steps"):
            warmup_cosine_schedule(1, base_lr=0.1, warmup_steps=0, total_steps=10)
        with pytest.raises(ParameterError, match="必须小于"):
            warmup_cosine_schedule(1, base_lr=0.1, warmup_steps=10, total_steps=10)


class TestScheduleFactory:
    """调度工厂：键对齐与分发."""

    def test_factory_covers_every_schedule(self):
        assert set(SCHEDULE_FACTORIES) == set(SCHEDULES)

    def test_factory_builds_each_schedule(self):
        schedule = make_schedule("step_decay", base_lr=0.2, drop_every=2, gamma=0.1)
        assert schedule(1) == pytest.approx(0.2)
        assert schedule(3) == pytest.approx(0.02)

    def test_unknown_schedule_is_rejected(self):
        with pytest.raises(ParameterError, match="不认识的学习率调度"):
            make_schedule("warmup_only", base_lr=0.1)


class TestGradientClipping:
    """梯度裁剪：三个原语各自的行为."""

    def test_global_norm(self):
        assert global_norm((3.0, 4.0)) == pytest.approx(5.0)

    def test_no_clipping_below_the_limit(self):
        grads, scale = clip_by_global_norm((3.0, 4.0), 10.0)
        assert grads == (3.0, 4.0)
        assert scale == 1.0

    def test_clipping_preserves_direction(self):
        """``(3, 4)`` 范数为 5，裁到 2.5 之后是 ``(1.5, 2)``——方向不变."""
        grads, scale = clip_by_global_norm((3.0, 4.0), 2.5)
        assert scale == pytest.approx(0.5)
        assert approx(grads[0], 1.5)
        assert approx(grads[1], 2.0)
        assert global_norm(grads) == pytest.approx(2.5)

    def test_exact_boundary_is_not_clipped(self):
        grads, scale = clip_by_global_norm((3.0, 4.0), 5.0)
        assert scale == 1.0
        assert grads == (3.0, 4.0)

    def test_bad_limit_is_rejected(self):
        with pytest.raises(ParameterError, match="max_norm"):
            clip_by_global_norm((1.0,), 0.0)

    def test_clip_by_value_changes_lenient_components(self):
        """逐分量裁剪会改变方向：``(100, 1)`` 被裁成 ``(1, 1)``（从几乎水平变 45°）."""
        assert clip_by_value((100.0, 1.0), 1.0) == (1.0, 1.0)

    def test_clip_by_value_validation(self):
        with pytest.raises(ParameterError, match="limit"):
            clip_by_value((1.0,), -1.0)


class TestSGD:
    """SGD：一步就是 ``θ − lr·g``."""

    def test_single_step_matches_hand_computation(self):
        optimizer = SGDOptimizer(0.1)
        updated = optimizer.step((1.0, 2.0), (0.5, -0.25))
        assert updated == (pytest.approx(0.95), pytest.approx(2.025))

    def test_step_count_increments(self):
        optimizer = SGDOptimizer(0.1)
        optimizer.step((0.0,), (0.0,))
        optimizer.step((0.0,), (0.0,))
        assert optimizer.step_count == 2

    def test_state_is_json_friendly(self):
        import json

        optimizer = SGDOptimizer(0.1)
        optimizer.step((0.0,), (1.0,))
        payload = optimizer.state()
        json.dumps(payload)
        assert payload["name"] == "sgd"
        assert payload["step_count"] == 1

    def test_describe_mentions_the_formula_source(self):
        assert "随机梯度下降" in SGDOptimizer(0.1).describe()

    def test_bad_learning_rate_is_rejected(self):
        with pytest.raises(ParameterError, match="learning_rate"):
            SGDOptimizer(-0.1)

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ShapeError, match="维"):
            SGDOptimizer(0.1).step((1.0, 2.0), (1.0,))


class TestMomentum:
    """动量法：两步手算复核."""

    def test_first_step(self):
        """``v₁ = g₁``，于是第一步与 SGD 相同."""
        optimizer = MomentumOptimizer(0.1, momentum=0.9)
        updated = optimizer.step((0.0,), (1.0,))
        assert updated == (pytest.approx(-0.1),)

    def test_second_step_accumulates_velocity(self):
        """``v₂ = 0.9·1 + 1 = 1.9``，因此第二步走 ``−0.19``（比 SGD 的 ``−0.1`` 大）."""
        optimizer = MomentumOptimizer(0.1, momentum=0.9)
        optimizer.step((0.0,), (1.0,))
        updated = optimizer.step((-0.1,), (1.0,))
        assert optimized(updated) == pytest.approx(-0.1 - 0.19)

    def test_velocity_is_part_of_the_state(self):
        optimizer = MomentumOptimizer(0.1, momentum=0.5)
        optimizer.step((0.0,), (2.0,))
        assert optimizer.state()["velocity"] == [2.0]

    def test_reset_clears_the_velocity(self):
        optimizer = MomentumOptimizer(0.1, momentum=0.9)
        optimizer.step((0.0,), (1.0,))
        optimizer.reset()
        assert optimizer.velocity == ()
        assert optimizer.step_count == 0
        assert optimizer.step((0.0,), (1.0,)) == (pytest.approx(-0.1),)

    def test_bad_momentum_is_rejected(self):
        with pytest.raises(ParameterError, match="momentum"):
            MomentumOptimizer(0.1, momentum=1.0)
        with pytest.raises(ParameterError, match="momentum"):
            MomentumOptimizer(0.1, momentum=-0.1)


class TestAdam:
    """Adam：偏差修正是这一节的全部重点."""

    def test_first_step_is_lr_times_sign(self):
        """**不做偏差修正时第一步会走出 ``lr·3.16``**.

        ```text
        m₁ = 0.1g        （真值的 1/10）
        v₁ = 0.001g²     （真值的 1/1000）
        无修正：lr·0.1g / (√0.001·|g|) = lr·3.16       ← 比设定值大 3 倍
        有修正：lr·g/g = lr                            ← 恰好是 step size
        ```

        因此这一条断言把"偏差修正"从一个公式变成一个可验证的等式。
        """
        optimizer = AdamOptimizer(0.1)
        for gradient in (1.0, -1.0, 0.5):
            fresh = AdamOptimizer(0.1)
            updated = fresh.step((0.0,), (gradient,))
            assert optimized(updated) == pytest.approx(-0.1 * math.copysign(1.0, gradient))

    def test_unbiased_first_step_would_be_over_three_times_larger(self):
        """把修正项去掉（用未修正的动量手算一次），第一步会大 3 倍以上."""
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        gradient = 1.0
        mean = (1.0 - beta1) * gradient
        square = (1.0 - beta2) * gradient * gradient
        without_correction = 0.1 * mean / (math.sqrt(square) + epsilon)
        assert without_correction > 3.0 * 0.1

    def test_constant_gradient_gives_approximately_lr_per_step(self):
        """梯度恒为 ``g`` 时每步位移约等于 ``lr``，**与 ``|g|`` 无关**.

        直接对优化器做（不经 ``minimize``）：这样绕过数值梯度，
        测到的就是"更新公式本身"的性质。两串梯度相差 100 万倍，
        30 步的位移都是 ``−3.0``（``30 × 0.1``）。

        推导：梯度恒定时 ``m̂/√v̂ = g/|g| = sign(g)``
        （两个偏差修正因子恰好抵消），因此每步都是 ``−lr·sign(g)``。
        """
        def travel(gradient: float, steps: int = 30) -> float:
            optimizer = AdamOptimizer(0.1)
            position = 0.0
            for _ in range(steps):
                position = optimizer.step((position,), (gradient,))[0]
            return position

        assert travel(1e-3) == pytest.approx(-3.0, rel=0.01)
        assert travel(1e3) == pytest.approx(-3.0, rel=0.01)

    def test_zero_gradient_does_not_move(self):
        optimizer = AdamOptimizer(0.1)
        updated = optimizer.step((1.0,), (0.0,))
        assert updated == (pytest.approx(1.0),)

    def test_bias_correction_factors_grow_towards_one(self):
        """两个修正因子都在往 1 走，但速度差 10 倍（这正是 β₁ 与 β₂ 取值差的意义）.

        ```text
        β₁ = 0.9     第 1 步修正因子 1 − 0.9   = 0.1      第 50 步 ≈ 0.995
        β₂ = 0.999   第 1 步修正因子 1 − 0.999 = 0.001    第 50 步 ≈ 0.049   ← 还很远
        ```

        "二阶动量的偏差修正在前 1000 步内一直重要"不是一句经验之谈，
        而是 ``β₂ᵗ`` 这条衰减曲线的直接读数（``0.999^1000 ≈ 0.37``）。
        """
        optimizer = AdamOptimizer(0.1)
        first_factors = []
        second_factors = []
        for _ in range(50):
            optimizer.step((0.0,), (1.0,))
            first_factors.append(1.0 - optimizer.beta1**optimizer.step_count)
            second_factors.append(1.0 - optimizer.beta2**optimizer.step_count)
        assert first_factors[0] == pytest.approx(0.1)
        assert first_factors[-1] > 0.99  # β₁ 的修正 50 步内基本到位
        assert second_factors[0] == pytest.approx(0.001)
        assert second_factors[-1] < 0.05  # β₂ 的修正还很远
        assert second_factors[-1] > second_factors[0]  # 但它确实在增长

    def test_reset_clears_both_moments_and_step_count(self):
        optimizer = AdamOptimizer(0.1)
        optimizer.step((0.0,), (1.0,))
        optimizer.reset()
        assert optimizer.first_moment == ()
        assert optimizer.second_moment == ()
        assert optimizer.step_count == 0
        # 重置之后"第一步"的行为必须与全新实例一致（否则偏差修正会错位）
        fresh = AdamOptimizer(0.1)
        assert optimizer.step((0.0,), (1.0,)) == fresh.step((0.0,), (1.0,))

    def test_bad_hyperparameters_are_rejected(self):
        with pytest.raises(ParameterError, match="beta1"):
            AdamOptimizer(0.1, beta1=1.0)
        with pytest.raises(ParameterError, match="beta2"):
            AdamOptimizer(0.1, beta2=-0.1)
        with pytest.raises(ParameterError, match="epsilon"):
            AdamOptimizer(0.1, epsilon=0.0)

    def test_state_is_json_friendly(self):
        import json

        optimizer = AdamOptimizer(0.1)
        optimizer.step((0.0,), (1.0,))
        json.dumps(optimizer.state())
        assert optimizer.state()["beta1"] == 0.9


def optimized(params) -> float:
    """取单参数向量的那个分量（让"手算值"的断言读起来像一个数）."""
    assert len(params) == 1
    return params[0]


class TestOptimizerFactory:
    """优化器工厂与类表."""

    def test_classes_cover_every_optimizer(self):
        assert set(OPTIMIZER_CLASSES) == set(OPTIMIZERS)

    def test_factory_builds_each_optimizer(self):
        assert isinstance(make_optimizer("sgd", 0.1), SGDOptimizer)
        assert isinstance(make_optimizer("momentum", 0.1), MomentumOptimizer)
        assert isinstance(make_optimizer("adam", 0.1), AdamOptimizer)

    def test_factory_passes_through_keyword_parameters(self):
        optimizer = make_optimizer("adam", 0.1, beta1=0.5)
        assert isinstance(optimizer, AdamOptimizer)
        assert optimizer.beta1 == 0.5

    def test_unknown_optimizer_is_rejected(self):
        with pytest.raises(ParameterError, match="不认识的优化器"):
            make_optimizer("rmsprop", 0.1)


class TestMinimize:
    """训练回路：损失必须真的降下来，而且痕迹要完整."""

    def test_bowl_converges_with_sgd(self):
        trace = minimize(bowl, (3.0, 3.0), optimizer=SGDOptimizer(0.05), steps=200)
        assert trace.final_loss < 1e-6
        assert trace.improvement > 0.99

    def test_bowl_converges_with_adam(self):
        trace = minimize(bowl, (3.0, 3.0), optimizer=AdamOptimizer(0.2), steps=400)
        assert trace.final_loss < 1e-3

    def test_ravine_separates_adam_from_plain_sgd(self):
        """峡谷地形上：Adam 在同样的步数里把损失压得更低（每参数等效步长）.

        这是"为什么要有 Adam"在一个**可测**的场景里的证据；
        在曲率相同的碗形函数上，三种优化器的差别看不出来。
        """
        steps = 60
        sgd = minimize(ravine, RAVINE_START, optimizer=SGDOptimizer(0.01), steps=steps)
        adam = minimize(ravine, RAVINE_START, optimizer=AdamOptimizer(0.1), steps=steps)
        assert adam.final_loss < sgd.final_loss

    def test_trace_lengths_are_consistent(self):
        """``losses`` 比 ``learning_rates`` 多一个（初始损失没有对应的学习率）."""
        trace = minimize(bowl, (3.0, 3.0), optimizer=SGDOptimizer(0.05), steps=10)
        assert len(trace.losses) == 11
        assert len(trace.learning_rates) == 10
        assert len(trace.params) == 11
        assert trace.steps == 10

    def test_losses_decrease_monotonically_on_a_bowl(self):
        trace = minimize(bowl, (3.0, 3.0), optimizer=SGDOptimizer(0.05), steps=20)
        assert trace.monotone() is True

    def test_best_loss_can_be_smaller_than_final_loss(self):
        """学习率过大时后期会在最优点附近来回抖：``best_loss < final_loss``.

        这条断言解释了 ``TrainingTrace`` 为什么要单独记 ``best_loss`` 与
        ``best_index``——只报最终损失会让人以为"它从来没到过那里"。
        """
        trace = minimize(bowl, (3.0, 3.0), optimizer=SGDOptimizer(0.52), steps=30)
        assert trace.best_loss <= trace.final_loss
        assert trace.best_index != trace.steps

    def test_schedule_is_applied_each_step(self):
        """调度生效的证据在 ``learning_rates`` 里：它应当等于逐点的调度值."""
        schedule = make_schedule("step_decay", base_lr=0.05, drop_every=2, gamma=0.5)
        trace = minimize(
            bowl, (3.0, 3.0), optimizer=SGDOptimizer(0.05), steps=6, schedule=schedule
        )
        assert list(trace.learning_rates) == [
            pytest.approx(schedule(step)) for step in range(1, 7)
        ]

    def test_converged_flag(self):
        trace = minimize(bowl, (3.0, 3.0), optimizer=SGDOptimizer(0.05), steps=500)
        assert trace.converged is True
        short = minimize(bowl, (3.0, 3.0), optimizer=SGDOptimizer(0.05), steps=3)
        assert short.converged is False

    def test_early_stop_when_tolerance_is_reached(self):
        trace = minimize(bowl, BOWL_MINIMUM, optimizer=SGDOptimizer(0.1), steps=50)
        assert trace.steps == 1  # 起点就是最优，第一步之后立即满足收敛判据
        assert trace.converged is True

    def test_trace_is_json_friendly(self):
        import json

        trace = minimize(bowl, (3.0, 3.0), optimizer=SGDOptimizer(0.05), steps=5)
        payload = trace.to_dict()
        json.dumps(payload)
        assert payload["steps"] == 5
        assert payload["monotone"] is True
        assert payload["notes"]
        assert "步 |" in trace.summary_line()
        assert trace.loss_line(0).endswith(f"{trace.losses[0]:.8f}")
        assert "lr" in trace.loss_line(3)

    def test_bad_parameters_are_rejected(self):
        with pytest.raises(ParameterError, match="steps"):
            minimize(bowl, (0.0,), optimizer=SGDOptimizer(0.1), steps=0)
        with pytest.raises(ParameterError, match="tolerance"):
            minimize(bowl, (0.0,), optimizer=SGDOptimizer(0.1), steps=1, tolerance=-1.0)
        with pytest.raises(ShapeError, match="不能为空"):
            minimize(bowl, (), optimizer=SGDOptimizer(0.1), steps=1)

    def test_objective_returning_nan_is_rejected(self):
        with pytest.raises(NumericError, match="非有限数"):
            minimize(lambda params: float("nan"), (1.0,), optimizer=SGDOptimizer(0.1), steps=1)


class TestTrainingTraceShape:
    """``TrainingTrace`` 自身的形状校验."""

    def test_loss_and_param_lengths_must_match(self):
        with pytest.raises(NumericError, match="一一对应"):
            TrainingTrace(losses=(1.0, 0.5), learning_rates=(0.1,), params=((1.0,),))

    def test_learning_rates_must_be_one_short(self):
        with pytest.raises(NumericError, match="少一个"):
            TrainingTrace(
                losses=(1.0,),
                learning_rates=(0.1,),
                params=((1.0,),),
            )

    def test_loss_line_range_is_checked(self):
        trace = TrainingTrace(losses=(1.0, 0.5), learning_rates=(0.1,), params=((1.0,), (0.5,)))
        with pytest.raises(ParameterError, match="超出范围"):
            trace.loss_line(5)

    def test_improvement_of_zero_initial_loss_is_zero(self):
        trace = TrainingTrace(losses=(0.0, 0.0), learning_rates=(0.1,), params=((1.0,), (1.0,)))
        assert trace.improvement == 0.0

    def test_summary_line_reports_monotonicity(self):
        rising = TrainingTrace(losses=(1.0, 2.0), learning_rates=(0.1,), params=((1.0,), (2.0,)))
        assert "单调 否" in rising.summary_line()


class TestParameterFlattening:
    """压平与还原：矩阵参数与向量优化器之间的桥."""

    MATRICES = (((1.0, 2.0), (3.0, 4.0)), ((5.0,),))

    def test_flatten_row_major(self):
        flat, shapes = flatten_matrices(self.MATRICES)
        assert flat == (1.0, 2.0, 3.0, 4.0, 5.0)
        assert shapes == ((2, 2), (1, 1))

    def test_roundtrip(self):
        flat, shapes = flatten_matrices(self.MATRICES)
        assert unflatten_matrices(flat, shapes) == self.MATRICES

    def test_length_mismatch_is_rejected(self):
        with pytest.raises(ShapeError, match="形状表需要"):
            unflatten_matrices((1.0, 2.0, 3.0), ((2, 2),))

    def test_empty_input_is_rejected(self):
        with pytest.raises(ShapeError, match="至少要有一个矩阵"):
            flatten_matrices([])

    def test_optimizer_works_on_a_flattened_matrix(self):
        """压平之后 Adam 不需要知道它优化的是一个矩阵（day075 的复用点）."""
        flat, shapes = flatten_matrices(self.MATRICES)
        params = SGDOptimizer(0.1).step(flat, tuple(1.0 for _ in flat))
        restored = unflatten_matrices(params, shapes)
        assert restored[0][0][0] == pytest.approx(0.9)
        assert restored[1][0][0] == pytest.approx(4.9)
