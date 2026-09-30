"""induction 任务、批量梯度与训练回路（day075）.

这一份测试回答一个**必须被回答**的问题：这一层真的能学吗？

```text
四个读数     损失 / 峰值权重 / 命中率（三个一起看） / 参数块（冻结了哪些）
两个对账     解析梯度 vs 数值梯度（用 SGD，误差不被归一化放大）
一个反例     "损失到 0"不等于"学到机制"——冻结 value/output 时损失降不下去，
             而四块全训时损失到 0 却只把峰值推到 0.44
```

训练本身是确定性的（样本用 LCG 数列生成、初始化写死、优化器无随机性），
因此这里的数字可以被**复核**，而不是"这次跑出来的"。
"""

from __future__ import annotations

import json
import math

import pytest

from smart_research_agent.math_foundations.calculus import gradient
from smart_research_agent.math_foundations.errors import ShapeError as MathShapeError
from smart_research_agent.math_foundations.optim import (
    AdamOptimizer,
    SGDOptimizer,
    cosine_schedule,
)
from smart_research_agent.transformer_core.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.transformer_core.layers import self_attention
from smart_research_agent.transformer_core.train import (
    DEFAULT_INIT_SCALE,
    GRADIENT_SOURCES,
    GRADIENT_SOURCE_ANALYTIC,
    GRADIENT_SOURCE_NUMERIC,
    PARAMETER_BLOCKS,
    AttentionTrainingReport,
    InductionTask,
    analytic_objective,
    batch_accuracy,
    batch_gradients,
    batch_loss,
    batch_mean_peak_weight,
    default_parameters,
    make_induction_batch,
    make_induction_task,
    train_attention,
)
from tests.attention_samples import ONE_HOT_INPUTS, SMALL_TARGET, approx, induction_tasks


class TestInductionTask:
    """样本的结构与四条校验."""

    def test_generated_shape(self):
        task = make_induction_task(seed=42)
        assert task.length == 5
        assert task.supervised == (3, 4)
        assert task.vocabulary == 6
        assert len(task.inputs) == 5

    def test_tokens_are_one_hot_rows(self):
        task = make_induction_task(seed=42)
        for row, token in zip(task.inputs, task.tokens):
            assert row[token] == 1.0
            assert math.fsum(row) == 1.0

    def test_supervised_tokens_appeared_earlier(self):
        """induction 的定义：答案必须能在上文里被找到."""
        task = make_induction_task(seed=42)
        for position in task.supervised:
            assert task.tokens[position] in task.tokens[:position]

    def test_head_is_made_of_distinct_tokens(self):
        task = make_induction_task(seed=42, sources=3, repeats=2)
        head = task.tokens[:3]
        assert len(set(head)) == 3

    def test_target_is_the_input(self):
        """目标是"把那个 token 复制出来"，因此它就是输入本身（**同一份数据**）."""
        task = make_induction_task(seed=42)
        assert task.target is task.inputs

    def test_deterministic_across_calls(self):
        first = make_induction_task(seed=42)
        second = make_induction_task(seed=42)
        assert first == second

    def test_different_seeds_give_different_samples(self):
        first = make_induction_task(seed=42)
        second = make_induction_task(seed=43)
        assert first.tokens != second.tokens

    def test_describe_and_to_dict(self):
        task = make_induction_task(seed=42)
        assert "^" in task.describe()
        payload = task.to_dict()
        json.dumps(payload)
        assert payload["length"] == 5
        assert payload["supervised"] == [3, 4]

    def test_token_out_of_range_is_rejected(self):
        with pytest.raises(ParameterError, match="token 下标"):
            InductionTask(
                vocabulary=2,
                tokens=(0, 1, 3),
                inputs=((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
                supervised=(2,),
                seed=0,
            )

    def test_vocabulary_must_be_at_least_two(self):
        with pytest.raises(ParameterError, match="vocabulary"):
            make_induction_task(seed=0, vocabulary=1)

    def test_vocabulary_must_be_an_integer(self):
        with pytest.raises(ParameterError, match="整数"):
            make_induction_task(seed=0, vocabulary=6.0)  # type: ignore[arg-type]

    def test_empty_tokens_is_rejected(self):
        with pytest.raises(ShapeError, match="不能为空"):
            InductionTask(vocabulary=2, tokens=(), inputs=(), supervised=(0,), seed=0)

    def test_input_shape_must_match(self):
        with pytest.raises(ShapeError, match="输入形状"):
            InductionTask(
                vocabulary=2,
                tokens=(0, 1),
                inputs=((1.0, 0.0),),
                supervised=(1,),
                seed=0,
            )

    def test_empty_supervision_is_rejected(self):
        with pytest.raises(ShapeError, match="监督位置不能为空"):
            InductionTask(
                vocabulary=2,
                tokens=(0, 1, 0),
                inputs=((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
                supervised=(),
                seed=0,
            )

    def test_supervised_position_without_an_earlier_match_is_rejected(self):
        """没有"上文里出现过"的位置没有可学的机制——当场拒绝而不是让它去猜."""
        with pytest.raises(ShapeError, match="之前没有出现过"):
            InductionTask(
                vocabulary=4,
                tokens=(0, 1, 2, 3),
                inputs=(
                    (1.0, 0.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                supervised=(3,),
                seed=0,
            )

    def test_supervised_position_out_of_range_is_rejected(self):
        with pytest.raises(ShapeError, match="之外"):
            InductionTask(
                vocabulary=2,
                tokens=(0, 1, 0),
                inputs=((1.0, 0.0), (0.0, 1.0), (1.0, 0.0)),
                supervised=(5,),
                seed=0,
            )

    def test_repeats_and_sources_are_validated(self):
        with pytest.raises(ParameterError, match="repeats"):
            make_induction_task(seed=0, repeats=0)
        with pytest.raises(ParameterError, match="sources"):
            make_induction_task(seed=0, sources=0)

    def test_more_repeats_than_sources_cycles(self):
        """``repeats > sources`` 时按位置循环复用（不会越界）."""
        task = make_induction_task(seed=42, sources=2, repeats=4)
        assert task.length == 6
        assert task.supervised == (2, 3, 4, 5)
        for position in task.supervised:
            assert task.tokens[position] in task.tokens[:position]


class TestInductionBatch:
    """批量生成."""

    def test_count_and_seeds(self):
        tasks = make_induction_batch(3, seed=42)
        assert len(tasks) == 3
        assert tasks[0].seed == 42
        assert tasks[1].seed == 49

    def test_deterministic(self):
        assert make_induction_batch(2, seed=42) == make_induction_batch(2, seed=42)

    def test_zero_count_is_rejected(self):
        with pytest.raises(ParameterError, match="count"):
            make_induction_batch(0)


class TestDefaultParameters:
    """确定性初始化."""

    def test_shapes(self):
        params = default_parameters(6)
        assert params.shape.inputs == 6
        assert params.parameter_count() == 4 * 36

    def test_deterministic(self):
        assert default_parameters(6, seed=7) == default_parameters(6, seed=7)

    def test_different_seeds_differ(self):
        assert default_parameters(6, seed=7) != default_parameters(6, seed=8)

    def test_values_stay_within_the_scale(self):
        params = default_parameters(6, seed=7, scale=0.25)
        for matrix in params.matrices():
            for row in matrix:
                for value in row:
                    assert -0.25 <= value < 0.25

    def test_scale_is_not_one_hot_identity(self):
        """**不是**"把 W_v/W_o 初始化成单位矩阵"：那样命中率一开始就是 1.0，
        三个读数里有两个不会动，"训练有没有效果"就看不出来了。
        """
        params = default_parameters(4, seed=7)
        identity = tuple(
            tuple(1.0 if row == column else 0.0 for column in range(4)) for row in range(4)
        )
        assert params.w_value != identity
        assert params.w_output != identity

    def test_scale_validation(self):
        with pytest.raises(ParameterError, match="scale"):
            default_parameters(6, scale=0.0)
        with pytest.raises(ParameterError, match="scale"):
            default_parameters(6, scale=1.0)

    def test_vocabulary_validation(self):
        with pytest.raises(ParameterError, match="vocabulary"):
            default_parameters(1)


class TestBatchReadings:
    """四个批量读数（损失 / 梯度 / 命中率 / 峰值权重）."""

    def test_loss_is_a_positive_number(self):
        value = batch_loss(default_parameters(6, seed=7), induction_tasks())
        assert 0.0 < value < 1.0

    def test_accuracy_is_a_fraction(self):
        value = batch_accuracy(default_parameters(6, seed=7), induction_tasks())
        assert 0.0 <= value <= 1.0

    def test_peak_weight_is_between_uniform_and_one(self):
        """初始峰值在"均匀分布"附近（每行候选数 4~5 → 均匀峰值 0.20~0.25）."""
        value = batch_mean_peak_weight(default_parameters(6, seed=7), induction_tasks())
        assert 0.15 < value < 0.4

    def test_gradient_length_matches_parameters(self):
        params = default_parameters(6, seed=7)
        grads = batch_gradients(params, induction_tasks())
        assert len(grads) == len(params.flatten()[0]) == 4 * 36

    def test_gradient_is_not_all_zero(self):
        grads = batch_gradients(default_parameters(6, seed=7), induction_tasks())
        assert any(value != 0.0 for value in grads)

    def test_empty_batch_is_rejected(self):
        with pytest.raises(ShapeError, match="样本批不能为空"):
            batch_loss(default_parameters(6), ())

    def test_mixed_vocabularies_are_rejected(self):
        tasks = (make_induction_task(seed=42, vocabulary=6), make_induction_task(seed=43, vocabulary=4))
        with pytest.raises(ShapeError, match="词表"):
            batch_loss(default_parameters(6), tasks)

    def test_unknown_gradient_source_is_rejected(self):
        with pytest.raises(ParameterError, match="梯度来源"):
            batch_gradients(default_parameters(6), induction_tasks(), source="magic")

    def test_analytic_and_numeric_gradients_match(self):
        """两种梯度源在同一批样本上的最大绝对差在数值分辨率量级.

        实测约 ``2e-11``——它由数值差分自身误差决定（day074 的分辨率公式），
        而不是实现差异。
        """
        params = default_parameters(6, seed=7)
        tasks = induction_tasks()
        analytic = batch_gradients(params, tasks, source=GRADIENT_SOURCE_ANALYTIC)
        numeric = batch_gradients(params, tasks, source=GRADIENT_SOURCE_NUMERIC)
        worst = max(abs(a - b) for a, b in zip(analytic, numeric, strict=True))
        assert worst < 1e-8
        assert worst > 0.0


class TestTrainableBlocks:
    """冻结：不训练的块梯度**恰好是 0**."""

    def test_parameter_blocks_are_named(self):
        assert PARAMETER_BLOCKS == ("w_query", "w_key", "w_value", "w_output")

    def test_frozen_blocks_get_zero_gradient(self):
        from smart_research_agent.math_foundations.optim import unflatten_matrices

        params = default_parameters(6, seed=7)
        shapes = params.flatten()[1]
        grads = batch_gradients(params, induction_tasks(), trainable=("w_query", "w_key"))
        blocks = unflatten_matrices(grads, shapes)
        assert any(value != 0.0 for row in blocks[0] for value in row)
        assert any(value != 0.0 for row in blocks[1] for value in row)
        assert all(value == 0.0 for row in blocks[2] for value in row)
        assert all(value == 0.0 for row in blocks[3] for value in row)

    def test_unknown_block_is_rejected(self):
        with pytest.raises(ParameterError, match="不认识的参数块"):
            batch_gradients(default_parameters(6), induction_tasks(), trainable=("w_magic",))

    def test_duplicate_block_is_rejected(self):
        with pytest.raises(ParameterError, match="重复"):
            batch_gradients(
                default_parameters(6), induction_tasks(), trainable=("w_query", "w_query")
            )

    def test_empty_trainable_is_rejected(self):
        with pytest.raises(ParameterError, match="至少要训练一块"):
            batch_gradients(default_parameters(6), induction_tasks(), trainable=())


class TestTraining:
    """训练回路：损失必须真的降下来，而且三个读数都要动."""

    def _run(self, *, steps: int = 200):
        return train_attention(
            default_parameters(6, seed=7),
            induction_tasks(),
            optimizer=AdamOptimizer(0.05),
            steps=steps,
        )

    def test_loss_decreases(self):
        report = self._run()
        assert report.ok is True
        assert report.final_loss < report.initial_loss

    def test_accuracy_improves(self):
        report = self._run()
        assert report.final_accuracy > report.initial_accuracy
        assert report.final_accuracy == 1.0

    def test_peak_weight_rises(self):
        """注意力确实变尖了（但**不会**到 1.0——见下面的"另一条路"那条测试）."""
        report = self._run()
        assert report.final_peak_weight > report.initial_peak_weight

    def test_trace_lengths_are_consistent(self):
        report = self._run(steps=20)
        assert len(report.trace.losses) == 21
        assert len(report.peak_weights) == 21
        assert len(report.accuracies) == 21
        assert report.steps == 20

    def test_epoch_line_readings(self):
        report = self._run(steps=5)
        assert "step" in report.epoch_line(0)
        assert "lr" in report.epoch_line(3)
        with pytest.raises(ParameterError, match="超出范围"):
            report.epoch_line(99)

    def test_report_is_json_friendly(self):
        payload = self._run(steps=5).to_dict()
        json.dumps(payload)
        assert payload["steps"] == 5
        assert payload["frozen"] == []
        assert payload["notes"]
        assert "命中率" in self._run(steps=5).summary_line()

    def test_final_parameters_are_returned(self):
        report = self._run(steps=3)
        assert report.final_parameters.shape.inputs == 6
        assert report.final_parameters != default_parameters(6, seed=7)

    def test_steps_validation(self):
        with pytest.raises(ParameterError, match="steps"):
            train_attention(
                default_parameters(6), induction_tasks(), optimizer=AdamOptimizer(0.05), steps=0
            )

    def test_source_validation(self):
        with pytest.raises(ParameterError, match="梯度来源"):
            train_attention(
                default_parameters(6),
                induction_tasks(),
                optimizer=AdamOptimizer(0.05),
                steps=1,
                source="magic",
            )

    def test_schedule_is_applied(self):
        report = train_attention(
            default_parameters(6, seed=7),
            induction_tasks(),
            optimizer=AdamOptimizer(0.05),
            steps=4,
            schedule=lambda step: cosine_schedule(step, base_lr=0.05, total_steps=10),
        )
        assert list(report.trace.learning_rates) == [
            pytest.approx(cosine_schedule(step, base_lr=0.05, total_steps=10))
            for step in range(1, 5)
        ]

    def test_schedule_hitting_zero_is_rejected(self):
        """调度给出非正学习率时当场拒绝（"这一步没走"看起来像"梯度错了"）."""
        with pytest.raises(ParameterError, match="非正学习率"):
            train_attention(
                default_parameters(6, seed=7),
                induction_tasks(),
                optimizer=AdamOptimizer(0.05),
                steps=4,
                schedule=lambda step: cosine_schedule(step, base_lr=0.05, total_steps=4),
            )

    def test_sgd_also_learns_but_slower(self):
        """同一个任务、同样 200 步：SGD 也在降，但明显不如 Adam（day074 的结论在这里重现）."""
        sgd = train_attention(
            default_parameters(6, seed=7),
            induction_tasks(),
            optimizer=SGDOptimizer(0.05),
            steps=200,
        )
        adam = self._run()
        assert sgd.final_loss < sgd.initial_loss
        assert adam.final_loss < sgd.final_loss

    def test_freezing_value_and_output_stalls_the_loss(self):
        """**"损失到 0"不等于"学到机制"**：只训 q/k 时注意力变尖，但损失降不下去.

        ```text
        四块全训  损失 → 0.0000   峰值 → 0.444
        只训 q/k  损失 → 0.1554   峰值 → 0.587
        ```

        两个方向都反直觉：全训时模型靠 value 路径上的"抵消"把损失压到 0，
        而注意力并没有变得很尖；只训 q/k 时注意力变尖了，但"看对了地方"
        只是必要条件——还要 value/output 把看到的东西映射成目标。
        """
        frozen = train_attention(
            default_parameters(6, seed=7),
            induction_tasks(),
            optimizer=AdamOptimizer(0.05),
            steps=200,
            trainable=("w_query", "w_key"),
        )
        assert frozen.frozen == ("w_value", "w_output")
        assert frozen.peak_weights[-1] > frozen.peak_weights[0]
        assert frozen.final_loss > 0.1

    def test_both_gradient_sources_agree_with_sgd(self):
        """两种梯度源的损失轨迹必须一致——**用 SGD 验证**.

        为什么不是 Adam：Adam 按 ``√v̂`` 归一化，会放大梯度分量的**相对**误差，
        于是两种来源的轨迹差约 1e-11（SGD 是 1e-14）——两支都收敛，
        但"验证梯度实现"这件事应当用不带归一化的优化器做。
        """
        first = train_attention(
            default_parameters(6, seed=7),
            induction_tasks(),
            optimizer=SGDOptimizer(0.05),
            steps=6,
        )
        second = train_attention(
            default_parameters(6, seed=7),
            induction_tasks(),
            optimizer=SGDOptimizer(0.05),
            steps=6,
            source=GRADIENT_SOURCE_NUMERIC,
        )
        assert first.gradient_source == GRADIENT_SOURCE_ANALYTIC
        assert second.gradient_source == GRADIENT_SOURCE_NUMERIC
        worst = max(
            abs(a - b) for a, b in zip(first.trace.losses, second.trace.losses, strict=True)
        )
        assert worst < 1e-9

    def test_both_gradient_sources_agree_with_adam_to_a_looser_tolerance(self):
        first = train_attention(
            default_parameters(6, seed=7), induction_tasks(), optimizer=AdamOptimizer(0.05), steps=6
        )
        second = train_attention(
            default_parameters(6, seed=7),
            induction_tasks(),
            optimizer=AdamOptimizer(0.05),
            steps=6,
            source=GRADIENT_SOURCE_NUMERIC,
        )
        worst = max(
            abs(a - b) for a, b in zip(first.trace.losses, second.trace.losses, strict=True)
        )
        assert worst < 1e-6


class TestTrainingReportValidation:
    """报告的三个长度必须一致、两个枚举必须合法."""

    def _report(self):
        return train_attention(
            default_parameters(6, seed=7),
            induction_tasks(),
            optimizer=AdamOptimizer(0.05),
            steps=3,
        )

    def test_length_mismatch_is_rejected(self):
        report = self._report()
        with pytest.raises(NumericError, match="长度必须一致"):
            AttentionTrainingReport(
                trace=report.trace,
                final_parameters=report.final_parameters,
                peak_weights=report.peak_weights[:-1],
                accuracies=report.accuracies,
            )

    def test_unknown_gradient_source_is_rejected(self):
        report = self._report()
        with pytest.raises(ParameterError, match="梯度来源"):
            AttentionTrainingReport(
                trace=report.trace,
                final_parameters=report.final_parameters,
                peak_weights=report.peak_weights,
                accuracies=report.accuracies,
                gradient_source="magic",
            )

    def test_unknown_trainable_block_is_rejected(self):
        report = self._report()
        with pytest.raises(ParameterError, match="不认识的参数块"):
            AttentionTrainingReport(
                trace=report.trace,
                final_parameters=report.final_parameters,
                peak_weights=report.peak_weights,
                accuracies=report.accuracies,
                trainable=("w_magic",),
            )

    def test_readings_and_improvement_ratio(self):
        report = self._report()
        assert report.initial_accuracy == report.accuracies[0]
        assert report.final_accuracy == report.accuracies[-1]
        assert report.initial_peak_weight == report.peak_weights[0]
        assert report.final_peak_weight == report.peak_weights[-1]
        assert report.improvement_ratio > 0.0

    def test_gradient_sources_table(self):
        assert GRADIENT_SOURCES == ("analytic", "numeric")

    def test_default_init_scale_is_documented(self):
        assert 0.0 < DEFAULT_INIT_SCALE < 1.0


class TestAnalyticObjective:
    """``analytic_objective`` 能被 day074 的数值梯度直接吃（手工核对用）."""

    def test_objective_is_a_flat_function(self):
        tasks = induction_tasks()
        params = default_parameters(6, seed=7)
        objective = analytic_objective(tasks)
        flat, _shapes = params.flatten()
        assert objective(flat) == pytest.approx(batch_loss(params, tasks))

    def test_objective_gradient_matches_the_batch_gradient(self):
        """``calculus.gradient`` 对目标函数求出的数值梯度 ≈ 批量解析梯度.

        两条路的差别只来自数值差分本身（约 1e-10），因此阈值取 1e-6。
        """
        tasks = induction_tasks(2)
        params = default_parameters(6, seed=7)
        objective = analytic_objective(tasks)
        flat, _shapes = params.flatten()
        numeric = gradient(objective, flat)
        analytic = batch_gradients(params, tasks)
        worst = max(abs(a - b) for a, b in zip(analytic, numeric, strict=True))
        assert worst < 1e-6


class TestForwardConsistency:
    """训练里用的前向与直接调用 ``self_attention`` 必须一致（没有第二条路径）."""

    def test_batch_loss_uses_the_layer_forward(self):
        task = make_induction_task(seed=42)
        params = default_parameters(6, seed=7)
        forward = self_attention(params, task.inputs, causal=True)
        assert forward.tokens == task.length == 5
        assert approx(forward.scale, 1.0 / math.sqrt(6))
        assert batch_loss(params, (task,)) > 0.0

    def test_scale_matches_the_vocabulary_dimension(self):
        """``d_k = vocabulary``（四个投影都是方阵），因此缩放系数是 ``1/√vocab``."""
        params = default_parameters(4, seed=7)
        task = make_induction_task(seed=42, vocabulary=4)
        forward = self_attention(params, task.inputs, causal=True)
        assert approx(forward.scale, 1.0 / math.sqrt(4))
        assert approx(forward.scale, 0.5)
