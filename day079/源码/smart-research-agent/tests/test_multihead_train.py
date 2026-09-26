"""训练回路与"同参数量对照"（day076 / M7-D2）.

这一份测试的最后两组（``TestCompareHeads`` / ``TestParameterCountParity``）守的是
这一课最重要的一句工程结论：

```text
heads 只决定"维度被切成几段" → 参数量与 heads **无关**
因此 H=1/2/3 的对照是**同参数量对照**：
它比较的是"注意力被组织成几份"，而不是"参数多了多少"。
```

为了让测试跑得快，这里用 **30~60 步**的小训练——它已经足够让损失明显下降
（day075 的读数表明第 8 步命中率就到 100%），而读数之间的关系与长训练一致。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.math_foundations.optim import (
    AdamOptimizer,
    SGDOptimizer,
    unflatten_matrices,
)
from smart_research_agent.multi_head.errors import (
    NumericError,
    ParameterError,
    PartitionError,
    ShapeError,
)
from smart_research_agent.multi_head.layers import multi_head_attention
from smart_research_agent.multi_head.train import (
    DEFAULT_INIT_SCALE,
    MULTIHEAD_GRADIENT_SOURCES,
    MULTIHEAD_PARAMETER_BLOCKS,
    HeadsComparison,
    HeadsComparisonRow,
    MultiHeadTrainingReport,
    analytic_objective,
    batch_accuracy,
    batch_gradients,
    batch_loss,
    batch_mean_disagreement,
    batch_mean_peak_weight,
    compare_heads,
    matrix_totals,
    train_multi_head,
)
from smart_research_agent.multi_head.types import MultiHeadShape
from smart_research_agent.multi_head.verify import head_disagreement
from smart_research_agent.transformer_core.layers import (
    masked_mean_squared_error,
    row_argmax_hits,
)
from smart_research_agent.transformer_core.train import (
    DEFAULT_INIT_SCALE as CLASSIC_INIT_SCALE,
)
from smart_research_agent.transformer_core.train import (
    GRADIENT_SOURCES,
    PARAMETER_BLOCKS,
    make_induction_task,
    train_attention,
)
from smart_research_agent.transformer_core.train import (
    batch_gradients as classic_batch_gradients,
)
from tests.multihead_samples import (
    HEAD_COUNTS,
    induction_tasks,
    toy_parameters,
)


class TestConstants:
    """三个常量与 day075 同值——但各自另立一份，并由断言钉在一起."""

    def test_init_scale_matches_day075(self):
        assert DEFAULT_INIT_SCALE == CLASSIC_INIT_SCALE

    def test_parameter_blocks_match_day075(self):
        assert MULTIHEAD_PARAMETER_BLOCKS == PARAMETER_BLOCKS
        assert MULTIHEAD_PARAMETER_BLOCKS == ("w_query", "w_key", "w_value", "w_output")

    def test_gradient_sources_match_day075(self):
        assert MULTIHEAD_GRADIENT_SOURCES == GRADIENT_SOURCES


class TestBatchReadouts:
    """批量读数：每一项都必须能被单独复算出来（否则它只是一个可疑的数）."""

    def test_loss_is_the_mean_of_the_per_task_losses(self):
        params = toy_parameters(6)
        tasks = induction_tasks(3)
        expected = math.fsum(
            masked_mean_squared_error(
                multi_head_attention(params, task.inputs, heads=2, causal=True).output,
                task.target,
                task.supervised,
            )
            for task in tasks
        ) / len(tasks)
        assert batch_loss(params, tasks, heads=2) == pytest.approx(expected, abs=1e-15)

    def test_accuracy_is_the_mean_of_the_per_task_hits(self):
        params = toy_parameters(6)
        tasks = induction_tasks(3)
        expected = math.fsum(
            row_argmax_hits(
                multi_head_attention(params, task.inputs, heads=2, causal=True).output,
                task.target,
                task.supervised,
            )
            for task in tasks
        ) / len(tasks)
        assert batch_accuracy(params, tasks, heads=2) == expected

    def test_peak_weight_is_a_probability(self):
        value = batch_mean_peak_weight(toy_parameters(6), induction_tasks(2), heads=3)
        assert 0.0 < value <= 1.0

    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    def test_disagreement_is_the_mean_of_the_per_task_values(self, heads):
        params = toy_parameters(6)
        tasks = induction_tasks(2)
        expected = (
            0.0
            if heads == 1
            else math.fsum(
                head_disagreement(
                    multi_head_attention(params, task.inputs, heads=heads, causal=True)
                ).mean_total_variation
                for task in tasks
            )
            / len(tasks)
        )
        assert batch_mean_disagreement(params, tasks, heads=heads) == pytest.approx(
            expected, abs=1e-15
        )

    def test_disagreement_of_a_single_head_is_zero(self):
        value = batch_mean_disagreement(toy_parameters(6), induction_tasks(1), heads=1)
        assert value == 0.0

    @pytest.mark.parametrize(
        "value,error",
        [(0, ParameterError), (-1, ParameterError), (5, PartitionError)],
    )
    def test_heads_must_divide_the_dimensions(self, value, error):
        """``heads <= 0`` 是普通的参数越界，``heads = 5`` 是**划分**失败——两族不同."""
        with pytest.raises(error):
            batch_loss(toy_parameters(6), induction_tasks(1), heads=value)

    def test_empty_batch_is_rejected(self):
        with pytest.raises(ShapeError):
            batch_loss(toy_parameters(6), (), heads=2)

    def test_mixed_vocabularies_are_rejected(self):
        tasks = (induction_tasks(1)[0], make_induction_task(seed=5, vocabulary=4))
        with pytest.raises(ShapeError, match="词表"):
            batch_loss(toy_parameters(6), tasks, heads=2)

    def test_parameters_must_be_attention_params(self):
        with pytest.raises(ShapeError):
            batch_loss("no", induction_tasks(1), heads=1)  # type: ignore[arg-type]


class TestTrainableBlocks:
    """"冻结"在数值上的全部含义：把不训练的那几块梯度**置 0**."""

    def test_freezing_zeroes_exactly_those_blocks(self):
        params = toy_parameters(6)
        frozen = batch_gradients(params, induction_tasks(2), heads=2, trainable=["w_query"])
        blocks = unflatten_matrices(frozen, params.flatten()[1])
        assert any(value != 0.0 for row in blocks[0] for value in row)
        for block in blocks[1:]:
            assert all(value == 0.0 for row in block for value in row)

    def test_unknown_block_is_rejected(self):
        with pytest.raises(ParameterError):
            batch_gradients(
                toy_parameters(6), induction_tasks(1), heads=2, trainable=["w_bias"]
            )

    def test_duplicate_block_is_rejected(self):
        with pytest.raises(ParameterError):
            batch_gradients(
                toy_parameters(6), induction_tasks(1), heads=2, trainable=["w_key", "w_key"]
            )

    def test_empty_trainable_is_rejected(self):
        with pytest.raises(ParameterError, match="至少要训练一块"):
            batch_gradients(toy_parameters(6), induction_tasks(1), heads=2, trainable=[])

    def test_heads_one_matches_day075_freeze_semantics(self):
        params = toy_parameters(6)
        tasks = induction_tasks(2)
        mine = batch_gradients(params, tasks, heads=1, trainable=["w_value", "w_output"])
        theirs = classic_batch_gradients(params, tasks, trainable=["w_value", "w_output"])
        assert max(abs(a - b) for a, b in zip(mine, theirs, strict=True)) < 1e-12


class TestTraining:
    """训练回路：四列读数必须一起记、一起读."""

    def _report(self, heads: int = 2, *, steps: int = 40, **kwargs):
        return train_multi_head(
            toy_parameters(6),
            induction_tasks(4),
            heads=heads,
            optimizer=AdamOptimizer(0.05),
            steps=steps,
            **kwargs,
        )

    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    def test_loss_decreases(self, heads):
        report = self._report(heads)
        assert report.ok
        assert report.final_loss < report.initial_loss
        assert report.improvement_ratio > 0.9

    @pytest.mark.parametrize("heads", HEAD_COUNTS)
    def test_four_columns_have_the_same_length(self, heads):
        report = self._report(heads, steps=8)
        expected = report.steps + 1
        assert len(report.trace.losses) == expected
        assert len(report.accuracies) == expected
        assert len(report.peak_weights) == expected
        assert len(report.disagreements) == expected

    def test_learning_rates_trail_the_losses_by_one(self):
        report = self._report(2, steps=5)
        assert len(report.trace.learning_rates) == report.steps
        assert report.trace.learning_rates == (0.05,) * 5

    def test_single_head_disagreement_is_zero_everywhere(self):
        report = self._report(1, steps=20)
        assert all(value == 0.0 for value in report.disagreements)

    def test_two_heads_develop_diversity(self):
        """两个头最终**确实**学到了不一样的分布（否则多头就是白买）."""
        report = self._report(2, steps=60)
        assert report.final_disagreement > 0.02
        assert report.final_disagreement > report.disagreements[0]

    def test_parameter_count_matches_the_formula(self):
        report = self._report(3, steps=3)
        assert report.parameter_count == 4 * 6 * 6

    def test_shape_reports_the_heads(self):
        report = self._report(3, steps=3)
        assert report.shape.heads == 3
        assert report.shape.head_dim == 2

    def test_frozen_blocks_are_reported(self):
        report = train_multi_head(
            toy_parameters(6),
            induction_tasks(2),
            heads=2,
            optimizer=AdamOptimizer(0.05),
            steps=20,
            trainable=["w_value", "w_output"],
        )
        assert report.frozen == ("w_query", "w_key")
        assert report.trainable == ("w_value", "w_output")

    def test_freezing_attention_weights_keeps_the_pattern_frozen(self):
        """冻结 q/k 时**注意力模式**完全不变（只有 value 路径在动）."""
        report = train_multi_head(
            toy_parameters(6),
            induction_tasks(4),
            heads=2,
            optimizer=AdamOptimizer(0.05),
            steps=40,
            trainable=["w_value", "w_output"],
        )
        assert report.final_loss < report.initial_loss
        assert report.final_disagreement == report.disagreements[0]
        assert report.final_peak_weight == pytest.approx(report.initial_peak_weight)

    def test_epoch_line(self):
        report = self._report(2, steps=3)
        assert "step    0" in report.epoch_line(0)
        assert "0.050000" in report.epoch_line(1)
        assert report.epoch_line(3)

    def test_epoch_line_range(self):
        report = self._report(2, steps=3)
        with pytest.raises(ParameterError):
            report.epoch_line(4)
        with pytest.raises(ParameterError):
            report.epoch_line(-1)

    def test_summary_line(self):
        line = self._report(2, steps=3).summary_line()
        assert "heads=2" in line
        assert "头间差异" in line

    def test_to_dict(self):
        payload = self._report(3, steps=3).to_dict()
        assert payload["heads"] == 3
        assert payload["parameter_count"] == 144
        assert len(payload["disagreements"]) == 4
        assert payload["frozen"] == []

    @pytest.mark.parametrize("source", ["analytic", "numeric"])
    def test_both_gradient_sources_train(self, source):
        report = self._report(2, steps=6, source=source)
        assert report.gradient_source == source
        assert report.ok

    def test_steps_must_be_positive(self):
        with pytest.raises(ParameterError):
            self._report(2, steps=0)

    def test_unknown_source_is_rejected(self):
        with pytest.raises(ParameterError):
            self._report(2, steps=2, source="magic")

    def test_schedule_must_stay_positive(self):
        with pytest.raises(ParameterError, match="非正学习率"):
            self._report(2, steps=3, schedule=lambda step: 0.0)

    def test_schedule_is_honoured(self):
        report = self._report(2, steps=4, schedule=lambda step: 0.01 * step)
        assert report.trace.learning_rates == (0.01, 0.02, 0.03, 0.04)

    def test_sgd_also_converges(self):
        report = train_multi_head(
            toy_parameters(6),
            induction_tasks(4),
            heads=2,
            optimizer=SGDOptimizer(0.5),
            steps=40,
        )
        assert report.ok


class TestTrainingReportValidation:
    """报告构造的校验：长度错位、非法来源、单头非零差异都要拒绝."""

    def _report(self):
        return train_multi_head(
            toy_parameters(6),
            induction_tasks(2),
            heads=2,
            optimizer=AdamOptimizer(0.05),
            steps=3,
        )

    def _kwargs(self, **overrides):
        report = self._report()
        payload = {
            "trace": report.trace,
            "final_parameters": report.final_parameters,
            "heads": report.heads,
            "peak_weights": report.peak_weights,
            "accuracies": report.accuracies,
            "disagreements": report.disagreements,
        }
        payload.update(overrides)
        return payload

    def test_heads_must_be_positive(self):
        with pytest.raises(ParameterError):
            MultiHeadTrainingReport(**self._kwargs(heads=0))

    @pytest.mark.parametrize("field", ["peak_weights", "accuracies", "disagreements"])
    def test_columns_must_not_shift(self, field):
        kwargs = self._kwargs()
        kwargs[field] = tuple(kwargs[field])[:-1]
        with pytest.raises(NumericError, match="长度必须一致"):
            MultiHeadTrainingReport(**kwargs)

    def test_unknown_source_is_rejected(self):
        with pytest.raises(ParameterError):
            MultiHeadTrainingReport(**self._kwargs(gradient_source="magic"))

    def test_single_head_must_have_zero_disagreement(self):
        """一个非零值意味着这个读数统计了不该统计的东西——当场拒绝."""
        kwargs = self._kwargs(heads=1)
        kwargs["disagreements"] = (0.0, 0.0, 0.0, 0.5)
        with pytest.raises(NumericError, match="恒为 0.0"):
            MultiHeadTrainingReport(**kwargs)

    def test_unknown_trainable_block_is_rejected(self):
        with pytest.raises(ParameterError):
            MultiHeadTrainingReport(**self._kwargs(trainable=("w_bias",)))

    def test_notes_are_stringified(self):
        report = MultiHeadTrainingReport(**self._kwargs(notes=(1,)))
        assert report.notes == ("1",)


class TestAnalyticObjective:
    """调试入口：把批量损失包成一个只吃压平参数的目标函数."""

    def test_matches_batch_loss(self):
        params = toy_parameters(6)
        tasks = induction_tasks(2)
        objective = analytic_objective(tasks, heads=2)
        flat, _shapes = params.flatten()
        assert objective(flat) == pytest.approx(
            batch_loss(params, tasks, heads=2), abs=1e-15
        )

    def test_heads_must_be_positive(self):
        with pytest.raises(ParameterError):
            analytic_objective(induction_tasks(1), heads=0)

    def test_empty_batch_is_rejected(self):
        with pytest.raises(ShapeError):
            analytic_objective((), heads=2)


class TestCompareHeads:
    """同参数量对照表：这一课的实验产物."""

    def _comparison(self, heads_list=(1, 2, 3), *, steps: int = 40):
        return compare_heads(
            toy_parameters(6),
            induction_tasks(4),
            heads_list=heads_list,
            optimizer_factory=lambda: AdamOptimizer(0.05),
            steps=steps,
        )

    def test_rows_follow_the_requested_heads(self):
        comparison = self._comparison()
        assert tuple(row.heads for row in comparison.rows) == (1, 2, 3)

    def test_all_heads_solve_the_task(self):
        comparison = self._comparison(steps=60)
        assert all(row.final_accuracy == 1.0 for row in comparison.rows)
        assert all(row.final_loss < row.initial_loss for row in comparison.rows)

    def test_disagreement_is_zero_for_a_single_head_only(self):
        comparison = self._comparison()
        by_heads = {row.heads: row for row in comparison.rows}
        assert by_heads[1].final_disagreement == 0.0
        assert by_heads[2].final_disagreement > 0.02
        assert by_heads[3].final_disagreement > 0.0

    def test_parameter_counts_are_identical(self):
        """**同参数量**：heads 只改变"维度被切成几段"，不改变参数多少."""
        comparison = self._comparison()
        assert comparison.comparable is True
        assert comparison.parameter_counts == (144, 144, 144)

    def test_best_and_most_diverse_rows(self):
        comparison = self._comparison()
        assert comparison.best_final_loss.heads in (1, 2, 3)
        assert comparison.most_diverse.heads in (1, 2, 3)

    def test_table_lines(self):
        lines = self._comparison().table_lines()
        assert len(lines) == 5  # 表头 + 分隔线 + 三行
        assert "heads" in lines[0]
        assert all(line.startswith("  ") for line in lines)

    def test_summary_line(self):
        line = self._comparison().summary_line()
        assert "3 行对照" in line
        assert "参数量 144" in line

    def test_to_dict(self):
        payload = self._comparison().to_dict()
        assert payload["comparable"] is True
        assert len(payload["rows"]) == 3
        assert payload["rows"][0]["heads"] == 1

    def test_notes_explain_it_is_one_observation(self):
        comparison = self._comparison()
        assert any("一次观测" in item for item in comparison.notes)
        assert any("同参数量" in item for item in comparison.notes)

    def test_single_heads_value_is_rejected(self):
        """只有一行的"对照"回答不了任何问题——当场拒绝."""
        with pytest.raises(ParameterError, match="至少要两个"):
            self._comparison(heads_list=(2,), steps=2)

    def test_indivisible_heads_is_rejected(self):
        with pytest.raises(PartitionError):
            self._comparison(heads_list=(2, 4), steps=2)

    def test_row_serialises(self):
        payload = self._comparison(heads_list=(1, 2), steps=3).rows[0].to_dict()
        assert payload["heads"] == 1
        assert payload["parameter_count"] == 144
        assert payload["steps"] == 3


class TestComparisonValidation:
    """对照表与行的构造校验（工厂造不出的表由这几条守住）."""

    def _row(self, **overrides):
        payload = {
            "heads": 1,
            "parameter_count": 144,
            "initial_loss": 0.2,
            "final_loss": 0.1,
            "improvement_ratio": 0.5,
            "initial_accuracy": 0.0,
            "final_accuracy": 1.0,
            "initial_peak_weight": 0.2,
            "final_peak_weight": 0.4,
            "final_disagreement": 0.0,
            "steps": 10,
        }
        payload.update(overrides)
        return HeadsComparisonRow(**payload)

    def test_comparison_requires_two_rows(self):
        with pytest.raises(ParameterError, match="至少要两行"):
            HeadsComparison(rows=(self._row(),), steps=10, gradient_source="analytic")

    def test_steps_must_be_positive(self):
        with pytest.raises(ParameterError):
            HeadsComparison(
                rows=(self._row(heads=1), self._row(heads=2)),
                steps=0,
                gradient_source="analytic",
            )

    @pytest.mark.parametrize("overrides", [{"heads": 0}, {"parameter_count": 0}])
    def test_row_bounds(self, overrides):
        with pytest.raises(ParameterError):
            self._row(**overrides)

    def test_asymmetric_table_is_flagged_by_the_property(self):
        """``comparable`` 是"这份对照表可不可比"的唯一判据（工厂造不出不对称的表）."""
        comparison = HeadsComparison(
            rows=(self._row(heads=1), self._row(heads=2, parameter_count=145)),
            steps=10,
            gradient_source="analytic",
        )
        assert comparison.comparable is False
        assert comparison.parameter_counts == (144, 145)

    def test_symmetric_table_is_comparable(self):
        comparison = HeadsComparison(
            rows=(self._row(heads=1), self._row(heads=2)),
            steps=10,
            gradient_source="analytic",
        )
        assert comparison.comparable is True
        assert comparison.best_final_loss.heads == 1

    def test_notes_are_stringified(self):
        comparison = HeadsComparison(
            rows=(self._row(heads=1), self._row(heads=2)),
            steps=10,
            gradient_source="analytic",
            notes=(1,),
        )
        assert comparison.notes == ("1",)


class TestParameterCountParity:
    """参数量与 heads 无关——这条结论的三个可执行版本."""

    @pytest.mark.parametrize("heads", HEAD_COUNTS + (6,))
    def test_parameter_count_only_depends_on_the_vocabulary(self, heads):
        params = toy_parameters(6)
        shape = MultiHeadShape(params.shape, heads)
        assert shape.heads == heads
        assert params.parameter_count() == 4 * 6 * 6

    def test_vocabulary_changes_the_count(self):
        assert toy_parameters(4).parameter_count() == 4 * 4 * 4

    def test_matrix_totals(self):
        assert matrix_totals(((1.0, 2.0), (3.0, 4.0))) == 10.0

    def test_matrix_totals_of_a_single_row(self):
        assert matrix_totals(((0.5, 0.25),)) == pytest.approx(0.75)


def test_single_head_report_equals_day075_training():
    """heads=1 的训练轨迹必须与 day075 的实现**逐位相同**（同种子、同优化器）."""
    params = toy_parameters(6)
    tasks = induction_tasks(4)
    mine = train_multi_head(
        params,
        tasks,
        heads=1,
        optimizer=AdamOptimizer(0.05),
        steps=5,
    )
    theirs = train_attention(params, tasks, optimizer=AdamOptimizer(0.05), steps=5)
    assert mine.trace.losses == theirs.trace.losses
    assert mine.accuracies == theirs.accuracies
    assert mine.peak_weights == theirs.peak_weights
    assert mine.final_parameters.flatten()[0] == theirs.final_parameters.flatten()[0]
