"""训练回路、冻结与两组对照实验（day078 / M7-D3）.

这一份测试守这一课的两组实验，它们都落在同一句话上：

```text
实验一 compare_encodings      四个刻度：不加 / 正弦 / 可学习（随机）/ 可学习（从正弦出发）
实验二 symmetry_floor_study   三个变体：等变 / 正弦 / 因果掩码——下界的判决书
```

两组实验都用同一个任务、同一批初始参数、同一个优化器配置，**只换一个旋钮**
（day068 起反复强调的纪律：一次只改一个）。

为了让测试跑得快，这里统一用 **20 步**的小训练——它已经足够让损失明显下降、
让对照表出现"越界 / 不越界"的分野，而读数之间的关系与长训练一致。
"""

from __future__ import annotations

import pytest

from smart_research_agent.math_foundations.optim import AdamOptimizer
from smart_research_agent.positional_encoding import (
    ENCODING_LABEL_DESCRIPTIONS,
    ENCODING_LABELS,
    GRAD_TABLE,
    GRADIENT_SOURCE_ANALYTIC,
    INIT_SCALE,
    LABEL_LEARNABLE_FROM_SINE,
    LABEL_LEARNABLE_RANDOM,
    LABEL_NONE,
    LABEL_SINUSOIDAL,
    TRAINABLE_BLOCKS,
    EncodingComparison,
    EncodingComparisonRow,
    NumericError,
    ParameterError,
    PositionalTrainingReport,
    analytic_objective,
    compare_encodings,
    default_parameters,
    flatten_parameters,
    freeze_blocks,
    is_equivariant_table,
    learnable_table,
    orbit_gradients,
    orbit_losses,
    resolve_trainable,
    symmetry_floor_study,
    train_positions,
)
from smart_research_agent.positional_encoding.train import orbit_loss_vector, task_batch
from smart_research_agent.transformer_core.train import (
    DEFAULT_INIT_SCALE as CLASSIC_INIT_SCALE,
)
from smart_research_agent.transformer_core.train import (
    PARAMETER_BLOCKS as CORE_PARAMETER_BLOCKS,
)
from tests.position_samples import (
    LEARNING_RATE,
    LENGTH,
    READOUT_FLOOR,
    TRAIN_STEPS,
    VOCABULARY,
    approx,
    position_parameters,
    readout_task,
    sinusoidal,
    zeros,
)


def _optimizer() -> AdamOptimizer:
    """每一次训练都用**新的**优化器（带状态的实例会把动量带进下一段）."""
    return AdamOptimizer(LEARNING_RATE)


def _train(table=None, *, steps: int = TRAIN_STEPS, trainable=None, causal: bool = False):
    resolved = zeros(LENGTH, VOCABULARY) if table is None else table
    return train_positions(
        position_parameters(VOCABULARY),
        resolved,
        readout_task(),
        optimizer=_optimizer(),
        steps=steps,
        causal=causal,
        trainable=TRAINABLE_BLOCKS[:-1] if trainable is None else trainable,
    )


class TestConstants:
    """四个刻度、五个可训练块与两个跨层常量（各自另立一份并由断言钉住）."""

    def test_encoding_labels(self):
        assert ENCODING_LABELS == (
            LABEL_NONE,
            LABEL_SINUSOIDAL,
            LABEL_LEARNABLE_RANDOM,
            LABEL_LEARNABLE_FROM_SINE,
        )
        assert set(ENCODING_LABELS) == set(ENCODING_LABEL_DESCRIPTIONS)

    def test_trainable_blocks_extend_the_day075_blocks(self):
        """前四块**逐字**等于 day075 的参数块，第五块是本课新增的位置表."""
        assert TRAINABLE_BLOCKS == CORE_PARAMETER_BLOCKS + (GRAD_TABLE,)
        assert TRAINABLE_BLOCKS == ("w_query", "w_key", "w_value", "w_output", "table")

    def test_gradient_source_is_analytic_only(self):
        assert GRADIENT_SOURCE_ANALYTIC == "analytic"

    def test_init_scale_matches_day075(self):
        assert INIT_SCALE == CLASSIC_INIT_SCALE


class TestResolveTrainable:
    """"这一轮训哪几块"：缺省全训、子集保序、非法输入当场拒绝."""

    def test_none_means_everything(self):
        assert resolve_trainable(None) == TRAINABLE_BLOCKS

    def test_subset_is_preserved_in_order(self):
        assert resolve_trainable(["table", "w_value"]) == ("table", "w_value")

    def test_unknown_block_is_rejected(self):
        with pytest.raises(ParameterError, match="不是可训练的参数块"):
            resolve_trainable(["w_bias"])

    def test_duplicate_block_is_rejected(self):
        with pytest.raises(ParameterError, match="重复"):
            resolve_trainable(["w_key", "w_key"])

    def test_empty_selection_is_rejected(self):
        with pytest.raises(ParameterError, match="至少要训练一块"):
            resolve_trainable([])

    def test_inputs_is_never_trainable(self):
        with pytest.raises(ParameterError, match="不是可训练的参数块"):
            resolve_trainable(["inputs"])


class TestFreezeBlocks:
    """"冻结"在数值上的全部含义：把不训练的那几块梯度**置 0**（形状不变）."""

    def _flat_and_shapes(self):
        return flatten_parameters(position_parameters(VOCABULARY), sinusoidal(LENGTH, VOCABULARY))

    def test_freezing_a_block_zeroes_exactly_that_block(self):
        flat, shapes = self._flat_and_shapes()
        trainable = ["table"]
        frozen = freeze_blocks(flat, shapes, trainable)
        projection_size = 4 * VOCABULARY * VOCABULARY
        assert all(value == 0.0 for value in frozen[:projection_size])
        assert any(value != 0.0 for value in frozen[projection_size:])

    def test_freezing_nothing_returns_the_input_unchanged(self):
        flat, shapes = self._flat_and_shapes()
        assert freeze_blocks(flat, shapes, TRAINABLE_BLOCKS) == flat

    def test_shape_table_must_have_five_blocks(self):
        flat, shapes = self._flat_and_shapes()
        with pytest.raises(NumericError, match="5 个"):
            freeze_blocks(flat, shapes[:4], ["table"])

    def test_shape_table_must_account_for_every_number(self):
        flat, _shapes = self._flat_and_shapes()
        with pytest.raises(NumericError, match="元素个数"):
            freeze_blocks(flat, ((VOCABULARY, VOCABULARY),) * 5, ["table"])

    def test_resolve_trainable_is_reused(self):
        flat, shapes = self._flat_and_shapes()
        with pytest.raises(ParameterError):
            freeze_blocks(flat, shapes, ["w_bias"])


class TestIsEquivariantTable:
    """一张表 + 掩码设置是否落在下界适用的那个类里（全零 + 无掩码）."""

    def test_zero_table_without_a_mask_is_equivariant(self):
        assert is_equivariant_table(zeros(LENGTH, VOCABULARY), causal=False) is True

    def test_sinusoidal_table_is_not_equivariant(self):
        assert is_equivariant_table(sinusoidal(LENGTH, VOCABULARY), causal=False) is False

    def test_random_learnable_table_is_not_equivariant(self):
        table = learnable_table(LENGTH, VOCABULARY, initializer="random", seed=7)
        assert is_equivariant_table(table, causal=False) is False

    def test_a_causal_mask_removes_equivariance(self):
        assert is_equivariant_table(zeros(LENGTH, VOCABULARY), causal=True) is False


class TestOrbitGradients:
    """轨道上**平均**的六块梯度：与损失对同一个量求导."""

    def test_length_matches_the_flattened_parameters(self):
        params = position_parameters(VOCABULARY)
        table = sinusoidal(LENGTH, VOCABULARY)
        flat, _shapes = flatten_parameters(params, table)
        assert len(orbit_gradients(params, table, readout_task())) == len(flat)

    def test_is_deterministic(self):
        params = position_parameters(VOCABULARY)
        table = sinusoidal(LENGTH, VOCABULARY)
        first = orbit_gradients(params, table, readout_task())
        second = orbit_gradients(params, table, readout_task())
        assert first == second

    def test_table_gradient_is_non_zero_on_a_zero_table(self):
        """全零表不是"没有表可学"：它的**梯度非零**（表会开始学）."""
        params = position_parameters(VOCABULARY)
        table = zeros(LENGTH, VOCABULARY)
        flat, _shapes = flatten_parameters(params, table)
        gradient = orbit_gradients(params, table, readout_task())
        assert any(value != 0.0 for value in gradient)

    def test_objective_matches_the_orbit_mean_loss(self):
        params = position_parameters(VOCABULARY)
        table = sinusoidal(LENGTH, VOCABULARY)
        task = readout_task()
        flat, _shapes = flatten_parameters(params, table)
        objective = analytic_objective(params, table, task)
        from smart_research_agent.positional_encoding import orbit_mean_loss

        assert approx(objective(flat), orbit_mean_loss(params, table, task), tolerance=1e-12)

    def test_objective_honours_the_causal_flag(self):
        params = position_parameters(VOCABULARY)
        table = sinusoidal(LENGTH, VOCABULARY)
        task = readout_task()
        flat, _shapes = flatten_parameters(params, table)
        assert analytic_objective(params, table, task, causal=True)(flat) != analytic_objective(
            params, table, task
        )(flat)


class TestTraining:
    """训练回路：三列读数必须一起记、一起读."""

    def test_loss_decreases_and_the_report_is_ok(self):
        report = _train()
        assert report.ok is True
        assert report.final_loss < report.initial_loss
        assert report.improvement_ratio > 0.0

    def test_trajectories_have_one_more_entry_than_steps(self):
        report = _train(steps=8)
        assert report.steps == 8
        assert len(report.trajectory) == 9
        assert len(report.accuracies) == 9

    def test_equivariant_run_stays_on_the_floor(self):
        """冻结位置表（全零）的等变训练**不可能**低于下界——低于它就是实现有问题."""
        report = _train()
        assert report.equivariant_class is True
        assert report.final_loss >= READOUT_FLOOR
        assert approx(report.floor, READOUT_FLOOR)

    def test_frozen_table_does_not_move(self):
        report = _train()
        assert report.frozen == (GRAD_TABLE,)
        assert report.table_movement == 0.0

    def test_training_the_table_moves_it(self):
        report = _train(trainable=TRAINABLE_BLOCKS)
        assert report.frozen == ()
        assert report.table_movement > 0.0

    def test_sinusoidal_table_beats_the_floor(self):
        """破对称的一次训练**可以**跑到界下——这是位置信息被用上的证据."""
        report = _train(sinusoidal(LENGTH, VOCABULARY))
        assert report.equivariant_class is False
        assert report.final_loss < READOUT_FLOOR

    def test_schedule_is_honoured(self):
        report = train_positions(
            position_parameters(VOCABULARY),
            zeros(LENGTH, VOCABULARY),
            readout_task(),
            optimizer=_optimizer(),
            steps=3,
            trainable=TRAINABLE_BLOCKS[:-1],
            schedule=lambda step: 0.01,
        )
        assert len(report.trajectory) == 4

    def test_a_non_positive_learning_rate_from_the_schedule_is_rejected(self):
        with pytest.raises(ParameterError, match="不是正的有限数"):
            train_positions(
                position_parameters(VOCABULARY),
                zeros(LENGTH, VOCABULARY),
                readout_task(),
                optimizer=_optimizer(),
                steps=2,
                trainable=TRAINABLE_BLOCKS[:-1],
                schedule=lambda step: 0.0,
            )

    @pytest.mark.parametrize("value", [0, -1, True, 2.5])
    def test_steps_must_be_a_positive_integer(self, value):
        with pytest.raises(ParameterError, match="steps"):
            _train(steps=value)

    def test_epoch_line_reports_the_loss_and_accuracy(self):
        report = _train(steps=3)
        assert "step    0" in report.epoch_line(0)
        assert "损失" in report.epoch_line(1)

    def test_epoch_line_range_is_checked(self):
        report = _train(steps=3)
        with pytest.raises(ParameterError, match="越界"):
            report.epoch_line(99)

    def test_summary_and_serialisation(self):
        report = _train(steps=3)
        assert "冻结" in report.summary_line()
        payload = report.to_dict()
        assert payload["steps"] == 3
        assert payload["floor"] == READOUT_FLOOR
        assert payload["frozen"] == [GRAD_TABLE]

    def test_final_parameters_are_returned(self):
        report = _train(steps=3)
        assert report.final_parameters.parameter_count() == 4 * VOCABULARY * VOCABULARY
        assert isinstance(report.final_table.table[0][0], float)


class TestTrainingReportValidation:
    """报告构造的校验：长度错位、非法来源、两类"不正常"都要被看见."""

    def _report(self):
        return _train(steps=3)

    def _kwargs(self, **overrides):
        report = self._report()
        payload = {
            "task": report.task,
            "causal": report.causal,
            "trajectory": report.trajectory,
            "accuracies": report.accuracies,
            "initial_table": report.initial_table,
            "final_table": report.final_table,
            "final_parameters": report.final_parameters,
            "gradient_source": report.gradient_source,
            "trainable": report.trainable,
        }
        payload.update(overrides)
        return payload

    def test_columns_must_not_shift(self):
        report = self._report()
        with pytest.raises(NumericError, match="长度不一致"):
            PositionalTrainingReport(**self._kwargs(accuracies=report.accuracies[:-1]))

    def test_trajectory_needs_two_readings(self):
        with pytest.raises(NumericError, match="两个读数"):
            PositionalTrainingReport(**self._kwargs(trajectory=(0.1,), accuracies=(0.1,)))

    def test_unknown_gradient_source_is_rejected(self):
        with pytest.raises(ParameterError, match="梯度来源"):
            PositionalTrainingReport(**self._kwargs(gradient_source="numeric"))

    def test_unknown_trainable_block_is_rejected(self):
        with pytest.raises(ParameterError):
            PositionalTrainingReport(**self._kwargs(trainable=("w_bias",)))

    def test_notes_are_stringified(self):
        assert PositionalTrainingReport(**self._kwargs(notes=(1,))).notes == ("1",)

    def test_a_rising_loss_is_not_ok(self):
        report = PositionalTrainingReport(
            **self._kwargs(trajectory=(0.10, 0.20), accuracies=(0.25, 0.25))
        )
        assert report.ok is False

    def test_an_equivariant_run_below_the_floor_is_not_ok(self):
        """全零表 + 无掩码 = 等变类；它跑到 0.10 < 0.125 → 报告必须判"不正常"."""
        report = self._report()
        flat_zero = zeros(LENGTH, VOCABULARY)
        not_ok = PositionalTrainingReport(
            **self._kwargs(
                trajectory=(0.16, 0.10),
                accuracies=(0.25, 0.25),
                initial_table=flat_zero,
                final_table=flat_zero,
                causal=False,
            )
        )
        assert not_ok.equivariant_class is True
        assert not_ok.ok is False
        assert report.ok is True


class TestEncodingComparisonRow:
    """一个"编码刻度"的读数：低于下界、降幅、打印与序列化."""

    def _row(self, **overrides):
        payload = {
            "label": LABEL_NONE,
            "description": ENCODING_LABEL_DESCRIPTIONS[LABEL_NONE],
            "table_kind": "learnable",
            "table_parameters": 0,
            "table_trainable": False,
            "equivariant": True,
            "floor": READOUT_FLOOR,
            "initial_loss": 0.1608,
            "final_loss": 0.1300,
            "accuracy": 0.25,
            "table_movement": 0.0,
        }
        payload.update(overrides)
        return EncodingComparisonRow(**payload)

    def test_below_floor_and_improvement(self):
        assert self._row().below_floor is False
        assert approx(self._row().improvement_ratio, 1.0 - 0.1300 / 0.1608)
        assert self._row(final_loss=0.0800).below_floor is True

    def test_improvement_of_a_zero_initial_loss_is_zero(self):
        assert self._row(initial_loss=0.0).improvement_ratio == 0.0

    def test_summary_and_serialisation(self):
        row = self._row()
        assert "表参数" in row.summary_line()
        assert row.to_dict()["below_floor"] is False

    def test_unknown_label_is_rejected(self):
        with pytest.raises(ParameterError, match="未知的编码标签"):
            self._row(label="magic")

    def test_non_finite_numbers_are_rejected(self):
        with pytest.raises(NumericError, match="final_loss"):
            self._row(final_loss=float("inf"))


class TestEncodingComparison:
    """对照表本身的校验（工厂造不出的表由这几条守住）."""

    def _row(self, label=LABEL_NONE):
        return EncodingComparisonRow(
            label=label,
            description=ENCODING_LABEL_DESCRIPTIONS[label],
            table_kind="learnable",
            table_parameters=0,
            table_trainable=False,
            equivariant=True,
            floor=READOUT_FLOOR,
            initial_loss=0.16,
            final_loss=0.13,
            accuracy=0.25,
            table_movement=0.0,
        )

    def test_at_least_two_rows_are_required(self):
        with pytest.raises(ParameterError, match="至少要有两行"):
            EncodingComparison(task=readout_task(), rows=(self._row(),), steps=10)

    def test_duplicate_labels_are_rejected(self):
        with pytest.raises(NumericError, match="重复的标签"):
            EncodingComparison(
                task=readout_task(), rows=(self._row(), self._row()), steps=10
            )

    def test_steps_must_be_positive(self):
        with pytest.raises(ParameterError, match="steps"):
            EncodingComparison(
                task=readout_task(),
                rows=(self._row(), self._row(LABEL_SINUSOIDAL)),
                steps=0,
            )

    def test_notes_are_stringified(self):
        comparison = EncodingComparison(
            task=readout_task(),
            rows=(self._row(), self._row(LABEL_SINUSOIDAL)),
            steps=10,
            notes=(1,),
        )
        assert comparison.notes == ("1",)


class TestCompareEncodings:
    """实验一：四个刻度、只换一个旋钮."""

    def _comparison(self, *, steps: int = TRAIN_STEPS):
        return compare_encodings(
            position_parameters(VOCABULARY),
            readout_task(),
            optimizer_factory=_optimizer,
            steps=steps,
        )

    def test_rows_follow_the_four_labels(self):
        comparison = self._comparison()
        assert tuple(row.label for row in comparison.rows) == ENCODING_LABELS
        assert comparison.steps == TRAIN_STEPS

    def test_table_parameter_counts(self):
        comparison = self._comparison()
        counts = {row.label: row.table_parameters for row in comparison.rows}
        assert counts[LABEL_NONE] == 0
        assert counts[LABEL_SINUSOIDAL] == 0
        assert counts[LABEL_LEARNABLE_RANDOM] == LENGTH * VOCABULARY
        assert counts[LABEL_LEARNABLE_FROM_SINE] == LENGTH * VOCABULARY

    def test_only_the_zero_table_row_is_equivariant(self):
        comparison = self._comparison()
        assert {row.label for row in comparison.rows if row.equivariant} == {LABEL_NONE}

    def test_equivariant_row_never_crosses_the_floor(self):
        comparison = self._comparison()
        assert comparison.equivariant_violations == ()
        assert comparison.ok is True
        by_label = {row.label: row for row in comparison.rows}
        assert by_label[LABEL_NONE].below_floor is False

    def test_position_aware_rows_cross_the_floor(self):
        comparison = self._comparison()
        by_label = {row.label: row for row in comparison.rows}
        assert by_label[LABEL_SINUSOIDAL].below_floor is True
        assert by_label[LABEL_SINUSOIDAL].final_loss < by_label[LABEL_NONE].final_loss

    def test_table_lines_and_summary(self):
        comparison = self._comparison()
        assert len(comparison.table_lines()) == len(ENCODING_LABELS) + 2
        assert "下界" in comparison.summary_line()
        assert comparison.floor == READOUT_FLOOR

    def test_serialisation_and_notes(self):
        comparison = self._comparison(steps=3)
        payload = comparison.to_dict()
        assert payload["ok"] is True
        assert len(payload["rows"]) == 4
        assert any("一次只改一个" in item or "唯一的旋钮" in item for item in comparison.notes)

    def test_steps_must_be_positive(self):
        with pytest.raises(ParameterError, match="steps"):
            self._comparison(steps=0)

    def test_each_row_is_reproducible(self):
        first = self._comparison(steps=3)
        second = self._comparison(steps=3)
        assert first.rows == second.rows


class TestSymmetryFloorStudy:
    """实验二：三个变体的下界判决书（等变 / 正弦 / 因果掩码）."""

    def _study(self, *, steps: int = TRAIN_STEPS):
        return symmetry_floor_study(
            position_parameters(VOCABULARY),
            readout_task(),
            optimizer_factory=_optimizer,
            steps=steps,
        )

    def test_three_variants_are_reported(self):
        study = self._study()
        assert len(study.variants) == 3
        assert tuple(item.name for item in study.variants) == (
            "no_mask_no_positions",
            "no_mask_sinusoidal",
            "causal_no_positions",
        )

    def test_floor_is_tight_and_the_verdict_holds(self):
        study = self._study()
        assert study.floor == READOUT_FLOOR
        assert study.witness == READOUT_FLOOR
        assert study.tight is True
        assert study.ok is True

    def test_equivariant_variant_stays_on_the_floor(self):
        study = self._study()
        (equivariant,) = study.equivariant_variants
        assert equivariant.name == "no_mask_no_positions"
        assert equivariant.final_loss >= READOUT_FLOOR
        assert study.floor_violations == ()

    def test_both_broken_symmetry_routes_cross_the_floor(self):
        """位置编码与因果掩码**两条路**都能越过下界——它正是"另一条路"的对照."""
        study = self._study()
        assert {item.name for item in study.crossers} == {
            "no_mask_sinusoidal",
            "causal_no_positions",
        }
        for item in study.crossers:
            assert item.final_loss < READOUT_FLOOR

    def test_table_lines_and_serialisation(self):
        study = self._study(steps=3)
        assert len(study.table_lines()) == 5
        payload = study.to_dict()
        assert payload["tight"] is True
        assert len(payload["variants"]) == 3

    def test_steps_must_be_positive(self):
        with pytest.raises(ParameterError, match="steps"):
            self._study(steps=0)

    def test_summary_line(self):
        assert "位置选择下界" in self._study(steps=3).summary_line()


class TestTaskHelpers:
    """把轨道变成批、以及两条"同一批损失"的入口对账."""

    def test_task_batch_pairs_every_shift_with_its_target(self):
        batch = task_batch(readout_task())
        assert len(batch) == LENGTH
        expected = tuple((item.inputs, item.target) for item in readout_task().orbit())
        assert batch == expected

    def test_orbit_loss_vector_matches_orbit_losses(self):
        params = position_parameters(VOCABULARY)
        table = sinusoidal(LENGTH, VOCABULARY)
        task = readout_task()
        assert orbit_loss_vector(params, table, task) == orbit_losses(params, table, task)

    def test_default_parameters_is_re_exported(self):
        """训练的初始参数来自 day075 的 ``default_parameters``（本包只转发）."""
        assert default_parameters(4).parameter_count() == 4 * 4 * 4
