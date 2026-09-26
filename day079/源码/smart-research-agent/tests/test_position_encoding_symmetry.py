"""位置选择任务、轨道、下界、见证与三个变体（day078 / M7-D3）.

这一份测试守这一课唯一"值钱"的结论——**一个可证且紧的下界**：

```text
等变模型（无掩码 + 不加位置编码）在位移轨道上的平均损失 >= (n − 1) / (n·V)
本课样本 n = 4、V = 6 时它是 3/24 = 0.125
而它**是紧的**：袋子预测器（每一行输出袋子分布）恰好达到它，且它本身也是等变的
```

"等变模型低于下界"在数学上不可能——因此那是一条**实现有问题**的读数，
而不是"模型很聪明"。破对称的变体（正弦表 / 因果掩码）不受它约束，
"跑到界下"才是位置信息真的被用上的证据。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.positional_encoding import (
    FLOOR_TOLERANCE,
    VARIANT_CAUSAL,
    VARIANT_DESCRIPTIONS,
    VARIANT_EQUIVARIANT,
    VARIANT_SINUSOIDAL,
    FloorVariant,
    NumericError,
    ParameterError,
    ReadoutTask,
    ShapeError,
    SymmetryFloorReport,
    assemble_floor_report,
    bag_predictor,
    check_task_is_a_position_readout,
    floor_witness,
    make_readout_task,
    orbit_accuracy,
    orbit_floor,
    orbit_loss_spread,
    orbit_losses,
    orbit_mean_loss,
    positional_forward,
    sample_positions,
)
from smart_research_agent.positional_encoding.symmetry import _orbit_mean_loss_of_outputs
from tests.position_samples import (
    BAG_PREDICTOR_ROW,
    LENGTH,
    READOUT_FLOOR,
    READOUT_ORBIT_TOKENS,
    READOUT_TOKENS,
    VOCABULARY,
    approx,
    position_parameters,
    readout_bag_row,
    readout_orbit,
    readout_task,
    sinusoidal,
    zeros,
)


def _variant(**overrides) -> FloorVariant:
    """一个"等变 + 停在界上"的变体（只用来做报告层面的对照，不训练）."""
    payload = {
        "name": VARIANT_EQUIVARIANT,
        "description": VARIANT_DESCRIPTIONS[VARIANT_EQUIVARIANT],
        "equivariant": True,
        "causal": False,
        "table_kind": None,
        "floor": READOUT_FLOOR,
        "initial_loss": 0.1608,
        "final_loss": 0.1300,
        "accuracy": 0.25,
        "trajectory": (0.1608, 0.1300),
    }
    payload.update(overrides)
    return FloorVariant(**payload)


class TestReadoutTask:
    """一条基序列：写死的 token、手算的轨道与下界."""

    def test_base_sequence_is_hand_visible(self):
        assert readout_task().tokens == READOUT_TOKENS

    def test_orbit_is_the_cycle_of_shifts(self):
        assert tuple(item.tokens for item in readout_orbit()) == READOUT_ORBIT_TOKENS

    def test_readout_target_is_the_zeroth_token_in_every_row(self):
        task = readout_task()
        head = task.tokens[0]
        assert task.target == tuple(
            tuple(1.0 if column == head else 0.0 for column in range(VOCABULARY))
            for _ in range(LENGTH)
        )
        assert check_task_is_a_position_readout(task) == READOUT_TOKENS

    def test_inputs_are_one_hot_rows(self):
        task = readout_task()
        for token, row in zip(task.tokens, task.inputs, strict=True):
            assert row == tuple(
                1.0 if column == token else 0.0 for column in range(VOCABULARY)
            )

    def test_every_row_is_supervised(self):
        assert readout_task().supervised == (0, 1, 2, 3)

    def test_shift_wraps_around_the_length(self):
        task = readout_task()
        assert task.shifted(LENGTH).tokens == task.tokens
        assert task.shifted(1).tokens == READOUT_ORBIT_TOKENS[1]
        assert task.shifted(2 * LENGTH).tokens == task.tokens

    def test_shift_validates_the_offset(self):
        with pytest.raises(ParameterError, match="必须是整数"):
            readout_task().shifted(True)
        with pytest.raises(ParameterError, match="必须 >= 0"):
            readout_task().shifted(-1)

    def test_describe_and_to_dict(self):
        task = readout_task()
        assert "位置选择任务" in task.describe()
        assert "基序列 (1, 0, 3, 2)" in task.describe()
        payload = task.to_dict()
        assert payload["tokens"] == [1, 0, 3, 2]
        assert len(payload["orbit"]) == LENGTH

    def test_length_must_match_the_token_count(self):
        with pytest.raises(ShapeError, match="不一致"):
            ReadoutTask(vocabulary=VOCABULARY, length=3, tokens=(1, 0))

    def test_tokens_must_be_distinct(self):
        with pytest.raises(ShapeError, match="互不相同"):
            ReadoutTask(vocabulary=VOCABULARY, length=2, tokens=(1, 1))

    def test_tokens_must_be_integers(self):
        with pytest.raises(ShapeError, match="必须是整数"):
            ReadoutTask(vocabulary=6, length=4, tokens=(1.5, 0.5, 2.5, 3.5))

    def test_booleans_are_not_integers(self):
        with pytest.raises(ShapeError, match="必须是整数"):
            ReadoutTask(vocabulary=6, length=4, tokens=(True, 2, 3, 4))

    def test_tokens_must_stay_inside_the_vocabulary(self):
        with pytest.raises(ShapeError, match="超出"):
            ReadoutTask(vocabulary=VOCABULARY, length=4, tokens=(1, 0, 3, 9))


class TestMakeReadoutTask:
    """构造入口：缺省值写死，越界的长度/词表当场拒绝."""

    def test_defaults_match_the_sample(self):
        assert make_readout_task().tokens == READOUT_TOKENS

    def test_is_deterministic(self):
        first = make_readout_task(seed=42, vocabulary=VOCABULARY, length=LENGTH)
        second = make_readout_task(seed=42, vocabulary=VOCABULARY, length=LENGTH)
        assert first.tokens == second.tokens

    @pytest.mark.parametrize("value", ["6", True, 1.5])
    def test_vocabulary_must_be_an_integer(self, value):
        with pytest.raises(ParameterError, match="vocabulary 必须是整数"):
            make_readout_task(vocabulary=value)

    def test_vocabulary_must_be_at_least_two(self):
        with pytest.raises(ParameterError, match="vocabulary 必须 >= 2"):
            make_readout_task(vocabulary=1)

    def test_length_must_be_an_integer(self):
        with pytest.raises(ParameterError, match="length 必须是整数"):
            make_readout_task(length="4")

    def test_length_must_be_at_least_two(self):
        """长度为 1 时轨道只有一个元素，下界退化成 0（那一条就没有内容了）."""
        with pytest.raises(ParameterError, match="length 必须 >= 2"):
            make_readout_task(length=1)

    def test_length_must_not_exceed_the_vocabulary(self):
        with pytest.raises(ParameterError, match="length 必须 <= vocabulary"):
            make_readout_task(vocabulary=4, length=5)

    def test_seed_must_be_an_integer(self):
        with pytest.raises(ParameterError, match="seed"):
            make_readout_task(seed="x")


class TestOrbitReadouts:
    """轨道上的四个读数：损失、极差、命中率、平均损失."""

    def _params(self):
        return position_parameters(VOCABULARY)

    def test_losses_have_one_entry_per_shift(self):
        losses = orbit_losses(self._params(), sinusoidal(LENGTH, VOCABULARY), readout_task())
        assert len(losses) == LENGTH
        assert all(math.isfinite(value) and value > 0.0 for value in losses)

    def test_losses_match_the_per_shift_forward(self):
        params = self._params()
        table = sinusoidal(LENGTH, VOCABULARY)
        task = readout_task()
        expected = tuple(
            positional_forward(params, table, item.inputs, target=item.target).loss
            for item in task.orbit()
        )
        assert orbit_losses(params, table, task) == expected

    def test_mean_loss_is_the_arithmetic_mean(self):
        params = self._params()
        table = sinusoidal(LENGTH, VOCABULARY)
        task = readout_task()
        losses = orbit_losses(params, table, task)
        assert orbit_mean_loss(params, table, task) == math.fsum(losses) / len(losses)

    def test_equivariant_orbit_mean_loss_respects_the_floor(self):
        """等变变体（全零表）的轨道平均损失必须 **>= 0.125**——低于它说明实现有问题."""
        value = orbit_mean_loss(self._params(), zeros(LENGTH, VOCABULARY), readout_task())
        assert value >= READOUT_FLOOR - FLOOR_TOLERANCE
        assert approx(value, 0.1608, tolerance=1e-2)

    def test_broken_symmetry_moves_the_loss(self):
        params = self._params()
        task = readout_task()
        equivariant = orbit_mean_loss(params, zeros(LENGTH, VOCABULARY), task)
        injected = orbit_mean_loss(params, sinusoidal(LENGTH, VOCABULARY), task)
        assert injected != equivariant

    def test_loss_spread_is_non_negative(self):
        assert orbit_loss_spread(self._params(), zeros(LENGTH, VOCABULARY), readout_task()) > 0.0

    def test_accuracy_is_a_fraction(self):
        params = self._params()
        task = readout_task()
        value = orbit_accuracy(params, zeros(LENGTH, VOCABULARY), task)
        assert 0.0 <= value <= 1.0
        assert value == 0.25  # 四个位移里命中一个

    def test_causal_accuracy_is_not_worse(self):
        params = self._params()
        task = readout_task()
        causal = orbit_accuracy(params, zeros(LENGTH, VOCABULARY), task, causal=True)
        assert causal >= orbit_accuracy(params, zeros(LENGTH, VOCABULARY), task)

    def test_sample_positions_returns_one_start(self):
        assert sample_positions(1) == (0,)

    @pytest.mark.parametrize("value", [0, True, 1.5])
    def test_sample_positions_validates_the_count(self, value):
        with pytest.raises(ParameterError, match="count"):
            sample_positions(value)


class TestFloor:
    """下界的闭式、它的紧性（见证）以及袋子预测器."""

    def test_closed_form_is_hand_computed(self):
        assert orbit_floor(readout_task()) == READOUT_FLOOR

    def test_closed_form_scales_with_n_and_v(self):
        assert orbit_floor(make_readout_task(vocabulary=10, length=5)) == 4 / 50

    def test_witness_equals_the_closed_form(self):
        """下界**是紧的**：袋子预测器恰好达到它（不是"某个够不着的数"）."""
        task = readout_task()
        assert floor_witness(task) == READOUT_FLOOR
        assert abs(floor_witness(task) - orbit_floor(task)) <= FLOOR_TOLERANCE

    def test_bag_predictor_is_hand_computed(self):
        assert readout_bag_row() == BAG_PREDICTOR_ROW
        assert bag_predictor(readout_task())[0] == BAG_PREDICTOR_ROW
        assert approx(math.fsum(readout_bag_row()), 1.0)

    def test_bag_predictor_ignores_the_row_order(self):
        """常数预测器（每一行都输出袋子分布）**本身也是等变的**."""
        task = readout_task()
        rows = bag_predictor(task)
        assert rows == (rows[0],) * LENGTH

    def test_witness_is_the_orbit_mean_loss_of_the_bag_predictor(self):
        task = readout_task()
        assert floor_witness(task) == _orbit_mean_loss_of_outputs(
            task, [bag_predictor(task)] * LENGTH
        )

    def test_witness_checks_the_output_count(self):
        task = readout_task()
        with pytest.raises(ShapeError, match="输出个数"):
            _orbit_mean_loss_of_outputs(task, [bag_predictor(task)] * (LENGTH - 1))

    def test_witness_checks_the_output_shape(self):
        task = readout_task()
        with pytest.raises(ShapeError, match="输出形状"):
            _orbit_mean_loss_of_outputs(task, [((1.0,),)] * LENGTH)


class TestFloorVariant:
    """一个变体的读数：是否越过下界、降幅、判决."""

    def test_equivariant_variant_stops_at_the_floor(self):
        variant = _variant()
        assert variant.below_floor is False
        assert variant.respects_floor is True
        assert variant.verdict == "受下界约束（停在界上）"

    def test_equivariant_variant_below_the_floor_is_flagged(self):
        variant = _variant(final_loss=0.10)
        assert variant.below_floor is True
        assert variant.respects_floor is False
        assert "实现有问题" in variant.verdict

    def test_breaking_variant_crosses_the_floor(self):
        variant = _variant(
            name=VARIANT_SINUSOIDAL,
            description=VARIANT_DESCRIPTIONS[VARIANT_SINUSOIDAL],
            equivariant=False,
            table_kind="sinusoidal",
            final_loss=0.0813,
        )
        assert variant.below_floor is True
        assert variant.verdict == "不受下界约束（**越过了**）"

    def test_breaking_variant_that_does_not_cross(self):
        variant = _variant(
            name=VARIANT_CAUSAL,
            description=VARIANT_DESCRIPTIONS[VARIANT_CAUSAL],
            equivariant=False,
            causal=True,
            final_loss=0.1300,
        )
        assert variant.below_floor is False
        assert variant.verdict == "不受下界约束（没有越过）"

    def test_improvement_ratio(self):
        variant = _variant(initial_loss=0.2, final_loss=0.1)
        assert approx(variant.improvement, 0.5)

    def test_summary_and_serialisation(self):
        payload = _variant().to_dict()
        assert payload["name"] == VARIANT_EQUIVARIANT
        assert payload["below_floor"] is False
        assert "下界" in _variant().summary_line()

    def test_unknown_name_is_rejected(self):
        with pytest.raises(ParameterError, match="未知的变体名"):
            _variant(name="magic")

    @pytest.mark.parametrize("field", ["floor", "initial_loss", "final_loss", "accuracy"])
    def test_finite_numbers_are_required(self, field):
        with pytest.raises(NumericError, match=field):
            _variant(**{field: float("nan")})

    def test_accuracy_must_be_a_probability(self):
        with pytest.raises(NumericError, match="accuracy"):
            _variant(accuracy=2.0)

    def test_trajectory_needs_two_readings(self):
        with pytest.raises(NumericError, match="两个读数"):
            _variant(trajectory=(0.1,))

    def test_notes_are_stringified(self):
        assert _variant(notes=(1,)).notes == ("1",)


class TestSymmetryFloorReport:
    """三个变体的对照（**这一课的结论就落在这里**）."""

    def _report(self) -> SymmetryFloorReport:
        equivariant = _variant()
        injected = _variant(
            name=VARIANT_SINUSOIDAL,
            description=VARIANT_DESCRIPTIONS[VARIANT_SINUSOIDAL],
            equivariant=False,
            table_kind="sinusoidal",
            initial_loss=0.1661,
            final_loss=0.0813,
            accuracy=0.69,
            trajectory=(0.1661, 0.0813),
        )
        causal = _variant(
            name=VARIANT_CAUSAL,
            description=VARIANT_DESCRIPTIONS[VARIANT_CAUSAL],
            equivariant=False,
            causal=True,
            initial_loss=0.1591,
            final_loss=0.0317,
            accuracy=0.94,
            trajectory=(0.1591, 0.0317),
        )
        return assemble_floor_report(readout_task(), (equivariant, injected, causal))

    def test_floor_and_witness_are_recomputed(self):
        report = self._report()
        assert report.floor == orbit_floor(readout_task())
        assert report.witness == floor_witness(readout_task())
        assert report.tight is True

    def test_verdict_is_ok_when_equivariant_stops_and_something_crosses(self):
        report = self._report()
        assert report.ok is True
        assert report.floor_violations == ()
        assert {item.name for item in report.crossers} == {
            VARIANT_SINUSOIDAL,
            VARIANT_CAUSAL,
        }

    def test_variant_partition(self):
        report = self._report()
        assert tuple(item.name for item in report.equivariant_variants) == (VARIANT_EQUIVARIANT,)
        assert {item.name for item in report.breaking_variants} == {
            VARIANT_SINUSOIDAL,
            VARIANT_CAUSAL,
        }

    def test_floor_violation_is_reported(self):
        """等变变体低于下界 = 数学上不可能 ⇒ 报告必须把它挑出来."""
        report = assemble_floor_report(
            readout_task(),
            (_variant(final_loss=0.10),),
        )
        assert len(report.floor_violations) == 1
        assert report.ok is False

    def test_table_lines(self):
        lines = self._report().table_lines()
        assert len(lines) == 5  # 表头 + 分隔线 + 三行
        assert all(line.startswith("  ") for line in lines)

    def test_summary_and_serialisation(self):
        report = self._report()
        assert "2 个越界" in report.summary_line()
        payload = report.to_dict()
        assert payload["ok"] is True
        assert payload["tight"] is True
        assert len(payload["variants"]) == 3

    def test_empty_report_is_rejected(self):
        with pytest.raises(ParameterError, match="至少要有一个变体"):
            SymmetryFloorReport(
                task=readout_task(),
                floor=READOUT_FLOOR,
                witness=READOUT_FLOOR,
                variants=(),
            )

    def test_three_duplicate_variants_are_rejected(self):
        variant = _variant()
        with pytest.raises(NumericError, match="三个变体必须齐备"):
            SymmetryFloorReport(
                task=readout_task(),
                floor=READOUT_FLOOR,
                witness=READOUT_FLOOR,
                variants=(variant, variant, variant),
            )

    def test_a_single_variant_report_is_allowed(self):
        report = SymmetryFloorReport(
            task=readout_task(), floor=READOUT_FLOOR, witness=READOUT_FLOOR, variants=(_variant(),)
        )
        assert report.ok is False  # 没有破对称变体越过 → 判决不成立
        assert "1 个变体" in report.summary_line()

    def test_notes_explain_the_two_routes(self):
        notes = " ".join(self._report().notes)
        assert "常数预测器" in notes
        assert "因果掩码" in notes
