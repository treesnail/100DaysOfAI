"""七条性质、两条正弦表定律与置换缺口（day078 / M7-D3）.

这一份测试守的是"判据要能反向检验"：

```text
前三条与第七条   只在**对齐频率的偶数维正弦表**上成立
                  错位频率（day073 的写法）必须被它们**一起挡住**
后四条           对任何表都成立（加法反向 / 表长硬边界 / 打破等变性）
```

只在对齐表上测过的性质无法区分"实现对了"与"判据太松"——
因此每一条正向性质都配一次**反向检验**：错位表、可学习表、全零表各自必须让它们失败。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.positional_encoding import (
    FLOOR_TOLERANCE,
    OFFSET_LAW_TOLERANCE,
    POSITIONAL_PROPERTIES,
    POSITIONAL_PROPERTY_DESCRIPTIONS,
    PROPERTY_ADDITIVE_BACKWARD,
    PROPERTY_BREAKS_EQUIVARIANCE,
    PROPERTY_CONSTANT_NORM,
    PROPERTY_LENGTH_IS_A_HARD_BOUND,
    PROPERTY_OFFSET_ONLY,
    PROPERTY_SHIFT_IS_ROTATION,
    PROPERTY_STAGGERED_BREAKS_THE_LAWS,
    ROW_NORM_TOLERANCE,
    PositionalPropertyOutcome,
    PositionalPropertyReport,
    SymmetryBreak,
    check_additive_backward,
    check_injection_breaks_equivariance,
    check_offset_law,
    check_properties,
    check_row_norm_is_constant,
    check_shift_is_rotation,
    check_staggered_pairing_breaks_both_laws,
    check_table_length_is_a_hard_bound,
    closed_form_offset_inner,
    default_permutation,
    loss_gradient,
    permute_rows,
    positional_backward,
    positional_forward,
    rotation_factor,
    symmetry_break,
)
from smart_research_agent.positional_encoding.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.positional_encoding.verify import (
    ROTATION_TOLERANCE,
    _accumulated_table_gradient,
    _overwrite_table_gradient,
)
from tests.position_samples import (
    DIMENSION,
    LENGTH,
    OFFSET_INNER_ONE,
    REVERSAL_PERMUTATION,
    VOCABULARY,
    approx,
    learnable,
    position_parameters,
    readout_task,
    sinusoidal,
    staggered,
    zeros,
)

#: 位置互不相同的显式序列——"按行覆盖"在这个样本上与累加**恰好相同**.
UNIQUE_POSITIONS: tuple[int, ...] = (0, 1, 2, 3)

#: 位置重复的显式序列——"按行覆盖"与累加在这里必然分家.
DUPLICATE_POSITIONS: tuple[int, ...] = (0, 1, 0, 1)


def _tasks():
    """一份固定的（参数, 输入, 目标, 监督行）四元组."""
    task = readout_task()
    params = position_parameters(VOCABULARY)
    return params, task.inputs, task.target, task.supervised


def _forward(table, *, positions=None):
    """对齐正弦表上的完整前向（带目标，供反向性质用）."""
    params, inputs, target, supervised = _tasks()
    return positional_forward(
        params,
        table,
        inputs,
        positions=positions,
        target=target,
        supervised=supervised,
    )


def _transpose(matrix):
    """转置（只用来核对 ``R_δᵀ·R_δ = I``）."""
    return tuple(zip(*matrix))


def _matmul(left, right):
    """两块方阵相乘（测试里只用来核对旋转矩阵的正交性）."""
    size = len(left)
    return tuple(
        tuple(
            math.fsum(left[row][inner] * right[inner][column] for inner in range(size))
            for column in range(size)
        )
        for row in range(size)
    )


def _row_gradients():
    """四行的"逐行梯度"（两列）：位置 ``(0, 1, 0, 1)`` 下累加与覆盖会分家."""
    return ((1.0, 0.5), (0.25, 2.0), (3.0, 0.0), (0.5, 0.5))


class TestPropertyTables:
    """七条性质的名单与说明必须逐键对齐（顺序就是报告里的顺序）."""

    def test_names_and_descriptions_align(self):
        assert set(POSITIONAL_PROPERTIES) == set(POSITIONAL_PROPERTY_DESCRIPTIONS)
        assert len(POSITIONAL_PROPERTIES) == 7

    def test_order_is_constant_five_then_the_two_extremes(self):
        assert POSITIONAL_PROPERTIES == (
            PROPERTY_CONSTANT_NORM,
            PROPERTY_OFFSET_ONLY,
            PROPERTY_SHIFT_IS_ROTATION,
            PROPERTY_BREAKS_EQUIVARIANCE,
            PROPERTY_ADDITIVE_BACKWARD,
            PROPERTY_LENGTH_IS_A_HARD_BOUND,
            PROPERTY_STAGGERED_BREAKS_THE_LAWS,
        )

    def test_every_property_has_a_non_empty_description(self):
        for name in POSITIONAL_PROPERTIES:
            assert POSITIONAL_PROPERTY_DESCRIPTIONS[name].strip()


class TestConstantNorm:
    """第 1 条：正弦表每一行的 L2 范数**恰好**是 ``√(d/2)``（``sin² + cos² = 1``）."""

    def test_aligned_table_passes(self):
        outcome = check_row_norm_is_constant(sinusoidal(LENGTH, DIMENSION))
        assert outcome.name == PROPERTY_CONSTANT_NORM
        assert outcome.passed is True
        assert "检查 4 行" in outcome.evidence
        assert "0.00e+00" in outcome.evidence

    def test_staggered_table_fails(self):
        """错位频率：``cos`` 用了隔壁的频率，``sin² + cos² = 1`` 不再成立."""
        outcome = check_row_norm_is_constant(staggered(LENGTH, DIMENSION))
        assert outcome.passed is False
        assert "偏差" in outcome.evidence
        assert "对齐频率" in outcome.detail

    def test_zero_table_fails(self):
        assert check_row_norm_is_constant(zeros(LENGTH, DIMENSION)).passed is False

    def test_learnable_from_sine_passes(self):
        assert check_row_norm_is_constant(learnable(LENGTH, DIMENSION)).passed is True

    def test_tolerance_must_be_positive(self):
        with pytest.raises(ParameterError, match="正的有限数"):
            check_row_norm_is_constant(sinusoidal(LENGTH, DIMENSION), tolerance=0.0)

    def test_tolerance_is_the_pinned_constant(self):
        assert ROW_NORM_TOLERANCE == 1e-12


class TestOffsetLaw:
    """第 2 条：``<PE(p), PE(q)>`` 只依赖 ``p − q``（相对距离读得出来）."""

    def test_aligned_table_passes_and_matches_the_closed_form(self):
        outcome = check_offset_law(sinusoidal(LENGTH, DIMENSION))
        assert outcome.name == PROPERTY_OFFSET_ONLY
        assert outcome.passed is True
        assert "检查 6 对位置" in outcome.evidence

    def test_closed_form_anchor(self):
        assert closed_form_offset_inner(DIMENSION, 1) == OFFSET_INNER_ONE

    def test_learnable_table_is_skipped_and_fails(self):
        outcome = check_offset_law(learnable(LENGTH, DIMENSION))
        assert outcome.passed is False
        assert "跳过不成立" in outcome.evidence
        assert outcome.skipped is True

    def test_staggered_table_fails(self):
        outcome = check_offset_law(staggered(LENGTH, DIMENSION))
        assert outcome.passed is False
        assert "对齐频率" in outcome.evidence

    def test_offsets_must_be_at_least_one(self):
        with pytest.raises(ParameterError, match="必须是 >= 1 的整数"):
            check_offset_law(sinusoidal(LENGTH, DIMENSION), offsets=(0,))

    def test_tolerance_is_the_pinned_constant(self):
        assert OFFSET_LAW_TOLERANCE == 1e-12


class TestShiftIsRotation:
    """第 3 条：``PE(p + δ)`` 逐位等于 ``R_δ · PE(p)``——位移律的**构造性理由**."""

    def test_aligned_table_passes(self):
        outcome = check_shift_is_rotation(sinusoidal(LENGTH, DIMENSION))
        assert outcome.name == PROPERTY_SHIFT_IS_ROTATION
        assert outcome.passed is True
        # (4−1) + (4−2) + (4−3) = 6 对，每对 4 维 → 24 个逐点
        assert "检查 24 个逐点" in outcome.evidence
        assert "正交" in outcome.detail

    def test_rotation_factor_is_orthogonal(self):
        """``R_δᵀ·R_δ = I`` ⇒ 它保持范数（第 1 条）与内积（第 2 条）——两条性质同源."""
        factor = rotation_factor(DIMENSION, 1)
        identity = tuple(
            tuple(1.0 if row == column else 0.0 for column in range(DIMENSION))
            for row in range(DIMENSION)
        )
        assert ROTATION_TOLERANCE == 1e-12
        for row, expected_row in zip(
            _matmul(_transpose(factor), factor), identity, strict=True
        ):
            for value, expected in zip(row, expected_row, strict=True):
                assert approx(value, expected, tolerance=1e-9)

    def test_learnable_table_fails(self):
        outcome = check_shift_is_rotation(learnable(LENGTH, DIMENSION))
        assert outcome.passed is False
        assert "跳过不成立" in outcome.evidence

    def test_staggered_table_fails(self):
        assert check_shift_is_rotation(staggered(LENGTH, DIMENSION)).passed is False

    def test_offsets_must_be_at_least_one(self):
        with pytest.raises(ParameterError, match="必须是 >= 1 的整数"):
            check_shift_is_rotation(sinusoidal(LENGTH, DIMENSION), offsets=(0,))


class TestEquivarianceBreak:
    """第 4 条：注入之后置换**不再**等变——这是本课要的效果，不是 bug."""

    def test_base_layer_gap_is_exactly_zero(self):
        """基准层（无位置编码）的置换缺口**恰好** 0.0：无掩码时注意力是置换等变的."""
        params, inputs, _target, _supervised = _tasks()
        report = symmetry_break(params, zeros(LENGTH, VOCABULARY), inputs)
        assert report.base_gap == 0.0
        assert "基准层逐位为 0" in report.notes[0]

    def test_injection_breaks_equivariance(self):
        params, inputs, _target, _supervised = _tasks()
        report = symmetry_break(params, sinusoidal(LENGTH, VOCABULARY), inputs)
        assert report.base_gap == 0.0
        assert report.injected_gap > 1e-6
        assert report.broken is True
        assert report.permutation == REVERSAL_PERMUTATION

    def test_zero_table_does_not_break_equivariance(self):
        params, inputs, _target, _supervised = _tasks()
        report = symmetry_break(params, zeros(LENGTH, VOCABULARY), inputs)
        assert report.base_gap == 0.0
        assert report.injected_gap == 0.0
        assert report.broken is False

    def test_explicit_permutation_is_honoured(self):
        params, inputs, _target, _supervised = _tasks()
        report = symmetry_break(
            params, sinusoidal(LENGTH, VOCABULARY), inputs, permutation=(1, 0, 3, 2)
        )
        assert report.permutation == (1, 0, 3, 2)

    def test_invalid_permutation_is_rejected(self):
        params, inputs, _target, _supervised = _tasks()
        with pytest.raises(ParameterError, match="不是 0"):
            symmetry_break(
                params, sinusoidal(LENGTH, VOCABULARY), inputs, permutation=(0, 0, 1, 2)
            )

    def test_check_returns_skipped_under_a_causal_mask(self):
        """``causal=True``：因果掩码本身就破坏了等变性，这一条**跳过**（而不是通过）."""
        params, inputs, _target, _supervised = _tasks()
        outcome = check_injection_breaks_equivariance(
            params, sinusoidal(LENGTH, VOCABULARY), inputs, causal=True
        )
        assert outcome.name == PROPERTY_BREAKS_EQUIVARIANCE
        assert outcome.passed is True
        assert "跳过" in outcome.evidence
        assert outcome.skipped is True

    def test_check_passes_without_a_mask(self):
        params, inputs, _target, _supervised = _tasks()
        outcome = check_injection_breaks_equivariance(
            params, sinusoidal(LENGTH, VOCABULARY), inputs
        )
        assert outcome.passed is True
        assert outcome.skipped is False
        assert "基准层逐位为 0" in outcome.evidence

    def test_default_permutation_is_the_reversal(self):
        assert default_permutation(4) == REVERSAL_PERMUTATION

    def test_default_permutation_validates(self):
        with pytest.raises(ParameterError):
            default_permutation(0)

    def test_permute_rows_round_trip(self):
        matrix = ((1.0,), (2.0,), (3.0,))
        assert permute_rows(matrix, (2, 1, 0)) == ((3.0,), (2.0,), (1.0,))

    def test_permute_rows_validates_length_and_entries(self):
        with pytest.raises(ShapeError, match="置换长度"):
            permute_rows(((1.0,), (2.0,)), (0,))
        with pytest.raises(ParameterError, match="不是 0"):
            permute_rows(((1.0,), (2.0,)), (0, 0))

    def test_symmetry_break_record_validation(self):
        with pytest.raises(NumericError, match="非负"):
            SymmetryBreak(permutation=(0,), base_gap=-1.0, injected_gap=0.0)
        with pytest.raises(NumericError, match="非负"):
            SymmetryBreak(permutation=(0,), base_gap=0.0, injected_gap=float("nan"))

    def test_symmetry_break_serialises(self):
        params, inputs, _target, _supervised = _tasks()
        report = symmetry_break(params, sinusoidal(LENGTH, VOCABULARY), inputs)
        payload = report.to_dict()
        assert payload["broken"] is True
        assert payload["base_gap"] == 0.0
        assert "置换" in report.summary_line()


class TestAdditiveBackward:
    """第 5 条：``dx`` 逐位传回、``dTable`` 按位置累加（漏掉不会报错的那一处）."""

    def test_identity_and_accumulation_hold(self):
        outcome = check_additive_backward(_forward(sinusoidal(LENGTH, VOCABULARY)))
        assert outcome.name == PROPERTY_ADDITIVE_BACKWARD
        assert outcome.passed is True
        assert "恰好 0.0" in outcome.evidence
        assert "重复位置 0 处" in outcome.evidence
        assert "与'按行覆盖'的差 0.00e+00" in outcome.evidence

    def test_duplicate_positions_expose_the_overwrite_gap(self):
        outcome = check_additive_backward(
            _forward(sinusoidal(LENGTH, VOCABULARY), positions=DUPLICATE_POSITIONS)
        )
        assert outcome.passed is True
        assert "重复位置 2 处" in outcome.evidence
        assert "与'按行覆盖'的差 0.00e+00" not in outcome.evidence

    def test_overwrite_equals_accumulate_without_duplicates(self):
        """位置互不相同的两条路径给出**同一张表梯度**——因此覆盖写法在这里躲得过去."""
        rows = _row_gradients()
        assert _overwrite_table_gradient(
            rows, UNIQUE_POSITIONS, table_positions=4
        ) == _accumulated_table_gradient(rows, UNIQUE_POSITIONS, table_positions=4)

    def test_overwrite_differs_from_accumulate_with_duplicates(self):
        rows = _row_gradients()
        overwritten = _overwrite_table_gradient(
            rows, DUPLICATE_POSITIONS, table_positions=4
        )
        accumulated = _accumulated_table_gradient(
            rows, DUPLICATE_POSITIONS, table_positions=4
        )
        assert overwritten != accumulated
        assert accumulated[0] == (4.0, 0.5)  # 第 0 行收到了两份梯度之和
        assert overwritten[0] == (3.0, 0.0)  # 覆盖写法只留下**最后一次**写入

    def test_dx_equals_dinjected_bit_for_bit(self):
        forward = _forward(sinusoidal(LENGTH, VOCABULARY), positions=DUPLICATE_POSITIONS)
        gradients = positional_backward(forward, loss_gradient(forward))
        assert gradients.grad_table == _accumulated_table_gradient(
            gradients.grad_inputs,
            DUPLICATE_POSITIONS,
            table_positions=forward.table.positions,
        )

    def test_needs_a_target(self):
        params, inputs, _target, _supervised = _tasks()
        forward = positional_forward(params, sinusoidal(LENGTH, VOCABULARY), inputs)
        with pytest.raises(ParameterError, match="没有 target"):
            check_additive_backward(forward)


class TestHardBound:
    """第 6 条：可学习表对越界**拒绝**，而正弦表对同一个位置**算得出来**."""

    def test_learnable_table_refuses(self):
        outcome = check_table_length_is_a_hard_bound(learnable(LENGTH, DIMENSION))
        assert outcome.name == PROPERTY_LENGTH_IS_A_HARD_BOUND
        assert outcome.passed is True
        assert "RangeError" in outcome.evidence

    def test_zero_table_also_refuses(self):
        outcome = check_table_length_is_a_hard_bound(zeros(LENGTH, DIMENSION))
        assert outcome.passed is True
        assert "RangeError" in outcome.evidence

    def test_sinusoidal_table_still_computes(self):
        outcome = check_table_length_is_a_hard_bound(sinusoidal(LENGTH, DIMENSION))
        assert outcome.passed is True
        assert "算得出来" in outcome.evidence


class TestStaggeredBreaksBothLaws:
    """第 7 条：错位频率让"行范数恒定"与"位移律"**一起失效**（判据能反向检验）."""

    def test_aligned_reference_passes(self):
        outcome = check_staggered_pairing_breaks_both_laws(sinusoidal(LENGTH, DIMENSION))
        assert outcome.name == PROPERTY_STAGGERED_BREAKS_THE_LAWS
        assert outcome.passed is True
        assert "错位表的行范数偏差" in outcome.evidence
        assert "两条判据都抓得住它" in outcome.evidence

    def test_staggered_table_cannot_serve_as_the_reference(self):
        """对照必须有一侧是"已知正确"的——用错位表当基准只是在自我循环."""
        outcome = check_staggered_pairing_breaks_both_laws(staggered(LENGTH, DIMENSION))
        assert outcome.passed is False
        assert "对照基准" in outcome.evidence

    def test_explicit_base_is_honoured(self):
        outcome = check_staggered_pairing_breaks_both_laws(
            sinusoidal(LENGTH, DIMENSION), base=10000.0
        )
        assert outcome.passed is True


class TestCheckProperties:
    """七条性质一起跑：报告必须能区分"通过 / 失败 / 跳过"三种状态."""

    def test_aligned_table_passes_all_seven(self):
        params, inputs, target, supervised = _tasks()
        report = check_properties(
            params, sinusoidal(LENGTH, VOCABULARY), inputs, target=target, supervised=supervised
        )
        assert report.ok is True
        assert tuple(item.name for item in report.outcomes) == POSITIONAL_PROPERTIES
        assert report.failures == ()
        assert report.skipped == ()
        assert "通过 7、失败 0、跳过 0" in report.summary_line()

    def test_causal_mask_skips_the_equivariance_property(self):
        params, inputs, target, supervised = _tasks()
        report = check_properties(
            params,
            sinusoidal(LENGTH, VOCABULARY),
            inputs,
            target=target,
            supervised=supervised,
            causal=True,
        )
        assert report.ok is True
        assert tuple(item.name for item in report.skipped) == (PROPERTY_BREAKS_EQUIVARIANCE,)
        assert "跳过 1" in report.summary_line()

    def test_learnable_table_fails_the_sinusoidal_properties(self):
        """可学习表不满足前三条与第七条——**那是正确的结果**，不是实现错了."""
        params, inputs, target, supervised = _tasks()
        report = check_properties(
            params, learnable(LENGTH, VOCABULARY), inputs, target=target, supervised=supervised
        )
        assert report.ok is False
        assert {item.name for item in report.failures} == {
            PROPERTY_OFFSET_ONLY,
            PROPERTY_SHIFT_IS_ROTATION,
            PROPERTY_STAGGERED_BREAKS_THE_LAWS,
        }
        assert len(report.skipped) == 3

    def test_without_a_target_the_backward_property_is_skipped(self):
        params, inputs, _target, _supervised = _tasks()
        report = check_properties(params, sinusoidal(LENGTH, VOCABULARY), inputs)
        skipped = {item.name for item in report.skipped}
        assert skipped == {PROPERTY_ADDITIVE_BACKWARD}
        assert report.ok is True

    def test_notes_explain_the_preconditions(self):
        params, inputs, target, supervised = _tasks()
        report = check_properties(
            params, sinusoidal(LENGTH, VOCABULARY), inputs, target=target, supervised=supervised
        )
        assert any("对齐频率" in note for note in report.notes)
        assert any("跳过" in note for note in report.notes)

    def test_report_serialises(self):
        params, inputs, target, supervised = _tasks()
        payload = check_properties(
            params, sinusoidal(LENGTH, VOCABULARY), inputs, target=target, supervised=supervised
        ).to_dict()
        assert payload["ok"] is True
        assert len(payload["outcomes"]) == 7
        assert payload["outcomes"][0]["description"]


class TestPropertyRecords:
    """性质记录的构造校验：不认识的项、缺项的汇总都要拒绝."""

    def test_unknown_name_is_rejected(self):
        with pytest.raises(ParameterError, match="未知的性质名"):
            PositionalPropertyOutcome(name="nope", passed=True, evidence="x")

    def test_description_comes_from_the_table(self):
        outcome = PositionalPropertyOutcome(
            name=PROPERTY_CONSTANT_NORM, passed=True, evidence="x"
        )
        assert "范数" in outcome.description

    def test_skipped_is_read_from_the_evidence(self):
        outcome = PositionalPropertyOutcome(
            name=PROPERTY_BREAKS_EQUIVARIANCE, passed=True, evidence="本次前向（跳过）"
        )
        assert outcome.skipped is True
        assert outcome.to_dict()["skipped"] is True

    def test_empty_report_is_rejected(self):
        with pytest.raises(ParameterError, match="不能为空"):
            PositionalPropertyReport(outcomes=())

    def test_incomplete_report_is_rejected(self):
        outcome = PositionalPropertyOutcome(
            name=PROPERTY_CONSTANT_NORM, passed=True, evidence="x"
        )
        with pytest.raises(NumericError, match="名单不完整"):
            PositionalPropertyReport(outcomes=(outcome,))

    def test_failing_outcome_marks_the_report(self):
        outcomes = tuple(
            PositionalPropertyOutcome(
                name=name, passed=name != PROPERTY_OFFSET_ONLY, evidence="e"
            )
            for name in POSITIONAL_PROPERTIES
        )
        report = PositionalPropertyReport(outcomes=outcomes)
        assert not report.ok
        assert len(report.failures) == 1
        assert "通过 6、失败 1" in report.summary_line()

    def test_notes_are_stringified(self):
        outcomes = tuple(
            PositionalPropertyOutcome(name=name, passed=True, evidence="e")
            for name in POSITIONAL_PROPERTIES
        )
        assert PositionalPropertyReport(outcomes=outcomes, notes=(1,)).notes == ("1",)

    def test_stale_floor_tolerance_constant(self):
        """下界的判据常量（``FLOOR_TOLERANCE``）必须足够紧：它在 1e-15 与 1e-9 之间."""
        assert 1e-15 < FLOOR_TOLERANCE < 1e-9


def test_symmetry_break_matches_the_property_evidence():
    """两条入口必须给同一个读数：``symmetry_break`` 与第 4 条性质的证据一致."""
    params, inputs, _target, _supervised = _tasks()
    table = sinusoidal(LENGTH, VOCABULARY)
    report = symmetry_break(params, table, inputs)
    outcome = check_injection_breaks_equivariance(params, table, inputs)
    assert f"{report.injected_gap:.2e}" in outcome.evidence
    assert report.base_gap == 0.0
    assert approx(report.injected_gap, 3.2e-3, tolerance=1e-1)
