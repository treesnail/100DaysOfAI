"""``math_foundations.types``：形状校验与四张口径表（day073）.

本文件钉住四类性质：

```text
① 四张封闭表逐键对齐（少一个键 = 某一项没有解释、也没有公式）
② 形状校验拒绝"看起来像、其实不对"的输入（NaN、不等长、空、字符串）
③ 比较与误差的口径只有一份实现（close / relative_error / normalize_rows）
④ 两个报告形状在构造期校验自己的不变量（权重行和为 1、派生量长度一致）
```
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.math_foundations.errors import (
    MathError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.math_foundations.types import (
    ATTENTION_VARIANTS,
    ATTENTION_VARIANT_DESCRIPTIONS,
    BRIDGE_TARGETS,
    BRIDGE_TARGET_DESCRIPTIONS,
    BRIDGE_TARGET_SOURCES,
    CHECK_LENGTH_MATCHES_LABELS,
    CHECK_NON_EMPTY,
    CHECK_NON_NEGATIVE,
    CHECK_SUMS_TO_ONE,
    DEFAULT_TOLERANCE,
    DISTRIBUTION_CHECKS,
    DISTRIBUTION_CHECK_DESCRIPTIONS,
    LINALG_OPS,
    LINALG_OP_DESCRIPTIONS,
    LINALG_OP_FORMULAS,
    AttentionReport,
    VectorSummary,
    check_distribution,
    close,
    is_finite,
    is_square,
    matrix_shape,
    normalize_rows,
    relative_error,
    require_same_dimension,
    require_square,
    row_sums,
    validate_matrix,
    validate_vector,
)
from tests.math_samples import approx, uniform


class TestClosedTables:
    """四张口径表必须逐键对齐（缺一个键 = 文档里少一行，而没人会发现）."""

    def test_linalg_tables_align(self) -> None:
        assert set(LINALG_OPS) == set(LINALG_OP_DESCRIPTIONS) == set(LINALG_OP_FORMULAS)
        assert len(LINALG_OPS) == 10

    def test_every_operator_has_a_formula(self) -> None:
        for name in LINALG_OPS:
            formula = LINALG_OP_FORMULAS[name]
            assert formula and ("=" in formula or "Σ" in formula), name

    def test_attention_tables_align(self) -> None:
        assert set(ATTENTION_VARIANTS) == set(ATTENTION_VARIANT_DESCRIPTIONS)
        assert len(ATTENTION_VARIANTS) == 4

    def test_bridge_tables_align(self) -> None:
        assert (
            set(BRIDGE_TARGETS)
            == set(BRIDGE_TARGET_DESCRIPTIONS)
            == set(BRIDGE_TARGET_SOURCES)
        )
        assert len(BRIDGE_TARGETS) == 8
        for name in BRIDGE_TARGETS:
            assert BRIDGE_TARGET_SOURCES[name].startswith("smart_research_agent.")

    def test_distribution_check_tables_align(self) -> None:
        assert set(DISTRIBUTION_CHECKS) == set(DISTRIBUTION_CHECK_DESCRIPTIONS)
        assert len(DISTRIBUTION_CHECKS) == 4


class TestShapeValidation:
    """向量/矩阵的形状与有限性校验（**每一个拒绝都要给得出为什么**）."""

    def test_vector_is_normalized_to_tuple_of_floats(self) -> None:
        assert validate_vector([1, 2.5]) == (1.0, 2.5)

    @pytest.mark.parametrize("value", ["abc", 3.0, None])
    def test_non_iterable_vectors_are_rejected(self, value: object) -> None:
        with pytest.raises(ShapeError):
            validate_vector(value)

    def test_empty_vector_is_rejected(self) -> None:
        with pytest.raises(ShapeError):
            validate_vector(())

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    def test_non_finite_components_are_rejected(self, value: float) -> None:
        with pytest.raises(NumericError):
            validate_vector((1.0, value))

    def test_bool_is_not_a_number(self) -> None:
        assert is_finite(1) is True
        assert is_finite(True) is False
        assert is_finite("1") is False

    def test_matrix_requires_rectangular_rows(self) -> None:
        assert validate_matrix([[1.0, 2.0], [3.0, 4.0]]) == ((1.0, 2.0), (3.0, 4.0))
        with pytest.raises(ShapeError):
            validate_matrix([[1.0, 2.0], [3.0]])

    def test_empty_matrix_is_rejected(self) -> None:
        with pytest.raises(ShapeError):
            validate_matrix([])

    def test_matrix_shape_and_squareness(self) -> None:
        assert matrix_shape(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))) == (2, 3)
        assert matrix_shape(()) == (0, 0)
        assert is_square(((1.0, 0.0), (0.0, 1.0))) is True
        assert is_square(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))) is False
        require_square(((1.0, 0.0), (0.0, 1.0)))
        with pytest.raises(ShapeError):
            require_square(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))

    def test_dimension_mismatch_is_rejected(self) -> None:
        require_same_dimension((1.0, 2.0), (3.0, 4.0))
        with pytest.raises(ShapeError):
            require_same_dimension((1.0, 2.0), (3.0, 4.0, 5.0))


class TestComparisons:
    """比较口径：容差必须为正、相对误差在 0 附近退化成绝对误差."""

    def test_close_uses_relative_and_absolute(self) -> None:
        assert close(1.0, 1.0 + 1e-12) is True
        assert close(1.0, 1.1) is False
        assert close(float("nan"), float("nan")) is False

    def test_close_rejects_non_positive_tolerance(self) -> None:
        with pytest.raises(ParameterError):
            close(1.0, 1.0, tolerance=0.0)

    def test_relative_error_near_zero_uses_absolute(self) -> None:
        # 精确值 0、近似值 1e-12：相对误差公式在这里会给出 1（100%），
        # 而本课的口径是"分母取 max(1, |r|)"，于是它退化成绝对误差 1e-12。
        assert relative_error(1e-12, 0.0) == pytest.approx(1e-12)
        assert relative_error(1.0, 2.0) == pytest.approx(0.5)
        assert relative_error(3.0, 2.0) == pytest.approx(0.5)

    def test_relative_error_rejects_non_finite(self) -> None:
        with pytest.raises(NumericError):
            relative_error(float("nan"), 0.0)

    def test_normalize_rows_scales_each_row(self) -> None:
        normalized = normalize_rows(((1.0, 3.0), (0.0, 0.0), (2.0, 2.0)))
        assert approx(normalized[0][0], 0.25)
        assert approx(normalized[0][1], 0.75)
        assert normalized[1] == (0.0, 0.0)  # 全零行原样保留（由调用方判）
        assert approx(normalized[2][0], 0.5)

    def test_row_sums(self) -> None:
        assert row_sums(((1.0, 2.0), (3.0, 4.0))) == (3.0, 7.0)


class TestDistributionCheck:
    """分布的四条判据（顺序固定、命中即抛，且**只有这一处实现**）."""

    def test_valid_distribution_passes(self) -> None:
        assert check_distribution(uniform(4)) == uniform(4)

    def test_empty_is_rejected(self) -> None:
        with pytest.raises(NumericError) as excinfo:
            check_distribution(())
        assert CHECK_NON_EMPTY in str(excinfo.value)

    def test_negative_is_rejected(self) -> None:
        with pytest.raises(NumericError) as excinfo:
            check_distribution((1.5, -0.5))
        assert CHECK_NON_NEGATIVE in str(excinfo.value)

    def test_unnormalized_is_rejected(self) -> None:
        with pytest.raises(NumericError) as excinfo:
            check_distribution((0.5, 0.4))
        assert CHECK_SUMS_TO_ONE in str(excinfo.value)

    def test_label_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(NumericError) as excinfo:
            check_distribution(uniform(3), labels=("a", "b"))
        assert CHECK_LENGTH_MATCHES_LABELS in str(excinfo.value)

    def test_tolerance_is_used(self) -> None:
        assert check_distribution((0.5, 0.5 + 1e-12)) == (0.5, 0.5 + 1e-12)


class TestVectorSummary:
    """向量画像：维度、模长、均值、极值、是否单位向量."""

    def test_summary_of_a_unit_vector(self) -> None:
        summary = VectorSummary.of((1.0, 0.0, 0.0))
        assert summary.dimension == 3
        assert approx(summary.norm, 1.0)
        assert summary.is_unit is True
        assert approx(summary.mean, 1.0 / 3.0)
        assert summary.minimum == 0.0 and summary.maximum == 1.0

    def test_summary_of_a_non_unit_vector(self) -> None:
        summary = VectorSummary.of((3.0, 4.0))
        assert approx(summary.norm, 5.0)
        assert summary.is_unit is False
        assert approx(summary.mean, 3.5)

    def test_dict_and_line(self) -> None:
        summary = VectorSummary.of((1.0, 0.0, 0.0))
        payload = summary.to_dict()
        assert payload["is_unit"] is True and payload["dimension"] == 3
        assert "单位向量" in summary.summary_line()


class TestAttentionReport:
    """注意力报告的不变量：权重每行和为 1、派生量长度与行数一致."""

    def _report(self, **overrides: object) -> AttentionReport:
        fields: dict[str, object] = {
            "variant": "scaled",
            "queries": 2,
            "keys": 2,
            "head_dim": 2,
            "scale": 0.7071067811865476,
            "weights": ((0.5, 0.5), (0.25, 0.75)),
            "output": ((1.0, 0.0), (0.25, 0.75)),
            "entropies": (math.log(2.0), 0.5623351446188083),
            "peak_weights": (0.5, 0.75),
            "peak_indices": (0, 1),
            "causal": False,
        }
        fields.update(overrides)
        return AttentionReport(**fields)  # type: ignore[arg-type]

    def test_derived_quantities(self) -> None:
        report = self._report()
        assert approx(report.mean_entropy, (math.log(2.0) + 0.5623351446188083) / 2)
        assert approx(report.max_entropy(), math.log(2.0))
        assert 0.0 <= report.focus_ratio() <= 1.0
        assert "scaled" in report.summary_line()

    def test_uniform_rows_sit_on_the_entropy_ceiling(self) -> None:
        report = self._report(
            weights=((0.5, 0.5), (0.5, 0.5)),
            entropies=(math.log(2.0), math.log(2.0)),
            peak_weights=(0.5, 0.5),
        )
        assert approx(report.mean_entropy, math.log(2.0))
        assert approx(report.focus_ratio(), 0.0)

    def test_single_key_has_no_focus_to_measure(self) -> None:
        # keys = 1 时 max_entropy() = ln(1) = 0，"集中度"没有定义 → 返回 0.0
        single = self._report(
            keys=1,
            weights=((1.0,), (1.0,)),
            entropies=(0.0, 0.0),
            peak_weights=(1.0, 1.0),
            peak_indices=(0, 0),
        )
        assert single.focus_ratio() == 0.0
        assert single.max_entropy() == 0.0

    def test_row_sum_must_be_one(self) -> None:
        with pytest.raises(NumericError):
            self._report(weights=((0.5, 0.4), (0.25, 0.75)))

    def test_shape_must_match_declaration(self) -> None:
        with pytest.raises(ShapeError):
            self._report(queries=3)

    def test_derived_lengths_must_match(self) -> None:
        with pytest.raises(ShapeError):
            self._report(peak_indices=(0,))

    def test_unknown_variant_is_rejected(self) -> None:
        with pytest.raises(MathError):
            self._report(variant="triple")

    def test_dict_is_json_friendly(self) -> None:
        import json

        payload = self._report().to_dict()
        assert payload["variant"] == "scaled"
        json.dumps(payload)
        assert DEFAULT_TOLERANCE > 0
