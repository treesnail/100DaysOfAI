"""``arch_variants.masks`` 的测试：三张掩码、一次组合、以及“因果是表的性质”（day082）."""

from __future__ import annotations

import pytest

from smart_research_agent.arch_variants import masks
from smart_research_agent.arch_variants.errors import (
    AssemblyError,
    ParameterError,
    ShapeError,
)


class TestStructuralMasks:
    """两张结构性掩码（**转发** day073/075，因此这里验的是口径不变）."""

    def test_full_mask_allows_every_position(self) -> None:
        mask = masks.mask_of(masks.MASK_FULL, 4)
        assert masks.mask_allowed_counts(mask) == (4, 4, 4, 4)
        assert masks.mask_allowed_pairs(mask) == 16

    def test_causal_mask_allows_a_triangle(self) -> None:
        mask = masks.mask_of(masks.MASK_CAUSAL, 4)
        assert masks.mask_allowed_counts(mask) == (1, 2, 3, 4)
        assert masks.mask_allowed_pairs(mask) == 10

    def test_causal_mask_is_the_forwarded_one(self) -> None:
        """与 ``math_foundations.attention.causal_mask`` 逐位一致（转发而不是重写）."""
        from smart_research_agent.math_foundations.attention import causal_mask

        assert masks.mask_of(masks.MASK_CAUSAL, 5) == causal_mask(5)

    def test_full_mask_is_the_forwarded_one(self) -> None:
        """与 ``transformer_core.layers.full_mask`` 逐位一致."""
        from smart_research_agent.transformer_core.layers import full_mask

        assert masks.mask_of(masks.MASK_FULL, 3) == full_mask(3)

    def test_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            masks.mask_of("triangular", 4)

    def test_kind_must_be_a_string(self) -> None:
        with pytest.raises(ParameterError):
            masks.mask_of(3, 4)  # type: ignore[arg-type]

    @pytest.mark.parametrize("size", [0, -1])
    def test_bad_size_is_rejected(self, size: int) -> None:
        with pytest.raises(ParameterError):
            masks.mask_of(masks.MASK_FULL, size)

    def test_bool_size_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            masks.mask_of(masks.MASK_FULL, True)  # type: ignore[arg-type]

    def test_non_int_size_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            masks.mask_of(masks.MASK_FULL, 4.0)  # type: ignore[arg-type]

    def test_descriptions_cover_every_kind(self) -> None:
        assert set(masks.MASK_DESCRIPTIONS) == set(masks.MASK_KINDS)


class TestPaddingMask:
    """填充掩码：``allowed[i][j] = (not pads[j]) or (i == j)``."""

    def test_pad_columns_are_blocked(self) -> None:
        mask = masks.padding_mask((False, True, False, False))
        # 第 1 个位置是填充：**列** 1 上除了它自己（对角线），谁也不该看它
        assert [row[1] for row in mask] == [False, True, False, False]
        # 而**非填充**的列（0、2、3）每一行都允许看
        assert [row[0] for row in mask] == [True, True, True, True]
        assert [row[2] for row in mask] == [True, True, True, True]

    def test_padded_row_still_sees_the_real_positions(self) -> None:
        """第 ③ 条：填充行仍然看得到**非填充**的位置——它的输出没有意义，该被丢掉."""
        mask = masks.padding_mask((True, True, False))
        assert masks.mask_allowed_counts(mask) == (2, 2, 1)
        assert mask[0] == (True, False, True)
        assert mask[2] == (False, False, True)

    def test_all_pads_is_still_defined(self) -> None:
        """整条序列都是填充时，每一行仍然有定义（day075 会拒绝全 False 行）."""
        mask = masks.padding_mask((True, True, True))
        assert masks.mask_allowed_counts(mask) == (1, 1, 1)

    def test_empty_pads_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            masks.padding_mask(())

    def test_non_bool_entry_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            masks.padding_mask((False, 1, False))  # type: ignore[arg-type]


class TestCombineMasks:
    """组合 = 逐位与（两张掩码同时生效时唯一自然的含义）."""

    def test_causal_and_padding(self) -> None:
        combined = masks.combine_masks(
            masks.mask_of(masks.MASK_CAUSAL, 3), masks.padding_mask((False, False, True))
        )
        # 第 2 行既是因果的末行、又是那个填充行 ⇒ 它保住对角线，因此三个位置都在
        assert masks.mask_allowed_counts(combined) == (1, 2, 3)
        assert masks.mask_is_causal(combined)

    def test_intersection_is_not_union(self) -> None:
        """写成并集的话，多看到的位置谁也不会报错——这里钉住“是交集”."""
        causal = masks.mask_of(masks.MASK_CAUSAL, 3)
        combined = masks.combine_masks(causal, masks.mask_of(masks.MASK_FULL, 3))
        assert combined == causal

    def test_empty_call_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            masks.combine_masks()

    def test_shape_mismatch_is_rejected(self) -> None:
        with pytest.raises(ShapeError):
            masks.combine_masks(
                masks.mask_of(masks.MASK_FULL, 3), masks.mask_of(masks.MASK_FULL, 4)
            )

    def test_mask_with_empty_row_is_rejected(self) -> None:
        with pytest.raises(AssemblyError):
            masks.combine_masks(((True, True),), ((False, False),))

    def test_intersection_turning_a_row_empty_is_rejected(self) -> None:
        """**两张各自合法的掩码，交集却可以什么都没有**——这一条必须在组合处挡住."""
        with pytest.raises(AssemblyError):
            masks.combine_masks(((True, False),), ((False, True),))

    def test_empty_mask_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            masks.combine_masks(())

    def test_ragged_mask_is_rejected(self) -> None:
        with pytest.raises(ShapeError):
            masks.combine_masks(((True, True), (True,)))

    def test_zero_width_mask_is_rejected(self) -> None:
        with pytest.raises(ShapeError):
            masks.combine_masks(((), ()))


class TestMaskQueries:
    """掩码的各种读数（形状、转浮点、摘要、熵上限）."""

    def test_shape_of_mask(self) -> None:
        assert masks.shape_of_mask(masks.mask_of(masks.MASK_CAUSAL, 4)) == (4, 4)

    def test_rectangular_mask_has_no_causality(self) -> None:
        """长方形掩码没有“因果”可言（那份性质只在同一段序列内部有定义）."""
        rectangular = tuple(tuple(True for _ in range(5)) for _ in range(4))
        with pytest.raises(ShapeError):
            masks.mask_is_causal(rectangular)

    def test_rectangular_shape_query(self) -> None:
        rectangular = tuple(tuple(True for _ in range(5)) for _ in range(4))
        assert masks.mask_is_shape_only(rectangular, 4, 5)
        assert masks.shape_of_mask(rectangular) == (4, 5)

    def test_mask_to_floats(self) -> None:
        mask = masks.mask_of(masks.MASK_CAUSAL, 2)
        assert masks.mask_to_floats(mask) == ((1.0, 0.0), (1.0, 1.0))

    def test_causal_mask_has_no_leaks(self) -> None:
        assert masks.mask_leaks(masks.mask_of(masks.MASK_CAUSAL, 4)) == ()

    def test_full_mask_leaks_are_the_upper_triangle(self) -> None:
        leaks = masks.mask_leaks(masks.mask_of(masks.MASK_FULL, 3))
        assert leaks == ((0, 1), (0, 2), (1, 2))

    def test_mask_without_first_diagonal_is_not_causal(self) -> None:
        """``mask[0][0]`` 为假时第 0 行看不到自己——本包直接判“不是因果的”."""
        assert not masks.mask_is_causal(((False, True), (True, True)))

    def test_fits_sequence_checks_length(self) -> None:
        mask = masks.mask_of(masks.MASK_FULL, 4)
        assert masks.mask_fits_sequence(mask, 4) == mask
        with pytest.raises(ShapeError):
            masks.mask_fits_sequence(mask, 5)

    def test_summary_line_reports_the_shape_and_the_verdict(self) -> None:
        line = masks.mask_summary(masks.mask_of(masks.MASK_CAUSAL, 4))
        assert "4×4" in line and "因果" in line and "允许 10/16" in line

    def test_summary_of_rectangular_mask(self) -> None:
        rectangular = tuple(tuple(True for _ in range(5)) for _ in range(4))
        assert "非方阵" in masks.mask_summary(rectangular)

    def test_entropy_ceiling_of_full_is_ln_n(self) -> None:
        import math

        assert masks.mask_entropy_ceiling(masks.mask_of(masks.MASK_FULL, 4)) == pytest.approx(
            math.log(4)
        )

    def test_entropy_ceiling_of_causal_is_lower(self) -> None:
        import math

        causal = masks.mask_entropy_ceiling(masks.mask_of(masks.MASK_CAUSAL, 4))
        assert causal == pytest.approx(sum(math.log(i + 1) for i in range(4)) / 4)
        assert causal < masks.mask_entropy_ceiling(masks.mask_of(masks.MASK_FULL, 4))


class TestDependencyFromFloats:
    """把实测读数按阈值变成掩码（``tolerance = 0`` 时它是精确的）."""

    def test_zero_tolerance_is_exact(self) -> None:
        table = ((1.0, 0.0), (0.0, 1.0))
        assert masks.dependency_from_floats(table) == ((True, False), (False, True))

    def test_large_tolerance_swallows_everything(self) -> None:
        """**把阈值调大，这张表会退化**——接口存在的意义就是让人亲手看到这件事."""
        table = ((1.0, 0.0), (0.0, 1.0))
        assert masks.dependency_from_floats(table, tolerance=2.0) == (
            (False, False),
            (False, False),
        )

    def test_negative_tolerance_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            masks.dependency_from_floats(((1.0,),), tolerance=-0.5)

    def test_non_finite_tolerance_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            masks.dependency_from_floats(((1.0,),), tolerance=float("inf"))

    def test_non_number_tolerance_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            masks.dependency_from_floats(((1.0,),), tolerance="0")  # type: ignore[arg-type]

    def test_empty_table_is_rejected(self) -> None:
        with pytest.raises(ShapeError):
            masks.dependency_from_floats(())

    def test_ragged_table_is_rejected(self) -> None:
        with pytest.raises(ShapeError):
            masks.dependency_from_floats(((1.0, 0.0), (1.0,)))


class TestMaskDiscipline:
    """一条纪律：掩码是**表**，因此它自己也要被检查."""

    def test_unknown_kind_message_lists_the_options(self) -> None:
        with pytest.raises(ParameterError) as error:
            masks.mask_of("nope", 2)
        assert "full" in str(error.value) and "causal" in str(error.value)

    def test_padding_row_rule_is_documented_in_the_error(self) -> None:
        with pytest.raises(AssemblyError) as error:
            masks.combine_masks(((True, False),), ((False, True),))
        assert "对角线" in str(error.value) or "padding" in str(error.value)
