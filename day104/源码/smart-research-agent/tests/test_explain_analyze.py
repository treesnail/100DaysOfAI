"""``explainability.analyze`` 的测试：六个读数 + 头间余弦 + 偏移（day083）."""

from __future__ import annotations

import math

import pytest

from smart_research_agent.explainability import analyze
from smart_research_agent.explainability.errors import (
    AssemblyError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.explainability.types import HeadProfile
from tests import explain_samples as samples


class TestMassReadings:
    """对角质量与偏移质量（都能手算）."""

    def test_diagonal_mass_of_a_uniform_record(self) -> None:
        assert analyze.diagonal_mass(samples.uniform_record(4)) == pytest.approx(0.25)

    def test_diagonal_mass_of_a_one_hot_record(self) -> None:
        assert analyze.diagonal_mass(samples.one_hot_record(3)) == pytest.approx(1.0)

    def test_diagonal_mass_needs_a_square_table(self) -> None:
        with pytest.raises(AssemblyError):
            analyze.diagonal_mass(samples.cross_record(3, 5))

    def test_offset_zero_is_the_diagonal(self) -> None:
        record = samples.uniform_record(3)
        assert analyze.offset_mass(record, 0) == analyze.diagonal_mass(record)

    def test_offset_one_of_a_one_hot_record_is_zero(self) -> None:
        assert analyze.offset_mass(samples.one_hot_record(3), 1) == 0.0

    def test_offset_one_of_a_causal_record(self) -> None:
        """因果均匀表：``k = 1`` 时只有第 1 行与第 2 行有格子（各 ``1/(i+1)``）."""
        record = samples.causal_uniform_record(3)
        expected = (0.5 + 1.0 / 3.0) / 2.0
        assert analyze.offset_mass(record, 1) == pytest.approx(expected)

    def test_offset_beyond_the_table_is_zero(self) -> None:
        assert analyze.offset_mass(samples.uniform_record(3), 5) == 0.0

    def test_negative_offset_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            analyze.offset_mass(samples.uniform_record(3), -1)

    def test_bool_offset_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            analyze.offset_mass(samples.uniform_record(3), True)  # type: ignore[arg-type]

    def test_record_type_is_checked(self) -> None:
        with pytest.raises(ParameterError):
            analyze.diagonal_mass("record")  # type: ignore[arg-type]


class TestSupportAndShape:
    """支撑集、Frobenius、峰值."""

    def test_uniform_record_uses_every_cell(self) -> None:
        record = samples.uniform_record(3)
        assert analyze.support_of(record) == 9

    def test_one_hot_record_uses_the_diagonal_only(self) -> None:
        assert analyze.support_of(samples.one_hot_record(3)) == 3

    def test_threshold_can_hide_small_weights(self) -> None:
        record = samples.uniform_record(3)
        assert analyze.support_of(record, threshold=1.0) == 0

    def test_bad_threshold_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            analyze.support_of(samples.uniform_record(3), threshold=-1.0)

    def test_frobenius_of_a_uniform_record_is_one(self) -> None:
        """``Σ(1/n)² × n² = 1`` ⇒ 范数恰好是 1（手算）."""
        assert analyze.frobenius_of(samples.uniform_record(4)) == pytest.approx(1.0)

    def test_frobenius_of_a_one_hot_record_is_sqrt_n(self) -> None:
        assert analyze.frobenius_of(samples.one_hot_record(4)) == pytest.approx(2.0)

    def test_peak_of_a_uniform_record(self) -> None:
        value, index = analyze.peak_of(samples.uniform_record(3))
        assert value == pytest.approx(1 / 3) and index == 0

    def test_peak_of_a_one_hot_record(self) -> None:
        value, index = analyze.peak_of(samples.one_hot_record(3))
        assert value == 1.0 and index == 0

    def test_shape_of_record(self) -> None:
        assert analyze.shape_of_record(samples.cross_record(3, 5)) == (3, 5)


class TestProfiles:
    """profile 的构造与汇总."""

    def test_profile_of_a_uniform_record(self) -> None:
        profile = analyze.profile_of(samples.uniform_record(4))
        assert profile.entropy == pytest.approx(math.log(4))
        assert profile.ceiling == pytest.approx(math.log(4))
        assert profile.normalized_entropy == pytest.approx(1.0)
        assert profile.support == 16
        assert profile.diagonal_mass == pytest.approx(0.25)

    def test_profile_of_a_cross_record_has_no_diagonal(self) -> None:
        profile = analyze.profile_of(samples.cross_record(3, 5))
        assert profile.diagonal_mass == 0.0
        assert any("对角线" in note for note in profile.notes)

    def test_profiles_reject_empty(self) -> None:
        with pytest.raises(ParameterError):
            analyze.profiles_of(())

    def test_threshold_flows_into_the_notes(self) -> None:
        profile = analyze.profile_of(samples.uniform_record(3), threshold=0.2)
        assert "0.2" in profile.notes[0]
        assert profile.support == 9

    def test_a_degenerate_threshold_is_rejected_with_a_hint(self) -> None:
        """阈值大于每一格时支撑集为 0——**报告里不能出现"什么都没有"的读数**.

        这条拒绝的理由写在错误消息里：softmax 的每一行和为 1，
        因此"每一格都不超过阈值"只可能是阈值太大（而不是这张表有问题）。
        """
        with pytest.raises(NumericError) as error:
            analyze.profile_of(samples.uniform_record(3), threshold=0.9)
        assert "阈值" in str(error.value)


class TestRedundancy:
    """逐格余弦（"几个头在看同一件事"）."""

    def test_cosine_of_a_table_with_itself_is_one(self) -> None:
        record = samples.uniform_record(3)
        assert analyze.cosine_similarity(record, record) == pytest.approx(1.0)

    def test_cosine_of_uniform_against_one_hot(self) -> None:
        """**手算**：``dot = n·(1/n) = 1``，``|u| = 1``、``|o| = √n`` ⇒ ``1/√n``."""
        value = analyze.cosine_similarity(samples.uniform_record(4), samples.one_hot_record(4))
        assert value == pytest.approx(1.0 / math.sqrt(4))

    def test_cosine_checks_shapes(self) -> None:
        with pytest.raises(ShapeError):
            analyze.cosine_similarity(samples.uniform_record(3), samples.uniform_record(4))

    def test_redundancy_matrix_diagonal_is_one(self) -> None:
        matrix = analyze.head_redundancy(samples.sample_head_records(heads=2))
        assert matrix[0][0] == pytest.approx(1.0)
        assert matrix[0][1] == matrix[1][0]

    def test_redundancy_rejects_different_masks(self) -> None:
        """不同掩码的两张表之间比余弦，比的是"能看到多少"而不是"看了哪里"."""
        with pytest.raises(AssemblyError):
            analyze.head_redundancy(
                (samples.uniform_record(4), samples.causal_uniform_record(4))
            )

    def test_redundancy_rejects_different_shapes(self) -> None:
        with pytest.raises(ShapeError):
            analyze.head_redundancy((samples.uniform_record(3), samples.uniform_record(4)))

    def test_redundancy_rejects_empty(self) -> None:
        with pytest.raises(ParameterError):
            analyze.head_redundancy(())


class TestLayerProfiles:
    """逐层聚合（先平均头、再取读数）."""

    def test_three_groups_for_three_layers(self) -> None:
        profiles = analyze.layer_profiles(samples.sample_layer_records(heads=2))
        assert len(profiles) == 3
        assert all("平均 2 头" in profile.label for profile in profiles)

    def test_missing_separator_is_rejected(self) -> None:
        """标签里没有「 · 」时按层分组无从下手——因此当场拒绝，而不是丢掉那几条."""
        from smart_research_agent.explainability.types import AttentionRecord, full_mask

        odd = AttentionRecord(
            label="没有分隔符", weights=((1.0,),), mask=full_mask(1)
        )
        with pytest.raises(AssemblyError):
            analyze.layer_profiles((odd,))

    def test_empty_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            analyze.layer_profiles(())


class TestVerdicts:
    """判词与死头（"太均匀"的那几个头）."""

    def test_uniform_is_almost_flat(self) -> None:
        profile = analyze.profile_of(samples.uniform_record(3))
        assert analyze.focus_verdict(profile) == "接近均匀"

    def test_one_hot_is_sharp(self) -> None:
        profile = analyze.profile_of(samples.one_hot_record(3))
        assert analyze.focus_verdict(profile) == "尖"

    def test_verdict_threshold_must_fit(self) -> None:
        profile = analyze.profile_of(samples.uniform_record(3))
        with pytest.raises(ParameterError):
            analyze.focus_verdict(profile, sharp=1.5)

    def test_verdict_rejects_non_profile(self) -> None:
        with pytest.raises(ParameterError):
            analyze.focus_verdict("profile")  # type: ignore[arg-type]

    def test_dead_heads_finds_the_uniform_ones(self) -> None:
        dead = analyze.dead_heads(samples.sample_head_records(heads=2))
        assert len(dead) == 2

    def test_dead_heads_ignores_a_sharp_record(self) -> None:
        assert analyze.dead_heads((samples.one_hot_record(3),), threshold=0.5) == ()

    def test_dead_heads_threshold_must_be_at_most_one(self) -> None:
        with pytest.raises(ParameterError):
            analyze.dead_heads(samples.sample_head_records(heads=2), threshold=1.5)

    def test_dead_heads_threshold_type_is_checked(self) -> None:
        with pytest.raises(ParameterError):
            analyze.dead_heads(samples.sample_head_records(heads=2), threshold="0.9")

    def test_dead_threshold_constant_is_high(self) -> None:
        assert 0.0 < analyze.DEFAULT_DEAD_THRESHOLD <= 1.0


class TestGapAndSummary:
    """两个跨表比较与一行汇总."""

    def test_entropy_gap_is_positive_when_the_first_is_sharper(self) -> None:
        sharp = analyze.profile_of(samples.one_hot_record(3))
        blunt = analyze.profile_of(samples.uniform_record(3))
        assert analyze.entropy_gap(
            samples.one_hot_record(3), samples.uniform_record(3)
        ) == pytest.approx(blunt.normalized_entropy - sharp.normalized_entropy)
        assert sharp.normalized_entropy < blunt.normalized_entropy

    def test_summarise_names_the_extremes(self) -> None:
        profiles = (
            analyze.profile_of(samples.uniform_record(3)),
            analyze.profile_of(samples.one_hot_record(3)),
        )
        line = analyze.summarise(profiles)
        assert "最尖" in line and "最散" in line and "2 个读数" in line

    def test_summarise_rejects_empty(self) -> None:
        with pytest.raises(ParameterError):
            analyze.summarise(())

    def test_summarise_rejects_non_profiles(self) -> None:
        with pytest.raises(ParameterError):
            analyze.summarise((HeadProfile,))  # type: ignore[arg-type]

    def test_profile_shape_mismatch_is_visible(self) -> None:
        """两张不同形状的记录各有自己的尺寸——这本身就是一条读数."""
        left = analyze.profile_of(samples.uniform_record(3))
        right = analyze.profile_of(samples.cross_record(3, 5))
        assert (left.rows, left.columns) == (3, 3)
        assert (right.rows, right.columns) == (3, 5)
