"""``arch_variants.study`` 的测试：四张表（掩码的账、家底、前缀稳定性、用错的代价）."""

from __future__ import annotations

import pytest

from smart_research_agent.arch_variants import study
from smart_research_agent.arch_variants.errors import NumericError, ParameterError
from smart_research_agent.arch_variants.probe import STREAM_ENCODER, STREAM_SOURCE
from smart_research_agent.arch_variants.types import (
    VARIANTS,
    VARIANT_DECODER_ONLY,
    VARIANT_ENCODER_DECODER,
    VARIANT_ENCODER_ONLY,
    VARIANT_EXAMPLES,
)
from tests import arch_variants_samples as samples


class TestLeakStudy:
    """第 ① 组：五个正常行 + 两个反证行."""

    def test_study_is_green(self) -> None:
        result = study.leak_study(samples.sample_shape())
        assert result.ok
        assert len(result.rows) == 7
        assert len(result.normal_rows) == 5
        assert len(result.counterexample_rows) == 2

    def test_every_row_is_bit_exact(self) -> None:
        """**含反证行**：换错掩码并不改变“挡掉的格子逐位为 0”这件事."""
        result = study.leak_study(samples.sample_shape())
        assert result.exact_rows == 7
        assert all(row.blocked_gap == 0.0 for row in result.rows)

    def test_normal_rows_are_consistent(self) -> None:
        result = study.leak_study(samples.sample_shape())
        assert all(row.consistent for row in result.normal_rows)
        assert all(row.leaks == 0 and row.missing == 0 for row in result.normal_rows)

    def test_counterexamples_are_inconsistent_in_both_directions(self) -> None:
        """**用错的代价**：一个方向是越界（错看未来），另一个方向是缺失（丢掉双向）."""
        result = study.leak_study(samples.sample_shape())
        leaks = result.row_of("反证：decoder_only 用全开掩码")
        missing = result.row_of("反证：encoder_only 用因果掩码")
        assert leaks.leaks == 6 and leaks.missing == 0
        assert missing.leaks == 0 and missing.missing == 6
        assert leaks.worst_leak > 0.0
        assert "如期不一致" in leaks.verdict

    def test_reach_profiles_encode_the_variant(self) -> None:
        result = study.leak_study(samples.sample_shape())
        assert result.row_of("encoder_only（全开）").reach == (4, 4, 4, 4)
        assert result.row_of("decoder_only（因果）").reach == (1, 2, 3, 4)
        assert result.row_of("encoder_decoder（编码器流）").reach == (5, 5, 5, 5, 5)
        assert result.row_of("encoder_decoder（交叉那一路）").reach == (5, 5, 5, 5)

    def test_cross_row_is_rectangular(self) -> None:
        result = study.leak_study(samples.sample_shape())
        cross = result.row_of("encoder_decoder（交叉那一路）")
        assert cross.stream == STREAM_SOURCE
        assert cross.total_pairs == 20

    def test_encoder_row_uses_the_encoder_stream(self) -> None:
        result = study.leak_study(samples.sample_shape())
        assert result.row_of("encoder_decoder（编码器流）").stream == STREAM_ENCODER

    def test_unknown_label_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            study.leak_study(samples.sample_shape()).row_of("nope")

    def test_table_lines_and_dict(self) -> None:
        result = study.leak_study(samples.sample_shape())
        assert len(result.table_lines()) == 7
        payload = result.to_dict()
        assert payload["ok"] is True and len(payload["rows"]) == 7

    def test_empty_study_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            study.LeakStudy(shape=samples.sample_shape(), rows=())

    def test_row_to_dict_carries_the_numbers(self) -> None:
        row = study.leak_study(samples.sample_shape()).rows[0]
        payload = row.to_dict()
        assert payload["allowed_pairs"] == 16 and payload["verdict"] == "一致"

    def test_counts_against_checks_shapes(self) -> None:
        from smart_research_agent.arch_variants import probe

        report = probe.dependency_matrix(
            samples.sample_parameters(VARIANT_DECODER_ONLY), samples.sample_inputs()
        )
        with pytest.raises(NumericError):
            study._counts_against(report, ((True,),))

    def test_study_with_a_smaller_shape(self) -> None:
        result = study.leak_study(samples.sample_shape(tokens=3, sources=3, layers=2))
        assert result.ok
        assert result.row_of("decoder_only（因果）").reach == (1, 2, 3)


class TestCensusStudy:
    """第 ② 组：家底（逐个数 vs 公式算）."""

    def test_study_is_green_and_ordered(self) -> None:
        result = study.census_study(samples.sample_shape())
        assert result.ok
        assert tuple(row.variant for row in result.rows) == VARIANTS

    def test_examples_are_bert_gpt_t5(self) -> None:
        result = study.census_study(samples.sample_shape())
        assert result.examples == tuple(VARIANT_EXAMPLES[variant] for variant in VARIANTS)

    def test_sample_parameter_counts(self) -> None:
        result = study.census_study(samples.sample_shape())
        assert result.row_of(VARIANT_DECODER_ONLY).census.parameters == 1458
        assert result.row_of(VARIANT_ENCODER_DECODER).census.parameters == 3384

    def test_unknown_variant_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            study.census_study(samples.sample_shape()).row_of("bert")

    def test_table_lines_and_dict(self) -> None:
        result = study.census_study(samples.sample_shape())
        assert len(result.table_lines()) == 3
        assert result.to_dict()["shape"].startswith("tokens=4")

    def test_row_summary_line_mentions_the_parameters(self) -> None:
        row = study.census_study(samples.sample_shape()).rows[0]
        assert "参数" in row.summary_line()
        assert row.to_dict()["variant"] == row.variant


class TestPrefixStudy:
    """第 ③ 组：扰动最后一个 token，看哪几行动了."""

    def test_encoder_sees_the_whole_sequence(self) -> None:
        result = study.prefix_study(samples.sample_shape())
        assert result.row_of("encoder_only").changed_rows == (0, 1, 2, 3)

    def test_decoder_sees_only_itself(self) -> None:
        """**最直观的一张表**：扰动最后一个 token，前面三行的输出**逐位不变**."""
        result = study.prefix_study(samples.sample_shape())
        for variant in (VARIANT_DECODER_ONLY, VARIANT_ENCODER_DECODER):
            row = result.row_of(variant)
            assert row.changed_rows == (3,)
            assert row.changed_count == 1

    def test_first_row_change_is_exactly_zero_for_causal_variants(self) -> None:
        result = study.prefix_study(samples.sample_shape())
        assert result.row_of(VARIANT_DECODER_ONLY).first_row_change == 0.0
        assert result.row_of("encoder_only").first_row_change > 0.0

    def test_default_position_is_the_last_token(self) -> None:
        result = study.prefix_study(samples.sample_shape())
        assert all(row.position == 3 for row in result.rows)

    def test_explicit_position(self) -> None:
        result = study.prefix_study(samples.sample_shape(), position=0)
        assert all(row.position == 0 for row in result.rows)
        assert result.row_of("encoder_only").changed_rows == (0, 1, 2, 3)

    def test_position_out_of_range(self) -> None:
        with pytest.raises(ParameterError):
            study.prefix_study(samples.sample_shape(), position=9)

    def test_position_type_is_checked(self) -> None:
        with pytest.raises(ParameterError):
            study.prefix_study(samples.sample_shape(), position=True)  # type: ignore[arg-type]

    def test_unknown_variant_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            study.prefix_study(samples.sample_shape()).row_of("bert")

    def test_empty_study_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            study.PrefixStudy(shape=samples.sample_shape(), rows=())

    def test_table_lines_and_dict(self) -> None:
        result = study.prefix_study(samples.sample_shape())
        assert len(result.table_lines()) == 3
        payload = result.to_dict()
        assert payload["rows"][0]["changed_count"] == 4

    def test_row_to_dict(self) -> None:
        row = study.prefix_study(samples.sample_shape()).rows[1]
        assert row.to_dict()["changed_rows"] == [3]


class TestSmallTables:
    """教程里几张小的口径表（它们的数据源都必须可被断言）."""

    def test_objectives_table(self) -> None:
        table = study.objectives_table()
        assert len(table) == 3
        assert table[0][1] == "BERT" and "MLM" in table[0][3]
        assert table[1][1] == "GPT" and "CLM" in table[1][3]

    def test_variant_descriptions(self) -> None:
        descriptions = dict(study.variant_descriptions())
        assert set(descriptions) == set(VARIANTS)
        assert "因果" in descriptions[VARIANT_DECODER_ONLY]

    def test_mask_catalogue(self) -> None:
        catalogue = dict(study.mask_catalogue(samples.sample_shape()))
        assert "允许 16/16" in catalogue["full"]
        assert "允许 10/16" in catalogue["causal"]

    def test_entropy_ceilings(self) -> None:
        ceilings = dict(study.entropy_ceiling_table(samples.sample_shape()))
        assert ceilings["full"] == pytest.approx(1.3862943611198906)
        assert ceilings["causal"] == pytest.approx(0.7945134586148693)
        assert ceilings["causal"] < ceilings["full"]

    def test_mask_pair_ratio(self) -> None:
        assert study.mask_pair_ratio(samples.sample_shape()) == pytest.approx(0.625)

    def test_mask_pair_ratio_approaches_one_half(self) -> None:
        """序列越长，因果掩码砍掉的比例越接近一半——第 9 章那句话的数字基础."""
        long_shape = samples.sample_shape(tokens=64)
        assert study.mask_pair_ratio(long_shape) == pytest.approx(0.5078125)

    def test_variant_parameter_total(self) -> None:
        assert (
            study.variant_parameter_total(VARIANT_ENCODER_ONLY, samples.sample_shape()) == 1458
        )

    def test_keystone_check(self) -> None:
        params = samples.sample_parameters(VARIANT_ENCODER_ONLY)
        assert study.keystone_check(params) is True

    def test_study_summary(self) -> None:
        assert "arch_variants 实验" in study.study_summary(samples.sample_shape())


class TestStudyDiscipline:
    """四组实验的口径纪律（与前几课逐字相同）."""

    def test_studies_use_the_same_shape_by_default(self) -> None:
        result = study.leak_study()
        assert result.shape.source_length == 5

    def test_census_default_shape_matches_other_studies(self) -> None:
        assert study.census_study().shape == study.prefix_study().shape

    def test_leak_study_is_reproducible(self) -> None:
        first = study.leak_study(samples.sample_shape())
        second = study.leak_study(samples.sample_shape())
        assert first.to_dict() == second.to_dict()

    def test_notes_explain_the_two_directions(self) -> None:
        notes = " ".join(study.leak_study(samples.sample_shape()).notes)
        assert "越界" in notes and "缺失" in notes
