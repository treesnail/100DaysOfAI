"""``explainability.study`` 的测试：五张表（day083）."""

from __future__ import annotations

import math

import pytest

from smart_research_agent.arch_variants.types import (
    VARIANT_DECODER_ONLY,
    VARIANT_ENCODER_DECODER,
    VARIANT_ENCODER_ONLY,
)
from smart_research_agent.explainability import study
from smart_research_agent.explainability.errors import (
    AssemblyError,
    ParameterError,
    ShapeError,
)
from tests import explain_samples as samples


class TestHelpers:
    """样本与参数（**每一组实验都从这几处取**，因此换变体与换随机性不混）."""

    def test_default_shape_matches_the_day082_samples(self) -> None:
        shape = study.default_shape()
        assert (shape.tokens, shape.source_length, shape.hidden, shape.layers) == (4, 5, 6, 3)

    def test_study_params_are_reproducible(self) -> None:
        first = study.study_params(VARIANT_DECODER_ONLY).flatten()
        second = study.study_params(VARIANT_DECODER_ONLY).flatten()
        assert first == second

    def test_study_inputs_are_reproducible(self) -> None:
        assert study.study_inputs() == study.study_inputs()

    def test_study_params_rejects_an_unknown_variant(self) -> None:
        """变体名的白名单在 **day082** 那一层（两包的失败族是兄弟）."""
        from smart_research_agent.arch_variants.errors import ParameterError as VariantError

        with pytest.raises(VariantError):
            study.study_params("bert")

    def test_all_layer_records_counts(self) -> None:
        records = samples.sample_layer_records(heads=2)
        assert len(records) == 6  # 3 层 × 2 头
        assert records[0].label == "层 0 · 头 0"

    def test_all_layer_records_need_a_source_for_two_streams(self) -> None:
        with pytest.raises(AssemblyError):
            study.all_layer_records(
                study.study_params(VARIANT_ENCODER_DECODER), study.study_inputs()
            )

    def test_all_layer_records_rejects_non_params(self) -> None:
        with pytest.raises(ParameterError):
            study.all_layer_records("params", study.study_inputs())  # type: ignore[arg-type]


class TestLayerEntropyStudy:
    """第 ① 组：逐层熵与天花板."""

    def test_study_is_green_and_has_one_row_per_layer(self) -> None:
        result = study.layer_entropy_study()
        assert result.ok
        assert len(result.profiles) == 3
        assert len(result.table_lines()) == 3

    def test_records_count_and_heads(self) -> None:
        result = study.layer_entropy_study(heads=2)
        assert result.records == 6
        assert result.heads == 2

    def test_every_layer_sits_at_its_own_ceiling_direction(self) -> None:
        """未训练的模型接近均匀（归一化熵接近 1）——这是**可复核**的一句读数，不是断言"应该"."""
        result = study.layer_entropy_study()
        assert all(0.0 <= profile.normalized_entropy <= 1.0 for profile in result.profiles)
        assert all(verdict in ("尖", "中等", "接近均匀") for verdict in result.verdicts())

    def test_to_dict_carries_the_summary(self) -> None:
        payload = study.layer_entropy_study().to_dict()
        assert "归一化熵" in payload["summary"] or "平均归一化熵" in payload["summary"]
        assert len(payload["profiles"]) == 3
        assert payload["ok"] is True

    def test_empty_study_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            study.LayerEntropyStudy(variant=VARIANT_DECODER_ONLY, heads=1, profiles=(), records=0)

    def test_variant_is_validated(self) -> None:
        """变体名的白名单在 day082 那一层——因此这里抛的是那边的 ParameterError."""
        from smart_research_agent.arch_variants.errors import ParameterError as VariantError

        with pytest.raises(VariantError):
            study.LayerEntropyStudy(variant="bert", heads=1, profiles=(), records=0)

    def test_one_head_also_works(self) -> None:
        result = study.layer_entropy_study(heads=1)
        assert result.records == 3 and result.ok


class TestRedundancyStudy:
    """第 ② 组：头间余弦."""

    def test_diagonal_is_one_and_off_diagonal_is_high_for_untrained_heads(self) -> None:
        result = study.redundancy_study()
        assert result.diagonal_is_one
        assert 0.0 < result.off_diagonal_mean() <= 1.0

    def test_table_has_one_line_per_layer(self) -> None:
        assert len(study.redundancy_study().table_lines()) == 3

    def test_one_head_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            study.redundancy_study(heads=1)

    def test_to_dict(self) -> None:
        payload = study.redundancy_study().to_dict()
        assert payload["diagonal_is_one"] is True
        assert len(payload["matrices"]) == 3

    def test_empty_study_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            study.RedundancyStudy(variant=VARIANT_DECODER_ONLY, labels=(), matrices=())


class TestOffsetStudy:
    """第 ③ 组：偏移质量."""

    def test_three_offsets_per_layer(self) -> None:
        result = study.offset_study()
        assert result.offsets == (0, 1, 2)
        assert len(result.values) == 3
        assert result.value_at(0, 0) >= result.value_at(0, 2)

    def test_table_lines_include_a_header(self) -> None:
        lines = study.offset_study().table_lines()
        assert lines[0].startswith("偏移")
        assert len(lines) == 4

    def test_value_at_checks_bounds(self) -> None:
        result = study.offset_study()
        with pytest.raises(ParameterError):
            result.value_at(9, 0)
        with pytest.raises(ParameterError):
            result.value_at(0, 5)
        with pytest.raises(ParameterError):
            result.value_at(True, 0)  # type: ignore[arg-type]

    def test_empty_offsets_are_rejected(self) -> None:
        with pytest.raises(ParameterError):
            study.offset_study(offsets=())

    def test_to_dict(self) -> None:
        payload = study.offset_study().to_dict()
        assert payload["offsets"] == [0, 1, 2]
        assert len(payload["values"]) == 3

    def test_empty_values_are_rejected(self) -> None:
        with pytest.raises(ParameterError):
            study.OffsetStudy(variant=VARIANT_DECODER_ONLY, offsets=(0,), values=())

    def test_row_length_must_match_offsets(self) -> None:
        with pytest.raises(ShapeError):
            study.OffsetStudy(
                variant=VARIANT_DECODER_ONLY, offsets=(0, 1), values=((0.5,),)
            )

    def test_a_single_offset_is_allowed(self) -> None:
        result = study.offset_study(offsets=(0,))
        assert result.offsets == (0,)
        assert result.value_at(1, 0) > 0.0


class TestRolloutStudy:
    """第 ④ 组：滚动的集中度."""

    def test_four_lines_and_a_profile(self) -> None:
        result = study.rollout_study()
        assert len(result.table_lines()) == 4
        assert result.layers == 3

    def test_gain_is_the_difference_of_the_two_entropies(self) -> None:
        result = study.rollout_study()
        assert result.gain == pytest.approx(result.last_entropy - result.rolled_entropy)

    def test_more_focused_is_a_boolean_reading(self) -> None:
        result = study.rollout_study()
        assert isinstance(result.more_focused, bool)

    def test_to_dict(self) -> None:
        payload = study.rollout_study().to_dict()
        assert payload["alpha"] == 0.5
        assert "profile" in payload and "more_focused" in payload

    def test_alpha_flows_through(self) -> None:
        result = study.rollout_study(alpha=0.9)
        assert result.alpha == 0.9

    def test_variant_is_validated(self) -> None:
        from smart_research_agent.arch_variants.errors import ParameterError as VariantError

        with pytest.raises(VariantError):
            study.RolloutStudy(
                variant="bert",
                alpha=0.5,
                last_entropy=1.0,
                rolled_entropy=1.0,
                gain=0.0,
                profile=study.rollout_study().profile,
                layers=1,
            )


class TestCeilingStudy:
    """第 ⑤ 组：两种掩码的天花板."""

    def test_two_entries_and_a_ratio(self) -> None:
        result = study.ceiling_study()
        assert [kind for kind, _ in result.entries] == ["full", "causal"]
        assert result.ratio < 1.0

    def test_ceilings_are_hand_computable(self) -> None:
        result = study.ceiling_study()
        assert result.ceiling_of("full") == pytest.approx(math.log(4))
        expected = sum(math.log(index + 1) for index in range(4)) / 4
        assert result.ceiling_of("causal") == pytest.approx(expected)

    def test_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            study.ceiling_study().ceiling_of("triangular")

    def test_table_lines(self) -> None:
        lines = study.ceiling_study().table_lines()
        assert len(lines) == 3
        assert "因果 / 全开" in lines[-1]

    def test_to_dict(self) -> None:
        payload = study.ceiling_study().to_dict()
        assert payload["tokens"] == 4
        assert len(payload["entries"]) == 2

    def test_bad_tokens_are_rejected(self) -> None:
        with pytest.raises(ParameterError):
            study.CeilingStudy(tokens=True, entries=(("full", 1.0),), ratio=1.0)
        with pytest.raises(ParameterError):
            study.CeilingStudy(tokens=0, entries=(("full", 1.0),), ratio=1.0)

    def test_empty_entries_are_rejected(self) -> None:
        with pytest.raises(ParameterError):
            study.CeilingStudy(tokens=4, entries=(), ratio=1.0)

    def test_a_longer_sequence_raises_the_causal_ceiling_share(self) -> None:
        """n 越大，因果掩码的天花板越接近全开的（"差别在长序列上更贵"的数字基础）."""
        short = study.ceiling_study(samples.sample_shape(tokens=4)).ratio
        long = study.ceiling_study(samples.sample_shape(tokens=64)).ratio
        assert long > short


class TestCatalogHelpers:
    """几张小的口径表（教程里引用的那几行都要有数据源）."""

    def test_how_to_read_has_five_steps(self) -> None:
        steps = study.how_to_read()
        assert len(steps) == 5
        assert steps[0].startswith("①")

    def test_variant_labels(self) -> None:
        assert study.variant_labels() == (
            VARIANT_ENCODER_ONLY,
            VARIANT_DECODER_ONLY,
            VARIANT_ENCODER_DECODER,
        )

    def test_all_ceilings_uses_the_matching_mask_per_variant(self) -> None:
        entries = dict((variant, ceiling) for variant, ceiling, _ in study.all_ceilings())
        assert entries[VARIANT_ENCODER_ONLY] == pytest.approx(math.log(4))
        assert entries[VARIANT_DECODER_ONLY] < entries[VARIANT_ENCODER_ONLY]

    def test_all_ceilings_baseline_is_one(self) -> None:
        assert all(baseline == 1.0 for _variant, _ceiling, baseline in study.all_ceilings())

    def test_study_seeds_are_documented(self) -> None:
        assert (study.PARAMETER_SEED, study.INPUT_SEED, study.SOURCE_SEED) == (7, 21, 13)
