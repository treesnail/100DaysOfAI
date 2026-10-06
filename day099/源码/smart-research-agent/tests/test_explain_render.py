"""``explainability.render`` 的测试：10 级、图例、以及**画回去**（day083）."""

from __future__ import annotations

import pytest

from smart_research_agent.explainability import render
from smart_research_agent.explainability.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.explainability.types import LEVELS, HeadProfile
from tests import explain_samples as samples


class TestLevelOf:
    """等级映射（10 级，第 k 级的区间是 ``[k/9, (k+1)/9)``）."""

    def test_zero_is_the_lowest_level(self) -> None:
        assert render.level_of(0.0) == 0
        assert render.level_char(0.0) == " "

    def test_one_is_the_highest_level(self) -> None:
        assert render.level_of(1.0) == 9
        assert render.level_char(1.0) == "@"

    def test_boundaries_use_floor(self) -> None:
        """第 k 级的**下界**属于第 k 级（向下取整），因此 1/9 是第 1 级."""
        assert render.level_of(1.0 / 9.0) == 1
        assert render.level_of(1.0 / 9.0 - 1e-12) == 0
        assert render.level_of(8.0 / 9.0) == 8

    def test_tiny_positive_is_still_level_zero(self) -> None:
        assert render.level_of(1e-12) == 0

    def test_monotone(self) -> None:
        levels = [render.level_of(index / 50.0) for index in range(51)]
        assert levels == sorted(levels)

    def test_above_one_is_rejected(self) -> None:
        """**一个权重不能大于 1**：越界只可能来自"两张表加在一起"或"没归一化"."""
        with pytest.raises(NumericError):
            render.level_of(1.0000001)

    def test_negative_is_rejected(self) -> None:
        with pytest.raises(NumericError):
            render.level_of(-0.1)

    def test_tolerance_widens_the_window(self) -> None:
        assert render.level_of(0.5, tolerance=1e-3) == 4
        assert render.level_of(-1e-9, tolerance=1e-3) == 0

    def test_non_number_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            render.level_of("0.5")  # type: ignore[arg-type]

    def test_non_finite_is_rejected(self) -> None:
        with pytest.raises(NumericError):
            render.level_of(float("nan"))


class TestCharLevel:
    """字符 → 等级（不认识的字符当场拒绝：解析侧不能猜）."""

    @pytest.mark.parametrize("level", range(len(LEVELS)))
    def test_round_trip_for_every_level(self, level: int) -> None:
        assert render.char_level(LEVELS[level]) == level

    def test_unknown_char_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            render.char_level("?")

    def test_multi_char_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            render.char_level("##")

    def test_non_string_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            render.char_level(3)  # type: ignore[arg-type]


class TestHeatmap:
    """热力图的渲染与解析（**画出来再读回去**）."""

    def test_block_has_one_line_per_row_plus_a_title_and_a_legend(self) -> None:
        record = samples.uniform_record(4)
        text = render.heatmap_block(record)
        lines = text.splitlines()
        assert record.summary_line() in lines[0]
        assert lines[-1] == render.LEGEND
        grid_rows = [line for line in lines if line.count("|") == 2]
        assert len(grid_rows) == 4

    def test_grid_rows_are_machine_parseable(self) -> None:
        record = samples.causal_uniform_record(4)
        text = render.heatmap_block(record)
        for line in text.splitlines():
            if line.count("|") == 2:
                cells = line.split("|")[1]
                assert len(cells) == 4

    def test_round_trip_of_a_uniform_record(self) -> None:
        record = samples.uniform_record(4)
        assert render.parse_heatmap(render.heatmap_block(record)) == render.normalized_levels(record)

    def test_round_trip_of_a_causal_record(self) -> None:
        """因果表的上三角是空格（第 0 级），而"空格"也要能读回来."""
        record = samples.causal_uniform_record(4)
        levels = render.parse_heatmap(render.heatmap_block(record))
        assert levels == render.normalized_levels(record)
        assert levels[0][1] == 0 and levels[1][2] == 0

    def test_round_trip_helper(self) -> None:
        assert render.round_trip(samples.uniform_record(3))
        assert render.round_trip(samples.one_hot_record(3))
        assert render.round_trip(samples.near_uniform_record(3))

    def test_parse_rejects_text_without_a_grid(self) -> None:
        with pytest.raises(ShapeError):
            render.parse_heatmap("这里没有表格")

    def test_parse_rejects_ragged_rows(self) -> None:
        text = "0 |###| x\n1 |##| x\n"
        with pytest.raises(ShapeError):
            render.parse_heatmap(text)

    def test_parse_rejects_a_bad_character(self) -> None:
        with pytest.raises(ParameterError):
            render.parse_heatmap("0 |?#?| x\n")

    def test_parse_rejects_an_empty_row(self) -> None:
        with pytest.raises(ShapeError):
            render.parse_heatmap("0 || x\n")

    def test_parse_rejects_non_text(self) -> None:
        with pytest.raises(ParameterError):
            render.parse_heatmap(123)  # type: ignore[arg-type]

    def test_parse_skips_the_title_and_the_legend(self) -> None:
        record = samples.one_hot_record(3)
        text = render.heatmap_block(record)
        assert len(render.parse_heatmap(text)) == 3

    def test_render_records_joins_blocks(self) -> None:
        records = samples.sample_head_records(heads=2)
        text = render.render_records(records)
        assert text.count(render.LEGEND) == 2

    def test_render_records_rejects_empty(self) -> None:
        with pytest.raises(ParameterError):
            render.render_records(())

    def test_render_rejects_non_record(self) -> None:
        with pytest.raises(ParameterError):
            render.heatmap_block("record")  # type: ignore[arg-type]

    def test_column_labels_can_be_turned_off(self) -> None:
        record = samples.uniform_record(3)
        with_labels = render.heatmap_block(record)
        without = render.heatmap_block(record, column_labels=False)
        assert len(with_labels.splitlines()) == len(without.splitlines()) + 1

    def test_explanation_of_a_value(self) -> None:
        assert "等级 0" in render.explanation_of(0.0)
        assert "等级" in render.explanation_of(0.5)


class TestProfileRendering:
    """profile 的条形与读数块."""

    def test_bar_of_a_uniform_record_is_full(self) -> None:
        """归一化熵贴着天花板 ⇒ 条形几乎全是 ``#``（浮点上偶尔差一格：``ln n / ln n`` 不是精确的 1）."""
        record = samples.uniform_record(3)
        from smart_research_agent.explainability.analyze import profile_of

        bar = render.profile_bar(profile_of(record), width=10)
        assert bar.count("#") >= 9

    def test_bar_of_a_one_hot_record_is_empty(self) -> None:
        record = samples.one_hot_record(3)
        from smart_research_agent.explainability.analyze import profile_of

        bar = render.profile_bar(profile_of(record), width=10)
        assert bar.strip().startswith("-" * 10)

    def test_bar_width_is_validated(self) -> None:
        record = samples.uniform_record(3)
        from smart_research_agent.explainability.analyze import profile_of

        profile = profile_of(record)
        with pytest.raises(ParameterError):
            render.profile_bar(profile, width=0)
        with pytest.raises(ParameterError):
            render.profile_bar(profile, width=True)  # type: ignore[arg-type]

    def test_bar_rejects_non_profile(self) -> None:
        with pytest.raises(ParameterError):
            render.profile_bar("profile")  # type: ignore[arg-type]

    def test_block_has_four_lines(self) -> None:
        from smart_research_agent.explainability.analyze import profile_of

        text = render.profile_block(profile_of(samples.uniform_record(3)))
        assert len(text.splitlines()) == 4

    def test_block_rejects_non_profile(self) -> None:
        with pytest.raises(ParameterError):
            render.profile_block(HeadProfile)  # type: ignore[arg-type]

    def test_render_profiles(self) -> None:
        from smart_research_agent.explainability.analyze import profiles_of

        profiles = profiles_of(samples.sample_head_records(heads=2))
        text = render.render_profiles(profiles)
        assert text.count("归一化熵") == 2

    def test_render_profiles_rejects_empty(self) -> None:
        with pytest.raises(ParameterError):
            render.render_profiles(())

    def test_legend_line(self) -> None:
        assert render.legend_line() == render.LEGEND
        assert "10 级" in render.legend_line()
