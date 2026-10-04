"""``hf_source.study``：七张表（day085 / M7-D9）."""

from __future__ import annotations

import math

import pytest

from smart_research_agent.hf_source import study
from smart_research_agent.hf_source.errors import ParameterError


def test_filter_study_covers_every_case() -> None:
    """过滤器表逐行对应一个配置，而且**两次调用逐位相同**（读数可复现）."""
    rows = study.filter_study()
    assert len(rows) == len(study.FILTER_CASES)
    assert rows == study.filter_study()
    assert rows[0].kept == len(study.FILTER_LOGITS)
    assert rows[3].kept == 2
    assert set(rows[0].to_dict()) == {"label", "kept", "top_probability", "entropy"}


def test_filter_study_shows_temperature_and_budget() -> None:
    """温度只改分布形状、不改保留个数；top-k 与 top-p 改的是保留个数."""
    rows = {row.label: row for row in study.filter_study()}
    assert rows["temperature=0.5"].kept == rows["raw"].kept
    assert rows["temperature=0.5"].entropy < rows["raw"].entropy < rows["temperature=2.0"].entropy
    assert rows["top_k=2"].kept < rows["top_k=4"].kept
    assert rows["top_p=0.5"].kept < rows["top_p=0.9"].kept


def test_top_k_study_keeps_a_constant_count() -> None:
    """top-k 的保留个数**完全由参数决定**（与数据无关）."""
    rows = study.top_k_study()
    assert [row.kept for row in rows] == [1, 2, 3, 6]
    assert rows[0].top_probability == 1.0
    assert rows[-1].kept == len(study.FILTER_LOGITS)


def test_nucleus_study_keeps_at_least_the_mass_of_p() -> None:
    """核的累积概率**永远不小于 p**——这是"留下的集合刚好超过 p"的直接推论."""
    rows = study.nucleus_study()
    for row in rows:
        assert row.cumulative >= row.top_p
        assert row.kept + row.removed == len(study.FILTER_LOGITS)
        assert row.kept >= 1


def test_nucleus_study_count_grows_with_p() -> None:
    """``p`` 越大留下的候选越多（保留个数是数据决定的，因此这个单调性可断言）."""
    rows = study.nucleus_study()
    counts = [row.kept for row in rows]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_sampling_study_matches_generate() -> None:
    """四行读数与直接调用 ``generate`` 一致（表不是另写一份实现）."""
    from smart_research_agent.hf_source.generation import generate

    rows = {row.strategy: row for row in study.sampling_study()}
    greedy = generate(
        study.toy_logits,
        (1, 3),
        study.GenerationSettings(max_new_tokens=6, do_sample=False, seed=7),  # type: ignore[attr-defined]
        vocab=study.TOY_VOCAB,
    )
    assert rows["greedy"].generated == greedy.generated
    assert len(rows) == len(study.SAMPLING_CASES)
    assert all(len(row.generated) == 6 for row in rows.values())


def test_sampling_study_shows_two_different_budgets() -> None:
    """``top_k=2`` 每步只留 2 个候选；``top_p`` 的保留个数**随步变化**（由数据决定）."""
    rows = {row.strategy: row for row in study.sampling_study()}
    assert set(rows["top_k=2"].kept) == {2}
    assert set(rows["sample/T=1"].kept) == {study.TOY_VOCAB}
    assert max(rows["top_p=0.7"].kept) < study.TOY_VOCAB
    assert len(set(rows["top_p=0.7"].kept)) > 1


def test_beam_study_covers_widths_and_penalties() -> None:
    """beam 表含贪心一行与三行 beam，每行都带归一化分数."""
    rows = study.beam_study()
    assert len(rows) == len(study.BEAM_CASES)
    assert rows[0].beams == 1
    assert all(len(row.generated) == 5 for row in rows)
    assert any("长度惩罚" in row.note for row in rows[1:])


def test_beam_study_differs_from_greedy() -> None:
    """beam 的第一步**不是**贪心的第一步——它优化的是整条序列的累积对数概率."""
    rows = study.beam_study()
    greedy = rows[0].generated
    beam = rows[1].generated
    assert greedy != beam
    assert greedy[0] != beam[0]
    assert set(beam[1:]) == {beam[0]}


def test_activation_study_anchors() -> None:
    """``x = 0`` 处差为 0；整张表的最大差落在 1e-4 量级（不是"差不多"）."""
    rows = study.activation_study()
    zero = [row for row in rows if row.point == 0.0][0]
    assert zero.gap == 0.0
    assert zero.gelu == 0.0 and zero.gelu_new == 0.0
    worst = max(row.gap for row in rows)
    assert 1e-5 < worst < 1e-3
    assert set(rows[0].to_dict()) == {"point", "gelu", "gelu_new", "gap"}


def test_layer_norm_study_matches_the_formula() -> None:
    """实测方差与 ``σ²/(σ²+eps)`` 一致（day079 第 3.2 节那条公式在三个 eps 上复核）."""
    rows = study.layer_norm_study()
    assert [row.name for row in rows] == ["gpt2", "bert", "t5"]
    for row in rows:
        assert row.max_gap < 1e-12
        assert 0.0 < row.measured_variance <= 1.0
        assert row.distance_to_one > 0.0
        assert row.to_dict()["epsilon"] == row.epsilon


def test_layer_norm_study_distance_shrinks_with_epsilon() -> None:
    """eps 越小，"方差离 1 多远"越小——**它由 eps 决定，不是误差**."""
    rows = {row.name: row for row in study.layer_norm_study()}
    assert rows["bert"].distance_to_one < rows["t5"].distance_to_one < rows["gpt2"].distance_to_one
    assert rows["gpt2"].distance_to_one == pytest.approx(5e-6, rel=1e-3)


def test_block_study_reports_both_placements() -> None:
    """两个块在同一份参数下的增益各一行，摆放与激活都从画像来."""
    rows = study.block_study()
    assert [row.name for row in rows] == ["gpt2", "bert"]
    assert rows[0].placement == "pre" and rows[0].activation == "gelu_new"
    assert rows[1].placement == "post" and rows[1].activation == "gelu"
    assert all(row.gain > 0.0 for row in rows)
    assert set(rows[0].to_dict()) == {"name", "placement", "activation", "gain"}


def test_toy_logits_is_deterministic_and_guarded() -> None:
    """玩具模型完全确定（否则"同一个种子"就不是策略的性质）."""
    assert study.toy_logits((1, 3)) == study.toy_logits((1, 3))
    assert len(study.toy_logits((1, 3))) == study.TOY_VOCAB
    with pytest.raises(ParameterError):
        study.toy_logits(())


def test_study_lines_render_every_table() -> None:
    """文本渲染含七个小节标题（演示脚本直接打印它）."""
    lines = study.study_lines()
    headers = [line for line in lines if line.startswith("==")]
    assert len(headers) == 7
    assert any("四个 warper" in header for header in headers)
    assert any("LayerNorm 的三个 eps" in header for header in headers)
    assert all(
        isinstance(value, float)
        for row in study.layer_norm_study()
        for value in (row.measured_variance, row.theory_variance)
    )
    assert math.isfinite(study.activation_study()[0].gap)
