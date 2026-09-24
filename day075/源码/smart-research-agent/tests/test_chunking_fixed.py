"""day062 固定长度分块（``chunking.fixed``）的单元测试.

两条主线：

1. ``window_plan`` 的算术——参数一确定，块数与放大量就能**算出来**；
2. 窗口切分的性质——长度不超预算、块文本是原文子串、覆盖率有上界、
   以及"它必然切开代码块"这条定义性弱点。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.chunking import (
    STRATEGY_FIXED,
    ChunkingError,
    ChunkPolicy,
    build_chunker,
    default_policy,
    window_plan,
)
from smart_research_agent.documents import BLOCK_CODE
from tests.chunking_samples import chars, guide_document, text_document

GUIDE_CHARS = 614  # GUIDE_MARKDOWN 规范化后的长度（见 samples 的注释）


def split_guide(policy: ChunkPolicy):  # type: ignore[no-untyped-def]
    """按给定参数切样本文档（测试里重复最多的一行，抽出来）."""
    return build_chunker(STRATEGY_FIXED, policy, measurer=chars()).split(guide_document())


# --------------------------------------------------------------------------- #
# window_plan：参数 → 代价的算术
# --------------------------------------------------------------------------- #


def test_window_plan_matches_the_measured_chunk_count() -> None:
    """实测校验：614 字符 / 320 预算 / 48 重叠 → 3 块（与真实切分一致）."""
    policy = default_policy(STRATEGY_FIXED)
    plan = window_plan(GUIDE_CHARS, policy)
    assert plan["step"] == 272
    assert plan["chunks"] == 3
    assert split_guide(policy).count == plan["chunks"]


def test_window_plan_formula_boundaries() -> None:
    """三个边界都要对：空文档 / 刚好装下 / 多一个字符."""
    policy = ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=320, overlap_tokens=48)
    assert window_plan(0, policy)["chunks"] == 0
    assert window_plan(320, policy)["chunks"] == 1
    assert window_plan(321, policy)["chunks"] == 2
    # 与"按步长向上取整"的朴素公式一致（两种写法必须给出同一个数）
    assert window_plan(614, policy)["chunks"] == math.ceil((614 - 320) / 272) + 1


def test_window_plan_reports_overlap_cost() -> None:
    """放大量 = overlap / step：默认参数下 48/272 ≈ 17.6%."""
    policy = default_policy(STRATEGY_FIXED)
    plan = window_plan(GUIDE_CHARS, policy)
    assert plan["amplification"] == pytest.approx(0.1765, abs=1e-4)
    assert plan["overlap_units"] == (plan["chunks"] - 1) * 48
    assert plan["total_units"] == GUIDE_CHARS + plan["overlap_units"]


def test_window_plan_without_overlap_has_no_amplification() -> None:
    """重叠为 0 时放大量为 0（这是"成本最小"的配置）."""
    plan = window_plan(GUIDE_CHARS, ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=320))
    assert plan["amplification"] == 0.0
    assert plan["overlap_units"] == 0
    assert plan["total_units"] == GUIDE_CHARS


# --------------------------------------------------------------------------- #
# 切分
# --------------------------------------------------------------------------- #


def test_fixed_chunks_stay_within_budget() -> None:
    """每一块都不超预算——**固定窗口最硬的一条性质**."""
    result = split_guide(
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=60, overlap_tokens=12)
    )
    assert result.count == 13
    assert all(chunk.token_count <= 60 for chunk in result.chunks)
    assert result.token_stats()["max"] == 60


def test_fixed_chunk_text_is_an_exact_substring_of_the_document() -> None:
    """``document.text[start:end] == chunk.text``：引用能逐字对回原文."""
    document = guide_document()
    result = build_chunker(
        STRATEGY_FIXED,
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=80, overlap_tokens=16),
        measurer=chars(),
    ).split(document)
    for chunk in result.chunks:
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text


def test_fixed_coverage_has_a_provable_upper_bound_on_the_shortfall() -> None:
    """覆盖率不会显著低于 1：差额上界是 ``2 × (块数 - 1) + 1``.

    这个上界来自"块首尾的空白被修剪"（块之间的空行与原文结尾的换行）。
    **有了它，"覆盖率 98% 是否正常"就不再是一个感觉问题。**
    """
    result = split_guide(
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=60, overlap_tokens=12)
    )
    shortfall = result.doc_chars - result.total_chars
    assert shortfall <= 2 * (result.count - 1) + 1
    assert result.coverage >= 1.0


def test_fixed_without_overlap_is_a_partition() -> None:
    """重叠为 0 时块是原文的一个**划分**：总长 ≈ 原长 - 被修剪的空白."""
    document = guide_document()
    result = build_chunker(
        STRATEGY_FIXED,
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=64, overlap_tokens=0),
        measurer=chars(),
    ).split(document)
    assert result.count == math.ceil(len(document.text) / 64)
    assert result.duplication_ratio == 0.0
    assert result.doc_chars - result.total_chars <= 2 * (result.count - 1) + 1


def test_fixed_overlap_raises_cost_by_the_predicted_ratio() -> None:
    """重叠的代价可以被预测：实测放大量与 ``overlap/step`` 相差不超过几个字符."""
    document = guide_document()
    policy = ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=64, overlap_tokens=16)
    result = build_chunker(STRATEGY_FIXED, policy, measurer=chars()).split(document)
    plan = window_plan(len(document.text), policy)
    assert result.total_chars <= plan["total_units"]
    assert plan["total_units"] - result.total_chars <= 2 * result.count
    assert result.duplication_ratio > 0


def test_fixed_split_cuts_inside_the_code_block() -> None:
    """固定窗口**不认识代码块**：它必然把代码从中间切开（这是它的定义）.

    这条测试把"弱点"变成可核对的证据：同一份文档在结构策略下代码块
    是完整的（见 ``test_chunking_structural``），在这里必然被切碎——
    **没有任何一块装得下整段代码，而代码的第一行与最后一行落在两块里。**
    """
    document = guide_document()
    code_block = next(block for block in document.blocks if block.kind == BLOCK_CODE)
    result = build_chunker(
        STRATEGY_FIXED,
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=60, overlap_tokens=0),
        measurer=chars(),
    ).split(document)
    assert not any(code_block.text in chunk.text for chunk in result.chunks)
    start = document.text.find(code_block.text)
    assert start >= 0
    covered = [
        chunk
        for chunk in result.chunks
        if chunk.start_char < start + len(code_block.text) and chunk.end_char > start
    ]
    assert len(covered) >= 2


def test_fixed_needs_no_separators_at_all() -> None:
    """一份没有标点、没有换行的长串照样能切（固定窗口不依赖任何结构）."""
    document = text_document("a" * 200, source="docs/log.txt")
    result = build_chunker(
        STRATEGY_FIXED,
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=64, overlap_tokens=0),
        measurer=chars(),
    ).split(document)
    assert result.count == math.ceil(200 / 64)
    assert all(chunk.token_count <= 64 for chunk in result.chunks)


def test_fixed_describe_states_its_weaknesses() -> None:
    """自述表要同时写出强项与弱项（选型依据来自它，而不是来自感觉）."""
    described = build_chunker(STRATEGY_FIXED, measurer=chars()).describe()
    assert described["name"] == STRATEGY_FIXED
    assert described["supports_overlap"] is True
    assert described["uses_similarity_percentile"] is False
    assert any("句子中间" in item for item in described["weaknesses"])
    assert any("代码" in item for item in described["weaknesses"])


def test_fixed_policy_rejects_overlap_equal_to_budget() -> None:
    """``overlap == max_tokens`` 时步长为 0，切分不会终止——构造期就报错."""
    with pytest.raises(ChunkingError) as excinfo:
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=64, overlap_tokens=64)
    assert "原地踏步" in str(excinfo.value)
