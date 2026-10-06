"""day062 ``chunking.base`` 里的策略参数与收尾器的单元测试.

覆盖 ``ChunkPolicy`` 的七条校验、``default_policy`` 的按策略默认值、
``Span`` / ``_trim`` / ``_merge`` / ``_assert_spans`` / ``_dedupe_starts``
这几个"收尾六步"里可单独测的部分，以及不变量被违反时的报错。
"""

from __future__ import annotations

import pytest

from smart_research_agent.chunking import (
    BOUNDARY_CHARS,
    CHUNKING_LIMITATIONS,
    CHUNKING_OUT_OF_SCOPE,
    DEFAULT_POLICIES,
    STRATEGIES,
    STRATEGY_FIXED,
    STRATEGY_RECURSIVE,
    STRATEGY_SEMANTIC,
    STRATEGY_STRUCTURAL,
    Chunker,
    ChunkerRegistry,
    ChunkingError,
    ChunkPolicy,
    Span,
    UnsupportedStrategy,
    build_chunker,
    chunking_boundaries,
    default_policy,
    default_registry,
    heading_positions,
    paragraph_spans,
    path_for_offset,
    snap_to_boundary,
)
from tests.chunking_samples import chars, guide_document, text_document


class ProbeChunker(Chunker):
    """一个最小的具体分块器：把给定的区间原样交出去（用来单测收尾逻辑）."""

    name = STRATEGY_FIXED

    def __init__(self, policy: ChunkPolicy, spans: list[Span], **kwargs) -> None:
        super().__init__(policy, **kwargs)
        self._given = spans

    def _spans(self, document):  # type: ignore[no-untyped-def]
        return list(self._given)

    def describe(self):  # type: ignore[no-untyped-def]
        return {"name": self.name}


class BigMeasurer:
    """一个"每个字符花 10 个单位"的度量器：用来把预算逼到装不下一个字符."""

    name = "big"

    def count(self, text: str) -> int:
        return len(text) * 10


# --------------------------------------------------------------------------- #
# ChunkPolicy 的校验
# --------------------------------------------------------------------------- #


def test_default_policy_is_per_strategy_not_global() -> None:
    """**默认值是策略的一部分**：结构策略的重叠是 0，语义策略的下限是 60."""
    assert default_policy(STRATEGY_STRUCTURAL).overlap_tokens == 0
    assert default_policy(STRATEGY_RECURSIVE).overlap_tokens == 48
    assert default_policy(STRATEGY_FIXED).overlap_tokens == 48
    assert default_policy(STRATEGY_SEMANTIC).min_tokens == 60
    assert set(DEFAULT_POLICIES) == set(STRATEGIES)


def test_bare_policy_is_the_conservative_one() -> None:
    """直接构造 ChunkPolicy 得到最保守的一套（不重叠），策略默认值走没门.

    这条区分很重要：字段默认值 = "最不会出错的配置"，
    ``DEFAULT_POLICIES`` = "这种策略的合理起点"。
    """
    policy = ChunkPolicy()
    assert policy.strategy == STRATEGY_RECURSIVE
    assert policy.overlap_tokens == 0
    assert policy.max_tokens == 320
    assert policy.step == 320


def test_default_policy_accepts_overrides() -> None:
    """``default_policy`` 的覆盖项生效，未覆盖的保留策略默认值."""
    policy = default_policy(STRATEGY_SEMANTIC, max_tokens=128)
    assert policy.max_tokens == 128
    assert policy.min_tokens == 60
    assert policy.overlap_tokens == 40


def test_default_policy_rejects_unknown_strategy() -> None:
    """未知策略要列出可选值（与 ``UnsupportedStrategy`` 的分工一致）."""
    with pytest.raises(UnsupportedStrategy):
        default_policy("semantic_v2")


def test_policy_rejects_unknown_strategy() -> None:
    with pytest.raises(UnsupportedStrategy):
        ChunkPolicy(strategy="rolling")


def test_policy_rejects_overlap_not_smaller_than_budget() -> None:
    """``overlap >= max_tokens`` 会让步长为 0：窗口原地踏步，切分不终止."""
    with pytest.raises(ChunkingError) as excinfo:
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=64, overlap_tokens=64)
    assert "原地踏步" in str(excinfo.value)


def test_policy_rejects_min_tokens_above_budget() -> None:
    """``min_tokens > max_tokens`` 会让合并后的块必然超预算."""
    with pytest.raises(ChunkingError):
        ChunkPolicy(max_tokens=32, min_tokens=64)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tokens": 0},
        {"max_tokens": -1},
        {"overlap_tokens": -1},
        {"min_tokens": -1},
        {"measurer": "wordpiece"},
    ],
)
def test_policy_rejects_invalid_numbers_and_measurers(kwargs: dict) -> None:
    with pytest.raises(ChunkingError):
        ChunkPolicy(**kwargs)


def test_policy_rejects_percentile_outside_unit_range() -> None:
    """分位数是比例：传 25 而不是 0.25 是最常见的笔误，必须报错."""
    with pytest.raises(ChunkingError) as excinfo:
        ChunkPolicy(similarity_percentile=25)
    assert "0.25" in str(excinfo.value)


def test_policy_step_and_to_dict() -> None:
    """``step`` 是派生量，``to_dict`` 把它一起带出去（报告要按它核对）."""
    policy = ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=320, overlap_tokens=48)
    assert policy.step == 272
    payload = policy.to_dict()
    assert payload["step"] == 272
    assert payload["strategy"] == STRATEGY_FIXED
    assert set(payload) == {
        "strategy",
        "max_tokens",
        "overlap_tokens",
        "min_tokens",
        "measurer",
        "model",
        "similarity_percentile",
        "step",
    }


def test_policy_replace_requires_known_fields() -> None:
    """派生参数时拼错字段名要报错，**不能静默忽略**."""
    policy = default_policy(STRATEGY_RECURSIVE)
    assert policy.replace(max_tokens=100).max_tokens == 100
    with pytest.raises(ChunkingError) as excinfo:
        policy.replace(max_token=100)
    assert "max_token" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 区间与收尾
# --------------------------------------------------------------------------- #


def test_span_only_carries_a_range() -> None:
    """``Span`` 刻意不带文本：文本只能由 ``document.text[start:end]`` 现取."""
    span = Span(start_char=1, end_char=4, reason="fits")
    assert span.start_block == -1 and span.end_block == -1
    assert not span.oversized


def test_finalize_returns_empty_set_for_blank_document() -> None:
    """空文档返回**空集合**而不是抛错：它是一个结论，不是一个错误."""
    chunker = build_chunker(STRATEGY_FIXED, measurer=chars())
    result = chunker.split(text_document("   \n\n  ", source="docs/blank.txt"))
    assert result.count == 0
    assert result.doc_chars == 0
    assert result.metadata["empty"] == "true"
    assert result.coverage == 0.0
    assert result.duplication_ratio == 0.0


def test_finalize_rejects_spans_with_no_progress() -> None:
    """起点倒退会被拦下：那意味着窗口没有前进，切分在原地打转.

    刻意用一个"后一块更短"的样本：只有这种情况不该被 ``_dedupe_starts``
    自愈（自愈仅限"后一块完全包含前一块"）。
    """
    document = text_document("abcdefghij", source="docs/x.txt")
    chunker = ProbeChunker(
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=100),
        [Span(3, 10), Span(1, 5)],
        measurer=chars(),
    )
    with pytest.raises(ChunkingError) as excinfo:
        chunker.split(document)
    assert "严格递增" in str(excinfo.value)


def test_finalize_rejects_uncovered_content() -> None:
    """丢了内容要**直接报错**，而不是给一个 coverage=0.94 的报告."""
    document = text_document("abcdefghij", source="docs/x.txt")
    chunker = ProbeChunker(
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=100),
        [Span(0, 5)],
        measurer=chars(),
    )
    with pytest.raises(ChunkingError) as excinfo:
        chunker.split(document)
    assert "没有被任何块覆盖" in str(excinfo.value)


def test_finalize_rejects_out_of_range_spans() -> None:
    """越界区间被拦下——这道检查直接测 ``_assert_spans``.

    为什么不通过 ``split`` 触发：``_trim`` 在收尾的第一步就把区间
    **夹到合法范围内**（``min(end, len(text))``），因此越界到不了
    ``_assert_spans``。它是接口的最后一道防线，所以直接调用它来验证——
    **一道永远走不到的护栏仍然值得存在，但要有测试证明它真的会拦。**
    """
    document = text_document("abcdefghij", source="docs/x.txt")
    chunker = ProbeChunker(
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=100), [], measurer=chars()
    )
    with pytest.raises(ChunkingError) as excinfo:
        chunker._assert_spans(document.text, [Span(0, 99)])
    assert "越界" in str(excinfo.value)
    with pytest.raises(ChunkingError):
        chunker._assert_spans(document.text, [Span(-1, 3)])


def test_finalize_clamps_out_of_range_spans() -> None:
    """``_trim`` 把越界区间夹回合法范围（因此上层看不到越界错误）."""
    document = text_document("abcdefghij", source="docs/x.txt")
    result = ProbeChunker(
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=100),
        [Span(0, 999)],
        measurer=chars(),
    ).split(document)
    assert [chunk.text for chunk in result.chunks] == ["abcdefghij"]


def test_finalize_rejects_empty_span_list_on_non_empty_document() -> None:
    """原文非空却切不出块：说明切分逻辑把内容全丢了."""
    document = text_document("abcdefghij", source="docs/x.txt")
    chunker = ProbeChunker(
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=100), [], measurer=chars()
    )
    with pytest.raises(ChunkingError) as excinfo:
        chunker.split(document)
    assert "没有产出任何块" in str(excinfo.value)


def test_finalize_trims_whitespace_and_drops_blank_spans() -> None:
    """修剪去掉区间首尾空白（否则块文本会以空行开头），全空白的块被丢掉.

    样本里的 ``[2, 4)`` 恰好是那段空行：它会被 ``_trim`` 判为空并丢弃，
    而两个真正的段落各自成块——**丢弃是安全的**，因为覆盖率只要求
    非空白字符被覆盖（见 ``_assert_spans``）。
    """
    document = text_document("aa\n\nbb", source="docs/x.txt")
    chunker = ProbeChunker(
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=100),
        [Span(0, 2), Span(2, 4), Span(4, 6)],
        measurer=chars(),
    )
    result = chunker.split(document)
    assert [chunk.text for chunk in result.chunks] == ["aa", "bb"]
    assert [chunk.start_char for chunk in result.chunks] == [0, 4]


def test_merge_joins_short_chunks_without_exceeding_budget() -> None:
    """``min_tokens`` 让过短的块与后一块合并；合并**不越过预算**."""
    document = text_document("a\n\nb\n\nc", source="docs/x.txt")
    spans = [Span(0, 1), Span(3, 4), Span(6, 7)]
    merged = ProbeChunker(
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=5, min_tokens=2),
        spans,
        measurer=chars(),
    ).split(document)
    assert [chunk.text for chunk in merged.chunks] == ["a\n\nb", "c"]
    assert all(chunk.token_count <= 5 for chunk in merged.chunks)


def test_merge_skips_oversized_neighbours() -> None:
    """原子块（oversized）不参与合并：把它并进来只会得到更大的块."""
    document = text_document("a\n\n" + "b" * 20, source="docs/x.txt")
    spans = [Span(0, 1), Span(3, 23, oversized=True)]
    merged = ProbeChunker(
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=10, min_tokens=5),
        spans,
        measurer=chars(),
    ).split(document)
    assert [chunk.oversized for chunk in merged.chunks] == [False, True]


def test_dedupe_starts_drops_the_contained_span() -> None:
    """起点撞在一起时丢掉被包含的那一块（``_dedupe_starts`` 的自愈路径）."""
    document = text_document("aabbccdd", source="docs/x.txt")
    chunker = ProbeChunker(
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=100),
        [Span(0, 2), Span(2, 4), Span(2, 8)],
        measurer=chars(),
    )
    result = chunker.split(document)
    assert [chunk.text for chunk in result.chunks] == ["aa", "bbccdd"]


def test_hard_split_reports_impossible_budget() -> None:
    """预算连一个字符都装不下时要报错，并且错误信息给出下一步动作."""
    document = text_document("abcdef", source="docs/x.txt")
    chunker = ProbeChunker(
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=1),
        [Span(0, 6)],
        measurer=BigMeasurer(),
    )
    with pytest.raises(ChunkingError) as excinfo:
        chunker._hard_split(document.text, 0, 6, "test")
    assert "装不下" in str(excinfo.value)
    assert "max_tokens" in str(excinfo.value)


def test_pack_merges_adjacent_spans_up_to_budget() -> None:
    """``_pack`` 把相邻小区间合并到预算上限（"尽量填满，但不超"）."""
    document = text_document("abcd" + "\n\n" + "efgh", source="docs/x.txt")
    chunker = ProbeChunker(
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=100), [], measurer=chars()
    )
    packed = chunker._pack(
        document.text, [Span(0, 4), Span(6, 10)], "+pack"
    )
    assert len(packed) == 1
    assert packed[0].reason == "+pack"


# --------------------------------------------------------------------------- #
# 共用原语
# --------------------------------------------------------------------------- #


def test_paragraph_spans_finds_offsets_not_just_text() -> None:
    """按空行切出**偏移区间**（不是文本片段）——偏移是本层一切产物的身份."""
    text = "aa\n\nbb\n\n\ncc"
    assert paragraph_spans(text) == [(0, 2), (4, 6), (9, 11)]
    assert paragraph_spans(text, 4, 6) == [(4, 6)]


def test_paragraph_spans_skips_blank_only_segments() -> None:
    """只有空白的段落被丢掉：它覆盖的是空白，而覆盖率只要求非空白字符."""
    assert paragraph_spans("aa\n\n  \n\nbb") == [(0, 2), (8, 10)]


def test_snap_to_boundary_prefers_a_natural_cut() -> None:
    """重叠回带的落点要对齐到断点字符（中英文标点都在表里）.

    ``"第一句。第二句。"`` 的 ``。`` 落在下标 3 之后，因此
    ``[1, 5)`` 里第一个"紧跟断点"的位置是 4。搜索范围是**左闭右开**的：
    落点必须严格小于块的起点（``_with_overlap`` 传的 ``high`` 就是起点），
    否则"回带"会把起点推到起点本身。
    """
    text = "第一句。第二句。"
    assert snap_to_boundary(text, 1, 5) == 4
    assert snap_to_boundary(text, 1, 4) == 1
    assert "。" in BOUNDARY_CHARS
    assert "\n" in BOUNDARY_CHARS


def test_snap_to_boundary_falls_back_to_low() -> None:
    """窗口内找不到断点时退回 ``low``：宁可落点不完美，也不把重叠翻倍."""
    assert snap_to_boundary("abcdefgh", 3, 5) == 3
    assert snap_to_boundary("第一句。第二句。", 5, 8) == 5
    assert snap_to_boundary("abcdefgh", 9, 9) == 8


def test_heading_positions_and_path_lookup() -> None:
    """标题栈与偏移：``level >= 当前`` 就弹栈（与结构策略同一套规则）."""
    document = guide_document()
    positions = heading_positions(document)
    assert [path for _, path in positions] == [
        ("检索手册",),
        ("检索手册", "分块"),
        ("检索手册", "检索"),
        ("检索手册", "成本"),
    ]
    assert positions[0][0] == 0
    assert path_for_offset(0, positions) == ("检索手册",)
    assert path_for_offset(positions[2][0] + 1, positions) == ("检索手册", "检索")


def test_path_for_offset_before_any_heading() -> None:
    """第一个标题之前的内容没有路径（空元组）."""
    assert path_for_offset(0, [(5, ("A",))]) == ()
    assert path_for_offset(5, [(5, ("A",))]) == ("A",)


# --------------------------------------------------------------------------- #
# 注册表与工厂
# --------------------------------------------------------------------------- #


def test_default_registry_registers_four_strategies_in_order() -> None:
    """四种策略齐全，且**按 STRATEGIES 的顺序**返回（不按注册顺序）."""
    registry = default_registry(measurer=chars())
    assert registry.strategies == STRATEGIES
    assert [row["name"] for row in registry.table()] == list(STRATEGIES)


def test_registry_rejects_duplicate_registration() -> None:
    """同名重复注册直接报错：否则 split 的结果取决于导入顺序."""
    registry = ChunkerRegistry()
    registry.register(build_chunker(STRATEGY_FIXED, measurer=chars()))
    with pytest.raises(ChunkingError) as excinfo:
        registry.register(build_chunker(STRATEGY_FIXED, measurer=chars()))
    assert "已经注册" in str(excinfo.value)


def test_registry_lists_candidates_for_unknown_strategy() -> None:
    registry = ChunkerRegistry()
    with pytest.raises(UnsupportedStrategy) as excinfo:
        registry.get("structural_v2")
    assert "当前可用" in str(excinfo.value)


def test_build_chunker_rejects_unknown_strategy() -> None:
    with pytest.raises(UnsupportedStrategy):
        build_chunker("nope", measurer=chars())


def test_build_chunker_rejects_policy_strategy_mismatch() -> None:
    """把固定长度的参数塞给结构分块器要报错，而不是"静默按参数跑"."""
    with pytest.raises(ChunkingError) as excinfo:
        build_chunker(STRATEGY_STRUCTURAL, default_policy(STRATEGY_FIXED))
    assert "只处理" in str(excinfo.value)


def test_chunking_boundaries_describes_the_whole_capability() -> None:
    """自述表与文档同源：策略、度量器、默认参数、边界、不变量都在一张表里."""
    boundaries = chunking_boundaries()
    assert boundaries["strategies"] == list(STRATEGIES)
    assert boundaries["measurers"]
    assert set(boundaries["default_policies"]) == set(STRATEGIES)
    assert boundaries["separators"][0] == "\n\n"
    assert "chunk.text" in boundaries["coverage_invariant"]
    assert list(boundaries["limitations"]) == list(CHUNKING_LIMITATIONS)
    assert list(boundaries["out_of_scope"]) == list(CHUNKING_OUT_OF_SCOPE)
    assert "day064" in boundaries["next_step"]
