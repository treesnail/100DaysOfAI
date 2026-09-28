"""day062 递归分块（``chunking.recursive``）的单元测试.

三条主线：分隔符表的**内容**（中文标点必须在）、递归下切的**收敛性**
（用完就硬切）、以及"先切碎再合并"这个容易漏掉的一步。
"""

from __future__ import annotations

import pytest

from smart_research_agent.chunking import (
    DEFAULT_SEPARATORS,
    SEPARATOR_LABELS,
    STRATEGY_RECURSIVE,
    ChunkingError,
    ChunkPolicy,
    build_chunker,
    default_policy,
    separator_histogram,
    split_keep_separator,
)
from tests.chunking_samples import chars, guide_document, text_document


def recursive(policy: ChunkPolicy):  # type: ignore[no-untyped-def]
    return build_chunker(STRATEGY_RECURSIVE, policy, measurer=chars())


# --------------------------------------------------------------------------- #
# 分隔符表本身
# --------------------------------------------------------------------------- #


def test_default_separators_contain_cjk_punctuation() -> None:
    """**中文标点必须在表里**：否则中文文档一路退到硬切，而且不会报错."""
    for separator in ("\n\n", "\n", "。", "！", "？", "；", "，"):
        assert separator in DEFAULT_SEPARATORS
    assert DEFAULT_SEPARATORS[0] == "\n\n"
    assert DEFAULT_SEPARATORS[-1] == ""
    assert len(set(DEFAULT_SEPARATORS)) == len(DEFAULT_SEPARATORS)


def test_separator_labels_cover_every_separator() -> None:
    """每个分隔符都要有可读名字：报告里的 ``reason`` 直接用它."""
    assert set(SEPARATOR_LABELS) == set(DEFAULT_SEPARATORS)
    assert SEPARATOR_LABELS["\n\n"] == "段落"
    assert "硬切" in SEPARATOR_LABELS[""]


def test_split_keep_separator_keeps_the_text_reconstructible() -> None:
    """分隔符留在**前一片的尾部**：拼回去必须等于原文（否则块文本对不回原文）."""
    assert split_keep_separator("第一句。第二句。", "。") == ["第一句。", "第二句。"]
    assert "".join(split_keep_separator("第一句。第二句。", "。")) == "第一句。第二句。"
    assert split_keep_separator("没有分隔符", "。") == ["没有分隔符"]
    assert split_keep_separator("abc", "") == ["abc"]


def test_separator_histogram_explains_degradation() -> None:
    """直方图回答"这份文档为什么退化了"：``count=0`` 的那一档就是原因."""
    rows = separator_histogram("第一段。\n\n第二段，还有逗号。")
    counts = {row["label"]: row["count"] for row in rows}
    assert counts["段落"] == 1
    assert counts["中文句号"] == 2
    assert counts["中文逗号"] == 1
    assert counts["空格"] == 0
    assert [row["separator"] for row in rows] == list(DEFAULT_SEPARATORS)


# --------------------------------------------------------------------------- #
# 切分
# --------------------------------------------------------------------------- #


def test_recursive_chunks_respect_the_budget_and_the_contract() -> None:
    """每一块不超预算；块文本是原文的连续子串."""
    document = guide_document()
    result = recursive(
        ChunkPolicy(strategy=STRATEGY_RECURSIVE, max_tokens=60, overlap_tokens=12)
    ).split(document)
    assert result.count > 1
    assert all(chunk.token_count <= 60 for chunk in result.chunks)
    for chunk in result.chunks:
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text


def test_recursive_merges_neighbours_to_fill_the_budget() -> None:
    """**先切碎再合并**：不合并的话一份 200 段的文档会切出 200 个碎块.

    样本刻意做成"前两个自然段合起来刚好装进预算、第三个装不进"：
    ``"aa\\n\\nbb\\n\\ncc"`` 在预算 8 下第一块是 ``aa\\n\\nbb``（合并成功），
    第二块是 ``cc``——**合并不是无条件的，它的边界就是预算本身。**
    """
    document = text_document("aa\n\nbb\n\ncc", source="docs/p.txt")
    result = recursive(
        ChunkPolicy(strategy=STRATEGY_RECURSIVE, max_tokens=8, overlap_tokens=0)
    ).split(document)
    assert [chunk.text for chunk in result.chunks] == ["aa\n\nbb", "cc"]
    assert "+pack" in result.chunks[0].reason


def test_recursive_reason_records_where_it_cut() -> None:
    """``reason`` 是可解释性的落点：报告里能直接看到"这一刀切在段落上"."""
    document = text_document("甲" * 8 + "\n\n" + "乙" * 8, source="docs/p.txt")
    result = recursive(
        ChunkPolicy(strategy=STRATEGY_RECURSIVE, max_tokens=10, overlap_tokens=0)
    ).split(document)
    assert [chunk.text for chunk in result.chunks] == ["甲" * 8, "乙" * 8]
    assert all("sep:段落" in chunk.reason for chunk in result.chunks)


def test_recursive_falls_back_to_hard_split_without_separators() -> None:
    """没有任何分隔符的长串走硬切（最后一档），并且每刀都前进."""
    document = text_document("x" * 200, source="docs/log.txt")
    result = recursive(
        ChunkPolicy(strategy=STRATEGY_RECURSIVE, max_tokens=64, overlap_tokens=0)
    ).split(document)
    assert result.count == 4
    assert [chunk.char_count for chunk in result.chunks] == [64, 64, 64, 8]
    assert all("char" in chunk.reason for chunk in result.chunks)


def test_recursive_degrades_one_level_at_a_time() -> None:
    """这一档没有该分隔符时**降到下一档**，而不是就地硬切.

    ``"a\nb\nc"`` 里没有 ``\\n\\n``，但有 ``\\n``——因此切点应当落在换行上
    （``sep:换行``），而不是落在字符上。
    """
    document = text_document("aaaa\nbbbb\ncccc", source="docs/lines.txt")
    result = recursive(
        ChunkPolicy(strategy=STRATEGY_RECURSIVE, max_tokens=5, overlap_tokens=0)
    ).split(document)
    assert all("换行" in chunk.reason for chunk in result.chunks)
    assert [chunk.text for chunk in result.chunks] == ["aaaa", "bbbb", "cccc"]


def test_recursive_split_splits_chinese_sentences() -> None:
    """中文句号可切：一份没有空行、只有句号的中文文档不会退化成硬切."""
    text = "第一句足够长。" * 4
    document = text_document(text, source="docs/cn.txt")
    result = recursive(
        ChunkPolicy(strategy=STRATEGY_RECURSIVE, max_tokens=8, overlap_tokens=0)
    ).split(document)
    assert result.count == 4
    assert all("中文句号" in chunk.reason for chunk in result.chunks)


def test_recursive_overlap_pulls_starts_back_and_still_matches_the_source() -> None:
    """重叠靠"把下一块的起点往回挪"实现，因此块仍是区间（可核对、可计费）."""
    document = guide_document()
    result = recursive(
        ChunkPolicy(strategy=STRATEGY_RECURSIVE, max_tokens=120, overlap_tokens=40)
    ).split(document)
    assert result.duplication_ratio > 0
    assert "+overlap" in result.chunks[1].reason
    for chunk in result.chunks:
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text
    assert result.chunks[1].start_char < result.chunks[0].end_char


def test_recursive_overlap_never_duplicates_a_whole_chunk() -> None:
    """护栏之一：回带之后"仍然前进"，不会出现两块起点相同."""
    document = text_document("第一段。\n\n第二段。\n\n第三段。", source="docs/s.txt")
    result = recursive(
        ChunkPolicy(strategy=STRATEGY_RECURSIVE, max_tokens=20, overlap_tokens=8)
    ).split(document)
    starts = [chunk.start_char for chunk in result.chunks]
    assert starts == sorted(set(starts))


def test_recursive_rejects_overlap_half_of_the_budget() -> None:
    """``2 × overlap >= max_tokens`` 时每一块都会变成上一块的复制品.

    这是**参数看起来完全合法**的一类错误：``max_tokens=60, overlap_tokens=48``
    两个数各自都没问题，合起来却让核心窗口只剩 12 个单位。
    """
    with pytest.raises(ChunkingError) as excinfo:
        build_chunker(
            STRATEGY_RECURSIVE, default_policy(STRATEGY_RECURSIVE, max_tokens=60)
        )
    assert "2 × overlap" in str(excinfo.value)


def test_recursive_min_tokens_merges_short_chunks() -> None:
    """``min_tokens`` 把过短的块与后一块合并（避免"阈值设为 0.85"这种碎块）.

    预算 6 下 ``ab`` 与 ``cd`` 本来各成一块（合起来 6 恰好卡在边界上，
    而分段切分时它们先各自成段），``min_tokens=5`` 把它们并成一个。
    """
    document = text_document("ab\n\ncd", source="docs/m.txt")
    without_floor = recursive(
        ChunkPolicy(strategy=STRATEGY_RECURSIVE, max_tokens=6, overlap_tokens=0)
    ).split(document)
    assert [chunk.text for chunk in without_floor.chunks] == ["ab", "cd"]
    with_floor = recursive(
        ChunkPolicy(
            strategy=STRATEGY_RECURSIVE, max_tokens=6, overlap_tokens=0, min_tokens=5
        )
    ).split(document)
    assert [chunk.text for chunk in with_floor.chunks] == ["ab\n\ncd"]


def test_recursive_describe_lists_the_separator_table() -> None:
    """自述表里带完整的优先级表——**它才是这个策略真正的参数**."""
    described = recursive(default_policy(STRATEGY_RECURSIVE)).describe()
    assert described["supports_overlap"] is True
    assert [row["separator"] for row in described["separators"]] == list(
        DEFAULT_SEPARATORS
    )
    assert any("标题" in item for item in described["weaknesses"])
