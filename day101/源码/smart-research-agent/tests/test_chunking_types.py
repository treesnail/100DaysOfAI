"""day062 ``chunking.types`` 的单元测试：Chunk 与 ChunkSet 的身份与度量.

全部离线、确定性。这一层的测试重点是**字段之间的分工**：
``chunk_id``（位置身份）与 ``fingerprint``（内容身份）、
``text``（原文）与 ``retrieval_text``（检索视图）、
``coverage``（有没有丢内容）与 ``duplication_ratio``（重叠花了多少）。
"""

from __future__ import annotations

from typing import Any

import pytest

from smart_research_agent.chunking import (
    CHUNK_ID_LENGTH,
    STRATEGIES,
    STRATEGY_FIXED,
    STRATEGY_RECURSIVE,
    STRATEGY_STRUCTURAL,
    Chunk,
    ChunkingError,
    ChunkSet,
    chunk_id_for,
)
from smart_research_agent.documents.types import content_id


def make_chunk(**overrides: Any) -> Chunk:
    """造一个合法的块（``chunk_id`` 默认按公式算，覆盖时才用给定值）."""
    payload: dict[str, Any] = {
        "doc_id": "a" * CHUNK_ID_LENGTH,
        "source": "docs/x.md",
        "text": "一段正文",
        "index": 0,
        "strategy": STRATEGY_RECURSIVE,
        "token_count": 4,
        "token_measurer": "chars",
        "start_char": 0,
        "end_char": 4,
    }
    payload.update(overrides)
    if "chunk_id" not in overrides:
        payload["chunk_id"] = chunk_id_for(
            payload["doc_id"], payload["strategy"], payload["index"], payload["text"]
        )
    return Chunk(**payload)


def make_set(**overrides: Any) -> ChunkSet:
    """造一个合法的块集（默认一块）."""
    chunks = overrides.pop("chunks", (make_chunk(),))
    payload: dict[str, Any] = {
        "doc_id": "a" * CHUNK_ID_LENGTH,
        "source": "docs/x.md",
        "strategy": STRATEGY_RECURSIVE,
        "chunks": chunks,
        "token_measurer": "chars",
        "doc_chars": 4,
    }
    payload.update(overrides)
    return ChunkSet(**payload)


# --------------------------------------------------------------------------- #
# 身份：chunk_id 与 fingerprint 是两个问题
# --------------------------------------------------------------------------- #


def test_chunk_id_is_deterministic() -> None:
    """同样的四元组永远得到同一个 id（知识库的去重键依赖它）."""
    first = chunk_id_for("doc", STRATEGY_FIXED, 0, "文本")
    second = chunk_id_for("doc", STRATEGY_FIXED, 0, "文本")
    assert first == second
    assert len(first) == CHUNK_ID_LENGTH
    assert all(character in "0123456789abcdef" for character in first)


@pytest.mark.parametrize(
    "changes",
    [
        {"doc_id": "other"},
        {"strategy": STRATEGY_STRUCTURAL},
        {"index": 1},
        {"text": "另一段正文"},
    ],
)
def test_chunk_id_changes_with_position(changes: dict) -> None:
    """文档 / 策略 / 序号 / 文本任一变化，id 就变（缺一不可）."""
    base = {"doc_id": "doc", "strategy": STRATEGY_FIXED, "index": 0, "text": "文本"}
    assert chunk_id_for(**{**base, **changes}) != chunk_id_for(**base)


def test_index_is_part_of_the_identity() -> None:
    """**同一段话在文档里出现两次时，两块各有各的 id**（不会被互相覆盖）.

    这是把 ``index`` 放进 id 的全部理由：只按文本算 id 时，
    工程文档里反复出现的"注意"小节会得到同一个 id，向量库按 id 覆盖，
    "命中了 3 次"在库里只剩 1 条。
    """
    first = chunk_id_for("doc", STRATEGY_FIXED, 0, "注意：阈值设为 0.85")
    second = chunk_id_for("doc", STRATEGY_FIXED, 3, "注意：阈值设为 0.85")
    assert first != second


def test_fingerprint_is_content_only() -> None:
    """而跨文档去重要的是相反的东西：同样的内容必须得到同样的指纹."""
    first = make_chunk(text="重复的一段话", index=0)
    second = make_chunk(text="重复的一段话", index=5, doc_id="b" * CHUNK_ID_LENGTH)
    assert first.chunk_id != second.chunk_id
    assert first.fingerprint == second.fingerprint == content_id("重复的一段话")


# --------------------------------------------------------------------------- #
# Chunk 的校验
# --------------------------------------------------------------------------- #


def test_chunk_rejects_unknown_strategy() -> None:
    with pytest.raises(ChunkingError) as excinfo:
        make_chunk(strategy="rolling")
    assert "可选" in str(excinfo.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"text": "   "},
        {"text": "\n\n"},
        {"index": -1},
        {"token_count": -1},
        {"start_char": 5, "end_char": 2},
        {"chunk_id": "short"},
    ],
)
def test_chunk_rejects_invalid_fields(overrides: dict) -> None:
    """六个字段各有各的拒绝理由（空文本 / 负序号 / 负预算 / 区间倒挂 / id 长度）."""
    with pytest.raises(ChunkingError):
        make_chunk(**overrides)


def test_chunk_rejects_block_range_upside_down() -> None:
    """块区间是结构策略的产物，倒挂意味着定位逻辑出错."""
    with pytest.raises(ChunkingError):
        make_chunk(start_block=3, end_block=1)


def test_chunk_allows_missing_block_range() -> None:
    """``-1`` 是"本策略不追踪块区间"的哨兵值（0 是合法下标，不能混用）."""
    chunk = make_chunk()
    assert chunk.start_block == -1 and chunk.end_block == -1


# --------------------------------------------------------------------------- #
# Chunk 的派生量与两个视图
# --------------------------------------------------------------------------- #


def test_headings_become_a_breadcrumb_in_the_retrieval_view() -> None:
    """标题路径进 ``retrieval_text`` 而**不进** ``text``（引用必须能对回原文）."""
    chunk = make_chunk(text="阈值设为 0.85", heading_path=("检索手册", "缓存"))
    assert chunk.heading_text == "检索手册 > 缓存"
    assert chunk.text == "阈值设为 0.85"
    assert chunk.retrieval_text == "检索手册 > 缓存\n阈值设为 0.85"
    assert chunk.char_count == len(chunk.text)


def test_retrieval_text_without_headings_is_the_raw_text() -> None:
    """没有标题时检索视图等于原文，不额外加任何字符."""
    chunk = make_chunk(text="一段正文")
    assert chunk.heading_path == ()
    assert chunk.heading_text == ""
    assert chunk.retrieval_text == chunk.text


def test_chunk_to_dict_hides_text_on_demand() -> None:
    """``include_text=False`` 只给度量与身份（一份 500 块的响应不该带全文）."""
    chunk = make_chunk(text="一段正文", heading_path=("A",), reason="sep:段落")
    full = chunk.to_dict()
    assert full["text"] == "一段正文"
    assert full["retrieval_text"] == "A\n一段正文"
    assert full["heading_path"] == ["A"]
    assert full["fingerprint"] == content_id("一段正文")
    lean = chunk.to_dict(include_text=False)
    assert "text" not in lean and "retrieval_text" not in lean
    assert lean["char_count"] == 4


def test_chunk_summary_line_marks_oversized() -> None:
    """超预算的块在报告里带一个 ``!`` 前缀（"为什么这块这么大"要一眼可见）."""
    normal = make_chunk(text="一段正文")
    oversized = make_chunk(text="一段很长的正文", oversized=True, index=1)
    assert normal.summary_line().startswith(" ")
    assert oversized.summary_line().startswith("!")
    assert "（无标题）" in normal.summary_line()


# --------------------------------------------------------------------------- #
# ChunkSet 的校验
# --------------------------------------------------------------------------- #


def test_chunk_set_rejects_unknown_strategy() -> None:
    with pytest.raises(ChunkingError):
        make_set(strategy="rolling")


def test_chunk_set_rejects_negative_doc_chars() -> None:
    with pytest.raises(ChunkingError):
        make_set(doc_chars=-1)


def test_chunk_set_requires_continuous_indices() -> None:
    """序号必须是 0..n-1：跳号意味着"有块丢了"，而那在下游只是少了几个片段."""
    chunks = (make_chunk(index=0), make_chunk(index=2, text="后一块"))
    with pytest.raises(ChunkingError) as excinfo:
        make_set(chunks=chunks)
    assert "连续递增" in str(excinfo.value)


def test_chunk_set_rejects_foreign_chunk() -> None:
    """块与集合的 doc_id 必须一致（否则是两批块被拼到了一起）."""
    with pytest.raises(ChunkingError) as excinfo:
        make_set(chunks=(make_chunk(doc_id="b" * CHUNK_ID_LENGTH),))
    assert "doc_id" in str(excinfo.value)


def test_chunk_set_rejects_mixed_strategies() -> None:
    with pytest.raises(ChunkingError) as excinfo:
        make_set(chunks=(make_chunk(strategy=STRATEGY_FIXED),))
    assert "策略" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# ChunkSet 的度量
# --------------------------------------------------------------------------- #


def test_chunk_set_metrics_on_a_two_chunk_set() -> None:
    """覆盖 / 重复 / 超预算三个数一起看：它们回答三个不同的问题."""
    chunks = (
        make_chunk(text="abcdefgh", index=0, start_char=0, end_char=8, token_count=8),
        make_chunk(text="cdefghijkl", index=1, start_char=6, end_char=16, token_count=10),
    )
    chunk_set = make_set(chunks=chunks, doc_chars=12)
    assert chunk_set.count == 2
    assert chunk_set.total_chars == 18
    assert chunk_set.total_tokens == 18
    assert chunk_set.coverage == 1.5
    assert chunk_set.duplication_ratio == 0.3333
    assert chunk_set.oversized_count == 0
    assert chunk_set.token_stats() == {
        "min": 8.0,
        "p50": 9.0,
        "p90": 9.8,
        "max": 10.0,
        "mean": 9.0,
    }


def test_chunk_set_metrics_on_empty_set() -> None:
    """空集合的每个度量都有定义（报告的表头不该因为空文档而缺列）."""
    empty = make_set(chunks=(), doc_chars=0)
    assert empty.count == 0
    assert empty.coverage == 0.0
    assert empty.duplication_ratio == 0.0
    assert empty.token_stats() == {"min": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    assert empty.summary_line().endswith("chars")


def test_chunk_set_duplication_ratio_is_clipped_at_zero() -> None:
    """块首尾空白被修剪会让差额略小于 0——那与重叠无关，不能显示成负数."""
    chunk_set = make_set(chunks=(make_chunk(text="abc", token_count=3),), doc_chars=10)
    assert chunk_set.duplication_ratio == 0.0


def test_heading_paths_and_oversized_listing() -> None:
    chunks = (
        make_chunk(text="一", index=0, heading_path=("A", "B")),
        make_chunk(text="二", index=1, heading_path=("A",)),
        make_chunk(text="三", index=2, heading_path=("A", "B"), oversized=True),
    )
    chunk_set = make_set(chunks=chunks, doc_chars=3)
    assert chunk_set.heading_paths() == [("A",), ("A", "B")]
    assert [chunk.index for chunk in chunk_set.oversized()] == [2]
    assert chunk_set.oversized_count == 1


# --------------------------------------------------------------------------- #
# 知识库记录：与 day061 的形状对齐
# --------------------------------------------------------------------------- #


def test_knowledge_records_keep_the_day061_shape() -> None:
    """**字段名不变**，只是 ``doc_id`` 位置换了身份：整份文档 → 片段."""
    chunk = make_chunk(
        text="阈值设为 0.85", heading_path=("检索手册", "缓存"), index=0
    )
    record = make_set(chunks=(chunk,), doc_chars=6).knowledge_records()[0]
    assert set(record) == {"doc_id", "source", "text", "metadata"}
    assert record["doc_id"] == chunk.chunk_id
    assert record["text"] == chunk.text
    assert record["metadata"]["parent_doc_id"] == chunk.doc_id
    assert record["metadata"]["heading_path"] == "检索手册 > 缓存"
    assert record["metadata"]["retrieval_text"] == chunk.retrieval_text
    assert record["metadata"]["oversized"] == "false"
    assert record["metadata"]["token_measurer"] == "chars"


def test_chunk_set_table_is_report_ready() -> None:
    """逐块视图里带 ``start:end``：**"块是原文连续子串"在这列上肉眼可验**."""
    rows = make_set().table()
    assert rows[0]["range"] == "0:4"
    assert rows[0]["preview"] == "一段正文"
    assert rows[0]["oversized"] is False


def test_chunk_set_to_dict_carries_identity_and_stats() -> None:
    payload = make_set().to_dict()
    assert payload["strategy"] == STRATEGY_RECURSIVE
    assert payload["count"] == 1
    assert payload["token_measurer"] == "chars"
    assert len(payload["chunks"][0]["chunk_id"]) == CHUNK_ID_LENGTH
    assert "text" not in payload["chunks"][0]
    assert payload["chunks"][0]["char_count"] == 4


def test_strategies_constant_is_the_single_source_of_truth() -> None:
    """四种策略的名字只定义一次（端点的列序、注册表的顺序都按它）."""
    assert STRATEGIES == ("fixed", "recursive", "structural", "semantic")
