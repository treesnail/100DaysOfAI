"""day062 语义分块（``chunking.semantic``）的单元测试.

语义分块是四种策略里唯一会调用外部能力的一种，因此测试的重点是
**可复现性**：注入确定性的 embedding（``MockEmbedding`` / 记录型包装），
然后把"分位数阈值""标题额外断开""退化为预算切分"三条行为钉住。
"""

from __future__ import annotations

import pytest

from smart_research_agent.chunking import (
    STRATEGY_SEMANTIC,
    ChunkingError,
    ChunkPolicy,
    SemanticChunker,
    build_chunker,
    default_policy,
    percentile,
)
from smart_research_agent.llm.embedding import MockEmbedding
from tests.chunking_samples import chars, guide_document, text_document


class RecordingEmbedding(MockEmbedding):
    """记录**每一次被编码的文本**的 MockEmbedding（用来核对"编码的是什么"）."""

    def __init__(self, dimension: int = 32) -> None:
        super().__init__(dimension=dimension)
        self.seen: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.seen.append(text)
        return super().embed(text)


def semantic(policy: ChunkPolicy, **kwargs: object) -> SemanticChunker:
    kwargs.setdefault("measurer", chars())
    kwargs.setdefault("embedding", MockEmbedding())
    return SemanticChunker(policy, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 分位数
# --------------------------------------------------------------------------- #


def test_percentile_uses_linear_interpolation() -> None:
    """``rank = fraction × (n - 1)`` + 线性插值（与块长统计同一个口径）."""
    assert percentile([], 0.5) == 0.0
    assert percentile([5.0], 0.9) == 5.0
    assert percentile([0.0, 10.0], 0.5) == 5.0
    assert percentile([0.0, 10.0], 0.25) == 2.5
    assert percentile([10.0, 0.0, 20.0], 1.0) == 20.0
    assert percentile([10.0, 0.0, 20.0], 0.0) == 0.0


# --------------------------------------------------------------------------- #
# 切分
# --------------------------------------------------------------------------- #


def test_semantic_chunks_respect_the_budget_and_the_contract() -> None:
    """每一块不超预算；块文本是原文的连续子串（与其它三种策略同一条契约）."""
    document = guide_document()
    result = semantic(
        ChunkPolicy(strategy=STRATEGY_SEMANTIC, max_tokens=200, overlap_tokens=0)
    ).split(document)
    assert result.count >= 1
    assert all(chunk.token_count <= 200 for chunk in result.chunks)
    for chunk in result.chunks:
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text


def test_semantic_records_how_it_decided() -> None:
    """产物里要能回答"它是怎么切的"：自然段数、断层数、阈值、embedding 是谁."""
    result = semantic(default_policy(STRATEGY_SEMANTIC)).split(guide_document())
    metadata = result.metadata
    assert metadata["embedding"] == "MockEmbedding"
    assert metadata["embedding_dimension"] == "64"
    assert int(metadata["units"]) >= 2
    assert float(metadata["threshold"]) != 0.0
    assert metadata["similarity_percentile"] == "0.25"


def test_semantic_encodes_exactly_the_paragraphs() -> None:
    """被编码的是**自然段**（逐段一次），不是整份文档也不是块文本."""
    document = text_document("甲甲甲。\n\n乙乙乙。\n\n丙丙丙。", source="docs/s.txt")
    recorder = RecordingEmbedding()
    semantic(
        ChunkPolicy(strategy=STRATEGY_SEMANTIC, max_tokens=100, overlap_tokens=0),
        embedding=recorder,
    ).split(document)
    assert recorder.seen == ["甲甲甲。", "乙乙乙。", "丙丙丙。"]


def test_semantic_without_similarity_breaks_produces_one_chunk() -> None:
    """所有自然段完全相同时相似度恒为 1，``< 阈值`` 永不成立 → 没有任何断层.

    此时整篇落在一个组里，按预算合并成一块。**这是"相对判据"的正常行为**：
    没有明显的相似度低谷，就不该凭空造一个边界出来。
    """
    document = text_document(
        "重复的一段话。\n\n重复的一段话。\n\n重复的一段话。", source="docs/same.txt"
    )
    result = semantic(
        ChunkPolicy(strategy=STRATEGY_SEMANTIC, max_tokens=100, overlap_tokens=0)
    ).split(document)
    assert result.metadata["breaks"] == "0"
    assert result.metadata["heading_breaks"] == "0"
    assert result.count == 1


def test_semantic_breaks_at_headings_even_without_a_similarity_gap() -> None:
    """标题处**额外**断开：``heading_path`` 是块级属性，一块装不下两条路径.

    这条规则让"标题丰富的文档"在语义策略下自然接近
    "结构策略 + 相似度细分"。
    """
    document = text_document(
        "# 甲\n\n第一段内容。\n\n# 乙\n\n第二段内容。",
        source="docs/heads.md",
        media_type="text/markdown",
    )
    result = semantic(
        ChunkPolicy(strategy=STRATEGY_SEMANTIC, max_tokens=100, overlap_tokens=0)
    ).split(document)
    assert result.metadata["heading_breaks"] == "1"
    assert int(result.metadata["breaks"]) >= 1
    assert [chunk.heading_path for chunk in result.chunks] == [("甲",), ("乙",)]


def test_semantic_single_paragraph_has_no_criterion_to_apply() -> None:
    """只有一个自然段时语义判据没有施展空间：``units=1 / breaks=0 / threshold=n/a``.

    这不是错误，是一个结论——报告里能直接看到"这次等于没切"。
    """
    document = text_document("整篇只有一段话，没有空行。", source="docs/one.txt")
    result = semantic(default_policy(STRATEGY_SEMANTIC)).split(document)
    assert result.count == 1
    assert result.metadata["units"] == "1"
    assert result.metadata["breaks"] == "0"
    assert result.metadata["threshold"] == "n/a"


def test_semantic_falls_back_to_budget_when_a_group_is_too_large() -> None:
    """装不下的组按预算下切（理由记成 ``topic-overflow``，与"话题边界"区分开）.

    样本刻意做成"一整段没有任何空行"：语义策略只认识自然段，
    因此它在这里必然退化为按预算切——**日志类语料就是这样退化的。**
    """
    document = text_document("这是一句很长的话。" * 10, source="docs/long.txt")
    result = semantic(
        ChunkPolicy(strategy=STRATEGY_SEMANTIC, max_tokens=40, overlap_tokens=0)
    ).split(document)
    assert result.count == 3
    assert all("topic-overflow" in chunk.reason for chunk in result.chunks)
    assert all(chunk.token_count <= 40 for chunk in result.chunks)


def test_semantic_never_produces_oversized_chunks() -> None:
    """语义策略**没有原子块保护**，因此它不会产出超预算的块.

    这与结构策略正好互补：那里"宁可不切"，这里"宁可切碎"。
    两种取舍都写下来，选型时才有依据。
    """
    result = semantic(
        ChunkPolicy(strategy=STRATEGY_SEMANTIC, max_tokens=120, overlap_tokens=20)
    ).split(guide_document())
    assert result.oversized_count == 0
    assert all(chunk.token_count <= 120 for chunk in result.chunks)


def test_semantic_overlap_is_applied_after_trimming() -> None:
    """重叠在收尾的"修剪之后"统一回带：块不以空行开头，起点严格递增."""
    document = guide_document()
    result = semantic(
        ChunkPolicy(strategy=STRATEGY_SEMANTIC, max_tokens=100, overlap_tokens=20)
    ).split(document)
    assert result.count >= 2
    assert result.duplication_ratio > 0
    for chunk in result.chunks:
        assert not chunk.text.startswith("\n")
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text
    starts = [chunk.start_char for chunk in result.chunks]
    assert starts == sorted(set(starts))


def test_semantic_rejects_overlap_half_of_the_budget() -> None:
    """与递归策略同一条护栏：``2 × overlap >= max_tokens`` 时块会互相复制."""
    with pytest.raises(ChunkingError) as excinfo:
        SemanticChunker(
            default_policy(STRATEGY_SEMANTIC, max_tokens=64, overlap_tokens=32),
            measurer=chars(),
            embedding=MockEmbedding(),
        )
    assert "2 × overlap" in str(excinfo.value)


def test_semantic_describe_names_the_embedding_provider() -> None:
    """自述表要报出当前用的是哪个提供方（确定性来源必须可见）."""
    described = semantic(default_policy(STRATEGY_SEMANTIC)).describe()
    assert described["uses_similarity_percentile"] is True
    assert described["embedding"] == "MockEmbedding"
    assert described["supports_overlap"] is True
    assert "embedded" in described["deterministic"] or "embedding" in described[
        "deterministic"
    ]
    assert any("代码" in item for item in described["weaknesses"])


def test_semantic_is_offline_by_default_with_a_deterministic_result() -> None:
    """不注入提供方时走 ``default_embedding()``（离线环境是 char-ngram）.

    同一份文档切两次必须得到同一批 ``chunk_id``——**这是"确定性来自哪里"
    这条纪律最直接的检验**。
    """
    document = text_document("甲。乙。\n\n丙。丁。\n\n戊。己。", source="docs/d.txt")
    first = build_chunker(STRATEGY_SEMANTIC, measurer=chars()).split(document)
    second = build_chunker(STRATEGY_SEMANTIC, measurer=chars()).split(document)
    assert [chunk.chunk_id for chunk in first.chunks] == [
        chunk.chunk_id for chunk in second.chunks
    ]
