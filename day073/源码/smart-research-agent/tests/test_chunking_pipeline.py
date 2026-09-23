"""day062 ``chunking.pipeline`` 与 ``chunking.evaluate`` 的单元测试.

两个模块都在"单个分块器"之上：``pipeline`` 负责批量与对照，
``evaluate`` 负责用探针集给策略打分。测试的重点是**报告里的三列**
（规模、重复、超预算）与**评估里的两条纪律**（探针唯一、指标方向分开）。
"""

from __future__ import annotations

import pytest

from smart_research_agent.chunking import (
    STRATEGIES,
    STRATEGY_FIXED,
    STRATEGY_RECURSIVE,
    STRATEGY_STRUCTURAL,
    ChunkingError,
    ChunkPipeline,
    ChunkPolicy,
    ChunkReport,
    RetrievalProbe,
    ambiguous_probes,
    build_index,
    chunking_plan,
    default_policy,
    evaluate_strategies,
    knowledge_record_shape,
    score_probes,
)
from smart_research_agent.llm.embedding import MockEmbedding
from smart_research_agent.memory.vector_store import InMemoryVectorStore
from tests.chunking_samples import (
    GUIDE_MARKDOWN,
    PLAIN_TEXT,
    chars,
    guide_document,
    text_document,
)


@pytest.fixture
def pipeline() -> ChunkPipeline:
    """离线分块编排器（字符度量器 + MockEmbedding，完全确定性）."""
    return ChunkPipeline(measurer=chars(), embedding=MockEmbedding())


# --------------------------------------------------------------------------- #
# ChunkPipeline：单份与批量
# --------------------------------------------------------------------------- #


def test_chunk_uses_the_strategy_defaults(pipeline: ChunkPipeline) -> None:
    """不给参数时用**策略默认值**（结构策略的重叠默认是 0）."""
    chunk_set = pipeline.chunk(guide_document(), STRATEGY_STRUCTURAL)
    assert chunk_set.metadata["policy"]["overlap_tokens"] == 0
    assert chunk_set.strategy == STRATEGY_STRUCTURAL


def test_chunk_with_explicit_policy_does_not_mutate_the_registry(
    pipeline: ChunkPipeline,
) -> None:
    """显式参数**现建一个分块器**：改注册表里那份会让下一次的默认值悄悄变掉."""
    pipeline.chunk(
        guide_document(),
        STRATEGY_RECURSIVE,
        ChunkPolicy(strategy=STRATEGY_RECURSIVE, max_tokens=64, overlap_tokens=8),
    )
    again = pipeline.chunk(guide_document(), STRATEGY_RECURSIVE)
    assert again.metadata["policy"]["max_tokens"] == 320


def test_chunk_rejects_unknown_strategy(pipeline: ChunkPipeline) -> None:
    with pytest.raises(ChunkingError):
        pipeline.chunk(guide_document(), "semantic_v2")


def test_chunk_all_covers_every_strategy_in_order(pipeline: ChunkPipeline) -> None:
    """同一份文档跑四个策略：报告里的策略顺序固定（便于逐列对比）."""
    report = pipeline.chunk_all(guide_document())
    assert report.strategies == STRATEGIES
    assert report.documents == 1
    assert report.count == sum(item.count for item in report.sets)


def test_chunk_all_can_be_limited_to_a_subset(pipeline: ChunkPipeline) -> None:
    report = pipeline.chunk_all(
        guide_document(), strategies=[STRATEGY_FIXED, STRATEGY_STRUCTURAL]
    )
    assert report.strategies == (STRATEGY_FIXED, STRATEGY_STRUCTURAL)
    assert report.to_dict()["strategies"] == [STRATEGY_FIXED, STRATEGY_STRUCTURAL]


def test_chunk_documents_keeps_one_set_per_document(pipeline: ChunkPipeline) -> None:
    report = pipeline.chunk_documents(
        [guide_document(), text_document(PLAIN_TEXT, source="docs/plain.txt")],
        STRATEGY_RECURSIVE,
    )
    assert report.documents == 2
    assert report.strategies == (STRATEGY_RECURSIVE,)
    assert len(report.by_strategy(STRATEGY_RECURSIVE)) == 2


def test_ingest_and_chunk_runs_the_day061_to_day062_pipeline(
    pipeline: ChunkPipeline,
) -> None:
    """day061 入库（去重 + 三类状态）→ day062 分块，两条报告分开返回."""
    ingest, chunked = pipeline.ingest_and_chunk(
        [
            ("a.md", GUIDE_MARKDOWN.encode("utf-8")),
            ("copy.md", GUIDE_MARKDOWN.encode("utf-8")),
            ("plain.txt", PLAIN_TEXT.encode("utf-8")),
        ],
        STRATEGY_RECURSIVE,
    )
    assert ingest.status_counts() == {"ok": 2, "duplicate": 1, "error": 0}
    assert chunked.documents == ingest.unique_documents
    assert chunked.count >= 2


# --------------------------------------------------------------------------- #
# ChunkReport：三列与三个派生视图
# --------------------------------------------------------------------------- #


def test_strategy_summary_reports_size_duplication_and_oversized(
    pipeline: ChunkPipeline,
) -> None:
    """报告的每一行回答一个独立的问题：索引多大、重叠白花多少、有多少块会被截断."""
    storage = ChunkPolicy(strategy=STRATEGY_STRUCTURAL, max_tokens=40)
    report = pipeline.chunk_all(
        guide_document(), policies={STRATEGY_STRUCTURAL: storage}
    )
    rows = {row["strategy"]: row for row in report.strategy_summary()}
    assert rows[STRATEGY_STRUCTURAL]["oversized_count"] == 1
    assert rows[STRATEGY_STRUCTURAL]["duplication_ratio"] == 0.0
    assert rows[STRATEGY_FIXED]["duplication_ratio"] > 0
    assert all(row["chunks"] >= 1 for row in rows.values())


def test_report_oversized_and_markdown(pipeline: ChunkPipeline) -> None:
    report = pipeline.chunk_all(
        guide_document(),
        strategies=[STRATEGY_STRUCTURAL],
        policies={STRATEGY_STRUCTURAL: ChunkPolicy(strategy=STRATEGY_STRUCTURAL, max_tokens=40)},
    )
    assert len(report.oversized()) == 1
    markdown = report.render_markdown()
    assert "# 分块报告" in markdown
    assert "超预算的块" in markdown
    assert "`structural`" in markdown


def test_report_finds_duplicate_content_across_chunks(pipeline: ChunkPipeline) -> None:
    """``fingerprint`` 才能回答"同一段话存了几份"（``chunk_id`` 各有各的位置）."""
    document = text_document("同样的一段话。\n\n同样的一段话。", source="docs/dup.txt")
    report = pipeline.chunk_documents(
        [document],
        STRATEGY_RECURSIVE,
        ChunkPolicy(strategy=STRATEGY_RECURSIVE, max_tokens=8, overlap_tokens=0),
    )
    groups = report.duplicate_chunks()
    assert len(groups) == 1
    assert len(next(iter(groups.values()))) == 2
    assert report.summary_line().count("重复内容组 1") == 1


def test_report_knowledge_records_use_chunk_identity(pipeline: ChunkPipeline) -> None:
    """记录里的 ``doc_id`` 是 ``chunk_id``：知识库的唯一键就是片段."""
    report = pipeline.chunk_all(guide_document(), strategies=[STRATEGY_STRUCTURAL])
    records = report.knowledge_records()
    chunk_ids = {chunk.chunk_id for chunk in report.chunks()}
    assert {record["doc_id"] for record in records} == chunk_ids
    assert all(record["metadata"]["parent_doc_id"] for record in records)
    assert report.to_dict()["chunks"] == len(records)


def test_empty_report_is_well_defined() -> None:
    """空报告不该崩（表头恒存在，指标取 0）."""
    report = ChunkReport()
    assert report.count == 0
    assert report.strategies == ()
    assert report.duplicate_chunks() == {}
    assert report.strategy_summary() == []
    assert "0 份文档" in report.summary_line()


# --------------------------------------------------------------------------- #
# 事前估算与记录形状
# --------------------------------------------------------------------------- #


def test_chunking_plan_is_exact_for_fixed_and_an_upper_bound_elsewhere() -> None:
    """``estimated_chunks`` 的适用范围必须写清楚（否则它会被当成实测值）."""
    fixed = chunking_plan(614, default_policy(STRATEGY_FIXED))
    assert fixed["estimated_chunks_exact"] is True
    assert fixed["chunks"] == 3
    recursive = chunking_plan(614, default_policy(STRATEGY_RECURSIVE))
    assert recursive["estimated_chunks_exact"] is False
    assert "通常更少" in recursive["note"]
    assert recursive["measurer"] == "chars"


def test_knowledge_record_shape_names_the_embedding_input() -> None:
    """把"存的"与"embed 的"分开写出来——**不写就会被"修"成一个 bug**."""
    shape = knowledge_record_shape()
    assert shape["embedding_input"] == "metadata.retrieval_text（含标题面包屑），不是 text"
    assert set(shape["fields"]) == {"doc_id", "source", "text", "metadata"}
    assert "day064" in shape["next_step"]


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #


def test_retrieval_probe_requires_question_and_expectation() -> None:
    """没有期望片段就无法判定命中——"人工看一眼觉得还行"不是判据."""
    with pytest.raises(ValueError):
        RetrievalProbe(question="", expect="片段")
    with pytest.raises(ValueError):
        RetrievalProbe(question="问题", expect="  ")


def test_build_index_embeds_the_retrieval_view_but_stores_the_raw_text() -> None:
    """向量用 ``retrieval_text``（含面包屑），载荷用 ``text``（可直接引用）.

    两处必须一致：``ChunkSet.knowledge_records`` 也是这么分的，
    否则评估出来的分数对应的不是线上真正会用的那份文本。
    """
    document = guide_document()
    chunk_set = ChunkPipeline(measurer=chars(), embedding=MockEmbedding()).chunk(
        document, STRATEGY_STRUCTURAL
    )
    calls: list[str] = []

    class Recorder(MockEmbedding):
        def embed(self, text: str) -> list[float]:
            calls.append(text)
            return super().embed(text)

    store = build_index(chunk_set, Recorder())
    assert isinstance(store, InMemoryVectorStore)
    assert len(calls) == chunk_set.count
    assert calls == [chunk.retrieval_text for chunk in chunk_set.chunks]
    assert calls[1] != chunk_set.chunks[1].text  # 带上了标题面包屑


def test_score_probes_counts_a_chunk_that_was_cut_in_half_as_a_miss() -> None:
    """**期望片段被切开 = 必然 miss**：这是分块策略最直接的检验方式.

    小预算的固定长度策略会把"相似度低于 0.35 的直接丢掉"这句话切开，
    于是任何块都不含完整片段——此时向量检索再准也没用。
    """
    document = guide_document()
    probes = [
        RetrievalProbe(question="阈值设多少", expect="相似度低于 0.35 的直接丢掉")
    ]
    provider = MockEmbedding()
    chopped = ChunkPipeline(measurer=chars(), embedding=provider).chunk(
        document,
        STRATEGY_FIXED,
        # 预算 12 个字符，而期望片段有 15 个字符：**它不可能完整地落进任何一块**
        ChunkPolicy(strategy=STRATEGY_FIXED, max_tokens=12, overlap_tokens=0),
    )
    intact = ChunkPipeline(measurer=chars(), embedding=provider).chunk(
        document, STRATEGY_STRUCTURAL
    )
    assert score_probes(chopped, probes, provider).hit_count == 0
    assert score_probes(intact, probes, provider).hit_count == 1


def test_score_probes_rejects_a_zero_depth() -> None:
    """``top_k < 1`` 是一个参数矛盾（"看前 0 条"没有意义）."""
    document = guide_document()
    provider = MockEmbedding()
    chunk_set = ChunkPipeline(measurer=chars(), embedding=provider).chunk(
        document, STRATEGY_RECURSIVE
    )
    with pytest.raises(ValueError):
        score_probes(chunk_set, [], provider, top_k=0)


def test_evaluate_strategies_separates_chunk_quality_from_retrieval_luck() -> None:
    """把 ``top_k`` 放大到覆盖全部块，分数就只反映**片段有没有被完整保留**.

    这一步很关键：用 ``MockEmbedding``（没有真实语义）时，top-3 的差距里
    混进了"向量检索的运气"，而**分块策略真正的贡献是"片段是否被切碎"**。
    两个深度各测一次，正好把两件事分开：

    - ``top_k=8``（覆盖全部块）：四个策略都应当 100% 命中——
      这些期望片段在各自的切法下都被完整保留了下来；
    - ``top_k=3``：块越多、越容易有"对的块排在第 4 位"的漏网，
      于是命中率会低于 100%——**这是检索深度的问题，不是分块的问题。**
    """
    document = guide_document()
    probes = [
        RetrievalProbe(question="分块预算怎么定", expect="固定长度分块每 320 个字符切一刀"),
        RetrievalProbe(question="相似度阈值设多少", expect="相似度低于 0.35 的直接丢掉"),
        RetrievalProbe(
            question="重叠多花多少钱", expect="重叠 15% 意味着其中约 1.8 万条是重复内容"
        ),
    ]
    comprehensive = evaluate_strategies(
        document, probes, top_k=8, embedding=MockEmbedding()
    )
    assert all(score.hit_rate == 1.0 for score in comprehensive.scores)
    assert all(score.mrr > 0 for score in comprehensive.scores)
    narrow = evaluate_strategies(document, probes, top_k=3, embedding=MockEmbedding())
    assert min(score.hit_rate for score in narrow.scores) < 1.0


def test_evaluation_keeps_metric_directions_apart() -> None:
    """质量指标取最大、成本指标取最小：**同一个排序函数处理两个方向一定会出错**."""
    evaluation = evaluate_strategies(
        guide_document(),
        [RetrievalProbe(question="阈值设多少", expect="相似度低于 0.35")],
        top_k=8,
        embedding=MockEmbedding(),
    )
    best_mrr = evaluation.best("mrr")
    assert best_mrr.mrr == max(score.mrr for score in evaluation.scores)
    cheapest = evaluation.best("total_tokens")
    assert cheapest.total_tokens == min(
        score.total_tokens for score in evaluation.scores
    )
    smallest_chunks = evaluation.best("avg_hit_chars")
    assert smallest_chunks.avg_hit_chars == min(
        score.avg_hit_chars for score in evaluation.scores
    )
    assert evaluation.summary_line().startswith("4 个策略")
    markdown = evaluation.render_markdown()
    assert "# 分块策略检索评估" in markdown
    assert "hit@k" in markdown


def test_evaluate_strategies_rejects_an_unknown_metric() -> None:
    evaluation = evaluate_strategies(
        guide_document(),
        [RetrievalProbe(question="分块", expect="分块")],
        embedding=MockEmbedding(),
    )
    with pytest.raises(ValueError):
        evaluation.best("precision")


def test_evaluate_strategies_marks_ambiguous_probes_separately() -> None:
    """出现多次的期望片段会让分数**虚高且毫不显眼**，因此必须单独列出来."""
    document = text_document("重复的一句。\n\n重复的一句。", source="docs/amb.txt")
    probes = [
        RetrievalProbe(question="重复的一句", expect="重复的一句。"),
        RetrievalProbe(question="独一无二", expect="并不存在的片段"),
    ]
    assert ambiguous_probes(document, probes) == [
        "重复的一句 → 期望片段在原文中出现 2 次"
    ]
    evaluation = evaluate_strategies(
        document, probes, strategies=[STRATEGY_RECURSIVE], embedding=MockEmbedding()
    )
    assert evaluation.ambiguous_probes == (
        "重复的一句 → 期望片段在原文中出现 2 次",
    )
    assert "歧义探针" in evaluation.render_markdown()
    assert evaluation.scores[0].missed()[0].question == "独一无二"


def test_evaluation_with_no_hits_still_reports_zeroes() -> None:
    """一条都没命中时指标是 0 而不是崩（``avg_hit_chars`` 没有样本时取 0）."""
    evaluation = evaluate_strategies(
        guide_document(),
        [RetrievalProbe(question="无关问题", expect="原文里没有这段话")],
        strategies=[STRATEGY_FIXED],
        embedding=MockEmbedding(),
    )
    score = evaluation.scores[0]
    assert score.hit_rate == 0.0
    assert score.mrr == 0.0
    assert score.avg_hit_chars == 0.0
    assert "未命中明细" in evaluation.render_markdown()


def test_evaluate_strategies_rejects_a_zero_top_k() -> None:
    with pytest.raises(ValueError):
        evaluate_strategies(
            guide_document(),
            [RetrievalProbe(question="分块", expect="分块")],
            top_k=0,
            embedding=MockEmbedding(),
        )
