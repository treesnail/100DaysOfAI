"""day064 ``vectorstore.pipeline`` 的摄取测试：逐条编码 + 三种失败分家.

本文件盯住的是 day064 → day065 的那条接缝，因此断言分四组：

```text
计数与幂等   seen/written/unchanged/embedding_calls 各自对得上；重放同一批全是 unchanged
编码的是谁   retrieval_text 优先、缺它才回落 text；"存的"与"编码的"不是同一份
三种分家     缺 id / 空白文本 → skipped；维度不一致 / 后端拒绝 → failed；
             非 VectorStoreError 一律上抛（不许被记成 failed）
检索         查询走同一条编码路径；top_k / min_score 的三级优先级；维度不一致报 VectorError
```

全部离线、确定性、零网络：向量由 ``TableEmbedding``（或本文件里两个更小的假编码器）
写死，期望值手算得出来，不需要真实 embedding 服务。
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.vectorstore.errors import VectorError, VectorStoreError
from smart_research_agent.vectorstore.flat import FlatVectorStore
from smart_research_agent.vectorstore.pipeline import (
    EMBEDDING_TEXT_FIELD,
    FALLBACK_TEXT_FIELD,
    RECORD_ID_FIELD,
    VectorIngestPipeline,
    embedding_input_field,
    embedding_text,
    record_from_knowledge,
    record_id_of,
)
from smart_research_agent.vectorstore.types import SearchResult, VectorRecord
from tests.vectorstore_samples import (
    RECORD_IDS,
    TableEmbedding,
    knowledge_records,
    sample_vectors,
)

#: 与样本一致的维度（``vectorstore_samples.VECTOR_DIMENSION`` 是 8）.
DIMENSION = 8

#: 一个只在本文件里用作查询的文本；它的向量与 ``r_a`` / ``r_d`` 同向，
#: 于是"余弦并列"这件事必须由 id 升序来打破。
QUERY_TEXT = "查询：检索手册里怎么配阈值"
QUERY_VECTOR = (1.0,) + (0.0,) * (DIMENSION - 1)

#: 专门给 ``text`` 字段准备的"另一个"向量：最后一个分量是 1，
#: 与六条样本向量（以及它们的归一化结果）都不相同——于是"库里的向量
#: 到底来自哪一份文本"是一个能一眼看出来、且不会误判的事实。
TEXT_VARIANT_VECTOR = (0.0,) * (DIMENSION - 1) + (1.0,)


# --------------------------------------------------------------------------- #
# 本文件专用的假编码器（都不是真实模型，因此完全离线且确定性）
# --------------------------------------------------------------------------- #


class WideTextEmbedding(EmbeddingProvider):
    """对**某一段**文本返回 16 维、其余返回 8 维（制造一条维度不一致的脏数据）.

    ``dimension`` 属性故意报 8：它在演示"编码器自己声明的维度"与
    "它这次实际给出的向量"可以是两回事。本模块因此用**真实向量长度**判维度，
    而不是相信 ``dimension``——相信它的后果是那条 16 维的向量一路走到写库，
    然后在整批 upsert 里把**所有**干净记录一起炸掉。
    """

    def __init__(
        self,
        wide_text: str,
        table: dict[str, tuple[float, ...]],
        dimension: int = DIMENSION,
    ) -> None:
        self._wide_text = wide_text
        self._table = dict(table)
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        if text == self._wide_text:
            return [1.0] + [0.0] * 15
        return list(self._table.get(text, [1.0] + [0.0] * (DIMENSION - 1)))


class BatchCounterEmbedding(EmbeddingProvider):
    """既能逐条也能批量的编码器，并分别记下两条路径被调用的次数.

    用途是把"本课确实在逐条编码"变成一个**可断言**的事实：如果 pipeline
    图省事去调一次 ``embed_batch``，``batch_calls`` 会变成 1、
    ``embedding_calls`` 会变成 1（而不是条数）——那这两个字段就失去了
    "能对账"的意义（day065 要靠它们算"省下了多少次编码"）。
    """

    def __init__(
        self,
        table: dict[str, tuple[float, ...]],
        dimension: int = DIMENSION,
    ) -> None:
        self._table = dict(table)
        self._dimension = dimension
        self.single_calls = 0
        self.batch_calls = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        self.single_calls += 1
        return list(self._table.get(text, [1.0] + [0.0] * (self._dimension - 1)))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        return [self.embed(text) for text in texts]


# --------------------------------------------------------------------------- #
# 构造样本与工具
# --------------------------------------------------------------------------- #


def unit(vector: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """L2 归一化（与 ``types.make_record`` 用的公式逐字相同）."""
    total = math.fsum(float(value) * float(value) for value in vector)
    return tuple(float(value) / math.sqrt(total) for value in vector)


def retrieval_table() -> dict[str, tuple[float, ...]]:
    """``retrieval_text`` → 样本向量（每条记录拿到它"该有"的那条向量）."""
    vectors = sample_vectors()
    return {
        record["metadata"][EMBEDDING_TEXT_FIELD]: vectors[record[RECORD_ID_FIELD]]
        for record in knowledge_records()
    }


def seam_table() -> dict[str, tuple[float, ...]]:
    """``retrieval_text`` → 样本向量；``text`` → 另一个完全不同的向量.

    两张表**同时**挂在同一个编码器上，于是"编码用的是哪一份文本"这件事
    在库里有可见的后果（否则两条路径给出同一个向量，断言就只是在复述实现）。
    """
    vectors = sample_vectors()
    table: dict[str, tuple[float, ...]] = {}
    for record in knowledge_records():
        table[record["metadata"][EMBEDDING_TEXT_FIELD]] = vectors[record[RECORD_ID_FIELD]]
        table[record["text"]] = TEXT_VARIANT_VECTOR
    return table


def configured_pipeline(**kwargs: Any) -> VectorIngestPipeline:
    """装好六条样本、并且**能回答查询**的摄取器（查询文本映射到 q_axis）."""
    store = FlatVectorStore(metric="cosine")
    table = retrieval_table()
    table[QUERY_TEXT] = QUERY_VECTOR
    embedding = TableEmbedding(table)
    pipeline = VectorIngestPipeline(store, embedding, **kwargs)
    assert pipeline.ingest(knowledge_records()).written == 6
    return pipeline


# --------------------------------------------------------------------------- #
# 一、计数与幂等：embedding_calls 的含义
# --------------------------------------------------------------------------- #


def test_ingest_writes_every_record_and_counts_embedding_calls() -> None:
    """六条记录 → 全部写入，且 ``embedding_calls == 6``（逐条编码）."""
    store = FlatVectorStore(metric="cosine")
    pipeline = VectorIngestPipeline(store, TableEmbedding(retrieval_table()))

    report = pipeline.ingest(knowledge_records())

    assert report.seen == 6
    assert report.written == 6
    assert report.unchanged == 0
    assert report.skipped == 0
    assert report.failed == 0
    assert report.embedding_calls == 6
    assert report.backend == "flat"
    assert report.metric == "cosine"
    assert report.dimension == DIMENSION
    assert report.failures == ()
    assert report.ok is True
    assert store.count() == 6
    assert store.ids() == sorted(RECORD_IDS)
    # 报告的自检点：每条记录都要有一个归宿
    assert report.seen == report.written + report.unchanged + report.skipped + report.failed


def test_report_projects_to_json_and_a_readable_line() -> None:
    """报告要能直接进响应体（``json.dumps``）与日志（一行摘要）."""
    store = FlatVectorStore(metric="cosine")
    pipeline = VectorIngestPipeline(store, TableEmbedding(retrieval_table()))

    report = pipeline.ingest(knowledge_records())
    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["embedding_calls"] == 6
    assert payload["ok"] is True
    assert payload["failures"] == []
    assert set(payload) >= {
        "seen",
        "written",
        "unchanged",
        "skipped",
        "failed",
        "embedding_calls",
        "backend",
        "metric",
        "dimension",
        "ok",
        "failures",
    }
    line = report.summary_line()
    assert "6" in line and "flat" in line


def test_embedding_calls_counts_per_record_calls_not_records() -> None:
    """用一个**同时支持批量**的编码器断言：走的仍是逐条路径.

    这是本文件最重要的一条"语义断言"：``embedding_calls`` 不是
    ``len(records)`` 的另一种写法，它是"实际发生了多少次编码"的事实。
    """
    embedding = BatchCounterEmbedding(retrieval_table())
    store = FlatVectorStore(metric="cosine")
    pipeline = VectorIngestPipeline(store, embedding)

    report = pipeline.ingest(knowledge_records())

    assert report.embedding_calls == 6
    assert embedding.single_calls == 6
    assert embedding.batch_calls == 0  # 批量入口一次都没走（那是 day065 的事）


def test_replaying_the_same_batch_reports_unchanged() -> None:
    """重放同一批 → ``written=0``、``unchanged=6``（day065 增量索引的地基）.

    注意这里同时验证了本课的另一半代价：**重放仍然编码了一遍**
    （``embedding_calls`` 仍是 6），但库一个字节都没动——
    "编码开销"与"库是否变化"是两件事，报告要能分别回答。
    """
    store = FlatVectorStore(metric="cosine")
    embedding = TableEmbedding(retrieval_table())
    pipeline = VectorIngestPipeline(store, embedding)

    pipeline.ingest(knowledge_records())
    replay = pipeline.ingest(knowledge_records())

    assert replay.written == 0
    assert replay.unchanged == 6
    assert replay.skipped == 0
    assert replay.failed == 0
    assert replay.embedding_calls == 6
    assert replay.ok is True
    assert store.count() == 6


# --------------------------------------------------------------------------- #
# 二、编码的是哪一份文本：retrieval_text 优先
# --------------------------------------------------------------------------- #


def test_retrieval_text_is_what_gets_embedded_and_text_is_what_gets_stored() -> None:
    """编码用 ``retrieval_text``、落库存 ``text``——两份不同的文本，两件不同的事."""
    store = FlatVectorStore(metric="cosine")
    pipeline = VectorIngestPipeline(store, TableEmbedding(seam_table()))
    records = knowledge_records()

    report = pipeline.ingest(records)

    assert report.written == 6
    for record in records:
        record_id = record[RECORD_ID_FIELD]
        stored = store.get(record_id)
        assert stored is not None
        # 报告可核对：这条走的是 retrieval_text
        assert embedding_input_field(record) == EMBEDDING_TEXT_FIELD
        # 库里的向量来自 retrieval_text（cosine 下已归一化），不是来自 text
        assert stored.vector == pytest.approx(unit(sample_vectors()[record_id]))
        assert stored.vector != pytest.approx(TEXT_VARIANT_VECTOR)
        # 而库里存下来的原文仍然是 text 那一份
        assert stored.text == record["text"]


def test_missing_or_blank_retrieval_text_falls_back_to_text() -> None:
    """缺 ``retrieval_text``（或它只有空白）时回落到 ``text``，且这是可查的."""
    record: dict[str, Any] = {
        "doc_id": "0f1e2d3c4b5a6978",
        "source": "docs/manual.md",
        "text": "只有正文的一条记录，没有检索视图。",
        "metadata": {"strategy": "fixed"},
    }
    assert embedding_text(record) == record["text"]
    assert embedding_input_field(record) == FALLBACK_TEXT_FIELD

    blank_view = dict(record, metadata={"retrieval_text": "   \n\t "})
    assert embedding_text(blank_view) == record["text"]
    assert embedding_input_field(blank_view) == FALLBACK_TEXT_FIELD

    store = FlatVectorStore(metric="cosine")
    embedding = TableEmbedding({record["text"]: TEXT_VARIANT_VECTOR})
    pipeline = VectorIngestPipeline(store, embedding)

    report = pipeline.ingest([record])

    assert report.written == 1
    stored = store.get(record["doc_id"])
    assert stored is not None
    assert stored.vector == pytest.approx(TEXT_VARIANT_VECTOR)  # 用的是 text 的那条向量
    assert stored.text == record["text"]


def test_record_id_of_treats_blank_ids_as_missing() -> None:
    """id 的"缺"包括缺失、``None`` 与纯空白——三者都拿不到一个可用的键."""
    assert record_id_of({"doc_id": "abc"}) == "abc"
    assert record_id_of({"doc_id": "  abc \n"}) == "abc"
    assert record_id_of({}) == ""
    assert record_id_of({"doc_id": None}) == ""
    assert record_id_of({"doc_id": "   "}) == ""


def test_record_from_knowledge_keeps_the_original_text_and_folds_source() -> None:
    """纯函数逐项核对：id 不改写、存原文、元数据原样 + 折入 source."""
    raw = knowledge_records()[0]
    vector = [1.0] + [0.0] * (DIMENSION - 1)

    record = record_from_knowledge(raw, vector)

    assert isinstance(record, VectorRecord)
    assert record.record_id == RECORD_IDS[0]
    assert record.text == raw["text"]
    assert record.metadata["source"] == raw["source"]
    assert record.metadata["retrieval_text"] == raw["metadata"][EMBEDDING_TEXT_FIELD]
    assert record.dimension == DIMENSION

    # metadata 不是字典时不炸：当成空字典，再补上 source（后面的校验会响该响的）
    loose = {
        "doc_id": "d1b2c3d4e5f60718",
        "source": "docs/y.md",
        "text": "正文",
        "metadata": None,
    }
    assert record_from_knowledge(loose, vector).metadata == {"source": "docs/y.md"}

    # 没有 source 就不补（"没有来源"与"来源是空串"是两件事，不替它编一个）
    no_source = {"doc_id": "f1b2c3d4e5f60718", "text": "正文", "metadata": {}}
    assert record_from_knowledge(no_source, vector).metadata == {}

    # 原始 metadata 里已经有 source 时以它为准（顶层的那个不覆盖它）
    own_source = {
        "doc_id": "a2b3c4d5e6f70819",
        "source": "docs/top.md",
        "text": "正文",
        "metadata": {"source": "docs/meta.md"},
    }
    assert record_from_knowledge(own_source, vector).metadata["source"] == "docs/meta.md"

    # 顶层与 metadata 都查：retrieval_text 在顶层时也认得（防御性，换产出方时不至于静默回落）
    top_level = {
        "doc_id": "e1b2c3d4e5f60718",
        "text": "正文",
        "retrieval_text": "顶层检索视图",
        "metadata": {},
    }
    assert embedding_text(top_level) == "顶层检索视图"
    assert embedding_text({"doc_id": "x", "text": None, "metadata": {}}) == ""


# --------------------------------------------------------------------------- #
# 三、三种"没做成"必须分家
# --------------------------------------------------------------------------- #


def test_record_without_doc_id_is_skipped_before_encoding() -> None:
    """缺 id → ``skipped``，而且**连一次编码都不发起**（不浪费调用）."""
    store = FlatVectorStore(metric="cosine")
    embedding = TableEmbedding(retrieval_table())
    pipeline = VectorIngestPipeline(store, embedding)
    nameless = {
        "source": "docs/x.md",
        "text": "有正文但没有 id。",
        "metadata": {"retrieval_text": "有检索视图但没有 id。"},
    }

    report = pipeline.ingest([nameless])

    assert report.seen == 1
    assert report.skipped == 1
    assert report.written == 0
    assert report.failed == 0
    assert report.embedding_calls == 0
    assert report.ok is False
    assert store.count() == 0
    assert report.failures[0][0] == ""
    assert "跳过" in report.failures[0][1]


def test_blank_text_and_retrieval_text_is_skipped() -> None:
    """两份文本都是空白（或都不存在）→ ``skipped``，同样不发起编码."""
    store = FlatVectorStore(metric="cosine")
    pipeline = VectorIngestPipeline(store, TableEmbedding(retrieval_table()))
    blank_both = {
        "doc_id": "a1b2c3d4e5f60718",
        "source": "docs/x.md",
        "text": "   \n ",
        "metadata": {"retrieval_text": "\t \n"},
    }
    no_text_at_all = {"doc_id": "b1b2c3d4e5f60718", "source": "docs/x.md", "metadata": {}}

    report = pipeline.ingest([blank_both, no_text_at_all])

    assert report.seen == 2
    assert report.skipped == 2
    assert report.written == 0
    assert report.embedding_calls == 0
    assert store.count() == 0
    assert [record_id for record_id, _ in report.failures] == [
        "a1b2c3d4e5f60718",
        "b1b2c3d4e5f60718",
    ]
    assert all("跳过" in reason for _, reason in report.failures)


def test_skipped_and_failed_are_counted_separately() -> None:
    """一条缺 id、一条维度不一致 → 两个计数各记各的，谁也不吞谁."""
    records = knowledge_records()
    wide_text = records[0]["metadata"][EMBEDDING_TEXT_FIELD]
    store = FlatVectorStore(metric="cosine", dimension=DIMENSION)
    embedding = WideTextEmbedding(wide_text, retrieval_table())
    pipeline = VectorIngestPipeline(store, embedding)
    nameless = {"source": "docs/x.md", "text": "有正文但没有 id。", "metadata": {}}

    report = pipeline.ingest([nameless, *records[:2]])

    assert report.seen == 3
    assert report.skipped == 1
    assert report.failed == 1
    assert report.written == 1
    assert report.embedding_calls == 2  # 被跳过的那条没编码，另外两条都编了


def test_one_bad_dimension_does_not_kill_the_batch() -> None:
    """维度不一致 → 单条 ``failed``，其余记录照常写入（脏数据不该毁掉整批）."""
    records = knowledge_records()
    wide_text = records[0]["metadata"][EMBEDDING_TEXT_FIELD]
    # 库已是 8 维：这是"拿一个 16 维的向量去打 8 维的库"那件事故的前提
    store = FlatVectorStore(metric="cosine", dimension=DIMENSION)
    embedding = WideTextEmbedding(wide_text, retrieval_table())
    pipeline = VectorIngestPipeline(store, embedding)

    report = pipeline.ingest(records)

    assert report.seen == 6
    assert report.failed == 1
    assert report.written == 5
    assert report.skipped == 0
    assert report.embedding_calls == 6  # 脏数据也是在编码之后才发现维度不对的
    assert store.count() == 5
    assert store.get(RECORD_IDS[0]) is None
    assert report.failures[0][0] == RECORD_IDS[0]
    assert "维度" in report.failures[0][1]
    assert report.ok is False
    assert report.seen == report.written + report.unchanged + report.skipped + report.failed


def test_record_rejected_at_construction_is_failed_not_raised() -> None:
    """元数据类型越界（``RecordError``）也属于 ``VectorStoreError`` → 记 ``failed``."""
    raw = {
        "doc_id": "c1b2c3d4e5f60718",
        "source": "docs/x.md",
        "text": "正文里带一个嵌套的元数据值。",
        "metadata": {"nested": {"a": 1}, "retrieval_text": "正文里带一个嵌套的元数据值。"},
    }
    store = FlatVectorStore(metric="cosine")
    pipeline = VectorIngestPipeline(store, TableEmbedding({raw["text"]: TEXT_VARIANT_VECTOR}))

    report = pipeline.ingest([raw])

    assert report.failed == 1
    assert report.written == 0
    assert store.count() == 0
    assert "记录构造被拒" in report.failures[0][1]
    assert "nested" in report.failures[0][1]  # 报错要点出是哪个键


def test_unexpected_backend_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """``upsert`` 抛 ``RuntimeError`` → **向上抛**，绝不被记成 ``failed``.

    这是"只吞 VectorStoreError"这条纪律的反面用例：如果这里被吞掉，
    后端实现里的真 bug 会永远以一行 ``failed=1`` 的形式藏着。
    """
    store = FlatVectorStore(metric="cosine")
    pipeline = VectorIngestPipeline(store, TableEmbedding(retrieval_table()))

    def explode(_records: Any) -> Any:
        raise RuntimeError("后端炸了：这个异常不属于 VectorStoreError")

    monkeypatch.setattr(store, "upsert", explode)

    with pytest.raises(RuntimeError, match="后端炸了"):
        pipeline.ingest(knowledge_records())


def test_batch_level_vectorstore_error_is_recorded_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """后端对**整批**的拒绝（``VectorStoreError``）→ 记 ``failed``，不打断调用方.

    与上一条对照着看：同一次调用里，认得的失败变成报告，不认识的失败变成异常。
    """
    store = FlatVectorStore(metric="cosine")
    pipeline = VectorIngestPipeline(store, TableEmbedding(retrieval_table()))

    def reject(_records: Any) -> Any:
        raise VectorStoreError("后端拒绝了这一整批")

    monkeypatch.setattr(store, "upsert", reject)

    report = pipeline.ingest(knowledge_records())

    assert report.written == 0
    assert report.unchanged == 0
    assert report.failed == 6
    assert report.embedding_calls == 6  # 编码已经发生，失败发生在写入这一步
    assert len(report.failures) == 6
    assert all("拒绝了这一整批" in reason for _, reason in report.failures)
    assert store.count() == 0


# --------------------------------------------------------------------------- #
# 四、检索：同一条编码路径 + 三级优先级
# --------------------------------------------------------------------------- #


def test_search_uses_the_same_encoding_path_and_ranks_by_score_then_id() -> None:
    """查询文本走同一个编码器；并列的余弦由 id 升序定胜负（排序规则是确定的）."""
    pipeline = configured_pipeline()
    embedding = pipeline.embedding

    result = pipeline.search(QUERY_TEXT)

    assert isinstance(result, SearchResult)
    assert result.metric == "cosine"
    assert result.top_k == 5  # settings.vector_default_top_k
    assert result.count == 5
    assert result.ids()[0] == RECORD_IDS[0]
    assert result.ids()[1] == RECORD_IDS[3]
    assert isinstance(embedding, TableEmbedding)
    assert embedding.single_calls == 7  # 六条入库 + 一次查询，都在 embed 这一条路径上


def test_top_k_priority_is_call_then_constructor_then_settings() -> None:
    """``top_k`` 的三级优先级：调用参数 > 构造参数 > ``settings``."""
    assert configured_pipeline().search(QUERY_TEXT).top_k == 5

    configured = configured_pipeline(top_k=3)
    assert configured.search(QUERY_TEXT).top_k == 3
    assert configured.search(QUERY_TEXT).count == 3
    assert configured.search(QUERY_TEXT, top_k=1).top_k == 1
    assert configured.search(QUERY_TEXT, top_k=1).count == 1


def test_min_score_priority_is_call_then_constructor_then_settings() -> None:
    """``min_score`` 的三级优先级；默认是"不设阈值"而不是 0."""
    plain = configured_pipeline()
    assert plain.search(QUERY_TEXT).count == 5  # 默认不设阈值（只被 top_k 截断）

    configured = configured_pipeline(min_score=0.99)
    assert configured.search(QUERY_TEXT).count == 2  # 只有两条余弦是 1.0
    # 调用参数把构造参数整个盖掉（0.0 是"阈值 0"，不是"没给"）
    assert configured.search(QUERY_TEXT, top_k=6, min_score=0.0).count == 6
    assert configured.search(QUERY_TEXT, min_score=0.9).count == 2


def test_search_rejects_a_query_vector_of_the_wrong_dimension() -> None:
    """查询向量的维度与库不一致 → ``VectorError``（而不是一条空结果）."""
    pipeline = configured_pipeline()
    pipeline.embedding = TableEmbedding(dimension=16)  # 换成一个 16 维编码器

    with pytest.raises(VectorError) as excinfo:
        pipeline.search(QUERY_TEXT)

    message = str(excinfo.value)
    assert "16 维" in message and "8 维" in message  # 两个数字都要点出来


def test_stats_exposes_the_ingest_contract() -> None:
    """``stats()`` 的键集合与内容（端点直接回显它）."""
    embedding = TableEmbedding(retrieval_table())
    store = FlatVectorStore(metric="cosine")
    pipeline = VectorIngestPipeline(store, embedding, top_k=4, min_score=0.25)

    report = pipeline.ingest(knowledge_records())
    stats = pipeline.stats()

    assert set(stats) >= {
        "backend",
        "metric",
        "dimension",
        "count",
        "top_k",
        "min_score",
        "embedding",
        "embedding_dimension",
        "embedding_input_field",
    }
    assert stats["backend"] == "flat"
    assert stats["metric"] == "cosine"
    assert stats["dimension"] == DIMENSION
    assert stats["count"] == 6
    assert stats["top_k"] == 4
    assert stats["min_score"] == 0.25
    assert stats["embedding"] == "TableEmbedding"
    assert stats["embedding_dimension"] == DIMENSION
    assert stats["embedding_input_field"] == EMBEDDING_TEXT_FIELD
    assert report.embedding_calls == 6
    assert json.loads(json.dumps(stats))["count"] == 6
