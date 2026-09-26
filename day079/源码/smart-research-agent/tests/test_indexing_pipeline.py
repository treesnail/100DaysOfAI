"""``indexing.pipeline`` 的测试：day061 → day062 → day064 → day065 的链子接上了没有（M6-D4）.

本文件只回答一个问题：**四天的产物串起来之后还认不认得彼此**。
因此它的重心不在"数字算得对不对"（那是 ``test_indexing_builder.py`` 的事），
而在三处接缝：

```text
文档 → 块           default_registry() 加载的 Document 能被 ChunkPipeline 切
块 → 记录           ChunkSet.knowledge_records() 的形状能被 builder 消费
记录 → 库 → 检索    建出来的库能被 VectorIngestPipeline 搜到（复用同一条编码路径）
```

另外两条用例守的是"装配"这件事本身：``stats()`` 的键集合完整、
``default_indexing_pipeline()`` 在默认 settings 下**可构造且不建任何目录**
（"问一下状态"不该在磁盘上留下东西）。

全部离线、确定性、零网络：默认 settings 就是"flat 后端 + char-ngram 编码器"，
两者都不需要网络与可选依赖。落盘只发生在 ``tmp_path`` 上。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_research_agent.chunking.pipeline import ChunkPipeline
from smart_research_agent.config import settings
from smart_research_agent.documents import default_registry
from smart_research_agent.documents.types import Document
from smart_research_agent.indexing.cache import EmbeddingCache
from smart_research_agent.indexing.encoder import describe_embedding
from smart_research_agent.indexing.pipeline import (
    BACKUP_DIR_NAME,
    IndexingPipeline,
    default_indexing_pipeline,
)
from smart_research_agent.llm.embedding import CharNgramEmbedding, default_embedding
from tests.test_indexing_builder import Rig, base_records, make_rig

#: 测试用 Markdown：两个标题（"检索手册" 与 "成本"）+ 若干自然段.
#:
#: 它刻意写得比一份 ``max_tokens=320`` 的预算长，因此**必然切出多块**——
#: 单块的文档会让"块 → 记录 → 库"这条链子只剩下一次搬运，
#: 那样连"条数对不对"都验不出来。
PIPELINE_MARKDOWN = """# 检索手册

本文给出一套可复现的检索参数，所有数字都在同一台机器上实测。

## 分块

固定长度分块每 320 个字符切一刀，重叠 48 个字符。它不认识标点，因此一定会切在句子中间，
而重叠只能保证被切断的句子在至少一块里是完整的。

递归分块优先在段落边界切开，其次换行，然后是句号与逗号。中文标点必须在分隔符表里，
否则一份中文文档会一路退到按字符硬切，而那种退化不会有任何报错。

语义分块按相邻段落的相似度找断层，它需要 embedding；而结构分块直接用标题层级，
因此两者对"一段话属于哪一节"的答案可能不同，那份差异必须能被报告出来。

## 成本

一次全量重建索引大约 12 万条记录，按每条 320 token 计，embedding 花费约 3.8 美元。

而重叠 15% 意味着其中约 1.8 万条是重复内容，这笔钱换来的只是边界处不丢句子。
"""


def guide_document(source: str = "docs/pipeline.md") -> Document:
    """把 ``PIPELINE_MARKDOWN`` 解析成归一化文档（走 day061 的真实加载器）."""
    return default_registry().load_bytes(
        PIPELINE_MARKDOWN.encode("utf-8"), source=source, media_type="text/markdown"
    )


@dataclass
class PipelineRig:
    """一份装配好的流水线 + 它背后那套零件（断言时直接读它们）."""

    rig: Rig
    pipeline: IndexingPipeline

    @property
    def backend(self) -> Any:
        return self.rig.backend

    @property
    def versions(self) -> Any:
        return self.rig.versions


def make_pipeline_rig(
    tmp_path: Any = None,
    *,
    embedding: Any = None,
    backend: Any = None,
    with_versions: bool = True,
    with_backups: bool = False,
) -> PipelineRig:
    """复用 ``test_indexing_builder`` 的装配器，再套一层流水线.

    版本表与备份表**两边各传一次**（builder 用来写，pipeline 用来读），
    正常路径上它们是同一份对象——``default_indexing_pipeline()`` 就是这么做的。
    两个测试文件共用同一份假提供方与假后端，理由与本课"一条规则只有一个实现"
    是同一条纪律：样本里的一个问题要修两处，漏掉的那处只会表现为覆盖变少。
    """
    rig = make_rig(
        tmp_path,
        embedding=embedding,
        backend=backend,
        with_versions=with_versions,
        with_backups=with_backups,
    )
    pipeline = IndexingPipeline(rig.builder, versions=rig.versions, backups=rig.backups)
    return PipelineRig(rig=rig, pipeline=pipeline)


# --------------------------------------------------------------------------- #
# build_from_records：与 builder 逐字段一致
# --------------------------------------------------------------------------- #


def test_build_from_records_matches_the_builder_behaviour() -> None:
    """一次正向 + 一次重放：数字与直接调 builder 完全相同."""
    rig = make_pipeline_rig()

    first = rig.pipeline.build_from_records(base_records())

    assert first.mode == "incremental"
    assert first.seen == 3
    assert first.written == 3
    assert first.unchanged == 0
    assert first.encoded == 3
    assert first.batches == 1
    assert first.failed == 0
    assert rig.backend.count() == 3

    second = rig.pipeline.build_from_records(base_records())

    assert second.written == 0
    assert second.unchanged == 3
    assert second.encoded == 0
    assert second.batches == 0
    assert second.parent_version == first.version_id
    assert second.version_id == first.version_id
    assert rig.backend.count() == 3


# --------------------------------------------------------------------------- #
# build_from_documents：端到端
# --------------------------------------------------------------------------- #


def test_build_from_documents_is_end_to_end() -> None:
    """文档 → 块 → 记录 → 库 → 清单：条数一路对得上，且 id 就是 chunk_id."""
    document = guide_document()
    # 期望值由 day062 的真实分块器独立算出来（不是复述流水线的内部计数）
    expected = ChunkPipeline().chunk(document, "recursive")
    assert expected.count >= 2, "样本本来就该切出多块（见 PIPELINE_MARKDOWN 的说明）"

    rig = make_pipeline_rig()
    report = rig.pipeline.build_from_documents([document], strategy="recursive")

    assert report.seen == expected.count
    assert report.written == expected.count
    assert report.encoded == expected.count
    assert report.failed == 0
    assert report.ok is True
    # 库里、清单里、块集里说的是同一批 id
    assert rig.backend.count() == expected.count
    assert set(rig.backend.ids()) == {chunk.chunk_id for chunk in expected.chunks}
    stats = rig.pipeline.stats()
    assert stats["manifest"]["count"] == expected.count
    assert stats["manifest"]["version_id"] == report.version_id
    # 落库的是**原文片段**（不是带面包屑的 retrieval_text）：命中之后要还给用户看的是它
    stored = rig.backend.get(expected.chunks[0].chunk_id)
    assert stored is not None
    assert stored.text == expected.chunks[0].text


def test_build_from_documents_defaults_to_the_settings_strategy() -> None:
    """``strategy`` 缺省取 ``settings.chunking_strategy``（本课缺省是 recursive）."""
    document = guide_document()
    expected = ChunkPipeline().chunk(document, settings.chunking_strategy)

    rig = make_pipeline_rig()
    report = rig.pipeline.build_from_documents([document])

    assert report.seen == expected.count
    assert report.written == expected.count


def test_build_from_documents_replays_for_free() -> None:
    """同一份文档再走一次：一条都不重编码（缓存 + 差集都生效）."""
    document = guide_document()
    rig = make_pipeline_rig()

    first = rig.pipeline.build_from_documents([document], strategy="recursive")
    second = rig.pipeline.build_from_documents([document], strategy="recursive")

    assert first.seen == second.seen
    assert second.written == 0
    assert second.unchanged == first.seen
    assert second.encoded == 0
    assert second.batches == 0
    assert second.version_id == first.version_id


# --------------------------------------------------------------------------- #
# search：复用 day064 的检索路径
# --------------------------------------------------------------------------- #


def test_search_finds_the_chunk_that_contains_the_word() -> None:
    """查一个出现在某块里的词 → 有条数、有分数、且排序是降序的."""
    rig = make_pipeline_rig(embedding=CharNgramEmbedding(dimension=64))
    rig.pipeline.build_from_documents([guide_document()], strategy="recursive")

    result = rig.pipeline.search("重叠 48 个字符", top_k=3)

    assert result.count > 0
    assert result.metric == "cosine"
    assert result.top_k == 3
    assert result.filter_applied is False
    assert result.candidates == rig.backend.count()
    assert any("重叠 48 个字符" in hit.record.text for hit in result.hits)
    # ids() 与 hits 的顺序一致；分数非递增（排序规则与 types.sort_hits 相同）
    assert result.ids() == [hit.record.record_id for hit in result.hits]
    assert result.scores == sorted(result.scores, reverse=True)
    assert result.top() is not None


def test_search_passes_where_through_to_the_backend() -> None:
    """``where`` 是元数据过滤（不是访问控制），按 ``strategy`` 过滤能命中."""
    rig = make_pipeline_rig(embedding=CharNgramEmbedding(dimension=64))
    rig.pipeline.build_from_documents([guide_document()], strategy="recursive")

    filtered = rig.pipeline.search("重叠", where={"strategy": "recursive"})

    assert filtered.filter_applied is True
    assert filtered.count > 0
    assert all(hit.record.metadata["strategy"] == "recursive" for hit in filtered.hits)


def test_search_without_a_built_index_returns_no_hits() -> None:
    """空库上检索返回 0 条而不是报错（"查不到"是一个正常结果）."""
    rig = make_pipeline_rig(embedding=CharNgramEmbedding(dimension=64))

    result = rig.pipeline.search("任何查询")

    assert result.count == 0
    assert result.candidates == 0


# --------------------------------------------------------------------------- #
# stats / history
# --------------------------------------------------------------------------- #


def test_stats_keys_and_empty_state() -> None:
    """没构建过时：键集合完整、``manifest`` 是 ``None``、``history()`` 是空列表."""
    rig = make_pipeline_rig()

    stats = rig.pipeline.stats()

    assert set(stats) == {
        "backend",
        "metric",
        "dimension",
        "count",
        "identity",
        "manifest",
        "versions",
        "backups",
        "cache",
    }
    assert stats["backend"] == "spy-flat"
    assert stats["metric"] == "cosine"
    assert stats["dimension"] == 0  # 库还没定维（编码器的维度在 identity 里）
    assert stats["count"] == 0
    assert set(stats["identity"]) == {"provider", "model", "dimension", "key"}
    assert stats["manifest"] is None
    assert stats["versions"] == 0
    assert stats["backups"] == 0
    assert "identity_key" in stats["cache"]
    assert stats["cache"]["size"] == 0
    assert rig.pipeline.history() == []


def test_stats_and_history_track_every_version() -> None:
    """构建之后：manifest 是本次那一版（不含 entries），history 逐版给出摘要."""
    rig = make_pipeline_rig()

    first = rig.pipeline.build_from_records(base_records())
    stats = rig.pipeline.stats()

    assert stats["count"] == 3
    assert stats["dimension"] == 4  # 写过一次之后库的维度就定下来了
    assert stats["manifest"] is not None
    assert stats["manifest"]["version_id"] == first.version_id
    assert stats["manifest"]["count"] == 3
    assert "entries" not in stats["manifest"]  # include_entries=False
    assert stats["versions"] == 1
    assert stats["cache"]["size"] == 3

    history = rig.pipeline.history()
    assert len(history) == 1
    assert set(history[0]) == {"version_id", "summary_line"}
    assert history[0]["version_id"] == first.version_id
    assert "版本" in history[0]["summary_line"]

    changed = base_records()
    changed[0]["text"] = "改写了第一条的内容。"
    second = rig.pipeline.build_from_records(changed, reason="测试第二版")

    assert rig.pipeline.stats()["manifest"]["version_id"] == second.version_id
    history = rig.pipeline.history()
    assert [item["version_id"] for item in history] == [first.version_id, second.version_id]
    assert second.parent_version == first.version_id


def test_pipeline_without_a_version_store_reports_nothing() -> None:
    """没接版本表时 ``history()`` 是空列表、版本数报 0（"没有表"与"表是空的"同形）."""
    rig = make_pipeline_rig(with_versions=False)

    report = rig.pipeline.build_from_records(base_records())

    assert report.written == 3
    assert rig.pipeline.history() == []
    assert rig.pipeline.stats()["versions"] == 0
    assert report.version_id  # 没接版本表也照样算得出这一版的版本号


# --------------------------------------------------------------------------- #
# default_indexing_pipeline：装配
# --------------------------------------------------------------------------- #


def test_default_indexing_pipeline_creates_no_directories(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """默认 settings 下可构造；**构造过程不创建任何目录**（"问状态"不留痕）."""
    target = tmp_path / "index"
    monkeypatch.setattr(settings, "indexing_dir", str(target))
    monkeypatch.setattr(settings, "indexing_cache_path", "")

    pipeline = default_indexing_pipeline()

    assert isinstance(pipeline, IndexingPipeline)
    assert not target.exists()
    assert not (target / BACKUP_DIR_NAME).exists()
    stats = pipeline.stats()
    assert stats["versions"] == 0
    assert stats["backups"] == 0
    assert set(stats) >= {"backend", "identity", "cache"}
    assert stats["identity"]["provider"] == "CharNgramEmbedding"


def test_default_indexing_pipeline_builds_and_searches(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """默认 settings（flat + char-ngram）下建一份索引并搜到东西——全程离线."""
    monkeypatch.setattr(settings, "indexing_dir", str(tmp_path / "index"))
    monkeypatch.setattr(settings, "indexing_cache_path", "")
    pipeline = default_indexing_pipeline()

    report = pipeline.build_from_documents([guide_document()], strategy="recursive")

    assert report.failed == 0
    assert report.seen > 0
    assert pipeline.stats()["count"] == report.seen
    result = pipeline.search("重叠 48 个字符", top_k=3)
    assert result.count > 0
    assert any("重叠" in hit.record.text for hit in result.hits)


def test_default_indexing_pipeline_loads_a_matching_cache(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """磁盘上有一份**同身份**的缓存 → 构造时载入它（否则那份文件等于白写）."""
    cache_path = tmp_path / "cache.json"
    identity = describe_embedding(default_embedding())
    seed = EmbeddingCache(
        path=str(cache_path),
        provider=identity.provider,
        model=identity.model,
        dimension=identity.dimension,
    )
    seed.put("预先算好的一段文本", [1.0] + [0.0] * (identity.dimension - 1))
    seed.persist()

    monkeypatch.setattr(settings, "indexing_dir", str(tmp_path / "index"))
    monkeypatch.setattr(settings, "indexing_cache_path", str(cache_path))

    pipeline = default_indexing_pipeline()

    stats = pipeline.stats()
    assert stats["cache"]["size"] == 1
    assert stats["cache"]["path"] == str(cache_path)
    assert stats["identity"]["provider"] == identity.provider
