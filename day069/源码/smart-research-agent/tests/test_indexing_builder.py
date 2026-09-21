"""``indexing.builder`` 的测试：装配线的承诺逐条钉住（M6-D4）.

本文件按"承诺"分八组，每一组都对应 ``builder.py`` 里一条**写下来就必须被守住**
的行为——它们共同的检验标准是"报告里的数字能不能被独立算出来"：

```text
首次构建与重放    encoded / batches / written / unchanged 四个数逐条对得上
改一条 / 删一条   只碰该碰的那一条（用 spy 记录 _upsert 真的收到了哪些 id）
模式阈值          变更比例超阈值时切 full，原因进 note；没超就仍是 incremental
维度护栏          库 8 维、编码器 16 维 → EncodingError，且**一次编码都没发生**
三处降级          空白文本 / 批编码失败 / 批写入失败 → 只丢那一条；别的异常上抛
版本与备份        register + adopt；备份里的清单 version_id 与本次一致
失败条目的归宿    增量下保留上一版的条目，且不借它的名义删库里的旧记录
verify            对账能发现"库里被人删了一条"
```

全部离线、确定性、零网络：向量由本文件里的假提供方写死，
**"提供方被调用了几次"本身就是要断言的期望值**（那是本课省钱的唯一证据）。
落盘只发生在 ``tmp_path`` 上（备份那两条用例）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import pytest

from smart_research_agent.config import settings
from smart_research_agent.indexing.backup import IndexBackupStore
from smart_research_agent.indexing.builder import (
    BUILD_MODES,
    IndexBuilder,
    split_usable_records,
)
from smart_research_agent.indexing.cache import EmbeddingCache
from smart_research_agent.indexing.encoder import describe_embedding
from smart_research_agent.indexing.errors import EncodingError, IndexingError
from smart_research_agent.indexing.versioning import IndexVersionStore
from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.vectorstore.errors import BackendUnavailable, VectorError
from smart_research_agent.vectorstore.flat import FlatVectorStore
from smart_research_agent.vectorstore.types import WriteReport

#: 假提供方的维度。取 4 而不是 384：期望值能手算，向量也一眼能看完。
DIMENSION = 4

#: 三个 16 位十六进制 id（形态取自 day062 的 ``chunk_id``）.
ID_A = "0a1b2c3d4e5f6071"
ID_B = "1b2c3d4e5f607182"
ID_C = "2c3d4e5f60718293"

#: 三条记录共用的文本（改动只发生在其中一条上）。
TEXTS: dict[str, str] = {
    ID_A: "固定长度分块每 320 个字符切一刀，重叠 48 个字符。",
    ID_B: "默认返回前 5 条，相似度低于 0.35 的直接丢掉。",
    ID_C: "一次全量重建索引大约 12 万条记录，按每条 320 token 计。",
}


def vector_for(text: str, dimension: int = DIMENSION) -> list[float]:
    """由文本确定性地造一个**非零、有限**的向量（每个分量落在 [0.1, 0.9]）.

    用 sha256 而不是"字符码求和再取模"：后者的碰撞会让
    "两段不同文本应当得到不同向量"这条前提失效，而本文件好几条断言靠它。
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [((digest[index % len(digest)] % 9) + 1) / 10.0 for index in range(dimension)]


# --------------------------------------------------------------------------- #
# 假提供方与假后端（都不是真实实现，因此完全离线且确定性）
# --------------------------------------------------------------------------- #


class CountingEmbedding(EmbeddingProvider):
    """确定性编码器：把"被调用了几次"记下来，并可对特定文本抛错.

    ```text
    fail_texts  对这段文本抛 EncodingError（模拟"提供方拒绝这一条"）
    boom_texts  对这段文本抛 RuntimeError（模拟"提供方实现里有 bug"）
    ```

    两种错误的区别是本文件一组用例的核心：前者该被记进 ``failures``，
    后者必须**向上抛**（见模块 docstring 三处降级的第三条）。
    """

    def __init__(
        self,
        dimension: int = DIMENSION,
        *,
        fail_texts: tuple[str, ...] = (),
        boom_texts: tuple[str, ...] = (),
    ) -> None:
        self._dimension = dimension
        self._fail = set(fail_texts)
        self._boom = set(boom_texts)
        #: 逐条 ``embed`` 的调用次数（含 ``embed_batch`` 内部的逐条展开）
        self.embed_calls = 0
        #: ``embed_batch`` 被调用了几次（批的边界是成本的一部分）
        self.batch_calls = 0
        self.batch_sizes: list[int] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        if text in self._boom:
            raise RuntimeError(f"假提供方内部错误：{text[:12]!r}")
        if text in self._fail:
            raise EncodingError(f"假提供方拒绝编码 {text[:12]!r}：这一段读不出来")
        return vector_for(text, self._dimension)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        self.batch_sizes.append(len(texts))
        return [self.embed(text) for text in texts]


class SpyBackend(FlatVectorStore):
    """``flat`` 后端 + 一份"谁被写过、谁被删过"的台账.

    它继承真实的 ``flat`` 而不是自己写一份，是为了让"库最终是什么样"
    这件事仍然由 day064 的实现回答——本文件要断言的额外事实只有
    "``_upsert`` 收到了哪些 id"（省钱的证据）与"哪个 id 会让写入失败"。
    """

    name = "spy-flat"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        #: 每次 ``_upsert`` 收到的 id 列表（逐次批量）
        self.upserted: list[list[str]] = []
        #: 每次 ``_remove`` 收到的 id 列表（``clear()`` 也会走这里）
        self.deleted: list[list[str]] = []
        #: 让写入在这个 id 上抛 ``VectorError``（模拟"整批被拒"）
        self.fail_upsert_ids: set[str] = set()
        #: 让写入在这个 id 上抛 ``RuntimeError``（模拟真 bug）
        self.boom_upsert_ids: set[str] = set()
        #: 让后端**假装**护栏跳过这些 id（不写进库，但记进 skipped）
        self.skip_upsert_ids: set[str] = set()

    def _upsert(self, records: Any) -> Any:
        items = list(records)
        self.upserted.append([record.record_id for record in items])
        # 先扫一遍再写：这样"整批被拒"是真的什么都没写进去，
        # 逐条重试才能干净地定位到那一条（见 builder 的三处降级）。
        for record in items:
            if record.record_id in self.boom_upsert_ids:
                raise RuntimeError(f"假后端内部错误（{record.record_id}）")
            if record.record_id in self.fail_upsert_ids:
                raise VectorError(f"假后端拒绝写入 {record.record_id}：模拟整批被拒")
        if not self.skip_upsert_ids:
            return super()._upsert(items)
        dropped = [item for item in items if item.record_id in self.skip_upsert_ids]
        kept = [item for item in items if item.record_id not in self.skip_upsert_ids]
        return super()._upsert(kept).merge(WriteReport(skipped=len(dropped)))

    def _remove(self, ids: Any) -> int:
        self.deleted.append(list(ids))
        return super()._remove(list(ids))


class InMemoryOnlyBackend(FlatVectorStore):
    """声明"不能落盘"的假后端（用来验证备份被**跳过**而不是报错）."""

    name = "memory-only"
    supports_persistence = False

    def persist(self, path: str | None = None) -> str:
        raise BackendUnavailable("memory-only 后端不支持持久化")


# --------------------------------------------------------------------------- #
# 样本与装配
# --------------------------------------------------------------------------- #


def knowledge_record(
    record_id: str,
    text: str,
    *,
    retrieval_text: str | None = None,
    fingerprint: str | None = None,
    token_count: Any = None,
) -> dict[str, Any]:
    """造一条 day062 ``knowledge_records()`` 形状的记录（四个键）.

    每个可选参数都对应一个**独立**的取数口径，因此用例能把
    "内容"与"该编码的文本"分离开来单独检验。
    """
    metadata: dict[str, Any] = {}
    if retrieval_text is not None:
        metadata["retrieval_text"] = retrieval_text
    if fingerprint is not None:
        metadata["fingerprint"] = fingerprint
    if token_count is not None:
        metadata["token_count"] = token_count
    return {
        "doc_id": record_id,
        "source": "docs/demo.md",
        "text": text,
        "metadata": metadata,
    }


def base_records() -> list[dict[str, Any]]:
    """三条记录：它们是下面大多数用例的共同起点（改动只发生在其中一条上）."""
    return [
        knowledge_record(ID_A, TEXTS[ID_A], token_count=24),
        knowledge_record(ID_B, TEXTS[ID_B], token_count=22),
        knowledge_record(ID_C, TEXTS[ID_C], token_count=25),
    ]


@dataclass
class Rig:
    """一份装配好的构建器 + 它用到的全部零件（断言时直接读它们）."""

    backend: Any
    embedding: CountingEmbedding
    cache: EmbeddingCache
    versions: IndexVersionStore | None
    backups: IndexBackupStore | None
    builder: IndexBuilder


def make_rig(
    tmp_path: Any = None,
    *,
    backend: Any = None,
    embedding: CountingEmbedding | None = None,
    batch_size: int = 4,
    identity: Any = None,
    with_versions: bool = True,
    with_backups: bool = False,
) -> Rig:
    """装配一份测试用的构建器.

    三条默认值各有理由：

    ```text
    cache 与 identity 一起给定   缓存的键里含身份，两者必须来自同一个 EmbeddingIdentity
    batch_size=4                 三条记录装得下一批（批次边界不是本文件的重点）
    versions 默认给、backups 默认不给
                                 "第二版"大多数用例都需要；备份只有那几条用例才碰磁盘
    ```
    """
    resolved_embedding = embedding if embedding is not None else CountingEmbedding()
    resolved_identity = identity or describe_embedding(resolved_embedding)
    cache = EmbeddingCache(
        provider=resolved_identity.provider,
        model=resolved_identity.model,
        dimension=resolved_identity.dimension,
    )
    resolved_backend = backend if backend is not None else SpyBackend()
    versions = IndexVersionStore() if with_versions else None
    backups = None
    if with_backups:
        assert tmp_path is not None, "备份会落盘，必须给 tmp_path"
        backups = IndexBackupStore(path=str(tmp_path / "backups"))
    builder = IndexBuilder(
        resolved_backend,
        resolved_embedding,
        cache=cache,
        identity=resolved_identity,
        batch_size=batch_size,
        versions=versions,
        backups=backups,
    )
    return Rig(
        backend=resolved_backend,
        embedding=resolved_embedding,
        cache=cache,
        versions=versions,
        backups=backups,
        builder=builder,
    )


# --------------------------------------------------------------------------- #
# 首次构建与重放：四个数字
# --------------------------------------------------------------------------- #


def test_first_build_writes_everything_and_says_it_is_first_version() -> None:
    """首次构建（parent=None）→ 全量语义：全部写入、全部编码、复用率为 0."""
    rig = make_rig()

    manifest, report = rig.builder.build(base_records())

    assert report.mode == "incremental"
    assert report.seen == 3
    assert report.written == 3
    assert report.unchanged == 0
    assert report.removed == 0
    assert report.failed == 0
    assert report.encoded == 3
    assert report.batches == 1
    assert report.cache_hits == 0
    assert report.reuse_ratio == 0.0
    assert report.ok is True
    # 清单与库说的是同一件事
    assert manifest.count == rig.backend.count() == 3
    assert manifest.verify_version_id() == manifest.version_id
    assert manifest.record_ids == (ID_A, ID_B, ID_C)
    assert report.version_id == manifest.version_id
    assert report.parent_version == ""
    # "没有任何来源"必须被写出来，而不是默默假设成首版
    assert "首版" in report.note


def test_replay_same_batch_encodes_nothing() -> None:
    """**重放同一批**：written=0、unchanged=N、encoded=0 且 batches=0.

    这是本课省钱的正面证据：day064 的基线在重放时会**重新发起 N 次编码调用**
    （"库没变、钱照花"），而这里一次都不发起——而且新版本号与上一版相同，
    说明"重放"没有凭空多出一个版本。
    """
    rig = make_rig()
    first, _ = rig.builder.build(base_records())
    calls_before = rig.embedding.batch_calls
    upserts_before = len(rig.backend.upserted)

    second, report = rig.builder.build(base_records())

    assert report.written == 0
    assert report.unchanged == 3
    assert report.removed == 0
    assert report.encoded == 0
    assert report.batches == 0
    assert report.reuse_ratio == 1.0
    assert report.parent_version == first.version_id
    assert second.version_id == first.version_id
    # 一次提供方调用都没发生（连空批都没有）
    assert rig.embedding.batch_calls == calls_before
    # 版本表里仍然只有一版（重放不该产生新版本）
    assert len(rig.versions) == 1
    # 未变的条目**一条都不碰库**：upsert 一次都没被调用过
    assert len(rig.backend.upserted) == upserts_before


def test_changed_text_updates_only_that_record() -> None:
    """改一条文本 → 只编码、只写那一条（spy 断言 ``_upsert`` 收到的 id）."""
    rig = make_rig()
    rig.builder.build(base_records())
    rig.backend.upserted.clear()

    changed = base_records()
    changed[0]["text"] = "固定长度的块每 320 个字符切一刀，重叠 48 个字符。"
    _, report = rig.builder.build(changed)

    assert report.encoded == 1
    assert report.batches == 1
    assert report.written == 1
    assert report.unchanged == 2
    assert report.removed == 0
    assert report.reuse_ratio == round(2 / 3, 4)
    assert rig.backend.upserted == [[ID_A]]


def test_removed_record_is_deleted_from_the_store() -> None:
    """少给一条 → ``plan.removed`` 里是它，库里那条也没了（其余一条都不碰）."""
    rig = make_rig()
    manifest, _ = rig.builder.build(base_records())
    rig.backend.upserted.clear()

    plan = rig.builder.plan(base_records()[:2], parent=manifest)
    assert plan.removed == (ID_C,)
    assert plan.unchanged == (ID_A, ID_B)

    _, report = rig.builder.build(base_records()[:2])

    assert report.removed == 1
    assert report.written == 0
    assert report.unchanged == 2
    assert report.encoded == 0
    assert ID_C not in rig.backend.ids()
    assert rig.backend.deleted == [[ID_C]]
    assert rig.backend.upserted == []


# --------------------------------------------------------------------------- #
# 模式判定：增量不是永远更省
# --------------------------------------------------------------------------- #


def test_change_ratio_over_threshold_switches_to_full(monkeypatch: pytest.MonkeyPatch) -> None:
    """变更比例超过阈值 → 切 full，原因进 note（含"阈值"两个字）."""
    monkeypatch.setattr(settings, "indexing_full_rebuild_threshold", 0.1)
    rig = make_rig()
    first, _ = rig.builder.build(base_records())

    changed = base_records()
    changed[0]["text"] = "改写了第一条的内容。"
    manifest, report = rig.builder.build(changed)

    assert report.mode == "full"
    assert "阈值" in report.note
    assert "incremental 切到 full" in report.note
    assert report.parent_version == first.version_id
    # 全量语义：清空重写，所以"写入"是全部、而"未变"恒为 0
    assert report.written == 3
    assert report.unchanged == 0
    # 但**全量重建不等于重新编码**：只有那一条改过的文本没命中缓存
    assert report.encoded == 1
    assert report.batches == 1
    assert report.cache_hits == 2
    # 全量下的复用率报的是"这一次靠缓存免掉了多少"
    assert report.reuse_ratio == round(2 / 3, 4)
    assert manifest.count == rig.backend.count() == 3


def test_change_ratio_below_threshold_stays_incremental(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """变更比例没超阈值 → 仍然是 incremental，且 note 里不出现阈值这句话."""
    monkeypatch.setattr(settings, "indexing_full_rebuild_threshold", 0.9)
    rig = make_rig()
    rig.builder.build(base_records())

    changed = base_records()
    changed[0]["text"] = "改写了第一条的内容。"
    _, report = rig.builder.build(changed)

    assert report.mode == "incremental"
    assert "阈值" not in report.note
    assert report.written == 1
    assert report.unchanged == 2
    assert report.reuse_ratio == round(2 / 3, 4)


def test_empty_records_neither_switch_mode_nor_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """空记录集：不换挡、不报错（``plan.total == 0`` 时分母不存在）."""
    monkeypatch.setattr(settings, "indexing_full_rebuild_threshold", 0.0)
    rig = make_rig()

    manifest, report = rig.builder.build([])

    assert report.mode == "incremental"
    assert report.seen == 0
    assert report.written == 0
    assert report.unchanged == 0
    assert report.removed == 0
    assert report.failed == 0
    assert report.encoded == 0
    assert report.batches == 0
    assert manifest.count == 0
    assert rig.embedding.batch_calls == 0


def test_full_mode_reuse_ratio_is_zero_without_requests() -> None:
    """全量 + 空记录集：复用率是 0.0（"没有请求"不是除零错误）."""
    rig = make_rig()

    manifest, report = rig.builder.build([], mode="full")

    assert report.mode == "full"
    assert report.seen == 0
    assert report.encoded == 0
    assert report.batches == 0
    assert report.reuse_ratio == 0.0
    assert manifest.count == 0
    assert rig.backend.count() == 0


# --------------------------------------------------------------------------- #
# 维度护栏：必须在编码之前
# --------------------------------------------------------------------------- #


def test_dimension_guard_fires_before_any_encoding() -> None:
    """库 8 维、编码器 16 维 → ``EncodingError``，且**一次编码都没发生**."""
    backend = SpyBackend(dimension=8)
    embedding = CountingEmbedding(dimension=16)
    builder = IndexBuilder(backend, embedding)

    with pytest.raises(EncodingError) as excinfo:
        builder.build(base_records())

    message = str(excinfo.value)
    assert "8 维" in message
    assert "16 维" in message
    assert "重建" in message
    # 这是本条用例的重点：钱一分都没花
    assert embedding.batch_calls == 0
    assert embedding.embed_calls == 0
    assert backend.count() == 0
    assert backend.upserted == []


# --------------------------------------------------------------------------- #
# 三处降级：坏数据不许毁掉整批
# --------------------------------------------------------------------------- #


def test_blank_text_record_goes_to_failures_without_encoding() -> None:
    """空白文本的记录直接进 failures（reason 里带 record_id），其余照常."""
    rig = make_rig()
    records = [
        knowledge_record(ID_A, TEXTS[ID_A]),
        knowledge_record(ID_B, "   \n  "),
        knowledge_record(ID_C, TEXTS[ID_C]),
    ]

    manifest, report = rig.builder.build(records)

    assert report.seen == 3
    assert report.failed == 1
    assert report.failures[0][0] == ID_B
    assert ID_B in report.failures[0][1]
    assert "空白" in report.failures[0][1]
    assert report.encoded == 2
    assert report.batches == 1
    assert report.written == 2
    assert report.ok is False
    assert rig.backend.ids() == [ID_A, ID_C]
    assert manifest.record_ids == (ID_A, ID_C)


def test_record_without_id_goes_to_failures() -> None:
    """没有 ``doc_id`` 的记录连计划都进不去（``IndexEntry`` 会直接拒绝空 id）."""
    rig = make_rig()
    records = [knowledge_record("", TEXTS[ID_A]), knowledge_record(ID_A, TEXTS[ID_A])]

    manifest, report = rig.builder.build(records)

    assert report.seen == 2
    assert report.failed == 1
    assert report.failures[0][0] == ""
    assert "doc_id" in report.failures[0][1]
    assert report.written == 1
    assert rig.backend.ids() == [ID_A]
    assert manifest.count == 1
    # 分流函数本身也可以被单独核对
    usable, failures = split_usable_records(records)
    assert len(usable) == 1 and len(failures) == 1


def test_record_with_illegal_metadata_goes_to_failures() -> None:
    """元数据不合法（嵌套对象）→ 记录构造被拒，也只丢这一条.

    向量库的元数据只接受扁平标量（day064 的公共契约：str / int / float /
    bool / 同类型标量数组）。这条用例把它踩在**编码之后、写入之前**那一层：
    向量已经算出来了，但这条记录进不了库——所以它必需进 failures，
    否则库里少一条而报告上什么都没有。
    """
    rig = make_rig()
    records = [knowledge_record(ID_A, TEXTS[ID_A]), knowledge_record(ID_B, TEXTS[ID_B])]
    records[1]["metadata"]["nested"] = {"a": 1}

    manifest, report = rig.builder.build(records)

    assert report.failed == 1
    assert report.failures[0][0] == ID_B
    assert "记录构造被拒" in report.failures[0][1]
    assert report.written == 1
    assert report.encoded == 2
    assert rig.backend.ids() == [ID_A]
    assert manifest.record_ids == (ID_A,)


def test_encoding_error_falls_back_to_per_record_retry() -> None:
    """一批编码失败 → 逐条重试：只有真正坏的那条进 failures，其余照常写入."""
    embedding = CountingEmbedding(fail_texts=(TEXTS[ID_B],))
    rig = make_rig(embedding=embedding)

    manifest, report = rig.builder.build(base_records())

    assert report.failed == 1
    assert report.failures[0][0] == ID_B
    assert report.written == 2
    assert report.encoded == 2
    assert rig.backend.ids() == [ID_A, ID_C]
    assert manifest.record_ids == (ID_A, ID_C)
    assert rig.builder.verify(manifest)["ok"] is True
    # 1 次整批 + 3 次逐条（失败的那次不计 batches，因为它没有产出报告）
    assert embedding.batch_calls == 4
    assert report.batches == 2
    assert "逐条重试" in report.note


def test_upsert_error_falls_back_to_per_record_retry() -> None:
    """一次 upsert 抛 ``VectorStoreError`` → 逐条重试：故障被定位到具体 id."""
    rig = make_rig()
    rig.backend.fail_upsert_ids = {ID_B}

    manifest, report = rig.builder.build(base_records())

    assert report.failed == 1
    assert report.failures[0][0] == ID_B
    assert "写入被拒" in report.failures[0][1]
    assert report.written == 2
    assert rig.backend.ids() == [ID_A, ID_C]
    assert manifest.record_ids == (ID_A, ID_C)
    assert rig.builder.verify(manifest)["ok"] is True
    # 先整批（三条一起），失败之后逐条——**包括那条坏的**（它要再响一次，
    # 否则"到底哪一条写不进去"就只能靠猜）
    assert rig.backend.upserted == [[ID_A, ID_B, ID_C], [ID_A], [ID_B], [ID_C]]
    assert "逐条重试" in report.note


def test_backend_guardrail_skips_are_noted_and_left_to_verify() -> None:
    """后端护栏跳过的条数进 note 而**不进 failures**（编不出逐条 id）.

    代价写在明面上：清单仍按输入记，而库里少若干条——`verify()` 会把这件
    事故指出来，这正是一个"清单与库对不上"的检查点存在的理由。
    """
    rig = make_rig()
    rig.backend.skip_upsert_ids = {ID_B}

    manifest, report = rig.builder.build(base_records())

    assert report.written == 2
    assert report.failed == 0
    assert "护栏跳过" in report.note
    assert manifest.count == 3
    problems = rig.builder.verify(manifest)["problems"]
    assert any(ID_B in problem for problem in problems)


def test_delete_mismatch_is_reported_in_the_note() -> None:
    """请求删 N 条、实际删掉 0 条 → note 里说清差额（库与清单已经不一致）."""
    rig = make_rig()
    rig.builder.build(base_records())
    # 有人绕过构建流程手工从库里删了一条（清单里仍然记着它）
    rig.backend.delete(ids=[ID_C])

    _, report = rig.builder.build(base_records()[:2])

    assert report.removed == 0
    assert report.unchanged == 2
    assert "实际删掉 0 条" in report.note


def test_unexpected_upsert_error_propagates() -> None:
    """非 ``VectorStoreError`` 的异常**向上抛**，不记进 failures."""
    rig = make_rig()
    rig.backend.boom_upsert_ids = {ID_B}

    with pytest.raises(RuntimeError):
        rig.builder.build(base_records())

    assert rig.backend.count() == 0


def test_unexpected_encoding_error_propagates() -> None:
    """提供方抛 ``RuntimeError``（实现里有 bug）同样上抛，不许被记成 failed=1."""
    embedding = CountingEmbedding(boom_texts=(TEXTS[ID_B],))
    rig = make_rig(embedding=embedding)

    with pytest.raises(RuntimeError):
        rig.builder.build(base_records())

    assert rig.backend.count() == 0


def test_failed_record_is_protected_by_the_previous_manifest() -> None:
    """增量下失败的条目：保留上一版清单里的那一条，且**不删库里已有的记录**."""
    rig = make_rig()
    rig.builder.build(base_records())

    rig.backend.fail_upsert_ids = {ID_A}
    changed = base_records()
    changed[0]["text"] = "改写了第一条的内容（这一次写不进去）。"
    manifest, report = rig.builder.build(changed)

    assert report.failed == 1
    assert report.failures[0][0] == ID_A
    assert report.written == 0
    assert report.removed == 0
    assert ID_A in rig.backend.ids()
    assert manifest.count == 3
    assert manifest.record_ids == (ID_A, ID_B, ID_C)
    assert rig.builder.verify(manifest)["ok"] is True


# --------------------------------------------------------------------------- #
# 版本表与备份
# --------------------------------------------------------------------------- #


def test_versions_register_and_adopt() -> None:
    """构建成功 → 登记 + 采纳；第二版的血缘接在上一版之后."""
    rig = make_rig()

    manifest, report = rig.builder.build(base_records())

    assert len(rig.versions) == 1
    assert rig.versions.current == report.version_id
    assert rig.versions.latest().version_id == report.version_id
    assert rig.versions.get(report.version_id).summary_line() == manifest.summary_line()

    changed = base_records()
    changed[0]["text"] = "改写了第一条的内容。"
    second, report2 = rig.builder.build(changed)

    assert report2.parent_version == report.version_id
    assert len(rig.versions) == 2
    assert rig.versions.current == report2.version_id
    assert rig.versions.latest().version_id == second.version_id
    assert [item.version_id for item in rig.versions.history()] == [
        report.version_id,
        second.version_id,
    ]


def test_backup_snapshots_the_store_with_the_same_version_id(tmp_path: Any) -> None:
    """``backup=True`` 且给了备份表 → 备份 +1，快照里的清单与本次同版."""
    store = FlatVectorStore(path=str(tmp_path / "store.json"))
    rig = make_rig(tmp_path=tmp_path, backend=store, with_backups=True)

    manifest, report = rig.builder.build(base_records(), backup=True, reason="手工触发")

    records = rig.backups.list()
    assert len(records) == 1
    assert records[0].version_id == report.version_id == manifest.version_id
    assert records[0].reason == "手工触发"
    assert records[0].files == ("store.json",)
    assert records[0].dimension == DIMENSION
    assert rig.backups.latest().backup_id == records[0].backup_id
    assert rig.backups.read_manifest(records[0].backup_id).version_id == report.version_id
    assert (tmp_path / "store.json").is_file()


def test_backup_is_skipped_when_the_backend_cannot_persist(tmp_path: Any) -> None:
    """后端不支持持久化 → 不报错、不备份，但在 note 里说明（没有文件可备份）."""
    rig = make_rig(tmp_path=tmp_path, backend=InMemoryOnlyBackend(), with_backups=True)

    manifest, report = rig.builder.build(base_records(), backup=True)

    assert rig.backups.list() == []
    assert "备份" in report.note
    assert "不支持持久化" in report.note
    assert report.failed == 0
    assert manifest.count == 3


def test_backup_is_skipped_when_persist_has_nowhere_to_write(tmp_path: Any) -> None:
    """声明支持持久化、但没有落盘位置 → ``persist()`` 抛错也被接住（另一条分支）."""
    rig = make_rig(tmp_path=tmp_path, backend=FlatVectorStore(), with_backups=True)

    _, report = rig.builder.build(base_records(), backup=True)

    assert rig.backups.list() == []
    assert "无法落盘" in report.note


def test_backup_without_a_backup_store_only_notes_it() -> None:
    """``backup=True`` 但没接备份表 → 只留一句话，构建照常成功."""
    rig = make_rig(with_backups=False)

    _, report = rig.builder.build(base_records(), backup=True)

    assert "没有接备份表" in report.note
    assert report.written == 3
    assert report.failed == 0


# --------------------------------------------------------------------------- #
# 全量路径：缓存照用
# --------------------------------------------------------------------------- #


def test_full_rebuild_still_reuses_the_cache() -> None:
    """显式全量重建：库被重写一遍，但**一条都没重新编码**（缓存全命中）."""
    rig = make_rig()
    rig.builder.build(base_records())
    calls_before = rig.embedding.batch_calls

    manifest, report = rig.builder.build(base_records(), mode="full")

    assert report.mode == "full"
    assert report.written == 3
    assert report.unchanged == 0
    assert report.encoded == 0
    assert report.batches == 0
    assert report.cache_hits == 3
    assert report.reuse_ratio == 1.0
    assert rig.embedding.batch_calls == calls_before
    assert rig.backend.count() == 3
    assert manifest.count == 3


def test_unknown_mode_is_rejected_before_touching_the_store() -> None:
    """``full`` / ``incremental`` 之外的值无条件报错（猜错的代价是一次全量重建）."""
    rig = make_rig()

    with pytest.raises(IndexingError) as excinfo:
        rig.builder.build(base_records(), mode="rebuild")

    assert BUILD_MODES == ("full", "incremental")
    assert "rebuild" in str(excinfo.value)
    assert "full / incremental" in str(excinfo.value)
    assert rig.backend.count() == 0
    assert rig.embedding.batch_calls == 0


# --------------------------------------------------------------------------- #
# parent 的解析顺序与两个只读属性
# --------------------------------------------------------------------------- #


def test_plan_prefers_explicit_parent_over_the_version_store() -> None:
    """``parent`` 的解析顺序：显式传入 > 版本表最近登记 > 首版."""
    rig = make_rig()
    manifest, _ = rig.builder.build(base_records())

    # 另一条装配线构建出的"只含一条"的清单，把它登记进同一个版本表当干扰项
    other = make_rig().builder.build(base_records()[:1])[0]
    assert other.version_id != manifest.version_id
    rig.versions.register(other)

    explicit = rig.builder.plan(base_records(), parent=manifest)
    assert explicit.unchanged == (ID_A, ID_B, ID_C)
    assert explicit.added == ()

    from_store = rig.builder.plan(base_records())
    assert from_store.unchanged == (ID_A,)
    assert from_store.added == (ID_B, ID_C)

    fresh = make_rig(with_versions=False).builder.plan(base_records())
    assert fresh.added == (ID_A, ID_B, ID_C)
    assert "首次构建" in fresh.reason


def test_cache_and_encoder_share_one_identity() -> None:
    """缓存与编码器用的是**同一个** ``EmbeddingIdentity``（键里含的就是它）."""
    rig = make_rig()

    assert rig.builder.cache is rig.builder.encoder.cache
    assert rig.builder.identity == rig.builder.encoder.identity
    assert rig.builder.cache.provider == "CountingEmbedding"
    assert rig.builder.cache.model == "default"
    assert rig.builder.cache.dimension == DIMENSION
    assert rig.builder.backend is rig.backend
    assert rig.builder.embedding is rig.embedding
    assert rig.builder.identity.key == rig.builder.cache.identity_key


# --------------------------------------------------------------------------- #
# verify：对账，不修复
# --------------------------------------------------------------------------- #


def test_verify_reports_a_manually_deleted_record() -> None:
    """正常 → ``ok``；人为删掉库里一条 → ``ok is False`` 且 problems 里含该 id."""
    rig = make_rig()
    manifest, _ = rig.builder.build(base_records())

    assert rig.builder.verify(manifest)["ok"] is True
    # 不给清单时按库现算一份（那里必然对得上：它是从库算出来的）
    assert rig.builder.verify()["ok"] is True

    rig.backend.delete(ids=[ID_B])

    result = rig.builder.verify(manifest)
    assert result["ok"] is False
    assert ID_B in result["checks"]["missing_in_store"]
    assert any(ID_B in problem for problem in result["problems"])
    # 不做任何修复：那条记录不该被"顺手补回来"
    assert rig.backend.get(ID_B) is None
