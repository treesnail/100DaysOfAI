"""索引构建器：把"一批记录"变成"一版索引 + 一份账"（M6-D4）.

到这一步为止，零件都齐了，但**没有人在装配**：

```text
planner     算出"哪些块要重算"          —— 一份计划
encoder     把"要重算的文本"变成向量      —— 一批向量 + 一笔编码账
cache       让同一段文本只算一次          —— 一张表
manifest    把"这一版长什么样"记成证据     —— 一份清单
versioning  让"回滚到上一版"有确定答案     —— 一张版本表
backup      让"还能回到上一版"能动手      —— 一份快照
vectorstore 真正存下向量与原文            —— 一个库
```

本模块是那条装配线，也是**唯一一处同时碰"库"与"账"的地方**。它的产物是两个
东西：一份 ``IndexManifest``（这一版是什么）与一份 ``IndexingReport``
（这一次花了什么）。

## 为什么本模块值得单独存在

把这些步骤写进一个 ``pipeline.ingest()`` 里看起来更省事，但会因为一件事而失败：
**增量构建的每一步都需要回答"谁说了算"，而这些答案彼此不同**：

```text
要不要重算     planner 的差集说了算（不是"库说了算"）
要不要编码     EncodeReport 的 cache_hits 说了算（不是"计划说了算"）
库里该有什么   清单说了算（不是"这次输入说了算"：未变的条目不进输入的那部分）
```

把三者揉在一起，最先坏掉的是**报告**：``written`` / ``unchanged`` / ``encoded``
会互相污染，于是"这次省了多少"永远算不出来。装配线单独一层，
才能把每个数字的来源写清楚（见 ``IndexingReport`` 的逐字段说明）。

## 五件必须在此处发生的事（每一件都有它的位置理由）

```text
1. 维度护栏      在最前面 —— 一次编码都不该白花（见 _guard_dimension）
2. 模式判定      plan 之后、编码之前 —— 它决定"要送多少条去编码"
3. 三处降级      空白文本 / 批编码失败 / 批写入失败（见 docstring 下文）
4. 清单与版本    写入之后 —— 清单描述的是"库**现在**是什么"
5. 备份          也是写入之后 —— 快照要与它配套的那份清单成对
```

## 增量不是永远更省：本模块的核心决策

直觉上"只重算变了的块"永远比"清空重写"便宜，工程上不是：

```text
90% 的块都变了 → 逐条 upsert 90% 的条目 + 逐条 delete 若干 + 维护差集
               → 开销高于"clear() + 一次性写回全部"
```

因此本模块有一条**自动换挡**：增量模式下若

```text
plan.changed / plan.total > settings.indexing_full_rebuild_threshold
```

就从 incremental 切到 full，并把原因写进 ``report.note``。
这个判断只有装配层做得了——planner 看不到"写库要花多少"，
后端也不知道"这批数据相对上一版变了多少"。**改走全量之后，缓存仍然照用**：
全量重建不等于重新编码，"把库重写一遍"与"把向量重算一遍"是两件事。

``plan.total == 0``（空记录集）时既不换挡也不报错：没有数据时两种模式等价。

## 三处降级：批量操作会把"一条坏数据"放大成"整批失败"

这是本模块第二重要的决策。三处降级的共同形状是"**先整批、失败再逐条**"：

```text
文本为空/全空白    直接进 failures，**一小段编码都不发起**（它本来就不该进编码器）
一批编码抛错       对该批逐条重试，真正失败的那几条进 failures，其余照常写入
一次 upsert 抛错    逐条重试，故障被定位到具体 record_id，其余照常写入
```

为什么偏向"逐条重试"而不是"整批重来"：**代价只在出错时付出，而收益是
把故障定位到具体 id**。反过来（整批失败就整批放弃）会让一份 1000 条的批次
里 1 条坏数据带走 999 条好数据，而报告上只写着"失败"。
注意逐条重试是**有界的**：每一条最多被重试一次，不存在重试风暴。

第三条纪律同样重要：**非 ``IndexingError``、非 ``VectorStoreError`` 的异常
一律向上抛**（不记进 ``failures``）。一个宽泛的 ``except Exception`` 会把
"后端实现里有个 ``TypeError``"这种真 bug 记成一行 ``failed=1``，
于是它永远不会被修——**报告只能收留我们认识的失败。**

## 全量 / 增量下四个数字的口径不同（**必须写在明面上**）

```text
written      两模式一致：本次真正写进库的条数（added + updated）
unchanged    增量：plan.unchanged 的条数（这些条目**一条都没碰库**）
             全量：恒为 0 —— clear() 之后库里没有"没动过"的东西
removed      增量：delete() **实际**删掉的条数（诚实于返回值，见 base.delete）
             全量：plan.removed 的条数（它们是 clear() 顺带删掉的）
reuse_ratio  增量：plan.reuse_ratio（这一版相对上一版有多少条完全没变）
             全量：EncodeReport.hit_ratio（**这一次**请求的文本里有多少条靠缓存免掉）
```

最后一行的理由是 ``types.IndexPlan.reuse_ratio`` 的说明里那句：全量重建时
plan 的复用率没有意义——库里已经清空了，"相对上一版没变"不再是一个
能描述本次成本的事实。全量模式真正决定成本的是缓存命中率，所以报它。

## 失败条目与清单的关系（一条容易被绕过去的规则）

清单要回答"库里现在有什么"，因此**失败的条目不能进清单**（它没写进去，
写进去就会让 ``verify`` 报"清单里有、库里没有"）。但"失败"分两种：

```text
这一条在上一版清单里      → 库里那条旧记录还在（我们没碰它，也没删它）
                          → **保留上一版的条目**（它仍然准确地描述了库）
                          → 它的 id 从待删列表里摘出来：坏数据不该毁掉整批，
                            也不该借它的名义把库里已有的东西删掉
这一条不在上一版清单里     → 它本来就不该在库里，直接缺席
```

全量模式下没有这条保护：``clear()`` 已经把旧记录清掉了，保留它的条目
会让清单与库**立刻**不一致。这个不对称是刻意的，也是两种模式"谁说了算"
不同的又一处体现。

## 放弃了什么（代价写在明面上）

| 放弃的东西 | 代价 | 为什么可以接受 |
|-----------|------|---------------|
| 一次构建的原子性 | 写入中途失败会留下部分新数据 | 清单只在成功路径上生成，verify 会当场指出 |
| 逐条重试的代价上限 | 最坏情况退化成 N 次单条调用 | 只在**出错**时才走这条路，且每条约一次 |
| 失败条目的精确成本 | 整批失败时 encoded 偏低（那批的账丢了） | 报告里用 note 说明"已逐条重试" |
| 自动修复脏状态 | 清单与库对不上时不"尽力而为" | 那是 ``verify`` 的结论，不该由构建路径顺手掩盖 |
| 并发构建 | 单进程使用，没有锁 | 与 day064 的向量库保持同一假设 |

## 谁依赖它

```text
indexing.pipeline（装配）  IndexingPipeline 用它把"记录 / 文档"变成索引
/indexing/build 端点       直接返回 IndexingReport.to_dict()
scripts / 演示脚本         build_from_records → report.summary_line()
tests/                     全部离线：假提供方把向量写死，假后端把失败点写死
```
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from smart_research_agent.config import settings
from smart_research_agent.indexing.backup import IndexBackupStore, utc_now_iso
from smart_research_agent.indexing.cache import EmbeddingCache
from smart_research_agent.indexing.encoder import BatchEncoder
from smart_research_agent.indexing.errors import EncodingError, IndexingError
from smart_research_agent.indexing.manifest import (
    build_manifest,
    manifest_from_store,
    verify_index,
)
from smart_research_agent.indexing.planner import entry_from_record, plan_index
from smart_research_agent.indexing.types import (
    EmbeddingIdentity,
    IndexingReport,
    IndexManifest,
    IndexPlan,
)
from smart_research_agent.indexing.versioning import IndexVersionStore
from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.vectorstore.base import VectorBackend
from smart_research_agent.vectorstore.errors import VectorStoreError
from smart_research_agent.vectorstore.pipeline import (
    embedding_text,
    record_from_knowledge,
    record_id_of,
)
from smart_research_agent.vectorstore.types import VectorRecord, WriteReport

#: 全量重建：``clear()`` 之后把这一批**全部**写回去（缓存照用）。
BUILD_MODE_FULL = "full"

#: 增量更新：只编码与写入 ``plan.to_encode``，未变的条目一条都不碰库。
BUILD_MODE_INCREMENTAL = "incremental"

#: 合法的构建模式（**顺序就是错误信息里的顺序**）。
BUILD_MODES: tuple[str, ...] = (BUILD_MODE_FULL, BUILD_MODE_INCREMENTAL)

#: ``parent`` 的三个来源（报告里的 ``note`` 与调试都按这三个值说话）.
PARENT_EXPLICIT = "explicit"
PARENT_VERSIONS = "versions"
PARENT_NONE = "none"


def _brief(value: Any, limit: int = 80) -> str:
    """把一段可能很长的文本压成一行短句（``note`` 里只放摘要）.

    ``note`` 是一句给人读的解释，而异常消息可能有三四百字（本项目的报错
    都带着"下一步做什么"）。整段塞进 ``note`` 会把 ``summary_line``
    挤成一段读不完的话——**报告要能被人读完，细节由异常与日志承担。**
    """
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit] + "…"


def split_usable_records(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """把记录分成"能进计划的"与"当场就判失败的"两份.

    唯一的分流条件是 **id 是否可用**（``doc_id`` 缺失 / ``None`` / 纯空白）。
    理由与 day064 的 ``record_id_of`` 逐字相同：id 是更新与删除的唯一依据，
    没有 id 的记录**连进计划都进不去**——``IndexEntry`` 会在构造时直接拒绝
    空 ``record_id``。与其让它把整批计划炸掉，不如在这里就把它记成失败。

    文本空白**不在这里分流**：那一条只对"要送进编码器的记录"才有意义
    （未变的条目根本不经过编码器，见 ``build`` 的说明），
    因此它由 ``build`` 在确定目标集合之后再判。
    """
    usable: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for record in records:
        record_id = record_id_of(record)
        if not record_id:
            failures.append(
                (
                    record_id,
                    "失败：记录缺少 doc_id（知识库记录的唯一键就是 chunk_id）。"
                    "本模块不会替它编一个 id——自动拼出来的 id 在下一批就对不上，"
                    "同一条内容会在库里出现两次，而那比丢掉这一条更难查。",
                )
            )
            continue
        usable.append(record)
    return usable, failures


class IndexBuilder:
    """索引构建器：一个后端 + 一个编码器 + 三个可选的账本（见模块 docstring）.

    四个依赖全部由构造注入，因此本类**完全离线可测**：注入一个把向量写死的
    提供方与一个内存后端，``encoded`` / ``batches`` / ``written`` 这些数字
    本身就是可断言的期望值（它们是本课"省了多少"的唯一证据）。

    ``versions`` / ``backups`` 允许为空：构建一份能用的索引**不需要**版本表，
    版本表与备份是"可运维性"的投入。反过来，只有两样都给了，
    "回滚"这条路径才是完整的（见 ``versioning`` / ``backup`` 的 docstring）。
    """

    def __init__(
        self,
        backend: VectorBackend,
        embedding: EmbeddingProvider,
        *,
        cache: EmbeddingCache | None = None,
        identity: EmbeddingIdentity | None = None,
        batch_size: int | None = None,
        versions: IndexVersionStore | None = None,
        backups: IndexBackupStore | None = None,
        clock: Callable[[], str] = utc_now_iso,
    ) -> None:
        """装配构建器；四个"账本"依赖都是可选的.

        ```text
        cache       缺省由 BatchEncoder 新建一份**带本编码器身份**的空缓存
        identity    缺省 describe_embedding(embedding)；显式传入即覆盖（见 encoder）
        batch_size  缺省取 settings.indexing_batch_size（<= 0 由 encoder 报错）
        versions    给了就登记 + 采纳；没给就只构建、不记账
        backups     与 backup=True 配合使用；没给时 backup=True 只会在 note 里说明
        clock       注入的时钟：缺省 utc_now_iso；测试用它断言 created_at
        ```

        ``cache`` 与 ``identity`` 必须**一起考虑**：缓存键里含身份
        （见 ``types.vector_key``），因此"给了一份与提供方不匹配的缓存"
        会让键错位。最省心的用法是显式传 ``identity=`` 并让缓存由本类新建；
        若自己造缓存，请用同一个 ``EmbeddingIdentity`` 的三个字段
        （``indexing.pipeline.default_indexing_pipeline`` 就是这么做的）。
        """
        self._backend = backend
        self._embedding = embedding
        self._encoder = BatchEncoder(
            embedding,
            batch_size=batch_size,
            cache=cache,
            identity=identity,
        )
        self._versions = versions
        self._backups = backups
        self._clock = clock

    # ------------------------------------------------------------------ 只读属性

    @property
    def identity(self) -> EmbeddingIdentity:
        """本构建器的编码器身份（进向量键、清单与报告）."""
        return self._encoder.identity

    @property
    def encoder(self) -> BatchEncoder:
        """批量编码器（``encoded`` / ``batches`` 的来源就在它上面）."""
        return self._encoder

    @property
    def cache(self) -> EmbeddingCache:
        """向量缓存（命中/未命中计数就在这里，``report.cache_hits`` 读它）."""
        return self._encoder.cache

    @property
    def backend(self) -> VectorBackend:
        """目标向量库（``pipeline.search`` 需要它来复用 day064 的检索路径）.

        它不在 D.2 列出的三个属性里，但装配层必须拿得到：
        "编码哪段文本"与"检索用什么口径"都只能有一处定义
        （见 ``indexing.pipeline`` 的复用说明），而那处定义需要后端与提供方。
        """
        return self._backend

    @property
    def embedding(self) -> EmbeddingProvider:
        """编码提供方（与 ``backend`` 一起交给 ``VectorIngestPipeline``）."""
        return self._embedding

    # ------------------------------------------------------------------ 计划

    def plan(
        self,
        records: Sequence[dict[str, Any]],
        *,
        parent: IndexManifest | None = None,
    ) -> IndexPlan:
        """算这一批记录相对上一版的差集（**不编码、不写库**）.

        它存在的意义是"先看计划再决定要不要动手"：``plan.summary_line()``
        能回答"这次要重算几条"，而那个数字决定了这次构建的价格。
        ``build`` 走的是同一个函数，因此计划里看到什么、构建就做什么。

        ``parent`` 的**解析顺序**（本模块对外的一致口径，``build`` 也用它）：

        ```text
        显式传入 parent        → 用它（调用方最清楚"上一版"是哪一版）
        versions.latest()      → 版本表里最近登记的一版
        None                   → 视作首版（没有任何来源时**明确**当首版，不默默假设）
        ```

        "显式传入优先于版本表"是刻意的：版本表的 ``latest()`` 是"最近**登记**的"，
        而调用方手上那份可能是"最近一次**成功**的"——两者在构建失败过之后
        会不同（见 ``versioning`` 里 v3 那个反例）。

        id 不可用的记录会先被摘掉（见 ``split_usable_records``），
        因此计划里的条数可能少于入参：**报告里的 ``seen`` 数的是入参**，
        两者的差额就是"当场判失败"的那部分。
        """
        resolved, _ = self._resolve_parent(parent)
        usable, _ = split_usable_records(list(records))
        return plan_index(resolved, usable, self.identity)

    # ------------------------------------------------------------------ 构建

    def build(
        self,
        records: Sequence[dict[str, Any]],
        *,
        mode: str | None = None,
        parent: IndexManifest | None = None,
        reason: str = "",
        backup: bool = False,
    ) -> tuple[IndexManifest, IndexingReport]:
        """构建一版索引，返回 ``(清单, 报告)``（语义见模块 docstring）.

        九个步骤的**顺序不能换**，每一步都有一条"为什么在这个位置"：

        ```text
        1. 维度护栏          在任何编码之前 —— 别白花一次编码的钱（见 _guard_dimension）
        2. 模式校验          "full / incremental" 之外的值无条件报错
        3. 解析 parent       显式 > 版本表 > 首版（与 plan 同一口径）
        4. 分流记录          id 不可用的当场进 failures，不进计划
        5. 算差集            决定"哪些块要重算"
        6. 模式换挡          变更比例超阈值时从 incremental 切到 full，并把原因写进 note
        7. 编码 + 写入       增量只碰 to_encode；全量 clear() 后重写全部（缓存照用）
        8. 删除              只剩"输入里确实没有了的"那些 id（失败的 id 被摘出来）
        9. 清单 / 版本 / 备份 清单描述库**现在**是什么；版本表登记并采纳；可选地备份
        ```

        第 7 步与第 8 步之间有一个不对称：``plan.removed`` 是"上一版有、
        这一版没有"，但其中**失败的那些不是"没有了"，而是"这次读不进来"**。
        把它们从待删列表里摘出来，是为了不借一条坏数据的名义删掉库里
        已经有的东西（见模块 docstring 的"失败条目与清单的关系"）。

        ``reason`` 只被备份用（写进 ``BackupRecord.reason``，回答"这份快照是
        因为什么建的"），它**不进清单**：清单的 ``metadata`` 不参与版本号，
        但把一次构建的临时理由写进版本记录会让版本表变成日志。
        """
        records = list(records)
        seen = len(records)

        # 1. 维度护栏：放在最前面，见 _guard_dimension。
        self._guard_dimension()
        # 2. 模式校验：拼错的模式在任何数据上都是错的，先报出来。
        requested_mode = self._validate_mode(mode)
        # 3. 解析上一版。
        resolved_parent, parent_source = self._resolve_parent(parent)
        # 4. 分流：id 不可用的记录不进计划。
        usable, failures = split_usable_records(records)
        record_map = {record_id_of(record): record for record in usable}
        # 5. 差集。
        plan = plan_index(resolved_parent, usable, self.identity)
        # 6. 换挡。
        mode, mode_note = self._apply_rebuild_threshold(requested_mode, plan, resolved_parent)

        notes: list[str] = []
        if parent_source == PARENT_NONE:
            notes.append(
                "没有上一版清单（未显式传入 parent，版本表里也没有已登记的版本）："
                "本次按**首版**构建，parent_version 为空"
            )
        if mode_note:
            notes.append(mode_note)

        entries_by_id = {
            record_id: entry_from_record(record, self.identity)
            for record_id, record in record_map.items()
        }

        # 7a. 确定"要送进编码器"的 id：增量只看 to_encode；全量要重写全部。
        target_ids = (
            tuple(sorted(entries_by_id))
            if mode == BUILD_MODE_FULL
            else tuple(plan.to_encode)
        )
        pending: list[tuple[str, str]] = []
        for record_id in target_ids:
            text = embedding_text(record_map[record_id])
            if not text.strip():
                failures.append(
                    (
                        record_id,
                        f"失败：记录 {record_id!r} 的文本为空或全空白"
                        "（text 与 retrieval_text 都没有内容），未送进编码器。"
                        "编码空文本只会得到零向量，而零向量没有方向"
                        "（余弦相似度对它是 0/0）——请在上游修数据。",
                    )
                )
                continue
            pending.append((record_id, text))

        # 7b. 编码（整批优先，失败逐条重试）。
        vectors, cache_hits, encoded, batches, encode_notes = self._encode(pending, failures)
        notes.extend(encode_notes)

        # 7c. 写入（增量只写 to_encode；全量先清空再全写）。
        prepared: list[VectorRecord] = []
        for record_id, _ in pending:
            vector = vectors.get(record_id)
            if vector is None:
                continue  # 编码失败，已在 failures 里
            try:
                prepared.append(
                    record_from_knowledge(
                        record_map[record_id], vector, metric=self._backend.metric
                    )
                )
            except VectorStoreError as exc:
                # RecordError 也属于这一族（id 超长、元数据类型越界、零向量…）：
                # 它们的共同点是"构造这条记录时就知道它进不去"，归宿与写入失败相同。
                failures.append((record_id, f"失败：记录构造被拒：{exc}"))

        # 全量路径在这里清库，**恰好在写入之前**：清早了会让"编码失败"
        # 变成"库先被清空、再发现写不回去"；清晚了就成了增量。
        # ``clear()`` 保留后端的度量与维度设定（见 base.clear），
        # 因此紧跟其后的写入不会因为"维度未定"而改变行为。
        if mode == BUILD_MODE_FULL:
            self._backend.clear()
        write, write_notes = self._write(prepared, failures)
        notes.extend(write_notes)
        # 8. 删除：把"失败的 id"从待删列表里摘出来（见方法 docstring 与模块说明）。
        failed_ids = {record_id for record_id, _ in failures}
        prior_entries = resolved_parent.entry_map() if resolved_parent is not None else {}
        protected = {record_id for record_id in failed_ids if record_id in prior_entries}
        removed, remove_notes = self._remove_stale(plan, mode, protected)
        notes.extend(remove_notes)

        # 9. 清单：库里现在有什么（失败的不进清单；增量下保留被保护的上一版条目）。
        entries = [
            entry
            for record_id, entry in sorted(entries_by_id.items())
            if record_id not in failed_ids
        ]
        if mode == BUILD_MODE_INCREMENTAL:
            entries.extend(prior_entries[record_id] for record_id in sorted(protected))

        manifest = build_manifest(
            entries,
            identity=self.identity,
            metric=self._backend.metric,
            backend=self._backend.name,
            parent_version=resolved_parent.version_id if resolved_parent is not None else "",
            created_at=self._clock(),
        )

        if self._versions is not None:
            # 登记与采纳分两步（见 versioning）：登记是"存在过"，采纳是"生效了"。
            # 构建成功才走到这里，因此本次构建是**最后一步失败也不留半版**的。
            self._versions.register(manifest)
            self._versions.adopt(manifest.version_id)

        if backup:
            note = self._maybe_backup(manifest, reason)
            if note:
                notes.append(note)

        report = IndexingReport(
            mode=mode,
            version_id=manifest.version_id,
            parent_version=manifest.parent_version,
            backend=self._backend.name,
            metric=self._backend.metric,
            dimension=self.identity.dimension,
            provider=self.identity.provider,
            seen=seen,
            cache_hits=cache_hits,
            encoded=encoded,
            batches=batches,
            written=write.added + write.updated,
            unchanged=len(plan.unchanged) if mode == BUILD_MODE_INCREMENTAL else 0,
            removed=removed,
            failed=len(failures),
            reuse_ratio=self._reuse_ratio(mode, plan, cache_hits, len(pending)),
            failures=tuple(failures),
            note="；".join(notes),
        )
        return manifest, report

    def verify(self, manifest: IndexManifest | None = None) -> dict[str, Any]:
        """把清单与库对账，返回 ``{"ok", "checks", "problems"}``（**不做任何修复**）.

        ``manifest`` 为空时用 ``manifest_from_store`` 现算一份：那是"清单丢了、
        库还在"时的补救路径（见 ``manifest`` 模块）。注意现算出来的清单是
        **"从现在起"的基线**，不是历史的那一版——所以"库里被人删了一条"
        这种问题只有拿**原来那份清单**来对账才看得出来
        （``verify()`` 无参调用必然对得上，因为清单是从库现算的）。

        "不做任何修复"是刻意的：本方法的名字是 verify，不是 repair。
        顺手补一条缺失的记录会让"库与清单不一致"这件事**下一次也查不出来**
        ——它的成因（半途失败的构建、人工改库）才是要修的东西。
        """
        target = manifest
        if target is None:
            target = manifest_from_store(
                self._backend, identity=self.identity, created_at=self._clock()
            )
        return verify_index(target, self._backend)

    # ------------------------------------------------------------------ 内部：判定

    def _guard_dimension(self) -> None:
        """维度护栏：库与编码器的维度不一致时**立刻**抛 ``EncodingError``.

        这是本模块唯一一条"宁可什么都不做"的检查，理由很直接：
        维度不同意味着库里已有的向量与当前编码器算出来的不是同一套，
        混着写会让余弦相似度的排序整体失去意义（而它**不报错**，
        只表现为检索变差）。此时**一次编码都不该发生**——
        编码是要花钱的，花完再拦等于把一次构建的成本白扔掉。

        消息里必须给出两条出路（换回旧编码器 / 用当前编码器重建），
        因为"维度不符"这句话本身不指向任何动作。
        """
        identity = self.identity
        if self._backend.has_dimension and self._backend.dimension != identity.dimension:
            raise EncodingError(
                f"维度护栏：库是 {self._backend.dimension} 维、编码器是 "
                f"{identity.dimension} 维（{identity.summary_line()}）。"
                "换过 embedding 提供方就必须重建库，不要把两种向量混着写——"
                "维度不同的向量混在一个库里，余弦相似度会整体失去意义，"
                "而检索不会报任何错，只会给出另一批结果。"
                "出路：用当前编码器重建（build(..., mode='full')），"
                "或换回建库时那个编码器。"
            )

    def _validate_mode(self, mode: str | None) -> str:
        """解析并校验构建模式：缺省取 ``settings.indexing_mode``，其它值报错.

        **不做大小写与别名的宽容**（"Full" / "rebuild" 一律拒绝）：
        模式决定的是"要不要清空整个库"，一个靠猜解析出来的模式
        一旦猜错，代价是一次全量重建；宁可让调用方改一个字符串。
        """
        raw = settings.indexing_mode if mode is None else mode
        resolved = str(raw).strip()
        if resolved not in BUILD_MODES:
            raise IndexingError(
                f"未知构建模式 {mode!r}（解析后 {resolved!r}），"
                f"可选：{' / '.join(BUILD_MODES)}。"
                "full 会先清空整个库再写回全部记录，因此本模块不接受"
                "任何'看起来差不多'的值——猜错的代价是一次全量重建。"
            )
        return resolved

    def _resolve_parent(
        self, parent: IndexManifest | None
    ) -> tuple[IndexManifest | None, str]:
        """按"显式 > 版本表 > 首版"的顺序解析上一版（返回值第二项是来源）."""
        if parent is not None:
            return parent, PARENT_EXPLICIT
        if self._versions is not None:
            latest = self._versions.latest()
            if latest is not None:
                return latest, PARENT_VERSIONS
        return None, PARENT_NONE

    def _apply_rebuild_threshold(
        self,
        mode: str,
        plan: IndexPlan,
        parent: IndexManifest | None,
    ) -> tuple[str, str]:
        """增量模式下按变更比例决定要不要换挡到全量（见模块 docstring）.

        四个前置条件缺一不可，其中第三个最容易被漏掉：

        ```text
        mode 是 incremental   全量模式不需要换挡（它已经是全量了）
        parent 存在            没有上一版时"变更比例"没有分母（全部是新增）
        plan.total > 0         空记录集时两种模式等价，不换挡也不报错
        比例 > 阈值（严格大于）等于阈值时不换挡：阈值是"超过它才换"的那条线
        ```

        换挡只改"怎么写库"，不改"写了什么"：条目集合、版本号都不受影响
        （``full`` 与 ``incremental`` 写完之后库的内容相同，差别只在过程与成本）。
        """
        if mode != BUILD_MODE_INCREMENTAL or parent is None or plan.total <= 0:
            return mode, ""
        ratio = plan.changed / plan.total
        threshold = float(settings.indexing_full_rebuild_threshold)
        if ratio > threshold:
            return BUILD_MODE_FULL, (
                f"变更比例 {ratio:.1%} 超过阈值 {threshold:.2f}，"
                "已从 incremental 切到 full："
                "增量不是永远更省——变更比例很高时，逐条 upsert 加逐条 delete 的开销"
                "反而高于清空重写"
            )
        return mode, ""

    def _reuse_ratio(
        self, mode: str, plan: IndexPlan, cache_hits: int, requested: int
    ) -> float:
        """按模式给出复用率（两套口径的理由见模块 docstring）.

        ```text
        增量：plan.reuse_ratio = 未变条目 / 总条目（这一版有多少条完全没变）
        全量：cache_hits / requested（这一次请求的文本里有多少条靠缓存免掉）
        ```

        全量模式下 ``requested == 0``（空记录集）时返回 0.0 而不是除零错误：
        "没有请求"与"请求了但一条都没命中"在本课是同一个数字（都没省下东西）。
        """
        if mode == BUILD_MODE_INCREMENTAL:
            return plan.reuse_ratio
        if requested <= 0:
            return 0.0
        return round(cache_hits / requested, 4)

    # ------------------------------------------------------------------ 内部：编码

    def _encode(
        self,
        pending: Sequence[tuple[str, str]],
        failures: list[tuple[str, str]],
    ) -> tuple[dict[str, list[float]], int, int, int, list[str]]:
        """编码 ``pending``（整批优先，失败逐条重试），返回向量表与四个计数.

        返回值里的计数是 ``EncodeReport`` 那四个数的聚合：``cache_hits`` /
        ``encoded`` / ``batches``，以及一条（可能有，也可能没有的）说明。

        逐条重试的代价必须写清楚：**失败的那一条不计入 ``encoded`` /
        ``batches``**（编码器在抛出之前没有产出报告），因此出错时的
        ``encoded`` 会**低于**真实花掉的调用次数。这不是可以糊过去的差额，
        所以它进 ``note``：看到"已逐条重试"这句话的人就知道那个数字偏小。
        """
        cache_hits = 0
        encoded = 0
        batches = 0
        notes: list[str] = []
        if not pending:
            return {}, cache_hits, encoded, batches, notes

        texts = [text for _, text in pending]
        try:
            batch_vectors, report = self._encoder.encode(texts)
        except EncodingError as exc:
            notes.append(
                f"{len(pending)} 条整批编码失败（{_brief(exc)}），已逐条重试"
                "（失败的那条不计入 encoded / batches，这次报告的编码成本偏低）"
            )
            vectors: dict[str, list[float]] = {}
            for record_id, text in pending:
                try:
                    one_vectors, one = self._encoder.encode([text])
                except EncodingError as item_exc:
                    failures.append((record_id, f"失败：编码被拒：{_brief(item_exc, 200)}"))
                    continue
                cache_hits += one.cache_hits
                encoded += one.encoded
                batches += one.batches
                vectors[record_id] = one_vectors[0]
            return vectors, cache_hits, encoded, batches, notes

        cache_hits += report.cache_hits
        encoded += report.encoded
        batches += report.batches
        # 编码器保证"结果与入参逐位对应"（见 encoder 的承诺），因此这里可以放心 zip。
        return (
            {record_id: vector for (record_id, _), vector in zip(pending, batch_vectors)},
            cache_hits,
            encoded,
            batches,
            notes,
        )

    # ------------------------------------------------------------------ 内部：写入

    def _write(
        self,
        prepared: Sequence[VectorRecord],
        failures: list[tuple[str, str]],
    ) -> tuple[WriteReport, list[str]]:
        """把记录写进库：整批优先，``VectorStoreError`` 时逐条重试.

        为什么整批优先而不是永远逐条：批量的收益是真实的（后端可以一次
        建索引、一次落盘），而失败是少见的。**逐条只在出错时付出**，
        并且它把故障从"这一批 32 条都不见了"缩小到"这一条写不进去"。

        非 ``VectorStoreError`` 的异常一律向上抛（见模块 docstring 第三条）：
        后端实现里的 ``TypeError`` 是真 bug，它必须让调用栈喊出来，
        而不是变成报告里一行安静的 ``failed=1``。
        """
        notes: list[str] = []
        if not prepared:
            return WriteReport(), notes
        try:
            write = self._backend.upsert(prepared)
        except VectorStoreError as exc:
            notes.append(
                f"整批写入被拒（{_brief(exc)}），已逐条重试："
                "批量操作会把一条坏数据放大成整批失败，而逐条重试能把故障定位到具体 id"
            )
            write = WriteReport()
            for record in prepared:
                try:
                    one = self._backend.upsert([record])
                except VectorStoreError as item_exc:
                    failures.append(
                        (record.record_id, f"失败：写入被拒：{_brief(item_exc, 200)}")
                    )
                    continue
                write = write.merge(one)
        if write.skipped:
            # 后端自己跳过的条数（护栏）不并进 failed：我们没有它的逐条 id，
            # 编一个 id 出来只会让 failures 变成一份不可信的名册。如实说一声，
            # 并把这个差额留给 verify() —— 它是"清单与库对不上"的检查点。
            notes.append(f"后端护栏跳过了 {write.skipped} 条（详见后端的 skipped 计数）")
        return write, notes

    # ------------------------------------------------------------------ 内部：删除

    def _remove_stale(
        self,
        plan: IndexPlan,
        mode: str,
        protected: set[str],
    ) -> tuple[int, list[str]]:
        """删除"上一版有、这一版没有"的 id，返回**真正删掉的条数**.

        ```text
        增量  对 plan.removed 调用 backend.delete(ids=...)；返回值就是结论
              请求数与实际删除数不同时补一条 note —— 那说明库与清单已经不一致
              （有人手工删过记录），它不该被静默吞掉
        全量  不单独删：clear() 已经把它们（以及所有旧记录）一起清掉了，
              因此这里报的是 plan.removed 的**条数**而不是一次 delete 的返回值
        ```

        ``protected``（失败的、且上一版清单里有的那些 id）**不在删除范围内**：
        见模块 docstring 的"失败条目与清单的关系"。
        """
        if mode != BUILD_MODE_INCREMENTAL:
            return len(plan.removed), []
        to_delete = tuple(record_id for record_id in plan.removed if record_id not in protected)
        if not to_delete:
            return 0, []
        removed = self._backend.delete(ids=list(to_delete))
        notes: list[str] = []
        if removed != len(to_delete):
            notes.append(
                f"请求删除 {len(to_delete)} 条，实际删掉 {removed} 条："
                "差额说明库与清单已经不一致（有人手工删过记录，或上一次构建半途失败），"
                "请用 verify() 对账后再做增量"
            )
        return removed, notes

    # ------------------------------------------------------------------ 内部：备份

    def _maybe_backup(self, manifest: IndexManifest, reason: str) -> str:
        """按需拷一份快照，返回一条说明（空串表示无需说明）.

        三种"不备份"都必须**留下说明而不是报错**：

        ```text
        没有备份表（backups=None）          → 调用方没接这条线，构建照常成功
        后端不支持持久化                    → **本来就没有文件可备份**
        后端声称支持但没有落盘位置（抛错）    → 同上，只是发现得晚一点
        ```

        第三种用 ``try/except`` 而不是提前判 ``supports_persistence``：
        一个"声明支持持久化但没配路径"的后端只会在 ``persist()`` 那里响，
        而提前判标志位会漏掉它。

        快照在**写入之后**建：它固化的是"这一版建成时的库文件"，
        与随之登记的这份清单成对（``BackupRecord.version_id`` 就是它的版本号）。
        想要"重建之前的快照"（那个库还是旧的那一版）请在**调用 build 之前**
        自己调 ``backups.create(...)``——构建器不在事前动手，
        因为那时它还没有一份清单可以配给快照。
        """
        if self._backups is None:
            return "没有接备份表（backups=None），本次未备份——构建照常完成"
        if not self._backend.supports_persistence:
            return (
                f"后端 {self._backend.name} 不支持持久化，没有文件可备份，已跳过备份"
                "（不能落盘的后端本来就没有快照可做，这不是一个错误）"
            )
        try:
            location = self._backend.persist()
        except VectorStoreError as exc:
            return (
                f"后端 {self._backend.name} 无法落盘（{_brief(exc)}），已跳过备份；"
                "构建本身不受影响，但这一版没有快照"
            )
        self._backups.create(
            source_files=[location],
            manifest=manifest,
            reason=reason or f"{manifest.backend} 索引 {manifest.version_id}",
        )
        return ""


__all__ = [
    "BUILD_MODE_FULL",
    "BUILD_MODE_INCREMENTAL",
    "BUILD_MODES",
    "PARENT_EXPLICIT",
    "PARENT_NONE",
    "PARENT_VERSIONS",
    "IndexBuilder",
    "split_usable_records",
]
