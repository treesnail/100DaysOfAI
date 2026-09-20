"""索引构建流水线：把 day061 → day062 → day064 → day065 四天的链子接在一起（M6-D4）.

```text
day061  documents   文件 → Document（全文 + 块序列 + 内容指纹）
day062  chunking    Document → ChunkSet → knowledge_records()（doc_id + text + metadata）
day064  vectorstore 记录 → 向量 → 向量库（逐条编码：库没变、钱照花）
day065  indexing    记录 → 计划 → 批量编码 + 缓存 → 增量对账 → 清单 + 版本 + 备份
```

本模块是这条链子的**最高一层**，它只做三件事：

```text
1. 接上游     把"文档"或"记录"变成一次构建（build_from_documents / build_from_records）
2. 报现状     stats() / history()：这个索引现在是什么、有哪几版
3. 检索        search()：复用 day064 的检索路径，一行转发（见下）
```

真正的构建逻辑全部在 ``builder`` 里。这一层刻意**薄**：它的价值不在
"多写一段流程"，而在"让调用方只需要认识一个入口"。装配阶段（``__init__.py``
与端点）要的是"给我一个能构建、能查、能报状态的索引对象"，
而不是七个必须按顺序调用的函数。

## 为什么 ``search`` 必须复用 ``VectorIngestPipeline``

day064 的 ``VectorIngestPipeline`` 已经定义了"**哪段文本被编码**"
（``retrieval_text`` 优先，回落 ``text``）与"检索用什么口径"
（``settings.vector_default_top_k`` / ``vector_min_score``）。本模块
**一行转发，不重写**：

```python
return self._ingest.search(query, top_k=top_k, where=where)
```

自己重写一遍编码+检索的代价不是"多几行"，而是**同一条规则有了两个实现**：

```text
写入侧用 retrieval_text（面包屑 + 正文），查询侧用纯 query
→ 这是对的（查询本来就没有面包屑）
写入侧用 A 编码器、构建侧又拿 B 编码器去查
→ 这是错的，而它的表现只是"排序不太对"，没有任何异常
```

"该编码哪段文本"这条规则只能有一处定义（``vectorstore.pipeline.embedding_text``，
day065 的 ``planner`` 也是复用它的）。查询侧与写入侧共用同一个编码器实例，
是把"两边口径一致"从一句承诺变成结构上的必然。

## 与 builder 的关系：一个只读的观测面

```text
IndexBuilder.build(...)     真正动手（编码、写库、登记、备份）
IndexingPipeline         只负责：接上游 / 转发行 / 报现状
```

因此本层**不做任何判定**：它不会"帮"构建器决定模式，也不会"帮"版本表
决定采纳哪一版。唯一的例外是 ``build_from_documents`` 里那次分块——
那是本层不可推卸的职责（builder 不认识 ``Document``）。

## 版本表与备份表为什么在这里**再注入一次**

``IndexBuilder`` 已经持有它们（构建时登记与备份），本层再注入一遍看起来重复。
但两者的**用途不同**：

```text
builder 手里的两份   用来**写**（register / adopt / create）
pipeline 手里的两份  用来**读**（history / stats 的版本数、备份数）
```

本层刻意不去翻 builder 的私有字段：那会让"这个 pipeline 报的是哪个版本表"
变成一个需要读两个类才能回答的问题。两个入口由 ``default_indexing_pipeline()``
统一装配，因此正常路径上它们**是同一份对象**（见那个函数的说明）。

## 放弃了什么（代价写在明面上）

| 放弃的东西 | 代价 | 为什么可以接受 |
|-----------|------|---------------|
| 自己拿 embedding 去分块 | semantic 下按 settings 另取一个编码器 | 切分用与入库用的向量是两件事 |
| 自动读盘 | 构造后版本数报 0（即使磁盘上有历史） | 读盘是显式动作：versions.load() |
| 目录的提前创建 | 导入期不建任何目录 | 只在真正落盘时创建，见 ``default_indexing_pipeline`` |
| 构建的自动重试 | 失败就失败，报告里如实记录 | 重试属于运维策略，不该藏在流水线里 |
| 构建历史的缓存 | 每次构建都真跑一遍 | 构建本来就慢，缓存一个慢结果没有意义 |

## 谁依赖它

```text
day065 装配阶段（indexing/__init__.py）  导出 IndexingPipeline / default_indexing_pipeline
/indexing/* 端点                        构建、检索、状态、版本历史都由它回答
scripts / 演示脚本                      default_indexing_pipeline() 一行拿到装配好的对象
tests/                                  全部离线：默认 settings 下可用（flat + char-ngram）
```
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from smart_research_agent.chunking.base import ChunkPolicy, default_policy
from smart_research_agent.chunking.pipeline import ChunkPipeline
from smart_research_agent.config import settings
from smart_research_agent.documents.types import Document
from smart_research_agent.indexing.backup import IndexBackupStore
from smart_research_agent.indexing.builder import IndexBuilder
from smart_research_agent.indexing.cache import EmbeddingCache
from smart_research_agent.indexing.encoder import describe_embedding
from smart_research_agent.indexing.types import (
    EmbeddingIdentity,
    IndexingReport,
    IndexManifest,
)
from smart_research_agent.indexing.versioning import IndexVersionStore
from smart_research_agent.llm.embedding import EmbeddingProvider, default_embedding
from smart_research_agent.vectorstore.base import VectorBackend
from smart_research_agent.vectorstore.pipeline import VectorIngestPipeline
from smart_research_agent.vectorstore.registry import create_backend
from smart_research_agent.vectorstore.types import SearchResult

#: 备份目录在 ``settings.indexing_dir`` 下的名字。取子目录而不是与清单平铺：
#: 备份是**若干份快照**（每份一个目录），而清单/历史/指针是单文件——
#: 混在一起会让 ``ls`` 的结果既读不出"现在生效的是哪一版"，
#: 也读不出"还有几份能回去"。
BACKUP_DIR_NAME = "backups"


class IndexingPipeline:
    """构建 / 检索 / 报状态的统一入口（见模块 docstring）.

    它**只做编排与转发**，不做任何判定——所有口径（差集、批量、缓存、
    版本、备份）都在被它调用的那几个模块里。因此本类的用例集中在**接缝**上：
    "文档能不能被切成块""块能不能变成记录""库能不能被搜到"，
    而每个模块内部的判定由它们各自的测试文件负责
    （见 ``test_indexing_pipeline.py`` 的说明）。
    """

    def __init__(
        self,
        builder: IndexBuilder,
        *,
        versions: IndexVersionStore | None = None,
        backups: IndexBackupStore | None = None,
    ) -> None:
        """装配流水线：一个构建器 + 两个**只读**用的账本.

        ``versions`` / ``backups`` 只被 ``stats()`` / ``history()`` /
        ``build_from_records`` 的备份开关使用。它们为 ``None`` 时：

        ```text
        history()   返回空列表（"没有版本表"与"有版本表但一版都没有"在本层同形）
        stats()     版本数 / 备份数报 0
        构建        不做备份（构建器那边同样会因此跳过备份并留一句 note）
        ```

        ``search`` 用的摄取流水线在这里就建好（它不碰库、不做 IO），
        因此构造本对象是**无副作用**的：不建目录、不读盘、不写任何东西。
        """
        self._builder = builder
        self._versions = versions
        self._backups = backups
        self._manifest: IndexManifest | None = None
        self._last_report: IndexingReport | None = None
        # 检索复用 day064 的摄取流水线（见模块 docstring）：
        # 同一个后端 + **同一个编码器实例**，因此查询侧与写入侧的口径不可能分叉。
        self._ingest = VectorIngestPipeline(builder.backend, builder.embedding)

    # ------------------------------------------------------------------ 构建

    def build_from_records(
        self,
        records: Sequence[dict[str, Any]],
        *,
        mode: str | None = None,
        reason: str = "",
    ) -> IndexingReport:
        """从一批知识库记录构建索引（转发给 ``builder.build``）.

        三个参数的去向各不相同，写清楚是因为它们**都不在本层解释**：

        ```text
        records  builder 按 planner 的口径读（doc_id / text / metadata）
        mode     None = 按 settings.indexing_mode；换挡逻辑在 builder 里
        reason   只进备份记录（"这份快照是因为什么建的"），不进清单
        ```

        备份开关是**由装配决定**的：接了备份表就每次构建都备份一份
        （保留份数由备份表的 ``keep`` 管，见 ``backup`` 模块"备份不是日志"）。
        这样调用方不必每次都记得传一个参数——而"忘了传"的后果是
        事故当天才发现没有快照可用。

        ``parent`` 不在这里暴露：本层一律让 builder 走"显式 > 版本表 > 首版"
        的解析顺序。要指定一个特定的上一版，请直接用 builder（本层是薄壳，
        不该长出第二条计分路径）。
        """
        manifest, report = self._builder.build(
            records,
            mode=mode,
            reason=reason,
            backup=self._backups is not None,
        )
        self._manifest = manifest
        self._last_report = report
        return report

    def build_from_documents(
        self,
        documents: Sequence[Document],
        *,
        strategy: str | None = None,
        policy: ChunkPolicy | None = None,
        mode: str | None = None,
    ) -> IndexingReport:
        """**端到端**：``Document`` → 分块 → 知识库记录 → 构建索引.

        ```text
        Document                      day061 的产物（全文 + 块 + 内容指纹）
          ↓ ChunkPipeline.chunk(...)   day062：按策略切，产出 ChunkSet
          ↓ ChunkSet.knowledge_records()  day062：doc_id + source + text + metadata
          ↓ build_from_records(...)    day065：计划 → 编码 → 写库 → 清单
        ```

        这条链子上的**每一个环节都还在**，本函数只是把它们接上：

        - 分块走 ``ChunkPipeline``（本课唯一的切分实现，四种策略都在它后面）；
        - 记录形状走 ``ChunkSet.knowledge_records()``（含 ``fingerprint``
          与 ``retrieval_text``——前者进清单的内容身份，后者进编码器）；
        - 入库走 ``build_from_records``（因此增量、缓存、清单、版本、备份
          这些行为与"直接喂记录"**逐字相同**）。

        ``strategy`` / ``policy`` 的缺省：``settings.chunking_strategy`` 与
        ``default_policy(strategy)``。给了 ``policy`` 而没给 ``strategy`` 时，
        策略取 ``policy.strategy``——否则这对参数里必然有一组会直接报错
        （分块器只接受与自己同名的策略），而那种报错对调用方毫无信息量。

        分块用的 embedding 刻意**不是** builder 的那个（本层拿不到它，
        也不该去拿）：分块的向量只用来找切分点，入库的向量才是索引的东西，
        把两者绑成一个会凭空多出一条"换切分器就得换索引编码器"的约束。
        默认策略 ``recursive`` 完全不需要 embedding。
        """
        resolved_strategy = strategy or (policy.strategy if policy is not None else None)
        if not resolved_strategy:
            resolved_strategy = settings.chunking_strategy
        resolved_policy = policy if policy is not None else default_policy(resolved_strategy)

        chunker = ChunkPipeline()
        chunk_sets = [
            chunker.chunk(document, resolved_strategy, resolved_policy)
            for document in documents
        ]
        records = [
            record for chunk_set in chunk_sets for record in chunk_set.knowledge_records()
        ]
        chunks = sum(chunk_set.count for chunk_set in chunk_sets)
        reason = (
            f"由 {len(chunk_sets)} 份文档按 {resolved_strategy} 策略切出 {chunks} 块后构建"
        )
        return self.build_from_records(records, mode=mode, reason=reason)

    # ------------------------------------------------------------------ 检索

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> SearchResult:
        """按查询文本检索，返回 ``SearchResult``（**一行转发**，见模块 docstring）.

        ``top_k`` 的三级优先级由摄取流水线给出（调用参数 > 构造参数 >
        ``settings.vector_default_top_k``）；本层不提供构造参数，
        因此实际是"调用参数 > settings"。

        ``min_score`` 不暴露：默认不设阈值（``settings.vector_min_score``
        为 ``None``）。理由见 ``vectorstore.pipeline._resolve_min_score``——
        在 ``l2`` 度量下分数是负数，一个"顺手加的"阈值会把全部命中切掉。
        要设阈值请直接用 ``VectorIngestPipeline``。
        """
        return self._ingest.search(query, top_k=top_k, where=where)

    # ------------------------------------------------------------------ 状态

    def stats(self) -> dict[str, Any]:
        """当前状态：库、编码器身份、清单、版本数、备份数、缓存（可直接 json.dumps）.

        两处刻意的不对称，都写在这里免得被当成 bug：

        ```text
        dimension   报的是**库**的维度（库还没建时是 0）；编码器的维度在
                    identity.dimension 里。两者不一致正是维度护栏要拦的状态，
                    把它们合并成一个字段会让这种脏状态消失
        manifest    报的是**这个 pipeline 亲手构建过的**那一版；它没有构建过
                    时是 None。它**不去磁盘上找**"当前生效的清单"——那需要回答
                    "清单文件在哪、它是不是现在生效的那份"，而那两个问题的
                    答案在版本表与配置文件里（history() 与 settings.indexing_dir）
        ```

        ``cache`` 直接用 ``EmbeddingCache.stats()``：它的 ``hit_ratio`` 与
        ``report.cache_hits`` 是同一个口径（后者是"某一次构建内"的，
        前者是"这份缓存实例累计的"）。
        """
        builder = self._builder
        return {
            "backend": builder.backend.name,
            "metric": builder.backend.metric,
            "dimension": builder.backend.dimension,
            "count": builder.backend.count(),
            "identity": builder.identity.to_dict(),
            "manifest": (
                self._manifest.to_dict(include_entries=False)
                if self._manifest is not None
                else None
            ),
            "versions": len(self._versions) if self._versions is not None else 0,
            "backups": len(self._backups.list()) if self._backups is not None else 0,
            "cache": builder.cache.stats(),
        }

    def history(self) -> list[dict[str, Any]]:
        """版本表里每一版的 ``version_id`` 与 ``summary_line``（**旧 → 新**）.

        顺序取版本表的**登记顺序**而不是版本号排序：版本号是内容摘要，
        它的字典序没有任何时间含义（见 ``versioning.IndexVersionStore.history``）。

        只给两个字段是刻意的：版本表里的每一版都是一份完整清单，
        把它们全部塞进响应体等于把历史索引的全文都发出去，
        而这里要回答的只是"有哪几版、各是什么"。
        """
        if self._versions is None:
            return []
        return [
            {"version_id": manifest.version_id, "summary_line": manifest.summary_line()}
            for manifest in self._versions.history()
        ]


def default_indexing_pipeline(
    *,
    embedding: EmbeddingProvider | None = None,
    backend: VectorBackend | None = None,
    cache: EmbeddingCache | None = None,
) -> IndexingPipeline:
    """按 ``settings`` 装配一条完整的索引流水线（端点与演示脚本的统一入口）.

    装配清单（每一项都来自配置，而不是写死的）：

    ```text
    backend    create_backend()                vector_backend / vector_metric / vector_persist_path
    embedding  default_embedding()             embedding_provider（缺省 char-ngram，离线可用）
    cache      EmbeddingCache(...)             indexing_cache_path（空串 = 只用内存）
    batch_size settings.indexing_batch_size    一次送进提供方的文本条数
    versions   <indexing_dir>                  版本历史与 current 指针
    backups    <indexing_dir>/backups          若干份快照（保留 indexing_backups_keep 份）
    ```

    三个有一句必须解释清楚的地方：

    **1. 缓存的三个身份字段来自编码器，而不是配置。** 缓存键里含身份
    （``types.vector_key``），因此"缓存属于谁"必须由 ``describe_embedding``
    当场问出来——用配置里的字符串（例如 ``embedding_provider="char-ngram"``）
    拼一个身份，会在"同一个提供方换了维度设置"时给出**同一个键**，
    于是旧向量被复用，而检索结果只是"变得不太对"。

    磁盘上已经有缓存文件时**载入它**（身份不符会被 ``EmbeddingCache.load``
    当场拒读——那是它的设计，见那个模块的第 2 条行为）；
    没有就当空缓存开始。缓存是可以随时删的，因此这里不做任何补救。

    **2. 版本表与备份表只装配、不读盘。** 两个模块的构造器本来就刻意不读盘
    （见各自 docstring 的取舍表：悄悄用磁盘上的旧指针回滚是一类真实的错误），
    因此"有历史"这件事必须由调用方显式``load()``。本函数保持这个语义。

    **3. 目录只在真正落盘时创建。** 本函数**一个目录都不建**：
    它可能只是被某个只读端点调用（回答"现在是什么状态"），
    而"问一下状态"不该在磁盘上留下任何东西——``IndexVersionStore.persist``
    与 ``IndexBackupStore._base_dir`` 会在需要时自己 ``mkdir``。
    """
    resolved_backend = backend if backend is not None else create_backend()
    resolved_embedding = embedding if embedding is not None else default_embedding()
    identity = describe_embedding(resolved_embedding)
    resolved_cache = cache if cache is not None else _default_cache(identity)

    directory = str(settings.indexing_dir)
    versions = IndexVersionStore(path=directory)
    backups = IndexBackupStore(path=str(Path(directory) / BACKUP_DIR_NAME))

    builder = IndexBuilder(
        resolved_backend,
        resolved_embedding,
        cache=resolved_cache,
        identity=identity,
        batch_size=int(settings.indexing_batch_size),
        versions=versions,
        backups=backups,
    )
    return IndexingPipeline(builder, versions=versions, backups=backups)


def _default_cache(identity: EmbeddingIdentity) -> EmbeddingCache:
    """按编码器身份建一份缓存，并在磁盘上已有同身份文件时载入它.

    单独抽出来是因为它有一处**顺序**要求：缓存必须先有身份、再谈载入。
    反过来（先载入、后补身份）会让载入校验拿一份空身份去比，
    而那样比出来的结论永远是"身份不符"或者（更糟）"恰好相符"。
    """
    path = str(settings.indexing_cache_path or "")
    cache = EmbeddingCache(
        path=path,
        provider=identity.provider,
        model=identity.model,
        dimension=identity.dimension,
    )
    if path and Path(path).exists():
        cache.load()
    return cache


__all__ = [
    "BACKUP_DIR_NAME",
    "IndexingPipeline",
    "default_indexing_pipeline",
]
