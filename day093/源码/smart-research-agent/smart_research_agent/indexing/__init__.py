"""Embedding 索引构建包（M6-D4）：给向量库补上一本**账**.

day064 交付的是一个库：存了向量与原文，能按相似度查。
但库本身**答不了**三个问题，而这三个问题决定了它能不能被运营：

```text
1. 这份索引是用哪个编码器的哪个版本建的？   → 库不认识"编码器"
2. 上一版到这一版，哪些块没变？             → 库没有"上一版"这个概念
3. 我能不能回到上一版，或者照清单重建一次？   → 库没有历史，也不能被 diff
```

今天这 11 个模块就是这三问的答案。而它们要解决的核心矛盾可以写成一句话：

> **"有没有变"与"该不该重算向量"是两个问题。**

```text
内容身份（fingerprint）   这段文本是不是另一段了        → 决定"要不要更新这条记录"
向量键（vector_key）      这个向量还能不能用              → 决定"要不要重新编码"
```

两者**必须分开判**：只比内容身份，会漏掉"换了 embedding 模型"
（内容没变、向量作废）；只比向量键，会漏掉"内容变了但长度相同"
（覆盖写场景里 `vector_key` 也可能相同）。而两种漏判都不报错，
只表现为"检索质量下降"——所以它们在 `planner` 里被写成两个独立判据。

## 十一个模块，每个回答一个问题

```text
errors.py     这一层会怎么失败（编码/清单/版本/备份 四族，修复人各不相同）
types.py      账的形状：清单、条目、差集、编码报告；时间戳为什么不能进版本号
cache.py      文本 → 向量的查表；键里为什么必须有编码器身份
encoder.py    批量编码：批大小的代价、同批去重、三项返回值校验
planner.py    差集：四条判据、两个身份的分别判定、编码器变了要单独说出来
manifest.py   清单：可被任何人重算的版本号、与库的一致性体检
versioning.py 版本表：血缘（回滚必须沿血缘，不能按时间倒序）
backup.py     备份：为什么不设保留上限是错的、backup_id 里为什么没有时间
builder.py    装配：维度护栏、阈值换挡（增量不是永远更省）、三处降级
pipeline.py   端到端：Document → Chunk → 记录 → 索引；检索复用向量层
```

## 三条贯穿全包的纪律

1. **身份进键**。`vector_key` 包含 provider / model / dimension，
   `index_version_id` 包含编码器身份 / 度量 / 后端 / 内容摘要。
   于是"换配置"会**必然**让旧向量作废，而不是靠人记得重建索引
   （day062 把 `strategy` 与 `index` 放进 `chunk_id`、day058 把三元组放进版本键，是同一个动作）；
2. **能重算的才叫版本号**。时间戳、机器名、耗时一律不进身份——
   同一份内容在任何时间任何机器上必须算出同一个 16 位十六进制数，
   否则"这个版本在哪台机器上建的"会变成一个必须回答的问题；
3. **"没做成"要分家**。`IndexPlan` 把未变/新增/更新/删除分成四组，
   `IndexingReport` 再把"数据问题"与"环境问题"分开，
   剩下不认识的异常一律上抛。**只有分开，"这次为什么什么都没做"才能被回答**——
   而重放同一批时 `encoded == 0` 正是这一课要拿出来的那个数字。

## 与既有包的接缝

- **上游**：`chunking`（day062）的 `knowledge_records()`；`llm.embedding`（day041）；
  `documents`（day061）负责把文件变成 `Document`；
- **脚下**：`vectorstore`（day064）提供库与六个原语，本包**不改它**，只是在它之上记账；
- **下游**：day066 的检索器消费本包写入的索引；day071 的 RAG 评估用
  `manifest.compare_with_store` 与 `builder.verify` 判断"索引有没有漂移"；
- **端点**：`api.routes` 的 `/indexing/*`；演示脚本 `scripts/indexing_demo.py`；
  手册在 `docs/embedding_index.md`。
"""

from __future__ import annotations

from smart_research_agent.indexing.backup import (
    BACKUP_INDEX_FILE,
    BACKUP_VERSION,
    DEFAULT_KEEP,
    BackupRecord,
    IndexBackupStore,
    utc_now_iso,
)
from smart_research_agent.indexing.builder import (
    BUILD_MODE_FULL,
    BUILD_MODE_INCREMENTAL,
    BUILD_MODES,
    PARENT_EXPLICIT,
    PARENT_NONE,
    PARENT_VERSIONS,
    IndexBuilder,
    split_usable_records,
)
from smart_research_agent.indexing.cache import CACHE_VERSION, EmbeddingCache
from smart_research_agent.indexing.encoder import (
    DEFAULT_BATCH_SIZE,
    MODEL_LABEL_FALLBACK,
    BatchEncoder,
    describe_embedding,
)
from smart_research_agent.indexing.errors import (
    BackupError,
    EncodingError,
    IndexingError,
    ManifestError,
    VersionError,
)
from smart_research_agent.indexing.manifest import (
    MANIFEST_FILE,
    MANIFEST_HISTORY_FILE,
    build_manifest,
    compare_with_store,
    manifest_from_store,
    read_manifest,
    verify_index,
    write_manifest,
)
from smart_research_agent.indexing.pipeline import (
    BACKUP_DIR_NAME,
    IndexingPipeline,
    default_indexing_pipeline,
)
from smart_research_agent.indexing.planner import (
    entry_from_record,
    plan_index,
    plan_reason,
)
from smart_research_agent.indexing.types import (
    ACTIONS,
    ACTION_ADDED,
    ACTION_REMOVED,
    ACTION_UNCHANGED,
    ACTION_UPDATED,
    MANIFEST_VERSION,
    VECTOR_KEY_LENGTH,
    EmbeddingIdentity,
    EncodeReport,
    IndexEntry,
    IndexManifest,
    IndexPlan,
    IndexingReport,
    entries_digest,
    index_version_id,
    vector_key,
)
from smart_research_agent.indexing.versioning import CURRENT_FILE, IndexVersionStore

__all__ = [
    "ACTIONS",
    "ACTION_ADDED",
    "ACTION_REMOVED",
    "ACTION_UNCHANGED",
    "ACTION_UPDATED",
    "BACKUP_DIR_NAME",
    "BACKUP_INDEX_FILE",
    "BACKUP_VERSION",
    "BUILD_MODES",
    "BUILD_MODE_FULL",
    "BUILD_MODE_INCREMENTAL",
    "CACHE_VERSION",
    "CURRENT_FILE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_KEEP",
    "MANIFEST_FILE",
    "MANIFEST_HISTORY_FILE",
    "MANIFEST_VERSION",
    "MODEL_LABEL_FALLBACK",
    "PARENT_EXPLICIT",
    "PARENT_NONE",
    "PARENT_VERSIONS",
    "VECTOR_KEY_LENGTH",
    "BackupError",
    "BackupRecord",
    "BatchEncoder",
    "EmbeddingCache",
    "EmbeddingIdentity",
    "EncodeReport",
    "EncodingError",
    "IndexBackupStore",
    "IndexBuilder",
    "IndexEntry",
    "IndexManifest",
    "IndexPlan",
    "IndexVersionStore",
    "IndexingError",
    "IndexingPipeline",
    "IndexingReport",
    "ManifestError",
    "VersionError",
    "build_manifest",
    "compare_with_store",
    "default_indexing_pipeline",
    "describe_embedding",
    "entries_digest",
    "entry_from_record",
    "index_version_id",
    "manifest_from_store",
    "plan_index",
    "plan_reason",
    "read_manifest",
    "split_usable_records",
    "utc_now_iso",
    "vector_key",
    "verify_index",
    "write_manifest",
]
