"""增量计划：算出"这一版与上一版之间，哪些块要重算"（M6-D4）.

``IndexPlan`` 是差集，本模块是**算差集的那一步**。它回答一个直接决定成本的问题：

```text
上一版建好之后，我只改了一段话 —— 这次要重新编码几条？
```

## 两个问题必须分开判，这是本模块唯一容易写错的地方

每一条记录身上有两个指纹，它们看起来都在说"变了没有"，实则回答两个问题：

```text
fingerprint   sha256(原文)[:16]                    ← 「内容变了没有」
vector_key    sha256(身份 | 编码文本)[:16]          ← 「该不该复用向量」
```

只比其中一个，都会漏掉一整类变更，而且**两类漏判都不报错**：

| 只比 | 漏掉的变更 | 症状 |
|------|-----------|------|
| ``fingerprint`` | 换了编码器（身份变了，原文没变） | 新旧向量混在一个库里，排序整体错乱 |
| ``vector_key`` | 内容变了但编码文本长度不变 | 库里留着旧内容的向量，命中的却是新原文 |

所以判据必须是**两条分别判、任一不同即 ``updated``**（见 ``plan_index``）。
"两个字段"这件事本身不是冗余，而是把两个不同的问题各留了一份证据——
与 day062 把 ``chunk_id``（哪一块）与 ``fingerprint``（哪段内容）分开是同一条纪律。

## ``vector_key`` 里含身份，于是"换编码器"自动变成"全部重算"

这是"身份进键"的设计在增量这一层的兑现：``identity`` 一变，所有
``vector_key`` 都变，于是所有条目落进 ``updated``。好处是正确性由结构保证
（想漏判都漏判不了）；坏处是**报告里只看到"更新了 1200 条"，
看不出这是配置变更而不是数据变更**。因此 ``plan_reason`` 把这一种情况
单独写出来——它不改变数字，只改变"看到数字之后该做什么"。

## 这一层不做什么

```text
不读缓存、不编码      → encoder.py 的事
不碰向量库            → builder 拿着本模块的 to_encode 去删/去写
不算版本号与清单       → manifest.py 的事（本模块只产出 IndexEntry 的形状）
```

## 放弃了什么

| 放弃的东西 | 代价 | 为什么可以接受 |
|-----------|------|---------------|
| 记录级去重提示 | 同一 id 出现两次时后者覆盖前者 | id 唯一性由 day062 的 ``chunk_id`` 保证 |
| 只读一个字段的"快路径" | 每条要算两次哈希 | 相对一次编码调用可忽略，省掉它会漏判 |
| 记录 ``updated`` 的原因 | ``IndexPlan`` 只有 id 分组 | 见 ``plan_reason`` 的汇总 |

## 谁依赖它

```text
builder（day065）        拿 plan.to_encode 决定这一轮要编码哪些块
manifest.compare_*       用同一套 fingerprint/vector_key 口径与库对账
reports / 端点           直接打印 plan.summary_line() 与 plan_reason()
```
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from smart_research_agent.documents.types import content_id
from smart_research_agent.indexing.types import (
    EmbeddingIdentity,
    IndexEntry,
    IndexManifest,
    IndexPlan,
    vector_key,
)
from smart_research_agent.vectorstore.pipeline import (
    FALLBACK_TEXT_FIELD,
    embedding_text,
    record_id_of,
)


def entry_from_record(record: dict[str, Any], identity: EmbeddingIdentity) -> IndexEntry:
    """把一条知识库记录变成一个清单条目（``IndexEntry``）.

    形状与 day062 ``ChunkSet.knowledge_records()`` / day064 ``vectorstore.pipeline``
    完全一致（``doc_id`` + ``text`` + ``metadata``），并且**复用它们的两个函数**：

    ```text
    record_id_of(record)      id 在哪一层、缺了怎么算 —— 与摄取层逐字相同
    embedding_text(record)    该编码哪段文本（retrieval_text 优先）—— 只有一处定义
    ```

    "该编码哪段文本"这条规则如果在本模块重写一遍，第一次两边一致、
    第二次有人改了其中一处，症状是"计划里算出的向量键与实际写的向量对不上"——
    而它不报错，只表现为增量更新偶尔漏掉几条。**一条规则只能有一个实现。**

    三个字段的来源各自回答一个问题：

    ```text
    record_id    record_id_of(record)       这是哪一条
    fingerprint  metadata["fingerprint"]    内容变了没有（day062 给的内容身份）
                 ↓ 取不到时
                 content_id(原文 text)      现场按原文算（**不是 retrieval_text**）
    vector_key   vector_key(identity, embedding_text(record))   该不该复用向量
    ```

    注意 ``fingerprint`` 的回落用的是**原文**而不是编码文本：day062 的
    ``fingerprint = content_id(chunk.text)`` 就是原文。两条文本分别对应
    "内容"与"该编码什么"，混用会让"只改检索视图"被误判成"内容变了"。

    ``token_count`` 只用于统计（成本估算），``metadata`` 里给了就带上，
    给不出或不是整数就记 0——它**不进摘要**，取值不影响版本号。
    """
    return IndexEntry(
        record_id=record_id_of(record),
        fingerprint=_fingerprint_of(record),
        vector_key=vector_key(
            identity.provider,
            identity.model,
            identity.dimension,
            embedding_text(record),
        ),
        token_count=_token_count_of(record),
    )


def plan_index(
    previous: IndexManifest | None,
    records: Sequence[dict[str, Any]],
    identity: EmbeddingIdentity,
) -> IndexPlan:
    """算这一批记录相对上一版清单的差集（判据见模块 docstring）.

    四条判据的**顺序**不能换，每一条的"为什么在这个位置"：

    ```text
    previous 为空                  → 全部 added（没有上一版，每一条都是新的）
    previous 里有、新记录里没有    → removed（先判"谁不在了"，与在不在集里无关）
    fingerprint 变了               → updated（内容变了，向量必须重算）
    vector_key 变了                → updated（编码器身份或编码文本变了）
    两者都没变                     → unchanged（**复用向量**，本课省钱的来源）
    ```

    先判 ``fingerprint`` 还是先判 ``vector_key`` 不影响结果（两个都变也只记一次
    ``updated``），但两者必须**分别判**：合并成一个"变了吗"的判断，
    就会按上表漏掉某一类变更。列表按 id 升序排序后再交给 ``IndexPlan``：
    它的构造器会校验"升序、无重复、跨动作不重叠"，那些校验是**兜底**，
    不该由正常路径去触发（触发一次就说明本函数写错了）。

    同一 id 在 ``records`` 里出现两次时后者覆盖前者：id 的唯一性是
    day062 ``chunk_id`` 的契约，本层报错也救不回被覆盖的那条。
    """
    entries: dict[str, IndexEntry] = {}
    for record in records:
        entry = entry_from_record(record, identity)
        entries[entry.record_id] = entry

    if previous is None:
        plan = IndexPlan(added=tuple(sorted(entries)))
    else:
        prior = previous.entry_map()
        added: list[str] = []
        updated: list[str] = []
        unchanged: list[str] = []
        for record_id, entry in entries.items():
            before = prior.get(record_id)
            if before is None:
                added.append(record_id)
            elif before.fingerprint != entry.fingerprint:
                updated.append(record_id)
            elif before.vector_key != entry.vector_key:
                updated.append(record_id)
            else:
                unchanged.append(record_id)
        plan = IndexPlan(
            added=tuple(sorted(added)),
            updated=tuple(sorted(updated)),
            removed=tuple(sorted(record_id for record_id in prior if record_id not in entries)),
            unchanged=tuple(sorted(unchanged)),
        )
    return replace(plan, reason=plan_reason(previous, identity, plan))


def plan_reason(
    previous: IndexManifest | None,
    identity: EmbeddingIdentity,
    plan: IndexPlan,
) -> str:
    """把差集翻译成一句"看到这个数字之后该做什么"的话.

    三种必须说清的情况，前两种**不能只看数字**：

    ```text
    previous 为空        → 首次构建：全部要编码，没有可复用的东西
    身份变了             → **配置变更，不是数据变更**（见下）
    正常的增量 / 无变化   → 按四条计数如实描述
    ```

    第二种是这个函数存在的**主要理由**：身份一变，所有 ``vector_key``
    都变，所有条目落进 ``updated``——报告上写着"更新 1200 条"，
    看起来像"数据动了很多"。但真相是编码器换了，**这一次重建是配置的
    后果而不是数据的后果**，两者的下一步动作完全不同（前者要重新标定
    阈值，后者不用）。这句话必须由本函数说出，因为 ``IndexPlan`` 里
    只剩下一批 id，已经看不出原因了。
    """
    if previous is None:
        return (
            f"首次构建：{plan.total} 条全部需要编码"
            f"（编码器 {identity.summary_line()}），没有可复用的条目"
        )
    if previous.identity.key != identity.key:
        return (
            f"编码器身份变了（旧 {previous.identity.summary_line()} → "
            f"新 {identity.summary_line()}）：全部 {plan.total} 条需重算，"
            "这不是数据变更——换模型或换输出维度必须整库重建，"
            "旧向量不能再复用"
        )
    if plan.changed == 0:
        return (
            f"没有变化：{plan.total} 条全部复用（复用率 {plan.reuse_ratio:.2%}），"
            "本次不需要任何编码调用"
        )
    return (
        f"增量更新：新增 {len(plan.added)} 条 / 更新 {len(plan.updated)} 条"
        f"（内容或编码文本变了）/ 删除 {len(plan.removed)} 条，"
        f"复用 {len(plan.unchanged)} 条（复用率 {plan.reuse_ratio:.2%}）"
    )


# --------------------------------------------------------------------------- #
# 内部工具：两处"取哪个字段"的口径（与 pipeline 保持一致）
# --------------------------------------------------------------------------- #


def _metadata_of(record: dict[str, Any]) -> dict[str, Any]:
    """取记录的 ``metadata``；不是字典时当成空字典.

    ``pipeline._metadata_of`` 是私有的，本模块没有复用它的入口，
    因此在这里留一份**行为逐字相同**的镜像（非字典 → 空字典）。
    这不是"又一处定义"：它不决定任何业务口径，只做类型收窄。
    """
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _raw_text_of(record: dict[str, Any]) -> str:
    """取**原文** ``text``（先看 ``metadata``，再看记录顶层）.

    与 ``pipeline.embedding_text`` 的差别是"取哪一个字段"：那里取的是
    编码视图（``retrieval_text`` 优先），这里取的是原文（``text``）。
    两个位置都查，理由与 pipeline 相同：同一个字段可能按产出方不同
    落在两层中的任一层，取值函数必须同时服务两种位置。
    """
    metadata = _metadata_of(record)
    if FALLBACK_TEXT_FIELD in metadata:
        value: Any = metadata[FALLBACK_TEXT_FIELD]
    else:
        value = record.get(FALLBACK_TEXT_FIELD)
    return "" if value is None else str(value)


def _fingerprint_of(record: dict[str, Any]) -> str:
    """内容身份：优先 ``metadata["fingerprint"]``，没有才按原文现算.

    ``metadata`` 里那个值是 day062 算好的（``content_id(chunk.text)``），
    用它而不是重算，是为了让"内容身份"在整条链路上**只有一次计算**：
    重新算一遍在今天就等价，但等到 day062 的规范化规则改动时
    （那会让全量 ``doc_id`` 失效），两处实现会给出两个答案。
    """
    declared = _metadata_of(record).get("fingerprint")
    value = "" if declared is None else str(declared).strip()
    if value:
        return value
    return content_id(_raw_text_of(record))


def _token_count_of(record: dict[str, Any]) -> int:
    """取 ``token_count``（只用于统计，不进摘要）.

    ``metadata`` 优先、顶层兜底；取不到或不是整数记 0。**不做四舍五入**：
    一个 ``"320.7"`` 这样的值说明上游口径有问题，静默取整会让报告里的
    成本估算看起来比实际精确。
    """
    raw = _metadata_of(record).get("token_count", record.get("token_count"))
    if raw is None:
        return 0
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 0
    return value if value >= 0 else 0


__all__ = [
    "entry_from_record",
    "plan_index",
    "plan_reason",
]
