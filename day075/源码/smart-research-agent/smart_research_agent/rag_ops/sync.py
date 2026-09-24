"""增量同步：两份快照的差集，以及"这次该怎么建"的那一个决定（M6-D10 / day072）.

本模块只有两个纯函数，加上一个把差集翻成人话的小工具。三个都不是"流程"，
而是"判据"——流程在 ``pipeline`` 里，它按这里的结论去调 day065 的构建器。

```text
plan_sync(previous, current)   两次快照 → 一份四组差集（增 / 改 / 删 / 未变）
suggest_mode(plan, threshold)  差集 + 变更阈值 → "整库重建"还是"按增量走"
changed_lines(plan)            差集 → 一份逐条人话（报告与演示脚本用）
```

## 为什么差集要在这里算，而不交给 day065 的 ``plan_index``

``plan_index`` 比的是**块**，它需要**当前这一批记录**与**上一版清单**——
也就是说，它必须先"把语料全部解析、全部切块"，才能回答"要重算哪些向量"。
本模块比的是**文件**，只要两圈扫盘的结果就够了。这个先后关系决定了：

```text
语料一份没变   → 本模块先给出"不用跑" → 一行都没解析、没切块、没编码
day065 的增量  → 语料变了之后，在**同一批**语料内部把重算量压到最小
```

把两件事合成一件，就只剩两种做法，而两种都错：要么每次都跑一遍完整解析
（"没变"这件事被浪费掉），要么只按文件粒度重建（每次改一个字就把整份文档
的所有块重算一遍）。**两级增量各管一段**才是这套设计的完整形态。

## ``suggest_mode`` 为什么只是一个建议

``plan_sync`` 的输出是**事实**（谁变了），``suggest_mode`` 的输出是**决定**
（这次怎么建）。决定被单独抽成一个函数，是因为它要被记录、被覆盖、被讨论：

```text
返回 None                    按 settings.indexing_mode 走（缺省 incremental）
返回 BUILD_MODE_FULL         变更比例超过阈值，整库重建更省
```

阈值来自 ``settings.rag_ops_churn_full_rebuild``（缺省 0.5）。它与 day065 的
``indexing_full_rebuild_threshold`` 是**同一个口径的两个使用点**，
而不是重复：那边按**块**算（回答"要不要清空重写库"），
这边按**文件**算（回答"要不要连解析与分块都省掉"）——
文件级先判一次的价值在于，它能在"一半以上的文档都换了"时
避免一轮完整的解析与切块（那是这一层最贵的一段）。
"""

from __future__ import annotations

from smart_research_agent.config import settings
from smart_research_agent.indexing.builder import BUILD_MODE_FULL, BUILD_MODE_INCREMENTAL
from smart_research_agent.rag_ops.errors import SyncError
from smart_research_agent.rag_ops.types import (
    ACTION_ADDED,
    ACTION_REMOVED,
    ACTION_UNCHANGED,
    ACTION_UPDATED,
    SYNC_ACTION_DESCRIPTIONS,
    CorpusSnapshot,
    SyncPlan,
)


def plan_sync(previous: CorpusSnapshot | None, current: CorpusSnapshot) -> SyncPlan:
    """把两份快照折成一份四组差集（纯函数：不读盘、不碰网络、不看 settings）.

    ``previous`` 为 ``None`` 表示**没有水位**（首次同步，或账本被删了），
    此时全部来源都是"新增"——这正是"账本丢了会全量重跑一次"的实现，
    也是为什么账本丢了不算事故（重跑幂等）。

    四条判据的顺序就是四条动作的顺序，且每一条都必须有：

    ```text
    路径只在 current 里             → added
    路径只在 previous 里            → removed
    两边都有、指纹不同              → updated
    两边都有、指纹相同              → unchanged
    ```

    指纹相同即"没变"这一条，是本模块全部节省的来源：它让"改了一份文档"
    只引起那一份文档的重建，而不是整个语料库的重建。
    """
    before = previous if previous is not None else CorpusSnapshot()
    before_map = before.by_source()
    after_map = current.by_source()

    added = sorted(set(after_map) - set(before_map))
    removed = sorted(set(before_map) - set(after_map))
    updated: list[str] = []
    unchanged: list[str] = []
    for source in sorted(set(before_map) & set(after_map)):
        if before_map[source].fingerprint == after_map[source].fingerprint:
            unchanged.append(source)
        else:
            updated.append(source)

    if previous is None:
        reason = (
            f"没有水位（首次同步或账本不存在）：{len(added)} 份来源按**新增**处理，"
            "本次会走一次完整的解析与入库"
        )
    elif not (added or removed or updated):
        reason = f"水位 {before.digest} 与当前 {current.digest} 一致：一份来源都没变"
    else:
        reason = (
            f"水位 {before.digest} → {current.digest}："
            f"新增 {len(added)}、更新 {len(updated)}、删除 {len(removed)}、"
            f"未变 {len(unchanged)}"
        )

    return SyncPlan(
        added=tuple(added),
        updated=tuple(updated),
        removed=tuple(removed),
        unchanged=tuple(unchanged),
        previous_id=before.digest if previous is not None else "",
        current_id=current.digest,
        previous_sources=before.sources,
        current_sources=current.sources,
        reason=reason,
    )


def suggest_mode(plan: SyncPlan, *, threshold: float | None = None) -> str | None:
    """按变更比例建议构建模式：``None``（照配置走）或 ``BUILD_MODE_FULL``.

    只在"变更比例**超过**阈值"时给建议，且给的建议永远是"整库重建"——
    反向的建议（"其实你可以走增量"）没有意义：增量的代价永远不会比全量高，
    而全量的收益是"少掉一层对账"。因此本函数要么不提，要么提全量。

    边界取值 ``threshold=0.0`` 会让任何一次变化都触发全量重建；
    ``threshold=1.0`` 会让它永不触发（变更比例的上界就是 1.0）。
    两种都不是"关掉它"的写法——要关掉请传 ``None`` 之外的方式：
    直接把本函数的结果丢掉（``mode=None`` 传进构建器），而不是调阈值。
    """
    limit = settings.rag_ops_churn_full_rebuild if threshold is None else float(threshold)
    if not 0.0 <= limit <= 1.0:
        raise SyncError(
            f"整库重建阈值必须落在 [0, 1]，收到 {limit}："
            "它是变更比例的**比例**，超过 1 的阈值永远不会触发（而那看起来只是'没生效'）。"
        )
    if plan.empty:
        return None
    if plan.churn_ratio > limit:
        return BUILD_MODE_FULL
    return None


def resolved_mode(plan: SyncPlan, *, threshold: float | None = None) -> tuple[str, str]:
    """把 :func:`suggest_mode` 的建议折成"最终用哪个模式 + 一句理由".

    ``None`` 的建议在这里被解释成 :data:`BUILD_MODE_INCREMENTAL` 并**带上理由**：
    报告里必须能读到"这次为什么走增量"，否则"增量"与"忘了换挡"长得一样。
    """
    suggestion = suggest_mode(plan, threshold=threshold)
    if suggestion == BUILD_MODE_FULL:
        return (
            BUILD_MODE_FULL,
            f"变更比例 {plan.churn_ratio:.1%} 超过阈值，本次按**整库重建**"
            "（逐条 upsert 的开销在高变更比例下反而高于清空重写）",
        )
    return (
        BUILD_MODE_INCREMENTAL,
        f"变更比例 {plan.churn_ratio:.1%} 未超过阈值，本次按**增量**对账"
        "（未变的块会由 day065 的 planner 判成 unchanged，不重新编码）",
    )


def changed_lines(plan: SyncPlan, *, limit: int = 20) -> list[str]:
    """把差集翻成逐条人话（**按动作分组**，动作内的顺序与报告一致）.

    ``limit`` 是**总数上限**，不是每组的上限：四组各截断到 ``limit`` 条
    会让"最多显示 20 条"这句话在四组都满的时候变成 80 条——而那个函数的
    调用方（报告与演示脚本）要的恰恰是"输出多长是可预期的"。
    被截断时会补一条说明，免得读的人以为"就这些"。
    """
    if limit < 0:
        raise SyncError(f"changed_lines 的 limit 不能为负：{limit}。")
    marks = {
        ACTION_ADDED: "+",
        ACTION_UPDATED: "~",
        ACTION_REMOVED: "-",
        ACTION_UNCHANGED: "=",
    }
    total = plan.changed_count + len(plan.unchanged)
    lines: list[str] = []
    for action in (ACTION_ADDED, ACTION_UPDATED, ACTION_REMOVED, ACTION_UNCHANGED):
        for source in plan.sources(action):
            if len(lines) >= limit:
                break
            lines.append(f"{marks[action]} {source}  — {SYNC_ACTION_DESCRIPTIONS[action]}")
    hidden = total - len(lines)
    if hidden > 0:
        lines.append(f"…另有 {hidden} 条未列出（本行只截断展示，不影响判定）")
    return lines


__all__ = [
    "changed_lines",
    "plan_sync",
    "resolved_mode",
    "suggest_mode",
]
