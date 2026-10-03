"""``flat`` 后端：一张字典 + 一次全量扫描，作为另外两个后端的**参照实现**（M6-D3）.

``flat`` 存在的理由不是"它快"，恰恰相反——**它慢、且慢得可控**：

```text
faiss    近似索引（IVF/HNSW 之后）  换一个索引类型就可能换一批结果
chroma   自带 HNSW + 自带的距离口径  依赖一个外部服务的版本与默认参数
flat     一次全量点积                结果只由 metrics.py 的公式决定
```

因此这一课把它当作**基准**：``faiss`` 与 ``chroma`` 的任何差异都要拿它对照。
这一条在所有验证里都会用到（``evaluate.verify_parity``、各后端的测试），
所以它必须满足两个比"快"重要得多的性质：

```text
零可选依赖    只用标准库 → 在任何一台机器上都能跑（CI 里最可靠的那一个）
逐位可复现    ids() 升序、排序规则与 types.sort_hits 相同 → 两次运行给出同一份结果
```

## 它放弃了什么（这些代价是刻意的）

| 放弃的东西 | 具体代价 | 为什么在这里可以接受 |
|-----------|---------|--------------------|
| 近似检索 | 每次查询都要扫全库（O(N·d)） | 它的职责是给出正确答案，不是给出快答案 |
| 增量落盘 | ``persist`` 每次都重写整份 JSON | 快照是"可读的证据"，不是生产存储（见下） |
| 并发 | 没有锁，多线程写会破坏 ``_records`` 的一致性 | 单进程脚本与测试才是它的使用场景 |
| 部分写入 | 一次 ``upsert`` 要么全算完要么报错 | 批量原子性比"尽量成功"更容易解释 |

## 内部就一张表，而且**向量与元数据在同一张表里**

```text
self._records: dict[str, VectorRecord]
   record_id ─┬─► vector     排序时读它
              └─► text/metadata   命中之后读它
```

两者在同一个对象上，所以本后端的 ``native_metadata`` 是 ``True``：
``_find_ids`` 直接扫这张表就够了，不需要"先问索引、再回表"的两趟。
**这也是它不适合当生产后端的原因**：这张表必须整个放进内存。

## 快照是"两个库"这个问题最直白的答案

``base.py`` 的模块 docstring 说"向量库其实是两个库"。
FAISS 把这句话写在了两个文件里（索引文件 + 旁挂记录文件），
``flat`` 则把它压进**同一份 JSON 的一个数组**里：

```json
{"version": 1, "backend": "flat", "metric": "cosine", "dimension": 8, "count": 6,
 "records": [{"record_id": "...", "vector": [...], "text": "...", "metadata": {...}}]}
```

JSON 而不是 pickle：快照的价值在于**它能被人打开看一眼**——
一份读不懂的快照，在"为什么重启之后少了两条"这种问题前面毫无帮助。

## 谁依赖它

```text
registry（D 号）        ``vector_backend="flat"`` 是缺省值
evaluate.verify_parity  三个后端互相对账时的基准
tests/                  六个测试文件都用它做"正确答案"
scripts/vectorstore_demo.py  演示脚本的缺省后端
```
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from smart_research_agent.vectorstore import metrics as metrics_module
from smart_research_agent.vectorstore.base import VectorBackend
from smart_research_agent.vectorstore.errors import (
    BackendUnavailable,
    VectorError,
    VectorStoreError,
)
from smart_research_agent.vectorstore.filters import compile_filter
from smart_research_agent.vectorstore.types import VectorRecord, WriteReport

#: 快照格式版本。什么时候才该动它是有明确边界的：
#:
#: ```text
#: 加字段（读的时候给默认值）   → 不用升版：旧快照仍然读得懂
#: 改字段含义 / 删字段 / 换编码 → 必须升版：旧快照读进来是"错得看不出来"
#: ```
#:
#: 因此 ``load`` 的第一件事就是核对它，而不是"尽量读读看"——
#: 把一份语义已经变了的旧快照当成新快照读进来，症状是**检索结果整体错位**，
#: 而没有任何一条异常指向"快照格式换了"。
SNAPSHOT_VERSION = 1


class FlatVectorStore(VectorBackend):
    """暴力精确检索的向量库（参照实现）.

    六个原语全部在本类里实现，公共行为（校验、归一化、过滤、阈值、报告）
    仍然由 ``VectorBackend`` 提供——**这正是"参照实现"的意义**：
    它不覆写任何公共行为，所以它与 faiss/chroma 的差异只可能来自
    "一次向量检索"和"一次持久化"，而不是来自三层各写一遍的校验。

    构造期的维度校验（``dimension`` 必须 >= 1）也属于这些公共行为：
    ``base.VectorBackend._check_dimension`` 是**唯一**的实现，
    本类不再重复覆写它——校验留在基类，才会对"将来新增的第四个后端"也成立。
    """

    name = "flat"

    #: 全靠 Python 的 ``==`` 与字典查找，没有任何可选依赖。
    requires: tuple[str, ...] = ()

    supports_filter = True
    supports_delete = True
    supports_persistence = True
    #: 向量与元数据在同一张记录表里（见模块 docstring）。
    native_metadata = True

    def __init__(
        self,
        *,
        metric: str = metrics_module.METRIC_COSINE,
        dimension: int | None = None,
        path: str = "",
    ) -> None:
        """构造一个空库；``path`` 指向的快照若已存在就**自动载入**.

        为什么是"自动 load"而不是"显式调 load()"：调用方写
        ``FlatVectorStore(path=...)`` 时的意图几乎不可能是"我只要路径、
        不要数据"。让"打开一个已存在的库"变成一步操作，可以少掉一种
        真实存在的故障——**忘了 load，于是对着空库跑完了一整轮评估**，
        而那种评估的报告看起来只是"检索效果很差"。
        """
        super().__init__(metric=metric, dimension=dimension, location=path)
        self._records: dict[str, VectorRecord] = {}
        if path and Path(path).exists():
            self.load(path)

    # ------------------------------------------------------------------ 原语

    def count(self) -> int:
        """库里的记录条数（字典长度，O(1)）."""
        return len(self._records)

    def ids(self) -> list[str]:
        """全部 id，**升序**.

        ``sorted`` 而不是 ``list(self._records)``：字典的迭代顺序是插入序，
        而插入序会随"批量怎么分批"变化。升序让"这批 id"成为一个
        **可以逐位比较的对象**——两次导入同一份数据之后 ``ids()`` 相等，
        是索引可复现性最基本的检查（见 ``base.ids`` 的说明）。
        """
        return sorted(self._records)

    def get(self, record_id: str) -> VectorRecord | None:
        """取一条记录；不存在返回 ``None``（查不到是正常结果，不是异常）."""
        return self._records.get(record_id)

    def _upsert(self, records: Sequence[VectorRecord]) -> WriteReport:
        """逐条判定 added / updated / unchanged 并写入.

        判据是"与新记录逐项比较"，三项**全部相同才算 unchanged**：

        ```text
        vector    逐位相等（``tuple`` 的 ``==``）
        text      完全相同
        metadata  字典相等（含其中的数组逐元素比较）
        ```

        ``unchanged`` 的记录**不写回库**（``continue``），这一条不是省一次
        赋值那么简单：day065 的增量索引要靠"库里那个对象没被换过"来判断
        "这一块的向量不需要重算"。如果这里改成"无条件写回一个新对象"，
        观察到的计数仍然是 ``unchanged=6``，但**对象的身份变了**，
        而所有基于身份的复用都会静默失效。

        ``metadata`` 里含数组时用的是 Python 的 ``==``：它与
        ``base._prepare`` 的重建过程是同一个口径（两边都来自
        ``types.make_record``），因此"重放同一批"必然落到 ``unchanged``。
        """
        added = updated = unchanged = 0
        for record in records:
            existing = self._records.get(record.record_id)
            if existing is None:
                added += 1
            elif self._same_record(existing, record):
                unchanged += 1
                # 刻意不写回：见 docstring（对象身份是 day065 的地基）。
                continue
            else:
                updated += 1
            self._records[record.record_id] = record
        return WriteReport(added=added, updated=updated, unchanged=unchanged)

    @staticmethod
    def _same_record(existing: VectorRecord, incoming: VectorRecord) -> bool:
        """两条记录是否**逐项相同**（``_upsert`` 判 ``unchanged`` 的唯一依据）.

        三个字段分开比较而不是比较整个 ``VectorRecord``：后者依赖
        dataclass 自动生成的 ``__eq__``，一旦将来给记录加一个"仅用于展示"
        的字段（比如入库时间），整条比较会**静默地永远不相等**，
        于是每次重放都报 ``updated``——而 day065 正是要看这个数字。

        向量用 ``tuple`` 的 ``==``：这是逐位比较，也是本包对"相同"
        唯一说得清的定义。唯一的边界是 ``0.0 == -0.0`` 为真，
        但写入路径上两条记录都经过同一个 ``make_record``，
        不会出现"一边 +0.0、一边 -0.0"这种输入。
        """
        return (
            existing.vector == incoming.vector
            and existing.text == incoming.text
            and existing.metadata == incoming.metadata
        )

    def _remove(self, ids: Sequence[str]) -> int:
        """删除给定 id，返回**真正删掉的条数**.

        不存在的 id 静默跳过：删除是幂等操作，删一个已经被删掉的 id
        不该报错。但返回值必须诚实——"请求删 5 条、实际删掉 0 条"
        是一个调用方需要看见的事实（与 ``WriteReport.unchanged`` 同一思路）。
        """
        removed = 0
        for record_id in ids:
            if record_id in self._records:
                del self._records[record_id]
                removed += 1
        return removed

    def _ranked_ids(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        restrict: frozenset[str] | None = None,
    ) -> list[tuple[str, float]]:
        """暴力打分 + 排序，返回最多 ``limit`` 个 ``(id, score)``.

        排序规则与 ``types.sort_hits`` **逐字相同**：分数降序、同分按 id 升序。
        第二关键字在这里是必需的而不是装饰——同一段文本用确定性编码器
        会得到逐位相同的向量，于是分数**精确相等**；没有第二关键字时，
        顺序取决于 ``dict`` 的迭代顺序（即插入顺序），
        而插入顺序在分批重放之后会变。

        ``restrict`` 的处理是本方法唯一需要小心的部分：**先在候选集里打分、
        再排序、最后截断**，而不是"先取全库 top-limit 再和 restrict 求交集"。
        后者会少返回——被截断掉的那些本来排得进 top-limit，只是不在
        restrict 里；而少返回不报错，只表现为"加了过滤之后结果变少了"。
        """
        if restrict is None:
            items: Any = self._records.items()
        else:
            items = (
                (record_id, self._records[record_id])
                for record_id in restrict
                if record_id in self._records
            )
        scored = [
            (record_id, metrics_module.score(self.metric, vector, record.vector))
            for record_id, record in items
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:limit]

    def _find_ids(self, where: dict[str, Any] | None) -> set[str] | None:
        """用 ``filters.compile_filter`` 扫记录表（本后端原生存元数据）.

        覆写 ``base._find_ids`` 只是省掉一层 ``get()`` 间接调用；
        **语义必须与默认实现完全一致**，否则 delete(where=...) 与
        query(where=...) 会给出不同的候选集，而那种不一致非常难查。
        """
        predicate = compile_filter(where)
        return {
            record_id
            for record_id, record in self._records.items()
            if predicate(record.metadata)
        }

    # ------------------------------------------------------------------ 元信息

    def describe_extra(self) -> dict[str, str]:
        """``/vectorstore/stats`` 的后端专属字段.

        ``snapshot_version`` 放进这里而不是放进 ``StoreInfo`` 的固定字段：
        它是 ``flat`` 特有的概念，FAISS 的快照由它自己管、Chroma 根本没有
        "快照"这个东西。**把后端的私有概念塞进公共形状，
        会让另外两个后端永远带着一个空字符串。**
        """
        return {
            "records": str(self.count()),
            "snapshot_version": str(SNAPSHOT_VERSION),
            "storage": "dict（向量与元数据同表）",
        }

    # ------------------------------------------------------------------ 持久化

    def persist(self, path: str | None = None) -> str:
        """把整库写成一份 JSON 快照，返回实际写入的路径.

        三个刻意的决定：

        - **每次全量重写**：没有"追加"这种模式。一份快照要么完整要么不存在，
          不存在"读的时候发现少了几条"的中间态；
        - **``parents=True, exist_ok=True``**：第一次运行时目录通常还不存在，
          把建目录放在这里，调用方就不必记住"先 mkdir"；
        - **记录按 id 升序写**：让同一次状态的两次快照**逐字节相同**。
          调试"为什么两次快照不一样"时，diff 里如果全是顺序变化，
          真正的差异就被淹没了。

        未给路径且构造时 ``path`` 为空 → ``BackendUnavailable``：
        绝不静默降级成"写到哪里去了都不知道"。
        """
        target = str(path or self.location or "")
        if not target:
            raise BackendUnavailable(
                "flat 后端没有可写的路径：构造时 path 为空，本次也没有传入 path。"
                "请用 FlatVectorStore(path=...) 或 persist(path=...) 指定落盘位置。"
            )
        file_path = Path(target)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "version": SNAPSHOT_VERSION,
            "backend": self.name,
            "metric": self.metric,
            "dimension": self.dimension,
            "count": self.count(),
            "records": [
                {
                    "record_id": record_id,
                    "vector": list(self._records[record_id].vector),
                    "text": self._records[record_id].text,
                    "metadata": dict(self._records[record_id].metadata),
                }
                for record_id in sorted(self._records)
            ],
        }
        file_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._location = str(file_path)
        return self._location

    def load(self, path: str | None = None) -> None:
        """读回快照；**任何不一致都在写入内存之前报错**（库保持原样）.

        三道校验，各挡一种"读进来之后错得看不出来"：

        ```text
        version    格式换了          → VectorStoreError（报文里点明"快照版本"）
        metric     度量不同          → VectorError（同一批向量在两种度量下近邻不同）
        dimension  维度不同          → VectorError（库与编码器不是同一套）
        ```

        校验按"从最外层到最内层"的顺序做：**先证明这份文件是不是我能读的，
        再判断它是不是我该读的**。反过来写的话，一份打不开的文件会先被
        拿去做度量比较，报出来的错就指向了一个跟真正病因无关的地方。

        清空再载入而不是合并：``load`` 的语义是"把库变成磁盘上那份状态"，
        合并会让"库里多出来的那几条"永远删不掉，而调用方以为它们是旧的。
        """
        target = str(path or self.location or "")
        if not target:
            raise BackendUnavailable(
                "flat 后端没有可读的路径：构造时 path 为空，本次也没有传入 path。"
            )
        file_path = Path(target)
        if not file_path.exists():
            raise BackendUnavailable(f"flat 快照文件不存在：{target}")
        payload = json.loads(file_path.read_text(encoding="utf-8"))

        version = payload.get("version")
        if version != SNAPSHOT_VERSION:
            raise VectorStoreError(
                f"快照版本与当前代码不匹配：文件是 version={version!r}，"
                f"当前代码是 version={SNAPSHOT_VERSION}。"
                "请用与快照同版本的代码读取，或用当前代码重新生成快照——"
                "强行读取会让字段含义错位，而症状只是检索结果整体变差。"
            )

        snapshot_metric = metrics_module.normalize_metric(str(payload.get("metric")))
        if snapshot_metric != self.metric:
            raise VectorError(
                f"快照的 metric={snapshot_metric!r} 与本实例的 metric={self.metric!r} "
                "不一致：同一批向量在两种度量下的最近邻不是同一批。"
                f"出路是**用 snapshot 的 metric 新建实例**"
                f"（例如 FlatVectorStore(metric={snapshot_metric!r}, path=...)），"
                "而不是让一个已有实例改度量。"
            )

        snapshot_dimension = int(payload.get("dimension", 0))
        if self.has_dimension and snapshot_dimension != self.dimension:
            raise VectorError(
                f"快照的 dimension={snapshot_dimension} 与本实例的 dimension="
                f"{self.dimension} 不一致：库与当前编码器不是同一套。"
                f"出路是**用 snapshot 的 dimension 新建实例**"
                "（或对相同 dimension 的实例 load），不要截断或补零。"
            )

        records = payload.get("records", [])
        declared = int(payload.get("count", len(records)))
        if declared != len(records):
            raise VectorStoreError(
                f"快照自相矛盾：count={declared}，但 records 有 {len(records)} 条。"
                "这是一份被截断或手改过的文件，不能作为建库依据。"
            )

        loaded: dict[str, VectorRecord] = {}
        for item in records:
            record = VectorRecord(
                record_id=item["record_id"],
                vector=tuple(float(value) for value in item["vector"]),
                text=str(item.get("text", "")),
                metadata=dict(item.get("metadata", {})),
            )
            loaded[record.record_id] = record
        self._records = loaded
        if not self.has_dimension and snapshot_dimension:
            self._dimension = snapshot_dimension
        self._location = str(file_path)


def build_flat_store(
    records: Sequence[VectorRecord] | None = None,
    *,
    metric: str = metrics_module.METRIC_COSINE,
    dimension: int | None = None,
    path: str = "",
) -> FlatVectorStore:
    """建一个 flat 库（可选带一批初始数据）——端点与演示脚本的统一入口.

    初始数据走 ``upsert`` 而不是 ``add``：``build_*`` 的语义是
    "给我一个状态是这样的库"，不是"这批数据必须是新的"。
    如果 ``path`` 指向的快照已经被自动载入，``upsert`` 会让这次调用
    变成一次幂等的对齐（重新跑一遍脚本不会报错）。
    """
    store = FlatVectorStore(metric=metric, dimension=dimension, path=path)
    if records:
        store.upsert(records)
    return store


__all__ = [
    "SNAPSHOT_VERSION",
    "FlatVectorStore",
    "build_flat_store",
]
