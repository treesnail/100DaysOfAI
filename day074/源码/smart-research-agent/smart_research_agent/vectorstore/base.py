"""后端抽象：把"向量库"拆成**两个库**，并规定它们的公共行为（M6-D3）.

在写这个文件之前，最值得先说清楚的一件事是：

```text
一个向量库其实是两个库拼起来的
├─ 向量索引    只认 (int64 id, float32 向量)      → 负责"谁离得最近"
└─ 记录表      id → (原文, 元数据)                → 负责"命中的那条是什么"
```

Chroma 把两者做在一个进程/一个目录里（``documents`` 与 ``metadatas``
直接跟着 ``embeddings`` 存），**FAISS 只做前者**——它的 ``IndexFlatIP``
甚至不存 id，必须先用 ``IndexIDMap2`` 包一层。
真实项目里配 FAISS 的两件套是"FAISS 管向量 + SQLite 管文本"
（这也是社区最常见的那种 ``faiss_store.py``）。

因此本包的 ``VectorBackend`` 抽象里同时存在这两层语义，
并且**把"谁是主"这件事定死**：

> **记录表是主，向量索引是从。**
> 删除以记录表为准（先删表，再让索引跟随）；查询以记录表为准
> （命中 id 之后回来取原文与元数据）。

反过来设计（以索引为主）会出现一个很难查的故障：
索引里还有那条向量、但记录表里已经没有它的原文了，
于是检索命中一个"没有内容的 id"——**要么在应用层抛 KeyError，
要么返回一条空文本的命中**，而两种表现都不指向"删除没做干净"。

## 四个抽象原语，以及为什么是这四个

后端只需实现四个东西，公共行为（校验、过滤、阈值、排序、报告）全部在本文件里：

```text
count()                                  有多少条
ids()                                    都是谁（升序，保证可复现）
get(record_id)                           取一条完整记录
_upsert(records) -> WriteReport          写入（已校验、已归一化）
_remove(ids) -> int                      删除
_ranked_ids(vector, limit, restrict)     向量排序（唯一需要"近似"的地方）
_find_ids(where) -> set[str] | None      元数据预筛（能用原生过滤就覆写）
```

**为什么排序原语的返回值是 id 而不是记录**：向量索引手里只有 id。
如果让它返回记录，每个后端都要自己维护一份"记录表"的引用，
而 Chroma 根本不需要（它自己就是表）——那就变成了重复实现。
让排序只回答"**谁**离得最近"，记录的统一由 ``query`` 取回来，
三个后端的差异就被压缩到"一次向量检索"这一件事上。

## 过滤为什么必须在这一层做

FAISS **没有任何元数据过滤能力**。于是"带过滤的检索"只有一个正确做法：
先按条件筛出候选 id，再在候选里排序。而"先排序取 top-k、再过滤掉不符合的"
会让**返回条数少于 top_k，尽管库里还有更多符合条件的记录排在更后面**
——这个 bug 不报错、只表现为"过滤之后结果变少了"（见 ``faiss_backend``）。

因此 ``_find_ids`` 是抽象的一部分：**能用原生过滤的后端必须覆写它**
（Chroma 会把它翻译成一次 ``collection.get(where=...)``），
不能的就老老实实扫记录表。两者语义必须一致，
由 ``evaluate.verify_parity`` 在测试里逐条对账。

## 这一层刻意不做的事

```text
不管批量编码      → 那是 day065 的 indexing 包（本层只接收算好的向量）
不管分块          → day062 的事
不管重排序        → day068 的事
不管多路召回融合   → day067 的事
```

``scripts/vectorstore_demo.py`` 里有一句原文解释这条边界：
**"本层只回答'给一个向量、返回最像的 K 条'，再往前一步的问题
（向量从哪来、命中之后怎么用）都属于别的模块"**——
这条边界一旦不写清楚，向量库会慢慢长成一个框架。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

from smart_research_agent.vectorstore import filters as filter_module
from smart_research_agent.vectorstore.errors import (
    BackendUnavailable,
    FilterError,
    RecordError,
    VectorError,
)
from smart_research_agent.vectorstore.metrics import METRIC_COSINE, distance, normalize_metric
from smart_research_agent.vectorstore.types import (
    SearchHit,
    SearchResult,
    StoreInfo,
    VectorRecord,
    WriteReport,
    make_record,
    sort_hits,
)

#: 维度未知时的取值。用 0 而不是 ``None``，是为了让"未定维"这件事
#: 在报告里是一个**数字**（``dimension=0``），而不是一个需要分支处理的空值。
UNKNOWN_DIMENSION = 0

#: 默认检索深度。取 5 而不是 3：day062 的探针实验说明
#: top-3 与 top-8 的差距经常来自"检索深度不够"，而不是数据质量本身。
DEFAULT_TOP_K = 5

#: 单次查询允许的最大 ``top_k``（防御性上限）。
#: 一份 10 万块的库取 ``top_k=100000`` 会把整库搬进响应体里。
MAX_TOP_K = 1000

#: 本层的边界（与 documents / chunking 各留一份，风格一致）。
VECTORSTORE_LIMITATIONS: tuple[str, ...] = (
    "只做暴力/近似最近邻检索，不提供倒排索引与稀疏检索（day067 的混合检索另建一层）",
    "不带重排序：命中的顺序完全由向量度量决定（day068 加交叉编码器）",
    "不管批量编码与向量缓存：本层接收算好的向量（day065 的 indexing 包负责）",
    "不做权限过滤：where 是元数据筛选，不是访问控制（那属于 day015 的工具权限层）",
)

#: 明确排除在范围外的能力（写下来，避免"这不算 bug"的争论）。
VECTORSTORE_OUT_OF_SCOPE: tuple[str, ...] = (
    "分布式分片与副本：本层是单进程视角，多副本一致性不在课程范围",
    "训练型索引（PQ/OPQ 的码本训练）：需要样本量与离线训练，属于生产调优",
    "GPU 索引：FAISS 的 GPU 索引不支持 write_index，序列化路径与 CPU 不同",
    "跨语言/跨进程的实时同步：Chroma 的服务端模式留到 day072 的部署一章",
)


class VectorBackend(ABC):
    """向量库后端的公共骨架（见模块 docstring 的四个原语）.

    子类只需要实现 6 个方法，其余行为（校验、归一化、过滤、阈值、
    稳定排序、写入报告、维度一致性）全部由本类提供——
    **这样三个后端才会给出逐条可比的结果**，而不是"看起来差不多"。
    """

    #: 后端名（``registry`` 用它做键，端点会回显它）.
    name: ClassVar[str] = "abstract"

    #: 需要的可选依赖包名（缺失时 ``registry`` 会给出安装指引）.
    requires: ClassVar[tuple[str, ...]] = ()

    #: 是否支持 ``where`` 过滤（本包三个后端都支持，但语义强弱不同）.
    supports_filter: ClassVar[bool] = True

    #: 是否支持按 id 删除.
    supports_delete: ClassVar[bool] = True

    #: 是否能把状态落到磁盘.
    supports_persistence: ClassVar[bool] = False

    #: 元数据是否与向量存在**同一个存储里**（Chroma 是，FAISS 不是）.
    #: 这个标志决定了 ``_find_ids`` 是"扫记录表"还是"问原生库"。
    native_metadata: ClassVar[bool] = False

    def __init__(
        self,
        *,
        metric: str = METRIC_COSINE,
        dimension: int | None = None,
        location: str = "",
    ) -> None:
        self._metric = normalize_metric(metric)
        self._location = location
        self._dimension = UNKNOWN_DIMENSION
        if dimension is not None:
            self._dimension = self._check_dimension(dimension)

    def _check_dimension(self, dimension: int) -> int:
        """校验构造期给定的维度.

        维度从 1 起计数：``0`` 是本包的"维度未定"哨兵（``UNKNOWN_DIMENSION``），
        不是一个可以建库的维度。把这个校验放在基类里而不是各后端里，
        理由是**它必须对第四个后端也成立**——新建一个后端的人不会想起来
        "还得自己写一个维度校验"，而漏掉它的表现是"维度传 0 时不报错"。
        """
        value = int(dimension)
        if value < 1:
            raise VectorError(
                f"dimension 必须是 >= 1 的整数，收到 {dimension!r}。"
                "0 在本包里是「维度未定」的哨兵值（见 UNKNOWN_DIMENSION），"
                "不是一个可以建库的维度。"
            )
        return value

    # ------------------------------------------------------------------ 元信息

    @property
    def metric(self) -> str:
        """本库的度量（构造时定下，运行期不再改变）.

        **不允许运行期切换**：同一批向量在两种度量下有不同的"最近邻"，
        让度量可变会让"换个度量再搜一次"变成一次静默的结果错乱。
        需要另一个度量就用另一个后端实例。
        """
        return self._metric

    @property
    def dimension(self) -> int:
        """向量维度；尚未写入任何记录且构造时未指定时为 0."""
        return self._dimension

    @property
    def has_dimension(self) -> bool:
        """维度是否已经确定."""
        return self._dimension > 0

    @property
    def location(self) -> str:
        """持久化位置（内存后端为空串）."""
        return self._location

    def info(self) -> StoreInfo:
        """当前状态（``/vectorstore/stats`` 直接返回它）."""
        return StoreInfo(
            backend=self.name,
            metric=self.metric,
            dimension=self.dimension,
            count=self.count(),
            persistent=self.supports_persistence and bool(self._location),
            location=self._location,
            extra=self.describe_extra(),
        )

    def describe_extra(self) -> dict[str, str]:
        """后端特有的状态字段（子类覆写；默认空）."""
        return {}

    # ------------------------------------------------------------ 必须实现的原语

    @abstractmethod
    def count(self) -> int:
        """库里的记录条数."""

    @abstractmethod
    def ids(self) -> list[str]:
        """全部记录 id（**升序**：让"这批 id"成为一个可比较的对象）.

        返回升序而不是插入序，是因为插入序会随批量重放而变化，
        而"两次运行的 id 列表相同"是索引可复现性最基本的检查。
        """

    @abstractmethod
    def get(self, record_id: str) -> VectorRecord | None:
        """取一条记录；不存在返回 ``None``（不抛异常：查不到是正常结果）."""

    @abstractmethod
    def _upsert(self, records: Sequence[VectorRecord]) -> WriteReport:
        """写入已校验、已归一化的记录（子类实现）."""

    @abstractmethod
    def _remove(self, ids: Sequence[str]) -> int:
        """删除给定 id 并返回真正删掉的条数.

        **返回"真正删掉的"而不是"请求删掉的"**：删一个不存在的 id
        是幂等操作，不该报错；但报告里必须能看出"这次其实什么都没删"
        （与 ``WriteReport.unchanged`` 是同一个思路）。
        """

    @abstractmethod
    def _ranked_ids(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        restrict: frozenset[str] | None = None,
    ) -> list[tuple[str, float]]:
        """向量排序：返回最多 ``limit`` 个 ``(id, score)``，**分数降序**.

        ``restrict`` 非空时只在这个集合内排序。**必须由后端保证
        "返回的就是这个集合内最像的 limit 条"**——不许先取全库 top-limit
        再与 restrict 求交集（那会少返回，见模块 docstring）。
        """

    def _find_ids(self, where: dict[str, Any] | None) -> set[str] | None:
        """按 ``where`` 预筛候选 id；返回 ``None`` 表示"不过滤".

        默认实现扫记录表（对不使用原生元数据的后端是正确的）；
        使用原生元数据的后端应当覆写它，把条件翻译成库自己的查询。
        """
        predicate = filter_module.compile_filter(where)
        return {record_id for record_id in self.ids() if predicate(self._metadata_of(record_id))}

    def _metadata_of(self, record_id: str) -> dict[str, Any]:
        """取一条记录的元数据（默认实现走 ``get``；子类可覆写为更省的一趟）."""
        record = self.get(record_id)
        return dict(record.metadata) if record is not None else {}

    # -------------------------------------------------------------- 公共写入路径

    def upsert(self, records: Iterable[VectorRecord]) -> WriteReport:
        """写入或覆盖（**推荐入口**：重跑同一批数据不会报错，也不会重复）.

        逐条判定三种去向（与 ``WriteReport`` 的三个字段一一对应）：

        ```text
        库里没有这个 id                       → added
        有，且向量/原文/元数据有一个不同       → updated
        有，且三者逐位相同                    → unchanged（不碰库）
        ```

        ``unchanged`` 这一态是 day065 增量索引的地基：**只有它能回答
        "这次重建实际上省掉了多少次编码"**。
        """
        return self._write(records, allow_existing=True)

    def add(self, records: Iterable[VectorRecord]) -> WriteReport:
        """只新增：任何一个 id 已存在就**整体拒绝**（不做部分写入）.

        与 ``upsert`` 的差别是"批量语义"：``upsert`` 是"把这份数据对齐到库里"，
        ``add`` 是"这批数据必须是新的"。后者在**第一次建库**时用来防错：
        如果建库脚本被误跑第二次，``upsert`` 会把"重复建库"变成一个
        静默的正常结果（全部 ``unchanged``），而 ``add`` 会直接响。

        整体拒绝而不是跳过冲突项：部分写入会让调用方以为整批成功，
        而实际上库里只有一部分数据——**让一次批量操作是原子的**，
        比让它"尽量成功"更重要。
        """
        return self._write(records, allow_existing=False)

    def _write(self, records: Iterable[VectorRecord], *, allow_existing: bool) -> WriteReport:
        prepared = [self._prepare(record) for record in self._iter_records(records)]
        if not allow_existing:
            conflicts = [record.record_id for record in prepared if self.get(record.record_id)]
            if conflicts:
                raise RecordError(
                    f"add() 撞上 {len(conflicts)} 个已存在的 id（例如 {conflicts[0]!r}）："
                    "add 表示'这批数据必须是新的'。要覆盖请用 upsert()。"
                )
        return self._upsert(prepared)

    def _iter_records(self, records: Iterable[VectorRecord]) -> list[VectorRecord]:
        """把入参收敛成一个列表，并把"传了字典"这件事变成明确的报错.

        允许传 ``dict`` 看似方便，实则是把类型错误推迟到运行期深处。
        本层要求**先构造 ``VectorRecord``**：那条构造函数里有关键校验
        （id 长度、元数据类型、向量有限性、零向量）。
        """
        items = list(records)
        for item in items:
            if not isinstance(item, VectorRecord):
                raise RecordError(
                    f"写入的必须是 VectorRecord，收到 {type(item).__name__}。"
                    "字典形状的输入请先经 ``vectorstore.pipeline.record_from_knowledge`` "
                    "或 ``types.make_record`` 转换——那里会做 id/元数据/向量的校验。"
                )
        return items

    def _prepare(self, record: VectorRecord) -> VectorRecord:
        """写入前的统一处理：维度一致性 + 按度量归一化.

        维度**在第一次写入时定下**：之后任何一次不同维度的写入都是
        ``VectorError``，因为那意味着库与编码器不是同一套
        （见 ``errors.VectorError`` 的说明）。空库不设限：那时维度还没有意义。
        """
        if self.has_dimension and record.dimension != self._dimension:
            raise VectorError(
                f"维度不一致：本库是 {self._dimension} 维，收到 {record.dimension} 维"
                f"（记录 {record.record_id!r}）。换过 embedding 提供方就要重建库，"
                "截断或补零都会让排序静默失去意义。"
            )
        prepared = make_record(
            record.record_id,
            list(record.vector),
            record.text,
            dict(record.metadata),
            metric=self.metric,
        )
        if not self.has_dimension:
            self._dimension = prepared.dimension
        return prepared

    # ------------------------------------------------------------------ 公共删除

    def delete(
        self,
        ids: Sequence[str] | None = None,
        *,
        where: dict[str, Any] | None = None,
    ) -> int:
        """按 id 或 ``where`` 删除，返回真正删掉的条数.

        两种方式**必须二选一**：同时给出时无法判断调用方想删的是什么。
        如果允许"两者取交集"，一次 ``delete(ids=[...], where={...})``
        在条件写错时会变成"什么都没删"，而调用方以为删了。
        """
        if not self.supports_delete:
            raise BackendUnavailable(f"后端 {self.name} 不支持删除")
        if ids is not None and where is not None:
            raise RecordError(
                "delete 只能按 ids 或按 where 之一删除，二者不能同时给出："
                "同时给会让'到底删了什么'变得不确定。"
            )
        if ids is not None:
            targets = [str(record_id) for record_id in ids]
            if not targets:
                return 0
            return self._remove(targets)
        if where is not None:
            matched = self._find_ids(where)
            if not matched:
                return 0
            return self._remove(sorted(matched))
        raise RecordError("delete 需要 ids 或 where 中的一个；两者都没给就不会删任何东西")

    def get_many(self, ids: Sequence[str]) -> list[VectorRecord]:
        """批量取记录（**跳过不存在的**，顺序与入参一致）.

        跳过而不是报错：批量取的典型场景是"把上一次的命中重新取回来展示"，
        中间被删掉一条不应该让整次展示失败。要严格校验请自己比对长度。
        """
        found: list[VectorRecord] = []
        for record_id in ids:
            record = self.get(record_id)
            if record is not None:
                found.append(record)
        return found

    # ------------------------------------------------------------------ 公共查询

    def query(
        self,
        vector: Sequence[float],
        top_k: int = DEFAULT_TOP_K,
        *,
        where: dict[str, Any] | None = None,
        min_score: float | None = None,
    ) -> SearchResult:
        """检索：给一个查询向量，返回最像的 ``top_k`` 条（可带元数据过滤与阈值）.

        参数校验的顺序刻意是"先便宜的后贵的"：``top_k`` 越界（一次比较）
        在开库之前就被拦掉；维度不一致要等拿到查询向量才能判断。
        """
        if top_k < 1:
            raise FilterError(
                f"top_k 必须 >= 1，收到 {top_k}。"
                "'要 0 条'不是一个有意义的检索请求；要探测'库里有没有数据'请用 count()。"
            )
        if top_k > MAX_TOP_K:
            raise FilterError(
                f"top_k={top_k} 超过上限 {MAX_TOP_K}：一次把整库搬进响应体没有意义，"
                "要遍历请用 ids() + get_many()。"
            )
        query_vector = [float(x) for x in vector]
        if self.has_dimension and len(query_vector) != self._dimension:
            raise VectorError(
                f"查询向量是 {len(query_vector)} 维，本库是 {self._dimension} 维："
                "查询用的编码器与建库用的不是同一个。"
            )
        if self.metric == METRIC_COSINE:
            query_vector = _l2_normalize(query_vector)

        filter_applied = bool(where)
        restrict = self._find_ids(where) if filter_applied else None
        if restrict is not None and not restrict:
            return SearchResult(
                hits=(),
                metric=self.metric,
                top_k=top_k,
                candidates=0,
                filter_applied=True,
            )

        ranked = self._ranked_ids(query_vector, limit=top_k, restrict=restrict)
        hits: list[SearchHit] = []
        for record_id, raw_score in ranked:
            record = self.get(record_id)
            if record is None:
                # 记录表与向量索引不同步。这里**跳过并继续**而不是抛错：
                # 查询是读路径，读路径不该因为一条脏数据整体失败；
                # 而这件事必须可见——evaluate.verify_parity 会把它查出来。
                continue
            hits.append(
                SearchHit(
                    record=record,
                    score=float(raw_score),
                    distance=distance(self.metric, query_vector, record.vector),
                    rank=0,
                )
            )

        if min_score is not None:
            hits = [hit for hit in hits if hit.score >= float(min_score)]
        ordered = sort_hits(hits)
        return SearchResult(
            hits=tuple(ordered),
            metric=self.metric,
            top_k=top_k,
            candidates=len(restrict) if restrict is not None else self.count(),
            filter_applied=filter_applied,
        )

    def clear(self) -> None:
        """清空（保留度量与维度的设定，便于复用同一个实例重建）."""
        existing = self.ids()
        if existing:
            self._remove(existing)

    # ------------------------------------------------------------------ 持久化

    def persist(self, path: str | None = None) -> str:
        """把状态写到磁盘，返回实际写入的位置.

        ``supports_persistence`` 为 False 或未给路径时抛 ``BackendUnavailable``
        ——**不做静默降级**：一个"以为落盘了其实还在内存里"的向量库，
        会在进程重启后才暴露，而那时你已经删掉了重建脚本。
        """
        raise BackendUnavailable(
            f"后端 {self.name} 不支持持久化（现状：location={self.location!r}）"
        )

    def load(self, path: str | None = None) -> None:
        """从磁盘读回状态."""
        raise BackendUnavailable(f"后端 {self.name} 不支持从磁盘读取")


def _l2_normalize(vector: list[float]) -> list[float]:
    """查询向量的 L2 归一化（与写入侧 ``make_record`` 用的是同一个口径）."""
    total = sum(value * value for value in vector)
    if total == 0.0:
        return list(vector)
    length = total**0.5
    return [value / length for value in vector]


__all__ = [
    "DEFAULT_TOP_K",
    "MAX_TOP_K",
    "UNKNOWN_DIMENSION",
    "VECTORSTORE_LIMITATIONS",
    "VECTORSTORE_OUT_OF_SCOPE",
    "VectorBackend",
]
