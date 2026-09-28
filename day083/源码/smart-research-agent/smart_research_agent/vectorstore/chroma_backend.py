"""Chroma 后端：把本包的六个原语翻译成**一次** Chroma 调用（M6-D3）.

到这一步为止，``flat`` 是"自己实现一切"的参照、``faiss`` 是"只管向量、
其余自己扛"。Chroma 是第三种形态：**它自己就是那张记录表**——
``documents`` 与 ``metadatas`` 直接跟着 ``embeddings`` 存在同一个集合里，
所以 ``native_metadata = True``，而"两个库"这件事在它身上被合并成了一个。

```text
本包的抽象          Chroma 的对应物
-----------------  ------------------------------------------------------------
count()            collection.count()
ids()              collection.get(include=[])                  → sorted()
get(record_id)     collection.get(ids=[...], include=[embeddings, documents, metadatas])
_upsert(records)   collection.upsert(ids, embeddings, metadatas, documents)
_remove(ids)       collection.delete(ids=[...])  → 返回 None，条数靠 count() 前后差
_ranked_ids(...)   collection.query(query_embeddings=[[...]], n_results=深度)
_find_ids(where)   collection.get(where={...}, include=[])["ids"]
query(...)         collection.query(..., where=...)            ← 覆写，见下一节
```

## 为什么必须覆写 ``query()``，而不是只实现 ``_ranked_ids``

``base`` 的公共查询路径是"先 ``_find_ids`` 取候选 id、再 ``_ranked_ids`` 排序"。
对 FAISS 这是唯一正确的做法（它**没有**元数据过滤），对 Chroma 却是
"两趟且更贵"：第一趟把满足条件的 id 全部搬回 Python，第二趟再按向量排序，
而 Chroma 的 ``where`` 是原生的，一次 ``query(n_results=k, where=...)`` 就够。

更关键的是**正确性**，不是性能：

```text
n_results 只有在 where 一起交给原生查询时，才是「过滤之后要的条数」
```

若先排序取 top-k 再过滤，"最像的 k 条里有 k−1 条不满足条件"会让返回条数
少于 k——库里还有更多符合条件的记录，只是它们排在更后面。
这个故障不报错，只表现为"加了过滤之后结果变少了"（见 ``base`` 模块 docstring）。

覆写时仍复用 ``metrics.similarity_from_distance`` 把 ``distances`` 折成
本包的 ``score``，并用 ``types.sort_hits`` 统一排序，**保证与 flat / faiss
逐条一致**——否则"换个后端"会顺手换掉平局的断法。

## 距离：Chroma 的三种 space 都是"越小越近"

```text
space     官方公式                                  与 FAISS 的关系
l2        Σ(ai − bi)²                               IndexFlatL2 的 D
ip        1 − Σ(ai·bi)                              IndexFlatIP 的 D 是**正号**，恰好相反
cosine    1 − Σ(ai·bi)/(√Σai²·√Σbi²)                需要自己归一化再取内积
```

因此 ``SearchHit.distance`` 直接透传 Chroma 返回的那个数，
``score`` 由 ``similarity_from_distance`` 折出来。**不自己再算一遍距离**：
再算一遍会让"库返回的距离"与"报告里的距离"变成两个数，
而二者不一致时你无法判断是谁错了——透传才有对账价值。

## 建集合时 ``embedding_function=None`` 是必须的

Chroma 的默认 embedding function 是 Sentence Transformers 的
``all-MiniLM-L6-v2``。它本地运行，但**第一次用会去下载模型权重**。
本包永远自己提供向量（``day065`` 的 indexing 包负责编码），
所以这里显式传 ``embedding_function=None``：

> 否则一个"只是建了个集合"的动作会触发一次网络下载——
> 在离线环境里表现为卡住，在受限网络里表现为一个与业务无关的超时。

集合也**只传 ``configuration`` 而不传 collection metadata**：两者都能设
``hnsw.space``，同时给会被真库判为冲突；而集合已存在时 ``metadata`` 会被忽略，
写进去只会让 ``describe_extra()`` 读出一个与磁盘不一致的说法。

## 持久化不是"写一次快照"

``PersistentClient(path=...)`` 本身就是持久的：数据一落就写进 ``path``。
因此本后端的 ``persist()`` 不写快照，而是**返回那个目录并说明数据已在盘上**；
构造时 ``path`` 为空（``chromadb.Client()`` 内存客户端）则抛
``BackendUnavailable``——**不做静默降级**：一个"以为落盘了其实还在内存里"
的向量库，要到进程重启后才暴露，而那时你已经删掉了重建脚本。

## 这一层刻意不做的事

```text
不调用库自带的编码器      → 向量永远由调用方提供（embedding_function=None）
不实现服务端 HttpClient   → day072 的部署一章，本课只做单进程
不做 where_document       → 本包的 where 只筛元数据（见 filters.py 的运算符表）
不替调用方重试/降级       → 依赖缺失与目录为空都直接报错，不猜
```
"""

from __future__ import annotations

import ipaddress
import math
from collections.abc import Sequence
from typing import Any, ClassVar

from smart_research_agent.llm.embedding import l2_normalize
from smart_research_agent.vectorstore.base import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    VectorBackend,
)
from smart_research_agent.vectorstore.errors import (
    BackendUnavailable,
    FilterError,
    RecordError,
    VectorError,
)
from smart_research_agent.vectorstore.metrics import (
    METRIC_COSINE,
    METRIC_INNER_PRODUCT,
    METRIC_L2,
    similarity_from_distance,
)
from smart_research_agent.vectorstore.types import (
    SearchHit,
    SearchResult,
    VectorRecord,
    WriteReport,
    sort_hits,
)

#: 集合名的长度约束（官方：3~512）。取 3 是因为两字符的名字几乎一定来自
#: "随手写了个缩写"，而它没有任何可读性收益；取 512 是官方的硬上限。
COLLECTION_NAME_MIN_LENGTH = 3
COLLECTION_NAME_MAX_LENGTH = 512

#: 默认集合名。与 ``config.vector_collection`` 的默认值逐字一致——
#: 两处不同会让"配置里没写"与"代码里没写"落到两个不同的集合上。
DEFAULT_COLLECTION_NAME = "smart_research_agent"

#: 本包的度量 → Chroma 的 ``hnsw.space``。**是恒等映射，仍然显式写出来**：
#: 三个名字恰好一样是巧合（``ip`` 在 FAISS 里是"越大越近"，在 Chroma 里
#: 是"越小越近"），写出来的作用是让"这里做过一次翻译"这件事在代码里可见，
#: 而不是靠"名字一样所以直接传过去"。
SPACE_BY_METRIC: dict[str, str] = {
    METRIC_COSINE: "cosine",
    METRIC_INNER_PRODUCT: "ip",
    METRIC_L2: "l2",
}

#: 判定"记录没变"时向量的相对/绝对容差。取值理由是 Chroma 的存储精度：
#: 它把向量存成 **float32**（约 7 位十进制有效数字），读回来必然带量化误差。
#: 若按逐位比较，重放同一批数据会被报成 ``updated``，
#: **``unchanged`` 这一态在 Chroma 上就永远不可达**——而它正是 day065
#: 增量索引回答"这次重建省掉了多少次编码"的唯一依据。
#: 1e-5 比 float32 的精度宽一个量级，又远小于任何真实的内容变化。
VECTOR_EQUAL_TOLERANCE = 1e-5

#: 集合名允许出现的字符（首尾另有约束：必须是小写字母或数字）。
_NAME_ALLOWED = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._-")


def validate_collection_name(name: str) -> str:
    """按官方约束校验集合名；不合法时抛 ``RecordError`` 并指出**是哪一条**违反.

    约束来自 Chroma 的服务端校验，五条各自对应一种"看起来能跑但会在建表时炸"：

    ```text
    长度 3~512          太短的名字几乎一定来自随手写的缩写
    首尾是小写字母/数字  首尾是 . - _ 的名字在文件系统里会长成奇怪的东西
    中间只允许 . - _     空格与中文会让"集合名"变成一句自由文本
    不能连续两个点       Chroma 明确禁止（``..`` 在名字里没有意义）
    不能是合法 IP        与它内部的路径/主机名规则冲突
    ```

    校验放在适配器这一层而不是"交给库去报错"，是因为库的报错发生在
    ``create_collection`` 那一刻——那时调用方已经认为自己写对了集合名，
    而错误信息里通常只有一句正则。这里把它提前成一个能自己改对的说明。

    注意它**不做任何"修复"**（不转小写、不替换非法字符）：
    静默改名会让"我明明指定了 collection=X"与"实际用到的集合"不一致，
    而那种不一致只会在下一次"数据集怎么空了"时才暴露。
    """
    if not isinstance(name, str):
        raise RecordError(
            f"collection 必须是字符串，收到 {type(name).__name__}。"
            f"例如 collection={DEFAULT_COLLECTION_NAME!r}。"
        )
    length = len(name)
    if length < COLLECTION_NAME_MIN_LENGTH or length > COLLECTION_NAME_MAX_LENGTH:
        raise RecordError(
            f"集合名 {name!r} 长度 {length} 不在允许范围 "
            f"{COLLECTION_NAME_MIN_LENGTH}~{COLLECTION_NAME_MAX_LENGTH} 之内。"
        )
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        address = None
    if address is not None:
        raise RecordError(
            f"集合名 {name!r} 是一个合法 IP 地址，Chroma 不允许"
            "（它内部的路径与主机名规则会与 IP 冲突）。请换一个名字。"
        )
    if not (name[0].isascii() and name[0].isalnum() and not name[0].isupper()):
        raise RecordError(
            f"集合名 {name!r} 必须以小写字母或数字开头。"
        )
    if not (name[-1].isascii() and name[-1].isalnum() and not name[-1].isupper()):
        raise RecordError(
            f"集合名 {name!r} 必须以小写字母或数字结尾（不能以 '.' / '-' / '_' 收尾）。"
        )
    if ".." in name:
        raise RecordError(f"集合名 {name!r} 里不能出现连续两个点（'..'）。")
    illegal = sorted({character for character in name if character not in _NAME_ALLOWED})
    if illegal:
        raise RecordError(
            f"集合名 {name!r} 含有不允许的字符 {'、'.join(repr(c) for c in illegal)}："
            "中间只允许小写字母、数字与 '.' / '-' / '_'（大写字母与空格都会被拒）。"
        )
    return name


class ChromaVectorStore(VectorBackend):
    """Chroma 后端（``native_metadata = True``：元数据与向量在同一个集合里）.

    ``dimension`` 可以缺省：Chroma 在建集合时并不需要 d，它在**第一次写入**时
    自行确定。因此本类在构造时若发现集合里已有数据，会把维度读回来
    （``_detect_dimension``）——这一步不是可选的：
    少了它，``base._prepare`` 的维度护栏在"重开一个已有库"时会失效，
    而失效的表现是**库自己抛一个英文的维度错误**，而不是本包的 ``VectorError``。

    ``dimension`` **给了值时**的合法性校验由基类负责：
    ``base.VectorBackend._check_dimension`` 是**唯一**的实现
    （维度从 1 起计数，``0`` 是 ``UNKNOWN_DIMENSION`` 的哨兵值），
    本类不再重复覆写它。
    """

    name: ClassVar[str] = "chroma"
    requires: ClassVar[tuple[str, ...]] = ("chromadb",)
    supports_filter: ClassVar[bool] = True
    supports_delete: ClassVar[bool] = True
    supports_persistence: ClassVar[bool] = True
    #: 元数据与向量存在同一个库里：``_find_ids`` 因此可以翻译成原生 ``get(where=...)``。
    native_metadata: ClassVar[bool] = True

    def __init__(
        self,
        *,
        metric: str = METRIC_COSINE,
        dimension: int | None = None,
        path: str = "",
        collection: str = DEFAULT_COLLECTION_NAME,
        client: Any = None,
        chromadb_module: Any = None,
    ) -> None:
        super().__init__(metric=metric, dimension=dimension, location=str(path or ""))
        self._collection_name = validate_collection_name(collection)
        self._space = SPACE_BY_METRIC[self.metric]
        if client is None:
            module = _resolve_chromadb(chromadb_module)
            self._client = _create_client(module, self._location)
        else:
            # 注入客户端（测试与自定义部署）：模块级检查跳过——本类需要的能力
            # 全在这个对象上，而它可能来自一个自建的封装而不是某个模块。
            module = chromadb_module
            self._client = client
            if not self._location:
                discovered = getattr(client, "path", "")
                if discovered:
                    self._location = str(discovered)
        self._chromadb_module = module
        self._collection = self._acquire_collection()
        self._detect_dimension()

    # ------------------------------------------------------------------ 元信息

    def describe_extra(self) -> dict[str, str]:
        """后端特有的状态（``/vectorstore/stats`` 直接回显）.

        ``embedding_function`` 固定为 ``"none"``：它是**本次部署的一条事实**，
        而不是一个可配项——本包永远自己提供向量，库自带的编码器一次都不该被调用。
        """
        return {
            "collection": self._collection_name,
            "space": self._space,
            "embedding_function": "none",
            "native_where": "true",
            "client": "persistent" if self._location else "memory",
        }

    # ------------------------------------------------------------ 必须实现的原语

    def count(self) -> int:
        """集合里的条数（Chroma 自己维护，不需要扫记录表）."""
        return int(self._collection.count())

    def ids(self) -> list[str]:
        """全部记录 id，**升序**（``base.ids`` 的契约，见那里的说明）.

        ``include=[]`` 是刻意的：只要 id 就不要把整库的正文与元数据搬进内存——
        一个 10 万块的库带上 documents 会让"列一下 id"变成一次几百兆的传输。
        """
        result = self._collection.get(include=[])
        return sorted(str(record_id) for record_id in (result.get("ids") or []))

    def get(self, record_id: str) -> VectorRecord | None:
        """取一条记录；不存在返回 ``None``."""
        return self._records([record_id]).get(str(record_id))

    def _upsert(self, records: Sequence[VectorRecord]) -> WriteReport:
        """写入：逐条与库里的现状比较，得出 ``added / updated / unchanged``.

        **``unchanged`` 的记录完全不碰库**：这是它与 ``upsert`` 调用的区别所在。
        如果为了省一次比较而把整批直接交给 ``collection.upsert``，
        报告里的 ``updated`` 会把"一个字节都没变"的批量重放也算进去，
        而 day065 要用这个数字算"增量索引省掉了多少编码"。
        """
        payload: list[VectorRecord] = []
        added = updated = unchanged = 0
        for record in records:
            existing = self.get(record.record_id)
            if existing is None:
                added += 1
                payload.append(record)
            elif _records_equal(existing, record):
                unchanged += 1
            else:
                updated += 1
                payload.append(record)
        if payload:
            self._collection.upsert(
                ids=[record.record_id for record in payload],
                embeddings=[list(record.vector) for record in payload],
                metadatas=[dict(record.metadata) for record in payload],
                documents=[record.text for record in payload],
            )
        return WriteReport(added=added, updated=updated, unchanged=unchanged)

    def _remove(self, ids: Sequence[str]) -> int:
        """删除并返回**真正删掉的**条数.

        ``collection.delete`` 返回的是 ``None``（真库如此，删不存在的 id
        是幂等的、不报错），所以条数只能靠"删除前后各 ``count()`` 一次"得到。
        直接返回 ``len(ids)`` 会把"请求删 3 条、库里有 1 条"报成 3——
        与 ``WriteReport.unchanged`` 是同一个思路：**报告里的数字必须是真的。**
        """
        before = self.count()
        self._collection.delete(ids=[str(record_id) for record_id in ids])
        return before - self.count()

    def _ranked_ids(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        restrict: frozenset[str] | None = None,
    ) -> list[tuple[str, float]]:
        """向量排序（``query()`` 不走这里，它是给"按 id 集合过滤"留的退路）.

        ``restrict`` 是**记录 id 的集合**，而 Chroma 的 ``query`` 只能按
        **元数据**过滤（``where``），没有任何"只在这些 id 里搜"的原生参数。
        因此这里有 ``restrict`` 时只能全量排一遍再筛：

        ```text
        restrict is None   → n_results = min(limit, count)
        restrict 非空      → n_results = count（全量排序）再按 id 筛、取前 limit
        ```

        这条退路的代价是真实的（一次查询可能把整库排一遍），因此
        ``query()`` 走的是 ``where`` 而不是 ``restrict``：
        **能用元数据表达的约束，就不要退化成"把 id 搬回 Python 再筛"。**
        """
        total = self.count()
        if total == 0 or limit < 1:
            return []
        depth = min(limit, total) if restrict is None else total
        ranked = self._native_rank(vector, depth)
        if restrict is not None:
            ranked = [item for item in ranked if item[0] in restrict]
        return ranked[:limit]

    def _find_ids(self, where: dict[str, Any] | None) -> set[str] | None:
        """按 ``where`` 取候选 id（**原生过滤**：一次 ``get(where=...)``）.

        ``None`` 表示"不过滤"（与 ``base._find_ids`` 的契约一致）：
        返回空集合会被 ``base.delete`` 当成"一条都没匹配上"，
        而两者在调用方看来是完全不同的结论。

        ``include=[]`` 只取 id、不取正文与元数据：这条路径上
        ``delete(where=...)`` 与"报告候选数"都只要 id，
        而一次命中 10 万条的 ``get`` 若默认带上 documents，
        会把整库的正文搬进内存——**为了数一遍 id 而拖来整库的文本**。
        """
        if where is None:
            return None
        result = self._collection.get(where=where, include=[])
        return {str(record_id) for record_id in (result.get("ids") or [])}

    # ------------------------------------------------------------------ 公共查询

    def query(
        self,
        vector: Sequence[float],
        top_k: int = DEFAULT_TOP_K,
        *,
        where: dict[str, Any] | None = None,
        min_score: float | None = None,
    ) -> SearchResult:
        """检索：把 ``where`` 一次性下推给原生查询（见模块 docstring 的取舍）.

        与 ``base.query`` 的三处差别，每一处都有理由：

        ```text
        1. where 下推       → 一趟而不是两趟；n_results 是"过滤之后"要的条数
        2. distance 透传    → SearchHit.distance 就是 Chroma 给的那个数，便于对账
        3. 批量取记录       → 命中的 k 条用一次 get(ids=[...]) 取回，而不是 k 次
        ```

        参数校验的顺序与措辞**与 ``base.query`` 逐字一致**：同一件事
        在两个入口不能有两套说法，否则调用方得按后端分别写错误处理。
        """
        _validate_top_k(top_k)
        query_vector = [float(x) for x in vector]
        if self.has_dimension and len(query_vector) != self._dimension:
            raise VectorError(
                f"查询向量是 {len(query_vector)} 维，本库是 {self._dimension} 维："
                "查询用的编码器与建库用的不是同一个。"
            )
        if self.metric == METRIC_COSINE:
            # 与写入侧同一个口径（``types.make_record`` 用的也是这个函数）
            query_vector = l2_normalize(query_vector)

        filter_applied = bool(where)
        matched = self._find_ids(where) if filter_applied else None
        available = self.count() if matched is None else len(matched)
        if available == 0:
            return SearchResult(
                hits=(),
                metric=self.metric,
                top_k=top_k,
                candidates=0,
                filter_applied=filter_applied,
            )

        n_results = min(top_k, available)
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where=where if filter_applied else None,
            include=["documents", "metadatas", "distances"],
        )
        ids_row = _first_row(result.get("ids"))
        distances_row = _first_row(result.get("distances"))
        # 库给的顺序只用于"取哪些 id"；最终顺序一律由 sort_hits 决定，
        # 这样三个后端在平局（分数精确相等）时的断法必然一致。
        records = self._records(ids_row)
        hits: list[SearchHit] = []
        for position, raw_id in enumerate(ids_row):
            record = records.get(str(raw_id))
            if record is None:
                # 集合里有一条取不回来的向量：跳过并继续，与 base.query 对脏数据
                # 的态度一致（读路径不该因为一条脏数据整体失败）。
                continue
            distance_value = float(distances_row[position])
            hits.append(
                SearchHit(
                    record=record,
                    score=similarity_from_distance(self.metric, distance_value),
                    distance=distance_value,
                    rank=position,
                )
            )
        if min_score is not None:
            hits = [hit for hit in hits if hit.score >= float(min_score)]
        ordered = sort_hits(hits)
        return SearchResult(
            hits=tuple(ordered),
            metric=self.metric,
            top_k=top_k,
            candidates=available,
            filter_applied=filter_applied,
        )

    # ------------------------------------------------------------------ 持久化

    def persist(self, path: str | None = None) -> str:
        """返回数据所在的目录（**不写快照**：``PersistentClient`` 已经在写了）.

        这是 Chroma 与其他两个后端最不一样的一处：它的持久化不是"某个时刻
        导出一次"，而是"每写一条就落在盘上"。因此 ``persist()`` 的语义是
        "确认并回显那个目录"，而不是"生成一份可以搬走的文件"。

        内存客户端（构造时 ``path`` 为空）直接抛 ``BackendUnavailable``：
        一个"以为落盘了其实还在内存里"的库要到进程重启才暴露。
        """
        if not self._location:
            raise BackendUnavailable(
                "Chroma 的内存客户端不落盘：本实例构造时 path 为空"
                "（对应 chromadb.Client()），数据随进程消失。"
                "请用 path=... 构造（对应 chromadb.PersistentClient(path=...)，"
                "数据从第一条起就在盘上），或换用 flat / faiss 后端。"
            )
        if path is not None and str(path) != self._location:
            raise BackendUnavailable(
                f"Chroma 的落盘目录在客户端构造时就已定死（当前 {self._location!r}），"
                f"没法把同一个实例改存到 {path!r}。"
                f"请用 path={path!r} 新建一个 ChromaVectorStore。"
            )
        return self._location

    def load(self, path: str | None = None) -> None:
        """重新拿到集合句柄（**不是**把快照读进内存）.

        对 Chroma 而言"加载"没有单独的一步：目录里的数据在
        ``PersistentClient(path=...)`` 构造时就可见了。这里做的只有两件事：
        重新取一次集合句柄（若集合是别人刚建的），并把维度从已有数据读回来。

        给了不同的 ``path`` 时直接报错而不是静默忽略：Chroma 的目录绑定在
        客户端上，让一个已打开的实例"改读另一个目录"会让它此后读的
        与调用方以为的完全是两份数据。
        """
        if path is not None and str(path) != self._location:
            raise BackendUnavailable(
                f"Chroma 的数据绑定在构造时的客户端目录上"
                f"（当前 {self._location or '内存（无目录）'}），同一个实例没法改读 "
                f"{path!r}：请用 path={path!r} 新建一个 ChromaVectorStore"
                "——数据已经在盘上，不需要重新写入。"
            )
        if not self._location:
            raise BackendUnavailable(
                "本实例是内存客户端（构造时 path 为空），没有可读取的目录："
                "请用 path=... 构造。"
            )
        self._collection = self._acquire_collection()
        self._detect_dimension()

    # ------------------------------------------------------------------ 内部工具

    def _acquire_collection(self) -> Any:
        """取得集合句柄（不存在就建）.

        ``embedding_function=None`` 与"只传 ``configuration``"这两条的理由
        都写在模块 docstring 里（模型下载、与 collection metadata 的冲突）。
        """
        return self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": self._space}},
        )

    def _detect_dimension(self) -> None:
        """集合里已有数据但实例还不知道维度时，把维度读回来（取一条探针）."""
        if self.has_dimension or self.count() == 0:
            return
        probe = self._collection.get(limit=1, include=["embeddings"])
        embeddings = probe.get("embeddings") or []
        if embeddings:
            self._dimension = len(embeddings[0])

    def _records(self, record_ids: Sequence[str]) -> dict[str, VectorRecord]:
        """按 id 批量取记录（**一次** ``get``，不是每个 id 一次）.

        一次 ``top_k=5`` 的查询若逐条取记录，会变成 6 次跨库调用；
        在 Chroma 的服务端模式下那就是 6 次 HTTP 往返。批量取只多一行代码，
        代价却从"随 top_k 线性增长"变成"常数次"。
        """
        wanted = [str(record_id) for record_id in record_ids]
        if not wanted:
            return {}
        result = self._collection.get(
            ids=wanted,
            include=["embeddings", "documents", "metadatas"],
        )
        return _records_from(result)

    def _native_rank(self, vector: Sequence[float], n_results: int) -> list[tuple[str, float]]:
        """一次原生 ``query``，把 ``distances`` 折成 ``(id, score)``（分数降序）.

        ``n_results`` 的下界由 ``_ranked_ids`` 保证（那里是唯一的入口）：
        Chroma 对 ``n_results=0`` 会直接报错，因此护栏只放一处、不重复放两处。
        """
        result = self._collection.query(
            query_embeddings=[[float(x) for x in vector]],
            n_results=n_results,
            include=["distances"],
        )
        ids_row = _first_row(result.get("ids"))
        distances_row = _first_row(result.get("distances"))
        ranked = [
            (
                str(record_id),
                similarity_from_distance(self.metric, float(distances_row[position])),
            )
            for position, record_id in enumerate(ids_row)
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked


def _validate_top_k(top_k: int) -> None:
    """``top_k`` 的两条边界（措辞与 ``base.query`` 逐字一致，见 ``query`` 的说明）."""
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


def _records_from(result: dict[str, Any]) -> dict[str, VectorRecord]:
    """把一次 ``get`` 的**一维平铺**结果解析成 ``id → VectorRecord``.

    这里刻意**直接构造 ``VectorRecord`` 而不是 ``make_record``**：
    写入侧已经按度量归一化过一次，读路径再归一化会让 ``cosine`` 下的向量
    每经历一次"读回来 → 写回去"就漂移一点点（除以一个 ~1.0 的模长）。
    漂移本身很小，但它会让"什么都没改"最终被报成 ``updated``——
    一个只有靠容差才压得住的、静默退化的故障。
    """
    ids = list(result.get("ids") or [])
    embeddings = list(result.get("embeddings") or [])
    documents = list(result.get("documents") or [])
    metadatas = list(result.get("metadatas") or [])
    records: dict[str, VectorRecord] = {}
    for position, raw_id in enumerate(ids):
        if position >= len(embeddings) or embeddings[position] is None:
            # 没有向量的行构不成记录（``include`` 里没要 embeddings 时就会这样）。
            # 跳过而不是抛错：读路径不该因为一行缺字段整体失败。
            continue
        record_id = str(raw_id)
        document = documents[position] if position < len(documents) else None
        metadata = metadatas[position] if position < len(metadatas) else None
        records[record_id] = VectorRecord(
            record_id=record_id,
            vector=tuple(float(x) for x in embeddings[position]),
            text=document if isinstance(document, str) else "",
            metadata=dict(metadata or {}),
        )
    return records


def _first_row(value: Any) -> list[Any]:
    """取 ``query`` 返回的二维结构的**第一组**（本包一次只查一个向量）."""
    if not value:
        return []
    return list(value[0])


def _records_equal(existing: VectorRecord, incoming: VectorRecord) -> bool:
    """两条记录是否"逐位相同"（向量按 float32 精度比较，见 ``VECTOR_EQUAL_TOLERANCE``）.

    比较的不是三个字段的哈希，而是"这三个字段有没有实质变化"：
    向量容差是**必须的**（存储精度所致），而 ``text`` / ``metadata`` 仍然精确比较
    ——它们是被原样存取的，出现差异就是真的有差异。
    """
    if existing.text != incoming.text:
        return False
    if existing.metadata != incoming.metadata:
        return False
    if existing.dimension != incoming.dimension:
        return False
    return all(
        math.isclose(a, b, rel_tol=VECTOR_EQUAL_TOLERANCE, abs_tol=VECTOR_EQUAL_TOLERANCE)
        for a, b in zip(existing.vector, incoming.vector)
    )


def _resolve_chromadb(module: Any) -> Any:
    """拿到可用的 ``chromadb`` 模块；缺失或残缺时给出**下一步做什么**.

    两层检查对应两类不同的失败：

    ```text
    模块不在           → 环境缺依赖：pip install chromadb
    模块在但缺 API     → 版本过旧或 wheel 残缺：pip install -U chromadb
    ```

    第二条不是洁癖：本后端声明 ``supports_persistence = True``，而"数据在盘上"
    这句话只能由 ``PersistentClient`` 兑现。一个没有它的安装会让
    ``persist()`` 的成功返回值变成一句假话——所以在这里就拦掉，
    而不是等 ``persist()`` 那一刻再报一个看不懂的 ``AttributeError``。
    """
    if module is not None:
        return _require_persistent_client(module)
    try:
        import chromadb  # noqa: PLC0415  （延迟导入：本后端是可选依赖）
    except ImportError as exc:
        raise BackendUnavailable(
            f"缺少可选依赖 chromadb（{exc}）：本后端需要它才能读写向量库。"
            "安装：pip install chromadb。"
            "若只是想在本地跑通，flat 后端零依赖、结果逐位可复现，可以先用它。"
        ) from exc
    return _require_persistent_client(chromadb)


def _require_persistent_client(module: Any) -> Any:
    if not hasattr(module, "PersistentClient"):
        raise BackendUnavailable(
            f"chromadb 已安装但缺少 PersistentClient（{module!r}）："
            "通常是版本过旧或安装残缺。"
            "本后端的数据落盘完全依赖它，请升级：pip install -U chromadb。"
        )
    return module


def _create_client(module: Any, path: str) -> Any:
    """按 ``path`` 决定建持久客户端还是内存客户端.

    Chroma 没有"打开或新建"的统一入口，这一步的选择就决定了整个实例的
    生命周期：``path`` 非空 → 数据在盘上；``path`` 为空 → 数据随进程消失。
    本包不替调用方猜（例如"path 为空就落到 .chroma"）：真库确实有那样一个
    默认值，但它会让"我没写路径"变成一个**静默写进当前目录**的行为。
    """
    if path:
        return module.PersistentClient(path=path)
    return module.Client()


__all__ = [
    "COLLECTION_NAME_MAX_LENGTH",
    "COLLECTION_NAME_MIN_LENGTH",
    "DEFAULT_COLLECTION_NAME",
    "SPACE_BY_METRIC",
    "VECTOR_EQUAL_TOLERANCE",
    "ChromaVectorStore",
    "validate_collection_name",
]
