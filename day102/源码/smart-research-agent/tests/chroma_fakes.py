"""假 chromadb：一份"够用且忠实"的最小实现（M6-D3，C 号任务专用）.

本机没有安装 ``chromadb``（也**不应该**为了跑测试去装：装它只是第一步，
它默认的 embedding function 还会在"只是建了个集合"的时候去下载
``all-MiniLM-L6-v2`` 的权重）。因此测试用一个假模块顶替真库。

``tests/chroma_fakes.py`` 这个名字不带 ``test_`` 前缀，是刻意的：
``pyproject.toml`` 里 ``python_files = ["test_*.py"]`` 只收集测试文件，
而**假库不是测试**——它是被测试注入的对象。名字以 ``test_`` 开头会让
pytest 试图收集它里面的 ``FakeCollection`` 之类的类（``python_classes = ["Test*"]``
只认 ``Test`` 前缀，所以这次侥幸不会出事），但"靠侥幸"不是一条能长期成立的规则。

## 忠实到什么程度，以及为什么必须是这个程度

假的忠实度不是"越像越好"，而是**恰好让适配器的每一处假设都能被证伪**：

```text
真实行为                    假库怎么做                   它让哪条断言成为可能
-------------------------  ---------------------------  --------------------------
集合名有三条命名约束         不校验（校验在适配器里）       非法名在建库之前就被拒
向量存成 float32            struct 做一次 float32 往返    重放同一批仍要报 unchanged
query 返回二维列表          外层按查询分组、内层按名次     防止适配器把二维当一维用
get 返回一维平铺            与 query 的形状刻意不同       防止两个入口共用解析代码
三种 distance 都是越小越近   三套公式照官方文档实现        适配器把方向折成"越大越近"
距离默认 space 是 l2        未给 configuration 时按 l2    适配器总是显式传 space
未知关键字直接报错           所有方法都是显式签名          适配器不许"发明参数"
metadata 只收标量/同类型数组  _validate_metadata 逐键校验   公共契约与真实库一致
embedding 维度写死在集合级    维度不一致就报错              适配器的 VectorError 先触发
```

## 假库刻意**不**做的事

```text
不实现 HNSW 的近似性     → 这里永远是精确检索（近似是性能问题，不是接口问题）
不实现服务端 HttpClient  → 服务端模式留到 day072 的部署一章
不实现 where_document    → 本包的 where 只筛元数据（filters.py 的运算符表）
不实现 embedding function 的调用 → 传进来的向量就是向量，绝不"顺手算一下"
```

最后一条最要紧：假库**只接受显式传入的 ``embeddings``**，若适配器漏传向量
（指望库自己算），假库会直接报错而不是悄悄成功——这正是"必须显式传
``embedding_function=None``"那条要求在测试里的抓手。
"""

from __future__ import annotations

import math
import pickle
import struct
from pathlib import Path
from typing import Any

#: 真库的版本号（仅用于让假模块看起来像一个模块；断言不依赖它）.
FAKE_CHROMADB_VERSION = "1.5.9"

#: ``collection.query`` 的默认 ``n_results``。真库默认 10，不是 5——
#: 这个默认值很容易被"以为库会给我全部"的调用方忽略。
DEFAULT_N_RESULTS = 10

#: ``get`` 与 ``query`` 各自允许的 ``include`` 取值（真库列出的是同一批字段的
#: 不同子集：``query`` 多一个 ``distances``）。分开写是为了让"传错入口"
#: 也能在假库上被发现。
GET_INCLUDE_FIELDS: tuple[str, ...] = ("documents", "embeddings", "metadatas", "uris", "data")
QUERY_INCLUDE_FIELDS: tuple[str, ...] = (
    "documents",
    "embeddings",
    "metadatas",
    "distances",
    "uris",
    "data",
)

#: 三种 ``hnsw.space`` 的取值。默认 ``l2``（真库如此），因此适配器若忘了显式传
#: space，在假库上会**静默按 l2 排序**——那正是"忘传配置"在生产里的表现。
SPACES: tuple[str, ...] = ("l2", "ip", "cosine")

#: 元数据允许的标量类型。``bool`` 必须排在 ``int`` 之前（``isinstance(True, int)``
#: 为真），否则布尔值会被当成整数报错。
_METADATA_SCALARS: tuple[type, ...] = (bool, int, float, str)


def to_float32(value: float) -> float:
    """把双精度浮点做一次 float32 往返（用 ``struct``，不引入 numpy）.

    Chroma 的底层存储是 float32，因此**写进去的向量读回来必然带上量化误差**。
    假库必须复现这一条，否则"重放同一批 → ``unchanged``"这条断言会变成
    "因为假库恰好没量化所以过了"，而真库上它会变成一堆 ``updated``。
    """
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _to_float32_vector(vector: list[float]) -> list[float]:
    return [to_float32(float(x)) for x in vector]


def _validate_metadata(metadata: Any) -> None:
    """按真库的规则校验一份 metadata（不合法就抛 ``ValueError``）.

    真库的报错是 ``ValueError``/``InvalidArgumentError``，消息里指出
    "Expected metadata value to be ..."。这里保留英文关键片段，
    因为**适配器的职责不是改写库的错误，而是让这类数据根本不进来**
    （``types.VectorRecord`` 在构造时已经拦下同一批形状）。
    """
    if metadata is None:
        return
    if not isinstance(metadata, dict):
        raise ValueError(f"Expected metadata to be a dict, got {type(metadata).__name__}")
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"Expected metadata key to be a non-empty str, got {key!r}")
        if isinstance(value, _METADATA_SCALARS):
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(
                    f"Expected metadata value for {key!r} to be a non-empty list "
                    "(empty arrays are rejected by Chroma)"
                )
            kinds = {_scalar_kind(item) for item in value}
            if len(kinds) != 1 or None in kinds:
                raise ValueError(
                    f"Expected metadata value for {key!r} to be a list of one scalar "
                    f"type, got {sorted(kinds, key=str)}"
                )
            continue
        raise ValueError(
            f"Expected metadata value for {key!r} to be a str, int, float, bool or a "
            f"list of these types, got {type(value).__name__}"
        )


def _scalar_kind(value: Any) -> str | None:
    """数组元素的"类型名"（``None`` 表示不可作为元数据标量）."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return None


def _match_where(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """假库自己的 ``where`` 求值器（**不复用** ``vectorstore.filters``）.

    刻意独立实现：本包把 Chroma 的语法当作规范语法，如果假库直接调用
    ``filters.compile_filter``，"适配器把 where 原样下推"这条断言就退化成
    "适配器调用了自己家的编译器"，失去意义。这里重写一遍运算符表，
    相当于给那份语法配了第二个实现（与 ``brute_force_top`` 的思路一致）。
    """
    if not where:
        return True
    for key, value in where.items():
        if key == "$and":
            if not all(_match_where(metadata, item) for item in value):
                return False
        elif key == "$or":
            if not any(_match_where(metadata, item) for item in value):
                return False
        elif not _match_field(metadata, key, value):
            return False
    return True


def _match_field(metadata: dict[str, Any], field: str, condition: Any) -> bool:
    if not isinstance(condition, dict):
        # 直接写标量等价于 $eq；字段不存在则不等（与真库一致）
        return field in metadata and metadata[field] == condition
    for operator, expected in condition.items():
        present = field in metadata
        value = metadata.get(field)
        if operator == "$eq":
            if not (present and value == expected):
                return False
        elif operator == "$ne":
            if not (not present or value != expected):
                return False
        elif operator in ("$gt", "$gte", "$lt", "$lte"):
            if not present:
                return False
            try:
                matched = {
                    "$gt": value > expected,
                    "$gte": value >= expected,
                    "$lt": value < expected,
                    "$lte": value <= expected,
                }[operator]
            except TypeError:
                # 类型不可比 → 判为不命中（真库/本包都是这条取舍）
                return False
            if not matched:
                return False
        elif operator == "$in":
            if not present or value not in expected:
                return False
        elif operator == "$nin":
            # 字段不存在也算"不在集合内"（与 filters.py 的语义一致）
            if present and value in expected:
                return False
        elif operator == "$contains":
            if not isinstance(value, (list, tuple)) or expected not in value:
                return False
        elif operator == "$not_contains":
            if isinstance(value, (list, tuple)) and expected in value:
                return False
        else:
            raise ValueError(f"Expected where operator, got unknown operator {operator!r}")
    return True


class FakeCollection:
    """一个内存集合：向量 + 文档 + 元数据 + 一份"插入顺序"记录.

    所有方法都写成**显式签名**（没有 ``**kwargs``）：适配器一旦发明了
    真库没有的参数，Python 会在这里抛 ``TypeError``，而不是等到上线后
    才发现参数被静默忽略。
    """

    def __init__(
        self,
        name: str,
        embedding_function: Any = None,
        metadata: dict[str, Any] | None = None,
        configuration: dict[str, Any] | None = None,
        *,
        owner: FakeClient | None = None,
    ) -> None:
        self.name = name
        self.metadata = dict(metadata or {})
        self.configuration = dict(configuration or {})
        self.embedding_function = embedding_function
        self._owner = owner
        #: 调用流水（测试用它们断言"适配器到底把什么交给了原生查询"）.
        self.query_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.add_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self._vectors: dict[str, list[float]] = {}
        self._documents: dict[str, str | None] = {}
        self._metadatas: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []

    # ------------------------------------------------------------------ 状态

    def _space(self) -> str:
        """本集合的 ``hnsw.space``（未显式配置时按真库的默认值 ``l2``）."""
        hnsw = self.configuration.get("hnsw") or {}
        space = hnsw.get("space") or self.configuration.get("space") or "l2"
        if space not in SPACES:
            raise ValueError(f"Expected hnsw space to be one of {SPACES}, got {space!r}")
        return str(space)

    def _dimension(self) -> int:
        """集合级维度（空集合为 0）。真库把它在建集合后第一次写入时定死."""
        for vector in self._vectors.values():
            return len(vector)
        return 0

    def _require_embedding_dimension(self, vector: list[float]) -> None:
        existing = self._dimension()
        if existing and len(vector) != existing:
            raise ValueError(
                f"Dimensionality of ({len(vector)}) does not match index dimensionality "
                f"({existing})"
            )

    def _touch(self) -> None:
        if self._owner is not None:
            self._owner._save()

    # ------------------------------------------------------------------ 写入

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """新增（id 已存在 → 报错）。真库在这一点上**不会**替你覆盖."""
        self.add_calls.append({"ids": list(ids), "documents": documents, "metadatas": metadatas})
        self._validate_batch(ids, embeddings, metadatas, documents)
        duplicated = [record_id for record_id in ids if record_id in self._vectors]
        if duplicated:
            raise ValueError(f"IDs already exist in collection {self.name}: {duplicated[0]!r}")
        self._write(ids, embeddings, metadatas, documents)
        self._touch()

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """存在则覆盖、不存在则插入（**本包推荐的写入入口**）."""
        self.add_calls.append({"ids": list(ids), "documents": documents, "metadatas": metadatas})
        self._validate_batch(ids, embeddings, metadatas, documents)
        self._write(ids, embeddings, metadatas, documents)
        self._touch()

    def update(
        self,
        ids: list[str],
        embeddings: list[list[float]] | None = None,
        metadatas: list[dict[str, Any]] | None = None,
        documents: list[str] | None = None,
    ) -> None:
        """只更新已存在的 id；有一条不存在就整体报错（真库如此）."""
        self._validate_batch(ids, embeddings, metadatas, documents, allow_missing=False)
        missing = [record_id for record_id in ids if record_id not in self._vectors]
        if missing:
            raise ValueError(f"Could not find ids in collection: {missing[0]!r}")
        self._write(ids, embeddings, metadatas, documents)
        self._touch()

    def _validate_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]] | None,
        metadatas: list[dict[str, Any]] | None,
        documents: list[str] | None,
        *,
        allow_missing: bool = True,
    ) -> None:
        if not ids:
            raise ValueError("Expected IDs to be a non-empty list")
        if len(set(ids)) != len(ids):
            raise ValueError("Expected IDs to be unique within a batch")
        if embeddings is None:
            # 传了 embedding_function 的真库会在这里自己算向量；本包**必须**
            # 自己给向量（见模块 docstring 关于模型下载的说明），因此直接拒。
            raise ValueError("Expected embeddings to be provided (this fake never embeds)")
        if len(embeddings) != len(ids):
            raise ValueError(
                f"Expected embeddings to have the same length as ids "
                f"({len(embeddings)} vs {len(ids)})"
            )
        if metadatas is not None:
            if len(metadatas) != len(ids):
                raise ValueError(
                    f"Expected metadatas to have the same length as ids "
                    f"({len(metadatas)} vs {len(ids)})"
                )
            for metadata in metadatas:
                _validate_metadata(metadata)
        if documents is not None and len(documents) != len(ids):
            raise ValueError(
                f"Expected documents to have the same length as ids "
                f"({len(documents)} vs {len(ids)})"
            )
        for vector in embeddings:
            self._require_embedding_dimension([float(x) for x in vector])
        del allow_missing  # 只影响调用方的语义，形状校验与它无关

    def _write(
        self,
        ids: list[str],
        embeddings: list[list[float]] | None,
        metadatas: list[dict[str, Any]] | None,
        documents: list[str] | None,
    ) -> None:
        assert embeddings is not None  # 已由 _validate_batch 保证
        for position, record_id in enumerate(ids):
            vector = _to_float32_vector(list(embeddings[position]))
            if record_id not in self._vectors:
                self._order.append(record_id)
            self._vectors[record_id] = vector
            if documents is not None:
                self._documents[record_id] = documents[position]
            if metadatas is not None:
                self._metadatas[record_id] = dict(metadatas[position] or {})

    # ------------------------------------------------------------------ 删除

    def delete(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        """删除（真库返回 ``None``，删了几条要自己 ``count()`` 前后对比）."""
        self.delete_calls.append({"ids": ids, "where": where})
        if ids is None and where is None:
            raise ValueError("Expected ids or where to be provided")
        if ids is not None and where is not None:
            raise ValueError("Expected exactly one of ids or where")
        targets = list(ids) if ids is not None else self._ids_matching(where)
        for record_id in targets:
            if record_id in self._vectors:
                self._vectors.pop(record_id)
                self._documents.pop(record_id, None)
                self._metadatas.pop(record_id, None)
                self._order.remove(record_id)
        self._touch()

    def _ids_matching(self, where: dict[str, Any] | None) -> list[str]:
        return [
            record_id
            for record_id in self._order
            if _match_where(self._metadatas.get(record_id, {}), where)
        ]

    # ------------------------------------------------------------------ 读取

    def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """读取：**一维平铺**（与 ``query`` 的二维形状刻意不同）."""
        self.get_calls.append(
            {"ids": ids, "where": where, "limit": limit, "offset": offset, "include": include}
        )
        fields = list(GET_INCLUDE_FIELDS[:2]) if include is None else list(include)
        _require_include(fields, GET_INCLUDE_FIELDS)
        if ids is not None:
            rows = [record_id for record_id in ids if record_id in self._vectors]
        elif where is not None:
            rows = self._ids_matching(where)
        else:
            rows = list(self._order)
        if offset:
            rows = rows[offset:]
        if limit is not None:
            rows = rows[:limit]
        return self._project(rows, fields)

    def peek(self, limit: int = DEFAULT_N_RESULTS) -> dict[str, Any]:
        """前 ``limit`` 条（真库用它做"这个集合里到底有什么"的快速一瞥）."""
        return self._project(self._order[:limit], ["documents", "metadatas", "embeddings"])

    def _project(self, rows: list[str], fields: list[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {"ids": list(rows)}
        if "embeddings" in fields:
            payload["embeddings"] = [list(self._vectors[record_id]) for record_id in rows]
        if "documents" in fields:
            payload["documents"] = [self._documents.get(record_id) for record_id in rows]
        if "metadatas" in fields:
            payload["metadatas"] = [dict(self._metadatas.get(record_id, {})) for record_id in rows]
        if "uris" in fields:
            payload["uris"] = [None for _ in rows]
        if "data" in fields:
            payload["data"] = [None for _ in rows]
        return payload

    def count(self) -> int:
        """条数."""
        return len(self._vectors)

    # ------------------------------------------------------------------ 检索

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int = DEFAULT_N_RESULTS,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, Any]:
        """向量检索：**返回二维列表**（外层每个查询一组）.

        真库的 ``ids`` / ``documents`` / ``metadatas`` / ``distances`` 全都是
        ``list[list[...]]``，即使只查一个向量。适配器若按一维解析，在真库上
        拿到的会是"整批的第一个元素"，而不是"第一条命中"。
        """
        self.query_calls.append(
            {
                "query_embeddings": [list(vector) for vector in query_embeddings],
                "n_results": n_results,
                "where": where,
                "include": include,
            }
        )
        if not query_embeddings:
            raise ValueError("Expected query_embeddings to be a non-empty list")
        if n_results < 1:
            raise ValueError(f"Expected n_results to be >= 1, got {n_results}")
        fields = ["documents", "metadatas", "distances"] if include is None else list(include)
        _require_include(fields, QUERY_INCLUDE_FIELDS)

        candidates = self._ids_matching(where)
        space = self._space()
        ids_rows: list[list[str]] = []
        distance_rows: list[list[float]] = []
        for raw_query in query_embeddings:
            vector = _to_float32_vector(list(raw_query))
            self._require_embedding_dimension(vector)
            scored = [
                (record_id, _distance(space, vector, self._vectors[record_id]))
                for record_id in candidates
            ]
            # 距离升序 + id 升序：与本包的 sort_hits 是同一个"平局怎么断"的规则
            scored.sort(key=lambda item: (item[1], item[0]))
            top = scored[:n_results]
            ids_rows.append([record_id for record_id, _ in top])
            distance_rows.append([value for _, value in top])

        payload: dict[str, Any] = {"ids": ids_rows}
        if "embeddings" in fields:
            payload["embeddings"] = [
                [list(self._vectors[record_id]) for record_id in row] for row in ids_rows
            ]
        if "documents" in fields:
            payload["documents"] = [
                [self._documents.get(record_id) for record_id in row] for row in ids_rows
            ]
        if "metadatas" in fields:
            payload["metadatas"] = [
                [dict(self._metadatas.get(record_id, {})) for record_id in row] for row in ids_rows
            ]
        if "distances" in fields:
            payload["distances"] = distance_rows
        if "uris" in fields:
            payload["uris"] = [[None for _ in row] for row in ids_rows]
        if "data" in fields:
            payload["data"] = [[None for _ in row] for row in ids_rows]
        return payload

    @property
    def last_query(self) -> dict[str, Any] | None:
        """最近一次 ``query`` 收到的关键字参数（断言"where 被原样下推"用）."""
        return self.query_calls[-1] if self.query_calls else None

    # ------------------------------------------------------------------ 落盘

    def _state(self) -> dict[str, Any]:
        return {
            "vectors": {key: list(value) for key, value in self._vectors.items()},
            "documents": dict(self._documents),
            "metadatas": {key: dict(value) for key, value in self._metadatas.items()},
            "order": list(self._order),
        }

    def _restore(self, state: dict[str, Any]) -> None:
        self._vectors = {key: list(value) for key, value in state["vectors"].items()}
        self._documents = dict(state["documents"])
        self._metadatas = {key: dict(value) for key, value in state["metadatas"].items()}
        self._order = list(state["order"])


def _distance(space: str, query: list[float], vector: list[float]) -> float:
    """按 ``space`` 算距离（**三种都是"越小越近"**，见 ``metrics`` 模块 docstring）.

    ```text
    l2       Σ(ai − bi)²
    ip       1 − Σ(ai·bi)
    cosine   1 − Σ(ai·bi) / (√Σai² · √Σbi²)
    ```
    """
    products = math.fsum(a * b for a, b in zip(query, vector))
    if space == "l2":
        return math.fsum((a - b) ** 2 for a, b in zip(query, vector))
    if space == "ip":
        return 1.0 - products
    query_norm = math.sqrt(math.fsum(a * a for a in query))
    vector_norm = math.sqrt(math.fsum(b * b for b in vector))
    if query_norm == 0.0 or vector_norm == 0.0:
        return 1.0
    return 1.0 - products / (query_norm * vector_norm)


def _require_include(fields: list[str], allowed: tuple[str, ...]) -> None:
    unknown = [field for field in fields if field not in allowed]
    if unknown:
        raise ValueError(f"Expected include to be a subset of {allowed}, got {unknown[0]!r}")


class FakeClient:
    """内存客户端（``path`` 非空时把状态 pickle 到该目录，使"持久化"可测）.

    真库的 ``PersistentClient(path=...)`` 与 ``Client()`` 的差别**只有一件事**：
    前者把数据写进 ``path``，后者随进程消失。假库把这条差别如实复现，
    于是"``persist()`` 在内存客户端上必须报错"才有一个可断言的对象。
    """

    def __init__(self, path: str = "") -> None:
        self.path = str(path)
        #: 建集合时收到的关键字参数（断言 ``embedding_function=None`` 用）.
        self.collection_kwargs: list[dict[str, Any]] = []
        self.deleted_collections: list[str] = []
        self.heartbeat_calls = 0
        self._collections: dict[str, FakeCollection] = {}
        if self.path:
            Path(self.path).mkdir(parents=True, exist_ok=True)
            self._load()

    # ------------------------------------------------------------------ 集合

    def create_collection(
        self,
        name: str,
        embedding_function: Any = None,
        metadata: dict[str, Any] | None = None,
        configuration: dict[str, Any] | None = None,
    ) -> FakeCollection:
        self.collection_kwargs.append(
            {
                "name": name,
                "embedding_function": embedding_function,
                "metadata": metadata,
                "configuration": configuration,
            }
        )
        if name in self._collections:
            raise ValueError(f"Collection {name} already exists")
        collection = FakeCollection(
            name=name,
            embedding_function=embedding_function,
            metadata=metadata,
            configuration=configuration,
            owner=self,
        )
        self._collections[name] = collection
        self._save()
        return collection

    def get_or_create_collection(
        self,
        name: str,
        embedding_function: Any = None,
        metadata: dict[str, Any] | None = None,
        configuration: dict[str, Any] | None = None,
    ) -> FakeCollection:
        self.collection_kwargs.append(
            {
                "name": name,
                "embedding_function": embedding_function,
                "metadata": metadata,
                "configuration": configuration,
            }
        )
        if name not in self._collections:
            self._collections[name] = FakeCollection(
                name=name,
                embedding_function=embedding_function,
                metadata=metadata,
                configuration=configuration,
                owner=self,
            )
            self._save()
        return self._collections[name]

    def get_collection(self, name: str) -> FakeCollection:
        if name not in self._collections:
            raise ValueError(f"Collection {name} does not exist")
        return self._collections[name]

    def list_collections(self) -> list[FakeCollection]:
        return list(self._collections.values())

    def delete_collection(self, name: str) -> None:
        if name not in self._collections:
            raise ValueError(f"Collection {name} does not exist")
        self._collections.pop(name)
        self.deleted_collections.append(name)
        self._save()

    def heartbeat(self) -> int:
        self.heartbeat_calls += 1
        return self.heartbeat_calls

    @property
    def last_collection(self) -> FakeCollection | None:
        """最近一次建出来的集合（``list_collections`` 的顺序即创建顺序）."""
        return list(self._collections.values())[-1] if self._collections else None

    # ------------------------------------------------------------------ 落盘

    def _state_file(self) -> Path:
        return Path(self.path) / "fake_chroma.pkl"

    def _save(self) -> None:
        if not self.path:
            return
        payload = {
            name: {
                "metadata": collection.metadata,
                "configuration": collection.configuration,
                "embedding_function_is_none": collection.embedding_function is None,
                "state": collection._state(),
            }
            for name, collection in self._collections.items()
        }
        self._state_file().write_bytes(pickle.dumps(payload))

    def _load(self) -> None:
        state_file = self._state_file()
        if not state_file.exists():
            return
        payload = pickle.loads(state_file.read_bytes())
        for name, entry in payload.items():
            collection = FakeCollection(
                name=name,
                embedding_function=None if entry["embedding_function_is_none"] else object(),
                metadata=entry["metadata"],
                configuration=entry["configuration"],
                owner=self,
            )
            collection._restore(entry["state"])
            self._collections[name] = collection


class FakeConfigModule:
    """``chromadb.config`` 的最小替身（只暴露 ``Settings``）."""

    class Settings:
        """真库的 ``Settings`` 是 pydantic 模型；这里只需要"能收关键字"."""

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = dict(kwargs)
            self.allow_reset = bool(kwargs.get("allow_reset", False))
            self.anonymized_telemetry = bool(kwargs.get("anonymized_telemetry", True))

        def __repr__(self) -> str:
            return f"FakeSettings({self.kwargs!r})"


class FakeChromaModule:
    """组装成 ``chromadb`` 模块的样子（``PersistentClient`` / ``Client`` / ``config``）.

    ``PersistentClient`` 与 ``Client`` 是**实例方法**而不是类：真库的模块属性
    本来就是个可调用对象，调用形态 ``chromadb.PersistentClient(path=...)``
    在两种写法下完全一致。每个造出来的客户端都留在 ``clients`` 里，
    测试因此不必把客户端一路传下去也能拿到它。
    """

    def __init__(self) -> None:
        self.__version__ = FAKE_CHROMADB_VERSION
        self.config = FakeConfigModule()
        self.clients: list[FakeClient] = []

    # 下面三个名字是**真库的公开 API**（``chromadb.PersistentClient`` 等），
    # 大小写必须照抄：改成 snake_case 就不叫假库了。因此这里显式豁免 N802。
    def PersistentClient(self, path: str = ".chroma") -> FakeClient:  # noqa: N802
        client = FakeClient(str(path))
        self.clients.append(client)
        return client

    def Client(self) -> FakeClient:  # noqa: N802
        client = FakeClient("")
        self.clients.append(client)
        return client

    def EphemeralClient(self) -> FakeClient:  # noqa: N802
        return self.Client()

    @property
    def last_client(self) -> FakeClient | None:
        """最近造出来的客户端（等价于适配器正在用的那一个）."""
        return self.clients[-1] if self.clients else None


class FakeChromaModuleMissingPersistentClient(FakeChromaModule):
    """缺少 ``PersistentClient`` 的变体：模拟"装了但版本过旧/安装不完整".

    用 ``property`` 抛 ``AttributeError`` 来表达"这个模块没有这个名字"——
    ``hasattr`` 在属性访问抛 ``AttributeError`` 时返回 ``False``，
    与"模块里根本没有这个属性"在调用方看来**完全一样**。
    """

    @property
    def PersistentClient(self) -> Any:  # type: ignore[override]  # noqa: N802
        raise AttributeError("module 'chromadb' has no attribute 'PersistentClient'")


__all__ = [
    "DEFAULT_N_RESULTS",
    "FAKE_CHROMADB_VERSION",
    "GET_INCLUDE_FIELDS",
    "QUERY_INCLUDE_FIELDS",
    "SPACES",
    "FakeChromaModule",
    "FakeChromaModuleMissingPersistentClient",
    "FakeClient",
    "FakeCollection",
    "FakeConfigModule",
    "to_float32",
]
