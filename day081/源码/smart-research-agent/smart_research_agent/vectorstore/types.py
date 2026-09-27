"""向量库里的四种形状：记录、命中、结果、写入报告（M6-D3）.

day062 交出来的 ``knowledge_records()`` 是**四个键的字典**
（``doc_id`` / ``source`` / ``text`` / ``metadata``）。今天要把它落到一个
真正的向量库里，于是必须先回答一个被那四个键绕过去的问题：

```text
"一条向量库记录"到底由什么构成？
```

答案是三件东西，而且**缺一不可**：

| 成分 | 为什么必须有 | 少了它会怎样 |
|------|-------------|-------------|
| id | 更新/删除都按它定位 | 只能整库重建；"这份文档变了"无法表达 |
| vector | 它是被检索的东西 | 只能做关键词检索（那就不是向量库了） |
| text | 命中之后要还给人看 | 检索到了却展示不出内容，得回原文再查一次 |
| metadata | 过滤与溯源 | 无法按 ``strategy`` 过滤、无法回指 ``parent_doc_id`` |

## 元数据为什么必须比 Python 允许的更严

Python 的字典什么都能装；**Chroma 的 metadata 只能装
字符串、整数、浮点数、布尔值，以及同类型标量的数组**（不允许嵌套、
不允许空数组）。本包把这条更严的规则提升为**公共契约**：

```text
VectorRecord.metadata 只允许：str / int / float / bool / list[同类型标量]
```

这个选择的代价是"在纯内存后端上本来能存下的东西被拒了"，
收益是**同一个 ``VectorRecord`` 在三个后端上行为一致**。
如果不这么做，会出现一个非常典型的故障：本地用 flat 后端跑得好好的，
换 Chroma 之后在某一次 ``upsert`` 上抛类型错误——
而那时数据已经生产出来了，问题却指向一个跟业务无关的类型细节。

**因此这条约束的报错必须发生在构造记录的那一刻**，
而不是等到写库；报错信息里要指出是哪个键、什么类型
（见 ``assert_metadata``）。

## ``score`` 与 ``distance`` 同时保留

``SearchHit`` 两个都带。这不是冗余：``score`` 是本包的统一口径（越大越近），
``distance`` 是与外部库对账用的原始口径（越小越近，与 Chroma 同名同义）。
只留一个的后果在 day064 的核对里会具体出现——**与 Chroma 对账时
你会需要那个"越小越近"的数**（见 ``metrics`` 模块 docstring）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.vectorstore.errors import RecordError
from smart_research_agent.vectorstore.metrics import (
    METRIC_COSINE,
    distance,
    normalize_metric,
    score,
)

#: 记录 id 的长度上限。取值与 Chroma 集合名的量级无关，
#: 只为了防止"有人把整段文本当 id"这种会撑爆持久化文件的用法。
MAX_RECORD_ID_LENGTH = 128

#: 元数据允许的标量类型。``bool`` 必须排在 ``int`` 之前判断——
#: 在 Python 里 ``isinstance(True, int)`` 是 ``True``，
#: 顺序写反会让布尔值被当成整数报错（一个真实且很容易踩的坑）。
METADATA_SCALAR_TYPES: tuple[type, ...] = (bool, int, float, str)


def metadata_type_name(value: Any) -> str:
    """给元数据值起一个人类可读的类型名（错误信息里用它）."""
    if value is None:
        return "NoneType（向量库不存 null，请改用空串或 0）"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple)):
        element = sorted({metadata_type_name(item) for item in value})
        return f"list[{', '.join(element)}]"
    if isinstance(value, dict):
        return "dict（不允许嵌套：向量库的 metadata 是**扁平**的一层）"
    return type(value).__name__


def is_valid_metadata_value(value: Any) -> bool:
    """判断一个元数据值是否符合公共契约（见模块 docstring）."""
    if isinstance(value, METADATA_SCALAR_TYPES):
        return True
    if isinstance(value, (list, tuple)):
        if not value:
            return False
        # 数组必须同类型：混类型的数组在 Chroma 上会被拒（见其 metadata 类型规则）
        kinds = {metadata_type_name(item) for item in value}
        return len(kinds) == 1 and all(
            isinstance(item, METADATA_SCALAR_TYPES) for item in value
        )
    return False


def assert_metadata(metadata: dict[str, Any]) -> None:
    """逐键校验元数据；不合法时抛 ``RecordError`` 并指出是哪个键、什么类型."""
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise RecordError(
                f"metadata 的键必须是非空字符串，收到 {key!r}（类型 {type(key).__name__}）"
            )
        if not is_valid_metadata_value(value):
            raise RecordError(
                f"metadata[{key!r}] 的类型不合法：{metadata_type_name(value)}。"
                "向量库只接受 str / int / float / bool 及**同类型**标量数组"
                "（不接受空数组、嵌套 list、dict、None）——"
                "复杂结构请先序列化成字符串再存。"
            )


def validate_vector(vector: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    """校验向量并冻结成 tuple（``VectorRecord`` 是 frozen 的，可变值会破坏它）.

    三条检查，每一条都对应一种"不报错但结果全错"的失败：

    ```text
    空向量        → 检索时永远返回空集，看起来像"库里没数据"
    非有限数      → nan/inf 参与的比较恒为 False，排序位置随机
    零向量        → 余弦相似度 0/0，被所有查询"平等地命中"或不命中
    ```
    """
    values = tuple(float(x) for x in vector)
    if not values:
        raise RecordError("向量不允许为空：一个 0 维向量既不能比距离，也不能建索引")
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise RecordError(
                f"向量的第 {index} 个分量不是有限数（{value!r}）。"
                "nan/inf 不会让写入失败，但会让**此后每一次排序都不可信**："
                "nan 参与的任何比较都返回 False，这条记录会落到任意位置。"
            )
    if norm_is_zero(values):
        raise RecordError(
            "向量是零向量（模长为 0）：它没有方向，余弦相似度对它是 0/0。"
            "零向量通常意味着编码器返回了空结果，请检查 embedding 提供方。"
        )
    return values


def norm_is_zero(vector: tuple[float, ...]) -> bool:
    """模长是否为 0（用 ``math.fsum`` 累加，避免"极小分量被浮点抵消成 0"）."""
    return math.fsum(value * value for value in vector) == 0.0


@dataclass(frozen=True)
class VectorRecord:
    """向量库里的一条记录：**id + 向量 + 原文 + 元数据**.

    字段分成两组，服务两组不同的读者：

    ```text
    检索侧   record_id / vector         → 排序与定位
    展示侧   text / metadata            → 命中之后还给用户看什么
    ```

    ``text`` 允许为空串（有些场景只做向量过滤，不需要原文），
    但**不允许只有空白**：一个只有空白的 ``text`` 会让"命中之后展示什么"
    这个问题在运行期才暴露。
    """

    record_id: str
    vector: tuple[float, ...]
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 向量校验放在这里而不是只放在 ``make_record`` 里：
        # 只要 ``VectorRecord`` 能被直接构造，就一定会有人直接构造它
        # （测试、脚本、调试）。**契约必须挂在类型上，而不是挂在某个工厂上**——
        # 挂在工厂上的检查可以被绕过，而绕过它的人不会收到任何提示。
        #
        # 顺带把 list 冻结成 tuple：frozen dataclass 的字段如果是 list，
        # 外部一句 ``record.vector[0] = 0.0`` 就能改掉"不可变"的记录，
        # 而那个改动会绕过所有校验。
        object.__setattr__(self, "vector", validate_vector(self.vector))
        if not isinstance(self.record_id, str) or not self.record_id.strip():
            raise RecordError(
                f"record_id 必须是非空字符串，收到 {self.record_id!r}。"
                "id 是更新与删除的唯一依据——没有它，任何一条记录都无法被单独替换。"
            )
        if len(self.record_id) > MAX_RECORD_ID_LENGTH:
            raise RecordError(
                f"record_id 长度 {len(self.record_id)} 超过上限 {MAX_RECORD_ID_LENGTH}："
                "id 应该是短标识（本项目用 day062 的 chunk_id，16 位十六进制），"
                "而不是一整段文本。"
            )
        assert_metadata(self.metadata)

    @property
    def dimension(self) -> int:
        """向量维度."""
        return len(self.vector)

    @property
    def char_count(self) -> int:
        """``text`` 的字符数（不含任何前缀：那是检索视图，不是内容）."""
        return len(self.text)

    def to_dict(self, *, include_vector: bool = False) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典.

        ``include_vector`` 默认 False：一个 384 维向量序列化成 JSON 之后
        大约 3 KB，一份 2000 块的库就是 6 MB——**报告里几乎从不需要它**，
        而带上它会让任何一次"看一眼统计"变成一次大响应。
        """
        payload: dict[str, Any] = {
            "record_id": self.record_id,
            "dimension": self.dimension,
            "char_count": self.char_count,
            "metadata": dict(self.metadata),
        }
        if self.text:
            payload["text"] = self.text
        if include_vector:
            payload["vector"] = list(self.vector)
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要（演示脚本逐条打印）."""
        preview = self.text.replace("\n", " ")[:40]
        return f"{self.record_id} | {self.dimension}d | {self.char_count} 字 | {preview}"


@dataclass(frozen=True)
class SearchHit:
    """一条命中：记录 + 两个口径的分数 + 名次.

    ``rank`` 从 0 开始，与 FAISS ``search`` 返回的**位置**一致、
    与它返回的 ``-1``（表示"结果不足，此处为空"）**不是一个东西**——
    ``-1`` 在进入本结构之前就已经被适配器丢掉了（见 ``faiss_backend``）。
    """

    record: VectorRecord
    score: float
    distance: float
    rank: int

    def to_dict(self, *, include_text: bool = True, include_vector: bool = False) -> dict[str, Any]:
        """投影为字典（``include_text=False`` 用于只看排序不变量的场景）."""
        payload: dict[str, Any] = {
            "rank": self.rank,
            "record_id": self.record.record_id,
            "score": round(self.score, 6),
            "distance": round(self.distance, 6),
            "dimension": self.record.dimension,
            "metadata": dict(self.record.metadata),
        }
        if include_text:
            payload["text"] = self.record.text
        if include_vector:
            payload["vector"] = list(self.record.vector)
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        preview = self.record.text.replace("\n", " ")[:36] if self.record.text else ""
        return (
            f"#{self.rank} {self.record.record_id} score={self.score:+.6f} "
            f"distance={self.distance:.6f} | {preview}"
        )


@dataclass(frozen=True)
class SearchResult:
    """一次查询的完整结果（含"这次是在什么条件下搜的"）.

    ``candidates`` 与 ``filter_applied`` 是这一层最重要的两个字段：
    它们把"**为什么只返回到 2 条**"变成一个可以直接读出来的事实。

    ```text
    candidates=0  且 filter_applied=True   → 过滤器把所有记录都排除了
    candidates=0  且 filter_applied=False  → 库是空的（或 top_k 之外没有数据）
    candidates=5  hits 只有 3 条           → min_score 阈值切掉了 2 条
    ```
    没有这两个字段时，三种情形在返回体里长得一样：``hits`` 很短。
    """

    hits: tuple[SearchHit, ...]
    metric: str
    top_k: int
    candidates: int = 0
    filter_applied: bool = False

    @property
    def count(self) -> int:
        """命中条数."""
        return len(self.hits)

    @property
    def scores(self) -> list[float]:
        """各命中的分数（统一口径：越大越近）."""
        return [hit.score for hit in self.hits]

    def ids(self) -> list[str]:
        """命中的记录 id，按名次排列（与外部库对账时直接用）."""
        return [hit.record.record_id for hit in self.hits]

    def top(self) -> SearchHit | None:
        """第一名（没有命中时返回 ``None``，而不是抛异常）."""
        return self.hits[0] if self.hits else None

    def to_dict(self, *, include_text: bool = True, include_vector: bool = False) -> dict[str, Any]:
        """投影为字典."""
        return {
            "metric": self.metric,
            "top_k": self.top_k,
            "candidates": self.candidates,
            "filter_applied": self.filter_applied,
            "count": self.count,
            "hits": [
                hit.to_dict(include_text=include_text, include_vector=include_vector)
                for hit in self.hits
            ],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        head = ", ".join(
            f"{hit.record.record_id}:{hit.score:+.4f}" for hit in self.hits[:3]
        )
        return (
            f"{self.metric} | top_k={self.top_k} | 候选 {self.candidates} | "
            f"命中 {self.count} | {head or '（无）'}"
        )


@dataclass(frozen=True)
class WriteReport:
    """一次写入的结果：四种去向，各自是一个数字.

    ```text
    added      新插入的记录
    updated    同一 id 被覆盖（内容确实变了）
    unchanged  同一 id、内容逐位相同 → **没有改动任何东西**
    skipped    被调用方或护栏拒绝的记录（例如超过单批上限）
    removed    delete 删掉的条数
    ```

    ``unchanged`` 单独一态是刻意的：**它是"这次写入其实什么都没做"
    的唯一证据**。day065 的增量索引就靠它来回答"哪些块不需要重新编码"——
    如果把它并进 ``updated``，"重建索引的代价"这个数就永远算不准
    （每个块都被报告成更新过，而实际上一个字节都没变）。
    """

    added: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped: int = 0
    removed: int = 0

    def __post_init__(self) -> None:
        for name in ("added", "updated", "unchanged", "skipped", "removed"):
            value = getattr(self, name)
            if value < 0:
                raise RecordError(f"WriteReport.{name} 必须非负，收到 {value}")

    @property
    def written(self) -> int:
        """实际写入库的条数（新增 + 覆盖）."""
        return self.added + self.updated

    @property
    def total(self) -> int:
        """本次调用处理过的记录总数."""
        return self.added + self.updated + self.unchanged + self.skipped

    @property
    def changed(self) -> int:
        """让库发生变化的条数（不含 ``unchanged``，含删除）."""
        return self.added + self.updated + self.removed

    def merge(self, other: WriteReport) -> WriteReport:
        """合并两批写入（分批写库时用）."""
        return WriteReport(
            added=self.added + other.added,
            updated=self.updated + other.updated,
            unchanged=self.unchanged + other.unchanged,
            skipped=self.skipped + other.skipped,
            removed=self.removed + other.removed,
        )

    def to_dict(self) -> dict[str, int]:
        """投影为字典（含两个派生量，便于直接断言）."""
        return {
            "added": self.added,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "skipped": self.skipped,
            "removed": self.removed,
            "written": self.written,
            "total": self.total,
            "changed": self.changed,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"新增 {self.added} / 覆盖 {self.updated} / 未变 {self.unchanged} / "
            f"跳过 {self.skipped} / 删除 {self.removed}"
        )


@dataclass(frozen=True)
class StoreInfo:
    """一个后端的当前状态（``/vectorstore/stats`` 直接返回它）."""

    backend: str
    metric: str
    dimension: int
    count: int
    persistent: bool = False
    location: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """投影为字典."""
        return {
            "backend": self.backend,
            "metric": self.metric,
            "dimension": self.dimension,
            "count": self.count,
            "persistent": self.persistent,
            "location": self.location,
            "extra": dict(self.extra),
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        where = self.location or "（内存）"
        return (
            f"{self.backend} | {self.metric} | {self.dimension}d | "
            f"{self.count} 条 | 持久化 {'是' if self.persistent else '否'} | {where}"
        )


def make_record(
    record_id: str,
    vector: list[float] | tuple[float, ...],
    text: str = "",
    metadata: dict[str, Any] | None = None,
    *,
    metric: str = METRIC_COSINE,
) -> VectorRecord:
    """构造一条记录，并按 ``metric`` 决定要不要先归一化.

    归一化只对 ``cosine`` 做：``ip`` 的语义**就是**不归一化的内积，
    替它归一化会把"两个度量不同"这件事悄悄抹掉——而保留这个差别
    正是验证"提供方到底归没归一化"的方法（见 ``metrics``）。

    写成工厂函数而不是让调用方自己调 ``VectorRecord``，是为了把
    "归不归一化"这个决定**集中在一处**：散到各个调用点之后，
    总会出现"写入时归一化了、查询时忘了"这一类不对称的 bug。
    """
    resolved = normalize_metric(metric)
    values = list(vector)
    if resolved == METRIC_COSINE:
        values = _l2_normalize(values)
    return VectorRecord(
        record_id=record_id,
        vector=tuple(values),
        text=text,
        metadata=dict(metadata or {}),
    )


def score_hit(
    record: VectorRecord,
    query: list[float] | tuple[float, ...],
    metric: str,
    rank: int,
) -> SearchHit:
    """给一条记录算出命中结构（两个口径都算，见模块 docstring）."""
    return SearchHit(
        record=record,
        score=score(metric, query, record.vector),
        distance=distance(metric, query, record.vector),
        rank=rank,
    )


def sort_hits(hits: list[SearchHit]) -> list[SearchHit]:
    """按"分数降序 + id 升序"排序并重排名次.

    第二个键不是装饰：**浮点分数相等的情形一定会出现**——相同文本
    用同一个确定性编码器会得到逐位相同的向量，于是分数精确相等。
    没有第二键时，``sorted`` 的顺序取决于字典的迭代顺序，
    而字典顺序取决于插入顺序；插入顺序在批量重放之后会变，
    于是"同一次查询两次运行给出不同的 top-1"。

    这属于"确定性要能被回答"这条纪律的边界情形：
    **不确定性不是来自算法，而是来自没写下来的排序规则。**
    """
    ordered = sorted(hits, key=lambda hit: (-hit.score, hit.record.record_id))
    return [
        SearchHit(
            record=hit.record,
            score=hit.score,
            distance=hit.distance,
            rank=position,
        )
        for position, hit in enumerate(ordered)
    ]


def _l2_normalize(vector: list[float]) -> list[float]:
    """L2 归一化；零向量原样返回（它的真正拒绝在 ``validate_vector`` 里）."""
    total = math.fsum(value * value for value in vector)
    if total == 0.0:
        return list(vector)
    length = math.sqrt(total)
    return [value / length for value in vector]


__all__ = [
    "MAX_RECORD_ID_LENGTH",
    "METADATA_SCALAR_TYPES",
    "SearchHit",
    "SearchResult",
    "StoreInfo",
    "VectorRecord",
    "WriteReport",
    "assert_metadata",
    "is_valid_metadata_value",
    "make_record",
    "metadata_type_name",
    "norm_is_zero",
    "score_hit",
    "sort_hits",
    "validate_vector",
]
