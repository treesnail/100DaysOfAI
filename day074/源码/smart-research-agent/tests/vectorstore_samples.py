"""day064 测试用的向量库样本构造器（记录表 + 查询 + 手工向量表 + 参照实现）.

放在单独模块里而不是复制到每个测试文件，理由与 ``chunking_samples`` 一样：
**同一份样本要被六七个测试文件用到**（types / metrics / filters / 三个后端 /
parity / 端点），复制多份会让"样本里加一个字段"变成要改许多处，
而漏改的那一处只表现为"某个测试不再覆盖那条语义"。

## 这批样本刻意埋了四个差异点

| 差异点 | 怎么埋的 | 它让哪一条断言成为可能 |
|--------|---------|----------------------|
| 余弦**并列** | ``r_a`` 与 ``r_d`` 对同一查询的余弦都是 1.0 | 平局的排序必须由 id 决定（不是由插入顺序） |
| 内积与余弦**不一致** | ``r_d`` 的模长是 10，其余是 1 | ``ip`` 的 top-1 与 ``cosine`` 的 top-1 不是同一条 |
| L2 与余弦**不一致** | 同一个 ``r_d`` 在 L2 下离得最远 | 三种度量不能互相"顺手替换" |
| 元数据类型齐全 | 字符串 / 整数 / 布尔 / **同类型数组** | ``$gt`` / ``$in`` / ``$contains`` 各自有用武之地 |

## 为什么这里自带一份"参照实现"

``brute_force_top`` 是**不复用 ``metrics.py`` 的一段独立计算**：
只用 ``math`` 与最朴素的循环。它的用途是给测试提供一个"不算错也写不对"
的对照物——如果它和 ``metrics`` 用的是同一段代码，
那么"两者一致"就只能证明"这段代码等于它自己"。

这与 day062 的探针集、day063 的只读核对是同一条方法论：
**判定的依据必须与被判定的实现相互独立，否则它只是在复述实现。**
"""

from __future__ import annotations

import math
from typing import Any

from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.vectorstore import VectorRecord, make_record

#: 样本向量维度。取 8 而不是 384：期望值能手算，
#: 而"手算得出来的期望值"是这一课所有断言的立足点。
VECTOR_DIMENSION = 8

#: 六条样本记录的 id（形态取自 day062 的 ``chunk_id``：16 位十六进制）.
RECORD_IDS: tuple[str, ...] = (
    "4a6680cde33da45e",
    "90a20c398e18fb0b",
    "1c9f4b7a2e6d8035",
    "7b3e0d5c9a14f286",
    "e2a94f61c7d30b58",
    "35d8c1b0a6e47f92",
)

#: 手工向量表：``(id, 向量, 文本, 元数据)``.
#:
#: ``r_a``（下标 0）与 ``r_d``（下标 3）都是"与查询 q1 完全同向"，
#: 但 ``r_d`` 的模长是 10 倍 —— 这就是"余弦并列、内积分家"的来源。
RECORD_SPECS: tuple[tuple[str, tuple[float, ...], str, dict[str, Any]], ...] = (
    (
        RECORD_IDS[0],
        (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "固定长度分块每 320 个字符切一刀，重叠 48 个字符。",
        {
            "parent_doc_id": "7feafd1c95ab3577",
            "strategy": "recursive",
            "index": 0,
            "token_count": 319,
            "heading_path": "检索手册 > 分块",
            "topic": "chunking",
            "tags": ["guide", "chunk"],
            "oversized": False,
        },
    ),
    (
        RECORD_IDS[1],
        (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "默认返回前 5 条，相似度低于 0.35 的直接丢掉。",
        {
            "parent_doc_id": "7feafd1c95ab3577",
            "strategy": "recursive",
            "index": 1,
            "token_count": 210,
            "heading_path": "检索手册 > 检索",
            "topic": "retrieval",
            "tags": ["guide", "search"],
            "oversized": False,
        },
    ),
    (
        RECORD_IDS[2],
        (0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "一次全量重建索引大约 12 万条记录，按每条 320 token 计。",
        {
            "parent_doc_id": "7feafd1c95ab3577",
            "strategy": "fixed",
            "index": 2,
            "token_count": 160,
            "heading_path": "检索手册 > 成本",
            "topic": "cost",
            "tags": ["guide", "cost"],
            "oversized": False,
        },
    ),
    (
        RECORD_IDS[3],
        (10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "阈值用分位数标定，换 embedding 模型必须重新标定。",
        {
            "parent_doc_id": "b41ad29c8f70e613",
            "strategy": "structural",
            "index": 0,
            "token_count": 120,
            "heading_path": "语义缓存 > 阈值",
            "topic": "cache",
            "tags": ["guide", "cache"],
            "oversized": False,
        },
    ),
    (
        RECORD_IDS[4],
        (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "术语表列出本手册用到的全部缩写。",
        {
            "parent_doc_id": "b41ad29c8f70e613",
            "strategy": "semantic",
            "index": 0,
            "token_count": 90,
            "heading_path": "附录 > 术语表",
            "topic": "glossary",
            "tags": ["appendix"],
            "oversized": False,
        },
    ),
    (
        RECORD_IDS[5],
        (0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0),
        "参数表把默认值与取值范围并排列出。",
        {
            "parent_doc_id": "b41ad29c8f70e613",
            "strategy": "structural",
            "index": 1,
            "token_count": 64,
            "heading_path": "附录 > 参数表",
            "topic": "glossary",
            "tags": ["appendix", "cost"],
            "oversized": True,
        },
    ),
)

#: 两个查询向量。
#:
#: ``q_axis`` 与 ``r_a`` / ``r_d`` 完全同向（用来制造并列）；
#: ``q_tilt`` 偏向 ``(0.6, 0.8)``，让 ``r_c`` 成为最近的那条。
SAMPLE_QUERIES: dict[str, tuple[float, ...]] = {
    "q_axis": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "q_tilt": (0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "q_orthogonal": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
}

#: 记录 id → 文本（``TableEmbedding`` 的查表键就是这段文本）.
RECORD_TEXTS: dict[str, str] = {
    spec[0]: spec[2] for spec in RECORD_SPECS
}


def sample_vectors() -> dict[str, tuple[float, ...]]:
    """id → 原始向量（**未归一化**：``make_record`` 会按度量决定要不要归一化）."""
    return {spec[0]: spec[1] for spec in RECORD_SPECS}


def sample_metadata() -> dict[str, dict[str, Any]]:
    """id → 元数据（覆盖 str / int / bool / 同类型数组四种类型）."""
    return {spec[0]: dict(spec[3]) for spec in RECORD_SPECS}


def sample_records(*, metric: str = "cosine") -> list[VectorRecord]:
    """六条 ``VectorRecord``（按 ``RECORD_IDS`` 顺序）."""
    return [
        make_record(
            record_id=spec[0],
            vector=spec[1],
            text=spec[2],
            metadata=dict(spec[3]),
            metric=metric,
        )
        for spec in RECORD_SPECS
    ]


def knowledge_records() -> list[dict[str, Any]]:
    """day062 ``ChunkSet.knowledge_records()`` 形状的样本（四个键）.

    形状与 day062 的产物逐键一致（``doc_id`` / ``source`` / ``text`` /
    ``metadata``），因为它就是 day064 的**真实**输入——
    用别的形状造样本会让"两个模块能接上"这件事永远没被验证过。
    """
    records: list[dict[str, Any]] = []
    for spec in RECORD_SPECS:
        metadata = {
            "parent_doc_id": spec[3]["parent_doc_id"],
            "strategy": spec[3]["strategy"],
            "index": str(spec[3]["index"]),
            "token_count": str(spec[3]["token_count"]),
            "token_measurer": "chars",
            "heading_path": spec[3]["heading_path"],
            "fingerprint": spec[0][::-1],
            "oversized": "true" if spec[3]["oversized"] else "false",
            # 检索视图与正文不同：pipeline 必须优先用这一个（day062 的约定）
            "retrieval_text": f"{spec[3]['heading_path']}\n{spec[2]}",
        }
        records.append(
            {
                "doc_id": spec[0],
                "source": f"docs/{spec[3]['parent_doc_id']}.md",
                "text": spec[2],
                "metadata": metadata,
            }
        )
    return records


class TableEmbedding(EmbeddingProvider):
    """按文本查表的确定性编码器：向量由样本写死（**"语义"由人给定**）.

    为什么不用 ``MockEmbedding``：它把文本哈希成向量，于是"哪条最近"
    只能靠实际跑一遍才知道——那样写出来的断言是在复述实现。
    这里把向量与文本的对应关系写死在样本里，**期望值是手算出来的**。

    它顺带记下每次 ``embed_batch`` 收到的文本（``batches``）：
    批量边界是查询方与编码方之间的一份契约（day065 会把它做成可配置的
    批大小），本课只需要它能被断言"一次给定的批没有被拆散"。
    """

    def __init__(
        self,
        table: dict[str, tuple[float, ...]] | None = None,
        *,
        default: tuple[float, ...] | None = None,
        dimension: int | None = None,
    ) -> None:
        entries = dict(table or sample_vectors())
        if default is None:
            resolved_dimension = dimension or len(next(iter(entries.values())))
            default = (1.0,) + (0.0,) * (resolved_dimension - 1)
        elif dimension is not None and len(default) != dimension:
            raise ValueError("default 的长度与 dimension 不一致")
        self._table = entries
        self._default = tuple(float(x) for x in default)
        self._dimension = dimension or len(self._default)
        #: 每次 ``embed_batch`` 的入参快照（断言"批没有被拆散"）
        self.batches: list[list[str]] = []
        #: 逐条 ``embed`` 的调用次数（用来对比批与逐条两条路径）
        self.single_calls = 0

    @property
    def dimension(self) -> int:
        """向量维度."""
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """按文本查表；查不到用 ``default``（并记一次逐条调用）."""
        self.single_calls += 1
        return list(self._table.get(text, self._default))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码：记录这一批的入参，再逐条查表."""
        self.batches.append(list(texts))
        return [self.embed(text) for text in texts]


def table_embedding() -> TableEmbedding:
    """``TableEmbedding`` 的便捷构造器（用样本向量表）."""
    return TableEmbedding()


def record_embedding_texts() -> list[str]:
    """六条样本记录的正文（按 ``RECORD_IDS`` 顺序）."""
    return [RECORD_TEXTS[record_id] for record_id in RECORD_IDS]


def record_embedding_vectors() -> list[list[float]]:
    """六条样本记录的向量（按 ``RECORD_IDS`` 顺序）."""
    vectors = sample_vectors()
    return [list(vectors[record_id]) for record_id in RECORD_IDS]


def brute_force_top(
    records: list[VectorRecord],
    query: list[float] | tuple[float, ...],
    metric: str,
    top_k: int,
) -> list[tuple[str, float]]:
    """独立的暴力检索参照实现（**不复用 ``metrics.py``**）.

    只用 ``math`` 与最朴素的循环写一遍三种度量的公式：

    ```text
    cosine   cos = Σ(a·b) / (|a| · |b|)
    ip       dot = Σ(a·b)
    l2       score = −Σ(a − b)²        （取负号让"越大越近"统一）
    ```

    排序规则也与本包一致：**分数降序、id 升序**。
    它与 ``metrics.py`` 的关系是"两份独立实现互相对账"，
    而不是"一份实现被调用两次"。
    """
    scored: list[tuple[str, float]] = []
    for record in records:
        value = _reference_score(query, record.vector, metric)
        scored.append((record.record_id, value))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:top_k]


def _reference_score(
    query: list[float] | tuple[float, ...],
    vector: tuple[float, ...],
    metric: str,
) -> float:
    if len(query) != len(vector):
        raise ValueError("参照实现收到维度不一致的向量")
    products = math.fsum(float(a) * float(b) for a, b in zip(query, vector))
    if metric == "cosine":
        query_norm = math.sqrt(math.fsum(float(a) * float(a) for a in query))
        vector_norm = math.sqrt(math.fsum(float(b) * float(b) for b in vector))
        if query_norm == 0.0 or vector_norm == 0.0:
            return 0.0
        return products / (query_norm * vector_norm)
    if metric == "ip":
        return products
    if metric == "l2":
        return -math.fsum((float(a) - float(b)) ** 2 for a, b in zip(query, vector))
    raise ValueError(f"参照实现不支持度量 {metric!r}")


__all__ = [
    "RECORD_IDS",
    "RECORD_SPECS",
    "RECORD_TEXTS",
    "SAMPLE_QUERIES",
    "VECTOR_DIMENSION",
    "TableEmbedding",
    "brute_force_top",
    "knowledge_records",
    "record_embedding_texts",
    "record_embedding_vectors",
    "sample_metadata",
    "sample_records",
    "sample_vectors",
    "table_embedding",
]
