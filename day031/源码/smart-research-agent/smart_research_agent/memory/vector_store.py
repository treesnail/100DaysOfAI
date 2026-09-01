"""纯 Python 向量库：余弦相似度检索，零外部依赖的教学版实现."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VectorRecord:
    id: str
    text: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """余弦相似度：值域 [-1, 1]，1 表示方向完全一致."""
    if len(a) != len(b):
        raise ValueError("向量维度不一致")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    """内存向量库：暴力最近邻搜索."""

    def __init__(self) -> None:
        self._records: dict[str, VectorRecord] = {}

    def add(self, record: VectorRecord) -> None:
        self._records[record.id] = record

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[VectorRecord, float]]:
        scored = [(r, cosine_similarity(query_vector, r.vector)) for r in self._records.values()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def delete(self, record_id: str) -> None:
        self._records.pop(record_id, None)

    def __len__(self) -> int:
        return len(self._records)
