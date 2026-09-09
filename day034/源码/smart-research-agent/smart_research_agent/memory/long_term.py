"""长期记忆：向量化存储与按需检索."""

from __future__ import annotations

import uuid
from typing import Any

from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.memory.vector_store import InMemoryVectorStore, VectorRecord


class LongTermMemory:
    """长期记忆 = Embedding 编码 + 向量库存储 + 相似度检索."""

    def __init__(self, embedding: EmbeddingProvider, store: InMemoryVectorStore | None = None):
        self._embedding = embedding
        self._store = store or InMemoryVectorStore()

    def remember(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        record_id = uuid.uuid4().hex[:12]
        vector = self._embedding.embed(text)
        self._store.add(VectorRecord(id=record_id, text=text, vector=vector, metadata=metadata or {}))
        return record_id

    def recall(self, query: str, top_k: int = 3) -> list[str]:
        query_vector = self._embedding.embed(query)
        return [r.text for r, _score in self._store.search(query_vector, top_k=top_k)]
