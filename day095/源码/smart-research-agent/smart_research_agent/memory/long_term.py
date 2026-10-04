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
        # 必须用 is None 判断：空向量库 __len__ == 0 是假值，
        # `store or ...` 会把调用方传入的空库静默换成新库（day041 修复）
        self._store = store if store is not None else InMemoryVectorStore()

    def remember(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        record_id = uuid.uuid4().hex[:12]
        vector = self._embedding.embed(text)
        self._store.add(VectorRecord(id=record_id, text=text, vector=vector, metadata=metadata or {}))
        return record_id

    def remember_many(
        self, texts: list[str], metadatas: list[dict[str, Any]] | None = None
    ) -> list[str]:
        """批量写入长期记忆（day041）：一次 embed_batch 编码整批.

        相比循环调用 ``remember``，神经/云端 embedding 只跑一次前向或
        一次 HTTP 请求。``metadatas`` 给出时必须与 ``texts`` 等长。
        """
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas 长度必须与 texts 一致")
        vectors = self._embedding.embed_batch(list(texts))
        record_ids: list[str] = []
        for i, (text, vector) in enumerate(zip(texts, vectors)):
            record_id = uuid.uuid4().hex[:12]
            metadata = metadatas[i] if metadatas is not None else {}
            self._store.add(
                VectorRecord(id=record_id, text=text, vector=vector, metadata=metadata)
            )
            record_ids.append(record_id)
        return record_ids

    def recall(self, query: str, top_k: int = 3, min_score: float | None = None) -> list[str]:
        """检索最相关的记忆文本；``min_score`` 透传向量库的阈值过滤（day041）."""
        query_vector = self._embedding.embed(query)
        return [
            r.text
            for r, _score in self._store.search(query_vector, top_k=top_k, min_score=min_score)
        ]
