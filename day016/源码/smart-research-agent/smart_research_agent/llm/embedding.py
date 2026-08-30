"""Embedding 提供方抽象与离线确定性实现."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """文本向量化提供方抽象."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """把文本编码为定长向量."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度."""


class MockEmbedding(EmbeddingProvider):
    """基于 SHA-256 哈希的确定性伪向量.

    离线、可复现，用于教学与测试。注意：它没有真实语义，
    语义相近的文本不会得到相近的向量；生产环境应替换为真实 embedding 服务。
    """

    def __init__(self, dimension: int = 64):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(digest[i % len(digest)] / 255.0) * 2 - 1 for i in range(self._dimension)]
