"""Embedding 提供方抽象与五种实现（day045 起含本地 Ollama 实现）.

提供方按"教学 → 生产"排列：

- ``MockEmbedding``：SHA-256 哈希伪向量。离线、确定性，但**没有语义**——
  语义相近的文本不会得到相近的向量，只适合测试占位；
- ``CharNgramEmbedding``：字符 n-gram 哈希 + L2 归一化。离线、确定性，
  具备"词汇层面"相似度——共享字符 n-gram 越多的文本向量越接近；
- ``SentenceTransformerEmbedding``：本地神经语义向量，封装
  sentence-transformers（可选依赖，构造时才惰性导入）；
- ``OpenAIEmbedding``：云端 embedding API（text-embedding-3-* 系列）；
- ``OllamaEmbedding``：本地 Ollama embedding（day045，OpenAI 兼容端点，
  ``/v1/embeddings``，维度按模型惰性探测）。

选型统一走 ``default_embedding()`` 工厂 + ``settings.embedding_provider``，
消费方（长期记忆、RAG 评估、API 层）只依赖 ``EmbeddingProvider`` 抽象。
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any

from smart_research_agent.config import settings
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    """文本向量化提供方抽象."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """把文本编码为定长向量."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码（day041）.

        默认实现是逐条调用 ``embed``——功能正确但不一定最优。神经模型与
        云端 API 可以覆写它以获得真正的批处理收益（一次前向/一次 HTTP
        处理整批）。设计成**带默认实现的具体方法**而非抽象方法，是为了
        向后兼容：已有的提供方（如 MockEmbedding）不覆写也能继续工作，
        与 day040 ``supports_vision`` 默认值的设计哲学一致。
        """
        return [self.embed(text) for text in texts]


def l2_normalize(vector: list[float]) -> list[float]:
    """L2 归一化：把向量缩放到单位长度（模长为 1）.

    归一化后余弦相似度退化为点积（``cos(a, b) = a·b``），计算更快、
    且所有文本的向量落在同一尺度上，相似度阈值（如 min_score=0.5）
    才有稳定的物理含义。零向量原样返回，避免除零。
    """
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return list(vector)
    return [x / norm for x in vector]


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


class CharNgramEmbedding(EmbeddingProvider):
    """字符 n-gram 哈希 embedding：离线、确定性、有词汇层面语义（day041）.

    原理（hashing trick）：

    1. 把文本切成若干长度为 n 的字符片段（n-gram），n 取 2 与 3——
       二元组捕捉局部搭配，三元组捕捉更长的词形；
    2. 每个 n-gram 用 MD5 哈希到 ``dimension`` 个桶之一并累计计数；
    3. 计数向量做 L2 归一化，得到单位长度的输出。

    两段文本共享的 n-gram 越多，计数向量方向越接近——因此
    "我爱机器学习"与"我喜欢机器学习"的相似度，会显著高于它们与
    "今天晚饭吃什么"的相似度。这是 MockEmbedding 不具备的性质。

    局限：相似度来自**字面重叠**，同义不同形（"西红柿"与"番茄"）
    得不到高相似度——这正是神经 embedding（第三章）要解决的问题。
    """

    def __init__(self, dimension: int = 256, ngram_sizes: tuple[int, ...] = (2, 3)):
        if dimension <= 0:
            raise ValueError("dimension 必须为正整数")
        if not ngram_sizes or any(n <= 0 for n in ngram_sizes):
            raise ValueError("ngram_sizes 必须为非空的正整数序列")
        self._dimension = dimension
        self._ngram_sizes = ngram_sizes

    @property
    def dimension(self) -> int:
        return self._dimension

    def _iter_ngrams(self, text: str):
        """产出文本的全部字符 n-gram.

        文本比某个 n 还短时，把整段文本当作一个 gram 产出——保证
        单字文本也有非零向量，而不是被静默丢弃。
        """
        for n in self._ngram_sizes:
            if len(text) < n:
                yield text
                continue
            for i in range(len(text) - n + 1):
                yield text[i : i + n]

    def embed(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self._dimension
        vector = [0.0] * self._dimension
        for gram in self._iter_ngrams(text):
            bucket = int.from_bytes(hashlib.md5(gram.encode("utf-8")).digest()[:4], "big")
            vector[bucket % self._dimension] += 1.0
        return l2_normalize(vector)


class SentenceTransformerEmbedding(EmbeddingProvider):
    """本地神经语义向量（day041）：封装 sentence-transformers.

    sentence-transformers 是**可选依赖**（需 torch，体积大）：本类在
    构造时才惰性导入它，未安装会在构造点抛出带安装指引的清晰错误，
    而不是让整个包在 import 时失败。测试通过 ``model`` 参数注入假模型
    对象，无需真实权重即可覆盖全部逻辑。

    默认模型 ``paraphrase-multilingual-MiniLM-L12-v2``：384 维、
    支持 50+ 语言（含中文）、体积小（~470MB），是中文场景常用的
    起步选择；英文专用可换 ``all-MiniLM-L6-v2``（384 维，~90MB）。
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        model: Any | None = None,
    ):
        self._model_name = model_name
        if model is not None:
            self._model = model
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "未安装 sentence-transformers，无法构建本地神经 embedding。"
                    "请先执行: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(model_name)
        self._dimension = self._resolve_dimension()

    def _resolve_dimension(self) -> int:
        """解析模型输出维度，兼容 sentence-transformers 新旧版本.

        v5.4 起 ``get_sentence_embedding_dimension()`` 更名为
        ``get_embedding_dimension()``（旧名仍可用但会告警）。优先用
        新方法，缺失时退回旧方法。
        """
        getter = getattr(self._model, "get_embedding_dimension", None)
        if getter is None:
            getter = self._model.get_sentence_embedding_dimension
        dim = getter()
        if not dim:
            raise RuntimeError(f"无法确定模型 {self._model_name} 的输出维度")
        return int(dim)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """一次前向编码整批（神经模型的批处理收益所在）.

        ``normalize_embeddings=True`` 让输出直接是单位向量，与
        余弦相似度/点积检索对齐。注意 ``encode`` 对"单个字符串"返回
        一维数组、对"字符串列表"返回二维数组——因此 ``embed`` 一律
        走列表入口，避免形状分支。
        """
        vectors = self._model.encode(list(texts), normalize_embeddings=True)
        return [[float(x) for x in row] for row in vectors]

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


#: OpenAI embedding 模型的已知维度表（官方规格）：
#: text-embedding-3-small 1536 维（$0.02/1M tokens），
#: text-embedding-3-large 3072 维（$0.13/1M tokens），
#: text-embedding-ada-002 1536 维（$0.10/1M tokens，旧版）。
OPENAI_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedding(EmbeddingProvider):
    """云端 embedding API（day041）：OpenAI ``embeddings.create``.

    一次请求可携带整批文本（``input`` 为列表），响应 ``data`` 数组与
    输入顺序一一对应。``text-embedding-3-*`` 系列还支持 ``dimensions``
    参数截断输出维度（Matryoshka 表示学习）——本实现按模型默认维度
    请求，截断需求可在构造后自行扩展。
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
        dimension: int | None = None,
    ):
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=api_key or settings.openai_api_key or "sk-missing",
                base_url=base_url or settings.openai_base_url,
            )
        self._model = model
        resolved = dimension or OPENAI_EMBEDDING_DIMENSIONS.get(model)
        if resolved is None:
            raise ValueError(f"未知模型 {model} 的向量维度，请显式传入 dimension")
        self._dimension = resolved

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self) -> str:
        return self._model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """一次 HTTP 调用编码整批；空输入直接返回，不发请求."""
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        return [list(item.embedding) for item in resp.data]

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


class OllamaEmbedding(EmbeddingProvider):
    """本地 Ollama embedding（day045）：把向量化也搬回自己机器.

    Ollama 同时提供原生 ``/api/embed`` 与 OpenAI 兼容的 ``/v1/embeddings``；
    这里走兼容端点，因此 **API 调用代码与 ``OpenAIEmbedding`` 完全同构**，
    差别只在 base_url 指向本机、api_key 是占位值——又一次印证"协议兼容
    比 SDK 封装更重要"。

    唯一真正不同的地方是**维度**：云端 ``text-embedding-3-*`` 的维度可以
    从常量表查（1536/3072），而本地 embedding 模型的输出维度取决于权重
    （``nomic-embed-text`` 为 768，其他模型各异），既没有权威常量表、
    也不该写死。所以本实现的维度是**首次调用后从响应推断**的惰性属性：
    ``dimension`` 在未显式传入且尚未调用过时，会触发一次极短文本的探测。

    维度为什么要紧：向量库（day041 的 ``VectorStore``）与语义缓存
    （day043）都按维度对齐存储，维度不一致的向量混进同一个库，检索结果
    就是静默错乱——这类 bug 不报错，只变笨。
    """

    #: Ollama 上最常用的 embedding 模型（137M 参数，768 维，约 274MB）
    DEFAULT_MODEL = "nomic-embed-text"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        dimension: int | None = None,
    ):
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=api_key or settings.local_api_key or "ollama",
                base_url=base_url or settings.local_base_url,
            )
        self._model = model
        self._dimension = dimension

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        """向量维度：显式传入则直接返回，否则探测一次并缓存."""
        if self._dimension is None:
            self.embed("dimension probe")
        assert self._dimension is not None  # embed 成功后必然已回填
        return self._dimension

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """一次调用编码整批；顺带在首次响应里补齐维度."""
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        vectors = [list(item.embedding) for item in resp.data]
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])
        return vectors

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]


def default_embedding() -> EmbeddingProvider:
    """按 ``settings.embedding_provider`` 构建默认 embedding 提供方（day041）.

    取值与退避策略：

    - ``mock`` / ``char-ngram``：离线实现，无条件可用（默认 char-ngram）；
    - ``sentence-transformer``：需要已安装 sentence-transformers 与模型权重；
    - ``ollama``：本地 Ollama embedding（day045），需要本机服务在跑；
    - ``openai``：需要 API Key；未配置密钥时退回 CharNgramEmbedding 并
      记录告警——与 ``default_llm`` 的"离线可启动"哲学一致。
    """
    provider = settings.embedding_provider
    if provider == "mock":
        return MockEmbedding()
    if provider == "char-ngram":
        return CharNgramEmbedding()
    if provider == "ollama":
        return OllamaEmbedding(model=settings.local_embedding_model)
    if provider == "sentence-transformer":
        return SentenceTransformerEmbedding(model_name=settings.embedding_model)
    if provider == "openai":
        if not settings.openai_api_key:
            logger.warning("未配置 OPENAI_API_KEY，embedding 退回 CharNgramEmbedding 离线实现")
            return CharNgramEmbedding()
        return OpenAIEmbedding(model=settings.openai_embedding_model)
    raise ValueError(
        f"未知 embedding_provider: {provider}，"
        "可选值: mock / char-ngram / sentence-transformer / ollama / openai"
    )
