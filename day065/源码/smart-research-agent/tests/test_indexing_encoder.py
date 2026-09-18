"""``indexing.encoder`` 的测试：批量、缓存、去重、四项校验（M6-D4）.

断言分四组，每一组都对应 encoder.py 里一条**写下来就必须被守住**的承诺：

```text
批量与成本    encoded / batches 按批大小算得出来；全命中时提供方一次都不被调用
缓存与顺序    命中不改变结果；批内重复只编码一次；结果与入参逐位对应
四项校验      条数 / 维度 / 非有限 / 零向量 → EncodingError，且脏数据不进缓存
身份          describe_embedding 只用公开属性；identity= 覆盖时以覆盖值为准
```

全部离线、确定性、零网络：向量由本文件里几个假提供方写死，
"提供方被调用了几次"本身就是要断言的期望值（那是本课省钱的唯一证据）。
"""

from __future__ import annotations

import hashlib

import pytest

from smart_research_agent.config import settings
from smart_research_agent.indexing.cache import EmbeddingCache
from smart_research_agent.indexing.encoder import (
    DEFAULT_BATCH_SIZE,
    MODEL_LABEL_FALLBACK,
    BatchEncoder,
    describe_embedding,
)
from smart_research_agent.indexing.errors import EncodingError, IndexingError
from smart_research_agent.indexing.types import EmbeddingIdentity
from smart_research_agent.llm.embedding import (
    CharNgramEmbedding,
    EmbeddingProvider,
    MockEmbedding,
)

#: 全部假提供方的维度。取 4 而不是 384：期望值能手算，向量也一眼能看完。
DIMENSION = 4


def _vector_for(text: str, dimension: int = DIMENSION) -> list[float]:
    """由文本确定性地造一个**非零、有限**的向量（每个分量落在 [0.1, 0.9]）.

    用 sha256 而不是"字符码求和再取模"：后者在``seed`` 与 ``seed + 9``
    之间有系统性碰撞（实测 "alpha" 与 "beta" 会得到同一个向量），
    而"两段不同文本应当得到不同向量"正是本文件好几条断言的前提。
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [((digest[index % len(digest)] % 9) + 1) / 10.0 for index in range(dimension)]


# --------------------------------------------------------------------------- #
# 假提供方（都不是真实模型，因此完全离线且确定性）
# --------------------------------------------------------------------------- #


class CountingEmbedding(EmbeddingProvider):
    """确定性编码器：分别记下 ``embed`` / ``embed_batch`` 被调用了几次.

    "提供方一次都没被调用"这句话在本课是一个**可断言的期望值**，
    而不是一种感觉——它必须有地方被记下来。
    """

    def __init__(self, dimension: int = DIMENSION) -> None:
        self._dimension = dimension
        self.embed_calls = 0
        self.batch_calls = 0
        self.batch_sizes: list[int] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return _vector_for(text, self._dimension)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        self.batch_sizes.append(len(texts))
        return [self.embed(text) for text in texts]


class NamedEmbedding(EmbeddingProvider):
    """带公开 ``model`` 属性的提供方（模拟 OpenAIEmbedding / OllamaEmbedding）."""

    def __init__(self, model: str = "fake-model", dimension: int = DIMENSION) -> None:
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self) -> str:
        return self._model

    def embed(self, text: str) -> list[float]:
        return _vector_for(text, self._dimension)


class ModelNameEmbedding(EmbeddingProvider):
    """带公开 ``model_name`` 属性的提供方（模拟 SentenceTransformerEmbedding）."""

    def __init__(self, model_name: str = "fake-name", dimension: int = DIMENSION) -> None:
        self._model_name = model_name
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, text: str) -> list[float]:
        return _vector_for(text, self._dimension)


class BadCountEmbedding(EmbeddingProvider):
    """批量入口**少返回一条**：数量不符就无法按序对应."""

    def __init__(self, dimension: int = DIMENSION) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        return _vector_for(text, self._dimension)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts][:1]


class BadDimensionEmbedding(EmbeddingProvider):
    """声明的维度是对的，实际给出的向量多一维（维度护栏要拦的第一种脏数据）."""

    def __init__(self, declared: int = DIMENSION) -> None:
        self._declared = declared

    @property
    def dimension(self) -> int:
        return self._declared

    def embed(self, text: str) -> list[float]:
        return _vector_for(text, self._declared + 1)


class NanEmbedding(EmbeddingProvider):
    """第一条分量是 ``nan``：写进库之后所有相似度都会变成 nan."""

    def __init__(self, dimension: int = DIMENSION) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        vector = _vector_for(text, self._dimension)
        vector[0] = float("nan")
        return vector


class ZeroEmbedding(EmbeddingProvider):
    """零向量：没有方向，余弦相似度对它是 0/0."""

    def __init__(self, dimension: int = DIMENSION) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        return [0.0] * self._dimension


# --------------------------------------------------------------------------- #
# 批量与成本：encoded / batches / 提供方调用次数
# --------------------------------------------------------------------------- #


def test_batches_follow_batch_size_32() -> None:
    """5 条文本、批大小 32 → 一批装下，``batches == 1``."""
    provider = CountingEmbedding()
    encoder = BatchEncoder(provider, batch_size=32)
    vectors, report = encoder.encode([f"t{index}" for index in range(5)])

    assert len(vectors) == 5
    assert report.texts == 5
    assert report.encoded == 5
    assert report.batches == 1
    assert report.cache_hits == 0
    assert report.dimension == DIMENSION
    assert report.provider == "CountingEmbedding"
    assert provider.batch_sizes == [5]


def test_batches_follow_batch_size_2() -> None:
    """同样 5 条、批大小 2 → ``batches == ceil(5 / 2) == 3``，且最后一批只有 1 条."""
    provider = CountingEmbedding()
    encoder = BatchEncoder(provider, batch_size=2)
    _, report = encoder.encode([f"t{index}" for index in range(5)])

    assert report.encoded == 5
    assert report.batches == 3
    assert provider.batch_sizes == [2, 2, 1]
    assert provider.batch_calls == 3


def test_default_batch_size_comes_from_settings() -> None:
    """缺省批大小取 ``settings.indexing_batch_size``，文档常量与它对齐."""
    assert DEFAULT_BATCH_SIZE == 32
    assert BatchEncoder(CountingEmbedding()).batch_size == settings.indexing_batch_size


def test_empty_input_does_not_call_provider() -> None:
    """空输入是一个合法请求：返回空列表，且不发起任何调用."""
    provider = CountingEmbedding()
    vectors, report = BatchEncoder(provider).encode([])

    assert vectors == []
    assert report.texts == 0
    assert report.encoded == 0
    assert report.batches == 0
    assert report.hit_ratio == 0.0
    assert provider.batch_calls == 0
    assert provider.embed_calls == 0


@pytest.mark.parametrize("bad_size", [0, -1])
def test_non_positive_batch_size_is_rejected(bad_size: int) -> None:
    """``batch_size <= 0`` 是配置错误，不许被静默改成 1."""
    with pytest.raises(IndexingError) as excinfo:
        BatchEncoder(CountingEmbedding(), batch_size=bad_size)

    assert "batch_size" in str(excinfo.value)
    assert f"收到 {bad_size!r}" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 缓存与顺序：命中不改变结果；批内重复只编码一次
# --------------------------------------------------------------------------- #


def test_duplicate_texts_are_encoded_only_once() -> None:
    """``[A, B, A]`` → 编码 2 条、命中至少 1 次，且结果与入参逐位对应."""
    provider = CountingEmbedding()
    encoder = BatchEncoder(provider, batch_size=32)
    texts = ["alpha", "beta", "alpha"]

    vectors, report = encoder.encode(texts)

    assert report.encoded == 2
    assert report.cache_hits >= 1
    assert report.batches == 1
    assert provider.embed_calls == 2
    # 顺序保持：第 0 条与第 2 条是同一段文本，第 1 条不是。
    assert vectors[0] == vectors[2]
    assert vectors[0] != vectors[1]


def test_cache_hit_returns_bitwise_identical_vector() -> None:
    """缓存只省调用，不改变结果：第二次编码的向量与第一次逐位相同."""
    provider = CountingEmbedding()
    encoder = BatchEncoder(provider)
    texts = ["t1", "t2"]

    first_vectors, first_report = encoder.encode(texts)
    second_vectors, second_report = encoder.encode(texts)

    assert second_vectors == first_vectors
    assert first_report.encoded == 2
    assert second_report.encoded == 0
    assert second_report.batches == 0
    assert second_report.cache_hits == 2
    assert second_report.hit_ratio == 1.0
    # 第二次一共只调用了两次（第一次那两条），没有新增调用。
    assert provider.embed_calls == 2


def test_full_hit_never_calls_the_provider() -> None:
    """全命中时提供方**一次都没被调用**（换一个实例、共用同一份缓存）."""
    cache = EmbeddingCache(
        provider="CountingEmbedding",
        model=MODEL_LABEL_FALLBACK,
        dimension=DIMENSION,
    )
    warm = CountingEmbedding()
    warm_encoder = BatchEncoder(warm, cache=cache)
    texts = ["t1", "t2", "t3"]
    warm_vectors, _ = warm_encoder.encode(texts)
    assert warm.embed_calls == 3

    cold = CountingEmbedding()
    cold_encoder = BatchEncoder(cold, cache=cache)
    vectors, report = cold_encoder.encode(texts)

    assert vectors == warm_vectors
    assert report.encoded == 0
    assert report.batches == 0
    assert report.cache_hits == 3
    assert cold.batch_calls == 0
    assert cold.embed_calls == 0


def test_default_cache_carries_the_encoder_identity() -> None:
    """缺省缓存带上本编码器的身份（键里含身份，因此天生与其他编码器隔离）."""
    encoder = BatchEncoder(CountingEmbedding())

    assert encoder.cache.provider == "CountingEmbedding"
    assert encoder.cache.model == MODEL_LABEL_FALLBACK
    assert encoder.cache.dimension == DIMENSION
    assert encoder.cache.identity_key == encoder.identity.key


# --------------------------------------------------------------------------- #
# 输入与返回值：四项校验都必须在缓存之前发生
# --------------------------------------------------------------------------- #


def test_blank_text_is_rejected_and_provider_is_not_called() -> None:
    """第 2 条是纯空白 → 报错并点名序号；且不浪费一次编码调用."""
    provider = CountingEmbedding()
    encoder = BatchEncoder(provider)

    with pytest.raises(EncodingError) as excinfo:
        encoder.encode(["正常文本", "   "])

    message = str(excinfo.value)
    assert "第 2 条" in message
    assert "空白" in message
    assert provider.embed_calls == 0
    assert provider.batch_calls == 0


def test_empty_string_is_rejected_with_preview() -> None:
    """空串同样被拒，消息里带前 20 字预览."""
    with pytest.raises(EncodingError) as excinfo:
        BatchEncoder(CountingEmbedding()).encode([""])

    message = str(excinfo.value)
    assert "第 1 条" in message
    assert "''" in message


def test_non_string_input_is_rejected() -> None:
    """非字符串会造出一条语义上不存在的记录，必须在编码前拒掉."""
    with pytest.raises(EncodingError) as excinfo:
        BatchEncoder(CountingEmbedding()).encode(["ok", None])  # type: ignore[list-item]

    message = str(excinfo.value)
    assert "第 2 条" in message
    assert "NoneType" in message


def test_returned_count_mismatch_is_rejected() -> None:
    """返回条数与请求条数不符：无法按序对应，必须整体拒绝."""
    encoder = BatchEncoder(BadCountEmbedding())

    with pytest.raises(EncodingError) as excinfo:
        encoder.encode(["t1", "t2"])

    message = str(excinfo.value)
    assert "返回 1 条" in message
    assert "请求了 2 条" in message


def test_dimension_mismatch_is_rejected() -> None:
    """某条向量的维度与 identity 不符 → 报错，且两个维度都写在消息里."""
    encoder = BatchEncoder(BadDimensionEmbedding(declared=DIMENSION))

    with pytest.raises(EncodingError) as excinfo:
        encoder.encode(["t1"])

    message = str(excinfo.value)
    assert f"{DIMENSION + 1} 维" in message
    assert f"{DIMENSION} 维" in message


def test_non_finite_vector_is_rejected() -> None:
    """含 ``nan`` 的向量被拒（它会污染所有与它比较的相似度）."""
    encoder = BatchEncoder(NanEmbedding())

    with pytest.raises(EncodingError, match="非有限"):
        encoder.encode(["t1"])


def test_zero_vector_is_rejected_and_not_cached() -> None:
    """零向量被拒，且**没有进入缓存**（脏数据不会被长期复用）."""
    encoder = BatchEncoder(ZeroEmbedding())

    with pytest.raises(EncodingError, match="零向量"):
        encoder.encode(["阈值设为 0.85"])

    assert encoder.cache.size == 0


def test_encode_one_uses_cache() -> None:
    """单条路径也走缓存：第二次不产生新的编码调用."""
    provider = CountingEmbedding()
    encoder = BatchEncoder(provider)

    first = encoder.encode_one("t1")
    assert first == _vector_for("t1", DIMENSION)
    assert provider.embed_calls == 1

    second = encoder.encode_one("t1")
    assert second == first
    assert provider.embed_calls == 1


def test_encode_one_rejects_blank_text() -> None:
    """单条路径与批量路径共用同一条文本校验."""
    with pytest.raises(EncodingError, match="全空白"):
        BatchEncoder(CountingEmbedding()).encode_one("  ")


# --------------------------------------------------------------------------- #
# 编码器身份：只用公开属性；可被显式覆盖
# --------------------------------------------------------------------------- #


def test_describe_embedding_charges_ngram_provider_to_fallback() -> None:
    """``CharNgramEmbedding`` 没有模型名 → ``model`` 落到 ``"default"``."""
    identity = describe_embedding(CharNgramEmbedding())

    assert identity.provider == "CharNgramEmbedding"
    assert identity.model == MODEL_LABEL_FALLBACK
    assert identity.dimension == 256


def test_describe_embedding_reads_mock_provider() -> None:
    """``MockEmbedding`` 同样没有模型名，维度 64."""
    identity = describe_embedding(MockEmbedding())

    assert (identity.provider, identity.model, identity.dimension) == (
        "MockEmbedding",
        MODEL_LABEL_FALLBACK,
        64,
    )


def test_describe_embedding_prefers_public_model_attribute() -> None:
    """有公开 ``model`` 属性时用它（不碰任何私有字段）."""
    identity = describe_embedding(NamedEmbedding(model="text-embedding-fake"))

    assert identity.provider == "NamedEmbedding"
    assert identity.model == "text-embedding-fake"
    assert identity.dimension == DIMENSION


def test_describe_embedding_falls_back_to_model_name() -> None:
    """没有 ``model`` 时退一级取 ``model_name``."""
    identity = describe_embedding(ModelNameEmbedding(model_name="mini-lm-fake"))

    assert identity.model == "mini-lm-fake"
    assert identity.provider == "ModelNameEmbedding"


def test_identity_key_is_stable_and_matches_manual_identity() -> None:
    """``identity.key`` 可复算、可复现；维度不同则 key 不同."""
    first = describe_embedding(CharNgramEmbedding())
    second = describe_embedding(CharNgramEmbedding())

    assert first == second
    assert first.key == second.key
    assert (
        first.key
        == EmbeddingIdentity(
            provider="CharNgramEmbedding", model=MODEL_LABEL_FALLBACK, dimension=256
        ).key
    )
    assert describe_embedding(CharNgramEmbedding(dimension=128)).key != first.key


def test_identity_override_wins() -> None:
    """``identity=`` 覆盖时以覆盖值为准（代价的出口，见 encoder 模块 docstring）."""
    provider = CountingEmbedding()
    override = EmbeddingIdentity(
        provider="CountingEmbedding", model="pin-v3", dimension=DIMENSION
    )

    encoder = BatchEncoder(provider, identity=override)

    assert encoder.identity is override
    assert encoder.identity.model == "pin-v3"
    _, report = encoder.encode(["t1"])
    assert report.provider == "CountingEmbedding"
    assert report.dimension == DIMENSION
