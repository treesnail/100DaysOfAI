"""批量编码器：把"一批文本"变成"一批向量"，并把这笔账记清楚（M6-D4）.

day064 的 ``VectorIngestPipeline`` 逐条调用 ``embedding.embed``，于是


```text
重放同一批：输入 6 → 写入 0（未变 6）| 编码调用 6 次
```

**库没变，钱照花。** 本模块与 ``cache.EmbeddingCache`` 合起来回答那笔账：
``EncodeReport`` 的四个数——``texts``（要多少）、``cache_hits``（白拿多少）、
``encoded``（真花了多少）、``batches``（分几次花）——就是"省下了什么"的全部证据。

## 三段流水线，每段都有它非在此处不可的理由

```text
1. 判文本      空 / 全空白 / 非字符串 → EncodingError（before 任何调用）
2. 查缓存      命中的直接拿，未命中的去重后进 pending
3. 批量编码    pending 按 batch_size 分批送 embed_batch，逐批校验后写缓存
4. 还原顺序    按**入参顺序**把结果拼回去（缓存命中与新鲜编码混在一起）
```

第 1 段必须在最前面：**空白文本不该换来一次计费请求**。第 2 段与第 3 段的
先后不能换：先查缓存才能算出"真正要编码多少条"，而"要编码 0 条"时
**一次提供方调用都不该发生**（``batches == 0``）——空调用在真实 API 上
是一次真实账单，也可能被某些提供方当成非法请求。第 4 段是这一层的核心承诺：
调用方拿到的是一个与 ``texts`` **逐位对应**的列表，而不是"命中在前、
新算在后"的拼接结果。

## 同一批里的重复文本只编码一次

``[A, B, A]`` 的正确账是 ``encoded == 2``。第二次出现的 ``A`` 既不是
"缓存直接命中"（查它的时候 A 还没写进去），也不该再花一次编码——
本模块把它记作一次 ``cache_hits``：**它省下的确实是一次调用**，
只是省下的方式与"上次已经存过"不同。这条差别在报告里不重要，
在账单里是同一个数字。

## 校验为什么必须在这里，而不是留给向量库

提供方返回的结果有三种"坏法"：条数对不上（无法按序对应）、
维度与身份不符、某条含 ``nan``/``inf`` 或是零向量。day064 的
``VectorRecord`` 已经拦过一部分，但那时**脏向量可能已经被写进缓存**——
而缓存条目活得很久，下一次构建会直接命中它。所以校验必须发生在
``cache.put`` 之前：**缓存一旦装进脏数据，错误就从"一次"变成"每次"。**

零向量尤其要点名：它没有方向，余弦相似度对它是 ``0/0``。某些提供方
对空串或纯符号返回零向量，且**不报错**——它只会让"这条记录永远检索不到"，
而报告里一切正常。

## 编码器身份的代价写在明面上

``describe_embedding`` 只用提供方的**公开属性**（``model`` → ``model_name``
→ 回落 ``"default"``）。代价是：拿不到模型名的提供方（``MockEmbedding``、
``CharNgramEmbedding`` 都是）会让"换了配置但类名与维度没变"这件事
**无法被发现**——身份相同 → 向量键相同 → 旧向量被复用，而检索结果
只是"变得不太对"。因此身份必须可被覆盖：``BatchEncoder(identity=...)``
是这条代价的出口，调用方在知道真实配置（例如 n-gram 阶数）时可以
显式钉死身份，让"换配置"重新变成一次必然的整库重算。

## 放弃了什么

| 放弃的东西 | 代价 | 谁来还 |
|-----------|------|--------|
| 逐条失败定位 | 一条坏向量让整批抛 ``EncodingError`` | 提供方修好后整批重来 |
| 重试与退避 | 一次失败即上抛，不做内部重试 | 上层构建流程（重试属于编排） |
| 缓存维度联动 | 传入缓存的键里含身份，天然按身份隔离 | ``cache.EmbeddingCache`` 的键 |

## 谁依赖它

```text
day065 的 builder        用 BatchEncoder 把"要编码的 id 集合"变成向量
day065 的 planner        不依赖本模块，但两者共享同一份 EmbeddingIdentity
tests/                   全部离线：假提供方把向量写死，调用次数可断言
```
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from smart_research_agent.config import settings
from smart_research_agent.indexing.cache import EmbeddingCache
from smart_research_agent.indexing.errors import EncodingError, IndexingError
from smart_research_agent.indexing.types import EmbeddingIdentity, EncodeReport
from smart_research_agent.llm.embedding import EmbeddingProvider

#: 一次送进提供方的文本条数（文档常量）。真实取值走 ``settings.indexing_batch_size``，
#: 这里保留它，是为了让"默认是 32"这件事在代码里有一个可被引用的名字
#: （配置项的注释与本模块的说明引用同一个数，不会各自漂移）。
DEFAULT_BATCH_SIZE = 32

#: 拿不到模型名时的标签。**它不是一个好标签，只是唯一诚实的标签**：
#: 与其猜一个（"char-ngram-2-3"）让身份看起来比实际更精确，不如让它显式地
#: 暴露"这个提供方没有可报的模型名"，从而提示调用方用 ``identity=`` 覆盖。
MODEL_LABEL_FALLBACK = "default"

#: 构造身份时依次尝试的公开属性名（见模块 docstring 的"身份"一节）。
_MODEL_ATTRIBUTES: tuple[str, ...] = ("model", "model_name")


def describe_embedding(embedding: EmbeddingProvider) -> EmbeddingIdentity:
    """从提供方读出它的身份（``provider`` / ``model`` / ``dimension``）.

    三个字段的取法都是**能从外面看见的**事实，本函数刻意不碰任何私有字段：

    ```text
    provider   type(embedding).__name__       实现类名，永远拿得到
    model      model → model_name → "default" 公开属性，拿不到就回落
    dimension  embedding.dimension            提供方自己声明的输出维度
    ```

    "只用公开属性"是一条纪律而不是洁癖：``_ngram_sizes``、``_model_name``
    这类私有字段是**实现细节**，今天读它、明天重命名就会静默失效
    （拿到 ``AttributeError`` 还算好的，更坏的是拿到另一个含义相同的值）。
    代价是``CharNgramEmbedding`` 这类"配置藏在私有字段里"的提供方
    只能落到 ``"default"``——它的 n-gram 阶数改了，身份却不变，
    于是**"换配置"这件事无法被发现**（``model`` 相同 → 向量键相同 →
    旧向量被复用），而检索结果只是"变得不太对"。

    因此本函数是**建议值而不是判决**：调用方知道真实配置时应当构造
    ``EmbeddingIdentity`` 自己传进 ``BatchEncoder(identity=...)``，
    把"换配置"重新变回一次必然的整库重算（这也是 ``types.py``
    把身份做成独立对象的全部理由）。
    """
    return EmbeddingIdentity(
        provider=type(embedding).__name__,
        model=_model_label(embedding),
        dimension=int(embedding.dimension),
    )


def _model_label(embedding: EmbeddingProvider) -> str:
    """按 ``model`` → ``model_name`` 的顺序取模型标识，取不到回落常量.

    只接受**非空字符串**：一个返回 ``None`` 或某个对象的同名属性不算数
    （``str()`` 它会让身份变成一个包含内存地址或类名的噪声串，
    比 ``"default"`` 更糟——噪声每天都在变，而身份本该稳定）。
    """
    for name in _MODEL_ATTRIBUTES:
        value = getattr(embedding, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return MODEL_LABEL_FALLBACK


class BatchEncoder:
    """批量编码器：缓存优先、批内去重、按序还原，并逐批校验（见模块 docstring）.

    它**只负责把文本变成向量**，不碰清单、不碰向量库、不做增量判定——
    那些是 ``planner`` / ``manifest`` / ``builder`` 的事。这条边界让本类
    可以被单独测试：给一个假提供方与一个空缓存，全部行为都能被断言。
    """

    def __init__(
        self,
        embedding: EmbeddingProvider,
        *,
        batch_size: int | None = None,
        cache: EmbeddingCache | None = None,
        identity: EmbeddingIdentity | None = None,
    ) -> None:
        """构造编码器；三段依赖都是可注入的.

        ```text
        embedding  真正干活的提供方（离线假实现同样合法）
        batch_size 缺省取 settings.indexing_batch_size；<= 0 → IndexingError
        cache      缺省新建一份**带本编码器身份**的空缓存（键里含身份，因此天生隔离）
        identity   缺省 describe_embedding(embedding)；显式传入即覆盖（见模块 docstring）
        ```
        """
        self._embedding = embedding
        self._identity = identity or describe_embedding(embedding)
        self._batch_size = _resolve_batch_size(batch_size)
        self._cache = cache if cache is not None else EmbeddingCache(
            provider=self._identity.provider,
            model=self._identity.model,
            dimension=self._identity.dimension,
        )

    # ------------------------------------------------------------------ 只读属性

    @property
    def identity(self) -> EmbeddingIdentity:
        """本编码器的身份（进向量键与报告，见模块 docstring）."""
        return self._identity

    @property
    def cache(self) -> EmbeddingCache:
        """本编码器使用的向量缓存（命中/未命中计数就在它上面）."""
        return self._cache

    @property
    def batch_size(self) -> int:
        """一次送进提供方的文本条数（恒 >= 1）."""
        return self._batch_size

    # ------------------------------------------------------------------ 编码

    def encode(self, texts: Sequence[str]) -> tuple[list[list[float]], EncodeReport]:
        """编码一批文本，返回**与 ``texts`` 逐位对应**的向量与一份账.

        四个数各自回答一个问题，缺一个都会让报告变成一句感觉：

        ```text
        texts       这次要了多少条
        cache_hits  其中多少条没有花编码调用（含批内重复）
        encoded     真正送给提供方的条数
        batches     向提供方发起了几批（encoded=6/batches=1 vs 6/6 是两种成本）
        ```

        ``cache_hits`` 读的是**本次调用内的差值**（调用前后的 ``cache.hits``
        之差），再补上批内重复那部分：调用方可能没有 ``reset_counters``
        （它有权保留跨次统计），而报告里的数字必须只描述这一次。

        任一条校验不过即抛 ``EncodingError``，**整批不写缓存**：
        半批写进去会让"重放能得到什么"取决于上次写到哪一条。
        """
        text_list = self._validate_texts(texts)

        hits_before = self._cache.hits
        cached: list[list[float] | None] = [None] * len(text_list)
        pending: list[str] = []
        pending_seen: set[str] = set()
        for index, text in enumerate(text_list):
            value = self._cache.get(text)
            if value is not None:
                cached[index] = value
                continue
            # 批内去重：同一个文本只进一次 pending，其余的等到这里把它写进缓存后
            # 自然成为"命中"。放在循环里做而不是事后 set()，是为了保住首次出现的顺序。
            if text not in pending_seen:
                pending_seen.add(text)
                pending.append(text)

        direct_hits = self._cache.hits - hits_before
        encoded_texts: dict[str, list[float]] = {}
        batches = 0
        for start in range(0, len(pending), self._batch_size):
            chunk = pending[start : start + self._batch_size]
            returned = self._embedding.embed_batch(list(chunk))
            batches += 1
            if len(returned) != len(chunk):
                raise EncodingError(
                    f"提供方返回 {len(returned)} 条向量，而这一批请求了 {len(chunk)} 条"
                    f"（第一条文本前 20 字：{chunk[0][:20]!r}）："
                    "数量不符就无法按序对应，把结果硬拼回去会让向量与文本错位——"
                    "那种错位不会报错，只会让每条记录都检索到自己不该有的邻居。"
                )
            for text, vector in zip(chunk, returned):
                values = [float(value) for value in vector]
                self._check_vector(values, where=f"文本 {text[:20]!r}")
                encoded_texts[text] = values

        # 先整批校验、再整批写缓存：见 docstring 的"整批不写缓存"。
        self._cache.put_vectors((text, encoded_texts[text]) for text in pending)

        vectors: list[list[float]] = []
        for index, text in enumerate(text_list):
            value = cached[index]
            vectors.append(value if value is not None else encoded_texts[text])

        encoded = len(pending)
        # 批内重复：既没命中缓存（查它时还没写进去），也没花一次编码——
        # 它省下的调用与"上次就存过"是同一个数字，因此一并计入 cache_hits。
        intra_batch_hits = (len(text_list) - direct_hits) - encoded
        return vectors, EncodeReport(
            texts=len(text_list),
            cache_hits=direct_hits + intra_batch_hits,
            encoded=encoded,
            batches=batches,
            dimension=self._identity.dimension,
            provider=self._identity.provider,
        )

    def encode_one(self, text: str) -> list[float]:
        """编码单条文本（同样是缓存优先 + 编码前校验）.

        存在的理由不是"方便"：单条路径若绕过缓存，调用方就会在
        "批量走缓存、单条不走"之间得到两套成本，而它们看起来都正常。
        """
        items = self._validate_texts([text])
        cached = self._cache.get(items[0])
        if cached is not None:
            return cached
        vector = [float(value) for value in self._embedding.embed(items[0])]
        self._check_vector(vector, where=f"文本 {items[0][:20]!r}")
        self._cache.put(items[0], vector)
        return vector

    # ------------------------------------------------------------------ 内部校验

    def _validate_texts(self, texts: Sequence[str]) -> list[str]:
        """把入参收成列表，并逐条判掉"不值得编码"的文本.

        三种输入都要在**任何**提供方调用之前被判掉：

        ```text
        非字符串   拼成 "None" 去编码会得到一条语义上不存在的记录
        空串       多数提供方返回零向量（或直接报错），两种都不该发生在这里
        全空白     与空串等价：它编码出来不会带来任何话题信息
        ```

        报错信息带序号与前 20 字预览：一批 32 条里有一条空白时，
        "有一处空白"这句话对修复毫无帮助。
        """
        items = list(texts)
        for index, text in enumerate(items):
            if not isinstance(text, str):
                raise EncodingError(
                    f"第 {index + 1} 条输入不是字符串（收到 {type(text).__name__}）："
                    "批量编码只接受文本；把别的类型 str() 之后送进来，"
                    "会凭空造出一条语义上并不存在的记录。"
                )
            if not text.strip():
                raise EncodingError(
                    f"第 {index + 1} 条文本为空或全空白（前 20 字：{text[:20]!r}）："
                    "编码空文本没有意义——多数提供方会给零向量，"
                    "而零向量没有方向，余弦相似度对它是 0/0。请在上游修数据。"
                )
        return items

    def _check_vector(self, vector: list[float], *, where: str) -> None:
        """三条硬校验：维度、有限性、非零（任何一条不过即拒收，见模块 docstring）.

        三条都发生在 ``cache.put`` **之前**：缓存的条目会被长期复用，
        脏数据一旦进去，代价就从"这一次"变成"每一次"。
        """
        identity = self._identity
        if len(vector) != identity.dimension:
            raise EncodingError(
                f"编码结果维度不符：{where} 得到 {len(vector)} 维，"
                f"而编码器身份（{identity.summary_line()}）声明的是 {identity.dimension} 维。"
                "维度不同说明提供方与身份声明的不是同一套（换过模型或输出维度设置）——"
                "请重建索引，不要截断或补零。"
            )
        if not all(math.isfinite(value) for value in vector):
            raise EncodingError(
                f"编码结果里有非有限数（nan / inf）：{where}。"
                "这样的向量写进库之后，所有与它比较的相似度都会变成 nan，"
                "而排序不会报错——它只会让这一条永远排不出确定的名次。"
            )
        if not any(value != 0.0 for value in vector):
            raise EncodingError(
                f"编码结果是零向量：{where}。零向量没有方向，余弦相似度对它是 0/0；"
                "day064 的 VectorRecord 会在写库时拦它，但那时它可能已经被写进缓存——"
                "所以必须在缓存之前拒收。"
            )


def _resolve_batch_size(batch_size: int | None) -> int:
    """解析批大小：显式参数 > ``settings.indexing_batch_size``，且必须为正.

    ``<= 0`` 报 ``IndexingError`` 而不是"当成 1"：批大小是成本参数，
    一个 0 或负数一定是配置写错了（例如环境变量留空被解析成 0）。
    悄悄改成 1 会让"逐条调用"以"配了个批大小"的样子继续跑下去，
    而报告里的 ``batches`` 看起来只是"批没生效"。
    """
    resolved = settings.indexing_batch_size if batch_size is None else batch_size
    value = int(resolved)
    if value <= 0:
        raise IndexingError(
            f"batch_size 必须是 >= 1 的整数，收到 {batch_size!r}（解析后为 {value}）。"
            "批大小是成本参数：0 或负数一定是配置写错了，"
            "本模块不会替你猜成 1——那会让逐条调用伪装成批量调用。"
        )
    return value


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "MODEL_LABEL_FALLBACK",
    "BatchEncoder",
    "describe_embedding",
]
