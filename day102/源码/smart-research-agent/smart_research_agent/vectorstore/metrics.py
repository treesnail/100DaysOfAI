"""相似度与距离：本包唯一一处定义"谁更近"的地方（M6-D3）.

这个模块存在的理由是一个非常具体的坑：**同一个向量库概念，
在三个库里用三种符号约定表达。**

```text
                FAISS（IndexFlatIP / IndexFlatL2）    Chroma（hnsw:space）
cosine          需要自己先归一化，再取内积（越大越近）   d = 1 − cos（越小越近）
ip              内积（越大越近）                        d = 1 − Σ(a·b)（越小越近）
l2              平方 L2（越小越近）                     d = Σ(ai − bi)²（越小越近）
```

三行合起来是一个很容易致命的结论：**FAISS 的"分数"越大越好，
Chroma 的"距离"越小越好，而两者都不叫 similarity。**
把 Chroma 的 ``distances[0][0] = 0.12`` 直接当相似度丢给阈值判断
（``if score > 0.8``）会让**最相似的记录被过滤掉**，而且不报错。

因此在 ``vectorstore`` 包里只保留两套口径，并且用名字把它钉死：

```text
score     统一口径：**越大越近**（本包内部排序、阈值、报告只用它）
distance  原始口径：**越小越近**（与 Chroma 的 distances 同名同义）
```

``SearchHit`` 两个都带（见 ``types.SearchHit``），因为报告需要前者、
而与外部库对账需要后者。两者之间的换算是这个模块的全部内容。

## 三个度量的取舍

| 度量 | 值域 | 需要归一化 | 什么时候用 |
|------|------|-----------|-----------|
| ``cosine`` | [-1, 1] | 是（本包自动做） | **默认**。文本 embedding 的标准用法 |
| ``ip`` | 无界 | 否 | 已归一化时与 cosine 等价；未归一化时"长向量占优" |
| ``l2`` | [0, ∞) | 否 | 图像/几何特征；对文本要先确认提供方的训练目标 |

默认取 ``cosine`` 而不是 ``ip``，理由是**阈值可解释**：
``cosine`` 的相似度有绝对上界 1（"方向完全一致"），
因此 ``min_score=0.8`` 是一句跨数据集都能说的话；
``ip`` 的值随向量模长浮动，同一个阈值换一批文档就换了含义
（与 day062 拒绝照抄 0.7 这个语义分块阈值是同一条纪律）。

``cosine`` 与 ``ip`` 的差别**只在有没有做 L2 归一化**：本包在写库与查询
之前统一调用 ``llm.embedding.l2_normalize``，因此两个度量在数值上一致，
但保留 ``ip`` 这个选项仍有意义——它正是 FAISS 里那个"能跑得更快"的入口
（不归一化就少一遍乘法），也是**验证"你的提供方到底归没归一化"的探针**。
"""

from __future__ import annotations

import math

from smart_research_agent.vectorstore.errors import FilterError, VectorError

#: 余弦相似度：与向量长度无关，值域 [-1, 1]。默认度量。
METRIC_COSINE = "cosine"
#: 内积（dot product）：值域无界；向量已归一化时与 ``cosine`` 数值相同。
METRIC_INNER_PRODUCT = "ip"
#: 平方欧氏距离（squared L2）：值域 [0, ∞)。**是平方，不是开方后的距离**。
METRIC_L2 = "l2"

#: 三种度量。**顺序就是报告与端点的顺序**，也是选型的阅读顺序。
METRICS: tuple[str, ...] = (METRIC_COSINE, METRIC_INNER_PRODUCT, METRIC_L2)

#: 别名表：把各处写惯的写法收敛到三个规范名。
#: 为什么要它：Chroma 的 ``configuration={"hnsw": {"space": ...}}`` 用 ``l2``/``ip``/``cosine``，
#: 而 scikit-learn / 各类教程常用 ``euclidean`` / ``dot``；同一个参数在不同来源
#: 有不同拼法，**静默地当成未知值处理会让"我明明设了 euclidean"变成一个查不出的问题**。
METRIC_ALIASES: dict[str, str] = {
    "cos": METRIC_COSINE,
    "cosine": METRIC_COSINE,
    "angular": METRIC_COSINE,
    "ip": METRIC_INNER_PRODUCT,
    "dot": METRIC_INNER_PRODUCT,
    "inner_product": METRIC_INNER_PRODUCT,
    "inner-product": METRIC_INNER_PRODUCT,
    "l2": METRIC_L2,
    "euclidean": METRIC_L2,
    "sqeuclidean": METRIC_L2,
    "squared_l2": METRIC_L2,
}

#: 每个度量的一句话说明（端点 ``/vectorstore/backends`` 直接回显）。
METRIC_NOTES: dict[str, str] = {
    METRIC_COSINE: "余弦相似度，值域 [-1, 1]，默认；库与查询向量都会做 L2 归一化",
    METRIC_INNER_PRODUCT: "内积，值域无界；归一化后与 cosine 数值一致，FAISS 上少一次乘法",
    METRIC_L2: "平方欧氏距离（不是开方后的距离）；小批量图像特征常用",
}


def normalize_metric(name: str) -> str:
    """把别名收敛成规范名；未知名字直接报错并列出可选值.

    与 day062 ``resolve_measurer`` 同一条纪律：**宁可报错，不要给一个
    看起来对的错答案**。若把 ``"euclidean2"`` 静默当成默认的 cosine，
    检索结果会以"排序不太对"的形式表现出来，而没有人会怀疑参数名拼错了。
    """
    key = str(name).strip().lower()
    if key in METRIC_ALIASES:
        return METRIC_ALIASES[key]
    raise FilterError(
        f"未知度量 {name!r}，可选 {', '.join(METRICS)}"
        f"（别名还有 {', '.join(sorted(set(METRIC_ALIASES) - set(METRICS)))}）"
    )


def dot(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    """点积（维度不一致时报 ``VectorError``，而不是抛 ``ValueError``）."""
    _require_same_dimension(a, b)
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a: list[float] | tuple[float, ...]) -> float:
    """L2 模长."""
    return math.sqrt(math.fsum(x * x for x in a))


def cosine_similarity(
    a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]
) -> float:
    """余弦相似度；任一侧是零向量时返回 0.0.

    零向量没有方向，"相似度"对它没有定义。这里返回 0.0 而不是抛错，
    是为了让**一条脏数据不要打断整批查询**；但它不应该是常态——
    写入侧由 ``types.VectorRecord`` 直接拒收零向量（见那里的说明）。
    """
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot(a, b) / (na * nb)


def squared_l2(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    """平方欧氏距离（与 Chroma ``l2`` 的公式一致：``Σ(ai − bi)²``）.

    刻意不取平方根：与 FAISS ``IndexFlatL2``、Chroma ``space="l2"``
    的返回口径保持一致，避免"对账时差一个开方"这类错误。
    """
    _require_same_dimension(a, b)
    return math.fsum((x - y) ** 2 for x, y in zip(a, b))


def score(
    metric: str,
    query: list[float] | tuple[float, ...],
    vector: list[float] | tuple[float, ...],
) -> float:
    """统一口径的分数：**越大越近**.

    ``l2`` 取负号是为了让整包只有一种排序方向。这个负号不是装饰：
    四个后端（flat / faiss / chroma / 将来的任何一个）的排序都调用它，
    于是"Chroma 的 l2 是越小越好"这件事只在 ``chroma_backend`` 的
    一次距离换算里出现，而不是散落在所有比较里。
    """
    resolved = normalize_metric(metric)
    if resolved == METRIC_COSINE:
        return cosine_similarity(query, vector)
    if resolved == METRIC_INNER_PRODUCT:
        return dot(query, vector)
    return -squared_l2(query, vector)


def distance(
    metric: str,
    query: list[float] | tuple[float, ...],
    vector: list[float] | tuple[float, ...],
) -> float:
    """原始口径的距离：**越小越近**（与 Chroma 的 ``1 − 相似度`` / 平方 L2 一致）."""
    resolved = normalize_metric(metric)
    if resolved == METRIC_COSINE:
        return 1.0 - cosine_similarity(query, vector)
    if resolved == METRIC_INNER_PRODUCT:
        return 1.0 - dot(query, vector)
    return squared_l2(query, vector)


def similarity_from_distance(metric: str, value: float) -> float:
    """把外部库返回的距离折回本包的 ``score``（越大越近）.

    两条换算规则对应上面那张表：

    ```text
    cosine / ip  →  1 − distance      （Chroma 就是这么定义它的距离的）
    l2           →  −distance         （平方 L2 越小越近，取负即"越大越近"）
    ```

    为什么必须集中在一处：Chroma 的 ``query`` 返回 ``distances``，
    FAISS 的 ``search`` 返回 ``D``，**两者的符号约定相反**。
    如果换算写在两个适配器里各写一遍，第二次写的时候抄错一个符号，
    结果就是"换后端之后检索质量突然变差"——而这种差异没有任何异常。
    """
    resolved = normalize_metric(metric)
    if resolved == METRIC_L2:
        return -float(value)
    return 1.0 - float(value)


def describe_metrics() -> list[dict[str, str]]:
    """三个度量的说明（端点与手册共用一份，避免文档与代码说法不一致）."""
    return [
        {
            "metric": name,
            "better": "越大越近",
            "needs_normalization": "true" if name == METRIC_COSINE else "false",
            "note": METRIC_NOTES[name],
        }
        for name in METRICS
    ]


def _require_same_dimension(
    a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]
) -> None:
    if len(a) != len(b):
        raise VectorError(
            f"向量维度不一致：查询 {len(a)} 维、库内 {len(b)} 维。"
            "维度不同说明**库与当前编码器不是同一套**——重新编码并重建索引即可，"
            "不要试图截断或补零（那会让排序结果静默地失去意义）。"
        )


__all__ = [
    "METRICS",
    "METRIC_ALIASES",
    "METRIC_COSINE",
    "METRIC_INNER_PRODUCT",
    "METRIC_L2",
    "METRIC_NOTES",
    "cosine_similarity",
    "describe_metrics",
    "distance",
    "dot",
    "norm",
    "normalize_metric",
    "score",
    "similarity_from_distance",
    "squared_l2",
]
