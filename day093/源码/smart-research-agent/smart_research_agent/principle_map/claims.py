"""``claims``：十二块拼图的定义与查询（day088 / M7-D12）.

十二块拼图按四层排列（条数 2 / 4 / 3 / 3）。每一块都带三个坐标：

```text
artifact     项目里的哪个函数（**字符串名字**，可被解析、可被检查）
probe        哪个探针去量它（reconcile 里的函数名）
application  它支撑哪个能力（六个应用之一）
```

## 一条纪律：**这里只有"命题"，没有"读数"**

读数全部在 :mod:`reconcile` 里由探针现场算出来。把读数写在本模块的后果是——
某一天底层实现改了，而这张表里的数字**悄悄地不再是真的**，
而它与"对上了"读起来一样。因此本模块只描述"命题是什么、落在哪里"，
"它现在读数是多少"必须每次重新量。
"""

from __future__ import annotations

from smart_research_agent.principle_map.errors import ReferenceError
from smart_research_agent.principle_map.types import (
    APP_AGENT_REASONING,
    APP_EXPLAINABILITY,
    APP_GENERATION,
    APP_RAG_RERANK,
    APP_RAG_RETRIEVAL,
    APP_SERVING,
    LAYER_ATTENTION,
    LAYER_INFERENCE,
    LAYER_MATH,
    LAYER_REPRESENTATION,
    PRINCIPLE_ATTENTION_IS_DIFFERENTIABLE_RETRIEVAL,
    PRINCIPLE_ATTENTION_ROWS_ARE_DISTRIBUTIONS,
    PRINCIPLE_CACHE_BYTES_IS_A_FORMULA,
    PRINCIPLE_CACHE_REUSE_IS_BITWISE_EXACT,
    PRINCIPLE_CAUSAL_MASK_BLOCKS_FUTURE,
    PRINCIPLE_COSINE_IS_NORMALIZED_DOT,
    PRINCIPLE_EMBEDDING_SIMILARITY_IS_DIRECTION,
    PRINCIPLE_GENERATION_RESPECTS_BUDGET_BREAKDOWN,
    PRINCIPLE_MULTI_HEAD_SPLITS_INSIDE_PROJECTION,
    PRINCIPLE_POSITION_ENCODING_BREAKS_PERMUTATION,
    PRINCIPLE_QUANTIZATION_ERROR_BOUNDED_BY_HALF_STEP,
    PRINCIPLE_RANK_ORDERING_MATCHES_ATTENTION_PEAKS,
    PRINCIPLES,
    Application,
    Principle,
    require_application,
    require_layer,
)

#: 十二块拼图的定义（**顺序 = 提纲里讲的顺序 = 图的拓扑序**）.
PRINCIPLE_DEFS: tuple[Principle, ...] = (
    Principle(
        id=PRINCIPLE_ATTENTION_IS_DIFFERENTIABLE_RETRIEVAL,
        layer=LAYER_MATH,
        statement="注意力 = 可微的检索：softmax(QKᵀ/√d_k)·V 是一组概率对 V 的加权平均，"
        "因此'想要什么'是一条能让梯度回传的分布，而不是一次离散的取舍。",
        artifact="math_foundations.attention.scaled_dot_product_attention",
        probe="probe_attention_is_differentiable_retrieval",
        application=APP_AGENT_REASONING,
        source_day="day073",
    ),
    Principle(
        id=PRINCIPLE_COSINE_IS_NORMALIZED_DOT,
        layer=LAYER_MATH,
        statement="余弦相似度 = 归一化之后的点积：cos(a,b) = a·b / (‖a‖‖b‖)，"
        "它只看方向、不看长度，值域 [-1, 1]。",
        artifact="math_foundations.linalg.cosine",
        probe="probe_cosine_is_normalized_dot",
        application=APP_RAG_RETRIEVAL,
        source_day="day073",
    ),
    Principle(
        id=PRINCIPLE_ATTENTION_ROWS_ARE_DISTRIBUTIONS,
        layer=LAYER_ATTENTION,
        statement="注意力权重矩阵的每一行是一个和为 1 的分布："
        "softmax 按行归一化，被掩码的位置权重恰好是 0.0。",
        artifact="transformer_core.types.assert_rows_are_distributions",
        probe="probe_attention_rows_are_distributions",
        application=APP_EXPLAINABILITY,
        source_day="day075",
    ),
    Principle(
        id=PRINCIPLE_CAUSAL_MASK_BLOCKS_FUTURE,
        layer=LAYER_ATTENTION,
        statement="因果掩码挡住未来：mask[i][j] = (j <= i)，被挡住的位置权重恰好是 0.0，"
        "因此第 i 行看不到任何 j > i 的位置。",
        artifact="transformer_core.layers.resolve_mask",
        probe="probe_causal_mask_blocks_future",
        application=APP_GENERATION,
        source_day="day075/079",
    ),
    Principle(
        id=PRINCIPLE_MULTI_HEAD_SPLITS_INSIDE_PROJECTION,
        layer=LAYER_ATTENTION,
        statement='"分头"发生在投影内部：它是一次 view + transpose，'
        "因此 merge_heads(split_heads(x)) == x 是一条逐位性质。",
        artifact="hf_source.attention.split_heads",
        probe="probe_multi_head_splits_inside_projection",
        application=APP_SERVING,
        source_day="day076/085",
    ),
    Principle(
        id=PRINCIPLE_POSITION_ENCODING_BREAKS_PERMUTATION,
        layer=LAYER_ATTENTION,
        statement="位置编码打破置换等变性：无编码时 attention(perm(x)) == perm(attention(x))，"
        "注入位置表之后这条等式不再成立——而『不再成立』正是模型知道顺序的证据。",
        artifact="positional_encoding.layers.inject",
        probe="probe_position_encoding_breaks_permutation",
        application=APP_GENERATION,
        source_day="day078",
    ),
    Principle(
        id=PRINCIPLE_EMBEDDING_SIMILARITY_IS_DIRECTION,
        layer=LAYER_REPRESENTATION,
        statement="嵌入相似度看的是方向：把任一侧放大 k > 0 倍，余弦相似度不变；"
        "因此向量入库前要归一化。",
        artifact="vectorstore.metrics.cosine_similarity",
        probe="probe_embedding_similarity_is_direction",
        application=APP_RAG_RETRIEVAL,
        source_day="day041/064",
    ),
    Principle(
        id=PRINCIPLE_RANK_ORDERING_MATCHES_ATTENTION_PEAKS,
        layer=LAYER_REPRESENTATION,
        statement="注意力的排序与检索的排序可以逐行对照："
        "当投影是恒等映射、行已归一化时，两者的排序完全一致（秩相关 = 1.0）。",
        artifact="transformer_core.verify.compare_with_retrieval",
        probe="probe_rank_ordering_matches_attention_peaks",
        application=APP_RAG_RERANK,
        source_day="day075",
    ),
    Principle(
        id=PRINCIPLE_CACHE_REUSE_IS_BITWISE_EXACT,
        layer=LAYER_REPRESENTATION,
        statement="缓存复用逐位相同：用缓存的 decode 与整段重算的最后一行 logits 逐位相等，"
        "因为两条路径的算术逐字相同，差别只有 K/V 从哪来。",
        artifact="inference_optim.cache.compare_with_recompute",
        probe="probe_cache_reuse_is_bitwise_exact",
        application=APP_SERVING,
        source_day="day087",
    ),
    Principle(
        id=PRINCIPLE_CACHE_BYTES_IS_A_FORMULA,
        layer=LAYER_INFERENCE,
        statement="缓存字节数是一条公式：2 · L · T · h · bytes，"
        "在任意长度上'公式'与'逐层相加'都必须整数相等。",
        artifact="inference_optim.types.cache_bytes",
        probe="probe_cache_bytes_is_a_formula",
        application=APP_SERVING,
        source_day="day087",
    ),
    Principle(
        id=PRINCIPLE_QUANTIZATION_ERROR_BOUNDED_BY_HALF_STEP,
        layer=LAYER_INFERENCE,
        statement="量化误差不超过半格：对称量化的实测误差 <= scale/2，"
        "它由 round 的误差 <= 0.5 推出，前提是每个元素都落在量程内。",
        artifact="inference_optim.quantize.error_bound",
        probe="probe_quantization_error_bounded_by_half_step",
        application=APP_SERVING,
        source_day="day087",
    ),
    Principle(
        id=PRINCIPLE_GENERATION_RESPECTS_BUDGET_BREAKDOWN,
        layer=LAYER_INFERENCE,
        statement="生成的预算是一条和：权重 + 缓存 + 激活 = 总量，"
        "而预算内最长能生成多少来自一次整除，两侧都要查。",
        artifact="inference_optim.budget.plan",
        probe="probe_generation_respects_budget_breakdown",
        application=APP_GENERATION,
        source_day="day087",
    ),
)

if len(PRINCIPLE_DEFS) != len(PRINCIPLES):  # pragma: no cover - 只在有人改表时触发
    raise ReferenceError(
        "原理定义表的条数与 types.PRINCIPLES 不一致："
        f"{len(PRINCIPLE_DEFS)} 与 {len(PRINCIPLES)}——"
        "少一条的那块拼图在图上会安静地消失。"
    )

_BY_ID: dict[str, Principle] = {item.id: item for item in PRINCIPLE_DEFS}

if set(_BY_ID) != set(PRINCIPLES):  # pragma: no cover - 只在有人改表时触发
    raise ReferenceError(
        "原理定义表的 id 集合与 types.PRINCIPLES 不一致："
        f"多 {sorted(set(_BY_ID) - set(PRINCIPLES))}、缺 {sorted(set(PRINCIPLES) - set(_BY_ID))}。"
    )


def principles() -> tuple[Principle, ...]:
    """十二块拼图（**顺序即提纲顺序**）."""
    return PRINCIPLE_DEFS


def principle(principle_id: str) -> Principle:
    """按 id 取一块拼图（未知 id 抛 :class:`errors.ReferenceError`）."""
    try:
        return _BY_ID[principle_id]
    except KeyError:
        raise ReferenceError(
            f"未知的原理 id {principle_id!r}：可选 {list(PRINCIPLES)}。"
        ) from None


def by_layer(layer: str) -> tuple[Principle, ...]:
    """某一层的全部拼图（未知层抛 ``ParameterError``）."""
    require_layer(layer)
    return tuple(item for item in PRINCIPLE_DEFS if item.layer == layer)


def by_application(application: str) -> tuple[Principle, ...]:
    """支撑某个应用的全部拼图（未知应用抛 ``ParameterError``）."""
    require_application(application)
    return tuple(item for item in PRINCIPLE_DEFS if item.application == application)


def layer_of(principle_id: str) -> str:
    """某条原理属于哪一层（未知 id 抛 ``ReferenceError``）."""
    return principle(principle_id).layer


def application_of(principle_id: str) -> str:
    """某条原理支撑哪个应用（未知 id 抛 ``ReferenceError``）."""
    return principle(principle_id).application


def applications() -> tuple[Application, ...]:
    """六个应用（``principles`` 先留空——填边是 :mod:`graph` 的事）."""
    from smart_research_agent.principle_map.types import APPLICATION_DESCRIPTIONS

    return tuple(
        Application(id=app, description=APPLICATION_DESCRIPTIONS[app], principles=())
        for app in APPLICATION_DESCRIPTIONS
    )


__all__ = [
    "PRINCIPLE_DEFS",
    "application_of",
    "applications",
    "by_application",
    "by_layer",
    "layer_of",
    "principle",
    "principles",
]
