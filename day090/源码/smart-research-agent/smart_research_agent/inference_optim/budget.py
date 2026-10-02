"""``budget``：三个账加起来的那个数（day087 / M7-D11）.

一次推理要在显存里放三样东西，而它们的**来源完全不同**：

```text
权重       参数量 × 每元素字节          —— 与序列长度**无关**
缓存       2·L·T·h·bytes               —— 与序列长度**线性相关**
激活       一次前向最宽的那个中间张量    —— 与批大小**线性相关**
```

第三样最容易写错，因此本课把它的口径写死：**按"某一层最宽的那个中间张量"估**，
也就是前馈的中间维 `ffn`（它比 hidden 宽 `ffn/h` 倍）。这是一条**保守**的估计——
真实实现的峰值取决于算子融合与内存分配器，而本课不承诺复现它们。

## 为什么"三项必须能被逐项相加"

因为放不下的时候要能回答**"是哪一项放不下"**：

```text
权重放不下       ⇒ 换更小的精度 / 换更小的模型
缓存放不下       ⇒ 缩短上下文 / 换能分页的路线
激活放不下       ⇒ 减小批大小
```

三种出路完全不同。把总量当成一个数（而不是三项之和）的后果是：
"显存不够"这句话没有任何下一步动作，而**偷偷把上下文截短**会看起来像一个正常的选择。

## 一个能被算出来的答案：最多能放多长

缓存与激活都**线性**于序列长度，因此"预算内最长能做多少 token"
是一个一次方程的整数解（而不是"试一试"）：

```text
weights + T·(batch·ffn·bytes + 2·L·h·bytes) ≤ budget
⇒ T_max = ⌊(budget − weights) / (batch·ffn·bytes + 2·L·h·bytes)⌋
```

本课把它算出来，并**验证**：`T_max` 放得下，而 `T_max + 1` 放不下。
"""

from __future__ import annotations

from smart_research_agent.hf_integration.types import ModelCard
from smart_research_agent.inference_optim.errors import ParameterError, ShapeError
from smart_research_agent.inference_optim.types import (
    LEVELS,
    LEVEL_FP32,
    BudgetBreakdown,
    cache_bytes,
    element_bytes,
)


def weight_bytes(card: ModelCard, *, level: str = LEVEL_FP32) -> int:
    """权重占多少字节（**与序列长度无关**，因此它是那个"地基"）."""
    return int(card.parameter_count * element_bytes(level))


def activation_bytes(
    card: ModelCard,
    *,
    batch: int = 1,
    tokens: int = 1,
    level: str = LEVEL_FP32,
) -> int:
    """激活占多少字节（**保守口径**：一层里最宽的那个中间张量 = 前馈的 ``ffn``）.

    这个口径必须写下来，否则"激活"会变成一个可以随便取值的数：
    换成 `hidden` 会让它小 `ffn/h` 倍，而换成"所有层一起"会让它大 `L` 倍——
    两个方向的差别都是数量级，而三种取法看起来都"合理"。
    """
    if batch < 1:
        raise ParameterError(f"batch 必须 >= 1，收到 {batch}。")
    if tokens < 1:
        raise ParameterError(f"tokens 必须 >= 1，收到 {tokens}。")
    return int(batch * tokens * card.ffn * element_bytes(level))


def cache_bytes_for(card: ModelCard, *, tokens: int, level: str = LEVEL_FP32) -> int:
    """缓存占多少字节（**直接调那个唯一的公式**，因此不可能与别处走散）."""
    return cache_bytes(card.layers, tokens, card.hidden, level)


def plan(
    card: ModelCard,
    *,
    level: str = LEVEL_FP32,
    tokens: int = 32,
    batch: int = 1,
    budget_bytes: int = 64 * 1024,
) -> BudgetBreakdown:
    """把一次部署按三项摊开（``fits`` 与余量都在返回的记录里）."""
    if budget_bytes < 0:
        raise ParameterError(f"预算不能为负，收到 {budget_bytes}。")
    if tokens < 0:
        raise ParameterError(f"tokens 不能为负，收到 {tokens}。")
    return BudgetBreakdown(
        level=level,
        parameters=card.parameter_count,
        tokens=tokens,
        hidden=card.hidden,
        layers=card.layers,
        batch=batch,
        weights_bytes=weight_bytes(card, level=level),
        cache_bytes=cache_bytes_for(card, tokens=tokens, level=level),
        activation_bytes=activation_bytes(card, batch=batch, tokens=max(tokens, 1), level=level),
        budget_bytes=budget_bytes,
    )


def per_token_cost(card: ModelCard, *, level: str, batch: int) -> int:
    """每多一个位置要多花多少字节（**缓存与激活两项之和**）.

    它是"最多能做多长"那个一次方程的斜率。两项都是整数，因此斜率也是整数——
    于是 `T_max` 是一个精确的整除结果，而不是一个估计。
    """
    if batch < 1:
        raise ParameterError(f"batch 必须 >= 1，收到 {batch}。")
    elements = int(2 * card.layers * card.hidden * element_bytes(level))
    activations = int(batch * card.ffn * element_bytes(level))
    return elements + activations


def max_tokens(
    card: ModelCard,
    *,
    level: str = LEVEL_FP32,
    batch: int = 1,
    budget_bytes: int = 64 * 1024,
) -> int:
    """预算内**最多**能做多长（整除，向下取整）.

    权重那一项不随长度变，因此先把它从预算里扣掉；剩下的钱按斜率买长度。
    若连权重都放不下，本函数返回 **0**——"放不下"这件事由
    :meth:`BudgetBreakdown.require_fits` 或 :func:`check_max_tokens` 报出来，
    而不是在这里抛错（一个返回 0 的函数比一个抛错的函数更容易被算进表里）。
    """
    available = budget_bytes - weight_bytes(card, level=level)
    cost = per_token_cost(card, level=level, batch=batch)
    if available < 0 or cost <= 0:  # pragma: no cover - cost 恒为正
        return 0
    return available // cost


def check_max_tokens(
    card: ModelCard,
    *,
    level: str = LEVEL_FP32,
    batch: int = 1,
    budget_bytes: int = 64 * 1024,
) -> str:
    """一行读数：``T_max`` 放得下，而 ``T_max + 1`` 放不下（**边界两侧都查**）.

    两条一起印是刻意的：只查"放得下"的话，一个偏小的 `T_max` 也会通过——
    而"保守"与"算错"读起来一样。
    """
    longest = max_tokens(card, level=level, batch=batch, budget_bytes=budget_bytes)
    inside = plan(
        card, level=level, tokens=longest, batch=batch, budget_bytes=budget_bytes
    )
    outside = plan(
        card, level=level, tokens=longest + 1, batch=batch, budget_bytes=budget_bytes
    )
    if longest == 0:
        raise ShapeError(
            f"{level} 下连权重都放不下（权重 {weight_bytes(card, level=level)} 字节 "
            f"> 预算 {budget_bytes} 字节）：这时候'能做多长'这个问题没有答案。"
        )
    if not inside.fits or outside.fits:
        # 一个真实的边界错误：说明这个"最大值"不是最大
        raise ShapeError(
            f"边界两侧没对上：T={longest} 放得下={inside.fits}、"
            f"T={longest + 1} 放得下={outside.fits}。"
            "（'保守'与'算错'读起来一样，因此这里必须两侧都查。）"
        )
    return (
        f"{level} 下最多 {longest} 个位置：T={longest} 合计 {inside.total_bytes} ✓、"
        f"T={longest + 1} 合计 {outside.total_bytes} ✗"
    )


def level_table(
    card: ModelCard,
    *,
    tokens: int = 32,
    batch: int = 1,
    budget_bytes: int = 64 * 1024,
) -> tuple[BudgetBreakdown, ...]:
    """四种精度各算一次（**同一份权重、同一个预算、只换精度**）."""
    return tuple(
        plan(card, level=level, tokens=tokens, batch=batch, budget_bytes=budget_bytes)
        for level in LEVELS
    )


def compression_ratio(baseline: str, other: str) -> float:
    """``other`` 相对 ``baseline`` 的权重压缩比（fp32 → int4 是 8 倍）."""
    return element_bytes(baseline) / element_bytes(other)


def level_line(breakdown: BudgetBreakdown) -> str:
    """一行读数：一个精度下的三项与它的占比."""
    share = breakdown.share
    return (
        breakdown.line()
        + f" | 占比 权重 {share['weights']:.1%} / 缓存 {share['cache']:.1%} / "
        f"激活 {share['activations']:.1%}"
    )


__all__ = [
    "activation_bytes",
    "cache_bytes_for",
    "check_max_tokens",
    "compression_ratio",
    "level_line",
    "level_table",
    "max_tokens",
    "per_token_cost",
    "plan",
    "weight_bytes",
]
