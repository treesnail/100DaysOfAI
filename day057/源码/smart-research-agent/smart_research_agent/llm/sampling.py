"""采样参数：SamplingParams 校验 + temperature/top_p 的数学演示工具（day033）.

本模块把 OpenAI 兼容 API 的三个核心采样参数显式化：
- ``temperature``：温度缩放，控制概率分布的"尖锐/平坦"；
- ``top_p``：核采样，按累积概率截断候选集；
- ``max_tokens``：生成token数上限，直接决定成本与截断风险。

``softmax_with_temperature`` 与 ``nucleus_filter`` 是纯函数，
供教程与测试离线演示参数的数学效果，无需任何真实模型。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingParams:
    """OpenAI 兼容 API 的采样参数（带取值校验，非法参数在构造时即失败）.

    取值范围（与主流 OpenAI 兼容 API 对齐）：
      - temperature: [0.0, 2.0]，0 表示贪心解码（总取概率最大的 token）；
      - top_p: (0.0, 1.0]，1.0 表示不截断；
      - max_tokens: >= 1。
    """

    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature 必须在 [0.0, 2.0] 内，收到 {self.temperature}"
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p 必须在 (0.0, 1.0] 内，收到 {self.top_p}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens 必须 >= 1，收到 {self.max_tokens}")


def softmax_with_temperature(logits: list[float], temperature: float = 1.0) -> list[float]:
    """温度缩放的 softmax：p_i = exp(z_i / T) / Σ_j exp(z_j / T).

    数学含义：
      - T = 1：原始分布；
      - T → 0：所有 logit 差异被放大，概率质量集中到最大值（趋于 one-hot，即贪心）；
      - T → ∞：差异被抹平，分布趋于均匀（输出近乎随机）。

    实现上先减去 max(logits) 再取 exp，是 softmax 的数值稳定标准做法
    （不改变结果，只防止 exp 溢出）。temperature 必须 > 0；
    T = 0 的贪心语义请在调用侧用 argmax 表达，不属于本函数。
    """
    if not logits:
        raise ValueError("logits 不能为空")
    if temperature <= 0:
        raise ValueError(f"temperature 必须 > 0（T=0 请用 argmax 贪心），收到 {temperature}")
    scaled = [z / temperature for z in logits]
    peak = max(scaled)
    exps = [math.exp(z - peak) for z in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def nucleus_filter(probs: list[float], top_p: float) -> list[float]:
    """核采样（top-p）截断：只保留累积概率达到 top_p 的最小候选集，并重新归一化.

    过程：按概率降序排序 → 逐个累加，累积和达到 top_p 即截断
    （使累积和 >= top_p 的那个 token 保留在集合内）→ 其余置零 → 归一化。

    与 top-k（固定保留 k 个）相比，核采样的候选集大小随分布形状自适应：
    分布尖锐时只留一两个，分布平坦时保留一大串。
    """
    if not probs:
        raise ValueError("probs 不能为空")
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p 必须在 (0.0, 1.0] 内，收到 {top_p}")
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    kept = [0.0] * len(probs)
    cumulative = 0.0
    for i in order:
        kept[i] = probs[i]
        cumulative += probs[i]
        if cumulative >= top_p:
            break
    total = sum(kept)
    return [p / total for p in kept]
