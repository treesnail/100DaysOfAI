"""``study``：六张表（day087 / M7-D11）.

前十五天每天最后都留下几张表，因为**一个数只有被印出来才可能被反驳**。
今天六张表各回答一个"这笔账算对了吗"的问题：

```text
缓存步表      每一步的缓存字节数（prefill 一步到位、之后每步一个固定增量）
缓存公式表    长度 × 每步 == 总量（对几个长度各核一次）
量化表        四种组合（2 个位数 × 2 个方案 × 2 种粒度）的误差与 SNR
粒度收益表    只换粒度：同样位数下 per_tensor 与 per_channel 的误差之差
批表          静态批 vs 连续批的步数、占用率与吞吐
预算表        四种精度下的三项之和、余量、T_max
```

## 一条纪律：每一行都要带**两个数**

与 day082 的探针同源——一行只写"通过"的表是没法反驳的。
因此每张表的每一行都至少有两列：**读数**与**参照**
（每步/公式、实测/上界、静态/连续、总量/预算）。
"""

from __future__ import annotations

from dataclasses import dataclass

from smart_research_agent.hf_integration.forward import ModelWeights
from smart_research_agent.hf_integration.types import ModelCard
from smart_research_agent.inference_optim import batching, budget, cache as cache_module
from smart_research_agent.inference_optim.quantize import QuantSpec, measure
from smart_research_agent.inference_optim.types import (
    GRANULARITIES,
    LEVELS,
    LEVEL_FP32,
    LEVEL_INT4,
    LEVEL_INT8,
    OPTIM_NOTES,
    SCHEMES,
    SCHEME_ABSMAX,
    SCHEME_ZERO_POINT,
    StepRow,
)

#: 缓存步表用的提示长度与步数（与演示脚本同源，因此读数可复算）。
PREFILL_TOKENS = 3
DECODE_STEPS = 4

#: 缓存公式表核对的几个长度。
FORMULA_TOKENS: tuple[int, ...] = (0, 1, 5, 16, 32)

#: 批表用的长度分布（刻意不整齐：整齐的长度让"填充"这件事消失）。
BATCH_LENGTHS: tuple[int, ...] = (9, 2, 5, 1, 7)

#: 批表用的批大小。
BATCH_MAX = 3

#: 预算表用的 token 数与预算是 64 KiB（与测试同源）。
BUDGET_TOKENS = 32
BUDGET_BYTES = 64 * 1024


@dataclass(frozen=True)
class CacheStepTable:
    """缓存步表：每一步一行 + 一行公式核对."""

    rows: tuple[StepRow, ...]
    formula_line: str

    def lines(self) -> tuple[str, ...]:
        """逐行读数."""
        return tuple(row.line() for row in self.rows) + (self.formula_line,)


@dataclass(frozen=True)
class QuantRow:
    """量化表的一行：一种组合的四个读数."""

    level: str
    scheme: str
    granularity: str
    max_abs_error: float
    mean_abs_error: float
    snr_db: float
    bound: float

    @property
    def satisfies_bound(self) -> bool:
        """实测误差是否不超过 ``scale/2``（**带方向的那一列**）."""
        return self.max_abs_error <= self.bound + 1e-12

    def line(self) -> str:
        """``int8 absmax   per_tensor  最大 1.9e-03 ≤ 上界 1.9e-03 ✓ | SNR 52.4 dB``."""
        mark = "✓" if self.satisfies_bound else "✗"
        return (
            f"{self.level:<5} {self.scheme:<10} {self.granularity:<11} "
            f"最大 {self.max_abs_error:.3e} ≤ {self.bound:.3e} {mark} | "
            f"均值 {self.mean_abs_error:.3e} | SNR {self.snr_db:6.2f} dB"
        )


@dataclass(frozen=True)
class GranularityRow:
    """粒度收益表的一行：同一个位数下两种粒度的**均值**误差之比.

    刻意的选择：用**均值**而不是最大值。最大值被那个"带离群值的那一行"占住，
    而两种粒度下它都是同一行——因此最大值会把收益完全掩盖。
    均值看的是**其余那些行细了多少倍**，而那正是粒度买来的东西。
    """

    level: str
    scheme: str
    per_tensor_error: float
    per_channel_error: float

    @property
    def ratio(self) -> float:
        """``per_tensor / per_channel``（> 1 表示细粒度更好）."""
        if self.per_channel_error == 0.0:
            return float("inf")
        return self.per_tensor_error / self.per_channel_error

    def line(self) -> str:
        """``int8 absmax  per_tensor 1.9e-03 vs per_channel 2.4e-04 ⇒ 7.9 倍``."""
        return (
            f"{self.level:<5} {self.scheme:<10} per_tensor {self.per_tensor_error:.3e} vs "
            f"per_channel {self.per_channel_error:.3e} ⇒ {self.ratio:.1f} 倍"
        )


@dataclass(frozen=True)
class BudgetRow:
    """预算表的一行：一个精度下的三项、余量与 T_max."""

    level: str
    weights_bytes: int
    cache_bytes: int
    activation_bytes: int
    total_bytes: int
    headroom_bytes: int
    fits: bool
    longest: int

    def line(self) -> str:
        """``int8  | 权重 11 536 + 缓存 512 + 激活 2 048 = 14 096 / 65 536 ✓ | T_max 1428``."""
        mark = "✓" if self.fits else "✗"
        return (
            f"{self.level:<5} | 权重 {self.weights_bytes:>8} + 缓存 {self.cache_bytes:>8} + "
            f"激活 {self.activation_bytes:>8} = {self.total_bytes:>9} / {BUDGET_BYTES:>9} {mark}"
            f" | 余量 {self.headroom_bytes:>8} | T_max {self.longest}"
        )


def cache_step_table(
    card: ModelCard,
    weights: ModelWeights,
    *,
    level: str = LEVEL_FP32,
    prompt_tokens: int = PREFILL_TOKENS,
    steps: int = DECODE_STEPS,
) -> CacheStepTable:
    """缓存步表：跑一次真生成，把每一步的缓存字节数印出来.

    它调 :func:`cache.generate_with_cache`（真跑前向），因此表里的数是**算出来的**
    而不是"按公式填的"——公式那一侧由最后一行核对。
    """
    prompt = tuple(range(1, prompt_tokens + 1))
    _logits, _state, rows = cache_module.generate_with_cache(
        card, weights, prompt, steps=steps, level=level
    )
    return CacheStepTable(
        rows=rows,
        formula_line=cache_module.cache_account_line(
            card, tokens=prompt_tokens + steps, level=level
        ),
    )


def cache_formula_rows(card: ModelCard, *, level: str = LEVEL_FP32) -> tuple[str, ...]:
    """缓存公式表：对几个长度各核一次 ``长度 × 每步 == 总量``."""
    per_step = int(2 * card.layers * card.hidden * budget.element_bytes(level))
    lines: list[str] = []
    for tokens in FORMULA_TOKENS:
        total = cache_module.make_cache(card, level=level).total_bytes
        formula = cache_module.cache_account_line(card, tokens=tokens, level=level)
        expected = total + per_step * tokens if tokens else 0
        lines.append(
            f"T={tokens:<3} 总量 {expected:<8} == 每步 {per_step} × {tokens} "
            f"| {formula}"
        )
    return tuple(lines)


def quant_rows(matrix: tuple[tuple[float, ...], ...]) -> tuple[QuantRow, ...]:
    """量化表：四种组合（2 个位数 × 2 个方案 × 2 种粒度）各一行."""
    rows: list[QuantRow] = []
    for level in (LEVEL_INT8, LEVEL_INT4):
        for scheme in SCHEMES:
            for granularity in GRANULARITIES:
                quantized, stats = measure(
                    matrix, QuantSpec(level=level, scheme=scheme, granularity=granularity)
                )
                rows.append(
                    QuantRow(
                        level=level,
                        scheme=scheme,
                        granularity=granularity,
                        max_abs_error=stats.max_abs_error,
                        mean_abs_error=stats.mean_abs_error,
                        snr_db=stats.snr_db,
                        bound=quantized.scale / 2.0,
                    )
                )
    return tuple(rows)


def skewed_matrix(
    *,
    rows: int = 4,
    columns: int = 16,
    base: float = 0.05,
    outlier: float = 4.0,
    outlier_rows: int = 1,
) -> tuple[tuple[float, ...], ...]:
    """一份**带离群值**的矩阵（前 ``outlier_rows`` 行各有一个放大到 ``outlier`` 的元素）.

    它存在的理由是一条被量出来的事实：在"每一行的分布都相同、没有离群值"的
    权重上，``per_tensor`` 与 ``per_channel`` 的误差**几乎一样**
    （读数 ≈ 1.0 倍）——因为两种粒度算出的 scale 本来就差不多。

    粒度只在**行与行的动态范围差得很大**时才值钱，而真实权重里正是这样：
    少数通道的值比其余大一到两个数量级。因此"粒度比位数更值钱"这句话
    必须带上这个前提，而它需要一个专门造出来的样本才量得出来。

    ``outlier_rows=1`` 是刻意的：**只有一行**带离群值时，per_tensor 的 scale
    被那一行抬高，其余各行都跟着变粗——那才是粒度要解决的那个问题。
    """
    from smart_research_agent.math_foundations.probability import uniforms

    raw = uniforms(rows * columns, seed=17)
    out: list[tuple[float, ...]] = []
    for row in range(rows):
        chunk = list(raw[row * columns : (row + 1) * columns])
        if row < outlier_rows:
            chunk[0] = outlier * (1.0 if row % 2 == 0 else -1.0)
        out.append(tuple((value * 2.0 - 1.0) * base for value in chunk))
    return tuple(out)


def granularity_rows(
    matrix: tuple[tuple[float, ...], ...] | None = None,
) -> tuple[GranularityRow, ...]:
    """粒度收益表：只换粒度（位数与方案都不动），把收益量出来.

    它存在的理由是"粒度比位数更值钱"这句话需要一个读数——
    而它只有在**同一个位数**下比才有意义（换位数时收益来自两个地方，无法归因）。
    默认样本用 :func:`skewed_matrix`（**带离群值**）：在均匀分布的权重上，
    两种粒度的差别会小到读数 ≈ 1.0 倍，那句结论也就量不出来。
    """
    sample = skewed_matrix() if matrix is None else matrix
    rows: list[GranularityRow] = []
    for level in (LEVEL_INT8, LEVEL_INT4):
        for scheme in (SCHEME_ABSMAX, SCHEME_ZERO_POINT):
            coarse = measure(
                sample,
                QuantSpec(level=level, scheme=scheme, granularity="per_tensor"),
            )[1].mean_abs_error
            fine = measure(
                sample,
                QuantSpec(level=level, scheme=scheme, granularity="per_channel"),
            )[1].mean_abs_error
            rows.append(
                GranularityRow(
                    level=level,
                    scheme=scheme,
                    per_tensor_error=coarse,
                    per_channel_error=fine,
                )
            )
    return tuple(rows)


def batch_rows(
    lengths: tuple[int, ...] = BATCH_LENGTHS,
    *,
    max_batch: int = BATCH_MAX,
) -> tuple[str, ...]:
    """批表：静态与连续两种做法的步数、占用率与吞吐."""
    plan = batching.plan_batches(lengths, max_batch=max_batch)
    schedule = batching.continuous_batching_schedule(lengths, max_batch=max_batch)
    rate, occupied, total = batching.occupancy(schedule)
    lines = plan.lines()
    lines = lines + (
        f"连续批的处理表：{len(schedule)} 步 × {max_batch} 槽位 = {total} | "
        f"占用 {occupied}（{rate:.1%}）",
    )
    for index, row in enumerate(schedule):
        lines = lines + (
            f"  步 {index}：{['空' if member < 0 else member for member in row]}",
        )
    return lines


def budget_rows(
    card: ModelCard,
    *,
    tokens: int = BUDGET_TOKENS,
    budget_bytes: int = BUDGET_BYTES,
) -> tuple[BudgetRow, ...]:
    """预算表：四种精度各一行（**同一份权重、同一个预算，只换精度**）."""
    rows: list[BudgetRow] = []
    for level in LEVELS:
        breakdown = budget.plan(
            card, level=level, tokens=tokens, batch=1, budget_bytes=budget_bytes
        )
        rows.append(
            BudgetRow(
                level=level,
                weights_bytes=breakdown.weights_bytes,
                cache_bytes=breakdown.cache_bytes,
                activation_bytes=breakdown.activation_bytes,
                total_bytes=breakdown.total_bytes,
                headroom_bytes=breakdown.headroom_bytes,
                fits=breakdown.fits,
                longest=budget.max_tokens(
                    card, level=level, batch=1, budget_bytes=budget_bytes
                ),
            )
        )
    return tuple(rows)


def note_lines(limit: int | None = None) -> tuple[str, ...]:
    """优化笔记逐行印出（十条，顺序即写入顺序）."""
    items = tuple(OPTIM_NOTES.items())
    if limit is not None:
        items = items[:limit]
    return tuple(f"{index:>2}. {value}" for index, (_key, value) in enumerate(items, start=1))


def study_lines(
    card: ModelCard,
    weights: ModelWeights,
    matrix: tuple[tuple[float, ...], ...],
    *,
    level: str = LEVEL_FP32,
) -> tuple[str, ...]:
    """一次跑完六张表（演示脚本与教程引用的是同一批读数）."""
    lines: list[str] = []
    lines.append(f"== 1. 缓存步表（{card.layers} 层、{card.hidden} 维、{level}）")
    for line in cache_step_table(card, weights, level=level).lines():
        lines.append("  " + line)
    lines.append("== 2. 缓存公式表（长度 × 每步 == 总量）")
    for line in cache_formula_rows(card, level=level):
        lines.append("  " + line)
    lines.append("== 3. 量化表（位数 × 方案 × 粒度）")
    for row in quant_rows(matrix):
        lines.append("  " + row.line())
    lines.append("== 4. 粒度收益表（只换粒度；样本带离群值）")
    for row in granularity_rows():
        lines.append("  " + row.line())
    lines.append("== 5. 批表（静态 vs 连续）")
    for line in batch_rows():
        lines.append("  " + line)
    lines.append("== 6. 预算表（四种精度、同一个预算）")
    for row in budget_rows(card):
        lines.append("  " + row.line())
    return tuple(lines)


__all__ = [
    "BATCH_LENGTHS",
    "BATCH_MAX",
    "BUDGET_BYTES",
    "BUDGET_TOKENS",
    "DECODE_STEPS",
    "FORMULA_TOKENS",
    "PREFILL_TOKENS",
    "BudgetRow",
    "CacheStepTable",
    "GranularityRow",
    "QuantRow",
    "batch_rows",
    "budget_rows",
    "cache_formula_rows",
    "cache_step_table",
    "granularity_rows",
    "note_lines",
    "quant_rows",
    "skewed_matrix",
    "study_lines",
]
