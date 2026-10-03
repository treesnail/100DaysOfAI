"""七条性质与三类对账（day087 / M7-D11）.

本模块是"这三本账算对了吗"的判据所在。三类判据，三种比法：

```text
逐位（``==``）    缓存路径 vs 整段路径（同一次算术）、int4 打包往返
整数相等          每步的字节增量 vs 2·L·h·bytes、预算三项之和 vs 总量、
                  处理表的占用槽位数 vs 逐行数出来的数
上界判定          反量化误差 <= scale/2（**这是本轮里唯一"两边不等"的判据**）
```

## 今天新出现的一类判据：**上界**

前十七天里，判据都是"两个数相等"（逐位或整数）。
今天多了一种：**"误差不超过某个上界"**——而它的两个操作数根本不该相等
（一个是实测误差，一个是推导出来的界）。

它比相等更难写对，因为它有一个**方向**：

```text
实测 <= 界      通过        界是推导出来的，实测必须落在它里面
实测 >  界      失败        要么实现错了，要么前提破了（例如输入里有 inf）
```

因此本课把它写成带方向的一行（``读数 / 上界 / 满足``），
而不是一句"误差很小"——后者无法被反驳。

## 三条对账（两条跨天）

```text
缓存 vs 整段      cache.compare_with_recompute ↔ day086 的 forward.logits（**逐位**）
prefill vs 整段    cache.prefill 的 hidden states ↔ day086 的 hidden_states（**逐位**）
量化 vs 原权重     quantify.measure 的误差 ↔ 上界 scale/2（**上界**，不是相等）
```

前两条是本课最强的一类证据：两条路径的算术**逐字相同**（都走 day085 的
`project` / `split_heads` / `row_softmax` / `merge_heads`），
因此它们必须在每一个浮点数上一致。缓存接错一侧（在归一化之前取、K/V 弄反）
时形状完全合法，而这两个向量会立刻分家。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.hf_integration.forward import ModelWeights, hidden_states
from smart_research_agent.hf_integration.types import ModelCard
from smart_research_agent.inference_optim import batching, cache as cache_module
from smart_research_agent.inference_optim.quantize import (
    error_bound,
    measure,
    pack_int4,
    packed_length,
    unpack_int4,
)
from smart_research_agent.inference_optim.budget import (
    activation_bytes,
    cache_bytes_for,
    max_tokens,
    plan,
    weight_bytes,
)
from smart_research_agent.inference_optim.errors import AssemblyError, NumericError
from smart_research_agent.inference_optim.types import (
    GRANULARITY_CHANNEL,
    GRANULARITY_TENSOR,
    LEVEL_FP32,
    LEVEL_INT4,
    LEVEL_INT8,
    OPTIM_PROPERTIES,
    PROPERTY_BUDGET_IS_EXACTLY_ACCOUNTED,
    PROPERTY_CACHED_EQUALS_RECOMPUTE,
    PROPERTY_CACHE_GROWTH_IS_A_FORMULA,
    PROPERTY_DEQUANT_ERROR_WITHIN_HALF_SCALE,
    PROPERTY_INT4_PACK_IS_LOSSLESS,
    PROPERTY_PREFILL_MATCHES_FULL,
    PROPERTY_SCHEDULE_OCCUPANCY_IS_COUNTED,
    SCHEME_ABSMAX,
    SCHEME_ZERO_POINT,
    CacheState,
    QuantSpec,
)

#: 打包往返要覆盖的边界值（有符号 4 位的两个端点必须出现）。
INT4_BOUNDARY = (-8, 7, 0, -1, 1, -7, 6)

#: 无符号那一侧的边界值（0 与 15）——两条式子的端点都要出现，
#: 因为"有符号的 -8 与无符号的 8 是同一个字节码"这件事正是最容易错的地方。
INT4_UNSIGNED_BOUNDARY = (0, 15, 1, 14, 7, 8)


@dataclass(frozen=True)
class CrossCheck:
    """两个来源之间的一次对账：来源、读数、判据（**含方向与上界**）.

    与 day085/day086 的同类多一个字段：``upper_bound``。
    有它的时候判据是"实测 <= 上界"，没有的时候是"两个数相等"。
    """

    name: str
    left: str
    right: str
    reading: float | int
    expected: float | int
    exact: bool = True
    upper_bound: float | None = None

    @property
    def passed(self) -> bool:
        """相等（逐位 / 整数）或"不超过上界"两种判据."""
        if self.upper_bound is not None:
            return float(self.reading) <= float(self.upper_bound) + 1e-12
        if self.exact:
            return self.reading == self.expected
        return abs(float(self.reading) - float(self.expected)) <= 1e-12

    def line(self) -> str:
        """一行可读的读数（带上两个来源与那个上界）."""
        verdict = "满足" if self.passed else "不满足"
        if self.upper_bound is not None:
            return (
                f"[{verdict}] {self.name}: 读数 {self.reading:.3e} ≤ 上界 "
                f"{self.upper_bound:.3e}（{self.left} vs {self.right}）"
            )
        return (
            f"[{verdict}] {self.name}: {self.left} vs {self.right} | "
            f"读数 {self.reading} / 期望 {self.expected}"
        )


@dataclass(frozen=True)
class PropertyOutcome:
    """一条性质的结论：适用性 + 结果 + 证据行."""

    name: str
    applicable: bool
    passed: bool
    evidence: tuple[str, ...] = field(default_factory=tuple)
    cross_check: CrossCheck | None = None

    def __post_init__(self) -> None:
        if not self.applicable and self.passed:
            raise NumericError(
                f"性质 {self.name!r} 标了'不适用'却又标了'通过'：两者必须分开——"
                "否则'把这条检查删掉'与'它通过了'在报告里长得一模一样。"
            )

    def line(self) -> str:
        """一行可读的结论（不适用也要印出来）."""
        if not self.applicable:
            return f"[不适用] {self.name} | {'；'.join(self.evidence)}"
        verdict = "通过" if self.passed else "失败"
        detail = "；".join(self.evidence)
        suffix = f" | {detail}" if detail else ""
        return f"[{verdict}] {self.name}{suffix}"


@dataclass(frozen=True)
class PropertyReport:
    """一组性质的报告（``ok`` 要求**所有适用**的都通过）."""

    outcomes: tuple[PropertyOutcome, ...]

    @property
    def applicable(self) -> tuple[PropertyOutcome, ...]:
        """适用（``applicable=True``）的那些性质."""
        return tuple(outcome for outcome in self.outcomes if outcome.applicable)

    @property
    def ok(self) -> bool:
        """是否全部通过——不适用不算通过、也不算失败."""
        return all(outcome.passed for outcome in self.applicable)

    def require_ok(self) -> None:
        """不通过时抛 :class:`errors.AssemblyError`."""
        if self.ok:
            return
        failures = [outcome.line() for outcome in self.applicable if not outcome.passed]
        raise AssemblyError("性质检查未全部通过：" + "；".join(failures))

    def lines(self) -> tuple[str, ...]:
        """逐行文本（**先印不适用**，让"这一次没查它"一眼可见）."""
        return tuple(
            outcome.line()
            for outcome in sorted(self.outcomes, key=lambda outcome: outcome.applicable)
        )

    def to_dict(self) -> dict[str, object]:
        """摊平成可 JSON 化的字段."""
        return {
            "ok": self.ok,
            "counts": {"total": len(self.outcomes), "applicable": len(self.applicable)},
            "lines": list(self.lines()),
        }


def check_cached_equals_recompute(
    card: ModelCard,
    weights: ModelWeights,
    prompt_ids: tuple[int, ...],
    new_token: int,
) -> PropertyOutcome:
    """缓存路径的最后一步 logits 与整段重算**逐位**相同.

    这条判据的强度来自"两条路径的算术逐字相同"：都走 day085 的那七个阶段。
    差别只有一处——K/V 是"从输入现场投影"还是"从参数传进来"。
    因此它抓的是缓存**接错位置**（在归一化之前取值）或**接错顺序**（K/V 弄反）。
    """
    cached, full = cache_module.compare_with_recompute(card, weights, prompt_ids, new_token)
    same = cached == full
    gap = max(
        (abs(a - b) for a, b in zip(cached, full, strict=True)),
        default=0.0,
    )
    check = CrossCheck(
        name="缓存路径 vs 整段重算",
        left="cache.decode_step（K/V 从缓存读）",
        right="hf_integration.forward.logits（整段重算）",
        reading=0.0 if same else 1.0,
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_CACHED_EQUALS_RECOMPUTE,
        applicable=True,
        passed=same,
        evidence=(
            f"提示 {len(prompt_ids)} 个 token + 新 token {new_token}：逐位相同 {same}",
            f"两边的宽度都是 {len(cached)}（= 词表），最大差 {gap:.6e}",
        ),
        cross_check=check,
    )


def check_prefill_matches_full(
    card: ModelCard,
    weights: ModelWeights,
    prompt_ids: tuple[int, ...],
) -> PropertyOutcome:
    """prefill 的输出与整段前向**逐位**相同，且缓存长度恰好等于提示长度.

    它同时查三件事：输出逐位、缓存长度、以及**每一层的长度都相同**——
    第三条是"追加只走一条路径"的直接后果（某些层多一行时，前向照样跑完）。
    """
    cache = cache_module.make_cache(card)
    states, _logits = cache_module.prefill(card, weights, cache, prompt_ids)
    reference = hidden_states(card, weights, (tuple(prompt_ids),)).rows[0]
    same = states == reference
    lengths = tuple(len(cache.keys_of(index)) for index in range(cache.depth))
    aligned = len(set(lengths)) == 1 and lengths[0] == len(prompt_ids)
    check = CrossCheck(
        name="prefill 的 hidden states vs 整段前向",
        left="cache.prefill",
        right="hf_integration.forward.hidden_states",
        reading=0.0 if same else 1.0,
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_PREFILL_MATCHES_FULL,
        applicable=True,
        passed=same and aligned,
        evidence=(
            f"{len(prompt_ids)} 个 token、{card.layers} 层：逐位相同 {same}",
            f"各层长度 {list(lengths)}，齐且等于提示长度：{aligned}",
        ),
        cross_check=check,
    )


def check_cache_growth_is_a_formula(card: ModelCard, *, level: str = LEVEL_FP32) -> PropertyOutcome:
    """每步新增字节数 == ``2·L·h·bytes``（整数相等），且总量 == 长度 × 每步.

    两处都是整数，因此没有"差不多"的余地。它抓的错法是
    "把 K 与 V 只算了一份"（少一半）或"漏乘层数"（差一个整数倍）。
    """
    cache = cache_module.make_cache(card, level=level)
    from smart_research_agent.inference_optim.types import element_bytes

    expected = int(2 * card.layers * card.hidden * element_bytes(level))
    actual = cache.bytes_per_step
    by_layer = sum(
        int(2 * card.hidden * element_bytes(level)) for _ in range(card.layers)
    )
    check = CrossCheck(
        name="每步的字节增量",
        left="KVCache.bytes_per_step",
        right="2·L·h·bytes",
        reading=actual,
        expected=expected,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_CACHE_GROWTH_IS_A_FORMULA,
        applicable=True,
        passed=check.passed and by_layer == expected,
        evidence=(
            f"{level}：L={card.layers} h={card.hidden} ⇒ 每步 {actual} 字节"
            f"（逐层相加 {by_layer}）",
            f"长度 5 时总量 {cache_module.cache_account_line(card, tokens=5, level=level)}",
        ),
        cross_check=check,
    )


def check_dequant_error_within_half_scale(
    matrix: tuple[tuple[float, ...], ...],
    *,
    level: str = LEVEL_INT8,
    scheme: str = SCHEME_ABSMAX,
    granularity: str = GRANULARITY_TENSOR,
) -> PropertyOutcome:
    """反量化的误差不超过 ``scale/2``（**上界判定**：两个操作数不该相等）.

    它必须**四种组合都跑**（两种方案 × 两种粒度）：因为"上界成立"这件事
    依赖 `round` 的误差 ≤ 0.5 格，而这条前提对两种方案都成立、
    与粒度无关——"与粒度无关"正是需要被验证的那句话。
    """
    spec = QuantSpec(level=level, scheme=scheme, granularity=granularity)
    quantized, stats = measure(matrix, spec)
    bound = error_bound(quantized.scale)
    check = CrossCheck(
        name=f"{level} {scheme} {granularity} 的最大误差",
        left="实测最大绝对误差",
        right="上界 scale/2",
        reading=stats.max_abs_error,
        expected=bound,
        upper_bound=bound,
    )
    return PropertyOutcome(
        name=PROPERTY_DEQUANT_ERROR_WITHIN_HALF_SCALE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"{spec.line()} | 最大误差 {stats.max_abs_error:.3e} ≤ {bound:.3e}",
            f"均值 {stats.mean_abs_error:.3e} | SNR {stats.snr_db:.2f} dB | "
            f"{stats.elements} 个元素",
        ),
        cross_check=check,
    )


def check_int4_pack_is_lossless(values: tuple[int, ...] = INT4_BOUNDARY) -> PropertyOutcome:
    """两个 4 位数打包成一个字节再解回来必须**逐位**还原（含边界值）.

    边界值必须出现在样本里：``-8`` 与 ``7`` 是补码的两个端点，
    而"负数的低 4 位会串到高位"这类错误**只在负数上**出现——
    用一组非负样本试，永远试不出来。
    """
    packed = pack_int4(values, signed=True)
    restored = unpack_int4(packed, len(values), signed=True)
    unsigned = pack_int4(INT4_UNSIGNED_BOUNDARY, signed=False)
    again = unpack_int4(unsigned, len(INT4_UNSIGNED_BOUNDARY), signed=False)
    check = CrossCheck(
        name="int4 打包往返",
        left=f"{len(values)} 个有符号 4 位数",
        right=f"{len(packed)} 个字节（每字节 2 个）",
        reading=0 if restored == values else 1,
        expected=0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_INT4_PACK_IS_LOSSLESS,
        applicable=True,
        passed=(
            restored == values
            and again == INT4_UNSIGNED_BOUNDARY
            and len(packed) == packed_length(len(values))
        ),
        evidence=(
            f"有符号往返逐位还原：{restored == values} | 无符号那侧："
            f"{again == INT4_UNSIGNED_BOUNDARY}",
            f"两组样本各自含端点：{-8 in values and 7 in values} / "
            f"{0 in INT4_UNSIGNED_BOUNDARY and 15 in INT4_UNSIGNED_BOUNDARY} | "
            f"{len(values)} 个元素 ⇒ {len(packed)} 字节",
        ),
        cross_check=check,
    )


def check_schedule_occupancy_is_counted(
    lengths: tuple[int, ...] = (9, 2, 5, 1, 7),
    *,
    max_batch: int = 3,
) -> PropertyOutcome:
    """处理表的占用率 == 占位槽位 / 总槽位，且静态批步数 == Σ 每组最长.

    三条都是**整数**判据：处理表里 ``>= 0`` 的格子数、每一组最长之和、
    以及**每一条请求出现的次数必须恰好等于它的长度**。
    第三条是最值钱的那一条：没有它，"某一条被提前扔掉"与"它算完了"
    在表里长得一样（前者会让占用率虚高，而**总步数不变**）。
    """
    schedule = batching.continuous_batching_schedule(lengths, max_batch=max_batch)
    rate, occupied, total = batching.occupancy(schedule)
    counted = sum(1 for row in schedule for member in row if member >= 0)
    per_request = batching.per_request_steps(schedule)
    plan = batching.plan_batches(lengths, max_batch=max_batch)
    static_by_hand = sum(max(lengths[index] for index in group.members) for group in plan.groups)
    check = CrossCheck(
        name="占用槽位数",
        left="occupancy() 数出来的",
        right="逐格数出来的",
        reading=occupied,
        expected=counted,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_SCHEDULE_OCCUPANCY_IS_COUNTED,
        applicable=True,
        passed=(
            check.passed
            and static_by_hand == plan.static_steps
            and rate == occupied / total
            and per_request == lengths
        ),
        evidence=(
            f"长度 {list(lengths)}、批大小 {max_batch}：{len(schedule)} 步 × {max_batch} 槽位",
            f"占用 {occupied}/{total} = {rate:.1%} | 静态批 {plan.static_steps} 步（手算 "
            f"{static_by_hand}）| 连续批下界 {plan.continuous_steps} 步、实际 "
            f"{len(schedule)} 步（省 {plan.static_steps - len(schedule)}）",
            f"每条请求在表里出现的次数 {list(per_request)} == 各自的长度 {list(lengths)}："
            f"{per_request == lengths}",
        ),
        cross_check=check,
    )


def check_budget_is_exactly_accounted(
    card: ModelCard,
    *,
    level: str = LEVEL_FP32,
    tokens: int = 32,
    batch: int = 1,
    budget_bytes: int = 64 * 1024,
) -> PropertyOutcome:
    """权重 + 缓存 + 激活 == 总量（整数相等），且 ``T_max`` 的两侧都对得上.

    第二半是这条判据里最值钱的部分：只查"``T_max`` 放得下"的话，
    一个**偏小**的最大值也会通过——而"保守"与"算错"读起来一样。
    因此这里两侧都查（`max_tokens` 内部已经做了，这里再核对一次它的返回）。
    """
    breakdown = plan(
        card, level=level, tokens=tokens, batch=batch, budget_bytes=budget_bytes
    )
    summed = (
        weight_bytes(card, level=level)
        + cache_bytes_for(card, tokens=tokens, level=level)
        + activation_bytes(card, batch=batch, tokens=max(tokens, 1), level=level)
    )
    longest = max_tokens(card, level=level, batch=batch, budget_bytes=budget_bytes)
    inside = plan(
        card, level=level, tokens=longest, batch=batch, budget_bytes=budget_bytes
    ).fits
    outside = plan(
        card, level=level, tokens=longest + 1, batch=batch, budget_bytes=budget_bytes
    ).fits
    check = CrossCheck(
        name="预算的三项之和",
        left="Breakdown.total_bytes",
        right="权重 + 缓存 + 激活",
        reading=breakdown.total_bytes,
        expected=summed,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_BUDGET_IS_EXACTLY_ACCOUNTED,
        applicable=True,
        passed=check.passed and inside and not outside,
        evidence=(
            f"{level}：{breakdown.total_bytes} = {breakdown.weights_bytes} + "
            f"{breakdown.cache_bytes} + {breakdown.activation_bytes} | 逐项相加 {summed}",
            f"T_max={longest}：放得下 {inside}、T_max+1 放得下 {outside}",
            f"占比 权重 {breakdown.share['weights']:.1%} / 缓存 {breakdown.share['cache']:.1%} / "
            f"激活 {breakdown.share['activations']:.1%}",
        ),
        cross_check=check,
    )


def check_all(
    card: ModelCard,
    weights: ModelWeights,
    *,
    prompt_ids: tuple[int, ...],
    new_token: int,
    quant_matrix: tuple[tuple[float, ...], ...],
    levels: tuple[str, ...] = (LEVEL_INT8, LEVEL_INT4),
    schemes: tuple[str, ...] = (SCHEME_ABSMAX, SCHEME_ZERO_POINT),
    granularities: tuple[str, ...] = (GRANULARITY_TENSOR, GRANULARITY_CHANNEL),
    budget_bytes: int = 64 * 1024,
) -> PropertyReport:
    """一次跑完七条性质（顺序与 :data:`types.OPTIM_PROPERTIES` 一致）.

    量化那一条按 ``levels × schemes × granularities`` 展开成多条读数，
    但**性质名只有一个**——因为它们是同一条判据在不同方案下的重复，
    而不是七条不同的判据（报告里按名去重，读数留在证据行里）。
    """
    quant_outcomes = [
        check_dequant_error_within_half_scale(
            quant_matrix, level=level, scheme=scheme, granularity=granularity
        )
        for level in levels
        for scheme in schemes
        for granularity in granularities
    ]
    quant_ok = all(outcome.passed for outcome in quant_outcomes)
    outcomes = (
        check_cached_equals_recompute(card, weights, prompt_ids, new_token),
        check_prefill_matches_full(card, weights, prompt_ids),
        check_cache_growth_is_a_formula(card),
        PropertyOutcome(
            name=PROPERTY_DEQUANT_ERROR_WITHIN_HALF_SCALE,
            applicable=True,
            passed=quant_ok,
            evidence=(
                f"{len(quant_outcomes)} 种组合（{len(levels)} 个位数 × {len(schemes)} 个方案 × "
                f"{len(granularities)} 种粒度）全部满足上界：{quant_ok}",
                "最差的一种："
                + max(
                    quant_outcomes,
                    key=lambda outcome: outcome.cross_check.reading  # type: ignore[union-attr]
                ).line(),
            ),
        ),
        check_int4_pack_is_lossless(),
        check_schedule_occupancy_is_counted(),
        check_budget_is_exactly_accounted(card, budget_bytes=budget_bytes),
    )
    names = {outcome.name for outcome in outcomes}
    extra = names - set(OPTIM_PROPERTIES)
    missing = set(OPTIM_PROPERTIES) - names
    if extra or missing:
        raise AssemblyError(
            "性质名单与 types.OPTIM_PROPERTIES 不一致："
            f"多 {sorted(extra)}、缺 {sorted(missing)}。"
            "名单对不上时，报告里那七行会安静地少一行或多一行。"
        )
    return PropertyReport(outcomes=tuple(outcomes))


def quant_summary(matrix: tuple[tuple[float, ...], ...]) -> tuple[str, ...]:
    """把四种组合逐个印一遍（**报告里读它**：一个方案一行，方向都是 lower）."""
    lines: list[str] = []
    for level in (LEVEL_INT8, LEVEL_INT4):
        for scheme in (SCHEME_ABSMAX, SCHEME_ZERO_POINT):
            for granularity in (GRANULARITY_TENSOR, GRANULARITY_CHANNEL):
                _quantized, stats = measure(
                    matrix, QuantSpec(level=level, scheme=scheme, granularity=granularity)
                )
                lines.append(stats.line() + f" | 方向 {stats.direction}")
    return tuple(lines)


def finite_or_raise(values: tuple[float, ...], *, name: str) -> None:
    """校验一串数全为有限（报告端的最后一道闸）."""
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise NumericError(f"{name} 的第 {index} 个值不是有限数：{value!r}。")


def cache_state_line(state: CacheState) -> str:
    """一行读数：缓存的四个上下文读数."""
    return (
        f"{state.depth} 层 × {state.length}/{state.capacity} 位置 @ {state.level} | "
        f"总量 {state.total_bytes} 字节 | 还能装 {state.free} 个位置"
    )


__all__ = [
    "INT4_BOUNDARY",
    "INT4_UNSIGNED_BOUNDARY",
    "CrossCheck",
    "PropertyOutcome",
    "PropertyReport",
    "cache_state_line",
    "check_all",
    "check_budget_is_exactly_accounted",
    "check_cache_growth_is_a_formula",
    "check_cached_equals_recompute",
    "check_dequant_error_within_half_scale",
    "check_int4_pack_is_lossless",
    "check_prefill_matches_full",
    "check_schedule_occupancy_is_counted",
    "finite_or_raise",
    "quant_summary",
]
