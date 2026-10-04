"""七条性质与两条跨天对账（day085 / M7-D9）.

本模块是"读源码读对了吗"的判据所在。它有两类：

```text
六条**自洽**性质   只涉及本包的输出：行随机、掩码外恰好 0、因果前缀稳定、
                  分头拼回恒等、融合等于分离、与 day076 的单头一致
一条**跨天**对账   把本包的块与 day079 的 encoder_block 放在一起比
```

## 两类判据用不同的比法，而这件事必须写清楚

```text
逐位（``==``）      两个量来自**同一次**算术：分头/拼回、掩码外是否为 0、
                    因果前缀、融合与分离（拼接顺序一致时必须逐位相同）
容差（``1e-12``）   两个量来自**两份独立实现**：本包与 day076 的注意力、
                    本包与 day079 的块。两处的浮点运算顺序不必相同——
                    比如一处乘 ``1/√head_dim``、一处除 ``√head_dim``。
```

day080 的那条纪律在这里第三次兑现：**两份实现之间的差异要么被消灭，
要么被写下来。** 本包把它写成一条带 `max_gap` 的读数，而不是一句"应该一样"。

## 一条"不可能失败"的读数也要印出来

`PropertyReport.ok` 要求"所有**适用**的性质都通过"，而 `applicable=False` 的性质
**不允许**同时也是 `passed=True`（构造期就拒绝）——理由与 day082 第 6.2 节逐字相同：
若"不适用"直接记成"通过"，那么"把这条检查删掉"与"它通过了"在报告里长得一模一样。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.encoder_decoder.layers import block_attention, encoder_block
from smart_research_agent.encoder_decoder.types import BlockParameters
from smart_research_agent.hf_source.attention import (
    SINGLE_HEAD_TOLERANCE,
    HfAttentionForward,
    fused_weight,
    hf_attention,
    merge_heads,
    project,
    split_fused,
    split_heads,
)
from smart_research_agent.hf_source.blocks import (
    BLOCK_TOLERANCE,
    gpt2_block,
    max_abs_gap,
)
from smart_research_agent.hf_source.errors import AssemblyError, NumericError, ShapeError
from smart_research_agent.hf_source.types import (
    PROPERTY_CAUSAL_PREFIX_IS_STABLE,
    PROPERTY_FUSED_MATCHES_SEPARATE,
    PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO,
    PROPERTY_PRE_NORM_BLOCK_MATCHES_DAY079,
    PROPERTY_ROWS_ARE_DISTRIBUTIONS,
    PROPERTY_SINGLE_HEAD_MATCHES_MULTI_HEAD,
    PROPERTY_SPLIT_MERGE_ROUND_TRIP,
    SOURCE_PROPERTIES,
    SourceShape,
)
from smart_research_agent.math_foundations.types import Matrix, validate_matrix
from smart_research_agent.multi_head.layers import multi_head_attention
from smart_research_agent.transformer_core.types import AttentionParams

#: 行随机的容差（浮点求和的有理误差，与 day083 同口径）.
ROW_SUM_TOLERANCE = 1e-9

#: 扰动幅度（与 day082 的探针同源：把一行乘 1.5）.
PERTURBATION = 1.5


@dataclass(frozen=True)
class CrossCheck:
    """两条实现之间的一次对账：两个来源、一个最大差、一个容差."""

    name: str
    left: str
    right: str
    max_gap: float
    tolerance: float

    @property
    def passed(self) -> bool:
        """最大差是否落在容差之内（``<=``，边界算通过）."""
        return self.max_gap <= self.tolerance

    def line(self) -> str:
        """一行可读的读数（**带上两个来源**，否则这个数无法被追溯）."""
        verdict = "一致" if self.passed else "不一致"
        return (
            f"[{verdict}] {self.name}: {self.left} vs {self.right} | "
            f"最大差 {self.max_gap:.3e} | 容差 {self.tolerance:.0e}"
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
        """不通过时抛 :class:`AssemblyError`（消息里带上失败那几条的原文）."""
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


def check_rows_are_distributions(forward: HfAttentionForward) -> PropertyOutcome:
    """每一行非负、和为 1、全部有限（注意力权重是一组分布）."""
    checked = 0
    worst = 0.0
    for head in forward.weights:
        for row in head:
            for value in row:
                if not math.isfinite(value):
                    raise NumericError(
                        "权重里出现了非有限数：一个 nan 会顺着'平均熵'污染整张表"
                        "（day073 第七章的同一条）。"
                    )
                if value < 0.0:
                    raise NumericError(
                        f"权重出现了负数 {value!r}：softmax 的输出不可能为负——"
                        "出现负数说明某个位置在归一化之前被减过。"
                    )
            total = math.fsum(row)
            checked += 1
            worst = max(worst, abs(total - 1.0))
    return PropertyOutcome(
        name=PROPERTY_ROWS_ARE_DISTRIBUTIONS,
        applicable=True,
        passed=worst <= ROW_SUM_TOLERANCE,
        evidence=(
            f"{forward.heads} 头 × {len(forward.weights[0])} 行 = {checked} 行",
            f"最大行和误差 {worst:.3e}（容差 {ROW_SUM_TOLERANCE:.0e}）",
        ),
    )


def check_masked_entries_are_exact_zero(forward: HfAttentionForward) -> PropertyOutcome:
    """被加性掩码挡掉的格子**逐位为 0.0**（不是"很小"）.

    没有掩码时这条性质**不适用**——而"不适用"要如实印出来，
    因为它与"查了、通过了"是两件不同的事（day083 第 7.2 节）。
    """
    if forward.bias is None:
        return PropertyOutcome(
            name=PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO,
            applicable=False,
            passed=False,
            evidence=("这一次前向没有掩码：掩码外共 0 个格子，没有东西可查",),
        )
    blocked = 0
    worst = 0.0
    for head in forward.weights:
        for index, row in enumerate(head):
            for column, value in enumerate(row):
                if forward.bias[index][column] < 0.0:
                    blocked += 1
                    worst = max(worst, abs(value))
    return PropertyOutcome(
        name=PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO,
        applicable=True,
        passed=worst == 0.0,
        evidence=(f"掩码外共 {blocked} 个格子，最大读数 {worst:.6e}",),
    )


def check_causal_prefix_is_stable(
    params: AttentionParams,
    inputs: Matrix,
    shape: SourceShape,
) -> PropertyOutcome:
    """扰动最后一个输入位置后，它**前面**那些位置的输出逐位不变.

    这条性质是 day082 那句"因果性是用扰动量出来的"在本包的写法：
    被掩码挡掉的格子 `scores[i][j]` 虽然被算出来了，但 `exp / sum / 加权`
    三步都不碰它，因此改它不会改变输出的**任何一个浮点**。
    """
    checked = validate_matrix(inputs, name="inputs")
    if not shape.causal_default:
        return PropertyOutcome(
            name=PROPERTY_CAUSAL_PREFIX_IS_STABLE,
            applicable=False,
            passed=False,
            evidence=("这一次前向是双向的（编码器一侧）：没有'未来'可以被扰动",),
        )
    before = hf_attention(params, checked, shape, causal=True)
    perturbed = tuple(checked[:-1]) + (
        tuple(value * PERTURBATION for value in checked[-1]),
    )
    after = hf_attention(params, perturbed, shape, causal=True)
    stable = sum(
        1
        for row_before, row_after in zip(before.output[:-1], after.output[:-1])
        if row_before == row_after
    )
    changed = before.output[-1] != after.output[-1]
    total = len(before.output) - 1
    return PropertyOutcome(
        name=PROPERTY_CAUSAL_PREFIX_IS_STABLE,
        applicable=True,
        passed=stable == total and changed,
        evidence=(
            f"扰动第 {len(checked) - 1} 行（×{PERTURBATION:g}）",
            f"前 {total} 行逐位不变：{stable}/{total}",
            f"被扰动那一行确实变了：{changed}",
        ),
    )


def check_split_merge_round_trip(matrix: Matrix, heads: int) -> PropertyOutcome:
    """``split_heads`` 之后 ``merge_heads`` 必须逐位还原（分头只是记账）."""
    checked = validate_matrix(matrix, name="matrix")
    restored = merge_heads(split_heads(checked, heads))
    return PropertyOutcome(
        name=PROPERTY_SPLIT_MERGE_ROUND_TRIP,
        applicable=True,
        passed=restored == checked,
        evidence=(
            f"heads={heads}，head_dim={len(checked[0]) // heads}",
            f"逐位还原：{restored == checked}",
        ),
    )


def check_fused_matches_separate(params: AttentionParams, inputs: Matrix) -> PropertyOutcome:
    """融合投影（GPT-2 的 ``c_attn``）切回去后，与三次独立投影**逐位相同**.

    这条性质是本课最值钱的一条"逐位"判据：GPT-2 只写了一次投影、BERT 写了三次，
    而它们必须是同一件事。拼接顺序错了**不会报错**——`split` 之后拿到的 q
    其实是最初的 v，三个角色互换，注意力照样跑。
    """
    checked = validate_matrix(inputs, name="inputs")
    fused = fused_weight(params.w_query, params.w_key, params.w_value)
    hidden = len(params.w_query)
    back_query, back_key, back_value = split_fused(fused, hidden)
    identical = (
        back_query == validate_matrix(params.w_query, name="w_query")
        and back_key == validate_matrix(params.w_key, name="w_key")
        and back_value == validate_matrix(params.w_value, name="w_value")
    )
    gap = max(
        max_abs_gap(project(checked, back_query), project(checked, params.w_query)),
        max_abs_gap(project(checked, back_key), project(checked, params.w_key)),
        max_abs_gap(project(checked, back_value), project(checked, params.w_value)),
    )
    return PropertyOutcome(
        name=PROPERTY_FUSED_MATCHES_SEPARATE,
        applicable=True,
        passed=identical and gap == 0.0,
        evidence=(
            f"融合宽度 {hidden * 3} = 3×{hidden}",
            f"切回后逐位相同：{identical}",
            f"三条投影路径的最大差 {gap:.6e}",
        ),
    )


def check_single_head_matches_multi_head(
    params: AttentionParams,
    inputs: Matrix,
    shape: SourceShape,
    *,
    causal: bool = False,
) -> PropertyOutcome:
    """``heads=1`` 时本包的注意力与 day076 的 ``multi_head_attention`` 一致.

    这是**跨天对账**：两份独立实现，因此判据是容差（:data:`SINGLE_HEAD_TOLERANCE`），
    而不是逐位。差异的来源可以逐条说出来（乘与除 `√head_dim`、求和顺序），
    因此它是一个"被写下来的差异"，而不是一个"应该一样的期望"。
    """
    if shape.heads != 1:
        return PropertyOutcome(
            name=PROPERTY_SINGLE_HEAD_MATCHES_MULTI_HEAD,
            applicable=False,
            passed=False,
            evidence=(f"这一次 heads={shape.heads}：两边只承诺在 heads=1 上一致",),
        )
    mine = hf_attention(params, inputs, shape, causal=causal)
    reference = multi_head_attention(params, inputs, heads=1, causal=causal)
    gap = max_abs_gap(mine.output, reference.output)
    check = CrossCheck(
        name="hf_attention vs multi_head_attention（heads=1）",
        left="hf_source.attention",
        right="multi_head.layers",
        max_gap=gap,
        tolerance=SINGLE_HEAD_TOLERANCE,
    )
    return PropertyOutcome(
        name=PROPERTY_SINGLE_HEAD_MATCHES_MULTI_HEAD,
        applicable=True,
        passed=check.passed,
        evidence=(f"最大差 {gap:.3e}（容差 {SINGLE_HEAD_TOLERANCE:.0e}）",),
        cross_check=check,
    )


def check_pre_norm_block_matches_day079(
    block_params: BlockParameters,
    attention_params: AttentionParams,
    inputs: Matrix,
    shape: SourceShape,
    *,
    activation: str = "relu",
) -> PropertyOutcome:
    """pre 摆放的本包块与 day079 的 ``encoder_block`` 一致（跨天对账）.

    两边的六个阶段逐字相同，因此这条判据把"本课从 HF 读出来的接线"
    与"本课程自己写过的那个块"钉在一起：**如果两边的块不一样，
    那一定有一个人在某个位置多放或少放了一次 LN**。
    """
    if shape.heads != 1:
        return PropertyOutcome(
            name=PROPERTY_PRE_NORM_BLOCK_MATCHES_DAY079,
            applicable=False,
            passed=False,
            evidence=(f"这一次 heads={shape.heads}：day079 的块只接受单头注意力账",),
        )
    mine = gpt2_block(
        block_params,
        attention_params,
        inputs,
        shape,
        activation=activation,
        causal=False,
    )
    day079_attention = block_attention(
        block_params, inputs, attention_params, placement="pre", causal=False
    )
    reference = encoder_block(
        block_params, inputs, day079_attention, placement="pre", activation=activation
    )
    gap = max_abs_gap(mine.output, reference.output)
    check = CrossCheck(
        name=f"gpt2_block vs encoder_block（pre，{activation}）",
        left="hf_source.blocks",
        right="encoder_decoder.layers",
        max_gap=gap,
        tolerance=BLOCK_TOLERANCE,
    )
    return PropertyOutcome(
        name=PROPERTY_PRE_NORM_BLOCK_MATCHES_DAY079,
        applicable=True,
        passed=check.passed,
        evidence=(f"最大差 {gap:.3e}（容差 {BLOCK_TOLERANCE:.0e}）",),
        cross_check=check,
    )


def check_all(
    params: AttentionParams,
    inputs: Matrix,
    shape: SourceShape,
    *,
    block_params: BlockParameters | None = None,
    heads_for_round_trip: int = 2,
    activation: str = "relu",
) -> PropertyReport:
    """一次跑完七条性质（顺序与 :data:`types.SOURCE_PROPERTIES` 一致）.

    ``block_params=None`` 时**少一条**（那条跨天对账需要 day079 的块参数）。
    本函数因此校验"结论集合必须是名单的子集，且那一条恰好按参数给没给出现"——
    名单对不上时，报告里那七行会安静地少一行或多一行。
    """
    forward = hf_attention(params, inputs, shape, causal=shape.causal_default)
    outcomes: list[PropertyOutcome] = [
        check_rows_are_distributions(forward),
        check_masked_entries_are_exact_zero(forward),
        check_causal_prefix_is_stable(params, inputs, shape),
        check_split_merge_round_trip(inputs, heads_for_round_trip),
        check_fused_matches_separate(params, inputs),
        check_single_head_matches_multi_head(params, inputs, shape),
    ]
    if block_params is not None:
        outcomes.append(
            check_pre_norm_block_matches_day079(
                block_params, params, inputs, shape, activation=activation
            )
        )
    names = {outcome.name for outcome in outcomes}
    extra = names - set(SOURCE_PROPERTIES)
    missing = set(SOURCE_PROPERTIES) - names
    expected_missing = (
        set() if block_params is not None else {PROPERTY_PRE_NORM_BLOCK_MATCHES_DAY079}
    )
    if extra or missing != expected_missing:
        raise ShapeError(
            "性质名单与 types.SOURCE_PROPERTIES 不一致："
            f"多 {sorted(extra)}、缺 {sorted(missing - expected_missing)}。"
            "名单对不上时，报告里那七行会安静地少一行或多一行。"
        )
    return PropertyReport(outcomes=tuple(outcomes))


def default_inputs(tokens: int, hidden: int, *, step: float = 0.2, drift: float = 0.1) -> Matrix:
    """写死的非平凡输入（**每一行必须有非零方差**，否则 LN 会退化）.

    与 day079 的样本同源：全零行会让 softmax 给出均匀分布、也让 LayerNorm
    的分母退化——那时"看起来正常"的读数是假象。
    """
    if tokens <= 0 or hidden <= 0:
        raise ShapeError(f"tokens 与 hidden 必须为正，收到 {tokens} / {hidden}。")
    return tuple(
        tuple(step * (row + 1) + drift * (column + 1) for column in range(hidden))
        for row in range(tokens)
    )


def scale_agreement_line(head_dim: int) -> str:
    """三处缩放系数的对账（本包 / day076 / day075）.

    三者必须**逐位**相同——它们由同一个 ``1/√head_dim`` 派生。
    这条读数存在的理由与 day077 第 4.2 节逐字相同：
    **一个量只要被写了两遍，就必须有一条断言让它无法分家。**
    """
    from smart_research_agent.hf_source.attention import hf_scale
    from smart_research_agent.multi_head.layers import head_scale
    from smart_research_agent.transformer_core.types import softmax_shape_scale

    values = (hf_scale(head_dim), head_scale(head_dim), softmax_shape_scale(head_dim))
    same = values[0] == values[1] == values[2]
    return (
        f"head_dim={head_dim}：hf_scale={values[0]!r} / head_scale={values[1]!r} / "
        f"softmax_shape_scale={values[2]!r} | 逐位相同：{same}"
    )


__all__ = [
    "PERTURBATION",
    "ROW_SUM_TOLERANCE",
    "CrossCheck",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_causal_prefix_is_stable",
    "check_fused_matches_separate",
    "check_masked_entries_are_exact_zero",
    "check_pre_norm_block_matches_day079",
    "check_rows_are_distributions",
    "check_single_head_matches_multi_head",
    "check_split_merge_round_trip",
    "default_inputs",
    "scale_agreement_line",
]
