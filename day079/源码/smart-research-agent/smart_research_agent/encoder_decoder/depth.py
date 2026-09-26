"""``encoder_decoder`` 的**深度实验**：残差是梯度的那条高速路（day079 / M7-D4）.

## 问题

一个块能被堆叠，而“堆起来”之后会出什么事？答案在反向里：
每经过一个块，输入梯度都要穿过两个子层（``LN → 分支 → 加``）。
如果那里只有分支，梯度就每层被“分支的雅可比”乘一次；而残差在**每一次**乘法旁边
加了一条**恒等**的路：

```text
y = x + F(x)        ⇒    dx = dy + dF       ← 那个 dy 不带任何雅可比
y = F(x)            ⇒    dx = dF            ← 只剩雅可比连乘
```

于是实验只有一个量：**堆 N 层之后，最底层输入梯度的范数是多少**。

## 三个变体（一次只改一个旋钮）

```text
pre_residual    y = x + F(LN(x))    现代实现的主流
post_residual   y = LN(x + F(x))    原论文的写法
bare            y = F(LN(x))        把那一项 +x 直接关掉（同一条代码路径）
```

## 为什么这件事值得单独一章

它解释了一个**看起来微不足道的改动**为什么变成了现代实现的默认：
pre 与 post 的形状完全一样、参数量完全一样、前向的输出分布也差不多，
而它们的**输入梯度范数**在深堆叠下差出量级。

```text
判据      同一个初始化、同一个目标、同一个深度序列，只换摆放位置与残差开关
读数      ‖∂loss/∂x‖（最底层输入的梯度范数）与它相对第 1 层的比值
```

## 这一课**不回答**什么

```text
它回答      “梯度能不能传到最底层”（一个**数值**问题，与前向的损失无关）
它不回答    “这个堆叠能不能训好”——那要看损失曲线、学习率与数据，
            而“梯度大”也可能是爆炸而不是好事（第 14 章的总结里写了这条边界）
```
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from smart_research_agent.encoder_decoder.errors import NumericError, ParameterError
from smart_research_agent.encoder_decoder.layers import (
    encoder_block,
    encoder_block_backward,
    layer_norm,
)
from smart_research_agent.encoder_decoder.types import (
    ACTIVATION_RELU,
    NORM_POST,
    NORM_PRE,
    BlockForward,
    BlockParameters,
    BlockShape,
    FFNWeights,
    _checked_activation,
    _checked_placement,
)
from smart_research_agent.math_foundations.probability import uniforms
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.transformer_core.layers import mean_squared_error, mse_gradient, self_attention
from smart_research_agent.transformer_core.train import default_parameters
from smart_research_agent.transformer_core.types import AttentionForward, AttentionParams

#: 三个变体的名字.
VARIANT_PRE_RESIDUAL = "pre_residual"
VARIANT_POST_RESIDUAL = "post_residual"
VARIANT_BARE = "bare"

VARIANTS: tuple[str, ...] = (
    VARIANT_PRE_RESIDUAL,
    VARIANT_POST_RESIDUAL,
    VARIANT_BARE,
)

VARIANT_DESCRIPTIONS: dict[str, str] = {
    VARIANT_PRE_RESIDUAL: "y = x + F(LN(x))：残差在**最后**（现代实现的主流）",
    VARIANT_POST_RESIDUAL: "y = LN(x + F(x))：残差之后再过 LN（原论文的写法）",
    VARIANT_BARE: "y = F(LN(x))：**把 +x 关掉**（同一个摆放位置、同一条代码路径）",
}

#: 默认的深度序列（都是 2 的幂附近，便于看“每层衰减多少倍”）.
DEFAULT_DEPTHS: tuple[int, ...] = (1, 2, 3, 4, 6, 8)

#: 参数初始化的幅度（与 day075 的 ``DEFAULT_INIT_SCALE`` 同值）.
DEFAULT_INIT_SCALE = 0.25


def _random_matrix(rows: int, columns: int, *, scale: float, seed: int) -> Matrix:
    """确定性随机矩阵（``[-scale, scale)``，用 day073 那串 LCG 随机数）."""
    if rows < 1 or columns < 1:
        raise ParameterError(f"矩阵形状必须为正，收到 ({rows}, {columns})。")
    if not math.isfinite(scale) or scale <= 0:
        raise ParameterError(f"scale 必须是正的有限数，收到 {scale!r}。")
    raw = uniforms(rows * columns, seed=seed)
    out: list[Vector] = []
    cursor = 0
    for _ in range(rows):
        out.append(tuple((value * 2.0 - 1.0) * scale for value in raw[cursor : cursor + columns]))
        cursor += columns
    return tuple(out)


def make_block_parameters(
    shape: BlockShape,
    *,
    seed: int = 7,
    scale: float = DEFAULT_INIT_SCALE,
) -> BlockParameters:
    """一个块的初始参数（**两个 LN 的 γ/β 取恒等**，前馈取小随机数）.

    γ = 1、β = 0 是 LayerNorm 的自然初始点：它让“初始时 LN 就是标准化本身”，
    于是三个变体的差别只来自**摆放位置**与**残差开关**，而不是来自初始化的运气。
    """
    return BlockParameters(
        norm1_gamma=tuple(1.0 for _ in range(shape.hidden)),
        norm1_beta=tuple(0.0 for _ in range(shape.hidden)),
        ffn_w_in=_random_matrix(shape.ffn, shape.hidden, scale=scale, seed=seed),
        ffn_b_in=tuple(0.0 for _ in range(shape.ffn)),
        ffn_w_out=_random_matrix(shape.hidden, shape.ffn, scale=scale, seed=seed + 1),
        ffn_b_out=tuple(0.0 for _ in range(shape.hidden)),
        norm2_gamma=tuple(1.0 for _ in range(shape.hidden)),
        norm2_beta=tuple(0.0 for _ in range(shape.hidden)),
    )


def make_stack_parameters(
    shape: BlockShape,
    layers: int,
    *,
    seed: int = 7,
    scale: float = DEFAULT_INIT_SCALE,
) -> tuple[tuple[BlockParameters, ...], tuple[AttentionParams, ...]]:
    """一摞块各自独立的参数（每一层一个种子，避免“层与层完全一样”）."""
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ParameterError(f"layers 必须是 >= 1 的整数，收到 {layers!r}。")
    blocks = tuple(
        make_block_parameters(shape, seed=seed + index * 13, scale=scale)
        for index in range(layers)
    )
    attentions = tuple(
        default_parameters(shape.hidden, seed=seed + index * 17) for index in range(layers)
    )
    return blocks, attentions


def target_matrix(shape: BlockShape, *, seed: int = 23) -> Matrix:
    """固定的目标矩阵（三个变体共用同一个目标，读数的差别只能来自模型）."""
    return _random_matrix(shape.tokens, shape.hidden, scale=0.5, seed=seed)


def stack_forward(
    blocks: Sequence[BlockParameters],
    attentions: Sequence[AttentionParams],
    inputs: Matrix,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
) -> tuple[Matrix, tuple[tuple[BlockForward, AttentionForward, BlockParameters], ...]]:
    """前向穿过整摞块，并把**每一层的账**留下来（反向需要它们）.

    ``attn = self_attention(attn_params, LN(current))``（pre）或
    ``self_attention(attn_params, current)``（post）——这一行是**摆放位置**在堆叠里的样子。
    """
    resolved_placement = _checked_placement(placement)
    resolved_activation = _checked_activation(activation)
    if len(blocks) != len(attentions):
        raise ParameterError(
            f"块参数 {len(blocks)} 份与注意力参数 {len(attentions)} 份不一致。"
        )
    checked = validate_matrix(inputs, name="inputs")
    forwards: list[tuple[BlockForward, AttentionForward, BlockParameters]] = []
    current = checked
    for params, attn_params in zip(blocks, attentions, strict=True):
        if resolved_placement == NORM_PRE:
            normed, _cache = layer_norm(
                current, gamma=params.norm1_gamma, beta=params.norm1_beta
            )
        else:
            normed = current
        attention = self_attention(attn_params, normed, causal=False)
        forward = encoder_block(
            params,
            current,
            attention,
            placement=resolved_placement,
            use_residual=use_residual,
            activation=resolved_activation,
        )
        forwards.append((forward, attention, params))
        current = forward.output
    return current, tuple(forwards)


def stack_loss(
    blocks: Sequence[BlockParameters],
    attentions: Sequence[AttentionParams],
    inputs: Matrix,
    target: Matrix,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
) -> float:
    """整摞块上的 MSE（与 ``encoder_block`` 用同一个损失函数）."""
    output, _forwards = stack_forward(
        blocks,
        attentions,
        inputs,
        placement=placement,
        use_residual=use_residual,
        activation=activation,
    )
    return mean_squared_error(output, target)


def stack_input_gradient_norm(
    blocks: Sequence[BlockParameters],
    attentions: Sequence[AttentionParams],
    inputs: Matrix,
    target: Matrix,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
) -> tuple[float, Matrix]:
    """穿过整摞块反向，返回 ``(最底层输入梯度的 Frobenius 范数, 那个梯度矩阵)``.

    这一趟反向是这一课最长的链：每一层都要走
    ``输出 → 两个 LN 的反向 → 两条残差路 → 前馈的反向 → 注意力那层的 grad_inputs``，
    而**每一层的那条 +1 路**都会在这里出现一次。
    """
    resolved_placement = _checked_placement(placement)
    resolved_activation = _checked_activation(activation)
    output, forwards = stack_forward(
        blocks,
        attentions,
        inputs,
        placement=resolved_placement,
        use_residual=use_residual,
        activation=resolved_activation,
    )
    grad = mse_gradient(output, target)
    for forward, attention, params in reversed(forwards):
        grads = encoder_block_backward(
            forward,
            params,
            grad,
            activation=resolved_activation,
        )
        grad = grads.grad_inputs
    norm = math.sqrt(math.fsum(value * value for row in grad for value in row))
    return norm, grad


@dataclass(frozen=True)
class DepthRow:
    """一个变体在一个深度上的一行读数."""

    variant: str
    layers: int
    input_gradient_norm: float
    ratio_to_first: float
    loss: float

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ParameterError(f"未知的变体名 {self.variant!r}。")
        for name in ("input_gradient_norm", "ratio_to_first", "loss"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise NumericError(f"{name} 必须是有限数，收到 {value!r}。")

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "variant": self.variant,
            "layers": self.layers,
            "input_gradient_norm": self.input_gradient_norm,
            "ratio_to_first": self.ratio_to_first,
            "loss": self.loss,
        }

    def summary_line(self) -> str:
        """一行说明：``pre_residual 8 层 | ‖dx‖ 1.234567 | 相对第 1 层 1.02e+00 | 损失 0.123456``."""
        return (
            f"{self.variant:<14} {self.layers} 层 | ‖dx‖ {self.input_gradient_norm:.6f} | "
            f"相对第 1 层 {self.ratio_to_first:.2e} | 损失 {self.loss:.6f}"
        )


@dataclass(frozen=True)
class DepthStudy:
    """三个变体在一条深度序列上的对照（**这一课的值钱结论落在这里**）."""

    shape: BlockShape
    rows: tuple[DepthRow, ...]
    depths: tuple[int, ...]
    seed: int
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        resolved = tuple(self.rows)
        if not resolved:
            raise ParameterError("深度实验至少要有一行读数。")
        if {row.variant for row in resolved} != set(VARIANTS):
            raise NumericError(
                f"三个变体必须齐备：缺少的对照会让'残差有用'这句话失去参照物。"
            )
        object.__setattr__(self, "rows", resolved)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    def ratios(self, variant: str) -> tuple[float, ...]:
        """某个变体在各深度上的“相对第 1 层”比值（按下标顺序）."""
        return tuple(row.ratio_to_first for row in self.rows if row.variant == variant)

    @property
    def deepest(self) -> int:
        """最深的那一层."""
        return max(row.layers for row in self.rows)

    @property
    def verdict_ok(self) -> bool:
        """判决：最深那一层上，**关掉残差的变体**衰减得明显更快.

        判据是“快一个数量级以上”（`bare` 的比值 < `pre` 的 1/10）——
        这个距离足够远，因此它不会因为换一组随机种子而翻面。
        """
        bare = self.ratios(VARIANT_BARE)[-1]
        pre = self.ratios(VARIANT_PRE_RESIDUAL)[-1]
        return bare < pre / 10.0

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "shape": self.shape.to_dict(),
            "depths": list(self.depths),
            "seed": self.seed,
            "verdict_ok": self.verdict_ok,
            "rows": [row.to_dict() for row in self.rows],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``深度实验：1~8 层 | 最深一层 ‖dx‖ 比：pre 1.02e+00、post 3.1e-01、bare 4.2e-06``."""
        parts = []
        for variant in VARIANTS:
            parts.append(f"{variant.split('_')[0]} {self.ratios(variant)[-1]:.2e}")
        return (
            f"深度实验：{self.depths[0]}~{self.deepest} 层 | 最深一层 ‖dx‖ 比："
            + "、".join(parts)
        )

    def table_lines(self) -> tuple[str, ...]:
        """把三个变体排成一张表（每一行一个变体 × 深度）."""
        header = f"  {'变体':<14} | {'层数':>4} | {'‖dx‖':>12} | {'相对第 1 层':>11} | {'损失':>10}"
        lines = [header, "  " + "-" * 68]
        for row in self.rows:
            lines.append(
                f"  {row.variant:<14} | {row.layers:>4} | {row.input_gradient_norm:>12.6f} | "
                f"{row.ratio_to_first:>11.2e} | {row.loss:>10.6f}"
            )
        return tuple(lines)


def depth_study(
    shape: BlockShape,
    *,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    seed: int = 7,
    scale: float = DEFAULT_INIT_SCALE,
    activation: str = ACTIVATION_RELU,
) -> DepthStudy:
    """跑一遍深度实验：三个变体 × 一条深度序列.

    三个变体共用**同一个目标**与**同一批初始参数**（同一层的种子相同），
    因此表里的差别只来自两个旋钮：摆放位置、残差开关。
    """
    resolved_depths = tuple(depths)
    if not resolved_depths:
        raise ParameterError("深度序列不能为空。")
    if sorted(set(resolved_depths)) != list(resolved_depths):
        raise ParameterError(f"深度序列必须严格递增且不重复，收到 {resolved_depths}。")
    target = target_matrix(shape, seed=seed + 5)
    inputs = _random_matrix(shape.tokens, shape.hidden, scale=0.5, seed=seed + 9)
    rows: list[DepthRow] = []
    for variant in VARIANTS:
        placement, use_residual = _variant_settings(variant)
        first_norm: float | None = None
        for layers in resolved_depths:
            blocks, attentions = make_stack_parameters(
                shape, layers, seed=seed, scale=scale
            )
            norm, _grad = stack_input_gradient_norm(
                blocks,
                attentions,
                inputs,
                target,
                placement=placement,
                use_residual=use_residual,
                activation=activation,
            )
            loss = stack_loss(
                blocks,
                attentions,
                inputs,
                target,
                placement=placement,
                use_residual=use_residual,
                activation=activation,
            )
            if first_norm is None:
                first_norm = norm
            base = first_norm if first_norm > 0 else 1.0
            rows.append(
                DepthRow(
                    variant=variant,
                    layers=layers,
                    input_gradient_norm=norm,
                    ratio_to_first=norm / base,
                    loss=loss,
                )
            )
    return DepthStudy(
        shape=shape,
        rows=tuple(rows),
        depths=resolved_depths,
        seed=seed,
        notes=(
            "三个变体共用同一个目标与同一批初始参数，只差摆放位置与残差开关",
            "读数是最底层输入梯度的 Frobenius 范数（**不是损失**）：它回答'梯度能不能传到最底层'",
            "梯度大也可能是爆炸——这一条读数不承诺'训练一定更好'",
        ),
    )


def _variant_settings(variant: str) -> tuple[str, bool]:
    """变体名 → ``(placement, use_residual)``."""
    if variant == VARIANT_PRE_RESIDUAL:
        return NORM_PRE, True
    if variant == VARIANT_POST_RESIDUAL:
        return NORM_POST, True
    if variant == VARIANT_BARE:
        return NORM_PRE, False
    raise ParameterError(
        f"未知的变体名 {variant!r}：可选 {', '.join(VARIANTS)}——"
        "本包不为未知变体挑一个默认设置。"
    )


def block_parameter_count(shape: BlockShape, layers: int) -> int:
    """``layers`` 层块一共新增多少参数（前馈 + 三个 LN 的 γ/β）."""
    if isinstance(layers, bool) or not isinstance(layers, int) or layers < 1:
        raise ParameterError(f"layers 必须是 >= 1 的整数，收到 {layers!r}。")
    return layers * shape.parameter_count


def ffn_output_scale(weights: FFNWeights, scale: float) -> FFNWeights:
    """把前馈的**输出层**整体缩放（第 8.7 节的“残差值钱”那条性质用它）."""
    if not math.isfinite(scale):
        raise ParameterError(f"scale 必须是有限数，收到 {scale!r}。")
    return FFNWeights(
        w_in=weights.w_in,
        b_in=weights.b_in,
        w_out=tuple(tuple(value * scale for value in row) for row in weights.w_out),
        b_out=tuple(value * scale for value in weights.b_out),
    )


def matrix_frobenius(matrix: Matrix) -> float:
    """Frobenius 范数（读数用）."""
    checked = validate_matrix(matrix, name="matrix")
    return math.sqrt(math.fsum(value * value for row in checked for value in row))


__all__ = [
    "DEFAULT_DEPTHS",
    "DEFAULT_INIT_SCALE",
    "VARIANTS",
    "VARIANT_BARE",
    "VARIANT_DESCRIPTIONS",
    "VARIANT_POST_RESIDUAL",
    "VARIANT_PRE_RESIDUAL",
    "DepthRow",
    "DepthStudy",
    "block_parameter_count",
    "depth_study",
    "ffn_output_scale",
    "make_block_parameters",
    "make_stack_parameters",
    "matrix_frobenius",
    "stack_forward",
    "stack_input_gradient_norm",
    "stack_loss",
    "target_matrix",
]
