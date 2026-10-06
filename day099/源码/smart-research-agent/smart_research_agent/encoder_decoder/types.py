"""``encoder_decoder`` 的形状、口径表与记录（day079 / M7-D4）.

day075 交出一层可训练的注意力，day078 交出“把位置加进输入”的那一行。
今天把三样东西拼成一个**块**，而“块”是 Transformer 论文里唯一被复制的单位：

```text
          ┌──────────────── 一个 Encoder 块 ────────────────┐
x ──► LN ──► 注意力 ──► ⊕ ──► LN ──► 前馈 ──► ⊕ ──► y        ⊕ 是残差（跨层直连）
          └──────────────────────────────────────────────┘
```

## 一、三个子层，每一个都值得单独讲

```text
① 残差      y = x + F(x)     —— 它在反向里多了一条**不减值的路**（第 4 章）
② LayerNorm 逐 token 标准化  —— 它**逐行独立**，因此不破坏置换等变性（第 3 章）
③ 前馈      FFN(x) 逐位置    —— 它也是逐行的，因此同样不破坏置换等变性
```

后两条合起来给出一个跨天的对照：**真正打破置换等变性的只有位置编码与掩码**，
而 LN 与 FFN 都是“逐行”算子（day078 的第 4 条性质与今天第 4、5 条性质是同一句话）。

## 二、pre 与 post：LN 站在哪一边

```text
post-LN   y = LN(x + F(x))     原论文的写法
pre-LN    y = x + F(LN(x))     现代实现的主流
```

两者的**形状完全一样**，区别只在 LN 的位置。第 9 章会把这件事量出来：
同一个初始化下堆 8 层，post-LN 的输入梯度范数明显衰减，而 pre-LN 基本平稳。
**一个只差“子层顺序”的改动，效果差一个数量级**——这就是它值得单独成章的理由。

## 三、九项梯度校验（新增的八块 + 输入）

```text
norm1_gamma → norm1_beta → ffn_w_in → ffn_b_in → ffn_w_out → ffn_b_out
                         → norm2_gamma → norm2_beta → inputs
```

注意力那一层的四个投影**不在名单里**：它的反向在 day075 已经逐项验过，
本课把它当**给定函数**（用它的 ``grad_inputs`` 接进程），
于是这九项全部落在今天新增的三个子层上。

## 四、八条性质（四条“逐行”，两条“残差”，两条“交叉”）

```text
norm_rows_are_standardized          每一行均值 0、方差 1
norm_is_invariant_to_a_constant_shift   LN(x + c·1) == LN(x)
norm_is_equivariant_to_positive_scaling LN(λx) == LN(x)，λ > 0
norm_is_row_independent             换行序 → 输出**逐位**跟着换（与 BatchNorm 的对照）
feed_forward_is_position_wise       同上，**逐位**
residual_is_the_identity_when_the_branch_vanishes  分支为 0 时 y == x、dx == dy
residual_keeps_a_unit_path_in_the_gradient         反向里那条 **+1** 的路（第 4 章量它）
cross_attention_must_not_be_causal  交叉注意力**不加**因果掩码；加了在 n_tgt == n_src 时是静默错误
```
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from smart_research_agent.encoder_decoder.errors import (
    AssemblyError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.math_foundations.optim import flatten_matrices, unflatten_matrices
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
    validate_vector,
)

#: ``eps``：LayerNorm 分母里的那个小量（**PyTorch ``nn.LayerNorm`` 的默认值**）.
#:
#: 原论文的实现用的是 ``1e-6``；本课取 ``1e-5`` 并把它**记在记录里**——
#: 它是一次调用的判据，不是配置项（与 day073~078 同一条纪律）。
DEFAULT_EPSILON = 1e-5

#: 前馈的扩张比 ``d_ff / d_model``（原论文取 4）.
DEFAULT_FFN_RATIO = 4

#: LN 的两种摆放位置.
NORM_PRE = "pre"
NORM_POST = "post"
NORM_PLACEMENTS: tuple[str, ...] = (NORM_PRE, NORM_POST)

NORM_PLACEMENT_DESCRIPTIONS: dict[str, str] = {
    NORM_PRE: "pre-LN：y = x + F(LN(x))——LN 在子层**之前**，残差是最后一步（现代实现的主流）",
    NORM_POST: "post-LN：y = x + 的分支再 LN，即 y = LN(x + F(x))——原论文的写法",
}

#: 两种激活函数（都逐元素，因此都是“逐行”算子）.
ACTIVATION_RELU = "relu"
ACTIVATION_GELU = "gelu"
ACTIVATIONS: tuple[str, ...] = (ACTIVATION_RELU, ACTIVATION_GELU)

ACTIVATION_DESCRIPTIONS: dict[str, str] = {
    ACTIVATION_RELU: "ReLU：max(0, x)——在 0 处不可导（本包取次梯度 0），因此梯度是**分段常数**",
    ACTIVATION_GELU: "GELU：0.5x(1+erf(x/√2))——处处可导，且在 0 处的导数是 0.5",
}

#: 编码器块的六个阶段（**顺序就是数据流的顺序**；post-LN 的差别见 ``NORM_PLACEMENTS``）.
STAGE_NORM1 = "norm1"
STAGE_BRANCH1 = "branch1"
STAGE_ADD1 = "add1"
STAGE_NORM2 = "norm2"
STAGE_BRANCH2 = "branch2"
STAGE_ADD2 = "add2"

ENCODER_BLOCK_STAGES: tuple[str, ...] = (
    STAGE_NORM1,
    STAGE_BRANCH1,
    STAGE_ADD1,
    STAGE_NORM2,
    STAGE_BRANCH2,
    STAGE_ADD2,
)

STAGE_DESCRIPTIONS: dict[str, str] = {
    STAGE_NORM1: "第一个 LN：把每一行标准化（**逐行**，因此与其它行无关）",
    STAGE_BRANCH1: "第一个分支：本课是注意力那一层（day075 的实现，一行未改）",
    STAGE_ADD1: "第一个残差：y = x + 分支(x)——**跨层直连的那一条路从这里开始**",
    STAGE_NORM2: "第二个 LN：作用在第一个残差的输出上",
    STAGE_BRANCH2: "第二个分支：前馈网络（逐位置，先扩张 4 倍再压回来）",
    STAGE_ADD2: "第二个残差：块的输出",
}

STAGE_SHAPES: dict[str, str] = {
    STAGE_NORM1: "(n, d) → (n, d)（逐行标准化，形状不变）",
    STAGE_BRANCH1: "(n, d) × 四个投影 → (n, d)",
    STAGE_ADD1: "(n, d) + (n, d) → (n, d)",
    STAGE_NORM2: "(n, d) → (n, d)",
    STAGE_BRANCH2: "(n, d) → (n, 4d) → (n, d)",
    STAGE_ADD2: "(n, d) + (n, d) → (n, d)",
}

#: 解码器块比编码器块多一个**交叉注意力**子层，因此是九个阶段.
DECODER_BLOCK_STAGES: tuple[str, ...] = (
    "norm1",
    "self_branch",
    "add1",
    "norm2",
    "cross_branch",
    "add2",
    "norm3",
    "ffn_branch",
    "add3",
)

DECODER_STAGE_DESCRIPTIONS: dict[str, str] = {
    "norm1": "第一个 LN（作用在解码器自己的流上）",
    "self_branch": "因果自注意力：Q/K/V 都来自解码器流，**必须**加因果掩码",
    "add1": "第一个残差",
    "norm2": "第二个 LN",
    "cross_branch": "交叉注意力：Q 来自解码器、K/V 来自编码器输出——**绝不能加因果掩码**",
    "add2": "第二个残差",
    "norm3": "第三个 LN",
    "ffn_branch": "前馈（与编码器块同一个实现）",
    "add3": "第三个残差：解码器块的输出",
}

DECODER_STAGE_SHAPES: dict[str, str] = {
    "norm1": "(n_tgt, d) → (n_tgt, d)",
    "self_branch": "(n_tgt, d) × 四个投影 → (n_tgt, d)（权重是 (n_tgt, n_tgt) 的下三角）",
    "add1": "(n_tgt, d) + (n_tgt, d) → (n_tgt, d)",
    "norm2": "(n_tgt, d) → (n_tgt, d)",
    "cross_branch": "Q:(n_tgt, d) × K/V:(n_src, d) → (n_tgt, d)（**权重是 (n_tgt, n_src)**）",
    "add2": "(n_tgt, d) + (n_tgt, d) → (n_tgt, d)",
    "norm3": "(n_tgt, d) → (n_tgt, d)",
    "ffn_branch": "(n_tgt, d) → (n_tgt, 4d) → (n_tgt, d)",
    "add3": "(n_tgt, d) + (n_tgt, d) → (n_tgt, d)",
}

#: 九项梯度校验的名字（**顺序是压平时的顺序**）.
GRAD_NORM1_GAMMA = "norm1_gamma"
GRAD_NORM1_BETA = "norm1_beta"
GRAD_FFN_W_IN = "ffn_w_in"
GRAD_FFN_B_IN = "ffn_b_in"
GRAD_FFN_W_OUT = "ffn_w_out"
GRAD_FFN_B_OUT = "ffn_b_out"
GRAD_NORM2_GAMMA = "norm2_gamma"
GRAD_NORM2_BETA = "norm2_beta"
GRAD_INPUTS = "inputs"

BLOCK_GRADIENT_TARGETS: tuple[str, ...] = (
    GRAD_NORM1_GAMMA,
    GRAD_NORM1_BETA,
    GRAD_FFN_W_IN,
    GRAD_FFN_B_IN,
    GRAD_FFN_W_OUT,
    GRAD_FFN_B_OUT,
    GRAD_NORM2_GAMMA,
    GRAD_NORM2_BETA,
    GRAD_INPUTS,
)

BLOCK_GRADIENT_FORMULAS: dict[str, str] = {
    GRAD_NORM1_GAMMA: "dGamma = Σ_rows dOut_norm ⊙ x̂（**按行相加**：gamma 被所有行共用）",
    GRAD_NORM1_BETA: "dBeta = Σ_rows dOut_norm（同理）",
    GRAD_FFN_W_IN: "dW_in = (dPre ⊙ act')ᵀ · ln1_out；dPre 由 dOut 经 W_out 回传",
    GRAD_FFN_B_IN: "dB_in = Σ_rows (dPre ⊙ act')（偏置被所有行共用）",
    GRAD_FFN_W_OUT: "dW_out = dOutᵀ · hidden",
    GRAD_FFN_B_OUT: "dB_out = Σ_rows dOut",
    GRAD_NORM2_GAMMA: "同第一项，但 dOut_norm 来自**块输出**那一路",
    GRAD_NORM2_BETA: "同第二项",
    GRAD_INPUTS: "dx = dOut + dNorm1_inputs + dNorm2_inputs——**第一项就是残差那条 +1 的路**",
}

BLOCK_GRADIENT_DESCRIPTIONS: dict[str, str] = {
    GRAD_NORM1_GAMMA: "第一个 LN 的缩放：四项里唯一乘以 x̂ 的那一项",
    GRAD_NORM1_BETA: "第一个 LN 的平移：它**不经过**任何 x̂，因此是纯求和",
    GRAD_FFN_W_IN: "前馈的第一层：把 d 扩到 4d 的那一次",
    GRAD_FFN_B_IN: "前馈第一层的偏置",
    GRAD_FFN_W_OUT: "前馈的第二层：把 4d 压回 d",
    GRAD_FFN_B_OUT: "前馈第二层的偏置",
    GRAD_NORM2_GAMMA: "第二个 LN 的缩放",
    GRAD_NORM2_BETA: "第二个 LN 的平移",
    GRAD_INPUTS: "输入：**残差项 dOut 与两条 LN 链之和**——少掉 dOut 那一项不会报错，只会让梯度变小",
}

#: 交叉注意力的六项梯度（比自注意力多了一路：来源不同）.
GRAD_W_QUERY = "w_query"
GRAD_W_KEY = "w_key"
GRAD_W_VALUE = "w_value"
GRAD_W_OUTPUT = "w_output"
GRAD_TARGET_INPUTS = "target_inputs"
GRAD_SOURCE_INPUTS = "source_inputs"

CROSS_GRADIENT_TARGETS: tuple[str, ...] = (
    GRAD_W_QUERY,
    GRAD_W_KEY,
    GRAD_W_VALUE,
    GRAD_W_OUTPUT,
    GRAD_TARGET_INPUTS,
    GRAD_SOURCE_INPUTS,
)

CROSS_GRADIENT_FORMULAS: dict[str, str] = {
    GRAD_W_QUERY: "dW_q = dQᵀ · target（Q 只来自 target 那一侧）",
    GRAD_W_KEY: "dW_k = dKᵀ · source",
    GRAD_W_VALUE: "dW_v = dVᵀ · source",
    GRAD_W_OUTPUT: "dW_o = dOutᵀ · context",
    GRAD_TARGET_INPUTS: "dTarget = dQ · W_q（**只有一条链**）",
    GRAD_SOURCE_INPUTS: "dSource = dK · W_k + dV · W_v（**两条链之和**——少一条不报错）",
}

#: 八条性质的名字与前缀说明.
PROPERTY_ROWS_STANDARDIZED = "norm_rows_are_standardized"
PROPERTY_SHIFT_INVARIANT = "norm_is_invariant_to_a_constant_shift"
PROPERTY_SCALE_EQUIVARIANT = "norm_is_equivariant_to_positive_scaling"
PROPERTY_NORM_ROW_INDEPENDENT = "norm_is_row_independent"
PROPERTY_FFN_POSITION_WISE = "feed_forward_is_position_wise"
PROPERTY_RESIDUAL_IDENTITY = "residual_is_the_identity_when_the_branch_vanishes"
PROPERTY_RESIDUAL_UNIT_PATH = "residual_keeps_a_unit_path_in_the_gradient"
PROPERTY_CROSS_NOT_CAUSAL = "cross_attention_must_not_be_causal"

ENCODER_DECODER_PROPERTIES: tuple[str, ...] = (
    PROPERTY_ROWS_STANDARDIZED,
    PROPERTY_SHIFT_INVARIANT,
    PROPERTY_SCALE_EQUIVARIANT,
    PROPERTY_NORM_ROW_INDEPENDENT,
    PROPERTY_FFN_POSITION_WISE,
    PROPERTY_RESIDUAL_IDENTITY,
    PROPERTY_RESIDUAL_UNIT_PATH,
    PROPERTY_CROSS_NOT_CAUSAL,
)

PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_ROWS_STANDARDIZED: "LayerNorm 之后每一行的均值恰好为 0、方差恰好为 1（容差内）",
    PROPERTY_SHIFT_INVARIANT: "给一整行加同一个常数不改变输出——**平移不是信息**",
    PROPERTY_SCALE_EQUIVARIANT: "把一整行乘 λ > 0 不改变输出——**尺度也不是信息**",
    PROPERTY_NORM_ROW_INDEPENDENT: "换行序 → 输出**逐位**跟着换（每一行只依赖它自己）",
    PROPERTY_FFN_POSITION_WISE: "同上（前馈是逐位置的）——这两条都不破坏置换等变性",
    PROPERTY_RESIDUAL_IDENTITY: "分支为 0 时 y 与 x **逐位**相等、dx 与 dy **逐位**相等",
    PROPERTY_RESIDUAL_UNIT_PATH: "反向里存在一条**不随分支缩放**的 +1 路——深堆叠时它决定梯度能不能活",
    PROPERTY_CROSS_NOT_CAUSAL: "交叉注意力的权重是 (n_tgt, n_src)；误加因果掩码在 n_tgt == n_src 时**不报错**",
}

#: LayerNorm 与残差的四条纪律（供报告与教程引用）.
ENCODER_DECODER_NOTES: tuple[str, ...] = (
    "LayerNorm 是**逐行**的：它不跨行取统计量，因此不破坏置换等变性（BatchNorm 会）",
    "残差是**恒等映射加一个分支**：分支为 0 时块退化成恒等，而这一条可以用 == 断言",
    "pre 与 post 的差别只在 LN 的位置，形状完全一样——而深堆叠下梯度差一个数量级",
    "交叉注意力的 K/V 来自**另一路**：它的四个投影列数可以不同，照抄自注意力的参数结构会被拒",
)


def _checked_positive_int(value: Any, *, name: str) -> int:
    """校验“正整数”（非整数或 < 1 抛 ``ParameterError``）."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError(f"{name} 必须是整数，收到 {value!r}。")
    if value < 1:
        raise ParameterError(f"{name} 必须 >= 1，收到 {value}。")
    return value


def _checked_tolerance(value: Any, *, name: str = "tolerance") -> float:
    """校验容差是正的有限数."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"{name} 必须是数，收到 {value!r}。")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ParameterError(f"{name} 必须是正的有限数，收到 {value!r}。")
    return resolved


def _checked_placement(value: Any) -> str:
    """校验 LN 的摆放位置."""
    if value not in NORM_PLACEMENTS:
        raise AssemblyError(
            f"未知的 LN 摆放位置 {value!r}：可选 {', '.join(NORM_PLACEMENTS)}——"
            "pre 与 post 的形状完全一样，给错时**不会报任何形状错误**，"
            "因此必须在入口处拒绝。"
        )
    return str(value)


def _checked_activation(value: Any) -> str:
    """校验激活函数."""
    if value not in ACTIVATIONS:
        raise ParameterError(
            f"未知的激活函数 {value!r}：可选 {', '.join(ACTIVATIONS)}——"
            "本包不为未知激活挑一个默认值。"
        )
    return str(value)


@dataclass(frozen=True)
class BlockShape:
    """一个块的三个维度.

    ```text
    hidden   d     隐藏维（块内所有张量的宽度，也是四个投影的列数）
    ffn      4d    前馈的中间维（原论文取 4 倍）
    tokens   n     序列长度
    ```

    ``tokens`` 放进形状里而不是让每个函数各自去数，是为了让**“块不改变形状”**
    这件事变成一条可断言的性质：``encoder_block`` 的输入输出形状必须一致。
    """

    hidden: int
    ffn: int
    tokens: int

    def __post_init__(self) -> None:
        for name in ("hidden", "ffn", "tokens"):
            _checked_positive_int(getattr(self, name), name=name)

    @property
    def ffn_ratio(self) -> float:
        """``ffn / hidden``（原论文的取值是 4.0）."""
        return self.ffn / self.hidden

    @property
    def parameter_count(self) -> int:
        """块里**新增**的可训练参数量（两个 LN 的 γ/β + 前馈的四块）.

        注意力那一层的参数不计：它的四个投影在 day075 就已经在那里了，
        而“今天多了多少参数”是这一课该说清的事。
        """
        return 4 * self.hidden + 2 * self.hidden * self.ffn + self.ffn + self.hidden

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "hidden": self.hidden,
            "ffn": self.ffn,
            "tokens": self.tokens,
            "ffn_ratio": self.ffn_ratio,
            "parameter_count": self.parameter_count,
        }

    def summary_line(self) -> str:
        """一行说明：``n=4 d=6 d_ff=24（4.0×）| 本块新增参数 150 个``."""
        return (
            f"n={self.tokens} d={self.hidden} d_ff={self.ffn}"
            f"（{self.ffn_ratio:.1f}×）| 本块新增参数 {self.parameter_count} 个"
        )


@dataclass(frozen=True)
class NormCache:
    """一次 LayerNorm 前向留下的账（反向只需要它）.

    ```text
    inputs      输入（原样存下：反传 dGamma/dBeta 时要乘以它）
    mean        每一行的均值
    variance    每一行的**有偏**方差（除以 n，与 PyTorch 一致）
    normalized  x̂ = (x − mean) / √(variance + eps)
    gamma      缩放向量（None 表示全 1）
    epsilon     分母里的那个小量（记下来，否则报告里说不清“这次用的是多少”）
    ```
    """

    inputs: Matrix
    mean: Vector
    variance: Vector
    normalized: Matrix
    epsilon: float
    gamma: Vector | None = None

    def __post_init__(self) -> None:
        checked_inputs = validate_matrix(self.inputs, name="inputs")
        checked_normalized = validate_matrix(self.normalized, name="normalized")
        if matrix_shape(checked_inputs) != matrix_shape(checked_normalized):
            raise ShapeError(
                f"输入 {matrix_shape(checked_inputs)} 与标准化结果 "
                f"{matrix_shape(checked_normalized)} 形状不一致。"
            )
        rows, columns = matrix_shape(checked_inputs)
        if len(self.mean) != rows or len(self.variance) != rows:
            raise ShapeError(
                f"均值/方差各应有 {rows} 个（每一行一个），收到 {len(self.mean)} 与 "
                f"{len(self.variance)}。"
            )
        if self.gamma is not None and len(self.gamma) != columns:
            raise ShapeError(
                f"gamma 的长度 {len(self.gamma)} 与列数 {columns} 不一致："
                "LayerNorm 的 γ/β 是**逐维**的（不是逐行），因此长度等于隐藏维。"
            )
        _checked_tolerance(self.epsilon, name="epsilon")
        object.__setattr__(self, "inputs", checked_inputs)
        object.__setattr__(self, "normalized", checked_normalized)
        object.__setattr__(self, "mean", validate_vector(self.mean, name="mean"))
        object.__setattr__(self, "variance", validate_vector(self.variance, name="variance"))
        if self.gamma is not None:
            object.__setattr__(self, "gamma", validate_vector(self.gamma, name="gamma"))

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "mean": list(self.mean),
            "variance": list(self.variance),
            "epsilon": self.epsilon,
            "normalized": [list(row) for row in self.normalized],
            "gamma": list(self.gamma) if self.gamma is not None else None,
        }

    def summary_line(self) -> str:
        """一行说明：``4 行 | 均值最大偏离 0.00e+00 | 方差最大偏离 0.00e+00（eps=1e-05）``."""
        return (
            f"{len(self.mean)} 行 | 均值最大偏离 {max(abs(v) for v in self.mean):.2e} | "
            f"方差最大偏离 {max(abs(v - 1.0) for v in self.variance):.2e}"
            f"（eps={self.epsilon:g}）"
        )


@dataclass(frozen=True)
class NormGradients:
    """一次 LayerNorm 反向的三块."""

    grad_gamma: Vector
    grad_beta: Vector
    grad_inputs: Matrix

    def __post_init__(self) -> None:
        object.__setattr__(self, "grad_gamma", validate_vector(self.grad_gamma, name="grad_gamma"))
        object.__setattr__(self, "grad_beta", validate_vector(self.grad_beta, name="grad_beta"))
        object.__setattr__(self, "grad_inputs", validate_matrix(self.grad_inputs, name="grad_inputs"))

    def summary_line(self) -> str:
        """一行说明（三块各自的最大绝对值）."""
        return (
            f"dGamma {max(abs(v) for v in self.grad_gamma):.6f} | "
            f"dBeta {max(abs(v) for v in self.grad_beta):.6f} | "
            f"dx {max(abs(v) for row in self.grad_inputs for v in row):.6f}"
        )


@dataclass(frozen=True)
class FFNWeights:
    """前馈的四块参数（两层线性 + 两个偏置，都带）.

    ```text
    w_in    (ffn, hidden)    w_out  (hidden, ffn)
    b_in    (ffn,)           b_out  (hidden,)
    ```
    """

    w_in: Matrix
    b_in: Vector
    w_out: Matrix
    b_out: Vector

    def __post_init__(self) -> None:
        checked_in = validate_matrix(self.w_in, name="w_in")
        checked_out = validate_matrix(self.w_out, name="w_out")
        hidden = matrix_shape(checked_out)[0]
        ffn = matrix_shape(checked_in)[0]
        if matrix_shape(checked_in)[1] != hidden:
            raise ShapeError(
                f"w_in 的列数 {matrix_shape(checked_in)[1]} 与 w_out 的行数 {hidden} "
                "不一致：前馈是 (hidden → ffn → hidden) 的一个往返。"
            )
        if matrix_shape(checked_out)[1] != ffn:
            raise ShapeError(
                f"w_out 的列数 {matrix_shape(checked_out)[1]} 与 w_in 的行数 {ffn} 不一致。"
            )
        if len(self.b_in) != ffn or len(self.b_out) != hidden:
            raise ShapeError(
                f"偏置长度不匹配：b_in 应为 {ffn}、b_out 应为 {hidden}，"
                f"收到 {len(self.b_in)} 与 {len(self.b_out)}。"
            )
        object.__setattr__(self, "w_in", checked_in)
        object.__setattr__(self, "w_out", checked_out)
        object.__setattr__(self, "b_in", validate_vector(self.b_in, name="b_in"))
        object.__setattr__(self, "b_out", validate_vector(self.b_out, name="b_out"))

    @property
    def parameter_count(self) -> int:
        """四块的元素个数之和."""
        return (
            matrix_shape(self.w_in)[0] * matrix_shape(self.w_in)[1]
            + len(self.b_in)
            + matrix_shape(self.w_out)[0] * matrix_shape(self.w_out)[1]
            + len(self.b_out)
        )

    def matrices(self) -> tuple[Matrix, ...]:
        """按“压平时的顺序”给出四块（两个矩阵 + 两个偏置各作一行）."""
        return (self.w_in, (self.b_in,), self.w_out, (self.b_out,))

    def summary_line(self) -> str:
        """一行说明."""
        return (
            f"w_in {matrix_shape(self.w_in)} | b_in {len(self.b_in)} | "
            f"w_out {matrix_shape(self.w_out)} | b_out {len(self.b_out)} | "
            f"{self.parameter_count} 个参数"
        )


@dataclass(frozen=True)
class FFNCache:
    """一次前馈前向留下的账."""

    inputs: Matrix
    pre_activation: Matrix
    hidden: Matrix
    weights: FFNWeights
    activation: str = ACTIVATION_RELU

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", validate_matrix(self.inputs, name="inputs"))
        object.__setattr__(self, "pre_activation", validate_matrix(self.pre_activation, name="pre_activation"))
        object.__setattr__(self, "hidden", validate_matrix(self.hidden, name="hidden"))
        if matrix_shape(self.pre_activation) != matrix_shape(self.hidden):
            raise ShapeError("pre_activation 与 hidden 的形状必须一致（激活是逐元素的）。")
        _checked_activation(self.activation)

    def summary_line(self) -> str:
        """一行说明：``4×6 → 4×24 → 4×6（relu，激活后零点占比 50.0%）``.

        零点占比是 ReLU 的一个便宜读数：它等于“这一批里有多少个神经元被关掉”，
        而关掉的神经元在反向里梯度**恰好是 0**（见第 6 章）。
        """
        total = 0
        zeros = 0
        for row in self.hidden:
            for value in row:
                total += 1
                if value == 0.0:
                    zeros += 1
        ratio = zeros / total if total else 0.0
        return (
            f"{matrix_shape(self.inputs)} → {matrix_shape(self.pre_activation)} → "
            f"{matrix_shape(self.hidden)}（{self.activation}，激活后零点占比 {ratio:.1%}）"
        )


@dataclass(frozen=True)
class FFNGradients:
    """一次前馈反向的五块."""

    grad_w_in: Matrix
    grad_b_in: Vector
    grad_w_out: Matrix
    grad_b_out: Vector
    grad_inputs: Matrix

    def __post_init__(self) -> None:
        object.__setattr__(self, "grad_w_in", validate_matrix(self.grad_w_in, name="grad_w_in"))
        object.__setattr__(self, "grad_w_out", validate_matrix(self.grad_w_out, name="grad_w_out"))
        object.__setattr__(self, "grad_b_in", validate_vector(self.grad_b_in, name="grad_b_in"))
        object.__setattr__(self, "grad_b_out", validate_vector(self.grad_b_out, name="grad_b_out"))
        object.__setattr__(self, "grad_inputs", validate_matrix(self.grad_inputs, name="grad_inputs"))

    def summary_line(self) -> str:
        """一行说明（五块各自的最大绝对值）."""
        return (
            f"dW_in {max(abs(v) for row in self.grad_w_in for v in row):.6f} | "
            f"db_in {max(abs(v) for v in self.grad_b_in):.6f} | "
            f"dW_out {max(abs(v) for row in self.grad_w_out for v in row):.6f} | "
            f"db_out {max(abs(v) for v in self.grad_b_out):.6f} | "
            f"dx {max(abs(v) for row in self.grad_inputs for v in row):.6f}"
        )


@dataclass(frozen=True)
class BlockParameters:
    """一个编码器块里**今天新增**的八块参数（两个 LN 的 γ/β + 前馈的四块）.

    注意力那一层的四个投影**不在这里**：它是 day075 的参数，
    本课把它当给定函数（见模块顶部的说明）。
    """

    norm1_gamma: Vector
    norm1_beta: Vector
    ffn_w_in: Matrix
    ffn_b_in: Vector
    ffn_w_out: Matrix
    ffn_b_out: Vector
    norm2_gamma: Vector
    norm2_beta: Vector

    def __post_init__(self) -> None:
        object.__setattr__(self, "norm1_gamma", validate_vector(self.norm1_gamma, name="norm1_gamma"))
        object.__setattr__(self, "norm2_gamma", validate_vector(self.norm2_gamma, name="norm2_gamma"))
        if len(self.norm1_gamma) != len(self.norm2_gamma):
            raise ShapeError(
                f"两个 LN 的 gamma 长度必须相同（同宽）：{len(self.norm1_gamma)} 与 "
                f"{len(self.norm2_gamma)}。"
            )
        object.__setattr__(self, "norm1_beta", validate_vector(self.norm1_beta, name="norm1_beta"))
        object.__setattr__(self, "norm2_beta", validate_vector(self.norm2_beta, name="norm2_beta"))
        # 借用 FFNWeights 的形状校验：前馈的形状契约只有一处实现
        FFNWeights(w_in=self.ffn_w_in, b_in=self.ffn_b_in, w_out=self.ffn_w_out, b_out=self.ffn_b_out)

    @property
    def ffn(self) -> FFNWeights:
        """把四块前馈参数打包（形状校验已经在 ``__post_init__`` 里跑过）."""
        return FFNWeights(
            w_in=self.ffn_w_in, b_in=self.ffn_b_in, w_out=self.ffn_w_out, b_out=self.ffn_b_out
        )

    @property
    def hidden(self) -> int:
        """隐藏维（= gamma 的长度）."""
        return len(self.norm1_gamma)

    @property
    def parameter_count(self) -> int:
        """八块的元素个数之和."""
        vectors = (
            self.norm1_gamma,
            self.norm1_beta,
            self.ffn_b_in,
            self.ffn_b_out,
            self.norm2_gamma,
            self.norm2_beta,
        )
        matrices = (self.ffn_w_in, self.ffn_w_out)
        return sum(len(v) for v in vectors) + sum(
            matrix_shape(m)[0] * matrix_shape(m)[1] for m in matrices
        )

    def matrices(self) -> tuple[Matrix, ...]:
        """按“压平时的顺序”给出八块（偏置各作一行矩阵）."""
        return (
            (self.norm1_gamma,),
            (self.norm1_beta,),
            self.ffn_w_in,
            (self.ffn_b_in,),
            self.ffn_w_out,
            (self.ffn_b_out,),
            (self.norm2_gamma,),
            (self.norm2_beta,),
        )

    def flatten(self) -> tuple[Vector, tuple[tuple[int, int], ...]]:
        """压平成一串数（**顺序与 ``BLOCK_GRADIENT_TARGETS`` 逐项对齐**）."""
        return flatten_matrices(self.matrices())

    @classmethod
    def unflatten(
        cls,
        flat: Vector,
        shapes: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    ) -> BlockParameters:
        """把一串数还原成八块（形状对不上时当场报错）."""
        parts = unflatten_matrices(flat, shapes)
        if len(parts) != 8:
            raise ShapeError(f"参数块个数 {len(parts)} 与预期的 8 个不一致。")
        return cls(
            norm1_gamma=parts[0][0],
            norm1_beta=parts[1][0],
            ffn_w_in=parts[2],
            ffn_b_in=parts[3][0],
            ffn_w_out=parts[4],
            ffn_b_out=parts[5][0],
            norm2_gamma=parts[6][0],
            norm2_beta=parts[7][0],
        )

    def with_ffn(self, weights: FFNWeights) -> BlockParameters:
        """换一组前馈参数（其余不动）."""
        return replace(
            self,
            ffn_w_in=weights.w_in,
            ffn_b_in=weights.b_in,
            ffn_w_out=weights.w_out,
            ffn_b_out=weights.b_out,
        )

    def summary_line(self) -> str:
        """一行说明."""
        return (
            f"d={self.hidden} d_ff={matrix_shape(self.ffn_w_in)[0]} | "
            f"γ/β 各 2 组 | {self.parameter_count} 个参数"
        )


@dataclass(frozen=True)
class BlockForward:
    """一次编码器块前向的账（六个阶段逐项留下）."""

    shape: BlockShape
    placement: str
    use_residual: bool
    inputs: Matrix
    attention: Any
    norm1: Matrix
    branch1: Matrix
    residual1: Matrix
    norm2: Matrix
    branch2: Matrix
    output: Matrix
    attention_layers: int = 1
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        _checked_placement(self.placement)
        for name in (
            "inputs",
            "norm1",
            "branch1",
            "residual1",
            "norm2",
            "branch2",
            "output",
        ):
            object.__setattr__(self, name, validate_matrix(getattr(self, name), name=name))
        if matrix_shape(self.inputs) != matrix_shape(self.output):
            raise ShapeError(
                f"块的输入 {matrix_shape(self.inputs)} 与输出 {matrix_shape(self.output)} "
                "形状必须一致：**块不改变形状**（否则它没法堆叠）。"
            )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def tokens(self) -> int:
        """行数（序列长度）."""
        return matrix_shape(self.inputs)[0]

    def summary_line(self) -> str:
        """一行说明：``pre-LN 残差 开 | n=4 d=6 | 输出与输入形状一致``."""
        residual = "开" if self.use_residual else "**关**"
        return (
            f"{self.placement}-LN 残差 {residual} | n={self.tokens} d={self.shape.hidden}"
            f" | d_ff={self.shape.ffn}"
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "shape": self.shape.to_dict(),
            "placement": self.placement,
            "use_residual": self.use_residual,
            "inputs": [list(row) for row in self.inputs],
            "norm1": [list(row) for row in self.norm1],
            "branch1": [list(row) for row in self.branch1],
            "residual1": [list(row) for row in self.residual1],
            "norm2": [list(row) for row in self.norm2],
            "branch2": [list(row) for row in self.branch2],
            "output": [list(row) for row in self.output],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class BlockGradients:
    """一次编码器块反向的账：**九块**（八块参数 + 输入）."""

    grad_norm1_gamma: Vector
    grad_norm1_beta: Vector
    grad_ffn_w_in: Matrix
    grad_ffn_b_in: Vector
    grad_ffn_w_out: Matrix
    grad_ffn_b_out: Vector
    grad_norm2_gamma: Vector
    grad_norm2_beta: Vector
    grad_inputs: Matrix

    def __post_init__(self) -> None:
        for name in (
            "grad_norm1_gamma",
            "grad_norm1_beta",
            "grad_ffn_b_in",
            "grad_ffn_b_out",
            "grad_norm2_gamma",
            "grad_norm2_beta",
        ):
            object.__setattr__(self, name, validate_vector(getattr(self, name), name=name))
        for name in ("grad_ffn_w_in", "grad_ffn_w_out", "grad_inputs"):
            object.__setattr__(self, name, validate_matrix(getattr(self, name), name=name))

    def matrices(self) -> tuple[Matrix, ...]:
        """按“压平时的顺序”给出九块（**顺序与 ``BLOCK_GRADIENT_TARGETS`` 对齐**）."""
        return (
            (self.grad_norm1_gamma,),
            (self.grad_norm1_beta,),
            self.grad_ffn_w_in,
            (self.grad_ffn_b_in,),
            self.grad_ffn_w_out,
            (self.grad_ffn_b_out,),
            (self.grad_norm2_gamma,),
            (self.grad_norm2_beta,),
            self.grad_inputs,
        )

    def flatten(self) -> Vector:
        """压平成一串数."""
        flat: list[float] = []
        for matrix in self.matrices():
            for row in matrix:
                flat.extend(row)
        return tuple(flat)

    def as_dict(self) -> dict[str, Matrix]:
        """按名字给出九块（键与 ``BLOCK_GRADIENT_TARGETS`` 一致）."""
        return {
            GRAD_NORM1_GAMMA: (self.grad_norm1_gamma,),
            GRAD_NORM1_BETA: (self.grad_norm1_beta,),
            GRAD_FFN_W_IN: self.grad_ffn_w_in,
            GRAD_FFN_B_IN: (self.grad_ffn_b_in,),
            GRAD_FFN_W_OUT: self.grad_ffn_w_out,
            GRAD_FFN_B_OUT: (self.grad_ffn_b_out,),
            GRAD_NORM2_GAMMA: (self.grad_norm2_gamma,),
            GRAD_NORM2_BETA: (self.grad_norm2_beta,),
            GRAD_INPUTS: self.grad_inputs,
        }

    def summary_line(self) -> str:
        """一行说明（九块各自的最大绝对值）."""
        names = (
            ("dγ1", (self.grad_norm1_gamma,)),
            ("dβ1", (self.grad_norm1_beta,)),
            ("dW_in", self.grad_ffn_w_in),
            ("db_in", (self.grad_ffn_b_in,)),
            ("dW_out", self.grad_ffn_w_out),
            ("db_out", (self.grad_ffn_b_out,)),
            ("dγ2", (self.grad_norm2_gamma,)),
            ("dβ2", (self.grad_norm2_beta,)),
            ("dx", self.grad_inputs),
        )
        parts = []
        for label, matrix in names:
            largest = 0.0
            for row in matrix:
                for value in row:
                    largest = max(largest, abs(value))
            parts.append(f"{label} {largest:.6f}")
        return " | ".join(parts)


@dataclass(frozen=True)
class CrossShape:
    """交叉注意力的形状：**两路**，因此比自注意力多两个维度."""

    targets: int
    sources: int
    dimension: int

    def __post_init__(self) -> None:
        for name in ("targets", "sources", "dimension"):
            _checked_positive_int(getattr(self, name), name=name)

    @property
    def weights_shape(self) -> tuple[int, int]:
        """权重表的形状 ``(n_tgt, n_src)``——**它是长方形的**（只要两路长度不同）."""
        return (self.targets, self.sources)

    def summary_line(self) -> str:
        """一行说明：``Q 4×6 | K/V 6×6 | 权重 4×6``."""
        return (
            f"Q {self.targets}×{self.dimension} | K/V {self.sources}×{self.dimension} | "
            f"权重 {self.weights_shape[0]}×{self.weights_shape[1]}"
        )


@dataclass(frozen=True)
class CrossParameters:
    """交叉注意力的四个投影（**列数可以不同**：K/V 作用在另一路上）.

    ```text
    w_query   (d_k, d_tgt)      w_key   (d_k, d_src)   ← 两者列数不同，这是与自注意力的关键差别
    w_value   (d_v, d_src)      w_output (d_out, d_v)
    ```
    """

    w_query: Matrix
    w_key: Matrix
    w_value: Matrix
    w_output: Matrix

    def __post_init__(self) -> None:
        checked_q = validate_matrix(self.w_query, name="w_query")
        checked_k = validate_matrix(self.w_key, name="w_key")
        checked_v = validate_matrix(self.w_value, name="w_value")
        checked_o = validate_matrix(self.w_output, name="w_output")
        if matrix_shape(checked_q)[0] != matrix_shape(checked_k)[0]:
            raise ShapeError(
                f"W_q 的行数 {matrix_shape(checked_q)[0]} 与 W_k 的行数 "
                f"{matrix_shape(checked_k)[0]} 必须相同（打分是同一个空间里的点积）。"
            )
        if matrix_shape(checked_v)[0] != matrix_shape(checked_o)[1]:
            raise ShapeError(
                f"W_v 的行数 {matrix_shape(checked_v)[0]} 与 W_o 的列数 "
                f"{matrix_shape(checked_o)[1]} 必须相同。"
            )
        if matrix_shape(checked_v)[1] != matrix_shape(checked_k)[1]:
            raise AssemblyError(
                f"W_v 的列数 {matrix_shape(checked_v)[1]} 与 W_k 的列数 "
                f"{matrix_shape(checked_k)[1]} 必须相同："
                "交叉注意力的 K 与 V **都来自 source 那一侧**——"
                "一个作用在 target 上的 K/V 会让'看哪些源位置'与'取哪些源内容'指向两拨数据。"
            )
        object.__setattr__(self, "w_query", checked_q)
        object.__setattr__(self, "w_key", checked_k)
        object.__setattr__(self, "w_value", checked_v)
        object.__setattr__(self, "w_output", checked_o)

    @property
    def parameter_count(self) -> int:
        """四个投影的元素个数之和（不含偏置——与 day075 同口径）."""
        total = 0
        for matrix in (self.w_query, self.w_key, self.w_value, self.w_output):
            rows, columns = matrix_shape(matrix)
            total += rows * columns
        return total

    def matrices(self) -> tuple[Matrix, ...]:
        """按“压平时的顺序”给出四块."""
        return (self.w_query, self.w_key, self.w_value, self.w_output)

    def summary_line(self) -> str:
        """一行说明."""
        return (
            f"W_q {matrix_shape(self.w_query)} | W_k {matrix_shape(self.w_key)} | "
            f"W_v {matrix_shape(self.w_value)} | W_o {matrix_shape(self.w_output)} | "
            f"{self.parameter_count} 个参数（无偏置）"
        )


@dataclass(frozen=True)
class CrossForward:
    """一次交叉注意力前向的账."""

    shape: CrossShape
    params: CrossParameters
    target_inputs: Matrix
    source_inputs: Matrix
    queries: Matrix
    keys: Matrix
    values: Matrix
    scores: Matrix
    weights: Matrix
    context: Matrix
    output: Matrix
    scale: float
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        for name in (
            "target_inputs",
            "source_inputs",
            "queries",
            "keys",
            "values",
            "scores",
            "weights",
            "context",
            "output",
        ):
            object.__setattr__(self, name, validate_matrix(getattr(self, name), name=name))
        if matrix_shape(self.weights) != self.shape.weights_shape:
            raise ShapeError(
                f"权重形状 {matrix_shape(self.weights)} 与交叉注意力的期望 "
                f"{self.shape.weights_shape} 不一致：**权重是 (n_tgt, n_src)**，"
                "它是长方形的，而不是方阵。"
            )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def row_sums(self) -> Vector:
        """每一行权重之和（应恰好为 1）."""
        return tuple(math.fsum(row) for row in self.weights)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "shape": self.shape.summary_line(),
            "scale": self.scale,
            "weights": [list(row) for row in self.weights],
            "output": [list(row) for row in self.output],
            "row_sums": list(self.row_sums),
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``Q 4×6 ← K/V 6×6 | 权重 4×6 | 每行和 1.000000``."""
        worst = max(abs(value - 1.0) for value in self.row_sums)
        return (
            f"Q {self.shape.targets}×{self.shape.dimension} ← K/V "
            f"{self.shape.sources}×{self.shape.dimension} | 权重 "
            f"{self.shape.weights_shape[0]}×{self.shape.weights_shape[1]} | "
            f"每行和最大偏离 {worst:.2e}"
        )


@dataclass(frozen=True)
class CrossGradients:
    """一次交叉注意力反向的六块（**两路各有一块**）."""

    grad_w_query: Matrix
    grad_w_key: Matrix
    grad_w_value: Matrix
    grad_w_output: Matrix
    grad_target_inputs: Matrix
    grad_source_inputs: Matrix

    def __post_init__(self) -> None:
        for name in (
            "grad_w_query",
            "grad_w_key",
            "grad_w_value",
            "grad_w_output",
            "grad_target_inputs",
            "grad_source_inputs",
        ):
            object.__setattr__(self, name, validate_matrix(getattr(self, name), name=name))

    def matrices(self) -> tuple[Matrix, ...]:
        """按“压平时的顺序”给出六块（**顺序与 ``CROSS_GRADIENT_TARGETS`` 对齐**）."""
        return (
            self.grad_w_query,
            self.grad_w_key,
            self.grad_w_value,
            self.grad_w_output,
            self.grad_target_inputs,
            self.grad_source_inputs,
        )

    def flatten(self) -> Vector:
        """压平成一串数."""
        flat: list[float] = []
        for matrix in self.matrices():
            for row in matrix:
                flat.extend(row)
        return tuple(flat)

    def as_dict(self) -> dict[str, Matrix]:
        """按名字给出六块."""
        return {
            GRAD_W_QUERY: self.grad_w_query,
            GRAD_W_KEY: self.grad_w_key,
            GRAD_W_VALUE: self.grad_w_value,
            GRAD_W_OUTPUT: self.grad_w_output,
            GRAD_TARGET_INPUTS: self.grad_target_inputs,
            GRAD_SOURCE_INPUTS: self.grad_source_inputs,
        }

    def summary_line(self) -> str:
        """一行说明（六块各自的最大绝对值）."""
        names = (
            ("dW_q", self.grad_w_query),
            ("dW_k", self.grad_w_key),
            ("dW_v", self.grad_w_value),
            ("dW_o", self.grad_w_output),
            ("dTarget", self.grad_target_inputs),
            ("dSource", self.grad_source_inputs),
        )
        parts = []
        for label, matrix in names:
            largest = 0.0
            for row in matrix:
                for value in row:
                    largest = max(largest, abs(value))
            parts.append(f"{label} {largest:.6f}")
        return " | ".join(parts)


@dataclass(frozen=True)
class DecoderGradients:
    """一次解码器块反向的账：**两路输入**各一块 + 三段子层各自的账.

    ```text
    grad_decoder_inputs   解码器自己的输入收到的梯度（三段链之和 + 三条残差路）
    grad_encoder_outputs  **编码器的输出收到的梯度**（经交叉注意力的 source 那一路）
    grad_self_params      day075 的四个投影（因果自注意力）
    grad_cross            CrossGradients（四个投影 + 两路输入）
    grad_ffn              FFNGradients（前馈五块）
    grad_norm_gammas      三个 LN 的 dγ
    grad_norm_betas       三个 LN 的 dβ
    ```

    ``grad_encoder_outputs`` 是本课的一个副产品：**“编码器只被训练一次、
    解码器只读它“这句话在反向里不成立**——编码器输出会收到一股梯度，
    而它是否会传回编码器的参数取决于调用方怎么用它（第 9 章的第 3 条读数）。
    """

    grad_decoder_inputs: Matrix
    grad_encoder_outputs: Matrix
    grad_self_params: Any
    grad_cross: CrossGradients
    grad_ffn: FFNGradients
    grad_norm_gammas: tuple[Vector, Vector, Vector]
    grad_norm_betas: tuple[Vector, Vector, Vector]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "grad_decoder_inputs", validate_matrix(self.grad_decoder_inputs, name="d_decoder")
        )
        object.__setattr__(
            self, "grad_encoder_outputs", validate_matrix(self.grad_encoder_outputs, name="d_encoder")
        )
        for name in ("grad_norm_gammas", "grad_norm_betas"):
            values = getattr(self, name)
            if len(values) != 3:
                raise ShapeError(
                    f"{name} 必须有 3 个（解码器块有三个 LN），收到 {len(values)}。"
                )
            for index, value in enumerate(values):
                validate_vector(value, name=f"{name}[{index}]")

    def summary_line(self) -> str:
        """一行说明（两路输入与三段子层的量级）."""
        decoder = max(abs(v) for row in self.grad_decoder_inputs for v in row)
        encoder = max(abs(v) for row in self.grad_encoder_outputs for v in row)
        return (
            f"dDecoder {decoder:.6f} | dEncoder {encoder:.6f} | "
            f"{self.grad_cross.summary_line()}"
        )


def matrix_max_absolute(matrix: Matrix) -> float:
    """矩阵里的最大绝对值（转发 day075 的实现：口径必须只有一处）."""
    from smart_research_agent.transformer_core.types import matrix_max_absolute as core

    return core(matrix)


def relative_matrix_error(approximate: Matrix, reference: Matrix) -> float:
    """两块同形矩阵的最大相对逐点误差（转发 day075 的实现）."""
    from smart_research_agent.transformer_core.types import relative_matrix_error as core

    return core(approximate, reference)


def validate_epsilon(epsilon: Any) -> float:
    """校验 ``eps`` 是正的有限数."""
    return _checked_tolerance(epsilon, name="epsilon")


__all__ = [
    "ACTIVATIONS",
    "ACTIVATION_DESCRIPTIONS",
    "ACTIVATION_GELU",
    "ACTIVATION_RELU",
    "BLOCK_GRADIENT_DESCRIPTIONS",
    "BLOCK_GRADIENT_FORMULAS",
    "BLOCK_GRADIENT_TARGETS",
    "CROSS_GRADIENT_FORMULAS",
    "CROSS_GRADIENT_TARGETS",
    "DECODER_BLOCK_STAGES",
    "DECODER_STAGE_DESCRIPTIONS",
    "DECODER_STAGE_SHAPES",
    "DEFAULT_EPSILON",
    "DEFAULT_FFN_RATIO",
    "ENCODER_BLOCK_STAGES",
    "ENCODER_DECODER_NOTES",
    "ENCODER_DECODER_PROPERTIES",
    "GRAD_FFN_B_IN",
    "GRAD_FFN_B_OUT",
    "GRAD_FFN_W_IN",
    "GRAD_FFN_W_OUT",
    "GRAD_INPUTS",
    "GRAD_NORM1_BETA",
    "GRAD_NORM1_GAMMA",
    "GRAD_NORM2_BETA",
    "GRAD_NORM2_GAMMA",
    "GRAD_SOURCE_INPUTS",
    "GRAD_TARGET_INPUTS",
    "GRAD_W_KEY",
    "GRAD_W_OUTPUT",
    "GRAD_W_QUERY",
    "GRAD_W_VALUE",
    "NORM_PLACEMENTS",
    "NORM_PLACEMENT_DESCRIPTIONS",
    "NORM_POST",
    "NORM_PRE",
    "PROPERTY_CROSS_NOT_CAUSAL",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_FFN_POSITION_WISE",
    "PROPERTY_NORM_ROW_INDEPENDENT",
    "PROPERTY_RESIDUAL_IDENTITY",
    "PROPERTY_RESIDUAL_UNIT_PATH",
    "PROPERTY_ROWS_STANDARDIZED",
    "PROPERTY_SCALE_EQUIVARIANT",
    "PROPERTY_SHIFT_INVARIANT",
    "STAGE_ADD1",
    "STAGE_ADD2",
    "STAGE_BRANCH1",
    "STAGE_BRANCH2",
    "STAGE_DESCRIPTIONS",
    "STAGE_NORM1",
    "STAGE_NORM2",
    "STAGE_SHAPES",
    "BlockForward",
    "BlockGradients",
    "BlockParameters",
    "BlockShape",
    "CrossForward",
    "CrossGradients",
    "CrossParameters",
    "CrossShape",
    "DecoderGradients",
    "FFNCache",
    "FFNGradients",
    "FFNWeights",
    "NormCache",
    "NormGradients",
    "matrix_max_absolute",
    "relative_matrix_error",
    "validate_epsilon",
]
