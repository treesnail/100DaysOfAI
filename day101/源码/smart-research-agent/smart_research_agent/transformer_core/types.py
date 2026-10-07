"""``transformer_core`` 的形状、口径表与四条记录（day075 / M7-D1）.

day073 的 ``math_foundations.attention`` 给了**没有参数的注意力**：
一堆纯函数，输入 Q/K/V、输出权重与加权平均。今天的差别只有一处，但它改变了整件事：

```text
day073   Q/K/V 是**给定的输入**          →  只能验证"公式算得对"
day075   Q/K/V 是**参数投影出来的**       →  可以问"权重该往哪调"（可训练）
```

于是这一层多出三样东西，它们都要有形状：

```text
AttentionShape      一次注意力的四个维度：输入 / 打分 / value / 输出
AttentionParams     四个投影矩阵（W_q / W_k / W_v / W_o）
AttentionForward    前向的全部中间量（反向传播要用，见第五章）
ParameterGradients  反向的全部结果（四块参数梯度 + 一块输入梯度）
```

## 口径表：三张表，逐键对齐

```text
ATTENTION_STAGES    七个阶段（投影 / 打分 / 缩放 / 掩码 / softmax / 加权 / 输出投影）
GRADIENT_TARGETS    五项梯度校验（四个参数矩阵 + 输入）
PROPERTY_CHECKS     四项性质（行和为 1 / 非负 / 因果无泄漏 / 置换等变）
```

少一个键**不会让任何测试变红**，只会让那一项在报告里失去"它能被复核"的部分——
与 day073 的四张表同一取向。

## 一个必须写下来的乘法口径：``(d_out, d_in)``

本包一律用与 ``nn.Linear`` 一致的约定：

```text
权重 W 的形状是 (d_out, d_in)
投影结果 = x · Wᵀ                     （x 的形状是 (n, d_in)）

于是   dW = gradᵀ · x     （形状 (d_out, d_in)）
       dx = grad · W      （形状 (n, d_in)）
```

两条式子是这一课**唯一**需要"逐元素推一遍"的地方，因此它们在
``layers.py`` 里各有一条独立的测试（用手算的小矩阵核对）。

另一个约定：**四个投影都不带偏置**。现代 Transformer 普遍省略注意力与 FFN 的偏置，
而省略之后有一条立刻可验证的性质——**全零输入得到全零输出**。
带偏置时它不成立（偏置会把 0 抬起来），而"全零进全零出"是这一层很多推理的基础。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.math_foundations.linalg import matmul, transpose
from smart_research_agent.math_foundations.optim import (
    flatten_matrices,
    unflatten_matrices,
)
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    is_finite,
    matrix_shape,
    row_sums,
    validate_matrix,
)
from smart_research_agent.transformer_core.errors import (
    NumericError,
    ParameterError,
    ShapeError,
    TransformerError,
)

#: 权重行和为 1 的判据容差（比逐位比较松一档：softmax 的归一化会累积舍入误差）.
DEFAULT_SUM_TOLERANCE = 1e-9

# --------------------------------------------------------------------------- #
# 闭合表 1：七个阶段
# --------------------------------------------------------------------------- #

STAGE_PROJECT = "project"
STAGE_SCORE = "score"
STAGE_SCALE = "scale"
STAGE_MASK = "mask"
STAGE_SOFTMAX = "softmax"
STAGE_MIX = "mix"
STAGE_OUTPUT = "output"

#: 七个阶段（顺序 = 一次前向的执行顺序，也是教程与手册里的排列）.
ATTENTION_STAGES: tuple[str, ...] = (
    STAGE_PROJECT,
    STAGE_SCORE,
    STAGE_SCALE,
    STAGE_MASK,
    STAGE_SOFTMAX,
    STAGE_MIX,
    STAGE_OUTPUT,
)

#: 每个阶段的一句话解释.
ATTENTION_STAGE_DESCRIPTIONS: dict[str, str] = {
    STAGE_PROJECT: "投影：用四个矩阵把输入变成 Q/K/V（这一步让注意力可训练）",
    STAGE_SCORE: "打分：raw = Q·Kᵀ，每一行是一个 query 对所有 key 的点积",
    STAGE_SCALE: "缩放：scores = raw / √d_k，把打分的方差拉回与维度无关",
    STAGE_MASK: "掩码：因果模式下把上三角（j > i）标成不允许看",
    STAGE_SOFTMAX: "softmax：按行归一化，被掩码的位置权重**恰好是 0.0**",
    STAGE_MIX: "加权：context = weights · V，用这行分布对 value 做加权平均",
    STAGE_OUTPUT: "输出投影：y = context · W_oᵀ，让模型自己决定怎么用混合结果",
}

#: 每个阶段的形状变化（报告里能读到"这一步之后形状是什么"）.
ATTENTION_STAGE_SHAPES: dict[str, str] = {
    STAGE_PROJECT: "(n, d_in) → Q/K/V 各 (n, d_k)/(n, d_k)/(n, d_v)",
    STAGE_SCORE: "(n, d_k) × (n, d_k)ᵀ → (n, n)",
    STAGE_SCALE: "(n, n) → (n, n)",
    STAGE_MASK: "(n, n) → (n, n)（掩码位置不参与 softmax 的分母）",
    STAGE_SOFTMAX: "(n, n) → (n, n)，每一行和为 1",
    STAGE_MIX: "(n, n) × (n, d_v) → (n, d_v)",
    STAGE_OUTPUT: "(n, d_v) × (d_out, d_v)ᵀ → (n, d_out)",
}

if not (
    set(ATTENTION_STAGES)
    == set(ATTENTION_STAGE_DESCRIPTIONS)
    == set(ATTENTION_STAGE_SHAPES)
):
    raise TransformerError(
        "注意力阶段的两张表不一致：ATTENTION_STAGES / ATTENTION_STAGE_DESCRIPTIONS / "
        "ATTENTION_STAGE_SHAPES 必须逐键对齐，否则某个阶段在报告里只有名字、"
        "没有解释也没有形状，而'缺一行'与'这一项没问题'读起来一样。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 2：五项梯度校验
# --------------------------------------------------------------------------- #

GRAD_W_QUERY = "w_query"
GRAD_W_KEY = "w_key"
GRAD_W_VALUE = "w_value"
GRAD_W_OUTPUT = "w_output"
GRAD_INPUTS = "inputs"

#: 五项（顺序 = 反向传播的执行顺序：输出投影 → value → 权重 → 打分 → 输入投影）.
GRADIENT_TARGETS: tuple[str, ...] = (
    GRAD_W_OUTPUT,
    GRAD_W_VALUE,
    GRAD_W_QUERY,
    GRAD_W_KEY,
    GRAD_INPUTS,
)

#: 每一项的解析梯度公式（**这一课的主角**：它必须与数值差分对上）.
GRADIENT_TARGET_FORMULAS: dict[str, str] = {
    GRAD_W_OUTPUT: "dW_o = (dContext)ᵀ · W_o 的反向：dW_o = dOutᵀ · context",
    GRAD_W_VALUE: "dV = weightsᵀ · dContext；dW_v = dVᵀ · x",
    GRAD_W_QUERY: "dScores ← softmax 反向；dQ = dRaw · K；dW_q = dQᵀ · x",
    GRAD_W_KEY: "dK = dRawᵀ · Q；dW_k = dKᵀ · x",
    GRAD_INPUTS: "dx = dQ·W_q + dK·W_k + dV·W_v + 反向传回的其余部分",
}

#: 每一项的说明（它回答"这一块梯度是怎么来的"）.
GRADIENT_TARGET_DESCRIPTIONS: dict[str, str] = {
    GRAD_W_OUTPUT: "输出投影：只有它直接吃损失对输出的梯度，因此它是反向的第一站",
    GRAD_W_VALUE: "value 投影：把'输出该混哪些位置'翻译成'value 该怎么调'",
    GRAD_W_QUERY: "query 投影：它决定'这一行想问什么'，因此它的梯度要穿过 softmax",
    GRAD_W_KEY: "key 投影：它决定'这一行有多容易被问中'，与 query 共享同一份软打分的梯度",
    GRAD_INPUTS: "输入：四条投影链的和（少了任何一条都不会报错，只会给出偏小的梯度）",
}

if not (
    set(GRADIENT_TARGETS)
    == set(GRADIENT_TARGET_FORMULAS)
    == set(GRADIENT_TARGET_DESCRIPTIONS)
):
    raise TransformerError(
        "梯度校验项的三张表不一致：GRADIENT_TARGETS / GRADIENT_TARGET_FORMULAS / "
        "GRADIENT_TARGET_DESCRIPTIONS 必须逐键对齐——少一个键的那一块梯度会静默地"
        "不被校验，而'没校验'与'校验通过'在报告里长得一样。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 3：四项性质
# --------------------------------------------------------------------------- #

PROPERTY_ROW_STOCHASTIC = "row_stochastic"
PROPERTY_NON_NEGATIVE = "non_negative"
PROPERTY_CAUSAL_NO_LEAK = "causal_no_leak"
PROPERTY_PERMUTATION_EQUIVARIANCE = "permutation_equivariance"

#: 四项性质（顺序 = 从"每一行的形状'到'整张表在换序下怎么变"）.
PROPERTY_CHECKS: tuple[str, ...] = (
    PROPERTY_ROW_STOCHASTIC,
    PROPERTY_NON_NEGATIVE,
    PROPERTY_CAUSAL_NO_LEAK,
    PROPERTY_PERMUTATION_EQUIVARIANCE,
)

#: 每条性质的一句话解释.
PROPERTY_CHECK_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_ROW_STOCHASTIC: "每一行权重之和为 1（它是一个条件分布，不是一组打分）",
    PROPERTY_NON_NEGATIVE: "每个权重 >= 0（负权重会让'加权平均'变成一个减法）",
    PROPERTY_CAUSAL_NO_LEAK: "因果模式下上三角（j > i）**恰好是 0.0**——位置 i 看不到未来",
    PROPERTY_PERMUTATION_EQUIVARIANCE: "无掩码时，把输入行换序，输出行按同样方式换序"
    "（注意力本身不知道顺序——这正是需要位置编码的原因）",
}

if set(PROPERTY_CHECKS) != set(PROPERTY_CHECK_DESCRIPTIONS):
    raise TransformerError(
        "性质的两张表不一致：PROPERTY_CHECKS / PROPERTY_CHECK_DESCRIPTIONS 必须逐键对齐。"
    )


# --------------------------------------------------------------------------- #
# 形状
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AttentionShape:
    """一次注意力的四个维度（**四个数字各自管什么，必须说清**）.

    ```text
    inputs     d_in    输入行宽（= 四个投影矩阵的列数）
    keys       d_k     打分维度（Q 与 K 的输出维；也是缩放的依据 √d_k）
    values     d_v     value 维度（= context 的宽度；决定"混合出来的东西有多宽"）
    outputs    d_out   最终输出维度（= W_o 的输出维）
    ```

    为什么要四个而不是一个 ``d``：**它们的来源不同**。
    ``d_k`` 只影响打分的尺度、``d_v`` 只影响混合结果的宽度、
    ``d_out`` 只影响最终输出的宽度——把三者混成一个 ``d`` 之后，
    "把 d_k 调大一点会不会更慢"这类问题就没有答案了（答案是：会，且只影响打分那一步）。
    """

    inputs: int
    keys: int
    values: int
    outputs: int

    def __post_init__(self) -> None:
        for name in ("inputs", "keys", "values", "outputs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ParameterError(f"{name} 必须是整数，收到 {value!r}。")
            if value < 1:
                raise ParameterError(
                    f"{name} 必须 >= 1，收到 {value}："
                    "零宽的向量上没有点积、没有分布、也没有加权平均。"
                )

    @property
    def scale(self) -> float:
        """缩放系数 ``1/√d_k``（**注意分母是 d_k，不是 d_in 或 d_v**）."""
        return 1.0 / math.sqrt(self.keys)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "inputs": self.inputs,
            "keys": self.keys,
            "values": self.values,
            "outputs": self.outputs,
            "scale": self.scale,
        }

    def summary_line(self) -> str:
        """一行说明：``d_in=6 d_k=6 d_v=6 d_out=6 | scale=0.408248``."""
        return (
            f"d_in={self.inputs} d_k={self.keys} d_v={self.values} d_out={self.outputs} | "
            f"scale={self.scale:.6f}"
        )


# --------------------------------------------------------------------------- #
# 参数、前向记录、梯度
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AttentionParams:
    """一层自注意力的全部参数：四个投影矩阵（**不带偏置**）.

    ```text
    w_query   (d_k,   d_in)
    w_key     (d_k,   d_in)      与 w_query 的行数必须相同（打分是同一个空间里的点积）
    w_value   (d_v,   d_in)
    w_output  (d_out, d_v)       列数必须等于 d_v（它消费的是 context）
    ```

    四个矩阵的**列数必须相同**（``d_in``）：它们都作用在同一个输入上。
    这四条约束加起来正好是"这个式子有定义"的条件，因此它们在 ``__post_init__`` 里
    一次性校验——而不是等到某个乘法把两个对不上的维度静默错配。
    """

    w_query: Matrix
    w_key: Matrix
    w_value: Matrix
    w_output: Matrix

    def __post_init__(self) -> None:
        checked = {
            "w_query": validate_matrix(self.w_query, name="w_query"),
            "w_key": validate_matrix(self.w_key, name="w_key"),
            "w_value": validate_matrix(self.w_value, name="w_value"),
            "w_output": validate_matrix(self.w_output, name="w_output"),
        }
        for name, matrix in checked.items():
            object.__setattr__(self, name, matrix)
        inputs = matrix_shape(checked["w_query"])[1]
        for name, matrix in checked.items():
            if matrix_shape(matrix)[1] != inputs:
                raise ShapeError(
                    f"{name} 的列数 {matrix_shape(matrix)[1]} 与 w_query 的列数 {inputs} "
                    "不一致：四个投影都作用在**同一个输入**上，列数就是输入的行宽。"
                )
        if matrix_shape(checked["w_key"])[0] != matrix_shape(checked["w_query"])[0]:
            raise ShapeError(
                f"w_key 的输出维 {matrix_shape(checked['w_key'])[0]} 与 w_query 的 "
                f"{matrix_shape(checked['w_query'])[0]} 不一致："
                "Q 与 K 必须在同一个空间里才能做点积（打分就是那个点积）。"
            )
        if matrix_shape(checked["w_output"])[1] != matrix_shape(checked["w_value"])[0]:
            raise ShapeError(
                f"w_output 的列数 {matrix_shape(checked['w_output'])[1]} 与 w_value 的"
                f"输出维 {matrix_shape(checked['w_value'])[0]} 不一致："
                "W_o 消费的是 context，而 context 的宽度就是 d_v。"
            )

    @property
    def shape(self) -> AttentionShape:
        """四个维度（从四个矩阵反推，**不重复声明**）."""
        return AttentionShape(
            inputs=matrix_shape(self.w_query)[1],
            keys=matrix_shape(self.w_query)[0],
            values=matrix_shape(self.w_value)[0],
            outputs=matrix_shape(self.w_output)[0],
        )

    def matrices(self) -> tuple[Matrix, ...]:
        """四个矩阵（顺序固定：q / k / v / o）——压平与还原都依赖这个顺序."""
        return (self.w_query, self.w_key, self.w_value, self.w_output)

    def flatten(self) -> tuple[Vector, tuple[tuple[int, int], ...]]:
        """压平成优化器能吃的向量，并交出形状表（day074 的 ``flatten_matrices``）.

        为什么优化器不需要知道参数是矩阵：更新公式 ``θ ← θ − lr·g`` 是**逐分量**的，
        与形状无关。压平之后 day074 的 ``AdamOptimizer`` 可以直接用在这一层上。
        """
        return flatten_matrices(self.matrices())

    @classmethod
    def unflatten(
        cls,
        flat: Vector,
        shapes: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    ) -> AttentionParams:
        """把一串数还原成四个矩阵（形状表由 :meth:`flatten` 给出）."""
        parts = unflatten_matrices(flat, shapes)
        if len(parts) != 4:
            raise ShapeError(
                f"还原出了 {len(parts)} 个矩阵，需要 4 个（q / k / v / o）："
                "形状表与参数布局必须一一对应，否则某一层的权重会被装到另一层上。"
            )
        return cls(w_query=parts[0], w_key=parts[1], w_value=parts[2], w_output=parts[3])

    def parameter_count(self) -> int:
        """参数量 ``d_k·d_in + d_k·d_in + d_v·d_in + d_out·d_v``（**手算可复核**）."""
        shape = self.shape
        return (
            shape.keys * shape.inputs * 2
            + shape.values * shape.inputs
            + shape.outputs * shape.values
        )

    def describe(self) -> str:
        """一行说明（进报告：报告里要能读出"这一层是什么形状的"）."""
        return (
            f"自注意力 {self.shape.summary_line()} | 参数 {self.parameter_count()} 个"
            f"（四个投影，无偏置）"
        )


@dataclass(frozen=True)
class AttentionForward:
    """一次前向的全部记录（**反向传播要用到的每一个中间量**）.

    ```text
    inputs      (n, d_in)   输入（反向时要用它算 dW = dQᵀ·x）
    queries     (n, d_k)    x · W_qᵀ
    keys        (n, d_k)    x · W_kᵀ
    values      (n, d_v)    x · W_vᵀ
    raw         (n, n)      Q·Kᵀ（**未缩放**：反向要乘回 scale）
    scores      (n, n)      raw · scale（softmax 的直接输入）
    weights     (n, n)      按行 softmax 之后（被掩码的位置**恰好 0.0**）
    context     (n, d_v)    weights · V
    output      (n, d_out)  context · W_oᵀ
    ```

    为什么要把 ``raw`` 与 ``scores`` 都留下：反向里只有一步要用它们
    （``dRaw = dScores · scale``），而"缩放系数是乘在打分上的"这件事
    一旦在反向里被漏掉，梯度会**整体差一个 √d_k 倍**——
    那不会报错，只会让训练"慢一点"。留下中间量是最便宜的护栏。
    """

    params: AttentionParams
    inputs: Matrix
    queries: Matrix
    keys: Matrix
    values: Matrix
    raw: Matrix
    scores: Matrix
    weights: Matrix
    context: Matrix
    output: Matrix
    causal: bool
    mask: tuple[tuple[bool, ...], ...]
    scale: float
    row_entropies: Vector
    peak_weights: Vector
    peak_indices: tuple[int, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        rows, columns = matrix_shape(self.weights)
        if rows != columns:
            raise ShapeError(
                f"自注意力的权重矩阵必须是方阵（自己看自己），收到 {matrix_shape(self.weights)}。"
            )
        if matrix_shape(self.output)[0] != rows:
            raise ShapeError("output 的行数必须与权重行数一致（一行输入一行输出）。")
        if len(self.mask) != rows or (rows and len(self.mask[0]) != columns):
            raise ShapeError(f"掩码形状与权重矩阵 {matrix_shape(self.weights)} 不一致。")
        if not (
            len(self.row_entropies) == len(self.peak_weights) == len(self.peak_indices) == rows
        ):
            raise ShapeError("row_entropies / peak_weights / peak_indices 的长度必须等于行数。")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def tokens(self) -> int:
        """行数（= 序列长度）."""
        return matrix_shape(self.weights)[0]

    @property
    def mean_entropy(self) -> float:
        """各行的平均熵（nats）：0 = 每行全押一个位置，ln(n) = 完全均匀."""
        if not self.row_entropies:
            return 0.0
        return math.fsum(self.row_entropies) / len(self.row_entropies)

    def max_entropy(self) -> float:
        """理论上界 ``ln(n)``（均匀分布时的熵）."""
        return math.log(self.tokens) if self.tokens > 0 else 0.0

    def focus_ratio(self) -> float:
        """集中度 ``1 − 平均熵/ln(n)``：0 = 完全均匀，1 = 每行只看一处."""
        ceiling = self.max_entropy()
        if ceiling <= 0:
            return 0.0
        return max(0.0, 1.0 - self.mean_entropy / ceiling)

    def summary_line(self) -> str:
        """一行说明：``causal | n=5 → d_out=6 | 平均熵 0.8123 | 集中度 49.5%``."""
        return (
            f"{'causal' if self.causal else 'full'} | n={self.tokens} → "
            f"d_out={matrix_shape(self.output)[1]} | 平均熵 {self.mean_entropy:.4f} | "
            f"集中度 {self.focus_ratio():.1%}"
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（含形状、缩放与两张派生量）."""
        return {
            "causal": self.causal,
            "tokens": self.tokens,
            "scale": self.scale,
            "shape": self.params.shape.to_dict(),
            "weights": [list(row) for row in self.weights],
            "output": [list(row) for row in self.output],
            "row_entropies": list(self.row_entropies),
            "peak_weights": list(self.peak_weights),
            "peak_indices": list(self.peak_indices),
            "mean_entropy": self.mean_entropy,
            "focus_ratio": self.focus_ratio(),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ParameterGradients:
    """一次反向的全部结果：四块参数梯度 + 一块输入梯度.

    顺序与 :meth:`AttentionParams.matrices` 一致（q / k / v / o），
    因此"压平参数'与'压平梯度"用的是同一张形状表——
    顺序错了不会报错，只会让某一层的权重按另一层的梯度更新。
    """

    grad_w_query: Matrix
    grad_w_key: Matrix
    grad_w_value: Matrix
    grad_w_output: Matrix
    grad_inputs: Matrix

    def matrices(self) -> tuple[Matrix, ...]:
        """四块参数梯度（顺序与参数一致）."""
        return (self.grad_w_query, self.grad_w_key, self.grad_w_value, self.grad_w_output)

    def flatten(self) -> Vector:
        """压平成一串数（顺序与 :meth:`AttentionParams.flatten` 逐位对齐）."""
        flat, _shapes = flatten_matrices(self.matrices())
        return flat

    def max_absolute(self) -> float:
        """四块参数梯度的最大绝对值（"这一次反向有没有给出东西"的一个读数）."""
        worst = 0.0
        for matrix in self.matrices():
            for row in matrix:
                for value in row:
                    worst = max(worst, abs(value))
        return worst

    def summary_line(self) -> str:
        """一行说明：``|dW_q|=max 0.0123 |dW_k|=… | 输入梯度 max 0.0456``."""
        parts = []
        for name, matrix in zip(
            ("dW_q", "dW_k", "dW_v", "dW_o"), self.matrices(), strict=True
        ):
            worst = max((abs(value) for row in matrix for value in row), default=0.0)
            parts.append(f"|{name}|={worst:.6f}")
        input_worst = max((abs(value) for row in self.grad_inputs for value in row), default=0.0)
        return " | ".join(parts) + f" | 输入梯度 max {input_worst:.6f}"

    def as_dict(self) -> dict[str, Matrix]:
        """按名字取四块参数梯度（生成报告与逐项校验用）."""
        return {
            GRAD_W_QUERY: self.grad_w_query,
            GRAD_W_KEY: self.grad_w_key,
            GRAD_W_VALUE: self.grad_w_value,
            GRAD_W_OUTPUT: self.grad_w_output,
        }


def relative_matrix_error(approximate: Matrix, reference: Matrix) -> float:
    """两块同形矩阵的最大**相对**逐点误差（分母取 ``max(1, |r|)``）.

    与 day074 的 ``gradcheck.max_scaled_gap`` 同一口径：
    "导数是 100 的量'与'导数是 1 的量"不该被同一个绝对阈值卡。
    这里额外校验两边同形——形状不同时直接报错而不是返回 ``inf``：
    梯度校验里"形状都不一样'是**实现错了**，不是'结果不一致"。
    """
    if matrix_shape(approximate) != matrix_shape(reference):
        raise ShapeError(
            f"两块矩阵形状不同：{matrix_shape(approximate)} 与 {matrix_shape(reference)}——"
            "在梯度校验里这意味着实现算错了（而不是'对不上'），因此当场拒绝。"
        )
    worst = 0.0
    for left_row, right_row in zip(approximate, reference, strict=True):
        for left, right in zip(left_row, right_row, strict=True):
            if not (is_finite(left) and is_finite(right)):
                return math.inf
            worst = max(worst, abs(left - right) / max(1.0, abs(right)))
    return worst


def matrix_max_absolute(matrix: Matrix) -> float:
    """矩阵逐元素绝对值的最大值（空矩阵给 0.0）."""
    return max((abs(value) for row in matrix for value in row), default=0.0)


def check_weights_are_a_distribution(
    weights: Matrix,
    *,
    tolerance: float = DEFAULT_SUM_TOLERANCE,
) -> None:
    """校验"每一行是一个概率分布"（非负 + 和为 1 + 有限）——**只有一处实现**.

    与 day073 的 ``check_distribution`` 是同一件事在两个尺度上的重复：
    那里校验"一个向量是一个分布'，这里校验'一张表的每一行是一个分布"。
    """
    checked = validate_matrix(weights, name="weights")
    for index, row in enumerate(checked):
        for column, value in enumerate(row):
            if value < 0:
                raise NumericError(
                    f"权重第 {index} 行第 {column} 列是负数（{value}）："
                    "负权重会让'加权平均'在某处变成减法，而整行看起来仍然是一条曲线。"
                )
        total = math.fsum(row)
        if abs(total - 1.0) > tolerance:
            raise NumericError(
                f"权重第 {index} 行的和是 {total!r}（容差 {tolerance}）："
                "每一行是一个**条件分布**（'这一行怎么看各个位置'），"
                "和前不为 1 会让后续加权求和变成一次缩放错误的组合。"
            )


def softmax_shape_scale(head_dim: int) -> float:
    """缩放系数 ``1/√d_k``（``head_dim < 1`` 当场报错）.

    这个函数在 day073 里叫 ``scaling_factor``。今天重写一遍不是重复：
    day073 的版本服务于"给定 Q/K/V 的注意力'，今天的版本服务于'参数化的注意力"，
    而两者的**调用点**不同（前者在纯函数里，后者在 ``AttentionShape.scale`` 上）。
    保留两处实现的风险是"某一天有人改了一处"——因此
    ``tests/test_attention_gradients.py`` 里有一条断言把两者逐位对上。
    """
    if isinstance(head_dim, bool) or not isinstance(head_dim, int):
        raise ParameterError(f"head_dim 必须是整数，收到 {head_dim!r}。")
    if head_dim < 1:
        raise ParameterError(f"head_dim 必须 >= 1，收到 {head_dim}。")
    return 1.0 / math.sqrt(head_dim)


def project(inputs: Matrix, weight: Matrix) -> Matrix:
    """一次投影 ``x · Wᵀ``（**唯一的乘法口径**，反向的推导以它为准）.

    写成 ``x·Wᵀ`` 而不是 ``x·W`` 的理由是形状的可读性：
    权重 ``(d_out, d_in)`` 的第一维就是"这一层输出多少维"，
    与 ``nn.Linear`` 一致——于是"某层是不是 768→768"这个问题看形状就能回答。
    """
    checked_inputs = validate_matrix(inputs, name="inputs")
    checked_weight = validate_matrix(weight, name="weight")
    if matrix_shape(checked_inputs)[1] != matrix_shape(checked_weight)[1]:
        raise ShapeError(
            f"输入行宽 {matrix_shape(checked_inputs)[1]} 与权重列数 "
            f"{matrix_shape(checked_weight)[1]} 不一致：投影是 x·Wᵀ，"
            "两者必须共用同一个输入维度。"
        )
    return matmul(checked_inputs, transpose(checked_weight))


def assert_rows_are_distributions(weights: Matrix) -> Vector:
    """返回每一行的和（**校验过之后再给读数**）——报告里用它作为证据."""
    check_weights_are_a_distribution(weights)
    return row_sums(weights)


__all__ = [
    "ATTENTION_STAGES",
    "ATTENTION_STAGE_DESCRIPTIONS",
    "ATTENTION_STAGE_SHAPES",
    "DEFAULT_SUM_TOLERANCE",
    "GRADIENT_TARGETS",
    "GRADIENT_TARGET_DESCRIPTIONS",
    "GRADIENT_TARGET_FORMULAS",
    "GRAD_INPUTS",
    "GRAD_W_KEY",
    "GRAD_W_OUTPUT",
    "GRAD_W_QUERY",
    "GRAD_W_VALUE",
    "PROPERTY_CHECKS",
    "PROPERTY_CHECK_DESCRIPTIONS",
    "PROPERTY_CAUSAL_NO_LEAK",
    "PROPERTY_NON_NEGATIVE",
    "PROPERTY_PERMUTATION_EQUIVARIANCE",
    "PROPERTY_ROW_STOCHASTIC",
    "STAGE_MASK",
    "STAGE_MIX",
    "STAGE_OUTPUT",
    "STAGE_PROJECT",
    "STAGE_SCALE",
    "STAGE_SCORE",
    "STAGE_SOFTMAX",
    "AttentionForward",
    "AttentionParams",
    "AttentionShape",
    "ParameterGradients",
    "assert_rows_are_distributions",
    "check_weights_are_a_distribution",
    "matrix_max_absolute",
    "project",
    "relative_matrix_error",
    "softmax_shape_scale",
]
