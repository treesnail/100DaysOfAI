"""``transformer_stack`` 的记录与口径表：把一个块复制成一条链（day080 / M7-D5）.

## 今天新增的唯一结构

day079 交出一个**块**，而块的价值在于它是**唯一被复制的单位**：

```text
day079   块 = LN → 注意力 → ⊕ → LN → 前馈 → ⊕        一个块的六个阶段
day080   链 = 块 × N                                  ——今天的全部结构
```

“复制 N 份”听上去像一行 ``for`` 循环，而它引进了三个**只在堆叠里才存在**的读数：

```text
增益     ‖y_i‖ / ‖x_i‖     这一层把输入放大/缩小了多少倍（逐层可见，前向）
直通占比 ‖x_i‖ / (‖x_i‖ + ‖branch1‖ + ‖branch2‖)
                          这一层的输出里有多少是**沿着残差直通过去**的（前向）
层梯度   ‖∂loss/∂x_i‖      每一层入口处的梯度范数（反向，day079 只看过最底层那一个）
```

第三个是今天最要紧的一个：day079 的深度实验只印了**最底层**那一个数，
而今天把**每一层**都留下来——于是“梯度在第几层开始塌”从一个结论变成了一个序列。

## 五张口径表

```text
STACK_STAGES        一次堆叠前向的五个阶段（进入 / 块 / 记账 / 传递 / 出口）
STAGE_SHAPES        每个阶段两侧的形状（**每一阶段都保形**）
STACK_PROPERTIES    六条性质（保形 / 与手写循环逐位一致 / 确定性 / 分支全零即恒等 /
                    参数量与构造对齐 / 生成的 PyTorch 脚本带着同一个形状）
STACK_GRADIENT_TARGETS  两处梯度校验的名单（整条链的输入 / 某一层的输入）
STACK_NOTES         三条边界（这套读数回答什么、不回答什么）
```

## 为什么“与手写循环逐位一致”值得是一条性质

``stack_forward`` 是 ``encoder_block`` 外面套了一个 ``for``。这句话听上去平凡，
而以“**逐位相等**”而不是“误差小于某值”作为判据，会顺手钉住三件事：

```text
① 累积顺序       第 i 层的累计结果必须**原样**交给第 i+1 层（不许中间过一次浮点运算）
② 参数对应关系    第 i 层用的必须是第 i 份参数（错位一份在数值上仍然"像样"）
③ 账的完整性      留下的每一层账必须能重放出同一串输出（反向要用它）
```

②是最阴的一类错误：把 ``blocks`` 与 ``attentions`` 的 zip 顺序写反、或者少一层，
结果**不会报形状错误**，只会得到一摞“每层都用错参数”的模型——而它的损失曲线
看起来和正确的差不多。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from smart_research_agent.encoder_decoder.types import (
    BLOCK_GRADIENT_TARGETS,
    GRAD_INPUTS,
    BlockForward,
    BlockGradients,
    BlockParameters,
    BlockShape,
    _checked_activation,
    _checked_placement,
    matrix_max_absolute,
)
from smart_research_agent.encoder_decoder.types import (
    relative_matrix_error as _relative_matrix_error,
)
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
    validate_vector,
)
from smart_research_agent.transformer_core.types import AttentionForward, AttentionParams
from smart_research_agent.transformer_stack.errors import (
    AssemblyError,
    NumericError,
    ParameterError,
    ShapeError,
)

# ---------------------------------------------------------------------- 五个阶段

#: ① 进入：校验输入形状，把它当成“第 0 层的输出”.
STAGE_ENTER = "enter"
#: ② 块：第 i 层内部的那六个子层（day079 的 ``encoder_block``）.
STAGE_BLOCK = "block"
#: ③ 记账：把这一层的入/出范数与两个分支的范数记下来.
STAGE_CENSUS = "census"
#: ④ 传递：把这一层的输出交给下一层（**块保形 → 这一步永远成立**）.
STAGE_CARRY = "carry"
#: ⑤ 出口：返回最后一层的输出（形状与最开始的输入一致）.
STAGE_EXIT = "exit"

STACK_STAGES: tuple[str, ...] = (
    STAGE_ENTER,
    STAGE_BLOCK,
    STAGE_CENSUS,
    STAGE_CARRY,
    STAGE_EXIT,
)

STAGE_DESCRIPTIONS: dict[str, str] = {
    STAGE_ENTER: "校验输入形状；把它当作第 0 层的输出（这一步只做一次）",
    STAGE_BLOCK: "第 i 层：LN → 注意力 → ⊕ → LN → 前馈 → ⊕（day079 的六个阶段）",
    STAGE_CENSUS: "记下这一层的 ‖x_i‖、‖y_i‖、两个分支的范数与直通占比",
    STAGE_CARRY: "把 y_i 原样交给第 i+1 层——块保形，因此这一步在数学上不可能失败",
    STAGE_EXIT: "返回 y_N：它与第 0 层的输出（也就是输入）逐位同形",
}

STAGE_SHAPES: dict[str, str] = {
    STAGE_ENTER: "(n, d) → (n, d)：进入不改形状",
    STAGE_BLOCK: "(n, d) → (n, d)：块的九个形状断言（day079）",
    STAGE_CENSUS: "(n, d) → (n, d) + 一行读数：记账**不碰**张量",
    STAGE_CARRY: "(n, d) → (n, d)：同形传递，链上没有任何一次 reshape",
    STAGE_EXIT: "(n, d) → (n, d)：出口与入口同形（这是“可堆叠”的定义）",
}

# ---------------------------------------------------------------------- 六条性质

PROPERTY_SHAPE_PRESERVED = "stack_preserves_shape_at_every_layer"
PROPERTY_MATCHES_LOOP = "stack_matches_a_hand_written_loop"
PROPERTY_DETERMINISTIC = "stack_is_deterministic"
PROPERTY_IDENTITY_WHEN_BRANCHES_VANISH = "stack_is_the_identity_when_every_branch_vanishes"
PROPERTY_PARAMETER_COUNT_MATCHES = "analytic_parameter_count_matches_construction"
PROPERTY_ASSEMBLY_CARRIES_SHAPE = "generated_torch_assembly_carries_the_same_shape"

STACK_PROPERTIES: tuple[str, ...] = (
    PROPERTY_SHAPE_PRESERVED,
    PROPERTY_MATCHES_LOOP,
    PROPERTY_DETERMINISTIC,
    PROPERTY_IDENTITY_WHEN_BRANCHES_VANISH,
    PROPERTY_PARAMETER_COUNT_MATCHES,
    PROPERTY_ASSEMBLY_CARRIES_SHAPE,
)

PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_SHAPE_PRESERVED: "每一层的输出与下一层的输入**逐位**同形（逐层断言，不是只查第一层）",
    PROPERTY_MATCHES_LOOP: "``stack_forward`` 与手写逐层调用 ``encoder_block`` **逐位**一致",
    PROPERTY_DETERMINISTIC: "同一组参数跑两次，输出与每一行读数**逐位**相同",
    PROPERTY_IDENTITY_WHEN_BRANCHES_VANISH: "把**每一层**的两个分支都置零，整条链退化成恒等映射",
    PROPERTY_PARAMETER_COUNT_MATCHES: "解析式算出的参数量与实际构造出来的逐块数一致",
    PROPERTY_ASSEMBLY_CARRIES_SHAPE: "生成的 PyTorch 脚本里的 d / d_ff / n / N 与 ``StackShape`` 同值",
}

STACK_NOTES: tuple[str, ...] = {
    "今天回答的是'把块堆起来之后，账与梯度在每一层各是多少'（一个**数值**问题）",
    "今天**不**回答'这摞块该训多少步、用什么学习率'——那是 day081 的事，"
    "本课只把每一层的读数交出来，好让 day081 有地方下手",
    "增益大于 1 也可能是坏事（那叫放大，不叫信息）——本课不承诺'越大越好'",
}

# ---------------------------------------------------------------------- 两处梯度校验

#: 整条链的输入梯度（day079 只量过它的**范数**，今天与数值差分**逐项**比）.
GRAD_STACK_INPUTS = "stack_inputs"

#: 报告的种类（决定取哪一张名单）.
KIND_STACK = "stack"
KIND_LAYER = "layer"

KINDS: tuple[str, ...] = (KIND_STACK, KIND_LAYER)

#: **某一层**的八块参数梯度（不含"输入"那一块：中间层的输入不是自变量，
#: 它由前一层算出来——把"某层的输入"当成可扰动对象会在链上指两个不同的东西）.
LAYER_GRADIENT_TARGETS: tuple[str, ...] = tuple(
    name for name in BLOCK_GRADIENT_TARGETS if name != GRAD_INPUTS
)

#: 两类报告各自的名单（**名单是构造参数**：缺项检查只有一处实现）.
STACK_GRADIENT_TARGETS: dict[str, tuple[str, ...]] = {
    KIND_STACK: (GRAD_STACK_INPUTS,),
    KIND_LAYER: LAYER_GRADIENT_TARGETS,
}

#: 默认的初始化幅度（与 day075~079 的 ``DEFAULT_INIT_SCALE`` 同值）.
DEFAULT_INIT_SCALE = 0.25

#: 层数少于这个值时报 ``AssemblyError``：一层都堆不起来时“堆叠”这个词没有意义.
MINIMUM_LAYERS = 1


def _checked_scale(scale: Any) -> float:
    """初始化幅度：必须落在 ``(0, 1)``（与 day075 的 ``default_parameters`` 同一条纪律）."""
    if isinstance(scale, bool) or not isinstance(scale, (int, float)):
        raise ParameterError(f"scale 必须是数，收到 {scale!r}。")
    value = float(scale)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ParameterError(
            f"scale 必须落在 (0, 1)，收到 {value!r}："
            "初始化太大时第一层的打分会被推到 softmax 的饱和区，而饱和区里的梯度极小。"
        )
    return value


def _checked_layers(value: Any) -> int:
    """层数：``>= 1`` 的整数（``0`` 层在数学上是恒等，在工程上是一次笔误）."""
    if isinstance(value, bool) or not isinstance(value, int) or value < MINIMUM_LAYERS:
        raise ParameterError(
            f"layers 必须是 >= {MINIMUM_LAYERS} 的整数，收到 {value!r}："
            "零层堆叠在数学上是一个恒等映射，而它更可能是把某个计数写漏了。"
        )
    return value


def _checked_finite(value: Any, *, name: str) -> float:
    """一个必须是有限实数的读数."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NumericError(f"{name} 必须是数，收到 {value!r}。")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise NumericError(f"{name} 必须是有限数，收到 {value!r}。")
    return resolved


def _checked_non_negative(value: Any, *, name: str) -> float:
    """一个必须非负的读数（范数就属于这一类）."""
    resolved = _checked_finite(value, name=name)
    if resolved < 0.0:
        raise NumericError(f"{name} 必须非负，收到 {resolved!r}：范数不可能是负数。")
    return resolved


def _checked_positive_dimension(value: Any, *, name: str) -> int:
    """维度：``>= 1`` 的整数，**由本模块抛本族的 ``ParameterError``**.

    day079 的 ``_checked_positive_int`` 抛的是 ``encoder_decoder`` 那一族，
    而本族与它是**兄弟**（都继承 day075 的 ``ParameterError``）——
    因此 ``except transformer_stack.ParameterError`` 兜不住它。
    本模块自己校验三个维度，正是为了让“谁的错”与“哪一族”对齐。
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ParameterError(f"{name} 必须是 >= 1 的整数，收到 {value!r}。")
    return value


def _checked_tolerance(value: Any, *, name: str = "tolerance") -> float:
    """容差：必须是正的有限数（**不接受 0**：那会把“无误差”当成唯一合格）."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"{name} 必须是数，收到 {value!r}。")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ParameterError(
            f"{name} 必须是正的有限数，收到 {value!r}："
            "容差为 0 意味着'必须逐位相等'，那样一条数值差分永远过不了。"
        )
    return resolved


@dataclass(frozen=True)
class StackShape:
    """一摞块的四个维度：``hidden`` / ``ffn`` / ``tokens`` / ``layers``.

    ```text
    hidden   d     隐藏维（块内所有张量的宽度，也是注意力四个投影的列数）
    ffn      4d    前馈的中间维
    tokens   n     序列长度（每一层都必须保持不变）
    layers   N     块被复制几份——**今天唯一新增的那个自由度**
    ```

    参数量被拆成三个**可以分别手算**的量，因为它们的来源不同：

    ```text
    block_parameter_count      块自己新增的：4d（两个 LN 的 γ/β）+ 2·d·d_ff（两层线性）
                               + d_ff（b_in）+ d（b_out）
    attention_parameter_count  注意力那四个 (d, d) 投影：4d²（day075 起就在那里）
    layer_parameter_count      上面两者之和（一层一次）
    ```
    """

    hidden: int
    ffn: int
    tokens: int
    layers: int

    def __post_init__(self) -> None:
        for name in ("hidden", "ffn", "tokens"):
            _checked_positive_dimension(getattr(self, name), name=name)
        _checked_layers(self.layers)

    @property
    def ffn_ratio(self) -> float:
        """``ffn / hidden``（原论文的取值是 4.0）."""
        return self.ffn / self.hidden

    @property
    def block_parameter_count(self) -> int:
        """**一个块**自己新增的参数（不含注意力那四个投影）."""
        return 5 * self.hidden + 2 * self.hidden * self.ffn + self.ffn

    @property
    def attention_parameter_count(self) -> int:
        """**一层**里注意力那四个 ``(d, d)`` 投影的参数个数."""
        return 4 * self.hidden * self.hidden

    @property
    def layer_parameter_count(self) -> int:
        """一层的全部参数（块 + 注意力）."""
        return self.block_parameter_count + self.attention_parameter_count

    @property
    def total_parameter_count(self) -> int:
        """整条链的全部参数（``layers`` 倍）."""
        return self.layers * self.layer_parameter_count

    @property
    def block_shape(self) -> BlockShape:
        """交给 day079 的 ``encoder_block`` 用的三种维度（**不含层数**）."""
        return BlockShape(hidden=self.hidden, ffn=self.ffn, tokens=self.tokens)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "hidden": self.hidden,
            "ffn": self.ffn,
            "tokens": self.tokens,
            "layers": self.layers,
            "ffn_ratio": self.ffn_ratio,
            "block_parameter_count": self.block_parameter_count,
            "attention_parameter_count": self.attention_parameter_count,
            "total_parameter_count": self.total_parameter_count,
        }

    def summary_line(self) -> str:
        """一行说明：``N=4 层 | n=4 d=6 d_ff=24（4.0×） | 每层 486 个参数 | 合计 1944``."""
        return (
            f"N={self.layers} 层 | n={self.tokens} d={self.hidden} d_ff={self.ffn}"
            f"（{self.ffn_ratio:.1f}×） | 每层 {self.layer_parameter_count} 个参数 | "
            f"合计 {self.total_parameter_count}"
        )


@dataclass(frozen=True)
class LayerCensus:
    """**一层的读数**——今天把它从“只看最底层”扩到“每一层都有一行”.

    ```text
    index           层号（0 是最下面那一层）
    input_norm      ‖x_i‖    这一层入口的 Frobenius 范数
    output_norm     ‖y_i‖    这一层出口的 Frobenius 范数
    branch1_norm    ‖注意力那一支的输出‖
    branch2_norm    ‖前馈那一支的输出‖
    max_abs_input   入口元素的最大绝对值（范数是平方和，这个是**单点**的读数）
    ```

    两个派生量：

    ```text
    gain         ‖y_i‖ / ‖x_i‖                这一层把“整体尺度”放大了多少倍
    carry_share  ‖x_i‖ / (‖x_i‖ + ‖b1‖ + ‖b2‖)  输出里有多少是沿着残差直通过去的
    ```

    ``carry_share`` 是今天最有意思的一个读数：**残差开时它严格落在 (0, 1)**，
    而残差关时它恰好是 ``0.0``（入口那一份没有被传下去）——于是
    “残差到底有没有生效”从一个布尔量变成了一个可以画出来的比例。
    """

    index: int
    input_norm: float
    output_norm: float
    branch1_norm: float
    branch2_norm: float
    max_abs_input: float
    use_residual: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ParameterError(f"层号必须是非负整数，收到 {self.index!r}。")
        if not isinstance(self.use_residual, bool):
            raise ParameterError(
                f"use_residual 必须是布尔量，收到 {self.use_residual!r}："
                "直通占比要靠它判断'入口那一份到底有没有被传下去'。"
            )
        for name in ("input_norm", "output_norm", "branch1_norm", "branch2_norm"):
            object.__setattr__(
                self, name, _checked_non_negative(getattr(self, name), name=name)
            )
        object.__setattr__(
            self, "max_abs_input", _checked_non_negative(self.max_abs_input, name="max_abs_input")
        )

    @property
    def gain(self) -> float:
        """``‖y_i‖ / ‖x_i‖``；入口范数为 0 时定义为 ``0.0``（**不是** 1，也不是 nan）.

        ``x_i`` 全零时增益没有定义。本包不返回 ``nan``（它会静默污染整张表），
        也不返回 ``1.0``（那看起来像“这一层什么都没干”）；它返回 ``0.0``，
        而 ``0.0`` 在一张“增益表”里是**显眼的**：任何一次真实的堆叠都不会出现它。
        """
        if self.input_norm == 0.0:
            return 0.0
        return self.output_norm / self.input_norm

    @property
    def carry_share(self) -> float:
        """输出里“沿着残差直通过去”的那一份占比.

        ```text
        残差开    ‖x_i‖ / (‖x_i‖ + ‖b1‖ + ‖b2‖)      一个严格落在 (0, 1) 的比例
        残差关    0.0                                入口那一份**确实**没有被传下去
        ```

        ``use_residual`` 被记进读数里，而不是留给读表的人去猜：否则同一张表里
        “0.30”既可能是“三成来自直通”，也可能是“残差其实关着”——它们必须是两个数。
        """
        if not self.use_residual:
            return 0.0
        denominator = self.input_norm + self.branch1_norm + self.branch2_norm
        if denominator == 0.0:
            return 0.0
        return self.input_norm / denominator

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "index": self.index,
            "input_norm": self.input_norm,
            "output_norm": self.output_norm,
            "branch1_norm": self.branch1_norm,
            "branch2_norm": self.branch2_norm,
            "max_abs_input": self.max_abs_input,
            "use_residual": self.use_residual,
            "gain": self.gain,
            "carry_share": self.carry_share,
        }

    def summary_line(self) -> str:
        """一行说明：``层 0 | ‖x‖ 1.234567 → ‖y‖ 1.345678（增益 1.09×） | 直通占比 0.42``."""
        return (
            f"层 {self.index} | ‖x‖ {self.input_norm:.6f} → ‖y‖ {self.output_norm:.6f}"
            f"（增益 {self.gain:.6f}×） | 直通占比 {self.carry_share:.4f}"
        )


@dataclass(frozen=True)
class StackedLayer:
    """链上的**一环**：第 ``index`` 层的账（前向的账 + 它那一层的参数）."""

    index: int
    block: BlockForward
    attention: AttentionForward
    params: BlockParameters

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ParameterError(f"层号必须是非负整数，收到 {self.index!r}。")
        if self.block.shape.hidden != self.params.hidden:
            raise ShapeError(
                f"第 {self.index} 层的账说 d={self.block.shape.hidden}，"
                f"而参数说 d={self.params.hidden}：账与参数必须来自同一次前向。"
            )

    @property
    def hidden(self) -> int:
        """这一层的隐藏维."""
        return self.block.shape.hidden

    def summary_line(self) -> str:
        """一行说明."""
        return f"第 {self.index} 层：{self.block.summary_line()}（{self.params.parameter_count} 个参数）"


@dataclass(frozen=True)
class StackParameters:
    """一摞块的参数：``blocks`` 与 ``attentions`` **逐层对齐**.

    两份参数分开存（而不是揉成一个记录），理由是它们的**来源**不同：
    ``blocks`` 是 day079 那一族（前馈 + 两个 LN），``attentions`` 是 day075 那一族
    （四个 ``(d, d)`` 投影）。堆叠要把它们逐层配起来，而“配错了”这件事
    不会以形状错误的形式暴露——因此这里在构造那一刻就把两份的长度钉死。
    """

    blocks: tuple[BlockParameters, ...]
    attentions: tuple[AttentionParams, ...]

    def __post_init__(self) -> None:
        resolved_blocks = tuple(self.blocks)
        resolved_attentions = tuple(self.attentions)
        if not resolved_blocks:
            raise ParameterError("一摞块至少要有一层参数。")
        if len(resolved_blocks) != len(resolved_attentions):
            raise ShapeError(
                f"块参数 {len(resolved_blocks)} 份与注意力参数 "
                f"{len(resolved_attentions)} 份不一致："
                "两份参数必须逐层对齐，少一份会在**前向跑到那一层时**才炸。"
            )
        hidden = resolved_blocks[0].hidden
        for index, (block, attention) in enumerate(
            zip(resolved_blocks, resolved_attentions, strict=True)
        ):
            if block.hidden != hidden:
                raise AssemblyError(
                    f"第 {index} 层的块宽度 {block.hidden} 与第 0 层的 {hidden} 不一致："
                    "链上每一层都必须同宽，否则第 i 层的输出接不进第 i+1 层。"
                )
            width = matrix_shape(attention.w_query)[1]
            if width != hidden:
                raise AssemblyError(
                    f"第 {index} 层的注意力投影列数 {width} 与块宽度 {hidden} 不一致："
                    "LN 的输出要接进注意力，两者的宽度必须一致。"
                )
        object.__setattr__(self, "blocks", resolved_blocks)
        object.__setattr__(self, "attentions", resolved_attentions)

    @property
    def layers(self) -> int:
        """层数（= 参数份数）."""
        return len(self.blocks)

    def shape_for(self, tokens: int) -> StackShape:
        """配上序列长度得到完整的形状（``tokens`` **不在参数里**，只能由调用方给）.

        参数里没有“序列长度”这件事是刻意的：``tokens`` 是**数据**的属性，
        不是模型的属性。让参数自己猜一个 ``tokens``（比如“反正第一层会告诉我”）
        会让“形状”在构造与运行两个时刻指两个不同的东西。
        """
        first = self.blocks[0]
        return StackShape(
            hidden=first.hidden,
            ffn=matrix_shape(first.ffn_w_in)[0],
            tokens=tokens,
            layers=self.layers,
        )

    def replace_layer(self, index: int, params: BlockParameters) -> StackParameters:
        """换掉第 ``index`` 层的块参数（其余不动）——数值梯度校验靠它逐块扰动."""
        resolved = _checked_layer_index(index, self.layers)
        if params.hidden != self.blocks[0].hidden:
            raise ShapeError(
                f"新参数的宽度 {params.hidden} 与链上的宽度 {self.blocks[0].hidden} 不一致。"
            )
        updated = list(self.blocks)
        updated[resolved] = params
        return replace(self, blocks=tuple(updated))

    @property
    def parameter_count(self) -> int:
        """整条链的参数总量（逐块数出来的，不是解析式）."""
        block_total = sum(params.parameter_count for params in self.blocks)
        attention_total = sum(
            sum(matrix_shape(matrix)[0] * matrix_shape(matrix)[1] for matrix in _projections(item))
            for item in self.attentions
        )
        return block_total + attention_total

    @property
    def block_parameter_count(self) -> int:
        """只数**块参数**的那一份（= :meth:`flatten` 出来的向量长度）."""
        return sum(params.parameter_count for params in self.blocks)

    def flatten(self) -> tuple[Vector, tuple[tuple[tuple[int, int], ...], ...]]:
        """压平成一串数，返回 ``(向量, 逐层的形状表)``.

        **只压平块参数**（每层 8 块、共 ``layers × block_parameter_count`` 个数）：
        注意力那四个投影不进来，因为 :meth:`unflatten` 会把它原样带回——
        而“哪一份参数需要被优化器逐分量更新”与“哪一份只是被携带”是两件事。

        形状表是**逐层**的（每一层 8 块），因此还原时不会把第 i 层的数填进第 j 层。
        """
        flat: list[float] = []
        shapes: list[tuple[tuple[int, int], ...]] = []
        for params in self.blocks:
            layer_flat, layer_shapes = params.flatten()
            flat.extend(layer_flat)
            shapes.append(layer_shapes)
        return tuple(flat), tuple(shapes)

    @classmethod
    def unflatten(
        cls,
        flat: Vector,
        shapes: tuple[tuple[tuple[int, int], ...], ...] | list[tuple[tuple[int, int], ...]],
        attentions: tuple[AttentionParams, ...],
    ) -> StackParameters:
        """把一串数还原成一摞块参数（注意力参数原样带入）."""
        resolved_shapes = tuple(shapes)
        if not resolved_shapes:
            raise ParameterError("形状表不能为空：没有形状就不知道该怎么切。")
        cursor = 0
        blocks: list[BlockParameters] = []
        for layer_index, layer_shapes in enumerate(resolved_shapes):
            required = sum(rows * columns for rows, columns in layer_shapes)
            piece = tuple(flat[cursor : cursor + required])
            if len(piece) != required:
                raise ShapeError(
                    f"第 {layer_index} 层需要 {required} 个数，而剩下的只有 {len(piece)} 个。"
                )
            blocks.append(BlockParameters.unflatten(piece, layer_shapes))
            cursor += required
        if cursor != len(flat):
            raise ShapeError(f"形状表只解释了 {cursor} 个数，而向量有 {len(flat)} 个。")
        return cls(blocks=tuple(blocks), attentions=attentions)

    def summary_line(self) -> str:
        """一行说明：``4 层 | 每层 d=6 d_ff=24 | 合计 1944 个参数``."""
        first = self.blocks[0]
        return (
            f"{self.layers} 层 | 每层 d={first.hidden} d_ff={matrix_shape(first.ffn_w_in)[0]} | "
            f"合计 {self.parameter_count} 个参数"
        )


def _projections(params: AttentionParams) -> tuple[Matrix, ...]:
    """注意力那四个投影（顺序固定：q / k / v / o）."""
    return (params.w_query, params.w_key, params.w_value, params.w_output)


def _checked_layer_index(index: Any, layers: int) -> int:
    """层号必须落在 ``[0, layers)``（越界时当场报错，而不是静默地改最后一层）."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise ParameterError(f"层号必须是整数，收到 {index!r}。")
    if not 0 <= index < layers:
        raise ParameterError(f"层号必须落在 [0, {layers})，收到 {index}。")
    return index


@dataclass(frozen=True)
class StackForward:
    """一次堆叠前向留下的**全部账**：每一环的六个阶段 + 每一行读数.

    ``layers`` 里的下标就是层号（0 在最下面），而 ``censuses`` 与它**同序**——
    两串东西分开存是刻意的：读数是要给人看的表，账是要给反向用的中间量，
    把它们揉进一个记录里会让“读表”与“反传”互相牵制。
    """

    shape: StackShape
    placement: str
    use_residual: bool
    activation: str
    inputs: Matrix
    output: Matrix
    layers: tuple[StackedLayer, ...]
    censuses: tuple[LayerCensus, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        _checked_placement(self.placement)
        _checked_activation(self.activation)
        checked_inputs = validate_matrix(self.inputs, name="inputs")
        checked_output = validate_matrix(self.output, name="output")
        resolved_layers = tuple(self.layers)
        resolved_censuses = tuple(self.censuses)
        if matrix_shape(checked_inputs) != matrix_shape(checked_output):
            raise ShapeError(
                f"链的输入 {matrix_shape(checked_inputs)} 与输出 "
                f"{matrix_shape(checked_output)} 形状不一致：块保形 ⇒ 整条链也保形。"
            )
        if matrix_shape(checked_inputs) != (self.shape.tokens, self.shape.hidden):
            raise ShapeError(
                f"形状说 (n, d) = ({self.shape.tokens}, {self.shape.hidden})，"
                f"而张量是 {matrix_shape(checked_inputs)}：形状与账必须来自同一次前向。"
            )
        if len(resolved_layers) != self.shape.layers:
            raise ShapeError(
                f"账里有 {len(resolved_layers)} 环，而形状说要 {self.shape.layers} 层。"
            )
        if len(resolved_censuses) != len(resolved_layers):
            raise ShapeError(
                f"读数 {len(resolved_censuses)} 行与账 {len(resolved_layers)} 环不一致："
                "每一层都必须有一行读数，否则'第几层开始塌'这句话就没有依据。"
            )
        for position, layer in enumerate(resolved_layers):
            if layer.index != position:
                raise ShapeError(
                    f"第 {position} 环的层号是 {layer.index}：账必须按层号顺序存放。"
                )
            if matrix_shape(layer.block.inputs) != matrix_shape(checked_inputs):
                raise ShapeError(
                    f"第 {position} 层的输入 {matrix_shape(layer.block.inputs)} 与链的输入"
                    f" {matrix_shape(checked_inputs)} 不同形：链上不许出现 reshape。"
                )
        for position in range(len(resolved_layers) - 1):
            current = resolved_layers[position].block.output
            following = resolved_layers[position + 1].block.inputs
            if current != following:
                raise ShapeError(
                    f"第 {position} 层的输出与第 {position + 1} 层的输入不符："
                    "链上传递的必须是**同一个值**（逐位相等），而不是重算一遍的近似。"
                )
        if resolved_layers[-1].block.output != checked_output:
            raise ShapeError("最后一层的输出与 ``output`` 字段不是同一份张量。")
        object.__setattr__(self, "inputs", checked_inputs)
        object.__setattr__(self, "output", checked_output)
        object.__setattr__(self, "layers", resolved_layers)
        object.__setattr__(self, "censuses", resolved_censuses)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def depth(self) -> int:
        """层数（= ``shape.layers``）."""
        return len(self.layers)

    @property
    def tokens(self) -> int:
        """序列长度."""
        return matrix_shape(self.inputs)[0]

    def census_at(self, index: int) -> LayerCensus:
        """取第 ``index`` 层的读数（越界当场报错）."""
        return self.censuses[_checked_layer_index(index, self.depth)]

    def layer_at(self, index: int) -> StackedLayer:
        """取第 ``index`` 环的账（越界当场报错）."""
        return self.layers[_checked_layer_index(index, self.depth)]

    def gain_profile(self) -> tuple[float, ...]:
        """逐层增益（从第 0 层到第 N-1 层）."""
        return tuple(row.gain for row in self.censuses)

    def carry_profile(self) -> tuple[float, ...]:
        """逐层直通占比（从第 0 层到第 N-1 层）."""
        return tuple(row.carry_share for row in self.censuses)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "shape": self.shape.to_dict(),
            "placement": self.placement,
            "use_residual": self.use_residual,
            "activation": self.activation,
            "inputs": [list(row) for row in self.inputs],
            "output": [list(row) for row in self.output],
            "censuses": [row.to_dict() for row in self.censuses],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``4 层 | pre-LN 残差 开 | n=4 d=6 | 输出与输入同形``."""
        residual = "开" if self.use_residual else "**关**"
        return (
            f"{self.depth} 层 | {self.placement}-LN 残差 {residual} | "
            f"n={self.tokens} d={self.shape.hidden} | 输出与输入同形"
        )

    def census_table(self) -> tuple[str, ...]:
        """把逐层读数排成一张表（每一行一层）."""
        header = (
            f"  {'层':>3} | {'‖x‖':>11} | {'‖y‖':>11} | {'增益':>9} | "
            f"{'‖分支一‖':>10} | {'‖分支二‖':>10} | {'直通占比':>8}"
        )
        lines = [header, "  " + "-" * 82]
        for row in self.censuses:
            lines.append(
                f"  {row.index:>3} | {row.input_norm:>11.6f} | {row.output_norm:>11.6f} | "
                f"{row.gain:>9.6f} | {row.branch1_norm:>10.6f} | "
                f"{row.branch2_norm:>10.6f} | {row.carry_share:>8.4f}"
            )
        return tuple(lines)


@dataclass(frozen=True)
class StackGradients:
    """一次堆叠反向留下的账：**整条链的输入梯度 + 每一环的九块**.

    ``layers`` 与 ``StackForward.layers`` **同序**（下标即层号），
    而反向的**计算**顺序是倒着的——这个差别由 ``stack_backward`` 负责，
    记录里只保留“按层号排好”的样子，因为读数是要按层号看的。
    """

    shape: StackShape
    grad_inputs: Matrix
    layers: tuple[BlockGradients, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        checked = validate_matrix(self.grad_inputs, name="grad_inputs")
        resolved = tuple(self.layers)
        if len(resolved) != self.shape.layers:
            raise ShapeError(
                f"梯度账里有 {len(resolved)} 环，而形状说要 {self.shape.layers} 层。"
            )
        for position, item in enumerate(resolved):
            if matrix_shape(item.grad_inputs) != matrix_shape(checked):
                raise ShapeError(
                    f"第 {position} 环的输入梯度 {matrix_shape(item.grad_inputs)} 与"
                    f"整条链的 {matrix_shape(checked)} 不同形：链上每一环都保形。"
                )
        if resolved and resolved[0].grad_inputs != checked:
            raise ShapeError(
                "``grad_inputs`` 必须是第 0 层的输入梯度——"
                "否则“整条链的输入梯度”与“最底层那一环”会指两个不同的东西。"
            )
        object.__setattr__(self, "grad_inputs", checked)
        object.__setattr__(self, "layers", resolved)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def depth(self) -> int:
        """层数."""
        return len(self.layers)

    def norms(self) -> tuple[float, ...]:
        """逐层入口梯度的范数（按层号顺序，**含第 0 层**）."""
        return tuple(frobenius(item.grad_inputs) for item in self.layers)

    def layer_at(self, index: int) -> BlockGradients:
        """取第 ``index`` 环的九块梯度."""
        return self.layers[_checked_layer_index(index, self.depth)]

    def summary_line(self) -> str:
        """一行说明：``4 层 | 逐层 ‖dx‖ 从 3.1e-01 到 1.2e-02 | 最底层 3.1e-01``."""
        norms = self.norms()
        return (
            f"{self.depth} 层 | 逐层 ‖dx‖ 从 {norms[0]:.2e} 到 {norms[-1]:.2e} | "
            f"整条链输入梯度 {frobenius(self.grad_inputs):.6f}"
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（逐层只留范数：九块矩阵太大）. """
        return {
            "shape": self.shape.to_dict(),
            "depth": self.depth,
            "input_gradient_norm": frobenius(self.grad_inputs),
            "layer_norms": list(self.norms()),
            "notes": list(self.notes),
        }


def frobenius(matrix: Matrix) -> float:
    """Frobenius 范数 ``√Σx²``（用 ``math.fsum`` 求和，避免长序列上的累积误差）."""
    checked = validate_matrix(matrix, name="matrix")
    return math.sqrt(math.fsum(value * value for row in checked for value in row))


def max_absolute(matrix: Matrix) -> float:
    """元素绝对值的最大值（转发 ``encoder_decoder`` 的同名读数，口径只有一处）."""
    return matrix_max_absolute(matrix)


def relative_matrix_error(approximate: Matrix, reference: Matrix) -> float:
    """相对矩阵误差（口径与 day075~079 一致：**分母是参考的范数**，两边共享实现）."""
    return _relative_matrix_error(approximate, reference)


__all__ = [
    "DEFAULT_INIT_SCALE",
    "GRAD_STACK_INPUTS",
    "KINDS",
    "KIND_LAYER",
    "KIND_STACK",
    "LAYER_GRADIENT_TARGETS",
    "MINIMUM_LAYERS",
    "PROPERTY_ASSEMBLY_CARRIES_SHAPE",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_DETERMINISTIC",
    "PROPERTY_IDENTITY_WHEN_BRANCHES_VANISH",
    "PROPERTY_MATCHES_LOOP",
    "PROPERTY_PARAMETER_COUNT_MATCHES",
    "PROPERTY_SHAPE_PRESERVED",
    "STACK_GRADIENT_TARGETS",
    "STACK_NOTES",
    "STACK_PROPERTIES",
    "STACK_STAGES",
    "STAGE_BLOCK",
    "STAGE_CARRY",
    "STAGE_CENSUS",
    "STAGE_DESCRIPTIONS",
    "STAGE_ENTER",
    "STAGE_EXIT",
    "STAGE_SHAPES",
    "LayerCensus",
    "StackForward",
    "StackGradients",
    "StackParameters",
    "StackShape",
    "StackedLayer",
    "frobenius",
    "max_absolute",
    "relative_matrix_error",
]
