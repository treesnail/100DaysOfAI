"""把多头注意力训练起来：同参数量下的 heads 对照（day076 / M7-D2）.

前八章把"多头前向、多头反向、六条性质、可达集合"都写成了可以核对的东西。
今天最后一步与 day075 相同：**让它真的学起来**——并且这次多问一句：

> 同一批参数、同一个参数量，把 ``heads`` 从 1 调到 2 调到 3，会有什么不同？

## 一、这一次实验为什么必须在"同参数量"下做

```text
四块投影的形状      W_q/W_k: (d_k, d_in)      W_v: (d_v, d_in)      W_o: (d_out, d_v)
```

它们**完全不含 heads**——``heads`` 只决定"这些维度被切成几段"。
因此下面这三个配置的参数量一字不差：

```text
heads = 1    1 头 × 宽度 6
heads = 2    2 头 × 宽度 3
heads = 3    3 头 × 宽度 2
```

这不是巧合，而是本课的核心命题之一：**多头不是"更多参数买更多能力"**。
[``HeadsComparison``][HeadsComparison] 里因此有一条 ``comparable`` 检查——
"参数量必须全部相同"，不同就当场拒绝出报告：一份在参数量上不可比的对照表，
无论数字多好看都不回答任何问题。

## 二、四个读数一起看（比 day075 多一个）

```text
损失        混出来的向量离目标多远            优化在不在动
峰值权重    这一行最看重谁、看重多少          注意力有没有变尖
命中率      最大的那个分量对上了吗            结果对不对
头间差异    各头的分布有多不一样（平均 TV）    多头有没有退化成单头
```

第四个读数是这一课新增的，也是它存在的理由：**"多头"这个结构本身可能白买**——
如果所有头都学到了同一个分布，那么损失照样降、命中率照样涨，
而这一层退化成了一个单头注意力（付了 heads 份参数、只用了一份注意力）。

## 三、一次刻意标注的"不回答"

```text
本模块回答     "在同一个任务、同一批参数上，不同 heads 的训练读数各是多少"
本模块不回答   "heads 越多越好"——那需要多个种子、多个任务与统计检验
```

``compare_heads`` 的返回值里因此带一条固定注记：
**它是同一颗种子下的一次观测，不是一项结论。**
这一条纪律与 day075 的"三个读数一起看"是同一种：
**不要用一个读数说一件它证明不了的事。**
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.math_foundations.optim import (
    Objective,
    Optimizer,
    Schedule,
    TrainingTrace,
    flatten_matrices,
    unflatten_matrices,
)
from smart_research_agent.math_foundations.types import Matrix, Vector, matrix_shape
from smart_research_agent.multi_head.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.multi_head.layers import multi_head_attention, multi_head_backward
from smart_research_agent.multi_head.types import MultiHeadShape
from smart_research_agent.multi_head.verify import (
    head_disagreement,
    numerical_multihead_gradients,
)
from smart_research_agent.transformer_core.layers import (
    masked_mean_squared_error,
    masked_mse_gradient,
    row_argmax_hits,
)
from smart_research_agent.transformer_core.train import (
    DEFAULT_INIT_SCALE as CLASSIC_INIT_SCALE,
)
from smart_research_agent.transformer_core.train import (
    GRADIENT_SOURCE_ANALYTIC,
    GRADIENT_SOURCES,
    PARAMETER_BLOCKS,
    InductionTask,
)
from smart_research_agent.transformer_core.types import (
    AttentionParams,
    ParameterGradients,
)

#: 两种梯度来源（与 day075 逐字相同；运行时有一条断言把两者钉成相等）.
MULTIHEAD_GRADIENT_SOURCES: tuple[str, ...] = GRADIENT_SOURCES

#: 缺省初始化的小扰动幅度（与 day075 同值，同样另立一个常量以便逐位对账）.
DEFAULT_INIT_SCALE = CLASSIC_INIT_SCALE

#: 四个参数块的名字（顺序 = ``AttentionParams.matrices`` 的顺序）.
MULTIHEAD_PARAMETER_BLOCKS: tuple[str, ...] = PARAMETER_BLOCKS


def _resolve_trainable(trainable: Sequence[str] | None) -> tuple[str, ...]:
    """把"这次训练哪几块参数"规范成一个元组（``None`` = 全部）.

    与 day075 的 ``_resolve_trainable`` 是**故意的重复**（两处的判据完全一致）：
    它决定的是"冻结"这件事在数值上的含义，而"复制一份"的风险在于
    "某天有人只改了一处"——测试里有一条断言把两者的输出逐位对上。

    多头下"冻结"多了一层用处：**冻结 q/k 只训练 v/o** 能回答
    "如果注意力模式不变，多头还剩多少能力"——那是可达集合那一章的训练版对照。
    """
    if trainable is None:
        return MULTIHEAD_PARAMETER_BLOCKS
    checked: list[str] = []
    for name in trainable:
        if name not in MULTIHEAD_PARAMETER_BLOCKS:
            raise ParameterError(
                f"不认识的参数块 {name!r}：可用取值 {list(MULTIHEAD_PARAMETER_BLOCKS)}。"
            )
        if name in checked:
            raise ParameterError(
                f"参数块 {name!r} 重复：重复不会让它'更重要'，只会让读数难以解释。"
            )
        checked.append(name)
    if not checked:
        raise ParameterError(
            "至少要训练一块参数：一块都不训练时损失不会变，"
            "而'损失没变'看起来像'学习率太小'。"
        )
    return tuple(checked)


def _mask_untrained_blocks(
    flat_grad: Vector,
    shapes: Sequence[tuple[int, int]],
    trainable: tuple[str, ...],
) -> Vector:
    """把不训练的那几块梯度**置 0**（"冻结"在数值上的全部含义）.

    与 day075 的同名函数逐行同构。多头下它多一个用途：
    冻结 ``w_query``/``w_key`` 就能做"注意力模式固定、只看 value 路径"的对照。
    """
    blocks = unflatten_matrices(flat_grad, shapes)
    if len(blocks) != len(MULTIHEAD_PARAMETER_BLOCKS):
        raise ShapeError(
            f"形状表给出了 {len(blocks)} 块，而参数块有 "
            f"{len(MULTIHEAD_PARAMETER_BLOCKS)} 个。"
        )
    kept = tuple(
        matrix if name in trainable else _zero_matrix(matrix)
        for name, matrix in zip(MULTIHEAD_PARAMETER_BLOCKS, blocks, strict=True)
    )
    flattened, _shapes = flatten_matrices(kept)
    return flattened


def _zero_matrix(matrix: Matrix) -> Matrix:
    """同形状的全零矩阵."""
    return tuple(tuple(0.0 for _ in row) for row in matrix)


def _checked_tasks(tasks: Sequence[InductionTask]) -> tuple[InductionTask, ...]:
    """校验样本批：非空、词表一致（与 day075 同一判据）."""
    if not tasks:
        raise ShapeError("样本批不能为空：空批的平均损失没有定义。")
    vocabulary = tasks[0].vocabulary
    for index, task in enumerate(tasks):
        if task.vocabulary != vocabulary:
            raise ShapeError(
                f"第 {index} 条样本的词表是 {task.vocabulary}，而第一条是 {vocabulary}："
                "参数是按词表大小定的形状，混进来会让矩阵乘法对不上——"
                "或者更糟：对得上但语义错了。"
            )
    return tuple(tasks)


def _checked_heads(params: AttentionParams, heads: int) -> MultiHeadShape:
    """把（参数, 头数）变成形状记录（不可整除时在这里就报错）.

    先看一眼类型：``params.shape`` 在传错对象时抛的是 ``AttributeError``——
    而一个"属性不存在"的错误信息不会告诉调用方"你传的东西不是一层参数"。
    """
    if not isinstance(params, AttentionParams):
        raise ShapeError(
            f"params 必须是 AttentionParams，收到 {type(params).__name__}。"
        )
    return MultiHeadShape(params.shape, heads)


# --------------------------------------------------------------------------- #
# 批量的损失、梯度与读数
# --------------------------------------------------------------------------- #


def batch_loss(
    params: AttentionParams,
    tasks: Sequence[InductionTask],
    *,
    heads: int = 1,
    causal: bool = True,
) -> float:
    """一批样本上的平均监督 MSE（**损失是唯一被优化的东西**）."""
    checked = _checked_tasks(tasks)
    _checked_heads(params, heads)
    total = 0.0
    for task in checked:
        forward = multi_head_attention(params, task.inputs, heads=heads, causal=causal)
        total += masked_mean_squared_error(forward.output, task.target, task.supervised)
    return total / len(checked)


def batch_accuracy(
    params: AttentionParams,
    tasks: Sequence[InductionTask],
    *,
    heads: int = 1,
    causal: bool = True,
) -> float:
    """监督位置上的平均命中率（``argmax`` 是否与目标一致）."""
    checked = _checked_tasks(tasks)
    _checked_heads(params, heads)
    total = 0.0
    for task in checked:
        forward = multi_head_attention(params, task.inputs, heads=heads, causal=causal)
        total += row_argmax_hits(forward.output, task.target, task.supervised)
    return total / len(checked)


def batch_mean_peak_weight(
    params: AttentionParams,
    tasks: Sequence[InductionTask],
    *,
    heads: int = 1,
    causal: bool = True,
) -> float:
    """监督位置上"这一行每一头最大的那个权重"的平均值.

    分母是 ``监督位置数 × heads``：多头下每一头都有自己的一份分布，
    因此峰值必须**按头**统计——只统计第一头的读数在多头下没有意义
    （它会把"第二头很尖"这件事完全漏掉）。
    """
    checked = _checked_tasks(tasks)
    _checked_heads(params, heads)
    values: list[float] = []
    for task in checked:
        forward = multi_head_attention(params, task.inputs, heads=heads, causal=causal)
        for position in task.supervised:
            for head in range(forward.heads):
                values.append(forward.head_peaks[head][position])
    return math.fsum(values) / len(values)


def batch_mean_disagreement(
    params: AttentionParams,
    tasks: Sequence[InductionTask],
    *,
    heads: int = 1,
    causal: bool = True,
) -> float:
    """一批样本上"头间差异"的平均值（``heads=1`` 时恒为 0.0）.

    单头时返回 0.0 而不是抛异常：那是一个**真实读数**（"没有第二个头可比"），
    而把它当错误会让"跑一遍 heads=1 的对照"这件事变得不可能——
    而那一行正是对照表的第一行。
    """
    checked = _checked_tasks(tasks)
    _checked_heads(params, heads)
    if heads == 1:
        return 0.0
    total = 0.0
    for task in checked:
        forward = multi_head_attention(params, task.inputs, heads=heads, causal=causal)
        total += head_disagreement(forward).mean_total_variation
    return total / len(checked)


def batch_gradients(
    params: AttentionParams,
    tasks: Sequence[InductionTask],
    *,
    heads: int = 1,
    causal: bool = True,
    source: str = GRADIENT_SOURCE_ANALYTIC,
    trainable: Sequence[str] | None = None,
) -> Vector:
    """一批样本上的平均参数梯度（压平成一串数，与 ``AttentionParams.flatten`` 对齐）.

    ``source`` 的两种取值与 day075 逐字相同：

    ```text
    analytic   本课实现的九步反向：一次前向 + 一次反向就能拿到全部参数梯度
    numeric    数值差分：2 × 参数量 次前向（慢，但不依赖任何推导）
    ```

    ``heads`` 在这里是一个**必须与解析侧一致**的参数：数值侧忘了传它时，
    它算的是单头前向，而"多头梯度对不上"会表现为 ``1e-2`` 量级的差，
    看起来像推导写错了（day075 已经因为"两侧算法不同"栽过一次）。
    """
    if source not in MULTIHEAD_GRADIENT_SOURCES:
        raise ParameterError(
            f"不认识的梯度来源 {source!r}：可用取值 {list(MULTIHEAD_GRADIENT_SOURCES)}。"
        )
    checked = _checked_tasks(tasks)
    _checked_heads(params, heads)
    checked_trainable = _resolve_trainable(trainable)
    shapes = params.flatten()[1]
    accumulated: Vector = tuple(0.0 for _ in range(sum(r * c for r, c in shapes)))
    for task in checked:
        if source == GRADIENT_SOURCE_ANALYTIC:
            forward = multi_head_attention(params, task.inputs, heads=heads, causal=causal)
            gradient_matrix = masked_mse_gradient(
                forward.output, task.target, task.supervised
            )
            contributions = multi_head_backward(forward, gradient_matrix).flatten()
        else:
            numeric = numerical_multihead_gradients(
                params,
                task.inputs,
                task.target,
                heads=heads,
                causal=causal,
                supervised=task.supervised,
            )
            contributions = _flatten_parameter_gradients(numeric)
        accumulated = tuple(
            total + value for total, value in zip(accumulated, contributions, strict=True)
        )
    averaged = tuple(value / len(checked) for value in accumulated)
    return _mask_untrained_blocks(averaged, shapes, checked_trainable)


def _flatten_parameter_gradients(gradients: ParameterGradients) -> Vector:
    """把四块参数梯度按 ``AttentionParams`` 的顺序压平（与参数同一张形状表）."""
    flat, _shapes = flatten_matrices(gradients.matrices())
    return flat


# --------------------------------------------------------------------------- #
# 训练回路与报告
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MultiHeadTrainingReport:
    """一次多头训练的全过程（比 day075 多一列"头间差异"）.

    四列读数的长度都必须一致（``steps + 1``：含**第 0 步**）：

    ```text
    trace.losses      损失（沿用 day074 的 TrainingTrace）
    peak_weights      每一步之后的平均峰值权重（机制有没有变尖）
    accuracies        每一步之后的命中率（结果对不对）
    disagreements     每一步之后的平均头间差异（多头有没有退化）
    ```
    """

    trace: TrainingTrace
    final_parameters: AttentionParams
    heads: int
    peak_weights: tuple[float, ...]
    accuracies: tuple[float, ...]
    disagreements: tuple[float, ...]
    gradient_source: str = GRADIENT_SOURCE_ANALYTIC
    trainable: tuple[str, ...] = MULTIHEAD_PARAMETER_BLOCKS
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.heads < 1:
            raise ParameterError(f"头数必须 >= 1，收到 {self.heads}。")
        expected = self.trace.steps + 1
        lengths = {
            "峰值权重": len(self.peak_weights),
            "命中率": len(self.accuracies),
            "头间差异": len(self.disagreements),
        }
        for name, length in lengths.items():
            if length != expected:
                raise NumericError(
                    f"{name}有 {length} 项，而损失有 {expected} 项（含第 0 步）："
                    "四列读数的长度必须一致，否则报告里的某两列会错开一行——"
                    "而错开一行在图上只是'曲线略陡'。"
                )
        if self.gradient_source not in MULTIHEAD_GRADIENT_SOURCES:
            raise ParameterError(
                f"不认识的梯度来源 {self.gradient_source!r}："
                f"可用取值 {list(MULTIHEAD_GRADIENT_SOURCES)}。"
            )
        for name in self.trainable:
            if name not in MULTIHEAD_PARAMETER_BLOCKS:
                raise ParameterError(
                    f"不认识的参数块 {name!r}：可用取值 "
                    f"{list(MULTIHEAD_PARAMETER_BLOCKS)}。"
                )
        if self.heads == 1 and any(value != 0.0 for value in self.disagreements):
            raise NumericError(
                "单头训练的头间差异必须恒为 0.0：没有第二个头可比——"
                "一个非零值意味着这个读数统计了不该统计的东西。"
            )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def frozen(self) -> tuple[str, ...]:
        """这一次**没有**被训练的参数块."""
        return tuple(
            name for name in MULTIHEAD_PARAMETER_BLOCKS if name not in self.trainable
        )

    @property
    def shape(self) -> MultiHeadShape:
        """这一层实际的形状（含头数）."""
        return MultiHeadShape(self.final_parameters.shape, self.heads)

    @property
    def parameter_count(self) -> int:
        """参数量（**与 heads 无关**——这正是"同参数量对照"的物理基础）."""
        return self.final_parameters.parameter_count()

    @property
    def steps(self) -> int:
        """实际走过的步数."""
        return self.trace.steps

    @property
    def initial_loss(self) -> float:
        """初始损失（第 0 步）."""
        return self.trace.initial_loss

    @property
    def final_loss(self) -> float:
        """最终损失."""
        return self.trace.final_loss

    @property
    def improvement_ratio(self) -> float:
        """相对下降比例."""
        return self.trace.improvement

    @property
    def initial_accuracy(self) -> float:
        """初始命中率."""
        return self.accuracies[0]

    @property
    def final_accuracy(self) -> float:
        """最终命中率."""
        return self.accuracies[-1]

    @property
    def initial_peak_weight(self) -> float:
        """初始平均峰值权重."""
        return self.peak_weights[0]

    @property
    def final_peak_weight(self) -> float:
        """最终平均峰值权重."""
        return self.peak_weights[-1]

    @property
    def final_disagreement(self) -> float:
        """最终平均头间差异（``heads=1`` 时恒为 0.0）."""
        return self.disagreements[-1]

    @property
    def ok(self) -> bool:
        """这次训练是否真的让损失下降了（**这一层的护栏**）."""
        return self.final_loss < self.initial_loss

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "steps": self.steps,
            "heads": self.heads,
            "parameter_count": self.parameter_count,
            "gradient_source": self.gradient_source,
            "trainable": list(self.trainable),
            "frozen": list(self.frozen),
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "improvement_ratio": self.improvement_ratio,
            "initial_accuracy": self.initial_accuracy,
            "final_accuracy": self.final_accuracy,
            "initial_peak_weight": self.initial_peak_weight,
            "final_peak_weight": self.final_peak_weight,
            "final_disagreement": self.final_disagreement,
            "losses": list(self.trace.losses),
            "accuracies": list(self.accuracies),
            "peak_weights": list(self.peak_weights),
            "disagreements": list(self.disagreements),
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``200 步（heads=2）| 损失 0.1634 → 0.0000（↓100.0%）| …``."""
        return (
            f"{self.steps} 步（heads={self.heads}，参数 {self.parameter_count} 个）| "
            f"损失 {self.initial_loss:.4f} → {self.final_loss:.4f}"
            f"（↓{self.improvement_ratio:.1%}）| "
            f"命中率 {self.initial_accuracy:.0%} → {self.final_accuracy:.0%} | "
            f"峰值 {self.initial_peak_weight:.3f} → {self.final_peak_weight:.3f} | "
            f"头间差异 {self.final_disagreement:.4f}"
        )

    def epoch_line(self, index: int) -> str:
        """第 ``index`` 步的一行读数（含四列读数与当步学习率）."""
        if index < 0 or index >= len(self.accuracies):
            raise ParameterError(f"步号 {index} 超出范围 [0, {len(self.accuracies) - 1}]。")
        rate = "—" if index == 0 else f"{self.trace.learning_rates[index - 1]:.6f}"
        return (
            f"step {index:>4} | lr {rate:>9} | loss {self.trace.losses[index]:.6f} | "
            f"命中率 {self.accuracies[index]:.0%} | 峰值 {self.peak_weights[index]:.4f} | "
            f"头间差异 {self.disagreements[index]:.4f}"
        )


def train_multi_head(
    initial: AttentionParams,
    tasks: Sequence[InductionTask],
    *,
    heads: int,
    optimizer: Optimizer,
    steps: int,
    causal: bool = True,
    source: str = GRADIENT_SOURCE_ANALYTIC,
    schedule: Schedule | None = None,
    trainable: Sequence[str] | None = None,
) -> MultiHeadTrainingReport:
    """用多头解析梯度训练这一层，返回全过程报告（四列读数一起记）.

    与 day075 的 ``train_attention`` 只有两处不同：

    ```text
    ① heads 进了前向与反向的调用点（其余一行未改）
    ② 每步多记一列"头间差异"——它必须与另外三列**同长度**，
       否则报告里"多头有没有退化"这个问题会与曲线错开一行
    ```
    """
    checked = _checked_tasks(tasks)
    _checked_heads(initial, heads)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ParameterError(f"steps 必须是 >= 1 的整数，收到 {steps!r}。")
    if source not in MULTIHEAD_GRADIENT_SOURCES:
        raise ParameterError(
            f"不认识的梯度来源 {source!r}：可用取值 {list(MULTIHEAD_GRADIENT_SOURCES)}。"
        )
    checked_trainable = _resolve_trainable(trainable)
    flat, shapes = initial.flatten()
    losses: list[float] = []
    rates: list[float] = []
    history: list[Vector] = [flat]
    peaks: list[float] = []
    accuracies: list[float] = []
    disagreements: list[float] = []

    def _record(current: AttentionParams, loss: float) -> None:
        """把这一步的四个读数一起记下来（**必须一起记**，否则会错行）."""
        losses.append(loss)
        peaks.append(batch_mean_peak_weight(current, checked, heads=heads, causal=causal))
        accuracies.append(batch_accuracy(current, checked, heads=heads, causal=causal))
        disagreements.append(
            batch_mean_disagreement(current, checked, heads=heads, causal=causal)
        )

    current = initial
    _record(current, batch_loss(current, checked, heads=heads, causal=causal))
    for index in range(1, steps + 1):
        if schedule is not None:
            optimizer.learning_rate = schedule(index)
            if optimizer.learning_rate <= 0:
                raise ParameterError(f"第 {index} 步的调度给出了非正学习率。")
        grads = batch_gradients(
            current,
            checked,
            heads=heads,
            causal=causal,
            source=source,
            trainable=checked_trainable,
        )
        rates.append(optimizer.learning_rate)
        flat = optimizer.step(flat, grads)
        current = AttentionParams.unflatten(flat, shapes)
        history.append(flat)
        _record(current, batch_loss(current, checked, heads=heads, causal=causal))
    frozen_names = "、".join(
        name for name in MULTIHEAD_PARAMETER_BLOCKS if name not in checked_trainable
    )
    trace = TrainingTrace(
        losses=tuple(losses),
        learning_rates=tuple(rates),
        params=tuple(history),
        converged=losses[-1] <= 1e-12,
        notes=(
            f"梯度来源：{source}"
            "（analytic = 本课的九步反向；numeric = 数值差分，用于交叉验证）",
            "损失/命中率/峰值/头间差异四列长度一致，且第 0 项是"
            "'还没训练时'的读数——没有第 0 步就读不出「涨了多少」",
            "参数压平之后交给 day074 的 Optimizer：更新公式是逐分量的，与形状无关——"
            "**多头不改变优化器那一层**",
            f"本次训练的参数块：{list(checked_trainable)}"
            f"（冻结 {frozen_names or '无'}）",
        ),
    )
    return MultiHeadTrainingReport(
        trace=trace,
        final_parameters=current,
        heads=heads,
        peak_weights=tuple(peaks),
        accuracies=tuple(accuracies),
        disagreements=tuple(disagreements),
        gradient_source=source,
        trainable=checked_trainable,
        notes=(
            "四个读数一起看：损失（混得多准）、峰值（注意力多尖）、"
            "命中率（argmax 对不对）、头间差异（多头有没有退化）",
            "heads 只改变「维度被切成几段」，**不改变参数量**——"
            "因此 heads=1/2/3 的对照是同参数量对照",
            "头间差异接近 0 意味着多头退化：损失照样降、命中率照样涨，"
            "而这一层等价于一个单头注意力（付了 heads 份参数、只用了一份注意力）",
            "本报告是**同一颗种子下的一次观测**，不是「heads 越多越好」的结论："
            "那需要多个种子、多个任务与统计检验",
        ),
    )


# --------------------------------------------------------------------------- #
# 同参数量对照
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HeadsComparisonRow:
    """对照表的一行：一个 heads 取值下的全部读数."""

    heads: int
    parameter_count: int
    initial_loss: float
    final_loss: float
    improvement_ratio: float
    initial_accuracy: float
    final_accuracy: float
    initial_peak_weight: float
    final_peak_weight: float
    final_disagreement: float
    steps: int

    def __post_init__(self) -> None:
        if self.heads < 1:
            raise ParameterError(f"头数必须 >= 1，收到 {self.heads}。")
        if self.parameter_count < 1:
            raise ParameterError(f"参数量必须 >= 1，收到 {self.parameter_count}。")

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "heads": self.heads,
            "parameter_count": self.parameter_count,
            "steps": self.steps,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "improvement_ratio": self.improvement_ratio,
            "initial_accuracy": self.initial_accuracy,
            "final_accuracy": self.final_accuracy,
            "initial_peak_weight": self.initial_peak_weight,
            "final_peak_weight": self.final_peak_weight,
            "final_disagreement": self.final_disagreement,
        }


@dataclass(frozen=True)
class HeadsComparison:
    """同参数量下的 heads 对照表（:func:`compare_heads` 的产物）.

    ``comparable`` 是这张表的**前置条件**而不是一句修饰：
    参数量不同的行放在一张表里，比较的就是"更多的参数"而不是"不同的头数"。
    """

    rows: tuple[HeadsComparisonRow, ...]
    steps: int
    gradient_source: str
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if len(self.rows) < 2:
            raise ParameterError(
                f"对照至少要两行（收到 {len(self.rows)}）："
                "只有一行的'对照'回答不了任何问题。"
            )
        if self.steps < 1:
            raise ParameterError(f"步数必须 >= 1，收到 {self.steps}。")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def parameter_counts(self) -> tuple[int, ...]:
        """每一行的参数量（**必须全部相同**）."""
        return tuple(row.parameter_count for row in self.rows)

    @property
    def comparable(self) -> bool:
        """所有行的参数量是否相同（不同的对照在参数量上不可比）."""
        return len(set(self.parameter_counts)) == 1

    @property
    def best_final_loss(self) -> HeadsComparisonRow:
        """最终损失最小的那一行（并列时取头数较小的一行，保证可复现）."""
        return min(self.rows, key=lambda row: (row.final_loss, row.heads))

    @property
    def most_diverse(self) -> HeadsComparisonRow:
        """头间差异最大的那一行（``heads=1`` 且只有它时是它自己）."""
        return max(self.rows, key=lambda row: (row.final_disagreement, -row.heads))

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "comparable": self.comparable,
            "parameter_counts": list(self.parameter_counts),
            "steps": self.steps,
            "gradient_source": self.gradient_source,
            "rows": [row.to_dict() for row in self.rows],
            "notes": list(self.notes),
        }

    def table_lines(self) -> tuple[str, ...]:
        """可打印的对照表（演示脚本第 9 节用它）."""
        header = (
            f"{'heads':>5} | {'参数量':>5} | {'初始损失':>9} | {'最终损失':>9} | "
            f"{'降幅':>7} | {'命中率':>7} | {'峰值':>7} | {'头间差异':>8}"
        )
        lines = ["  " + header, "  " + "-" * len(header)]
        for row in self.rows:
            lines.append(
                f"  {row.heads:>5} | {row.parameter_count:>5} | "
                f"{row.initial_loss:>9.6f} | {row.final_loss:>9.6f} | "
                f"{row.improvement_ratio:>6.1%} | {row.final_accuracy:>6.0%} | "
                f"{row.final_peak_weight:>7.4f} | {row.final_disagreement:>8.4f}"
            )
        return tuple(lines)

    def summary_line(self) -> str:
        """一行说明：``3 行对照（参数量 156 全部相同）| 最小损失 heads=2``."""
        counts = set(self.parameter_counts)
        parity = f"参数量 {counts.pop()}" if len(set(self.parameter_counts)) == 1 else "参数量不同"
        return (
            f"{len(self.rows)} 行对照（{parity}，{self.steps} 步，{self.gradient_source}）| "
            f"最小损失 heads={self.best_final_loss.heads}"
        )


def compare_heads(
    initial: AttentionParams,
    tasks: Sequence[InductionTask],
    *,
    heads_list: Sequence[int],
    optimizer_factory: Callable[[], Optimizer],
    steps: int,
    causal: bool = True,
    source: str = GRADIENT_SOURCE_ANALYTIC,
    trainable: Sequence[str] | None = None,
) -> HeadsComparison:
    """在**同一批参数、同一个任务**上跑几个 heads 取值，返回对照表.

    ``optimizer_factory`` 是一个**每次都新建优化器**的工厂而不是一个实例：
    优化器带状态（动量、二阶动量），复用一个实例会让第二次训练带着
    第一次的动量起步——而那个差别表现为"某一行收敛得莫名其妙地快"。
    这与 day074 的 ``Optimizer.reset`` 是同一条纪律：**换任务时状态必须清空**。

    三处显式拒绝：

    ```text
    头数表少于两行        "对照"至少要两行
    某个 heads 不能整除   由 MultiHeadShape 当场拒绝（PartitionError）
    参数量不一致          在报告里由 comparable 暴露（并写进注记）
    ```
    """
    checked = _checked_tasks(tasks)
    checked_heads = tuple(heads_list)
    if len(checked_heads) < 2:
        raise ParameterError(
            f"对照至少要两个 heads 取值（收到 {len(checked_heads)}）："
            "只有一行的'对照'回答不了任何问题。"
        )
    for heads in checked_heads:
        _checked_heads(initial, heads)

    rows: list[HeadsComparisonRow] = []
    for heads in checked_heads:
        report = train_multi_head(
            initial,
            checked,
            heads=heads,
            optimizer=optimizer_factory(),
            steps=steps,
            causal=causal,
            source=source,
            trainable=trainable,
        )
        rows.append(
            HeadsComparisonRow(
                heads=heads,
                parameter_count=report.parameter_count,
                initial_loss=report.initial_loss,
                final_loss=report.final_loss,
                improvement_ratio=report.improvement_ratio,
                initial_accuracy=report.initial_accuracy,
                final_accuracy=report.final_accuracy,
                initial_peak_weight=report.initial_peak_weight,
                final_peak_weight=report.final_peak_weight,
                final_disagreement=report.final_disagreement,
                steps=report.steps,
            )
        )
    return HeadsComparison(
        rows=tuple(rows),
        steps=steps,
        gradient_source=source,
        notes=(
            "这是**同参数量对照**：heads 只改变「维度被切成几段」，"
            "四个投影矩阵的形状完全不含 heads——因此 ``parameter_counts`` "
            "必然全部相同（这里**不再写一遍运行时校验**：一个永远为真的分支"
            "在覆盖率上看起来与「防御写得全」一模一样）",
            "同一颗种子、同一个任务、同一个优化器配置——"
            "因此它是**一次观测**，不是「heads 越多越好」的结论",
            "读四点：最终损失（准不准）、命中率（对不对）、峰值（尖不尖）、"
            "头间差异（多头有没有退化）——**没有一个能单独说明「多头有没有用」**",
        ),
    )


def analytic_objective(
    tasks: Sequence[InductionTask],
    *,
    heads: int,
    causal: bool = True,
) -> Objective:
    """把"这一批样本上的多头损失"包成一个只吃压平参数的目标函数（调试用）.

    用途与 day075 的对应函数相同：把一个压平向量喂给 ``calculus.gradient``
    就得到数值梯度，与 :func:`batch_gradients` 的解析结果比一比。
    """
    checked = _checked_tasks(tasks)
    if heads < 1:
        raise ParameterError(f"头数必须 >= 1，收到 {heads}。")

    def objective(flat: Vector) -> float:
        """按固定形状还原参数并算多头损失."""
        rebuilt = AttentionParams.unflatten(flat, _shape_table(checked[0]))
        return batch_loss(rebuilt, checked, heads=heads, causal=causal)

    return objective


def _shape_table(task: InductionTask) -> tuple[tuple[int, int], ...]:
    """一条样本对应的参数形状表（四个 ``(vocab, vocab)``）."""
    size = task.vocabulary
    return ((size, size), (size, size), (size, size), (size, size))


def matrix_totals(matrix: Matrix) -> float:
    """矩阵逐元素之和（演示脚本用它核对"两头的贡献之和 = 输出"）."""
    rows, columns = matrix_shape(matrix)
    return math.fsum(
        matrix[row][column] for row in range(rows) for column in range(columns)
    )


__all__ = [
    "DEFAULT_INIT_SCALE",
    "MULTIHEAD_GRADIENT_SOURCES",
    "MULTIHEAD_PARAMETER_BLOCKS",
    "HeadsComparison",
    "HeadsComparisonRow",
    "MultiHeadTrainingReport",
    "analytic_objective",
    "batch_accuracy",
    "batch_gradients",
    "batch_loss",
    "batch_mean_disagreement",
    "batch_mean_peak_weight",
    "compare_heads",
    "matrix_totals",
    "train_multi_head",
]
