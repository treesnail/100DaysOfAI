"""把一层注意力训练起来：induction 任务、损失、梯度与训练报告（day075 / M7-D1）.

前六章把"一层自注意力"写成了可以做前向也能做反向的东西。今天最后一步：
**让它真的学起来**——否则"可微"这个形容词仍然没有被验证过。

## 任务：induction（"这个 token 之前出现过，找出它"）

```text
序列      a  b  c  a  b            （sources=3 个不同的 token，repeats=2 次重复）
监督位置  第 3、4 位                 （只有"重复出现"的位置有确定答案）
目标      第 i 位的输出应当等于**它自己那个 token 的 one-hot**
```

这个任务的名字来自"induction head"——真实模型里确实会出现一种专门做这件事的注意力头。
它的好处是**答案可以被逐位复核**：一个只依赖"找同一个 token"的机制就能做对，
因此"训练有没有效果"不会与"任务的模糊性"混在一起。

## 为什么这一课用 MSE 而不是交叉熵

MSE 的梯度是 ``2(y−t)/N``——**一行就能手算**。于是这一课里
"外部输入的梯度"是平凡的，任何梯度校验失败都只可能来自注意力那七步。
（这与 day074 用"碗形函数"验证优化器是同一个手法：**把不确定性一次只留一个**。）

## 三种读数，各自回答一个问题

```text
损失       混出来的向量离目标多远          —— 优化在不在动
峰值权重   这一行最看重谁、看重多少        —— 注意力有没有变尖（机制在不在学）
命中率     最大的那个分量对上了吗          —— 结果对不对
```

三者一起看才能区分"学会了、但损失还差一点"与"损失降了、但机制没学到"。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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
from smart_research_agent.math_foundations.probability import uniforms
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.transformer_core.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.transformer_core.layers import (
    attention_backward,
    masked_mean_squared_error,
    masked_mse_gradient,
    row_argmax_hits,
    self_attention,
)
from smart_research_agent.transformer_core.types import (
    AttentionParams,
    ParameterGradients,
)
from smart_research_agent.transformer_core.verify import numerical_parameter_gradients

#: 梯度来源：解析反向（本课的产物）或数值差分（校验用，慢但不依赖推导）.
GRADIENT_SOURCE_ANALYTIC = "analytic"
GRADIENT_SOURCE_NUMERIC = "numeric"

#: 两种梯度来源（顺序 = 从"生产用"到"校验用"）。
GRADIENT_SOURCES: tuple[str, ...] = (GRADIENT_SOURCE_ANALYTIC, GRADIENT_SOURCE_NUMERIC)

#: 缺省初始化的小扰动幅度（0.25 = 初值落在 [-0.25, 0.25) 上）。
DEFAULT_INIT_SCALE = 0.25

#: 四个参数块的名字（顺序 = ``AttentionParams.matrices`` 的顺序）。
PARAMETER_BLOCKS: tuple[str, ...] = ("w_query", "w_key", "w_value", "w_output")


def _resolve_trainable(trainable: Sequence[str] | None) -> tuple[str, ...]:
    """把"这次训练哪几块参数"规范成一个元组（``None`` = 全部）.

    **冻结**是一项真实的技术（LoRA 就是"冻结主干、只训练低秩旁路"），
    而在这一课里它还有一个诊断用途——**它把"三个读数"拆开了**：

    ```text
    四块全训（Adam 0.05，200 步）  损失 0.1634 → 0.0000   命中率 38% → 100%   峰值 0.232 → 0.444
    只训 q/k（同样配置）            损失 0.1634 → 0.1554   命中率 38% →  12%   峰值 0.232 → 0.587
    ```

    读法（**两个方向都反直觉**）：

    ```text
    只训 q/k      注意力确实变尖了（峰值 0.59），但损失几乎不动——
                  "看对了地方"只是必要条件，还要 value/output 把看到的东西映射成目标
    四块全训      损失降到 0、命中率 100%，而峰值只有 0.44——
                  模型找到了另一条路（在 value 路径上把不需要的分量抵消掉），
                  因此"损失为 0"不等于"学到了 induction 机制"
    ```

    结论：**没有任何一个读数能单独说明"学到了什么"。**
    """
    if trainable is None:
        return PARAMETER_BLOCKS
    checked: list[str] = []
    for name in trainable:
        if name not in PARAMETER_BLOCKS:
            raise ParameterError(
                f"不认识的参数块 {name!r}：可用取值 {list(PARAMETER_BLOCKS)}。"
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

    压平之后每一块的位置由形状表决定，因此"置 0"必须按块做，
    而不是按某个分量区间猜——猜错的后果是某一层的权重按另一层的梯度更新，
    而它只会表现为"训练变慢"。
    """
    blocks = unflatten_matrices(flat_grad, shapes)
    if len(blocks) != len(PARAMETER_BLOCKS):
        raise ShapeError(
            f"形状表给出了 {len(blocks)} 块，而参数块有 {len(PARAMETER_BLOCKS)} 个。"
        )
    kept = tuple(
        matrix if name in trainable else _zero_matrix(matrix)
        for name, matrix in zip(PARAMETER_BLOCKS, blocks, strict=True)
    )
    flattened, _shapes = flatten_matrices(kept)
    return flattened


def _zero_matrix(matrix: Matrix) -> Matrix:
    """同形状的全零矩阵."""
    return tuple(tuple(0.0 for _ in row) for row in matrix)


@dataclass(frozen=True)
class InductionTask:
    """一条 induction 样本：一段 token 序列 + 哪些位置是监督位置.

    ```text
    tokens      (length,)           token 下标（可重复）
    inputs      (length, vocab)     每个 token 的 one-hot 向量
    supervised  (repeats,)          监督位置（只有"重复出现"的位置有确定答案）
    ```

    ``target`` 就是 ``inputs``：**目标是把那个 token 自己复制出来**。
    写成"目标 = 输入"而不是另存一份，是因为它们必须是同一份数据——
    两份各写一次迟早会在某次改动里悄悄分家，而"目标错了"只会表现为"学不会"。
    """

    vocabulary: int
    tokens: tuple[int, ...]
    inputs: Matrix
    supervised: tuple[int, ...]
    seed: int

    def __post_init__(self) -> None:
        if isinstance(self.vocabulary, bool) or not isinstance(self.vocabulary, int):
            raise ParameterError(f"vocabulary 必须是整数，收到 {self.vocabulary!r}。")
        if self.vocabulary < 2:
            raise ParameterError(
                f"vocabulary 必须 >= 2，收到 {self.vocabulary}："
                "只有一个 token 的词表里没有'匹配'这件事可言。"
            )
        if not self.tokens:
            raise ShapeError("token 序列不能为空。")
        for token in self.tokens:
            if not 0 <= token < self.vocabulary:
                raise ParameterError(
                    f"token 下标 {token} 落在 [0, {self.vocabulary}) 之外。"
                )
        if matrix_shape(self.inputs) != (len(self.tokens), self.vocabulary):
            raise ShapeError(
                f"输入形状 {matrix_shape(self.inputs)} 与 "
                f"({len(self.tokens)}, {self.vocabulary}) 不一致。"
            )
        if not self.supervised:
            raise ShapeError(
                "监督位置不能为空：没有任何监督位置时损失没有定义——"
                "返回 0.0 会让'这一批样本什么也没教'伪装成'损失完美'。"
            )
        for position in self.supervised:
            if not 0 <= position < len(self.tokens):
                raise ShapeError(f"监督位置 {position} 落在 [0, {len(self.tokens)}) 之外。")
            if not _earlier_matches(self.tokens, position):
                raise ShapeError(
                    f"监督位置 {position} 的 token 在它之前没有出现过："
                    "induction 任务要求'答案能在上下文里被找到'——"
                    "否则这一位没有可学的机制，模型只能靠记忆。"
                )

    @property
    def length(self) -> int:
        """序列长度."""
        return len(self.tokens)

    @property
    def target(self) -> Matrix:
        """目标（= 输入）：把那个 token 复制出来."""
        return self.inputs

    def describe(self) -> str:
        """一行说明（进报告：报告里要能读出"这一条样本长什么样"）."""
        marks = "".join(
            "^" if position in self.supervised else " "
            for position in range(self.length)
        )
        return f"tokens={self.tokens} 监督='{marks}'（vocab={self.vocabulary}）"

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "seed": self.seed,
            "vocabulary": self.vocabulary,
            "tokens": list(self.tokens),
            "supervised": list(self.supervised),
            "length": self.length,
        }


def _earlier_matches(tokens: tuple[int, ...], position: int) -> tuple[int, ...]:
    """在**严格更早**的位置（0..position−1）里，与 ``tokens[position]`` 相同的下标.

    "严格更早"这三个字是这条例子的全部内容：把 ``position`` 自己算进去，
    "每个位置都能在上下文里找到答案"就变成了**恒真**——而一个恒真的校验
    与"没有校验"在报告里长得一模一样。
    """
    target = tokens[position]
    return tuple(index for index in range(position) if tokens[index] == target)


def _distinct_offsets(count: int, size: int, *, seed: int) -> tuple[int, ...]:
    """用确定性数列抽 ``count`` 个**互不相同**的下标（都在 ``[0, size)`` 内）.

    为什么不用 ``random.sample``：day073 起全包只用 ``uniforms`` 那串
    LCG 数列，因此"这一次抽到了什么"永远可以被复核。
    抽样方式是"抽到重复就丢掉再抽"（拒绝采样）——它在 ``count <= size`` 时必然终止，
    而超过时这里直接报错（不进入死循环）。

    **返回的是下标（位置），不是值**：这一点在 induction 任务的生成里很要紧——
    第一步要抽"哪几个 token 出现在头部"（值域是词表），
    第二步要抽"哪几个头部位置被重复"（值域是头部长度）。
    两者混用会得到一个"值当成了下标"的越界错误——而它只在某些种子下出现，
    因此更容易被误当成"某条样本的数据问题"。
    """
    if count < 1:
        raise ParameterError(f"count 必须 >= 1，收到 {count}。")
    if size < 1:
        raise ParameterError(f"size 必须 >= 1，收到 {size}。")
    if count > size:
        raise ParameterError(
            f"要抽 {count} 个互不相同的下标，而范围只有 {size} 个："
            "拒绝采样在这里会一直抽不到，因此当场拒绝而不是死循环。"
        )
    chosen: list[int] = []
    raw = uniforms(count * size * 2 + 16, seed=seed)
    for value in raw:
        candidate = int(value * size)
        if candidate >= size:  # pragma: no cover - uniforms 落在 [0,1)
            candidate = size - 1
        if candidate not in chosen:
            chosen.append(candidate)
        if len(chosen) == count:
            break
    if len(chosen) < count:  # pragma: no cover - 防御式：数列长度已给足
        raise NumericError(
            f"确定性数列没有凑够 {count} 个互不相同的下标"
            "（这不应当发生：本函数的数列长度是按最坏情况给的）。"
        )
    return tuple(chosen)


def make_induction_task(
    *,
    seed: int,
    vocabulary: int = 6,
    sources: int = 3,
    repeats: int = 2,
) -> InductionTask:
    """造一条 induction 样本：``sources`` 个不同的 token + ``repeats`` 个重复.

    ```text
    sources=3, repeats=2, vocabulary=6
    tokens = [1, 0, 3, 0, 1]        监督位置 = {3, 4}
              ↑ 头部（互不相同）    ↑ 尾部：从头部里挑几个重复一遍
    ```

    两步抽样都用同一个确定性数列（``seed`` 与 ``seed + 1``），
    因此"第 3 位是 0"这件事可以被复核。
    """
    if repeats < 1:
        raise ParameterError(f"repeats 必须 >= 1，收到 {repeats}。")
    if sources < 1:
        raise ParameterError(f"sources 必须 >= 1，收到 {sources}。")
    if isinstance(vocabulary, bool) or not isinstance(vocabulary, int):
        raise ParameterError(f"vocabulary 必须是整数，收到 {vocabulary!r}。")
    if vocabulary < 2:
        raise ParameterError(
            f"vocabulary 必须 >= 2，收到 {vocabulary}："
            "只有一个 token 的词表里没有'匹配'这件事可言。"
        )
    head = _distinct_offsets(sources, vocabulary, seed=seed)
    repeated_positions = _distinct_offsets(
        min(repeats, sources), sources, seed=seed + 1
    )
    tail = tuple(
        head[repeated_positions[index % len(repeated_positions)]]
        for index in range(repeats)
    )
    tokens = head + tail
    inputs = tuple(
        tuple(1.0 if position == token else 0.0 for position in range(vocabulary))
        for token in tokens
    )
    supervised = tuple(range(sources, sources + repeats))
    return InductionTask(
        vocabulary=vocabulary,
        tokens=tokens,
        inputs=inputs,
        supervised=supervised,
        seed=seed,
    )


def make_induction_batch(
    count: int,
    *,
    seed: int = 42,
    vocabulary: int = 6,
    sources: int = 3,
    repeats: int = 2,
) -> tuple[InductionTask, ...]:
    """造一批样本（每条一个 ``seed + index``，因此整批可复现）."""
    if count < 1:
        raise ParameterError(f"count 必须 >= 1，收到 {count}。")
    return tuple(
        make_induction_task(
            seed=seed + index * 7,
            vocabulary=vocabulary,
            sources=sources,
            repeats=repeats,
        )
        for index in range(count)
    )


def default_parameters(
    vocabulary: int,
    *,
    seed: int = 7,
    scale: float = DEFAULT_INIT_SCALE,
) -> AttentionParams:
    """确定性的小随机初始化（四个矩阵都在 ``[-scale, scale)`` 上）.

    **四个矩阵**都随机，而不是"把 W_v/W_o 初始化成单位矩阵"：
    后者会让"输出 = 目标 token 的混合"从一开始就大致正确，
    于是命中率一开始就是 1.0，"训练有没有让它变好"就看不出来了。
    随机初始化下命中率从 0 附近起步，因此三个读数都会真的动起来。

    ``seed`` 与样本的 seed 是独立的两个参数：样本变了、初始化不变，
    才能让"换一批样本"与"换一次初始化"两件事分开被观察。
    """
    if vocabulary < 2:
        raise ParameterError(f"vocabulary 必须 >= 2，收到 {vocabulary}。")
    if not 0.0 < scale < 1.0:
        raise ParameterError(
            f"scale 必须落在 (0, 1)，收到 {scale}："
            "初始化太大时第一层的打分会被推到 softmax 的饱和区，"
            "而饱和区里的梯度极小——训练会在第一步就显得'不动'。"
        )
    return AttentionParams(
        w_query=_random_matrix(vocabulary, vocabulary, seed=seed, scale=scale),
        w_key=_random_matrix(vocabulary, vocabulary, seed=seed + 1, scale=scale),
        w_value=_random_matrix(vocabulary, vocabulary, seed=seed + 2, scale=scale),
        w_output=_random_matrix(vocabulary, vocabulary, seed=seed + 3, scale=scale),
    )


def _random_matrix(rows: int, columns: int, *, seed: int, scale: float) -> Matrix:
    """确定性的小随机矩阵（``uniforms`` 的 [0,1) 映射到 [-scale, scale)）."""
    raw = uniforms(rows * columns, seed=seed)
    values = [(value * 2.0 - 1.0) * scale for value in raw]
    return tuple(
        tuple(values[row * columns : (row + 1) * columns]) for row in range(rows)
    )


# --------------------------------------------------------------------------- #
# 批量的损失、梯度与读数
# --------------------------------------------------------------------------- #


def _checked_tasks(tasks: Sequence[InductionTask]) -> tuple[InductionTask, ...]:
    """校验样本批：非空、词表一致（**不同词表的样本不能放在同一批里**）."""
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


def batch_loss(
    params: AttentionParams,
    tasks: Sequence[InductionTask],
    *,
    causal: bool = True,
) -> float:
    """一批样本上的平均监督 MSE（**损失是唯一被优化的东西**）."""
    checked = _checked_tasks(tasks)
    total = 0.0
    for task in checked:
        forward = self_attention(params, task.inputs, causal=causal)
        total += masked_mean_squared_error(forward.output, task.target, task.supervised)
    return total / len(checked)


def batch_gradients(
    params: AttentionParams,
    tasks: Sequence[InductionTask],
    *,
    causal: bool = True,
    source: str = GRADIENT_SOURCE_ANALYTIC,
    trainable: Sequence[str] | None = None,
) -> Vector:
    """一批样本上的平均参数梯度（压平成一串数，与 ``AttentionParams.flatten`` 对齐）.

    ``source`` 有两种取值，它们的存在本身就是这一课的结论：

    ```text
    analytic   本课实现的七步反向：一次前向 + 一次反向就能拿到全部参数梯度
    numeric    数值差分：2 × 参数量 次前向（慢，但不依赖任何推导）
    ```

    ``trainable`` 非空时，**不训练的那几块梯度被置 0**（冻结）。

    ## 两种梯度源的对账（实测）

    ```text
    SGD  0.05   6 步   损失轨迹最大差 5.09e-14
    Adam 0.05   6 步   损失轨迹最大差 4.62e-11
    Adam 0.05  30 步   损失轨迹最大差 7.05e-11
    Adam 0.10  30 步   损失轨迹最大差 1.41e-10
    ```

    差距的来源是**数值差分自身的误差**（约 ``1e-11``，day074 的分辨率公式），
    而 Adam 的差距比 SGD 大约 1000 倍——这正是"Adam 按 ``√v̂`` 归一化，
    会放大梯度分量的**相对**误差"这条性质的直接观测。
    两支都稳定收敛，因此这不是实现问题；但**要验证梯度实现，请优先用 SGD**：
    它的步长与梯度成线性，误差不会被放大。

    ## 一个真实踩过的坑：两侧必须算**同一个损失**

    第一版的数值侧算的是**全行** MSE，而解析侧算的是**监督行** MSE——
    于是"梯度最大绝对差 7.5e-3"（梯度范数只有 4.8e-2，即差 15%），
    而**两边都是对的**。补上 ``supervised`` 之后差值降到 ``2.1e-11``。
    一个"两边都对却对不上"的对照比"有一边错"的对照更难查：
    它会让人先去怀疑推导。**这就是为什么对照必须把"算的是哪个式子"写进参数名。**
    """
    if source not in GRADIENT_SOURCES:
        raise ParameterError(
            f"不认识的梯度来源 {source!r}：可用取值 {list(GRADIENT_SOURCES)}。"
        )
    checked = _checked_tasks(tasks)
    checked_trainable = _resolve_trainable(trainable)
    shapes = params.flatten()[1]
    accumulated: Vector = tuple(0.0 for _ in range(sum(r * c for r, c in shapes)))
    for task in checked:
        if source == GRADIENT_SOURCE_ANALYTIC:
            forward = self_attention(params, task.inputs, causal=causal)
            gradient_matrix = masked_mse_gradient(forward.output, task.target, task.supervised)
            contributions = attention_backward(forward, gradient_matrix).flatten()
        else:
            numeric = numerical_parameter_gradients(
                params,
                task.inputs,
                task.target,
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
    """把四块参数梯度按 ``AttentionParams`` 的顺序压平（**与参数同一张形状表**）."""
    flat, _shapes = flatten_matrices(gradients.matrices())
    return flat


def batch_accuracy(
    params: AttentionParams,
    tasks: Sequence[InductionTask],
    *,
    causal: bool = True,
) -> float:
    """监督位置上的平均命中率（``argmax`` 是否与目标一致）."""
    checked = _checked_tasks(tasks)
    total = 0.0
    for task in checked:
        forward = self_attention(params, task.inputs, causal=causal)
        total += row_argmax_hits(forward.output, task.target, task.supervised)
    return total / len(checked)


def batch_mean_peak_weight(
    params: AttentionParams,
    tasks: Sequence[InductionTask],
    *,
    causal: bool = True,
) -> float:
    """监督位置上"这一行最大的那个权重"的平均值（注意力有没有变尖）.

    ```text
    均匀分布（n 个候选）   峰值 = 1/n        看不出任何选择
    只盯一个位置           峰值 = 1.0        选择最锐利
    ```
    """
    checked = _checked_tasks(tasks)
    values: list[float] = []
    for task in checked:
        forward = self_attention(params, task.inputs, causal=causal)
        for position in task.supervised:
            values.append(forward.peak_weights[position])
    return math.fsum(values) / len(values)


# --------------------------------------------------------------------------- #
# 训练回路与报告
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AttentionTrainingReport:
    """一次训练的全过程：损失、学习率、参数（沿用 day074 的 ``TrainingTrace``）.

    在 trace 之上多两列**这一层特有的读数**：

    ```text
    peak_weights   每一步之后的平均峰值权重（机制有没有变尖）
    accuracies     每一步之后的命中率（结果对不对）
    ```

    三者长度一致（``steps + 1``：含**第 0 步**，即"还没训练时"的样子）——
    没有第 0 步就读不出"涨了多少"。
    """

    trace: TrainingTrace
    final_parameters: AttentionParams
    peak_weights: tuple[float, ...]
    accuracies: tuple[float, ...]
    gradient_source: str = GRADIENT_SOURCE_ANALYTIC
    trainable: tuple[str, ...] = PARAMETER_BLOCKS
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        expected = self.trace.steps + 1
        if len(self.peak_weights) != expected or len(self.accuracies) != expected:
            raise NumericError(
                f"峰值权重/命中率各有 {len(self.peak_weights)} 与 {len(self.accuracies)} 项，"
                f"而损失有 {expected} 项（含第 0 步）：三者的长度必须一致，"
                "否则报告里的某两列会错开一行——而错开一行在图上只是'曲线略陡'。"
            )
        if self.gradient_source not in GRADIENT_SOURCES:
            raise ParameterError(
                f"不认识的梯度来源 {self.gradient_source!r}：可用取值 {list(GRADIENT_SOURCES)}。"
            )
        for name in self.trainable:
            if name not in PARAMETER_BLOCKS:
                raise ParameterError(
                    f"不认识的参数块 {name!r}：可用取值 {list(PARAMETER_BLOCKS)}。"
                )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def frozen(self) -> tuple[str, ...]:
        """这一次**没有**被训练的参数块（报告里要能读出"哪几块被冻结"）."""
        return tuple(name for name in PARAMETER_BLOCKS if name not in self.trainable)

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
        """相对下降比例（初始损失为 0 时记 0.0）."""
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
    def ok(self) -> bool:
        """这次训练是否真的让损失下降了（**这是这一层的护栏**）."""
        return self.final_loss < self.initial_loss

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "steps": self.steps,
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
            "losses": list(self.trace.losses),
            "learning_rates": list(self.trace.learning_rates),
            "accuracies": list(self.accuracies),
            "peak_weights": list(self.peak_weights),
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``300 步 | 损失 0.3120 → 0.0041（↓98.7%）| 命中率 0% → 100%``."""
        return (
            f"{self.steps} 步 | 损失 {self.initial_loss:.4f} → {self.final_loss:.4f}"
            f"（↓{self.improvement_ratio:.1%}）| "
            f"命中率 {self.initial_accuracy:.0%} → {self.final_accuracy:.0%} | "
            f"峰值权重 {self.initial_peak_weight:.3f} → {self.final_peak_weight:.3f}"
            f" | 梯度源 {self.gradient_source}"
        )

    def epoch_line(self, index: int) -> str:
        """第 ``index`` 步的一行读数（含三个读数与当步学习率）."""
        if index < 0 or index >= len(self.accuracies):
            raise ParameterError(f"步号 {index} 超出范围 [0, {len(self.accuracies) - 1}]。")
        rate = "—" if index == 0 else f"{self.trace.learning_rates[index - 1]:.6f}"
        return (
            f"step {index:>4} | lr {rate:>9} | loss {self.trace.losses[index]:.6f} | "
            f"命中率 {self.accuracies[index]:.0%} | 峰值 {self.peak_weights[index]:.4f}"
        )


def train_attention(
    initial: AttentionParams,
    tasks: Sequence[InductionTask],
    *,
    optimizer: Optimizer,
    steps: int,
    causal: bool = True,
    source: str = GRADIENT_SOURCE_ANALYTIC,
    schedule: Schedule | None = None,
    trainable: Sequence[str] | None = None,
) -> AttentionTrainingReport:
    """用**解析梯度**（或数值梯度）训练一层自注意力，返回全过程报告.

    与 day074 的 ``optim.minimize`` 的关系值得说清：

    ```text
    minimize        目标函数 + **内部**用数值差分求梯度        —— 适合"不知道梯度公式"的场景
    train_attention 目标函数 + **外部**传入的梯度（本课的七步）—— 适合"梯度已经推出来了"的场景
    ```

    两者用的是同一批 ``Optimizer``（SGD / 动量 / Adam）与同一份 ``TrainingTrace``，
    因此"优化器那一层"完全不需要改动——**这就是 day074 把参数压平与优化器分离的回报**。

    ``trainable`` 可以让这次训练只更新其中几块参数（见 :func:`_resolve_trainable`）。
    """
    checked = _checked_tasks(tasks)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ParameterError(f"steps 必须是 >= 1 的整数，收到 {steps!r}。")
    if source not in GRADIENT_SOURCES:
        raise ParameterError(
            f"不认识的梯度来源 {source!r}：可用取值 {list(GRADIENT_SOURCES)}。"
        )
    checked_trainable = _resolve_trainable(trainable)
    flat, shapes = initial.flatten()
    losses: list[float] = []
    rates: list[float] = []
    history: list[Vector] = [flat]
    peaks: list[float] = []
    accuracies: list[float] = []

    def _record(current: AttentionParams, loss: float) -> None:
        """把这一步的三个读数一起记下来（**必须一起记**，否则会错行）."""
        losses.append(loss)
        peaks.append(batch_mean_peak_weight(current, checked, causal=causal))
        accuracies.append(batch_accuracy(current, checked, causal=causal))

    current = initial
    _record(current, batch_loss(current, checked, causal=causal))
    for index in range(1, steps + 1):
        if schedule is not None:
            optimizer.learning_rate = schedule(index)
            if optimizer.learning_rate <= 0:
                raise ParameterError(f"第 {index} 步的调度给出了非正学习率。")
        grads = batch_gradients(
            current, checked, causal=causal, source=source, trainable=checked_trainable
        )
        rates.append(optimizer.learning_rate)
        flat = optimizer.step(flat, grads)
        current = AttentionParams.unflatten(flat, shapes)
        history.append(flat)
        _record(current, batch_loss(current, checked, causal=causal))
    trace = TrainingTrace(
        losses=tuple(losses),
        learning_rates=tuple(rates),
        params=tuple(history),
        converged=losses[-1] <= 1e-12,
        notes=(
            f"梯度来源：{source}"
            "（analytic = 本课的七步反向；numeric = 数值差分，用于交叉验证）",
            "losses/accuracies/peak_weights 三者长度一致，且第 0 项是'还没训练时'的读数",
            "参数压平之后交给 day074 的 Optimizer：更新公式是逐分量的，与形状无关",
            f"本次训练的参数块：{list(checked_trainable)}"
            f"（冻结 {'、'.join(name for name in PARAMETER_BLOCKS if name not in checked_trainable) or '无'}）",
        ),
    )
    return AttentionTrainingReport(
        trace=trace,
        final_parameters=current,
        peak_weights=tuple(peaks),
        accuracies=tuple(accuracies),
        gradient_source=source,
        trainable=checked_trainable,
        notes=(
            "三个读数一起看：损失（混得多准）、峰值权重（注意力多尖）、命中率（argmax 对不对）",
            "命中率一开始可能为 0：随机初始化下输出的最大分量与目标无关，"
            "因此'训练让它变好'这件事在三个读数上都能看见",
            "只训练 q/k 时注意力会变尖（峰值升到 0.59）而损失几乎不动："
            "'看对了地方'只是必要条件，还要 value/output 把看到的东西映射成目标——"
            "反过来，四块全训能把损失压到 0 而峰值只有 0.44。"
            "**没有任何一个读数能单独说明'学到了什么'**",
        ),
    )


def analytic_objective(
    tasks: Sequence[InductionTask], *, causal: bool = True
) -> Objective:
    """把"这一批样本上的损失"包成一个只吃压平参数的目标函数（调试用）.

    它的用途是**手工核对**：把一个压平向量喂给 ``calculus.gradient``
    就能得到数值梯度，与 :func:`batch_gradients` 的解析结果比一比——
    ``tests/test_attention_training.py`` 里那条"两种梯度源轨迹一致"的断言
    用的就是这两条路。
    """
    checked = _checked_tasks(tasks)

    def objective(flat: Vector) -> float:
        """按固定形状还原参数并算损失."""
        rebuilt = AttentionParams.unflatten(flat, _shape_table(checked[0]))
        return batch_loss(rebuilt, checked, causal=causal)

    return objective


def _shape_table(task: InductionTask) -> tuple[tuple[int, int], ...]:
    """一条样本对应的参数形状表（四个 ``(vocab, vocab)``）."""
    size = task.vocabulary
    return ((size, size), (size, size), (size, size), (size, size))


__all__ = [
    "DEFAULT_INIT_SCALE",
    "GRADIENT_SOURCES",
    "GRADIENT_SOURCE_ANALYTIC",
    "GRADIENT_SOURCE_NUMERIC",
    "PARAMETER_BLOCKS",
    "AttentionTrainingReport",
    "InductionTask",
    "analytic_objective",
    "batch_accuracy",
    "batch_gradients",
    "batch_loss",
    "batch_mean_peak_weight",
    "default_parameters",
    "make_induction_batch",
    "make_induction_task",
    "train_attention",
]
