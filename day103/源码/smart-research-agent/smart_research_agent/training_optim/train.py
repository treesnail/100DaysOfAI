"""``training_optim`` 的训练循环：把一条链更新 T 次（day081 / M7-D6）.

## 今天的全部新增代码：**一行**

```text
前向     layer_i 的账算完之后，把它的**输出**过一次 dropout 再交给下一层
反向     每一层反传之前，先把梯度**穿过那一层的掩码**
```

其余部分**原封不动**地调用 day079/080 的 ``block_attention`` / ``encoder_block`` /
``encoder_block_backward``——因为"加一个 dropout"不该顺手把九个梯度公式重写一遍。

```python
# 前向（每一层多一行）
forward = encoder_block(params_i, current, attention, ...)
current, mask_i, scale_i = dropout_forward(forward.output, rate=..., seed=...)

# 反向（每一层多一行，注意顺序：**先穿掩码，再反传**）
current = dropout_backward(current, mask_i, scale_i)
layer_grads = encoder_block_backward(layer.block, layer.params, current)
current = layer_grads.grad_inputs
```

## 一条必须写下来的边界：本课把 dropout 放在**层与层之间**

```text
经典 Transformer   每个子层之后、残差相加之前各放一次（块**内部**）
本课               层与层之间放一次（块**外部**）
```

这不是等价的做法，而是一个**刻意的取舍**：块内部的 dropout 要让掩码穿过
``encoder_block_backward`` 的九个公式（``dResidual1 = dOutput + dNorm2`` 那一项
也要先乘掩码），而那样一来"掩码该在哪一步乘"就和"九个梯度公式"混在一起了。
放在层与层之间，**"掩码必须穿过反向"这件事被孤立成一件事**——

```text
它错了会怎样   形状全对、也真的在跑，只是梯度少了一次逐元素的缩放
谁会抓到它     drop-out 掩码被固定之后，梯度**可以**被数值差分逐项核对（第 7 节）
```

## 三条口实：这一课训的是什么

```text
训的       块参数（前馈 + 两个 LN 的 γ/β），向量长度 = N × 342 = 1368
不训       注意力那四个投影（day079 起就是"给定函数"）
损失       MSE（与 day079/080 同一个函数，因此两条曲线可比）
```

## 每步换一个掩码，而它由 ``(seed, step)`` 唯一决定

```text
同一份 config 跑两次      每一步的损失**逐位**相同（第 6 节的性质 1）
换一个 step              掩码不同 ⇒ 这才是 dropout，而不是"固定丢掉同一批单元"
```
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.encoder_decoder.layers import (
    block_attention,
    encoder_block,
    encoder_block_backward,
)
from smart_research_agent.encoder_decoder.types import (
    ACTIVATION_RELU,
    NORM_PRE,
    BlockGradients,
)
from smart_research_agent.math_foundations.calculus import DEFAULT_STEP
from smart_research_agent.math_foundations.optim import global_norm
from smart_research_agent.math_foundations.types import Matrix, Vector, validate_matrix
from smart_research_agent.transformer_core.layers import mean_squared_error, mse_gradient
from smart_research_agent.transformer_stack import (
    StackGradients,
    StackParameters,
    StackShape,
    StackedLayer,
    stack_forward,
)
from smart_research_agent.training_optim.controls import (
    EarlyStopping,
    build_optimizer,
    clip_gradients,
    learning_rate_at,
)
from smart_research_agent.training_optim.dropout import (
    PHASE_EVAL,
    PHASE_TRAIN,
    dropout_backward,
    dropout_forward,
    expected_keep,
    kept_fraction,
)
from smart_research_agent.training_optim.errors import (
    DivergenceError,
    NumericError,
    ParameterError,
    ShapeError,
    TrainingError,
)
from smart_research_agent.training_optim.init import initialize_parameters
from smart_research_agent.training_optim.types import (
    EpochRecord,
    TrainingConfig,
    TrainingCurve,
    check_no_divergence,
)

#: 逐层的掩码种子步长（第 i 层用 ``seed + i * 31``；步与步之间再乘一个步长）.
LAYER_MASK_STRIDE = 31
#: 步与步之间的掩码种子步长（**每步换一个掩码**，否则那不是 dropout）.
STEP_MASK_STRIDE = 1009


@dataclass(frozen=True)
class StepCache:
    """一步前向留下的账：每一环的账 + 每一层的掩码与缩放.

    ``masks`` 与 ``layers`` 同序（下标即层号）。反向要用它们，
    而**把它们记下来**这件事本身就是"梯度可被校验"的前提（见 :mod:`.dropout`）。
    """

    output: Matrix
    layers: tuple[StackedLayer, ...]
    masks: tuple[Matrix, ...]
    scales: tuple[float, ...]
    kept: float

    def __post_init__(self) -> None:
        validate_matrix(self.output, name="output")
        if not (len(self.layers) == len(self.masks) == len(self.scales)):
            raise ShapeError(
                f"账 {len(self.layers)} 环、掩码 {len(self.masks)} 份、缩放 "
                f"{len(self.scales)} 份：三者必须逐层对齐。"
            )
        if not 0.0 <= float(self.kept) <= 1.0:
            raise ParameterError(f"kept 必须落在 [0, 1]，收到 {self.kept!r}。")

    @property
    def depth(self) -> int:
        """层数."""
        return len(self.layers)

    def summary_line(self) -> str:
        """一行说明."""
        return f"{self.depth} 层 | 平均保留比例 {self.kept:.4f}"


def mask_seed(config: TrainingConfig, step: int, layer: int) -> int:
    """第 ``step`` 步、第 ``layer`` 层的掩码种子（**由两个数唯一决定 ⇒ 可复现**）."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ParameterError(f"step 必须是 >= 1 的整数，收到 {step!r}。")
    if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
        raise ParameterError(f"layer 必须是非负整数，收到 {layer!r}。")
    return int(config.seed) + step * STEP_MASK_STRIDE + layer * LAYER_MASK_STRIDE


def training_forward(
    params: StackParameters,
    inputs: Matrix,
    *,
    config: TrainingConfig,
    step: int = 1,
    phase: str = PHASE_TRAIN,
    activation: str = ACTIVATION_RELU,
    placement: str = NORM_PRE,
) -> StepCache:
    """一次带 dropout 的链式前向（**``dropout = 0`` 时与 ``stack_forward`` 逐位一致**）.

    与 day080 的 ``stack_forward`` 只差一行：每一层的输出先过一次 dropout 再往下传。
    ``dropout = 0`` 时那一行退化成逐元素乘 ``1.0``，因此两边逐位相同——
    这条等式是一条可失败的断言（性质 2），也是"我们训的是同一个模型"的唯一证据。
    """
    checked = validate_matrix(inputs, name="inputs")
    layers: list[StackedLayer] = []
    masks: list[Matrix] = []
    scales: list[float] = []
    kepts: list[float] = []
    current = checked
    for index, (block_params, attention_params) in enumerate(
        zip(params.blocks, params.attentions, strict=True)
    ):
        attention = block_attention(block_params, current, attention_params, placement=placement)
        forward = encoder_block(
            block_params,
            current,
            attention,
            placement=placement,
            use_residual=True,
            activation=activation,
        )
        dropped, mask, scale = dropout_forward(
            forward.output,
            rate=config.dropout,
            seed=mask_seed(config, step, index),
            phase=phase,
        )
        layers.append(
            StackedLayer(index=index, block=forward, attention=attention, params=block_params)
        )
        masks.append(mask)
        scales.append(scale)
        kepts.append(kept_fraction(mask))
        current = dropped
    mean_kept = sum(kepts) / len(kepts)
    return StepCache(
        output=current,
        layers=tuple(layers),
        masks=tuple(masks),
        scales=tuple(scales),
        kept=mean_kept,
    )


def training_backward(
    cache: StepCache,
    params: StackParameters,
    grad_output: Matrix,
    *,
    activation: str = ACTIVATION_RELU,
) -> StackGradients:
    """链式反向 + 每一层先穿过它自己的掩码（**顺序不能反**）.

    ```text
    对的     current = dropout_backward(current, mask_i, scale_i)
             layer_grads = encoder_block_backward(账_i, 参数_i, current)
    错的     先反传、再穿掩码   —— 掩码乘在了"已经穿过九个公式"的梯度上
    ```

    错法在 ``dropout = 0`` 时**完全正确**（掩码是全 1），
    因此它只在 `rate > 0` 时才暴露——这正是"掩码固定之后梯度可校验"要抓的东西。
    """
    checked_grad = validate_matrix(grad_output, name="grad_output")
    if cache.depth != params.layers:
        raise ShapeError(
            f"账里有 {cache.depth} 层，而参数有 {params.layers} 份："
            "反向必须与产生这份账的那次前向用同一摞参数。"
        )
    grads: list[BlockGradients | None] = [None] * cache.depth
    current = checked_grad
    for index in reversed(range(cache.depth)):
        layer = cache.layers[index]
        current = dropout_backward(current, cache.masks[index], cache.scales[index])
        layer_grads = encoder_block_backward(layer.block, layer.params, current, activation=activation)
        grads[index] = layer_grads
        current = layer_grads.grad_inputs
    resolved: list[BlockGradients] = []
    for index, item in enumerate(grads):
        if item is None:  # pragma: no cover - 循环长度由 depth 决定，走不完才可能触发
            raise ParameterError(f"第 {index} 层的梯度没有被算到。")
        resolved.append(item)
    return StackGradients(
        shape=params.shape_for(cache.layers[0].block.shape.tokens),
        grad_inputs=resolved[0].grad_inputs,
        layers=tuple(resolved),
        notes=(
            "每一层先穿过自己的掩码，再调用 encoder_block_backward",
            "dropout = 0 时这一趟与 day080 的 stack_backward 逐位一致",
        ),
    )


def flatten_block_gradients(grads: StackGradients, params: StackParameters) -> Vector:
    """把逐层梯度压成**与 ``params.flatten()`` 同序**的一串数.

    ``BlockGradients.flatten()`` 的顺序是"八块参数 + 输入梯度"，
    而 ``BlockParameters.flatten()`` 只有那八块——因此这里取**前**
    ``block_parameter_count`` 个数即可。这个"取前 N 个"不是随手写的：

    ```text
    八块的顺序两边来自同一张表（BLOCK_GRADIENT_TARGETS），输入梯度排在最后
    ⇒ 前 342 个数逐位对应那一层的八块参数
    ```
    """
    if not isinstance(grads, StackGradients):
        raise ParameterError(f"grads 必须是 StackGradients，收到 {type(grads).__name__}。")
    if grads.depth != params.layers:
        raise ShapeError(f"梯度账 {grads.depth} 层与参数 {params.layers} 份不一致。")
    per_layer = params.block_parameter_count // params.layers
    flat: list[float] = []
    for index in range(grads.depth):
        flat.extend(grads.layer_at(index).flatten()[:per_layer])
    if len(flat) != params.block_parameter_count:  # pragma: no cover - 由上面的切片保证
        raise NumericError(
            f"压平后的梯度有 {len(flat)} 个数，而参数有 {params.block_parameter_count} 个。"
        )
    return tuple(flat)


def step_loss(
    params: StackParameters,
    inputs: Matrix,
    target: Matrix,
    *,
    config: TrainingConfig,
    step: int,
    phase: str,
    placement: str = NORM_PRE,
    activation: str = ACTIVATION_RELU,
) -> tuple[float, StepCache]:
    """一步的损失与账（损失口径与 day079/080 **同一个函数**）.

    ``placement`` 必须一路传下来：第一版漏了它，于是"post-LN"那几行
    实际上跑的是 pre-LN，而**它们看起来完全正常**（连读数都一模一样）——
    两条本该不同的曲线重合成一条，是这类漏传最典型的信号。
    """
    cache = training_forward(
        params,
        inputs,
        config=config,
        step=step,
        phase=phase,
        placement=placement,
        activation=activation,
    )
    return mean_squared_error(cache.output, target), cache


def _ensure_finite(values: Vector, *, what: str, step: int, learning_rate: float) -> None:
    """状态护栏：**参数或梯度一旦出现非有限数，就把它归入发散族**.

    这条护栏是被一次真实的失败逼出来的：学习率放到 ``0.8`` 时，
    参数还在（看起来）有限的范围内，而**梯度已经是 ``inf``**——
    于是 ``clip_gradients`` 里的数值守卫先抛 ``NumericError``，
    报出来的信息是"original_norm 必须是非负有限数"，而不是"这一步的步长太大"。

    ```text
    NumericError     调用方给的数据有问题（该改数据）
    DivergenceError  过程本身走坏了（该改超参）——哪怕它表现为一个 inf
    ```

    按"该谁去修"分族，这里的 inf 属于后者，因此本函数把它转成 ``DivergenceError``。
    """
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise DivergenceError(
                f"训练在第 {step} 步让{what}变成了非有限数（第 {index} 个分量是 {value!r}）："
                f"学习率 {learning_rate:g} 太大，参数被推到了浮点范围之外。"
                "这与'调用方给的数据有问题'不是一回事——修法是改超参。"
            )


def _ensure_finite_norm(values: Vector, *, step: int, learning_rate: float) -> float:
    """梯度范数的护栏：**分量都有限、而它们的平方和已经溢出**时也归入发散族.

    这一条是第二次被真实失败逼出来的：``0.8`` 的学习率下，每一个梯度的分量
    仍然是有限数，而 ``Σ g²`` 已经溢出成 ``inf``——于是报错来自 ``ClipReport``
    的数值守卫（"original_norm 必须是非负有限数"），而不是"这一步太长"。

    两个护栏形状不同、抓的也是不同的东西：

    ```text
    逐分量检查    某个分量本身是 inf/nan（参数被推到浮点范围之外）
    范数检查      分量都有限，但**规模**已经大到平方和溢出
    ```
    """
    length = global_norm(values)
    if not math.isfinite(length):
        raise DivergenceError(
            f"训练在第 {step} 步的梯度范数溢出成 {length!r}："
            f"学习率 {learning_rate:g} 太大——每一个分量都还是有限数，"
            "而它们的平方和已经越过了浮点的范围。修法同样是改超参。"
        )
    return length


def train(
    shape: StackShape,
    inputs: Matrix,
    target: Matrix,
    *,
    config: TrainingConfig | None = None,
    params: StackParameters | None = None,
    activation: str = ACTIVATION_RELU,
    placement: str = NORM_PRE,
) -> TrainingCurve:
    """把一条链训练 ``config.steps`` 步，返回**每一步的读数**.

    流程与任何训练循环一样，而三处顺序值得点出：

    ```text
    ① 学习率先取、再更新      第 t 步用的是调度在 t 上的值（不是 t−1 的）
    ② 裁剪在优化器之前        优化器拿到的必须是**裁剪后**的梯度
    ③ 评估用**推理相**        每一步都再跑一次无 dropout 的前向，得到 eval_loss
                            ——只看训练损失会让 dropout 看起来有害（记录里两个口径都给）
    """
    resolved_config = TrainingConfig() if config is None else config
    if not isinstance(resolved_config, TrainingConfig):
        raise ParameterError(
            f"config 必须是 TrainingConfig，收到 {type(resolved_config).__name__}。"
        )
    checked_inputs = validate_matrix(inputs, name="inputs")
    checked_target = validate_matrix(target, name="target")
    if resolved_config.dropout > 0.0 and expected_keep(resolved_config.dropout) >= 1.0:
        raise ParameterError("dropout 必须小于 1：整层全丢的模型没有可学的东西。")
    current = (
        initialize_parameters(
            shape,
            scheme=resolved_config.init_scheme,
            seed=resolved_config.seed,
            scale=resolved_config.init_scale,
        )
        if params is None
        else params
    )
    flat, shapes = current.flatten()
    optimizer = build_optimizer(resolved_config)
    stopper = (
        EarlyStopping(resolved_config.patience, min_delta=resolved_config.min_delta)
        if resolved_config.patience is not None
        else None
    )
    records: list[EpochRecord] = []
    stop_report = None
    for step in range(1, resolved_config.steps + 1):
        learning_rate = learning_rate_at(resolved_config, step)
        optimizer.learning_rate = learning_rate
        try:
            train_loss, cache = step_loss(
                current,
                checked_inputs,
                checked_target,
                config=resolved_config,
                step=step,
                phase=PHASE_TRAIN,
                placement=placement,
                activation=activation,
            )
            grads = training_backward(
                cache, current, mse_gradient(cache.output, checked_target), activation=activation
            )
            flat_grads = flatten_block_gradients(grads, current)
            _ensure_finite(flat_grads, what="梯度分量", step=step, learning_rate=learning_rate)
            _ensure_finite_norm(flat_grads, step=step, learning_rate=learning_rate)
            clipped, clip_report = clip_gradients(flat_grads, resolved_config)
            flat = optimizer.step(flat, clipped)
            _ensure_finite(flat, what="参数", step=step, learning_rate=learning_rate)
            current = StackParameters.unflatten(flat, shapes, current.attentions)
            eval_loss, _eval_cache = step_loss(
                current,
                checked_inputs,
                checked_target,
                config=resolved_config,
                step=step,
                phase=PHASE_EVAL,
                placement=placement,
                activation=activation,
            )
        except OverflowError as error:
            # 学到的边界之一：参数被推到太大之后，**数值溢出会先于发散判据发生**
            # （Python 的 float 会抛 OverflowError，而 payload 里已经全是 inf）。
            # 按"该谁去修"分族，它属于 DivergenceError：每一步都合法，
            # 是这一步的步长让整段过程不收敛——修法是改超参（学习率/裁剪/热身）。
            raise DivergenceError(
                f"训练在第 {step} 步数值溢出（{error}）：学习率 {learning_rate:g} 太大，"
                "参数被推到了浮点范围之外——**溢出发生在发散判据之前**，"
                "因此这里把它归入同一个族（修法是改超参，不是改数据）。"
            ) from error
        except ValueError as error:
            # 学到的边界之二：``math.fsum`` 在遇到 ``-inf + inf`` 时抛的是
            # **``ValueError``**（"−inf + inf in fsum"），而不是 OverflowError。
            # 但它同时也是本包自己那五族的基类（``TrainingError`` 继承 ``ValueError``），
            # 因此这里必须先把"我们自己的错"放过去——否则真 bug 会被改写成发散。
            if isinstance(error, TrainingError):
                raise
            raise DivergenceError(
                f"训练在第 {step} 步让数值失去意义（{error}）：学习率 {learning_rate:g} "
                "太大——反向里已经出现 ±inf 的组合（fsum 拒绝把 −inf 与 +inf 相加）。"
                "这是一个**非本包**的 ValueError，因此按'该谁去修'归入发散族。"
            ) from error
        records.append(
            EpochRecord(
                step=step,
                loss=train_loss,
                eval_loss=eval_loss,
                learning_rate=learning_rate,
                grad_norm=clip_report.original_norm,
                clip_scale=clip_report.scale,
                kept_fraction=cache.kept,
            )
        )
        if stopper is not None and stopper.update(step, train_loss):
            stop_report = stopper.report()
            break
    curve = TrainingCurve(
        config=resolved_config,
        shape=shape,
        records=tuple(records),
        early_stop=stop_report,
        notes=(
            "loss 是训练相（带 dropout）、eval_loss 是推理相（无 dropout）",
            "训练的是块参数（前馈 + 两个 LN 的 γ/β），注意力是给定函数",
        ),
    )
    check_no_divergence(curve)
    return curve


def forward_output_at(
    params: StackParameters,
    inputs: Matrix,
    *,
    config: TrainingConfig,
    phase: str = PHASE_EVAL,
    activation: str = ACTIVATION_RELU,
    placement: str = NORM_PRE,
) -> Matrix:
    """一次前向的输出（``phase`` 决定要不要 dropout；**推理相与 day080 的链逐位一致**）."""
    return training_forward(
        params,
        inputs,
        config=config,
        step=1,
        phase=phase,
        activation=activation,
        placement=placement,
    ).output


def plain_stack_output(
    params: StackParameters,
    inputs: Matrix,
    *,
    activation: str = ACTIVATION_RELU,
    placement: str = NORM_PRE,
) -> Matrix:
    """day080 的那条链的输出（**对照用**：不带 dropout、逐位口径唯一）."""
    return stack_forward(
        params, inputs, placement=placement, use_residual=True, activation=activation
    ).output


def numerical_training_input_gradient(
    params: StackParameters,
    inputs: Matrix,
    target: Matrix,
    *,
    config: TrainingConfig,
    step: int = 1,
    placement: str = NORM_PRE,
    activation: str = ACTIVATION_RELU,
    step_size: float = DEFAULT_STEP,
) -> Matrix:
    """带 dropout 的那条链的**输入梯度**（中心差分，掩码被固定）.

    这是"掩码固定 ⇒ 梯度可校验"那句话的实现：掩码由 ``(seed, step, layer)``
    唯一决定，因此在一次数值差分里它是一个**常数矩阵**——整个算子退化成
    "逐元素乘以已知常数"，于是中心差分给出的答案必须与解析梯度逐项对上。

    它同时是"每一层先穿掩码、再反传"那一行代码的**唯一证据**：
    顺序写反时这个检查会亮红（而 ``dropout = 0`` 时它永远通过）。
    """
    from smart_research_agent.math_foundations.calculus import gradient
    from smart_research_agent.math_foundations.optim import (
        flatten_matrices,
        unflatten_matrices,
    )

    checked_inputs = validate_matrix(inputs, name="inputs")
    flat_inputs, input_shapes = flatten_matrices((checked_inputs,))

    def objective(theta: Vector) -> float:
        """给定压平后的输入，返回带 dropout 的损失（掩码由 step 固定）."""
        rebuilt = unflatten_matrices(theta, input_shapes)[0]
        return step_loss(
            params,
            rebuilt,
            target,
            config=config,
            step=step,
            phase=PHASE_TRAIN,
            placement=placement,
            activation=activation,
        )[0]

    numeric = gradient(objective, flat_inputs, step=step_size)
    return unflatten_matrices(numeric, input_shapes)[0]


def curve_summary(curve: TrainingCurve) -> dict[str, Any]:
    """把一条曲线压成一个字典（演示脚本并排打印多条曲线时用）."""
    if not isinstance(curve, TrainingCurve):
        raise ParameterError(f"curve 必须是 TrainingCurve，收到 {type(curve).__name__}。")
    return {
        "config": curve.config.summary_line(),
        "steps": len(curve.records),
        "first_loss": curve.first_loss,
        "best_loss": curve.best_loss,
        "best_step": curve.best_step,
        "last_loss": curve.last_loss,
        "improvement_ratio": curve.improvement_ratio,
        "mean_gap": curve.mean_gap(),
        "diverged": curve.diverged,
    }


__all__ = [
    "LAYER_MASK_STRIDE",
    "STEP_MASK_STRIDE",
    "StepCache",
    "curve_summary",
    "flatten_block_gradients",
    "forward_output_at",
    "mask_seed",
    "plain_stack_output",
    "step_loss",
    "train",
    "training_backward",
    "training_forward",
]
