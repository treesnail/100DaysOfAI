"""把梯度变成一次**真的下降**（day090 / M8-D2）.

到这一章为止，这一课已经有了三样东西：前向（day089）、导数的解析式（``gradients``）、
逐层的回传（``layers`` / ``network``）。它们合起来只回答了一个问题——
"损失对每个参数有多敏感"。今天最后一步回答另一个问题：**沿这个方向走一步，
损失真的会小吗？**

```text
θ ← θ − lr·g          g 来自本包的**解析反向**（不是数值差分，也不是自动微分）
```

## 一、为什么这一步值得单独写一遍

day074 的 ``optim.optimize`` 里已经有一个训练回路，但它的梯度来自**数值差分**：

```text
day074 minimize(objective, ...)   每一步 2n 次函数求值（n = 参数个数）——慢，但不依赖任何推导
day090 train_mlp(...)             每一步 1 次前向 + 1 次反向——快，但依赖"推导是对的"
```

两者的差别不是速度，而是**由谁保证正确性**：

```text
数值差分版    正确性由"尺子"保证（慢，但不会错）
解析反向版    正确性由"对账"保证（快，但必须与尺子比过一次）
```

因此这一课的顺序永远是：**先写反向 → 再与尺子比 → 比过了才敢用它下降**。
没比过就开始训练，会得到一个"损失确实在下降"的模型——而它可能正在往一个错误的方向走，
只是恰好比起点好。

## 二、三条护栏（少任何一条，训练失败都会伪装成"调参没调好"）

```text
护栏一   梯度必须有限      出现 nan / inf 时当场抛 NumericError（不是让参数变成 nan）
护栏二   参数更新后必须有限  一步走完参数成了 inf ⇒ StepError（学习率太大，不是数据坏了）
护栏三   长度必须对得上    压平 / 还原之间长度不符 ⇒ ShapeError，绝不"按顺序切"
```

三条都来自同一个观察：**训练里的错误几乎都不会当场报错**，
它们只会让曲线"看起来不太对"。把这三条写成异常，是让它们能被看见的最便宜的办法。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.backprop import layers as layer_ops
from smart_research_agent.backprop.errors import NumericError, ParameterError, StepError
from smart_research_agent.backprop.network import (
    chain_forward,
    chain_forward_with_cache,
    chain_from_parameters,
    mlp_loss_gradients,
    spec_shapes,
)
from smart_research_agent.backprop.types import LOSS_MSE
from smart_research_agent.math_foundations.optim import Optimizer, make_optimizer
from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.neural_basics.layers import initialize
from smart_research_agent.neural_basics.losses import mse as mse_loss
from smart_research_agent.neural_basics.types import MLPSpec

#: 训练默认的优化器（sgd 是基线，与 day074 的口径相同：没有基线就说不清"优化器有没有用"）.
DEFAULT_OPTIMIZER = "sgd"


@dataclass(frozen=True)
class TrainReport:
    """一次解析反向训练的全过程：每一刻的损失、学习率与梯度范数.

    ```text
    losses            长度 steps + 1（含**初始损失**，否则"下降了没有"无法判断）
    learning_rates    长度 steps（每一步实际用的 lr）
    gradient_norms    长度 steps（每一步的梯度整体范数——它比损失更早暴露问题）
    ```
    """

    losses: tuple[float, ...]
    learning_rates: tuple[float, ...]
    gradient_norms: tuple[float, ...]
    optimizer_name: str = DEFAULT_OPTIMIZER
    converged: bool = False
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if len(self.losses) != len(self.learning_rates) + 1:
            raise NumericError(
                f"损失有 {len(self.losses)} 个而学习率有 {len(self.learning_rates)} 个："
                "学习率比损失**少一个**（初始损失没有对应的学习率）。"
            )
        if len(self.gradient_norms) != len(self.learning_rates):
            raise NumericError(
                f"梯度范数有 {len(self.gradient_norms)} 个而学习率有 {len(self.learning_rates)} 个："
                "两者都是'每一步一个'。"
            )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def steps(self) -> int:
        """实际走过的步数."""
        return len(self.learning_rates)

    @property
    def initial_loss(self) -> float:
        """初始损失（"下降了没有"的参照点）."""
        return self.losses[0]

    @property
    def final_loss(self) -> float:
        """最终损失."""
        return self.losses[-1]

    @property
    def improvement(self) -> float:
        """相对下降比例 ``(L₀ − L_T) / L₀``（初始损失为 0 时记 0.0）."""
        if self.initial_loss == 0.0:
            return 0.0
        return (self.initial_loss - self.final_loss) / abs(self.initial_loss)

    def monotone(self) -> bool:
        """损失是否**一路不升**（更严格的判断：每一步都比上一步小）."""
        return all(
            following <= previous for previous, following in zip(self.losses, self.losses[1:])
        )

    def to_dict(self) -> dict[str, object]:
        """可 json.dumps 的形状."""
        return {
            "steps": self.steps,
            "optimizer": self.optimizer_name,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "improvement": self.improvement,
            "monotone": self.monotone(),
            "converged": self.converged,
            "learning_rates": list(self.learning_rates),
            "gradient_norms": list(self.gradient_norms),
            "losses": list(self.losses),
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``20 步 sgd | 损失 4.520000 → 0.000123（↓100.0%）| 单调 是``."""
        return (
            f"{self.steps} 步 {self.optimizer_name} | 损失 {self.initial_loss:.6f} → "
            f"{self.final_loss:.6f}（↓{self.improvement:.1%}）| "
            f"单调 {'是' if self.monotone() else '否'}"
        )

    def step_line(self, index: int) -> str:
        """第 ``index`` 步的一行读数（含该步的学习率与梯度范数）."""
        if index < 0 or index >= len(self.losses):
            raise ParameterError(f"步号 {index} 超出范围 [0, {len(self.losses) - 1}]。")
        if index == 0:
            return f"step {index:>3} | lr      — | ‖g‖      — | loss {self.losses[index]:.8f}"
        return (
            f"step {index:>3} | lr {self.learning_rates[index - 1]:.6f} | "
            f"‖g‖ {self.gradient_norms[index - 1]:.6e} | loss {self.losses[index]:.8f}"
        )


def _checked_loss(value: float) -> float:
    """一次损失读数必须有限（非有限时当场拒绝，而不是让它一路传到报告里）."""
    if not math.isfinite(value):
        raise NumericError(
            f"训练中出现非有限的损失（{value!r}）："
            "在 inf / nan 上继续下降会得到 nan 参数，根因却在更早的一次除法或 log 上。"
        )
    return float(value)


def train_mlp(
    spec: MLPSpec,
    inputs: Matrix,
    targets: Matrix,
    *,
    steps: int,
    learning_rate: float = 0.1,
    optimizer_name: str = DEFAULT_OPTIMIZER,
    optimizer_params: dict[str, float] | None = None,
    optimizer: Optimizer | None = None,
    tolerance: float = 1e-12,
) -> TrainReport:
    """用**解析反向**跑 ``steps`` 步梯度下降，返回全过程 :class:`TrainReport`.

    ``optimizer`` 可以显式注入（测试用它来构造"参数被推成非有限数"的情形）；
    不注入时按 ``optimizer_name`` 走 day074 的 ``make_optimizer``
    （``sgd`` / ``momentum`` / ``adam``）——本课不重新实现优化器：
    这一课要证明的是"梯度是对的"，而优化器是对错的**无关变量**。
    把它复用过来，恰恰是为了让"损失下降"这件事只可能归因于梯度。

    损失固定为 MSE（这一课只做回归式的最小实验：它让"损失降到了多少"
    有一个可以手算的参照）。
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ParameterError(f"steps 必须是 >= 1 的整数，收到 {steps!r}。")
    if not inputs:
        raise ParameterError("训练集不能为空：没有样本就没有「要拟合的东西」。")
    shapes = spec_shapes(spec)
    activations = tuple(layer.activation for layer in spec.layers)
    flat, _ = layer_ops.flatten_parameters(
        tuple(
            initialize(layer.init, layer.out_features, layer.in_features, seed=layer.seed)
            for layer in spec.layers
        )
    )
    resolved = (
        optimizer
        if optimizer is not None
        else make_optimizer(optimizer_name, learning_rate, **(optimizer_params or {}))
    )

    def loss_of(flat_params: Vector) -> float:
        params = layer_ops.unflatten_parameters(flat_params, shapes)
        chain = chain_from_parameters(params, activations)
        return _checked_loss(mse_loss(chain_forward(chain, inputs), targets))

    losses: list[float] = [loss_of(flat)]
    rates: list[float] = []
    norms: list[float] = []
    for _ in range(steps):
        params = layer_ops.unflatten_parameters(flat, shapes)
        chain = chain_from_parameters(params, activations)
        cache = chain_forward_with_cache(chain, inputs)
        trace = mlp_loss_gradients(cache, targets)
        grads = trace.flat_parameter_gradients()
        if any(not math.isfinite(value) for value in grads):  # pragma: no cover - 防御式分支
            raise NumericError(
                "反向给出的梯度里出现非有限数："
                "它通常来自更早的一次 log(0) 或除以 0，而不是优化器。"
            )
        rates.append(resolved.learning_rate)
        norms.append(math.sqrt(math.fsum(value * value for value in grads)))
        flat = resolved.step(flat, grads)
        if any(not math.isfinite(value) for value in flat):
            raise StepError(
                f"第 {len(rates)} 步之后参数出现非有限数："
                f"这一步的 lr={resolved.learning_rate:g}、‖g‖={norms[-1]:.6e}——"
                "先调小学习率或加梯度裁剪，而不是去查数据。"
            )
        losses.append(loss_of(flat))
    converged = losses[-1] <= tolerance
    return TrainReport(
        losses=tuple(losses),
        learning_rates=tuple(rates),
        gradient_norms=tuple(norms),
        optimizer_name=optimizer_name if optimizer is None else resolved.name,
        converged=converged,
        notes=(
            "梯度来自本包的**解析反向**（Dense 的 dW/db/dx + 逐元素激活的导数），"
            "不是数值差分——数值差分在这一课的角色是**尺子**（见 verify 的第 6 条性质）",
            f"优化器复用 day074 的 {optimizer_name if optimizer is None else resolved.name}："
            "这一课要证明的是'梯度是对的'，优化器是对错的无关变量",
            f"初始损失记在 losses[0]：没有它，'下降了多少'就没有参照点（收敛判据 {tolerance:g}）",
        ),
    )


def evaluate_loss(spec: MLPSpec, inputs: Matrix, targets: Matrix) -> float:
    """按 spec 造一份**初始**参数并算一次 MSE（演示与测试共用的"起点读数"）."""
    params = tuple(
        initialize(layer.init, layer.out_features, layer.in_features, seed=layer.seed)
        for layer in spec.layers
    )
    chain = chain_from_parameters(params, tuple(layer.activation for layer in spec.layers))
    return _checked_loss(mse_loss(chain_forward(chain, inputs), targets))


#: 本课实现的损失名（供报告标注"这一次下降优化的是哪个损失"）.
TRAIN_LOSS = LOSS_MSE

__all__ = [
    "DEFAULT_OPTIMIZER",
    "TRAIN_LOSS",
    "TrainReport",
    "evaluate_loss",
    "train_mlp",
]
