"""``training_optim`` 的三个控制器：调度、裁剪、早停（day081 / M7-D6）.

三个控制器各自只做一件事，而它们都是**判据**而不是结果：

```text
learning_rate_at  第 t 步的学习率是多少      ——交给 day074 的调度函数，本包不重写公式
clip_gradients    第 t 步的梯度该不该缩短     ——交给 day074 的整体范数裁剪，本包只加读数
EarlyStopping     第 t 步之后该不该停         ——本包新增：它需要一个"等了多少步"的计数器
```

## 为什么调度与裁剪是转发而不是重写

day074 的 ``math_foundations.optim`` 里已经有四种调度与两种裁剪，而它们的公式
在那一课被逐条推过、被逐项测过。今天重写一遍的代价不是"多写 30 行"，
而是**两处口径会慢慢分开**：某一天有人在一边改了公式，而另一边还是旧的，
于是"调度"这个词在两个地方指两件事。

因此本包只做三件事，且都能被测试逐位核对：

```text
learning_rate_at   把 config 里的名字与参数交给 make_schedule，并把它变成"第 t 步的值"
clip_gradients     把 config 里的阈值交给 clip_by_global_norm，并**把读数留下来**
EarlyStopping      新增（day074 没有"过程级"的判据）
```

## 早停的两条纪律

```text
① "变好"必须带一个 min_delta     ——浮点噪声级别的"变好"不是变好（否则早停永不触发）
② 它记录 best_step 而不是 last_step ——"最好的一步"才是该保存的那份参数
```

第 ② 条有一个具体的后果：**训练结束时手上的参数可能不是最好的那一份**。
EarlyStopping 因此把 ``best_step`` 与 ``best_loss`` 一起交给报告，
而"要不要回滚到那一步"是调用方的决定（本课不回滚，只把这件事写进报告）。
"""

from __future__ import annotations

import math
from typing import Any

from smart_research_agent.math_foundations.optim import (
    Optimizer,
    clip_by_global_norm,
    global_norm,
    make_optimizer,
    make_schedule,
)
from smart_research_agent.math_foundations.types import Vector, validate_vector
from smart_research_agent.training_optim.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.training_optim.types import (
    ClipReport,
    EarlyStopReport,
    TrainingConfig,
    _checked_non_negative_float,
    _checked_positive_int,
)


def learning_rate_at(config: TrainingConfig, step: int) -> float:
    """第 ``step`` 步的学习率（**转发 day074 的调度函数**）.

    步号是 1-based（与 ``math_foundations.optim`` 的纪律一致）：
    第 1 步就是第一次更新。0-based 与 1-based 混用时"热身少一步、退火早一步"，
    而两条曲线都看起来正常——这条边界在 day074 已经被写下来过。

    学习率允许是 **0**：``warmup_cosine`` 的 ``min_lr`` 默认是 ``0.0``，
    因此退火的**最后一步**拿到 0 是它故意的端点（那一步不做更新）。
    本包只拒绝负数与非有限数——把 0 也拒掉会让"退火到 0"这个合法配置无法跑完。
    """
    if not isinstance(config, TrainingConfig):
        raise ParameterError(f"config 必须是 TrainingConfig，收到 {type(config).__name__}。")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ParameterError(f"step 必须是 >= 1 的整数，收到 {step!r}。")
    try:
        schedule = make_schedule(
            config.schedule, base_lr=config.learning_rate, **config.schedule_kwargs
        )
        value = float(schedule(step))
    except TypeError as error:
        # 注意：缺参数这件事**不在**造曲线时暴露——``make_schedule`` 只是把参数
        # 记进 lambda，真正的调用发生在 ``schedule(step)`` 那一刻。
        # 第一版只把 ``make_schedule`` 包进 try，于是报出来的是一个原始 TypeError 栈。
        raise ParameterError(
            f"调度 {config.schedule!r} 的参数对不上：{error}——"
            "本包不替调用方补默认参数（那样'这一次到底用了什么'就不可读了）。"
        ) from error
    if not math.isfinite(value) or value < 0.0:
        raise NumericError(
            f"第 {step} 步的学习率是 {value!r}：调度必须给出**非负的有限数**——"
            "负数会把下降变成上升，而它不会报错，只会让损失一路变大。"
        )
    return value


def build_optimizer(config: TrainingConfig) -> Optimizer:
    """按配置造一个优化器（转发 ``make_optimizer``）."""
    if not isinstance(config, TrainingConfig):
        raise ParameterError(f"config 必须是 TrainingConfig，收到 {type(config).__name__}。")
    return make_optimizer(
        config.optimizer, config.learning_rate, **config.optimizer_kwargs
    )


def clip_gradients(grads: Vector, config: TrainingConfig) -> tuple[Vector, ClipReport]:
    """按配置裁剪梯度，返回 ``(裁剪后的梯度, 读数)``.

    没开裁剪时也返回一份读数（缩放 1.0、阈值 ``None``）——
    **"没有裁剪"与"裁剪了但没生效"必须是两个可读的状态**：
    只返回梯度的话，两者看起来完全一样。
    """
    checked = validate_vector(grads, name="grads")
    if not isinstance(config, TrainingConfig):
        raise ParameterError(f"config 必须是 TrainingConfig，收到 {type(config).__name__}。")
    length = global_norm(checked)
    if config.max_norm is None:
        return checked, ClipReport(original_norm=length, max_norm=None, scale=1.0)
    clipped, scale = clip_by_global_norm(checked, config.max_norm)
    return clipped, ClipReport(original_norm=length, max_norm=config.max_norm, scale=scale)


def direction_cosine(before: Vector, after: Vector) -> float:
    """裁剪前后两个向量的余弦（**整体范数裁剪应当给出 1.0**）.

    这个读数是"裁剪只改长度、不改方向"这条性质的证据：两种情况都成立：

    ```text
    ‖g‖ <= max_norm    余弦 = 1（两个向量逐位相同，cos 的 0/0 由定义补成 1）
    ‖g‖ >  max_norm    余弦 = 1（整体乘一个正标量）
    ```

    而逐分量裁剪（``clip_by_value``）会在这里给出小于 1 的数——
    day074 已经量过那个差别（``(100, 1)`` 裁到 ``(1, 1)`` 之后方向从"几乎水平"变成 45°）。
    """
    checked_before = validate_vector(before, name="before")
    checked_after = validate_vector(after, name="after")
    if len(checked_before) != len(checked_after):
        raise ShapeError(
            f"两个向量的长度必须相同：{len(checked_before)} 与 {len(checked_after)}。"
        )
    left = math.sqrt(math.fsum(value * value for value in checked_before))
    right = math.sqrt(math.fsum(value * value for value in checked_after))
    if left == 0.0 or right == 0.0:
        return 1.0
    dot = math.fsum(a * b for a, b in zip(checked_before, checked_after, strict=True))
    return dot / (left * right)


class EarlyStopping:
    """早停：连续 ``patience`` 步没有变好 ``min_delta`` 以上，就建议停下.

    ```text
    update(step, loss) -> bool      返回"该不该停"（**它只回答这个**）
    report()                        把"为什么停在这里"交出去
    ```

    "变好"的判据是 ``loss < best_loss − min_delta``：

    ```text
    min_delta = 0     任何一点下降都算变好 —— 浮点噪声会被当成进展
    min_delta > 0     要求"明显"变好（本课默认 0，而把这件事写下来）
    ```
    """

    def __init__(self, patience: int, *, min_delta: float = 0.0) -> None:
        self.patience = _checked_positive_int(patience, name="patience")
        self.min_delta = _checked_non_negative_float(min_delta, name="min_delta")
        self.best_step: int | None = None
        self.best_loss: float | None = None
        self.waited = 0
        self.triggered = False

    def update(self, step: int, loss: float) -> bool:
        """喂一步读数，返回"该不该停"（一旦触发过就一直是 ``True``）."""
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise ParameterError(f"step 必须是 >= 1 的整数，收到 {step!r}。")
        if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
            raise NumericError(f"loss 必须是有限数，收到 {loss!r}。")
        value = float(loss)
        if self.best_loss is None or value < self.best_loss - self.min_delta:
            self.best_step = step
            self.best_loss = value
            self.waited = 0
            return False
        self.waited += 1
        if self.waited >= self.patience:
            self.triggered = True
        return self.triggered

    def report(self) -> EarlyStopReport:
        """把早停的读数交出去（**没有喂过任何一步时不许造报告**）."""
        if self.best_step is None or self.best_loss is None:
            raise ParameterError(
                "还没有喂过任何一步：早停报告里的 best_step 会是一个凭空的数。"
            )
        return EarlyStopReport(
            patience=self.patience,
            best_step=self.best_step,
            best_loss=self.best_loss,
            waited=self.waited,
            triggered=self.triggered,
            min_delta=self.min_delta,
        )

    def state(self) -> dict[str, Any]:
        """可 json.dumps 的状态（只含跨步保留的东西）."""
        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_step": self.best_step,
            "best_loss": self.best_loss,
            "waited": self.waited,
            "triggered": self.triggered,
        }

    def describe(self) -> str:
        """一行说明."""
        if self.best_step is None or self.best_loss is None:
            return f"早停（耐心 {self.patience}）：还没有喂过一步"
        return (
            f"早停（耐心 {self.patience}，min_delta {self.min_delta:g}）："
            f"最好第 {self.best_step} 步（{self.best_loss:.6f}），已等 {self.waited} 步"
        )


__all__ = [
    "EarlyStopping",
    "build_optimizer",
    "clip_gradients",
    "direction_cosine",
    "learning_rate_at",
]
