"""``device``：把"参数量 / 批大小 / 宽度 / 精度"换成字节数，并与设备预算对账（day096 / M8-D7）.

```text
tensor_bytes(shape, dtype)           bytes = (Π shape_i) × DTYPE_BYTES[dtype]
parameter_bytes(count, dtype)        bytes = count × DTYPE_BYTES[dtype]
activation_bytes(batch, width, dtype) bytes = batch × width × DTYPE_BYTES[dtype]
plan_device(params, …)               params + activations + optimizer_multiplier × params
```

## 一、"装不下"不是异常，而是一个读数

```text
放不进预算   ⇒  device_plan.fits is False，并给出一行 warn_line()
             ⇒ **不抛异常**：它是可以换批大小 / 换精度解决的工程选择
```

把它做成异常会让"先跑一份小配置看看"这件事变成"必须先算准预算"；
而预算本身是**算**出来的估计，不是一个硬约束。

## 二、未知设备 / 未知精度才抛 ``DeviceError``

```text
设备名与精度名都是**白名单**：cuda / cpu / mps、float32 / float64 / float16 / bfloat16。
把它们写错（``CUDA `` / ``float32x``）不是"数值范围"问题、也不是"形状"问题——
它是"这台机器上没有你要的东西"，因此独立成本包的 :class:`DeviceError`。
```

## 三、这一层**不探测硬件**

它一次 ``import torch`` 都不做、一次 ``cuda`` 查询都不发：所有读数都是**算术**，
因此可以在任何机器上逐位复算。这也正是"设备预算"进得了性质表的理由。
"""

from __future__ import annotations

import math

from dataclasses import dataclass

from smart_research_agent.torch_pipeline.errors import (
    DeviceError,
    ParameterError,
)
from smart_research_agent.torch_pipeline.types import (
    DEVICE_CPU,
    DEVICE_KINDS,
    DEVICE_PREFIXES,
    DTYPE_BYTES,
    DTYPE_FLOAT32,
)

#: 缺省的设备预算（字节）：**纯算术表**，不是运行时探测值.
#:
#: 它们足够大，所以本课的小模型一定 ``fits``；要看不 ``fits`` 的一行，
#: 要么把 ``batch_size`` / ``width`` 调大，要么显式传一个更小的 ``budget_bytes``。
DEFAULT_DEVICE_BUDGETS: dict[str, int] = {
    DEVICE_CPU: 8 * 1024**3,
    "cuda": 4 * 1024**3,
    "mps": 2 * 1024**3,
}


def _checked_dtype(dtype: object) -> str:
    """精度名必须在 :data:`types.DTYPE_BYTES` 里（**白名单**，写错抛 ``DeviceError``）."""
    if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
        raise DeviceError(
            f"未知的精度 {dtype!r}：可用取值 {sorted(DTYPE_BYTES)}——"
            "精度名是一张白名单，写错了不该被当成'某个数值越界'。"
        )
    return dtype


def _checked_device(device: object) -> str:
    """设备名必须在 :data:`types.DEVICE_KINDS` 里（**白名单**，写错抛 ``DeviceError``）."""
    if not isinstance(device, str) or device not in DEVICE_KINDS:
        raise DeviceError(
            f"未知的设备 {device!r}：可用取值 {list(DEVICE_KINDS)}——"
            "本课只做设备与预算的**算术**，不做硬件探测。"
        )
    return device


def _checked_count(value: object, *, name: str, minimum: int = 0) -> int:
    """非负整数校验（布尔不是整数）."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError(f"{name} 必须是整数，收到 {value!r}。")
    if value < minimum:
        raise ParameterError(f"{name} 必须 >= {minimum}，收到 {value}。")
    return int(value)


def tensor_bytes(shape: tuple[int, ...], *, dtype: str = DTYPE_FLOAT32) -> int:
    """一个张量的字节数：``(Π shape_i) × DTYPE_BYTES[dtype]``.

    ``shape`` 是一串非负整数（标量的空 shape 记 1 个元素）。
    """
    checked_dtype = _checked_dtype(dtype)
    if isinstance(shape, (str, bytes)) or not hasattr(shape, "__iter__"):
        raise ParameterError(f"shape 必须是可迭代的整数串，收到 {type(shape).__name__}。")
    elements = 1
    for index, dimension in enumerate(shape):
        elements *= _checked_count(dimension, name=f"shape[{index}]")
    return elements * DTYPE_BYTES[checked_dtype]


def parameter_bytes(count: int, *, dtype: str = DTYPE_FLOAT32) -> int:
    """``count`` 个参数的字节数（每参数一个元素）."""
    checked_dtype = _checked_dtype(dtype)
    return _checked_count(count, name="count") * DTYPE_BYTES[checked_dtype]


def activation_bytes(
    batch_size: int, width: int, *, dtype: str = DTYPE_FLOAT32
) -> int:
    """一层 ``batch_size × width`` 中间结果的字节数（激活随批大小线性增长）."""
    checked_dtype = _checked_dtype(dtype)
    checked_batch = _checked_count(batch_size, name="batch_size", minimum=1)
    checked_width = _checked_count(width, name="width", minimum=1)
    return checked_batch * checked_width * DTYPE_BYTES[checked_dtype]


@dataclass(frozen=True)
class DevicePlan:
    """一次设备对账的读数：三块字节数 + 设备预算（``params`` / ``activations`` 都是**字节**）."""

    device: str
    dtype: str
    params: int
    activations: int
    optimizer_bytes: int
    budget_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _checked_device(self.device))
        object.__setattr__(self, "dtype", _checked_dtype(self.dtype))
        for name in ("params", "activations", "optimizer_bytes"):
            object.__setattr__(self, name, _checked_count(getattr(self, name), name=name))
        object.__setattr__(self, "budget_bytes", _checked_count(self.budget_bytes, name="budget_bytes"))

    @property
    def total_bytes(self) -> int:
        """三块之和：参数 + 激活 + 优化器状态."""
        return self.params + self.activations + self.optimizer_bytes

    @property
    def fits(self) -> bool:
        """是否装得下预算（**放不下不抛异常**：它只是一个读数）."""
        return self.total_bytes <= self.budget_bytes

    @property
    def headroom_bytes(self) -> int:
        """预算余量（可以为负——负号表示"超出多少"）."""
        return self.budget_bytes - self.total_bytes

    def line(self) -> str:
        """一行说明：``cuda/float32 | 参数 32 B + 激活 24 B + 优化器 96 B = 152 B / 4.0 GiB | fits=True``."""
        mark = "fits=True" if self.fits else "fits=False"
        return (
            f"{self.device}/{self.dtype} | 参数 {self.params} B + 激活 {self.activations} B + "
            f"优化器 {self.optimizer_bytes} B = {self.total_bytes} B / "
            f"{self.budget_bytes / 1024**3:.1f} GiB | {mark}"
        )

    def warn_line(self) -> str:
        """一条**警告行**：装得下时报余量，装不下时报超出多少（永不抛异常）."""
        if self.fits:
            return f"预算充足：余量 {self.headroom_bytes} 字节（{self.device}/{self.dtype}）"
        return (
            f"警告：超出预算 {-self.headroom_bytes} 字节（{self.device}/{self.dtype}）——"
            "先降 batch_size、再降 width，最后才考虑换精度"
        )


def plan_device(
    params: int,
    *,
    batch_size: int,
    width: int,
    device: str,
    dtype: str = DTYPE_FLOAT32,
    optimizer_multiplier: int = 3,
    budget_bytes: int | None = None,
) -> DevicePlan:
    """把"参数量 + 批大小 + 宽度 + 设备 + 精度"算成一份 :class:`DevicePlan`.

    ``optimizer_multiplier`` 的缺省是 3（Adam 的一阶 + 二阶动量 + 参数本身，
    约等于 3 倍参数量）；``budget_bytes`` 缺省取 :data:`DEFAULT_DEVICE_BUDGETS`。

    预算不足**不抛异常**——由 :attr:`DevicePlan.fits` 暴露，另给一条
    :meth:`DevicePlan.warn_line`（这是本课刻意选的口径：装不下是一个可解决的读数）。
    """
    checked_device = _checked_device(device)
    checked_dtype = _checked_dtype(dtype)
    checked_params = _checked_count(params, name="params", minimum=1)
    checked_multiplier = _checked_count(optimizer_multiplier, name="optimizer_multiplier")
    if budget_bytes is None:
        budget_bytes = DEFAULT_DEVICE_BUDGETS[checked_device]
    checked_budget = _checked_count(budget_bytes, name="budget_bytes", minimum=1)
    return DevicePlan(
        device=checked_device,
        dtype=checked_dtype,
        params=parameter_bytes(checked_params, dtype=checked_dtype),
        activations=activation_bytes(batch_size, width, dtype=checked_dtype),
        optimizer_bytes=checked_multiplier * parameter_bytes(checked_params, dtype=checked_dtype),
        budget_bytes=checked_budget,
    )


def move_report(plan: DevicePlan) -> tuple[str, ...]:
    """"把这条链搬到设备上"的可读行（CPU 一律 0 字节，其余给出总搬运量）."""
    if not isinstance(plan, DevicePlan):
        raise ParameterError(f"plan 必须是 DevicePlan，收到 {type(plan).__name__}。")
    if plan.device == DEVICE_CPU:
        return ("留在 CPU：搬运 0 字节（本课所有读数都在这里复算）",)
    prefix = DEVICE_PREFIXES[plan.device]
    return (
        f"{prefix}：搬运 {plan.total_bytes} 字节"
        f"（参数 {plan.params} + 激活 {plan.activations} + 优化器 {plan.optimizer_bytes}）",
        f"精度 {plan.dtype}：每个元素 {DTYPE_BYTES[plan.dtype]} 字节",
    )


if not math.isfinite(float(sum(DEFAULT_DEVICE_BUDGETS.values()))):  # pragma: no cover - 防御式
    raise DeviceError("缺省预算表里出现了非有限数。")


__all__ = [
    "DEFAULT_DEVICE_BUDGETS",
    "DevicePlan",
    "activation_bytes",
    "move_report",
    "parameter_bytes",
    "plan_device",
    "tensor_bytes",
]
