"""微分：把"变化率"写成一串可以相减的函数值（day074 / Math-D2）.

导数的定义是一句极限：

```text
f'(x) = lim_{h→0} (f(x + h) − f(x)) / h
```

计算机不会取极限。它只会做两件事：**算函数值**与**相减**。于是"求导"这件事
在代码里变成一个**近似**，而这一课的第一件事就是把近似的代价量出来：

```text
前向差分    (f(x+h) − f(x)) / h                截断误差 O(h)
后向差分    (f(x) − f(x−h)) / h                截断误差 O(h)
中心差分    (f(x+h) − f(x−h)) / (2h)           截断误差 O(h²)   ← 默认
```

## 为什么"中心差分"值得多算一次函数值

把两个式子各自做泰勒展开（``h`` 的同阶项一个保留、一个抵消）：

```text
f(x+h) = f(x) + h f'(x) + h² f''(x)/2 + O(h³)
f(x−h) = f(x) − h f'(x) + h² f''(x)/2 + O(h³)
────────────────────────────────────────────── 相减
f(x+h) − f(x−h) = 2h f'(x) + O(h³)         → 误差 O(h²)
```

前向差分里那个 ``h f''(x)/2`` 项在中心差分里**被减掉了**——代价是 h 每缩小 10 倍，
误差缩小 100 倍而不是 10 倍。这一点不靠"记住结论"，而是靠
:func:`step_size_study` **量出来**（``observed_error_order`` 会给出实测阶数）。

## 为什么步长不能一直缩小（这一课最容易忽略的一条）

``h`` 小到一定程度后误差**反而上升**，因为浮点数相减会丢有效位
（``f(x+h) − f(x−h)`` 是两个几乎相等的数相减，"灾难性抵消"）：

```text
h = 1e-3    截断误差主导      误差 ≈ 1e-6（中心差分）
h = 1e-6    两边打平          误差 ≈ 1e-9  ← 中心差分的最优区间
h = 1e-15   舍入误差主导      误差 ≈ 1e-1（已经把 f 的有效位全减没了）
```

最优步长的量级是 ``eps^{1/3} ≈ 6e-6``（中心差分）与 ``eps^{1/2} ≈ 1.5e-8``
（前向差分）——本模块的常量 :data:`DEFAULT_STEP` 取 ``1e-6``，
在"够小以压住截断误差"与"够大以留住有效位"之间。

## 这一课与后面每一课的关系

```text
梯度下降（optim.py）   往哪走 = 梯度（今天这一层算出来的东西）
反向传播（autograd.py）链式法则自动跑一遍 = 把"求导"从数值近似换成解析式
注意力（day075）       注意力权重对打分的导数 = softmax 的雅可比（gradcheck 会验证）
训练（day080 起）      每个参数一个偏导数，几千个参数一起用同一套规则
```

**数值微分不是生产实现，而是"校准尺"**：任何一份解析梯度都可以拿它来验
（:mod:`~smart_research_agent.math_foundations.gradcheck` 做的就是这件事）。
它慢、它只精确到 1e-9，但它不依赖任何推导——
因此它是唯一能独立验证"你的推导对不对"的方案。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.math_foundations.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.math_foundations.types import (
    BACKWARD,
    CALCULUS_METHOD_DESCRIPTIONS,
    CALCULUS_METHODS,
    CENTRAL,
    FLOAT_EPSILON,
    FORWARD,
    Matrix,
    Vector,
    close,
    is_finite,
    validate_vector,
)

#: 标量函数（一个实数进、一个实数出）.
ScalarFunction = Callable[[float], float]

#: 向量函数（一个向量进、一个实数出）——梯度就是它对每个分量的偏导数.
VectorFunction = Callable[[Vector], float]

#: 向量到向量的函数（雅可比矩阵的适用对象）.
VectorToVector = Callable[[Vector], Vector]

#: 缺省步长。中心差分的最优步长量级是 ``eps^{1/3} ≈ 6e-6``，
#: 这里取 ``1e-6``：比最优值略小一点，但在手算对比时更好看
#: （误差恰在 1e-9 附近，与 :data:`types.DEFAULT_TOLERANCE` 同量级）。
DEFAULT_STEP = 1e-6

#: 误差阶数实验中用的步长序列（从大到小；**写死**以保证可复现）.
DEFAULT_STEP_SIZES: tuple[float, ...] = (
    1e-1,
    1e-2,
    1e-3,
    1e-4,
    1e-5,
    1e-6,
    1e-7,
    1e-8,
    1e-9,
    1e-10,
    1e-12,
    1e-14,
)


def _checked_step(step: float) -> float:
    """校验步长：必须是正的有限数（**不接受 0**，也不接受"取个默认值兜底"）."""
    if not is_finite(step) or step <= 0:
        raise ParameterError(
            f"步长必须为正的有限数，收到 {step!r}："
            "h = 0 时分子恒为 0，(f(x+h) − f(x))/h 会变成 0/0——"
            "导数在一点上的定义要求 h → 0 但不等于 0。"
        )
    return float(step)


def _checked_point(x: float, *, name: str = "点") -> float:
    """校验求导点：必须是有限实数（NaN/inf 会让整条链路变成 NaN）."""
    if not is_finite(x):
        raise NumericError(
            f"{name}必须是有限实数，收到 {x!r}："
            "在 NaN 上求导会得到 NaN，而 NaN 会顺着整条链路把均值、梯度范数全变成 NaN。"
        )
    return float(x)


def _evaluate(function: ScalarFunction, x: float) -> float:
    """调用一次函数，并把"返回了非有限数"这件事当场拦下.

    不拦的后果很具体：``f`` 在 ``x + h`` 处溢出成 ``inf`` 时，
    ``(inf − f(x)) / h`` 得到 ``inf``，于是**导数报告成一个无穷大**，
    而根因（这个函数在该点溢出）要在很远的地方才看得出来。
    """
    value = function(x)
    if not is_finite(value):
        raise NumericError(
            f"函数在 x = {x!r} 处的取值不是有限实数（收到 {value!r}）："
            "差分公式会用两个非有限数相减，得到一个看起来像'导数很大'的结果。"
        )
    return float(value)


def forward_difference(function: ScalarFunction, x: float, *, step: float = DEFAULT_STEP) -> float:
    """前向差分 ``(f(x+h) − f(x)) / h``（截断误差 ``O(h)``，只需一次额外求值）."""
    h = _checked_step(step)
    point = _checked_point(x)
    return (_evaluate(function, point + h) - _evaluate(function, point)) / h


def backward_difference(function: ScalarFunction, x: float, *, step: float = DEFAULT_STEP) -> float:
    """后向差分 ``(f(x) − f(x−h)) / h``（截断误差 ``O(h)``）.

    它比前向差分多一个用途：**定义域端点上的单侧导数**。
    在右端点（例如 ``√(1−x)`` 在 ``x = 1``）上，前向差分会去取 ``f(x+h)``——
    那一点根本不在定义域里；而后向差分只用到 ``f(x)`` 与 ``f(x−h)``，
    因此它是端点处唯一可用的那一侧。
    """
    h = _checked_step(step)
    point = _checked_point(x)
    return (_evaluate(function, point) - _evaluate(function, point - h)) / h


def central_difference(function: ScalarFunction, x: float, *, step: float = DEFAULT_STEP) -> float:
    """中心差分 ``(f(x+h) − f(x−h)) / (2h)``（截断误差 ``O(h²)``，代价是两次额外求值）.

    这是本模块的缺省方法：多花一次函数求值，换来误差从 ``O(h)`` 降到 ``O(h²)``。
    ``h = 1e-6`` 时前向差分的误差在 1e-6 量级，中心差分在 1e-9 量级——
    这个差距在"用数值梯度验证解析梯度"的场景里是决定性的。
    """
    h = _checked_step(step)
    point = _checked_point(x)
    return (_evaluate(function, point + h) - _evaluate(function, point - h)) / (2.0 * h)


def second_difference(function: ScalarFunction, x: float, *, step: float = DEFAULT_STEP) -> float:
    """二阶中心差分 ``(f(x+h) − 2f(x) + f(x−h)) / h²``（截断误差 ``O(h²)``）.

    二阶导数是"变化率的变化率"，它在优化里的位置是**曲率**：
    曲率大说明这一步迈大了会冲过头（曲线在这里拐得急）。
    本模块用它来做一件具体的事：把"中心差分的最优步长"算出来
    （见 :func:`best_step_for_central` 的说明）。
    """
    h = _checked_step(step)
    point = _checked_point(x)
    return (
        _evaluate(function, point + h)
        - 2.0 * _evaluate(function, point)
        + _evaluate(function, point - h)
    ) / (h * h)


def derivative(
    function: ScalarFunction,
    x: float,
    *,
    step: float = DEFAULT_STEP,
    method: str = CENTRAL,
) -> float:
    """按指定方法求一阶导数（``method`` 取值见 :data:`types.CALCULUS_METHODS`）."""
    if method not in CALCULUS_METHODS:
        raise ParameterError(
            f"不认识的差分方法 {method!r}：可用取值 {list(CALCULUS_METHODS)}"
            f"（{CALCULUS_METHOD_DESCRIPTIONS[FORWARD][:12]}…）。"
        )
    if method == FORWARD:
        return forward_difference(function, x, step=step)
    if method == BACKWARD:
        return backward_difference(function, x, step=step)
    return central_difference(function, x, step=step)


def gradient(
    function: VectorFunction,
    point: Vector,
    *,
    step: float = DEFAULT_STEP,
    method: str = CENTRAL,
) -> Vector:
    """梯度 ``∇f = (∂f/∂x_1, …, ∂f/∂x_n)``：对每个分量分别做一元差分.

    形状是最容易搞错的地方：``∇f`` 与 ``x`` **同形**（都是 ``n`` 维），
    而每一步只扰动一个分量。这也是"维度灾难"的算术来源——
    ``n`` 维梯度要 ``2n`` 次函数求值（中心差分），所以生产里不用数值梯度，
    只在验证时用（:mod:`gradcheck` 也是这么用的）。
    """
    checked = validate_vector(point, name="point")
    h = _checked_step(step)
    if method not in CALCULUS_METHODS:
        raise ParameterError(f"不认识的差分方法 {method!r}：可用取值 {list(CALCULUS_METHODS)}。")
    partials: list[float] = []
    for index in range(len(checked)):
        def shifted(offset: float) -> float:
            """只把第 ``index`` 个分量平移 ``offset``，其余不动."""
            bumped = list(checked)
            bumped[index] = checked[index] + offset
            return _evaluate(function, tuple(bumped))

        if method == FORWARD:
            partials.append((shifted(h) - _evaluate(function, checked)) / h)
        elif method == BACKWARD:
            partials.append((_evaluate(function, checked) - shifted(-h)) / h)
        else:
            partials.append((shifted(h) - shifted(-h)) / (2.0 * h))
    return tuple(partials)


def directional_derivative(
    function: VectorFunction,
    point: Vector,
    direction: Vector,
    *,
    step: float = DEFAULT_STEP,
) -> float:
    """方向导数 ``D_u f = ∇f · û``：沿某个**单位**方向走一步，函数值变化多快.

    两个约定写在这里：

    ```text
    ① direction 会被归一化   否则"方向导数"会随方向的长度改变，
                            而"方向"本身不该带长度信息（那是步长的事）
    ② 零方向当场报错         零向量没有方向，沿它没有"变化率"可言
                            （与 linalg.normalize 拒绝零向量是同一条纪律）
    ```

    它也是"梯度为什么是最陡方向"的可验证形式：
    在单位方向里取遍所有方向，``D_u f`` 的最大值恰好在 ``u = ∇f/‖∇f‖``
    处取到，最大值等于 ``‖∇f‖``（柯西–施瓦茨不等式）。
    演示脚本会把这件事扫一遍（第 5 节）。
    """
    checked_point_vector = validate_vector(point, name="point")
    checked_direction = validate_vector(direction, name="direction")
    if len(checked_direction) != len(checked_point_vector):
        raise ShapeError(
            f"方向 {len(checked_direction)} 维而点是 {len(checked_point_vector)} 维："
            "方向导数只在同一维空间里有定义。"
        )
    length = math.sqrt(math.fsum(value * value for value in checked_direction))
    if length == 0.0:
        raise NumericError(
            "方向是零向量：沿零向量没有'变化率'——"
            "如果你要的是'所有方向的平均'，请显式取一组单位方向再平均。"
        )
    unit = tuple(value / length for value in checked_direction)
    h = _checked_step(step)

    def along(offset: float) -> float:
        """沿单位方向平移 ``offset`` 之后的函数值."""
        moved = tuple(x + offset * d for x, d in zip(checked_point_vector, unit))
        return _evaluate(function, moved)

    return (along(h) - along(-h)) / (2.0 * h)


def jacobian(
    function: VectorToVector,
    point: Vector,
    *,
    step: float = DEFAULT_STEP,
) -> Matrix:
    """雅可比矩阵 ``J_ij = ∂f_i/∂x_j``（形状 ``(len(f), len(x))``）.

    "梯度"是雅可比在 ``f`` 取标量时的特例（此时形状退化成一行）。
    本模块实现它的理由只有一个：**softmax 的雅可比可以手算**，
    而它是"``∂L/∂z = p − y`` 这个著名结论"的全部来源
    （见 :mod:`gradcheck` 的 ``softmax`` 与 ``cross_entropy`` 两项）。
    """
    checked = validate_vector(point, name="point")
    h = _checked_step(step)
    outputs = validate_vector(function(checked), name="function(point)")
    rows: list[Vector] = []
    for index in range(len(outputs)):
        row: list[float] = []
        for position in range(len(checked)):
            # 只把第 position 个输入分量平移，取第 index 个输出分量。
            # 每个 (index, position) 组合独立做一次中心差分——
            # 这是 2·len(f)·len(x) 次函数求值，因此它只适合小规模校验。
            def output_at(offset: float, row_index: int = index, column: int = position) -> float:
                """把第 ``column`` 个输入分量平移 ``offset`` 后，取第 ``row_index`` 个输出."""
                bumped = list(checked)
                bumped[column] = checked[column] + offset
                return _evaluate(lambda point: function(point)[row_index], tuple(bumped))

            row.append((output_at(h) - output_at(-h)) / (2.0 * h))
        rows.append(tuple(row))
    return tuple(rows)


# --------------------------------------------------------------------------- #
# 步长实验：把"误差随 h 怎么变"量出来
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StepRecord:
    """一个步长上三种方法的绝对误差（外加两条归一化列）.

    ```text
    forward_error / backward_error    O(h)   截断误差主导时，"误差/h"应当大致是常数
    central_error                     O(h²)  截断误差主导时，"误差/h²"应当大致是常数
    ```

    两条归一化列（``forward_error / h`` 与 ``central_error / h²``）是这一层的
    "证据"：只看误差数字，读的人无法分辨"误差在按 O(h) 还是 O(h²) 下降"。
    """

    step: float
    forward_error: float
    backward_error: float
    central_error: float

    @property
    def forward_scaled(self) -> float:
        """``误差 / h``（前向差分在截断误差主导区应当是常数）."""
        return self.forward_error / self.step

    @property
    def central_scaled(self) -> float:
        """``误差 / h²``（中心差分在截断误差主导区应当是常数）."""
        return self.central_error / (self.step * self.step)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "step": self.step,
            "forward_error": self.forward_error,
            "backward_error": self.backward_error,
            "central_error": self.central_error,
            "forward_scaled": self.forward_scaled,
            "central_scaled": self.central_scaled,
        }

    def summary_line(self) -> str:
        """一行说明：``h=1e-06 | 前向 4.20e-06 | 后向 4.20e-06 | 中心 3.10e-10``."""
        return (
            f"h={self.step:.0e} | 前向 {self.forward_error:.2e} | "
            f"后向 {self.backward_error:.2e} | 中心 {self.central_error:.2e}"
        )


@dataclass(frozen=True)
class StepSizeStudy:
    """一次"步长 vs 误差"的实测：**这就是"为什么不能把 h 一直缩小"的证据**.

    ```text
    records              每个 h 上一行读数（步长从大到小）
    best_central         实测误差最小的那个 h（中心差分）
    observed_order       实测误差阶数：截断误差主导区里 log(e1/e2) / log(h1/h2)
    ```

    ``observed_order`` 只在**截断误差主导**的两个点上算（取步长最大的前两条）：
    在舍入误差主导的区域算阶数会得到接近 0（误差不再随 h 下降），
    把两者混在一起平均，得到的数字既不是 1 也不是 2，而是一条没有意义的数。
    """

    x: float
    exact_derivative: float
    records: tuple[StepRecord, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.records:
            raise ParameterError("步长实验至少要有一个读数：没有读数的实验无法支撑任何结论。")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    def _error(self, record: StepRecord, method: str) -> float:
        """按方法取一行读数里的误差列."""
        if method == FORWARD:
            return record.forward_error
        if method == BACKWARD:
            return record.backward_error
        if method == CENTRAL:
            return record.central_error
        raise ParameterError(f"不认识的差分方法 {method!r}：可用取值 {list(CALCULUS_METHODS)}。")

    def observed_order(self, method: str = CENTRAL, *, index: int = 0) -> float:
        """实测误差阶数：取第 ``index`` 与 ``index + 1`` 两条读数算 ``log(e₁/e₂)/log(h₁/h₂)``.

        缺省取最初两条（步长最大）——那里是截断误差主导区。
        前向/后向应当是 1，中心应当是 2。
        """
        if index < 0 or index + 1 >= len(self.records):
            raise ParameterError(
                f"阶数需要相邻两条读数，index={index} 超出范围 [0, {len(self.records) - 2}]。"
            )
        first, second = self.records[index], self.records[index + 1]
        error_first = self._error(first, method)
        error_second = self._error(second, method)
        if error_first <= 0 or error_second <= 0:
            raise NumericError(
                f"两条读数的误差是 {error_first!r} 与 {error_second!r}："
                "阶数是比值取对数，误差为 0 时它没有定义（说明这一步恰好精确命中）。"
            )
        return math.log(error_first / error_second) / math.log(first.step / second.step)

    def best(self, method: str = CENTRAL) -> StepRecord:
        """误差最小的那一条读数（"实践上该取多大 h"的答案）."""
        return min(self.records, key=lambda record: self._error(record, method))

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（含实测阶数与最优步长）."""
        return {
            "x": self.x,
            "exact_derivative": self.exact_derivative,
            "observed_order_forward": self.observed_order(FORWARD),
            "observed_order_backward": self.observed_order(BACKWARD),
            "observed_order_central": self.observed_order(CENTRAL),
            "best_central_step": self.best(CENTRAL).step,
            "best_central_error": self.best(CENTRAL).central_error,
            "records": [record.to_dict() for record in self.records],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``实测阶数：前向 1.00、中心 2.00 | 最优 h=1e-06（误差 3.10e-10）``."""
        best = self.best(CENTRAL)
        return (
            f"实测阶数：前向 {self.observed_order(FORWARD):.2f}、"
            f"中心 {self.observed_order(CENTRAL):.2f} | "
            f"最优 h={best.step:.0e}（中心误差 {best.central_error:.2e}）"
        )

    def summary_lines(self) -> tuple[str, ...]:
        """逐行读数（进演示脚本与报告）."""
        return tuple(record.summary_line() for record in self.records)


def step_size_study(
    function: ScalarFunction,
    x: float,
    exact_derivative: float,
    *,
    steps: Sequence[float] = DEFAULT_STEP_SIZES,
) -> StepSizeStudy:
    """对三种差分方法扫一遍步长，把"误差随 h 的变化"量出来.

    ``exact_derivative`` 必须由调用方给出（例如 ``exp`` 在 ``x`` 处的导数就是
    ``exp(x)``）——**没有真值就谈不上误差**，而"看起来收敛得不错"这句话
    在没有真值时是没有内容的。

    这个实验回答三个问题，每一个都可以被断言：

    ```text
    ① 误差随 h 怎么下降      前向 O(h)、中心 O(h²)（observed_order 给出实测值）
    ② 误差在哪里触底         中心差分约 1e-6、前向约 1e-8（best 给出实测值）
    ③ 触底之后为什么上升     舍入误差主导：f(x+h) 与 f(x−h) 的有效位被减掉了
    ```
    """
    point = _checked_point(x)
    exact = _checked_point(exact_derivative, name="精确导数值")
    if not steps:
        raise ParameterError("步长序列不能为空：没有步长的实验没有读数可以比较。")
    records: list[StepRecord] = []
    for raw in steps:
        h = _checked_step(raw)
        records.append(
            StepRecord(
                step=h,
                forward_error=abs(forward_difference(function, point, step=h) - exact),
                backward_error=abs(backward_difference(function, point, step=h) - exact),
                central_error=abs(central_difference(function, point, step=h) - exact),
            )
        )
    return StepSizeStudy(
        x=point,
        exact_derivative=exact,
        records=tuple(records),
        notes=(
            "前向/后向差分的截断误差是 O(h)，中心差分是 O(h²)——"
            "因为 f(x+h) 与 f(x−h) 展开后那个 h f''(x) 项在相减时被消掉了",
            "舍入误差与截断误差方向相反：h 越小截断误差越小、"
            "但 f(x+h) − f(x−h) 两个几乎相等的数相减丢掉的位越多，"
            "两条曲线的交点就是实测最优步长（中心差分约 1e-6）",
            "阶数只在截断误差主导区有意义：在触底之后的读数上算阶数会得到接近 0",
        ),
    )


def best_step_for_central(function: ScalarFunction, x: float) -> float:
    """按曲率算中心差分的**理论**最优步长 ``(eps/|f''(x)|)^{1/3}``.

    推导用的是"两条误差量级相等"：截断误差 ``≈ |f'''| h²/6`` 与
    舍入误差 ``≈ eps·|f(x)|/h`` 在交点处同阶。工程上常用的化简版是

    ```text
    h ≈ eps^{1/3} ≈ 6.06e-6        （|f| 与 |f''| 都是 1 的量级时）
    ```

    ## 一个必须处理的问题：``f''`` 是**量出来**的，而测量有噪声

    ``f''(x)`` 由 :func:`second_difference` 给出，它的分母是 ``h²``——
    于是**舍入误差被放大 1/h² 倍**，噪声地板大约是 ``4·eps·|f(x)|/h²``：

    ```text
    f(x) = 3x + 1（直线，真 f'' = 0）    h = 1e-6
    真值                     0
    量出来的 |f''|           ≈ 1.8e-3      ← 全是噪声（比真值大 15 个数量级）
    ```

    如果只判断"``|f''| <= eps``"，这条噪声会**永远**通不过，
    于是"接近直线的函数"会得到一个由噪声决定的最优步长——
    而那是一个与函数本身无关的数。因此这里把量出来的曲率与
    **它的噪声地板**比较：低于地板就当作 0，返回 ``eps^{1/3}``。
    """
    point = _checked_point(x)
    curvature = abs(second_difference(function, point))
    magnitude = abs(_evaluate(function, point))
    noise_floor = 4.0 * FLOAT_EPSILON * max(magnitude, 1.0) / (DEFAULT_STEP * DEFAULT_STEP)
    if curvature <= noise_floor:
        return FLOAT_EPSILON ** (1.0 / 3.0)
    return (FLOAT_EPSILON / curvature) ** (1.0 / 3.0)


# --------------------------------------------------------------------------- #
# 链式法则：反向传播的全部内容
# --------------------------------------------------------------------------- #


def compose(*functions: ScalarFunction) -> ScalarFunction:
    """把若干一元函数按书写顺序**从右到左**复合：``compose(f, g)(x) == f(g(x))``.

    与数学上的 ``f ∘ g`` 同一个方向（右到左），而不是 ``f(g(…))`` 的书写顺序——
    这条约定必须写下来：两种读法都"看起来对"，而它们的数值结果完全不同。
    """
    if not functions:
        raise ParameterError("至少要有一个函数：空复合没有定义。")
    if len(functions) == 1:
        return functions[0]

    def composed(value: float) -> float:
        """从最右边的函数开始，把结果一路喂给左边的函数."""
        result = value
        for function in reversed(functions):
            result = function(result)
        return result

    return composed


def chain_rule(
    functions: Sequence[ScalarFunction],
    x: float,
    *,
    step: float = DEFAULT_STEP,
) -> tuple[float, float]:
    """链式法则的两条路，返回 ``(整体导数, 局部导数之积)``.

    ```text
    整体      d/dx fₙ(…f₁(x))        直接对复合函数做差分
    局部之积  fₙ'(…f₁(x)) · … · f₁'(x) 每一层的导数在**正确的点上**取值
    ```

    两者必须相等，而"在正确的点上取值"正是最容易写错的地方：
    ``f₂'(f₁(x))`` 而不是 ``f₂'(x)``。**反向传播做的事就是自动取对每一层的点**
    （它先算完整条前向链，再沿着链把局部导数乘回去），
    所以 :class:`autograd.Scalar` 的 ``backward()`` 可以看作
    "把这一节的乘法顺序自动化"。
    """
    if not functions:
        raise ParameterError("至少要有一个函数：空复合没有导数。")
    point = _checked_point(x)
    h = _checked_step(step)

    def composed(value: float) -> float:
        """按 compose 的同一方向（右到左）求值."""
        result = value
        for function in reversed(tuple(functions)):
            result = function(result)
        return result

    overall = central_difference(composed, point, step=h)

    # 逐层推进：每到一个新点，就用**这个点**上的局部导数乘上去
    product = 1.0
    current = point
    for function in reversed(tuple(functions)):
        product *= central_difference(function, current, step=h)
        current = _evaluate(function, current)
    return overall, product


def linear_approximation(function: ScalarFunction, x0: float, x: float, *, step: float = DEFAULT_STEP) -> float:
    """一阶泰勒近似 ``f(x) ≈ f(x₀) + f'(x₀)(x − x₀)``（切线的解析式）.

    为什么梯度下降敢"沿着梯度迈一步就相信损失会下降"：因为损失在
    ``x₀`` 附近**近似是一条直线**，而直线的下降方向就是梯度方向。
    迈得太远时近似失效（曲面弯了），所以有学习率上限——
    这一条把 :mod:`optim` 里的学习率与这里的二阶项连起来：

    ```text
    f(x₀ + Δ) ≈ f(x₀) + f'(x₀)Δ + f''(x₀)Δ²/2
                                               └── 这项被忽略掉了
    Δ 越小，忽略的那一项越小（这就是"小步走"的数学理由）
    ```
    """
    point = _checked_point(x0, name="展开点")
    target = _checked_point(x, name="目标点")
    return _evaluate(function, point) + central_difference(function, point, step=step) * (
        target - point
    )


def taylor_accuracy(
    function: ScalarFunction,
    x0: float,
    x: float,
    *,
    step: float = DEFAULT_STEP,
) -> tuple[float, float]:
    """一阶近似的误差与二阶项上界，返回 ``(实测误差, 二阶项量级)``.

    这两个数字放在一起才说明问题：实测误差应当与二阶项**同量级**，
    而"距离越大误差越大"的速率由 ``|f''(x₀)|·(x−x₀)²/2`` 给出。
    只报误差看不出它是从哪里来的；只报二阶项则没有验证过。
    """
    point = _checked_point(x0, name="展开点")
    target = _checked_point(x, name="目标点")
    approximate = linear_approximation(function, point, target, step=step)
    actual = _evaluate(function, target)
    curvature = abs(second_difference(function, point, step=step))
    gap = target - point
    return abs(actual - approximate), curvature * gap * gap / 2.0


def is_close(left: float, right: float, *, tolerance: float = 1e-6) -> bool:
    """两个数是否在容差内相等（**与** :func:`types.close` **同一口径**的转发）.

    留一个转发入口是因为差分结果常与手算值比较，而两处各写一遍判据
    迟早会在某一天给出不一致的结论（一处宽松、一处严格）。
    """
    return close(left, right, tolerance=tolerance)


__all__ = [
    "DEFAULT_STEP",
    "DEFAULT_STEP_SIZES",
    "ScalarFunction",
    "StepRecord",
    "StepSizeStudy",
    "VectorFunction",
    "VectorToVector",
    "backward_difference",
    "best_step_for_central",
    "central_difference",
    "chain_rule",
    "compose",
    "derivative",
    "directional_derivative",
    "forward_difference",
    "gradient",
    "is_close",
    "jacobian",
    "linear_approximation",
    "second_difference",
    "step_size_study",
    "taylor_accuracy",
]
