"""反向传播：把链式法则自动跑一遍（day074 / Math-D2）.

``calculus`` 用**数值差分**求导：它不依赖任何推导，但"算一次梯度要 2n 次函数求值"、
而且只精确到 1e-9。这一模块走另一条路——**解析式 + 自动微分**：

```text
每个数记两样东西   value（它现在的值）   grad（损失对它有多敏感）
每个运算记一条规则 一个乘法的局部导数是"另一个乘数"，一个 exp 的局部导数是"它自己"
反向传播         沿计算图**从输出往输入**走一遍，每一步把局部导数乘进上一步的梯度
```

于是导数不再是一个近似，而是**一份推导的可执行形式**：
``d(x·x)/dx = 2x`` 不是"记住的"，而是从"乘法的局部导数是另一个乘数"这条规则里
自己长出来的（``x·x`` 的两条边会把 ``x`` 的梯度各加一份，正好是 ``2x``）。

## 这一课最值钱的一句：链式法则 × 累积

反向传播的公式只有一条：**上游梯度 × 局部导数**。真正需要小心的是第二件事——
**一个值被用到 k 次，它的梯度是 k 条路径之和**：

```text
y = x · x            两条边都通向 x：dy/dx = x + x = 2x（在 x = 3 处是 6）
y = x + x            两条边都通向 x：dy/dx = 1 + 1 = 2
y = exp(x) + log(x)  两条边都通向 x：dy/dx = exp(x) + 1/x
```

因此 :meth:`Scalar.backward` 里对父节点一律用 ``+=`` 而不是 ``=``。
写成 ``=`` 的后果非常具体：**它不会报错，只会给出一个偏小的梯度**，
而"梯度偏小"看起来像"学习率设小了"——于是有人去调学习率。

## 为什么这一层是"教学实现"而不是"能用的框架"

```text
只做标量         一个参数一个节点，10 万个参数的模型会有 10 万个对象（生产用张量）
不做原地复用     同一个节点被跑两次 backward 会叠加（必须显式 zero_grad()）
不做设备/类型    没有 float32、没有 GPU——那些是"规模"的问题，不是"导数"的问题
```

它的用途与 ``calculus`` 的数值差分是**一对**：数值差分是"校准尺"（不依赖推导），
自动微分是"推导的执行"（不依赖近似）。两者对上，才算把链式法则这件事说清楚了
（对照代码见 :mod:`~smart_research_agent.math_foundations.gradcheck`）。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from smart_research_agent.math_foundations.errors import NumericError, ParameterError

#: 逗号分隔的一串标量节点（计算图的输入）.
ScalarInputs = Sequence["Scalar"]

#: 接受若干标量节点、返回一个标量节点的函数（要被求梯度的"前向表达式"）.
ScalarExpression = Callable[[list["Scalar"]], "Scalar"]


@dataclass(frozen=True)
class TraceRow:
    """计算图上一行账：这个节点是什么运算、值多少、梯度多少.

    "把图打印出来"这件事在教学里的价值被低估了：一张 6 个节点的图打印出来之后，
    "为什么 ``x`` 被用了两次，梯度就是 2 倍"是一眼能看出来的；
    而只给一个数字 ``6.0``，读的人只能选择相信。
    """

    label: str
    operation: str
    value: float
    grad: float

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "label": self.label,
            "operation": self.operation,
            "value": self.value,
            "grad": self.grad,
        }

    def summary_line(self) -> str:
        """一行说明：``x                 value=+3.000000 grad=+6.000000``."""
        return (
            f"{self.label:<16} {self.operation:<12} "
            f"value={self.value:+.6f} grad={self.grad:+.6f}"
        )


class Scalar:
    """计算图上的一个标量节点：值 + 梯度 + 它的父节点 + 一条局部导数规则.

    三个约定写下来（它们的反面都是"不报错但结果错"）：

    ```text
    ① 值一旦建图就不改（需要新值就新建节点）：原地改 value 会让已算出的
       梯度与新的前向值不对应，而两者都还是"看起来正常的数"
    ② 梯度用 += 累积：一个值被用 k 次，梯度是 k 条路径之和
    ③ requires_grad=False 的节点是**常量**：反向传播不会往它里面写梯度
    ```

    运算符齐全（``+ - * / **`` 与 ``-x``）、初等函数齐全
    （``exp`` / ``log`` / ``sqrt`` / ``relu`` / ``sigmoid`` / ``tanh``）——
    这些正好是"后面要自己实现的那些层"用得到的全部零件。
    """

    __slots__ = ("value", "grad", "label", "operation", "_parents", "_propagate", "requires_grad")

    def __init__(
        self,
        value: float,
        *,
        label: str = "",
        operation: str = "leaf",
        requires_grad: bool = True,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NumericError(
                f"Scalar 的值必须是实数，收到 {type(value).__name__}（{value!r}）："
                "字符串或 None 会在第一次运算时变成 TypeError，而栈会深到看不出根因。"
            )
        if not math.isfinite(float(value)):
            raise NumericError(
                f"Scalar 的值必须是有限实数，收到 {value!r}："
                "NaN/inf 会顺着整条链把梯度全变成 NaN，"
                "而'梯度是 NaN'看起来像学习率爆炸，根因却在更早的一次除法上。"
            )
        self.value = float(value)
        self.grad = 0.0
        self.label = label or f"node{operation}"
        self.operation = operation
        self.requires_grad = requires_grad
        self._parents: tuple[Scalar, ...] = ()
        self._propagate: Callable[[float], None] = lambda upstream: None

    # ------------------------------------------------------------------ #
    # 建图入口：本模块唯一的"内部构造函数"
    # ------------------------------------------------------------------ #

    @classmethod
    def _from(
        cls,
        value: float,
        *,
        operation: str,
        parents: tuple[Scalar, ...],
        propagate: Callable[[float], None],
        label: str = "",
    ) -> Scalar:
        """造一个运算节点（值 + 父节点 + 局部导数规则），供所有算子复用.

        两个约定必须写下来，因为它们各自对应一类"不报错但结果错"：

        ```text
        ① propagate 接收的是**本节点自己的梯度**（上游梯度），
           而不是"某个父节点的梯度"。写成"闭包里读 self.grad"是这一层
           最容易犯的错——self 在闭包里指向左操作数，于是加法会把
           左操作数的旧梯度当成上游梯度传下去；在只有一条链、且左操作数
           恰好是叶子时，它甚至能算出正确的结果（叶子的梯度初值 0），
           因此在简单的例子上根本发现不了。
        ② 只在需要求梯度的父节点上参与建图：全部父节点都是常量时，
           这个节点自己就没有可回传的东西（requires_grad=False），
           于是"常量表达式"不会在图上留下无用节点。
        ```
        """
        node = cls(value, label=label, operation=operation)
        node.requires_grad = any(parent.requires_grad for parent in parents)
        node._parents = parents
        node._propagate = propagate
        return node

    # ------------------------------------------------------------------ #
    # 运算符
    # ------------------------------------------------------------------ #

    def __add__(self, other: Scalar | float) -> Scalar:
        other_node = ensure_scalar(other, name="加数")
        return Scalar._from(
            self.value + other_node.value,
            operation="add",
            parents=(self, other_node),
            propagate=lambda upstream: _accumulate((self, other_node), (upstream, upstream)),
        )

    def __radd__(self, other: Scalar | float) -> Scalar:
        return ensure_scalar(other, name="加数") + self

    def __mul__(self, other: Scalar | float) -> Scalar:
        other_node = ensure_scalar(other, name="乘数")
        # 乘法的局部导数：对左乘数是"右乘数的值"，对右乘数是"左乘数的值"。
        # 一处最容易写错的细节：局部导数取的是**前向值**，与梯度无关。
        return Scalar._from(
            self.value * other_node.value,
            operation="mul",
            parents=(self, other_node),
            propagate=lambda upstream: _accumulate(
                (self, other_node), (upstream * other_node.value, upstream * self.value)
            ),
        )

    def __rmul__(self, other: Scalar | float) -> Scalar:
        return ensure_scalar(other, name="乘数") * self

    def __sub__(self, other: Scalar | float) -> Scalar:
        other_node = ensure_scalar(other, name="减数")
        return Scalar._from(
            self.value - other_node.value,
            operation="sub",
            parents=(self, other_node),
            propagate=lambda upstream: _accumulate((self, other_node), (upstream, -upstream)),
        )

    def __rsub__(self, other: Scalar | float) -> Scalar:
        return ensure_scalar(other, name="被减数") - self

    def __truediv__(self, other: Scalar | float) -> Scalar:
        other_node = ensure_scalar(other, name="除数")
        if other_node.value == 0.0:
            raise NumericError(
                "除数是 0：除法在这个点没有定义，而'先算出来再变成 inf'"
                "会让错误出现在很远的地方（inf 会一路污染到梯度）。"
            )
        return Scalar._from(
            self.value / other_node.value,
            operation="div",
            parents=(self, other_node),
            propagate=lambda upstream: _accumulate(
                (self, other_node),
                (
                    upstream / other_node.value,
                    -upstream * self.value / (other_node.value * other_node.value),
                ),
            ),
        )

    def __rtruediv__(self, other: Scalar | float) -> Scalar:
        return ensure_scalar(other, name="被除数") / self

    def __pow__(self, exponent: float) -> Scalar:
        if isinstance(exponent, bool) or not isinstance(exponent, (int, float)):
            raise ParameterError(
                f"指数必须是实数，收到 {type(exponent).__name__}："
                "本模块不支持'指数也是一个节点'（那需要换一套实现，见模块说明）。"
            )
        power = float(exponent)
        if not math.isfinite(power):
            raise ParameterError(f"指数必须是有限实数，收到 {power!r}。")
        if self.value == 0.0 and power < 1.0:
            raise NumericError(
                f"0 的 {power} 次幂没有定义：导数公式 n·x^(n−1) 在这一步会变成 "
                "0 的负数次幂（除零）。"
            )
        if self.value < 0.0 and not float(power).is_integer():
            raise NumericError(
                f"负数的 {power} 次幂在实数范围内没有定义："
                "请显式取绝对值或改写表达式，而不是得到一个复数或 nan。"
            )
        local = power * self.value ** (power - 1.0)
        return Scalar._from(
            self.value**power,
            operation=f"pow({power:g})",
            parents=(self,),
            propagate=lambda upstream: _accumulate((self,), (upstream * local,)),
        )

    def __neg__(self) -> Scalar:
        return self * -1.0

    def __repr__(self) -> str:  # pragma: no cover - 只影响调试输出
        return f"Scalar(value={self.value!r}, grad={self.grad!r}, label={self.label!r})"

    # ------------------------------------------------------------------ #
    # 初等函数
    # ------------------------------------------------------------------ #

    def exp(self) -> Scalar:
        """``exp``：局部导数就是它自己的值（这也是"exp 很好求导"的全部原因）."""
        result = math.exp(self.value)
        return Scalar._from(
            result,
            operation="exp",
            parents=(self,),
            propagate=lambda upstream: _accumulate((self,), (upstream * result,)),
        )

    def log(self) -> Scalar:
        """``log``：局部导数 ``1/x``；``x <= 0`` 当场报错（不是返回 ``-inf``）."""
        if self.value <= 0.0:
            raise NumericError(
                f"log 的自变量必须为正，收到 {self.value!r}："
                "x = 0 时对数是 −inf、x < 0 时在实数里没有定义——"
                "两者都会让梯度变成 NaN（−inf 与 0 相乘），"
                "而 NaN 看起来像学习率爆炸。"
            )
        local = 1.0 / self.value
        return Scalar._from(
            math.log(self.value),
            operation="log",
            parents=(self,),
            propagate=lambda upstream: _accumulate((self,), (upstream * local,)),
        )

    def sqrt(self) -> Scalar:
        """``sqrt``：局部导数 ``1/(2√x)``（在 ``x = 0`` 处无定义，当场报错）."""
        if self.value <= 0.0:
            raise NumericError(
                f"sqrt 的自变量必须为正，收到 {self.value!r}："
                "0 处的导数是 1/(2·0)，那是一个无穷大的斜率——"
                "它在某些任务里是数学上真实存在的（如对比学习的 NT-Xent），"
                "但需要显式的稳定化处理，不能靠一个 nan 混过去。"
            )
        result = math.sqrt(self.value)
        local = 0.5 / result
        return Scalar._from(
            result,
            operation="sqrt",
            parents=(self,),
            propagate=lambda upstream: _accumulate((self,), (upstream * local,)),
        )

    def relu(self) -> Scalar:
        """``ReLU``：``max(0, x)``，局部导数在 ``x > 0`` 处是 1、否则是 0.

        在 ``x = 0`` 处导数不存在的处理方式是**取 0**（次梯度）。
        这是一个约定，不是一个事实，因此它必须写下来：
        取 1 也"说得通"（右侧导数），但那样一个恒为负的输入会一直得到梯度，
        与"不激活就不学习"的直觉相反。
        """
        active = self.value > 0.0
        local = 1.0 if active else 0.0
        return Scalar._from(
            self.value if active else 0.0,
            operation="relu",
            parents=(self,),
            propagate=lambda upstream: _accumulate((self,), (upstream * local,)),
        )

    def sigmoid(self) -> Scalar:
        """``σ(x) = 1/(1+e^{−x})``：局部导数 ``σ(x)(1−σ(x))``.

        按符号分支计算（``x`` 很负时先算 ``exp(x)`` 而不是 ``exp(−x)``）：
        ``exp(800)`` 会溢出成 ``inf``，虽然 ``1/(1+inf) = 0`` 结果是对的，
        但中间那一步 ``inf`` 在开启浮点陷阱的环境里会直接抛异常。
        """
        value = _sigmoid(self.value)
        local = value * (1.0 - value)
        return Scalar._from(
            value,
            operation="sigmoid",
            parents=(self,),
            propagate=lambda upstream: _accumulate((self,), (upstream * local,)),
        )

    def tanh(self) -> Scalar:
        """``tanh``：局部导数 ``1 − tanh²(x)``（值域 (−1, 1)，因此导数上界是 1）."""
        value = math.tanh(self.value)
        local = 1.0 - value * value
        return Scalar._from(
            value,
            operation="tanh",
            parents=(self,),
            propagate=lambda upstream: _accumulate((self,), (upstream * local,)),
        )

    def sin(self) -> Scalar:
        """``sin``：局部导数 ``cos``（链式法则里这一项让"周期性"进入梯度）."""
        local = math.cos(self.value)
        return Scalar._from(
            math.sin(self.value),
            operation="sin",
            parents=(self,),
            propagate=lambda upstream: _accumulate((self,), (upstream * local,)),
        )

    def cos(self) -> Scalar:
        """``cos``：局部导数 ``−sin``（符号是这一处最容易漏的地方）."""
        local = -math.sin(self.value)
        return Scalar._from(
            math.cos(self.value),
            operation="cos",
            parents=(self,),
            propagate=lambda upstream: _accumulate((self,), (upstream * local,)),
        )

    # ------------------------------------------------------------------ #
    # 反向传播
    # ------------------------------------------------------------------ #

    def zero_grad(self) -> None:
        """把整张图的梯度清零（**图可以复用，梯度不会自己清零**）.

        不清零的后果非常具体：第二次 ``backward()`` 会把梯度**加上去**，
        于是"这一步的梯度"变成"两步梯度之和"——
        它不会报错，只会让训练看起来"梯度比预期大两倍"。
        """
        for node in topological_order(self):
            node.grad = 0.0

    def backward(self, seed: float = 1.0) -> None:
        """从本节点出发做反向传播，把梯度累积到整张图上.

        ``seed`` 是"输出对输出自己的导数"（通常取 1.0）。
        它存在的意义只有一个：让"对一个非标量输出求梯度"这件事有一个显式入口
        （例如给每个输出配一个权重再求和，就等于给每个输出一个 seed）。

        顺序是**从根到叶**（:func:`topological_order` 的返回顺序）：
        每个节点在被处理时，它的梯度已经收齐了所有下游路径的贡献。
        在一条链上（``f₃(f₂(f₁(x)))``）这就是"从最外层往里乘"，
        与 :func:`calculus.chain_rule` 手写的那条乘法顺序完全一致。
        """
        if not math.isfinite(seed):
            raise ParameterError(f"seed 必须是有限实数，收到 {seed!r}。")
        self.grad = float(seed)
        for node in topological_order(self):
            node._propagate(node.grad)

    def trace(self) -> tuple[TraceRow, ...]:
        """把整张图打印成一张表（值 + 梯度 + 运算名），顺序是"从叶到根"."""
        rows = [
            TraceRow(
                label=node.label,
                operation=node.operation,
                value=node.value,
                grad=node.grad,
            )
            for node in reversed(topological_order(self))
        ]
        return tuple(rows)


def topological_order(root: Scalar) -> tuple[Scalar, ...]:
    """返回从 ``root`` 到叶子的拓扑顺序（**正是反向传播要的顺序**）.

    用迭代而不是递归：一条 10 万步的链会直接把递归深度打爆
    （Python 缺省上限 1000），而"层数一多就崩"这种失败很难与"模型太大"区分开。
    """
    order: list[Scalar] = []
    visited: set[int] = set()
    stack: list[tuple[Scalar, bool]] = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if id(node) in visited:
            continue
        visited.add(id(node))
        stack.append((node, True))
        for parent in node._parents:
            if id(parent) not in visited:
                stack.append((parent, False))
    order.reverse()  # 收集顺序是"叶在前"，翻转后是"根在前"
    return tuple(order)


def ensure_scalar(value: Scalar | float | int, *, name: str = "值") -> Scalar:
    """把一个普通数包成常量节点（已经是节点就原样返回）.

    常量节点的 ``requires_grad=False``：它参与建图（值要参与计算），
    但不接收梯度——否则每一处 ``x * 2`` 都会留下一个永远为 0 的梯度节点。
    """
    if isinstance(value, Scalar):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NumericError(
            f"{name}必须是实数或 Scalar，收到 {type(value).__name__}："
            "'字符串加数字'这类错误在这里拦下比在算梯度时拦下便宜得多。"
        )
    return Scalar(float(value), label=f"const({float(value)!r})", requires_grad=False)


def _accumulate(nodes: tuple[Scalar, ...], contributions: tuple[float, ...]) -> None:
    """把若干"上游梯度 × 局部导数"累加到对应节点上（**一致用 +=**）.

    这是全模块唯一写梯度的地方。合成一个函数而不是每处写两行，
    是为了让"用 ``+=`` 而不是 ``=``"这条纪律只有一处可能被写错。
    """
    for node, contribution in zip(nodes, contributions):
        if node.requires_grad:
            node.grad += contribution


def _sigmoid(value: float) -> float:
    """数值稳定的 sigmoid（按符号分支，``x`` 很负时中间量不溢出）."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def value_and_grad(
    expression: ScalarExpression,
    inputs: Sequence[float],
) -> tuple[float, tuple[float, ...]]:
    """对 ``expression`` 在 ``inputs`` 处求值并求梯度，返回 ``(值, 梯度元组)``.

    这是本模块最常用的入口：一串普通数进、一串普通数出，
    "计算图"这件事完全留在内部。它也是 :mod:`gradcheck` 里
    ``autograd_chain`` 那一项的一侧（另一侧是同一点的数值差分）。
    """
    if not inputs:
        raise ParameterError("至少需要一个输入：没有输入就谈不上'对谁求导'。")
    nodes = [
        Scalar(float(value), label=f"x{index}") for index, value in enumerate(inputs)
    ]
    output = expression(nodes)
    if not isinstance(output, Scalar):
        raise ParameterError(
            f"表达式必须返回一个 Scalar，收到 {type(output).__name__}："
            "返回一串数的表达式是向量函数，它的导数是雅可比矩阵（见 calculus.jacobian）。"
        )
    output.backward()
    return output.value, tuple(node.grad for node in nodes)


def gradients_of(root: Scalar) -> dict[str, float]:
    """把图上有名字的节点的梯度导成字典（``label -> grad``）.

    用于演示与报告：一个节点的标签重复时后写的覆盖前写的，
    因此这里的输出**适合看，不适合当数据源**（要做数据源请用 ``trace()``）。
    """
    return {node.label: node.grad for node in topological_order(root)}


__all__ = [
    "Scalar",
    "ScalarExpression",
    "ScalarInputs",
    "TraceRow",
    "ensure_scalar",
    "gradients_of",
    "topological_order",
    "value_and_grad",
]
