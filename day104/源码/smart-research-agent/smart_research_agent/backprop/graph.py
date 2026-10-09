"""一张**向量级**的计算图：把同一批梯度再自动算一遍（day090 / M8-D2）.

day074 的 ``math_foundations.autograd`` 是一张**标量**图：一个参数一个节点。
它把"链式法则可以被自动跑一遍"这件事讲清楚了，但它跑不动一层全连接——
一层 ``(32, 4)`` 的权重有两千多个数，标量图会需要两千多个节点。

今天补上另一半：**一张张量级（向量 / 矩阵为节点）的图**。

```text
节点          一个矩阵、一个向量或一个标量（用嵌套元组表示，不可变）
边            一次运算（matmul / 加偏置 / 激活 / softmax / 损失）
局部导数      每条边自己带来（就是 backprop.gradients 里那些式子）
反向          从标量损失出发，沿图从输出往输入走，把"上游 × 局部"**累加**到每个节点
```

## 一、为什么还要再做一张图（day074 已经有一张了）

```text
day074 标量图    讲清"链式法则"与"梯度必须累加"这两件事
day090 张量图    把这两件事**搬到一个真实的层上**：一次 matmul 的局部导数是一对公式
                （dA = dC·Bᵀ、dB = Aᵀ·dC），它们不是逐元素的，因此标量图里看不到
```

有了它，这一课就有了**两条独立算出梯度的路径**：

```text
路径 A   手写反向（layers / network 里逐层写出来的 dW、db、dx）
路径 B   计算图自动微分（本模块）
路径 C   数值差分（day074 的 calculus.gradient，**不依赖任何推导**）
```

三条路径在同一个点上给出同一个数——这是这一课唯一真正的判据（第 6、7 条性质）。
``GradientError`` 正是在"三条路径里有两条不一致"时才会被抛出来。

## 二、三个约定（每一个的反面都是"不报错但结果错"）

```text
① 值一旦建图就不改         原地改 value 会让已算出的梯度与新的前向值不对应
② 梯度一律用累加           一个值被用了 k 次，梯度是 k 条路径之和（写成 = 会让梯度偏小）
③ requires_grad=False 是常量  它参与计算，但不接收梯度——否则每处 ×2 都会留下一个 0 梯度节点
```

三条与 day074 的标量图**逐字相同**。同一套纪律换一个载体，仍然必须写下来：
"我知道标量图要这样，所以张量图也一样"是一个**假设**，而假设不会被测试覆盖。

## 三、这一层的边界（写下来，免得被当成框架）

```text
只支持本课用得到的算子    matmul / 加偏置 / 五个逐元素激活 / 逐行 softmax / mse / 交叉熵
不做广播                  加偏置是唯一的"广播"，且它被单独写成一个算子
不做原地复用              同一张图反向两次必须显式 zero_grad（与 day074 同一口径）
```
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from smart_research_agent.backprop import gradients as local
from smart_research_agent.backprop.errors import ChainError, ParameterError, ShapeError

#: 张量：标量（float）、向量（tuple[float, ...]）或矩阵（tuple[tuple[float, ...], ...]）.
Tensor = Any

#: 一条边的局部导数规则：接收本节点的梯度，把贡献写进父节点.
Propagate = Callable[[Tensor], None]


# --------------------------------------------------------------------------------------
# 1. 形状工具（张量是一层嵌套元组，因此形状要现场看出来）
# --------------------------------------------------------------------------------------


def is_scalar(value: Tensor) -> bool:
    """是不是一个标量（``int`` / ``float``，``bool`` 被显式排除）."""
    return not isinstance(value, bool) and isinstance(value, (int, float))


def is_vector(value: Tensor) -> bool:
    """是不是一个向量（非空、元素全是标量的元组）."""
    return (
        isinstance(value, tuple)
        and len(value) > 0
        and all(is_scalar(item) for item in value)
    )


def is_matrix(value: Tensor) -> bool:
    """是不是一个矩阵（非空、每行都是等长非空向量的元组）."""
    if not isinstance(value, tuple) or not value or not all(isinstance(row, tuple) for row in value):
        return False
    width = len(value[0])
    return width > 0 and all(is_vector(row) and len(row) == width for row in value)


def tensor_shape(value: Tensor) -> tuple[int, ...]:
    """张量的形状：``()`` / ``(n,)`` / ``(rows, columns)``（未知结构抛 ``ShapeError``）."""
    if is_scalar(value):
        return ()
    if is_vector(value):
        return (len(value),)
    if is_matrix(value):
        return (len(value), len(value[0]))
    raise ShapeError(f"无法识别的张量结构 {value!r}：只支持标量、向量与等长矩阵。")


def zeros_like(value: Tensor) -> Tensor:
    """同形状的全零张量（梯度初值）."""
    if is_scalar(value):
        return 0.0
    if is_vector(value):
        return tuple(0.0 for _ in value)
    if is_matrix(value):
        return tuple(tuple(0.0 for _ in row) for row in value)
    raise ShapeError(f"无法识别的张量结构 {value!r}。")


def ones_like(value: Tensor) -> Tensor:
    """同形状的全一张量（``sum_all`` 的反向把 ``1.0`` 撒回每一个位置）."""
    if is_scalar(value):
        return 1.0
    if is_vector(value):
        return tuple(1.0 for _ in value)
    if is_matrix(value):
        return tuple(tuple(1.0 for _ in row) for row in value)
    raise ShapeError(f"无法识别的张量结构 {value!r}。")


def add_values(left: Tensor, right: Tensor) -> Tensor:
    """两个**同形**张量逐元素相加（这是"梯度累加"的唯一实现处）.

    写成函数而不是每处写两层循环，是为了让"用累加而不是覆盖"这条纪律
    只有一处可能被写错。
    """
    if is_scalar(left) and is_scalar(right):
        return float(left) + float(right)
    if is_vector(left) and is_vector(right) and len(left) == len(right):
        return tuple(float(a) + float(b) for a, b in zip(left, right, strict=True))
    if is_matrix(left) and is_matrix(right) and tensor_shape(left) == tensor_shape(right):
        return tuple(
            tuple(float(a) + float(b) for a, b in zip(lrow, rrow, strict=True))
            for lrow, rrow in zip(left, right, strict=True)
        )
    raise ShapeError(f"形状不一致，无法累加：{tensor_shape(left)} 与 {tensor_shape(right)}。")


def flatten_value(value: Tensor) -> tuple[float, ...]:
    """把张量按行优先压平成一串数（对照两侧必须用**同一个顺序**）."""
    if is_scalar(value):
        return (float(value),)
    if is_vector(value):
        return tuple(float(item) for item in value)
    if is_matrix(value):
        return tuple(float(item) for row in value for item in row)
    raise ShapeError(f"无法识别的张量结构 {value!r}。")


def _matmul(left: Tensor, right: Tensor) -> Tensor:
    """矩阵乘 ``A·B``（内维必须一致）."""
    if not (is_matrix(left) and is_matrix(right)):
        raise ShapeError("节点级 matmul 只接受两个矩阵（加偏置是单独的算子）。")
    rows = len(left)
    inner = len(left[0])
    if len(right) != inner:
        raise ShapeError(f"矩阵乘的内维不一致：{inner} != {len(right)}。")
    columns = len(right[0])
    return tuple(
        tuple(
            math.fsum(float(left[row][k]) * float(right[k][column]) for k in range(inner))
            for column in range(columns)
        )
        for row in range(rows)
    )


def _transpose(matrix: Tensor) -> Tensor:
    """转置 ``Aᵀ``（两个矩阵乘法的局部导数各要用一次对方的转置）."""
    if not is_matrix(matrix):
        raise ShapeError("转置只接受矩阵。")
    return tuple(
        tuple(float(matrix[row][column]) for row in range(len(matrix)))
        for column in range(len(matrix[0]))
    )


# --------------------------------------------------------------------------------------
# 2. 节点
# --------------------------------------------------------------------------------------


class Node:
    """计算图上的一个张量节点：值 + 梯度 + 父节点 + 一条局部导数规则.

    ``Node`` 与 day074 的 ``Scalar`` 是同一个概念的两个载体：那个只装一个数，
    这个装一个矩阵。三个约定（不改值、累加梯度、常量不接收梯度）逐字相同。
    """

    __slots__ = ("value", "grad", "label", "operation", "_parents", "_propagate", "requires_grad")

    def __init__(
        self,
        value: Tensor,
        *,
        label: str = "",
        operation: str = "leaf",
        requires_grad: bool = True,
    ) -> None:
        if not (is_scalar(value) or is_vector(value) or is_matrix(value)):
            raise ShapeError(
                f"Node 的值必须是标量 / 向量 / 矩阵，收到 {type(value).__name__}（{value!r}）："
                "一个结构不明的值会在第一次运算时变成 IndexError，而栈会深到看不出根因。"
            )
        for item in flatten_value(value):
            if not math.isfinite(item):
                raise ShapeError(
                    f"Node 的值必须由有限数组成，收到 {item!r}："
                    "nan / inf 会顺着整条链把梯度全变成 nan，"
                    "而'梯度是 nan'看起来像学习率爆炸，根因却在更早的一次除法上。"
                )
        self.value = value if is_scalar(value) else tuple(value)
        self.grad: Tensor = zeros_like(self.value)
        self.label = label or f"node({operation})"
        self.operation = operation
        self.requires_grad = requires_grad
        self._parents: tuple[Node, ...] = ()
        self._propagate: Propagate = lambda upstream: None

    # ------------------------------------------------------------------ 建图入口

    @classmethod
    def _from(
        cls,
        value: Tensor,
        *,
        operation: str,
        parents: tuple[Node, ...],
        propagate: Propagate,
        label: str = "",
    ) -> Node:
        """造一个运算节点（值 + 父节点 + 局部导数规则），供所有算子复用.

        ``propagate`` 接收的是**本节点自己的梯度**（上游梯度），而不是"某个父节点的梯度"。
        写成"闭包里读某个父节点的 grad"是这一层最容易犯的错——它在只有一条链、
        且那个父节点恰好是叶子时甚至能算对（叶子的梯度初值是 0），
        因此简单的例子上根本发现不了（这条与 day074 的标量图同源）。
        """
        node = cls(value, label=label, operation=operation)
        node.requires_grad = any(parent.requires_grad for parent in parents)
        node._parents = parents
        node._propagate = propagate
        return node

    # ------------------------------------------------------------------ 反向

    def zero_grad(self) -> None:
        """把整张图的梯度清零（**图可以复用，梯度不会自己清零**）.

        不清零的后果非常具体：第二次 ``backward()`` 会把梯度**加上去**，
        于是"这一步的梯度"变成"两步梯度之和"——它不报错，只会让训练看起来
        "梯度比预期大两倍"。这一条属于 :class:`backprop.errors.ChainError` 描述的接线问题。
        """
        for node in topological_order(self):
            node.grad = zeros_like(node.value)

    def backward(self, seed: float = 1.0) -> None:
        """从本节点出发做反向传播（``seed`` 是"输出对输出自己的导数"，通常取 1.0）.

        顺序是 :func:`topological_order` 给出的"根在前"——每个节点在被处理时，
        它的梯度已经收齐了所有下游路径的贡献。**每次 backward 之前应当显式 zero_grad**。
        """
        if not math.isfinite(seed):
            raise ParameterError(f"seed 必须是有限实数，收到 {seed!r}。")
        if not is_scalar(self.value):
            raise ShapeError(
                f"backward 只能从**标量**出发，而本节点的形状是 {tensor_shape(self.value)}："
                "对一个非标量输出求梯度需要先给每个分量一个权重（那是 seed 的用处）。"
            )
        self.grad = float(seed)
        for node in topological_order(self):
            node._propagate(node.grad)

    def __repr__(self) -> str:  # pragma: no cover - 只影响调试输出
        return f"Node(op={self.operation!r}, shape={tensor_shape(self.value)}, label={self.label!r})"


def _accumulate(nodes: tuple[Node, ...], contributions: tuple[Tensor, ...]) -> None:
    """把若干"上游梯度 × 局部导数"**累加**到对应父节点上（全模块唯一写梯度的地方）.

    一致使用 :func:`add_values`（即 ``+=`` 的语义）。写成覆盖的后果是
    "一个值被用了 k 次时梯度只保留最后一条路径"——它不报错，只是偏小。
    """
    for node, contribution in zip(nodes, contributions, strict=True):
        if node.requires_grad:
            node.grad = add_values(node.grad, contribution)


def topological_order(root: Node) -> tuple[Node, ...]:
    """返回从 ``root`` 到叶子的拓扑顺序（**正是反向传播要的顺序**）.

    用迭代而不是递归：一条 10 万步的链会直接把递归深度打爆
    （Python 缺省上限 1000），而"层数一多就崩"这种失败很难与"模型太大"区分开。
    """
    order: list[Node] = []
    visited: set[int] = set()
    stack: list[tuple[Node, bool]] = [(root, False)]
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


# --------------------------------------------------------------------------------------
# 3. 算子（每一个都带自己的局部导数）
# --------------------------------------------------------------------------------------


def variable(value: Tensor, *, label: str = "") -> Node:
    """一个**需要梯度**的输入节点（叶子）."""
    return Node(value, label=label, operation="variable", requires_grad=True)


def constant(value: Tensor, *, label: str = "") -> Node:
    """一个**常量**节点：它参与计算，但不接收梯度.

    常量不接收梯度的理由与 day074 同源：否则每一处 ``x * 2`` 都会在图上留下
    一个永远为 0 的梯度节点，而"这个参数不学"与"它的梯度恰好是 0"在读表时长得一样。
    """
    return Node(value, label=label, operation="constant", requires_grad=False)


def ensure_node(value: Tensor | Node, *, name: str = "值") -> Node:
    """把一个裸张量包成常量节点（已经是节点就原样返回）."""
    if isinstance(value, Node):
        return value
    return constant(value, label=name)


def matmul(left: Tensor | Node, right: Tensor | Node) -> Node:
    """``C = A·B``：局部导数 ``dA = dC·Bᵀ``、``dB = Aᵀ·dC``.

    两条公式里各出现一次**对方的转置**——这是"矩阵乘法不是逐元素运算"的直接体现，
    也是标量图里看不到的东西。
    """
    left_node = ensure_node(left, name="A")
    right_node = ensure_node(right, name="B")
    product = _matmul(left_node.value, right_node.value)
    return Node._from(
        product,
        operation="matmul",
        parents=(left_node, right_node),
        propagate=lambda upstream: _accumulate(
            (left_node, right_node),
            (
                _matmul(upstream, _transpose(right_node.value)),
                _matmul(_transpose(left_node.value), upstream),
            ),
        ),
    )


def add_bias(inputs: Tensor | Node, bias: Tensor | Node) -> Node:
    """``Y = Z + b``（偏置**逐行**广播）：局部导数 ``dZ = dY``、``db = dY 的列求和``.

    偏置被所有行共用，因此它的梯度是"每一行贡献之和"——这是"广播算子的反向是求和"
    这条一般规则的最小例子。
    """
    input_node = ensure_node(inputs, name="Z")
    bias_node = ensure_node(bias, name="b")
    if not is_matrix(input_node.value) or not is_vector(bias_node.value):
        raise ShapeError("add_bias 需要一个矩阵与一个向量（偏置逐行广播）。")
    if len(bias_node.value) != len(input_node.value[0]):
        raise ShapeError(
            f"偏置长度 {len(bias_node.value)} 与列数 {len(input_node.value[0])} 不一致。"
        )
    value = tuple(
        tuple(float(cell) + float(b) for cell, b in zip(row, bias_node.value, strict=True))
        for row in input_node.value
    )

    def propagate(upstream: Tensor) -> None:
        columns = len(bias_node.value)
        grad_bias = tuple(
            math.fsum(float(upstream[row][column]) for row in range(len(upstream)))
            for column in range(columns)
        )
        _accumulate((input_node, bias_node), (upstream, grad_bias))

    return Node._from(value, operation="add_bias", parents=(input_node, bias_node), propagate=propagate)


def dense(inputs: Tensor | Node, weight: Tensor | Node, bias: Tensor | Node) -> Node:
    """``Y = X·Wᵀ + b``（与 ``nn.Linear`` / ``dense_linear`` **同口径**）.

    它是这一课最重要的一个节点算子，因为它把三块梯度**一次**全写下来——
    这也是"两条独立路径"里那条**自动微分**的路：

    ```text
    前向   Y[i][j] = Σ_k X[i][k]·W[j][k] + b[j]
    dX     dX[i][k] = Σ_j dY[i][j]·W[j][k]      （按 W 的行收）
    dW     dW[j][k] = Σ_i dY[i][j]·X[i][k]      （回传梯度的每一列 · 输入）
    db     db[j]    = Σ_i dY[i][j]              （偏置被所有行共用 ⇒ 按行求和）
    ```

    **它与 :func:`backprop.layers.dense_backward` 是两份独立的实现**：
    那一份是"手写反向"（本课的主角），这一份是"图的自动微分"。两份对上，
    才说明手写的那份没有抄错——否则"两边一起错"是可能的（这一点与 day074
    的标量图、``gradcheck`` 的关系完全同源）。
    """
    input_node = ensure_node(inputs, name="X")
    weight_node = ensure_node(weight, name="W")
    bias_node = ensure_node(bias, name="b")
    if not is_matrix(input_node.value) or not is_matrix(weight_node.value) or not is_vector(bias_node.value):
        raise ShapeError("dense 需要 输入矩阵、权重矩阵 与 偏置向量。")
    rows = len(input_node.value)
    in_features = len(input_node.value[0])
    out_features = len(weight_node.value)
    if len(weight_node.value[0]) != in_features:
        raise ShapeError(
            f"权重的列数 {len(weight_node.value[0])} 与输入的列数 {in_features} 不一致。"
        )
    if len(bias_node.value) != out_features:
        raise ShapeError(f"偏置长度 {len(bias_node.value)} 与权重的行数 {out_features} 不一致。")
    value = tuple(
        tuple(
            math.fsum(
                float(input_node.value[row][k]) * float(weight_node.value[column][k])
                for k in range(in_features)
            )
            + float(bias_node.value[column])
            for column in range(out_features)
        )
        for row in range(rows)
    )

    def propagate(upstream: Tensor) -> None:
        grad_inputs = tuple(
            tuple(
                math.fsum(
                    float(upstream[row][column]) * float(weight_node.value[column][k])
                    for column in range(out_features)
                )
                for k in range(in_features)
            )
            for row in range(rows)
        )
        grad_weight = tuple(
            tuple(
                math.fsum(
                    float(upstream[row][column]) * float(input_node.value[row][k])
                    for row in range(rows)
                )
                for k in range(in_features)
            )
            for column in range(out_features)
        )
        grad_bias = tuple(
            math.fsum(float(upstream[row][column]) for row in range(rows))
            for column in range(out_features)
        )
        _accumulate((input_node, weight_node, bias_node), (grad_inputs, grad_weight, grad_bias))

    return Node._from(
        value, operation="dense", parents=(input_node, weight_node, bias_node), propagate=propagate
    )


def sum_all(inputs: Tensor | Node) -> Node:
    """把任意形状的张量求和成**一个标量**（反向：每个元素都拿到 ``1.0``）.

    它是这一课最小的归约算子，存在有两个用途：

    ```text
    一  让"从任意张量出发做一次 backward"有入口（backward 只从标量出发）
    二  它是"广播/求和"这一类算子的最小例子：前向把很多数压成一个，
       反向把同一个数**原样撒回**每一个位置
    ```

    和函数的使用方式与其它算子一致：``sum_all(x).backward()``。
    """
    input_node = ensure_node(inputs, name="x")
    value = math.fsum(flatten_value(input_node.value))

    def propagate(upstream: Tensor) -> None:
        _accumulate((input_node,), (ones_like(input_node.value),))

    return Node._from(value, operation="sum_all", parents=(input_node,), propagate=propagate)


def _elementwise(function: Callable[[float], float], node_value: Tensor) -> Tensor:
    """逐元素地把一个一元函数作用在任意形状的张量上."""
    if is_scalar(node_value):
        return float(function(float(node_value)))
    if is_vector(node_value):
        return tuple(float(function(float(item))) for item in node_value)
    if is_matrix(node_value):
        return tuple(
            tuple(float(function(float(item))) for item in row) for row in node_value
        )
    raise ShapeError(f"无法识别的张量结构 {node_value!r}。")


def relu(inputs: Tensor | Node) -> Node:
    """``y = max(0, x)``：反向是一个**开关**（``x > 0`` 原样通过，否则恰好 0）."""
    input_node = ensure_node(inputs, name="x")
    reference = input_node.value
    value = _elementwise(lambda item: item if item > 0.0 else 0.0, reference)

    def propagate(upstream: Tensor) -> None:
        backward = local.elementwise_backward("relu", _as_matrix(reference), _as_matrix(upstream))
        _accumulate((input_node,), (_from_matrix(backward, reference),))

    return Node._from(value, operation="relu", parents=(input_node,), propagate=propagate)


def gelu(inputs: Tensor | Node) -> Node:
    """``y = gelu(x)``（精确 erf 式）：反向需要**激活前的值**（``x·e^{−x²/2}`` 那一项）."""
    from smart_research_agent.neural_basics.activations import gelu as gelu_forward

    input_node = ensure_node(inputs, name="x")
    reference = input_node.value
    value = _elementwise(gelu_forward, reference)

    def propagate(upstream: Tensor) -> None:
        backward = local.elementwise_backward("gelu", _as_matrix(reference), _as_matrix(upstream))
        _accumulate((input_node,), (_from_matrix(backward, reference),))

    return Node._from(value, operation="gelu", parents=(input_node,), propagate=propagate)


def softmax_rows(inputs: Tensor | Node) -> Node:
    """``P = softmax(Z)``（逐行）：反向 ``dZ = P⊙(dP − ⟨dP, P⟩行)``（不建 n×n 雅可比）."""
    from smart_research_agent.neural_basics.activations import softmax as softmax_forward

    input_node = ensure_node(inputs, name="Z")
    if not is_matrix(input_node.value):
        raise ShapeError("softmax_rows 需要一行以上的 logits（每一行各自归一）。")
    value = tuple(softmax_forward(row) for row in input_node.value)

    def propagate(upstream: Tensor) -> None:
        _accumulate((input_node,), (local.softmax_backward_rows(value, _as_matrix(upstream)),))

    return Node._from(value, operation="softmax_rows", parents=(input_node,), propagate=propagate)


def mse(predictions: Tensor | Node, targets: Sequence[Sequence[float]]) -> Node:
    """``L = mean((P − T)²)``：反向 ``dP = 2(P − T)/N``（``N`` = 全部元素个数）."""
    prediction_node = ensure_node(predictions, name="P")
    if not is_matrix(prediction_node.value):
        raise ShapeError("mse 需要一个矩阵作为预测。")
    reference = tuple(tuple(float(item) for item in row) for row in targets)
    if tensor_shape(prediction_node.value) != (len(reference), len(reference[0]) if reference else 0):
        raise ShapeError(
            f"预测的形状 {tensor_shape(prediction_node.value)} 与目标的形状 "
            f"{(len(reference), len(reference[0]) if reference else 0)} 不一致。"
        )
    value = math.fsum(
        (float(pred) - target) ** 2
        for pred_row, target_row in zip(prediction_node.value, reference, strict=True)
        for pred, target in zip(pred_row, target_row, strict=True)
    ) / sum(len(row) for row in reference)

    def propagate(upstream: Tensor) -> None:
        _accumulate((prediction_node,), (local.mse_grad(prediction_node.value, reference),))

    return Node._from(value, operation="mse", parents=(prediction_node,), propagate=propagate)


def cross_entropy(logits: Tensor | Node, targets: Sequence[int]) -> Node:
    """逐行交叉熵的**批均值**：反向 ``dZ = (p − onehot)/rows``.

    除以行数是因为损失是批均值。忘记这个分母会让梯度大 ``rows`` 倍，
    而它看起来只是"这一批的学习率需要除以 batch size"。
    """
    logit_node = ensure_node(logits, name="Z")
    reference = tuple(int(item) for item in targets)
    if not is_matrix(logit_node.value):
        raise ShapeError("cross_entropy 需要一行以上的 logits。")
    if len(logit_node.value) != len(reference):
        raise ShapeError(
            f"打分有 {len(logit_node.value)} 行而标签有 {len(reference)} 个：必须一一对应。"
        )
    from smart_research_agent.neural_basics.losses import cross_entropy as cross_entropy_forward

    total = math.fsum(
        cross_entropy_forward(row, target)
        for row, target in zip(logit_node.value, reference, strict=True)
    )
    value = total / len(reference)

    def propagate(upstream: Tensor) -> None:
        _accumulate((logit_node,), (local.cross_entropy_grad_rows(logit_node.value, reference),))

    return Node._from(value, operation="cross_entropy", parents=(logit_node,), propagate=propagate)


def _as_matrix(value: Tensor) -> tuple[tuple[float, ...], ...]:
    """把任意形状的张量看作矩阵（标量 → 1×1、向量 → 1×n），供逐元素反向复用.

    这不是"泛化"，而是**为了让 relu / gelu 的反向只需要一份实现**：
    它们逐元素，所以形状不需要被理解。
    """
    if is_scalar(value):
        return ((float(value),),)
    if is_vector(value):
        return (tuple(float(item) for item in value),)
    if is_matrix(value):
        return tuple(tuple(float(item) for item in row) for row in value)
    raise ShapeError(f"无法识别的张量结构 {value!r}。")


def _from_matrix(matrix: tuple[tuple[float, ...], ...], reference: Tensor) -> Tensor:
    """把 :func:`_as_matrix` 的结果还原成 ``reference`` 的形状（逐元素算子用）."""
    if is_scalar(reference):
        return matrix[0][0]
    if is_vector(reference):
        return tuple(matrix[0])
    return matrix


def grad_of(node: Node) -> Tensor:
    """读一个节点的梯度（不触发任何计算）."""
    if node.grad is None:  # pragma: no cover - grad 恒有初值，这里只是防御
        raise ChainError(f"节点 {node.label!r} 还没有梯度：反向传播可能根本没跑到它。")
    return node.grad


__all__ = [
    "Node",
    "Tensor",
    "add_bias",
    "add_values",
    "constant",
    "cross_entropy",
    "dense",
    "ensure_node",
    "flatten_value",
    "gelu",
    "grad_of",
    "is_matrix",
    "is_scalar",
    "is_vector",
    "matmul",
    "mse",
    "ones_like",
    "relu",
    "softmax_rows",
    "sum_all",
    "tensor_shape",
    "topological_order",
    "variable",
    "zeros_like",
]
