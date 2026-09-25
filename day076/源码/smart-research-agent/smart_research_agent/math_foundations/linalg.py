"""线性代数：从点积到低秩近似，十个算子全在一个文件里（day073 / Math-D1）.

这一课的第一半。十个算子的顺序不是随意的，它是**从一个向量走到一个矩阵的近似**
的一条路：

```text
dot → norm → normalize → cosine        两个向量之间的关系（几何）
matmul → transpose → projection        矩阵作为"线性变换"
softmax                                把打分变成概率（几何 → 概率的桥）
power_iteration → low_rank             反复乘一个矩阵会浮现出什么
```

## 三处刻意的实现选择

**1. 一律用纯 Python，不引入 numpy。**

```text
确定性    同一份输入永远同一份输出（浮点求和顺序固定，见下）
零依赖    这一层可以被任何环境导入，包括没有 numpy 的教学机
可读性    三行 Python 比一行 numpy 更接近公式本身——这一课的目的是看懂公式
```

代价是慢：低秩近似在几百维上就该换 numpy（本课只在小矩阵上跑）。
这条取舍必须写下来，否则"为什么不用 numpy"会变成一个说不清的问题。

**2. 求和一律用 ``math.fsum``，不用内置 ``sum``。**

``sum`` 是**顺序**累加，浮点误差会随维度线性累积；``fsum`` 用精确累加
（Shewchuk 算法）把误差压到与顺序无关。它与"检索里两处余弦必须一致"
是同一类要求：**同一个式子在不同维度上不该给出系统性偏差**。

**3. 零向量的处理与项目里既有实现**刻意不同**（见 ``errors`` 的说明）。**

```text
normalize(零向量)    → 报错      （零向量没有方向，这不是数学层能替调用方决定的事）
cosine(零向量, ·)    → 0.0       （与 vectorstore.metrics 的口径一致：让查询不中断）
```

两者不是矛盾：**一个"必须拒绝"，一个"必须容错"**——归一化的结果是一个方向，
而相似度的结果是"这两个东西像不像"（不像就是 0）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from smart_research_agent.math_foundations.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    require_same_dimension,
    row_sums,
    validate_matrix,
    validate_vector,
)

#: 幂迭代的缺省上限（小矩阵上几十步就够；给一个上限是为了"永不收敛"时有出路）.
DEFAULT_MAX_ITERATIONS = 200

#: 幂迭代的缺省收敛判据（相邻两次主向量的最大分量差）.
DEFAULT_ITERATION_TOLERANCE = 1e-12


# --------------------------------------------------------------------------- #
# 一、两个向量之间的关系
# --------------------------------------------------------------------------- #


def dot(left: Vector, right: Vector) -> float:
    """点积 ``a·b = Σ a_i b_i``（维度不一致抛 ``ShapeError``）.

    点积**只含方向信息**：把任一侧放大 k 倍，点积也放大 k 倍。
    因此"点积大"不等于"方向一致"——这正是要再除以两个模长的原因
    （看 :func:`cosine`）。
    """
    checked_left = validate_vector(left, name="left")
    checked_right = validate_vector(right, name="right")
    require_same_dimension(checked_left, checked_right)
    return math.fsum(x * y for x, y in zip(checked_left, checked_right))


def norm(vector: Vector) -> float:
    """L2 模长 ``‖a‖ = √(Σ a_i²)``（本课只做 L2，并把这条写在这里）.

    实现上先平方再开方，而不是用 ``math.hypot(*a)``：``hypot`` 在小维度上更稳，
    但它的实现细节（缩放策略）随 Python 版本变化，而这一课要求**逐位可复现**。
    """
    checked = validate_vector(vector)
    return math.sqrt(math.fsum(value * value for value in checked))


def scale(vector: Vector, factor: float) -> Vector:
    """标量乘法 ``k·a``（factor 必须是有限实数）."""
    checked = validate_vector(vector)
    if not math.isfinite(factor):
        raise NumericError(f"缩放系数必须是有限实数，收到 {factor!r}。")
    return tuple(value * factor for value in checked)


def add(left: Vector, right: Vector) -> Vector:
    """逐分量相加（维度必须一致）."""
    checked_left = validate_vector(left, name="left")
    checked_right = validate_vector(right, name="right")
    require_same_dimension(checked_left, checked_right)
    return tuple(x + y for x, y in zip(checked_left, checked_right))


def subtract(left: Vector, right: Vector) -> Vector:
    """逐分量相减（维度必须一致）."""
    checked_left = validate_vector(left, name="left")
    checked_right = validate_vector(right, name="right")
    require_same_dimension(checked_left, checked_right)
    return tuple(x - y for x, y in zip(checked_left, checked_right))


def normalize(vector: Vector) -> Vector:
    """归一化到单位长度（**零向量直接报错**，理由见模块 docstring）.

    ```text
    零向量 没有方向 → 归一化的结果应该是什么？没有人能回答
           原样返回（llm.embedding.l2_normalize 的选择）会让"单位向量"这个前提静默失效
           返回 0.0（vectorstore.metrics 的选择）是一个**相似度**的答案，不是方向的答案
    ```

    因此本函数拒绝它，并把"该怎么办"留给调用方：
    要么在上游把零向量拦掉（``vectorstore.types.VectorRecord`` 就是这么做的），
    要么显式用 :func:`cosine`（它对零向量的约定是 0.0）。
    """
    checked = validate_vector(vector)
    length = norm(checked)
    if length == 0.0:
        raise NumericError(
            "零向量没有方向，无法归一化："
            "如果你要的是'这两个东西像不像'，请用 cosine（它对零向量的约定是 0.0）；"
            "如果你要的是一个单位方向，请先在上游修掉那个零向量。"
        )
    return tuple(value / length for value in checked)


def cosine(left: Vector, right: Vector) -> float:
    """余弦相似度 ``cos(a,b) = a·b / (‖a‖‖b‖)``，值域 [-1, 1].

    与 ``vectorstore.metrics.cosine_similarity``（day064）**逐位一致**，
    包括零向量的约定：任一侧是零向量时返回 ``0.0``——
    "一条脏数据不要打断整批查询"（见那个模块的说明）。
    这条一致性由 ``bridge`` 模块逐点核对，而不是靠"两边都写得对"。
    """
    checked_left = validate_vector(left, name="left")
    checked_right = validate_vector(right, name="right")
    require_same_dimension(checked_left, checked_right)
    length_left = norm(checked_left)
    length_right = norm(checked_right)
    if length_left == 0.0 or length_right == 0.0:
        return 0.0
    return dot(checked_left, checked_right) / (length_left * length_right)


def projection(u: Vector, v: Vector) -> float:
    """把 ``u`` 投到 ``v`` 方向上的**标量**坐标 ``(u·v̂)``.

    返回标量而不是向量是刻意的：向量形式是 ``(u·v̂)·v̂``，只多一次乘法，
    而标量形式直接回答"在这个方向上有多长"——报告里要读的是这个数。
    ``v`` 是零向量时报错（没有"方向上"可言）。
    """
    return dot(u, normalize(v))


# --------------------------------------------------------------------------- #
# 二、矩阵作为线性变换
# --------------------------------------------------------------------------- #


def transpose(matrix: Matrix) -> Matrix:
    """转置 ``(Aᵀ)_ij = A_ji``（空矩阵给空矩阵）."""
    checked = validate_matrix(matrix)
    return tuple(zip(*checked))


def matmul(left: Matrix, right: Matrix) -> Matrix:
    """矩阵乘法 ``(AB)_ij = Σ_k A_ik B_kj``（内维不等长抛 ``ShapeError``）.

    ``zip`` 的静默截断是这里最想拦下的：3×4 乘 3×3 不会报错，
    它会**少算一列**（或把两个不同的 k 配到一起）。
    """
    checked_left = validate_matrix(left, name="left")
    checked_right = validate_matrix(right, name="right")
    rows_left, inner_left = matrix_shape(checked_left)
    rows_right, columns_right = matrix_shape(checked_right)
    if inner_left != rows_right:
        raise ShapeError(
            f"矩阵乘法内维不一致：左 {matrix_shape(checked_left)} × 右 "
            f"{matrix_shape(checked_right)}——需要左的列数等于右的行数，"
            "否则逐元素相乘会把 k 配错（结果仍然是一组正常的数）。"
        )
    right_columns = tuple(zip(*checked_right))
    return tuple(
        tuple(math.fsum(x * y for x, y in zip(row, column)) for column in right_columns)
        for row in checked_left
    )


def matvec(matrix: Matrix, vector: Vector) -> Vector:
    """矩阵乘向量 ``A·v``（把向量当作只有一列的矩阵）."""
    checked_matrix = validate_matrix(matrix)
    checked_vector = validate_vector(vector, name="vector")
    rows, columns = matrix_shape(checked_matrix)
    if columns != len(checked_vector):
        raise ShapeError(
            f"矩阵乘向量的维度不匹配：{matrix_shape(checked_matrix)} · {len(checked_vector)}，"
            "需要矩阵的列数等于向量的维度。"
        )
    assert rows == len(checked_matrix)
    return tuple(
        math.fsum(value * item for value, item in zip(row, checked_vector))
        for row in checked_matrix
    )


def outer(left: Vector, right: Vector) -> Matrix:
    """外积 ``(a bᵀ)_ij = a_i b_j``（低秩近似的每一层都是它）."""
    checked_left = validate_vector(left, name="left")
    checked_right = validate_vector(right, name="right")
    return tuple(tuple(x * y for y in checked_right) for x in checked_left)


def matrix_add(left: Matrix, right: Matrix) -> Matrix:
    """逐元素相加（形状必须一致）."""
    checked_left = validate_matrix(left, name="left")
    checked_right = validate_matrix(right, name="right")
    if matrix_shape(checked_left) != matrix_shape(checked_right):
        raise ShapeError(
            f"矩阵相加的形状不一致：{matrix_shape(checked_left)} != "
            f"{matrix_shape(checked_right)}。"
        )
    return tuple(
        tuple(x + y for x, y in zip(row_left, row_right))
        for row_left, row_right in zip(checked_left, checked_right)
    )


def matrix_scale(matrix: Matrix, factor: float) -> Matrix:
    """矩阵的标量乘法."""
    checked = validate_matrix(matrix)
    if not math.isfinite(factor):
        raise NumericError(f"缩放系数必须是有限实数，收到 {factor!r}。")
    return tuple(tuple(value * factor for value in row) for row in checked)


def identity(size: int) -> Matrix:
    """单位矩阵 ``I_size``（size < 1 抛 ``ParameterError``）."""
    if size < 1:
        raise ParameterError(f"单位矩阵的阶必须 >= 1，收到 {size}。")
    return tuple(
        tuple(1.0 if row == column else 0.0 for column in range(size)) for row in range(size)
    )


# --------------------------------------------------------------------------- #
# 三、从这里开始有概率的味道
# --------------------------------------------------------------------------- #


def softmax(logits: Sequence[float], *, temperature: float = 1.0) -> Vector:
    """数值稳定的 softmax（**先减最大值再取指数**）.

    ```text
    softmax(z)_i = exp(z_i / T) / Σ_j exp(z_j / T)
    ```

    减去最大值不改变结果（分子分母同乘 ``exp(−max)``），但把 ``exp`` 的
    自变量压到 ``<= 0``，从而不会上溢。**这条恒等式的价值只在于数值**：
    一个不懂它的实现会在 ``z = [1000, 1001]`` 上得到 ``[nan, nan]``
    （``exp(1000) = inf``、``inf/inf = nan``），而正确答案是 ``[0.269, 0.731]``。

    温度的作用（与 ``llm.sampling.softmax_with_temperature`` 同一口径）：

    ```text
    T → 0    概率质量集中到最大值上（趋近 one-hot，即贪心）
    T = 1    原始分布
    T → ∞    趋近均匀分布（输出近乎随机）
    ```
    """
    values = validate_vector(tuple(logits), name="logits")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ParameterError(
            f"温度必须为正的有限数，收到 {temperature!r}："
            "T = 0 的贪心语义请用 argmax 表达（本函数不做截断，"
            "否则'我调了温度'这件事会变成假的）。"
        )
    scaled = [value / temperature for value in values]
    peak = max(scaled)
    exponentials = [math.exp(value - peak) for value in scaled]
    total = math.fsum(exponentials)
    return tuple(value / total for value in exponentials)


def log_softmax(logits: Sequence[float]) -> Vector:
    """数值稳定的 log-softmax ``z − max − log Σ exp(z − max)``.

    直接算 ``log(softmax(z))`` 在概率极小时会先下溢成 ``0.0``、
    再取对数得到 ``−inf``；本表达式在同样的输入下仍然给出有限值。
    """
    values = validate_vector(tuple(logits), name="logits")
    peak = max(values)
    shifted = [value - peak for value in values]
    log_total = math.log(math.fsum(math.exp(value) for value in shifted))
    return tuple(value - log_total for value in shifted)


def argmax(vector: Vector) -> int:
    """最大值的位置（并列时取**最小的下标**——这条约定让结果可复现）."""
    checked = validate_vector(vector)
    best_index = 0
    for index in range(1, len(checked)):
        if checked[index] > checked[best_index]:
            best_index = index
    return best_index


def top_k_indices(vector: Vector, k: int) -> tuple[int, ...]:
    """按值降序取前 ``k`` 个下标（并列时按下标升序，结果确定）."""
    checked = validate_vector(vector)
    if k < 1 or k > len(checked):
        raise ParameterError(f"k 必须落在 [1, {len(checked)}]，收到 {k}。")
    order = sorted(range(len(checked)), key=lambda index: (-checked[index], index))
    return tuple(order[:k])


def row_normalize(weights: Matrix) -> Matrix:
    """把矩阵每一行归一化成和为 1 的分布（全零行会报错，不做静默兜底）."""
    checked = validate_matrix(weights)
    total = row_sums(checked)
    for index, value in enumerate(total):
        if value == 0.0:
            raise NumericError(
                f"第 {index} 行全为 0，无法归一化：一行全零在概率上没有意义"
                "（它既不是'不看任何位置'，也不是'均匀地看'）。"
            )
    return tuple(
        tuple(value / total[index] for value in row) for index, row in enumerate(checked)
    )


# --------------------------------------------------------------------------- #
# 四、反复乘一个矩阵会浮现出什么
# --------------------------------------------------------------------------- #


def power_iteration(
    matrix: Matrix,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_ITERATION_TOLERANCE,
) -> tuple[float, Vector, Vector]:
    """幂迭代：求**最大奇异值**三元组 ``(σ, u, v)``（``A v = σ u``）.

    做法是把"求 A 的主奇异向量"化成"求 ``AᵀA`` 的主特征向量"
    （后者对称半正定，一定有实特征值、且幂迭代一定收敛）：

    ```text
    v ← Aᵀ A v / ‖Aᵀ A v‖       反复做 → v 收敛到 AᵀA 的主特征向量
    σ = ‖A v‖                    （= √λ_max(AᵀA)）
    u = A v / σ
    ```

    三条拒绝：

    ```text
    迭代次数 <= 0               不知道怎么开始
    容差 <= 0                   零容差会让循环跑到上限（"收敛"永远不成立）
    矩阵全零                    归一化会除零；零矩阵的"主方向"没有定义
    ```

    **收敛性说明（写在这里而不是含糊过去）**：幂迭代的收敛速度取决于
    ``σ₂ / σ₁`` 的比值——两个最大的奇异值越接近，收敛越慢。
    返回值里不带"迭代了几次"，因为调用方真正要知道的是"σ 有多大"；
    要复核收敛请传一个更小的容差再跑一遍，两次结果一致才是真的收敛了。
    """
    checked = validate_matrix(matrix)
    if max_iterations <= 0:
        raise ParameterError(f"max_iterations 必须 >= 1，收到 {max_iterations}。")
    if tolerance <= 0:
        raise ParameterError(f"tolerance 必须为正，收到 {tolerance}。")
    columns = matrix_shape(checked)[1]
    transposed = transpose(checked)
    previous = tuple(1.0 / math.sqrt(columns) for _ in range(columns))
    vector = previous
    for _ in range(max_iterations):
        product = matvec(checked, vector)
        if all(value == 0.0 for value in product):
            raise NumericError(
                "矩阵（在起始方向上）乘出来是零向量：全零矩阵没有主方向，"
                "无法用幂迭代给出 σ / u / v——请先检查这份矩阵是怎么来的。"
            )
        candidate = matvec(transposed, product)
        length = norm(candidate)
        if length == 0.0:  # pragma: no cover - 与上面的零检测同源
            raise NumericError("幂迭代中出现零向量：矩阵的行空间在这一方向上是零。")
        vector = tuple(value / length for value in candidate)
        # 收敛判据用"两个单位向量是否平行"（|cos| → 1），而不是逐分量比大小：
        # 幂迭代的主向量可能在两步之间整体翻符号，逐分量比较会把那次翻转
        # 当成"还没收敛"，于是永远跑到迭代上限（而结果其实已经对了）。
        if abs(1.0 - abs(dot(vector, previous))) <= tolerance:
            break
        previous = vector
    product = matvec(checked, vector)
    sigma = norm(product)
    if sigma == 0.0:  # pragma: no cover - 与上面的零检测同源
        raise NumericError("σ = 0：这个矩阵的主奇异值为零（它在所有方向上都把向量压没了）。")
    u = tuple(value / sigma for value in product)
    return sigma, u, vector


def low_rank_approximation(
    matrix: Matrix,
    rank: int,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_ITERATION_TOLERANCE,
) -> tuple[Vector, tuple[Vector, ...], tuple[Vector, ...], Matrix]:
    """低秩近似：返回 ``(奇异值, u 列表, v 列表, A_r)``，``A_r = Σ_{i≤r} σ_i u_i v_iᵀ``.

    这是这一课的落点，也是 **LoRA（day051）的数学原型**：
    "一个矩阵可以被少数几个方向的叠加近似"，因此我们只需要学那几对方向，
    而不是整个矩阵。本课用**幂迭代 + 逐步正交化**实现（教学实现，不是 LAPACK）：

    ```text
    ① 对残差矩阵求主奇异三元组
    ② 把新方向对已找到的方向做 Gram–Schmidt 正交
    ③ 从残差里减掉这一层 → 回到 ①
    ④ r 层之后剩下的就是"被丢掉的能量"
    ```

    **正交化的理由必须写清**：没有它时，第二轮幂迭代会**再次收敛到同一个主方向**
    上（残差里那个方向并没有被真正去掉），于是 A_r 变成 σ₁u₁v₁ᵀ 的重复叠加——
    而它看起来仍然是一组正常的数、并且"误差也没大得离谱"。
    这类错误没有异常，只有"低秩近似好像不太准"。

    三条拒绝：``rank < 1``、``rank`` 超过矩阵列数、矩阵全零（没有方向可提）。
    """
    checked = validate_matrix(matrix)
    rows, columns = matrix_shape(checked)
    if rank < 1:
        raise ParameterError(f"秩必须 >= 1，收到 {rank}。")
    if rank > min(rows, columns):
        raise ParameterError(
            f"秩 {rank} 超过了矩阵的可用维数 {min(rows, columns)}"
            f"（形状 {matrix_shape(checked)}）：超过它只会提出零向量。"
        )
    residual = checked
    singular_values: list[float] = []
    left_vectors: list[Vector] = []
    right_vectors: list[Vector] = []
    for _ in range(rank):
        sigma, u, v = power_iteration(
            residual, max_iterations=max_iterations, tolerance=tolerance
        )
        if sigma <= tolerance:
            break
        for previous in right_vectors:
            coefficient = dot(v, previous)
            v = subtract(v, scale(previous, coefficient))
        length = norm(v)
        if length <= tolerance:
            break
        v = scale(v, 1.0 / length)
        # σ 与 u 都用**原矩阵**（而不是残差）算：递推式的每一步应该给出
        # "原矩阵在第 i 个方向上的真实强度"，而不是"残差里的强度"。
        product = matvec(checked, v)
        sigma = norm(product)
        if sigma <= tolerance:
            break
        u = scale(product, 1.0 / sigma)
        singular_values.append(sigma)
        left_vectors.append(u)
        right_vectors.append(v)
        residual = matrix_add(residual, matrix_scale(outer(u, v), -sigma))
    approximation = zeros(rows, columns)
    for sigma, u, v in zip(singular_values, left_vectors, right_vectors):
        approximation = matrix_add(approximation, matrix_scale(outer(u, v), sigma))
    return (
        tuple(singular_values),
        tuple(left_vectors),
        tuple(right_vectors),
        approximation,
    )


def zeros(rows: int, columns: int) -> Matrix:
    """全零矩阵（行/列数必须 >= 1）."""
    if rows < 1 or columns < 1:
        raise ParameterError(f"零矩阵的行列数必须 >= 1，收到 {rows}×{columns}。")
    return tuple(tuple(0.0 for _ in range(columns)) for _ in range(rows))


def frobenius_norm(matrix: Matrix) -> float:
    """Frobenius 范数 ``‖A‖_F = √Σ a_ij²``（低秩近似的误差就是按它量的）."""
    checked = validate_matrix(matrix)
    return math.sqrt(math.fsum(value * value for row in checked for value in row))


def reconstruction_error(matrix: Matrix, approximation: Matrix) -> float:
    """重建的**相对误差** ``‖A − A_r‖_F / ‖A‖_F``（分母为零时报错）."""
    checked = validate_matrix(matrix)
    approximation_shape = matrix_shape(approximation)
    if matrix_shape(checked) != approximation_shape:
        raise ShapeError(
            f"重建误差要求两个矩阵同形：{matrix_shape(checked)} != {approximation_shape}。"
        )
    base = frobenius_norm(checked)
    if base == 0.0:
        raise NumericError(
            "原矩阵的 Frobenius 范数为 0（它是全零矩阵）："
            "相对误差的分母为零，此时'误差'这个数没有意义。"
        )
    difference = matrix_add(checked, matrix_scale(validate_matrix(approximation), -1.0))
    return frobenius_norm(difference) / base


__all__ = [
    "DEFAULT_ITERATION_TOLERANCE",
    "DEFAULT_MAX_ITERATIONS",
    "add",
    "argmax",
    "cosine",
    "dot",
    "frobenius_norm",
    "identity",
    "log_softmax",
    "low_rank_approximation",
    "matmul",
    "matrix_add",
    "matrix_scale",
    "matvec",
    "norm",
    "normalize",
    "outer",
    "power_iteration",
    "projection",
    "reconstruction_error",
    "row_normalize",
    "scale",
    "softmax",
    "subtract",
    "top_k_indices",
    "transpose",
    "zeros",
]
