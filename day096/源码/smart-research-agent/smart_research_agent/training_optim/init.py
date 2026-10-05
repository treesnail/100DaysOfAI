"""``training_optim`` 的权重初始化：一个尺度、一条方差（day081 / M7-D6）.

## 初始化只在做一件事

```text
给每一层的权重挑一个**幅度**，而这个幅度决定"信号穿过这一层之后是变大还是变小"
```

day080 的逐层增益 ``‖y_i‖/‖x_i‖`` 是这件事的读数：初始化太大时它一路放大，
太小时一路缩小，而"刚好"的那一档有一个**可以手算的名字**：

```text
Xavier/Glorot    U(−a, a)，a = √(6/(fan_in + fan_out))   ⇒  std = √(2/(fan_in + fan_out))
Kaiming/He       U(−b, b)，b = √(6/fan_in)               ⇒  std = √(2/fan_in)
正态             N(0, 1/fan_in)                          ⇒  std = √(1/fan_in)
```

三条公式都来自同一句话：**让"每一层输出的方差 ≈ 输入的方差"**。
其中的差别只在"对 ReLU 把一半单元关掉"这件事做不做补偿：

```text
Xavier   假设激活是线性的（原论文配 tanh/sigmoid）      ⇒ 分母里 fan_in 与 fan_out 都进
Kaiming   假设激活是 ReLU（一半单元输出 0）             ⇒ 分子从 2 变成 2（把那一半补回来）
```

## 本包做三件事

```text
init_matrix            按方案与 fan 造一个矩阵（**确定性**：种子决定一切）
expected_std           给出**理论标准差**（上面那三条公式）
initialization_report  逐个矩阵把"实测标准差"与"理论值"并排印出来（比值应当在 1 附近）
```

第三条是这一课的护栏：**没有它，"测出来的 std 对不对"只能靠印象**。
论文里的公式是一个**在无穷多样本下的期望**，而一个 6×6 的矩阵只有 36 个数——
实测值与理论值差几个百分点是正常的，差一倍就不是了。

## 一个刻意的例外：两个 LN 的 γ/β 不被初始化

```text
γ = 1、β = 0        —— LayerNorm 的自然初始点（让初始时 LN 就是标准化本身）
```

这是 day079 定下的口径，今天不改：**它让"初始增益"这个读数只反映权重矩阵的幅度**，
而不是被 γ 的初值污染。同理，注意力那四个投影在本课是给定函数（day079 的选择），
它们的初值不进本包的实验。
"""

from __future__ import annotations

import math
from typing import Any

from smart_research_agent.math_foundations.probability import uniforms
from smart_research_agent.math_foundations.types import (
    Matrix,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.encoder_decoder.types import BlockParameters, BlockShape
from smart_research_agent.transformer_core.train import default_parameters
from smart_research_agent.transformer_stack import (
    DEFAULT_INIT_SCALE,
    StackParameters,
    StackShape,
)
from smart_research_agent.training_optim.errors import ParameterError

#: 四种初始化方案（**第一种是 day075 起的基线**，没有它就说不清"换初始化有没有用"）.
SCHEME_UNIFORM = "uniform"
SCHEME_XAVIER = "xavier_uniform"
SCHEME_KAIMING = "kaiming_uniform"
SCHEME_NORMAL = "normal"

INIT_SCHEMES: tuple[str, ...] = (
    SCHEME_UNIFORM,
    SCHEME_XAVIER,
    SCHEME_KAIMING,
    SCHEME_NORMAL,
)

SCHEME_DESCRIPTIONS: dict[str, str] = {
    SCHEME_UNIFORM: "U(−s, s)：day075 起的基线（s = 0.25），它的标准差与 fan 无关",
    SCHEME_XAVIER: "U(−√(6/(fan_in+fan_out)), +)：把 fan_in 与 fan_out 都算进去（配 tanh/sigmoid）",
    SCHEME_KAIMING: "U(−√(6/fan_in), +)：只按 fan_in（把 ReLU 关掉的那一半补回来）",
    SCHEME_NORMAL: "N(0, 1/√fan_in)：正态版本（用 Box–Muller 从同一串 LCG 变出来）",
}

#: 实测标准差与理论标准差之比必须落在这个区间内（6×6 只有 36 个数，差几个百分点正常）.
STD_RATIO_LOW = 0.5
STD_RATIO_HIGH = 2.0


def _checked_scheme(scheme: Any) -> str:
    """方案名必须是四种之一（**不给它挑一个默认值**）."""
    if scheme not in INIT_SCHEMES:
        raise ParameterError(f"不认识的初始化方案 {scheme!r}：可用取值 {list(INIT_SCHEMES)}。")
    return str(scheme)


def _checked_fan(value: Any, *, name: str) -> int:
    """``fan_in`` / ``fan_out``：``>= 1`` 的整数."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ParameterError(f"{name} 必须是 >= 1 的整数，收到 {value!r}。")
    return value


def expected_std(scheme: str, *, fan_in: int, fan_out: int, scale: float = DEFAULT_INIT_SCALE) -> float:
    """每种方案在**无穷多样本下**的期望标准差.

    ```text
    uniform          scale / √3                    （U(−s, s) 的标准差是 s/√3）
    xavier_uniform   √(2/(fan_in + fan_out))
    kaiming_uniform  √(2/fan_in)
    normal           1/√fan_in
    ```
    """
    resolved = _checked_scheme(scheme)
    resolved_in = _checked_fan(fan_in, name="fan_in")
    resolved_out = _checked_fan(fan_out, name="fan_out")
    if resolved == SCHEME_UNIFORM:
        if not 0.0 < float(scale) < 1.0:
            raise ParameterError(f"scale 必须落在 (0, 1)，收到 {scale!r}。")
        return float(scale) / math.sqrt(3.0)
    if resolved == SCHEME_XAVIER:
        return math.sqrt(2.0 / (resolved_in + resolved_out))
    if resolved == SCHEME_KAIMING:
        return math.sqrt(2.0 / resolved_in)
    return 1.0 / math.sqrt(resolved_in)


def _normal_pair(count: int, *, seed: int) -> tuple[float, ...]:
    """Box–Muller：把两串 ``[0, 1)`` 的均匀数变成一串标准正态数（确定性）."""
    left = uniforms(count, seed=seed)
    right = uniforms(count, seed=seed + 1009)
    return tuple(
        math.sqrt(-2.0 * math.log(max(a, 1e-12))) * math.cos(2.0 * math.pi * b)
        for a, b in zip(left, right, strict=True)
    )


def init_matrix(
    rows: int,
    columns: int,
    *,
    scheme: str = SCHEME_UNIFORM,
    seed: int = 7,
    scale: float = DEFAULT_INIT_SCALE,
) -> Matrix:
    """按方案造一个 ``(rows, columns)`` 的初始化矩阵（**确定性**）.

    ``rows`` 是 ``fan_out``、``columns`` 是 ``fan_in``——本包的权重按 ``y = x·Wᵀ`` 用，
    因此 ``W`` 的形状是 ``(out, in)``，与 PyTorch 的 ``nn.Linear`` 一致。
    """
    resolved = _checked_scheme(scheme)
    resolved_rows = _checked_fan(rows, name="rows")
    resolved_columns = _checked_fan(columns, name="columns")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ParameterError(f"seed 必须是整数，收到 {seed!r}。")
    count = resolved_rows * resolved_columns
    if resolved == SCHEME_NORMAL:
        std = expected_std(resolved, fan_in=resolved_columns, fan_out=resolved_rows)
        raw = tuple(value * std for value in _normal_pair(count, seed=seed))
    else:
        bound = _bound_of(resolved, resolved_columns, resolved_rows, scale)
        raw = tuple((2.0 * value - 1.0) * bound for value in uniforms(count, seed=seed))
    return tuple(
        raw[index * resolved_columns : (index + 1) * resolved_columns]
        for index in range(resolved_rows)
    )


def _bound_of(scheme: str, fan_in: int, fan_out: int, scale: float) -> float:
    """均匀方案的半宽 ``a``（三种均匀分布的半宽都从这里取，口径只有一处）."""
    if scheme == SCHEME_UNIFORM:
        if not 0.0 < float(scale) < 1.0:
            raise ParameterError(f"scale 必须落在 (0, 1)，收到 {scale!r}。")
        return float(scale)
    if scheme == SCHEME_XAVIER:
        return math.sqrt(6.0 / (fan_in + fan_out))
    return math.sqrt(6.0 / fan_in)


def measure_std(matrix: Matrix) -> float:
    """实测标准差（**有偏**：除以 n，与 day079 的 LayerNorm 口径一致）."""
    checked = validate_matrix(matrix, name="matrix")
    flat = [value for row in checked for value in row]
    if not flat:
        raise ParameterError("空矩阵没有标准差。")
    mean = math.fsum(flat) / len(flat)
    variance = math.fsum((value - mean) ** 2 for value in flat) / len(flat)
    return math.sqrt(variance)


def initialize_block(
    shape: BlockShape,
    *,
    scheme: str = SCHEME_UNIFORM,
    seed: int = 7,
    scale: float = DEFAULT_INIT_SCALE,
) -> BlockParameters:
    """按方案造一个块的八块参数（**两个 LN 保持 γ=1、β=0**）."""
    resolved = _checked_scheme(scheme)
    hidden = shape.hidden
    ffn = shape.ffn
    return BlockParameters(
        norm1_gamma=tuple(1.0 for _ in range(hidden)),
        norm1_beta=tuple(0.0 for _ in range(hidden)),
        ffn_w_in=init_matrix(ffn, hidden, scheme=resolved, seed=seed, scale=scale),
        ffn_b_in=tuple(0.0 for _ in range(ffn)),
        ffn_w_out=init_matrix(hidden, ffn, scheme=resolved, seed=seed + 1, scale=scale),
        ffn_b_out=tuple(0.0 for _ in range(hidden)),
        norm2_gamma=tuple(1.0 for _ in range(hidden)),
        norm2_beta=tuple(0.0 for _ in range(hidden)),
    )


def initialize_parameters(
    shape: StackShape,
    *,
    scheme: str = SCHEME_UNIFORM,
    seed: int = 7,
    scale: float = DEFAULT_INIT_SCALE,
) -> StackParameters:
    """按方案初始化整条链的块参数（注意力那一层沿用 day075 的给定参数）.

    两层种子各自步进（块 ``seed + i*13``、注意力 ``seed + i*17``），
    与 day080 的 ``make_stack_parameters`` 完全一致——因此换成 ``uniform`` 方案时，
    本函数与 ``make_stack_parameters`` 给出**逐位相同**的结果（第 4 节会验它）。
    """
    resolved = _checked_scheme(scheme)
    blocks = tuple(
        initialize_block(
            shape.block_shape, scheme=resolved, seed=seed + index * 13, scale=scale
        )
        for index in range(shape.layers)
    )
    attentions = tuple(
        default_parameters(
            shape.hidden, seed=seed + index * 17, scale=scale if scheme == SCHEME_UNIFORM else 0.25
        )
        for index in range(shape.layers)
    )
    return StackParameters(blocks=blocks, attentions=attentions)


def initialization_rows(
    shape: StackShape,
    *,
    scheme: str = SCHEME_UNIFORM,
    seed: int = 7,
    scale: float = DEFAULT_INIT_SCALE,
) -> tuple[tuple[str, float, float, float], ...]:
    """逐矩阵的 ``(名字, 理论 std, 实测 std, 比值)``.

    四列缺一不可：

    ```text
    理论 std    上面那三条公式给出的期望（**在无穷多样本下**）
    实测 std    这一个矩阵里真实的 36 个数算出来的
    比值        两者的商；它是"方案有没有写错"的唯一读数（`0.5 ~ 2` 之外就该怀疑）
    ```
    """
    resolved = _checked_scheme(scheme)
    block = shape.block_shape
    hidden, ffn = block.hidden, block.ffn
    specs = (
        ("ffn_w_in", ffn, hidden, seed),
        ("ffn_w_out", hidden, ffn, seed + 1),
    )
    rows: list[tuple[str, float, float, float]] = []
    for name, out_features, in_features, offset in specs:
        matrix = init_matrix(
            out_features, in_features, scheme=resolved, seed=offset, scale=scale
        )
        theoretical = expected_std(
            resolved, fan_in=in_features, fan_out=out_features, scale=scale
        )
        measured = measure_std(matrix)
        ratio = measured / theoretical if theoretical > 0 else 0.0
        rows.append((name, theoretical, measured, ratio))
    return tuple(rows)


def initialization_is_consistent(
    rows: tuple[tuple[str, float, float, float], ...],
    *,
    low: float = STD_RATIO_LOW,
    high: float = STD_RATIO_HIGH,
) -> bool:
    """所有比值都落在 ``[low, high]`` 之内（**一条可失败的判据**）."""
    if not rows:
        raise ParameterError("空的初始化报告没有可判的东西。")
    return all(low <= ratio <= high for _name, _theory, _measured, ratio in rows)


def matrix_std_ratio(matrix: Matrix, *, scheme: str, scale: float = DEFAULT_INIT_SCALE) -> float:
    """一个矩阵的实测/理论标准差之比（供演示脚本与校验使用）."""
    resolved = _checked_scheme(scheme)
    rows, columns = matrix_shape(matrix)
    theoretical = expected_std(resolved, fan_in=columns, fan_out=rows, scale=scale)
    if theoretical == 0.0:  # pragma: no cover - 三种方案的理论值都为正
        return 0.0
    return measure_std(matrix) / theoretical


__all__ = [
    "INIT_SCHEMES",
    "SCHEME_DESCRIPTIONS",
    "SCHEME_KAIMING",
    "SCHEME_NORMAL",
    "SCHEME_UNIFORM",
    "SCHEME_XAVIER",
    "STD_RATIO_HIGH",
    "STD_RATIO_LOW",
    "expected_std",
    "init_matrix",
    "initialization_is_consistent",
    "initialization_rows",
    "initialize_block",
    "initialize_parameters",
    "matrix_std_ratio",
    "measure_std",
]
