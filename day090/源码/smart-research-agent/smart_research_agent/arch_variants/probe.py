"""``probe.py``：用**扰动**量出“谁能看到谁”（day082 的核心判据）.

## 一、为什么不信 ``causal=True`` 这四个字

“这个模型是因果的”通常靠读代码确认。而代码可以有三处对不上：

```text
① 掩码传错了流（把因果掩码给了编码器、把填充掩码给了解码器）
② 掩码算对了但没传进去（函数签名里少了一个参数——形状照样合法）
③ 反向里没把掩码穿过去（前向因果、反向把未来位置的梯度也算上了）
```

第 ③ 条最隐蔽：**前向完全正确**，只有梯度里混进了“未来”的贡献。
因此本课换一条路取证：**扰动输入的第 j 个 token，重新前向，看输出第 i 行变了多少**。

```text
被掩码挡掉的 (i, j)      Δ **恰好 0.0**
允许的 (i, j)            Δ 明显大于 0
```

## 二、为什么“恰好 0.0”而不是“很小”

这不是数值上的巧合，而是实现上的一条硬事实：day073 的 ``masked_softmax_rows``
**只在允许的位置上做 softmax**，被屏蔽的那些打分**从来没有被读过**：

```text
scores[i][j] 算了（它在矩阵里）   但 exp/sum/加权三步都不碰它
⇒ 改它不会改变任何一个中间量 ⇒ 输出的每一个浮点都逐位相同
```

所以本课的容差是 ``0.0``——**逐位**判据。而它有一个反证：
把掩码换成全开之后，同一张表上那些格子立刻变成非零（第 5 章那张表里的第 5 行）。
一条判据如果两种情况下都给“通过”，它就没有分辨力。

## 三、两条流各量一次

```text
主流（自注意力）    mask 是 (n_tgt, n_tgt) 的方阵，量的是“第 i 个输出看不看第 j 个输入”
源流（交叉注意力）  没有掩码，量的是“第 i 个解码位置看不看第 j 个源位置”——
                    这张表是 **(n_tgt, n_src) 的长方形**，而它应当**一格 0 都没有**
```

后者正是 day079 “交叉注意力不能加因果掩码”那句拒绝的**正面读法**：
不加掩码 ⇒ 每个解码位置都覆盖整段源序列（第 4 条性质）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.arch_variants.errors import (
    AssemblyError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.arch_variants.masks import (
    Mask,
    dependency_from_floats,
    mask_allowed_pairs,
    mask_is_causal,
    mask_summary,
)
from smart_research_agent.arch_variants.stacks import (
    DEFAULT_SOURCE_SEED,
    VariantParameters,
    mask_for_streams,
    variant_forward,
)
from smart_research_agent.arch_variants.types import (
    VARIANT_ENCODER_DECODER,
    VariantShape,
    max_absolute,
    validate_variant,
)
from smart_research_agent.math_foundations.types import (
    Matrix,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.encoder_decoder.types import ACTIVATION_RELU
from smart_research_agent.encoder_decoder.types import (
    _checked_activation as _checked_activation,
)

#: 扰动的倍率（把第 j 行整体乘 1.5：确定性、量级足够、**不会把数值送进溢出**）.
DEFAULT_PERTURBATION = 1.5

#: 判“依赖”的阈值。默认 **0.0**：被掩码挡掉的格子必须逐位为 0（第 1 节）.
DEFAULT_PROBE_TOLERANCE = 0.0

#: 两条流的名字.
STREAM_MAIN = "main"
STREAM_SOURCE = "source"
#: 编码器那一条流（``encoder_decoder`` 专用；单流变体里它等于 ``main``）.
STREAM_ENCODER = "encoder"

STREAM_KINDS: tuple[str, ...] = (STREAM_MAIN, STREAM_SOURCE, STREAM_ENCODER)


def perturb_row(
    matrix: Matrix,
    index: int,
    *,
    factor: float = DEFAULT_PERTURBATION,
) -> Matrix:
    """把第 ``index`` 行整体乘 ``factor``（**只改一行**：其余行逐位不变）.

    ``factor`` 必须不等于 1：等于 1 时扰动为 0，而那时“所有格子都不依赖”
    会让每一条判据都通过——一张什么都没量的表。本包直接拒绝这个值。
    """
    checked = validate_matrix(matrix, name="matrix")
    rows, _columns = matrix_shape(checked)
    if isinstance(index, bool) or not isinstance(index, int):
        raise ParameterError(f"行号必须是整数，收到 {index!r}。")
    if not 0 <= index < rows:
        raise ParameterError(f"行号 {index} 越界：可选 0..{rows - 1}。")
    if isinstance(factor, bool) or not isinstance(factor, (int, float)):
        raise ParameterError(f"倍率必须是数，收到 {factor!r}。")
    resolved = float(factor)
    if not math.isfinite(resolved):
        raise NumericError(f"倍率必须是有限数，收到 {factor!r}。")
    if resolved == 1.0:
        raise ParameterError(
            "倍率不能是 1.0：那样扰动为 0，而'所有格子都不依赖'会让判据全部通过——"
            "一张什么都没量的表看起来和一张全 0 的表一模一样。"
        )
    return tuple(
        tuple(value * resolved for value in row) if position == index else row
        for position, row in enumerate(checked)
    )


@dataclass(frozen=True)
class DependencyReport:
    """一张**实测**依赖表：``matrix[i][j]`` = 扰动位置 ``j`` 之后第 ``i`` 行的最大变化.

    ```text
    mask            这一次前向用的掩码（**期望**：哪些格子该是 0）
    tolerance       判“有依赖”的阈值（默认 0.0 = 逐位）
    blocked_gap     被掩码挡掉的格子里最大的那个读数（**它必须是 0.0**）
    allowed_floor   允许的格子里最小的那个读数（**它必须是正的**）
    ```

    两个读数一起给出来才有分辨力：只看 ``blocked_gap = 0`` 时，
    “掩码把一切都挡住了”与“掩码正确”无法区分——而 ``allowed_floor > 0`` 才说明
    这张表真的在量东西。
    """

    variant: str
    stream: str
    shape: VariantShape
    matrix: Matrix
    mask: Mask
    perturbation: float = DEFAULT_PERTURBATION
    tolerance: float = DEFAULT_PROBE_TOLERANCE
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", validate_variant(self.variant))
        if not isinstance(self.shape, VariantShape):
            raise ParameterError(
                f"shape 必须是 VariantShape，收到 {type(self.shape).__name__}："
                "报告里的形状要在后面被读（层数、源长度），一个非形状对象会让报错"
                "发生在离这里很远的地方。"
            )
        if self.stream not in STREAM_KINDS:
            raise ParameterError(
                f"未知的流名 {self.stream!r}：可选 {', '.join(STREAM_KINDS)}。"
            )
        checked = validate_matrix(self.matrix, name="matrix")
        object.__setattr__(self, "matrix", checked)
        rows, columns = matrix_shape(checked)
        if (len(self.mask), len(self.mask[0])) != (rows, columns):
            raise ShapeError(
                f"依赖表 {rows}×{columns} 与掩码 {len(self.mask)}×{len(self.mask[0])} "
                "形状不一致：两张表必须逐格对齐，否则‘哪个格子该是 0’根本无从比较。"
            )
        if not math.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ParameterError(f"tolerance 必须是有限的非负数，收到 {self.tolerance!r}。")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def rows(self) -> int:
        """行数（输出那一侧的长度）."""
        return len(self.matrix)

    @property
    def columns(self) -> int:
        """列数（被扰动那一侧的长度）."""
        return len(self.matrix[0])

    def _blocked(self) -> tuple[float, ...]:
        return tuple(
            self.matrix[row][column]
            for row in range(self.rows)
            for column in range(self.columns)
            if not self.mask[row][column]
        )

    def _allowed(self) -> tuple[float, ...]:
        return tuple(
            self.matrix[row][column]
            for row in range(self.rows)
            for column in range(self.columns)
            if self.mask[row][column]
        )

    @property
    def blocked_gap(self) -> float:
        """被挡掉的格子里最大的读数（**逐位判据的对象**）."""
        return max(self._blocked(), default=0.0)

    @property
    def allowed_floor(self) -> float:
        """允许的格子里最小的读数（它必须 > 阈值，否则这张表没在量东西）."""
        return min(self._allowed(), default=0.0)

    @property
    def exact(self) -> bool:
        """被挡掉的那些格子是不是**逐位**为 0（``blocked_gap == 0.0``）."""
        return self.blocked_gap == 0.0

    @property
    def measured_mask(self) -> Mask:
        """按阈值把实测读数变成掩码（默认阈值下就是“这一格有没有依赖”）."""
        return dependency_from_floats(self.matrix, tolerance=self.tolerance)

    @property
    def matches_mask(self) -> bool:
        """实测依赖表与掩码是否**逐格一致**（这是第 2、3、4 条性质的主判据）."""
        measured = self.measured_mask
        return all(
            measured[row][column] == self.mask[row][column]
            for row in range(self.rows)
            for column in range(self.columns)
        )

    def reach(self, row: int) -> int:
        """第 ``row`` 个输出依赖了多少个输入位置（因果掩码下应当是 ``row + 1``）."""
        if isinstance(row, bool) or not isinstance(row, int):
            raise ParameterError(f"行号必须是整数，收到 {row!r}。")
        if not 0 <= row < self.rows:
            raise ParameterError(f"行号 {row} 越界：可选 0..{self.rows - 1}。")
        return sum(1 for column in range(self.columns) if self.matrix[row][column] > self.tolerance)

    def reach_profile(self) -> tuple[int, ...]:
        """每一行的可达位置数（``1, 2, 3, 4`` 或 ``4, 4, 4, 4``）."""
        return tuple(self.reach(row) for row in range(self.rows))

    def leaks(self) -> tuple[tuple[int, int], ...]:
        """**实测**越界的格子：掩码挡住了、而扰动却让输出变了.

        它对应实现里的第 ③ 类失败（反向没穿掩码时前向是干净的，
        因此更常见的是“该挡的没挡住”——例如把因果掩码写成了全开）。
        """
        return tuple(
            (row, column)
            for row in range(self.rows)
            for column in range(self.columns)
            if (not self.mask[row][column]) and self.matrix[row][column] > self.tolerance
        )

    def summary_line(self) -> str:
        """一行说明（实验表里用的就是这一行）."""
        return (
            f"{self.variant:<16} | {self.stream:<6} | 允许 {mask_allowed_pairs(self.mask):>2}"
            f"/{self.rows * self.columns:<2} | 挡掉的格子最大读数 {self.blocked_gap:.6f} | "
            f"允许的格子最小读数 {self.allowed_floor:.6f} | 可达 {self.reach_profile()}"
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（**矩阵本身也带上**：它是证据）."""
        return {
            "variant": self.variant,
            "stream": self.stream,
            "shape": self.shape.summary_line(),
            "matrix": [list(row) for row in self.matrix],
            "allowed_pairs": mask_allowed_pairs(self.mask),
            "blocked_gap": self.blocked_gap,
            "allowed_floor": self.allowed_floor,
            "exact": self.exact,
            "matches_mask": self.matches_mask,
            "reach": list(self.reach_profile()),
        }


def _row_change_profile(left: Matrix, right: Matrix) -> Matrix:
    """每一行一个数：这一行里变化最大的那个分量（``(rows, 1)`` 形状）.

    之所以按“行内取最大”而不是逐元素，是因为依赖是**按行**定义的：
    第 i 行的输出是一整个向量，而“它有没有变”应当由它最敏感的那个分量回答。
    """
    if matrix_shape(left) != matrix_shape(right):
        raise ShapeError(
            f"两次前向的输出形状不同：{matrix_shape(left)} 与 {matrix_shape(right)}。"
        )
    return tuple(
        (
            max(abs(a - b) for a, b in zip(left_row, right_row, strict=True)),
        )
        for left_row, right_row in zip(left, right, strict=True)
    )


def dependency_matrix(
    params: VariantParameters,
    inputs: Matrix,
    *,
    source: Matrix | None = None,
    pads: Any = None,
    mask_kind: str | None = None,
    perturbation: float = DEFAULT_PERTURBATION,
    activation: str = ACTIVATION_RELU,
    source_seed: int = DEFAULT_SOURCE_SEED,
) -> DependencyReport:
    """逐列扰动输入，量出“第 i 个输出对第 j 个输入的依赖”.

    实现是 ``n`` 次完整前向（``n = tokens``）：本课的规模很小，
    而**“重新前向一次”比“把掩码的语义解释一遍”可信得多**——
    前者跑的就是真实的那条代码路径。

    ``encoder_decoder`` 变体可以顺带给出一段源序列（它只影响数值，不影响主流掩码）；
    单流变体给 ``source`` 会被拒绝——与本包其它入口同一条纪律（**不做静默忽略**）。
    """
    if not isinstance(params, VariantParameters):
        raise ParameterError(f"params 必须是 VariantParameters，收到 {type(params).__name__}。")
    if source is not None and params.variant != VARIANT_ENCODER_DECODER:
        raise AssemblyError(
            f"{params.variant} 变体只有一条流，不该收到 source：交叉注意力要的是**第二路**。"
        )
    checked_inputs = validate_matrix(inputs, name="inputs")
    if matrix_shape(checked_inputs) != (params.shape.tokens, params.shape.hidden):
        raise ShapeError(
            f"输入形状 {matrix_shape(checked_inputs)} 与 "
            f"({params.shape.tokens}, {params.shape.hidden}) 不一致。"
        )
    resolved_activation = _checked_activation(activation)
    base = variant_forward(
        params,
        checked_inputs,
        source=source,
        pads=pads,
        mask_kind=mask_kind,
        activation=resolved_activation,
        source_seed=source_seed,
    )
    columns = params.shape.tokens
    rows: list[tuple[float, ...]] = [[] for _ in range(params.shape.tokens)]
    for column in range(columns):
        perturbed = perturb_row(checked_inputs, column, factor=perturbation)
        moved = variant_forward(
            params,
            perturbed,
            source=source,
            pads=pads,
            mask_kind=mask_kind,
            activation=resolved_activation,
            source_seed=source_seed,
        )
        profile = _row_change_profile(base.output, moved.output)
        for index in range(params.shape.tokens):
            rows[index].append(profile[index][0])
    return DependencyReport(
        variant=params.variant,
        stream=STREAM_MAIN,
        shape=params.shape,
        matrix=tuple(tuple(row) for row in rows),
        mask=base.mask,
        perturbation=perturbation,
        notes=(
            f"逐列扰动 {columns} 次，每次重跑一遍完整前向",
            f"掩码（主流）：{mask_summary(base.mask)}",
        ),
    )


def cross_dependency(
    params: VariantParameters,
    inputs: Matrix,
    source: Matrix,
    *,
    mask_kind: str | None = None,
    perturbation: float = DEFAULT_PERTURBATION,
    activation: str = "relu",
) -> DependencyReport:
    """扰动**源序列**的每一行，量“第 i 个解码位置对第 j 个源位置的依赖”.

    这张表是 ``(tokens, sources)`` 的**长方形**，而它的期望掩码是**全开的长方形**：
    交叉注意力没有掩码，因此每一个解码位置都应当对**每一个**源位置有依赖。
    这是第 4 条性质（``cross_spans_all_sources``）的实测形式——
    它与第 2 条（自注意力因果）在**同一个解码器**里同时成立。
    """
    if not isinstance(params, VariantParameters):
        raise ParameterError(f"params 必须是 VariantParameters，收到 {type(params).__name__}。")
    if params.variant != VARIANT_ENCODER_DECODER:
        raise AssemblyError(
            f"{params.variant} 变体没有交叉注意力：它的 K/V 只来自自己那一条流——"
            "只有 encoder_decoder 才有“第二路”可以被扰动。"
        )
    checked_inputs = validate_matrix(inputs, name="inputs")
    checked_source = validate_matrix(source, name="source")
    if matrix_shape(checked_source) != (params.shape.source_length, params.shape.hidden):
        raise ShapeError(
            f"源序列形状 {matrix_shape(checked_source)} 与 "
            f"({params.shape.source_length}, {params.shape.hidden}) 不一致。"
        )
    base = variant_forward(
        params,
        checked_inputs,
        source=checked_source,
        mask_kind=mask_kind,
        activation=activation,
    )
    rows: list[tuple[float, ...]] = [[] for _ in range(params.shape.tokens)]
    for column in range(params.shape.source_length):
        perturbed = perturb_row(checked_source, column, factor=perturbation)
        moved = variant_forward(
            params,
            checked_inputs,
            source=perturbed,
            mask_kind=mask_kind,
            activation=activation,
        )
        profile = _row_change_profile(base.output, moved.output)
        for index in range(params.shape.tokens):
            rows[index].append(profile[index][0])
    expected: Mask = tuple(
        tuple(True for _ in range(params.shape.source_length))
        for _ in range(params.shape.tokens)
    )
    return DependencyReport(
        variant=params.variant,
        stream=STREAM_SOURCE,
        shape=params.shape,
        matrix=tuple(tuple(row) for row in rows),
        mask=expected,
        perturbation=perturbation,
        notes=(
            f"扰动源序列 {params.shape.source_length} 行，量的是**交叉注意力**那一条路",
            "期望掩码是**全开的长方形**：交叉注意力没有掩码（day079 的组装拒绝）",
        ),
    )


def _mask_for_streams_of(
    params: VariantParameters, mask_kind: str | None
) -> tuple[Mask, Mask | None]:
    """取两条流的掩码（**转发 ``stacks.mask_for_streams``**：口径只有一处）."""
    return mask_for_streams(params.variant, params.shape, mask_kind=mask_kind)


def encoder_dependency(
    params: VariantParameters,
    inputs: Matrix,
    *,
    source: Matrix | None = None,
    mask_kind: str | None = None,
    perturbation: float = DEFAULT_PERTURBATION,
    activation: str = ACTIVATION_RELU,
    source_seed: int = DEFAULT_SOURCE_SEED,
) -> DependencyReport:
    """扰动**编码器那一条流**的每一行，量“编码器第 i 个输出对第 j 个输入的依赖”.

    它与 :func:`dependency_matrix` 的差别只有一处：量的是**编码器的输出**
    （``encoder_decoder`` 里那一份中间结果），而不是整条变体的输出。

    为什么需要它：``encoder_decoder`` 的**最终输出**来自解码器，那个输出对源位置的
    依赖走的是**交叉注意力**（第 4 条性质量的是它）；而“编码器那一侧是不是双向的”
    是另一件事——它只能在这一张表上被量出来（第 3 条性质）。
    """
    if not isinstance(params, VariantParameters):
        raise ParameterError(f"params 必须是 VariantParameters，收到 {type(params).__name__}。")
    checked_inputs = validate_matrix(inputs, name="inputs")
    resolved_activation = _checked_activation(activation)
    if params.variant == VARIANT_ENCODER_DECODER:
        if source is None:
            raise AssemblyError(
                "encoder_dependency 量的是编码器那一侧的账，因此 encoder_decoder 变体"
                "必须显式给出源序列——用默认样本量出来的表会与调用方的源序列无关。"
            )
        stream_input = validate_matrix(source, name="source")
        expected_length = params.shape.source_length
    else:
        if source is not None:
            raise AssemblyError(
                f"{params.variant} 变体没有第二路：它只有一条流，因此 source 不该给。"
            )
        stream_input = checked_inputs
        expected_length = params.shape.tokens
    if matrix_shape(stream_input) != (expected_length, params.shape.hidden):
        raise ShapeError(
            f"编码器那一侧的输入形状 {matrix_shape(stream_input)} 与 "
            f"({expected_length}, {params.shape.hidden}) 不一致。"
        )

    def _stream_output(matrix: Matrix) -> Matrix:
        if params.variant == VARIANT_ENCODER_DECODER:
            forward = variant_forward(
                params,
                checked_inputs,
                source=matrix,
                mask_kind=mask_kind,
                activation=resolved_activation,
                source_seed=source_seed,
            )
        else:
            forward = variant_forward(
                params,
                matrix,
                mask_kind=mask_kind,
                activation=resolved_activation,
                source_seed=source_seed,
            )
        return forward.output if forward.encoder_output is None else forward.encoder_output

    base = _stream_output(stream_input)
    rows: list[tuple[float, ...]] = [[] for _ in range(expected_length)]
    for column in range(expected_length):
        moved = _stream_output(perturb_row(stream_input, column, factor=perturbation))
        profile = _row_change_profile(base, moved)
        for index in range(expected_length):
            rows[index].append(profile[index][0])
    main_mask, source_mask = _mask_for_streams_of(params, mask_kind)
    effective = main_mask if source_mask is None else source_mask
    return DependencyReport(
        variant=params.variant,
        stream=STREAM_ENCODER,
        shape=params.shape,
        matrix=tuple(tuple(row) for row in rows),
        mask=effective,
        perturbation=perturbation,
        notes=(
            f"扰动编码器那一侧 {expected_length} 行，量的是**编码器的输出**",
            f"掩码（编码器流）：{mask_summary(effective)}",
        ),
    )


def row_change_profile(left: Matrix, right: Matrix) -> Matrix:
    """两次前向之间“每一行变了多少”（供 ``study`` 的稳定性实验使用）."""
    return _row_change_profile(left, right)


def changed_rows(left: Matrix, right: Matrix, *, tolerance: float = 0.0) -> tuple[int, ...]:
    """变了的那几行的行号（``tolerance = 0.0`` 时它就是“逐位变了吗”）."""
    profile = _row_change_profile(left, right)
    return tuple(
        index for index, row in enumerate(profile) if row[0] > tolerance
    )


def contract_gap(report: DependencyReport) -> float:
    """实测表与掩码的不一致程度：**``0.0`` 表示逐格一致**.

    ```text
    越界格子（掩码挡住了、而扰动让输出动了）  ⇒ 取最大读数（正数）
    该动没动的格子（掩码允许、而扰动没让它动） ⇒ 每出现一个记 1.0
    ```

    两个错一起收进一个标量：``0.0`` 只有在**两边都为空**时才成立，
    而单个标量做断言比两张表逐格比更不容易写错。
    """
    leaks = report.leaks()
    worst = max((report.matrix[row][column] for row, column in leaks), default=0.0)
    dead = sum(
        1
        for row in range(report.rows)
        for column in range(report.columns)
        if report.mask[row][column] and report.matrix[row][column] <= report.tolerance
    )
    return worst + float(dead)


def report_matches(report: DependencyReport) -> bool:
    """``matches_mask`` 的自由函数版本（供报告模板调用）."""
    return report.matches_mask


def dependency_of(report: DependencyReport, row: int, column: int) -> float:
    """取一个格子（越界当场拒绝：这张表是证据，索引错了就不该继续）."""
    if isinstance(row, bool) or not isinstance(row, int) or isinstance(column, bool) or not isinstance(column, int):
        raise ParameterError("行号与列号必须是整数。")
    if not 0 <= row < report.rows or not 0 <= column < report.columns:
        raise ParameterError(
            f"({row}, {column}) 越界：可选 0..{report.rows - 1} × 0..{report.columns - 1}。"
        )
    return report.matrix[row][column]


def is_rectangular(report: DependencyReport) -> bool:
    """这张表是不是长方形的（源流那一张是，主流那一张不是）."""
    return report.rows != report.columns


def causal_verdict(report: DependencyReport) -> bool:
    """把实测表按阈值变成掩码之后，它是不是因果的（**判据不读调用参数**）."""
    return mask_is_causal(report.measured_mask)


def largest_change(report: DependencyReport) -> float:
    """整张表里最大的读数（一行的量级说明）."""
    return max_absolute(report.matrix)


__all__ = [
    "DEFAULT_PERTURBATION",
    "DEFAULT_PROBE_TOLERANCE",
    "STREAM_KINDS",
    "STREAM_MAIN",
    "STREAM_SOURCE",
    "DependencyReport",
    "causal_verdict",
    "changed_rows",
    "contract_gap",
    "cross_dependency",
    "dependency_matrix",
    "dependency_of",
    "encoder_dependency",
    "is_rectangular",
    "largest_change",
    "perturb_row",
    "report_matches",
    "row_change_profile",
]
