"""``study.py``：四张表——**掩码的账、家底、前缀稳定性、用错的代价**（day082）.

四组实验共用同一个问题：**换一个变体（或换一张掩码），能看到的东西会变成什么样？**

```text
① 掩码的账       每个变体两条流各自的掩码、允许的位置对、实测的越界/缺失
② 家底           块数、子层数、参数量（逐个数 vs 公式算）
③ 前缀稳定性     扰动最后一个 token，看**哪几行**的输出变了
④ 用错的代价     把因果掩码换成全开（反之亦然），量出越界与缺失
```

四组同一条纪律（与 day079~081 逐字相同）：**一次只改一个旋钮**，
其余字段由 :meth:`VariantShape` 与同一颗种子固定下来——
于是“换变体”与“换随机性”不会混在一起。

第 ③ 组是本课最直观的一张表：扰动**最后一个** token 之后，

```text
encoder_only     四行全变（**双向**：每个位置都能看到它）
decoder_only     只有最后一行变（**因果**：前面三个位置看不到未来，逐位不变）
encoder_decoder  只有最后一行变（解码器那一流因果；编码器那一流另算，见第 ① 组第 4 行）
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.arch_variants.errors import NumericError, ParameterError
from smart_research_agent.arch_variants.masks import (
    MASK_CAUSAL,
    MASK_FULL,
    Mask,
    mask_allowed_pairs,
    mask_of,
    mask_summary,
)
from smart_research_agent.arch_variants.probe import (
    DEFAULT_PERTURBATION,
    STREAM_ENCODER,
    STREAM_MAIN,
    STREAM_SOURCE,
    DependencyReport,
    cross_dependency,
    dependency_matrix,
    encoder_dependency,
    perturb_row,
    row_change_profile,
)
from smart_research_agent.arch_variants.stacks import (
    VariantParameters,
    census_of,
    make_variant_parameters,
    make_variant_shape,
    sample_matrix,
    variant_forward,
)
from smart_research_agent.arch_variants.types import (
    DEFAULT_INIT_SCALE,
    DEFAULT_LAYERS,
    DEFAULT_SOURCES,
    VARIANTS,
    VARIANT_DECODER_ONLY,
    VARIANT_DESCRIPTIONS,
    VARIANT_ENCODER_DECODER,
    VARIANT_ENCODER_ONLY,
    VARIANT_EXAMPLES,
    VARIANT_MASKS,
    VARIANT_OBJECTIVES,
    VariantCensus,
    VariantShape,
)
from smart_research_agent.encoder_decoder.types import ACTIVATION_GELU
from smart_research_agent.math_foundations.types import Matrix

#: 输入样本的种子（与参数、源序列分开：三者各自可复现）.
DEFAULT_INPUT_SEED = 21
#: 源序列样本的种子.
DEFAULT_SOURCE_SEED_SAMPLE = 17


def _sample_inputs(shape: VariantShape, seed: int = DEFAULT_INPUT_SEED) -> Matrix:
    return sample_matrix(shape.tokens, shape.hidden, seed=seed, scale=DEFAULT_INIT_SCALE)


def _sample_source(
    shape: VariantShape, seed: int = DEFAULT_SOURCE_SEED_SAMPLE
) -> Matrix:
    return sample_matrix(shape.source_length, shape.hidden, seed=seed, scale=DEFAULT_INIT_SCALE)


# ---------------------------------------------------------------------- ① 掩码的账


@dataclass(frozen=True)
class LeakRow:
    """一行“掩码的账”：允许多少、实测越界多少、实测缺失多少.

    ``expect_mismatch=True`` 的那两行是**反证**：它们本来就不该一致
    （把掩码换错之后，同一张表必须亮红）。把它写成字段而不是写进标签里，
    是因为“这一行该不该一致”是一个**可被断言**的性质——
    否则 ``ok`` 会把它一起算进去，于是反证失败时整张表看起来“正常”。
    """

    label: str
    variant: str
    stream: str
    mask: Mask
    intended: Mask
    blocked_gap: float
    allowed_floor: float
    reach: tuple[int, ...]
    leaks: int
    missing: int
    worst_leak: float
    exact: bool
    expect_mismatch: bool = False

    @property
    def allowed_pairs(self) -> int:
        """这一次前向实际用的掩码允许的位置对."""
        return mask_allowed_pairs(self.mask)

    @property
    def total_pairs(self) -> int:
        """总位置对（行数 × 列数）."""
        return len(self.mask) * len(self.mask[0])

    @property
    def consistent(self) -> bool:
        """实测与期望是否逐格一致."""
        return self.leaks == 0 and self.missing == 0

    @property
    def verdict(self) -> str:
        """这一行的结论（反证行期待“不一致”）."""
        if self.expect_mismatch:
            return "**如期不一致**" if not self.consistent else "**反证失效**"
        return "一致" if self.consistent else "**不一致**"

    def summary_line(self) -> str:
        """一行说明（实验表里用的就是这一行）."""
        return (
            f"{self.label:<34} | 允许 {self.allowed_pairs:>2}/{self.total_pairs:<2} | "
            f"挡掉的格子最大读数 {self.blocked_gap:.6e} | 可达 {self.reach} | "
            f"越界 {self.leaks} · 缺失 {self.missing} | {self.verdict}"
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "label": self.label,
            "variant": self.variant,
            "stream": self.stream,
            "allowed_pairs": self.allowed_pairs,
            "total_pairs": self.total_pairs,
            "blocked_gap": self.blocked_gap,
            "allowed_floor": self.allowed_floor,
            "reach": list(self.reach),
            "leaks": self.leaks,
            "missing": self.missing,
            "worst_leak": self.worst_leak,
            "exact": self.exact,
            "expect_mismatch": self.expect_mismatch,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class LeakStudy:
    """第 ① 组：每个变体两条流各一行，外加两行反证."""

    shape: VariantShape
    rows: tuple[LeakRow, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.rows:
            raise ParameterError("掩码的账不能为空。")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def ok(self) -> bool:
        """**正常行**全部一致，且**反证行**全部如期不一致."""
        return all(row.consistent for row in self.normal_rows) and all(
            not row.consistent for row in self.counterexample_rows
        )

    @property
    def normal_rows(self) -> tuple[LeakRow, ...]:
        """该一致的那几行（五个变体 / 流的正常配置）."""
        return tuple(row for row in self.rows if not row.expect_mismatch)

    @property
    def counterexample_rows(self) -> tuple[LeakRow, ...]:
        """反证行（把它们写出来，是为了让“判据真的有分辨力”可被断言）."""
        return tuple(row for row in self.rows if row.expect_mismatch)

    @property
    def exact_rows(self) -> int:
        """“挡掉的格子逐位为 0”的行数（**含反证行**：换错掩码也不改变这一点）."""
        return sum(1 for row in self.rows if row.exact)

    def row_of(self, label: str) -> LeakRow:
        """按标签取一行（标签不认识时当场拒绝）."""
        for row in self.rows:
            if row.label == label:
                return row
        raise ParameterError(
            f"表里没有 {label!r}：可选 {', '.join(row.label for row in self.rows)}。"
        )

    def table_lines(self) -> tuple[str, ...]:
        """整张表（逐行）."""
        return tuple(row.summary_line() for row in self.rows)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "shape": self.shape.summary_line(),
            "ok": self.ok,
            "exact_rows": self.exact_rows,
            "rows": [row.to_dict() for row in self.rows],
        }


def _counts_against(report: DependencyReport, intended: Mask) -> tuple[int, int, float]:
    """实测表与**期望掩码**比：``(越界数, 缺失数, 最坏的越界读数)``.

    这是本课“用错掩码的代价”那句话的算法：越界 = 期望挡住而实测有依赖；
    缺失 = 期望允许而实测没有依赖（双向能力被砍掉的那种失败）。
    """
    rows, columns = report.rows, report.columns
    if (len(intended), len(intended[0])) != (rows, columns):
        raise NumericError(
            f"期望掩码 {len(intended)}×{len(intended[0])} 与实测表 {rows}×{columns} "
            "形状不一致：两张表必须逐格对齐。"
        )
    leaks = 0
    missing = 0
    worst = 0.0
    for row in range(rows):
        for column in range(columns):
            measured = report.matrix[row][column] > report.tolerance
            if measured and not intended[row][column]:
                leaks += 1
                worst = max(worst, report.matrix[row][column])
            elif not measured and intended[row][column]:
                missing += 1
    return leaks, missing, worst


def leak_study(
    shape: VariantShape | None = None,
    *,
    perturbation: float = DEFAULT_PERTURBATION,
    activation: str = ACTIVATION_GELU,
    input_seed: int = DEFAULT_INPUT_SEED,
    source_seed: int = DEFAULT_SOURCE_SEED_SAMPLE,
) -> LeakStudy:
    """跑完第 ① 组：五个“正常行” + 两个反证行."""
    resolved_shape = shape if shape is not None else make_variant_shape(
        layers=DEFAULT_LAYERS, sources=DEFAULT_SOURCES
    )
    inputs = _sample_inputs(resolved_shape, seed=input_seed)
    source = _sample_source(resolved_shape, seed=source_seed)
    rows: list[LeakRow] = []

    def add(
        label: str,
        report: DependencyReport,
        intended: Mask,
        *,
        stream: str | None = None,
        expect_mismatch: bool = False,
    ) -> None:
        leaks, missing, worst = _counts_against(report, intended)
        rows.append(
            LeakRow(
                label=label,
                variant=report.variant,
                stream=stream if stream is not None else report.stream,
                mask=report.mask,
                intended=intended,
                blocked_gap=report.blocked_gap,
                allowed_floor=report.allowed_floor,
                reach=report.reach_profile(),
                leaks=leaks,
                missing=missing,
                worst_leak=worst,
                exact=report.exact,
                expect_mismatch=expect_mismatch,
            )
        )

    # ① encoder_only：唯一那条流是全开的
    encoder_only = make_variant_parameters(resolved_shape, VARIANT_ENCODER_ONLY)
    encoder_report = dependency_matrix(
        encoder_only, inputs, perturbation=perturbation, activation=activation
    )
    add("encoder_only（全开）", encoder_report, mask_of(MASK_FULL, resolved_shape.tokens))

    # ② decoder_only：唯一那条流是因果的
    decoder_only = make_variant_parameters(resolved_shape, VARIANT_DECODER_ONLY)
    decoder_report = dependency_matrix(
        decoder_only, inputs, perturbation=perturbation, activation=activation
    )
    add("decoder_only（因果）", decoder_report, mask_of(MASK_CAUSAL, resolved_shape.tokens))

    # ③ encoder_decoder：主流（解码器）因果
    both = make_variant_parameters(resolved_shape, VARIANT_ENCODER_DECODER)
    main_report = dependency_matrix(
        both,
        inputs,
        source=source,
        perturbation=perturbation,
        activation=activation,
    )
    add("encoder_decoder（主流=解码器）", main_report, mask_of(MASK_CAUSAL, resolved_shape.tokens))

    # ④ encoder_decoder：编码器那一流全开
    encoder_stream = encoder_dependency(
        both,
        inputs,
        source=source,
        perturbation=perturbation,
        activation=activation,
    )
    add(
        "encoder_decoder（编码器流）",
        encoder_stream,
        mask_of(MASK_FULL, resolved_shape.source_length),
        stream=STREAM_ENCODER,
    )

    # ⑤ encoder_decoder：交叉那一路（长方形、无掩码）
    cross_report = cross_dependency(
        both, inputs, source, perturbation=perturbation, activation=activation
    )
    add(
        "encoder_decoder（交叉那一路）",
        cross_report,
        tuple(
            tuple(True for _ in range(resolved_shape.source_length))
            for _ in range(resolved_shape.tokens)
        ),
        stream=STREAM_SOURCE,
    )

    # ⑥ 反证一：把因果掩码换成全开 ⇒ **越界 6 格**（未来被看到了）
    wrong_full = dependency_matrix(
        decoder_only,
        inputs,
        mask_kind=MASK_FULL,
        perturbation=perturbation,
        activation=activation,
    )
    add(
        "反证：decoder_only 用全开掩码",
        wrong_full,
        mask_of(MASK_CAUSAL, resolved_shape.tokens),
        expect_mismatch=True,
    )

    # ⑦ 反证二：把全开掩码换成因果 ⇒ **缺失 6 格**（双向能力被砍掉）
    wrong_causal = dependency_matrix(
        encoder_only,
        inputs,
        mask_kind=MASK_CAUSAL,
        perturbation=perturbation,
        activation=activation,
    )
    add(
        "反证：encoder_only 用因果掩码",
        wrong_causal,
        mask_of(MASK_FULL, resolved_shape.tokens),
        expect_mismatch=True,
    )

    notes = (
        f"样本：{resolved_shape.summary_line()} | 扰动倍率 {perturbation} | 激活 {activation}",
        "越界 = 期望挡住而实测有依赖；缺失 = 期望允许而实测没有依赖（**两个方向都要量**）",
        "前五行是‘实测与声明一致’，后两行是反证：**换掉掩码之后同一条判据立刻亮红**",
    )
    return LeakStudy(shape=resolved_shape, rows=tuple(rows), notes=notes)


# ---------------------------------------------------------------------- ② 家底


@dataclass(frozen=True)
class CensusRow:
    """一行家底：变体、块数、子层数、参数量（逐个数 vs 公式）."""

    census: VariantCensus

    @property
    def variant(self) -> str:
        """变体名."""
        return self.census.variant

    def summary_line(self) -> str:
        """一行说明."""
        return self.census.summary_line()

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return self.census.to_dict()


@dataclass(frozen=True)
class CensusStudy:
    """第 ② 组：三个变体的家底（**参数量必须两条路径一致**）."""

    shape: VariantShape
    rows: tuple[CensusRow, ...]
    examples: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """所有行的“逐个数 == 公式算”."""
        return all(row.census.matches_analytic for row in self.rows)

    def row_of(self, variant: str) -> CensusRow:
        """按变体名取一行."""
        for row in self.rows:
            if row.variant == variant:
                return row
        raise ParameterError(
            f"表里没有变体 {variant!r}：可选 {', '.join(row.variant for row in self.rows)}。"
        )

    def table_lines(self) -> tuple[str, ...]:
        """整张表（逐行）."""
        return tuple(row.summary_line() for row in self.rows)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "shape": self.shape.summary_line(),
            "ok": self.ok,
            "rows": [row.to_dict() for row in self.rows],
        }


def census_study(
    shape: VariantShape | None = None,
    *,
    seed: int = 7,
    scale: float = DEFAULT_INIT_SCALE,
) -> CensusStudy:
    """跑完第 ② 组：三个变体各数一遍家底."""
    resolved_shape = shape if shape is not None else make_variant_shape(
        layers=DEFAULT_LAYERS, sources=DEFAULT_SOURCES
    )
    rows: list[CensusRow] = []
    for variant in VARIANTS:
        params = make_variant_parameters(resolved_shape, variant, seed=seed, scale=scale)
        rows.append(CensusRow(census=census_of(params)))
    return CensusStudy(
        shape=resolved_shape,
        rows=tuple(rows),
        examples=tuple(VARIANT_EXAMPLES[variant] for variant in VARIANTS),
    )


def objectives_table() -> tuple[tuple[str, str, str, str], ...]:
    """三行“变体 / 例子 / 掩码 / 训练目标”（教程第 1 章那张表的数据源）."""
    return tuple(
        (
            variant,
            VARIANT_EXAMPLES[variant],
            " + ".join(VARIANT_MASKS[variant]),
            VARIANT_OBJECTIVES[variant],
        )
        for variant in VARIANTS
    )


def variant_descriptions() -> tuple[tuple[str, str], ...]:
    """三个变体各自的一句话说明."""
    return tuple((variant, VARIANT_DESCRIPTIONS[variant]) for variant in VARIANTS)


# ---------------------------------------------------------------------- ③ 前缀稳定性


@dataclass(frozen=True)
class PrefixRow:
    """扰动**最后一个** token 之后，哪几行的输出变了."""

    variant: str
    position: int
    changed_rows: tuple[int, ...]
    row_change: tuple[float, ...]

    @property
    def first_row_change(self) -> float:
        """输出第 0 行变了多少（双向时明显大于 0，因果时恰好 0.0）."""
        return self.row_change[0] if self.row_change else 0.0

    @property
    def changed_count(self) -> int:
        """变了的行数."""
        return len(self.changed_rows)

    def summary_line(self) -> str:
        """一行说明."""
        return (
            f"{self.variant:<16} | 扰动位置 {self.position} | 变了的行 {self.changed_rows}"
            f"（{self.changed_count} 行）| 第 0 行的变化 {self.first_row_change:.6e}"
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "variant": self.variant,
            "position": self.position,
            "changed_rows": list(self.changed_rows),
            "changed_count": self.changed_count,
            "first_row_change": self.first_row_change,
        }


@dataclass(frozen=True)
class PrefixStudy:
    """第 ③ 组：前缀稳定性（**这张表用“哪几行动了”回答“因果吗”**）."""

    shape: VariantShape
    rows: tuple[PrefixRow, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.rows:
            raise ParameterError("前缀稳定性表不能为空。")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    def row_of(self, variant: str) -> PrefixRow:
        """按变体名取一行."""
        for row in self.rows:
            if row.variant == variant:
                return row
        raise ParameterError(
            f"表里没有变体 {variant!r}：可选 {', '.join(row.variant for row in self.rows)}。"
        )

    def table_lines(self) -> tuple[str, ...]:
        """整张表（逐行）."""
        return tuple(row.summary_line() for row in self.rows)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {"shape": self.shape.summary_line(), "rows": [row.to_dict() for row in self.rows]}


def prefix_study(
    shape: VariantShape | None = None,
    *,
    position: int | None = None,
    perturbation: float = DEFAULT_PERTURBATION,
    activation: str = ACTIVATION_GELU,
    input_seed: int = DEFAULT_INPUT_SEED,
    source_seed: int = DEFAULT_SOURCE_SEED_SAMPLE,
    tolerance: float = 0.0,
) -> PrefixStudy:
    """跑完第 ③ 组：扰动一个位置，看**哪几行**变了（默认扰动最后一个）."""
    resolved_shape = shape if shape is not None else make_variant_shape(
        layers=DEFAULT_LAYERS, sources=DEFAULT_SOURCES
    )
    inputs = _sample_inputs(resolved_shape, seed=input_seed)
    source = _sample_source(resolved_shape, seed=source_seed)
    resolved_position = resolved_shape.tokens - 1 if position is None else position
    if isinstance(resolved_position, bool) or not isinstance(resolved_position, int):
        raise ParameterError(f"位置必须是整数，收到 {resolved_position!r}。")
    if not 0 <= resolved_position < resolved_shape.tokens:
        raise ParameterError(
            f"位置 {resolved_position} 越界：可选 0..{resolved_shape.tokens - 1}。"
        )
    perturbed = perturb_row(inputs, resolved_position, factor=perturbation)
    rows: list[PrefixRow] = []
    for variant in VARIANTS:
        params = make_variant_parameters(resolved_shape, variant)
        kwargs = {"source": source} if variant == VARIANT_ENCODER_DECODER else {}
        before = variant_forward(params, inputs, activation=activation, **kwargs)
        after = variant_forward(params, perturbed, activation=activation, **kwargs)
        profile = tuple(
            value[0] for value in row_change_profile(before.output, after.output)
        )
        rows.append(
            PrefixRow(
                variant=variant,
                position=resolved_position,
                changed_rows=tuple(
                    index for index, value in enumerate(profile) if value > tolerance
                ),
                row_change=profile,
            )
        )
    notes = (
        f"扰动位置 {resolved_position}（最后一个 token），倍率 {perturbation}",
        "判据是 `> 0.0`（逐位）：因果变体上前面那些行的读数**恰好是 0.0**",
        "encoder_decoder 这一行量的是**解码器那一条流**；编码器那一流的读数见第 ① 组",
    )
    return PrefixStudy(shape=resolved_shape, rows=tuple(rows), notes=notes)


# ---------------------------------------------------------------------- ④ 汇总


def study_summary(shape: VariantShape | None = None) -> str:
    """四张表的标题行（脚本用它分段打印）."""
    resolved_shape = shape if shape is not None else make_variant_shape(
        layers=DEFAULT_LAYERS, sources=DEFAULT_SOURCES
    )
    return f"arch_variants 实验 | {resolved_shape.summary_line()}"


def mask_catalogue(shape: VariantShape | None = None) -> tuple[tuple[str, str], ...]:
    """三张掩码各自的摘要（教程第 2 章那张小表）."""
    resolved_shape = shape if shape is not None else make_variant_shape(
        layers=DEFAULT_LAYERS, sources=DEFAULT_SOURCES
    )
    lines: list[tuple[str, str]] = []
    for kind in (MASK_FULL, MASK_CAUSAL):
        mask = mask_of(kind, resolved_shape.tokens)
        lines.append((kind, mask_summary(mask)))
    return tuple(lines)


def entropy_ceiling_table(shape: VariantShape | None = None) -> tuple[tuple[str, float], ...]:
    """两张掩码各自的注意力熵上限（``ln n`` 与 ``(1/n)Σln(i+1)``）.

    它是 day083（可解释性与注意力可视化）的伏笔：**熵要跟天花板比**——
    因果掩码把第 0 行的天花板压到 ``ln 1 = 0``，因此“注意力很尖”这件事
    在因果变体上有一部分是**掩码造成的**，而不是模型学出来的。
    """
    from smart_research_agent.arch_variants.masks import mask_entropy_ceiling

    resolved_shape = shape if shape is not None else make_variant_shape(
        layers=DEFAULT_LAYERS, sources=DEFAULT_SOURCES
    )
    return tuple(
        (kind, mask_entropy_ceiling(mask_of(kind, resolved_shape.tokens)))
        for kind in (MASK_FULL, MASK_CAUSAL)
    )


def variant_parameter_total(
    variant: str,
    shape: VariantShape | None = None,
) -> int:
    """单个变体的参数量（教程里引用的那个数）."""
    resolved_shape = shape if shape is not None else make_variant_shape(
        layers=DEFAULT_LAYERS, sources=DEFAULT_SOURCES
    )
    params = make_variant_parameters(resolved_shape, variant)
    return params.parameter_count()


def mask_pair_ratio(shape: VariantShape | None = None) -> float:
    """因果掩码允许的位置对占全开的比例（``n(n+1)/2 / n²``）.

    ``n = 4`` 时它是 ``10/16 = 0.625``——而它随 ``n`` 增长趋近 ``1/2``：
    **序列越长，因果掩码砍掉的比例越接近一半**。这个数解释了第 9 章那句
    “BERT 与 GPT 的差别在长序列上更贵”。
    """
    resolved_shape = shape if shape is not None else make_variant_shape(
        layers=DEFAULT_LAYERS, sources=DEFAULT_SOURCES
    )
    full = mask_allowed_pairs(mask_of(MASK_FULL, resolved_shape.tokens))
    causal = mask_allowed_pairs(mask_of(MASK_CAUSAL, resolved_shape.tokens))
    if full == 0:  # pragma: no cover - tokens >= 1 时不可能
        raise NumericError("全开掩码允许的位置对不能是 0。")
    return causal / full


def keystone_check(params: VariantParameters, shape: VariantShape | None = None) -> bool:
    """一条总闸：这个变体的“逐个数 == 公式算”是否成立（实验脚本末尾印它）."""
    census = census_of(params)
    if census.layers != params.shape.layers:
        raise NumericError("census 的层数与形状声明不一致。")
    return census.matches_analytic


__all__ = [
    "DEFAULT_INPUT_SEED",
    "DEFAULT_SOURCE_SEED_SAMPLE",
    "CensusRow",
    "CensusStudy",
    "LeakRow",
    "LeakStudy",
    "PrefixRow",
    "PrefixStudy",
    "census_study",
    "entropy_ceiling_table",
    "keystone_check",
    "leak_study",
    "mask_catalogue",
    "mask_pair_ratio",
    "objectives_table",
    "prefix_study",
    "study_summary",
    "variant_descriptions",
    "variant_parameter_total",
]
