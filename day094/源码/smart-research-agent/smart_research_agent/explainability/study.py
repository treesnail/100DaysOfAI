"""``study.py``：五张表——**逐层熵、头间冗余、偏移质量、滚动的集中度、两种掩码的天花板**（day083）.

五组实验共用一个问题：**把权重画出来之后，能读出什么？**

```text
① 逐层熵        每层"平均头"的熵 / 天花板 / 归一化熵——**熵要跟天花板比**
② 头间冗余      同一层的几个头之间逐格余弦——"多头退化成一头"的读数
③ 偏移质量      质量落在离对角线 0 / 1 / 2 格上的比例——"它看的是相邻位置吗"
④ 滚动的集中度   滚动之后归一化熵变了多少（**可能是负的**：如实给三个数）
⑤ 天花板对照     全开 vs 因果两种掩码的熵天花板——解释"GPT 比 BERT 尖"里有多少是掩码造成的
```

四组同一条纪律（与前几天的实验逐字相同）：**一次只改一个旋钮**——
样本、种子、层数、头数都由函数参数固定，于是"换变体/换层"与"换随机性"不会混在一起。

第 ⑤ 组是本课最值钱的一张表，因为它把"注意力很尖"这件事拆成两半：

```text
模型学出来的部分    归一化熵（< 1 表示比"均匀"更尖）
掩码造成的部分      天花板本身（因果掩码下第 0 行是 ln 1 = 0）
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from smart_research_agent.arch_variants.masks import (
    MASK_CAUSAL,
    MASK_FULL,
    MASK_KINDS,
    mask_entropy_ceiling,
    mask_of,
)
from smart_research_agent.arch_variants.stacks import (
    VariantParameters,
    make_variant_parameters,
    make_variant_shape,
    sample_matrix,
)
from smart_research_agent.arch_variants.types import (
    DEFAULT_HIDDEN,
    DEFAULT_LAYERS,
    DEFAULT_SOURCES,
    DEFAULT_TOKENS,
    VARIANT_DECODER_ONLY,
    VARIANT_ENCODER_DECODER,
    VARIANTS,
    VariantShape,
    validate_variant,
)
from smart_research_agent.explainability.analyze import (
    focus_verdict,
    head_redundancy,
    layer_profiles,
    offset_mass,
    profile_of,
    summarise,
)
from smart_research_agent.explainability.errors import (
    AssemblyError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.explainability.extract import head_records
from smart_research_agent.explainability.rollout import (
    aggregate_by_layer,
    rollout_focus,
    rollout_record,
)
from smart_research_agent.explainability.types import (
    DEFAULT_ALPHA,
    DEFAULT_HEADS,
    AttentionRecord,
    HeadProfile,
    mean_of,
)
from smart_research_agent.math_foundations.types import Matrix

#: 三颗互不相干的种子（参数 / 输入 / 源序列）——与 day082 的样本同值，便于跨天对照.
PARAMETER_SEED = 7
INPUT_SEED = 21
SOURCE_SEED = 13

#: 偏移质量默认量的三条斜线（0 = 自看自，1 = 看前一个位置）.
DEFAULT_OFFSETS: tuple[int, ...] = (0, 1, 2)


def default_shape(
    *,
    tokens: int = DEFAULT_TOKENS,
    hidden: int = DEFAULT_HIDDEN,
    layers: int = DEFAULT_LAYERS,
    sources: int = DEFAULT_SOURCES,
) -> VariantShape:
    """默认形状（与 day082 的样本同规模：4 个 token、d = 6、3 层）."""
    return make_variant_shape(tokens=tokens, hidden=hidden, layers=layers, sources=sources)


def study_params(
    variant: str = VARIANT_DECODER_ONLY,
    shape: VariantShape | None = None,
    *,
    seed: int = PARAMETER_SEED,
) -> VariantParameters:
    """某一组实验用的参数（**每一组都从这一处取**：换变体与换随机性不混）."""
    resolved = shape if shape is not None else default_shape()
    return make_variant_parameters(resolved, validate_variant(variant), seed=seed)


def study_inputs(
    shape: VariantShape | None = None, *, seed: int = INPUT_SEED
) -> Matrix:
    """样本输入（同一颗种子：跨组可比）."""
    resolved = shape if shape is not None else default_shape()
    return sample_matrix(resolved.tokens, resolved.hidden, seed=seed)


def all_layer_records(
    params: VariantParameters,
    inputs: Matrix,
    *,
    heads: int = DEFAULT_HEADS,
    source: Matrix | None = None,
) -> tuple[AttentionRecord, ...]:
    """**逐层**跑一次多头读法（标签是 ``"层 i · 头 h"``，因此可以直接进滚动）.

    与 :func:`extract.head_records` 的分工：那个读**一层**，这一个把每层都读一遍——
    而"层 i"这套命名是滚动与逐层对照要的（``aggregate_by_layer`` 按它排序）。
    """
    if not isinstance(params, VariantParameters):
        raise ParameterError(f"params 必须是 VariantParameters，收到 {type(params).__name__}。")
    if params.variant == VARIANT_ENCODER_DECODER and source is None:
        raise AssemblyError(
            "encoder_decoder 变体的逐层读法要显式给出源序列（编码器那一流吃的是它）。"
        )
    records: list[AttentionRecord] = []
    for layer in range(params.shape.layers):
        records.extend(
            head_records(
                params, inputs, layer=layer, heads=heads, source=source
            )
        )
    return tuple(records)


@dataclass(frozen=True)
class LayerEntropyStudy:
    """第 ① 组：逐层熵与天花板."""

    variant: str
    heads: int
    profiles: tuple[HeadProfile, ...]
    records: int
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", validate_variant(self.variant))
        if not self.profiles:
            raise ParameterError("逐层熵表不能为空。")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def ok(self) -> bool:
        """每一层的熵都不超过它的天花板（否则权重与掩码来自两次前向）."""
        return all(profile.entropy <= profile.ceiling + 1e-12 for profile in self.profiles)

    def verdicts(self) -> tuple[str, ...]:
        """逐层判词（尖 / 中等 / 接近均匀）."""
        return tuple(focus_verdict(profile) for profile in self.profiles)

    def table_lines(self) -> tuple[str, ...]:
        """整张表（逐层一行）."""
        return tuple(
            f"{profile.summary_line()} | {verdict}"
            for profile, verdict in zip(self.profiles, self.verdicts(), strict=True)
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "variant": self.variant,
            "heads": self.heads,
            "records": self.records,
            "ok": self.ok,
            "profiles": [profile.to_dict() for profile in self.profiles],
            "verdicts": list(self.verdicts()),
            "summary": summarise(self.profiles),
        }


def layer_entropy_study(
    variant: str = VARIANT_DECODER_ONLY,
    shape: VariantShape | None = None,
    *,
    heads: int = DEFAULT_HEADS,
    source: Matrix | None = None,
) -> LayerEntropyStudy:
    """跑完第 ① 组（每层先做"平均头"，再做 profile）."""
    resolved_shape = shape if shape is not None else default_shape()
    params = study_params(variant, resolved_shape)
    inputs = study_inputs(resolved_shape)
    records = all_layer_records(params, inputs, heads=heads, source=source)
    profiles = layer_profiles(records)
    return LayerEntropyStudy(
        variant=params.variant,
        heads=heads,
        profiles=profiles,
        records=len(records),
        notes=(
            f"样本 {resolved_shape.summary_line()} | {heads} 头 | 记录 {len(records)} 条",
            "每层先做逐格平均（aggregate_heads），再做 profile——"
            "因为熵与天花板是逐行的量，混合头之前不能平均",
            "归一化熵 = 熵 / 天花板，因此**跨层可比**（天花板随掩码变化）",
        ),
    )


@dataclass(frozen=True)
class RedundancyStudy:
    """第 ② 组：同一层几个头之间的逐格余弦."""

    variant: str
    labels: tuple[str, ...]
    matrices: tuple[tuple[tuple[float, ...], ...], ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", validate_variant(self.variant))
        if not self.matrices:
            raise ParameterError("冗余表不能为空。")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def diagonal_is_one(self) -> bool:
        """对角线是否都是 1（余弦的自洽检查：自己与自己必然相同）."""
        return all(
            abs(matrix[index][index] - 1.0) <= 1e-9
            for matrix in self.matrices
            for index in range(len(matrix))
        )

    def off_diagonal_mean(self) -> float:
        """非对角元素的平均（**越大越冗余**：几个头在看同一件事）."""
        values = [
            matrix[row][column]
            for matrix in self.matrices
            for row in range(len(matrix))
            for column in range(len(matrix))
            if row != column
        ]
        if not values:
            return 1.0
        return mean_of(values)

    def table_lines(self) -> tuple[str, ...]:
        """整张表：每一层一行（把余弦矩阵压成一行）."""
        lines: list[str] = []
        for label, matrix in zip(self.labels, self.matrices, strict=True):
            cells = " ".join(
                f"{matrix[row][column]:.4f}"
                for row in range(len(matrix))
                for column in range(len(matrix))
                if row != column
            )
            lines.append(f"{label:<12} | 非对角余弦 {cells or '（只有一个头）'}")
        return tuple(lines)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "variant": self.variant,
            "labels": list(self.labels),
            "diagonal_is_one": self.diagonal_is_one,
            "off_diagonal_mean": self.off_diagonal_mean(),
            "matrices": [[list(row) for row in matrix] for matrix in self.matrices],
        }


def redundancy_study(
    variant: str = VARIANT_DECODER_ONLY,
    shape: VariantShape | None = None,
    *,
    heads: int = DEFAULT_HEADS,
    source: Matrix | None = None,
) -> RedundancyStudy:
    """跑完第 ② 组（逐层算一次头间余弦）."""
    resolved_shape = shape if shape is not None else default_shape()
    if heads < 2:
        raise ParameterError(
            f"heads = {heads}：只有一个头时「头间冗余」没有定义（对角线上只有 1.0）。"
        )
    params = study_params(variant, resolved_shape)
    inputs = study_inputs(resolved_shape)
    labels: list[str] = []
    matrices: list[tuple[tuple[float, ...], ...]] = []
    for layer in range(resolved_shape.layers):
        records = head_records(params, inputs, layer=layer, heads=heads, source=source)
        labels.append(f"层 {layer}")
        matrices.append(head_redundancy(records))
    return RedundancyStudy(
        variant=params.variant,
        labels=tuple(labels),
        matrices=tuple(matrices),
        notes=(
            f"{heads} 头 | 每层 {heads}×{heads} 个余弦",
            "对角线恒为 1（自己与自己）——因此**只看非对角**",
            "余弦度量'两张表的形状像不像'：它不区分'看同一个位置'与'看同一片区域'",
        ),
    )


@dataclass(frozen=True)
class OffsetStudy:
    """第 ③ 组：质量落在离对角线 k 格上的比例."""

    variant: str
    offsets: tuple[int, ...]
    values: tuple[tuple[float, ...], ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", validate_variant(self.variant))
        if not self.values:
            raise ParameterError("偏移质量表不能为空。")
        for index, row in enumerate(self.values):
            if len(row) != len(self.offsets):
                raise ShapeError(
                    f"第 {index} 层的读数有 {len(row)} 个，而偏移有 {len(self.offsets)} 个。"
                )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    def value_at(self, layer: int, offset: int) -> float:
        """某一层某个偏移上的质量（越界当场拒绝）."""
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ParameterError(f"层号必须是整数，收到 {layer!r}。")
        if not 0 <= layer < len(self.values):
            raise ParameterError(f"层号 {layer} 越界：可选 0..{len(self.values) - 1}。")
        if offset not in self.offsets:
            raise ParameterError(
                f"偏移 {offset} 不在这一张表里：可选 {self.offsets}。"
            )
        return self.values[layer][self.offsets.index(offset)]

    def table_lines(self) -> tuple[str, ...]:
        """整张表（逐层一行）."""
        header = " | ".join(f"k={offset}" for offset in self.offsets)
        lines = [f"偏移 {header}"]
        for index, row in enumerate(self.values):
            cells = " | ".join(f"{value:.6f}" for value in row)
            lines.append(f"层 {index}：{cells}")
        return tuple(lines)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "variant": self.variant,
            "offsets": list(self.offsets),
            "values": [list(row) for row in self.values],
        }


def offset_study(
    variant: str = VARIANT_DECODER_ONLY,
    shape: VariantShape | None = None,
    *,
    heads: int = DEFAULT_HEADS,
    offsets: Sequence[int] = DEFAULT_OFFSETS,
    source: Matrix | None = None,
) -> OffsetStudy:
    """跑完第 ③ 组（逐层、逐偏移量一次质量）."""
    resolved_shape = shape if shape is not None else default_shape()
    resolved_offsets = tuple(offsets)
    if not resolved_offsets:
        raise ParameterError("offsets 不能为空。")
    params = study_params(variant, resolved_shape)
    inputs = study_inputs(resolved_shape)
    values: list[tuple[float, ...]] = []
    for layer in range(resolved_shape.layers):
        records = head_records(params, inputs, layer=layer, heads=heads, source=source)
        merged = records[0] if len(records) == 1 else _merge(records)
        values.append(tuple(offset_mass(merged, offset) for offset in resolved_offsets))
    return OffsetStudy(
        variant=params.variant,
        offsets=resolved_offsets,
        values=tuple(values),
        notes=(
            f"{heads} 头（先逐格平均再量偏移） | 样本 {resolved_shape.summary_line()}",
            "k = 0 是自看自；k = 1 是看前一个位置（**归纳头**常在这里出现）",
            "因果掩码下 k >= i 的格子不存在，因此第 0 行的读数会拉低平均——这是掩码的账",
        ),
    )


def _merge(records: Sequence[AttentionRecord]) -> AttentionRecord:
    """把几个头逐格平均（转发 ``extract.aggregate_heads``，供本模块内部使用）."""
    from smart_research_agent.explainability.extract import aggregate_heads

    return aggregate_heads(records)


@dataclass(frozen=True)
class RolloutStudy:
    """第 ④ 组：滚动之后的集中度."""

    variant: str
    alpha: float
    last_entropy: float
    rolled_entropy: float
    gain: float
    profile: HeadProfile
    layers: int
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", validate_variant(self.variant))
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def more_focused(self) -> bool:
        """滚动之后是不是更尖（``gain > 0``）——**可能是假，而那不是错误**."""
        return self.gain > 0.0

    def table_lines(self) -> tuple[str, ...]:
        """整张表（两行：最后一层与滚动）."""
        return (
            f"最后一层的归一化熵：{self.last_entropy:.6f}",
            f"滚动（α={self.alpha}，{self.layers} 层）的归一化熵："
            f"{self.rolled_entropy:.6f}",
            f"集中度变化：{self.gain:+.6f}（{'更尖' if self.more_focused else '更散/持平'}）",
            self.profile.summary_line(),
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "variant": self.variant,
            "alpha": self.alpha,
            "layers": self.layers,
            "last_entropy": self.last_entropy,
            "rolled_entropy": self.rolled_entropy,
            "gain": self.gain,
            "more_focused": self.more_focused,
            "profile": self.profile.to_dict(),
        }


def rollout_study(
    variant: str = VARIANT_DECODER_ONLY,
    shape: VariantShape | None = None,
    *,
    heads: int = DEFAULT_HEADS,
    alpha: float = DEFAULT_ALPHA,
    source: Matrix | None = None,
) -> RolloutStudy:
    """跑完第 ④ 组（滚动一次，并把两个归一化熵并排给出）."""
    resolved_shape = shape if shape is not None else default_shape()
    params = study_params(variant, resolved_shape)
    inputs = study_inputs(resolved_shape)
    records = all_layer_records(params, inputs, heads=heads, source=source)
    last_entropy, rolled_entropy, gain = rollout_focus(records, alpha=alpha)
    rolled = rollout_record(records, alpha=alpha)
    return RolloutStudy(
        variant=params.variant,
        alpha=alpha,
        last_entropy=last_entropy,
        rolled_entropy=rolled_entropy,
        gain=gain,
        profile=profile_of(rolled),
        layers=len(aggregate_by_layer(records)),
        notes=(
            f"Â = α·A + (1−α)·I，R = Â_L ⋯ Â_1（{len(aggregate_by_layer(records))} 层）",
            "两个数都是**归一化熵**（熵 / 天花板）——否则掩码不同的两张表不能比",
            "gain 为负是合法的：一个已经很尖的模型滚动之后可能更散，"
            "本课如实给出三个数，判词留给调用方",
        ),
    )


@dataclass(frozen=True)
class CeilingStudy:
    """第 ⑤ 组：两种掩码的熵天花板（"尖"里有多少是掩码造成的）."""

    tokens: int
    entries: tuple[tuple[str, float], ...]
    ratio: float
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if isinstance(self.tokens, bool) or not isinstance(self.tokens, int) or self.tokens < 1:
            raise ParameterError(f"tokens 必须是 >= 1 的整数，收到 {self.tokens!r}。")
        if not self.entries:
            raise ParameterError("天花板对照不能为空。")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    def ceiling_of(self, kind: str) -> float:
        """某种掩码的天花板（不认识的名字当场拒绝）."""
        for name, value in self.entries:
            if name == kind:
                return value
        raise ParameterError(
            f"表里没有掩码 {kind!r}：可选 {[name for name, _ in self.entries]}。"
        )

    def table_lines(self) -> tuple[str, ...]:
        """整张表（每种掩码一行）."""
        lines = [
            f"{kind:<7} | 天花板 {value:.6f}"
            for kind, value in self.entries
        ]
        lines.append(f"因果 / 全开 = {self.ratio:.6f}（n 越大越接近 1）")
        return tuple(lines)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "tokens": self.tokens,
            "entries": [[kind, value] for kind, value in self.entries],
            "ratio": self.ratio,
        }


def ceiling_study(shape: VariantShape | None = None) -> CeilingStudy:
    """跑完第 ⑤ 组：两张结构性掩码的熵天花板."""
    resolved_shape = shape if shape is not None else default_shape()
    entries = tuple(
        (kind, mask_entropy_ceiling(mask_of(kind, resolved_shape.tokens)))
        for kind in MASK_KINDS
    )
    values = dict(entries)
    ratio = values[MASK_CAUSAL] / values[MASK_FULL]
    return CeilingStudy(
        tokens=resolved_shape.tokens,
        entries=entries,
        ratio=ratio,
        notes=(
            f"样本 tokens = {resolved_shape.tokens}",
            "全开掩码的天花板是 ln n；因果掩码是 (1/n)·Σ ln(i+1)",
            "**这张表解释了一半的'尖'**：因果变体的第 0 行天花板是 ln 1 = 0，"
            "因此它天然比全开变体尖",
        ),
    )


def how_to_read() -> tuple[str, ...]:
    """五张表的阅读顺序（演示脚本与教程共用同一段话）."""
    return (
        "① 先看天花板（第 ⑤ 组）：两个变体的'均匀基线'本来就不一样",
        "② 再看归一化熵（第 ① 组）：这才是在同一把尺子上比'谁更尖'",
        "③ 然后看头间冗余（第 ② 组）：几个头在看同一件事吗",
        "④ 再看偏移（第 ③ 组）：它看的是相邻位置还是远处",
        "⑤ 最后看滚动（第 ④ 组）：信息至少走了几跳",
    )


def variant_labels() -> tuple[str, ...]:
    """三个变体的名字（教程第 1 章那张表的数据源）."""
    return tuple(VARIANTS)


def all_ceilings(shape: VariantShape | None = None) -> tuple[tuple[str, float, float], ...]:
    """每个变体的``(变体, 天花板, 归一化基线)``：给"基线不同"这件事一个数.

    归一化基线就是"如果这一层完全均匀，它的归一化熵是多少"——
    对全开掩码是 1.0（`ln n / ln n`），对因果掩码也是 1.0。
    因此第 ① 组的归一化熵与 1.0 的差，才是"模型学出来的尖"。
    """
    resolved_shape = shape if shape is not None else default_shape()
    full = mask_entropy_ceiling(mask_of(MASK_FULL, resolved_shape.tokens))
    causal = mask_entropy_ceiling(mask_of(MASK_CAUSAL, resolved_shape.tokens))
    return tuple(
        (
            variant,
            causal if variant == VARIANT_DECODER_ONLY else full,
            1.0,
        )
        for variant in VARIANTS
    )


__all__ = [
    "DEFAULT_OFFSETS",
    "INPUT_SEED",
    "PARAMETER_SEED",
    "SOURCE_SEED",
    "CeilingStudy",
    "LayerEntropyStudy",
    "OffsetStudy",
    "RedundancyStudy",
    "RolloutStudy",
    "all_ceilings",
    "all_layer_records",
    "ceiling_study",
    "default_shape",
    "how_to_read",
    "layer_entropy_study",
    "offset_study",
    "redundancy_study",
    "rollout_study",
    "study_inputs",
    "study_params",
    "variant_labels",
]
