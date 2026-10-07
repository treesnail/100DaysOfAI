"""七条性质与三类判据（day099 / G1-D1）.

本模块是"这条链装对了吗"的判据所在。七条性质按三类判据分：

```text
相等（==）   ① capabilities_are_covered      覆盖计数 == 8
             ② stages_match_spec            阶段序列差异项数 == 0
             ③ assembly_is_reproducible     两次运行的差异项数 == 0
             ⑦ document_covers_all_capabilities  文档缺失能力数 == 0
下界（>=）   ④ retrieval_recall_meets_floor  召回 >= 1.0  ← 唯一一条下界
上界（<=）   ⑤ grounding_has_no_hallucination  幻觉引用 <= 0
             ⑥ cost_matches_hand_formula    成本相对差 <= 1e-12
```

## 一、为什么第 ④ 条必须是下界而不是相等

召回度量的是"金标准命中了没有"。写成"等于 1.0"的相等判据会在**取回更多条**
（更好）时误报失败——因为召回一旦到 1.0 就封顶，取回更多条不会让它超过 1.0，
但"等于"这种写法会让人误以为"比 1.0 大才算更好"。下界才是它真正的方向，
而这条判据在报告里的读数是"越低越危险"。

## 二、为什么第 ⑤ ⑥ 条是上界而不是相等

```text
幻觉引用    定义上应当恰好是 0；写成"<= 0"与"== 0"在这里等价，
            但写成上界能让"为什么不是下界"这件事在判据名上说清楚：
            它们是**越少越好**的量，因此兜的是上限。
成本相对差  是一个"应当为 0"的浮点差；用上界（<= 容差）而不是相等，
            是因为浮点相对差的最后几位在不同平台上可能不同——
            它在容差内即算对上，而容差本身写死成 1e-12（**不是**随意放宽）。
```

## 三、一条纪律：`CrossCheck.passed` 必须与 `PropertyOutcome.passed` 一致

一份报告里，"那一行说通过"与"它挂的对账说没通过"是最糟的状态——
读的人只会相信看起来更合理的那一个。因此 :class:`PropertyOutcome` 在
``__post_init__`` 里当场拒绝两者不一致的构造。

## 四、与既有包的接缝

- **上游**：:mod:`capstone.assembly`（``run`` / ``SystemRun``）、
  :mod:`capstone.manifest`（``Manifest``）、:mod:`capstone.document`（渲染与覆盖）、
  :mod:`capstone.adapters`（手算成本公式）；
- **脚下**：性质名单只有一处定义（``types.CAPSTONE_PROPERTIES``），
  :func:`check_all` 会在导入期与它对齐；
- **下游**：:mod:`capstone.study` 打印性质表，演示脚本逐条印读数行。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.capstone import adapters, assembly, document
from smart_research_agent.capstone.errors import (
    CapabilityError,
    DocumentError,
    NumericError,
    ParameterError,
)
from smart_research_agent.capstone.manifest import Manifest, build_manifest
from smart_research_agent.capstone.types import (
    ASSEMBLY_STAGES,
    CAPABILITY_ORDER,
    CAPSTONE_PROPERTIES,
    CRITERION_EQUALITY,
    CRITERION_LOWER_BOUND,
    CRITERION_UPPER_BOUND,
    PROPERTY_ASSEMBLY_IS_REPRODUCIBLE,
    PROPERTY_CAPABILITIES_ARE_COVERED,
    PROPERTY_COST_MATCHES_HAND_FORMULA,
    PROPERTY_DOCUMENT_COVERS_ALL_CAPABILITIES,
    PROPERTY_GROUNDING_HAS_NO_HALLUCINATION,
    PROPERTY_RETRIEVAL_RECALL_MEETS_FLOOR,
    PROPERTY_STAGES_MATCH_SPEC,
    STAGE_GROUND,
    STAGE_PACK,
    STAGE_RETRIEVE,
)

#: 召回的下界（金标准必须出现在名单里）.
RECALL_FLOOR = 1.0

#: 幻觉引用的上限（一条都不许有）.
HALLUCINATION_CEILING = 0.0

#: 成本相对差的容差（浮点对账；**写死一个很小的数**，而不是"随手放宽"）.
COST_TOLERANCE = 1e-12

#: 相等判据的浮点容差（逐位 / 整数相等时不走它）.
EQUALITY_TOLERANCE = 1e-12


@dataclass(frozen=True)
class CrossCheck:
    """一次对账：来源、读数、判据（**含方向**：相等 / 上界 / 下界）.

    三个字段里**最多只能给一个方向**：给 ``lower_bound`` 时判据是"读数 >= 下界"，
    给 ``upper_bound`` 时是"读数 <= 上界"，都不给时才是"读数 == 期望"。
    同时给两个方向会被当场拒绝——"既要又不要"的判据没有明确真值。
    """

    name: str
    left: str
    right: str
    reading: float
    expected: float
    exact: bool = True
    upper_bound: float | None = None
    lower_bound: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.left or not self.right:
            raise ParameterError("对账的来源与名字都不能为空。")
        if not math.isfinite(self.reading) or not math.isfinite(self.expected):
            raise NumericError(
                f"对账读数必须有限：reading={self.reading!r}、expected={self.expected!r}。"
            )
        if self.upper_bound is not None and self.lower_bound is not None:
            raise ParameterError(
                f"对账 {self.name!r} 同时给了上界与下界："
                "一个数不可能既是上界又是下界——请只给一个方向。"
            )
        if self.upper_bound is not None and (
            not math.isfinite(self.upper_bound) or self.upper_bound < 0
        ):
            raise NumericError(f"上界必须是有限非负数，收到 {self.upper_bound!r}。")
        if self.lower_bound is not None and not math.isfinite(self.lower_bound):
            raise NumericError(f"下界必须是有限数，收到 {self.lower_bound!r}。")

    @property
    def criterion(self) -> str:
        """这次用的是哪一类判据（相等 / 上界 / 下界）."""
        if self.lower_bound is not None:
            return CRITERION_LOWER_BOUND
        if self.upper_bound is not None:
            return CRITERION_UPPER_BOUND
        return CRITERION_EQUALITY

    @property
    def passed(self) -> bool:
        """三类判据各自的判定."""
        if self.lower_bound is not None:
            return float(self.reading) >= float(self.lower_bound) - EQUALITY_TOLERANCE
        if self.upper_bound is not None:
            return float(self.reading) <= float(self.upper_bound) + EQUALITY_TOLERANCE
        if self.exact:
            return self.reading == self.expected
        return abs(float(self.reading) - float(self.expected)) <= EQUALITY_TOLERANCE

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段."""
        return {
            "name": self.name,
            "criterion": self.criterion,
            "reading": self.reading,
            "expected": self.expected,
            "upper_bound": self.upper_bound,
            "lower_bound": self.lower_bound,
            "passed": self.passed,
        }

    def line(self) -> str:
        """一行可读读数（带上方向与来源）."""
        verdict = "满足" if self.passed else "不满足"
        if self.lower_bound is not None:
            return (
                f"[{verdict}] {self.name}: 读数 {self.reading:.6e} ≥ 下界 "
                f"{self.lower_bound:.6e}（{self.left} vs {self.right}）"
            )
        if self.upper_bound is not None:
            return (
                f"[{verdict}] {self.name}: 读数 {self.reading:.6e} ≤ 上界 "
                f"{self.upper_bound:.6e}（{self.left} vs {self.right}）"
            )
        return (
            f"[{verdict}] {self.name}: {self.left} vs {self.right} | "
            f"读数 {self.reading} / 期望 {self.expected}"
        )


@dataclass(frozen=True)
class PropertyOutcome:
    """一条性质的结论：适用性 + 结果 + 证据行 + 一次对账."""

    name: str
    applicable: bool
    passed: bool
    evidence: tuple[str, ...] = field(default_factory=tuple)
    cross_check: CrossCheck | None = None

    def __post_init__(self) -> None:
        if not self.applicable and self.passed:
            raise NumericError(
                f"性质 {self.name!r} 标了'不适用'却又标了'通过'：两者必须分开——"
                "否则'把这条检查删掉'与'它通过了'在报告里长得一模一样。"
            )
        if self.cross_check is not None and self.cross_check.passed != self.passed:
            raise NumericError(
                f"性质 {self.name!r} 的结论与它挂的对账不一致："
                f"outcome.passed={self.passed}、cross_check.passed={self.cross_check.passed}。"
                "一份报告里'那一行说通过'与'它挂的对账说没通过'是最糟的状态——"
                "读的人只会相信看起来更合理的那一个。"
            )

    @property
    def criterion(self) -> str:
        """这条性质用的是哪一类判据（没有对账时按相等记）."""
        return CRITERION_EQUALITY if self.cross_check is None else self.cross_check.criterion

    def line(self) -> str:
        """一行可读结论（不适用也要印出来）."""
        if not self.applicable:
            return f"[不适用] {self.name} | {'；'.join(self.evidence)}"
        verdict = "通过" if self.passed else "失败"
        detail = "；".join(self.evidence)
        suffix = f" | {detail}" if detail else ""
        return f"[{verdict}] {self.name}{suffix}"


@dataclass(frozen=True)
class PropertyReport:
    """一组性质的报告（``ok`` 要求**所有适用**的都通过）."""

    outcomes: tuple[PropertyOutcome, ...]

    @property
    def applicable(self) -> tuple[PropertyOutcome, ...]:
        """适用（``applicable=True``）的那些性质."""
        return tuple(outcome for outcome in self.outcomes if outcome.applicable)

    @property
    def ok(self) -> bool:
        """是否全部通过——不适用不算通过、也不算失败."""
        return all(outcome.passed for outcome in self.applicable)

    def require_ok(self) -> None:
        """不通过时抛 :class:`errors.DocumentError`（"拒绝交付"的那条路）."""
        if self.ok:
            return
        failures = [outcome.line() for outcome in self.applicable if not outcome.passed]
        raise DocumentError("性质检查未全部通过：" + "；".join(failures))

    def lines(self) -> tuple[str, ...]:
        """逐行文本（**先印不适用**，让"这一次没查它"一眼可见）."""
        return tuple(
            outcome.line()
            for outcome in sorted(self.outcomes, key=lambda outcome: outcome.applicable)
        )

    def to_dict(self) -> dict[str, object]:
        """摊平成可 JSON 化的字段."""
        return {
            "ok": self.ok,
            "counts": {"total": len(self.outcomes), "applicable": len(self.applicable)},
            "lines": list(self.lines()),
        }


# --------------------------------------------------------------------------- #
# 七条性质
# --------------------------------------------------------------------------- #


def check_capabilities_are_covered(manifest: Manifest | None = None) -> PropertyOutcome:
    """① 8 项能力全部被覆盖（每一项都有在场的承担子包）."""
    resolved = build_manifest() if manifest is None else manifest
    covered = len(resolved.covered)
    total = len(CAPABILITY_ORDER)
    check = CrossCheck(
        name="被覆盖的能力数",
        left="manifest.covered",
        right="CAPABILITY_ORDER 的 8 项",
        reading=float(covered),
        expected=float(total),
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_CAPABILITIES_ARE_COVERED,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"覆盖 {covered}/{total} 项能力",
            "未覆盖的：" + ("、".join(resolved.missing) if resolved.missing else "无"),
        ),
        cross_check=check,
    )


def check_stages_match_spec(run_result: assembly.SystemRun | None = None) -> PropertyOutcome:
    """② 运行的阶段序列与 ASSEMBLY_STAGES 逐位相同."""
    resolved = assembly.run() if run_result is None else run_result
    actual = resolved.stages
    mismatches = sum(1 for a, b in zip(actual, ASSEMBLY_STAGES, strict=True) if a != b)
    check = CrossCheck(
        name="阶段序列的差异项数",
        left="SystemRun.stages",
        right="ASSEMBLY_STAGES",
        reading=float(mismatches),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_STAGES_MATCH_SPEC,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"{len(actual)} 段与规格逐位相同：{mismatches == 0}",
            "序列：" + " → ".join(actual),
        ),
        cross_check=check,
    )


def check_assembly_is_reproducible(
    first: assembly.SystemRun | None = None,
    second: assembly.SystemRun | None = None,
) -> PropertyOutcome:
    """③ 同一输入两次运行**逐位相同**（装配链里没有任何未固定的随机性）."""
    left = assembly.run() if first is None else first
    right = assembly.run() if second is None else second
    diff = left.diff_count(right)
    check = CrossCheck(
        name="两次运行的差异项数",
        left="run() 第一次",
        right="run() 第二次",
        reading=float(diff),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_ASSEMBLY_IS_REPRODUCIBLE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"差异项数 {diff}（0 = 逐位相同）",
            f"摘要 {left.digest()} / {right.digest()}",
        ),
        cross_check=check,
    )


def check_retrieval_recall_meets_floor(run_result: assembly.SystemRun | None = None) -> PropertyOutcome:
    """④ **下界**：端到端检索的召回不低于 1.0（金标准必须出现在名单里）."""
    resolved = assembly.run() if run_result is None else run_result
    check = CrossCheck(
        name="端到端检索的召回",
        left="evaluation.rag_metrics.retrieval_recall",
        right=f"下界 {RECALL_FLOOR:.1f}",
        reading=resolved.recall,
        expected=RECALL_FLOOR,
        exact=False,
        lower_bound=RECALL_FLOOR,
    )
    return PropertyOutcome(
        name=PROPERTY_RETRIEVAL_RECALL_MEETS_FLOOR,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"召回 {resolved.recall:.4f} ≥ 下界 {RECALL_FLOOR:.1f}",
            f"检索读数 {resolved.reading_of(STAGE_RETRIEVE):.0f} 条 / "
            f"打包 {resolved.reading_of(STAGE_PACK):.0f} 条",
        ),
        cross_check=check,
    )


def check_grounding_has_no_hallucination(
    run_result: assembly.SystemRun | None = None,
) -> PropertyOutcome:
    """⑤ **上界**：答案里没有幻觉引用（引用的编号都能在提示词里找到）."""
    resolved = assembly.run() if run_result is None else run_result
    hallucinations = resolved.reading_of(STAGE_GROUND)
    check = CrossCheck(
        name="幻觉引用条数",
        left="GroundingReport.invalid",
        right=f"上界 {HALLUCINATION_CEILING:.0f}",
        reading=hallucinations,
        expected=HALLUCINATION_CEILING,
        exact=True,
        upper_bound=HALLUCINATION_CEILING,
    )
    return PropertyOutcome(
        name=PROPERTY_GROUNDING_HAS_NO_HALLUCINATION,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"幻觉引用 {hallucinations:.0f} ≤ 上界 {HALLUCINATION_CEILING:.0f}",
            resolved.record_of(STAGE_GROUND).detail,
        ),
        cross_check=check,
    )


def check_cost_matches_hand_formula(run_result: assembly.SystemRun | None = None) -> PropertyOutcome:
    """⑥ **上界**：记出来的费用与手算公式的相对差不超过容差."""
    resolved = assembly.run() if run_result is None else run_result
    hand = adapters.hand_cost_formula(resolved.prompt_tokens, resolved.completion_tokens)
    if hand == 0.0:
        relative = 0.0 if resolved.cost_usd == 0.0 else float("inf")
    else:
        relative = abs(resolved.cost_usd - hand) / abs(hand)
    check = CrossCheck(
        name="成本与手算公式的相对差",
        left="CostTracker.report().total_cost_usd",
        right="hand_cost_formula（token × 单价 / 1000）",
        reading=relative,
        expected=0.0,
        exact=False,
        upper_bound=COST_TOLERANCE,
    )
    return PropertyOutcome(
        name=PROPERTY_COST_MATCHES_HAND_FORMULA,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"相对差 {relative:.6e} ≤ 容差 {COST_TOLERANCE:.0e}",
            f"记账 ${resolved.cost_usd:.8f} / 手算 ${hand:.8f} | token {resolved.total_tokens}",
        ),
        cross_check=check,
    )


def check_document_covers_all_capabilities(
    documents: dict[str, str] | None = None,
) -> PropertyOutcome:
    """⑦ 渲染出的文档覆盖全部 8 项能力（每一个 id 都出现在正文里）."""
    resolved = document.render_documents() if documents is None else documents
    missing = document.missing_capabilities(resolved)
    check = CrossCheck(
        name="文档缺失的能力数",
        left="render_documents() 的正文",
        right="CAPABILITY_ORDER 的 8 项",
        reading=float(len(missing)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_DOCUMENT_COVERS_ALL_CAPABILITIES,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"合并正文 {sum(len(text) for text in resolved.values())} 字符；缺失 {len(missing)} 项",
            "缺失的：" + ("、".join(missing) if missing else "无"),
        ),
        cross_check=check,
    )


def check_all(
    run_result: assembly.SystemRun | None = None,
    manifest: Manifest | None = None,
    documents: dict[str, str] | None = None,
) -> PropertyReport:
    """一次跑完七条性质（顺序与 :data:`types.CAPSTONE_PROPERTIES` 一致）.

    第一、二次运行**共用同一份 run**（第 ③ 条自己再跑一次做对照），
    因此"7 条性质"总共只跑 2 次端到端链，而不是 6 次。
    """
    resolved_run = assembly.run() if run_result is None else run_result
    resolved_manifest = build_manifest() if manifest is None else manifest
    resolved_documents = document.render_documents(resolved_manifest) if documents is None else documents
    outcomes = (
        check_capabilities_are_covered(resolved_manifest),
        check_stages_match_spec(resolved_run),
        check_assembly_is_reproducible(resolved_run, None),
        check_retrieval_recall_meets_floor(resolved_run),
        check_grounding_has_no_hallucination(resolved_run),
        check_cost_matches_hand_formula(resolved_run),
        check_document_covers_all_capabilities(resolved_documents),
    )
    names = {outcome.name for outcome in outcomes}
    extra = names - set(CAPSTONE_PROPERTIES)
    missing = set(CAPSTONE_PROPERTIES) - names
    if extra or missing:  # pragma: no cover - 只在有人改性质名单时触发
        raise CapabilityError(
            "性质名单与 types.CAPSTONE_PROPERTIES 不一致："
            f"多 {sorted(extra)}、缺 {sorted(missing)}。"
            "名单对不上时，报告里那七行会安静地少一行或多一行。"
        )
    return PropertyReport(outcomes=outcomes)


__all__ = [
    "COST_TOLERANCE",
    "EQUALITY_TOLERANCE",
    "HALLUCINATION_CEILING",
    "RECALL_FLOOR",
    "CrossCheck",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_assembly_is_reproducible",
    "check_capabilities_are_covered",
    "check_cost_matches_hand_formula",
    "check_document_covers_all_capabilities",
    "check_grounding_has_no_hallucination",
    "check_retrieval_recall_meets_floor",
    "check_stages_match_spec",
]
