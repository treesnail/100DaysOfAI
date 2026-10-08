"""七条性质与两类判据（day088 / M7-D12）.

本模块是"这张图拼对了吗"的判据所在。七条性质分四类：

```text
存在性    每条原理都有实现落点（artifact 可解析）
         每个应用都至少被一条原理支撑（没有孤岛）
可复现性  同一个探针连续两次调用得到的证据**逐位相同**（单次计算、可重复）
跨天对账  注意力每一行是分布（调 transformer_core，偏差 <= 容差）
         缓存公式与 day087 的读数一致（调 inference_optim.cache_bytes，整数相等）
次序      提纲的层次序满足前置依赖（拓扑序检查）
完整性    原理文档覆盖全部十二条命题
```

## 判据分两类（与 day087 同源）

```text
相等（== / 容差）   读数与期望一致（存在性、可复现、整数、次序、覆盖）
上界（<=）          读数不超过某个界（注意力行和偏差 <= 容差）
```

因此 :class:`CrossCheck` 带 ``upper_bound`` 字段：有它时判据是"≤"，没有时才是"=="。
把两类混成一个判据，就会出现"实测误差恰好等于 0（因为输入全是 0）被当成通过"这种事。

## 两条跨天对账

```text
注意力行分布   reconcile.probe_attention_rows_are_distributions ↔ transformer_core
缓存字节公式   reconcile.probe_cache_bytes_is_a_formula       ↔ inference_optim（day087）
```

两条都调用**别的包**来算读数，因此它们能抓住"本包自己写错了一个数"——
自证是不成立的，跨包对账才是。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smart_research_agent.principle_map import graph as graph_module
from smart_research_agent.principle_map import outline as outline_module
from smart_research_agent.principle_map import reconcile
from smart_research_agent.principle_map.errors import CoverageError, NumericError
from smart_research_agent.principle_map.types import (
    APPLICATIONS,
    PRINCIPLE_PROPERTIES,
    PROPERTY_ATTENTION_ROWS_ARE_DISTRIBUTIONS,
    PROPERTY_CACHE_FORMULA_MATCHES_DAY087,
    PROPERTY_DOCUMENT_COVERS_ALL_PRINCIPLES,
    PROPERTY_EVIDENCE_IS_REPRODUCIBLE,
    PROPERTY_EVERY_APPLICATION_IS_SUPPORTED,
    PROPERTY_EVERY_PRINCIPLE_HAS_ARTIFACT,
    PROPERTY_OUTLINE_RESPECTS_DEPENDENCIES,
)

#: 行的容差（注意力行和与 1 允许的偏差）——它来自 ``transformer_core`` 的口径.
ROW_TOLERANCE = 1e-9


@dataclass(frozen=True)
class CrossCheck:
    """两个来源之间的一次对账：来源、读数、判据（**含方向与上界**）.

    与 day087 的同类逐字同源：有 ``upper_bound`` 的时候判据是"实测 <= 上界"，
    没有的时候是"两个数相等"。
    """

    name: str
    left: str
    right: str
    reading: float
    expected: float
    exact: bool = True
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        import math

        if not math.isfinite(self.reading) or not math.isfinite(self.expected):
            raise NumericError(
                f"对账读数必须有限：reading={self.reading!r}、expected={self.expected!r}。"
            )
        if self.upper_bound is not None and (
            not math.isfinite(self.upper_bound) or self.upper_bound < 0
        ):
            raise NumericError(f"上界必须是有限非负数，收到 {self.upper_bound!r}。")

    @property
    def passed(self) -> bool:
        """相等（逐位 / 整数 / 容差）或"不超过上界"两种判据."""
        if self.upper_bound is not None:
            return float(self.reading) <= float(self.upper_bound) + 1e-12
        if self.exact:
            return self.reading == self.expected
        return abs(float(self.reading) - float(self.expected)) <= 1e-12

    def line(self) -> str:
        """一行可读的读数（带上两个来源与那个上界）."""
        verdict = "满足" if self.passed else "不满足"
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
    """一条性质的结论：适用性 + 结果 + 证据行."""

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

    def line(self) -> str:
        """一行可读的结论（不适用也要印出来）."""
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
        """不通过时抛 :class:`errors.CoverageError`."""
        if self.ok:
            return
        failures = [outcome.line() for outcome in self.applicable if not outcome.passed]
        raise CoverageError("性质检查未全部通过：" + "；".join(failures))

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


def check_every_principle_has_artifact(
    graph: graph_module.PrincipleGraph | None = None,
) -> PropertyOutcome:
    """每条原理的 ``artifact`` 都能被解析成一个真实存在的对象.

    这是图上"原理 → 实现"那些箭头里最基础的一条：指不到东西的箭头是装饰。
    """
    resolved = graph or graph_module._default_graph()
    failures: list[str] = []
    for item in resolved.principles:
        try:
            graph_module.resolve_artifact(item.artifact)
        except Exception as error:  # noqa: BLE001 - 报告里只印第一行
            failures.append(f"{item.id}（{item.artifact}：{str(error).splitlines()[0]}）")
    check = CrossCheck(
        name="实现落点可解析",
        left="十二条 artifact",
        right="importlib 解析结果",
        reading=float(len(failures)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_EVERY_PRINCIPLE_HAS_ARTIFACT,
        applicable=True,
        passed=not failures,
        evidence=(
            f"解析成功 {len(resolved.principles) - len(failures)}/{len(resolved.principles)}",
            "失败：" + ("、".join(failures) if failures else "无"),
        ),
        cross_check=check,
    )


def check_every_application_is_supported(
    graph: graph_module.PrincipleGraph | None = None,
) -> PropertyOutcome:
    """六个应用每一个都至少被一条原理支撑（没有孤岛）."""
    resolved = graph or graph_module._default_graph()
    unsupported = resolved.unsupported
    check = CrossCheck(
        name="没有入边的应用数",
        left="图上的应用",
        right="APPLICATIONS 的六个",
        reading=float(len(unsupported)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_EVERY_APPLICATION_IS_SUPPORTED,
        applicable=True,
        passed=not unsupported,
        evidence=(
            f"有支撑的应用 {len(resolved.supported)}/{len(APPLICATIONS)}",
            "没有入边的：" + ("、".join(unsupported) if unsupported else "无"),
        ),
        cross_check=check,
    )


def check_evidence_is_reproducible() -> PropertyOutcome:
    """同一个探针连续两次调用得到的证据**逐位相同**（单次计算、可重复）."""
    first = reconcile.probe_all()
    second = reconcile.probe_all()
    mismatches = 0
    for left, right in zip(first, second, strict=True):
        same = (
            left.principle == right.principle
            and left.reading == right.reading
            and left.expected == right.expected
            and left.exact == right.exact
            and left.upper_bound == right.upper_bound
        )
        mismatches += int(not same)
    check = CrossCheck(
        name="两次探针的逐位相同",
        left="probe_all() 第一次",
        right="probe_all() 第二次",
        reading=float(mismatches),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_EVIDENCE_IS_REPRODUCIBLE,
        applicable=True,
        passed=mismatches == 0,
        evidence=(
            f"{len(first)} 条证据逐位相同：{mismatches == 0}（不等个数 {mismatches}）",
            "读数示例：" + "；".join(f"{item.principle}={item.reading}" for item in first[:3]),
        ),
        cross_check=check,
    )


def check_attention_rows_are_distributions() -> PropertyOutcome:
    """**跨天对账**：调 ``transformer_core``，判据是'行和与 1 的偏差 <= 容差'."""
    evidence = reconcile.probe_attention_rows_are_distributions()
    bound = evidence.upper_bound if evidence.upper_bound is not None else ROW_TOLERANCE
    check = CrossCheck(
        name="注意力行和与 1 的偏差",
        left="transformer_core.self_attention 的权重",
        right="每一行的和为 1",
        reading=evidence.reading,
        expected=0.0,
        exact=False,
        upper_bound=bound,
    )
    return PropertyOutcome(
        name=PROPERTY_ATTENTION_ROWS_ARE_DISTRIBUTIONS,
        applicable=True,
        passed=check.passed,
        evidence=(evidence.note, f"判据：偏差 <= {bound:.0e}"),
        cross_check=check,
    )


def check_cache_formula_matches_day087() -> PropertyOutcome:
    """**跨天对账**：调 ``inference_optim.cache_bytes``，整数相等（没有容差空间）."""
    evidence = reconcile.probe_cache_bytes_is_a_formula()
    check = CrossCheck(
        name="缓存公式的不等个数",
        left="inference_optim.types.cache_bytes",
        right="2 · L · T · h · bytes",
        reading=evidence.reading,
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_CACHE_FORMULA_MATCHES_DAY087,
        applicable=True,
        passed=check.passed,
        evidence=(evidence.note,),
        cross_check=check,
    )


def check_outline_respects_dependencies(
    sections: tuple[object, ...] | None = None,
) -> PropertyOutcome:
    """提纲的层次序满足前置依赖（拓扑序检查）."""
    resolved = outline_module.build_outline() if sections is None else tuple(sections)
    failure = ""
    try:
        outline_module.check_order(resolved)  # type: ignore[arg-type]
    except Exception as error:  # noqa: BLE001 - 报告里只印第一行
        failure = str(error).splitlines()[0]
    order = [getattr(section, "layer", "?") for section in resolved]
    check = CrossCheck(
        name="提纲次序",
        left="提纲的层次序",
        right="LAYER_ORDER（数学 → 注意力 → 表征 → 推理）",
        reading=0.0 if not failure else 1.0,
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_OUTLINE_RESPECTS_DEPENDENCIES,
        applicable=True,
        passed=not failure,
        evidence=(
            f"{len(resolved)} 节的层次序 {order}",
            f"总时长 {outline_module.outline_minutes(resolved)} 分钟" if resolved else "空提纲",
            failure or "拓扑序检查通过",
        ),
        cross_check=check,
    )


def check_document_covers_all_principles(
    graph: graph_module.PrincipleGraph | None = None,
) -> PropertyOutcome:
    """原理文档覆盖全部十二条命题（每一条的 id 都出现在正文里）."""
    document = outline_module.render_document(graph)
    missing = outline_module.missing_principles(document)
    check = CrossCheck(
        name="文档缺失的命题数",
        left="render_document() 的正文",
        right="types.PRINCIPLES 的十二条",
        reading=float(len(missing)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_DOCUMENT_COVERS_ALL_PRINCIPLES,
        applicable=True,
        passed=not missing,
        evidence=(
            f"文档 {len(document)} 字符、缺失 {len(missing)} 条",
            "缺失的：" + ("、".join(missing) if missing else "无"),
        ),
        cross_check=check,
    )


def check_all(graph: graph_module.PrincipleGraph | None = None) -> PropertyReport:
    """一次跑完七条性质（顺序与 :data:`types.PRINCIPLE_PROPERTIES` 一致）."""
    outcomes = (
        check_every_principle_has_artifact(graph),
        check_every_application_is_supported(graph),
        check_evidence_is_reproducible(),
        check_attention_rows_are_distributions(),
        check_cache_formula_matches_day087(),
        check_outline_respects_dependencies(),
        check_document_covers_all_principles(graph),
    )
    names = {outcome.name for outcome in outcomes}
    extra = names - set(PRINCIPLE_PROPERTIES)
    missing = set(PRINCIPLE_PROPERTIES) - names
    if extra or missing:  # pragma: no cover - 只在有人改性质名单时触发
        raise CoverageError(
            "性质名单与 types.PRINCIPLE_PROPERTIES 不一致："
            f"多 {sorted(extra)}、缺 {sorted(missing)}。"
            "名单对不上时，报告里那七行会安静地少一行或多一行。"
        )
    return PropertyReport(outcomes=outcomes)


__all__ = [
    "ROW_TOLERANCE",
    "CrossCheck",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_attention_rows_are_distributions",
    "check_cache_formula_matches_day087",
    "check_document_covers_all_principles",
    "check_evidence_is_reproducible",
    "check_every_application_is_supported",
    "check_every_principle_has_artifact",
    "check_outline_respects_dependencies",
]
