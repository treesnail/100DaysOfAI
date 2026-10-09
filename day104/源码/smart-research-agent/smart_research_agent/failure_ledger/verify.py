"""七条性质与三类判据（day102）.

本模块是"这份台账成不成立"的判据所在。七条性质按三类判据分：

```text
相等（==）   ① ledger_covers_all_error_modules     缺失数 == 0
             ② qualified_names_are_unique          重复的 包.名 数 == 0
             ③ ledger_is_reproducible              两次构建的差异项数 == 0
             ④ every_module_has_exactly_one_root   根数不是 1 的模块数 == 0
上界（<=）   ⑤ base_references_resolve              未解析基类数 <= 0
下界（>=）   ⑥ every_family_has_a_parent           最小基类数 >= 1
             ⑦ inheritance_depth_is_positive        最大继承深度 >= 1
```

## 一、为什么 ⑥ ⑦ 必须是下界

```text
"每个族至少一个基类"   基类多是正常的（多继承）⇒ 只能从下面兜住
"最大继承深度 >= 1"    "链存在"这件事只能从下面兜住（写成 == 1 会漏掉更深的链）
```

把它写成相等（``== 1``）会在"某个族写了两个基类"或"链变深"时误报失败。下界才是它们的方向。

## 二、为什么 ⑤ 是上界

"未解析基类数"是一条**越少越好**的量：0 才是健康的，越多越坏。
因此它兜的是上限。这与 day101 的"分数不超过 1"是同一种写法：
**判据的方向要与这个量的方向一致。**

## 三、一条纪律：`CrossCheck.passed` 必须与 `PropertyOutcome.passed` 一致

一份报告里，"那一行说通过"与"它挂的对账说没通过"是最糟的状态——
读的人只会相信看起来更合理的那一个。因此 :class:`PropertyOutcome` 在
``__post_init__`` 里当场拒绝两者不一致的构造。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.failure_ledger.errors import NumericError, ParameterError
from smart_research_agent.failure_ledger.ledger import Family, Ledger, build_ledger
from smart_research_agent.failure_ledger.scan import ScannedModule, expected_modules
from smart_research_agent.failure_ledger.types import (
    FAILURE_LEDGER_PROPERTIES,
    CRITERION_EQUALITY,
    CRITERION_LOWER_BOUND,
    CRITERION_UPPER_BOUND,
    PROPERTY_BASE_REFERENCES_RESOLVE,
    PROPERTY_EVERY_FAMILY_HAS_A_PARENT,
    PROPERTY_EVERY_MODULE_HAS_EXACTLY_ONE_ROOT,
    PROPERTY_INHERITANCE_DEPTH_IS_POSITIVE,
    PROPERTY_LEDGER_COVERS_ALL_ERROR_MODULES,
    PROPERTY_LEDGER_IS_REPRODUCIBLE,
    PROPERTY_QUALIFIED_NAMES_ARE_UNIQUE,
)

#: 相等判据的浮点容差（逐位 / 整数相等时不走它）.
EQUALITY_TOLERANCE = 1e-12

#: 未解析基类数的上界（一个都不许有）.
UNRESOLVED_CEILING = 0.0

#: 每个族至少要有几个基类（**下界**）.
PARENT_FLOOR = 1.0

#: 最大继承深度的下界（链至少要存在）.
DEPTH_FLOOR = 1.0


@dataclass(frozen=True)
class CrossCheck:
    """一次对账：来源、读数、判据（**含方向**：相等 / 上界 / 下界）."""

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
            raise ParameterError(f"对账 {self.name!r} 同时给了上界与下界：请只给一个方向。")
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

    def to_dict(self) -> dict[str, Any]:
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
                f"性质 {self.name!r} 标了'不适用'却又标了'通过'：两者必须分开。"
            )
        if self.cross_check is not None and self.cross_check.passed != self.passed:
            raise NumericError(
                f"性质 {self.name!r} 的结论与它挂的对账不一致："
                f"outcome.passed={self.passed}、cross_check.passed={self.cross_check.passed}。"
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

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "name": self.name,
            "criterion": self.criterion,
            "applicable": self.applicable,
            "passed": self.passed,
            "evidence": list(self.evidence),
            "cross_check": None if self.cross_check is None else self.cross_check.to_dict(),
        }


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
        """不通过时抛 :class:`errors.ScanError`（"拒绝交付"的那条路）."""
        if self.ok:
            return
        from smart_research_agent.failure_ledger.errors import ScanError

        failures = [outcome.line() for outcome in self.applicable if not outcome.passed]
        raise ScanError("性质检查未全部通过：" + "；".join(failures))

    def lines(self) -> tuple[str, ...]:
        """逐行文本（**先印不适用**）."""
        return tuple(
            outcome.line()
            for outcome in sorted(self.outcomes, key=lambda outcome: outcome.applicable)
        )

    def to_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 化的字段."""
        return {
            "ok": self.ok,
            "counts": {"total": len(self.outcomes), "applicable": len(self.applicable)},
            "lines": list(self.lines()),
        }


# --------------------------------------------------------------------------- #
# 七条性质
# --------------------------------------------------------------------------- #


def _resolve(ledger: Ledger | None) -> Ledger:
    """缺省时现场编一份台账."""
    return build_ledger() if ledger is None else ledger


def check_ledger_covers_all_error_modules(
    ledger: Ledger | None = None,
    *,
    expected: tuple[str, ...] | None = None,
) -> PropertyOutcome:
    """① 每一份 ``errors.py`` 都进了台账."""
    resolved = _resolve(ledger)
    expected_packages = expected_modules() if expected is None else expected
    present = set(resolved.modules)
    missing = tuple(name for name in expected_packages if name not in present)
    check = CrossCheck(
        name="模块缺失数",
        left="Ledger.modules",
        right="扫描到的 errors.py 包名",
        reading=float(len(missing)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_LEDGER_COVERS_ALL_ERROR_MODULES,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"模块 {resolved.module_count} 份 / 期望 {len(expected_packages)} 份 | 缺失 {len(missing)}",
            "缺失的：" + ("、".join(missing) if missing else "无"),
        ),
        cross_check=check,
    )


def check_qualified_names_are_unique(
    families: tuple[Family, ...] | None = None,
    *,
    ledger: Ledger | None = None,
) -> PropertyOutcome:
    """② ``包.名`` 在台账里唯一（裸名字允许跨包重名）."""
    resolved = _resolve(ledger).families if families is None else families
    qualified = [family.qualified for family in resolved]
    duplicated = sorted({name for name in qualified if qualified.count(name) > 1})
    check = CrossCheck(
        name="重复的 包.名 数",
        left="Ledger.families",
        right="唯一定义",
        reading=float(len(duplicated)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_QUALIFIED_NAMES_ARE_UNIQUE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"族 {len(resolved)} 个 | 不同的 包.名 {len(set(qualified))} 个",
            "重复的：" + ("、".join(duplicated) if duplicated else "无"),
        ),
        cross_check=check,
    )


def check_ledger_is_reproducible(
    first: Ledger | None = None,
    second: Ledger | None = None,
    *,
    modules: tuple[ScannedModule, ...] | None = None,
) -> PropertyOutcome:
    """③ 同一批文件两次构建台账**逐位相同**."""
    left = build_ledger(modules) if first is None else first
    right = build_ledger(modules) if second is None else second
    diff = left.diff_count(right)
    check = CrossCheck(
        name="两次构建的差异项数",
        left="build_ledger 第一次",
        right="build_ledger 第二次",
        reading=float(diff),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_LEDGER_IS_REPRODUCIBLE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"差异项数 {diff}（0 = 逐位相同）",
            f"摘要 {left.digest()} / {right.digest()}",
        ),
        cross_check=check,
    )


def check_every_module_has_exactly_one_root(ledger: Ledger | None = None) -> PropertyOutcome:
    """④ 每一份 ``errors.py`` 恰好有一个根族."""
    resolved = _resolve(ledger)
    bad = 0
    detail: list[str] = []
    for package in resolved.modules:
        roots = tuple(family for family in resolved.by_package(package) if family.is_root)
        if len(roots) != 1:
            bad += 1
            detail.append(f"{package}({len(roots)})")
    check = CrossCheck(
        name="根数不是 1 的模块数",
        left="Ledger.roots 按包分组",
        right="每包恰好 1 个根",
        reading=float(bad),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_EVERY_MODULE_HAS_EXACTLY_ONE_ROOT,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"模块 {resolved.module_count} 份 | 根 {len(resolved.roots)} 个 | 异常模块 {bad} 个",
            "异常模块：" + ("、".join(detail) if detail else "无"),
        ),
        cross_check=check,
    )


def check_base_references_resolve(ledger: Ledger | None = None) -> PropertyOutcome:
    """⑤ **上界**：每一个基类名都能归属到 local / builtin / imported."""
    resolved = _resolve(ledger)
    unresolved = resolved.unresolved_bases()
    check = CrossCheck(
        name="未解析的基类名数",
        left="Family.bases 里 kind=unresolved",
        right=f"上界 {UNRESOLVED_CEILING:.0f}",
        reading=float(len(unresolved)),
        expected=UNRESOLVED_CEILING,
        exact=False,
        upper_bound=UNRESOLVED_CEILING,
    )
    return PropertyOutcome(
        name=PROPERTY_BASE_REFERENCES_RESOLVE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"未解析 {len(unresolved)} 个 ≤ 上界 {UNRESOLVED_CEILING:.0f}",
            "未解析的：" + ("、".join(sorted(set(unresolved))) if unresolved else "无"),
        ),
        cross_check=check,
    )


def check_every_family_has_a_parent(ledger: Ledger | None = None) -> PropertyOutcome:
    """⑥ **下界**：每一个族至少有 1 个基类."""
    resolved = _resolve(ledger)
    fewest = float(resolved.min_bases())
    check = CrossCheck(
        name="最小的基类个数",
        left="Family.bases 的长度",
        right=f"下界 {PARENT_FLOOR:.0f}",
        reading=fewest,
        expected=PARENT_FLOOR,
        exact=False,
        lower_bound=PARENT_FLOOR,
    )
    return PropertyOutcome(
        name=PROPERTY_EVERY_FAMILY_HAS_A_PARENT,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"最小基类数 {fewest:.0f} ≥ 下界 {PARENT_FLOOR:.0f}",
            resolved.line(),
        ),
        cross_check=check,
    )


def check_inheritance_depth_is_positive(ledger: Ledger | None = None) -> PropertyOutcome:
    """⑦ **下界**：本模块内部真的存在一条继承链."""
    resolved = _resolve(ledger)
    deepest = float(resolved.max_depth())
    check = CrossCheck(
        name="最大继承深度",
        left="Ledger._depths()",
        right=f"下界 {DEPTH_FLOOR:.0f}",
        reading=deepest,
        expected=DEPTH_FLOOR,
        exact=False,
        lower_bound=DEPTH_FLOOR,
    )
    return PropertyOutcome(
        name=PROPERTY_INHERITANCE_DEPTH_IS_POSITIVE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"最大深度 {deepest:.0f} ≥ 下界 {DEPTH_FLOOR:.0f}",
            f"派生族 {len(resolved.subs)} 个 / 根族 {len(resolved.roots)} 个",
        ),
        cross_check=check,
    )


def check_all(
    *,
    ledger: Ledger | None = None,
    modules: tuple[ScannedModule, ...] | None = None,
    expected: tuple[str, ...] | None = None,
) -> PropertyReport:
    """一次跑完七条性质（顺序与 :data:`types.FAILURE_LEDGER_PROPERTIES` 一致）.

    七条性质**共用同一份台账**，因此报告读的是同一时刻的状态。
    """
    from smart_research_agent.failure_ledger.errors import LedgerError

    resolved = build_ledger(modules) if ledger is None else ledger
    outcomes = (
        check_ledger_covers_all_error_modules(resolved, expected=expected),
        check_qualified_names_are_unique(resolved.families),
        check_ledger_is_reproducible(resolved, resolved),
        check_every_module_has_exactly_one_root(resolved),
        check_base_references_resolve(resolved),
        check_every_family_has_a_parent(resolved),
        check_inheritance_depth_is_positive(resolved),
    )
    names = {outcome.name for outcome in outcomes}
    extra = names - set(FAILURE_LEDGER_PROPERTIES)
    missing = set(FAILURE_LEDGER_PROPERTIES) - names
    if extra or missing:  # pragma: no cover - 只在有人改性质名单时触发
        raise LedgerError(
            "性质名单与 types.FAILURE_LEDGER_PROPERTIES 不一致："
            f"多 {sorted(extra)}、缺 {sorted(missing)}。"
        )
    return PropertyReport(outcomes=outcomes)


def require_ok(report: PropertyReport | None = None) -> PropertyReport:
    """七条性质没全部通过时抛 :class:`ScanError`（"拒绝交付"的那条路）."""
    resolved = check_all() if report is None else report
    if not resolved.ok:
        resolved.require_ok()
    return resolved


__all__ = [
    "DEPTH_FLOOR",
    "EQUALITY_TOLERANCE",
    "PARENT_FLOOR",
    "UNRESOLVED_CEILING",
    "CrossCheck",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_base_references_resolve",
    "check_every_family_has_a_parent",
    "check_every_module_has_exactly_one_root",
    "check_inheritance_depth_is_positive",
    "check_ledger_covers_all_error_modules",
    "check_ledger_is_reproducible",
    "check_qualified_names_are_unique",
    "require_ok",
]
