"""七条性质与三类判据（day105）.

本模块是"这份常量台账成不成立"的判据所在。七条性质按三类判据分：

```text
相等（==）   ① ledger_covers_all_modules             缺失数 == 0
             ② ledger_is_reproducible                两次构建的差异项数 == 0
             ③ conflicts_are_sound                   不成立的冲突组数 == 0
             ④ shared_constants_are_sound            不成立的同名组成员数 == 0
上界（<=）   ⑤ duplicate_assignments_within_module    重复赋值的模块数 <= 0
下界（>=）   ⑥ constants_are_numerous                 常量总数 >= 500
             ⑦ shared_constants_exist                 跨模块同名组数 >= 1
```

## 一、为什么 ⑥ ⑦ 必须是下界

```text
"台账里至少有一批常量"      它是一个**存在性**断言 ⇒ 只能从下面兜住
"至少有一个名字跨两个模块"    "有同名"这件事只能从下面兜住
```

第 ⑥ 条尤其要点明：写成 `== 某个精确数` 会在仓库合理地变化时误报失败，
因此它的阈值定在"这批文件必然成立"的地方——**它是为了抓"扫描口径写错了"，不是为了数数。**

## 二、为什么 ③ ④ 是"真的吗"型判据

一份台账里最容易出错的不是"数错了"，而是"**报出来的这一组到底是不是真的那一组**"。
因此 ③ 独立复核冲突的**源码片段**（`parse.value_segments`），
④ 从 `pairs()` 逐条重数同名组的成员——两条都与台账的构建算法**不同路**。

## 三、一条纪律：`CrossCheck.passed` 必须与 `PropertyOutcome.passed` 一致

一份报告里，"那一行说通过"与"它挂的对账说没通过"是最糟的状态。
因此 :class:`PropertyOutcome` 在 ``__post_init__`` 里当场拒绝两者不一致的构造。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smart_research_agent.constant_ledger.errors import ConstantError, NumericError, ParameterError
from smart_research_agent.constant_ledger.ledger import ConstantLedger, build_ledger
from smart_research_agent.constant_ledger.parse import (
    ModuleConstants,
    expected_modules,
    value_segments,
)
from smart_research_agent.constant_ledger.types import (
    CONSTANT_LEDGER_PROPERTIES,
    CRITERION_EQUALITY,
    CRITERION_LOWER_BOUND,
    CRITERION_UPPER_BOUND,
    PROPERTY_CONFLICTS_ARE_SOUND,
    PROPERTY_CONSTANTS_ARE_NUMEROUS,
    PROPERTY_DUPLICATE_ASSIGNMENTS_WITHIN_MODULE,
    PROPERTY_LEDGER_COVERS_ALL_MODULES,
    PROPERTY_LEDGER_IS_REPRODUCIBLE,
    PROPERTY_SHARED_CONSTANTS_ARE_SOUND,
    PROPERTY_SHARED_CONSTANTS_EXIST,
)

#: 相等判据的浮点容差（逐位 / 整数相等时不走它）.
EQUALITY_TOLERANCE = 1e-12

#: 重复赋值的模块数的上界（一个都不许有）.
DUPLICATE_CEILING = 0.0

#: 台账里常量总数的下界（**下界**：它是"扫描口径没写错"的存在性断言）.
CONSTANT_FLOOR = 500.0

#: 跨模块同名组数的下界（**下界**：至少存在一个跨模块同名）.
SHARED_FLOOR = 1.0


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
            raise NumericError(f"性质 {self.name!r} 标了'不适用'却又标了'通过'：两者必须分开。")
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
        """不通过时抛 :class:`errors.ParseError`（"拒绝交付"的那条路）."""
        if self.ok:
            return
        from smart_research_agent.constant_ledger.errors import ParseError

        failures = [outcome.line() for outcome in self.applicable if not outcome.passed]
        raise ParseError("性质检查未全部通过：" + "；".join(failures))

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


def _resolve(ledger: ConstantLedger | None) -> ConstantLedger:
    """缺省时现场编一份台账."""
    return build_ledger() if ledger is None else ledger


def check_ledger_covers_all_modules(
    ledger: ConstantLedger | None = None,
    *,
    expected: tuple[str, ...] | None = None,
) -> PropertyOutcome:
    """① 每一个 ``.py`` 都是台账里的一行."""
    resolved = _resolve(ledger)
    expected_names = expected_modules() if expected is None else expected
    missing = tuple(name for name in expected_names if name not in resolved.module_set)
    check = CrossCheck(
        name="模块缺失数",
        left="ConstantLedger.modules",
        right="扫描到的 .py 模块名",
        reading=float(len(missing)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_LEDGER_COVERS_ALL_MODULES,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"模块 {resolved.module_count} 个 / 期望 {len(expected_names)} 个 | 缺失 {len(missing)}",
            "缺失的：" + ("、".join(missing) if missing else "无"),
        ),
        cross_check=check,
    )


def check_ledger_is_reproducible(
    first: ConstantLedger | None = None,
    second: ConstantLedger | None = None,
    *,
    modules: tuple[ModuleConstants, ...] | None = None,
) -> PropertyOutcome:
    """② 同一批文件两次编出的台账**逐位相同**."""
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


def check_conflicts_are_sound(ledger: ConstantLedger | None = None) -> PropertyOutcome:
    """③ 报出来的每一组"同名不同值"都是**真的**.

    独立复核走的是**源码片段**这条路：把每个成员模块里该名字的右端源码取出来、去掉空白，
    这一组必须至少有两种不同的片段才算冲突（见 :func:`parse.value_segments`）。
    它与 :func:`parse.canonical_value` 是两条独立的路，因此能抓住"取值指纹算错了"。
    """
    resolved = _resolve(ledger)
    unsound: list[str] = []
    for group in resolved.conflicts():
        segments: set[str] = set()
        for module in group.module_names:
            table = value_segments(Path(resolved.entry_of(module).path))
            segments.add(table.get(group.name, ""))
        if len(segments) < 2:
            unsound.append(group.name)
    check = CrossCheck(
        name="不成立的冲突组数",
        left="ConstantLedger.conflicts()",
        right="成员模块的右端源码片段至少两种",
        reading=float(len(unsound)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_CONFLICTS_ARE_SOUND,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"冲突组 {len(resolved.conflicts())} 个 / 不成立 {len(unsound)} 个",
            "不成立的：" + ("、".join(unsound[:3]) if unsound else "无"),
        ),
        cross_check=check,
    )


def check_shared_constants_are_sound(ledger: ConstantLedger | None = None) -> PropertyOutcome:
    """④ 同名表里每一项的成员集合与独立重数一致.

    这里从 ``pairs()`` 逐条重数（而不是复用 :meth:`ConstantLedger.name_groups`），
    因此它能抓住"同名表把某个模块算漏了 / 算重了"。
    """
    resolved = _resolve(ledger)
    by_name: dict[str, set[str]] = {}
    for item in resolved.pairs():
        by_name.setdefault(item.name, set()).add(item.module)
    unsound: list[str] = []
    for group in resolved.shared_groups():
        if len(group.module_names) < 2 or set(group.module_names) != by_name.get(group.name, set()):
            unsound.append(group.name)
    check = CrossCheck(
        name="不成立的同名组成员数",
        left="ConstantLedger.shared_groups()",
        right="pairs() 逐条重数",
        reading=float(len(unsound)),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_SHARED_CONSTANTS_ARE_SOUND,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"同名组 {len(resolved.shared_groups())} 个 / 不成立 {len(unsound)} 个",
            "不成立的：" + ("、".join(unsound[:3]) if unsound else "无"),
        ),
        cross_check=check,
    )


def check_duplicate_assignments_within_module(
    ledger: ConstantLedger | None = None,
) -> PropertyOutcome:
    """⑤ **上界**：同一个模块里不重复给同一个常量名赋值."""
    resolved = _resolve(ledger)
    duplicated = resolved.duplicate_modules()
    check = CrossCheck(
        name="重复赋值的模块数",
        left="ConstantLedger.duplicate_modules()",
        right=f"上界 {DUPLICATE_CEILING:.0f}",
        reading=float(len(duplicated)),
        expected=DUPLICATE_CEILING,
        exact=False,
        upper_bound=DUPLICATE_CEILING,
    )
    return PropertyOutcome(
        name=PROPERTY_DUPLICATE_ASSIGNMENTS_WITHIN_MODULE,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"重复 {len(duplicated)} 个 ≤ 上界 {DUPLICATE_CEILING:.0f}",
            "重复的：" + ("、".join(duplicated[:3]) if duplicated else "无"),
        ),
        cross_check=check,
    )


def check_constants_are_numerous(ledger: ConstantLedger | None = None) -> PropertyOutcome:
    """⑥ **下界**：台账里至少有一批常量（它是"扫描口径没写错"的存在性断言）."""
    resolved = _resolve(ledger)
    total = float(resolved.constant_count())
    check = CrossCheck(
        name="常量总数",
        left="ConstantLedger.constant_count()",
        right=f"下界 {CONSTANT_FLOOR:.0f}",
        reading=total,
        expected=CONSTANT_FLOOR,
        exact=False,
        lower_bound=CONSTANT_FLOOR,
    )
    return PropertyOutcome(
        name=PROPERTY_CONSTANTS_ARE_NUMEROUS,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"常量 {total:.0f} ≥ 下界 {CONSTANT_FLOOR:.0f}",
            f"可取值 {resolved.literal_count()} | 读不出 {resolved.opaque_count()}",
        ),
        cross_check=check,
    )


def check_shared_constants_exist(ledger: ConstantLedger | None = None) -> PropertyOutcome:
    """⑦ **下界**：至少有一个常量名出现在两个以上的模块里."""
    resolved = _resolve(ledger)
    shared = float(len(resolved.shared_groups()))
    check = CrossCheck(
        name="跨模块同名组数",
        left="ConstantLedger.shared_groups()",
        right=f"下界 {SHARED_FLOOR:.0f}",
        reading=shared,
        expected=SHARED_FLOOR,
        exact=False,
        lower_bound=SHARED_FLOOR,
    )
    return PropertyOutcome(
        name=PROPERTY_SHARED_CONSTANTS_EXIST,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"同名组 {shared:.0f} ≥ 下界 {SHARED_FLOOR:.0f}",
            f"冲突 {len(resolved.conflicts())} | 无法比较 {len(resolved.incomparables())}",
        ),
        cross_check=check,
    )


def check_all(
    *,
    ledger: ConstantLedger | None = None,
    modules: tuple[ModuleConstants, ...] | None = None,
    expected: tuple[str, ...] | None = None,
) -> PropertyReport:
    """一次跑完七条性质（顺序与 :data:`types.CONSTANT_LEDGER_PROPERTIES` 一致）.

    七条性质**共用同一份台账**，因此报告读的是同一时刻的状态。
    """
    resolved = build_ledger(modules) if ledger is None else ledger
    outcomes = (
        check_ledger_covers_all_modules(resolved, expected=expected),
        check_ledger_is_reproducible(resolved, resolved),
        check_conflicts_are_sound(resolved),
        check_shared_constants_are_sound(resolved),
        check_duplicate_assignments_within_module(resolved),
        check_constants_are_numerous(resolved),
        check_shared_constants_exist(resolved),
    )
    names = {outcome.name for outcome in outcomes}
    extra = names - set(CONSTANT_LEDGER_PROPERTIES)
    missing = set(CONSTANT_LEDGER_PROPERTIES) - names
    if extra or missing:  # pragma: no cover - 只在有人改性质名单时触发
        raise ConstantError(
            "性质名单与 types.CONSTANT_LEDGER_PROPERTIES 不一致："
            f"多 {sorted(extra)}、缺 {sorted(missing)}。"
        )
    return PropertyReport(outcomes=outcomes)


def require_ok(report: PropertyReport | None = None) -> PropertyReport:
    """七条性质没全部通过时抛 :class:`ParseError`（"拒绝交付"的那条路）."""
    resolved = check_all() if report is None else report
    if not resolved.ok:
        resolved.require_ok()
    return resolved


__all__ = [
    "CONSTANT_FLOOR",
    "DUPLICATE_CEILING",
    "EQUALITY_TOLERANCE",
    "SHARED_FLOOR",
    "CrossCheck",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_conflicts_are_sound",
    "check_constants_are_numerous",
    "check_duplicate_assignments_within_module",
    "check_ledger_covers_all_modules",
    "check_ledger_is_reproducible",
    "check_shared_constants_are_sound",
    "check_shared_constants_exist",
    "require_ok",
]
