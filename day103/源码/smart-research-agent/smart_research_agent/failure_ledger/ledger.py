"""``ledger``：把原始族编成一份**跨包的台账**（day102）.

台账只有三件事：

```text
编族   每个 ClassDef → 一个 :class:`Family`（名字 + 归属好的基类 + 首行自述）
定身份 没有 local 基类的是 root（根），否则是 sub（派生）
复算   同一批文件两次构建，台账的摘要必须逐位相同
```

## 一、今天最值钱的一句话

> **族的裸名字允许跨包重名，但 ``包.名`` 必须唯一——
> 这正是 day101 给文档名加种类前缀的同一条纪律。**

本仓库里 ``ShapeError`` / ``ParameterError`` / ``NumericError`` 各出现十几次；
把它们当成"重名"去消歧是错的，把它们当成"同一件事在每一层的同一个名字"才是对的。
因此 :class:`Ledger` 的唯一性检查只针对 ``包.名``。

## 二、一条纪律：**并列**要有一个确定的顺序

```text
families 的顺序 = 扫描顺序（包名升序）× 源码顺序
comparable() 里没有任何集合 / 字典：因此 repr 稳定、摘要才有意义
```

与 day101 的倒排表同源：把"集合 → 序列"的地方显式排序，
"两次构建逐位相同"就从"希望如此"变成"不可能不如此"。

## 三、与既有包的接缝

- **上游**：:mod:`failure_ledger.scan`（原始族）、:mod:`failure_ledger.resolve`（归属）、
  :mod:`failure_ledger.types`（两种身份）；
- **下游**：:mod:`failure_ledger.verify` 在它上面检查七条性质，
  :mod:`failure_ledger.study` 打印台账表。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from smart_research_agent.failure_ledger.errors import CoverageError, LedgerBuildError, ParameterError
from smart_research_agent.failure_ledger.resolve import BaseRef, resolve_family
from smart_research_agent.failure_ledger.scan import ScannedModule, expected_modules, scan_all
from smart_research_agent.failure_ledger.types import FAMILY_KIND_ROOT, FAMILY_KIND_SUB

#: 台账摘要取前多少位十六进制（与 day099 / day100 / day101 同一口径）.
LEDGER_DIGEST_LENGTH = 16


@dataclass(frozen=True)
class Family:
    """台账里的一个族：包名 + 名字 + 归属好的基类 + 首行自述."""

    package: str
    name: str
    bases: tuple[BaseRef, ...]
    doc_first_line: str = ""

    def __post_init__(self) -> None:
        if not self.package or not self.name:
            raise ParameterError("族的包名与名字都不能为空。")

    @property
    def qualified(self) -> str:
        """``包.名``（台账里的唯一键）."""
        return f"{self.package}.{self.name}"

    @property
    def local_parents(self) -> tuple[str, ...]:
        """本模块内部的父族名（顺序即源码顺序）."""
        return tuple(ref.target for ref in self.bases if ref.is_local)

    @property
    def is_root(self) -> bool:
        """没有 local 基类 ⇒ 根."""
        return not self.local_parents

    @property
    def kind(self) -> str:
        """两种身份之一（root / sub）."""
        return FAMILY_KIND_ROOT if self.is_root else FAMILY_KIND_SUB

    @property
    def cross_package_bases(self) -> tuple[BaseRef, ...]:
        """来自别的 ``errors.py`` 的基类（跨包契约）."""
        return tuple(ref for ref in self.bases if ref.kind == "imported")

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "package": self.package,
            "name": self.name,
            "qualified": self.qualified,
            "kind": self.kind,
            "bases": [ref.raw for ref in self.bases],
            "doc": self.doc_first_line,
        }

    def line(self) -> str:
        """一行读数：``transformer_stack.ShapeError | sub | CoreShapeError、StackError``."""
        rendered = "、".join(ref.raw for ref in self.bases) or "（无基类）"
        return f"{self.qualified:<34} | {self.kind:<4} | {rendered}"


@dataclass(frozen=True)
class Ledger:
    """一份台账：模块名单 + 全部族（``包.名`` 唯一）."""

    modules: tuple[str, ...]
    families: tuple[Family, ...]

    def __post_init__(self) -> None:
        if not self.families:
            raise LedgerBuildError("台账不能为空：空台账会让后面每一条性质都'通过'于无形。")
        if len(set(self.modules)) != len(self.modules):
            raise LedgerBuildError(f"模块名单里有重复：{sorted(self.modules)}。")
        qualified = [family.qualified for family in self.families]
        duplicated = sorted({name for name in qualified if qualified.count(name) > 1})
        if duplicated:
            raise LedgerBuildError(
                f"台账里有重复的 ``包.名``：{duplicated}——"
                "两个族共用一个键，台账里就没法区分它们。"
            )

    # ------------------------------------------------------------------ 只读视图

    @property
    def module_count(self) -> int:
        """模块数（份数）."""
        return len(self.modules)

    @property
    def family_count(self) -> int:
        """族数."""
        return len(self.families)

    def by_package(self, package: str) -> tuple[Family, ...]:
        """取一个包的全部族（顺序即源码顺序）."""
        rows = tuple(family for family in self.families if family.package == package)
        if not rows:
            raise ParameterError(f"台账里没有包 {package!r}。")
        return rows

    @property
    def roots(self) -> tuple[Family, ...]:
        """全部根族."""
        return tuple(family for family in self.families if family.is_root)

    @property
    def subs(self) -> tuple[Family, ...]:
        """全部派生族."""
        return tuple(family for family in self.families if not family.is_root)

    def root_of(self, package: str) -> Family:
        """取一个包唯一的那一个根族（0 个或 2 个都当场拒绝）."""
        roots = tuple(family for family in self.by_package(package) if family.is_root)
        if len(roots) != 1:
            raise ParameterError(
                f"包 {package!r} 有 {len(roots)} 个根族（应当恰好 1 个）——"
                "两个根意味着读者要先猜'我该 except 哪一个'。"
            )
        return roots[0]

    def family_of(self, qualified: str) -> Family:
        """按 ``包.名`` 取一个族（不在台账里当场拒绝）."""
        for family in self.families:
            if family.qualified == qualified:
                return family
        raise ParameterError(f"台账里没有族 {qualified!r}。")

    def unresolved_bases(self) -> tuple[str, ...]:
        """全部未解析的基类名（顺序即出现顺序）."""
        return tuple(
            ref.raw
            for family in self.families
            for ref in family.bases
            if ref.kind == "unresolved"
        )

    # ------------------------------------------------------------------ 继承深度

    def _depths(self) -> tuple[tuple[str, int], ...]:
        """每族的继承深度（根为 0，子比父大 1；出现环时当场拒绝）."""
        by_package: dict[str, dict[str, Family]] = {}
        for family in self.families:
            by_package.setdefault(family.package, {})[family.name] = family
        cache: dict[str, int] = {}

        def depth(family: Family, stack: tuple[str, ...]) -> int:
            if family.qualified in cache:
                return cache[family.qualified]
            if family.qualified in stack:
                raise LedgerBuildError(
                    f"继承里出现环：{' → '.join((*stack, family.qualified))}——"
                    "Python 不允许继承成环，能让静态扫描看见环的只有源码本身。"
                )
            if family.is_root:
                value = 0
            else:
                siblings = by_package[family.package]
                parents: list[Family] = []
                for name in family.local_parents:
                    if name not in siblings:
                        raise LedgerBuildError(
                            f"族 {family.qualified!r} 的 local 基类 {name!r} 不在同一包里——"
                            "local 的语义是'同一模块内部'，找不到它说明这条归属是坏的。"
                        )
                    parents.append(siblings[name])
                value = 1 + max(depth(parent, (*stack, family.qualified)) for parent in parents)
            cache[family.qualified] = value
            return value

        return tuple((family.qualified, depth(family, ())) for family in self.families)

    def depth_of(self, qualified: str) -> int:
        """取一个族的继承深度（不在台账里当场拒绝）."""
        for name, value in self._depths():
            if name == qualified:
                return value
        raise ParameterError(f"台账里没有族 {qualified!r}。")

    def max_depth(self) -> int:
        """最大继承深度（性质 ⑦ 读它）."""
        return max(value for _, value in self._depths())

    def min_bases(self) -> int:
        """最小的基类个数（性质 ⑥ 读它）."""
        return min(len(family.bases) for family in self.families)

    # ------------------------------------------------------------------ 复算口径

    def comparable(self) -> tuple[Any, ...]:
        """**只含确定性字段**的元组（两份台账逐位比较它）."""
        return (
            self.modules,
            tuple(
                (
                    family.qualified,
                    family.kind,
                    tuple((ref.raw, ref.kind, ref.target) for ref in family.bases),
                )
                for family in self.families
            ),
        )

    def digest(self) -> str:
        """台账摘要（两次构建摘要相同 ⇔ 逐位相同）."""
        return hashlib.sha256(repr(self.comparable()).encode("utf-8")).hexdigest()[
            :LEDGER_DIGEST_LENGTH
        ]

    def diff_count(self, other: Ledger) -> int:
        """与另一份台账在复算口径上**差了几项**（0 = 逐位相同）."""
        left = self.comparable()
        right = other.comparable()
        return sum(1 for a, b in zip(left, right, strict=True) if a != b)

    def to_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 化的字段."""
        return {
            "modules": self.module_count,
            "families": self.family_count,
            "roots": len(self.roots),
            "subs": len(self.subs),
            "digest": self.digest(),
            "packages": list(self.modules),
        }

    def line(self) -> str:
        """一行读数：``台账：35 份模块 | 187 个族 | 35 个根 | 152 个派生 | 摘要 xxxx``."""
        return (
            f"台账：{self.module_count} 份模块 | {self.family_count} 个族"
            f" | {len(self.roots)} 个根 | {len(self.subs)} 个派生 | 摘要 {self.digest()}"
        )


def build_ledger(modules: tuple[ScannedModule, ...] | None = None) -> Ledger:
    """把扫描结果编成一份 :class:`Ledger`（**确定性**）.

    ``modules`` 缺省时现场扫一遍；测试可以注入一组小模块。
    """
    resolved = scan_all() if modules is None else modules
    families: list[Family] = []
    for module in resolved:
        for family in module.families:
            families.append(
                Family(
                    package=module.package,
                    name=family.name,
                    bases=resolve_family(family, module),
                    doc_first_line=family.doc_first_line,
                )
            )
    return Ledger(modules=tuple(module.package for module in resolved), families=tuple(families))


def ledger_digest(ledger: Ledger | None = None) -> str:
    """台账摘要（缺省时现场编一份）."""
    resolved = build_ledger() if ledger is None else ledger
    return resolved.digest()


def require_reproducible(first: Ledger, second: Ledger) -> Ledger:
    """两份台账不一致时抛 :class:`LedgerBuildError`（"拒绝交付"的那条路）."""
    diff = first.diff_count(second)
    if diff == 0:
        return first
    raise LedgerBuildError(
        f"同一批文件编出的两份台账差了 {diff} 项——"
        "台账里混进了未固定的量（集合序 / 字典序），它还不是一份可以被别人重算的中间物。"
    )


def require_coverage(ledger: Ledger | None = None, *, expected: tuple[str, ...] | None = None) -> Ledger:
    """有 ``errors.py`` 没进台账时抛 :class:`CoverageError`（"拒绝交付"的那条路）."""
    resolved = build_ledger() if ledger is None else ledger
    expected_packages = expected_modules() if expected is None else expected
    missing = tuple(name for name in expected_packages if name not in set(resolved.modules))
    if missing:
        raise CoverageError(
            f"台账漏了 {len(missing)} 份 errors.py：{list(missing)}——"
            "一份没进台账的文件，与'这个子包没有失败族'在报告里读起来一样。"
        )
    return resolved


def ledger_lines(ledger: Ledger | None = None, *, limit: int | None = None) -> tuple[str, ...]:
    """把台账逐行印出来（可选截断）."""
    from smart_research_agent.failure_ledger.types import require_positive_int

    resolved = build_ledger() if ledger is None else ledger
    chosen = resolved.families if limit is None else resolved.families[: require_positive_int("limit", limit)]
    lines = ["台账表（包.名 | 身份 | 基类）："]
    lines.extend("  " + family.line() for family in chosen)
    lines.append("  " + resolved.line())
    return tuple(lines)


__all__ = [
    "LEDGER_DIGEST_LENGTH",
    "Family",
    "Ledger",
    "build_ledger",
    "ledger_digest",
    "ledger_lines",
    "require_coverage",
    "require_reproducible",
]
