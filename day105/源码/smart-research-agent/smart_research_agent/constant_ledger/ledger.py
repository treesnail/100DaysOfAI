"""``ledger``：把"谁把哪个名字钉成了什么值"编成一份**可复算**的台账（day105）.

台账只有三件事：

```text
编台账   每个模块一行：它定义了哪些模块级常量、各是什么取值形态
归同类   把**同名**的常量收到一组，判定它们的关系：unique / consistent / conflict / incomparable
摆不同   把"同名不同值"整张摆出来（这是本课最值钱的读数）
```

## 一、今天最值钱的一句话

> **"同名不同值"本身不是错误，它是一个读数——
> 但它是这一课唯一一条能让人立刻去翻两个包的读数。**

因此本课**不**写一条"同名必须同值"的性质去把它处理掉：只有调用点显式要求一致时，
:func:`require_consistent` 才把它变成一次拒绝交付（:class:`ConflictError`）。

## 二、一条纪律：**四类关系里，"冲突"与"无法比较"必须分开**

```text
unique          只在一个模块里出现
consistent      跨模块，且能比较的那些字面值全都相同
conflict        跨模块，且能比较的那些字面值不止一个
incomparable    跨模块，但至少一侧是 opaque ⇒ 这一课**拒绝下结论**
```

把 `incomparable` 并进 `conflict`，会让"我算不出来"被读成"它们对不上"——
一个没人说得清归属的名字，比一个报错更难被发现（day102 / day104 说过两次了）。

## 三、一条纪律：**并列**要有一个确定的顺序

```text
模块按名字排序、每个模块的常量按（名字, 行号）排序、同名组按（-出现模块数, 名字）排序
comparable() 里没有任何集合 / 字典 / 绝对路径：因此 repr 稳定、摘要才有意义
```

## 四、与既有包的接缝

- **上游**：:mod:`constant_ledger.parse`（模块级的常量定义与取值指纹）；
- **下游**：:mod:`constant_ledger.verify` 在它上面检查七条性质，
  :mod:`constant_ledger.study` 打印台账的表。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from smart_research_agent.constant_ledger.errors import ConflictError, LedgerBuildError, ParameterError
from smart_research_agent.constant_ledger.parse import ConstantDef, ModuleConstants, scan_all
from smart_research_agent.constant_ledger.types import (
    PACKAGE_NAME,
    RELATION_CONFLICT,
    RELATION_CONSISTENT,
    RELATION_DESCRIPTIONS,
    RELATION_INCOMPARABLE,
    RELATION_UNIQUE,
    VALUE_LITERAL,
    VALUE_OPAQUE,
    require_relation,
)

#: 台账摘要取前多少位十六进制（与 day099 ~ day104 同一口径）.
LEDGER_DIGEST_LENGTH = 16


def classify(members: tuple[ConstantDef, ...]) -> str:
    """一组同名常量的关系（判定顺序：unique → incomparable → conflict/consistent）.

    ```text
    只有一个模块        ⇒ unique
    有模块的取值读不出来 ⇒ incomparable（**拒绝下结论**）
    能比较且只有一个值   ⇒ consistent
    能比较且有多个值     ⇒ conflict
    ```
    """
    modules = {item.module for item in members}
    if not members:
        raise ParameterError("一组同名常量不能是空的。")
    if len(modules) < 2:
        return RELATION_UNIQUE
    if any(item.value_kind == VALUE_OPAQUE for item in members):
        return RELATION_INCOMPARABLE
    values = {item.literal for item in members}
    return RELATION_CONSISTENT if len(values) == 1 else RELATION_CONFLICT


@dataclass(frozen=True)
class NameGroup:
    """一组**同名**常量：名字 + 关系 + 成员（按（模块, 名字）排序）."""

    name: str
    relation: str
    members: tuple[ConstantDef, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ParameterError("同名组的名字不能为空。")
        require_relation(self.relation)
        if not self.members:
            raise ParameterError(f"同名组 {self.name!r} 不能没有成员。")
        if {item.name for item in self.members} != {self.name}:
            raise ParameterError(f"同名组 {self.name!r} 里混进了别的名字。")
        expected = classify(self.members)
        if expected != self.relation:
            raise LedgerBuildError(
                f"同名组 {self.name!r} 的关系标成了 {self.relation!r}，"
                f"按成员算出来应当是 {expected!r}——两者必须一致。"
            )

    @property
    def module_names(self) -> tuple[str, ...]:
        """成员所属模块（排序、去重）."""
        return tuple(sorted({item.module for item in self.members}))

    @property
    def is_conflict(self) -> bool:
        """这一组是不是"同名不同值"."""
        return self.relation == RELATION_CONFLICT

    def literals(self) -> tuple[str, ...]:
        """能比较的那些字面值（排序、去重；没有就是空）."""
        return tuple(sorted({item.literal for item in self.members if item.literal is not None}))

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "name": self.name,
            "relation": self.relation,
            "modules": list(self.module_names),
            "literals": list(self.literals()),
        }

    def line(self) -> str:
        """``EPSILON | conflict | 2 个模块 | 取值 ['1e-05', '1e-08']``."""
        values = "、".join(self.literals()) or "（读不出来）"
        return (
            f"{self.name:<32} | {self.relation:<12} | {len(self.module_names):>3} 个模块"
            f" | 取值 [{values}]"
        )


@dataclass(frozen=True)
class ConstantLedger:
    """一份常量台账：模块行（排序唯一）+ 由它派生的一组只读视图.

    ``entries`` 里的每个 :class:`~constant_ledger.parse.ModuleConstants`
    都是同一时刻解析出来的——因此整份台账读的是一个一致的状态。
    """

    entries: tuple[ModuleConstants, ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise LedgerBuildError("台账不能没有模块行。")
        names = [entry.module for entry in self.entries]
        if names != sorted(names) or len(set(names)) != len(names):
            raise LedgerBuildError(f"模块行必须是一份排序且不重复的名单：{len(names)} 行。")

    # ------------------------------------------------------------------ 只读视图

    @property
    def modules(self) -> tuple[str, ...]:
        """全部模块名（排序）."""
        return tuple(entry.module for entry in self.entries)

    @property
    def module_set(self) -> frozenset[str]:
        """模块名集合（查表用）."""
        return frozenset(self.modules)

    @property
    def module_count(self) -> int:
        """模块数（= 行数）."""
        return len(self.entries)

    def entry_of(self, module: str) -> ModuleConstants:
        """取一个模块的行（不在台账里当场拒绝）."""
        for entry in self.entries:
            if entry.module == module:
                return entry
        raise ParameterError(f"台账里没有模块 {module!r}。")

    def package_of(self, module: str) -> str:
        """一个模块的顶层子包名（不在台账里当场拒绝）."""
        self.entry_of(module)
        parts = module.split(".")
        return parts[1] if len(parts) > 1 else module

    def package_groups(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """按顶层子包分组（组名排序，组内成员排序）."""
        groups: dict[str, list[str]] = {}
        for entry in self.entries:
            parts = entry.module.split(".")
            key = parts[1] if len(parts) > 1 else entry.module
            groups.setdefault(key, []).append(entry.module)
        return tuple((name, tuple(sorted(groups[name]))) for name in sorted(groups))

    def pairs(self) -> tuple[ConstantDef, ...]:
        """每个 `(模块, 常量名)` 一条**主定义**（顺序 = 模块名升序 × 名字升序）."""
        return tuple(
            item for entry in self.entries for item in entry.primary_defs()
        )

    def constant_count(self) -> int:
        """台账里的 `(模块, 常量名)` 条数（= 每一行的名字数之和）."""
        return sum(len(entry.primary_defs()) for entry in self.entries)

    def assignment_count(self) -> int:
        """常量赋值语句总数（重复赋值也算，因此 >= :meth:`constant_count`）."""
        return sum(len(entry.constants) for entry in self.entries)

    def literal_count(self) -> int:
        """能求出取值的那些条数."""
        return sum(1 for item in self.pairs() if item.value_kind == VALUE_LITERAL)

    def opaque_count(self) -> int:
        """读不出取值的那些条数."""
        return sum(1 for item in self.pairs() if item.value_kind == VALUE_OPAQUE)

    def value_kind_counts(self) -> tuple[tuple[str, int], ...]:
        """两类取值形态各有几条（按 :data:`types.VALUE_KINDS` 的顺序）."""
        return (
            (VALUE_LITERAL, self.literal_count()),
            (VALUE_OPAQUE, self.opaque_count()),
        )

    # ------------------------------------------------------------------ 同名组

    def name_groups(self) -> tuple[NameGroup, ...]:
        """全部同名组（按 "出现模块数降序 × 名字升序" 排）."""
        table: dict[str, list[ConstantDef]] = {}
        for item in self.pairs():
            table.setdefault(item.name, []).append(item)
        groups = [
            NameGroup(
                name=name,
                relation=classify(tuple(members)),
                members=tuple(sorted(members, key=lambda item: (item.module, item.name))),
            )
            for name, members in table.items()
        ]
        return tuple(sorted(groups, key=lambda group: (-len(group.module_names), group.name)))

    def shared_groups(self) -> tuple[NameGroup, ...]:
        """出现在**两个以上**模块里的同名组（按 :meth:`name_groups` 的顺序）."""
        return tuple(group for group in self.name_groups() if len(group.module_names) >= 2)

    def conflicts(self) -> tuple[NameGroup, ...]:
        """**同名不同值**的那些组（按 :meth:`name_groups` 的顺序）."""
        return tuple(group for group in self.name_groups() if group.relation == RELATION_CONFLICT)

    def incomparables(self) -> tuple[NameGroup, ...]:
        """跨模块但至少一侧读不出来的那些组."""
        return tuple(
            group for group in self.name_groups() if group.relation == RELATION_INCOMPARABLE
        )

    def duplicate_modules(self) -> tuple[str, ...]:
        """把同一个常量名赋了两次以上的模块（排序）."""
        return tuple(sorted(entry.module for entry in self.entries if entry.duplicate_names()))

    # ------------------------------------------------------------------ 复算口径

    def comparable(self) -> tuple[Any, ...]:
        """**只含确定性字段**的元组（两份台账逐位比较它；不含路径、不含集合）."""
        return (
            self.modules,
            tuple(
                (
                    entry.module,
                    tuple(
                        (item.name, item.line, item.annotated, item.value_kind, item.literal)
                        for item in entry.constants
                    ),
                )
                for entry in self.entries
            ),
        )

    def digest(self) -> str:
        """台账摘要（两次构建摘要相同 ⇔ 逐位相同）."""
        return hashlib.sha256(repr(self.comparable()).encode("utf-8")).hexdigest()[
            :LEDGER_DIGEST_LENGTH
        ]

    def diff_count(self, other: ConstantLedger) -> int:
        """与另一份台账在复算口径上**差了几项**（0 = 逐位相同）."""
        left = self.comparable()
        right = other.comparable()
        return sum(1 for a, b in zip(left, right, strict=True) if a != b)

    def to_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 化的字段."""
        return {
            "modules": self.module_count,
            "packages": len(self.package_groups()),
            "constants": self.constant_count(),
            "literals": self.literal_count(),
            "opaques": self.opaque_count(),
            "shared": len(self.shared_groups()),
            "conflicts": len(self.conflicts()),
            "digest": self.digest(),
        }

    def line(self) -> str:
        """一行读数：``台账：464 个模块 | 常量 N | 可取值 M | 同名 S | 冲突 C | 摘要 xxxx``."""
        return (
            f"台账：{self.module_count} 个模块 | 常量 {self.constant_count()}"
            f" | 可取值 {self.literal_count()} | 读不出 {self.opaque_count()}"
            f" | 同名 {len(self.shared_groups())} | 冲突 {len(self.conflicts())}"
            f" | 摘要 {self.digest()}"
        )


def build_ledger(modules: tuple[ModuleConstants, ...] | None = None) -> ConstantLedger:
    """把解析结果编成一份 :class:`ConstantLedger`（**确定性**）.

    ``modules`` 缺省时现场扫一遍；测试可以注入一组小模块。
    """
    resolved = scan_all() if modules is None else modules
    return ConstantLedger(entries=tuple(sorted(resolved, key=lambda entry: entry.module)))


def ledger_digest(ledger: ConstantLedger | None = None) -> str:
    """台账摘要（缺省时现场编一份）."""
    resolved = build_ledger() if ledger is None else ledger
    return resolved.digest()


def require_reproducible(first: ConstantLedger, second: ConstantLedger) -> ConstantLedger:
    """两份台账不一致时抛 :class:`LedgerBuildError`（"拒绝交付"的那条路）."""
    diff = first.diff_count(second)
    if diff == 0:
        return first
    raise LedgerBuildError(
        f"同一批文件编出的两份台账差了 {diff} 项——"
        "台账里混进了未固定的量（集合序 / 字典序 / 绝对路径），它还不是一份可以被别人重算的中间物。"
    )


def require_consistent(ledger: ConstantLedger | None = None) -> ConstantLedger:
    """调用点**显式**要求"同名必须同值"时用：有冲突就抛 :class:`ConflictError`.

    这是本包唯一把"同名不同值"当成失败的入口——它是一次**调用点的决定**，
    不是本包的默认行为（见模块 docstring 第一节）。
    """
    resolved = build_ledger() if ledger is None else ledger
    conflicts = resolved.conflicts()
    if conflicts:
        rendered = "、".join(group.name for group in conflicts[:5])
        raise ConflictError(
            f"有 {len(conflicts)} 组同名常量取值不一致：{rendered}——"
            "要么把它们统一成同一个值，要么别要求它们一致。"
        )
    return resolved


def expected_packages(ledger: ConstantLedger | None = None) -> tuple[str, ...]:
    """读取一份台账的顶层子包名单（排序）."""
    resolved = build_ledger() if ledger is None else ledger
    return tuple(name for name, _members in resolved.package_groups())


def ledger_lines(ledger: ConstantLedger | None = None, *, limit: int | None = None) -> tuple[str, ...]:
    """把台账逐行印出来（可选截断）."""
    from smart_research_agent.constant_ledger.types import require_positive_int

    resolved = build_ledger() if ledger is None else ledger
    chosen = (
        resolved.entries if limit is None else resolved.entries[: require_positive_int("limit", limit)]
    )
    lines = ["常量台账（模块 | 常量数 | 可取值 | 读不出 | 重名）："]
    lines.extend("  " + entry.line() for entry in chosen)
    lines.append("  " + resolved.line())
    return tuple(lines)


def docs_of_relation(relation: str) -> str:
    """返回一类同名关系的一句话解释（未知类别当场拒绝）."""
    return RELATION_DESCRIPTIONS[require_relation(relation)]


def root_package() -> str:
    """被扫描的包名（台账里所有模块的前缀）."""
    return PACKAGE_NAME


__all__ = [
    "LEDGER_DIGEST_LENGTH",
    "ConstantLedger",
    "NameGroup",
    "build_ledger",
    "classify",
    "docs_of_relation",
    "expected_packages",
    "ledger_digest",
    "ledger_lines",
    "require_consistent",
    "require_reproducible",
    "root_package",
]
