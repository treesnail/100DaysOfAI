"""``catalog``：把"承诺"编成一份**可复算**的清单（day104）.

清单只有三件事：

```text
编清单   每个模块一行：它的承诺来源、声明了什么、手里有多少名字
找重名   同一个名字被多少个模块导出（跨包契约的入口）
找幽灵   承诺了、却在本模块里找不到落点的名字
```

## 一、今天最值钱的一句话

> **一份清单的价值不在"它好不好看"，而在"同一个仓库能不能被编出同一份清单"。**

因此 :meth:`SymbolCatalog.comparable` 里**一个集合、一个字典、一个绝对路径都没有**：
只有排序过的名字序列。于是"两次构建逐位相同"从"希望如此"变成"不可能不如此"。

## 二、一条纪律：**并列**要有一个确定的顺序

```text
模块按名字排序、声明按名字排序、重名表按（-出现模块数, 名字）排序——三处显式排序
```

这与 day101 的倒排表、day102 的台账、day103 的图是同一条纪律的**第四次**应用。

## 三、与既有包的接缝

- **上游**：:mod:`symbol_catalog.parse`（模块行的三类绑定与承诺来源）；
- **下游**：:mod:`symbol_catalog.verify` 在它上面检查七条性质，
  :mod:`symbol_catalog.study` 打印清单的表。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from smart_research_agent.symbol_catalog.errors import CatalogBuildError, ParameterError
from smart_research_agent.symbol_catalog.parse import ModuleBindings, scan_all
from smart_research_agent.symbol_catalog.types import (
    PACKAGE_NAME,
    SOURCE_DECLARED,
    SOURCE_DESCRIPTIONS,
    require_promise_source,
)

#: 清单摘要取前多少位十六进制（与 day099 ~ day103 同一口径）.
CATALOG_DIGEST_LENGTH = 16


def _package_of(module: str) -> str:
    """一个模块的顶层子包名（``smart_research_agent.api.app`` → ``api``）.

    根包 ``smart_research_agent``（自身只有一个点）返回它自己。
    """
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else module


@dataclass(frozen=True)
class SymbolCatalog:
    """一份符号清单：模块行（排序唯一）+ 由它派生的一组只读视图.

    ``entries`` 里的每个 :class:`~symbol_catalog.parse.ModuleBindings`
    都是同一时刻解析出来的——因此整份清单读的是一个一致的状态。
    """

    entries: tuple[ModuleBindings, ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise CatalogBuildError("清单不能没有模块行。")
        names = [entry.module for entry in self.entries]
        if names != sorted(names) or len(set(names)) != len(names):
            raise CatalogBuildError(f"模块行必须是一份排序且不重复的名单：{len(names)} 行。")

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

    def entry_of(self, module: str) -> ModuleBindings:
        """取一个模块的行（不在清单里当场拒绝）."""
        for entry in self.entries:
            if entry.module == module:
                return entry
        raise ParameterError(f"清单里没有模块 {module!r}。")

    def package_of(self, module: str) -> str:
        """一个模块的顶层子包名（不在清单里当场拒绝）."""
        self.entry_of(module)
        return _package_of(module)

    def package_groups(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """按顶层子包分组（组名排序，组内成员排序）."""
        groups: dict[str, list[str]] = {}
        for entry in self.entries:
            groups.setdefault(_package_of(entry.module), []).append(entry.module)
        return tuple((name, tuple(sorted(groups[name]))) for name in sorted(groups))

    def source_counts(self) -> tuple[tuple[str, int], ...]:
        """三种承诺来源各有几个模块（按 :data:`types.PROMISE_SOURCES` 的顺序）."""
        table: dict[str, int] = {source: 0 for source in SOURCE_DESCRIPTIONS}
        for entry in self.entries:
            table[entry.source] += 1
        return tuple((source, table[source]) for source in SOURCE_DESCRIPTIONS)

    def export_pairs(self) -> tuple[tuple[str, str], ...]:
        """全部 `(模块, 对外名字)`（顺序 = 模块名升序 × 名字升序）."""
        return tuple(
            (entry.module, name) for entry in self.entries for name in entry.exports()
        )

    def promise_count(self) -> int:
        """全部模块承诺的名字总数."""
        return sum(len(entry.exports()) for entry in self.entries)

    def declared_count(self) -> int:
        """写了字面量 ``__all__`` 的模块数."""
        return sum(1 for entry in self.entries if entry.source == SOURCE_DECLARED)

    # ------------------------------------------------------------------ 重名与幽灵

    def shared_names(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """被 **两个以上** 模块导出的名字（按 "出现模块数降序 × 名字升序" 排）.

        一个名字被几十个模块同时导出（如 ``ShapeError`` / ``NumericError``）既是
        "同一件事在很多层各写了一遍"的证据，也是跨包契约的入口。
        """
        table: dict[str, list[str]] = {}
        for module, name in self.export_pairs():
            table.setdefault(name, []).append(module)
        shared = [
            (name, tuple(sorted(set(modules))))
            for name, modules in table.items()
            if len(set(modules)) >= 2
        ]
        return tuple(sorted(shared, key=lambda row: (-len(row[1]), row[0])))

    def phantom_pairs(self) -> tuple[tuple[str, str], ...]:
        """全部幽灵导出 `(模块, 名字)`（顺序 = 模块名升序 × 名字升序）."""
        return tuple(
            (entry.module, name) for entry in self.entries for name in entry.phantoms()
        )

    def duplicate_declarations(self) -> tuple[str, ...]:
        """同一个 ``__all__`` 里重复声明了名字的模块（排序）."""
        return tuple(
            entry.module
            for entry in self.entries
            if len(set(entry.declared)) != len(entry.declared)
        )

    # ------------------------------------------------------------------ 复算口径

    def comparable(self) -> tuple[Any, ...]:
        """**只含确定性字段**的元组（两份清单逐位比较它；不含路径、不含集合）."""
        return (
            self.modules,
            tuple(
                (
                    entry.module,
                    entry.source,
                    entry.declared,
                    entry.defined,
                    entry.assigned,
                    entry.imported,
                )
                for entry in self.entries
            ),
        )

    def digest(self) -> str:
        """清单摘要（两次构建摘要相同 ⇔ 逐位相同）."""
        return hashlib.sha256(repr(self.comparable()).encode("utf-8")).hexdigest()[
            :CATALOG_DIGEST_LENGTH
        ]

    def diff_count(self, other: SymbolCatalog) -> int:
        """与另一份清单在复算口径上**差了几项**（0 = 逐位相同）."""
        left = self.comparable()
        right = other.comparable()
        return sum(1 for a, b in zip(left, right, strict=True) if a != b)

    def to_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 化的字段."""
        return {
            "modules": self.module_count,
            "packages": len(self.package_groups()),
            "declared_modules": self.declared_count(),
            "promises": self.promise_count(),
            "shared_names": len(self.shared_names()),
            "phantoms": len(self.phantom_pairs()),
            "digest": self.digest(),
        }

    def line(self) -> str:
        """一行读数：``清单：457 个模块 | 346 份字面量承诺 | 承诺 8377 | 重名 2980 | 幽灵 1``."""
        return (
            f"清单：{self.module_count} 个模块 | {self.declared_count()} 份字面量承诺"
            f" | 承诺 {self.promise_count()} | 重名 {len(self.shared_names())}"
            f" | 幽灵 {len(self.phantom_pairs())} | 摘要 {self.digest()}"
        )


def build_catalog(modules: tuple[ModuleBindings, ...] | None = None) -> SymbolCatalog:
    """把解析结果编成一份 :class:`SymbolCatalog`（**确定性**）.

    ``modules`` 缺省时现场扫一遍；测试可以注入一组小模块。
    """
    resolved = scan_all() if modules is None else modules
    return SymbolCatalog(entries=tuple(sorted(resolved, key=lambda entry: entry.module)))


def catalog_digest(catalog: SymbolCatalog | None = None) -> str:
    """清单摘要（缺省时现场编一份）."""
    resolved = build_catalog() if catalog is None else catalog
    return resolved.digest()


def require_reproducible(first: SymbolCatalog, second: SymbolCatalog) -> SymbolCatalog:
    """两份清单不一致时抛 :class:`CatalogBuildError`（"拒绝交付"的那条路）."""
    diff = first.diff_count(second)
    if diff == 0:
        return first
    raise CatalogBuildError(
        f"同一批文件编出的两份清单差了 {diff} 项——"
        "清单里混进了未固定的量（集合序 / 字典序 / 绝对路径），它还不是一份可以被别人重算的中间物。"
    )


def require_unique_declarations(catalog: SymbolCatalog | None = None) -> SymbolCatalog:
    """有模块重复声明同一个名字时抛 :class:`CatalogBuildError`（"拒绝交付"的那条路）."""
    resolved = build_catalog() if catalog is None else catalog
    duplicated = resolved.duplicate_declarations()
    if duplicated:
        rendered = "、".join(duplicated[:5])
        raise CatalogBuildError(
            f"有 {len(duplicated)} 个模块在同一个 __all__ 里重复声明了名字：{rendered}——"
            "一个名字出现两次，读的人就不知道这一条到底算几条。"
        )
    return resolved


def expected_packages(catalog: SymbolCatalog | None = None) -> tuple[str, ...]:
    """读取一份清单的顶层子包名单（排序）."""
    resolved = build_catalog() if catalog is None else catalog
    return tuple(name for name, _members in resolved.package_groups())


def catalog_lines(catalog: SymbolCatalog | None = None, *, limit: int | None = None) -> tuple[str, ...]:
    """把清单逐行印出来（可选截断）."""
    from smart_research_agent.symbol_catalog.types import require_positive_int

    resolved = build_catalog() if catalog is None else catalog
    chosen = (
        resolved.entries
        if limit is None
        else resolved.entries[: require_positive_int("limit", limit)]
    )
    lines = ["符号清单（模块 | 承诺来源 | 承诺数 | 落点数）："]
    lines.extend("  " + entry.line() for entry in chosen)
    lines.append("  " + resolved.line())
    return tuple(lines)


def docs_of_source(source: str) -> str:
    """返回一种承诺来源的一句话解释（未知来源当场拒绝）."""
    return SOURCE_DESCRIPTIONS[require_promise_source(source)]


def root_package() -> str:
    """被扫描的包名（清单里所有模块的前缀）."""
    return PACKAGE_NAME


__all__ = [
    "CATALOG_DIGEST_LENGTH",
    "SymbolCatalog",
    "build_catalog",
    "catalog_digest",
    "catalog_lines",
    "docs_of_source",
    "expected_packages",
    "require_reproducible",
    "require_unique_declarations",
    "root_package",
]
