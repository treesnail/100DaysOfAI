"""``manifest``：用 ``importlib`` 把八项能力**真的**落到子包上，产出清单与覆盖报告（day099）.

day088 的图上，落点是一个 ``"模块.函数"`` 字符串；今天上移一层——
落点是**子包**，因为结业项目要回答的问题是"这一项能力由哪个包负责"。
本模块只做三件事：

```text
解析    importlib.import_module("smart_research_agent." + name)
        —— 子包存在吗？它公开了多少个名字？
落点    把 types.CAPABILITIES 里的每一项能力，落到它的承担子包上
报告    哪些能力被覆盖、哪些子包"无人认领"（已交付但不属于这 8 项能力）
```

## 一、今天最值钱的一句话

> **"覆盖"必须由 ``importlib`` 说了算，而不是由一张手写的表说了算——
> 表可以写"已覆盖"，而 ``importlib`` 会当场告诉你那个包在不在。**

因此 :func:`build_manifest` **不静默跳过**解析失败：它把每一个承担子包的
"在不在 / 公开了多少个名字"都记进 :class:`ModuleInfo`，覆盖与否是**算出来的**。
一个解析不到的子包名是一条指向空气的边——它既不该被计进覆盖，
也不该让整份报告直接崩掉（那会让"哪一项能力没落地"这个问题反而答不出来）。
真想"宁可报错不可漏算"时调 :func:`require_manifest_complete`。

## 二、为什么"无人认领的子包"要单独列一份

```text
结业项目里已交付的东西 ≠ 这 8 项能力
indexing / mcp_server / finetune / api 都是真的写出来了的包，
但它们不在这 8 项能力的清单里——报告里必须把这件事**说出来**，
否则"12 个候选子包里有 4 个没人认领"会被误读成"有 4 个包是坏的"。
```

因此 :class:`Manifest` 同时给两支名单：被认领的（coverages 里 owners 的并集）
与无人认领的（候选池减去前者）。

## 三、与既有包的接缝

- **上游**：:mod:`capstone.types`（``CAPABILITIES`` / ``CANDIDATE_SUBPACKAGES``）；
- **下游**：:mod:`capstone.document` 用它渲染 README 与架构文档，
  :mod:`capstone.verify` 用它检查"8 项能力是否全部被覆盖"，
  :mod:`capstone.study` 用它打印清单表。
"""

from __future__ import annotations

import importlib
import pkgutil
import types
from dataclasses import dataclass
from typing import Any

from smart_research_agent.capstone.errors import ManifestError, ParameterError
from smart_research_agent.capstone.types import (
    CAPABILITIES,
    CAPABILITY_ORDER,
    CANDIDATE_SUBPACKAGES,
    SUBPACKAGE_DESCRIPTIONS,
    Capability,
)

#: 本项目的包前缀（子包名 = 前缀 + 名字）.
PACKAGE_PREFIX = "smart_research_agent."


@dataclass(frozen=True)
class ModuleInfo:
    """一个子包的解析结论：在不在 + 公开了多少个名字（**不抛异常，只记录**）.

    ``error`` 是解析失败时的第一行错误信息（成功时为空串）：
    把它**记下来**而不是抛出去，是为了让"哪一项能力没落地"这个问题可以先被回答。
    """

    name: str
    exists: bool
    symbol_count: int
    has_all: bool
    error: str = ""

    def __post_init__(self) -> None:
        if self.name not in CANDIDATE_SUBPACKAGES:
            raise ParameterError(
                f"未知的候选子包 {self.name!r}：可选 {list(CANDIDATE_SUBPACKAGES)}。"
            )
        if self.exists and self.symbol_count < 0:
            raise ParameterError(f"子包 {self.name!r} 的公开符号数不能为负。")
        if not self.exists and not self.error:
            raise ManifestError(
                f"子包 {self.name!r} 标了'不存在'却没有留错误信息："
                "一份只写'不存在'的报告没有任何下一步动作——要点名它为什么不在。"
            )

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "name": self.name,
            "exists": self.exists,
            "symbol_count": self.symbol_count,
            "has_all": self.has_all,
            "error": self.error,
            "description": SUBPACKAGE_DESCRIPTIONS[self.name],
        }

    def line(self) -> str:
        """一行读数：``security | 在场 | 公开 8 个符号 | 输入侧护栏……``."""
        state = "在场" if self.exists else "缺失"
        detail = f"公开 {self.symbol_count} 个符号" if self.exists else self.error
        return f"{self.name:<12} | {state} | {detail} | {SUBPACKAGE_DESCRIPTIONS[self.name]}"


def _public_symbol_count(module: types.ModuleType) -> int:
    """数一个模块**公开导出的名字**（有 ``__all__`` 就信它）.

    没有 ``__all__`` 的包（例如 ``agent`` / ``llm`` / ``tools``）改用**文件系统口径**：
    数它目录下有多少个子模块。理由只有一个——**确定性与导入顺序无关**：

    ```text
    按 vars() 数   随着别的模块把子模块 import 进来，包命名空间会“长”出新名字，
                   于是同一次运行里"先数"与"后数"会给出不同的数字（6 与 7 这种）
    按目录数      结果只取决于磁盘上有哪些文件，因此两次调用逐位相同
    ```

    "公开面"本来就包含子模块（它们是对外可 import 的东西），
    因此这个口径不是权宜之计，而是对无 ``__all__`` 包更准确的一种数法。
    """
    exported = getattr(module, "__all__", None)
    if exported is not None:
        return len(set(exported))
    package_paths = getattr(module, "__path__", None)
    if package_paths:
        return sum(1 for _ in pkgutil.iter_modules(package_paths))
    return sum(1 for name in vars(module) if not name.startswith("_"))


def resolve_subpackage(name: str) -> ModuleInfo:
    """解析一个子包：``importlib`` 真的去 import 一次，返回 :class:`ModuleInfo`.

    未知名字（不在 :data:`CANDIDATE_SUBPACKAGES` 里）当场抛 :class:`ManifestError`；
    "名字对但这个包 import 失败"则被**记录**进 ``error`` 而不抛出——
    这样覆盖报告仍然给得出来。
    """
    if name not in CANDIDATE_SUBPACKAGES:
        raise ManifestError(
            f"未知的候选子包 {name!r}：可选 {list(CANDIDATE_SUBPACKAGES)}——"
            "清单只认候选池里的名字；池子外的名字既不会被解析、也不会被计入覆盖。"
        )
    try:
        module = importlib.import_module(PACKAGE_PREFIX + name)
    except Exception as error:  # noqa: BLE001 —— 报告里只印第一行
        message = str(error).splitlines()[0] if str(error) else type(error).__name__
        return ModuleInfo(name=name, exists=False, symbol_count=0, has_all=False, error=message)
    return ModuleInfo(
        name=name,
        exists=True,
        symbol_count=_public_symbol_count(module),
        has_all=getattr(module, "__all__", None) is not None,
    )


@dataclass(frozen=True)
class CapabilityCoverage:
    """一项能力的覆盖结论：承担子包清单 + 每个子包的解析结论 + 是否被覆盖.

    ``covered`` 的定义只有一条：**每一个承担子包都在场**。
    写成"只要有一个在场就算覆盖"会让"某项能力有一半没接上"读起来像"接上了"。
    """

    capability: str
    modules: tuple[ModuleInfo, ...]

    def __post_init__(self) -> None:
        if self.capability not in CAPABILITIES:
            raise ParameterError(
                f"未知的能力 {self.capability!r}：可选 {list(CAPABILITY_ORDER)}。"
            )
        if not self.modules:
            raise ManifestError(f"能力 {self.capability!r} 没有任何承担子包。")

    @property
    def owners(self) -> tuple[str, ...]:
        """承担子包名（顺序与能力表里的一致）."""
        return tuple(module.name for module in self.modules)

    @property
    def covered(self) -> bool:
        """是否被覆盖 = 每一个承担子包都在场."""
        return all(module.exists for module in self.modules)

    @property
    def missing(self) -> tuple[str, ...]:
        """不在场的承担子包（应当为空）."""
        return tuple(module.name for module in self.modules if not module.exists)

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "capability": self.capability,
            "owners": list(self.owners),
            "covered": self.covered,
            "missing": list(self.missing),
            "modules": [module.to_dict() for module in self.modules],
        }

    def line(self) -> str:
        """一行读数：``input_guard | ✓ | security、llm | 公开 8 / 5 个符号``."""
        mark = "✓" if self.covered else "✗"
        counts = "、".join(f"{module.name}={module.symbol_count}" for module in self.modules)
        return f"{self.capability:<22} | {mark} | {'、'.join(self.owners):<28} | {counts}"


@dataclass(frozen=True)
class Manifest:
    """一份清单：8 项能力的覆盖结论 + 被认领 / 无人认领的子包两支名单.

    ``complete`` 是"8 项能力全部被覆盖"这一个条件——它是性质
    ``capabilities_are_covered`` 的读数的来源。
    """

    coverages: tuple[CapabilityCoverage, ...]
    claimed: tuple[str, ...]
    unclaimed: tuple[str, ...]

    def __post_init__(self) -> None:
        capability_ids = {coverage.capability for coverage in self.coverages}
        if capability_ids != set(CAPABILITY_ORDER):
            raise ManifestError(
                f"清单里的能力集合与 CAPABILITY_ORDER 不一致："
                f"多 {sorted(capability_ids - set(CAPABILITY_ORDER))}、"
                f"缺 {sorted(set(CAPABILITY_ORDER) - capability_ids)}——"
                "对不上名单的清单会让覆盖报告安静地少一行或多一行。"
            )
        if set(self.claimed) & set(self.unclaimed):
            raise ManifestError(
                f"同一子包不可能既被认领又无人认领：{sorted(set(self.claimed) & set(self.unclaimed))}"
            )
        if set(self.claimed) | set(self.unclaimed) != set(CANDIDATE_SUBPACKAGES):
            raise ManifestError(
                "被认领与无人认领的两支名单并起来必须恰好是候选池——"
                "漏掉的那个子包既不会出现在报告里、也不会被任何人负责。"
            )

    @property
    def covered(self) -> tuple[str, ...]:
        """被覆盖的能力 id（应当有 8 个）."""
        return tuple(coverage.capability for coverage in self.coverages if coverage.covered)

    @property
    def missing(self) -> tuple[str, ...]:
        """没有被覆盖的能力 id（应当为空）."""
        return tuple(coverage.capability for coverage in self.coverages if not coverage.covered)

    @property
    def complete(self) -> bool:
        """8 项能力是否全部被覆盖."""
        return not self.missing

    def coverage_of(self, capability: str) -> CapabilityCoverage:
        """取某一项能力的覆盖结论（未知能力当场拒绝）."""
        for coverage in self.coverages:
            if coverage.capability == capability:
                return coverage
        raise ParameterError(f"清单里没有能力 {capability!r}。")

    def module_names(self) -> tuple[str, ...]:
        """清单里出现过的全部子包（被认领的那些，去掉重复）."""
        seen: list[str] = []
        for coverage in self.coverages:
            for name in coverage.owners:
                if name not in seen:
                    seen.append(name)
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 化的字段."""
        return {
            "complete": self.complete,
            "covered": list(self.covered),
            "missing": list(self.missing),
            "claimed": list(self.claimed),
            "unclaimed": list(self.unclaimed),
            "coverages": [coverage.to_dict() for coverage in self.coverages],
        }

    def line(self) -> str:
        """一行读数：``覆盖 8/8 项能力 | 被认领 8 个包 | 无人认领 4 个``."""
        return (
            f"覆盖 {len(self.covered)}/{len(CAPABILITY_ORDER)} 项能力 | "
            f"被认领 {len(self.claimed)} 个包 | 无人认领 {len(self.unclaimed)} 个"
        )


def build_manifest(
    *,
    capabilities: dict[str, Capability] | None = None,
    candidates: tuple[str, ...] | None = None,
) -> Manifest:
    """按能力表与候选池解析子包，产出清单（**解析失败会被记录，不会中断报告**）.

    参数可注入（测试用它构造"某一项能力指到一个不存在的子包"的反例），
    默认取 :mod:`capstone.types` 的两张表。
    """
    table = CAPABILITIES if capabilities is None else capabilities
    pool = CANDIDATE_SUBPACKAGES if candidates is None else candidates
    cache: dict[str, ModuleInfo] = {}

    def resolve(name: str) -> ModuleInfo:
        if name not in cache:
            cache[name] = resolve_subpackage(name)
        return cache[name]

    coverages = tuple(
        CapabilityCoverage(
            capability=key,
            modules=tuple(resolve(owner) for owner in table[key].owners),
        )
        for key in CAPABILITY_ORDER
        if key in table
    )
    claimed: list[str] = []
    for coverage in coverages:
        for name in coverage.owners:
            if name not in claimed:
                claimed.append(name)
    unclaimed = tuple(name for name in pool if name not in claimed)
    return Manifest(
        coverages=coverages,
        claimed=tuple(claimed),
        unclaimed=unclaimed,
    )


def require_manifest_complete(manifest: Manifest | None = None) -> Manifest:
    """清单不完整时抛 :class:`ManifestError`（"宁可报错不可漏算"的那条路）.

    它与 :func:`build_manifest` 的分工是：build 只记录，require 才抛。
    这样"覆盖报告"能先被回答，而"要不要直接拒绝"是一次可选的、显式的决定。
    """
    resolved = build_manifest() if manifest is None else manifest
    if resolved.complete:
        return resolved
    failures = [coverage.line() for coverage in resolved.coverages if not coverage.covered]
    raise ManifestError("清单不完整（有能力的承担子包不在场）：" + "；".join(failures))


def manifest_lines(manifest: Manifest | None = None) -> tuple[str, ...]:
    """把清单逐行印出来（覆盖表 + 两支子包名单）."""
    resolved = build_manifest() if manifest is None else manifest
    lines: list[str] = ["覆盖表（8 项能力 → 承担子包）："]
    lines.extend("  " + coverage.line() for coverage in resolved.coverages)
    lines.append(
        "被认领的子包（" + str(len(resolved.claimed)) + "）：" + "、".join(resolved.claimed)
    )
    lines.append(
        "无人认领的子包（"
        + str(len(resolved.unclaimed))
        + "）："
        + ("、".join(resolved.unclaimed) if resolved.unclaimed else "（无）")
    )
    lines.append(resolved.line())
    return tuple(lines)


__all__ = [
    "PACKAGE_PREFIX",
    "CapabilityCoverage",
    "Manifest",
    "ModuleInfo",
    "build_manifest",
    "manifest_lines",
    "require_manifest_complete",
    "resolve_subpackage",
]
