"""``resolve``：把一个基类名归属到四类中的一类（day102）.

归属只有四类，判定顺序也是固定的：

```text
① local      名字就在同一个 errors.py 里（``class ShapeError(CoreShapeError, StackError)`` 的 StackError）
② builtin    名字在 :data:`types.BUILTIN_EXCEPTIONS` 白名单里（ValueError / Exception / RuntimeError ……）
③ imported   名字是某个 import 的别名（``ShapeError as CoreShapeError`` ⇒ 指向另一个 errors.py）
④ unresolved 以上三类都不是 ⇒ **当场点名**，绝不静默地当成类名
```

## 一、今天最值钱的一句话

> **一个"看起来像基类名"的字符串，如果没有人能说清它属于哪一类，它就不该被静默地当成类名。**

:func:`require_resolved` 是这条纪律的执行者：它是"拒绝交付"的那条路，
而 :class:`BaseRef` 带着 ``kind`` 与 ``target`` 是"这条归属是怎么来的"的证据。

## 二、为什么判定顺序不能换

```text
先 builtin 后 local   一个子包若定义了自己的 ``ValueError``（合法！），会被误认成内置异常
先 imported 后 local  本模块里的族会被误认成"从别处导进来的同名字"
```

因此顺序写死成 local → builtin → imported。这与 day087 的"位置必须由缓存长度决定"、
day101 的"分词只有一个交替正则"是同一条纪律：**把与正文无关的顺序显式定死。**

## 三、与既有包的接缝

- **上游**：:mod:`failure_ledger.scan`（原始族与别名表）、:mod:`failure_ledger.types`（白名单）；
- **下游**：:mod:`failure_ledger.ledger` 用归属结果建台账。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_research_agent.failure_ledger.errors import ResolveError
from smart_research_agent.failure_ledger.scan import RawFamily, ScannedModule
from smart_research_agent.failure_ledger.types import (
    BASE_KIND_BUILTIN,
    BASE_KIND_IMPORTED,
    BASE_KIND_LOCAL,
    BASE_KIND_UNRESOLVED,
    BUILTIN_EXCEPTIONS,
    require_base_kind,
)


@dataclass(frozen=True)
class BaseRef:
    """一条归属好的基类引用：源码里的名字 + 归属类别 + 去向.

    ``target`` 是"这条归属是怎么来的"的证据：
    local 指向本模块的族名、builtin 指向内置异常名、imported 指向带路径的目标，
    unresolved 原样保留那个名字（它正是要被点名的那一个）。
    """

    raw: str
    kind: str
    target: str

    def __post_init__(self) -> None:
        if not self.raw:
            raise ResolveError("基类名不能为空。")
        require_base_kind(self.kind)
        if not self.target:
            raise ResolveError(f"基类 {self.raw!r} 的归属 {self.kind!r} 没有去向。")
        if self.kind == BASE_KIND_UNRESOLVED and self.target != self.raw:
            raise ResolveError(
                f"未解析的基类 {self.raw!r} 的去向必须原样保留它自己，收到 {self.target!r}——"
                "被改写的名字会让'它到底是什么'这件事再也对不上。"
            )

    @property
    def is_local(self) -> bool:
        """是不是本模块内部的继承."""
        return self.kind == BASE_KIND_LOCAL

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {"raw": self.raw, "kind": self.kind, "target": self.target}

    def line(self) -> str:
        """一行读数：``StackError → local | ShapeError → imported(smart_research_agent...)``."""
        return f"{self.raw} → {self.kind} | {self.target}"


def resolve_base(raw: str, *, local_names: frozenset[str], module: ScannedModule) -> BaseRef:
    """归属一个基类名（判定顺序写死在 :mod:`failure_ledger.resolve` 的 docstring 里）."""
    if raw in local_names:
        return BaseRef(raw=raw, kind=BASE_KIND_LOCAL, target=raw)
    if raw in BUILTIN_EXCEPTIONS:
        return BaseRef(raw=raw, kind=BASE_KIND_BUILTIN, target=raw)
    imported = module.alias_of(raw)
    if imported is not None:
        return BaseRef(raw=raw, kind=BASE_KIND_IMPORTED, target=imported)
    return BaseRef(raw=raw, kind=BASE_KIND_UNRESOLVED, target=raw)


def local_names_of(module: ScannedModule) -> frozenset[str]:
    """一份模块里"本模块定义的"全部族名."""
    return frozenset(module.family_names)


def resolve_family(family: RawFamily, module: ScannedModule) -> tuple[BaseRef, ...]:
    """归属一个族的全部基类."""
    local = local_names_of(module)
    return tuple(resolve_base(raw, local_names=local, module=module) for raw in family.bases)


def unresolved_of(refs: tuple[BaseRef, ...]) -> tuple[str, ...]:
    """从一串归属里挑出未解析的那些名字（顺序即出现顺序）."""
    return tuple(ref.raw for ref in refs if ref.kind == BASE_KIND_UNRESOLVED)


def require_resolved(module: ScannedModule, refs: tuple[BaseRef, ...]) -> tuple[BaseRef, ...]:
    """有未解析的基类名时抛 :class:`ResolveError`（"拒绝交付"的那条路）.

    它与 :func:`resolve_base` 的分工是"给归属 vs 拒绝坏名字"：
    一个未解析的名字在报告里只是一行 ``unresolved``，而在本课它是一次真失败。
    """
    bad = unresolved_of(refs)
    if not bad:
        return refs
    raise ResolveError(
        f"{module.package}/{module.path} 里有无法归属的基类名：{list(bad)}——"
        "它既不是本模块的、也不是内置、也没有导入别名；要么补一条别名规则，要么删掉那个名字。"
    )


def resolve_lines(refs: tuple[BaseRef, ...]) -> tuple[str, ...]:
    """把一串归属逐行印出来."""
    lines = ["归属表（基类名 → 类别 | 去向）："]
    lines.extend("  " + ref.line() for ref in refs)
    return tuple(lines)


__all__ = [
    "BaseRef",
    "local_names_of",
    "require_resolved",
    "resolve_base",
    "resolve_family",
    "resolve_lines",
    "unresolved_of",
]
