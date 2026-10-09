"""``parse``：把 450 份 ``.py`` 读成"谁 import 了谁"（day103）.

解析只做三件事，而且**一行 ``import`` 都不执行**：

```text
收节点   列出 ``smart_research_agent/**/*.py``，每个文件是一个模块（``__init__.py`` 是包节点）
解析     用标准库 ``ast`` 把每份文件里的 import 语句读出来
定目标   把每条 import 的目标匹配到"已知模块 / 已知包"上（相对导入要按 level 减层）
```

## 一、今天最值钱的一句话

> **一份没进图的模块，与"这个模块不存在"在依赖图里读起来完全一样。**

因此 :func:`expected_modules`（本该进图的模块）与 :func:`unresolved_of`（谁对不上号）
是这一课的第一等公民：前者管"有没有漏"，后者管"有没有算错"。

## 二、相对导入是本课唯一"会算错而且不会报错"的地方

```text
在 smart_research_agent/foo/bar.py 里：
  from .base import X       level=1 ⇒ 目标是 smart_research_agent.foo.base
  from ..other import Y     level=2 ⇒ 目标是 smart_research_agent.other
  from . import baz         level=1 ⇒ 目标是 smart_research_agent.foo.baz（若它是个模块）
```

`level - 1` 这一层"点"必须减掉；少减一层，边就指到隔壁模块去了——
而依赖图照样能画出来、照样能排序，**只有"未解析目标数为 0"这条性质会把它抓住**。

## 三、与既有包的接缝

- **上游**：文件系统（``smart_research_agent/**/*.py``），不 import 任何子包；
- **下游**：:mod:`import_graph.graph` 用它建图，
  :mod:`import_graph.verify` 用它检查覆盖与"目标都能解析"。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smart_research_agent.import_graph.errors import ImportTargetError, ParseError
from smart_research_agent.import_graph.types import (
    EDGE_KIND_ABSOLUTE,
    EDGE_KIND_RELATIVE,
    INIT_STEM,
    MODULE_SUFFIX,
    PACKAGE_NAME,
    require_edge_kind,
)

#: 项目根目录（``parse.py`` 向上三级：import_graph → smart_research_agent → 项目根）.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 被扫描的包目录.
PACKAGE_DIR = PROJECT_ROOT / PACKAGE_NAME


@dataclass(frozen=True)
class ModuleIndex:
    """"已知模块 / 已知包"的索引（:func:`resolve_candidates` 在这里查表）."""

    modules: tuple[str, ...]
    packages: tuple[str, ...]
    prefixes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.modules:
            raise ParseError("模块索引不能为空。")
        for name, rows in (("modules", self.modules), ("packages", self.packages)):
            if len(set(rows)) != len(rows) or list(rows) != sorted(rows):
                raise ParseError(f"{name} 必须是一份排序且不重复的名单。")

    @property
    def module_set(self) -> frozenset[str]:
        """模块名的集合（查表用）."""
        return frozenset(self.modules)

    @property
    def package_set(self) -> frozenset[str]:
        """包节点的集合（``__init__.py`` 的那些模块）."""
        return frozenset(self.packages)

    @property
    def prefix_set(self) -> frozenset[str]:
        """全部"包的父路径"（用于识别没有 ``__init__.py`` 的目录）."""
        return frozenset(self.prefixes)

    def resolve_candidates(self, candidates: tuple[str, ...]) -> str | None:
        """在一串候选目标里挑一个（**长的优先**：越具体越准）.

        ```text
        优先挑一个"已知模块"（.py 文件）
        其次挑一个"已知包"（目录 / __init__.py 的父路径）
        都不是 ⇒ None（调用方把它记成"未解析"）
        ```

        **并列必须按名字兜顺序**：``sorted(..., key=len, reverse=True)`` 在"长度相同"的
        候选上会回到输入顺序，而输入来自一个 ``set``——它的迭代序随进程变化。
        本课真的撞到了这堵墙：``from . import a, b``（``a`` 与 ``b`` 同长且都是模块）
        会让边在两次运行之间**指向不同的目标**，而图照样画得出来。
        因此这里的排序键是 ``(-长度, 名字)``：把并列的不确定性显式消掉。
        """
        ordered = sorted({item for item in candidates if item}, key=lambda item: (-len(item), item))
        for candidate in ordered:
            if candidate in self.module_set:
                return candidate
        for candidate in ordered:
            if candidate in self.package_set or candidate in self.prefix_set:
                return candidate
        return None

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "modules": len(self.modules),
            "packages": len(self.packages),
            "prefixes": len(self.prefixes),
        }

    def line(self) -> str:
        """一行读数：``索引：450 个模块 | 55 个包 | 56 条前缀``."""
        return f"索引：{len(self.modules)} 个模块 | {len(self.packages)} 个包 | {len(self.prefixes)} 条前缀"


@dataclass(frozen=True)
class ImportRef:
    """一条 import：源码里写的目标 + 边种类 + 解析到的目标（没解析到是 ``None``）.

    ``internal`` 表示"这条 import 想指向本包内部"：绝对导入看前缀，相对导入一律算内部。
    只有 ``internal=True`` 而 ``target is None`` 的才叫**未解析**——外部的 import
    （``import os``）本来就不该进这张图。
    """

    raw: str
    kind: str
    target: str | None
    internal: bool

    def __post_init__(self) -> None:
        if not self.raw:
            raise ParseError("import 的目标文本不能为空。")
        require_edge_kind(self.kind)

    @property
    def resolved(self) -> bool:
        """是不是解析到了一个已知模块 / 包."""
        return self.target is not None

    @property
    def unresolved(self) -> bool:
        """是不是"想指向本包、却没对上号"（性质 ⑤ 读它）."""
        return self.internal and self.target is None

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {"raw": self.raw, "kind": self.kind, "target": self.target, "internal": self.internal}

    def line(self) -> str:
        """一行读数：``.base → relative | smart_research_agent.import_graph.types``."""
        return f"{self.raw} → {self.kind} | {self.target or '（未解析）'}"


@dataclass(frozen=True)
class ScannedModule:
    """一份被解析的 ``.py``：模块名 + 是否包 + 路径 + 若干 import."""

    module: str
    is_package: bool
    path: str
    refs: tuple[ImportRef, ...]

    def __post_init__(self) -> None:
        if not self.module:
            raise ParseError("模块名不能为空。")

    def targets(self) -> tuple[str, ...]:
        """解析到的目标（去掉重复、保留出现顺序）."""
        seen: list[str] = []
        for ref in self.refs:
            if ref.target is not None and ref.target != self.module and ref.target not in seen:
                seen.append(ref.target)
        return tuple(seen)

    def unresolved(self) -> tuple[ImportRef, ...]:
        """未解析的 import（性质 ⑤ 读它）."""
        return tuple(ref for ref in self.refs if ref.unresolved)

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "module": self.module,
            "is_package": self.is_package,
            "refs": len(self.refs),
            "targets": len(self.targets()),
        }

    def line(self) -> str:
        """一行读数：``smart_research_agent.import_graph.graph | 3 条目标``."""
        return f"{self.module:<58} | {len(self.targets()):>3} 条目标"


def package_dir(root: Path | None = None) -> Path:
    """返回被扫描的包目录（不存在时抛 :class:`ParseError`）."""
    resolved = PACKAGE_DIR if root is None else Path(root) / PACKAGE_NAME
    if not resolved.is_dir():
        raise ParseError(
            f"包目录不存在：{resolved}——"
            "一份读不到的源码与'这个包一条依赖都没有'在依赖图里读起来一样。"
        )
    return resolved


def module_files(root: Path | None = None) -> tuple[Path, ...]:
    """列出全部 ``*.py``（按相对路径排序，确定性）."""
    directory = package_dir(root)
    files = sorted(
        (path for path in directory.rglob(f"*{MODULE_SUFFIX}") if path.is_file()),
        key=lambda item: item.relative_to(directory).as_posix(),
    )
    if not files:
        raise ParseError(f"{directory} 下一份 {MODULE_SUFFIX} 都没有。")
    return tuple(files)


def module_name(path: Path, root: Path | None = None) -> str:
    """从路径求模块名（``__init__.py`` 折成包名，取不到时抛 :class:`ParseError`）."""
    directory = package_dir(root)
    try:
        relative = path.resolve().relative_to(directory.resolve())
    except ValueError as error:  # pragma: no cover - 只在传错路径时触发
        raise ParseError(f"{path} 不在被扫描的包目录 {directory} 下。") from error
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == INIT_STEM:
        parts = parts[:-1]
    if not parts:
        return PACKAGE_NAME
    return ".".join([PACKAGE_NAME, *parts])


def is_package_file(path: Path) -> bool:
    """这份 ``.py`` 是不是包入口（``__init__.py``）."""
    return path.stem == INIT_STEM


def module_index(root: Path | None = None) -> ModuleIndex:
    """建一份"已知模块 / 已知包"的索引（排序、不重复）."""
    modules: list[str] = []
    packages: list[str] = []
    prefixes: set[str] = set()
    for path in module_files(root):
        name = module_name(path, root)
        modules.append(name)
        if is_package_file(path):
            packages.append(name)
        parts = name.split(".")
        for index in range(1, len(parts)):
            prefixes.add(".".join(parts[:index]))
    return ModuleIndex(
        modules=tuple(sorted(modules)),
        packages=tuple(sorted(packages)),
        prefixes=tuple(sorted(prefixes)),
    )


def read_source(path: Path) -> str:
    """读一份源码（读不到当场抛 :class:`ParseError`）."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError as error:  # pragma: no cover - 只在文件系统异常时触发
        raise ParseError(f"读不到源码 {path}：{error}") from error


def parse_source(source: str, *, origin: str) -> ast.Module:
    """把源码解析成 AST（语法错当场抛 :class:`ParseError`）."""
    try:
        return ast.parse(source)
    except SyntaxError as error:
        raise ParseError(f"{origin} 语法错误（第 {error.lineno} 行）：{error.msg}") from error


def relative_base(*, module: str, is_package: bool, level: int, imported: str | None) -> str:
    """把相对导入的 ``level`` 折成一个绝对前缀（见模块 docstring 第二节）.

    ```text
    所在包 = module（包入口）或 module 的父路径（普通模块）
    去掉 level - 1 层"点"，再拼上 imported（如果写了 from ... import）
    ```
    """
    own = module if is_package else module.rsplit(".", 1)[0]
    parts = own.split(".")
    up = level - 1
    if up >= len(parts):
        return ""
    base = ".".join(parts[: len(parts) - up])
    if imported:
        return f"{base}.{imported}" if base else imported
    return base


def imports_of(
    tree: ast.Module,
    *,
    module: str,
    is_package: bool,
    index: ModuleIndex,
) -> tuple[ImportRef, ...]:
    """把一份 AST 里的 import 语句读成一串 :class:`ImportRef`（**保留出现顺序**）.

    ```text
    import a.b.c [as X]            ⇒ 目标 a.b.c
    from M import n1, n2           ⇒ 候选 [M, M.n1, M.n2]，挑最具体的一个
    from .b import X               ⇒ 候选 [所在包.b, 所在包.b.X]
    ```
    """
    refs: list[ImportRef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                refs.append(
                    ImportRef(
                        raw=alias.name,
                        kind=EDGE_KIND_ABSOLUTE,
                        target=index.resolve_candidates((alias.name,)),
                        internal=alias.name == PACKAGE_NAME or alias.name.startswith(f"{PACKAGE_NAME}."),
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            if node.level == 0:
                base = node.module or ""
                kind = EDGE_KIND_ABSOLUTE
                internal = base == PACKAGE_NAME or base.startswith(f"{PACKAGE_NAME}.")
            else:
                base = relative_base(
                    module=module, is_package=is_package, level=node.level, imported=node.module
                )
                kind = EDGE_KIND_RELATIVE
                # 相对导入**一律**算内部：一句 from ...baz import X 想指的永远是本包内部。
                # 如果它减完点之后落到了包根之外（base == ""），那不是"外部导入"，
                # 而是一条**解析不出来的包内 import** —— 它必须被性质 ⑤ 抓住。
                internal = True
            candidates = (base, *(f"{base}.{name}" for name in names)) if base else ()
            refs.append(
                ImportRef(
                    raw="." * node.level + (node.module or ""),
                    kind=kind,
                    target=index.resolve_candidates(candidates),
                    internal=internal,
                )
            )
    return tuple(refs)


def scan_module(path: Path, root: Path | None = None, *, index: ModuleIndex | None = None) -> ScannedModule:
    """解析一份 ``.py``（缺文件 / 语法错都在这里当场拒绝）."""
    resolved_index = module_index(root) if index is None else index
    tree = parse_source(read_source(path), origin=path.as_posix())
    package = is_package_file(path)
    return ScannedModule(
        module=module_name(path, root),
        is_package=package,
        path=path.as_posix(),
        refs=imports_of(
            tree,
            module=module_name(path, root),
            is_package=package,
            index=resolved_index,
        ),
    )


def scan_all(root: Path | None = None) -> tuple[ScannedModule, ...]:
    """解析全部 ``*.py``（**确定性**：两次调用逐位相同）."""
    index = module_index(root)
    return tuple(scan_module(path, root, index=index) for path in module_files(root))


def expected_modules(root: Path | None = None) -> tuple[str, ...]:
    """返回"本该进图的全部模块"（覆盖检查的参照名单）."""
    return module_index(root).modules


def unresolved_of(modules: tuple[ScannedModule, ...] | None = None) -> tuple[tuple[str, str], ...]:
    """全部未解析的 import，返回 ``(模块名, 目标文本)``（顺序即出现顺序）."""
    resolved = scan_all() if modules is None else modules
    return tuple((module.module, ref.raw) for module in resolved for ref in module.unresolved())


def require_resolved(modules: tuple[ScannedModule, ...] | None = None) -> tuple[ScannedModule, ...]:
    """有未解析的包内 import 时抛 :class:`ImportTargetError`（"拒绝交付"的那条路）."""
    resolved = scan_all() if modules is None else modules
    bad = unresolved_of(resolved)
    if bad:
        rendered = "、".join(f"{module} 里的 {raw!r}" for module, raw in bad[:5])
        raise ImportTargetError(
            f"有 {len(bad)} 条包内 import 的目标对不上号：{rendered}——"
            "先回到相对导入的解析规则（是不是少减了一层点），或修改那一条源码。"
        )
    return resolved


def scan_lines(modules: tuple[ScannedModule, ...] | None = None) -> tuple[str, ...]:
    """把解析结果逐行印出来（默认只印前若干行）."""
    resolved = scan_all() if modules is None else modules
    total = sum(len(module.targets()) for module in resolved)
    lines = ["解析表（每个模块 → 目标数）："]
    lines.extend("  " + module.line() for module in resolved[:12])
    lines.append(f"  {'（合计）':<56} | {total:>3} 条目标 | {len(resolved)} 个模块")
    return tuple(lines)


__all__ = [
    "PACKAGE_DIR",
    "PROJECT_ROOT",
    "ImportRef",
    "ModuleIndex",
    "ScannedModule",
    "expected_modules",
    "imports_of",
    "is_package_file",
    "module_files",
    "module_index",
    "module_name",
    "package_dir",
    "parse_source",
    "read_source",
    "relative_base",
    "require_resolved",
    "scan_all",
    "scan_lines",
    "scan_module",
    "unresolved_of",
]
