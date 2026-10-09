"""``parse``：把几百份 ``.py`` 读成"这个模块承诺了什么、这些承诺落到了哪里"（day104）.

解析只做三件事，而且**一行 ``import`` 都不执行**：

```text
收行     列出 ``smart_research_agent/**/*.py``，每个文件是一行
读承诺   用标准库 ``ast`` 读 ``__all__``（字面量？非字面量？根本没写？）
找落点   把这个模块**模块级作用域**里的 def / class、赋值与 import 别名收齐
```

## 一、今天最值钱的一句话

> **一份没进清单的模块，与"这个模块什么都没承诺"在清单里读起来完全一样。**

因此 :func:`expected_modules`（本该进清单的模块）与 :func:`phantom_exports`（谁承诺了却没落点）
是这一课的第一等公民：前者管"有没有漏"，后者管"有没有落空"。

## 二、为什么绑定要"只在模块级作用域"里找

```text
class Foo:
    def bar(...): ...        ⇒ bar **不算**本模块的落点（它在 Foo 里面）
def baz():
    import json             ⇒ json **不算**本模块的导入
try:
    from x import Y         ⇒ Y **算**（模块级作用域，即便写在 try 里）
```

一个名字的落点必须**在模块级**：写在函数体里的 import 或 def 属于那一层的实现细节，
把它算进来会让"承诺"看起来比实际更硬。这也是本课与 day103 的分界：
day103 关心的是**任意位置**的一条 import 语句（因为它要在图上画一条边），
本课关心的是**模块级**的一个绑定（因为它要回答"这个对外名字从哪里来"）。

## 三、与既有包的接缝

- **上游**：文件系统（``smart_research_agent/**/*.py``），不 import 任何子包；
- **下游**：:mod:`symbol_catalog.catalog` 用它编清单，
  :mod:`symbol_catalog.verify` 用它检查覆盖与"每个幽灵都是真的"。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smart_research_agent.symbol_catalog.errors import BindingError, DeclareError, ParseError
from smart_research_agent.symbol_catalog.types import (
    ALL_NAME,
    BINDING_ASSIGNED,
    BINDING_DEFINED,
    BINDING_IMPORTED,
    BINDING_UNRESOLVED,
    INIT_STEM,
    MODULE_SUFFIX,
    PACKAGE_NAME,
    SOURCE_COMPUTED,
    SOURCE_DECLARED,
    SOURCE_IMPLICIT,
    require_binding_kind,
)

#: 项目根目录（``parse.py`` 向上三级：symbol_catalog → smart_research_agent → 项目根）.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 被扫描的包目录.
PACKAGE_DIR = PROJECT_ROOT / PACKAGE_NAME

#: 模块级作用域里"还要往里看"的语句种类（**不进 def / class**）.
_NESTED_STATEMENTS: tuple[type[ast.stmt], ...] = (
    ast.If,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.For,
    ast.AsyncFor,
    ast.While,
)

#: 从一段文本里认出"标识符"的正则（幽灵导出的独立复核用）.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: 挖 `__all__` 片段时的哨兵：``ast.get_source_segment`` 对真实文本永远给得出片段，
#: 用一个不可能出现在源码里的字符串代替 ``None``，就不必为"拿不到片段"单写一条分支。
_NO_SEGMENT = "\x00\x00no-source-segment\x00\x00"


@dataclass(frozen=True)
class ModuleBindings:
    """一份被解析的 ``.py``：模块名 + 承诺来源 + 声明 + 模块级的三类绑定.

    ``declared`` 只在来源是 ``declared`` 时非空；三类绑定的名单都**排序**且去重。
    """

    module: str
    is_package: bool
    path: str
    source: str
    declared: tuple[str, ...] = ()
    defined: tuple[str, ...] = ()
    assigned: tuple[str, ...] = ()
    imported: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.module:
            raise ParseError("模块名不能为空。")
        if self.source not in (SOURCE_DECLARED, SOURCE_COMPUTED, SOURCE_IMPLICIT):
            raise DeclareError(f"未知的承诺来源 {self.source!r}。")
        if self.source == SOURCE_DECLARED and not self.declared:
            raise DeclareError(
                f"{self.module} 的来源是 declared，却没有一份声明名单——"
                "一份空的承诺与'没有承诺'在清单里读起来一样。"
            )
        for label, rows in (
            ("declared", self.declared),
            ("defined", self.defined),
            ("assigned", self.assigned),
            ("imported", self.imported),
        ):
            if list(rows) != sorted(rows) or len(set(rows)) != len(rows):
                raise ParseError(f"{self.module} 的 {label} 必须是一份排序且不重复的名单。")

    @property
    def bound_names(self) -> frozenset[str]:
        """本模块**手里有**的全部名字（前面三类绑定的并集）."""
        return frozenset(self.defined) | frozenset(self.assigned) | frozenset(self.imported)

    def binding_of(self, name: str) -> str:
        """一个名字的落点类别（顺序写死：先定义、再赋值、再导入、最后 unresolved）."""
        if name in set(self.defined):
            return BINDING_DEFINED
        if name in set(self.assigned):
            return BINDING_ASSIGNED
        if name in set(self.imported):
            return BINDING_IMPORTED
        return BINDING_UNRESOLVED

    def bindings(self) -> tuple[tuple[str, str], ...]:
        """声明名 → 落点类别（按名字排序；来源不是 declared 时为空）."""
        if self.source != SOURCE_DECLARED:
            return ()
        return tuple((name, self.binding_of(name)) for name in self.declared)

    def public_definitions(self) -> tuple[str, ...]:
        """顶层公开 def / class（名字不以 ``_`` 开头）."""
        return tuple(name for name in self.defined if not name.startswith("_"))

    def exports(self) -> tuple[str, ...]:
        """这个模块**对外**的名字：有字面量 ``__all__`` 时是它，否则退而取公开定义."""
        if self.source == SOURCE_DECLARED:
            return self.declared
        return self.public_definitions()

    def phantoms(self) -> tuple[str, ...]:
        """幽灵导出：声明了、却在本模块里找不到任何落点的名字（排序）."""
        if self.source != SOURCE_DECLARED:
            return ()
        return tuple(sorted(name for name in self.declared if self.binding_of(name) == BINDING_UNRESOLVED))

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "module": self.module,
            "is_package": self.is_package,
            "source": self.source,
            "declared": len(self.declared),
            "defined": len(self.defined),
            "assigned": len(self.assigned),
            "imported": len(self.imported),
            "bound": len(self.bound_names),
            "exports": len(self.exports()),
            "phantoms": len(self.phantoms()),
        }

    def line(self) -> str:
        """一行读数：``smart_research_agent.symbol_catalog.parse | declared | 承诺 12 | 落点 40``."""
        return (
            f"{self.module:<62} | {self.source:<8} | "
            f"承诺 {len(self.exports()):>3} | 落点 {len(self.bound_names):>3}"
        )


def package_dir(root: Path | None = None) -> Path:
    """返回被扫描的包目录（不存在时抛 :class:`ParseError`）."""
    resolved = PACKAGE_DIR if root is None else Path(root) / PACKAGE_NAME
    if not resolved.is_dir():
        raise ParseError(
            f"包目录不存在：{resolved}——"
            "一份读不到的源码与'这个包什么都没承诺'在清单里读起来一样。"
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


def iter_module_scope(statements: list[ast.stmt]):
    """造一个迭代器：只看**模块级**作用域（进 if / try / with，不进 def / class）.

    这条边界是本课与 day103 的分界（见模块 docstring 第二节）。
    """
    for node in statements:
        yield node
        if isinstance(node, _NESTED_STATEMENTS):
            yield from iter_module_scope(node.body)
            yield from iter_module_scope(getattr(node, "orelse", []) or [])
            for handler in getattr(node, "handlers", []) or []:
                yield from iter_module_scope(handler.body)
            yield from iter_module_scope(getattr(node, "finalbody", []) or [])


def target_names(node: ast.expr) -> tuple[str, ...]:
    """把一个赋值目标摊成裸名字（支持 ``a, b = ...`` 与 ``a, *b = ...``）."""
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.Tuple, ast.List)):
        collected: list[str] = []
        for element in node.elts:
            collected.extend(target_names(element))
        return tuple(collected)
    if isinstance(node, ast.Starred):
        return target_names(node.value)
    return ()


def read_all_literal(value: ast.expr | None) -> tuple[str, ...] | None:
    """把一个 ``__all__`` 的赋值右端读成一份名字表（读不成返回 ``None`` = computed）.

    字面量容器里的元素**必须都是字符串**；混进了别的东西就抛 :class:`DeclareError`——
    一份写不下来的 ``__all__`` 与"没写 ``__all__``"在清单里读起来一样，但它其实想说点什么。
    """
    if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        return None
    names: list[str] = []
    for element in value.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            names.append(element.value)
        else:
            raise DeclareError(
                f"字面量 {ALL_NAME} 里混进了非字符串元素（{type(element).__name__}）："
                "一份读不下来的承诺与'没有承诺'在清单里读起来一样。"
            )
    return tuple(names)


def collect_bindings(tree: ast.Module) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str, tuple[str, ...]]:
    """把一份 AST 读成"三类绑定 + 承诺来源 + 声明"（**全部排序**）.

    只扫**模块级**作用域；``__all__`` 被写两次（或写成 ``+=``）一律记成 ``computed``，
    因为这一课只承诺"读得出来就说得清，读不出来就说读不出来"。
    """
    defined: set[str] = set()
    assigned: set[str] = set()
    imported: set[str] = set()
    declared: tuple[str, ...] | None = None
    seen_all = 0

    for node in iter_module_scope(tree.body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names = target_names(target)
                assigned.update(names)
                if ALL_NAME in names:
                    seen_all += 1
                    read = read_all_literal(node.value)
                    if seen_all > 1 or read is None:
                        declared = None
                    else:
                        declared = read
        elif isinstance(node, ast.AnnAssign):
            names = target_names(node.target)
            assigned.update(names)
            if ALL_NAME in names:
                seen_all += 1
                read = read_all_literal(node.value)
                if seen_all > 1 or read is None:
                    declared = None
                else:
                    declared = read
        elif isinstance(node, ast.AugAssign):
            names = target_names(node.target)
            assigned.update(names)
            if ALL_NAME in names:
                seen_all += 1
                declared = None
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                imported.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            assigned.update(target_names(node.target))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    assigned.update(target_names(item.optional_vars))
        elif isinstance(node, ast.Global):
            assigned.update(node.names)

    # ``__all__`` 这个名字本身**不算**一个"模块手里的名字"：它是那份承诺的**容器**，
    # 把它算进来会让每个声明了承诺的模块凭空多一个落点（而它与承诺内容无关）。
    assigned.discard(ALL_NAME)
    defined.discard(ALL_NAME)
    imported.discard(ALL_NAME)

    if seen_all == 0:
        source = SOURCE_IMPLICIT
        declared_names: tuple[str, ...] = ()
    elif declared is None:
        source = SOURCE_COMPUTED
        declared_names = ()
    else:
        source = SOURCE_DECLARED
        declared_names = tuple(sorted(declared))

    return (
        tuple(sorted(defined)),
        tuple(sorted(assigned)),
        tuple(sorted(imported)),
        source,
        declared_names,
    )


def scan_module(path: Path, root: Path | None = None) -> ModuleBindings:
    """解析一份 ``.py``（缺文件 / 语法错 / 声明读不下来都在这里当场拒绝）."""
    tree = parse_source(read_source(path), origin=path.as_posix())
    defined, assigned, imported, source, declared = collect_bindings(tree)
    return ModuleBindings(
        module=module_name(path, root),
        is_package=is_package_file(path),
        path=path.as_posix(),
        source=source,
        declared=declared,
        defined=defined,
        assigned=assigned,
        imported=imported,
    )


def scan_all(root: Path | None = None) -> tuple[ModuleBindings, ...]:
    """解析全部 ``*.py``（**确定性**：两次调用逐位相同）."""
    return tuple(scan_module(path, root) for path in module_files(root))


def expected_modules(root: Path | None = None) -> tuple[str, ...]:
    """返回"本该进清单的全部模块"（覆盖检查的参照名单）."""
    return tuple(sorted(module_name(path, root) for path in module_files(root)))


def phantom_exports(modules: tuple[ModuleBindings, ...] | None = None) -> tuple[tuple[str, str], ...]:
    """全部幽灵导出，返回 ``(模块名, 名字)``（顺序 = 模块名升序 × 名字升序）."""
    resolved = scan_all() if modules is None else modules
    return tuple(
        (module.module, name)
        for module in resolved
        for name in module.phantoms()
    )


def outside_all_names(path: Path) -> frozenset[str]:
    """**独立复核**用：一份源码里"出现在 ``__all__`` 之外"的全部标识符.

    它走的是**纯文本**这条路——把每一段 ``__all__ = [...]`` 的源码片段从文本里挖掉，
    再正则找出剩下的标识符。因此它与 :meth:`ModuleBindings.binding_of` 是两条独立的路：
    前者看"这个名字在正文里出现过没有"，后者看"它被哪一类绑定兜住了"。

    性质 ③ 用它来判"报出来的幽灵是不是真的"：一个真幽灵，
    它的名字在 ``__all__`` 之外**一次都不该出现**。
    """
    text = read_source(path)
    tree = parse_source(text, origin=path.as_posix())
    stripped = text
    for node in iter_module_scope(tree.body):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        if not any(ALL_NAME in target_names(target) for target in targets):
            continue
        segment = ast.get_source_segment(text, node) or _NO_SEGMENT
        stripped = stripped.replace(segment, " ")
    return frozenset(_IDENTIFIER.findall(stripped))


def require_no_phantoms(modules: tuple[ModuleBindings, ...] | None = None) -> tuple[ModuleBindings, ...]:
    """有幽灵导出时抛 :class:`BindingError`（"拒绝交付"的那条路）."""
    resolved = scan_all() if modules is None else modules
    bad = phantom_exports(resolved)
    if bad:
        rendered = "、".join(f"{module} 里的 {name!r}" for module, name in bad[:5])
        raise BindingError(
            f"有 {len(bad)} 个承诺找不到落点：{rendered}——"
            "要么把这个名字真的定义出来，要么把它从 __all__ 里删掉。"
        )
    return resolved


def binding_kind_of(module: ModuleBindings, name: str) -> str:
    """一个名字的落点类别（未知类别由 :func:`types.require_binding_kind` 兜住）."""
    return require_binding_kind(module.binding_of(name))


def scan_lines(modules: tuple[ModuleBindings, ...] | None = None, *, limit: int | None = None) -> tuple[str, ...]:
    """把解析结果逐行印出来（默认只印前若干行）."""
    resolved = scan_all() if modules is None else modules
    chosen = resolved if limit is None else resolved[:limit]
    total = sum(len(module.exports()) for module in resolved)
    lines = ["解析表（每个模块 → 承诺数 / 落点数）："]
    lines.extend("  " + module.line() for module in chosen)
    lines.append(f"  {'（合计）':<60} | 承诺 {total:>4} | 模块 {len(resolved):>3}")
    return tuple(lines)


__all__ = [
    "PACKAGE_DIR",
    "PROJECT_ROOT",
    "ModuleBindings",
    "binding_kind_of",
    "collect_bindings",
    "expected_modules",
    "is_package_file",
    "iter_module_scope",
    "module_files",
    "module_name",
    "outside_all_names",
    "package_dir",
    "parse_source",
    "phantom_exports",
    "read_all_literal",
    "read_source",
    "require_no_phantoms",
    "scan_all",
    "scan_lines",
    "scan_module",
    "target_names",
]
