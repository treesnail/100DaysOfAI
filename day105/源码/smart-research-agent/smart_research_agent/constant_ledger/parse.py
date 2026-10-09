"""``parse``：把几百份 ``.py`` 读成"谁把哪个名字钉成了什么值"（day105）.

解析只做三件事，而且**一行 ``import`` 都不执行**：

```text
收行     列出 ``smart_research_agent/**/*.py``，每个文件是一行
读常量   用标准库 ``ast`` 收集**模块级**的 ``UPPER_CASE = 右端`` 赋值
定取值   用 ``ast.literal_eval`` 求值：求得出记 literal，求不出记 opaque
```

## 一、今天最值钱的一句话

> **一份没进台账的模块，与"这个模块一个常量都没定义"在台账里读起来完全一样。**

因此 :func:`expected_modules`（本该进台账的模块）与 :func:`duplicate_assignments`（谁把同一个名字钉了两次）
是这一课的第一等公民：前者管"有没有漏"，后者管"有没有含糊"。

## 二、为什么"取值"要有一个离线、确定的尺子

```text
EPSILON = 1e-5            ⇒ ast.literal_eval 给出 1e-05，取值指纹 = "1e-05"
ROOT = Path(__file__)...  ⇒ literal_eval 报错，取值指纹 = opaque（**不执行它**）
```

`literal_eval` **不执行代码、不 import、不联网**，因此"这个值是什么"有一个离线、确定的答案。
本课绝不用 `eval`：那会把"编一份台账"变成"跑一遍整个仓库"。

## 三、集合字面量是这一课真的撞到的墙

`literal_eval("{'a', 'b'}")` 给出的是一份 **`set`**，而 `set` 的 `repr` 顺序随进程变化
（与 day101 / day103 那两堵墙同源）。因此取值指纹**不能直接用 `repr`**：
:func:`canonical_value` 把 set / frozenset 的元素**先排序再拼接**，
于是"两次构建逐位相同"才不会被一个 `{}` 悄悄破坏。

## 四、与既有包的接缝

- **上游**：文件系统（``smart_research_agent/**/*.py``），不 import 任何子包；
- **下游**：:mod:`constant_ledger.ledger` 用它编台账，
  :mod:`constant_ledger.verify` 用它检查覆盖与"每个冲突都是真的"。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smart_research_agent.constant_ledger.errors import AssignError, LedgerBuildError, ParseError
from smart_research_agent.constant_ledger.types import (
    INIT_STEM,
    MIN_CONSTANT_NAME_LENGTH,
    MODULE_SUFFIX,
    PACKAGE_NAME,
    VALUE_LITERAL,
    VALUE_OPAQUE,
    require_value_kind,
)

#: 项目根目录（``parse.py`` 向上三级：constant_ledger → smart_research_agent → 项目根）.
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


def canonical_value(value: Any) -> str:
    """把一个**已求值**的字面量折成确定的字符串（集合先排序，见模块 docstring 第三节）.

    ```text
    set / frozenset   ⇒ {"a", "b"}（元素先按 repr 排序，再拼接）
    tuple / list      ⇒ 递归折每个元素
    dict              ⇒ 递归折键与值（键序保留源码顺序，因此本身已确定）
    其它              ⇒ repr(value)
    ```
    """
    if isinstance(value, (set, frozenset)):
        rendered = ", ".join(sorted(canonical_value(item) for item in value))
        return "{" + rendered + "}"
    if isinstance(value, (tuple, list)):
        rendered = ", ".join(canonical_value(item) for item in value)
        suffix = "," if isinstance(value, tuple) and len(value) == 1 else ""
        return f"({rendered}{suffix})" if isinstance(value, tuple) else f"[{rendered}]"
    if isinstance(value, dict):
        rendered = ", ".join(
            f"{canonical_value(key)}: {canonical_value(item)}" for key, item in value.items()
        )
        return "{" + rendered + "}"
    return repr(value)


def is_constant_name(name: str) -> bool:
    """这个名字算不算"被钉死的常量"（全大写标识符、长度 >= 2、不以 ``_`` 开头）."""
    return (
        name.isidentifier()
        and name.isupper()
        and not name.startswith("_")
        and len(name) >= MIN_CONSTANT_NAME_LENGTH
    )


def fingerprint(node: ast.expr | None) -> tuple[str, str | None]:
    """一条常量赋值的**取值指纹**：``(literal, 确定字符串)`` 或 ``(opaque, None)``."""
    if node is None:
        return VALUE_OPAQUE, None
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return VALUE_OPAQUE, None
    return VALUE_LITERAL, canonical_value(value)


@dataclass(frozen=True)
class ConstantDef:
    """一条常量定义：模块 + 名字 + 行号 + 是否带注解 + 取值指纹."""

    module: str
    name: str
    line: int
    annotated: bool
    value_kind: str
    literal: str | None

    def __post_init__(self) -> None:
        if not self.module or not self.name:
            raise AssignError("常量定义的模块名与常量名都不能为空。")
        if not is_constant_name(self.name):
            raise AssignError(
                f"{self.name!r} 不是一个『被钉死的常量名』："
                "常量名必须是全大写标识符、长度 >= 2，且不以 _ 开头。"
            )
        require_value_kind(self.value_kind)
        if self.value_kind == VALUE_LITERAL and self.literal is None:
            raise AssignError(f"{self.name} 记成 literal 却没有取值指纹。")
        if self.value_kind == VALUE_OPAQUE and self.literal is not None:
            raise AssignError(f"{self.name} 记成 opaque 却带着取值指纹。")
        if self.line < 1:
            raise AssignError(f"{self.name} 的行号必须是 >= 1 的整数，收到 {self.line!r}。")

    @property
    def key(self) -> tuple[str, int]:
        """排序键（名字, 行号）."""
        return (self.name, self.line)

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "module": self.module,
            "name": self.name,
            "line": self.line,
            "annotated": self.annotated,
            "value_kind": self.value_kind,
            "literal": self.literal,
        }

    def line_text(self) -> str:
        """一行读数：``EPSILON = 1e-05 | literal | 第 42 行``."""
        shown = self.literal if self.literal is not None else "（读不出来）"
        mark = "带注解" if self.annotated else "无注解"
        return f"{self.name:<34} = {shown:<24} | {self.value_kind:<7} | {mark} | 第 {self.line} 行"


@dataclass(frozen=True)
class ModuleConstants:
    """一份被解析的 ``.py``：模块名 + 它定义的全部模块级常量（按（名字, 行号）排序）."""

    module: str
    is_package: bool
    path: str
    constants: tuple[ConstantDef, ...]

    def __post_init__(self) -> None:
        if not self.module:
            raise ParseError("模块名不能为空。")
        keys = [item.key for item in self.constants]
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise ParseError(f"{self.module} 的常量必须按（名字, 行号）排序且不重复。")
        if any(item.module != self.module for item in self.constants):
            raise ParseError(f"{self.module} 的常量里混进了别的模块的定义。")

    def names(self) -> tuple[str, ...]:
        """全部常量名（排序、去重）."""
        return tuple(sorted({item.name for item in self.constants}))

    def defs_of(self, name: str) -> tuple[ConstantDef, ...]:
        """一个名字在本模块里的全部定义（按行号）."""
        return tuple(item for item in self.constants if item.name == name)

    def primary_of(self, name: str) -> ConstantDef:
        """一个名字的**主定义**：源码里最后一条（Python 导入期的最终取值）."""
        rows = self.defs_of(name)
        if not rows:
            raise ParseError(f"{self.module} 里没有常量 {name!r}。")
        return rows[-1]

    def primary_defs(self) -> tuple[ConstantDef, ...]:
        """每个名字一条主定义（按名字排序）."""
        return tuple(self.primary_of(name) for name in self.names())

    def duplicate_names(self) -> tuple[str, ...]:
        """在本模块里被赋了两次以上的常量名（排序）."""
        return tuple(sorted(name for name in self.names() if len(self.defs_of(name)) > 1))

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        primary = self.primary_defs()
        return {
            "module": self.module,
            "is_package": self.is_package,
            "constants": len(self.constants),
            "names": len(primary),
            "literal": sum(1 for item in primary if item.value_kind == VALUE_LITERAL),
            "opaque": sum(1 for item in primary if item.value_kind == VALUE_OPAQUE),
            "duplicates": len(self.duplicate_names()),
        }

    def line(self) -> str:
        """一行读数：``smart_research_agent.constant_ledger.types | 常量 41 | 可取值 40 | 读不出 1``."""
        primary = self.primary_defs()
        literal = sum(1 for item in primary if item.value_kind == VALUE_LITERAL)
        return (
            f"{self.module:<60} | 常量 {len(primary):>3} | 可取值 {literal:>3} | "
            f"读不出 {len(primary) - literal:>3} | 重名 {len(self.duplicate_names()):>2}"
        )


def package_dir(root: Path | None = None) -> Path:
    """返回被扫描的包目录（不存在时抛 :class:`ParseError`）."""
    resolved = PACKAGE_DIR if root is None else Path(root) / PACKAGE_NAME
    if not resolved.is_dir():
        raise ParseError(
            f"包目录不存在：{resolved}——"
            "一份读不到的源码与'这个包一个常量都没定义'在台账里读起来一样。"
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
    """造一个迭代器：只看**模块级**作用域（进 if / try / with，不进 def / class）."""
    for node in statements:
        yield node
        if isinstance(node, _NESTED_STATEMENTS):
            yield from iter_module_scope(node.body)
            yield from iter_module_scope(getattr(node, "orelse", []) or [])
            for handler in getattr(node, "handlers", []) or []:
                yield from iter_module_scope(handler.body)
            yield from iter_module_scope(getattr(node, "finalbody", []) or [])


def constant_defs(tree: ast.Module, *, module: str) -> tuple[ConstantDef, ...]:
    """把一份 AST 读成一串 :class:`ConstantDef`（**只在模块级作用域**，按（名字, 行号）排序）.

    ```text
    NAME = 右端               ⇒ 收（单个 Name 目标；解包赋值 A, B = ... 不收）
    NAME: 注解 = 右端          ⇒ 收（annotated=True）
    NAME += 右端 / NAME: T      ⇒ 收（取值记 opaque：它们没有可比较的右端）
    ```
    """
    collected: list[ConstantDef] = []
    for node in iter_module_scope(tree.body):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and is_constant_name(target.id):
                kind, literal = fingerprint(node.value)
                collected.append(
                    ConstantDef(
                        module=module,
                        name=target.id,
                        line=node.lineno,
                        annotated=False,
                        value_kind=kind,
                        literal=literal,
                    )
                )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if is_constant_name(node.target.id):
                kind, literal = fingerprint(node.value)
                collected.append(
                    ConstantDef(
                        module=module,
                        name=node.target.id,
                        line=node.lineno,
                        annotated=True,
                        value_kind=kind,
                        literal=literal,
                    )
                )
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            if is_constant_name(node.target.id):
                collected.append(
                    ConstantDef(
                        module=module,
                        name=node.target.id,
                        line=node.lineno,
                        annotated=False,
                        value_kind=VALUE_OPAQUE,
                        literal=None,
                    )
                )
    return tuple(sorted(collected, key=lambda item: item.key))


def value_segments(path: Path) -> dict[str, str]:
    """**独立复核**用：一份源码里每个模块级常量名的"右端源码片段"（去掉空白，最后一条算数）.

    它走的是**纯文本**这条路（`ast.get_source_segment`），与 :func:`fingerprint`
    求出来的取值指纹是两条独立的路：前者看"源码里写的是什么"，后者看"它求出来是什么"。
    性质 ③ 用它来判"报出来的冲突是不是真的"。
    """
    text = read_source(path)
    tree = parse_source(text, origin=path.as_posix())
    table: dict[str, str] = {}
    for node in iter_module_scope(tree.body):
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and is_constant_name(target.id):
            segment = ast.get_source_segment(text, value) if value is not None else None
            table[target.id] = " ".join((segment or "").split())
    return table


def scan_module(path: Path, root: Path | None = None) -> ModuleConstants:
    """解析一份 ``.py``（缺文件 / 语法错 / 常量名非法都在这里当场拒绝）."""
    tree = parse_source(read_source(path), origin=path.as_posix())
    module = module_name(path, root)
    return ModuleConstants(
        module=module,
        is_package=is_package_file(path),
        path=path.as_posix(),
        constants=constant_defs(tree, module=module),
    )


def scan_all(root: Path | None = None) -> tuple[ModuleConstants, ...]:
    """解析全部 ``*.py``（**确定性**：两次调用逐位相同）."""
    return tuple(scan_module(path, root) for path in module_files(root))


def expected_modules(root: Path | None = None) -> tuple[str, ...]:
    """返回"本该进台账的全部模块"（覆盖检查的参照名单）."""
    return tuple(sorted(module_name(path, root) for path in module_files(root)))


def duplicate_assignments(modules: tuple[ModuleConstants, ...] | None = None) -> tuple[str, ...]:
    """把同一个常量名赋了两次以上的模块（排序）."""
    resolved = scan_all() if modules is None else modules
    return tuple(
        sorted(entry.module for entry in resolved if entry.duplicate_names())
    )


def require_unique_assignments(
    modules: tuple[ModuleConstants, ...] | None = None,
) -> tuple[ModuleConstants, ...]:
    """有模块重复给同一个常量名赋值时抛 :class:`LedgerBuildError`（"拒绝交付"的那条路）."""
    resolved = scan_all() if modules is None else modules
    bad = duplicate_assignments(resolved)
    if bad:
        rendered = "、".join(bad[:5])
        raise LedgerBuildError(
            f"有 {len(bad)} 个模块把同一个常量名赋了两次以上：{rendered}——"
            "读的人就不知道台账该记哪一个数。"
        )
    return resolved


def scan_lines(modules: tuple[ModuleConstants, ...] | None = None, *, limit: int | None = None) -> tuple[str, ...]:
    """把解析结果逐行印出来（默认只印前若干行）."""
    resolved = scan_all() if modules is None else modules
    chosen = resolved if limit is None else resolved[:limit]
    total = sum(len(entry.primary_defs()) for entry in resolved)
    lines = ["解析表（每个模块 → 常量名数 / 可取值数）："]
    lines.extend("  " + entry.line() for entry in chosen)
    lines.append(f"  {'（合计）':<58} | 常量 {total:>4} | 模块 {len(resolved):>3}")
    return tuple(lines)


__all__ = [
    "PACKAGE_DIR",
    "PROJECT_ROOT",
    "ConstantDef",
    "ModuleConstants",
    "canonical_value",
    "constant_defs",
    "duplicate_assignments",
    "expected_modules",
    "fingerprint",
    "is_constant_name",
    "is_package_file",
    "iter_module_scope",
    "module_files",
    "module_name",
    "package_dir",
    "parse_source",
    "read_source",
    "require_unique_assignments",
    "scan_all",
    "scan_lines",
    "scan_module",
    "value_segments",
]
