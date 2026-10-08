"""``scan``：把 36 份 ``errors.py``（35 份历史 + 本课这一份）读成一群**原始族**（day102）.

扫描只做三件事，而且**一行 ``import`` 都不做**：

```text
发现   列出 ``smart_research_agent/*/errors.py``（排序，确定性）
解析   用标准库 ``ast`` 把每份文件读成若干 ``ClassDef``（名字 + 基类名 + 首行自述）
归属   把每份文件的 import 收成一张"别名 → 目标"表（供 resolve 用）
```

## 一、今天最值钱的一句话

> **一份没被扫到的 ``errors.py``，与"这个子包没有失败族"在台账里读起来完全一样。**

因此 :func:`error_module_paths` 与 :func:`expected_modules` 是这一课的第一等公民：
它们把"哪些文件必须进得来"变成一个**可以被点名的清单**，而不是一个沉默的缺口。

## 二、为什么"读源码"而不是"import 进来"

```text
import       会把被扫描的包真的执行一遍：一个语法错误会被 ImportError 掩盖，
             一个副作用（写文件 / 连网）会在"编台账"时被触发
ast.parse    只读文本：语法错误当场点名（:class:`ScanError`），不执行任何东西——确定性
```

## 三、与既有包的接缝

- **上游**：文件系统（``smart_research_agent/*/errors.py``），不 import 任何子包；
- **下游**：:mod:`failure_ledger.resolve` 用它的别名表归属基类，
  :mod:`failure_ledger.ledger` 用它建台账。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smart_research_agent.failure_ledger.errors import ScanError
from smart_research_agent.failure_ledger.types import PACKAGE_NAME, SCAN_FILENAME

#: 项目根目录（``scan.py`` 向上三级：failure_ledger → smart_research_agent → 项目根）.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 被扫描的包目录.
PACKAGE_DIR = PROJECT_ROOT / PACKAGE_NAME


@dataclass(frozen=True)
class RawFamily:
    """一份文件里的一个 ``ClassDef``（**还没归属**的原始记录）.

    ``bases`` 存的是**源码里写的名字**（例如 ``CoreShapeError``），
    它是不是"本模块的 / 内置的 / 从别处导进来的"由 :mod:`failure_ledger.resolve` 决定。
    """

    name: str
    bases: tuple[str, ...]
    doc_first_line: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ScanError("族名不能为空。")

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {"name": self.name, "bases": list(self.bases), "doc": self.doc_first_line}

    def line(self) -> str:
        """一行读数：``ShapeError(CoreShapeError, StackError)``."""
        return f"{self.name}({', '.join(self.bases)})"


@dataclass(frozen=True)
class ScannedModule:
    """一份被扫描的 ``errors.py``：包名 + 路径 + 若干原始族 + 别名表."""

    package: str
    path: str
    families: tuple[RawFamily, ...]
    aliases: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.package:
            raise ScanError("包名不能为空。")
        if not self.families:
            raise ScanError(
                f"{self.path} 里一个 ClassDef 都没有："
                "一份没有族的 errors.py 与'这个子包没有失败族'在台账里读起来一样。"
            )
        names = [family.name for family in self.families]
        if len(set(names)) != len(names):
            raise ScanError(f"{self.path} 里有重名的族：{sorted(names)}。")

    @property
    def family_names(self) -> tuple[str, ...]:
        """本模块里的族名（顺序即源码顺序）."""
        return tuple(family.name for family in self.families)

    def alias_of(self, raw: str) -> str | None:
        """查一个别名（没有时返回 ``None``）."""
        for alias, target in self.aliases:
            if alias == raw:
                return target
        return None

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {
            "package": self.package,
            "path": self.path,
            "families": len(self.families),
        }

    def line(self) -> str:
        """一行读数：``transformer_stack | 6 族 | errors.py``."""
        return f"{self.package:<24} | {len(self.families):>3} 族 | {SCAN_FILENAME}"


def package_dir(root: Path | None = None) -> Path:
    """返回被扫描的包目录（不存在时抛 :class:`ScanError`）."""
    resolved = PACKAGE_DIR if root is None else Path(root) / PACKAGE_NAME
    if not resolved.is_dir():
        raise ScanError(
            f"包目录不存在：{resolved}——"
            "一份读不到的源码与'这个包没有失败族'在台账里读起来一样。"
        )
    return resolved


def error_module_paths(root: Path | None = None) -> tuple[Path, ...]:
    """列出全部 ``errors.py``（按相对路径排序，确定性）."""
    directory = package_dir(root)
    paths = sorted(
        (path for path in directory.glob(f"*/{SCAN_FILENAME}") if path.is_file()),
        key=lambda item: item.as_posix(),
    )
    if not paths:
        raise ScanError(f"{directory} 下一份 {SCAN_FILENAME} 都没有。")
    return tuple(paths)


def package_label(path: Path, root: Path | None = None) -> str:
    """从路径里取包名（``.../transformer_stack/errors.py`` → ``transformer_stack``）."""
    directory = package_dir(root)
    relative = path.resolve().parent.relative_to(directory.resolve())
    label = relative.as_posix()
    return label if label != "." else "<root>"


def read_source(path: Path) -> str:
    """读一份源码（读不到当场抛 :class:`ScanError`）."""
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError as error:  # pragma: no cover - 只在文件系统异常时触发
        raise ScanError(f"读不到源码 {path}：{error}") from error


def parse_source(source: str, *, origin: str) -> ast.Module:
    """把源码解析成 AST（语法错当场抛 :class:`ScanError`）."""
    try:
        return ast.parse(source)
    except SyntaxError as error:
        raise ScanError(f"{origin} 语法错误（第 {error.lineno} 行）：{error.msg}") from error


def import_aliases(tree: ast.Module) -> tuple[tuple[str, str], ...]:
    """把 import 收成 ``(本地名, 目标)`` 别名表（按本地名排序，确定性）.

    ```text
    from a.b.errors import ShapeError as CoreShapeError   ⇒  ("CoreShapeError", "a.b.errors.ShapeError")
    import a.b.c as D                                     ⇒  ("D", "a.b.c")
    ```
    """
    rows: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                local = alias.asname or alias.name
                rows.append((local, f"{module}.{alias.name}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                rows.append((local, alias.name))
    return tuple(sorted(set(rows)))


def class_defs(tree: ast.Module) -> tuple[RawFamily, ...]:
    """取一份 AST 里**顶层**的 ``ClassDef``（嵌套类不算：它们不是失败族）."""
    families: list[RawFamily] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = tuple(ast.unparse(base) for base in node.bases)
        doc = ast.get_docstring(node, clean=True) or ""
        first_line = next((line.strip() for line in doc.splitlines() if line.strip()), "")
        families.append(RawFamily(name=node.name, bases=bases, doc_first_line=first_line))
    return tuple(families)


def scan_module(path: Path, root: Path | None = None) -> ScannedModule:
    """扫描一份 ``errors.py``（缺族 / 语法错 / 重名都在这里当场拒绝）."""
    tree = parse_source(read_source(path), origin=path.as_posix())
    return ScannedModule(
        package=package_label(path, root),
        path=path.as_posix(),
        families=class_defs(tree),
        aliases=import_aliases(tree),
    )


def scan_all(root: Path | None = None) -> tuple[ScannedModule, ...]:
    """扫描全部 ``errors.py``（**确定性**：两次调用逐位相同）."""
    return tuple(scan_module(path, root) for path in error_module_paths(root))


def expected_modules(root: Path | None = None) -> tuple[str, ...]:
    """返回"本该进台账的全部包名"（覆盖检查的参照名单）."""
    return tuple(package_label(path, root) for path in error_module_paths(root))


def packages_without_errors(root: Path | None = None) -> tuple[str, ...]:
    """列出**没有** ``errors.py`` 的子包（读数，不是失败）."""
    directory = package_dir(root)
    all_packages = sorted(
        child.name
        for child in directory.iterdir()
        if child.is_dir() and not child.name.startswith((".", "_"))
    )
    with_errors = set(expected_modules(root))
    return tuple(name for name in all_packages if name not in with_errors)


def scan_lines(modules: tuple[ScannedModule, ...] | None = None) -> tuple[str, ...]:
    """把扫描结果逐行印出来."""
    resolved = scan_all() if modules is None else modules
    total = sum(len(module.families) for module in resolved)
    lines = ["扫描表（每份 errors.py → 族数）："]
    lines.extend("  " + module.line() for module in resolved)
    lines.append(f"  {'（合计）':<22} | {total:>3} 族 | 合计 {len(resolved)} 份")
    return tuple(lines)


__all__ = [
    "PACKAGE_DIR",
    "PROJECT_ROOT",
    "RawFamily",
    "ScannedModule",
    "class_defs",
    "error_module_paths",
    "expected_modules",
    "import_aliases",
    "package_dir",
    "package_label",
    "packages_without_errors",
    "parse_source",
    "read_source",
    "scan_all",
    "scan_lines",
    "scan_module",
]
