"""``corpus``：把 100 天的材料收成一份语料（day101）.

语料有**两种**，都从仓库里**读**出来，没有一行是手写的：

```text
doc         仓库 ``docs/`` 下的 ``*.md`` 全文（课程手册）
subpackage  每个子包的 ``__init__.py`` docstring（包自述）
```

## 一、今天最值钱的一句话

> **一份没进语料的材料，与"这门课没有那份材料"在检索结果里读起来完全一样。**

因此 :meth:`Corpus.missing` 与 :func:`missing_names` 是这一课的第一等公民：
它们把"哪一份材料没进来"变成一个**可以被点名的清单**，而不是一个沉默的缺口。

## 二、一条纪律：文档名必须**带种类**

```text
docs/backprop.md            →  doc:backprop
smart_research_agent/backprop →  subpackage:backprop
```

裸名字会撞车——本仓库里 ``backprop`` / ``capstone`` / ``conv_net`` / ``graduation``
都同时是"一篇手册"与"一个子包"。撞车之后 `Corpus` 的"名字唯一"检查会抛
:class:`CorpusError`：**宁可当场变红，也不要让两件不同的东西共用一个名字**。

## 三、与既有包的接缝

- **上游**：文件系统（``docs/*.md``）与 ``importlib``（子包 docstring）；
- **下游**：:mod:`course_index.index` 用它建索引，
  :mod:`course_index.verify` 用它检查"两种语料都覆盖完整"。
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smart_research_agent.course_index.errors import CorpusError, ParameterError
from smart_research_agent.course_index.types import (
    DOC_KIND_DESCRIPTIONS,
    DOC_KIND_DOC,
    DOC_KIND_SUBPACKAGE,
    DOC_KINDS,
    require_doc_kind,
)

#: 项目根目录（``corpus.py`` 向上三级：course_index → smart_research_agent → 项目根）.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 被扫描的包名.
PACKAGE_NAME = "smart_research_agent"

#: 手册目录名与后缀.
DOCS_DIRNAME = "docs"
DOC_SUFFIX = ".md"

#: 文档名的分隔符（``doc:backprop`` 里的那个冒号）.
NAME_SEPARATOR = ":"


def qualify(kind: str, raw_name: str) -> str:
    """把"种类 + 裸名字"拼成一个**唯一的**文档名（见模块 docstring 第二节）."""
    require_doc_kind(kind)
    if not raw_name:
        raise ParameterError("裸名字不能为空：一个没有名字的文档不可能被检索到。")
    return f"{kind}{NAME_SEPARATOR}{raw_name}"


def unqualify(name: str) -> tuple[str, str]:
    """把文档名拆回 ``(种类, 裸名字)``（名字不合法当场拒绝）."""
    kind, separator, raw_name = name.partition(NAME_SEPARATOR)
    if not separator or not raw_name:
        raise ParameterError(
            f"文档名 {name!r} 不是 ``种类{NAME_SEPARATOR}裸名字`` 的形式："
            "名字里必须带种类，否则'一篇手册'与'一个子包'会共用一个名字。"
        )
    return require_doc_kind(kind), raw_name


@dataclass(frozen=True)
class Document:
    """一份语料：名字（带种类）+ 种类 + 正文.

    ``text`` 非空是一条硬要求：进了语料却一个字都没有的材料，
    会在"每份文档都至少被索引一个词"那条性质里变成一个**沉默的**失败。
    """

    name: str
    kind: str
    text: str

    def __post_init__(self) -> None:
        kind, _raw_name = unqualify(self.name)
        if kind != require_doc_kind(self.kind):
            raise ParameterError(
                f"文档 {self.name!r} 的种类字段是 {self.kind!r}，与名字里带的对不上："
                "两处各写一遍迟早会分家。"
            )
        if not self.text.strip():
            raise CorpusError(
                f"语料 {self.name!r} 的正文是空的：一份空材料在检索里"
                "与'它不存在'完全一样，但它会被计进语料总数。"
            )

    @property
    def raw_name(self) -> str:
        """裸名字（不带种类前缀）."""
        return unqualify(self.name)[1]

    @property
    def chars(self) -> int:
        """正文长度（字符数）."""
        return len(self.text)

    def to_dict(self) -> dict[str, Any]:
        """摊平成一行字段."""
        return {"name": self.name, "kind": self.kind, "chars": self.chars}

    def line(self) -> str:
        """一行读数：``doc:capstone              | doc        |  3751 字``."""
        return f"{self.name:<32} | {self.kind:<10} | {self.chars:>6} 字"


@dataclass(frozen=True)
class Corpus:
    """一份语料：若干份 :class:`Document`（名字唯一，且两种种类都可能有）."""

    documents: tuple[Document, ...]

    def __post_init__(self) -> None:
        if not self.documents:
            raise CorpusError("语料不能为空：空语料会让后面每一条性质都'通过'于无形。")
        names = [document.name for document in self.documents]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise CorpusError(
                f"语料里有重复的文档名：{duplicated}——"
                "两件不同的材料共用一个名字，检索结果里就没法区分它们。"
            )

    @property
    def names(self) -> tuple[str, ...]:
        """全部文档名（顺序即语料顺序）."""
        return tuple(document.name for document in self.documents)

    def of_kind(self, kind: str) -> tuple[Document, ...]:
        """取某一种语料（未知种类当场拒绝）."""
        return tuple(
            document for document in self.documents if document.kind == require_doc_kind(kind)
        )

    @property
    def docs(self) -> tuple[Document, ...]:
        """手册那一种语料."""
        return self.of_kind(DOC_KIND_DOC)

    @property
    def subpackages(self) -> tuple[Document, ...]:
        """子包那一种语料."""
        return self.of_kind(DOC_KIND_SUBPACKAGE)

    @property
    def total_chars(self) -> int:
        """全部语料的字符数之和."""
        return sum(document.chars for document in self.documents)

    def document_of(self, name: str) -> Document:
        """取一份语料（不在语料里当场拒绝）."""
        for document in self.documents:
            if document.name == name:
                return document
        raise ParameterError(f"语料里没有文档 {name!r}。")

    def missing(self, expected: tuple[str, ...]) -> tuple[str, ...]:
        """在期望名单里、却没有进语料的名字（应当为空）."""
        present = set(self.names)
        return tuple(name for name in expected if name not in present)

    def to_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 化的字段."""
        return {
            "documents": len(self.documents),
            "chars": self.total_chars,
            "kinds": {kind: len(self.of_kind(kind)) for kind in DOC_KINDS},
            "names": list(self.names),
        }

    def line(self) -> str:
        """一行读数：``语料 66 份 | 手册 14 / 子包 52 | 合计 123456 字``."""
        counts = " / ".join(f"{DOC_KIND_DESCRIPTIONS[kind].split('：')[0]} {len(self.of_kind(kind))}"
                            for kind in DOC_KINDS)
        return f"语料 {len(self.documents)} 份 | {counts} | 合计 {self.total_chars} 字"


def docs_dir(root: Path | None = None) -> Path:
    """返回手册目录（不存在时抛 :class:`CorpusError`）."""
    resolved = (PROJECT_ROOT if root is None else Path(root)) / DOCS_DIRNAME
    if not resolved.is_dir():
        raise CorpusError(
            f"手册目录不存在：{resolved}——"
            "一份读不到的语料与'这门课没有手册'在报告里读起来一样。"
        )
    return resolved


def discover_subpackages() -> tuple[str, ...]:
    """发现 ``smart_research_agent`` 下的全部一级子包（不 import，按名字排序）."""
    package = importlib.import_module(PACKAGE_NAME)
    paths = getattr(package, "__path__", None)
    if not paths:
        raise CorpusError(f"{PACKAGE_NAME} 不是一个包（没有 __path__），无法扫描子包。")
    names = [info.name for info in pkgutil.iter_modules(paths) if info.ispkg and not info.name.startswith("_")]
    return tuple(sorted(names))


def read_documents(root: Path | None = None) -> tuple[tuple[str, str], ...]:
    """读全部 ``docs/*.md``，返回 ``(裸名字, 正文)``（按名字排序，确定性）."""
    directory = docs_dir(root)
    rows: list[tuple[str, str]] = []
    for path in sorted(directory.glob(f"*{DOC_SUFFIX}"), key=lambda item: item.name):
        text = path.read_text(encoding="utf-8")
        rows.append((path.stem, text))
    if not rows:
        raise CorpusError(f"手册目录 {directory} 里一份 {DOC_SUFFIX} 都没有。")
    return tuple(rows)


def subpackage_docstrings() -> tuple[tuple[str, str], ...]:
    """取每个子包的 ``__init__.py`` docstring，返回 ``(子包名, docstring)``（按名字排序）."""
    rows: list[tuple[str, str]] = []
    for name in discover_subpackages():
        module = importlib.import_module(f"{PACKAGE_NAME}.{name}")
        docstring = (module.__doc__ or "").strip()
        if not docstring:
            raise CorpusError(
                f"子包 {name!r} 没有 docstring：它的自述永远不会被检索到——"
                "要么补一段自述，要么把它排除在语料之外（不能两头都不做）。"
            )
        rows.append((name, docstring))
    return tuple(rows)


def build_corpus(
    *,
    documents: tuple[tuple[str, str], ...] | None = None,
    subpackages: tuple[tuple[str, str], ...] | None = None,
) -> Corpus:
    """收集两种语料，产出一份 :class:`Corpus`（**确定性**：两次调用逐位相同）.

    ``documents`` / ``subpackages`` 可注入（测试用它构造反例）。
    """
    doc_rows = read_documents() if documents is None else documents
    sub_rows = subpackage_docstrings() if subpackages is None else subpackages
    collected = [
        Document(name=qualify(DOC_KIND_DOC, raw), kind=DOC_KIND_DOC, text=text)
        for raw, text in doc_rows
    ]
    collected.extend(
        Document(name=qualify(DOC_KIND_SUBPACKAGE, raw), kind=DOC_KIND_SUBPACKAGE, text=text)
        for raw, text in sub_rows
    )
    return Corpus(documents=tuple(collected))


def expected_names(
    *,
    documents: tuple[tuple[str, str], ...] | None = None,
    subpackages: tuple[tuple[str, str], ...] | None = None,
) -> tuple[str, ...]:
    """返回"本该进语料的全部名字"（覆盖检查的参照名单）."""
    doc_rows = read_documents() if documents is None else documents
    sub_rows = subpackage_docstrings() if subpackages is None else subpackages
    names = [qualify(DOC_KIND_DOC, raw) for raw, _ in doc_rows]
    names.extend(qualify(DOC_KIND_SUBPACKAGE, raw) for raw, _ in sub_rows)
    return tuple(names)


def corpus_lines(corpus: Corpus | None = None) -> tuple[str, ...]:
    """把语料逐行印出来."""
    resolved = build_corpus() if corpus is None else corpus
    lines = ["语料表（两种材料：手册全文 + 子包 docstring）："]
    lines.extend("  " + document.line() for document in resolved.documents)
    lines.append("  " + resolved.line())
    return tuple(lines)


__all__ = [
    "DOCS_DIRNAME",
    "DOC_SUFFIX",
    "NAME_SEPARATOR",
    "PACKAGE_NAME",
    "PROJECT_ROOT",
    "Corpus",
    "Document",
    "build_corpus",
    "corpus_lines",
    "discover_subpackages",
    "docs_dir",
    "expected_names",
    "qualify",
    "read_documents",
    "subpackage_docstrings",
    "unqualify",
]
