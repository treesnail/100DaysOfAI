"""``index``：把语料编成一份**倒排索引**（day101）.

一份索引只做三件事：

```text
分词   正文 → token 序列（ASCII 词 + 中文二元组，见 types 的第二节）
倒排   token → 出现它的文档与次数（按名字排序，去掉一切顺序不确定性）
复算   同一份语料两次构建，索引的摘要必须逐位相同
```

## 一、今天最值钱的一句话

> **索引是中间物：它的错只在检索时才显形，而那时现象是"某个查询少了一条命中"——
> 看起来像查询的问题。因此索引必须自己能被人单独复算。**

:func:`index_digest` 与 :func:`course_index.verify.check_index_is_reproducible` 是这件事的两端：
前者给出一个指纹，后者把它做成一条性质。day095 把"两相写成一相"变成可读读数、
day100 把"剧本重放"变成一条 `==`，今天是同一条纪律的第三次应用。

## 二、一条纪律：**并列**要有一个确定的顺序

```text
token → ({doc_a: 2, doc_b: 1})   ⇒  postings 里的文档名必须**排序**
正文明明一样，但两次构建给出不同的 postings ⇒ "索引可复算"这条性质永远失败
```

因此本模块在任何"集合 → 序列"的地方都显式排序；`repr` 出来的东西因此是稳定的。

## 三、与既有包的接缝

- **上游**：:mod:`course_index.corpus`（语料）、:mod:`course_index.types`（分词口径）；
- **下游**：:mod:`course_index.query` 在它上面检索，
  :mod:`course_index.study` 打印索引表。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from smart_research_agent.course_index.corpus import Corpus, build_corpus
from smart_research_agent.course_index.errors import IndexBuildError, NumericError, ParameterError
from smart_research_agent.course_index.types import MIN_WORD_LEN, NGRAM

#: 一个 token 的两种形态：ASCII / 数字 / 下划线组成的词，或连续的中日韩表意文字.
#:
#: 用**一个**交替模式（而不是两个 `findall`）是刻意的：两个独立的正则各扫一遍，
#: 会把"先 ASCII 后中文"这个与正文无关的顺序写进 token 序列，
#: 而 `query_tokens` / `matched` 是会被印出来的——它们必须跟着正文走。
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]+")

#: 索引摘要取前多少位十六进制（与 day099 / day100 同一口径）.
INDEX_DIGEST_LENGTH = 16


def tokenize(text: str) -> tuple[str, ...]:
    """把一段正文切成 token 序列（**确定性、零依赖、按正文顺序**）.

    规则只有两条：

    ```text
    ASCII 词     ``[A-Za-z0-9_]+``，小写化，长度 >= MIN_WORD_LEN
    中文串       长度 >= NGRAM 时切成全部相邻 NGRAM 元组；更短的整段作为一个 token
    ```
    """
    if not isinstance(text, str):
        raise ParameterError("tokenize 只接受字符串。")
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text):
        piece = match.group(0)
        if piece[0].isascii():
            lowered = piece.lower()
            if len(lowered) >= MIN_WORD_LEN:
                tokens.append(lowered)
        elif len(piece) < NGRAM:
            tokens.append(piece)
        else:
            tokens.extend(piece[i : i + NGRAM] for i in range(len(piece) - NGRAM + 1))
    return tuple(tokens)


def token_count(text: str) -> int:
    """一段正文切出来的 token 个数（性质 ⑥ 读它）."""
    return len(tokenize(text))


@dataclass(frozen=True)
class InvertedIndex:
    """一份倒排索引：文档名单 + 每份文档的词数 + ``token → ((文档, 次数), ...)``.

    ``postings`` 是**排序后**的元组元组：它是"两份索引逐位相同"这条性质比较的东西。
    """

    documents: tuple[str, ...]
    tokens_per_document: tuple[tuple[str, int], ...]
    postings: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]

    def __post_init__(self) -> None:
        if not self.documents:
            raise IndexBuildError("索引不能没有文档。")
        if len(set(self.documents)) != len(self.documents):
            raise IndexBuildError(f"索引里的文档名不唯一：{sorted(self.documents)}。")
        if tuple(name for name, _ in self.tokens_per_document) != self.documents:
            raise IndexBuildError(
                "tokens_per_document 的文档顺序与 documents 不一致："
                "两处各写一遍顺序，迟早会让'某份文档有多少词'读成邻居的数。"
            )
        if any(count < 0 for _, count in self.tokens_per_document):
            raise NumericError("每份文档的词数不能为负。")
        for token, pairs in self.postings:
            if not token:
                raise ParameterError("倒排表里出现空 token。")
            names = [name for name, _ in pairs]
            if names != sorted(names):
                raise IndexBuildError(
                    f"token {token!r} 的文档名没有排序：{names}——"
                    "并列的不确定性会让'两次构建逐位相同'永远失败。"
                )
            if any(count < 1 for _, count in pairs):
                raise NumericError(f"token {token!r} 的计数必须 >= 1。")

    # ------------------------------------------------------------------ 只读视图

    @property
    def document_count(self) -> int:
        """文档数."""
        return len(self.documents)

    @property
    def term_count(self) -> int:
        """不同的 token 个数（索引的"词典"大小）."""
        return len(self.postings)

    @property
    def total_tokens(self) -> int:
        """全部文档的 token 总数."""
        return sum(count for _, count in self.tokens_per_document)

    def postings_of(self, token: str) -> tuple[tuple[str, int], ...]:
        """取一个 token 的倒排表（没有这个词时返回空元组）."""
        for candidate, pairs in self.postings:
            if candidate == token:
                return pairs
        return ()

    def documents_containing(self, token: str) -> tuple[str, ...]:
        """取包含某个 token 的文档名."""
        return tuple(name for name, _ in self.postings_of(token))

    def tokens_of_document(self, name: str) -> int:
        """取某份文档的词数（不在索引里当场拒绝）."""
        for candidate, count in self.tokens_per_document:
            if candidate == name:
                return count
        raise ParameterError(f"索引里没有文档 {name!r}。")

    # ------------------------------------------------------------------ 复算口径

    def comparable(self) -> tuple[Any, ...]:
        """**只含确定性字段**的元组（两份索引逐位比较它）."""
        return (self.documents, self.tokens_per_document, self.postings)

    def digest(self) -> str:
        """索引摘要（两次构建摘要相同 ⇔ 逐位相同）."""
        return hashlib.sha256(repr(self.comparable()).encode("utf-8")).hexdigest()[
            :INDEX_DIGEST_LENGTH
        ]

    def diff_count(self, other: InvertedIndex) -> int:
        """与另一份索引在复算口径上**差了几项**（0 = 逐位相同）."""
        left = self.comparable()
        right = other.comparable()
        return sum(1 for a, b in zip(left, right, strict=True) if a != b)

    def to_dict(self) -> dict[str, Any]:
        """摊平成可 JSON 化的字段."""
        return {
            "document_count": self.document_count,
            "term_count": self.term_count,
            "total_tokens": self.total_tokens,
            "digest": self.digest(),
            "documents": list(self.documents),
        }

    def line(self) -> str:
        """一行读数：``索引：66 份文档 | 词典 12345 个词 | 词次 67890 | 摘要 xxxx``."""
        return (
            f"索引：{self.document_count} 份文档 | 词典 {self.term_count} 个词"
            f" | 词次 {self.total_tokens} | 摘要 {self.digest()}"
        )


def build_index(corpus: Corpus | None = None) -> InvertedIndex:
    """把语料编成一份 :class:`InvertedIndex`（**确定性**）.

    ``corpus`` 缺省时现场收一份语料；测试可以注入一份小语料。
    """
    resolved = build_corpus() if corpus is None else corpus
    counters: list[tuple[str, dict[str, int]]] = []
    for document in resolved.documents:
        counts: dict[str, int] = {}
        for token in tokenize(document.text):
            counts[token] = counts.get(token, 0) + 1
        counters.append((document.name, counts))
    terms = sorted({token for _, counts in counters for token in counts})
    postings: list[tuple[str, tuple[tuple[str, int], ...]]] = []
    for token in terms:
        pairs = tuple(
            sorted((name, counts[token]) for name, counts in counters if token in counts)
        )
        postings.append((token, pairs))
    return InvertedIndex(
        documents=tuple(document.name for document in resolved.documents),
        tokens_per_document=tuple((name, sum(counts.values())) for name, counts in counters),
        postings=tuple(postings),
    )


def index_digest(index: InvertedIndex | None = None) -> str:
    """索引摘要（缺省时现场建一份索引）."""
    resolved = build_index() if index is None else index
    return resolved.digest()


def require_reproducible(first: InvertedIndex, second: InvertedIndex) -> InvertedIndex:
    """两份索引不一致时抛 :class:`IndexBuildError`（"拒绝交付"的那条路）."""
    diff = first.diff_count(second)
    if diff == 0:
        return first
    raise IndexBuildError(
        f"同一份语料编出的两份索引差了 {diff} 项——"
        "索引里混进了未固定的量（集合序 / 字典序），它还不是一份可以被别人重算的中间物。"
    )


def index_lines(index: InvertedIndex | None = None, *, limit: int | None = None) -> tuple[str, ...]:
    """把索引逐行印出来（词典前若干条 + 汇总行）."""
    resolved = build_index() if index is None else index
    lines = ["倒排索引表（token → 出现它的文档与次数）："]
    chosen = resolved.postings if limit is None else resolved.postings[:limit]
    for token, pairs in chosen:
        rendered = "、".join(f"{name}={count}" for name, count in pairs[:4])
        more = "" if len(pairs) <= 4 else f"（+{len(pairs) - 4}）"
        lines.append(f"  {token:<24} | 文档 {len(pairs):>3} | {rendered}{more}")
    lines.append("  " + resolved.line())
    return tuple(lines)


__all__ = [
    "INDEX_DIGEST_LENGTH",
    "TOKEN_PATTERN",
    "InvertedIndex",
    "build_index",
    "index_digest",
    "index_lines",
    "require_reproducible",
    "token_count",
    "tokenize",
]
