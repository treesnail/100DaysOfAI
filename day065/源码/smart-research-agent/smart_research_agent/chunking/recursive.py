"""递归分块：按"语义强度递减"的分隔符表逐层下切（M6-D2）.

固定长度策略的致命伤是"不认识标点"。递归策略的全部内容就是给它加上一张
**分隔符优先级表**，然后从强到弱依次尝试：

```text
一段文本装不下预算
  ├─ 在 "\n\n"（段落边界）处切开 → 每一段装得下吗？装不下就 ↓
  ├─ 在 "\n"（换行）处切开   → 装不下就 ↓
  ├─ 在 "。" 处切开          → 装不下就 ↓
  ├─ 在 "，" 处切开          → 装不下就 ↓
  └─ 连空格都找不到          → 按字符硬切（最后的退路）
```

这就是 LangChain 的 ``RecursiveCharacterTextSplitter`` 的核心思想。
它之所以好用，是因为它把"优先在哪里切"这件本来需要模型判断的事，
退化成了一个**可以写下来的字符串表**——可读、可测、可复现。

## 三处必须说清楚的设计

### 1. 分隔符表必须含中文标点

只用英文标点（或只放 ``"\n"``）的表在中文文档上会**一路退到按字符硬切**：
中文的句末是 ``。``，而 ``, `` `. `` 都不是 ASCII。后果不是"切得差一点"，
而是"退化成固定长度分块"——而**退化的表现只是块边界变差，不会有任何报错**。
本模块的表里中英标点并列，正是为了避免这种沉默的退化
（``separator_histogram`` 可以把一份文档里每种分隔符的数量数出来，
"这份文档为什么退化了"因此可以当场核对）。

### 2. 分隔符保留在片段的**尾部**

``"第一句。第二句。"`` 按 ``。`` 切，得到的是 ``["第一句。", "第二句。"]``
而不是 ``["第一句", "第二句"]``。丢掉分隔符会让**产物再也拼不回原文**，
于是"块文本是原文的连续子串"这条不变量直接失效（见 types.py 的模块 docstring）。

### 3. 切完必须"合回去"（``+pack``）

第一刀在 ``\n\n`` 处切完之后，每个自然段各自成块——一份 200 段的文档
得到 200 个小块。递归切分的正确形状是"**先切碎、再按预算合并**"，
合并由 ``Chunker._pack`` 承担：相邻片段能装进同一预算就合成一块。
这一步决定了产物的平均块长，而它最容易被忘掉——忘了它，
块数会膨胀到与段落数同量级，embedding 账单同步翻倍。
"""

from __future__ import annotations

from typing import Any

from smart_research_agent.chunking.base import (
    Chunker,
    ChunkPolicy,
    Span,
    default_policy,
)
from smart_research_agent.chunking.errors import ChunkingError
from smart_research_agent.chunking.types import STRATEGY_RECURSIVE
from smart_research_agent.documents.types import Document

#: 分隔符表（**顺序就是优先级**）。最后一档是空串，表示"没有分隔符可用，
#: 按字符硬切"——把它写成表里的一行而不是一个 ``if``，是为了让
#: ``DEFAULT_SEPARATORS`` 本身就是一份完整的、可以整表打印的策略说明。
SEPARATOR_NOTES: tuple[tuple[str, str], ...] = (
    ("\n\n", "段落"),
    ("\n", "换行"),
    ("。", "中文句号"),
    ("！", "中文叹号"),
    ("？", "中文问号"),
    ("；", "中文分号"),
    (". ", "英文句号"),
    ("! ", "英文叹号"),
    ("? ", "英文问号"),
    ("; ", "英文分号"),
    ("，", "中文逗号"),
    (", ", "英文逗号"),
    (" ", "空格"),
    ("", "字符（无分隔符可用，硬切）"),
)

DEFAULT_SEPARATORS: tuple[str, ...] = tuple(item[0] for item in SEPARATOR_NOTES)

#: 分隔符 → 人类可读的名字（报告里的 ``reason`` 用它，如 ``sep:段落``）.
SEPARATOR_LABELS: dict[str, str] = {
    separator: label for separator, label in SEPARATOR_NOTES
}


def split_keep_separator(text: str, separator: str) -> list[str]:
    """按分隔符切分，但**把分隔符留在前一片的尾部**（见模块 docstring 第 2 条）.

    ``"a。b。"`` 按 ``。`` 切开得到 ``["a。", "b。"]``：拼回去等于原文。
    末尾的空片会被丢掉——它是分隔符造成的，不代表有内容。
    """
    if not separator:
        return [text]
    parts = text.split(separator)
    pieces = [part + separator for part in parts[:-1]]
    if parts[-1]:
        pieces.append(parts[-1])
    return pieces


def separator_histogram(text: str) -> list[dict[str, Any]]:
    """数一数这份文本里每种分隔符各出现多少次（按优先级排列）.

    它是"这份文档为什么退化成硬切"的答案：**``count=0`` 的那一档
    就是从这一层掉下去的原因**。报告里打印它，比打印一堆块长分布
    更能解释问题——**块长分布告诉你"结果不好"，直方图告诉你"为什么"。**
    """
    return [
        {
            "separator": separator,
            "label": label,
            "count": text.count(separator) if separator else 0,
        }
        for separator, label in SEPARATOR_NOTES
    ]


class RecursiveChunker(Chunker):
    """递归分块器（分隔符优先级表 + 预算合并 + 重叠回带）."""

    name = STRATEGY_RECURSIVE
    #: 重叠由收尾器统一施加（``finalize`` 的顺序：修剪 → 重叠 → 合并 → 校验）.
    apply_overlap = True

    def __init__(
        self,
        policy: ChunkPolicy | None = None,
        *,
        separators: tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy or default_policy(STRATEGY_RECURSIVE), **kwargs)
        self.separators = separators or DEFAULT_SEPARATORS
        # 重叠在**最后**统一回带（``_with_overlap``），因此这里先留出额度：
        # 核心片段按 max_tokens - overlap 切，回带之后合计不超过 max_tokens。
        self.window_budget = self.policy.max_tokens - self.policy.overlap_tokens
        self.window_overlap = 0
        if self.policy.overlap_tokens * 2 >= self.policy.max_tokens:
            # 这条护栏挡的是"重叠把块撑成上一块的复制品"：核心窗口只有
            # max - overlap 个单位，而回带 overlap 个单位——当
            # 2 × overlap >= max 时，后一块几乎整块都是前一块的内容，
            # 起点被推回上一块的起点附近，切分退化成"每块只前进几个字符"。
            # 真实表现是**块数暴涨 + 重复率接近 50%**，而参数看起来完全合法。
            raise ChunkingError(
                f"重叠 {self.policy.overlap_tokens} 相对预算 "
                f"{self.policy.max_tokens} 过大（要求 2 × overlap < max_tokens）："
                "核心窗口只有 max-overlap 个单位，回带之后每一块几乎都是上一块的复制品"
            )

    def _spans(self, document: Document) -> list[Span]:
        """递归下切（重叠由收尾器在修剪之后统一回带）."""
        text = document.text
        if not text.strip():
            return []
        return self._split_range(text, 0, len(text), self.separators)

    def _split_range(
        self, text: str, start: int, end: int, separators: tuple[str, ...]
    ) -> list[Span]:
        """把 ``[start, end)`` 切到每一片都装得进预算.

        递归的收敛性由两件事保证：**分隔符表有限**（用完就硬切）
        且**硬切每刀至少一个字符**（``_hard_split`` 的护栏）。
        没有这两条，一个不含任何分隔符的超长串会让递归永远进不去也出不来。
        """
        if start >= end:
            return []
        if self._fits(text, start, end):
            return [Span(start_char=start, end_char=end, reason="fits")]
        if not separators:
            return self._hard_split(text, start, end, "no-separator")
        separator, rest = separators[0], separators[1:]
        if not separator:
            return self._hard_split(text, start, end, "char")
        pieces = split_keep_separator(text[start:end], separator)
        if len(pieces) == 1:
            # 这一档分隔符在这段文本里不存在：直接降到下一档，
            # **不要在这里硬切**——硬切会跳过后面的优先档位。
            return self._split_range(text, start, end, rest)
        label = SEPARATOR_LABELS.get(separator, separator)
        spans: list[Span] = []
        cursor = start
        for piece in pieces:
            piece_start = cursor
            piece_end = cursor + len(piece)
            if self._fits(text, piece_start, piece_end):
                spans.append(
                    Span(
                        start_char=piece_start,
                        end_char=piece_end,
                        reason=f"sep:{label}",
                    )
                )
            else:
                spans.extend(self._split_range(text, piece_start, piece_end, rest))
            cursor = piece_end
        return self._pack(text, spans)

    def describe(self) -> dict[str, Any]:
        """策略自述（含分隔符表——它是这个策略真正的"参数"）."""
        return {
            "name": self.name,
            "unit": "分隔符优先级 + 预算",
            "where_to_cut": "按分隔符表从强到弱下切，再按预算合并相邻片段",
            "parameters": ["max_tokens", "overlap_tokens", "min_tokens", "measurer"],
            "uses_similarity_percentile": False,
            "supports_overlap": True,
            "deterministic": "yes（无随机、无模型调用）",
            "separators": [
                {"separator": separator, "label": label}
                for separator, label in SEPARATOR_NOTES
            ],
            "strengths": [
                "不切在句子中间的概率远高于固定长度策略",
                "不需要任何结构元数据，任何纯文本都能处理",
                "分隔符表就是全部参数，可读、可测、可复现",
            ],
            "weaknesses": [
                "不知道标题层级：标题与正文会被合进同一块（结构策略才认标题）",
                "分隔符表对语料有隐含假设，中英混排的表在纯代码语料上会退化",
                "合并是按预算贪心的，可能把两个话题的尾巴拼进同一块",
            ],
            "cost": "重叠放大量 = overlap/step（与固定长度同口径）",
        }


__all__ = [
    "DEFAULT_SEPARATORS",
    "SEPARATOR_LABELS",
    "SEPARATOR_NOTES",
    "RecursiveChunker",
    "separator_histogram",
    "split_keep_separator",
]
