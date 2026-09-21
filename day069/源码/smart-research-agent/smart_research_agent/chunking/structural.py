"""结构分块：按文档自己的标题层级切（M6-D2）.

前两种策略都在**纯文本**上工作：固定窗口不认识标点，递归策略认识标点
但不认识"这是第 3 章第 2 节"。而 day061 花了一整天把结构从八种格式里
抽出来，存进 ``Document.blocks`` —— 今天就是来花它的：

```text
blocks = [heading(1,"RAG 入门"), paragraph, code, heading(2,"检索"), paragraph, …]
                    │
                    ▼  按标题层级切小节
小节 1  (RAG 入门)           小节 2  (RAG 入门 > 检索)
  heading / paragraph          heading(2) / paragraph
```

## 标题路径（breadcrumb）是这一天最有价值的一列

每个块都带一条 ``heading_path``：``("RAG 入门", "检索")``。它的用途是
**给片段补上它自己说不出来的语境**：一句"阈值设为 0.85"单独看没有话题，
而它上面的两级标题才是"查缓存配置能命中它"的原因。因此它被拼进
``Chunk.retrieval_text``（检索视图），**但不进 ``Chunk.text``**——
后者必须能在原文里逐字找到（见 types.py 的取舍二）。

## 两条与"保护原子块"有关的设计

### 1. 代码块与表格不可切（``ATOMIC_KINDS``）

把一个函数从中间切开，得到的两半都是**语法不完整、语义不成立的文本**：
它们会各自被向量化、各自被检索到，然后拼进提示词里让模型读一段残码。
因此超预算的原子块**整块保留并标记 ``oversized``**。这是一个刻意的取舍：

```text
切开会得到两块「都不对」的文本     ← 更糟：错误很隐蔽
不切会得到一个超出预算的块         ← 更好：超预算是可见、可统计的
```

代价是那一块进提示词时可能被模型截断——所以 ``ChunkSet.oversized_count``
是一个必须被看见的数字（报告与端点都会打印它）。

### 2. 定位失败的小节退化为按段落切，并把块区间记 -1

"按标题切"需要一个前提：标题在 ``document.text`` 里的位置能找回来。
本模块的做法是把小节的块文本用 ``"\\n\\n"`` 拼起来，然后在全文里 ``find``：
**找得到就用精确路径**（块级打包，绝不在块内下刀），
**找不到就退化**（按段落切那一段范围，并把 ``start_block``/``end_block``
记为 ``-1``）。退化会被计数并写进 ``metadata.unlocated_sections``：

> **一次"没能精确定位"必须留下痕迹**，否则它只表现为"某些块的块区间
> 是 -1"，而下一个人根本不知道那是退化还是没实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from smart_research_agent.chunking.base import (
    Chunker,
    ChunkPolicy,
    Span,
    default_policy,
    paragraph_spans,
)
from smart_research_agent.chunking.errors import ChunkingError
from smart_research_agent.chunking.types import STRATEGY_STRUCTURAL
from smart_research_agent.documents.types import BLOCK_CODE, BLOCK_HEADING, BLOCK_TABLE, Document

#: 不可切的块类型：切开它们得到的不是"两段文本"，而是"两段残码/残表"。
ATOMIC_KINDS: tuple[str, ...] = (BLOCK_CODE, BLOCK_TABLE)


@dataclass(frozen=True)
class Section:
    """一个小节：一条标题路径 + 属于它的块下标（升序）.

    ``level = 0`` 表示"标题之前的内容"（前言）：它有块但没有标题，
    因此 ``heading_path`` 为空。**不把它丢掉**是刻意的——
    前言里经常放着整个文档的定位信息（"本文适用的版本是 …"）。
    """

    heading_path: tuple[str, ...]
    level: int
    title: str
    blocks: tuple[int, ...]


class StructuralChunker(Chunker):
    """结构分块器（标题切小节 + 小节内块级打包 + 原子块保护）."""

    name = STRATEGY_STRUCTURAL

    def __init__(self, policy: ChunkPolicy | None = None, **kwargs: Any) -> None:
        super().__init__(policy or default_policy(STRATEGY_STRUCTURAL), **kwargs)
        if self.policy.overlap_tokens:
            # 结构策略显式拒绝重叠（见 base.DEFAULT_POLICIES 的注释）：
            # 跨标题的重叠会把上一节的内容灌进下一节，而下一节的
            # heading_path 会让这些内容看起来"本来就属于这一节"——
            # 一个错误的来源标签比没有标签更糟。
            raise ChunkingError(
                "结构策略不支持重叠（overlap_tokens 必须为 0）："
                "标题边界就是语义边界，跨边界重叠会把上一节内容灌进下一节，"
                "而块的 heading_path 会让它看起来本来就属于这一节"
            )

    # ------------------------------------------------------------------ #
    # 切分
    # ------------------------------------------------------------------ #

    def _spans(self, document: Document) -> list[Span]:
        """切小节 → 定位 → 小节内打包（三步各自可单独测）."""
        if not document.text.strip():
            return []
        if not document.blocks:
            # 没有块序列（理论上不该发生：day061 的全文由块拼出），
            # 此时"结构"这个策略失去全部输入，退化为按段落切。
            self.stats = {"sections": "0", "degraded_reason": "no-blocks"}
            return self._paragraph_split(document.text, 0, len(document.text), ())
        sections = build_sections(document)
        ranges = self._locate(document, sections)
        spans: list[Span] = []
        unlocated = 0
        for section, (start, end, exact) in zip(sections, ranges):
            if not exact:
                unlocated += 1
            spans.extend(self._section_spans(document, section, start, end, exact))
        spans = self._pack(document.text, spans, "+section")
        self.stats = {
            "sections": str(len(sections)),
            "unlocated_sections": str(unlocated),
            "atomic_oversized": str(
                len([span for span in spans if span.oversized])
            ),
            "atomic_kinds": ",".join(ATOMIC_KINDS),
        }
        return spans

    def _locate(
        self, document: Document, sections: list[Section]
    ) -> list[tuple[int, int, bool]]:
        """把每个小节定位到 ``document.text`` 的一个区间上.

        两步：先顺序 ``find`` 精确路径（游标单调右移，因此不会把后一个小节
        定位到前一个小节之前）；**未命中的小节用前后已定位小节的空隙补齐**。

        补齐这一步是覆盖率的关键：小节区间**按构造拼满全文**，
        于是"原文每一个非空白字符都被覆盖"这条不变量与定位成功与否无关——
        **能靠构造保证的事不要靠运气。**
        """
        text = document.text
        located: list[tuple[int, int, bool]] = []
        cursor = 0
        for section in sections:
            joined = "\n\n".join(document.blocks[index].text for index in section.blocks)
            position = text.find(joined, cursor) if joined else -1
            if position >= 0:
                located.append((position, position + len(joined), True))
                cursor = position + len(joined)
                continue
            located.append((-1, -1, False))
        for index, (start, end, exact) in enumerate(located):
            if exact:
                continue
            previous_end = located[index - 1][1] if index > 0 else 0
            next_start = len(text)
            for later in located[index + 1 :]:
                if later[2]:
                    next_start = later[0]
                    break
            located[index] = (previous_end, max(previous_end, next_start), False)
        return located

    def _section_spans(
        self,
        document: Document,
        section: Section,
        start: int,
        end: int,
        exact: bool,
    ) -> list[Span]:
        """一个小节 → 若干区间（精确路径走块级打包，退化路径走段落切分）."""
        text = document.text
        if start >= end:
            return []
        if not exact:
            return self._paragraph_split(text, start, end, section.heading_path)
        offsets: list[tuple[int, int, int]] = []
        cursor = start
        for index in section.blocks:
            block_text = document.blocks[index].text
            offsets.append((cursor, cursor + len(block_text), index))
            cursor += len(block_text) + 2  # 块之间由 "\n\n" 相连（make_document）
        spans: list[Span] = []
        group: list[tuple[int, int, int]] = []
        for item in offsets:
            if group and not self._fits(text, group[0][0], item[1]):
                spans.append(self._emit_group(group, section))
                group = []
            if not group and not self._fits(text, item[0], item[1]):
                spans.extend(self._emit_single(document, item, section))
                continue
            group.append(item)
        if group:
            spans.append(self._emit_group(group, section))
        return spans

    def _emit_group(
        self, group: list[tuple[int, int, int]], section: Section
    ) -> Span:
        """一组装得下的块 → 一个区间（块区间精确可考）."""
        return Span(
            start_char=group[0][0],
            end_char=group[-1][1],
            reason=f"section:{section.title or '前言'}",
            heading_path=section.heading_path,
            start_block=group[0][2],
            end_block=group[-1][2],
        )

    def _emit_single(
        self, document: Document, item: tuple[int, int, int], section: Section
    ) -> list[Span]:
        """单个块就超预算时的两条出路：原子块整块保留，普通段落继续下切."""
        start, end, index = item
        kind = document.blocks[index].kind
        if kind in ATOMIC_KINDS:
            return [
                Span(
                    start_char=start,
                    end_char=end,
                    reason=f"atom:{kind}",
                    heading_path=section.heading_path,
                    start_block=index,
                    end_block=index,
                    oversized=True,
                )
            ]
        return self._paragraph_split(
            document.text, start, end, section.heading_path, start_block=index
        )

    def _paragraph_split(
        self,
        text: str,
        start: int,
        end: int,
        heading_path: tuple[str, ...],
        start_block: int = -1,
    ) -> list[Span]:
        """退化路径：按段落切，段落内部再装不下就硬切.

        它同时服务三处：定位失败的小节、单独的巨型段落块、以及
        "这份文档根本没有块序列"。**同一个退化路径服务三处**是刻意的——
        三处写三套会有三份不一样的边界行为，而它们本应是同一件事。
        """
        spans = [
            Span(
                start_char=left,
                end_char=right,
                reason="para",
                heading_path=heading_path,
                start_block=start_block,
                end_block=start_block,
            )
            for left, right in paragraph_spans(text, start, end)
        ]
        packed = self._pack(text, spans)
        result: list[Span] = []
        for span in packed:
            if self._fits(text, span.start_char, span.end_char):
                result.append(span)
                continue
            result.extend(
                self._hard_split(text, span.start_char, span.end_char, "oversized-para")
            )
        return result

    def describe(self) -> dict[str, Any]:
        """策略自述."""
        return {
            "name": self.name,
            "unit": "标题层级（小节）",
            "where_to_cut": "在标题处开新小节，小节内按块打包到预算",
            "parameters": ["max_tokens", "min_tokens", "measurer"],
            "uses_similarity_percentile": False,
            "supports_overlap": False,
            "atomic_kinds": list(ATOMIC_KINDS),
            "deterministic": "yes（无随机、无模型调用）",
            "strengths": [
                "块的边界与人的阅读单元一致（一节一段），引用时天然可定位",
                "每块带 heading_path，检索时能补上片段自己说不出的语境",
                "保护代码块与表格：超预算时整块保留，不产生残码",
            ],
            "weaknesses": [
                "完成依赖 day061 抽出的 blocks：纯文本输入下退化为段落切分",
                "没有标题的文档（如日志、FAQ）几乎等同于按段落切",
                "不支持重叠，段落被切断时两侧都读不到完整句",
            ],
            "cost": "重叠为 0，因此重复率恒为 0（成本最低）",
        }


def build_sections(document: Document) -> list[Section]:
    """按标题层级把块序列切成小节（**只依赖块序列，不依赖定位**）.

    标题栈的规则很简单：遇到 ``level`` 的标题时，把栈里所有
    ``level >= 当前 level`` 的项弹掉再压入——这正是 Markdown / HTML 的
    章节语义。**用栈而不是"记住上一个标题"**是必需的：一个 ``##`` 之后
    出现 ``###``，再出现另一个 ``##`` 时，第三级标题必须被弹掉，
    否则第二个 ``##`` 小节会带着上一个 ``###`` 的名字。
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    blocks: list[int] = []
    path: tuple[str, ...] = ()
    level = 0
    title = ""

    def flush() -> None:
        if blocks:
            sections.append(
                Section(
                    heading_path=path,
                    level=level,
                    title=title,
                    blocks=tuple(blocks),
                )
            )

    for index, block in enumerate(document.blocks):
        if block.kind == BLOCK_HEADING:
            flush()
            blocks = [index]
            heading_title = block.text.strip()
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            stack.append((block.level, heading_title))
            path = tuple(item[1] for item in stack)
            level = block.level
            title = heading_title
            continue
        blocks.append(index)
    flush()
    return sections


__all__ = ["ATOMIC_KINDS", "Section", "StructuralChunker", "build_sections"]
