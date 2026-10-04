"""HTML 加载器：用标准库 ``html.parser`` 把标签流还原成结构块（M6-D1）.

HTML 是 M6 里唯一的"**标记比内容多**"的格式。一份从网页另存的文档，
正文字符可能只有 3 KB，而标签有 30 KB；其中还有 ``<script>`` 里的
一整段 JavaScript 与 ``<style>`` 里的 CSS。如果不做剥离就入库，
检索"配置"这个词会命中页面的埋点脚本——**而那种命中看起来完全正常**。

## 为什么用 ``html.parser`` 而不是正则或第三方解析器

| 方案 | 问题 |
|------|------|
| 正则 | HTML 不是正则语言。``<p>`` 与 ``</p>`` 的配对、属性里的 ``>``、注释里的标签都会让它出错，而**出错的样子是"少了一段内容"** |
| BeautifulSoup / lxml | 引入依赖，且它们会顺手做很多我们不需要的事（编码嗅探、容错修复）——而"容错修复"会让"解析结果与原文哪里不同"这件事不可核对 |
| ``html.parser``（标准库） | 事件式（``handle_starttag`` / ``handle_endtag`` / ``handle_data``），**没有隐式修复**，行为完全可预测 |

标准库的代价是它**不做容错**：真实网页里常见的未闭合标签会让结构
还原得不够准。本模块的应对是"宁可少给结构，不可丢内容"——
``handle_endtag`` 时若当前没有打开对应块，就当作一次普通的块边界处理，
**已收集的文本照常落块**。

## 实体解码

``convert_charrefs=True``（默认）会把 ``&amp;`` / ``&#x4E2D;`` /
``&nbsp;`` 全部换成字符。这不是可选项：**中文网页里 ``&nbsp;`` 与
全角空格混用是常态**，不解码的话检索会拿一段 ``&nbsp;`` 去匹配。

## 表格

``<table>`` 里的 ``<tr>`` / ``<td>`` / ``<th>`` 被还原成 ``Block.rows``，
并且**与 Markdown 的表格共用一个渲染器**（``types.render_table``）：
三种格式的表格以同一种样子进提示词。
"""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any

from smart_research_agent.documents.base import MEDIA_HTML, DocumentLoader
from smart_research_agent.documents.text_loader import decode_bytes
from smart_research_agent.documents.types import (
    BLOCK_CODE,
    BLOCK_HEADING,
    BLOCK_LIST_ITEM,
    BLOCK_PARAGRAPH,
    BLOCK_QUOTE,
    BLOCK_TABLE,
    Block,
    Document,
    make_document,
    render_table,
)

#: 内容整段跳过的标签：**它们里面的东西不是给人读的**。
#: ``script`` 与 ``style`` 是显而易见的两项；``noscript`` 在启用 JS 的
#: 场景下用户根本看不到；``template`` 是未实例化的模板。
SKIP_TAGS: tuple[str, ...] = ("script", "style", "noscript", "template")

#: 标题标签 → 层级。
HEADING_TAGS: dict[str, int] = {f"h{level}": level for level in range(1, 7)}

#: 会结束当前块的容器标签（遇到它们时先落块，避免段落被粘成一条）.
CONTAINER_TAGS: tuple[str, ...] = (
    "div",
    "section",
    "article",
    "header",
    "footer",
    "main",
    "nav",
    "aside",
    "figure",
    "figcaption",
    "dl",
    "dt",
    "dd",
)


class HtmlStructureParser(HTMLParser):
    """把 HTML 事件流还原成块序列（一次通过，不建 DOM）.

    不建 DOM 的原因很实际：一份网页另存的文件可以被解析成几万个节点，
    而我们只关心"块边界在哪"。事件式解析让内存占用与标签深度无关。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self.title = ""
        self.lang = ""
        #: 当前正在收集的块（空串表示不在块内，文本落在"自由文本"缓冲里）
        self._kind = ""
        self._level = 0
        self._buffer: list[str] = []
        self._skip_depth = 0
        self._list_depth = 0
        self._in_pre = False
        self._in_title = False
        #: 表格状态：行 × 单元格
        self._rows: list[tuple[str, ...]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    # ------------------------------------------------------------------ 收集
    def _append(self, text: str) -> None:
        """把一段文本放进当前收集位置（单元格优先，其次块缓冲）."""
        if self._cell is not None:
            self._cell.append(text)
            return
        self._buffer.append(text)

    def _flush(self) -> None:
        """落一个块（按当前 ``_kind``）."""
        text = "".join(self._buffer).strip()
        self._buffer.clear()
        kind, self._kind = self._kind, ""
        level = self._level
        self._level = 0
        if not text:
            return
        # 标题与列表项**必须带上层级**：``Block`` 会校验标题层级落在 1~6，
        # 而漏传 level 的表现是一条"标题层级必须是 1~6，收到 0"的报错——
        # 那份文档会整份解析失败（本课第一次跑冒烟时就是这样丢了一整个 HTML）。
        if kind in (BLOCK_HEADING, BLOCK_LIST_ITEM):
            self.blocks.append(Block(kind=kind, text=text, level=level))
        else:
            self.blocks.append(Block(kind=kind or BLOCK_PARAGRAPH, text=text))

    def _open(self, kind: str, *, level: int = 0, force: bool = False) -> None:
        """开始一个新块（先把上一个落掉）.

        ``force=True`` 用于**同一类型的相邻块**：``<ul><li>外层<ul><li>内层``
        里两个 ``li`` 之间没有结束标签，因此"类型不同才落块"的规则会让它们
        粘成一个块（本课第一版就是这样，嵌套列表只剩一层）。
        相邻的 ``<p>`` 不会遇到这个问题——它们之间一定有一个 ``</p>``。
        """
        if force or (self._kind and self._kind != kind):
            self._flush()
        self._kind = kind
        self._level = level

    # ------------------------------------------------------------------ 事件
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if lowered == "html":
            for name, value in attrs:
                if name.lower() == "lang" and value:
                    self.lang = value
            return
        if lowered == "title":
            self._in_title = True
            return
        if lowered == "br":
            self._append("\n")
            return
        if lowered == "hr":
            self._flush()
            return
        if lowered in HEADING_TAGS:
            self._open(BLOCK_HEADING, level=HEADING_TAGS[lowered], force=True)
            return
        if lowered == "p":
            self._open(BLOCK_PARAGRAPH)
            return
        if lowered in ("ul", "ol"):
            self._list_depth += 1
            return
        if lowered == "li":
            self._open(BLOCK_LIST_ITEM, level=max(self._list_depth - 1, 0), force=True)
            return
        if lowered == "blockquote":
            self._open(BLOCK_QUOTE)
            return
        if lowered == "pre":
            self._in_pre = True
            self._open(BLOCK_CODE)
            return
        if lowered == "table":
            self._flush()
            self._rows = []
            return
        if lowered == "tr" and self._rows is not None:
            self._row = []
            return
        if lowered in ("td", "th") and self._row is not None:
            self._cell = []
            return
        if lowered in CONTAINER_TAGS:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in SKIP_TAGS:
            self._skip_depth = max(self._skip_depth - 1, 0)
            return
        if self._skip_depth:
            return
        if lowered == "title":
            self._in_title = False
            return
        if lowered in HEADING_TAGS or lowered in ("p", "li", "blockquote", "pre"):
            if lowered == "pre":
                self._in_pre = False
            self._flush()
            return
        if lowered in ("ul", "ol"):
            self._list_depth = max(self._list_depth - 1, 0)
            self._flush()
            return
        if lowered in ("td", "th") and self._cell is not None:
            value = "".join(self._cell).strip()
            self._cell = None
            if self._row is not None:
                self._row.append(value)
            return
        if lowered == "tr" and self._row is not None:
            # 空行（只含空白单元格）不进 rows：它在渲染出的 Markdown 表格里
            # 是一行全空，对读者与模型都是噪声。
            if any(cell for cell in self._row) and self._rows is not None:
                self._rows.append(tuple(self._row))
            self._row = None
            return
        if lowered == "table":
            rows = tuple(self._rows or ())
            self._rows = None
            if rows:
                self.blocks.append(
                    Block(kind=BLOCK_TABLE, text=render_table(rows), rows=rows)
                )
            return
        if lowered in CONTAINER_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        if not data.strip() and self._cell is None:
            # 只有空白的数据**在块内**也要保留（它是词与词之间的分隔），
            # 但不在块内时它只是标签之间的排版空白，丢掉。
            if self._kind:
                self._append(" ")
            return
        self._append(data)

    # ------------------------------------------------------------------ 收尾
    def close(self) -> None:  # noqa: D102 - 继承父类文档
        super().close()
        self._flush()


class HtmlLoader(DocumentLoader):
    """HTML 加载器：剥离脚本与样式，还原标题 / 段落 / 列表 / 引用 / 代码 / 表格."""

    media_types: tuple[str, ...] = (MEDIA_HTML,)
    suffixes: tuple[str, ...] = (".html", ".htm")
    description: str = "HTML：标准库事件式解析，脚本与样式整体跳过，表格保留行列"

    def load_bytes(self, data: bytes, *, source: str, media_type: str) -> Document:
        """解析 HTML 字节（先按纯文本的规则解码，因此同样受编码链保护）."""
        decoded = decode_bytes(data, source=source)
        parser = HtmlStructureParser()
        parser.feed(decoded.text)
        parser.close()
        blocks = list(parser.blocks)
        metadata = {
            "encoding": decoded.encoding,
            "encoding_confident": "true" if decoded.confident else "false",
            "lang": parser.lang,
            "title": parser.title.strip(),
            "table_count": str(sum(1 for block in blocks if block.kind == BLOCK_TABLE)),
        }
        return make_document(
            source=source,
            media_type=media_type,
            blocks=blocks,
            metadata={key: value for key, value in metadata.items() if value},
        )


def html_rules() -> list[dict[str, Any]]:
    """HTML 解析规则的对照表（报告与 ``/documents/loaders`` 端点同源）."""
    return [
        {
            "rule": "整体跳过",
            "tags": ", ".join(SKIP_TAGS),
            "reason": "它们里面的东西不是给人读的：埋点脚本会让「配置」这个词命中整个页面",
        },
        {
            "rule": "标题",
            "tags": "h1 ~ h6",
            "reason": "与 Markdown 的 ATX 标题收敛到同一个块类型，供结构分块使用",
        },
        {
            "rule": "段落与列表",
            "tags": "p / li（ul、ol 记层级）",
            "reason": "段落是正文的基本单位；列表的层级信息在 day062 分块时有用",
        },
        {
            "rule": "引用与代码",
            "tags": "blockquote / pre",
            "reason": "代码块必须整块保留，day062 不该把它从中间切开",
        },
        {
            "rule": "表格",
            "tags": "table / tr / td / th",
            "reason": "行列结构保留在 rows 里，text 与 Markdown 表格用同一个渲染器",
        },
        {
            "rule": "容器",
            "tags": ", ".join(CONTAINER_TAGS[:6]) + " …",
            "reason": "只作为块边界（先落块），不产生独立的块——它们没有语义内容",
        },
    ]


__all__ = [
    "CONTAINER_TAGS",
    "HEADING_TAGS",
    "SKIP_TAGS",
    "HtmlLoader",
    "HtmlStructureParser",
    "html_rules",
]
