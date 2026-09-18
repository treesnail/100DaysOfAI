"""Markdown 加载器：把标题、代码块、表格、列表解析成结构块（M6-D1）.

Markdown 是 M6 里**唯一一种"结构本来就写在文本里"的格式**——标题是 ``#``、
代码块是围栏、表格是管道符。因此它值得被认真解析：把结构读出来之后，
day062 的分块器可以按标题切、把整段代码留在一块里、把表格整张留给模型。

## 解析依据

| 结构 | 规范条款 |
|------|---------|
| ATX 标题 | CommonMark v0.31.2 §4.2（``#`` 到 ``######``，**``#`` 后必须有空格**） |
| 围栏代码块 | CommonMark v0.31.2 §4.5（````` ``` ````` 或 ``~~~``，可带信息串） |
| 表格 | GFM v0.29-gfm §4.10 Tables（表头 + 分隔行 + 数据行） |
| YAML front matter | Jekyll / Hugo 的约定（文件**开头**由 ``---`` 包围） |

## 三处刻意的取舍

### 1. 只做"结构抽取"，不做"Markdown 渲染"

本模块**不生成 HTML、不处理行内标记**（``**粗体**``、``[链接](url)`` 原样保留）。
理由：M6 的下游是分块与检索，它们要的是"这句话在哪一段、哪一节"，
而不是"这句话渲染出来长什么样"。做渲染会引入一整套 HTML 与转义逻辑，
而它的输出对检索毫无帮助。

行内标记原样保留还有一个副作用：**引用时能给出原文**——
day069 的 RAG 回答里若引用了包含 ``**重点**`` 的句子，保留标记比丢掉它更忠实。

### 2. 未识别的行**不会被丢掉**

任何不匹配已知结构的行都归入当前段落（或另起一个段落）。
"结构化解析"最容易犯的错是**静默丢内容**：一段没被识别的文字消失了，
而下游只会看到"这份文档比原文短"，谁也不知道少了什么。
本模块的规则是**只分类、不丢弃**，这一点由
``test_every_non_blank_line_survives_parsing`` 守着。

### 3. front matter 进元数据，不进正文

``---`` 包围的 YAML 头是**元数据**（title / tags / date），
把它当正文会让每一份文档的前几行都是同一堆字段名，而检索"tags"这种词
会命中所有文档。本模块把它切出来解析成 ``key=value`` 放进
``Document.metadata``，并且**不引入 YAML 解析器**——只识别
``key: value`` 与 ``key: [a, b]`` 这两种最常见的形式，其余原样保留成一个
``front_matter_raw`` 字段。理由与整门课的纪律一致：**不引入一个只为了
读六行头的依赖**。
"""

from __future__ import annotations

import re
from typing import Any

from smart_research_agent.documents.base import MEDIA_MARKDOWN, DocumentLoader
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

#: ATX 标题：``#{1,6}`` + **一个空格** + 文本（CommonMark §4.2）。
#: 那个空格是规范要求，也是实践中的分界线：``#标签`` 是话题标签而不是标题
#: （中文技术文档里 ``#话题`` 很常见），把它当标题会让文档结构凭空多出一节。
ATX_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")

#: 围栏代码块的起始行：````` ``` `` `` 或 ``~~~`` 后跟可选信息串（§4.5）。
FENCE_OPEN = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([^`]*)$")

#: GFM 表格分隔行：``|---|:---:|`` 这类（§4.10）。
TABLE_DELIMITER = re.compile(r"^[ \t]*\|?[ \t]*:?-{1,}:?[ \t]*(\|[ \t]*:?-{1,}:?[ \t]*)*\|?[ \t]*$")

#: 无序列表项：``- `` / ``* `` / ``+ ``。
BULLET_ITEM = re.compile(r"^([ \t]*)([-*+])[ \t]+(.*)$")

#: 有序列表项：``1. `` / ``1) ``。
ORDERED_ITEM = re.compile(r"^([ \t]*)(\d{1,9})[.)][ \t]+(.*)$")

#: 引用行：``> ``。
QUOTE_LINE = re.compile(r"^[ \t]*>[ \t]?(.*)$")

#: front matter 的分隔线（Jekyll 与 Hugo 都用三个短横）。
FRONT_MATTER_DELIMITER = "---"


def split_front_matter(text: str) -> tuple[dict[str, str], str]:
    """切出 YAML front matter，返回 ``(元数据, 剩余正文)``.

    只有"文件第一行就是 ``---``"才当作 front matter——这与 Jekyll 的约定
    一致（front matter 必须是**文件最开头**）。不这样限定的话，
    正文中间的一条水平分割线会被当成 front matter 的开始，
    而它后面的全部内容都会被吞进元数据里。
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != FRONT_MATTER_DELIMITER:
        return {}, text
    for index in range(1, len(lines)):
        if lines[index].strip() == FRONT_MATTER_DELIMITER:
            raw = "\n".join(lines[1:index])
            rest = "\n".join(lines[index + 1 :])
            metadata = parse_front_matter(raw)
            return metadata, rest
    # 没有闭合线：**整份文件都不是 front matter**，按普通正文处理。
    # 否则一份以 `---` 开头的普通文档会被整份吞掉（而那是"解析成功但内容为空"
    # 最典型的一种成因）。
    return {}, text


def parse_front_matter(raw: str) -> dict[str, str]:
    """解析 front matter 的简形式（``key: value`` 与 ``key: [a, b]``）.

    不引 YAML 解析器的理由写在模块 docstring 里。这里只做两件事：

    - ``key: value`` → ``{key: value}``（去引号）；
    - ``key: [a, b]`` → ``{key: "a, b"}``（**列表统一成逗号分隔的字符串**，
      因为 ``Document.metadata`` 的值类型是 ``str``，"元数据的形状只有一种"
      比"有时是列表"更容易被下游消费）。

    其余形式（嵌套、多行块）原样保留进 ``front_matter_raw``。
    """
    metadata: dict[str, str] = {}
    unparsed: list[str] = []
    for line in raw.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            unparsed.append(stripped)
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            unparsed.append(stripped)
            continue
        if value.startswith("[") and value.endswith("]"):
            value = ", ".join(
                item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()
            )
        metadata[key] = value.strip("\"'")
    if unparsed:
        metadata["front_matter_raw"] = "\n".join(unparsed)
    return metadata


def _split_table_row(line: str) -> tuple[str, ...]:
    """把一行表格按管道符切成单元格（去掉两端多余的管道符）."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return tuple(cell.strip() for cell in stripped.split("|"))


class MarkdownLoader(DocumentLoader):
    """Markdown 加载器：front matter + 六类结构块."""

    media_types: tuple[str, ...] = (MEDIA_MARKDOWN,)
    suffixes: tuple[str, ...] = (".md", ".markdown")
    description: str = "Markdown：ATX 标题 / 围栏代码块 / GFM 表格 / 列表 / 引用"

    def load_bytes(self, data: bytes, *, source: str, media_type: str) -> Document:
        """解析 Markdown 字节（先按纯文本的规则解码）."""
        decoded = decode_bytes(data, source=source)
        blocks = self.parse(decoded.text)
        front_matter, _ = split_front_matter(decoded.text)
        metadata = {
            "encoding": decoded.encoding,
            "encoding_confident": "true" if decoded.confident else "false",
            "has_front_matter": "true" if front_matter else "false",
            "heading_count": str(sum(1 for block in blocks if block.kind == BLOCK_HEADING)),
        }
        metadata.update({f"front_matter.{key}": value for key, value in front_matter.items()})
        return make_document(
            source=source, media_type=media_type, blocks=blocks, metadata=metadata
        )

    def parse(self, text: str) -> list[Block]:
        """把 Markdown 文本解析成块序列（**只分类，不丢弃**）."""
        _, body = split_front_matter(text)
        blocks: list[Block] = []
        paragraph: list[str] = []

        def flush() -> None:
            if paragraph:
                blocks.append(Block(kind=BLOCK_PARAGRAPH, text="\n".join(paragraph)))
                paragraph.clear()

        lines = body.split("\n")
        index = 0
        while index < len(lines):
            line = lines[index]
            fence = FENCE_OPEN.match(line)
            if fence:
                flush()
                marker = fence.group(1)[0] * 3
                language = fence.group(2).strip()
                code: list[str] = []
                index += 1
                while index < len(lines) and not lines[index].strip().startswith(marker):
                    code.append(lines[index])
                    index += 1
                index += 1  # 跳过收尾围栏（缺失时也会跳过一行，循环随即结束）
                # 去掉尾部空行：文件末尾的换行会让收集结果多一个空元素，
                # 而"``` 未闭合"与"``` 后面恰好有一个空行"是两件事。
                while code and not code[-1].strip():
                    code.pop()
                blocks.append(
                    Block(kind=BLOCK_CODE, text="\n".join(code), language=language)
                )
                continue

            heading = ATX_HEADING.match(line)
            if heading:
                flush()
                blocks.append(
                    Block(
                        kind=BLOCK_HEADING,
                        text=heading.group(2).strip(),
                        level=len(heading.group(1)),
                    )
                )
                index += 1
                continue

            if (
                "|" in line
                and index + 1 < len(lines)
                and TABLE_DELIMITER.match(lines[index + 1])
                and "|" in lines[index + 1]
            ):
                flush()
                rows = [_split_table_row(line)]
                index += 2  # 表头 + 分隔行
                while index < len(lines) and "|" in lines[index] and lines[index].strip():
                    rows.append(_split_table_row(lines[index]))
                    index += 1
                blocks.append(
                    Block(
                        kind=BLOCK_TABLE, text=render_table(rows), rows=tuple(rows)
                    )
                )
                continue

            quote = QUOTE_LINE.match(line)
            if quote:
                flush()
                collected = [quote.group(1)]
                index += 1
                while index < len(lines):
                    follow = QUOTE_LINE.match(lines[index])
                    if not follow:
                        break
                    collected.append(follow.group(1))
                    index += 1
                blocks.append(Block(kind=BLOCK_QUOTE, text="\n".join(collected)))
                continue

            bullet = BULLET_ITEM.match(line) or ORDERED_ITEM.match(line)
            if bullet:
                flush()
                blocks.append(
                    Block(
                        kind=BLOCK_LIST_ITEM,
                        text=bullet.group(3).strip(),
                        level=len(bullet.group(1).replace("\t", "    ")) // 2,
                    )
                )
                index += 1
                continue

            if line.strip():
                paragraph.append(line)
            else:
                flush()
            index += 1
        flush()
        return blocks


def markdown_syntax_table() -> list[dict[str, Any]]:
    """Markdown 结构与其规范条款的对照表（报告与端点同源）."""
    return [
        {
            "structure": "ATX 标题",
            "syntax": "# 到 ###### （**# 后必须有空格**）",
            "spec": "CommonMark v0.31.2 §4.2",
            "block": BLOCK_HEADING,
        },
        {
            "structure": "围栏代码块",
            "syntax": "``` 或 ~~~ 起止，可带信息串作为语言",
            "spec": "CommonMark v0.31.2 §4.5",
            "block": BLOCK_CODE,
        },
        {
            "structure": "表格",
            "syntax": "管道符行 + 分隔行（|---|）",
            "spec": "GFM v0.29-gfm §4.10",
            "block": BLOCK_TABLE,
        },
        {
            "structure": "列表项",
            "syntax": "- / * / + 与 1. / 1)",
            "spec": "CommonMark v0.31.2 §5.2 / §5.3",
            "block": BLOCK_LIST_ITEM,
        },
        {
            "structure": "引用",
            "syntax": "> 开头",
            "spec": "CommonMark v0.31.2 §5.1",
            "block": BLOCK_QUOTE,
        },
        {
            "structure": "front matter",
            "syntax": "文件最开头的 --- 包围块",
            "spec": "Jekyll / Hugo 约定",
            "block": "（进 metadata，不进正文）",
        },
    ]


__all__ = [
    "ATX_HEADING",
    "BULLET_ITEM",
    "FENCE_OPEN",
    "FRONT_MATTER_DELIMITER",
    "MarkdownLoader",
    "ORDERED_ITEM",
    "QUOTE_LINE",
    "TABLE_DELIMITER",
    "markdown_syntax_table",
    "parse_front_matter",
    "split_front_matter",
]
