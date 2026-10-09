"""DOCX 加载器：用 ``zipfile`` + ``xml.etree`` 直接读 OOXML（M6-D1）.

``.docx`` 的本质是一个 ZIP 包，正文在 ``word/document.xml`` 里。
Microsoft 官方文档给出的最小结构（引 ISO/IEC 29500）是：

```xml
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>文本在这里</w:t></w:r></w:p>
  </w:body>
</w:document>
```

其中 ``w:p`` 是段落、``w:r`` 是"格式相同的一段文本"（run）、``w:t`` 是文本本身。
命名空间 URI 就是上面那个，**必须按完整 URI 匹配**——``ElementTree`` 里
标签名是 ``{uri}local`` 的形式，用 ``"w:t"`` 去查会一个都找不到
（而"找不到"的表现是**解析成功但内容为空**，最难发现的一类）。

## 为什么不用 python-docx

python-docx 当前的最新版是 **1.2.0**（2025-06-17）。它很好用，但它把
"读取"与"生成/修改"绑在一起，而那正是本模块不需要的部分：
我们只要读三个标签，并且要能**离线、零依赖**地跑在 CI 里。
这与 day050 用 `CharTokenizer` 而不是 `transformers` 的分词器、
day060 用公开价格常量而不是账单 API 是同一种取舍：
**需要的能力边界只有一点点时，自己实现的那一点点比引入一整个依赖更可控。**

## 三处刻意的处理

1. **只读 ``word/document.xml``**：页眉页脚（``header1.xml``）、脚注、
   批注、文本框里的内容**读不到**。这是本模块的已知边界，写进元数据
   （``docx_parts_read``）而不是假装没有——**"没读到的部分"最容易被
   当成"文档里没有"**；
2. **``w:t`` 之间不加分隔符**：同一个段落里的多个 run 是"同一句话的
   不同格式片段"（例如"**注意**：……"），它们之间本来就没有空格，
   加分隔符会把一个词从中间切开；
3. **``w:tab`` / ``w:br`` 会变成空格与换行**：它们在 XML 里是独立标签、
   没有文本内容，不显式处理就会静默丢掉——而丢掉之后
   "第 1 行"与"第 2 行"会粘成"第 1 行第 2 行"。

## 关于 ``xml.etree`` 的安全性

``ElementTree`` 在 Python 3.8+ 上默认不解析外部实体，因此用它读
不受信任的 docx 是安全的（XXE 需要外部实体支持）。本模块另外再加一道
护栏：**只解压 ``word/document.xml`` 一个部件**，不做整包解压——
一个恶意的 zip 炸弹（高压缩比的超大文件）也就无从展开。
"""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET  # noqa: N817 - 与官方文档一致的别名
import zipfile
from typing import Any

from smart_research_agent.documents.base import MEDIA_DOCX, DocumentLoader
from smart_research_agent.documents.errors import UnsupportedDocument
from smart_research_agent.documents.text_loader import decode_bytes
from smart_research_agent.documents.types import (
    BLOCK_HEADING,
    BLOCK_LIST_ITEM,
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    Block,
    Document,
    make_document,
    render_table,
)
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: WordprocessingML 主命名空间（Microsoft 官方文档给出的原文）.
#: **必须用完整 URI**：ElementTree 的标签名是 ``{uri}local`` 形式。
WORDML_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

#: 正文部件在包里的路径。**只读它**（见模块 docstring 第 3 条）。
DOCUMENT_PART = "word/document.xml"

#: 本模块读得到的部件清单（进元数据，用于说清"没读到什么"）。
PARTS_READ: tuple[str, ...] = (DOCUMENT_PART,)

#: 本模块**读不到**的部件（写进元数据的已知边界）。
PARTS_NOT_READ: tuple[str, ...] = (
    "word/header*.xml",
    "word/footer*.xml",
    "word/footnotes.xml",
    "word/comments.xml",
    "word/embeddings/*",
)

#: ZIP 的本地文件头签名（同时也是 day061 类型识别里的魔数之一）。
ZIP_SIGNATURE = b"PK\x03\x04"

#: 段落样式 → 块类型的判断依据：``Heading 1`` / ``标题 1`` 这类样式名。
#: 大小写不敏感，因为 Word 的中文版写出的是 ``标题 1``。
HEADING_STYLE_PREFIXES: tuple[str, ...] = ("heading", "标题", "title")

#: 列表样式的前缀（Word 有 ``ListParagraph``，中文版是 ``列表段落``）。
LIST_STYLE_PREFIXES: tuple[str, ...] = ("listparagraph", "列表段落", "list paragraph")


def _q(local: str) -> str:
    """拼出带命名空间的完整标签名（``w:p`` → ``{uri}p``）."""
    return f"{{{WORDML_NAMESPACE}}}{local}"


def _text_of(element: ET.Element) -> str:
    """取一个元素里的全部文本（``w:t`` 直接拼接，``w:tab`` → 空格，``w:br`` → 换行）."""
    parts: list[str] = []
    for node in element.iter():
        if node.tag == _q("t"):
            parts.append(node.text or "")
        elif node.tag == _q("tab"):
            parts.append(" ")
        elif node.tag in (_q("br"), _q("cr")):
            parts.append("\n")
    return "".join(parts)


def _heading_level(paragraph: ET.Element) -> int:
    """判断段落是不是标题，并给出层级（不是标题时返回 0）.

    ``w:pPr/w:pStyle`` 的 ``w:val`` 就是样式名（如 ``Heading1`` / ``标题 2``）。
    末尾的数字是层级；没有数字的（如 ``Title``）按 1 级处理。
    """
    style = paragraph.find(f"{_q('pPr')}/{_q('pStyle')}")
    if style is None:
        return 0
    value = (style.get(_q("val")) or "").strip().lower()
    if not value:
        return 0
    for prefix in HEADING_STYLE_PREFIXES:
        if value.startswith(prefix):
            digits = "".join(char for char in value[len(prefix) :] if char.isdigit())
            level = int(digits) if digits else 1
            return min(max(level, 1), 6)
    return 0


def _is_list_paragraph(paragraph: ET.Element) -> bool:
    """判断段落是不是列表项（按样式名，或按 ``w:numPr`` 编号属性）."""
    style = paragraph.find(f"{_q('pPr')}/{_q('pStyle')}")
    if style is not None:
        value = (style.get(_q("val")) or "").strip().lower()
        if any(value.startswith(prefix) for prefix in LIST_STYLE_PREFIXES):
            return True
    return paragraph.find(f"{_q('pPr')}/{_q('numPr')}") is not None


def _table_rows(table: ET.Element) -> tuple[tuple[str, ...], ...]:
    """把 ``w:tbl`` 还原成行 × 单元格（单元格文本由其中所有段落拼成）."""
    rows: list[tuple[str, ...]] = []
    for row in table.findall(_q("tr")):
        cells: list[str] = []
        for cell in row.findall(_q("tc")):
            text = "\n".join(
                _text_of(paragraph).strip() for paragraph in cell.findall(_q("p"))
            ).strip()
            cells.append(text)
        if any(cells):
            rows.append(tuple(cells))
    return tuple(rows)


def parse_document_xml(xml_bytes: bytes, *, source: str = "") -> list[Block]:
    """解析 ``word/document.xml``，返回块序列.

    按 ``w:body`` 的**子元素顺序**遍历：段落（``w:p``）与表格（``w:tbl``）
    在原文档里是交错的，**顺序就是阅读顺序**。先取全部段落再取全部表格
    会把文档内容重排，而重排之后的"引用溯源"就指向了错误的位置。
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise UnsupportedDocument(
            f"{source or '（未知来源）'} 的 {DOCUMENT_PART} 不是合法 XML：{exc}"
        ) from exc

    body = root.find(_q("body"))
    if body is None:
        raise UnsupportedDocument(
            f"{source or '（未知来源）'} 的 {DOCUMENT_PART} 里没有 w:body："
            "缺了这个部件就不是一份可读的 Word 文档"
        )

    blocks: list[Block] = []
    for child in body:
        if child.tag == _q("p"):
            text = _text_of(child).strip()
            if not text:
                continue
            level = _heading_level(child)
            if level:
                blocks.append(Block(kind=BLOCK_HEADING, text=text, level=level))
            elif _is_list_paragraph(child):
                blocks.append(Block(kind=BLOCK_LIST_ITEM, text=text))
            else:
                blocks.append(Block(kind=BLOCK_PARAGRAPH, text=text))
        elif child.tag == _q("tbl"):
            rows = _table_rows(child)
            if rows:
                blocks.append(
                    Block(kind=BLOCK_TABLE, text=render_table(rows), rows=rows)
                )
    return blocks


class DocxLoader(DocumentLoader):
    """DOCX 加载器：只读 ``word/document.xml``，还原段落 / 标题 / 列表 / 表格."""

    media_types: tuple[str, ...] = (MEDIA_DOCX,)
    suffixes: tuple[str, ...] = (".docx",)
    description: str = "DOCX：读 OOXML 的 word/document.xml，按 body 子元素顺序还原结构"

    def load_bytes(self, data: bytes, *, source: str, media_type: str) -> Document:
        """解析 docx 字节；不是 ZIP、缺部件、XML 损坏都抛 ``UnsupportedDocument``."""
        if not data.startswith(ZIP_SIGNATURE):
            raise UnsupportedDocument(
                f"{source or '（未知来源）'} 不是 ZIP 包（头四个字节不是 PK\\x03\\x04）："
                "改名为 .docx 不会让一份旧版 .doc 变成 OOXML"
            )
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = archive.namelist()
                if DOCUMENT_PART not in names:
                    raise UnsupportedDocument(
                        f"{source or '（未知来源）'} 的包里没有 {DOCUMENT_PART}："
                        f"包里有 {len(names)} 个部件，但缺正文这一份"
                    )
                payload = archive.read(DOCUMENT_PART)
        except zipfile.BadZipFile as exc:
            raise UnsupportedDocument(
                f"{source or '（未知来源）'} 的 ZIP 结构损坏：{exc}"
            ) from exc

        blocks = parse_document_xml(payload, source=source)
        metadata = {
            # 编码固定是 UTF-8：OOXML 规范要求 XML 声明为 UTF-8，
            # 因此这里**不需要**走编码探测链——多一次"猜"只会多一次猜错的机会。
            "encoding": "utf-8",
            "encoding_confident": "true",
            "docx_parts_read": ", ".join(PARTS_READ),
            "docx_parts_not_read": ", ".join(PARTS_NOT_READ),
            "table_count": str(sum(1 for block in blocks if block.kind == BLOCK_TABLE)),
        }
        logger.debug("%s 读取部件 %s", source, PARTS_READ)
        return make_document(
            source=source, media_type=media_type, blocks=blocks, metadata=metadata
        )


def docx_scope() -> dict[str, Any]:
    """DOCX 支持的边界（报告与 ``/documents/loaders`` 端点同源）."""
    return {
        "namespace": WORDML_NAMESPACE,
        "parts_read": list(PARTS_READ),
        "parts_not_read": list(PARTS_NOT_READ),
        "structures": [
            "w:p → paragraph / heading（按 pStyle 判断层级）",
            "w:r + w:t → 段落内的文本（多个 run 直接拼接，不加分隔符）",
            "w:tab → 空格、w:br / w:cr → 换行",
            "w:tbl → table（w:tr / w:tc，单元格内多个段落用换行连接）",
        ],
        "not_supported": [
            "旧版二进制 .doc（它不是 ZIP，无法用本方式解析）",
            "页眉页脚、脚注、批注、文本框里的内容（不在 document.xml 里）",
            "图片与其说明（word/media/* 与嵌入对象）",
        ],
    }


__all__ = [
    "DOCUMENT_PART",
    "HEADING_STYLE_PREFIXES",
    "LIST_STYLE_PREFIXES",
    "PARTS_NOT_READ",
    "PARTS_READ",
    "WORDML_NAMESPACE",
    "ZIP_SIGNATURE",
    "DocxLoader",
    "docx_scope",
    "parse_document_xml",
]
