"""文档解析与加载包（M6-D1）：把八种格式收敛成一种下游能消费的形状.

M6 的第一天解决的是一个问题：**下游不该认识格式**。

```text
各种格式  →  Document（全文 + 块序列 + 元数据 + 内容指纹）  →  分块 / 向量化 / 检索
```

七个模块，每个回答一个问题：

```text
base.py             什么文件用哪个解析器（注册表 + 内容优先的类型识别 + 大小护栏）
types.py            归一化后的形状（六种块 / 内容指纹 / 唯一标准形）
text_loader.py      编码怎么定（固定顺序 + 「定不下来」要显式标记）
markdown_loader.py  结构写在文本里的格式（标题 / 围栏代码 / 表格 / 列表 / front matter）
html_loader.py      标记比内容多的格式（脚本与样式整体跳过，表格保留行列）
docx_loader.py      ZIP 包里的 OOXML（只读 document.xml，边界写进元数据）
pdf_loader.py       为打印而设计的格式（内容流 + zlib + 文本操作符，边界写满）
pipeline.py         一批文件怎么入库（内容指纹去重 + 三类状态 + 确定性遍历）
```

## 三条贯穿全包的纪律

1. **只分类，不丢弃**。三个解析器都不允许"未识别的行被静默吞掉"——
   "结构化解析"最容易犯的错就是让内容消失，而它的表现只是
   "这份文档比原文短"；
2. **能力边界写进元数据**。PDF 的 CID 字体、DOCX 读不到的部件、
   编码探测的"没把握"，全都以字段形式跟着文档走。
   **"没抽到的内容"最容易被当成"文档里本来就没有"**；
3. **同一份内容只有一个指纹**。``doc_id = sha256(规范化文本)[:16]``，
   与 day058 的三元组、day060 的部署绑定是同一种内容寻址。
   因此规范化规则是**接口的一部分**，改它等于让历史索引全部失效。

## 与既有包的接缝

- **上游**：``settings.documents_*``（文件大小上限、目录等）；
- **下游**：``IngestReport.knowledge_records()`` 产出的形状就是 day009 的
  ``VectorStore.add`` 需要的；day062 的分块器以 ``Block`` 为单位切分，
  接口形状不变；
- **端点**：``api.routes`` 的 ``/documents/*`` 只做**只读或纯计算**——
  真正的目录遍历与批量入库由 ``scripts/documents_demo.py`` 承担。
"""

from __future__ import annotations

from smart_research_agent.documents.base import (
    DECIDED_BY_FALLBACK,
    DECIDED_BY_MAGIC,
    DECIDED_BY_SUFFIX,
    MEDIA_DOCX,
    MEDIA_HTML,
    MEDIA_MARKDOWN,
    MEDIA_PDF,
    MEDIA_TEXT,
    MEDIA_TYPES,
    SUFFIX_MEDIA_TYPES,
    DocumentLoader,
    LoaderRegistry,
    MediaTypeDetection,
    default_registry,
    detect_media_type,
    magic_media_type,
    suffix_media_type,
)
from smart_research_agent.documents.docx_loader import (
    DOCUMENT_PART,
    WORDML_NAMESPACE,
    DocxLoader,
    docx_scope,
    parse_document_xml,
)
from smart_research_agent.documents.errors import DocumentError, UnsupportedDocument
from smart_research_agent.documents.html_loader import (
    SKIP_TAGS,
    HtmlLoader,
    HtmlStructureParser,
    html_rules,
)
from smart_research_agent.documents.markdown_loader import (
    MarkdownLoader,
    markdown_syntax_table,
    parse_front_matter,
    split_front_matter,
)
from smart_research_agent.documents.pdf_loader import (
    PDF_SIGNATURE,
    PdfLoader,
    PdfTextExtraction,
    decode_literal_string,
    extract_lines,
    extract_pdf_text,
    iter_streams,
    maybe_inflate,
    pdf_boundaries,
)
from smart_research_agent.documents.pipeline import (
    DOCUMENTS_LIMITATIONS,
    DOCUMENTS_OUT_OF_SCOPE,
    DocumentIngestor,
    IngestEntry,
    IngestReport,
    iter_supported_files,
    knowledge_record_shape,
)
from smart_research_agent.documents.text_loader import (
    CJK_FALLBACK_ENCODING,
    ENCODING_CHAIN,
    DecodedText,
    TextLoader,
    decode_bytes,
    encoding_chain,
    looks_binary,
    split_paragraphs,
)
from smart_research_agent.documents.types import (
    BLOCK_CODE,
    BLOCK_HEADING,
    BLOCK_KINDS,
    BLOCK_LIST_ITEM,
    BLOCK_PARAGRAPH,
    BLOCK_QUOTE,
    BLOCK_TABLE,
    DOC_ID_LENGTH,
    Block,
    Document,
    content_id,
    make_document,
    normalize_text,
    render_table,
)

__all__ = [
    "BLOCK_CODE",
    "BLOCK_HEADING",
    "BLOCK_KINDS",
    "BLOCK_LIST_ITEM",
    "BLOCK_PARAGRAPH",
    "BLOCK_QUOTE",
    "BLOCK_TABLE",
    "CJK_FALLBACK_ENCODING",
    "DECIDED_BY_FALLBACK",
    "DECIDED_BY_MAGIC",
    "DECIDED_BY_SUFFIX",
    "DOCUMENTS_LIMITATIONS",
    "DOCUMENTS_OUT_OF_SCOPE",
    "DOCUMENT_PART",
    "DOC_ID_LENGTH",
    "ENCODING_CHAIN",
    "MEDIA_DOCX",
    "MEDIA_HTML",
    "MEDIA_MARKDOWN",
    "MEDIA_PDF",
    "MEDIA_TEXT",
    "MEDIA_TYPES",
    "PDF_SIGNATURE",
    "SKIP_TAGS",
    "SUFFIX_MEDIA_TYPES",
    "WORDML_NAMESPACE",
    "Block",
    "DecodedText",
    "Document",
    "DocumentError",
    "DocumentIngestor",
    "DocumentLoader",
    "DocxLoader",
    "HtmlLoader",
    "HtmlStructureParser",
    "IngestEntry",
    "IngestReport",
    "LoaderRegistry",
    "MarkdownLoader",
    "MediaTypeDetection",
    "PdfLoader",
    "PdfTextExtraction",
    "TextLoader",
    "UnsupportedDocument",
    "content_id",
    "decode_bytes",
    "decode_literal_string",
    "default_registry",
    "detect_media_type",
    "docx_scope",
    "encoding_chain",
    "extract_lines",
    "extract_pdf_text",
    "html_rules",
    "iter_streams",
    "iter_supported_files",
    "knowledge_record_shape",
    "looks_binary",
    "magic_media_type",
    "make_document",
    "markdown_syntax_table",
    "maybe_inflate",
    "normalize_text",
    "parse_document_xml",
    "parse_front_matter",
    "pdf_boundaries",
    "render_table",
    "split_front_matter",
    "split_paragraphs",
    "suffix_media_type",
]
