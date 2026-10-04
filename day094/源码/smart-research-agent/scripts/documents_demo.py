#!/usr/bin/env python
"""day061 演示脚本：文档解析与加载（M6-D1）.

七节，全部**离线、确定性、零第三方依赖、零网络**：

```text
1. 五个加载器与编码探测链        类型识别、冲突上报、编码的"没把握"
2. 纯文本与 Markdown             front matter / 标题 / 围栏代码 / GFM 表格 / 列表 / 引用
3. HTML                          脚本与样式整体跳过、实体解码、表格保留行列
4. DOCX                          OOXML 的段落 / 标题 / 列表 / 表格，以及"没读到什么"
5. PDF                           内容流 + zlib + 文本操作符，以及它的四条边界
6. 批量入库                      内容指纹去重、三类状态、确定性遍历
7. 四个 HTTP 端点                进程内 ASGI 调用，不写盘、不遍历目录
```

运行::

    PYTHONPATH=. python scripts/documents_demo.py
"""

from __future__ import annotations

import base64
import io
import sys
import zipfile
import zlib
from pathlib import Path

from smart_research_agent.documents import (
    BLOCK_KINDS,
    CJK_FALLBACK_ENCODING,
    DOCUMENTS_LIMITATIONS,
    DocumentIngestor,
    decode_bytes,
    default_registry,
    detect_media_type,
    docx_scope,
    encoding_chain,
    html_rules,
    markdown_syntax_table,
    pdf_boundaries,
)

WORDML = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def title(text: str) -> None:
    """打印一节标题."""
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
# 样本构造（与 tests/document_samples.py 同一手法，演示脚本里自带一份）
# --------------------------------------------------------------------------- #

MARKDOWN = """---
title: 知识库设计
tags: [rag, chunk]
---

# 检索

检索是 RAG 的第一步。它把问题变成一段向量，再去库里找最近的片段。

## 分块策略

```python
def chunk(text):
    return [text[i : i + 200] for i in range(0, len(text), 200)]
```

| 策略 | 优点 | 缺点 |
| --- | --- | --- |
| 固定长度 | 简单 | 会切断句子 |
| 递归 | 尊重结构 | 参数多 |

- 固定长度适合日志
- 递归适合技术文档

> 分块决定检索的上限：切错了，后面的重排序也救不回来。

结尾段落。
"""

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <title>RAG 入门</title>
  <script>var tracking = "should not appear";</script>
  <style>p { color: red }</style>
</head>
<body>
  <h1>检索增强生成</h1>
  <p>RAG 的第一步是<b>解析</b>，第二步是分块 &amp; 向量化。</p>
  <ul><li>解析</li><li>分块</li></ul>
  <blockquote>没有解析，后面的都不成立。</blockquote>
  <table>
    <tr><th>环节</th><th>答案在谁手里</th></tr>
    <tr><td>检索</td><td>文档</td></tr>
  </table>
</body>
</html>
"""


def build_pdf(lines: list[str], *, compress: bool = True) -> bytes:
    """手工造一份最小 PDF（内容流用 Td/T* + Tj 输出每一行）."""
    parts = ["BT /F1 12 Tf 72 720 Td 14 TL"]
    for line in lines:
        parts.append(f"({line}) Tj")
        parts.append("T*")
    parts.append("ET")
    content = "\n".join(parts).encode("latin-1")
    payload = zlib.compress(content) if compress else content
    filter_entry = b"/Filter /FlateDecode " if compress else b""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Contents 4 0 R >>",
        b"<< /Length "
        + str(len(payload)).encode()
        + b" "
        + filter_entry
        + b">>\nstream\n"
        + payload
        + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    for index, body in enumerate(objects, start=1):
        out += str(index).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    out += b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
    return bytes(out)


def build_docx() -> bytes:
    """手工造一份最小 docx（两级标题 + 正文 + 列表 + 一张表）."""
    body = (
        f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>第一章 解析</w:t></w:r></w:p>'
        f'<w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>为什么要归一化</w:t></w:r></w:p>'
        f"<w:p><w:r><w:t>下游只应该认识一种形状。</w:t></w:r></w:p>"
        f'<w:p><w:pPr><w:pStyle w:val="ListParagraph"/></w:pPr><w:r><w:t>统一的结构</w:t></w:r></w:p>'
        "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>格式</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>解析器</w:t></w:r></w:p></w:tc></w:tr>"
        "<w:tr><w:tc><w:p><w:r><w:t>PDF</w:t></w:r></w:p></w:tc>"
        "<w:tc><w:p><w:r><w:t>PdfLoader</w:t></w:r></w:p></w:tc></w:tr></w:tbl>"
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{WORDML}"><w:body>{body}</w:body></w:document>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# 七节
# --------------------------------------------------------------------------- #


def section_1_loaders() -> None:
    """第 1 节：加载器表、类型识别与编码链."""
    title("1. 五个加载器：一个媒体类型对应一个解析器（一一对应是刻意的）")
    registry = default_registry()
    for row in registry.table():
        print(f"  {row['media_type']:<62} {row['loader']:<16} {row['suffixes']}")
    print()
    print("  类型识别：**内容优先，冲突必报**")
    cases = (
        ("guide.md", MARKDOWN.encode("utf-8")),
        ("fake.txt", b"%PDF-1.4\n"),                       # 后缀与内容冲突
        ("report.docx", build_docx()),                     # ZIP 签名与 docx 后缀不冲突
        ("noextension", b"just text\n"),                   # 只能兜底成 text/plain
    )
    for name, data in cases:
        detection = detect_media_type(name, data)
        print(f"    {name:<14} → {detection.summary_line()}")
    print()
    print("  编码探测链（固定顺序，每一步都有理由）：")
    for row in encoding_chain():
        print(f"    {row['step']}. {row['rule']}")
        print(f"       {row['reason']}")
    print()
    print("  三种真实字节序列的判定：")
    for label, data in (
        ("utf-8 中文", "中文".encode("utf-8")),
        ("GB18030 中文", "中文".encode(CJK_FALLBACK_ENCODING)),
        ("带 BOM 的 utf-16-le", b"\xff\xfe" + "小端".encode("utf-16-le")),
        ("两者都不成立", b"\x80\x80"),
    ):
        decoded = decode_bytes(data)
        print(f"    {label:<20} → {decoded.summary_line()}")


def section_2_text_markdown() -> None:
    """第 2 节：纯文本与 Markdown."""
    title("2. 纯文本与 Markdown：结构写在文本里的格式值得被认真解析")
    ingestor = DocumentIngestor()
    print("  Markdown 结构与其规范条款：")
    for row in markdown_syntax_table():
        print(f"    {row['structure']:<10} {row['syntax']:<28} {row['spec']}")
    print()
    report = ingestor.ingest_bytes(
        [
            ("docs/design.md", MARKDOWN.encode("utf-8")),
            ("docs/plain.txt", "第一段。\n\n第二段还在同一份文件里。\n".encode("utf-8")),
            ("docs/legacy.txt", "旧版中文编码。\n".encode(CJK_FALLBACK_ENCODING)),
        ]
    )
    for document in report.documents:
        print(f"  {document.summary_line()}")
    print()
    markdown_doc = next(
        document for document in report.documents if document.source.endswith(".md")
    )
    print("  逐块明细（Markdown）：")
    for block in markdown_doc.blocks:
        print(f"    {block.summary_line()}")
    print()
    print("  元数据（front matter 进元数据，不进正文）：")
    for key, value in sorted(markdown_doc.metadata.items()):
        print(f"    {key} = {value}")


def section_3_html() -> None:
    """第 3 节：HTML."""
    title("3. HTML：标记比内容多的格式（剥不干净会把埋点脚本喂给模型）")
    print("  解析规则：")
    for row in html_rules():
        print(f"    {row['rule']:<10} {row['tags']:<34} {row['reason']}")
    print()
    ingestor = DocumentIngestor()
    document = ingestor.ingest_bytes([("docs/page.html", HTML.encode("utf-8"))]).documents[0]
    print(f"  {document.summary_line()}")
    print("  逐块明细：")
    for block in document.blocks:
        print(f"    {block.summary_line()}")
    print()
    print(f"  元数据：title={document.metadata['title']!r} lang={document.metadata['lang']!r} "
          f"table_count={document.metadata['table_count']}")
    print(f"  埋点脚本是否被剥掉：{'should not appear' not in document.text}")


def section_4_docx() -> None:
    """第 4 节：DOCX."""
    title("4. DOCX：读 OOXML，并且**说清自己没读什么**")
    scope = docx_scope()
    print(f"  命名空间：{scope['namespace']}")
    print(f"  读到的部件：{', '.join(scope['parts_read'])}")
    print(f"  读不到的部件：{', '.join(scope['parts_not_read'])}")
    print("  能做到的：")
    for item in scope["structures"]:
        print(f"    - {item}")
    print("  做不到的：")
    for item in scope["not_supported"]:
        print(f"    - {item}")
    print()
    ingestor = DocumentIngestor()
    document = ingestor.ingest_bytes([("docs/report.docx", build_docx())]).documents[0]
    print(f"  {document.summary_line()}")
    print("  逐块明细（**顺序就是阅读顺序**）：")
    for block in document.blocks:
        print(f"    {block.summary_line()}")
    print(f"  元数据里的边界声明：{document.metadata['docx_parts_not_read'][:60]}…")


def section_5_pdf() -> None:
    """第 5 节：PDF."""
    title("5. PDF：零依赖文本抽取，以及四条必须说清的边界")
    boundaries = pdf_boundaries()
    print("  认识的操作符：")
    for row in boundaries["operators"]:
        print(f"    {row['operator']:<20} {row['meaning']:<44} {row['spec']}")
    print("  支持的过滤器：")
    for row in boundaries["filters"]:
        print(f"    {row['name']:<12} {row['implementation']:<46} {row['spec']}")
    print("  **做不到**的（这一段比上面重要）：")
    for item in boundaries["not_supported"]:
        print(f"    - {item}")
    print()
    ingestor = DocumentIngestor()
    report = ingestor.ingest_bytes(
        [
            ("docs/paper.pdf", build_pdf(["Retrieval starts with parsing.", "Chunking comes next."])),
            ("docs/raw.pdf", build_pdf(["Uncompressed content stream."], compress=False)),
            ("docs/cid.pdf", build_pdf(["x"], compress=False) + b"/Type0 /Identity-H"),
        ]
    )
    for document in report.documents:
        print(f"  {document.summary_line()}")
    print()
    for document in report.documents:
        name = Path(document.source).name
        print(f"  {name}：pages={document.metadata['pdf_pages']} "
              f"streams={document.metadata['pdf_streams']} "
              f"inflated={document.metadata['pdf_inflated_streams']} "
              f"cid={document.metadata['pdf_cid_font_detected']} "
              f"confident={document.metadata['encoding_confident']}")
        print(f"    文本：{document.text!r}")
        if document.metadata["pdf_warnings"]:
            print(f"    告警：{document.metadata['pdf_warnings'][:70]}…")


def section_6_ingest() -> None:
    """第 6 节：批量入库."""
    title("6. 批量入库：内容指纹去重 + 三类状态 + 确定性")
    items = [
        ("docs/guide.md", MARKDOWN.encode("utf-8")),
        ("docs/guide-copy.md", MARKDOWN.encode("utf-8")),      # 同内容不同名
        ("docs/page.html", HTML.encode("utf-8")),
        ("docs/report.docx", build_docx()),
        ("docs/paper.pdf", build_pdf(["PDF text here."])),
        ("docs/plain.txt", "纯文本一段。\n".encode("utf-8")),
        ("docs/legacy.txt", "旧编码一段。\n".encode(CJK_FALLBACK_ENCODING)),
        ("docs/broken.txt", b"\x00\x01\x02binary"),
        ("docs/fake.pdf", b"not a pdf at all"),
    ]
    report = DocumentIngestor().ingest_bytes(items)
    print(f"  {report.summary_line()}")
    print()
    for entry in report.entries:
        print(f"    {entry.summary_line()}")
    print()
    counts = report.status_counts()
    print(f"  状态计数（三个键恒存在）：{counts}")
    print(f"  块统计（六类恒存在）：{report.block_counts()}")
    print(f"  媒体类型：{ {k: v for k, v in report.media_type_counts().items() if v} }")
    print()
    print("  交给知识库的记录形状（day062 会在这里切分块）：")
    record = report.knowledge_records()[0]
    print(f"    doc_id={record['doc_id']} source={record['source']!r} "
          f"text={len(record['text'])} 字符 metadata={len(record['metadata'])} 项")
    print("  同一份内容换个文件名要求同样的指纹：")
    again = DocumentIngestor().ingest_bytes([("renamed.md", MARKDOWN.encode("utf-8"))])
    print(f"    {again.documents[0].fingerprint == record['doc_id']}")


def section_7_api() -> None:
    """第 7 节：四个端点."""
    title("7. 四个端点：只读或纯计算，不写盘、不遍历目录")
    from fastapi.testclient import TestClient

    from smart_research_agent.api.app import create_app
    from smart_research_agent.llm.mock import MockLLM

    client = TestClient(create_app(llm=MockLLM(default="offline")))
    loaders = client.get("/documents/loaders").json()
    print(f"  GET  /documents/loaders  → 媒体类型 {len(loaders['media_types'])} 种 / "
          f"加载器 {len(loaders['loaders'])} 个 / 编码链 {len(loaders['encoding_chain'])} 步 / "
          f"限制 {len(loaders['limitations'])} 条")
    detect = client.post(
        "/documents/detect",
        json={"filename": "fake.txt", "content_base64": base64.b64encode(b"%PDF-1.5\n").decode()},
    ).json()
    print(f"  POST /documents/detect   → {detect['summary']}")
    parse = client.post(
        "/documents/parse", json={"filename": "design.md", "text": MARKDOWN}
    ).json()
    print(f"  POST /documents/parse    → {parse['summary']}")
    print(f"       块类型：{[block['kind'] for block in parse['document']['blocks']]}")
    ingest = client.post(
        "/documents/ingest",
        json={
            "files": [
                {"filename": "a.md", "text": MARKDOWN},
                {"filename": "b.md", "text": MARKDOWN},
                {"filename": "c.txt", "content_base64": base64.b64encode(b"\x00\x01").decode()},
            ]
        },
    ).json()
    print(f"  POST /documents/ingest   → {ingest['summary']}")
    print(f"       状态计数：{ingest['report']['status_counts']}")
    print()
    print(f"  六种块类型：{', '.join(BLOCK_KINDS)}")
    print("  本日的四条限制里，最该记住的是：")
    print(f"    - {DOCUMENTS_LIMITATIONS[0][:64]}…")


def main() -> int:
    """按节运行演示（任何一节失败都返回非零退出码）."""
    for section in (
        section_1_loaders,
        section_2_text_markdown,
        section_3_html,
        section_4_docx,
        section_5_pdf,
        section_6_ingest,
        section_7_api,
    ):
        section()
    print(f"\n演示完成（工作目录 {Path.cwd()}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
