"""day061 测试用的手工样本构造器（PDF 与 DOCX）.

放在单独模块里而不是复制到每个测试文件：**同一份样本被三处用到**
（PDF 单测、HTML/DOCX 单测、入库与端点测试），复制三份会让一次格式调整
要改三处，而漏改的那一处会以"某个测试莫名失败"的形式出现。

两个构造器都**不依赖任何第三方库**，因此样本可以在离线 CI 里逐字节复现。
"""

from __future__ import annotations

import io
import zipfile
import zlib

WORDML_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def build_pdf(
    lines: list[str],
    *,
    compress: bool = True,
    extra: bytes = b"",
    line_spacing: float = 14.0,
    y_start: float = 720.0,
) -> bytes:
    """手工造一份最小 PDF：用 ``Td`` + ``T*`` + ``Tj`` 输出给定的每一行.

    ``compress=True`` 时内容流用 zlib 压缩并声明 ``/Filter /FlateDecode``，
    ``False`` 时用未压缩的流（两种都是合法写法，因此两种都要有样本）。

    ``line_spacing`` 用来构造"段落间距"：相邻行的纵坐标差就是它，
    而 ``_group_paragraphs`` 用中位数判定段落边界——因此把某一行之后
    的间距放大就会得到一个新的段落。
    """
    parts = [f"BT /F1 12 Tf 72 {y_start} Td {line_spacing} TL"]
    y = y_start
    for line in lines:
        parts.append(f"({line}) Tj")
        parts.append("T*")
        y -= line_spacing
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
    out += extra
    out += b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
    return bytes(out)


def build_pdf_with_gap(
    first: list[str], second: list[str], *, gap: float = 40.0, line_spacing: float = 14.0
) -> bytes:
    """造一份"两段"的 PDF：第二段的起始行与上一行之间有 ``gap`` 的额外间距."""
    lines = list(first)
    y = 720.0
    for _ in first:
        y -= line_spacing
    y -= gap
    lines.extend(second)
    parts = [f"BT /F1 12 Tf 72 720 Td {line_spacing} TL"]
    for index, line in enumerate(lines):
        if index == len(first):
            parts.append(f"1 0 0 1 72 {y:.1f} Tm")
        parts.append(f"({line}) Tj")
        parts.append("T*")
    parts.append("ET")
    content = "\n".join(parts).encode("latin-1")
    payload = zlib.compress(content)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Contents 4 0 R >>",
        b"<< /Length "
        + str(len(payload)).encode()
        + b" /Filter /FlateDecode >>\nstream\n"
        + payload
        + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    for index, body in enumerate(objects, start=1):
        out += str(index).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    out += b"trailer\n<< /Root 1 0 R >>\n%%EOF\n"
    return bytes(out)


def _paragraph(style: str, text: str) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t>{text}</w:t></w:r></w:p>"


def build_docx(
    paragraphs: list[tuple[str, str]],
    *,
    table: tuple[tuple[str, ...], ...] | None = (("A", "B"), ("1", "2")),
    include_document_part: bool = True,
    include_body: bool = True,
    extra_parts: dict[str, str] | None = None,
) -> bytes:
    """手工造一份最小 docx：段落列表为 ``(样式, 文本)``，可选一张表.

    ``include_document_part=False`` 用来造"缺 word/document.xml"的包，
    ``include_body=False`` 用来造"有 document.xml 但没有 w:body"的包——
    两种都是真实存在的坏文件，而且都必须在测试里各有一条用例。
    """
    body = "".join(_paragraph(style, text) for style, text in paragraphs)
    if table is not None:
        rows = "".join(
            "<w:tr>"
            + "".join(
                f"<w:tc><w:p><w:r><w:t>{cell}</w:t></w:r></w:p></w:tc>" for cell in row
            )
            + "</w:tr>"
            for row in table
        )
        body += f"<w:tbl>{rows}</w:tbl>"
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{WORDML_NAMESPACE}">'
        + (f"<w:body>{body}</w:body>" if include_body else "")
        + "</w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        if include_document_part:
            archive.writestr("word/document.xml", document)
        for name, payload in (extra_parts or {}).items():
            archive.writestr(name, payload)
    return buffer.getvalue()


__all__ = ["WORDML_NAMESPACE", "build_docx", "build_pdf", "build_pdf_with_gap"]
