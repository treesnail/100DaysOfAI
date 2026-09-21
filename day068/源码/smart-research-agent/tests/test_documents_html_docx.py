"""day061 ``documents.html_loader`` 与 ``documents.docx_loader`` 的单元测试."""

from __future__ import annotations

import pytest

from smart_research_agent.documents import (
    BLOCK_CODE,
    BLOCK_HEADING,
    BLOCK_LIST_ITEM,
    BLOCK_PARAGRAPH,
    BLOCK_QUOTE,
    BLOCK_TABLE,
    DOCUMENT_PART,
    MEDIA_DOCX,
    MEDIA_HTML,
    SKIP_TAGS,
    DocxLoader,
    HtmlLoader,
    UnsupportedDocument,
    docx_scope,
    html_rules,
    parse_document_xml,
)
from tests.document_samples import WORDML_NAMESPACE, build_docx

HTML_SAMPLE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <title>RAG 入门</title>
  <script>var tracking = "should not appear";</script>
  <style>p { color: red }</style>
</head>
<body>
  <h1>检索增强生成</h1>
  <p>RAG &amp; 检索  的第一步是<b>解析</b>。</p>
  <ul><li>分块</li><li>向量化</li></ul>
  <blockquote>引用的一段话</blockquote>
  <pre>def chunk(): pass</pre>
  <table>
    <tr><th>策略</th><th>优点</th></tr>
    <tr><td>固定</td><td>简单</td></tr>
    <tr><td></td><td></td></tr>
  </table>
</body>
</html>
""".encode("utf-8")


def _html_document():
    return HtmlLoader().load_bytes(HTML_SAMPLE, source="a.html", media_type=MEDIA_HTML)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #


def test_html_loader_declares_its_scope() -> None:
    loader = HtmlLoader()
    assert loader.media_types == (MEDIA_HTML,)
    assert set(loader.suffixes) == {".html", ".htm"}


def test_scripts_and_styles_are_skipped_entirely() -> None:
    """不剥离的话，检索「配置」这个词会命中页面的埋点脚本——而那种命中看起来很正常."""
    document = _html_document()
    assert "should not appear" not in document.text
    assert "color: red" not in document.text
    assert set(SKIP_TAGS) == {"script", "style", "noscript", "template"}


def test_html_structures_become_blocks() -> None:
    document = _html_document()
    kinds = [block.kind for block in document.blocks]
    assert kinds == [
        BLOCK_HEADING,
        BLOCK_PARAGRAPH,
        BLOCK_LIST_ITEM,
        BLOCK_LIST_ITEM,
        BLOCK_QUOTE,
        BLOCK_CODE,
        BLOCK_TABLE,
    ]
    assert document.blocks[0].level == 1
    assert document.blocks[0].text == "检索增强生成"


def test_html_entities_are_decoded() -> None:
    """``&amp;`` / ``&nbsp;`` 不解码的话，检索会拿一段实体去匹配."""
    document = _html_document()
    assert "RAG & 检索" in document.text
    assert "&amp;" not in document.text


def test_html_title_and_lang_go_to_metadata() -> None:
    document = _html_document()
    assert document.metadata["title"] == "RAG 入门"
    assert document.metadata["lang"] == "zh-CN"
    assert document.metadata["table_count"] == "1"


def test_html_table_keeps_rows_and_drops_blank_rows() -> None:
    document = _html_document()
    table = next(block for block in document.blocks if block.kind == BLOCK_TABLE)
    assert table.rows == (("策略", "优点"), ("固定", "简单"))
    assert table.text.startswith("| 策略 | 优点 |")


def test_html_br_becomes_a_newline() -> None:
    document = HtmlLoader().load_bytes(
        b"<p>line1<br>line2</p>", source="a.html", media_type=MEDIA_HTML
    )
    assert document.blocks[0].text == "line1\nline2"


def test_html_container_tags_only_end_blocks() -> None:
    """``div`` / ``section`` 没有语义内容：它们只作为块边界，不产生独立的块."""
    document = HtmlLoader().load_bytes(
        b"<div>first</div><div>second</div>", source="a.html", media_type=MEDIA_HTML
    )
    assert [block.kind for block in document.blocks] == [BLOCK_PARAGRAPH, BLOCK_PARAGRAPH]
    assert [block.text for block in document.blocks] == ["first", "second"]


def test_html_free_text_is_wrapped_into_a_paragraph() -> None:
    """没有块标签包裹的裸文本也要落块——**"只分类不丢弃"同样适用于 HTML**."""
    document = HtmlLoader().load_bytes(
        b"<body>bare text here</body>", source="a.html", media_type=MEDIA_HTML
    )
    assert document.blocks[0].kind == BLOCK_PARAGRAPH
    assert "bare text here" in document.blocks[0].text


def test_html_empty_input_yields_no_blocks() -> None:
    document = HtmlLoader().load_bytes(b"", source="a.html", media_type=MEDIA_HTML)
    assert document.blocks == ()


def test_html_nested_list_records_depth() -> None:
    document = HtmlLoader().load_bytes(
        b"<ul><li>top<ul><li>inner</li></ul></li></ul>",
        source="a.html",
        media_type=MEDIA_HTML,
    )
    levels = [block.level for block in document.blocks if block.kind == BLOCK_LIST_ITEM]
    assert levels == [0, 1]


def test_html_skips_nested_tags_inside_script() -> None:
    """``<script>`` 里的标签同样要跳过：只看开始标签会让跳过的深度错位."""
    document = HtmlLoader().load_bytes(
        b"<body><script>if (a<b) { document.write('<p>inner</p>'); }</script>"
        b"<p>kept</p></body>",
        source="a.html",
        media_type=MEDIA_HTML,
    )
    assert "inner" not in document.text
    assert document.blocks[0].text == "kept"


def test_html_without_lang_attribute_is_fine() -> None:
    document = HtmlLoader().load_bytes(
        b"<html><body><p>x</p></body></html>", source="a.html", media_type=MEDIA_HTML
    )
    assert "lang" not in document.metadata


def test_html_hr_flushes_the_current_block() -> None:
    """``<hr>`` 是显式的分隔线：它不产生块，但**它必须结束当前块**."""
    document = HtmlLoader().load_bytes(
        b"<p>before</p><hr><p>after</p>", source="a.html", media_type=MEDIA_HTML
    )
    assert [block.text for block in document.blocks] == ["before", "after"]


def test_html_rules_explain_every_decision() -> None:
    rows = html_rules()
    assert len(rows) == 6
    assert all(row["reason"] for row in rows)
    assert any("script" in row["tags"] for row in rows)


def test_html_of_gbk_page_decodes() -> None:
    data = "<html><body><p>中文页面</p></body></html>".encode("gb18030")
    document = HtmlLoader().load_bytes(data, source="a.html", media_type=MEDIA_HTML)
    assert document.metadata["encoding"] == "gb18030"
    assert "中文页面" in document.text


# --------------------------------------------------------------------------- #
# DOCX
# --------------------------------------------------------------------------- #


def _docx_document(**kwargs):
    data = build_docx(
        [("Heading1", "第一章"), ("Heading2", "子节"), ("", "正文一段。"), ("ListParagraph", "要点")],
        **kwargs,
    )
    return DocxLoader().load_bytes(data, source="a.docx", media_type=MEDIA_DOCX)


def test_docx_structures_become_blocks_in_document_order() -> None:
    """``w:p`` 与 ``w:tbl`` 在原文档里交错，**顺序就是阅读顺序**."""
    document = _docx_document()
    kinds = [block.kind for block in document.blocks]
    assert kinds == [
        BLOCK_HEADING,
        BLOCK_HEADING,
        BLOCK_PARAGRAPH,
        BLOCK_LIST_ITEM,
        BLOCK_TABLE,
    ]
    assert [block.level for block in document.blocks[:2]] == [1, 2]
    assert document.blocks[4].rows == (("A", "B"), ("1", "2"))


def test_docx_loader_declares_its_media_type() -> None:
    loader = DocxLoader()
    assert loader.media_types == (MEDIA_DOCX,)
    assert loader.suffixes == (".docx",)


def test_docx_metadata_names_what_was_and_was_not_read() -> None:
    """**"没读到的部分"最容易被当成"文档里没有"**，因此它必须进元数据."""
    document = _docx_document()
    assert document.metadata["docx_parts_read"] == DOCUMENT_PART
    assert "header" in document.metadata["docx_parts_not_read"]
    assert document.metadata["encoding"] == "utf-8"
    assert document.metadata["encoding_confident"] == "true"


def test_docx_heading_style_without_digit_defaults_to_level_one() -> None:
    data = build_docx([("Title", "文档标题")], table=None)
    document = DocxLoader().load_bytes(data, source="a.docx", media_type=MEDIA_DOCX)
    assert document.blocks[0].level == 1


def test_docx_empty_style_value_is_not_a_heading() -> None:
    """``w:pStyle`` 存在但 ``w:val`` 为空时**不能**当标题（否则层级校验会拦住整份文档）."""
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<w:document xmlns:w="{WORDML_NAMESPACE}"><w:body>'
        '<w:p><w:pPr><w:pStyle w:val=""/></w:pPr><w:r><w:t>正文</w:t></w:r></w:p>'
        "</w:body></w:document>"
    ).encode("utf-8")
    blocks = parse_document_xml(document_xml)
    assert blocks[0].kind == BLOCK_PARAGRAPH


def test_docx_style_that_is_not_a_heading_stays_a_paragraph() -> None:
    data = build_docx([("Quote", "引用样式")], table=None)
    document = DocxLoader().load_bytes(data, source="a.docx", media_type=MEDIA_DOCX)
    assert document.blocks[0].kind == BLOCK_PARAGRAPH


def test_docx_numbered_list_is_detected_by_numpr() -> None:
    """中文版 Word 写出的是「列表段落」，有时还只有 ``w:numPr`` 而没有样式名."""
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<w:document xmlns:w="{WORDML_NAMESPACE}"><w:body>'
        "<w:p><w:pPr><w:numPr><w:ilvl w:val=\"0\"/></w:numPr></w:pPr>"
        "<w:r><w:t>编号项</w:t></w:r></w:p></w:body></w:document>"
    ).encode("utf-8")
    blocks = parse_document_xml(document_xml, source="a.docx")
    assert blocks[0].kind == BLOCK_LIST_ITEM


def test_docx_rejects_a_non_zip_payload() -> None:
    with pytest.raises(UnsupportedDocument, match="不是 ZIP 包"):
        DocxLoader().load_bytes(b"just text", source="a.docx", media_type=MEDIA_DOCX)


def test_docx_rejects_a_package_without_the_document_part() -> None:
    data = build_docx([("", "x")], include_document_part=False)
    with pytest.raises(UnsupportedDocument, match="包里没有 word/document.xml"):
        DocxLoader().load_bytes(data, source="a.docx", media_type=MEDIA_DOCX)


def test_docx_rejects_a_document_without_body() -> None:
    data = build_docx([("", "x")], include_body=False)
    with pytest.raises(UnsupportedDocument, match="没有 w:body"):
        DocxLoader().load_bytes(data, source="a.docx", media_type=MEDIA_DOCX)


def test_docx_rejects_a_corrupt_zip() -> None:
    with pytest.raises(UnsupportedDocument, match="ZIP 结构损坏"):
        DocxLoader().load_bytes(
            b"PK\x03\x04" + b"\x00" * 32, source="a.docx", media_type=MEDIA_DOCX
        )


def test_parse_document_xml_rejects_malformed_xml() -> None:
    with pytest.raises(UnsupportedDocument, match="不是合法 XML"):
        parse_document_xml(b"<w:document", source="a.docx")


def test_docx_runs_are_concatenated_without_separators() -> None:
    """同一段的多个 run 是"同一句话的不同格式片段"，它们之间本来就没有空格."""
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<w:document xmlns:w="{WORDML_NAMESPACE}"><w:body><w:p>'
        "<w:r><w:t>加粗</w:t></w:r><w:r><w:t>紧跟的后半句</w:t></w:r>"
        "</w:p></w:body></w:document>"
    ).encode("utf-8")
    blocks = parse_document_xml(document_xml)
    assert blocks[0].text == "加粗紧跟的后半句"


def test_docx_tabs_and_breaks_are_preserved() -> None:
    """``w:tab`` 与 ``w:br`` 没有文本内容，不显式处理就会静默丢掉."""
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<w:document xmlns:w="{WORDML_NAMESPACE}"><w:body><w:p>'
        "<w:r><w:t>第一行</w:t><w:br/><w:t>第二行</w:t><w:tab/><w:t>制表后</w:t></w:r>"
        "</w:p></w:body></w:document>"
    ).encode("utf-8")
    blocks = parse_document_xml(document_xml)
    assert blocks[0].text == "第一行\n第二行 制表后"


def test_docx_empty_paragraphs_are_dropped() -> None:
    data = build_docx([("", "正文"), ("", "   ")], table=None)
    document = DocxLoader().load_bytes(data, source="a.docx", media_type=MEDIA_DOCX)
    assert len(document.blocks) == 1


def test_docx_scope_documents_the_boundary() -> None:
    scope = docx_scope()
    assert scope["namespace"] == WORDML_NAMESPACE
    assert scope["parts_read"] == [DOCUMENT_PART]
    assert any("header" in item for item in scope["parts_not_read"])
    assert any(".doc" in item for item in scope["not_supported"])
