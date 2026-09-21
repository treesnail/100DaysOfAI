"""day061 ``documents.base`` 与 ``documents.text_loader`` 的单元测试.

覆盖两件最容易静默失效的事：**类型识别的冲突**与**编码探测的"没把握"**。
"""

from __future__ import annotations

import pytest

from smart_research_agent.documents import (
    CJK_FALLBACK_ENCODING,
    DECIDED_BY_FALLBACK,
    DECIDED_BY_MAGIC,
    DECIDED_BY_SUFFIX,
    MEDIA_DOCX,
    MEDIA_HTML,
    MEDIA_MARKDOWN,
    MEDIA_PDF,
    MEDIA_TEXT,
    MEDIA_TYPES,
    DocumentError,
    LoaderRegistry,
    TextLoader,
    UnsupportedDocument,
    decode_bytes,
    default_registry,
    detect_media_type,
    encoding_chain,
    looks_binary,
    magic_media_type,
    suffix_media_type,
    split_paragraphs,
)


# --------------------------------------------------------------------------- #
# 类型识别：内容优先，冲突必报
# --------------------------------------------------------------------------- #


def test_suffix_table_covers_the_five_media_types() -> None:
    assert suffix_media_type("a.md") == MEDIA_MARKDOWN
    assert suffix_media_type("a.MARKDOWN") == MEDIA_MARKDOWN  # 大小写不敏感
    assert suffix_media_type("a.html") == MEDIA_HTML
    assert suffix_media_type("a.htm") == MEDIA_HTML
    assert suffix_media_type("a.txt") == MEDIA_TEXT
    assert suffix_media_type("a.log") == MEDIA_TEXT
    assert suffix_media_type("a.pdf") == MEDIA_PDF
    assert suffix_media_type("a.docx") == MEDIA_DOCX
    assert suffix_media_type("README") == ""


def test_magic_detects_pdf_zip_and_html() -> None:
    assert magic_media_type(b"%PDF-1.7\n") == MEDIA_PDF
    assert magic_media_type(b"PK\x03\x04rest") == "application/zip"
    assert magic_media_type(b"<!DOCTYPE html><html></html>") == MEDIA_HTML
    assert magic_media_type(b"  <html lang='zh'>") == MEDIA_HTML
    assert magic_media_type(b"\x00\x01\x02") == ""


def test_detection_prefers_content_and_reports_the_conflict() -> None:
    """文件名与内容说了两件不同的事时必须报出来（往往是"文件被人改过"的证据）."""
    detection = detect_media_type("fake.txt", b"%PDF-1.4\n")
    assert detection.media_type == MEDIA_PDF
    assert detection.decided_by == DECIDED_BY_MAGIC
    assert detection.conflict is True
    assert detection.suffix_type == MEDIA_TEXT
    assert "**冲突**" in detection.summary_line()
    assert detection.to_dict()["conflict"] is True


def test_docx_zip_signature_is_not_a_conflict() -> None:
    """docx 本身就是 ZIP：签名与后缀说的是同一件事，不是冲突."""
    detection = detect_media_type("a.docx", b"PK\x03\x04rest")
    assert detection.media_type == MEDIA_DOCX
    assert detection.decided_by == DECIDED_BY_SUFFIX
    assert detection.conflict is False


def test_detection_falls_back_to_the_suffix_or_to_plain_text() -> None:
    by_suffix = detect_media_type("a.pdf", b"not really a pdf")
    assert by_suffix.media_type == MEDIA_PDF
    assert by_suffix.decided_by == DECIDED_BY_SUFFIX
    assert by_suffix.magic_type == ""

    fallback = detect_media_type("noextension", b"just text")
    assert fallback.media_type == MEDIA_TEXT
    assert fallback.decided_by == DECIDED_BY_FALLBACK
    assert fallback.conflict is False
    assert "text/plain" in fallback.summary_line()


def test_detection_with_no_name_reports_fallback() -> None:
    detection = detect_media_type("", b"")
    assert detection.media_type == MEDIA_TEXT
    assert detection.decided_by == DECIDED_BY_FALLBACK


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #


def test_default_registry_registers_one_loader_per_media_type() -> None:
    registry = default_registry()
    assert registry.media_types == sorted(MEDIA_TYPES)
    assert len(registry.loaders) == 5
    assert registry.supports(MEDIA_PDF) is True
    assert registry.supports("application/zip") is False


def test_registry_table_is_stable_and_descriptive() -> None:
    rows = default_registry().table()
    assert [row["media_type"] for row in rows] == sorted(MEDIA_TYPES)
    pdf_row = next(row for row in rows if row["media_type"] == MEDIA_PDF)
    assert pdf_row["loader"] == "PdfLoader"
    assert ".pdf" in pdf_row["suffixes"]
    assert pdf_row["description"]


def test_registry_rejects_duplicate_media_types() -> None:
    """两个加载器抢同一种格式时，实际生效的取决于导入顺序——那是最难复现的一类问题."""
    registry = LoaderRegistry([TextLoader()])
    with pytest.raises(DocumentError, match="已由 TextLoader 注册"):
        registry.register(TextLoader())


def test_registry_rejects_a_loader_without_media_types() -> None:
    class _Nameless:
        media_types: tuple[str, ...] = ()
        suffixes: tuple[str, ...] = ()
        description = ""

        def load_bytes(self, data, *, source, media_type):  # pragma: no cover - 到不了
            raise NotImplementedError

    with pytest.raises(DocumentError, match="没有声明 media_types"):
        LoaderRegistry([_Nameless()])  # type: ignore[list-item]


def test_registry_unknown_media_type_lists_the_alternatives() -> None:
    """"你要的是不是其中之一"比 KeyError 有用得多（day045 列候选模型名同理）."""
    with pytest.raises(DocumentError) as excinfo:
        default_registry().get("application/zip")
    message = str(excinfo.value)
    assert "application/zip" in message
    assert MEDIA_PDF in message
    assert MEDIA_MARKDOWN in message


def test_registry_size_guard_blocks_before_parsing() -> None:
    registry = LoaderRegistry([TextLoader()], max_bytes=16)
    with pytest.raises(DocumentError, match="超过上限"):
        registry.load_bytes(b"x" * 64, source="big.txt")


def test_registry_load_bytes_records_the_detection_in_metadata() -> None:
    document = default_registry().load_bytes(b"hello", source="a.md")
    assert document.media_type == MEDIA_MARKDOWN
    assert document.metadata["detected_media_type"] == MEDIA_MARKDOWN
    assert document.metadata["media_type_conflict"] == "false"
    assert document.metadata["suffix_media_type"] == MEDIA_MARKDOWN


def test_registry_load_bytes_records_a_conflict_for_later_readers() -> None:
    document = default_registry().load_bytes(b"%PDF-1.4\n", source="a.txt")
    assert document.metadata["media_type_conflict"] == "true"
    assert document.metadata["magic_media_type"] == MEDIA_PDF
    assert document.metadata["suffix_media_type"] == MEDIA_TEXT


def test_registry_load_path_reads_a_real_file(tmp_path) -> None:
    target = tmp_path / "note.md"
    target.write_text("# 标题\n\n正文\n", encoding="utf-8")
    document = default_registry().load_path(target)
    assert document.source == str(target)
    assert document.media_type == MEDIA_MARKDOWN
    assert document.blocks[0].text == "标题"


def test_registry_load_path_rejects_a_directory(tmp_path) -> None:
    with pytest.raises(DocumentError, match="不是一个文件"):
        default_registry().load_path(tmp_path)


def test_loader_supported_line_is_readable() -> None:
    line = TextLoader().supported_line()
    assert "TextLoader" in line
    assert MEDIA_TEXT in line
    assert ".txt" in line


# --------------------------------------------------------------------------- #
# 编码探测链
# --------------------------------------------------------------------------- #


def test_encoding_chain_documents_five_ordered_steps() -> None:
    rows = encoding_chain()
    assert [row["step"] for row in rows] == ["1", "2", "3", "4", "5"]
    assert "utf-8 严格解码" in rows[2]["rule"]
    assert CJK_FALLBACK_ENCODING in rows[3]["rule"]
    assert "UTF-32 必须排在 UTF-16 之前" in rows[0]["reason"]
    assert all(row["reason"] for row in rows)


def test_decode_prefers_utf8_when_it_is_valid() -> None:
    decoded = decode_bytes("中文内容".encode("utf-8"))
    assert decoded.text == "中文内容"
    assert decoded.encoding == "utf-8"
    assert decoded.confident is True
    assert "可信" in decoded.summary_line()


def test_decode_falls_back_to_gb18030_for_legacy_chinese() -> None:
    """GB18030 是 GBK 与 GB2312 的超集，一个编码覆盖三代中文文档."""
    decoded = decode_bytes("旧版中文编码测试".encode("gb18030"))
    assert decoded.text == "旧版中文编码测试"
    assert decoded.encoding == CJK_FALLBACK_ENCODING
    assert decoded.confident is True


def test_decode_marks_latin1_fallback_as_untrusted() -> None:
    """latin-1 永不失败，因此它**只能排最后**，且结果必须被标记为不可信."""
    # 0x80 在 utf-8 与 gb18030 里都是非法起始字节，因此必然走到最后一步。
    # **注意不要用 \\xff\\xfe 开头**：那是 UTF-16-LE 的 BOM，会被第 1 步接走。
    decoded = decode_bytes(b"\x80\x80")
    assert decoded.encoding == "latin-1"
    assert decoded.decided_by == "latin-1"
    assert decoded.confident is False
    assert "不可信" in decoded.summary_line()


def test_decode_respects_a_utf8_bom() -> None:
    decoded = decode_bytes(b"\xef\xbb\xbf" + "带 BOM".encode("utf-8"))
    assert decoded.text == "带 BOM"
    assert decoded.encoding == "utf-8-sig"
    assert decoded.decided_by == "bom"


def test_decode_respects_utf16_boms() -> None:
    little = decode_bytes(b"\xff\xfe" + "小端".encode("utf-16-le"))
    assert little.text == "小端"
    assert little.encoding == "utf-16-le"
    big = decode_bytes(b"\xfe\xff" + "大端".encode("utf-16-be"))
    assert big.text == "大端"
    assert big.encoding == "utf-16-be"


def test_utf32_bom_must_be_checked_before_utf16() -> None:
    """UTF-32-LE 的 BOM 以 UTF-16-LE 的 BOM 开头：顺序写反会得到"每个字符后跟一个 NUL"."""
    decoded = decode_bytes(b"\xff\xfe\x00\x00" + "宽字符".encode("utf-32-le"))
    assert decoded.text == "宽字符"
    assert decoded.encoding == "utf-32-le"


def test_decode_rejects_binary_content() -> None:
    with pytest.raises(UnsupportedDocument, match="看起来是二进制文件"):
        decode_bytes(b"\x00\x01\x02\x03", source="fake.txt")


def test_decode_rejects_a_bom_that_does_not_match_the_content() -> None:
    """BOM 是明示：按它解不开说明文件坏了，而不是"我们猜错了"."""
    with pytest.raises(UnsupportedDocument, match="按它解码失败"):
        decode_bytes(b"\xfe\xff\x00\x41\x00", source="broken.txt")


def test_decode_of_empty_input_is_confident() -> None:
    decoded = decode_bytes(b"")
    assert decoded.text == ""
    assert decoded.confident is True


def test_looks_binary_uses_only_nul() -> None:
    assert looks_binary(b"abc\x00def") is True
    assert looks_binary(b"abc\x1bdef") is False   # 控制字符不算（误判代价不对称）
    assert looks_binary("中文".encode("utf-8")) is False


# --------------------------------------------------------------------------- #
# 纯文本加载器
# --------------------------------------------------------------------------- #


def test_split_paragraphs_uses_blank_lines() -> None:
    assert split_paragraphs("第一段\n还在第一段\n\n第二段") == ["第一段\n还在第一段", "第二段"]
    assert split_paragraphs("\n\n\n") == []
    assert split_paragraphs("") == []


def test_text_loader_builds_one_block_per_paragraph() -> None:
    document = TextLoader().load_bytes(
        "第一段。\n\n第二段。\n".encode("utf-8"), source="a.txt", media_type=MEDIA_TEXT
    )
    assert [block.kind for block in document.blocks] == ["paragraph", "paragraph"]
    assert document.metadata["encoding"] == "utf-8"
    assert document.metadata["encoding_confident"] == "true"
    assert document.metadata["paragraph_count"] == "2"


def test_text_loader_marks_untrusted_encoding_in_metadata() -> None:
    document = TextLoader().load_bytes(b"\x80\x80", source="a.txt", media_type=MEDIA_TEXT)
    assert document.metadata["encoding_confident"] == "false"
    assert document.metadata["encoding"] == "latin-1"


def test_decoded_text_to_dict_is_json_ready() -> None:
    payload = decode_bytes("中文".encode("utf-8")).to_dict()
    assert payload == {
        "text": "中文",
        "encoding": "utf-8",
        "decided_by": "strict",
        "confident": True,
    }


def test_text_loader_rejects_binary() -> None:
    with pytest.raises(UnsupportedDocument):
        TextLoader().load_bytes(b"\x00\x01", source="a.txt", media_type=MEDIA_TEXT)
