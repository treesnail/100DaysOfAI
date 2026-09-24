"""day061 ``documents.pdf_loader`` 的单元测试：零依赖文本抽取与它的边界."""

from __future__ import annotations

import pytest

from smart_research_agent.documents import (
    BLOCK_PARAGRAPH,
    MEDIA_PDF,
    PDF_SIGNATURE,
    PdfLoader,
    UnsupportedDocument,
    decode_literal_string,
    extract_lines,
    extract_pdf_text,
    iter_streams,
    maybe_inflate,
    pdf_boundaries,
)
from tests.document_samples import build_pdf, build_pdf_with_gap


def _load(data: bytes, source: str = "a.pdf"):
    return PdfLoader().load_bytes(data, source=source, media_type=MEDIA_PDF)


# --------------------------------------------------------------------------- #
# 字面字符串与转义
# --------------------------------------------------------------------------- #


def test_literal_string_decodes_latin1_by_default() -> None:
    assert decode_literal_string("Hello".encode("latin-1")) == "Hello"


def test_literal_string_decodes_utf16be_with_bom() -> None:
    """PDF 2.0 允许字符串以 ``FE FF`` 开头表示 UTF-16BE：非 ASCII 唯一可移植的写法."""
    assert decode_literal_string(b"\xfe\xff" + "中文".encode("utf-16-be")) == "中文"


def test_literal_escapes_are_decoded() -> None:
    content = rb"(line1\nline2\t\(paren\)\\backslash\053) Tj"
    lines, _ = extract_lines(content)
    assert lines[-1][1] == "line1\nline2\t(paren)\\backslash+"


def test_octal_escape_takes_at_most_three_digits() -> None:
    """``\\0536`` 是 ``+6`` 而不是一个码点 536（规范 §7.3.4.2）."""
    lines, _ = extract_lines(rb"(\0536) Tj")
    assert lines[-1][1] == "+6"


def test_hex_strings_are_supported() -> None:
    lines, _ = extract_lines(b"<48656C6C6F> Tj")
    assert lines[-1][1] == "Hello"


def test_hex_string_with_odd_digits_is_padded() -> None:
    lines, _ = extract_lines(b"<41424> Tj")
    assert lines[-1][1] == "AB@"


def test_backslash_at_end_of_line_continues_without_adding_anything() -> None:
    lines, _ = extract_lines(b"(abc\\\ndef) Tj")
    assert lines[-1][1] == "abcdef"


def test_truncated_content_stream_does_not_raise() -> None:
    """内容流被截断时**不抛异常**：截到什么算什么（后面还有别的流要处理）.

    ``(unterminated`` 没有 ``)``，因此词法分析器会把剩下的一切都当成字符串内容——
    它是对的：在没有收尾括号的情况下，**我们无法知道哪一位才是真正的结束**。
    """
    from smart_research_agent.documents.pdf_loader import _Lexer

    assert list(_Lexer(b"(unterminated").tokens()) == [("str", b"unterminated")]
    lines, warnings = extract_lines(b"BT (unterminated")
    assert lines == []          # 没有 Tj，因此什么都不显示
    assert warnings == []


# --------------------------------------------------------------------------- #
# 操作符
# --------------------------------------------------------------------------- #


def test_td_with_vertical_delta_starts_a_new_line() -> None:
    lines, _ = extract_lines(b"BT (first) Tj 0 -14 Td (second) Tj ET")
    assert [text for _, text in lines if text] == ["first", "second"]


def test_apostrophe_and_quote_operators_show_text_on_a_new_line() -> None:
    lines, _ = extract_lines(b"BT (one) Tj (two) ' (three) \" ET")
    assert [text for _, text in lines if text] == ["one", "two", "three"]


def test_tj_array_inserts_a_space_for_large_negative_offsets() -> None:
    """``TJ`` 数组里的负位移是**千分之一 em**，大于阈值即判为词间空格."""
    lines, _ = extract_lines(b"BT [(Hello) -300 (World)] TJ ET")
    assert lines[-1][1] == "Hello World"
    small, _ = extract_lines(b"BT [(Hel) -10 (lo)] TJ ET")
    assert small[-1][1] == "Hello"


def test_comments_and_whitespace_are_ignored() -> None:
    lines, _ = extract_lines(b"% comment\n BT (text) Tj ET")
    assert lines[-1][1] == "text"


def test_numbers_are_not_mistaken_for_text() -> None:
    lines, _ = extract_lines(b"BT 12 14 1.5 (real) Tj ET")
    assert [text for _, text in lines if text] == ["real"]


def test_operator_without_operands_does_not_crash() -> None:
    """缺操作数的操作符只留下一个空行——**空行是段落信号**，不是错误."""
    lines, _ = extract_lines(b"BT Td T* ET")
    assert [text for _, text in lines] == [""]


def test_line_length_is_capped() -> None:
    long_text = b"x" * 5000
    _, warnings = extract_lines(b"BT (" + long_text + b") Tj ET")
    assert any("超过上限" in item for item in warnings)


def test_repeated_empty_lines_are_collapsed() -> None:
    """``ET`` 与收尾各提交一次会产生两个空行，但**同一份内容不该有两种空行写法**."""
    lines, _ = extract_lines(b"BT (a) Tj ET T* T* ET")
    assert [text for _, text in lines] == ["a", ""]


def test_paragraph_grouping_uses_the_line_gap() -> None:
    """行距中位数的 1.5 倍以上判为段落边界——用中位数是因为标题会把平均值拉飞.

    样本：第一段三行（行距 14），随后一次性下移 40 再写两行。
    中位行距是 14，阈值 21，而 40 > 21，因此应当得到两段。
    """
    extraction = extract_pdf_text(build_pdf_with_gap(["a", "b", "c"], ["d", "e"]))
    assert extraction.paragraphs == ["a\nb\nc", "d\ne"]


def test_single_tm_jump_is_a_paragraph_boundary() -> None:
    """单行文档只有一个段落——**样本里刻意只用 ASCII**：本模块按 latin-1 处理
    字符串字面量，中文要靠 UTF-16BE 或 CID 字体，见 5.4 节的边界讨论。"""
    extraction = extract_pdf_text(build_pdf(["only one line"]))
    assert extraction.paragraphs == ["only one line"]


# --------------------------------------------------------------------------- #
# 流与解压
# --------------------------------------------------------------------------- #


def test_iter_streams_finds_each_stream() -> None:
    streams = list(iter_streams(b"stream\nAAA\nendstream\nstream\nBBB\nendstream"))
    assert streams == [b"AAA\n", b"BBB\n"]


def test_maybe_inflate_handles_compressed_and_raw() -> None:
    import zlib

    payload = zlib.compress(b"hello")
    assert maybe_inflate(payload) == (b"hello", True)
    assert maybe_inflate(b"not compressed") == (b"not compressed", False)
    assert maybe_inflate(b"") == (b"", False)


def test_compressed_pdf_is_extracted() -> None:
    document = _load(build_pdf(["First line.", "Second line."]))
    assert "First line." in document.text
    assert "Second line." in document.text
    assert document.metadata["pdf_pages"] == "1"
    assert document.metadata["pdf_streams"] == "1"
    assert document.metadata["pdf_inflated_streams"] == "1"
    assert document.metadata["pdf_cid_font_detected"] == "false"


def test_uncompressed_pdf_is_extracted_without_a_warning() -> None:
    """未压缩的内容流是合法写法：把它报成诊断会让真正的信号被噪声稀释."""
    document = _load(build_pdf(["Uncompressed line."], compress=False))
    assert "Uncompressed line." in document.text
    assert document.metadata["pdf_inflated_streams"] == "0"
    assert document.metadata["pdf_warnings"] == ""


# --------------------------------------------------------------------------- #
# 拒收与边界
# --------------------------------------------------------------------------- #


def test_pdf_loader_rejects_a_non_pdf() -> None:
    with pytest.raises(UnsupportedDocument, match="不是 PDF"):
        _load(b"not a pdf at all")


def test_encrypted_pdf_is_rejected_not_garbled() -> None:
    """返回一堆乱码比拒收更糟——乱码会一路走进索引，而且看不出来."""
    with pytest.raises(UnsupportedDocument, match="加密的 PDF"):
        _load(build_pdf(["x"], extra=b"/Encrypt 5 0 R"))


def test_cid_fonts_raise_a_loud_warning() -> None:
    """Identity-H 的字符码是字形索引：规范 §9.10.2 判定为不可确定.

    这是本模块最重要的边界，因此它必须出现在**三条**地方：
    元数据字段、告警文本、以及 ``pdf_boundaries()`` 的 not_supported 列表。
    """
    data = build_pdf(["x"], extra=b"/Type0 /Identity-H /ToUnicode")
    document = _load(data)
    assert document.metadata["pdf_cid_font_detected"] == "true"
    assert "§9.10.2" in document.metadata["pdf_warnings"]
    assert document.metadata["encoding_confident"] == "false"
    assert any("Identity-H" in item for item in pdf_boundaries()["not_supported"])


def test_pdf_without_text_streams_still_reports_diagnostics() -> None:
    data = b"%PDF-1.4\n1 0 obj\n<< /Type /Page >>\nendobj\nstream\n\x01\x02\nendstream\n%%EOF\n"
    document = _load(data)
    assert document.blocks == ()
    assert document.metadata["pdf_pages"] == "1"
    assert "既没有流解压成功" in document.metadata["pdf_warnings"]


def test_pdf_metadata_does_not_lie_about_encoding() -> None:
    """PDF 的字符串是"字节 + 字体编码"，因此这里**没有单一编码可报**."""
    document = _load(build_pdf(["text"]))
    assert "取决于字体" in document.metadata["encoding"]


def test_extraction_summary_line_is_readable() -> None:
    extraction = extract_pdf_text(build_pdf(["a", "b"]))
    line = extraction.summary_line()
    assert "1 页" in line
    assert "行" in line and "段" in line
    assert extraction.to_dict()["pages"] == 1
    assert PDF_SIGNATURE == b"%PDF-"


def test_pdf_boundaries_documents_operators_and_limits() -> None:
    boundaries = pdf_boundaries()
    operators = {row["operator"] for row in boundaries["operators"]}
    assert "BT / ET" in operators
    assert "Tj / ' / \" / TJ" in operators
    assert boundaries["filters"][0]["name"] == "FlateDecode"
    assert len(boundaries["not_supported"]) >= 5
    assert boundaries["reference_versions"]["pypdf"] == "6.18.0（2026-09-08）"


def test_pdf_paragraph_blocks_use_the_paragraph_kind() -> None:
    document = _load(build_pdf(["single paragraph"]))
    assert [block.kind for block in document.blocks] == [BLOCK_PARAGRAPH]
    assert document.blocks[0].text == "single paragraph"


# --------------------------------------------------------------------------- #
# 对损坏/异常内容流的容忍
# --------------------------------------------------------------------------- #


def test_nested_parentheses_inside_a_literal_string() -> None:
    """括号可以嵌套：``(a(b)c)`` 是一个字符串，不是三个记号."""
    lines, _ = extract_lines(b"(a(b)c) Tj")
    assert lines[-1][1] == "a(b)c"


def test_hex_strings_ignore_whitespace() -> None:
    lines, _ = extract_lines(b"<48 65 6C 6C 6F> Tj")
    assert lines[-1][1] == "Hello"


def test_invalid_hex_string_yields_empty_text_instead_of_raising() -> None:
    """``<zz>`` 不是合法十六进制：返回空串而不是让整份文档失败."""
    lines, _ = extract_lines(b"<zz> Tj")
    assert lines[-1][1] == ""


def test_stray_closing_delimiter_is_skipped() -> None:
    """孤立的 ``)`` 会被跳过：内容流损坏时**跳过坏字节**比抛异常有用."""
    from smart_research_agent.documents.pdf_loader import _Lexer

    assert list(_Lexer(b") Tj").tokens()) == [("op", "Tj")]


def test_dictionary_markers_are_recognized() -> None:
    from smart_research_agent.documents.pdf_loader import _Lexer

    tokens = list(_Lexer(b"<< /Type /Page >>").tokens())
    assert tokens[0] == ("op", "<<")
    assert ("name", "Type") in tokens


def test_pending_buffer_is_capped() -> None:
    """操作数栈最多保留 16 项：超出的丢掉也不会影响 Tj（它只看最近的那一项）."""
    operands = b" ".join(str(index).encode() for index in range(40))
    lines, _ = extract_lines(b"BT " + operands + b" (still here) Tj ET")
    assert lines[-1][1] == "still here"


def test_stream_without_endstream_is_ignored() -> None:
    """``endstream`` 缺失时不抛异常，只是这一条流拿不到——后面还有别的流要处理."""
    assert list(iter_streams(b"stream\nAAAA")) == []
    assert list(iter_streams(b"nothing here")) == []
    extraction = extract_pdf_text(
        b"%PDF-1.4\n1 0 obj\n<< /Type /Page >>\nstream\ntruncated\n%%EOF\n"
    )
    assert extraction.streams == 0
    assert extraction.paragraphs == []


def test_empty_lines_do_not_start_a_paragraph() -> None:
    """只有空行时没有段落——空行是"断开"的信号，不是内容."""
    extraction = extract_pdf_text(build_pdf([""]))
    assert extraction.paragraphs == []
    assert extraction.to_dict()["paragraphs"] == 0
