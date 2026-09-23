"""day061 ``documents.types`` 的单元测试：归一化形状与内容指纹."""

from __future__ import annotations

import json

import pytest

from smart_research_agent.documents import (
    BLOCK_HEADING,
    BLOCK_KINDS,
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    DOC_ID_LENGTH,
    Block,
    Document,
    DocumentError,
    content_id,
    make_document,
    normalize_text,
    render_table,
)


# --------------------------------------------------------------------------- #
# 规范化：doc_id 建立在它上面
# --------------------------------------------------------------------------- #


def test_normalize_unifies_line_endings() -> None:
    """Windows 与 Unix 的换行是同一份内容——不统一的话指纹会凭空不同."""
    assert normalize_text("a\r\nb") == normalize_text("a\nb") == "a\nb\n"
    assert normalize_text("a\rb") == "a\nb\n"


def test_normalize_strips_trailing_whitespace_per_line() -> None:
    assert normalize_text("a   \nb\t\n") == "a\nb\n"


def test_normalize_collapses_blank_runs() -> None:
    """连续空行折叠到一个：不同编辑器导出的空行数量本来就不同."""
    assert normalize_text("a\n\n\n\nb") == "a\n\nb\n"
    assert normalize_text("a\n\nb") == "a\n\nb\n"


def test_normalize_adds_a_single_trailing_newline() -> None:
    """有无结尾换行是**同一份内容**，统一补一个，让两者指纹相同."""
    assert normalize_text("a") == normalize_text("a\n") == normalize_text("a\n\n\n") == "a\n"


def test_normalize_does_not_touch_full_width_punctuation() -> None:
    """不做 Unicode 规范化：中英文混排里全角逗号常是内容差异而不是格式差异."""
    assert normalize_text("第一，第二") == "第一，第二\n"
    assert normalize_text("第一, 第二\t") == "第一, 第二\n"


def test_normalize_handles_empty_and_whitespace_only() -> None:
    assert normalize_text("") == ""
    assert normalize_text("   \n\n  ") == ""


def test_content_id_is_a_stable_short_hash() -> None:
    fingerprint = content_id(normalize_text("同一份内容"))
    assert fingerprint == content_id(normalize_text("同一份内容"))
    assert len(fingerprint) == DOC_ID_LENGTH == 16
    assert all(char in "0123456789abcdef" for char in fingerprint)


def test_content_id_changes_when_one_byte_changes() -> None:
    """改一个字节指纹就变——与 day058 的三元组"内容寻址"是同一种设计."""
    assert content_id("甲") != content_id("乙")


# --------------------------------------------------------------------------- #
# Block
# --------------------------------------------------------------------------- #


def test_block_kinds_are_exactly_six() -> None:
    assert BLOCK_KINDS == (
        "heading",
        "paragraph",
        "code",
        "list_item",
        "table",
        "quote",
    )


def test_block_rejects_unknown_kind() -> None:
    with pytest.raises(DocumentError, match="未知块类型"):
        Block(kind="footnote", text="x")


def test_block_rejects_heading_level_out_of_range() -> None:
    """标题只有 1~6 档：Markdown 与 HTML 都是这六档，别的值一定是解析错了."""
    Block(kind=BLOCK_HEADING, text="标题", level=6)
    with pytest.raises(DocumentError, match="标题层级必须是 1~6"):
        Block(kind=BLOCK_HEADING, text="标题", level=0)
    with pytest.raises(DocumentError, match="标题层级必须是 1~6"):
        Block(kind=BLOCK_HEADING, text="标题", level=7)


def test_table_block_requires_rows() -> None:
    """没有行列结构的表格与普通段落没有区别——那就该用 paragraph 而不是 table."""
    with pytest.raises(DocumentError, match="表格块必须有 rows"):
        Block(kind=BLOCK_TABLE, text="| a |")
    block = Block(kind=BLOCK_TABLE, text="| a |", rows=(("a",),))
    assert block.to_dict()["rows"] == [["a"]]


def test_block_is_empty_only_when_both_text_and_rows_are_blank() -> None:
    assert Block(kind=BLOCK_PARAGRAPH, text="   \n ").is_empty is True
    assert Block(kind=BLOCK_PARAGRAPH, text="x").is_empty is False
    assert Block(kind=BLOCK_TABLE, text="", rows=(("x",),)).is_empty is False


def test_block_summary_marks_level_language_and_table_size() -> None:
    assert "heading2" in Block(kind=BLOCK_HEADING, text="x", level=2).summary_line()
    code = Block(kind="code", text="print(1)", language="python").summary_line()
    assert "[python]" in code
    table = Block(kind=BLOCK_TABLE, text="| a |", rows=(("a", "b"), ("c", "d"))).summary_line()
    assert "2×2" in table
    long_block = Block(kind=BLOCK_PARAGRAPH, text="x" * 100).summary_line()
    assert len(long_block) < 80


# --------------------------------------------------------------------------- #
# Document 与组装
# --------------------------------------------------------------------------- #


def test_document_rejects_blank_identity_fields() -> None:
    with pytest.raises(DocumentError, match="文档必须有 source"):
        Document(source="", media_type="text/plain", text="x")
    with pytest.raises(DocumentError, match="文档必须有 media_type"):
        Document(source="a", media_type="", text="x")


def test_document_rejects_a_mismatched_doc_id() -> None:
    """外部传进来的 doc_id 与内容不符时必须拒绝：否则去重与溯源同时失效."""
    with pytest.raises(DocumentError, match="doc_id 与文本内容不符"):
        Document(source="a", media_type="text/plain", text="x", doc_id="0" * 16)


def test_make_document_derives_everything_from_blocks() -> None:
    document = make_document(
        source="a.md",
        media_type="text/markdown",
        blocks=[
            Block(kind=BLOCK_HEADING, text="标题", level=1),
            Block(kind=BLOCK_PARAGRAPH, text="正文"),
            Block(kind=BLOCK_PARAGRAPH, text="   "),
        ],
        metadata={"encoding": "utf-8"},
    )
    assert document.text == "标题\n\n正文\n"
    assert len(document.blocks) == 2  # 空块被丢掉
    assert document.doc_id == content_id(document.text)
    assert document.fingerprint == document.doc_id
    assert document.char_count == len(document.text)
    assert document.metadata["encoding"] == "utf-8"


def test_full_text_is_always_derived_from_the_blocks() -> None:
    """``make_document`` **不接受**一个覆盖用的 text：两份内容不一致是所有麻烦的开头.

    这条测试的写法是"接口形状本身"的断言：一旦有人加回 ``text`` 参数，
    它不会有任何表现——因此这里显式检查调用签名。
    """
    import inspect

    parameters = inspect.signature(make_document).parameters
    assert "text" not in parameters
    assert set(parameters) == {"source", "media_type", "blocks", "metadata"}


def test_headings_and_block_counts_are_complete() -> None:
    document = make_document(
        source="a.md",
        media_type="text/markdown",
        blocks=[
            Block(kind=BLOCK_HEADING, text="一", level=1),
            Block(kind=BLOCK_HEADING, text="二", level=2),
            Block(kind=BLOCK_PARAGRAPH, text="正文"),
        ],
    )
    assert [block.text for block in document.headings()] == ["一", "二"]
    counts = document.block_counts()
    assert set(counts) == set(BLOCK_KINDS)          # 六个键恒存在
    assert counts[BLOCK_HEADING] == 2
    assert counts["code"] == 0


def test_document_to_dict_can_omit_blocks() -> None:
    """一份几百页的文档，其块序列会比全文还大；报告通常只需要统计量."""
    document = make_document(
        source="a.md",
        media_type="text/markdown",
        blocks=[Block(kind=BLOCK_PARAGRAPH, text="正文")],
    )
    full = document.to_dict()
    slim = document.to_dict(include_blocks=False)
    assert "blocks" in full and "blocks" not in slim
    assert slim["block_count"] == 1
    assert slim["preview"].startswith("正文")
    assert json.loads(json.dumps(full))["doc_id"] == document.doc_id


def test_document_summary_lists_only_non_zero_block_kinds() -> None:
    document = make_document(
        source="a.md",
        media_type="text/markdown",
        blocks=[Block(kind=BLOCK_HEADING, text="一", level=1)],
    )
    line = document.summary_line()
    assert "heading=1" in line
    assert "code=" not in line
    assert "a.md" in line


def test_empty_block_list_yields_an_empty_but_valid_document() -> None:
    document = make_document(source="empty.txt", media_type="text/plain", blocks=[])
    assert document.text == ""
    assert document.blocks == ()
    assert document.char_count == 0
    assert document.fingerprint == content_id("")


# --------------------------------------------------------------------------- #
# 表格渲染：三个格式共用一份实现
# --------------------------------------------------------------------------- #


def test_render_table_is_markdown() -> None:
    rows = (("策略", "优点"), ("固定", "简单"), ("递归", "保留结构"))
    assert render_table(rows) == (
        "| 策略 | 优点 |\n| --- | --- |\n| 固定 | 简单 |\n| 递归 | 保留结构 |"
    )


def test_render_table_pads_short_rows() -> None:
    """行列不齐时补空单元格，而不是让渲染出的表格错位."""
    assert render_table((("a", "b", "c"), ("1",))).endswith("| 1 |  |  |")


def test_render_table_truncates_extra_cells() -> None:
    assert render_table((("a", "b"), ("1", "2", "3"))).endswith("| 1 | 2 |")


def test_render_table_handles_no_rows() -> None:
    assert render_table(()) == ""
    assert render_table([]) == ""
