"""day061 ``documents.markdown_loader`` 的单元测试."""

from __future__ import annotations

import pytest

from smart_research_agent.documents import (
    BLOCK_CODE,
    BLOCK_HEADING,
    BLOCK_LIST_ITEM,
    BLOCK_PARAGRAPH,
    BLOCK_QUOTE,
    BLOCK_TABLE,
    MEDIA_MARKDOWN,
    MarkdownLoader,
    markdown_syntax_table,
    parse_front_matter,
    split_front_matter,
)

SAMPLE = """---
title: 知识库设计
tags: [rag, chunk, "eval"]
author: 学习者
备注：
---

# 一级标题

正文第一段。

## 二级标题

```python
def chunk(text):
    return [text]
```

| 策略 | 优点 |
| --- | --- |
| 固定 | 简单 |
| 递归 | 保留结构 |

- 第一条
  - 子项
* 第二条

> 引用的一段话

结尾段落。
"""


def _blocks(text: str):
    return MarkdownLoader().parse(text)


def test_loader_declares_its_media_type_and_suffixes() -> None:
    loader = MarkdownLoader()
    assert loader.media_types == (MEDIA_MARKDOWN,)
    assert set(loader.suffixes) == {".md", ".markdown"}
    assert "CommonMark" not in loader.description  # 描述里放的是"做了什么"


def test_sample_yields_the_expected_block_sequence() -> None:
    kinds = [block.kind for block in _blocks(SAMPLE)]
    assert kinds == [
        BLOCK_HEADING,
        BLOCK_PARAGRAPH,
        BLOCK_HEADING,
        BLOCK_CODE,
        BLOCK_TABLE,
        BLOCK_LIST_ITEM,
        BLOCK_LIST_ITEM,
        BLOCK_LIST_ITEM,
        BLOCK_QUOTE,
        BLOCK_PARAGRAPH,
    ]


def test_atx_headings_need_a_space_after_the_hashes() -> None:
    """``#标签`` 是话题标签不是标题（CommonMark §4.2）——中文文档里它很常见."""
    blocks = _blocks("#真的不是标题\n\n# 是标题\n")
    assert [block.kind for block in blocks] == [BLOCK_PARAGRAPH, BLOCK_HEADING]
    assert blocks[0].text == "#真的不是标题"
    assert blocks[1].level == 1
    assert blocks[1].text == "是标题"


def test_heading_levels_are_recorded() -> None:
    blocks = _blocks("###### 六级\n\n####### 七级不是标题\n")
    assert blocks[0].kind == BLOCK_HEADING
    assert blocks[0].level == 6
    assert blocks[1].kind == BLOCK_PARAGRAPH


def test_closing_hashes_are_stripped() -> None:
    blocks = _blocks("## 标题 ##\n")
    assert blocks[0].text == "标题"
    assert blocks[0].level == 2


def test_fenced_code_keeps_language_and_whole_body() -> None:
    blocks = _blocks("```python\nprint(1)\nprint(2)\n```\n")
    assert len(blocks) == 1
    assert blocks[0].kind == BLOCK_CODE
    assert blocks[0].language == "python"
    assert blocks[0].text == "print(1)\nprint(2)"


def test_tilde_fences_work_too() -> None:
    blocks = _blocks("~~~\nplain\n~~~\n")
    assert blocks[0].kind == BLOCK_CODE
    assert blocks[0].language == ""


def test_unclosed_fence_still_yields_a_code_block() -> None:
    """没有收尾围栏时**内容照常落块**——"只分类不丢弃"是本模块的硬规则.

    同时验证尾部空行被去掉：``"``` 未闭合"`` 与 ``"``` 后面恰好有一个空行"``
    应当是同一份内容，不该让代码块里多出一个空行。
    """
    blocks = _blocks("```\n没有收尾\n")
    assert [block.kind for block in blocks] == [BLOCK_CODE]
    assert blocks[0].text == "没有收尾"


def test_gfm_table_keeps_rows_and_renders_markdown() -> None:
    blocks = _blocks("| A | B |\n| --- | --- |\n| 1 | 2 |\n")
    assert blocks[0].kind == BLOCK_TABLE
    assert blocks[0].rows == (("A", "B"), ("1", "2"))
    assert blocks[0].text.startswith("| A | B |")


def test_pipe_line_without_delimiter_is_a_paragraph() -> None:
    """没有分隔行的管道符行不是表格（GFM §4.10 要求分隔行）."""
    blocks = _blocks("a | b | c\n")
    assert blocks[0].kind == BLOCK_PARAGRAPH


def test_quote_block_joins_consecutive_quote_lines() -> None:
    blocks = _blocks("> 第一行\n> 第二行\n\n正文\n")
    assert blocks[0].kind == BLOCK_QUOTE
    assert blocks[0].text == "第一行\n第二行"
    assert blocks[1].kind == BLOCK_PARAGRAPH


def test_list_items_record_indentation_level() -> None:
    blocks = _blocks("- 顶层\n  - 子项\n1. 有序\n")
    assert [block.kind for block in blocks] == [BLOCK_LIST_ITEM] * 3
    assert blocks[0].level == 0
    assert blocks[1].level == 1
    assert blocks[2].text == "有序"


def test_front_matter_is_extracted_into_metadata_not_body() -> None:
    """front matter 是元数据：当正文会让每份文档的前几行都是同一堆字段名."""
    document = MarkdownLoader().load_bytes(
        SAMPLE.encode("utf-8"), source="a.md", media_type=MEDIA_MARKDOWN
    )
    assert document.metadata["has_front_matter"] == "true"
    assert document.metadata["front_matter.title"] == "知识库设计"
    assert document.metadata["front_matter.tags"] == "rag, chunk, eval"
    assert "title:" not in document.text
    assert document.text.startswith("一级标题")   # 标题的 `#` 是标记，不进文本


def test_split_front_matter_requires_the_first_line() -> None:
    """正文中间的分割线**不是** front matter（Jekyll 的约定是"文件最开头"）."""
    metadata, body = split_front_matter("正文\n---\ntitle: x\n---\n后面\n")
    assert metadata == {}
    assert body.startswith("正文")


def test_split_front_matter_without_closing_line_keeps_everything() -> None:
    """没有闭合线时整份文件都是正文——否则一份以 ``---`` 开头的文档会被整份吞掉."""
    metadata, body = split_front_matter("---\ntitle: x\n正文\n")
    assert metadata == {}
    assert body.startswith("---")


def test_parse_front_matter_handles_simple_forms_only() -> None:
    metadata = parse_front_matter(
        "# 注释\ntitle: \"引号标题\"\ntags: [a, b]\nnested: {x: 1}\n裸行\n"
    )
    assert metadata["title"] == "引号标题"
    assert metadata["tags"] == "a, b"
    assert metadata["nested"] == "{x: 1}"
    assert metadata["front_matter_raw"] == "裸行"


def test_parse_front_matter_ignores_blank_and_comment_lines() -> None:
    assert parse_front_matter("\n   \n# c\nkey: v\n") == {"key": "v"}


def test_parse_front_matter_with_empty_key_goes_to_raw() -> None:
    metadata = parse_front_matter(": 没有键\nk: v\n")
    assert metadata["k"] == "v"
    assert metadata["front_matter_raw"] == ": 没有键"


def _content_of(line: str) -> str:
    """去掉行首的 Markdown 标记，只留内容（用于"内容没被丢掉"的检查）.

    标记本身（``#`` / ``-`` / ``>`` / ``| --- |``）不进块的 ``text``——
    标题块的 text 是标题文字，表格块的 text 是重新渲染后的 Markdown。
    因此这里比的是**内容**，而不是原始字节。
    """
    stripped = line.strip()
    for prefix in ("```", "~~~", "######", "#####", "####", "###", "##", "#", ">", "-", "*", "+"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].strip()
            break
    return stripped


def test_every_non_blank_line_survives_parsing() -> None:
    """**本模块最硬的一条规则**：结构化解析最容易犯的错是让内容静默消失.

    因此这里逐行检查（去掉 Markdown 标记之后）：原文里每一个非空行
    都必须能在某一块里找到。
    """
    document = MarkdownLoader().load_bytes(
        SAMPLE.encode("utf-8"), source="a.md", media_type=MEDIA_MARKDOWN
    )
    # 代码块的"语言"存在 ``block.language`` 而不是 text 里，因此它也要进检查的靶子。
    joined = "\n".join(
        f"{block.text}\n{block.language}" for block in document.blocks
    )
    body = SAMPLE.split("---\n", 2)[2]
    for line in body.split("\n"):
        content = _content_of(line)
        if content and not set(content) <= {"|", "-", " "}:
            assert content in joined, f"这一行被丢掉了：{line!r}"


def test_loader_metadata_counts_headings() -> None:
    document = MarkdownLoader().load_bytes(
        b"# a\n\n## b\n\ntext\n", source="a.md", media_type=MEDIA_MARKDOWN
    )
    assert document.metadata["heading_count"] == "2"
    assert document.metadata["has_front_matter"] == "false"
    assert document.metadata["encoding"] == "utf-8"


def test_loader_survives_crlf_and_gbk() -> None:
    document = MarkdownLoader().load_bytes(
        "# 标题\r\n\r\n正文。\r\n".encode("gb18030"), source="a.md", media_type=MEDIA_MARKDOWN
    )
    assert document.blocks[0].text == "标题"
    assert document.metadata["encoding"] == "gb18030"


def test_markdown_syntax_table_cites_specs() -> None:
    rows = markdown_syntax_table()
    structures = {row["structure"]: row for row in rows}
    assert "§4.2" in structures["ATX 标题"]["spec"]
    assert "§4.5" in structures["围栏代码块"]["spec"]
    assert "§4.10" in structures["表格"]["spec"]
    assert all(row["block"] for row in rows)


def test_parse_of_empty_text_yields_no_blocks() -> None:
    assert _blocks("") == []
    assert _blocks("\n\n\n") == []


def test_table_row_with_extra_cells_is_padded_by_the_renderer() -> None:
    blocks = _blocks("| A | B |\n| --- | --- |\n| 1 |\n")
    assert blocks[0].rows == (("A", "B"), ("1",))
    assert blocks[0].text.endswith("| 1 |  |")
