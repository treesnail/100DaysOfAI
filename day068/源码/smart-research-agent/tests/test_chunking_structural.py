"""day062 结构分块（``chunking.structural``）的单元测试.

四组重点：

1. 标题栈（``build_sections``）——``level >= 当前`` 就弹栈的规则；
2. 精确路径——块区间、breadcrumb、不切开代码块与表格；
3. 退化路径——小节定位失败时按段落切并把块区间记 ``-1``；
4. 与重叠的关系——结构策略**显式拒绝**非零重叠。
"""

from __future__ import annotations

import pytest

from smart_research_agent.chunking import (
    ATOMIC_KINDS,
    STRATEGY_STRUCTURAL,
    ChunkingError,
    ChunkPolicy,
    Section,
    build_chunker,
    build_sections,
    default_policy,
)
from smart_research_agent.documents import (
    BLOCK_CODE,
    BLOCK_HEADING,
    BLOCK_PARAGRAPH,
    BLOCK_TABLE,
    Block,
    Document,
    content_id,
    make_document,
)
from smart_research_agent.documents.types import BLOCK_QUOTE
from tests.chunking_samples import (
    big_code_document,
    chars,
    guide_document,
    plain_document,
    text_document,
)

LONG_TABLE = (
    "# 对照表\n\n"
    "| 维度 | 说明 |\n|------|------|\n"
    "| 召回 | " + "甲" * 24 + " |\n"
    "| 精度 | " + "乙" * 24 + " |\n\n"
    "表格之后的一段正文。\n"
)


def structural(policy: ChunkPolicy):  # type: ignore[no-untyped-def]
    return build_chunker(STRATEGY_STRUCTURAL, policy, measurer=chars())


# --------------------------------------------------------------------------- #
# 标题栈
# --------------------------------------------------------------------------- #


def test_build_sections_uses_a_heading_stack() -> None:
    """``h1 / h2 / h3 / h2``：第三个标题之后，``h3`` 必须被弹掉.

    用"记住上一个标题"的写法会让第二个 ``h2`` 小节错误地继承 ``h3`` 的名字。
    """
    document = guide_document()
    sections = build_sections(document)
    assert [section.heading_path for section in sections] == [
        ("检索手册",),
        ("检索手册", "分块"),
        ("检索手册", "检索"),
        ("检索手册", "成本"),
    ]
    assert [section.level for section in sections] == [1, 2, 2, 2]
    assert all(section.blocks for section in sections)


def test_build_sections_keeps_content_before_the_first_heading() -> None:
    """标题之前的内容自成一小节（``level=0``、路径为空），**不会被丢掉**.

    前言里经常放着整个文档的定位信息（"本文适用的版本是 …"）。
    """
    document = make_document(
        source="docs/preamble.md",
        media_type="text/markdown",
        blocks=[
            Block(kind=BLOCK_PARAGRAPH, text="本文适用 v2。"),
            Block(kind=BLOCK_HEADING, text="正文", level=1),
            Block(kind=BLOCK_PARAGRAPH, text="正文内容。"),
        ],
    )
    sections = build_sections(document)
    assert len(sections) == 2
    assert sections[0] == Section(
        heading_path=(), level=0, title="", blocks=(0,)
    )
    assert sections[1].heading_path == ("正文",)


def test_build_sections_returns_one_section_for_headingless_document() -> None:
    """没有标题的文档只有一个小节：**结构策略由此退化为按段落切**."""
    sections = build_sections(plain_document())
    assert len(sections) == 1
    assert sections[0].heading_path == ()
    assert sections[0].level == 0


# --------------------------------------------------------------------------- #
# 精确路径
# --------------------------------------------------------------------------- #


def test_structural_chunks_carry_their_breadcrumb() -> None:
    """每块带 ``heading_path``：一句"阈值设为 0.85"靠它才说得清自己在讲什么."""
    document = guide_document()
    result = structural(default_policy(STRATEGY_STRUCTURAL)).split(document)
    paths = [chunk.heading_path for chunk in result.chunks]
    assert ("检索手册", "检索") in paths
    assert ("检索手册", "成本") in paths
    assert any("检索手册 > 检索" in chunk.retrieval_text for chunk in result.chunks)


def test_structural_chunks_are_exact_substrings_with_precise_block_ranges() -> None:
    """块文本是原文的连续子串，而且**块区间精确可考**."""
    document = guide_document()
    result = structural(default_policy(STRATEGY_STRUCTURAL)).split(document)
    for chunk in result.chunks:
        assert document.text[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.start_block >= 0 and chunk.end_block >= chunk.start_block
        for index in range(chunk.start_block, chunk.end_block + 1):
            assert document.blocks[index].text  # 区间内的块都是真实存在的
    assert result.metadata["unlocated_sections"] == "0"
    assert result.metadata["sections"] == "4"


def test_structural_protects_the_code_block() -> None:
    """超预算的代码块**整块保留**并标记 ``oversized``：不产生残码."""
    document = big_code_document()
    code_block = next(block for block in document.blocks if block.kind == BLOCK_CODE)
    assert len(code_block.text) > 40
    result = structural(ChunkPolicy(strategy=STRATEGY_STRUCTURAL, max_tokens=40)).split(
        document
    )
    holders = [
        chunk for chunk in result.chunks if code_block.text in chunk.text
    ]
    assert len(holders) == 1
    assert holders[0].oversized is True
    assert holders[0].reason == "atom:code"
    assert result.oversized_count == 1
    assert holders[0].token_count > 40


def test_structural_protects_tables_too() -> None:
    """表格与代码块同属原子块：切开一个表格得到的是两段对不上的行列."""
    document = text_document(LONG_TABLE, source="docs/table.md", media_type="text/markdown")
    table_block = next(block for block in document.blocks if block.kind == BLOCK_TABLE)
    result = structural(ChunkPolicy(strategy=STRATEGY_STRUCTURAL, max_tokens=40)).split(
        document
    )
    holders = [chunk for chunk in result.chunks if table_block.text in chunk.text]
    assert len(holders) == 1
    assert holders[0].oversized is True
    assert holders[0].reason == "atom:table"
    assert BLOCK_TABLE in ATOMIC_KINDS and BLOCK_CODE in ATOMIC_KINDS


def test_structural_splits_an_oversized_paragraph_inside_the_block() -> None:
    """普通段落超预算时继续下切（只有代码与表格才享受"整块保留"）."""
    document = text_document(
        "# 长段落\n\n" + "这句话很长。" * 12 + "\n",
        source="docs/long.md",
        media_type="text/markdown",
    )
    result = structural(
        ChunkPolicy(strategy=STRATEGY_STRUCTURAL, max_tokens=30)
    ).split(document)
    assert result.count >= 2
    assert result.oversized_count == 0
    assert any("oversized-para" in chunk.reason for chunk in result.chunks)


def test_structural_degrades_when_a_section_cannot_be_located() -> None:
    """小节定位失败时按段落切，并把块区间记 ``-1``（**不猜一个值**）.

    样本刻意让某一个段落的文本在规范化后与原块不同（连续 4 个换行会被
    折叠成 1 个空行），于是"把这一节的块拼起来"在全文里找不到——
    这正是真实数据里"解析结果与全文有一处对不上"的样子。
    """
    document = make_document(
        source="docs/odd.md",
        media_type="text/markdown",
        blocks=[
            Block(kind=BLOCK_HEADING, text="标题", level=1),
            Block(kind=BLOCK_PARAGRAPH, text="a\n\n\n\nb"),
        ],
    )
    result = structural(ChunkPolicy(strategy=STRATEGY_STRUCTURAL, max_tokens=100)).split(
        document
    )
    assert result.metadata["unlocated_sections"] == "1"
    assert result.count == 1
    assert result.chunks[0].start_block == -1
    assert result.chunks[0].end_block == -1
    assert "para" in result.chunks[0].reason


def test_structural_degrades_to_paragraphs_without_a_block_sequence() -> None:
    """没有块序列时结构策略失去全部输入，退化为按段落切（并留下原因）."""
    document = Document(
        source="docs/no-blocks.md",
        media_type="text/plain",
        text="第一段。\n\n第二段。\n",
        doc_id=content_id("第一段。\n\n第二段。\n"),
    )
    result = structural(ChunkPolicy(strategy=STRATEGY_STRUCTURAL, max_tokens=4)).split(
        document
    )
    assert [chunk.text for chunk in result.chunks] == ["第一段。", "第二段。"]
    assert result.metadata["degraded_reason"] == "no-blocks"
    assert result.metadata["sections"] == "0"


# --------------------------------------------------------------------------- #
# 与重叠、合并的关系
# --------------------------------------------------------------------------- #


def test_structural_rejects_non_zero_overlap() -> None:
    """标题边界就是语义边界：跨边界回带会把上一节内容挂到下一节的路径下.

    报错信息必须说清**为什么**（而不是"不支持"），否则下一个人会去调小
    overlap 把它"绕过去"——而正确动作是不重叠。
    """
    with pytest.raises(ChunkingError) as excinfo:
        build_chunker(
            STRATEGY_STRUCTURAL,
            ChunkPolicy(strategy=STRATEGY_STRUCTURAL, max_tokens=100, overlap_tokens=10),
            measurer=chars(),
        )
    message = str(excinfo.value)
    assert "不支持重叠" in message
    assert "标题边界" in message


def test_structural_never_merges_across_sections() -> None:
    """``min_tokens`` 只在**同一小节内**合并：跨小节合并会造出一条错的来源路径."""
    document = make_document(
        source="docs/two.md",
        media_type="text/markdown",
        blocks=[
            Block(kind=BLOCK_HEADING, text="A", level=1),
            Block(kind=BLOCK_PARAGRAPH, text="短。"),
            Block(kind=BLOCK_HEADING, text="B", level=1),
            Block(kind=BLOCK_PARAGRAPH, text="也短。"),
        ],
    )
    result = structural(
        ChunkPolicy(strategy=STRATEGY_STRUCTURAL, max_tokens=100, min_tokens=50)
    ).split(document)
    assert [chunk.heading_path for chunk in result.chunks] == [("A",), ("B",)]
    assert result.count == 2


def test_structural_header_line_is_part_of_its_own_chunk() -> None:
    """标题行属于它自己那一节（否则"标题文字"就没有任何块覆盖它）."""
    document = guide_document()
    result = structural(default_policy(STRATEGY_STRUCTURAL)).split(document)
    first = result.chunks[0]
    assert first.text.startswith("检索手册")
    assert first.heading_path == ("检索手册",)
    assert result.coverage > 0.95


def test_structural_describe_states_atomic_kinds_and_no_overlap() -> None:
    """自述表要写明三件事：不重叠、保护哪些块、以及它的软肋在哪里."""
    described = structural(default_policy(STRATEGY_STRUCTURAL)).describe()
    assert described["supports_overlap"] is False
    assert described["atomic_kinds"] == list(ATOMIC_KINDS)
    assert any("heading_path" in item for item in described["strengths"])
    assert any("blocks" in item for item in described["weaknesses"])
    assert "0" in described["cost"]


def test_structural_handles_other_block_kinds_without_special_treatment() -> None:
    """引用块既不是标题也不是原子块：它与相邻段落一起按预算打包."""
    document = make_document(
        source="docs/quote.md",
        media_type="text/markdown",
        blocks=[
            Block(kind=BLOCK_QUOTE, text="> 引用一句"),
            Block(kind=BLOCK_PARAGRAPH, text="正文一段。"),
        ],
    )
    result = structural(ChunkPolicy(strategy=STRATEGY_STRUCTURAL, max_tokens=100)).split(
        document
    )
    assert result.count == 1
    assert "引用一句" in result.chunks[0].text
    assert result.chunks[0].reason == "section:前言"
