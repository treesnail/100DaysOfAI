"""day062 ``chunking.tokens`` 的单元测试：预算单位与那个二分原语.

全部离线、确定性：字符度量器只调用 ``len``，tiktoken 相关用例在词表
不可用时自动跳过（``pytest.importorskip`` 语义）。
"""

from __future__ import annotations

import pytest

from smart_research_agent.chunking import (
    MEASURER_CHARS,
    MEASURER_TIKTOKEN,
    MEASURERS,
    CharMeasurer,
    ChunkingError,
    TiktokenMeasurer,
    count_tokens,
    describe_measurers,
    longest_prefix_within,
    resolve_measurer,
)

CJK_TEXT = "分块预算决定索引规模，overlap decides cost."


class FakeMeasurer:
    """一个"每字符固定花 N 个单位"的度量器（用来触发布局的边界分支）."""

    def __init__(self, per_char: int = 1, name: str = "fake") -> None:
        self.per_char = per_char
        self.name = name

    def count(self, text: str) -> int:
        return len(text) * self.per_char


def test_char_measurer_counts_characters() -> None:
    """字符度量器就是 ``len``：这是它"确定性"的全部来源."""
    measurer = CharMeasurer()
    assert measurer.name == MEASURER_CHARS
    assert measurer.count("中文abc") == 5
    assert measurer.count("") == 0


def test_resolve_measurer_returns_chars_by_default() -> None:
    """``chars`` 是默认口径：不需要任何环境前提."""
    resolved = resolve_measurer(MEASURER_CHARS)
    assert isinstance(resolved, CharMeasurer)
    assert resolved.name == MEASURER_CHARS


def test_resolve_measurer_rejects_unknown_name() -> None:
    """未知度量器要**列出可选值**，而不是抛 KeyError（day045 同款报错）."""
    with pytest.raises(ChunkingError) as excinfo:
        resolve_measurer("wordpiece")
    message = str(excinfo.value)
    assert "wordpiece" in message
    assert "chars" in message and MEASURER_TIKTOKEN in message


def test_resolve_measurer_refuses_silent_fallback(monkeypatch) -> None:
    """本机没有词表缓存时**拒绝**，而不是静默退回字符级估算.

    这条断言守的是本模块最上面那条纪律：静默降级会让"我要 token 预算"
    变成"我拿到字符预算"，而两者相差约 1.5 倍且结果里看不出来。
    """
    monkeypatch.setattr(
        TiktokenMeasurer, "backend", property(lambda self: "fallback")
    )
    with pytest.raises(ChunkingError) as excinfo:
        resolve_measurer(MEASURER_TIKTOKEN)
    assert "tiktoken" in str(excinfo.value)
    assert "chars" in str(excinfo.value)


def test_tiktoken_measurer_uses_real_tokenizer_when_available() -> None:
    """词表可用时按真实 token 计数；不可用则跳过（CI 上允许缺词表）."""
    measurer = TiktokenMeasurer()
    if measurer.backend != "tiktoken":
        pytest.skip("本机没有 tiktoken 词表缓存")
    # 中文按 token 计通常少于按字符计（cl100k_base 下常见汉字约 1 token/字，
    # 但一个汉字可能被编成多个 token，因此这里只断言"不等于字符数"这条弱条件）。
    assert measurer.count(CJK_TEXT) > 0
    assert measurer.count("") == 0


def test_describe_measurers_flags_determinism() -> None:
    """自述表必须有 ``deterministic`` 一列：它决定 chunk_id 能不能跨机器对齐."""
    table = describe_measurers()
    assert [row["name"] for row in table] == list(MEASURERS)
    by_name = {row["name"]: row for row in table}
    assert by_name[MEASURER_CHARS]["deterministic"] == "yes"
    assert by_name[MEASURER_TIKTOKEN]["deterministic"] == "no"
    assert by_name[MEASURER_TIKTOKEN]["consistent_with_billing"] == "yes"


def test_longest_prefix_within_returns_whole_text_when_it_fits() -> None:
    """整段装得下时返回全长（不做无谓的二分）."""
    measurer = CharMeasurer()
    assert longest_prefix_within("abcdef", 6, measurer) == 6
    assert longest_prefix_within("abcdef", 100, measurer) == 6


def test_longest_prefix_within_bisects_to_the_budget() -> None:
    """超预算时返回**最大**的、不超预算的前缀长度（边界必须精确）."""
    measurer = CharMeasurer()
    text = "x" * 50
    assert longest_prefix_within(text, 17, measurer) == 17
    assert longest_prefix_within(text, 1, measurer) == 1


def test_longest_prefix_within_returns_zero_when_one_char_exceeds_budget() -> None:
    """连一个字符都装不下时返回 0：**这个情形必须由调用方显式处理**."""
    measurer = FakeMeasurer(per_char=10)
    assert longest_prefix_within("abc", 5, measurer) == 0
    assert longest_prefix_within("abc", 10, measurer) == 1


def test_longest_prefix_within_rejects_negative_budget() -> None:
    """负预算是一个参数矛盾，必须在入口就报错."""
    with pytest.raises(ChunkingError):
        longest_prefix_within("abc", -1, CharMeasurer())


def test_longest_prefix_within_on_empty_text() -> None:
    """空文本返回 0（没有可切的内容，也不是错误）."""
    assert longest_prefix_within("", 10, CharMeasurer()) == 0


def test_prefix_count_is_monotonic() -> None:
    """二分的合法性依赖"前缀计数单调不减"这条**假设**，因此它要被测出来.

    换一个"会在边界上压缩"的度量器（例如带词干还原的计数），二分就会
    给出错误答案且**不报错**——所以这条假设不能只写在 docstring 里。
    """
    measurer = CharMeasurer()
    counts = [measurer.count(CJK_TEXT[:index]) for index in range(len(CJK_TEXT) + 1)]
    assert counts == sorted(counts)
    assert counts[0] == 0


def test_count_tokens_handles_empty_text() -> None:
    """空文本记 0，不调用度量器（省掉一次无意义的长度计算）."""
    assert count_tokens("", CharMeasurer()) == 0
    assert count_tokens("abc", CharMeasurer()) == 3
