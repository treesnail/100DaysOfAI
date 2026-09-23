"""day067 ``retrieval.lexical.tokenize`` 的单元测试：分词规则与它的代价.

这一层只有一条函数，但它是**整个关键词路的地基**：分词错了，BM25 的分数、
词表、``matched_terms`` 的对账全部跟着错，而且不会报任何错
（只会表现为"关键词路召回得不太对"）。因此这里钉死三件事：

```text
1. 规则本身      拉丁/数字/下划线原样；汉字段 1 字保留单字、>=2 字出"单字 + 相邻 2-gram"
2. 确定性        同一个输入在任何机器上给出同一个列表，逐位可比
3. 边界与代价   标点为空、空白串为空、非字符串当场报错；
                 2-gram 会把"语义"与"义语"当两个词元（这条**真实**的代价也要被钉住）
```

期望值来自一份**独立实现**的切分逻辑（测试文件里自己写的正则，
不引用 ``lexical.TOKEN_PATTERN``）——两边算出的列表必须逐位相同。
否则用例只是在复述实现，而实现改错时用例会跟着一起错。

全部离线、确定性、零网络。
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from smart_research_agent.retrieval.errors import QueryError
from smart_research_agent.retrieval.lexical import tokenize
from tests.hybrid_samples import RECORD_TEXTS, query_vectors

#: 测试侧独立的切段正则（**刻意不引用** ``lexical.TOKEN_PATTERN``）.
LATIN_RUN = re.compile(r"[a-z0-9_]+")
CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


def latin_runs(text: str) -> list[str]:
    """文本里的拉丁/数字段（小写后），按出现顺序——独立实现的期望值来源之一."""
    return LATIN_RUN.findall(text.lower())


def cjk_runs(text: str) -> list[str]:
    """文本里的汉字段，按出现顺序——独立实现的期望值来源之一."""
    return CJK_RUN.findall(text)


def expected_tokens(text: str) -> list[str]:
    """按**规则原文**重写一遍切分（独立于实现），用来交叉核对 ``tokenize``.

    它按"先把拉丁段与汉字段挑出来、再按出现顺序铺开"的方式重算一遍：
    与实现里的 ``findall`` 顺序语义一致，但代码路径完全不同（这里是两趟扫描，
    实现里是一趟）。两边一致才说明"顺序"这件事没有被实现悄悄改动。
    """
    spans: list[tuple[int, str, bool]] = []
    for match in LATIN_RUN.finditer(text.lower()):
        spans.append((match.start(), match.group(), False))
    for match in CJK_RUN.finditer(text):
        spans.append((match.start(), match.group(), True))
    spans.sort(key=lambda item: item[0])
    tokens: list[str] = []
    for _position, chunk, is_cjk in spans:
        if not is_cjk or len(chunk) == 1:
            tokens.append(chunk)
            continue
        tokens.extend(chunk)
        tokens.extend(chunk[index : index + 2] for index in range(len(chunk) - 1))
    return tokens


# --------------------------------------------------------------------------- #
# 规则：一张写死的对照表（每条都是"规则原文 → 期望输出"）
# --------------------------------------------------------------------------- #

#: ``(输入, 期望词元)``。覆盖六类写法：纯拉丁、拉丁+数字、下划线、标点、
#: 中文单字/两字/多字、中英混排。
TOKENIZE_CASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("", ()),
    ("   ", ()),
    ("\n\t", ()),
    ("hello", ("hello",)),
    ("Hello", ("hello",)),
    ("HELLO world", ("hello", "world")),
    ("err", ("err",)),
    ("2043", ("2043",)),
    ("ERR-2043", ("err", "2043")),
    ("v1.2", ("v1", "2")),
    ("retry_budget", ("retry_budget",)),
    ("a_b_c", ("a_b_c",)),
    ("__init__", ("__init__",)),
    ("token 320", ("token", "320")),
    ("12 万", ("12", "万")),
    ("语", ("语",)),
    ("缓存", ("缓", "存", "缓存")),
    ("语义缓存", ("语", "义", "缓", "存", "语义", "义缓", "缓存")),
    ("语义缓存与检索", (
        "语", "义", "缓", "存", "与", "检", "索",
        "语义", "义缓", "缓存", "存与", "与检", "检索",
    )),
    ("缓存，检索", ("缓", "存", "缓存", "检", "索", "检索")),
    ("缓存、检索", ("缓", "存", "缓存", "检", "索", "检索")),
    ("缓存 检索", ("缓", "存", "缓存", "检", "索", "检索")),
    ("缓存\n检索", ("缓", "存", "缓存", "检", "索", "检索")),
    ("缓存。", ("缓", "存", "缓存")),
    ("，。！？", ()),
    ("；：（）【】", ()),
    ("ABC语义", ("abc", "语", "义", "语义")),
    ("语义ABC", ("语", "义", "语义", "abc")),
    ("语义 v1 缓存", ("语", "义", "语义", "v1", "缓", "存", "缓存")),
    # 空白是**分隔符**：'语义' 与 '缓存' 是两个汉字段，各自出单字与自己的 2-gram
    # （因此这里**没有**跨空白的 '义缓'——跨词边界的 2-gram 只在同一个汉字段内生成）。
    ("  语义  缓存  ", ("语", "义", "语义", "缓", "存", "缓存")),
    ("ERR-2043 表示向量维度不一致", (
        "err", "2043",
        "表", "示", "向", "量", "维", "度", "不", "一", "致",
        "表示", "示向", "向量", "量维", "维度", "度不", "不一", "一致",
    )),
    ("retry_budget 缺省 3 次", ("retry_budget", "缺", "省", "缺省", "3", "次")),
    ("编号、函数名、报错码", (
        "编", "号", "编号", "函", "数", "名", "函数", "数名",
        "报", "错", "码", "报错", "错码",
    )),
)


class TestTokenizeRules:
    """每一类写法一条用例（期望值见 ``TOKENIZE_CASES`` 的注释）. """

    @pytest.mark.parametrize(("text", "expected"), TOKENIZE_CASES)
    def test_matches_the_documented_rule(self, text: str, expected: tuple[str, ...]) -> None:
        assert tokenize(text) == list(expected)

    @pytest.mark.parametrize(("text", "expected"), TOKENIZE_CASES)
    def test_independent_reimplementation_agrees(
        self, text: str, expected: tuple[str, ...]
    ) -> None:
        """实现与测试侧独立重写的那份切分必须逐位相同（顺序也算）. """
        assert tokenize(text) == expected_tokens(text) == list(expected)


class TestTokenizeCJKExample:
    """手册里那个被反复引用的例子：``"语义缓存" → 语/义/缓/存/语义/义缓/缓存``."""

    def test_singles_come_before_bigrams(self) -> None:
        """单字在前、2-gram 在后——**顺序本身也是确定性的一部分**. """
        tokens = tokenize("语义缓存")

        assert tokens[:4] == ["语", "义", "缓", "存"]
        assert tokens[4:] == ["语义", "义缓", "缓存"]

    def test_bigram_count_is_length_minus_one(self) -> None:
        """n 字的汉字段恰好给出 n-1 个相邻 2-gram（滑动窗口的定义）. """
        tokens = tokenize("语义缓存与检索")
        bigrams = [token for token in tokens if len(token) == 2]

        assert bigrams == ["语义", "义缓", "缓存", "存与", "与检", "检索"]
        assert len(bigrams) == len("语义缓存与检索") - 1

    def test_single_character_run_is_kept_as_is(self) -> None:
        """长度 1 的汉字段保留单字（单字查询是真实存在的："熵"、"税"）. """
        assert tokenize("熵") == ["熵"]
        assert tokenize("税与熵") == ["税", "与", "熵", "税与", "与熵"]


class TestTokenizeProperties:
    """跨全部样本的**结构性质**：这些性质比逐条对照更能抓实现漂移. """

    @pytest.mark.parametrize("record_id", sorted(RECORD_TEXTS))
    def test_every_cjk_character_becomes_a_token(self, record_id: str) -> None:
        """每个汉字都必须出现在单字词元里（否则跨词边界的召回会丢）. """
        text = RECORD_TEXTS[record_id]
        tokens = set(tokenize(text))
        for chunk in cjk_runs(text):
            for character in chunk:
                assert character in tokens

    @pytest.mark.parametrize("record_id", sorted(RECORD_TEXTS))
    def test_each_cjk_run_yields_its_adjacent_bigrams(self, record_id: str) -> None:
        """每个汉字段的相邻 2-gram 一个都不能少（精确率靠它）. """
        text = RECORD_TEXTS[record_id]
        tokens = tokenize(text)
        for chunk in cjk_runs(text):
            expected = [chunk[index : index + 2] for index in range(len(chunk) - 1)]
            for bigram in expected:
                assert bigram in tokens

    @pytest.mark.parametrize("record_id", sorted(RECORD_TEXTS))
    def test_latin_runs_survive_verbatim_and_lowercased(self, record_id: str) -> None:
        """拉丁/数字段原样保留（下划线也在内），大小写先折叠. """
        text = RECORD_TEXTS[record_id]
        tokens = tokenize(text)
        for run in latin_runs(text):
            assert run in tokens

    @pytest.mark.parametrize("record_id", sorted(RECORD_TEXTS))
    def test_no_token_contains_whitespace(self, record_id: str) -> None:
        """词元里不允许出现空白（否则"一个词元"这件事就不成立了）. """
        for token in tokenize(RECORD_TEXTS[record_id]):
            assert token and not any(character.isspace() for character in token)

    @pytest.mark.parametrize("record_id", sorted(RECORD_TEXTS))
    def test_tokens_are_already_lowercased(self, record_id: str) -> None:
        """词元自身的 ``lower()`` 必须等于自身（幂等性：再切一次不会变）. """
        for token in tokenize(RECORD_TEXTS[record_id]):
            assert token == token.lower()

    @pytest.mark.parametrize("record_id", sorted(RECORD_TEXTS))
    def test_tokenize_is_deterministic_and_pure(self, record_id: str) -> None:
        """两次调用逐位相同，且输入字符串没有被改动（纯函数）. """
        text = RECORD_TEXTS[record_id]
        before = str(text)
        first = tokenize(text)
        second = tokenize(text)

        assert first == second
        assert text == before

    @pytest.mark.parametrize("query", sorted(query_vectors()))
    def test_query_texts_follow_the_same_rule(self, query: str) -> None:
        """查询侧与文档侧用的是同一个函数（两边的分词口径必须相同）. """
        assert tokenize(query) == expected_tokens(query)


class TestTokenizeSurfacingCosts:
    """分词法的**已知代价**：这些不是 bug，但必须被钉住（免得被当成"没实现"）. """

    def test_two_character_reversal_is_a_different_token(self) -> None:
        """``"语义"`` 与 ``"义语"`` 是两个词元：2-gram 的区分度正来自这里.

        代价是词表变大（一条 20 字的句子会产生 19 个 2-gram），
        收益是"缓存"与"存缓"不会被当成同一个词。
        """
        tokens = tokenize("语义义语")

        assert "语义" in tokens
        assert "义语" in tokens
        assert tokens.count("语义") == 1

    def test_no_word_segmentation_happens(self) -> None:
        """不做词形还原、不识别词边界：``"缓存"`` 在 ``"缓存的"`` 里仍切成两个单字加一个 2-gram. """
        assert tokenize("缓存的") == ["缓", "存", "的", "缓存", "存的"]

    def test_cross_boundary_bigram_is_generated(self) -> None:
        """跨词边界的 2-gram（``"存检"``）也会被生成——这是 2-gram 法提高召回的一面. """
        assert "存检" in tokenize("缓存检索")

    def test_repeated_word_keeps_duplicates(self) -> None:
        """重复词元**保留重复**：词频是 BM25 的两个输入之一，必须能被读出来. """
        tokens = tokenize("缓存缓存")

        assert tokens.count("缓存") == 2
        assert tokens.count("缓") == 2


class TestTokenizeErrors:
    """非字符串输入当场报 ``QueryError``（不留到打分那一步再炸）. """

    @pytest.mark.parametrize(
        "value", [None, 123, 1.5, True, ["缓存"], {"text": "缓存"}, b"cache"]
    )
    def test_non_string_is_rejected(self, value: Any) -> None:
        with pytest.raises(QueryError) as excinfo:
            tokenize(value)

        message = str(excinfo.value)
        assert "tokenize 需要字符串" in message
        assert "LexicalDocument" in message

    def test_error_message_names_the_type(self) -> None:
        with pytest.raises(QueryError) as excinfo:
            tokenize(2043)

        assert "int" in str(excinfo.value)
