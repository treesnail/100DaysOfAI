"""day066 ``retrieval.context`` 的单元测试：预算、截断、丢尾、引用编号（M6-D5）.

这一层要回答的是"提示词里到底放了什么、放了几个字、为什么只放了这些"，
因此断言按 **四条留了数字的规则** 分组（见 ``context.pack_context`` 的固定五步）：

```text
单条截断    超 per_hit_chars 的块被截断并记 truncated_hits（它仍然在包里）
整包丢尾    整包超预算 → 从**最后一条**开始整条丢，dropped_hits 里能看到是谁
至少保一条  预算紧到只放得下一条时也返回它（截到 max_chars），而不是返回空
预算下限    预算连 min_hit_chars 都放不下 → ContextError（配置错误不许伪装成空上下文）
```

另外两组盯住"编号"与"形状"：``Citation.marker`` 必须与 ``text`` 里的 ``[n]``
一一对应（**被丢掉的尾部不占编号**），而 ``PackedContext.__post_init__`` 会把
这份一致性在构造期再校验一遍——那些校验要有用例，否则它们只是注释里的承诺。

预算单位默认是字符数（``len``）；注入 ``measurer`` 之后 ``max_chars`` /
``per_hit_chars`` / ``min_hit_chars`` **三者同单位**，因此 token 口径的用例
全部显式传 ``min_hit_chars=1``：把 token 预算配上字符下限，会得到一个
"永远为真"的比较，那样"同单位"这条纪律就白写了。

两处**边界行为**（不是期望行为）也有用例，并在本轮报告里写明：
预算小到连 ``[1]`` 都放不下时标记本身会被切掉；以及 ``measure(标记) > limit``
的极端口径下"至少保一条"会退化成空文本。断言写的是**当前实际行为**，
理由与 ``test_retrieval_retriever`` 里那条"端到端不可达的 diversity 分支"相同：
把边界写下来，下一个人就不必再猜。

全部离线、确定性：命中是手工构造的 ``RetrievalHit``，期望值全部能手算。
"""

from __future__ import annotations

from typing import Any

import pytest

from smart_research_agent.llm.mock import estimate_tokens
from smart_research_agent.retrieval.context import (
    BLOCK_SEPARATOR,
    TRUNCATION_MARKER,
    Citation,
    PackedContext,
    pack_context,
)
from smart_research_agent.retrieval.errors import ContextError
from smart_research_agent.retrieval.types import DEFAULT_MIN_HIT_CHARS, RetrievalHit

#: 样本来源（所有手工命中的默认 ``metadata["source"]``）。
SOURCE = "docs/a.md"

#: 第一条命中的块头（``[1] docs/a.md``，13 个字符）——期望值就是拿它手算的。
HEADER_ONE = f"[1] {SOURCE}"

#: 块头 + 换行 + 10 字正文 = 24 字符：整包丢尾那一组用例的分母。
FLAT_BLOCK_CHARS = len(HEADER_ONE) + 1 + 10


def make_hit(
    record_id: str,
    text: str = "",
    *,
    score: float = 1.0,
    rank: int = 0,
    metadata: dict[str, Any] | None = None,
) -> RetrievalHit:
    """构造一条命中（默认只带 ``source``；``metadata=None`` 时如此）."""
    return RetrievalHit(
        record_id=record_id,
        score=score,
        rank=rank,
        text=text,
        metadata={"source": SOURCE} if metadata is None else dict(metadata),
    )


def flat_hits(count: int = 4, *, text: str = "x" * 10) -> list[RetrievalHit]:
    """``count`` 条**块长完全相同**的命中（块长 = ``FLAT_BLOCK_CHARS``）.

    id 取 ``h-1`` … ``h-9``（一位数），这样每条块头都是 13 个字符——
    整包丢尾的期望值因此可以写成"两块 50 字、三块 76 字"这样的数。
    """
    return [
        make_hit(f"h-{index}", text, score=1.0 - index * 0.01, rank=index - 1)
        for index in range(1, count + 1)
    ]


def make_citation(
    marker: int,
    record_id: str = "h-1",
    *,
    source: str = SOURCE,
    heading: str = "",
    score: float = 1.0,
) -> Citation:
    """手工构造一条引用（只给一致性校验用）."""
    return Citation(
        marker=marker,
        record_id=record_id,
        source=source,
        heading_path=heading,
        score=score,
    )


# --------------------------------------------------------------------------- #
# 第一条规则：先截断单条
# --------------------------------------------------------------------------- #


class TestSingleHitTruncation:
    """单条超 ``per_hit_chars`` → 截断 + 补标记 + ``truncated_hits`` 记一笔."""

    def test_truncates_an_over_long_hit_and_counts_it(self) -> None:
        """``per_hit_chars=60`` 时那条 114 字的块被截断，计数为 1."""
        result = pack_context(
            [make_hit("h-1", "甲" * 100)], max_chars=500, per_hit_chars=60
        )

        assert result.truncated_hits == 1, "被单条上限截断的条数必须被记下来"
        assert result.count == 1, "被截断的块**仍然在包里**（只有超预算才整条丢）"
        assert result.dropped_hits == (), "截断不是丢弃：dropped_hits 应为空"

    def test_truncated_text_ends_with_the_marker(self) -> None:
        """块尾必须补截断标记：模型要能看出"这段是半截的"."""
        result = pack_context(
            [make_hit("h-1", "甲" * 100)], max_chars=500, per_hit_chars=60
        )

        assert result.text.endswith(TRUNCATION_MARKER), (
            "截断标记是「这段是半截的」在提示词里唯一的证据"
        )

    def test_truncated_block_is_exactly_the_per_hit_cap(self) -> None:
        """恰好切到 60 字：标记占 1 字，因此正文留 59 字."""
        block = f"{HEADER_ONE}\n{'甲' * 100}"
        result = pack_context([make_hit("h-1", "甲" * 100)], max_chars=500, per_hit_chars=60)

        assert result.text == block[:59] + TRUNCATION_MARKER
        assert len(result.text) == 60 == result.char_count

    def test_short_hits_are_not_truncated(self) -> None:
        """块长小于上限时一个字都不动（``truncated_hits`` 保持 0）."""
        result = pack_context(
            [make_hit("h-1", "短正文")], max_chars=500, per_hit_chars=200
        )

        assert result.truncated_hits == 0
        assert result.text == f"{HEADER_ONE}\n短正文"

    def test_cap_applies_per_hit_not_to_the_whole_pack(self) -> None:
        """上限是**每条**的：三条 24 字的块拼成 76 字仍然一条都不截."""
        hits = flat_hits(3)
        result = pack_context(hits, max_chars=500, per_hit_chars=30)

        assert result.count == 3
        assert result.truncated_hits == 0, "per_hit_chars 是单条上限，不是整包上限"
        assert result.char_count == 76, "两块之间 2 个分隔符：24×3 + 2×2 = 76"

    def test_per_hit_cap_of_zero_is_rejected(self) -> None:
        """上限为 0 意味着没有任何片段能进上下文（配置错误当场报）."""
        with pytest.raises(ContextError) as excinfo:
            pack_context([make_hit("h-1", "正文")], per_hit_chars=0)

        assert "per_hit_chars=0" in str(excinfo.value)

    def test_per_hit_cap_must_be_an_int(self) -> None:
        """上限必须是整数（``True`` 也拦掉：它会让"预算"变成一个布尔）."""
        with pytest.raises(ContextError) as excinfo:
            pack_context([make_hit("h-1", "正文")], per_hit_chars=True)

        assert "per_hit_chars 必须是整数" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 第二条规则：整包从尾部丢
# --------------------------------------------------------------------------- #


class TestPackTailDrop:
    """整包超预算 → 从最后一条开始整条丢（丢的是尾部，不是中间或头部）."""

    def test_drops_from_the_tail_and_keeps_the_head(self) -> None:
        """预算 60 只放得下两块（50 字），因此后两条被整条丢掉."""
        hits = flat_hits(4)
        result = pack_context(hits, max_chars=60, per_hit_chars=800)

        assert result.count == 2, "50 ≤ 60 < 76：留下的正好是前两条"
        assert result.char_count == 50
        assert [hit.record_id for hit in result.dropped_hits] == ["h-3", "h-4"]

    def test_used_plus_dropped_equals_input_count(self) -> None:
        """进包 + 丢弃 == 入参条数：这一层不许**悄悄**少掉一条."""
        hits = flat_hits(4)
        result = pack_context(hits, max_chars=60, per_hit_chars=800)

        assert len(result.used_hits) + len(result.dropped_hits) == len(hits) == 4

    def test_dropped_hits_are_exactly_the_tail_slice(self) -> None:
        """丢掉的必须**逐条等于**入参的尾部切片（不是"随便丢两条"）."""
        hits = flat_hits(5)
        result = pack_context(hits, max_chars=60, per_hit_chars=800)

        kept = result.count
        assert result.used_hits == tuple(hits[:kept])
        assert result.dropped_hits == tuple(hits[kept:])

    def test_dropped_hits_keep_their_ranking_order(self) -> None:
        """被丢的那些保持名次顺序（报告要能一眼看出"最靠后的那几条"）."""
        hits = flat_hits(4)
        result = pack_context(hits, max_chars=60, per_hit_chars=800)

        scores = [hit.score for hit in result.dropped_hits]
        assert scores == sorted(scores, reverse=True), "入参已是名次顺序，丢尾不乱序"

    def test_no_truncation_is_counted_when_whole_hits_are_dropped(self) -> None:
        """整条丢与"截断"是两件事：``truncated_hits`` 保持 0."""
        result = pack_context(flat_hits(4), max_chars=60, per_hit_chars=800)

        assert result.truncated_hits == 0
        assert len(result.dropped_hits) == 2

    def test_truncated_hits_counts_only_the_hits_that_stayed(self) -> None:
        """单条上限切过三条、但后两条被整条丢掉 → 计数只算留下的那一条.

        ``truncated_hits`` 的定义是"**真的进了上下文的**那些里被单条上限截断的条数"
        （见 ``PackedContext`` 的字段说明）：合成一个数字会让"该调哪个参数"失去判据。
        """
        hits = flat_hits(3, text="甲" * 300)
        result = pack_context(hits, max_chars=60, per_hit_chars=60)

        assert result.count == 1
        assert result.dropped_hits == tuple(hits[1:])
        assert result.truncated_hits == 1, "只算留下的那一条（三条都被切过）"

    def test_a_bigger_budget_keeps_more_hits(self) -> None:
        """同一个库、同一批命中：预算变大 → 丢得更少（单调性）."""
        hits = flat_hits(4)

        tight = pack_context(hits, max_chars=60, per_hit_chars=800)
        loose = pack_context(hits, max_chars=200, per_hit_chars=800)

        assert tight.count == 2
        assert loose.count == 4
        assert loose.dropped_hits == ()


# --------------------------------------------------------------------------- #
# 第三条规则：至少保住一条
# --------------------------------------------------------------------------- #


class TestKeepAtLeastOne:
    """"预算紧"不等于"没有上下文"——这一层最重要的护栏."""

    def test_a_single_over_budget_hit_is_still_returned(self) -> None:
        """唯一的块有 314 字、预算只有 60：仍然返回它（而不是返回空上下文）."""
        result = pack_context(
            [make_hit("h-1", "乙" * 300)], max_chars=60, per_hit_chars=800
        )

        assert result.count == 1, "返回空上下文会让 RAG 链路走进「检索为空」的分支"
        assert result.char_count == 60 == result.max_chars
        assert result.dropped_hits == ()

    def test_the_kept_hit_is_cut_to_the_budget(self) -> None:
        """保下来的那一条被截到刚好占满预算（证据是 ``char_count == max_chars``）."""
        result = pack_context(
            [make_hit("h-1", "乙" * 300)], max_chars=60, per_hit_chars=800
        )

        assert result.fill_ratio == 1.0
        assert result.char_count == result.max_chars
        assert result.text.endswith(TRUNCATION_MARKER)

    def test_this_truncation_is_not_counted_as_truncated_hit(self) -> None:
        """"至少保一条"的这次截断**不计**进 ``truncated_hits``（见 docstring）."""
        result = pack_context(
            [make_hit("h-1", "乙" * 300)], max_chars=60, per_hit_chars=800
        )

        assert result.truncated_hits == 0, "它的证据是 char_count == max_chars，不是计数"

    def test_two_hits_shrink_to_one_when_the_budget_is_tight(self) -> None:
        """两条都超预算 → 先丢尾到一条，再把仅剩的一条截到预算."""
        hits = flat_hits(2, text="丙" * 300)
        result = pack_context(hits, max_chars=60, per_hit_chars=800)

        assert result.count == 1
        assert result.dropped_hits == (hits[1],)
        assert result.char_count == 60

    def test_budget_equal_to_the_floor_is_allowed(self) -> None:
        """预算 == ``min_hit_chars`` 是合法的（只有**小于**才拒绝）."""
        result = pack_context(
            [make_hit("h-1", "丁" * 200)], max_chars=DEFAULT_MIN_HIT_CHARS,
            min_hit_chars=DEFAULT_MIN_HIT_CHARS, per_hit_chars=800,
        )

        assert result.char_count == DEFAULT_MIN_HIT_CHARS == 32
        assert result.count == 1

    def test_a_budget_below_the_header_can_cut_the_marker(self) -> None:
        """边界：预算小到连块头都放不下时，``[1]`` 会被切掉（当前行为）.

        默认下限是 32 字（``DEFAULT_MIN_HIT_CHARS``），因此这条路只能靠
        **显式**把 ``min_hit_chars`` 调到 1~3 才走得到；本轮报告里记了这一点。
        """
        result = pack_context(
            [make_hit("h-1", "x" * 50)], max_chars=3, min_hit_chars=1, per_hit_chars=1000
        )

        assert result.count == 1
        assert result.text == "[1…"
        assert "[1]" not in result.text


# --------------------------------------------------------------------------- #
# 第四条规则：预算放不下一条 → ContextError
# --------------------------------------------------------------------------- #


class TestBudgetFloor:
    """预算小于 ``min_hit_chars`` → 报错，而不是静默返回空上下文."""

    def test_budget_below_floor_raises(self) -> None:
        """消息里必须同时有"需要多少"与"给了多少"."""
        with pytest.raises(ContextError) as excinfo:
            pack_context([make_hit("h-1", "正文")], max_chars=10, min_hit_chars=32)

        message = str(excinfo.value)
        assert "至少需要 32 字" in message, "报错要说清需要多少"
        assert "只给了 10 字" in message, "报错要说清给了多少"

    def test_error_message_offers_a_way_out(self) -> None:
        """四族错误都必须是"现象 + 出路"两句（见 ``errors`` 的模板）."""
        with pytest.raises(ContextError) as excinfo:
            pack_context([make_hit("h-1", "正文")], max_chars=10, min_hit_chars=32)

        assert "settings.retrieval_max_context_chars" in str(excinfo.value)

    def test_budget_of_zero_is_rejected(self) -> None:
        """0 与"不限"是两件事：要"不限"请给一个足够大的正数."""
        with pytest.raises(ContextError) as excinfo:
            pack_context([make_hit("h-1", "正文")], max_chars=0)

        assert "max_chars=0" in str(excinfo.value)

    def test_budget_must_be_an_int(self) -> None:
        """预算必须是整数（字符串形式的数字也不接受）."""
        with pytest.raises(ContextError) as excinfo:
            pack_context([make_hit("h-1", "正文")], max_chars="800")  # type: ignore[arg-type]

        assert "max_chars 必须是整数" in str(excinfo.value)

    def test_config_is_checked_even_without_hits(self) -> None:
        """**空命中不跳过配置校验**：配错的预算会在下一次有命中时以同样方式错下去."""
        with pytest.raises(ContextError):
            pack_context([], max_chars=0)

    def test_min_hit_chars_must_be_positive(self) -> None:
        """下限为 0 会让"预算够不够"这个判断失去尺度."""
        with pytest.raises(ContextError) as excinfo:
            pack_context([make_hit("h-1", "正文")], min_hit_chars=0)

        assert "min_hit_chars 必须是 >= 1 的整数" in str(excinfo.value)

    def test_min_hit_chars_must_be_an_int(self) -> None:
        """``True`` 是 bool 而不是整数（同一个坑在三个参数上都要堵住）."""
        with pytest.raises(ContextError):
            pack_context([make_hit("h-1", "正文")], min_hit_chars=True)

    def test_the_default_floor_is_the_documented_constant(self) -> None:
        """默认下限是常量 ``DEFAULT_MIN_HIT_CHARS``（31 字就该被拒）."""
        assert DEFAULT_MIN_HIT_CHARS == 32

        with pytest.raises(ContextError):
            pack_context([make_hit("h-1", "正文")], max_chars=DEFAULT_MIN_HIT_CHARS - 1)


# --------------------------------------------------------------------------- #
# 预算单位：默认 chars，可注入 measurer
# --------------------------------------------------------------------------- #


class TestMeasurerInjection:
    """``measurer`` 注入之后预算与计数**全部换成那个口径**（三者也必须同单位）."""

    def test_injected_measurer_decides_the_counted_number(self) -> None:
        """token 口径下 ``char_count`` 是 token 数（27），不是字符数（110）."""
        hits = flat_hits(2, text="甲" * 40)
        result = pack_context(
            hits, max_chars=30, per_hit_chars=1000, min_hit_chars=1, measurer=estimate_tokens
        )

        assert result.count == 2
        assert len(result.text) == 110, "两块 54 字的块 + 2 个分隔符"
        assert result.char_count == estimate_tokens(result.text) == 27
        assert result.char_count != len(result.text), "口径换了，数字就不该等于 len()"

    def test_injected_measurer_changes_which_hits_survive(self) -> None:
        """同一条预算数字、两种口径 → 不同的结果（这就是"必须同单位"的理由）."""
        hits = flat_hits(2, text="甲" * 40)

        tokens = pack_context(
            hits, max_chars=20, per_hit_chars=1000, min_hit_chars=1, measurer=estimate_tokens
        )
        chars = pack_context(hits, max_chars=120, per_hit_chars=1000, min_hit_chars=1)

        assert tokens.count == 1 and tokens.char_count == 13
        assert chars.count == 2 and chars.char_count == 110

    def test_injected_measurer_also_applies_to_the_per_hit_cap(self) -> None:
        """``per_hit_chars`` 也是那个口径：20 token 的块有 83 个字符（> 20）."""
        result = pack_context(
            [make_hit("h-1", "甲" * 400)], max_chars=500, per_hit_chars=20,
            min_hit_chars=1, measurer=estimate_tokens,
        )

        assert result.truncated_hits == 1
        assert estimate_tokens(result.text) <= 20
        assert len(result.text) == 83, "若按字符切，这里只会留 20 个字"

    def test_budget_is_recorded_in_the_measurer_unit(self) -> None:
        """``max_chars`` 原样记下那个口径的预算（报告里不该出现单位混用）."""
        result = pack_context(
            flat_hits(2, text="甲" * 40), max_chars=20, per_hit_chars=1000,
            min_hit_chars=1, measurer=estimate_tokens,
        )

        assert result.max_chars == 20

    def test_a_custom_measurer_is_used_verbatim(self) -> None:
        """注入的口径说了算：恒返回 1 的口径下，114 字的块"只占 1 个字"."""

        def one(_text: str) -> int:
            return 1

        result = pack_context(
            [make_hit("h-1", "x" * 100)], max_chars=5, per_hit_chars=1000,
            min_hit_chars=1, measurer=one,
        )

        assert result.char_count == 1
        assert len(result.text) == 114, "文本没有被切（1 ≤ 5），只是记的数字是 1"

    def test_a_measurer_that_cannot_fit_the_marker_returns_a_plain_prefix(self) -> None:
        """边界：``measure(标记) > limit`` 时退化成"能放下多少算多少"（可能是空串）.

        这条只有手工注入一个"单位比预算还大"的口径才走得到（模块 docstring
        第 3 条：``min_hit_chars`` 与预算也要同单位）；本轮报告里记了这一点。
        """

        def tenths(text: str) -> int:
            return len(text) * 10

        result = pack_context(
            [make_hit("h-1", "x" * 50)], max_chars=5, per_hit_chars=1000,
            min_hit_chars=1, measurer=tenths,
        )

        assert result.count == 1, "命中的账仍然记着它"
        assert result.text == "", "但这次连一个字都没能留下"
        assert result.char_count == 0

    def test_the_fallback_returns_a_plain_prefix_when_the_marker_does_not_fit(self) -> None:
        """同一条退化路径的另一半：文本自己还能放进一点时，返回**不带标记**的前缀.

        "标记放不下"与"文本放不下"是两件事：前者退化成后者（见 ``_truncate``），
        因此这里拿到的是一段半截得更彻底的文本——但预算这条硬约束没有破。
        """

        def marker_is_huge(text: str) -> int:
            return 100 if TRUNCATION_MARKER in text else len(text)

        result = pack_context(
            [make_hit("h-1", "x" * 50)], max_chars=5, per_hit_chars=1000,
            min_hit_chars=1, measurer=marker_is_huge,
        )

        assert result.text == "[1] d", "按测量口径能放下的 5 个字符"
        assert TRUNCATION_MARKER not in result.text
        assert result.char_count == 5 == result.max_chars

    def test_the_default_measurer_is_the_char_count(self) -> None:
        """默认口径就是 ``len()``：确定、离线、可复现（与 day062 的取向一致）."""
        result = pack_context(flat_hits(3))

        assert result.char_count == len(result.text)


# --------------------------------------------------------------------------- #
# 空命中
# --------------------------------------------------------------------------- #


class TestEmptyHits:
    """空 ``hits`` → 空上下文（不是异常）."""

    def test_empty_hits_give_an_empty_context(self) -> None:
        """没有命中可打包不是错误：``text=""``、``citations=()``."""
        result = pack_context([])

        assert result.text == ""
        assert result.citations == ()
        assert result.used_hits == () and result.dropped_hits == ()
        assert result.char_count == 0 and result.truncated_hits == 0

    def test_empty_context_is_empty_and_fills_nothing(self) -> None:
        """预算照常带在结果里（读的人知道"这次的预算是什么"）."""
        result = pack_context([])

        assert result.is_empty is True
        assert result.count == 0
        assert result.fill_ratio == 0.0
        assert result.max_chars == 2400, "默认预算来自 settings.retrieval_max_context_chars"

    def test_empty_hits_with_explicit_budgets_are_still_empty(self) -> None:
        """显式给出的预算与上限原样记下，不会因为没命中而被改写."""
        result = pack_context([], max_chars=1234, per_hit_chars=56)

        assert result.is_empty is True
        assert result.max_chars == 1234

    def test_every_marker_lookup_misses_on_an_empty_context(self) -> None:
        """空上下文里任何一个 id 都没有编号（``marker_for`` 返回 ``None``）."""
        assert pack_context([]).marker_for("h-1") is None


# --------------------------------------------------------------------------- #
# 引用编号
# --------------------------------------------------------------------------- #


class TestCitationMarkers:
    """编号是"在这一次提示词里的位置"：1 起连续，且与文本里的 ``[n]`` 一一对应."""

    def test_markers_are_one_based_and_contiguous(self) -> None:
        """三条命中 → ``(1, 2, 3)``（不是 0 起，也不跳号）."""
        result = pack_context(flat_hits(3))

        assert [citation.marker for citation in result.citations] == [1, 2, 3]
        assert result.text.startswith("[1] ")

    def test_each_marker_appears_exactly_once_in_the_text(self) -> None:
        """文本里每个 ``[n]`` 都只出现一次（出现两次会让溯源指向两处）."""
        result = pack_context(flat_hits(3))

        for marker in (1, 2, 3):
            assert result.text.count(f"[{marker}]") == 1

    def test_citations_follow_the_used_hits_order(self) -> None:
        """``citations`` 与 ``used_hits`` 逐条对齐（编号 i 对应第 i 条命中）."""
        result = pack_context(flat_hits(3))

        assert [citation.record_id for citation in result.citations] == [
            hit.record_id for hit in result.used_hits
        ]

    def test_citation_carries_the_source_and_the_score(self) -> None:
        """引用同时给"哪一份"与"多近"（score 保留 6 位）."""
        hits = [make_hit("h-1", "正文", score=0.8123456789)]
        result = pack_context(hits)

        assert result.citations[0].source == SOURCE
        assert result.citations[0].to_dict()["score"] == 0.812346

    def test_a_dropped_hit_has_no_marker_in_the_text(self) -> None:
        """被丢掉的尾部**不占编号**：提示词里不该出现一个指不到片段的 ``[3]``."""
        result = pack_context(flat_hits(4), max_chars=60, per_hit_chars=800)

        assert [citation.marker for citation in result.citations] == [1, 2]
        assert "[3]" not in result.text
        assert "[4]" not in result.text

    def test_markers_restart_from_one_and_end_at_the_count(self) -> None:
        """编号上界 == 进包条数（丢尾之后重编，绝不留下尾部的号）."""
        result = pack_context(flat_hits(5), max_chars=60, per_hit_chars=800)

        markers = [citation.marker for citation in result.citations]
        assert markers == list(range(1, result.count + 1))

    def test_blocks_are_joined_by_the_documented_separator(self) -> None:
        """块之间用空行相连：分隔符个数 == 条数 - 1（单换行会让块的起点看不清）."""
        result = pack_context(flat_hits(3))

        assert result.text.count(BLOCK_SEPARATOR) == result.count - 1 == 2
        assert len(result.text.split(BLOCK_SEPARATOR)) == 3


class TestMarkerFor:
    """``marker_for`` 是"从命中的 id 回到提示词里的位置"这一个方向的入口."""

    def test_marker_for_a_kept_hit(self) -> None:
        """进了上下文的命中返回它的编号."""
        result = pack_context(flat_hits(3))

        assert result.marker_for("h-1") == 1
        assert result.marker_for("h-3") == 3

    def test_marker_for_an_unknown_id_is_none(self) -> None:
        """库里没有这个 id 时返回 ``None``（"它不在里面"是一个正常答案）."""
        result = pack_context(flat_hits(3))

        assert result.marker_for("不存在的-id") is None

    def test_marker_for_a_dropped_hit_is_none(self) -> None:
        """被丢掉的尾部同样返回 ``None``（而不是 0 或抛异常）."""
        result = pack_context(flat_hits(4), max_chars=60, per_hit_chars=800)

        assert result.marker_for("h-3") is None
        assert result.marker_for("h-4") is None


# --------------------------------------------------------------------------- #
# 块渲染
# --------------------------------------------------------------------------- #


class TestRenderBlock:
    """``[n] source › heading_path`` + 换行 + 正文（回落规则与检索器同一份实现）."""

    def test_block_header_is_marker_plus_source(self) -> None:
        """没有 ``heading_path`` 时块头就只有"编号 + 来源"."""
        result = pack_context([make_hit("h-1", "正文")])

        assert result.text.splitlines()[0] == HEADER_ONE
        assert result.text.splitlines()[1] == "正文"

    def test_heading_path_is_appended_when_present(self) -> None:
        """day062 的 ``heading_path`` 终于在这里变成"这句话出自哪一节"."""
        hit = make_hit("h-1", "正文", metadata={"source": SOURCE, "heading_path": "手册 > 阈值"})
        result = pack_context([hit])

        assert result.text.splitlines()[0] == f"[1] {SOURCE} › 手册 > 阈值"
        assert result.citations[0].heading_path == "手册 > 阈值"

    def test_source_falls_back_to_doc_id(self) -> None:
        """``source`` 缺失时回落到 ``doc_id``（回落顺序只有一份实现）."""
        result = pack_context([make_hit("h-1", "正文", metadata={"doc_id": "doc-x"})])

        assert result.text.splitlines()[0] == "[1] doc-x"
        assert result.citations[0].source == "doc-x"

    def test_source_falls_back_to_record_id(self) -> None:
        """两个键都没有时回落到 ``record_id``（引用永远不会是空的一句）."""
        result = pack_context([make_hit("h-1", "正文", metadata={})])

        assert result.text.splitlines()[0] == "[1] h-1"
        assert result.citations[0].source == "h-1"

    def test_empty_text_renders_the_header_only(self) -> None:
        """正文为空时只留块头，不补一行空洞的换行."""
        result = pack_context([make_hit("h-1", "")])

        assert result.text == HEADER_ONE
        assert result.char_count == 13


# --------------------------------------------------------------------------- #
# Citation 自身
# --------------------------------------------------------------------------- #


class TestCitationShape:
    """``Citation`` 是"编号 → 命中"的对照表，它自己也校验两条纪律."""

    def test_citation_projects_to_a_dict(self) -> None:
        """``to_dict`` 的五个键与字段一一对应（score 保留 6 位）."""
        payload = make_citation(2, "h-2", heading="手册 > 阈值", score=0.5).to_dict()

        assert payload == {
            "marker": 2,
            "record_id": "h-2",
            "source": SOURCE,
            "heading_path": "手册 > 阈值",
            "score": 0.5,
        }

    def test_citation_summary_line_contains_the_marker(self) -> None:
        """一行摘要从编号开始（报告里逐行打印的就是它）."""
        line = make_citation(3, "h-3", heading="手册 > 阈值", score=-0.25).summary_line()

        assert line.startswith("[3] ")
        assert "手册 > 阈值" in line
        assert "-0.250000" in line

    def test_citation_summary_line_omits_the_arrow_without_heading(self) -> None:
        """没有标题路径时那一段自然消失（不会留下一个孤零零的 ``›``）."""
        line = make_citation(1).summary_line()

        assert "›" not in line
        assert line.startswith(f"[1] {SOURCE}")

    def test_marker_must_start_at_one(self) -> None:
        """0 或负数会指向一个不存在的片段."""
        with pytest.raises(ContextError) as excinfo:
            make_citation(0)

        assert "从 1 起" in str(excinfo.value)

    def test_marker_must_not_be_a_bool(self) -> None:
        """``True`` 是 bool：它会让编号在报告里变成 True/False."""
        with pytest.raises(ContextError):
            make_citation(True)  # type: ignore[arg-type]

    def test_record_id_must_be_non_empty(self) -> None:
        """没有 id 的引用无法回库取原文."""
        for record_id in ("", "   "):
            with pytest.raises(ContextError):
                make_citation(1, record_id)


# --------------------------------------------------------------------------- #
# PackedContext 的一致性强校验
# --------------------------------------------------------------------------- #


class TestPackedContextGuards:
    """``PackedContext.__post_init__`` 把"账要平"这件事在构造期钉死."""

    def test_a_consistent_context_can_be_built(self) -> None:
        """合法的一份：1 条引用 / 1 条命中 / 编号从 1 起."""
        context = PackedContext(
            text="[1] docs/a.md",
            citations=(make_citation(1, "h-1"),),
            used_hits=(make_hit("h-1"),),
            dropped_hits=(),
            truncated_hits=0,
            char_count=13,
            max_chars=100,
        )

        assert context.count == 1
        assert context.marker_for("h-1") == 1

    def test_citation_count_must_match_used_hits(self) -> None:
        """引用数 != 命中数 → 报错：多出来的引用指向提示词里不存在的片段."""
        with pytest.raises(ContextError) as excinfo:
            PackedContext(
                text="t",
                citations=(make_citation(1, "h-1"),),
                used_hits=(make_hit("h-1"), make_hit("h-2")),
                dropped_hits=(),
                truncated_hits=0,
                char_count=1,
                max_chars=100,
            )

        assert "引用 1 条" in str(excinfo.value)
        assert "一一对应" in str(excinfo.value)

    def test_markers_must_be_contiguous_from_one(self) -> None:
        """编号 (2, 2) 既不从 1 起也不连续 → 报错（中间断号会让模型写出空位的 ``[n]``）."""
        with pytest.raises(ContextError) as excinfo:
            PackedContext(
                text="t",
                citations=(make_citation(2, "h-1"), make_citation(2, "h-2")),
                used_hits=(make_hit("h-1"), make_hit("h-2")),
                dropped_hits=(),
                truncated_hits=0,
                char_count=1,
                max_chars=100,
            )

        assert "1 起连续" in str(excinfo.value)

    def test_counters_must_not_be_negative(self) -> None:
        """负数是报告里最没有意义的东西（两个计数字段都要挡住）."""
        for field in ("truncated_hits", "char_count"):
            payload: dict[str, Any] = {
                "text": "t",
                "citations": (),
                "used_hits": (),
                "dropped_hits": (),
                "truncated_hits": 0,
                "char_count": 0,
                "max_chars": 100,
            }
            payload[field] = -1
            with pytest.raises(ContextError) as excinfo:
                PackedContext(**payload)

            assert "计数字段非法" in str(excinfo.value)

    def test_max_chars_must_be_positive(self) -> None:
        """零预算是配置错误：它会让"被预算削过没有"这个事实无法被读出."""
        with pytest.raises(ContextError):
            PackedContext(
                text="t",
                citations=(),
                used_hits=(),
                dropped_hits=(),
                truncated_hits=0,
                char_count=0,
                max_chars=0,
            )

    def test_a_context_dropped_by_hand_can_be_built(self) -> None:
        """手工构造一份"有丢弃"的账也是合法的（端点与演示脚本会这么用）."""
        hits = flat_hits(2)
        context = PackedContext(
            text="[1] docs/a.md",
            citations=(make_citation(1, hits[0].record_id),),
            used_hits=(hits[0],),
            dropped_hits=(hits[1],),
            truncated_hits=0,
            char_count=13,
            max_chars=13,
        )

        assert context.count == 1
        assert len(context.dropped_hits) == 1
        assert context.fill_ratio == 1.0


class TestPackedContextProjection:
    """``to_dict`` / ``summary_line`` / 三个属性是这一层给人看的四个出口."""

    def test_to_dict_contains_the_whole_ledger(self) -> None:
        """账本四件套都必须出现：占用、预算、条数与两个"丢/截"的数字."""
        result = pack_context(flat_hits(4), max_chars=60, per_hit_chars=800)
        payload = result.to_dict()

        assert payload["count"] == 2
        assert payload["char_count"] == 50 and payload["max_chars"] == 60
        assert payload["truncated_hits"] == 0
        assert payload["used"] == ["h-1", "h-2"]
        assert payload["dropped"] == ["h-3", "h-4"]
        assert payload["text"] == result.text
        assert [item["marker"] for item in payload["citations"]] == [1, 2]

    def test_to_dict_is_json_serializable(self) -> None:
        """端点要直接返回它（``json.dumps`` 不许因为某个字段失败）."""
        import json

        payload = pack_context(flat_hits(2)).to_dict()
        assert json.loads(json.dumps(payload, ensure_ascii=False))["count"] == 2

    def test_summary_line_mentions_the_budget_and_the_counts(self) -> None:
        """一行摘要要同时给出占用率、引用数、截断数与丢尾数."""
        line = pack_context(flat_hits(4), max_chars=60, per_hit_chars=800).summary_line()

        assert "上下文 50/60 字" in line
        assert "引用 2 条" in line
        assert "截断 0 条" in line
        assert "丢尾 2 条" in line

    def test_fill_ratio_is_rounded_to_four_places(self) -> None:
        """``50/60`` 保留 4 位（"包很满"这件事要能与 ``dropped_hits`` 一起读）."""
        result = pack_context(flat_hits(4), max_chars=60, per_hit_chars=800)

        assert result.fill_ratio == 0.8333

    def test_count_and_is_empty_reflect_the_used_hits(self) -> None:
        """``count`` 数的是进包条数（不是入参条数）."""
        result = pack_context(flat_hits(4), max_chars=60, per_hit_chars=800)

        assert result.count == 2
        assert result.is_empty is False
        assert pack_context([]).is_empty is True
