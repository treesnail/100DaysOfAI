"""day069 ``RAGGenerator.generate`` 的九步流水线单元测试（六种回退 + 逐次覆盖）.

这一层的价值几乎全在**第 2 步的位置**与**六条回退的分岔**上：

```text
第 2 步（空上下文早退）       它是本课最重要的护栏：没有片段时一次 LLM 都不调
第 4~8 步（五条回退）         每一次都必须是"带原因的结果"，而不是一个栈
第 9 步（正常路径）           fallback_reason 是空串，且核对出了至少一条有效引用
```

因此本文件最重要的一组用例不是"正常答案长什么样"，而是
**用一个"一旦被调用就抛异常"的假 LLM 证明护栏生效**（``ExplodingLLM``，
从 ``test_retrieval_pipeline`` **复用**，不另写一套）：``llm.calls == 0``
是"模型没有机会自由发挥"的唯一证据——判定读的是它，不是"answer 等不等于兜底串"
（兜底文案改一个字，字符串比较那种判定就会静默失效）。

其余六组：

```text
[n] 解析      正常 / 部分引用 / 越界 / [0] / 重复 / 带空格 / 一段里多个编号
接地两侧      valid 非空且 invalid 空 → True；有 invalid → False；valid 空 → False
拒答判定      三个固定句式各一条 + 四条**反例**（含"资料"二字但没命中句式）
空回复 / 报错  空串与纯空白各一条；必然抛异常的假 LLM 走 llm_error
引用开关       require_citation 开/关 × 有/无合法引用的四格矩阵
逐次覆盖       三个键各自生效、清单外的键报错、覆盖必须进 notes
提示词版本     默认 v2；覆盖成 v1 时 v1 的特征串在场、v2 的不在场
链路集成       pipeline.RagPipeline 的回填、describe()、空检索一次都不调
```

**刻意不写"软断言"**（``is not None`` / 只判真假）：所有断言都指向具体数值、
具体次序、具体错误消息片段或具体的注记话术。

期望值来自 ``tests/generation_samples.py``（五条片段的编号表、六组核对输出、
七个答案脚本的手算推导都写在那里）。

全部离线、确定性、零网络。
"""

from __future__ import annotations

from typing import Any

import pytest

from smart_research_agent.config import settings
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.retrieval.errors import GenerationError
from smart_research_agent.retrieval.generation import (
    CURRENT_PROMPT_VERSION,
    DECLINE_MARKERS,
    FALLBACK_NO_CONTEXT,
    FALLBACK_REASON_EMPTY_REPLY,
    FALLBACK_REASON_LLM_ERROR,
    FALLBACK_REASON_MODEL_DECLINED,
    FALLBACK_REASON_NO_CONTEXT,
    FALLBACK_REASON_NONE,
    FALLBACK_REASON_UNUSABLE_CITATIONS,
    GENERATION_OVERRIDE_KEYS,
    PROMPT_VERSION_V1,
    PROMPT_VERSION_V2,
    RAG_PROMPT_V2,
    GroundingReport,
    RAGGenerator,
)
from smart_research_agent.retrieval.pipeline import RagPipeline
from smart_research_agent.retrieval.retriever import Retriever
from smart_research_agent.retrieval.types import RetrievalQuery
from tests.generation_samples import (
    ANSWER_ALL_FIVE,
    ANSWER_EMPTY,
    ANSWER_MIXED,
    ANSWER_MULTI,
    ANSWER_NO_CITATION,
    ANSWER_ONE,
    ANSWER_OUT_OF_RANGE,
    ANSWER_SPACED,
    ANSWER_TWO,
    ANSWER_ZERO_NOISE,
    CITATION_IDS,
    COVERAGE_HALF,
    COVERAGE_THREE,
    DECLINE_ANSWER_BY_MARKER,
    DECLINE_COUNTEREXAMPLES,
    GROUNDING_CASES,
    MARKER_TO_ID,
    QUESTION,
    ROUNDING_CASES,
    AnswerLLM,
    empty_packed_context,
    packed_context,
    prompt_of,
)
from tests.retrieval_samples import TableEmbedding, empty_store, sample_store
from tests.test_retrieval_pipeline import ExplodingLLM, RecordingLLM

# --------------------------------------------------------------------------- #
# 假 LLM 与装配助手
# --------------------------------------------------------------------------- #


class RaisingLLM(BaseLLM):
    """每次调用都抛出**指定异常**的假 LLM（用来钉住"宽捕获"的边界）."""

    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.calls = 0

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """被调用即抛出那个异常."""
        self.calls += 1
        raise self._error


def make_generator(llm: BaseLLM | None = None, **overrides: Any) -> RAGGenerator:
    """造一个生成器（缺省：按脚本返回"正常引用 [1]"的假 LLM）."""
    return RAGGenerator(llm if llm is not None else AnswerLLM(ANSWER_TWO), **overrides)


def make_pipeline(
    *,
    llm: BaseLLM | None = None,
    store: Any = None,
    **overrides: Any,
) -> RagPipeline:
    """装配一条 RAG 链路（缺省：样本库 + 一句带 [1] 的脚本回复）."""
    resolved = llm if llm is not None else AnswerLLM(ANSWER_TWO)
    retriever = Retriever(
        sample_store() if store is None else store, TableEmbedding()
    )
    return RagPipeline(retriever, resolved, **overrides)


# --------------------------------------------------------------------------- #
# 第 2 步：空上下文早退（本课最重要的护栏）
# --------------------------------------------------------------------------- #


class TestNoContextGuard:
    """没有片段时模型仍然会写出一段通顺的答案，而读的人看不出区别 —— 因此在门口拦住."""

    def test_the_input_context_really_is_empty(self) -> None:
        """先确认样本本身是那条路（``is_empty`` 为真、没有编号、文本为空）."""
        context = empty_packed_context()

        assert context.is_empty is True
        assert context.count == 0
        assert context.citations == ()
        assert context.text == ""

    def test_an_empty_context_never_touches_the_model(self) -> None:
        """``MockLLM`` 的 ``calls`` 是列表：次数必须是 0（不是"少调了一次"）."""
        llm = MockLLM(responses=[ANSWER_TWO])

        make_generator(llm).generate(QUESTION, empty_packed_context())

        assert llm.calls == []
        assert len(llm.calls) == 0

    def test_the_guard_also_holds_for_an_exploding_llm(self) -> None:
        """换成"一被调用就炸"的假 LLM 同样安静（护栏失效时它会立刻炸出来）."""
        llm = ExplodingLLM()

        make_generator(llm).generate(QUESTION, empty_packed_context())

        assert llm.calls == 0

    def test_the_guard_holds_for_the_sample_answer_llm_too(self) -> None:
        """第三种假实现（``AnswerLLM``）也一样：护栏不挑模型."""
        llm = AnswerLLM(ANSWER_TWO)

        make_generator(llm).generate(QUESTION, empty_packed_context())

        assert llm.calls == []

    def test_the_fallback_reason_is_no_context(self) -> None:
        """原因必须是 ``no_context``（最靠上游的那一个）."""
        generation = make_generator().generate(QUESTION, empty_packed_context())

        assert generation.fallback_reason == FALLBACK_REASON_NO_CONTEXT
        assert generation.is_fallback is True

    def test_llm_called_is_false(self) -> None:
        """它是护栏的唯一证据（与假 LLM 的 calls 计数一一对应）."""
        assert (
            make_generator().generate(QUESTION, empty_packed_context()).llm_called is False
        )

    def test_the_prompt_is_never_rendered(self) -> None:
        """``prompt_text`` 必须是空串：没渲染过提示词（护栏在渲染之前）."""
        generation = make_generator().generate(QUESTION, empty_packed_context())

        assert generation.prompt_text == ""
        assert generation.to_dict()["prompt_chars"] == 0

    def test_there_are_no_citations_to_hand_back(self) -> None:
        """没有片段就没有引用表（与 ``context.citations`` 一致，都是空元组）."""
        generation = make_generator().generate(QUESTION, empty_packed_context())

        assert generation.citations == ()
        assert generation.check.given == 0

    def test_the_answer_is_the_fallback_answer(self) -> None:
        """答案直接取兜底答复（它带着三条出路，而不是一句失败宣告）."""
        generation = make_generator().generate(QUESTION, empty_packed_context())

        assert generation.answer == FALLBACK_NO_CONTEXT
        assert "换一种说法" in generation.answer

    @pytest.mark.parametrize("fallback", ["本次没有可用资料。", "请稍后重试或换个问法。"])
    def test_an_injected_fallback_answer_is_delivered(self, fallback: str) -> None:
        """装配期注入的兜底答复才是交付出去的那一份（一处定义、两处引用）."""
        generation = make_generator(fallback_answer=fallback).generate(
            QUESTION, empty_packed_context()
        )

        assert generation.answer == fallback

    def test_the_check_is_an_empty_report_with_a_note(self) -> None:
        """核对报告的三组编号都只能是空的，且**注记要说明"没有东西可引"**."""
        check = make_generator().generate(QUESTION, empty_packed_context()).check

        assert isinstance(check, GroundingReport)
        assert check.cited == ()
        assert check.valid == ()
        assert check.invalid == ()
        assert check.unused == ()
        assert check.coverage == 0.0
        assert check.grounded is False
        assert check.checks == ()
        assert len(check.notes) == 1
        assert "没有任何片段可核对" in check.notes[0]
        assert "没有东西可引" in check.notes[0]

    def test_the_guard_writes_three_notes(self) -> None:
        """三条注记：为什么没调模型 / 答案取的是兜底 / 三个参数这一次都没生效."""
        generation = make_generator().generate(QUESTION, empty_packed_context())

        assert len(generation.notes) == 3
        assert any("一个 LLM 都没有调" in note for note in generation.notes)
        assert any("答案取的是兜底答复" in note for note in generation.notes)
        assert any("被忽略的参数" in note for note in generation.notes)

    def test_the_ignored_parameters_note_names_the_three_knobs(self) -> None:
        """被忽略的参数必须被说出来（否则调用方会以为它生效了）."""
        generation = make_generator(temperature=0.5, max_tokens=32).generate(
            QUESTION, empty_packed_context()
        )

        note = generation.notes[-1]
        assert "temperature=0.5" in note
        assert "max_tokens=32" in note
        assert "require_citation=False" in note

    def test_the_summary_line_says_the_model_was_not_called(self) -> None:
        """一行摘要里"调没调模型"必须看得见（护栏的唯一可见证据）."""
        line = make_generator().generate(QUESTION, empty_packed_context()).summary_line()

        assert "提示词 v2（未调用）" in line
        assert "接地未通过" in line
        assert line.endswith("| 回退 no_context")

    def test_the_explain_lists_the_way_out(self) -> None:
        """``explain()`` 里有回退那一段，且用的是同一本解释字典（出路必须跟着走）."""
        lines = make_generator().generate(QUESTION, empty_packed_context()).explain()

        assert len(lines) == 8
        assert lines[4].startswith("回退 no_context：")
        assert "调大打包预算" in lines[4]

    def test_the_guard_does_not_check_the_question_against_the_library(self) -> None:
        """护栏只认"上下文空不空"，不猜问题相不相关（相关性是检索那一层的事）."""
        generation = make_generator().generate(
            "一个与库毫无关系的问题？", empty_packed_context()
        )

        assert generation.fallback_reason == FALLBACK_REASON_NO_CONTEXT
        assert generation.question == "一个与库毫无关系的问题？"


# --------------------------------------------------------------------------- #
# 第 6 步：[n] 解析的手算核对
# --------------------------------------------------------------------------- #


class TestMarkerParsing:
    """核对只做**编号层面**的对账（纯字符串，不花一分钱），因此每个数都能手算."""

    @pytest.mark.parametrize("case", GROUNDING_CASES, ids=[c.label for c in GROUNDING_CASES])
    def test_every_hand_case_matches_the_generation(self, case: Any) -> None:
        """六组输出逐项与手算表一致（分母恒为 5，分子是手数的编号个数）."""
        check = make_generator(AnswerLLM(case.answer)).generate(
            QUESTION, packed_context()
        ).check

        assert check.cited == case.cited
        assert check.valid == case.valid
        assert check.invalid == case.invalid
        assert check.unused == case.unused
        assert check.coverage == case.coverage
        assert check.grounded is case.grounded
        assert len(check.notes) == case.report_notes

    @pytest.mark.parametrize("case", GROUNDING_CASES, ids=[c.label for c in GROUNDING_CASES])
    def test_every_hand_case_reaches_the_expected_reason(self, case: Any) -> None:
        """这些脚本都没有命中固定句式，也没有开开关：原因由"回没回复"决定.

        ```text
        有回复且 valid 非空 → 正常路径（reason 是空串）
        有回复但 valid 为空 → **也是**正常路径：开关没开时它照样交付，
                             只是报告里 coverage 是 0、grounded 是 False
        空回复（空串 / 纯空白）→ empty_reply：它是第 5 步先撞上的那条分支
        ```
        """
        generation = make_generator(AnswerLLM(case.answer)).generate(
            QUESTION, packed_context()
        )

        assert generation.llm_called is True
        assert generation.answer == case.answer
        if any(marker in case.answer for marker in DECLINE_MARKERS):
            assert generation.fallback_reason == FALLBACK_REASON_MODEL_DECLINED
        elif case.answer.strip():
            assert generation.fallback_reason == FALLBACK_REASON_NONE
        else:
            assert generation.fallback_reason == FALLBACK_REASON_EMPTY_REPLY

    def test_a_normal_answer_cites_two_of_the_five_clauses(self) -> None:
        """``[1] 与 [2]``：有效 2 条、幻觉 0 条、未引用 3 条、覆盖 0.4."""
        check = make_generator(AnswerLLM(ANSWER_TWO)).generate(
            QUESTION, packed_context()
        ).check

        assert check.cited == (1, 2)
        assert check.valid == (1, 2)
        assert check.unused == (3, 4, 5)
        assert check.coverage == 0.4
        assert check.summary_line() == (
            "核对：给了 5 条片段 | 答案引用 2 条（有效 2 / 幻觉 0）"
            " | 未引用 3 条 | 覆盖 40.0% | 接地通过"
        )

    def test_a_partial_answer_leaves_four_clauses_unused(self) -> None:
        """只引中间那一条（``[3]``）：有效 1、未引用 4、覆盖 0.2."""
        check = make_generator(AnswerLLM(ANSWER_ONE)).generate(
            QUESTION, packed_context()
        ).check

        assert check.valid == (3,)
        assert check.unused == (1, 2, 4, 5)
        assert check.coverage == 0.2

    def test_citing_all_five_reaches_the_full_coverage(self) -> None:
        """``[1][2][3][4][5]``：覆盖 1.0、未引用是空元组（分母的另一半为零）."""
        check = make_generator(AnswerLLM(ANSWER_ALL_FIVE)).generate(
            QUESTION, packed_context()
        ).check

        assert check.cited == (1, 2, 3, 4, 5)
        assert check.unused == ()
        assert check.coverage == 1.0
        assert check.given == 5

    def test_an_out_of_range_marker_is_a_hallucination(self) -> None:
        """``[9]``：提示词里只有 [1]~[5]，因此它落不到任何片段上（valid 为空）."""
        check = make_generator(AnswerLLM(ANSWER_OUT_OF_RANGE)).generate(
            QUESTION, packed_context()
        ).check

        assert check.cited == (9,)
        assert check.valid == ()
        assert check.invalid == (9,)
        assert check.coverage == 0.0

    def test_the_zero_marker_is_neither_a_citation_nor_a_hallucination(self) -> None:
        """``[0]`` 不是编号：它不进 cited / valid / invalid，只留一句注记."""
        check = make_generator(AnswerLLM(ANSWER_ZERO_NOISE)).generate(
            QUESTION, packed_context()
        ).check

        assert check.cited == (1,)
        assert check.invalid == ()
        assert 0 not in check.cited
        assert len(check.notes) == 1
        assert "[0]" in check.notes[0]
        assert "0 不是编号" in check.notes[0]

    def test_a_spaced_marker_is_understood(self) -> None:
        """``[ 1 ]`` 里的空格不影响解析（正则读的是编号语法）."""
        check = make_generator(AnswerLLM(ANSWER_SPACED)).generate(
            QUESTION, packed_context()
        ).check

        assert check.cited == (1, 2)
        assert check.valid == (1, 2)
        assert check.coverage == 0.4

    def test_duplicated_markers_are_deduplicated(self) -> None:
        """同一个编号写三遍也只算一条（升序去重是这一层的硬口径）."""
        check = make_generator(AnswerLLM("先说 [1]，再说 [1]，最后仍是 [1]。")).generate(
            QUESTION, packed_context()
        ).check

        assert check.cited == (1,)
        assert check.coverage == 0.2
        assert check.checks[0].used is True

    def test_many_markers_in_one_paragraph_are_all_counted(self) -> None:
        """一段里连写三个编号：三个都要被看见（不是"整段找一个"）."""
        check = make_generator(AnswerLLM(ANSWER_MULTI)).generate(
            QUESTION, packed_context()
        ).check

        assert check.cited == (1, 2, 3)
        assert check.valid == (1, 2, 3)
        assert check.unused == (4, 5)
        assert check.coverage == 0.6

    def test_a_two_digit_marker_is_parsed_as_one_number(self) -> None:
        """``[12]`` 是一个编号（不是 ``[1]`` 与 ``[2]``）——它照样是越界引用."""
        check = make_generator(AnswerLLM("见 [12]。")).generate(
            QUESTION, packed_context()
        ).check

        assert check.cited == (12,)
        assert check.invalid == (12,)

    def test_an_unbracketed_number_is_not_a_marker(self) -> None:
        """正文里的裸数字不是引用（否则每一条含数字的答案都会被误读成引用了片段）."""
        check = make_generator(AnswerLLM("默认返回 5 条，深度是 3 倍。")).generate(
            QUESTION, packed_context()
        ).check

        assert check.cited == ()
        assert check.unused == (1, 2, 3, 4, 5)
        assert len(check.notes) == 1

    @pytest.mark.parametrize(("count", "used", "expected"), ROUNDING_CASES)
    def test_the_coverage_matches_the_hand_rounded_value(
        self, count: int, used: int, expected: float
    ) -> None:
        """覆盖率 = 有效引用 / 给出去的片段数，**按 4 位小数**收敛.

        分子用 ``used`` 条（引到前 ``used`` 个编号），分母是这次的片段数 ``count``：
        ``round(used / count, 4)`` 的期望值写在 ``ROUNDING_CASES`` 里，逐位手算。
        """
        answer = " ".join(f"[{marker}]" for marker in range(1, used + 1))
        check = make_generator(AnswerLLM(answer)).generate(
            QUESTION, packed_context(count)
        ).check

        assert check.given == count
        assert check.valid == tuple(range(1, used + 1))
        assert check.coverage == expected
        assert check.grounded is True

    def test_the_two_clause_denominator_is_a_half(self) -> None:
        """分母换成 2 时覆盖率是 1/2 = 0.5（同一个分子在两个分母下是两个数）."""
        check = make_generator(AnswerLLM("[1] 支持这个结论。")).generate(
            QUESTION, packed_context(2)
        ).check

        assert check.given == 2
        assert check.unused == (2,)
        assert check.coverage == COVERAGE_HALF

    def test_the_three_marker_case_reaches_sixty_percent(self) -> None:
        """分母 5、分子 3 → 3/5 = 0.6（手算常量 ``COVERAGE_THREE``）."""
        check = make_generator(AnswerLLM(ANSWER_MULTI)).generate(
            QUESTION, packed_context()
        ).check

        assert check.coverage == COVERAGE_THREE
        assert len(check.valid) == 3
        assert len(check.unused) == 2

    def test_the_eight_clause_denominator_needs_three_decimals(self) -> None:
        """分母换成 8 时 1/8 = 0.125（round 到 4 位不会改动它）."""
        check = make_generator(AnswerLLM("[1] 支持这个结论。")).generate(
            QUESTION, packed_context(8)
        ).check

        assert check.given == 8
        assert check.unused == (2, 3, 4, 5, 6, 7, 8)
        assert check.coverage == 0.125

    @pytest.mark.parametrize(
        ("count", "answer", "cited", "valid", "invalid", "unused", "coverage"),
        [
            # 分母 2：[1][2] 全引 → 覆盖 1.0；只引 [2] → 0.5；引 [7] → 越界
            (2, "[1][2] 都支持。", (1, 2), (1, 2), (), (), 1.0),
            (2, "只靠 [2]。", (2,), (2,), (), (1,), 0.5),
            (2, "引了 [7]。", (7,), (), (7,), (1, 2), 0.0),
            # 分母 3：引 [1][3] → 2/3 = 0.6667；只引 [3] → 0.3333
            (3, "[1] 与 [3] 两句。", (1, 3), (1, 3), (), (2,), 0.6667),
            (3, "只引 [3]。", (3,), (3,), (), (1, 2), 0.3333),
            # 分母 5 的中间几档
            (5, "[4] 一条。", (4,), (4,), (), (1, 2, 3, 5), 0.2),
            (5, "[2][4] 两条。", (2, 4), (2, 4), (), (1, 3, 5), 0.4),
            (5, "[1][2][3][4] 四条。", (1, 2, 3, 4), (1, 2, 3, 4), (), (5,), 0.8),
            # 分母 8 的两档（8 条全在库里）
            (8, "[1] 与 [8]。", (1, 8), (1, 8), (), (2, 3, 4, 5, 6, 7), 0.25),
            (8, "[7] 与 [9]。", (7, 9), (7,), (9,), (1, 2, 3, 4, 5, 6, 8), 0.125),
        ],
    )
    def test_the_marker_table_for_other_context_sizes(
        self,
        count: int,
        answer: str,
        cited: tuple[int, ...],
        valid: tuple[int, ...],
        invalid: tuple[int, ...],
        unused: tuple[int, ...],
        coverage: float,
    ) -> None:
        """换一个分母（2 / 3 / 5 / 8）之后，六个输出仍然全部手算得出来.

        ```text
        未引用集合 = 1..count 减去 cited 里落在范围内的那些（越界编号不占位置）
        覆盖率     = len(valid) / count，按 4 位小数收敛
        ```
        """
        check = make_generator(AnswerLLM(answer)).generate(
            QUESTION, packed_context(count)
        ).check

        assert check.given == count
        assert check.cited == cited
        assert check.valid == valid
        assert check.invalid == invalid
        assert check.unused == unused
        assert check.coverage == coverage
        assert check.grounded is (bool(valid) and not invalid)


# --------------------------------------------------------------------------- #
# 第 7 步：接地判定的两侧
# --------------------------------------------------------------------------- #


class TestGroundedBothSides:
    """``grounded = valid 非空 且 invalid 为空``：两个前提各有一条用例."""

    def test_a_valid_citation_without_hallucination_is_grounded(self) -> None:
        """前半条成立、后半条也成立 → True（这是唯一判 True 的形状）."""
        check = make_generator(AnswerLLM(ANSWER_TWO)).generate(
            QUESTION, packed_context()
        ).check

        assert check.valid != ()
        assert check.invalid == ()
        assert check.grounded is True

    def test_any_hallucination_breaks_grounding(self) -> None:
        """有幻觉引用时即使 valid 非空也不是"有可核对依据"（两个数各说各的）."""
        check = make_generator(AnswerLLM(ANSWER_MIXED)).generate(
            QUESTION, packed_context()
        ).check

        assert check.valid == (1,)
        assert check.invalid == (9,)
        assert check.grounded is False
        assert check.coverage == 0.2

    def test_an_empty_valid_set_is_not_grounded(self) -> None:
        """一次没有依据的总结（一条 [n] 都没写）→ False，哪怕它读起来通顺."""
        check = make_generator(AnswerLLM(ANSWER_NO_CITATION)).generate(
            QUESTION, packed_context()
        ).check

        assert check.valid == ()
        assert check.invalid == ()
        assert check.grounded is False

    def test_only_out_of_range_markers_are_not_grounded(self) -> None:
        """全是幻觉引用（valid 空 + invalid 非空）→ False（两种失败同时发生）."""
        check = make_generator(AnswerLLM(ANSWER_OUT_OF_RANGE)).generate(
            QUESTION, packed_context()
        ).check

        assert check.valid == ()
        assert check.invalid == (9,)
        assert check.grounded is False
        assert check.hallucinated == 1

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            (ANSWER_TWO, True),
            (ANSWER_ONE, True),
            (ANSWER_ALL_FIVE, True),
            (ANSWER_MIXED, False),
            (ANSWER_OUT_OF_RANGE, False),
            (ANSWER_NO_CITATION, False),
            (ANSWER_ZERO_NOISE, True),
            (ANSWER_MULTI, True),
            (ANSWER_SPACED, True),
            ("结论见 [ 3 ]。", True),
            ("见 [12]。", False),
            ("[0] 一条格式噪声。", False),
            ("[1] 与 [12]。", False),
            ("[2][3][4] 三条依据。", True),
            ("这个配置项在片段里没有出现。", False),
            ("[1]，[1]，[1]。", True),
        ],
    )
    def test_the_grounding_truth_table(self, answer: str, expected: bool) -> None:
        """十六个脚本的接地判定逐条对照（真值表来自手算，不是跑一遍抄下来）.

        ```text
        引了不存在的编号（[12]）      → False（哪怕它同时引了合法的 [1]）
        只写了 [0]                    → False（0 不是编号，等于一条引用都没有）
        重复写同一个编号              → True（去重之后仍是一条有效引用）
        ```
        """
        generation = make_generator(AnswerLLM(answer)).generate(
            QUESTION, packed_context()
        )

        assert generation.check.grounded is expected
        assert generation.grounded is expected

    def test_grounding_is_a_projection_of_the_check(self) -> None:
        """``Generation.grounded`` 只是转发（不另存一份可能分家的副本）."""
        generation = make_generator(AnswerLLM(ANSWER_TWO)).generate(
            QUESTION, packed_context()
        )

        assert generation.grounded is generation.check.grounded


# --------------------------------------------------------------------------- #
# 第 7 步：拒答判定（固定句式）
# --------------------------------------------------------------------------- #


class TestDeclineDetection:
    """v2 要求模型资料不足时以固定句式开头：句式固定之后这件事才**可检测**."""

    @pytest.mark.parametrize(
        ("marker", "answer"),
        sorted(DECLINE_ANSWER_BY_MARKER.items()),
        ids=sorted(DECLINE_ANSWER_BY_MARKER),
    )
    def test_each_fixed_sentence_is_detected(self, marker: str, answer: str) -> None:
        """三个句式各一条：命中之后原因必须是 ``model_declined``."""
        generation = make_generator(AnswerLLM(answer)).generate(
            QUESTION, packed_context()
        )

        assert generation.fallback_reason == FALLBACK_REASON_MODEL_DECLINED
        assert generation.llm_called is True
        assert generation.answer == answer

    @pytest.mark.parametrize(
        ("marker", "answer"),
        sorted(DECLINE_ANSWER_BY_MARKER.items()),
        ids=sorted(DECLINE_ANSWER_BY_MARKER),
    )
    def test_the_note_names_the_sentence_that_was_hit(self, marker: str, answer: str) -> None:
        """报告里要写出"命中了哪一句"（否则"模型说资料不够"这个结论无法被复核）."""
        generation = make_generator(AnswerLLM(answer)).generate(
            QUESTION, packed_context()
        )

        assert any(f"命中固定句式 {marker!r}" in note for note in generation.notes)

    @pytest.mark.parametrize("answer", DECLINE_COUNTEREXAMPLES)
    def test_an_answer_that_merely_mentions_the_library_is_not_a_decline(
        self, answer: str
    ) -> None:
        """**反例**：含"资料"二字但没有命中固定句式 → 必须走正常路径.

        误判一次拒答等于把一次正常回答丢掉，而它的表现恰好是"看起来更谨慎了"。
        """
        generation = make_generator(AnswerLLM(answer)).generate(
            QUESTION, packed_context()
        )

        assert generation.fallback_reason == FALLBACK_REASON_NONE
        assert generation.answer == answer

    @pytest.mark.parametrize(
        "answer",
        [
            "资料中 没有相关内容，缺时间范围。",
            "资料中没有相关内容，",
            "（资料中没有相关内容）",
            "结论：资料未提及。",
            "无法依据资料回答：缺少配置项。",
        ],
    )
    def test_whitespace_and_punctuation_do_not_defeat_the_detection(self, answer: str) -> None:
        """归一化会去掉空白与标点（模型写"资料中没有相关内容，"也是拒答）."""
        generation = make_generator(AnswerLLM(answer)).generate(
            QUESTION, packed_context()
        )

        assert generation.fallback_reason == FALLBACK_REASON_MODEL_DECLINED

    def test_the_detection_is_a_containment_not_a_prefix(self) -> None:
        """判定是"包含"而不是"以它开头"（模型常在给出结论之后才说资料不足）."""
        answer = "先说结论：这一段没有直接依据。资料未提及该配置项。"

        generation = make_generator(AnswerLLM(answer)).generate(
            QUESTION, packed_context()
        )

        assert generation.fallback_reason == FALLBACK_REASON_MODEL_DECLINED
        assert any("命中固定句式 '资料未提及'" in note for note in generation.notes)

    def test_the_first_marker_in_the_list_order_wins(self) -> None:
        """一句里命中多个句式时，报出来的是 ``DECLINE_MARKERS`` 里**靠前**的那一个.

        ``资料未提及`` 的下标是 1、``无法依据资料回答`` 是 2，因此报 1。
        """
        answer = "无法依据资料回答，因为资料未提及这个配置项。"

        generation = make_generator(AnswerLLM(answer)).generate(
            QUESTION, packed_context()
        )

        assert any("命中固定句式 '资料未提及'" in note for note in generation.notes)
        assert all("命中固定句式 '无法依据资料回答'" not in note for note in generation.notes)

    def test_a_decline_is_reported_even_with_valid_citations(self) -> None:
        """拒答的判定**排在引用开关之前**：带着 [1] 的拒答也是拒答（先命中先处置）."""
        answer = "资料未提及该配置项，只能按 [1] 的默认值说明。"

        generation = make_generator(AnswerLLM(answer), require_citation=True).generate(
            QUESTION, packed_context()
        )

        assert generation.check.valid == (1,)
        assert generation.fallback_reason == FALLBACK_REASON_MODEL_DECLINED

    def test_a_decline_explains_the_two_possible_causes(self) -> None:
        """注记要说清"该查检索还是该改提示词"（两种原因对应两种动作）."""
        generation = make_generator(AnswerLLM("资料中 没有相关内容。")).generate(
            QUESTION, packed_context()
        )

        assert len(generation.notes) == 3
        assert any("多半是检索给的片段不相关" in note for note in generation.notes)
        assert any("这次给了 5 条片段、答案引用到 0 条" in note for note in generation.notes)


# --------------------------------------------------------------------------- #
# 第 5 步：空回复
# --------------------------------------------------------------------------- #


class TestEmptyReply:
    """空回复不是"知识库里没有相关内容"（那一类一次 LLM 都不会调）."""

    @pytest.mark.parametrize(("label", "reply"), [("empty", ANSWER_EMPTY), ("blank", "   \n\t  ")])
    def test_an_empty_reply_falls_back_but_keeps_the_reply(
        self, label: str, reply: str
    ) -> None:
        """``answer`` 必须**原样**是模型给的那个串（兜底答复是另一条路的事）."""
        generation = make_generator(AnswerLLM(reply)).generate(
            QUESTION, packed_context()
        )

        assert generation.fallback_reason == FALLBACK_REASON_EMPTY_REPLY
        assert generation.answer == reply
        assert generation.answer != FALLBACK_NO_CONTEXT
        assert generation.llm_called is True

    @pytest.mark.parametrize(("label", "reply"), [("empty", ANSWER_EMPTY), ("blank", "   ")])
    def test_the_prompt_was_still_rendered(self, label: str, reply: str) -> None:
        """空回复是"调了、答了、答的是空"——提示词与引用表都在（与 no_context 对照）."""
        generation = make_generator(AnswerLLM(reply)).generate(
            QUESTION, packed_context()
        )

        assert "## 约束" in generation.prompt_text
        assert len(generation.citations) == 5
        assert generation.check.given == 5

    @pytest.mark.parametrize(("label", "reply"), [("empty", ANSWER_EMPTY), ("blank", "   ")])
    def test_the_notes_say_it_is_not_the_same_as_no_context(
        self, label: str, reply: str
    ) -> None:
        """两条注记：现象 + "它**不是**知识库里没有内容"这条对照."""
        generation = make_generator(AnswerLLM(reply)).generate(
            QUESTION, packed_context()
        )

        assert len(generation.notes) == 2
        assert "模型返回了空回复" in generation.notes[0]
        assert "不是**'知识库里没有相关内容'" in generation.notes[1]
        assert "调大 max_tokens（当前 1024）" in generation.notes[1]

    def test_an_empty_reply_gets_no_grounding_note_either(self) -> None:
        """没有实测过一段答案时，核对那一段注记整段不渲染（"覆盖率为 0"在那时不是结论）."""
        generation = make_generator(AnswerLLM(ANSWER_EMPTY)).generate(
            QUESTION, packed_context()
        )

        assert generation.check.cited == ()
        assert generation.check.unused == (1, 2, 3, 4, 5)
        assert generation.check.coverage == 0.0
        assert all("覆盖率为 0" not in note for note in generation.notes)

    def test_the_max_tokens_hint_follows_the_actual_setting(self) -> None:
        """出路里那个数字必须是**这一次**的上限（否则调参的人会调错那一处）."""
        generation = make_generator(AnswerLLM(ANSWER_EMPTY), max_tokens=64).generate(
            QUESTION, packed_context()
        )

        assert "调大 max_tokens（当前 64）" in generation.notes[1]


# --------------------------------------------------------------------------- #
# 第 4 步：调用抛异常
# --------------------------------------------------------------------------- #


class TestLLMError:
    """超时 / 鉴权 / 网络 / 额度都是**可预期的失败**：必须变成一份带原因的结果."""

    def test_the_exploding_llm_lands_on_llm_error(self) -> None:
        """复用 ``test_retrieval_pipeline`` 的那个假 LLM（它的异常消息就是断言目标）."""
        llm = ExplodingLLM()

        generation = make_generator(llm).generate(QUESTION, packed_context())

        assert generation.fallback_reason == FALLBACK_REASON_LLM_ERROR
        assert llm.calls == 1

    def test_the_answer_falls_back_but_the_call_is_recorded(self) -> None:
        """这一次**确实**调用了模型（护栏管的是"该不该给片段"）."""
        generation = make_generator(ExplodingLLM()).generate(
            QUESTION, packed_context()
        )

        assert generation.answer == FALLBACK_NO_CONTEXT
        assert generation.llm_called is True
        assert "## 约束" in generation.prompt_text

    def test_the_notes_name_the_exception_class_and_message(self) -> None:
        """注记里必须写出**异常类名 + 消息**（否则"调用失败了"这句话没有依据）."""
        generation = make_generator(ExplodingLLM()).generate(
            QUESTION, packed_context()
        )

        assert len(generation.notes) == 3
        assert generation.notes[0].startswith(
            "回退原因 llm_error：调用模型时抛出 AssertionError: "
        )
        assert "护栏被绕过了" in generation.notes[0]
        assert "它们都**不是**'资料里没有相关内容'" in generation.notes[0]

    def test_the_error_note_says_the_model_was_called(self) -> None:
        """第二条注记把"调了"写死（与 no_context 的"一次都没调"形成对照）."""
        notes = make_generator(ExplodingLLM()).generate(
            QUESTION, packed_context()
        ).notes

        assert "这一次**确实**调用了模型（llm_called=True）" in notes[1]
        assert "检查提供方的配置与网络" in notes[1]

    def test_the_fallback_text_mismatch_is_pointed_out(self) -> None:
        """兜底答复的文案是为 no_context 写的：这一处不一致必须被说出来."""
        notes = make_generator(ExplodingLLM()).generate(
            QUESTION, packed_context()
        ).notes

        assert "答案取的是兜底答复，而它的文案是为 no_context 写的" in notes[2]
        assert "知识库中没有检索到与该问" in notes[2]

    @pytest.mark.parametrize(
        ("error", "name"),
        [
            (TimeoutError("调用超时"), "TimeoutError"),
            (PermissionError("鉴权失败"), "PermissionError"),
            (RuntimeError("额度不足"), "RuntimeError"),
            (ValueError("提供方返回了坏结构"), "ValueError"),
        ],
    )
    def test_a_wide_range_of_exceptions_becomes_the_same_reason(
        self, error: Exception, name: str
    ) -> None:
        """四类异常（超时 / 鉴权 / 额度 / 坏结构）走同一条路，且类名进注记."""
        generation = make_generator(RaisingLLM(error)).generate(
            QUESTION, packed_context()
        )

        assert generation.fallback_reason == FALLBACK_REASON_LLM_ERROR
        assert f"调用模型时抛出 {name}: {error}" in generation.notes[0]

    def test_the_check_is_built_from_an_empty_answer(self) -> None:
        """调用失败时核对的是空串：三组编号都空、unused 是全部片段."""
        check = make_generator(ExplodingLLM()).generate(
            QUESTION, packed_context()
        ).check

        assert check.cited == ()
        assert check.valid == ()
        assert check.unused == (1, 2, 3, 4, 5)
        assert check.coverage == 0.0
        assert len(check.notes) == 1
        assert "答案里没有出现任何 [n]" in check.notes[0]

    def test_a_base_exception_is_not_swallowed(self) -> None:
        """宽捕获只收 ``Exception``：``KeyboardInterrupt`` 必须继续往上抛.

        把"进程被中断"也折成一份带原因的答案，会让 Ctrl-C 失效——
        那不是"可预期的失败"，而是调用方要求停止。
        """
        with pytest.raises(KeyboardInterrupt):
            make_generator(RaisingLLM(KeyboardInterrupt())).generate(
                QUESTION, packed_context()
            )

    def test_the_latency_is_recorded_even_on_failure(self) -> None:
        """``llm_error`` 那条路上"等了多久才失败"本身就是有用的数字（超时多从这里看）."""
        generation = make_generator(ExplodingLLM()).generate(
            QUESTION, packed_context()
        )

        assert isinstance(generation.latency_ms, float)
        assert generation.latency_ms >= 0.0

    def test_the_summary_line_carries_the_reason(self) -> None:
        """一行摘要末尾写出 ``llm_error``（一次失败必须在最外层可见）."""
        line = make_generator(ExplodingLLM()).generate(
            QUESTION, packed_context()
        ).summary_line()

        assert line.endswith("| 回退 llm_error")
        assert "提示词 v2（已调用）" in line


# --------------------------------------------------------------------------- #
# 第 8 步：引用开关的四格矩阵
# --------------------------------------------------------------------------- #


class TestRequireCitationMatrix:
    """``require_citation`` 是一道**闸门**：开了它，"一条合法引用都没有"不能交付."""

    @pytest.mark.parametrize(
        ("required", "answer", "expected"),
        [
            (False, ANSWER_NO_CITATION, FALLBACK_REASON_NONE),
            (True, ANSWER_NO_CITATION, FALLBACK_REASON_UNUSABLE_CITATIONS),
            (True, ANSWER_OUT_OF_RANGE, FALLBACK_REASON_UNUSABLE_CITATIONS),
            (False, ANSWER_OUT_OF_RANGE, FALLBACK_REASON_NONE),
            (False, ANSWER_TWO, FALLBACK_REASON_NONE),
            (True, ANSWER_TWO, FALLBACK_REASON_NONE),
            (True, ANSWER_MIXED, FALLBACK_REASON_NONE),
            (False, ANSWER_MIXED, FALLBACK_REASON_NONE),
        ],
    )
    def test_the_four_cells_of_the_matrix(
        self, required: bool, answer: str, expected: str
    ) -> None:
        """开关 × 有无合法引用的交叉表（8 组，含"只有幻觉引用"这一格）.

        ```text
        开关关 + 没有合法引用   照常交付（报告里是 coverage 0.0、grounded False）
        开关开 + 没有合法引用   答案取兜底（含"引用了不存在的编号"这一格）
        开关开 + 有合法引用     照常交付（哪怕同时还有幻觉引用）
        ```
        """
        generation = make_generator(
            AnswerLLM(answer), require_citation=required
        ).generate(QUESTION, packed_context())

        assert generation.fallback_reason == expected
        assert generation.is_fallback is (expected != FALLBACK_REASON_NONE)

    def test_the_matrix_delivers_the_model_answer_when_the_gate_is_open(self) -> None:
        """闸门打开、答案不合格时答案被换成兜底答复（原文被丢弃）."""
        generation = make_generator(
            AnswerLLM(ANSWER_NO_CITATION), require_citation=True
        ).generate(QUESTION, packed_context())

        assert generation.answer == FALLBACK_NO_CONTEXT
        assert ANSWER_NO_CITATION not in generation.answer

    def test_the_matrix_keeps_the_model_answer_when_the_gate_is_closed(self) -> None:
        """闸门关着时同一段答案照常交付（它是报告里的一项，而不是一道闸门）."""
        generation = make_generator(
            AnswerLLM(ANSWER_NO_CITATION), require_citation=False
        ).generate(QUESTION, packed_context())

        assert generation.answer == ANSWER_NO_CITATION
        assert generation.check.coverage == 0.0
        assert generation.check.grounded is False

    def test_the_gate_counts_only_valid_citations(self) -> None:
        """开闸时"有引用"必须指**合法**引用：全是幻觉引用同样是不可交付."""
        generation = make_generator(
            AnswerLLM(ANSWER_OUT_OF_RANGE), require_citation=True
        ).generate(QUESTION, packed_context())

        assert generation.check.cited == (9,)
        assert generation.check.valid == ()
        assert generation.fallback_reason == FALLBACK_REASON_UNUSABLE_CITATIONS

    def test_the_gate_note_counts_the_discarded_characters(self) -> None:
        """注记要写出"丢了多少字"（那是被闸门拦下的正文长度）."""
        generation = make_generator(
            AnswerLLM(ANSWER_NO_CITATION), require_citation=True
        ).generate(QUESTION, packed_context())

        assert len(generation.notes) == 2
        assert f"模型给的 {len(ANSWER_NO_CITATION)} 字答案已被丢弃" in generation.notes[1]
        assert "把 require_citation 关掉" in generation.notes[1]

    def test_the_gate_still_reports_the_unused_clauses(self) -> None:
        """开了开关之后，"给了的片段有几条没被用上"正是需要看的账."""
        generation = make_generator(
            AnswerLLM(ANSWER_TWO), require_citation=True
        ).generate(QUESTION, packed_context())

        assert len(generation.notes) == 1
        assert generation.notes[0].startswith(
            "未引用的片段 3 条（编号 [3]、[4]、[5]）"
        )

    def test_the_unused_note_is_not_rendered_when_the_gate_is_closed(self) -> None:
        """开关关着时"引了几条"是常态：每一次都念一遍会把它变成噪音."""
        generation = make_generator(
            AnswerLLM(ANSWER_TWO), require_citation=False
        ).generate(QUESTION, packed_context())

        assert generation.notes == ()

    @pytest.mark.parametrize("answer", [ANSWER_NO_CITATION, ANSWER_OUT_OF_RANGE])
    def test_the_zero_coverage_note_is_always_rendered(self, answer: str) -> None:
        """覆盖率为 0 的两条成因必须被分开读（一个是没写引用，一个是引了不存在的）."""
        generation = make_generator(AnswerLLM(answer)).generate(
            QUESTION, packed_context()
        )

        assert any("覆盖率为 0" in note for note in generation.notes)
        assert any("invalid 非空" in note for note in generation.notes)
        assert any("cited 为空" in note for note in generation.notes)

    def test_the_hallucination_note_points_at_the_rows(self) -> None:
        """有幻觉引用时注记要点名编号，并指向 ``check.checks`` 里 ``record_id`` 为空的那几条."""
        generation = make_generator(AnswerLLM(ANSWER_MIXED)).generate(
            QUESTION, packed_context()
        )

        assert len(generation.notes) == 1
        assert "幻觉引用 1 条" in generation.notes[0]
        assert "编号 [9] 不在那次提示词里" in generation.notes[0]
        assert "（那次只给了 [1]~[5]）" in generation.notes[0]
        assert "record_id 为空的那几条" in generation.notes[0]

    def test_a_clean_answer_has_no_notes_at_all(self) -> None:
        """没有降级就没有注记（注记多了会被读的人忽略）."""
        generation = make_generator(AnswerLLM(ANSWER_ALL_FIVE)).generate(
            QUESTION, packed_context()
        )

        assert generation.notes == ()


# --------------------------------------------------------------------------- #
# 批量生成
# --------------------------------------------------------------------------- #


class TestGenerateMany:
    """``generate_many`` 刻意逐条：每一次的 ``llm_called`` 与覆盖率都是独立证据."""

    def test_returns_one_generation_per_item_in_order(self) -> None:
        """顺序与入参一致（否则报告里"第 2 个问题"就没法核对了）."""
        llm = RecordingLLM(["第一 [1]", "第二 [1]", "第三 [1]"])
        items = [
            ("第一问？", packed_context()),
            ("第二问？", packed_context(2)),
            ("第三问？", packed_context()),
        ]

        generations = make_generator(llm).generate_many(items)

        assert [item.answer for item in generations] == ["第一 [1]", "第二 [1]", "第三 [1]"]
        assert [item.question for item in generations] == ["第一问？", "第二问？", "第三问？"]
        assert len(llm.calls) == 3

    def test_every_item_uses_its_own_context(self) -> None:
        """每一次用的是**自己那份上下文**（条数与编号都跟着它走）."""
        llm = RecordingLLM(["答案 [1]", "答案 [1]"])
        items = [("五条片段的问题？", packed_context()), ("两条片段的问题？", packed_context(2))]

        first, second = make_generator(llm).generate_many(items)

        assert len(first.citations) == 5
        assert first.check.given == 5
        assert len(second.citations) == 2
        assert second.check.given == 2

    def test_every_item_gets_its_own_prompt(self) -> None:
        """两条不同的提示词（逐条调用，不是把多个问题塞进一次调用）."""
        llm = RecordingLLM(["答案 [1]", "答案 [1]"])

        make_generator(llm).generate_many(
            [("第一个问题？", packed_context()), ("第二个问题？", packed_context())]
        )

        first = str(llm.calls[0]["messages"][0].content)
        second = str(llm.calls[1]["messages"][0].content)

        assert "第一个问题？" in first and "第二个问题？" not in first
        assert "第二个问题？" in second and "第一个问题？" not in second

    def test_an_empty_batch_returns_an_empty_list(self) -> None:
        """空批次不报错（与 ``answer_many`` / ``retrieve_many`` 的取向一致）."""
        assert make_generator().generate_many([]) == []

    def test_an_empty_context_in_the_middle_does_not_stop_the_batch(self) -> None:
        """一条走空上下文护栏、其余照常（脚本只够两次也能过）."""
        llm = RecordingLLM(["第一 [1]", "第三 [1]"])

        generations = make_generator(llm).generate_many(
            [
                ("第一问？", packed_context()),
                ("第二问？", empty_packed_context()),
                ("第三问？", packed_context()),
            ]
        )

        assert len(llm.calls) == 2
        assert [item.llm_called for item in generations] == [True, False, True]
        assert generations[1].fallback_reason == FALLBACK_REASON_NO_CONTEXT
        assert generations[1].answer == FALLBACK_NO_CONTEXT

    @pytest.mark.parametrize("items", ["不是序列", 123, {"a": 1}])
    def test_the_batch_must_be_a_sequence(self, items: Any) -> None:
        """入参必须是序列（一次调用一条，顺序即结果顺序）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generator().generate_many(items)

        assert "generate_many 的入参必须是序列" in str(excinfo.value)

    @pytest.mark.parametrize("bad", ["x", ("只有一项",), ("a", "b", "c"), None])
    def test_every_item_must_be_a_pair(self, bad: Any) -> None:
        """形状不对的入参在**任何模型调用之前**报错（半路才炸会花掉前几次调用）."""
        llm = ExplodingLLM()

        with pytest.raises(GenerationError) as excinfo:
            make_generator(llm).generate_many([("好问题？", packed_context()), bad])

        assert "不是 (问题, 上下文) 二元组" in str(excinfo.value)

    def test_a_bad_question_is_caught_before_any_call(self) -> None:
        """第 2 条的问题不合法时，前 1 次调用也不该发生."""
        llm = ExplodingLLM()
        items = [("好问题？", packed_context()), (None, packed_context())]

        with pytest.raises(GenerationError) as excinfo:
            make_generator(llm).generate_many(items)

        assert "question 必须是字符串" in str(excinfo.value)
        assert llm.calls == 0

    def test_a_bad_context_is_caught_before_any_call(self) -> None:
        """上下文的形状同样在任何调用之前校验."""
        llm = ExplodingLLM()

        with pytest.raises(GenerationError) as excinfo:
            make_generator(llm).generate_many(
                [("好问题？", packed_context()), ("另一问？", "不是上下文")]
            )

        assert "context 必须是 PackedContext" in str(excinfo.value)
        assert llm.calls == 0

    def test_an_empty_question_is_rejected(self) -> None:
        """空白问题会让提示词里的"## 问题"一节变成空的（护栏在调用之前）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generator().generate_many([("   ", packed_context())])

        assert "空问题在生成这一层没有定义" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# 逐次覆盖（overrides）
# --------------------------------------------------------------------------- #


class TestGenerationOverrides:
    """三个键各自生效；清单外的键当场报错；覆盖生效必须进 notes."""

    def test_temperature_override_reaches_the_model(self) -> None:
        """覆盖必须**真的**传给模型（不是记在返回值里）."""
        llm = RecordingLLM(["答案 [1]"])

        make_generator(llm, temperature=0.2).generate(
            QUESTION, packed_context(), temperature=0.7
        )

        assert llm.calls[0]["temperature"] == 0.7

    def test_the_constructor_temperature_is_used_without_an_override(self) -> None:
        """不覆盖时用的是构造期那一个数."""
        llm = RecordingLLM(["答案 [1]"])

        make_generator(llm, temperature=0.2).generate(QUESTION, packed_context())

        assert llm.calls[0]["temperature"] == 0.2

    def test_max_tokens_override_reaches_the_model(self) -> None:
        """``max_tokens`` 同样逐次可覆盖（调小它会让答案变短，不是"看起来变短"）."""
        llm = RecordingLLM(["答案 [1]"])

        make_generator(llm, max_tokens=1024).generate(
            QUESTION, packed_context(), max_tokens=64
        )

        assert llm.calls[0]["max_tokens"] == 64

    @pytest.mark.parametrize("key", ["temperature", "max_tokens", "prompt_version"])
    def test_none_means_do_not_override(self, key: str) -> None:
        """``None`` 的语义统一：这一次不覆盖，用构造期的值（与构造参数一致）.

        渲染后的长度也能逐项加出来：v1 模板 238 字 - 占位符 19 字
        + 上下文 295 字 + 问题 15 字 = 529 字。
        """
        generation = make_generator(
            temperature=0.3, max_tokens=128, prompt_version=PROMPT_VERSION_V1
        ).generate(QUESTION, packed_context(), **{key: None})

        assert generation.prompt_version == PROMPT_VERSION_V1
        assert generation.to_dict()["prompt_chars"] == 238 - 19 + 295 + 15 == 529

    def test_a_temperature_override_is_clamped_by_the_same_validator(self) -> None:
        """覆盖路径与构造期共用同一套校验（越界一样当场报错）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generator().generate(QUESTION, packed_context(), temperature=2.5)

        assert "temperature=2.5 超出 [0, 2]" in str(excinfo.value)

    def test_a_max_tokens_override_must_be_positive(self) -> None:
        """0 不是"不限"（模型没有空间说话的表现是一次空回复）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generator().generate(QUESTION, packed_context(), max_tokens=0)

        assert "max_tokens=0" in str(excinfo.value)

    @pytest.mark.parametrize("key", ["top_k", "max_token", "promt_version", "require_citation"])
    def test_an_unknown_override_key_is_rejected(self, key: str) -> None:
        """清单外的键要报错并**列出合法取值**（静默忽略会让调用方以为它生效了）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generator().generate(QUESTION, packed_context(), **{key: "x"})

        message = str(excinfo.value)
        assert key in message
        assert "可用的键只有 ['temperature', 'max_tokens', 'prompt_version']" in message
        assert "静默忽略一个覆盖参数会让调用方以为它生效了" in message

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("top_k", 3),
            ("max_token", 64),
            ("temperature ", 0.5),
            ("PROMPT_VERSION", "v1"),
            ("prompt", "【资料】{context}【问】{question}"),
        ],
    )
    def test_the_rejected_key_never_reaches_the_model(self, key: str, value: Any) -> None:
        """拼错 / 大小写 / 尾随空格 / 想直接传模板：五种写法都在调用之前被拒.

        最后一条尤其重要：**请求体里不许传提示词文本**——提示词是"这一次模型
        到底被怎么问的"的唯一证据，能改它就等于版本号不再指着确定的模板。
        """
        llm = ExplodingLLM()

        with pytest.raises(GenerationError) as excinfo:
            make_generator(llm).generate(QUESTION, packed_context(), **{key: value})

        assert key in str(excinfo.value)
        assert llm.calls == 0

    def test_the_unknown_key_is_caught_before_any_llm_call(self) -> None:
        """覆盖参数的校验在**调用之前**（错误不该花掉一次模型调用）."""
        llm = ExplodingLLM()

        with pytest.raises(GenerationError):
            make_generator(llm).generate(QUESTION, packed_context(), top_k=3)

        assert llm.calls == 0

    def test_a_prompt_version_override_is_recorded_in_the_notes(self) -> None:
        """覆盖生效必须进 notes（否则调用方会以为"我指定了 v1"却没生效）."""
        generation = make_generator().generate(
            QUESTION, packed_context(), prompt_version=PROMPT_VERSION_V1
        )

        assert generation.prompt_version == PROMPT_VERSION_V1
        assert generation.notes[0].startswith("本次覆盖了提示词版本：v2 → v1。")
        assert "按版本分组时请以结果里的 Generation.prompt_version 为准" in generation.notes[0]

    def test_an_illegal_override_version_is_rejected(self) -> None:
        """清单外的版本号在覆盖路径上同样报错（校验只有一份实现）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generator().generate(QUESTION, packed_context(), prompt_version="v9")

        assert "合法取值是 ['v1', 'v2']" in str(excinfo.value)

    def test_temperature_and_max_tokens_overrides_do_not_add_notes(self) -> None:
        """只有"改模板"这类会改变答案文本的覆盖才进 notes（改采样参数不必念一遍）."""
        generation = make_generator().generate(
            QUESTION, packed_context(), temperature=0.4, max_tokens=256
        )

        assert generation.notes == ()

    @pytest.mark.parametrize("version", [PROMPT_VERSION_V1, PROMPT_VERSION_V2])
    def test_a_custom_template_ignores_the_version_override(self, version: str) -> None:
        """注入自定义模板时，逐次覆盖版本号**换不了模板**：它无法生效，且必须说出来."""
        generator = RAGGenerator(
            AnswerLLM(ANSWER_TWO), prompt="【资料】{context}【问】{question}"
        )

        generation = generator.generate(QUESTION, packed_context(), prompt_version=version)

        assert generation.prompt_version == "custom"
        assert generation.notes[0].startswith(
            f"本次把 prompt_version 覆盖成 {version!r}，但本生成器注入的是自定义模板"
        )
        assert "覆盖**无法生效**" in generation.notes[0]

    @pytest.mark.parametrize("version", ["v3", "", 123])
    def test_a_custom_template_does_not_check_the_version_override(
        self, version: Any
    ) -> None:
        """**已记录的源码缺口**（见测试报告的"发现的缺陷"一节）.

        构造期对"被忽略的版本号"是**过清单校验**的（``__init__`` 的注释写明了
        理由："拼错的版本号被记成'已忽略'会让调用方以为自己指定对了"），
        而逐次覆盖走的是另一条分支：自定义模板下这个版本号既不生效也不校验，
        ``prompt_version=123`` 这种形状同样被写进 notes 当作"覆盖过了"。
        这条断言把现状钉住：它一旦开始报错，说明缺口被补上，届时应改成反向断言。
        """
        generator = RAGGenerator(
            AnswerLLM(ANSWER_TWO), prompt="【资料】{context}【问】{question}"
        )

        generation = generator.generate(QUESTION, packed_context(), prompt_version=version)

        assert generation.prompt_version == "custom"
        assert f"本次把 prompt_version 覆盖成 {version!r}" in generation.notes[0]

    def test_both_prompt_and_version_at_construction_are_recorded(self) -> None:
        """构造期同时给两者时以 prompt 为准，且那句"被忽略的是版本号"要进 notes."""
        generator = RAGGenerator(
            AnswerLLM(ANSWER_TWO),
            prompt="【资料】{context}【问】{question}",
            prompt_version=PROMPT_VERSION_V1,
        )

        generation = generator.generate(QUESTION, packed_context())

        assert generation.prompt_version == "custom"
        assert generation.notes == (
            "构造期同时给了 prompt 与 prompt_version='v1'：**以 prompt 为准**，"
            "版本号记为 'custom'——被忽略的是那个版本号，不是那份模板。",
        )

    def test_the_custom_template_is_the_one_that_was_used(self) -> None:
        """自定义模板渲染出来的提示词要以它为骨架（版本号是标签，内容才算数）."""
        llm = AnswerLLM(ANSWER_TWO)
        generator = RAGGenerator(llm, prompt="【资料】{context}【问】{question}")

        generation = generator.generate(QUESTION, packed_context())

        assert generation.prompt_text.startswith("【资料】[1] ")
        assert generation.prompt_text.endswith(QUESTION)
        assert prompt_of(llm) == generation.prompt_text

    def test_all_three_keys_can_be_overridden_together(self) -> None:
        """三个键同时覆盖时互不干扰（版本号进 notes，采样值进模型入参）."""
        llm = RecordingLLM(["答案 [1]"])

        generation = make_generator(llm).generate(
            QUESTION,
            packed_context(),
            temperature=0.1,
            max_tokens=32,
            prompt_version=PROMPT_VERSION_V1,
        )

        assert llm.calls[0]["temperature"] == 0.1
        assert llm.calls[0]["max_tokens"] == 32
        assert generation.prompt_version == PROMPT_VERSION_V1
        assert len(generation.notes) == 1

    def test_the_override_keys_are_the_documented_three(self) -> None:
        """封闭清单的内容（与顺序）是接口的一部分."""
        assert GENERATION_OVERRIDE_KEYS == ("temperature", "max_tokens", "prompt_version")


# --------------------------------------------------------------------------- #
# 第 3 步：提示词渲染
# --------------------------------------------------------------------------- #


class TestPromptRendering:
    """渲染的结果**原样**存进结果："模型看到了什么"要能被读出来."""

    def test_the_default_version_is_v2(self) -> None:
        """不传版本号时读 settings（项目默认只有一处定义）."""
        generation = make_generator().generate(QUESTION, packed_context())

        assert generation.prompt_version == PROMPT_VERSION_V2
        assert generation.prompt_version == CURRENT_PROMPT_VERSION
        assert settings.retrieval_prompt_version == "v2"

    @pytest.mark.parametrize(
        "fragment", ["## 资料片段", "## 问题", "## 约束", "## 输出格式"]
    )
    def test_the_default_prompt_is_the_v2_text(self, fragment: str) -> None:
        """v2 的四段式标题都在提示词里（默认用的是那一段正文）."""
        generation = make_generator().generate(QUESTION, packed_context())

        assert fragment in generation.prompt_text

    def test_the_v2_only_sentence_is_present_by_default(self) -> None:
        """v2 独有的固定拒答句式要求也在场（v1 里没有这一句）."""
        generation = make_generator().generate(QUESTION, packed_context())

        assert "必须以固定句式开头" in generation.prompt_text
        assert "回答要求" not in generation.prompt_text

    def test_the_v1_override_swaps_the_template(self) -> None:
        """覆盖成 v1 时：v1 的特征串在场，v2 的特征串**不在场**."""
        generation = make_generator().generate(
            QUESTION, packed_context(), prompt_version=PROMPT_VERSION_V1
        )

        assert "回答要求" in generation.prompt_text
        assert "## 约束" not in generation.prompt_text
        assert "## 输出格式" not in generation.prompt_text
        assert "必须以固定句式开头" not in generation.prompt_text

    def test_a_v1_generator_renders_v1_by_default(self) -> None:
        """构造期指定版本的效果与逐次覆盖一致（两处走同一个取值入口）."""
        generation = make_generator(prompt_version=PROMPT_VERSION_V1).generate(
            QUESTION, packed_context()
        )

        assert generation.prompt_version == "v1"
        assert "回答要求" in generation.prompt_text
        assert "## 约束" not in generation.prompt_text

    @pytest.mark.parametrize(
        ("version", "present", "absent"),
        [
            ("v2", "## 资料片段", "回答要求："),
            ("v2", "## 问题", "资料片段（每条的编号写在方括号里）："),
            ("v2", "## 约束", "资料片段（每条的编号写在方括号里）"),
            ("v2", "## 输出格式", "回答要求："),
            ("v1", "回答要求：", "## 资料片段"),
            ("v1", "1. 只使用上面资料片段", "## 约束"),
            ("v1", "资料片段（每条的编号写在方括号里）", "## 输出格式"),
            ("v1", "问题：", "## 问题"),
        ],
    )
    def test_the_two_versions_use_different_section_labels(
        self, version: str, present: str, absent: str
    ) -> None:
        """两版模板的分节写法不同：v1 用行内标题，v2 用 ``##`` 小节.

        这八组是"逐次换版本**真的**换了模板"的锚点：只看版本号的话，
        一个把 v1 与 v2 指向同一份正文的实现也能全部通过。
        """
        generation = make_generator(prompt_version=version).generate(
            QUESTION, packed_context()
        )

        assert present in generation.prompt_text
        assert absent not in generation.prompt_text

    def test_the_prompt_carries_the_context_text_verbatim(self) -> None:
        """提示词里嵌入的必须是**打包出来的那段文本**（不是重新拼一遍）."""
        context = packed_context()
        generation = make_generator().generate(QUESTION, context)

        assert context.text in generation.prompt_text
        assert generation.prompt_text.count("[1] ") == 1

    def test_the_prompt_carries_the_numbered_blocks(self) -> None:
        """块头是 ``[n] 来源 › 标题路径``（编号已就位，答案里的 [n] 才指得到）."""
        generation = make_generator().generate(QUESTION, packed_context())

        assert "[1] docs/检索手册.md" in generation.prompt_text
        assert "[5] docs/附录.md" in generation.prompt_text

    def test_the_prompt_carries_the_question_verbatim(self) -> None:
        """问题原文必须逐字出现（编号之外的另一半锚点）."""
        generation = make_generator().generate("这个问题的原文？", packed_context())

        assert "这个问题的原文？" in generation.prompt_text

    def test_the_rendered_prompt_has_no_placeholders_left(self) -> None:
        """渲染之后不该再留下 ``{context}`` / ``{question}``（缺占位符在构造期已拦）."""
        generation = make_generator().generate(QUESTION, packed_context())

        assert "{context}" not in generation.prompt_text
        assert "{question}" not in generation.prompt_text

    def test_the_prompt_length_is_the_v2_length_plus_the_context_and_the_question(self) -> None:
        """提示词长度可以逐项加出来：模板 500 字 + 上下文 295 字 + 问题 15 字 - 占位符 19 字.

        ```text
        v2 模板 500 字，其中 {context} 占 9 字、{question} 占 10 字
        渲染后 = 500 - 9 - 10 + len(context.text) + len(question) = 791
        ```
        """
        context = packed_context()

        generation = make_generator().generate(QUESTION, context)

        assert len(RAG_PROMPT_V2) == 500
        assert len(context.text) == 295
        assert len(QUESTION) == 15
        assert len(generation.prompt_text) == 791
        assert len(generation.prompt_text) == 500 - 9 - 10 + 295 + 15
        assert generation.to_dict()["prompt_chars"] == 791

    def test_the_prompt_is_a_single_user_message(self) -> None:
        """一次问答只发一条 user 消息（模板里没有 system 段）."""
        llm = AnswerLLM(ANSWER_TWO)

        make_generator(llm).generate(QUESTION, packed_context())

        assert len(llm.calls) == 1
        assert len(llm.calls[0]["messages"]) == 1
        assert llm.calls[0]["messages"][0].role == "user"
        assert llm.calls[0]["messages"][0].content == prompt_of(llm)

    def test_the_rendered_prompt_is_stored_not_rebuilt(self) -> None:
        """结果里的 ``prompt_text`` 与模型收到的那一份**逐字相同**."""
        llm = AnswerLLM(ANSWER_TWO)
        generation = make_generator(llm).generate(QUESTION, packed_context())

        assert generation.prompt_text == prompt_of(llm)

    def test_a_custom_template_renders_both_placeholders(self) -> None:
        """两个占位符都在的自定义模板照它的样子渲染."""
        generation = RAGGenerator(
            AnswerLLM(ANSWER_TWO), prompt="<资料>{context}</资料><问>{question}</问>"
        ).generate("自定模板问题？", packed_context(2))

        assert generation.prompt_text.startswith("<资料>[1] ")
        assert generation.prompt_text.endswith("</资料><问>自定模板问题？</问>")
        assert generation.prompt_version == "custom"


# --------------------------------------------------------------------------- #
# 链路集成：pipeline.RagPipeline
# --------------------------------------------------------------------------- #


class TestPipelineIntegration:
    """``RagPipeline`` 从 day069 起是装配线：生成那一组整体交给生成器."""

    def test_a_default_pipeline_runs_without_an_injected_generator(self) -> None:
        """不传 ``generator`` 时本类按参数自建一个（缺省路径必须是通的）."""
        pipeline = make_pipeline()
        answer = pipeline.answer(QUESTION)

        assert answer.llm_called is True
        assert answer.answer == ANSWER_TWO
        assert len(answer.citations) == 5

    def test_the_answer_backfills_the_check(self) -> None:
        """``RagAnswer.check`` 是那两个新字段之一：三组编号与覆盖率逐项回填."""
        answer = make_pipeline().answer(QUESTION)

        assert isinstance(answer.check, GroundingReport)
        assert answer.check.cited == (1, 2)
        assert answer.check.valid == (1, 2)
        assert answer.check.unused == (3, 4, 5)
        assert answer.check.coverage == 0.4
        assert answer.check.grounded is True
        assert answer.grounded is True

    def test_the_answer_backfills_the_fallback_reason(self) -> None:
        """另一新字段 ``fallback_reason``：正常路径是空串（与生成器同一本词汇表）."""
        answer = make_pipeline().answer(QUESTION)

        assert answer.fallback_reason == FALLBACK_REASON_NONE
        assert answer.to_dict()["fallback_reason"] == ""

    def test_the_gate_reason_is_backfilled_too(self) -> None:
        """闸门拦下时那个原因要一路传上来（报告按它分层统计不可交付的那一批）."""
        answer = make_pipeline(llm=AnswerLLM(ANSWER_NO_CITATION), require_citation=True).answer(
            QUESTION
        )

        assert answer.fallback_reason == FALLBACK_REASON_UNUSABLE_CITATIONS
        assert answer.answer == FALLBACK_NO_CONTEXT
        assert answer.check is not None and answer.check.grounded is False

    @pytest.mark.parametrize(
        ("answer_text", "expected"),
        [
            (ANSWER_TWO, ""),
            (ANSWER_OUT_OF_RANGE, ""),
            ("资料未提及该配置项。", FALLBACK_REASON_MODEL_DECLINED),
        ],
    )
    def test_the_pipeline_mirrors_every_reason_it_gets(
        self, answer_text: str, expected: str
    ) -> None:
        """生成器给什么原因，链路就报什么原因（不回落到一个笼统的值）."""
        answer = make_pipeline(llm=AnswerLLM(answer_text)).answer(QUESTION)

        assert answer.fallback_reason == expected

    def test_the_prompt_version_property_reads_the_generator(self) -> None:
        """从 day069 起它来自生成器（两处各存一份会分家）."""
        pipeline = make_pipeline(prompt_version=PROMPT_VERSION_V1)

        assert pipeline.prompt_version == PROMPT_VERSION_V1
        assert pipeline.prompt_version == pipeline.generator.prompt_version
        assert pipeline.describe()["prompt_version"] == PROMPT_VERSION_V1

    def test_describe_carries_the_generator_group(self) -> None:
        """``describe()`` 里新增 ``generator`` 那一组（端点直接返回它）."""
        described = make_pipeline().describe()

        assert set(described) == {
            "prompt_version",
            "max_context_chars",
            "per_hit_chars",
            "temperature",
            "retriever",
            "generator",
        }
        assert described["generator"]["prompt_versions"] == ["v1", "v2"]
        assert described["generator"]["llm"] == "AnswerLLM"
        assert described["generator"]["prompt_version"] == "v2"

    def test_the_empty_retrieval_path_still_calls_no_llm(self) -> None:
        """**硬护栏**：检索为空时生成器一次都没被调用（``ExplodingLLM.calls == 0``）."""
        llm = ExplodingLLM()

        answer = make_pipeline(llm=llm, store=empty_store()).answer(QUESTION)

        assert llm.calls == 0
        assert answer.llm_called is False

    def test_the_empty_retrieval_path_leaves_the_check_unset(self) -> None:
        """``check is None`` 的含义是"这次没走到生成那一步"（不是"核对没通过"）."""
        answer = make_pipeline(llm=ExplodingLLM(), store=empty_store()).answer(QUESTION)

        assert answer.check is None
        assert answer.grounded is False
        assert answer.fallback_reason == FALLBACK_REASON_NO_CONTEXT
        assert answer.citations == ()

    def test_the_empty_retrieval_answer_is_the_generators_fallback(self) -> None:
        """护栏那一条路上的兜底答复与生成器是**同一份定义**（一处定义、两处引用）."""
        pipeline = make_pipeline(
            llm=ExplodingLLM(), store=empty_store(), fallback_answer="没资料。"
        )

        answer = pipeline.answer(QUESTION)

        assert answer.answer == "没资料。"
        assert pipeline.generator.fallback_answer == "没资料。"

    def test_the_temperature_is_forwarded_from_the_constructor(self) -> None:
        """构造期的温度进了生成器（本层不预先解析它：校验只有一份实现）."""
        llm = RecordingLLM(["答案 [1]"])
        make_pipeline(llm=llm, temperature=0.4).answer(QUESTION)

        assert llm.calls[0]["temperature"] == 0.4

    def test_the_temperature_override_is_forwarded(self) -> None:
        """逐次覆盖的 ``temperature`` 由本层**转发**给生成器."""
        llm = RecordingLLM(["答案 [1]"])
        make_pipeline(llm=llm, temperature=0.4).answer(QUESTION, temperature=0.9)

        assert llm.calls[0]["temperature"] == 0.9

    def test_the_default_temperature_is_zero(self) -> None:
        """默认温度 0.0：RAG 问答是有依据的复述（让它照抄依据）."""
        llm = RecordingLLM(["答案 [1]"])
        make_pipeline(llm=llm).answer(QUESTION)

        assert llm.calls[0]["temperature"] == 0.0

    def test_the_notes_are_ordered_like_the_chain(self) -> None:
        """注记顺序 = 链路顺序：先"这次打包削掉了什么"，再"这次回答与引用对得上吗"."""
        answer = make_pipeline(
            llm=AnswerLLM(ANSWER_TWO), per_hit_chars=50, require_citation=True
        ).answer(QUESTION)

        assert len(answer.notes) == 2
        assert "被单块上限 50 字截断" in answer.notes[0]
        assert "未引用的片段" in answer.notes[1]

    def test_the_to_dict_carries_both_new_fields_at_the_end(self) -> None:
        """两个新键加在**末尾**：键集合只增，老消费者读到的键一个都不少."""
        payload = make_pipeline().answer(QUESTION).to_dict()

        assert set(payload) == {
            "question",
            "answer",
            "llm_called",
            "prompt_version",
            "citations",
            "context",
            "retrieval",
            "notes",
            "check",
            "fallback_reason",
        }
        assert payload["check"]["given"] == 5
        assert payload["prompt_version"] == CURRENT_PROMPT_VERSION

    def test_the_summary_line_shows_the_coverage(self) -> None:
        """一行摘要里"接地 + 覆盖率"必须看得见（day069 把它压进了那一行）."""
        line = make_pipeline().answer(QUESTION).summary_line()

        assert "接地通过（覆盖 40.0%）" in line
        assert "引用 5 条" in line
        assert "LLM 已调用" in line

    def test_an_injected_generator_is_used_as_is(self) -> None:
        """注入生成器时整组生成参数由它说了算（本类不再逐项给）."""
        llm = AnswerLLM(ANSWER_ONE)
        generator = RAGGenerator(llm, prompt_version=PROMPT_VERSION_V1)

        pipeline = RagPipeline(
            Retriever(sample_store(), TableEmbedding()), llm, generator=generator
        )

        answer = pipeline.answer(QUESTION)

        assert answer.answer == ANSWER_ONE
        assert answer.check.coverage == 0.2
        assert pipeline.prompt_version == PROMPT_VERSION_V1

    def test_an_injected_generator_must_use_the_same_llm(self) -> None:
        """两个 LLM 会让"端点拦住的"与"真正发出去的"分家."""
        with pytest.raises(GenerationError) as excinfo:
            RagPipeline(
                Retriever(sample_store(), TableEmbedding()),
                AnswerLLM(ANSWER_TWO),
                generator=RAGGenerator(AnswerLLM(ANSWER_TWO)),
            )

        assert "不是同一个对象" in str(excinfo.value)

    def test_an_injected_generator_rejects_the_per_item_parameters(self) -> None:
        """两种装配方式只能选一种（同时给会让其中一边**看起来生效**）."""
        llm = AnswerLLM(ANSWER_TWO)

        with pytest.raises(GenerationError) as excinfo:
            RagPipeline(
                Retriever(sample_store(), TableEmbedding()),
                llm,
                generator=RAGGenerator(llm),
                temperature=0.2,
            )

        assert "不能再逐项给生成参数" in str(excinfo.value)
        assert "['temperature']" in str(excinfo.value)

    def test_the_pipeline_can_answer_a_query_object(self) -> None:
        """用 ``RetrievalQuery`` 提问时条数与引用都跟着它走（提示词也随之变短）."""
        answer = make_pipeline().answer(RetrievalQuery(text=QUESTION, top_k=2))

        assert answer.check is not None and answer.check.given == 2
        assert len(answer.citations) == 2
        assert answer.check.coverage == 1.0
        assert answer.check.unused == ()

    def test_the_pipeline_carries_the_retrieval_record(self) -> None:
        """检索的账（含索引状态）跟着回答一起走（本层不吞掉它）."""
        answer = make_pipeline().answer(QUESTION)

        assert answer.retrieval is not None
        assert answer.retrieval.ids() == list(CITATION_IDS)
        assert [citation.record_id for citation in answer.citations] == list(CITATION_IDS)

    def test_the_answer_many_path_uses_the_generator_per_item(self) -> None:
        """``answer_many`` 逐条调用：每条各自一次生成、各自一份核对."""
        llm = RecordingLLM(["甲 [1]", "乙 [1]"])
        answers = make_pipeline(llm=llm).answer_many([QUESTION, QUESTION])

        assert [item.answer for item in answers] == ["甲 [1]", "乙 [1]"]
        assert [item.check.coverage for item in answers] == [0.2, 0.2]
        assert len(llm.calls) == 2

    def test_a_mixed_batch_only_spends_one_call(self) -> None:
        """一批里有一条检索为空：它不消耗模型，其余照常."""
        llm = RecordingLLM(["有命中的那一次 [1]"])
        empty_retriever = Retriever(empty_store(), TableEmbedding())
        pipeline = RagPipeline(Retriever(sample_store(), TableEmbedding()), llm)
        empty_pipeline = RagPipeline(empty_retriever, llm)

        hit = pipeline.answer(QUESTION)
        miss = empty_pipeline.answer(QUESTION)

        assert len(llm.calls) == 1
        assert hit.fallback_reason == FALLBACK_REASON_NONE
        assert miss.fallback_reason == FALLBACK_REASON_NO_CONTEXT
        assert miss.check is None

    def test_the_answer_line_of_the_pipeline_records_the_citation_ids(self) -> None:
        """引用表的记录 id 与样本的手算表逐条一致（编号 → 记录 id）."""
        answer = make_pipeline().answer(QUESTION)

        mapping = {citation.marker: citation.record_id for citation in answer.citations}

        assert mapping == MARKER_TO_ID

    @pytest.mark.parametrize("version", [PROMPT_VERSION_V1, PROMPT_VERSION_V2])
    def test_the_pipeline_reports_the_version_it_actually_used(self, version: str) -> None:
        """报告里的版本号必须与生成器**实际渲染**的模板一致（三处读数同源）."""
        answer = make_pipeline(prompt_version=version).answer(QUESTION)
        generator = make_pipeline(prompt_version=version).generator

        assert generator.prompt_version == version
        assert generator.describe()["prompt_chars"] == (
            238 if version == PROMPT_VERSION_V1 else 500
        )
        assert answer.to_dict()["prompt_version"] == CURRENT_PROMPT_VERSION
        assert answer.check is not None

    @pytest.mark.parametrize("top_k", [1, 2, 3, 5])
    def test_the_coverage_denominator_follows_the_retrieved_clause_count(
        self, top_k: int
    ) -> None:
        """分母跟着**这次检索给了几条**走（答案是逐次生成的，不是缓存的）."""
        answer = make_pipeline().answer(RetrievalQuery(text=QUESTION, top_k=top_k))

        assert answer.check is not None
        assert answer.check.given == top_k
        assert len(answer.citations) == top_k
        expected = 1.0 if top_k <= 2 else round(2 / top_k, 4)
        assert answer.check.coverage == expected

    @pytest.mark.parametrize(
        ("answer_text", "coverage", "unused_ids", "grounded"),
        [
            (ANSWER_TWO, 0.4, [3, 4, 5], True),
            (ANSWER_ONE, 0.2, [1, 2, 4, 5], True),
            (ANSWER_ALL_FIVE, 1.0, [], True),
            (ANSWER_MIXED, 0.2, [2, 3, 4, 5], False),
        ],
    )
    def test_the_check_projection_carries_the_hand_checked_counts(
        self,
        answer_text: str,
        coverage: float,
        unused_ids: list[int],
        grounded: bool,
    ) -> None:
        """四个脚本的 ``check`` 投影与手算表逐位一致（它进公共字段，因此要准）."""
        payload = make_pipeline(llm=AnswerLLM(answer_text)).answer(QUESTION).to_dict()["check"]

        assert payload["given"] == 5
        assert payload["coverage"] == coverage
        assert payload["unused"] == unused_ids
        assert payload["grounded"] is grounded
        assert len(payload["unused"]) == len(unused_ids)
