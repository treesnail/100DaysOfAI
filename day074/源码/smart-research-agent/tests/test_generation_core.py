"""day069 ``retrieval.generation`` 的形状层与常量层单元测试（不含流水线集成）.

这一层被测的是"任何一个数字都能被纸笔验算"这件事的前提：**四份清单**、
**两版模板**与**三副形状**。因此断言分八组，每组都盯着"这一处被改坏之后，
报告里的哪句话会变成假的"：

```text
版本与清单   PROMPT_VERSIONS / PROMPT_CHANGELOG / RAG_PROMPTS 的键集合必须**逐项**
             相等（版本号一旦能被静默接受，'换一版提示词之后变好了吗'就无从分组）
回退族       FALLBACK_REASONS 的**顺序 = 判定优先级**，且需要与
             FALLBACK_REASON_DESCRIPTIONS 的键集合互相覆盖（含 NONE 这个哨兵值）
提示词文本   v1 与 pipeline 的转发名**逐字相同**；v2 比 v1 多出来的三件事
             （四段式 / 固定拒答句式 / 引用格式）各自有一条断言
模板取值     prompt_template_of 的两条合法路 + 一组非法版本名（消息里必须列出合法值）
CitationCheck 一条检查行的三条字段校验，以及"幻觉"与"未引用"是两件事的措辞
接地报告     三条不变量的正反例、coverage 的取值域与 round(..., 4)、
             given/hallucinated 两个**算出来的**数
一次生成     Generation 的每一条构造期校验（含与回退联动的两条纪律）、
             to_dict / to_summary / summary_line / explain 的键集合与行数
生成器       构造期的四类参数校验与 describe() 的键集合
```

**刻意不写"软断言"**（``is not None`` / 只判真假）：一条软断言在实现被改坏之后
仍然会通过，它证明不了任何事。所有断言都指向具体数值、具体次序或具体错误消息片段。

期望值来自 ``tests/generation_samples.py``（五条片段的编号表与六组核对输出都是手算常量）。

全部离线、确定性、零网络。
"""

from __future__ import annotations

from typing import Any

import pytest

from smart_research_agent.config import settings
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.retrieval import pipeline
from smart_research_agent.retrieval.context import Citation
from smart_research_agent.retrieval.errors import ContextError, GenerationError
from smart_research_agent.retrieval.generation import (
    CURRENT_PROMPT,
    CURRENT_PROMPT_VERSION,
    CUSTOM_PROMPT_VERSION,
    DECLINE_MARKERS,
    DEFAULT_PROMPT_VERSION,
    FALLBACK_NO_CONTEXT,
    FALLBACK_REASON_DESCRIPTIONS,
    FALLBACK_REASON_EMPTY_REPLY,
    FALLBACK_REASON_LLM_ERROR,
    FALLBACK_REASON_MODEL_DECLINED,
    FALLBACK_REASON_NO_CONTEXT,
    FALLBACK_REASON_NONE,
    FALLBACK_REASON_UNUSABLE_CITATIONS,
    FALLBACK_REASONS,
    GENERATION_OVERRIDE_KEYS,
    PROMPT_CHANGELOG,
    PROMPT_VERSION_V1,
    PROMPT_VERSION_V2,
    PROMPT_VERSIONS,
    RAG_PROMPT_V1,
    RAG_PROMPT_V2,
    RAG_PROMPTS,
    REQUIRED_PROMPT_FIELDS,
    CitationCheck,
    Generation,
    GroundingReport,
    RAGGenerator,
    prompt_template_of,
)
from tests.generation_samples import (
    ANSWER_ONE,
    ANSWER_TWO,
    CITATION_COUNT,
    CITATION_IDS,
    COVERAGE_ONE,
    COVERAGE_TWO,
    FULL_CITATION_IDS,
    GROUNDING_CASES,
    HAND_CHECKS_TWO,
    MARKER_TO_ID,
    QUESTION,
    ROUNDING_CASES,
    AnswerLLM,
    empty_packed_context,
    markers_of,
    packed_context,
    question,
    retrieval,
    sample_retriever,
)

# --------------------------------------------------------------------------- #
# 造样本的小工具（每个都只改一处，让"哪一支坏了"能一眼定位）
# --------------------------------------------------------------------------- #


def make_citation(marker: int, record_id: str | None = None) -> Citation:
    """造一条合法引用（``marker`` 与 ``record_id`` 之外的三项给固定值）."""
    resolved = record_id if record_id is not None else MARKER_TO_ID.get(marker, "c-x-99")
    return Citation(
        marker=marker,
        record_id=resolved,
        source="docs/检索手册.md",
        heading_path="检索手册 > 分册",
        score=1.0,
    )


def make_citations(count: int = 5) -> tuple[Citation, ...]:
    """造 ``count`` 条引用（编号 1..count，与 ``PackedContext`` 的形状同规）."""
    return tuple(make_citation(marker) for marker in range(1, count + 1))


def make_check(**overrides: Any) -> GroundingReport:
    """造一份合法报告（缺省：给了 5 条、引到 2 条、接地通过、带五行逐编号检查）.

    ``grounded`` 缺省按定义算出来（``valid 非空且 invalid 为空``），
    于是"三组编号与接地不一致"那条校验只有用例显式传它时才触发。
    """
    payload: dict[str, Any] = {
        "cited": (1, 2),
        "valid": (1, 2),
        "invalid": (),
        "unused": (3, 4, 5),
        "coverage": COVERAGE_TWO,
        "checks": make_checks(HAND_CHECKS_TWO),
    }
    payload.update(overrides)
    payload.setdefault("grounded", bool(payload["valid"]) and not payload["invalid"])
    return GroundingReport(**payload)


def make_no_context_generation(**overrides: Any) -> Generation:
    """造一次 ``no_context`` 的生成（三条"什么都没发生"的字段都在位）."""
    payload: dict[str, Any] = {
        "question": QUESTION,
        "answer": FALLBACK_NO_CONTEXT,
        "prompt_version": PROMPT_VERSION_V2,
        "prompt_text": "",
        "llm_called": False,
        "fallback_reason": FALLBACK_REASON_NO_CONTEXT,
        "citations": (),
        "check": GroundingReport(),
        "notes": (),
        "model": "",
        "latency_ms": 0.0,
    }
    payload.update(overrides)
    return Generation(**payload)


def make_generation(**overrides: Any) -> Generation:
    """造一次正常的生成（正常路径：答案来自模型且核对通过）."""
    payload: dict[str, Any] = {
        "question": QUESTION,
        "answer": ANSWER_TWO,
        "prompt_version": PROMPT_VERSION_V2,
        "prompt_text": "（提示词）",
        "llm_called": True,
        "fallback_reason": FALLBACK_REASON_NONE,
        "citations": make_citations(5),
        "check": make_check(),
        "notes": (),
        "model": "",
        "latency_ms": 0.0,
    }
    payload.update(overrides)
    return Generation(**payload)


def constraint_lines(prompt: str) -> list[str]:
    """模板里"约束"那一节的编号行（``1. ...`` 到 ``6. ...``）.

    v1 没有小节标题（四段式是 v2 才有的），因此它按**整段文本**数编号行；
    v2 只数 ``## 约束`` 与下一个 ``##`` 之间的行——``## 输出格式`` 那一节
    也有编号行，把它一起数进来会让"约束扩到 6 条"这件事失去区分度。
    """
    lines = prompt.splitlines()
    if "## 约束" in lines:
        start = lines.index("## 约束") + 1
        end = next(
            (index for index in range(start, len(lines)) if lines[index].startswith("## ")),
            len(lines),
        )
        lines = lines[start:end]
    return [line for line in lines if line[:2] in {f"{n}." for n in range(1, 10)}]


def make_checks(rows: tuple[tuple[int, str | None, bool], ...]) -> tuple[CitationCheck, ...]:
    """把样本里的手算检查行折成 ``CitationCheck``（越界那行的 ``record_id`` 是 None）."""
    return tuple(
        CitationCheck(marker=marker, record_id=record_id, used=used)
        for marker, record_id, used in rows
    )


# --------------------------------------------------------------------------- #
# 四份清单
# --------------------------------------------------------------------------- #


class TestPromptVersionConstants:
    """版本号是报告与评估的分组键：它必须是**封闭清单**，且当前版本指向唯一."""

    def test_the_managed_versions_are_exactly_v1_and_v2(self) -> None:
        """受管版本只有两个，且顺序就是报告里的顺序（不是字母序、不是长度序）."""
        assert PROMPT_VERSIONS == ("v1", "v2")
        assert PROMPT_VERSION_V1 == "v1"
        assert PROMPT_VERSION_V2 == "v2"

    def test_the_prompt_table_keys_equal_the_managed_versions(self) -> None:
        """``RAG_PROMPTS`` 的键集合必须**恰好**等于受管清单（多一个就是清单外的模板）."""
        assert tuple(RAG_PROMPTS) == PROMPT_VERSIONS
        assert set(RAG_PROMPTS) == set(PROMPT_VERSIONS)

    @pytest.mark.parametrize("version", PROMPT_VERSIONS)
    def test_each_version_points_to_its_own_template(self, version: str) -> None:
        """版本 → 模板是一一对应：两张表指向同一个对象会让"换版本"变成空动作."""
        assert RAG_PROMPTS[version] is (RAG_PROMPT_V1 if version == "v1" else RAG_PROMPT_V2)

    def test_the_default_version_is_v2(self) -> None:
        """缺省版本是 v2（它与``CURRENT_PROMPT_VERSION``今天相同，但服务两个问题）."""
        assert DEFAULT_PROMPT_VERSION == "v2"
        assert DEFAULT_PROMPT_VERSION in PROMPT_VERSIONS
        assert settings.retrieval_prompt_version == "v2"

    def test_the_current_prompt_is_v2(self) -> None:
        """当前生效的模板指向 v2，而 v1 作为历史版本仍然在位（永不删）."""
        assert CURRENT_PROMPT is RAG_PROMPT_V2
        assert CURRENT_PROMPT_VERSION == PROMPT_VERSION_V2
        assert CURRENT_PROMPT is not RAG_PROMPT_V1

    def test_the_custom_version_is_legal_but_not_managed(self) -> None:
        """``custom`` 是合法取值，但**不在**受管清单里（它不是一版受管模板）."""
        assert CUSTOM_PROMPT_VERSION == "custom"
        assert CUSTOM_PROMPT_VERSION not in PROMPT_VERSIONS
        assert CUSTOM_PROMPT_VERSION not in RAG_PROMPTS

    def test_the_changelog_covers_every_managed_version(self) -> None:
        """变更原因与版本号必须一一对应：漏掉一版，下一版的人就看不到取舍."""
        assert set(PROMPT_CHANGELOG) == set(PROMPT_VERSIONS)
        assert tuple(PROMPT_CHANGELOG) == PROMPT_VERSIONS

    @pytest.mark.parametrize("version", PROMPT_VERSIONS)
    def test_every_changelog_entry_says_the_day_and_the_reason(self, version: str) -> None:
        """每条变更原因都要写出"哪一天改的 + 改了什么"（照 agent.prompts 的口吻）."""
        entry = PROMPT_CHANGELOG[version]

        assert entry.startswith("day0")
        assert "：" in entry
        assert len(entry) > 30

    def test_the_required_placeholders_are_context_and_question(self) -> None:
        """构造期校验的就是这两个名字（清单住在模板旁边：模板住哪，清单住哪）."""
        assert REQUIRED_PROMPT_FIELDS == ("context", "question")


class TestFallbackConstants:
    """六种回退是一个封闭清单 + 固定优先级：判定顺序即 ``FALLBACK_REASONS`` 的顺序."""

    def test_the_reasons_are_the_five_documented_ones_in_priority_order(self) -> None:
        """顺序 = 判定优先级（最上游的先判），少一个都会让某一条回退没有名字."""
        assert FALLBACK_REASONS == (
            "no_context",
            "llm_error",
            "empty_reply",
            "model_declined",
            "unusable_citations",
        )

    @pytest.mark.parametrize(
        ("constant", "value"),
        [
            (FALLBACK_REASON_NO_CONTEXT, "no_context"),
            (FALLBACK_REASON_LLM_ERROR, "llm_error"),
            (FALLBACK_REASON_EMPTY_REPLY, "empty_reply"),
            (FALLBACK_REASON_MODEL_DECLINED, "model_declined"),
            (FALLBACK_REASON_UNUSABLE_CITATIONS, "unusable_citations"),
        ],
    )
    def test_each_reason_constant_equals_its_literal(self, constant: str, value: str) -> None:
        """常量的字面值进报告、进评估分组：改一个字就是一次静默的接口变更."""
        assert constant == value

    def test_none_is_the_empty_string_sentinel(self) -> None:
        """NONE 是空串（与 ``types.EMPTY_REASON_NONE`` 同一套写法）：永远有值可比."""
        assert FALLBACK_REASON_NONE == ""
        assert FALLBACK_REASON_NONE not in FALLBACK_REASONS

    def test_descriptions_cover_every_value_including_none(self) -> None:
        """字典的键集合要覆盖**每一个**可能出现的取值（含"没有回退"那一档）."""
        assert set(FALLBACK_REASON_DESCRIPTIONS) == {FALLBACK_REASON_NONE, *FALLBACK_REASONS}
        assert len(FALLBACK_REASON_DESCRIPTIONS) == len(FALLBACK_REASONS) + 1

    @pytest.mark.parametrize(
        ("reason", "fragments"),
        [
            (FALLBACK_REASON_NONE, ("没有回退", "有效引用")),
            (FALLBACK_REASON_NO_CONTEXT, ("打包为空", "一次 LLM 都不调", "兜底")),
            (FALLBACK_REASON_LLM_ERROR, ("抛异常", "超时", "网络", "额度")),
            (FALLBACK_REASON_EMPTY_REPLY, ("空串", "max_tokens", "提供方")),
            (FALLBACK_REASON_MODEL_DECLINED, ("固定句式", "覆盖")),
            (FALLBACK_REASON_UNUSABLE_CITATIONS, ("require_citation", "换一版模板", "开关")),
        ],
    )
    def test_every_description_names_the_symptom_and_the_way_out(
        self, reason: str, fragments: tuple[str, ...]
    ) -> None:
        """每条解释都是"现象 + 出路"：只有现象的话，读的人不知道该动哪个参数."""
        description = FALLBACK_REASON_DESCRIPTIONS[reason]

        assert isinstance(description, str)
        for fragment in fragments:
            assert fragment in description

    def test_the_fallback_answer_offers_three_ways_out(self) -> None:
        """兜底答复要给下一步动作（换说法 / 放宽过滤 / 查索引版本），不是一句失败宣告."""
        assert "知识库中没有检索到" in FALLBACK_NO_CONTEXT
        assert "换一种说法" in FALLBACK_NO_CONTEXT
        assert "放宽过滤条件" in FALLBACK_NO_CONTEXT
        assert "索引版本" in FALLBACK_NO_CONTEXT

    def test_the_decline_markers_are_the_three_fixed_sentences(self) -> None:
        """检测清单与 v2 第 2 条约束要求的三句式必须是**同一份**（顺序也一致）."""
        assert DECLINE_MARKERS == ("资料中没有相关内容", "资料未提及", "无法依据资料回答")

    def test_the_override_keys_are_the_three_documented_ones(self) -> None:
        """封闭清单：多一个键就报错，因此它的内容（与顺序）是接口的一部分."""
        assert GENERATION_OVERRIDE_KEYS == ("temperature", "max_tokens", "prompt_version")


# --------------------------------------------------------------------------- #
# 两版模板
# --------------------------------------------------------------------------- #


class TestPromptTexts:
    """v1 → v2 改了三件事（四段式 / 固定拒答句式 / 引用格式），每件都要能被读到."""

    def test_v1_is_identical_to_the_pipeline_alias(self) -> None:
        """``pipeline.RAG_ANSWER_PROMPT_V1`` 是**转发**：两个名字必须逐字相同."""
        assert RAG_PROMPT_V1 == pipeline.RAG_ANSWER_PROMPT_V1
        assert RAG_PROMPT_V1 is pipeline.RAG_ANSWER_PROMPT_V1

    def test_the_pipeline_aliases_point_at_the_current_version(self) -> None:
        """另外三个历史名字从 day069 起指向当前版本（调用方零改动）."""
        assert pipeline.RAG_ANSWER_PROMPT is CURRENT_PROMPT
        assert pipeline.RAG_ANSWER_PROMPT_VERSION == CURRENT_PROMPT_VERSION
        assert pipeline.REQUIRED_PROMPT_FIELDS == REQUIRED_PROMPT_FIELDS

    def test_v2_is_a_different_text_than_v1(self) -> None:
        """换版本必须真的换了正文：两份模板相同的话，"v1 vs v2"的实验是假的."""
        assert RAG_PROMPT_V2 != RAG_PROMPT_V1
        assert len(RAG_PROMPT_V2) == 500
        assert len(RAG_PROMPT_V1) == 238

    @pytest.mark.parametrize(
        "section", ["## 资料片段", "## 问题", "## 约束", "## 输出格式"]
    )
    def test_v2_has_the_four_sections(self, section: str) -> None:
        """四段式：喂进去的东西与要它做的事各自有标题（v1 只能靠位置猜）."""
        assert section in RAG_PROMPT_V2

    def test_the_four_sections_appear_in_that_order(self) -> None:
        """顺序 = 模型阅读的顺序（资料 → 问题 → 约束 → 输出格式）."""
        positions = [
            RAG_PROMPT_V2.index(section)
            for section in ("## 资料片段", "## 问题", "## 约束", "## 输出格式")
        ]

        assert positions == sorted(positions)
        assert len(set(positions)) == 4

    @pytest.mark.parametrize("placeholder", ["{context}", "{question}"])
    @pytest.mark.parametrize("version", PROMPT_VERSIONS)
    def test_both_templates_carry_both_placeholders(
        self, version: str, placeholder: str
    ) -> None:
        """占位符是构造期校验的对象：两版模板都必须同时带上这两个."""
        assert placeholder in RAG_PROMPTS[version]

    @pytest.mark.parametrize("version", PROMPT_VERSIONS)
    def test_both_templates_carry_the_decline_sentence(self, version: str) -> None:
        """两版都要求"资料不足就说没有"（v2 的差别在于**句式固定**）."""
        assert "资料中没有相关内容" in RAG_PROMPTS[version]

    def test_v2_has_six_constraints_and_v1_has_four(self) -> None:
        """约束从 4 条扩到 6 条（新增的两条：固定拒答句式、引用格式）."""
        assert len(constraint_lines(RAG_PROMPT_V2)) == 6
        assert len(constraint_lines(RAG_PROMPT_V1)) == 4

    @pytest.mark.parametrize(
        "fragment",
        [
            "只使用上面资料片段中出现的信息",
            "必须以固定句式开头",
            "编号必须来自上面的片段",
            "每条结论后面至少跟一个引用",
            "数字、日期、代码与专有名词",
            "不要整段复述原文",
        ],
    )
    def test_every_v2_constraint_is_present(self, fragment: str) -> None:
        """六条约束逐条在位（每条都注明了它挡哪一种真实失败）."""
        assert fragment in RAG_PROMPT_V2

    @pytest.mark.parametrize("marker", DECLINE_MARKERS)
    def test_v2_asks_for_each_decline_marker(self, marker: str) -> None:
        """**要求清单与检测清单是同一份**：v2 里出现的三个句式必须逐个能被检出."""
        assert f'"{marker}"' in RAG_PROMPT_V2

    def test_v2_states_the_output_shape(self) -> None:
        """新增的输出格式一节：v1 只管内容该怎样，没管这份答案长什么样."""
        tail = RAG_PROMPT_V2[RAG_PROMPT_V2.index("## 输出格式") :]

        assert "第一行：一句话结论" in tail
        assert "依据 [n]：" in tail
        assert "资料不足时" in tail


class TestPromptTemplateOf:
    """按版本号取受管模板：清单外的取值当场报错，并把合法取值列在消息里."""

    @pytest.mark.parametrize(
        ("version", "expected"),
        [(PROMPT_VERSION_V1, RAG_PROMPT_V1), (PROMPT_VERSION_V2, RAG_PROMPT_V2)],
    )
    def test_each_managed_version_resolves_to_its_template(
        self, version: str, expected: str
    ) -> None:
        """两条合法路各自一路（返回的是同一个对象，不是一份拷贝）."""
        assert prompt_template_of(version) is expected

    @pytest.mark.parametrize(
        "version", ["v3", "v10", "V1", "V2", "", "  ", "v2 ", " v2", "latest", "custom", None, 2]
    )
    def test_an_unmanaged_version_is_rejected_with_the_legal_values(
        self, version: Any
    ) -> None:
        """非法版本号（拼错 / 大写 / 空串 / 尾随空格 / None）都要当场报错.

        消息里必须**列出合法取值**：只报"未知的版本"的话，调用方只能去翻代码。
        """
        with pytest.raises(GenerationError) as excinfo:
            prompt_template_of(version)

        message = str(excinfo.value)
        assert "受管版本只有 ['v1', 'v2']" in message
        assert "'custom'" in message
        assert "RAG_PROMPTS" in message

    @pytest.mark.parametrize("version", ["v3", "V2", "  "])
    def test_the_error_quotes_the_offending_value(self, version: str) -> None:
        """消息要点名那个非法值（否则一次批量调用里不知道是哪一条坏了）."""
        with pytest.raises(GenerationError) as excinfo:
            prompt_template_of(version)

        assert repr(version) in str(excinfo.value)

    def test_custom_is_not_reachable_through_this_entry(self) -> None:
        """``custom`` 是合法版本号但**不是**受管模板：这个入口取不到它."""
        with pytest.raises(GenerationError) as excinfo:
            prompt_template_of(CUSTOM_PROMPT_VERSION)

        assert "custom" in str(excinfo.value)
        assert "它不是一版受管模板" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# CitationCheck：一条检查行
# --------------------------------------------------------------------------- #


class TestCitationCheckShape:
    """一条检查行回答"这个编号到底有没有落在那次提示词里"，两条分支不能混."""

    @pytest.mark.parametrize(
        ("marker", "record_id", "used", "hallucinated"),
        [
            (1, "c-a-01", True, False),
            (2, "c-b-01", False, False),
            (5, "c-c-01", False, False),
            (9, None, True, True),
            (12, None, True, True),
        ],
    )
    def test_a_check_row_carries_the_three_fields(
        self, marker: int, record_id: str | None, used: bool, hallucinated: bool
    ) -> None:
        """三个字段原样带着走，幻觉与否**由 record_id 是否为空推出**（不另存一份）."""
        check = CitationCheck(marker=marker, record_id=record_id, used=used)

        assert check.marker == marker
        assert check.record_id == record_id
        assert check.used is used
        assert check.hallucinated is hallucinated

    def test_to_dict_has_exactly_the_four_keys(self) -> None:
        """投影的键集合是接口的一部分（多一个别名会让下游读错）."""
        payload = CitationCheck(marker=1, record_id="c-a-01", used=True).to_dict()

        assert set(payload) == {"marker", "record_id", "hallucinated", "used"}
        assert payload == {
            "marker": 1,
            "record_id": "c-a-01",
            "hallucinated": False,
            "used": True,
        }

    def test_an_hallucinated_row_projects_a_null_record_id(self) -> None:
        """越界那条的 ``record_id`` 必须是 JSON 的 null（不是空串）."""
        payload = CitationCheck(marker=9, record_id=None, used=True).to_dict()

        assert payload["record_id"] is None
        assert payload["hallucinated"] is True
        assert payload["used"] is True

    def test_summary_line_names_the_hallucination(self) -> None:
        """幻觉那条的摘要要点名它是幻觉（否则一行行看下去看不出差别）."""
        line = CitationCheck(marker=9, record_id=None, used=True).summary_line()

        assert line == "[9] **幻觉引用**：这个编号不在那次提示词里"

    @pytest.mark.parametrize(
        ("used", "state"),
        [(True, "已引用"), (False, "未引用")],
    )
    def test_summary_line_names_the_record_and_the_state(self, used: bool, state: str) -> None:
        """非幻觉两态各自一行：``[n] 记录 id 已引用 / 未引用``."""
        line = CitationCheck(marker=2, record_id="c-b-01", used=used).summary_line()

        assert line == f"[2] c-b-01 {state}"

    @pytest.mark.parametrize("marker", [0, -1, -9, True, 1.0, "1", None])
    def test_the_marker_must_be_an_integer_from_one(self, marker: Any) -> None:
        """0 / 负数 / 布尔 / 字符串都不是编号：它们不该出现在这张表里."""
        with pytest.raises(GenerationError) as excinfo:
            CitationCheck(marker=marker, record_id="c-a-01", used=True)

        assert "必须是从 1 起的整数" in str(excinfo.value)
        assert repr(marker) in str(excinfo.value)

    @pytest.mark.parametrize("record_id", ["", "   ", 7, 1.5, b"c-a-01"])
    def test_the_record_id_must_be_a_non_empty_string_or_none(self, record_id: Any) -> None:
        """空串顶替 None 会被读成"这条记录没有 id"——两件事的处置完全不同."""
        with pytest.raises(GenerationError) as excinfo:
            CitationCheck(marker=1, record_id=record_id, used=True)

        assert "必须是非空字符串或 None" in str(excinfo.value)

    @pytest.mark.parametrize("used", [1, 0, "yes", None, "False"])
    def test_used_must_be_a_real_boolean(self, used: Any) -> None:
        """非布尔的写法在 if 里恒为真（"未引用"这件事从来没有发生过）."""
        with pytest.raises(GenerationError) as excinfo:
            CitationCheck(marker=1, record_id="c-a-01", used=used)

        assert "必须是布尔值" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# GroundingReport：一份核对报告
# --------------------------------------------------------------------------- #


class TestGroundingReportShape:
    """三条不变量一旦破了，报告里的数字就会互相矛盾，而读的人只会相信最合理那个."""

    def test_the_defaults_are_all_empty(self) -> None:
        """什么都不给时是一份"没有东西可核对"的报告（合法状态，不是异常）."""
        report = GroundingReport()

        assert report.cited == ()
        assert report.valid == ()
        assert report.invalid == ()
        assert report.unused == ()
        assert report.coverage == 0.0
        assert report.grounded is False
        assert report.notes == ()
        assert report.checks == ()
        assert report.given == 0
        assert report.hallucinated == 0

    @pytest.mark.parametrize("case", GROUNDING_CASES, ids=[c.label for c in GROUNDING_CASES])
    def test_the_hand_table_is_constructible(self, case: Any) -> None:
        """样本里的六组输出全部能被构造出来（账自洽），且两个算出来的数正确.

        ``given`` 必须等于 ``len(valid) + len(unused)``（= 给出去的条数），
        ``hallucinated`` 必须等于 ``len(invalid)``——它们是**算出来的**，
        多存一份就多一处可能分家的副本。
        """
        report = GroundingReport(
            cited=case.cited,
            valid=case.valid,
            invalid=case.invalid,
            unused=case.unused,
            coverage=case.coverage,
            grounded=case.grounded,
        )

        assert report.cited == case.cited
        assert report.valid == case.valid
        assert report.invalid == case.invalid
        assert report.unused == case.unused
        assert report.coverage == case.coverage
        assert report.grounded is case.grounded
        assert report.given == 5
        assert report.hallucinated == len(case.invalid)

    @pytest.mark.parametrize(
        ("label", "value"),
        [
            ("cited", [1, 2]),
            ("valid", [1]),
            ("invalid", []),
            ("unused", [3]),
        ],
    )
    def test_the_four_marker_groups_must_be_tuples(self, label: str, value: Any) -> None:
        """frozen 形状里塞一个可变对象，"这份核对不曾被改过"就不再成立."""
        payload: dict[str, Any] = {"cited": (1, 2), "valid": (1, 2)}
        payload[label] = value

        with pytest.raises(GenerationError) as excinfo:
            GroundingReport(**payload)

        assert f"GroundingReport.{label} 必须是 tuple" in str(excinfo.value)

    @pytest.mark.parametrize("label", ["cited", "valid"])
    @pytest.mark.parametrize("value", [(0,), (-1,), (True,), ("1",), (1, 0)])
    def test_a_marker_must_be_an_integer_from_one(self, label: str, value: Any) -> None:
        """0 / 负数 / 布尔 / 字符串都说明这份报告是手工拼出来的."""
        payload: dict[str, Any] = {"cited": (1,), "valid": (1,), "grounded": True}
        payload[label] = value

        with pytest.raises(GenerationError) as excinfo:
            GroundingReport(**payload)

        assert f"GroundingReport.{label} 里出现了非法编号" in str(excinfo.value)

    @pytest.mark.parametrize("value", [(2, 1), (1, 1), (1, 3, 2), (3, 1, 1)])
    def test_the_markers_must_be_sorted_and_unique(self, value: tuple[int, ...]) -> None:
        """顺序或重复会让"哪一条没被引用"读起来要先做一次心算."""
        with pytest.raises(GenerationError) as excinfo:
            GroundingReport(cited=value, valid=value)

        assert "必须是**升序去重**的" in str(excinfo.value)

    def test_valid_and_invalid_cannot_intersect(self) -> None:
        """同一个编号不可能既在提示词里、又不在提示词里."""
        with pytest.raises(GenerationError) as excinfo:
            GroundingReport(cited=(1,), valid=(1,), invalid=(1,), grounded=False)

        assert "valid=[1] 与 invalid=[1] 有交集" in str(excinfo.value)

    def test_the_union_of_valid_and_invalid_must_equal_cited(self) -> None:
        """漏掉的那个编号会让"有没有幻觉引用"这个问题答不出来."""
        with pytest.raises(GenerationError) as excinfo:
            GroundingReport(cited=(1, 2), valid=(1,), invalid=())

        assert "valid ∪ invalid = [1]，而 cited = [1, 2]" in str(excinfo.value)

    def test_unused_and_valid_cannot_intersect(self) -> None:
        """'给了它、模型没引用'与'模型引用了它'不可能同时成立."""
        with pytest.raises(GenerationError) as excinfo:
            GroundingReport(cited=(1,), valid=(1,), unused=(1,), coverage=0.5)

        assert "unused=[1] 与 valid=[1] 有交集" in str(excinfo.value)

    @pytest.mark.parametrize("value", ["0.5", True, None, [0.5]])
    def test_coverage_must_be_a_number(self, value: Any) -> None:
        """字符串会让报告里的百分号变成一句乱码；bool 是 int 的子类，更要单独拦."""
        with pytest.raises(GenerationError) as excinfo:
            GroundingReport(coverage=value)

        assert "coverage 必须是数字" in str(excinfo.value)

    @pytest.mark.parametrize("value", [1.5, -0.1, 2.0, float("nan"), float("inf")])
    def test_coverage_must_stay_inside_the_unit_interval(self, value: float) -> None:
        """它是"有效引用 / 给出去的片段数"：越界说明分母算错了."""
        with pytest.raises(GenerationError) as excinfo:
            GroundingReport(coverage=value)

        assert "必须落在 [0, 1]" in str(excinfo.value)

    @pytest.mark.parametrize(
        "value",
        [0, 1, 0.0, 1.0, 0.25, 0.5, 0.75],
    )
    def test_coverage_bounds_are_inclusive(self, value: float) -> None:
        """两端都是闭区间：切到 0 与切到 1 都是合法的比例."""
        assert GroundingReport(coverage=value).coverage == value

    @pytest.mark.parametrize(
        ("valid", "invalid", "declared"),
        [((1,), (), False), ((), (9,), True), ((1,), (9,), True)],
    )
    def test_grounded_must_match_the_three_marker_groups(
        self, valid: tuple[int, ...], invalid: tuple[int, ...], declared: bool
    ) -> None:
        """接地判定**不看字符串**：它与三组编号必须严格一致（两个方向都拦）."""
        cited = tuple(sorted(set(valid) | set(invalid)))

        with pytest.raises(GenerationError) as excinfo:
            GroundingReport(cited=cited, valid=valid, invalid=invalid, grounded=declared)

        assert "与三组编号不一致" in str(excinfo.value)
        assert f"按定义应当是 {not declared}" in str(excinfo.value)

    @pytest.mark.parametrize("note", ["", "   "])
    def test_notes_cannot_contain_blank_entries(self, note: str) -> None:
        """注记是给人读的，空串只会让报告里多一行空白."""
        with pytest.raises(GenerationError) as excinfo:
            GroundingReport(notes=(note,))

        assert "notes 里出现了空条目" in str(excinfo.value)

    @pytest.mark.parametrize("note", [1, None, b"x"])
    def test_a_note_must_be_a_string(self, note: Any) -> None:
        """注记的**条目**必须是字符串（非字符串会把一行日志变成一段 repr）."""
        with pytest.raises(GenerationError) as excinfo:
            GroundingReport(notes=(note,))

        assert "notes 里出现了空条目" in str(excinfo.value)

    def test_the_notes_container_is_not_type_checked(self) -> None:
        """**已记录的源码缺口**（见测试报告的"发现的缺陷"一节）.

        四条编号组走 ``_validate_marker_tuple``（"必须是 tuple"），而 ``notes``
        只逐条校验**元素**——传一个 list 会被接受，于是 frozen 形状里就躺着一个
        可变对象（``Generation.notes`` 反倒是拦的）。这条断言把现状钉住：
        它一旦变成报错，说明缺口被补上了，届时这条用例应当改成反向断言。
        """
        report = GroundingReport(notes=["一条注记"])

        assert report.notes == ["一条注记"]
        assert report.to_dict()["notes"] == ["一条注记"]

    @pytest.mark.parametrize("check", [1, "x", None, (1,)])
    def test_the_checks_must_be_citation_checks(self, check: Any) -> None:
        """这张表只收 ``CitationCheck``（编号 / 记录 / 有没有被引用）."""
        with pytest.raises(GenerationError) as excinfo:
            GroundingReport(checks=(check,))

        assert "只收 CitationCheck" in str(excinfo.value)

    def test_to_dict_has_exactly_the_documented_keys(self) -> None:
        """投影的键集合全等：``given`` 在、三组编号在、``checks`` 逐行展开."""
        payload = make_check().to_dict()

        assert set(payload) == {
            "given",
            "cited",
            "valid",
            "invalid",
            "unused",
            "coverage",
            "grounded",
            "notes",
            "checks",
        }
        assert payload["given"] == 5
        assert payload["cited"] == [1, 2]
        assert payload["valid"] == [1, 2]
        assert payload["invalid"] == []
        assert payload["unused"] == [3, 4, 5]
        assert payload["coverage"] == 0.4
        assert payload["grounded"] is True
        assert payload["notes"] == []
        assert len(payload["checks"]) == 5

    def test_to_dict_projects_every_by_number_check_row(self) -> None:
        """逐编号的检查行是核对最细的一层：``checks`` 的长度 = 给出去的片段数 + 越界编号数."""
        report = GroundingReport(
            cited=(1, 9),
            valid=(1,),
            invalid=(9,),
            unused=(2, 3, 4, 5),
            coverage=0.2,
            grounded=False,
            checks=(
                CitationCheck(marker=1, record_id="c-a-01", used=True),
                CitationCheck(marker=9, record_id=None, used=True),
            ),
        )
        payload = report.to_dict()

        assert len(payload["checks"]) == 2
        assert payload["checks"][0] == {
            "marker": 1,
            "record_id": "c-a-01",
            "hallucinated": False,
            "used": True,
        }
        assert payload["checks"][1]["record_id"] is None

    @pytest.mark.parametrize(("count", "used"), [(1, 1), (2, 1), (3, 2), (5, 5), (8, 3)])
    def test_given_is_the_sum_of_valid_and_unused(self, count: int, used: int) -> None:
        """``given`` 是**算出来的**（valid + unused 恰好把 1..n 分完）."""
        valid = tuple(range(1, used + 1))
        report = GroundingReport(
            cited=valid,
            valid=valid,
            unused=tuple(range(used + 1, count + 1)),
            coverage=round(used / count, 4),
            grounded=True,
        )

        assert report.given == count
        assert report.given == len(report.valid) + len(report.unused)

    @pytest.mark.parametrize(("count", "used", "expected"), ROUNDING_CASES)
    def test_coverage_is_rounded_to_four_decimals(
        self, count: int, used: int, expected: float
    ) -> None:
        """覆盖率按 4 位小数四舍五入（照 ``PackedContext.fill_ratio`` 的口径）."""
        valid = tuple(range(1, used + 1))
        raw = used / count
        report = GroundingReport(
            cited=valid,
            valid=valid,
            unused=tuple(range(used + 1, count + 1)),
            coverage=raw,
            grounded=True,
        )

        assert report.coverage == expected
        assert report.to_dict()["coverage"] == expected
        assert report.coverage == round(raw, 4)

    def test_the_summary_line_is_the_hand_checked_sentence(self) -> None:
        """一行摘要里的每个数字都要能被手算（给了 5 / 引用 2 / 有效 2 / 未引用 3 / 40.0%）."""
        line = make_check().summary_line()

        assert line == (
            "核对：给了 5 条片段 | 答案引用 2 条（有效 2 / 幻觉 0）"
            " | 未引用 3 条 | 覆盖 40.0% | 接地通过"
        )

    def test_the_summary_line_says_not_passed_when_hallucinated(self) -> None:
        """有幻觉引用时摘要必须写"未通过"（覆盖率与接地是两个数，别混着读）."""
        report = make_check(
            cited=(1, 9), valid=(1,), invalid=(9,), unused=(2, 3, 4, 5), coverage=0.2
        )

        assert (
            report.summary_line()
            == "核对：给了 5 条片段 | 答案引用 2 条（有效 1 / 幻觉 1）"
            " | 未引用 4 条 | 覆盖 20.0% | 接地未通过"
        )

    def test_explain_renders_the_two_mandatory_lines_for_an_empty_report(self) -> None:
        """没有幻觉、没有未引用、没有注记时只有两句：口径与覆盖率的定义."""
        lines = GroundingReport().explain()

        assert len(lines) == 2
        assert lines[0].startswith("核对口径：") and "给了 0 条片段" in lines[0]
        assert lines[1].startswith("覆盖率 0.0%") and "接地未通过" in lines[1]

    def test_explain_adds_a_line_for_each_hallucinated_number(self) -> None:
        """幻觉那一段只在真有幻觉渲染，且**点名编号**（不是一句"存在幻觉"）."""
        lines = make_check(
            cited=(1, 9), valid=(1,), invalid=(9,), unused=(2, 3, 4, 5), coverage=0.2
        ).explain()

        assert len(lines) == 4
        assert lines[2] == (
            "幻觉引用：编号 [9] 不在那次提示词里"
            "—— 它不是'多写了一个数字'，而是**指到了一份并不存在的依据**"
        )
        assert "指到了一份并不存在的依据" in lines[2]

    def test_explain_counts_the_unused_clauses(self) -> None:
        """未引用那一段要写出**条数**（它是覆盖率的分母）——没有幻觉时只有三行."""
        lines = make_check().explain()
        tail = lines[2]

        assert len(lines) == 3
        assert tail.startswith("未引用的片段：编号 [3, 4, 5]（共 3 条）")
        assert "是覆盖率的分母" in tail

    @pytest.mark.parametrize("count", [1, 2, 3])
    def test_explain_appends_one_line_per_note(self, count: int) -> None:
        """注记有几条就多几行（"注记多了会被读的人忽略"是这一层最该守的纪律）."""
        notes = tuple(f"注记 {index}" for index in range(count))

        assert len(GroundingReport(notes=notes).explain()) == 2 + count

    def test_explain_does_not_render_the_hallucination_section_when_clean(self) -> None:
        """没有幻觉就不占一行（按情况渲染，而不是永远印一遍）."""
        lines = GroundingReport(cited=(1,), valid=(1,), coverage=1.0, grounded=True).explain()

        assert len(lines) == 2
        assert all("幻觉引用：" not in line for line in lines)
        assert all("未引用的片段" not in line for line in lines)


# --------------------------------------------------------------------------- #
# Generation：一次生成
# --------------------------------------------------------------------------- #


class TestGenerationShape:
    """一次生成把"答案 / 怎么问的 / 核对结果"钉在一起，还带两条与回退联动的纪律."""

    def test_a_normal_generation_reports_both_derived_flags(self) -> None:
        """``is_fallback`` 与 ``grounded`` 都是转发出来的（不另存字段）."""
        generation = make_generation()

        assert generation.is_fallback is False
        assert generation.grounded is True
        assert generation.llm_called is True
        assert generation.citations == make_citations(5)

    @pytest.mark.parametrize("reason", FALLBACK_REASONS)
    def test_every_fallback_reason_marks_the_generation_as_fallback(self, reason: str) -> None:
        """五个回退原因都是"不算数"（只有空串那个哨兵不是）."""
        generation = make_generation(
            fallback_reason=reason,
            llm_called=reason != FALLBACK_REASON_NO_CONTEXT,
            prompt_text="" if reason == FALLBACK_REASON_NO_CONTEXT else "P",
            citations=() if reason == FALLBACK_REASON_NO_CONTEXT else make_citations(5),
            check=GroundingReport() if reason == FALLBACK_REASON_NO_CONTEXT else make_check(),
        )

        assert generation.is_fallback is True
        assert generation.fallback_reason == reason

    @pytest.mark.parametrize(
        ("latency", "expected"),
        [(0.0, 0.0), (1.23456, 1.235), (1.2344, 1.234), (12.5, 12.5), (7, 7.0)],
    )
    def test_latency_is_rounded_to_three_decimals(
        self, latency: Any, expected: float
    ) -> None:
        """耗时是报告里的一个数字：按 3 位小数收敛（照 ``RerankResult`` 的口径）."""
        generation = make_generation(latency_ms=latency)

        assert generation.latency_ms == expected
        assert generation.to_dict()["latency_ms"] == expected

    def test_to_dict_carries_the_prompt_by_default(self) -> None:
        """默认带上提示词：排查一次回答时"模型到底被怎么问的"就是它."""
        payload = make_generation(prompt_text="一二三四五").to_dict()

        assert set(payload) == {
            "question",
            "answer",
            "prompt_version",
            "prompt_chars",
            "llm_called",
            "fallback_reason",
            "is_fallback",
            "grounded",
            "model",
            "latency_ms",
            "citations",
            "check",
            "notes",
            "prompt_text",
        }
        assert payload["prompt_text"] == "一二三四五"
        assert payload["prompt_chars"] == 5

    def test_to_dict_can_drop_the_prompt(self) -> None:
        """只要排序与统计时显式传 ``False``（一段两千字的提示词会把 diff 淹掉）."""
        payload = make_generation(prompt_text="一二三四五").to_dict(include_prompt=False)

        assert len(payload) == 13
        assert "prompt_text" not in payload
        assert payload["prompt_chars"] == 5

    def test_to_dict_projects_the_citations_and_the_check(self) -> None:
        """引用表与核对报告各自展开成 JSON 可序列化的形状."""
        payload = make_generation().to_dict()

        assert [item["marker"] for item in payload["citations"]] == [1, 2, 3, 4, 5]
        assert payload["citations"][0]["record_id"] == "c-a-01"
        assert payload["check"]["given"] == 5
        assert payload["check"]["coverage"] == 0.4
        assert payload["grounded"] is True
        assert payload["is_fallback"] is False
        assert payload["fallback_reason"] == ""

    def test_to_summary_has_exactly_the_eleven_public_keys(self) -> None:
        """摘要是上游的公共字段：形状只在这里定义一次（三份手写的摘要一定会分家）."""
        payload = make_generation().to_summary()

        assert set(payload) == {
            "prompt_version",
            "model",
            "llm_called",
            "fallback_reason",
            "grounded",
            "given",
            "coverage",
            "valid",
            "invalid",
            "unused",
            "latency_ms",
        }

    @pytest.mark.parametrize("key", ["prompt_text", "answer", "citations", "check", "notes"])
    def test_to_summary_deliberately_omits_the_bulky_fields(self, key: str) -> None:
        """**刻意不含提示词**（也没有答案）：否则摘要会变成第二份正文."""
        assert key not in make_generation().to_summary()

    def test_to_summary_reports_the_counts_not_the_lists(self) -> None:
        """摘要报的是**条数**（valid / invalid / unused 三个整数），不是编号列表."""
        payload = make_generation(
            check=make_check(
                cited=(1, 9), valid=(1,), invalid=(9,), unused=(2, 3, 4, 5), coverage=0.2
            )
        ).to_summary()

        assert payload["given"] == 5
        assert payload["valid"] == 1
        assert payload["invalid"] == 1
        assert payload["unused"] == 4
        assert payload["coverage"] == 0.2
        assert payload["grounded"] is False
        assert payload["fallback_reason"] == ""
        assert payload["llm_called"] is True

    def test_the_summary_line_is_the_hand_checked_sentence(self) -> None:
        """一行摘要 = 问 / 答（截断到 24、30 字）+ 版本与调用 + 核对那一行."""
        generation = make_generation(
            question="问题？", answer="答案 [1]", prompt_text="（提示词）"
        )

        assert generation.summary_line() == (
            "问：问题？ | 答：答案 [1] | 提示词 v2（已调用） | 核对：给了 5 条片段"
            " | 答案引用 2 条（有效 2 / 幻觉 0） | 未引用 3 条 | 覆盖 40.0% | 接地通过"
        )

    def test_the_summary_line_truncates_the_question_and_the_answer(self) -> None:
        """问题截到 24 字、答案截到 30 字，换行折成空格（一行摘要必须真是一行）."""
        question = "问" * 30
        answer = "答" * 40
        generation = make_generation(question=question, answer=answer)

        line = generation.summary_line()

        assert f"问：{'问' * 24} |" in line
        assert f"答：{'答' * 30} |" in line
        assert "问" * 25 not in line
        assert "答" * 31 not in line

    def test_the_summary_line_carries_the_fallback_reason(self) -> None:
        """走了回退时那一行末尾要写出原因（否则"这次不算数"看不见）."""
        line = make_no_context_generation().summary_line()

        assert "提示词 v2（未调用）" in line
        assert line.endswith("| 回退 no_context")

    def test_explain_renders_the_question_and_the_check(self) -> None:
        """正常路径的 ``explain()`` = 问法一行 + 核对那一段（不渲染回退与注记）."""
        lines = make_generation(prompt_text="一二三四五", model="mock-v1").explain()

        assert len(lines) == 4
        assert lines[0] == (
            "问法：提示词 v2（5 字） | 模型 mock-v1 | 耗时 0.0ms | LLM 已调用"
        )
        assert lines[1].startswith("核对口径：")
        assert lines[2].startswith("覆盖率 40.0%")
        assert lines[3].startswith("未引用的片段：")

    def test_explain_says_so_when_no_model_name_was_reported(self) -> None:
        """模型名空串表示"装配期没有报告"：报告里要写成括号里的那半句."""
        assert "模型 （未报告）" in make_generation().explain()[0]

    @pytest.mark.parametrize("count", [0, 1, 3])
    def test_explain_appends_one_line_per_note_and_one_for_the_fallback(
        self, count: int
    ) -> None:
        """回退与注记各自占行：问法 1 + 核对 2 + 回退 1 + 注记 count = 4 + count."""
        notes = tuple(f"第 {index} 条" for index in range(count))

        lines = make_no_context_generation(notes=notes).explain()

        assert len(lines) == 4 + count
        assert lines[-1 - count].startswith("回退 no_context：")
        assert lines[4:] == [f"注记：{note}" for note in notes]

    def test_explain_describes_the_fallback_with_the_shared_dictionary(self) -> None:
        """回退那一行直接引用 ``FALLBACK_REASON_DESCRIPTIONS``（同一句话不写两遍）."""
        lines = make_no_context_generation().explain()

        assert (
            f"回退 no_context：{FALLBACK_REASON_DESCRIPTIONS[FALLBACK_REASON_NO_CONTEXT]}"
            in lines
        )

    def test_the_no_context_generation_carries_the_three_empty_fields(self) -> None:
        """``no_context`` 的三条字段是护栏在数据上的形状（空提示词 + 空引用 + 没调用）."""
        generation = make_no_context_generation()

        assert generation.llm_called is False
        assert generation.prompt_text == ""
        assert generation.citations == ()
        assert generation.check.given == 0
        assert generation.check.grounded is False
        assert generation.answer == FALLBACK_NO_CONTEXT


class TestGenerationConstructionGuards:
    """构造期校验：每一条都对应"报告里哪句话会变成假的"."""

    @pytest.mark.parametrize("question", ["", "   ", "\n\t"])
    def test_the_question_must_be_non_empty(self, question: str) -> None:
        """一份不知在回答什么问题的答案无法被复核."""
        with pytest.raises(GenerationError) as excinfo:
            make_generation(question=question)

        assert "Generation.question 必须是非空字符串" in str(excinfo.value)

    @pytest.mark.parametrize("answer", [1, None, b"x"])
    def test_the_answer_must_be_a_string(self, answer: Any) -> None:
        """模型回复的形状已经被收敛成文本（其它形状说明上游漏了一步）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generation(answer=answer)

        assert "Generation.answer 必须是字符串" in str(excinfo.value)

    @pytest.mark.parametrize("prompt_text", [None, 1])
    def test_the_prompt_text_must_be_a_string(self, prompt_text: Any) -> None:
        """它是"模型看到了什么"的唯一证据（None 会让那次提问无从复现）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generation(prompt_text=prompt_text)

        assert "Generation.prompt_text 必须是字符串" in str(excinfo.value)

    @pytest.mark.parametrize("version", ["v3", "V2", "custom ", "latest"])
    def test_the_prompt_version_must_be_on_the_closed_list(self, version: str) -> None:
        """版本号是分组键：清单外的取值会让这次生成落进一个谁也不认识的分组."""
        with pytest.raises(GenerationError) as excinfo:
            make_generation(prompt_version=version)

        assert "不在受管清单里" in str(excinfo.value)
        assert "合法取值是 ['v1', 'v2']" in str(excinfo.value)

    @pytest.mark.parametrize("version", ["", "   "])
    def test_a_blank_prompt_version_is_rejected_as_a_blank_string(self, version: str) -> None:
        """空串走的是另一条分支（它连"清单外的版本号"都算不上）——两条消息都要给全."""
        with pytest.raises(GenerationError) as excinfo:
            make_generation(prompt_version=version)

        message = str(excinfo.value)
        assert "必须是非空字符串" in message
        assert "出路：写 ['v1', 'v2'] 里的一个" in message

    def test_the_custom_version_is_accepted_by_the_shape(self) -> None:
        """``custom`` 不在受管清单里但**是合法取值**（装配期注入模板时的版本号）."""
        assert make_generation(prompt_version=CUSTOM_PROMPT_VERSION).prompt_version == "custom"

    @pytest.mark.parametrize("value", [1, 0, "True", None])
    def test_llm_called_must_be_a_real_boolean(self, value: Any) -> None:
        """它是护栏的唯一证据（假 LLM 的 calls 计数与它一一对应）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generation(llm_called=value)

        assert "Generation.llm_called 必须是布尔值" in str(excinfo.value)

    @pytest.mark.parametrize("reason", ["bogus", "No_Context", "none", "empty", None])
    def test_an_unknown_fallback_reason_is_rejected_with_the_legal_values(
        self, reason: Any
    ) -> None:
        """回退原因是封闭清单：自由文本会让"这一批里几次不可交付"无法统计."""
        with pytest.raises(GenerationError) as excinfo:
            make_generation(fallback_reason=reason)

        message = str(excinfo.value)
        assert "未知的回退原因" in message
        assert "['no_context', 'llm_error', 'empty_reply'" in message

    def test_the_citations_must_be_a_tuple(self) -> None:
        """写法是 ``citations=context.citations``（一份 list 会让"不曾被改过"不成立）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generation(citations=[make_citation(1)], check=make_check())

        assert "Generation.citations 必须是 tuple" in str(excinfo.value)

    @pytest.mark.parametrize("check", [None, 1, "x"])
    def test_the_check_must_be_a_grounding_report(self, check: Any) -> None:
        """核对结论要跟答案一起走（None 会让"这份答案有没有依据"无从回答）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generation(check=check)

        assert "Generation.check 必须是 GroundingReport" in str(excinfo.value)

    @pytest.mark.parametrize(("given", "count"), [(0, 1), (5, 2), (2, 5)])
    def test_the_check_must_cover_the_same_context(self, given: int, count: int) -> None:
        """报告里的"给了 N 条"必须说的是**这一次**提问的片段（两者同源）."""
        report = GroundingReport(
            cited=tuple(range(1, min(given, count) + 1)),
            valid=tuple(range(1, min(given, count) + 1)),
            unused=tuple(range(min(given, count) + 1, given + 1)),
            coverage=1.0 if given else 0.0,
            grounded=bool(given),
        )

        with pytest.raises(GenerationError) as excinfo:
            make_generation(citations=make_citations(count), check=report)

        assert "Generation 的账不自洽" in str(excinfo.value)
        assert f"核对报告覆盖 {given} 条片段，而 citations 有 {count} 条" in str(excinfo.value)

    def test_the_notes_must_be_a_tuple(self) -> None:
        """注记组合是 tuple（写法：``notes=tuple(notes)``）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generation(notes=["注记"])

        assert "Generation.notes 必须是 tuple" in str(excinfo.value)

    @pytest.mark.parametrize("model", [7, None, ["m"]])
    def test_the_model_must_be_a_string(self, model: Any) -> None:
        """它只进报告（空串表示装配期没有报告模型名）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generation(model=model)

        assert "Generation.model 必须是字符串" in str(excinfo.value)

    @pytest.mark.parametrize("latency", ["1", None, True])
    def test_latency_must_be_a_number(self, latency: Any) -> None:
        """耗时是数字（字符串会让报告里的那半句变成一句乱码）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generation(latency_ms=latency)

        assert "latency_ms 必须是数字" in str(excinfo.value)

    @pytest.mark.parametrize("latency", [-1.0, -0.001, float("nan"), float("inf")])
    def test_latency_must_be_non_negative_and_finite(self, latency: float) -> None:
        """nan 与负数都只会让这个数字失去意义（它要进报告与统计）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generation(latency_ms=latency)

        assert "latency_ms 必须是非负有限数" in str(excinfo.value)


class TestTheTwoReasonDisciplines:
    """两条与回退联动的纪律：它们把"这次到底发生了什么"钉成一个可判定的事实."""

    @pytest.mark.parametrize(
        ("field", "value", "check"),
        [
            ("llm_called", True, GroundingReport()),
            ("prompt_text", "（提示词）", GroundingReport()),
            (
                "citations",
                make_citations(1),
                GroundingReport(
                    cited=(), valid=(), unused=(1,), coverage=0.0, grounded=False
                ),
            ),
        ],
    )
    def test_no_context_forbids_calling_the_model_or_rendering(
        self, field: str, value: Any, check: GroundingReport
    ) -> None:
        """三条必须同时成立：没渲染提示词、没给片段、没调模型.

        第三个组合的 ``check`` 与 ``citations`` 是**配套**的（报告里的"给了 N 条"
        必须先与引用表对上，否则先撞上的会是"账不自洽"那条更靠前的校验）。
        """
        payload: dict[str, Any] = {"check": check}
        payload[field] = value

        with pytest.raises(GenerationError) as excinfo:
            make_no_context_generation(**payload)

        assert "no_context 的这次生成必须满足三条" in str(excinfo.value)

    def test_no_context_accepts_the_three_empty_fields(self) -> None:
        """反例：三条都在位时它是合法的（这条护栏不是"凡 no_context 都报错"）."""
        assert make_no_context_generation().fallback_reason == FALLBACK_REASON_NO_CONTEXT

    @pytest.mark.parametrize("answer", ["", "   ", "\n"])
    def test_none_forbids_an_empty_answer(self, answer: str) -> None:
        """空串必须走 ``empty_reply``（它的处置动作是查提供方，不是"没事"）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generation(answer=answer)

        assert "报告'没有回退'却给了一份空答案" in str(excinfo.value)

    def test_none_forbids_llm_called_false(self) -> None:
        """一次正常的答案只可能来自模型（没调模型时必须落到某个回退原因上）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generation(llm_called=False)

        assert "报告'没有回退'却写着 llm_called=False" in str(excinfo.value)

    def test_none_forbids_an_empty_citation_table(self) -> None:
        """正常路径要求至少一条落在提示词里的引用（这是 grounded 的前半条）."""
        with pytest.raises(GenerationError) as excinfo:
            make_generation(
                citations=(),
                check=GroundingReport(cited=(9,), invalid=(9,), grounded=False),
            )

        assert "报告'没有回退'却没有任何引用" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# RAGGenerator：构造期与只读视图
# --------------------------------------------------------------------------- #


class TestGeneratorConstruction:
    """构造期的四类参数（模板 / 采样 / 交付 / 模型名）各自决定一件可讨论的事."""

    def test_describe_has_exactly_the_eight_documented_keys(self) -> None:
        """``describe()`` 是端点与演示脚本的统一入口：键名与规格一致."""
        described = RAGGenerator(MockLLM()).describe()

        assert set(described) == {
            "prompt_version",
            "prompt_chars",
            "temperature",
            "max_tokens",
            "require_citation",
            "model",
            "llm",
            "prompt_versions",
        }

    def test_describe_echoes_the_managed_versions(self) -> None:
        """端点收到一个版本号时要能当场列出合法取值（受管清单只有一个出处）."""
        described = RAGGenerator(MockLLM()).describe()

        assert described["prompt_versions"] == ["v1", "v2"]
        assert described["prompt_version"] == "v2"
        assert described["prompt_chars"] == len(RAG_PROMPT_V2) == 500
        assert described["llm"] == "MockLLM"

    def test_the_defaults_come_from_settings(self) -> None:
        """``None`` 的含义是"没指定，去读 settings"（项目默认只有一处定义）."""
        described = RAGGenerator(MockLLM()).describe()

        assert described["temperature"] == settings.retrieval_answer_temperature == 0.0
        assert described["max_tokens"] == settings.retrieval_answer_max_tokens == 1024
        assert described["require_citation"] is False
        assert described["model"] == ""

    @pytest.mark.parametrize("temperature", [0.0, 0.5, 1, 2.0, 2])
    def test_a_valid_temperature_is_accepted_and_normalized(self, temperature: Any) -> None:
        """整数温度会被收敛成 float（报告里的那个数必须是已解析的）."""
        assert RAGGenerator(MockLLM(), temperature=temperature).temperature == float(temperature)

    def test_the_prompt_version_property_reports_the_managed_version(self) -> None:
        """显式给版本号时用它，且模板与版本号必须配套."""
        generator = RAGGenerator(MockLLM(), prompt_version=PROMPT_VERSION_V1)

        assert generator.prompt_version == "v1"
        assert generator.describe()["prompt_chars"] == len(RAG_PROMPT_V1) == 238

    def test_a_custom_template_is_labelled_custom(self) -> None:
        """给了 prompt 就用它，版本号记为 ``custom``（它不在受管清单里但合法）."""
        generator = RAGGenerator(MockLLM(), prompt="【资料】{context}【问】{question}")

        assert generator.prompt_version == CUSTOM_PROMPT_VERSION
        assert generator.describe()["prompt_chars"] == 26

    def test_the_llm_property_returns_the_injected_object(self) -> None:
        """端点在"未配置 LLM"时要在构造前就拦住，因此注入的对象必须能读回来."""
        llm = MockLLM()

        assert RAGGenerator(llm).llm is llm

    def test_the_read_only_properties_expose_the_resolved_parameters(self) -> None:
        """四个旋钮都要能被读回来（端点与演示脚本用它们自述）."""
        generator = RAGGenerator(
            MockLLM(), temperature=0.25, max_tokens=64, require_citation=True, model="mock-v1"
        )

        assert generator.temperature == 0.25
        assert generator.max_tokens == 64
        assert generator.require_citation is True
        assert generator.fallback_answer == FALLBACK_NO_CONTEXT

    @pytest.mark.parametrize("llm", ["x", 1, None, object()])
    def test_the_llm_must_be_a_base_llm(self, llm: Any) -> None:
        """没有模型就没有答案可核对（任何实现 ``BaseLLM.chat`` 的对象都可以注入）."""
        with pytest.raises(GenerationError) as excinfo:
            RAGGenerator(llm)

        assert "llm 必须是 BaseLLM" in str(excinfo.value)
        assert type(llm).__name__ in str(excinfo.value)

    @pytest.mark.parametrize("temperature", [-0.1, 2.5, -1, 100])
    def test_temperature_out_of_range_is_rejected(self, temperature: float) -> None:
        """RAG 问答是有依据的复述：温度越高，模型越会改写片段里的事实."""
        with pytest.raises(GenerationError) as excinfo:
            RAGGenerator(MockLLM(), temperature=temperature)

        assert f"temperature={float(temperature)} 超出 [0, 2]" in str(excinfo.value)

    @pytest.mark.parametrize("temperature", [float("nan"), float("inf"), "0.5", True, None])
    def test_a_non_number_temperature_falls_back_to_the_settings(self, temperature: Any) -> None:
        """``None`` = 读 settings；其余非数字当场报错（nan 会让采样不可复现）."""
        if temperature is None:
            assert RAGGenerator(MockLLM(), temperature=None).temperature == 0.0
            return
        with pytest.raises(GenerationError) as excinfo:
            RAGGenerator(MockLLM(), temperature=temperature)

        message = str(excinfo.value)
        assert ("必须是数字" in message) or ("必须是有限数" in message)

    @pytest.mark.parametrize("max_tokens", [0, -3, -1])
    def test_max_tokens_must_be_positive(self, max_tokens: int) -> None:
        """0 会让模型"没有空间说话"，而它的表现是一次空回复（会被误读成拒答）."""
        with pytest.raises(GenerationError) as excinfo:
            RAGGenerator(MockLLM(), max_tokens=max_tokens)

        assert f"max_tokens={max_tokens}" in str(excinfo.value)
        assert "没有空间说话" in str(excinfo.value)

    @pytest.mark.parametrize("max_tokens", ["16", 1.5, True, None])
    def test_max_tokens_must_be_an_integer(self, max_tokens: Any) -> None:
        """``None`` 读 settings；其余非整数当场报错（浮点 token 上限没有定义）."""
        if max_tokens is None:
            assert RAGGenerator(MockLLM(), max_tokens=None).max_tokens == 1024
            return
        with pytest.raises(GenerationError) as excinfo:
            RAGGenerator(MockLLM(), max_tokens=max_tokens)

        assert "max_tokens 必须是整数" in str(excinfo.value)

    @pytest.mark.parametrize("value", ["true", 1, 0, None])
    def test_require_citation_must_be_a_real_boolean(self, value: Any) -> None:
        """它是开关：字符串 ``'false'`` 在 if 里恒为真，"关掉了"从没发生过."""
        with pytest.raises(GenerationError) as excinfo:
            RAGGenerator(MockLLM(), require_citation=value)

        assert "require_citation 必须是布尔值" in str(excinfo.value)

    @pytest.mark.parametrize("model", [3, None, ["mock"]])
    def test_the_model_name_must_be_a_string(self, model: Any) -> None:
        """它只进报告，不参与任何加载或路由（本模块不认识任何具体后端）."""
        with pytest.raises(GenerationError) as excinfo:
            RAGGenerator(MockLLM(), model=model)

        assert "model 必须是字符串" in str(excinfo.value)

    @pytest.mark.parametrize("fallback", ["", "   ", 1, None])
    def test_the_fallback_answer_must_be_non_empty_text(self, fallback: Any) -> None:
        """回退时返回一句空话与"不回答"是两件事（前者让用户以为系统坏了）."""
        with pytest.raises(GenerationError) as excinfo:
            RAGGenerator(MockLLM(), fallback_answer=fallback)

        assert "fallback_answer 必须是非空字符串" in str(excinfo.value)

    @pytest.mark.parametrize("version", ["v3", "V1", "custom ", "latest"])
    def test_an_illegal_version_is_rejected_even_when_a_prompt_is_given(
        self, version: str
    ) -> None:
        """被忽略的版本号**也要过清单校验**：拼错的版本号被记成"已忽略"更危险."""
        with pytest.raises(GenerationError) as excinfo:
            RAGGenerator(
                MockLLM(), prompt="【资料】{context}【问】{question}", prompt_version=version
            )

        message = str(excinfo.value)
        assert "prompt_version" in message
        assert "合法取值是 ['v1', 'v2']" in message

    @pytest.mark.parametrize("version", ["", "  "])
    def test_a_blank_ignored_version_is_rejected_too(self, version: str) -> None:
        """空串同样当场报错（"反正它不生效"不是放宽校验的理由）."""
        with pytest.raises(GenerationError) as excinfo:
            RAGGenerator(
                MockLLM(), prompt="【资料】{context}【问】{question}", prompt_version=version
            )

        assert "必须是非空字符串" in str(excinfo.value)

    @pytest.mark.parametrize(
        "template",
        [
            "只依据资料回答。",
            "资料片段：{context}",
            "问题是：{question}",
            "{{context}} 与 {{question}}",
        ],
    )
    def test_a_template_without_both_placeholders_is_rejected(self, template: str) -> None:
        """缺占位符的模板渲染出来是一段**没有资料**的提示词，而它不会报错."""
        with pytest.raises(ContextError) as excinfo:
            RAGGenerator(MockLLM(), prompt=template)

        assert "缺少占位符" in str(excinfo.value)

    def test_a_malformed_template_is_rejected(self) -> None:
        """单独的 ``{`` 不是合法格式串（不拦的话会在运行期炸，而那时检索已经跑完）."""
        with pytest.raises(ContextError) as excinfo:
            RAGGenerator(MockLLM(), prompt="资料 {")

        assert "不是合法的格式串" in str(excinfo.value)

    @pytest.mark.parametrize("template", [None, 123])
    def test_the_template_must_be_a_string(self, template: Any) -> None:
        """``None`` 的含义是"用受管模板"，因此它走的是另一条路（不是这里）."""
        if template is None:
            assert RAGGenerator(MockLLM(), prompt=None).prompt_version == "v2"
            return
        with pytest.raises(ContextError) as excinfo:
            RAGGenerator(MockLLM(), prompt=template)

        assert "prompt 必须是字符串" in str(excinfo.value)

    def test_the_generator_does_not_echo_the_template_text(self) -> None:
        """``describe()`` 只报长度不报模板正文（它是装配期的东西，不是这一次的账）."""
        described = RAGGenerator(MockLLM()).describe()

        assert "prompt" not in described
        assert all(isinstance(value, (str, int, float, bool, list)) for value in described.values())


class TestGeneratorAgainstTheSamples:
    """把样本表的六个输出值接到生成器上：这也验证了样本表本身是被实现承认的."""

    @pytest.mark.parametrize("case", GROUNDING_CASES, ids=[c.label for c in GROUNDING_CASES])
    def test_the_hand_table_matches_a_real_generation(self, case: Any) -> None:
        """每一组手算的六个数都要与真跑一次生成得到的一致."""
        generation = RAGGenerator(AnswerLLM(case.answer)).generate(
            QUESTION, packed_context()
        )

        assert generation.check.cited == case.cited
        assert generation.check.valid == case.valid
        assert generation.check.invalid == case.invalid
        assert generation.check.unused == case.unused
        assert generation.check.coverage == case.coverage
        assert generation.check.grounded is case.grounded
        assert len(generation.check.notes) == case.report_notes

    def test_the_one_citation_case_lands_on_the_documented_coverage(self) -> None:
        """抽查一条：只引中间那一条 → 覆盖 0.2、未引用 4 条、接地通过."""
        check = RAGGenerator(AnswerLLM(ANSWER_ONE)).generate(
            QUESTION, packed_context()
        ).check

        assert check.cited == (3,)
        assert check.valid == (3,)
        assert check.unused == (1, 2, 4, 5)
        assert check.coverage == COVERAGE_ONE
        assert check.grounded is True
        assert check.checks[2].record_id == MARKER_TO_ID[3]


class TestTheSampleFactories:
    """样本工厂本身也要自洽：它们是全部期望值的来源，改一个字所有断言都会跟着动."""

    def test_the_question_factory_returns_the_documented_query(self) -> None:
        """问题就是那条 ``TableEmbedding`` 查表用的文本（15 字，可逐字核对）."""
        assert question() == QUESTION
        assert len(question()) == 15
        assert question() == "检索手册里怎么配置时间范围过滤"

    def test_the_packed_context_factory_is_the_documented_one(self) -> None:
        """缺省上下文：5 条片段、295 字、编号 1..5 与手算清单一一对应."""
        context = packed_context()

        assert markers_of(context) == (1, 2, 3, 4, 5)
        assert [citation.record_id for citation in context.citations] == list(CITATION_IDS)
        assert context.char_count == 295
        assert context.is_empty is False

    def test_the_empty_packed_context_factory_is_empty(self) -> None:
        """空上下文那条路：没有任何命中，也没有任何编号（护栏的输入形状）."""
        context = empty_packed_context()

        assert context.is_empty is True
        assert markers_of(context) == ()
        assert context.text == ""

    def test_the_retrieval_factory_returns_the_documented_hits(self) -> None:
        """检索工厂给的是库里的前 5 条（顺序 = 分数降序 + 同分按 id）."""
        assert retrieval().ids() == list(CITATION_IDS)
        assert retrieval(2).ids() == list(CITATION_IDS[:2])
        assert retrieval(8).ids() == list(FULL_CITATION_IDS)

    def test_the_retriever_factory_uses_the_documented_defaults(self) -> None:
        """缺省检索器：top_k=5、名字是 default（与 ``settings`` 同一处定义）."""
        described = sample_retriever().describe()

        assert described["top_k"] == CITATION_COUNT
        assert described["name"] == "default"
