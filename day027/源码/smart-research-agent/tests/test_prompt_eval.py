"""PromptEvaluator 与提示词模板的测试：规则分支、评委解析、加权合成、版本对比."""

from __future__ import annotations

import pytest

from smart_research_agent.agent.prompts import (
    REACT_SYSTEM_PROMPT,
    REACT_SYSTEM_PROMPT_V1,
    REACT_SYSTEM_PROMPT_V2,
)
from smart_research_agent.evaluation.harness import EvalCase
from smart_research_agent.evaluation.prompt_eval import (
    JUDGE_FALLBACK_SCORE,
    JUDGE_WEIGHT,
    RULE_WEIGHT,
    PromptEvalHarness,
    PromptEvaluator,
    parse_judge_output,
)
from smart_research_agent.llm.mock import MockLLM

GOOD_JUDGE_JSON = (
    '{"clarity": 5, "consistency": 5, "ambiguity_free": 4, '
    '"issues": [], "suggestions": ["可补充 few-shot 示例"]}'
)
LOW_JUDGE_JSON = (
    '{"clarity": 1, "consistency": 2, "ambiguity_free": 1, '
    '"issues": ["读完不知道要做什么"], "suggestions": ["重写"]}'
)

#: 结构良好但缺少角色定义的提示词（长度 >= MIN_PROMPT_LENGTH）
NO_ROLE_PROMPT = (
    "请按照指定的 JSON 格式输出结果。必须严格遵循以下要求：逐条列出全部要点，"
    "保持字段命名一致，不要遗漏任何字段内容，完成后自行校验格式是否合法。"
)


def make_evaluator(*responses: str) -> PromptEvaluator:
    return PromptEvaluator(llm=MockLLM(responses=list(responses)))


class TestRuleChecks:
    """check_rules 的七个规则分支."""

    def test_short_prompt_flagged(self):
        report = make_evaluator().check_rules("你好。")
        assert report.clarity < 5.0
        assert any("过短" in issue for issue in report.issues)

    def test_missing_role_flagged(self):
        report = make_evaluator().check_rules(NO_ROLE_PROMPT)
        assert any("角色" in issue for issue in report.issues)
        assert report.clarity == 4.0

    def test_missing_task_or_format_flagged(self):
        prompt = "你是一个乐于助人的助手。必须严格遵循要求，禁止编造内容，不要离题。" * 2
        report = make_evaluator().check_rules(prompt)
        assert any("任务或输出格式" in issue for issue in report.issues)

    def test_missing_constraints_flagged(self):
        prompt = "你是翻译助手。请把用户输入的英文段落输出为通顺流畅的中文译文，" "保持原意不变。" * 3
        report = make_evaluator().check_rules(prompt)
        assert any("约束" in issue for issue in report.issues)
        assert report.ambiguity_free < 5.0

    def test_unclosed_placeholder_flagged(self):
        prompt = (
            "你是一个严谨的助手。请按照规定的格式输出结果，必须严格遵循要求，"
            "不要遗漏字段。可用工具列表：{tools"
        )
        report = make_evaluator().check_rules(prompt)
        assert any("占位符" in issue for issue in report.issues)
        assert report.consistency == 4.0

    def test_empty_placeholder_flagged(self):
        prompt = (
            "你是一个严谨的助手。请按照规定的格式输出结果，必须严格遵循要求，"
            "不要遗漏字段。可用工具列表：{}"
        )
        report = make_evaluator().check_rules(prompt)
        assert any("占位符" in issue for issue in report.issues)

    def test_contradiction_detected(self):
        prompt = (
            "你是评审员。请严格按照规定的格式输出评审结果，逐条给出评分理由，"
            "不要遗漏维度。输出格式可以随意调整。"
        )
        report = make_evaluator().check_rules(prompt)
        assert any("矛盾" in issue for issue in report.issues)
        assert report.consistency < 5.0

    def test_vague_wording_detected(self):
        prompt = (
            "你是写作助手。请按照规定的格式输出文章，必须严格遵循要求，"
            "不要跑题，内容尽量写得详细一些。"
        )
        report = make_evaluator().check_rules(prompt)
        assert any("模糊" in issue for issue in report.issues)
        assert report.ambiguity_free == 4.0

    def test_well_structured_prompt_gets_full_rule_marks(self):
        report = make_evaluator().check_rules(REACT_SYSTEM_PROMPT_V2)
        assert (report.clarity, report.consistency, report.ambiguity_free) == (5.0, 5.0, 5.0)
        assert report.issues == []


class TestParseJudgeOutput:
    """评委输出解析：正常、容错、兜底、钳制."""

    def test_parse_clean_json(self):
        result = parse_judge_output(GOOD_JUDGE_JSON)
        assert result["clarity"] == 5.0
        assert result["ambiguity_free"] == 4.0
        assert result["suggestions"] == ["可补充 few-shot 示例"]

    def test_parse_json_with_surrounding_text(self):
        text = f"好的，评审如下：\n{GOOD_JUDGE_JSON}\n以上。"
        result = parse_judge_output(text)
        assert result["clarity"] == 5.0

    def test_non_json_falls_back_to_neutral(self):
        result = parse_judge_output("我觉得这个提示词还行，给四分吧。")
        assert result["clarity"] == JUDGE_FALLBACK_SCORE
        assert result["consistency"] == JUDGE_FALLBACK_SCORE
        assert any("无法解析" in issue for issue in result["issues"])

    def test_missing_key_falls_back(self):
        result = parse_judge_output('{"clarity": 4}')
        assert result["consistency"] == JUDGE_FALLBACK_SCORE

    def test_out_of_range_scores_clamped(self):
        result = parse_judge_output('{"clarity": 9, "consistency": 0, "ambiguity_free": 3}')
        assert result["clarity"] == 5.0
        assert result["consistency"] == 1.0


class TestEvaluate:
    """规则侧与评委侧的加权合成."""

    def test_weighted_blend(self):
        evaluator = make_evaluator(GOOD_JUDGE_JSON)
        score = evaluator.evaluate(NO_ROLE_PROMPT)
        rule = evaluator.check_rules(NO_ROLE_PROMPT)
        expected_clarity = round(RULE_WEIGHT * rule.clarity + JUDGE_WEIGHT * 5.0, 2)
        assert score.clarity == expected_clarity
        assert score.overall == round((score.clarity + score.consistency + score.ambiguity_free) / 3, 2)

    def test_issues_and_suggestions_merged(self):
        score = make_evaluator(GOOD_JUDGE_JSON).evaluate(NO_ROLE_PROMPT)
        assert any("角色" in issue for issue in score.issues)  # 规则侧
        assert "可补充 few-shot 示例" in score.suggestions  # 评委侧

    def test_judge_failure_degrades_gracefully(self):
        score = make_evaluator("这不是 JSON").evaluate(REACT_SYSTEM_PROMPT_V2)
        # 规则满分 + 评委中性兜底：overall = (5*0.4+3*0.6) 加权后的均值
        assert 3.0 < score.overall < 5.0
        assert any("无法解析" in issue for issue in score.issues)

    def test_judge_prompt_contains_rubric_and_target(self):
        evaluator = make_evaluator(GOOD_JUDGE_JSON)
        evaluator.evaluate(NO_ROLE_PROMPT)
        sent = evaluator.llm.calls[0][0].content
        assert "clarity" in sent and "rubric" in sent
        assert NO_ROLE_PROMPT in sent


class TestCompare:
    """模板版本对比."""

    def test_better_prompt_wins(self):
        result = make_evaluator(GOOD_JUDGE_JSON, GOOD_JUDGE_JSON).compare(
            REACT_SYSTEM_PROMPT_V2, "你好。"
        )
        assert result.winner == "a"
        assert result.score_a.overall > result.score_b.overall

    def test_second_prompt_wins(self):
        result = make_evaluator(GOOD_JUDGE_JSON, GOOD_JUDGE_JSON).compare(
            "你好。", REACT_SYSTEM_PROMPT_V2
        )
        assert result.winner == "b"

    def test_identical_prompts_tie(self):
        result = make_evaluator(GOOD_JUDGE_JSON, GOOD_JUDGE_JSON).compare(
            REACT_SYSTEM_PROMPT_V2, REACT_SYSTEM_PROMPT_V2
        )
        assert result.winner == "tie"


class TestReactSystemPrompt:
    """用 PromptEvaluator 守卫 prompts.py 里的模板质量."""

    def test_v2_scores_high(self):
        score = make_evaluator(GOOD_JUDGE_JSON).evaluate(REACT_SYSTEM_PROMPT)
        assert score.overall >= 4.0

    def test_v2_not_worse_than_v1(self):
        result = make_evaluator(GOOD_JUDGE_JSON, GOOD_JUDGE_JSON).compare(
            REACT_SYSTEM_PROMPT_V2, REACT_SYSTEM_PROMPT_V1
        )
        assert result.winner in ("a", "tie")
        assert result.score_a.overall >= result.score_b.overall


class TestPromptEvalHarness:
    """提示词回归评估执行器."""

    def test_passing_case(self):
        harness = PromptEvalHarness(make_evaluator(GOOD_JUDGE_JSON))
        results = harness.run([EvalCase(name="react_v2", input=REACT_SYSTEM_PROMPT_V2, expected="4.0")])
        assert results[0].passed is True
        assert harness.summary(results)["pass_rate"] == 1.0

    def test_failing_case_records_issues(self):
        harness = PromptEvalHarness(make_evaluator(LOW_JUDGE_JSON))
        results = harness.run([EvalCase(name="bad", input="你好。", expected="4.0")])
        assert results[0].passed is False
        assert results[0].detail  # 问题清单写入 detail

    def test_default_threshold_when_expected_missing(self):
        harness = PromptEvalHarness(make_evaluator(GOOD_JUDGE_JSON), default_threshold=4.5)
        results = harness.run([EvalCase(name="no_expected", input=REACT_SYSTEM_PROMPT_V2)])
        assert results[0].passed is True
