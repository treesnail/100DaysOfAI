"""Planner 模块测试."""

from __future__ import annotations

import pytest

from smart_research_agent.agent.planner import PlanParseError, Planner, parse_plan
from smart_research_agent.llm.mock import MockLLM


class TestParsePlan:
    def test_parse_clean_json(self):
        assert parse_plan('["查资料", "写总结"]') == ["查资料", "写总结"]

    def test_parse_json_with_surrounding_text(self):
        text = '好的，这是计划：\n["步骤一", "步骤二", "步骤三"]\n请确认。'
        assert parse_plan(text) == ["步骤一", "步骤二", "步骤三"]

    def test_no_json_raises(self):
        with pytest.raises(PlanParseError, match="JSON"):
            parse_plan("没有任何数组")

    def test_empty_array_raises(self):
        with pytest.raises(PlanParseError, match="空"):
            parse_plan("[]")

    def test_non_string_items_raise(self):
        with pytest.raises(PlanParseError):
            parse_plan("[1, 2, 3]")


class TestPlanner:
    def test_plan_end_to_end(self):
        llm = MockLLM(responses=['["收集资料", "对比方案", "撰写报告"]'])
        plan = Planner(llm=llm).plan("调研 RAG 技术")
        assert plan.goal == "调研 RAG 技术"
        assert plan.steps == ["收集资料", "对比方案", "撰写报告"]

    def test_invalid_llm_output_raises(self):
        llm = MockLLM(responses=["我不知道怎么拆解"])
        with pytest.raises(PlanParseError):
            Planner(llm=llm).plan("任意目标")
