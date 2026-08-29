"""ReAct 解析器测试."""

from __future__ import annotations

import pytest

from smart_research_agent.agent.parser import ReActParseError, parse_react_output


class TestParseReactOutput:
    def test_parse_action_form(self):
        text = "Thought: 需要计算\nAction: calculator\nAction Input: 2+3"
        step = parse_react_output(text)
        assert step.thought == "需要计算"
        assert step.action == "calculator"
        assert step.action_input == "2+3"
        assert step.final_answer is None

    def test_parse_final_answer_form(self):
        text = "Thought: 已有答案\nFinal Answer: 42"
        step = parse_react_output(text)
        assert step.final_answer == "42"

    def test_missing_thought_raises(self):
        with pytest.raises(ReActParseError, match="Thought"):
            parse_react_output("Action: calculator")

    def test_missing_action_raises(self):
        with pytest.raises(ReActParseError, match="Action"):
            parse_react_output("Thought: 只是想")

    def test_multiline_thought(self):
        text = "Thought: 第一步分析\n第二步分析\nAction: calculator\nAction Input: 1+1"
        step = parse_react_output(text)
        assert "第二步分析" in step.thought
