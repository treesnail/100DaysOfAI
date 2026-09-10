"""Reflexion 反思模块测试（MockLLM 驱动，完全离线）."""

from __future__ import annotations

from smart_research_agent.agent.react_agent import ReactAgent
from smart_research_agent.agent.reflexion import Reflection, Reflector, parse_reflection
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry


def _make_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return registry


class TestParseReflection:
    def test_parse_clean_json(self):
        reason, suggestion = parse_reflection(
            '{"reason": "输入格式错误", "suggestion": "改用阿拉伯数字"}'
        )
        assert reason == "输入格式错误"
        assert suggestion == "改用阿拉伯数字"

    def test_parse_json_with_surrounding_text(self):
        text = '复盘如下：{"reason": "调用了不存在的工具", "suggestion": "只用清单内的工具"} 完毕。'
        assert parse_reflection(text) == ("调用了不存在的工具", "只用清单内的工具")

    def test_non_json_falls_back_to_raw_text(self):
        reason, suggestion = parse_reflection("模型直接输出了一段散文复盘")
        assert reason == "模型直接输出了一段散文复盘"
        assert suggestion == ""

    def test_empty_output_falls_back(self):
        reason, suggestion = parse_reflection("")
        assert reason == "LLM 未给出有效复盘"
        assert suggestion == ""


class TestReflector:
    def test_reflect_end_to_end(self):
        llm = MockLLM(responses=['{"reason": "表达式无法解析", "suggestion": "输入 2+3"}'])
        reflector = Reflector(llm=llm)
        reflection = reflector.reflect("2+3 等于几？", ["第1步 Action: calculator(二加三) | Observation: 计算失败"])
        assert reflection == Reflection(failed=True, reason="表达式无法解析", suggestion="输入 2+3")
        # 每次反思都被记录，便于审计与断言
        assert reflector.reflections == [reflection]

    def test_default_max_retries_is_two(self):
        assert Reflector(llm=MockLLM()).max_retries == 2

    def test_reflect_prompt_contains_task_and_trajectory(self):
        llm = MockLLM(responses=['{"reason": "r", "suggestion": "s"}'])
        Reflector(llm=llm).reflect("调研任务", ["轨迹1", "轨迹2"])
        prompt = llm.calls[0][0].content
        assert "调研任务" in prompt
        assert "轨迹1" in prompt and "轨迹2" in prompt


class TestReactAgentReflexion:
    def test_fail_reflect_retry_then_success(self):
        """完整场景：失败 → 反思 → 建议注入 → 重试成功."""
        agent_llm = MockLLM(
            responses=[
                # 第一次尝试：把中文数字喂给计算器，必然失败
                "Thought: 先算一下\nAction: calculator\nAction Input: 二加三",
                # 第二次尝试：按建议改用阿拉伯数字
                "Thought: 改用阿拉伯数字\nAction: calculator\nAction Input: 2+3",
                "Thought: 已得到结果\nFinal Answer: 2+3 等于 5",
            ]
        )
        reflector_llm = MockLLM(
            responses=['{"reason": "中文数字无法解析", "suggestion": "请使用阿拉伯数字表达式"}']
        )
        reflector = Reflector(llm=reflector_llm)
        agent = ReactAgent(llm=agent_llm, registry=_make_registry(), reflector=reflector)

        answer = agent.run("2+3 等于几？")

        assert "5" in answer
        # 反思被恰好触发一次
        assert len(reflector_llm.calls) == 1
        assert reflector.reflections[0].suggestion == "请使用阿拉伯数字表达式"
        # 建议被注入第二次尝试的 system 与 user 消息
        second_attempt = agent_llm.calls[1]
        assert any("请使用阿拉伯数字表达式" in m.content for m in second_attempt if m.role == "system")
        assert any("请使用阿拉伯数字表达式" in m.content for m in second_attempt if m.role == "user")

    def test_max_retries_limits_attempts(self):
        """一直失败时：最多 1 + max_retries 次尝试，反思恰好 max_retries 次."""
        agent_llm = MockLLM(
            responses=["Thought: 再试\nAction: calculator\nAction Input: 二加三"] * 10
        )
        reflector_llm = MockLLM(
            responses=['{"reason": "r", "suggestion": "s"}'] * 10
        )
        reflector = Reflector(llm=reflector_llm, max_retries=2)
        agent = ReactAgent(llm=agent_llm, registry=_make_registry(), reflector=reflector)

        answer = agent.run("2+3 等于几？")

        assert "失败" in answer
        assert len(agent_llm.calls) == 3  # 1 次原始尝试 + 2 次重试
        assert len(reflector_llm.calls) == 2  # 最后一次失败后不再反思
        assert len(reflector.reflections) == 2

    def test_no_final_answer_triggers_reflection(self):
        """循环耗尽仍未产出答案，同样触发反思."""
        agent_llm = MockLLM(
            responses=[
                "Thought: 一直算\nAction: calculator\nAction Input: 1+1",
                "Thought: 还在算\nAction: calculator\nAction Input: 1+1",
                "Thought: 重试后直接回答\nFinal Answer: 答案是 2",
            ]
        )
        reflector_llm = MockLLM(responses=['{"reason": "陷入重复", "suggestion": "直接给答案"}'])
        reflector = Reflector(llm=reflector_llm)
        agent = ReactAgent(
            llm=agent_llm, registry=_make_registry(), max_steps=2, reflector=reflector
        )

        answer = agent.run("1+1 等于几？")

        assert answer == "答案是 2"
        assert len(reflector_llm.calls) == 1

    def test_success_without_reflection(self):
        """一次成功的任务不触发任何反思."""
        agent_llm = MockLLM(responses=["Thought: 直接答\nFinal Answer: 完成"])
        reflector_llm = MockLLM()
        agent = ReactAgent(
            llm=agent_llm, registry=_make_registry(), reflector=Reflector(llm=reflector_llm)
        )

        assert agent.run("任意任务") == "完成"
        assert reflector_llm.calls == []
        assert agent.history[0].final_answer == "完成"

    def test_backward_compatible_without_reflector(self):
        """不传 reflector 时行为与 day006 一致：错误 Observation 回喂后继续循环."""
        agent = ReactAgent(
            llm=MockLLM(
                responses=[
                    "Thought: 试试不存在的工具\nAction: time_machine\nAction Input: now",
                    "Thought: 工具不存在，直接回答\nFinal Answer: 无法使用工具",
                ]
            ),
            registry=_make_registry(),
        )
        answer = agent.run("现在几点？")
        assert answer == "无法使用工具"
        second_call = agent.llm.calls[1]
        assert any("不存在" in m.content for m in second_call)

    def test_parse_error_in_reflection_output_still_retries(self):
        """反思输出不是 JSON 时软着陆：原文兜底，重试照常进行."""
        agent_llm = MockLLM(
            responses=[
                "Thought: 试试\nAction: time_machine\nAction Input: now",
                "Thought: 收到教训\nFinal Answer: 无法完成",
            ]
        )
        reflector_llm = MockLLM(responses=["这不是 JSON，是一段复盘散文"])
        reflector = Reflector(llm=reflector_llm)
        agent = ReactAgent(llm=agent_llm, registry=_make_registry(), reflector=reflector)

        answer = agent.run("现在几点？")

        assert answer == "无法完成"
        # 兜底：整段原文作为 reason，suggestion 为空
        assert reflector.reflections[0].reason == "这不是 JSON，是一段复盘散文"
        assert reflector.reflections[0].suggestion == ""
        # 第二次尝试正常进行（共 2 次 Agent LLM 调用）
        assert len(agent_llm.calls) == 2
