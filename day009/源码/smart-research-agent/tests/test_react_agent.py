"""ReactAgent 骨架测试."""

from __future__ import annotations

from smart_research_agent.agent.react_agent import ReactAgent
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.tools.registry import ToolRegistry


class TestReactAgentSkeleton:
    def test_default_init(self):
        agent = ReactAgent(llm=MockLLM(), registry=ToolRegistry())
        assert agent.max_steps == 10
        assert agent.history == []
