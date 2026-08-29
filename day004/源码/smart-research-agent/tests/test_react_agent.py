"""ReactAgent 骨架测试."""

from __future__ import annotations

import pytest

from smart_research_agent.agent.react_agent import ReactAgent


class TestReactAgentSkeleton:
    def test_default_init(self):
        agent = ReactAgent()
        assert agent.max_steps == 10
        assert agent.tools == []
        assert agent.history == []

    def test_run_not_implemented_yet(self):
        agent = ReactAgent()
        with pytest.raises(NotImplementedError, match="day006"):
            agent.run("任意任务")
