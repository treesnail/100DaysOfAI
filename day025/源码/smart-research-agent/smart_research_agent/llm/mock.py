"""MockLLM：离线测试用的确定性 LLM 实现."""

from __future__ import annotations

from smart_research_agent.llm.base import BaseLLM, Message


class MockLLM(BaseLLM):
    """按预设脚本回复的 LLM，用于无网络、无 API Key 的测试环境.

    用法::

        llm = MockLLM(responses=["回答1", "回答2"])
        llm.chat([Message(role="user", content="hi")])  # -> "回答1"
        llm.chat([Message(role="user", content="hi")])  # -> "回答2"
    """

    def __init__(self, responses: list[str] | None = None, default: str = "mock response"):
        self._responses = list(responses or [])
        self._default = default
        self.calls: list[list[Message]] = []

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append(messages)
        if self._responses:
            return self._responses.pop(0)
        return self._default
