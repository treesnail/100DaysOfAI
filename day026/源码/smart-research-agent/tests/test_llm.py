"""LLM 调用层测试（全部离线，不发起网络请求）."""

from __future__ import annotations

from smart_research_agent.llm.base import Message
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.llm.openai_compatible import OpenAICompatibleLLM


class TestMessage:
    def test_to_dict(self):
        assert Message(role="user", content="hi").to_dict() == {"role": "user", "content": "hi"}


class TestMockLLM:
    def test_scripted_responses_in_order(self):
        llm = MockLLM(responses=["r1", "r2"])
        assert llm.chat([Message(role="user", content="a")]) == "r1"
        assert llm.chat([Message(role="user", content="b")]) == "r2"

    def test_default_when_script_exhausted(self):
        llm = MockLLM(responses=["r1"], default="兜底")
        llm.chat([Message(role="user", content="a")])
        assert llm.chat([Message(role="user", content="b")]) == "兜底"

    def test_records_calls(self):
        llm = MockLLM()
        llm.chat([Message(role="user", content="你好")])
        assert len(llm.calls) == 1
        assert llm.calls[0][0].content == "你好"


class TestOpenAICompatibleLLM:
    def test_constructs_with_explicit_key(self):
        llm = OpenAICompatibleLLM(api_key="sk-test", model="test-model")
        assert llm._model == "test-model"
