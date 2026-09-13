"""LLM 调用层测试（全部离线，不发起网络请求）."""

from __future__ import annotations

import pytest

from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.llm.openai_compatible import OpenAICompatibleLLM


class _ChatOnlyLLM(BaseLLM):
    """只实现 chat、不覆写 stream 的最小实现，验证 BaseLLM 默认流式契约."""

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        return "完整回复"


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


class TestStream:
    """day039：流式输出协议（MockLLM 真分片 + BaseLLM 默认单片段）."""

    def test_mock_stream_reassembles_to_chat(self):
        llm = MockLLM(responses=["0123456789"])  # 10 字符，_CHUNK_SIZE=4 → 3 片
        chunks = list(llm.stream([Message(role="user", content="hi")]))
        assert "".join(chunks) == "0123456789"
        assert len(chunks) == 3
        assert chunks == ["0123", "4567", "89"]

    def test_mock_stream_consumes_script_like_chat(self):
        llm = MockLLM(responses=["first"])
        first = list(llm.stream([Message(role="user", content="a")]))
        assert "".join(first) == "first"
        # 脚本已被流式消费，下一次走 default 兜底
        assert llm.chat([Message(role="user", content="b")]) == "mock response"

    def test_mock_stream_records_usage_same_as_chat(self):
        llm = MockLLM(responses=["usage reply"])
        list(llm.stream([Message(role="user", content="hi")]))
        assert len(llm.usage_log) == 1
        assert llm.total_completion_tokens > 0

    def test_base_default_stream_yields_single_chunk(self):
        llm = _ChatOnlyLLM()
        assert list(llm.stream([Message(role="user", content="hi")])) == ["完整回复"]

    def test_base_chat_with_tools_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _ChatOnlyLLM().chat_with_tools([Message(role="user", content="hi")], tools=[])
