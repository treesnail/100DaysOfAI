"""本地模型调用层测试（day045，全部离线：客户端为桩对象，不发网络请求）."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from smart_research_agent.llm.base import Message
from smart_research_agent.llm.local_model import (
    DEFAULT_OLLAMA_API_KEY,
    DEFAULT_VLLM_API_KEY,
    LocalModel,
    LocalModelSpec,
    create_local_model,
    default_base_url,
    normalize_base_url,
)


class _StubDelta:
    def __init__(self, content):
        self.content = content


class _StubStreamChoice:
    def __init__(self, content):
        self.delta = _StubDelta(content)


class _StubStreamChunk:
    def __init__(self, content):
        self.choices = [_StubStreamChoice(content)]


class _StubFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _StubToolCall:
    def __init__(self, id_: str, name: str, arguments: str):
        self.id = id_
        self.type = "function"
        self.function = _StubFunction(name, arguments)


class _StubMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _StubCompletion:
    def __init__(self, message: _StubMessage):
        self.choices = [SimpleNamespace(message=message)]


class _StubCompletions:
    """桩的 /chat/completions：记录调用参数，按队列返回消息."""

    def __init__(self, queue=None):
        self.queue = list(queue or [])
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter([_StubStreamChunk("本地"), _StubStreamChunk("流式")])
        message = self.queue.pop(0) if self.queue else _StubMessage(content="默认回复")
        return _StubCompletion(message)


class _StubModels:
    """桩的 /models：可模拟探活成功与失败."""

    def __init__(self, ids=None, error: Exception | None = None):
        self.ids = ids or []
        self.error = error
        self.timeouts: list[float | None] = []

    def list(self, timeout=None):
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(data=[SimpleNamespace(id=i) for i in self.ids])


class StubClient:
    def __init__(self, ids=None, list_error=None, queue=None):
        self.models = _StubModels(ids, list_error)
        self.completions = _StubCompletions(queue)
        self.chat = SimpleNamespace(completions=self.completions)


class TestNormalizeBaseUrl:
    """端点规范化：本地部署最高频的配置错误就是漏写 /v1."""

    def test_appends_v1_when_missing(self):
        assert normalize_base_url("http://localhost:11434") == "http://localhost:11434/v1"

    def test_strips_trailing_slash(self):
        assert normalize_base_url("http://localhost:11434/v1/") == "http://localhost:11434/v1"

    def test_adds_scheme(self):
        assert normalize_base_url("localhost:11434") == "http://localhost:11434/v1"

    def test_keeps_custom_prefix(self):
        """反向代理前缀不应被改写（只在路径为空时补 /v1）."""
        assert (
            normalize_base_url("https://gw.example.com/proxy/llm")
            == "https://gw.example.com/proxy/llm"
        )

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="不能为空"):
            normalize_base_url("   ")

    def test_rejects_unparsable(self):
        with pytest.raises(ValueError, match="无法解析"):
            normalize_base_url("http://")

    def test_default_base_url_per_backend(self):
        assert default_base_url("ollama") == "http://localhost:11434/v1"
        assert default_base_url("vllm") == "http://localhost:8000/v1"


class TestLocalModelSpec:
    def test_ollama_defaults(self):
        spec = LocalModelSpec(name="qwen3:8b")
        assert spec.backend == "ollama"
        assert spec.base_url == "http://localhost:11434/v1"
        assert spec.api_key == DEFAULT_OLLAMA_API_KEY  # 占位密钥，服务端不校验
        assert spec.context_length == 4096  # Ollama 默认上下文窗口
        assert spec.keep_alive == "5m"  # Ollama 默认常驻时长

    def test_vllm_defaults(self):
        spec = LocalModelSpec(name="Qwen/Qwen3-8B", backend="vllm")
        assert spec.base_url == "http://localhost:8000/v1"
        assert spec.api_key == DEFAULT_VLLM_API_KEY

    def test_endpoint_description(self):
        assert LocalModelSpec(name="m").endpoint == "ollama@http://localhost:11434/v1"

    def test_rejects_unknown_backend(self):
        with pytest.raises(ValueError, match="未知的本地后端"):
            LocalModelSpec(name="m", backend="lmstudio")

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError, match="模型名不能为空"):
            LocalModelSpec(name="")

    def test_rejects_non_positive_context(self):
        with pytest.raises(ValueError, match="context_length"):
            LocalModelSpec(name="m", context_length=0)


class TestLocalModel:
    def test_properties_expose_deployment_facts(self):
        model = LocalModel(
            name="qwen3:8b",
            context_length=8192,
            keep_alive="-1",
            supports_vision=True,
            client=StubClient(),
        )
        assert model.backend == "ollama"
        assert model.model_name == "qwen3:8b"
        assert model.base_url == "http://localhost:11434/v1"
        assert model.context_length == 8192
        assert model.keep_alive == "-1"
        assert model.supports_vision is True

    def test_describe_returns_loggable_profile(self):
        model = LocalModel(name="qwen3:8b", client=StubClient())
        assert model.describe() == {
            "name": "qwen3:8b",
            "backend": "ollama",
            "base_url": "http://localhost:11434/v1",
            "context_length": 4096,
            "keep_alive": "5m",
            "supports_vision": False,
        }

    def test_is_available_true(self):
        client = StubClient(ids=["qwen3:8b"])
        model = LocalModel(name="qwen3:8b", client=client)
        assert model.is_available(timeout=1.5) is True
        assert client.models.timeouts == [1.5]

    def test_is_available_false_on_connection_error(self):
        """探活失败返回 False 而不是抛异常——调用方据此降级到云端."""
        model = LocalModel(name="m", client=StubClient(list_error=ConnectionError("拒绝连接")))
        assert model.is_available() is False

    def test_list_served_models(self):
        model = LocalModel(name="m", client=StubClient(ids=["qwen3:8b", "nomic-embed-text:latest"]))
        assert model.list_served_models() == ["qwen3:8b", "nomic-embed-text:latest"]

    def test_list_served_models_raises_with_endpoint_hint(self):
        model = LocalModel(name="m", client=StubClient(list_error=TimeoutError("超时")))
        with pytest.raises(RuntimeError, match="无法访问本地 ollama 服务"):
            model.list_served_models()

    def test_chat_uses_openai_compatible_protocol(self):
        client = StubClient(queue=[_StubMessage(content="本地回答")])
        model = LocalModel(name="qwen3:8b", client=client)
        reply = model.chat([Message(role="user", content="你好")])
        assert reply == "本地回答"
        assert client.completions.calls[0]["model"] == "qwen3:8b"
        assert client.completions.calls[0]["messages"] == [{"role": "user", "content": "你好"}]

    def test_stream_yields_increments(self):
        model = LocalModel(name="m", client=StubClient())
        assert list(model.stream([Message(role="user", content="hi")])) == ["本地", "流式"]

    def test_chat_with_tools_normalizes_tool_calls(self):
        message = _StubMessage(
            content=None,
            tool_calls=[_StubToolCall("call_0", "calculator", '{"expression": "1+1"}')],
        )
        client = StubClient(queue=[message])
        model = LocalModel(name="m", client=client)
        result = model.chat_with_tools(
            [Message(role="user", content="算一下")],
            tools=[{"type": "function", "function": {"name": "calculator"}}],
        )
        assert result["content"] is None
        assert result["tool_calls"] == [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'},
            }
        ]
        assert "tools" in client.completions.calls[0]

    def test_chat_with_tools_omits_empty_tool_calls(self):
        client = StubClient(queue=[_StubMessage(content="直接回答")])
        model = LocalModel(name="m", client=client)
        result = model.chat_with_tools([Message(role="user", content="你好")], tools=[])
        assert result == {"content": "直接回答"}


class TestCreateLocalModel:
    def test_uses_settings_defaults(self, monkeypatch: pytest.MonkeyPatch):
        from smart_research_agent.config import settings

        monkeypatch.setattr(settings, "local_model", "qwen3:8b")
        monkeypatch.setattr(settings, "local_base_url", "http://localhost:11434")
        monkeypatch.setattr(settings, "local_keep_alive", "1h")
        model = create_local_model(client=StubClient())
        # 工厂负责把 settings 映射成 spec，并补齐漏写的 /v1
        assert model.model_name == "qwen3:8b"
        assert model.base_url == "http://localhost:11434/v1"
        assert model.keep_alive == "1h"

    def test_explicit_arguments_win(self):
        model = create_local_model(
            "Qwen/Qwen3-8B",
            backend="vllm",
            base_url="http://127.0.0.1:9000/v1",
            supports_vision=False,
            context_length=32768,
            keep_alive="-1",
            client=StubClient(),
        )
        assert model.backend == "vllm"
        assert model.base_url == "http://127.0.0.1:9000/v1"
        assert model.context_length == 32768
        assert model.spec.api_key == DEFAULT_VLLM_API_KEY
