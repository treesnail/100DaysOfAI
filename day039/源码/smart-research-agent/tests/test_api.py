"""FastAPI 服务层测试（day038）：全部走 TestClient，离线、不起真实服务.

TestClient 基于 httpx 在进程内直接调用 ASGI 应用，覆盖完整的
HTTP 协议路径（路由匹配 -> 请求体校验 -> 依赖注入 -> 响应序列化），
但不需要监听端口、不需要网络。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from smart_research_agent.agent.fc_agent import FunctionCallingAgent
from smart_research_agent.api.app import API_VERSION, create_app
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry


def make_client(
    llm: MockLLM | None = None,
    agent_factory=None,
    models: dict | None = None,
    raise_server_exceptions: bool = False,
) -> TestClient:
    """组装一个注入 MockLLM 的测试客户端.

    models 是 ``model`` 参数的按名查找表（day039）；None 时走 create_app
    的 default_models()（离线环境只含 "mock" 占位模型）。

    raise_server_exceptions=False：让未捕获异常走全局异常处理器返回 500，
    而不是在测试里直接抛出（生产服务器的真实行为）。
    """
    app = create_app(llm=llm or MockLLM(), agent_factory=agent_factory, models=models)
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def scripted_agent_factory(tool_call_responses: list[dict]):
    """生成注入脚本化 MockLLM 的 Agent 工厂（每个请求独立 Agent 实例）."""

    def factory() -> FunctionCallingAgent:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        llm = MockLLM(tool_call_responses=list(tool_call_responses))
        return FunctionCallingAgent(llm, registry)

    return factory


class TestHealth:
    def test_health_ok(self):
        resp = make_client().get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "version": API_VERSION}


class TestChat:
    def test_chat_returns_reply(self):
        llm = MockLLM(responses=["你好，我是智研助手"])
        resp = make_client(llm=llm).post("/chat", json={"message": "你好"})
        assert resp.status_code == 200
        assert resp.json() == {"reply": "你好，我是智研助手"}

    def test_chat_respects_temperature_field(self):
        llm = MockLLM(candidates=["低温回答"])
        resp = make_client(llm=llm).post(
            "/chat", json={"message": "hi", "temperature": 0.2}
        )
        assert resp.status_code == 200
        assert resp.json()["reply"] == "低温回答"


class TestValidation:
    """Pydantic 校验层：不合法的请求在路由函数之前被 422 拦截."""

    @pytest.mark.parametrize(
        "payload",
        [
            {},  # 缺必填字段 message
            {"message": ""},  # 违反 min_length=1
            {"message": 123},  # 类型错误：int 不能隐式转 str
            {"message": "hi", "temperature": 3.0},  # 超出 le=2.0 上限
        ],
    )
    def test_chat_invalid_payload_422(self, payload):
        resp = make_client().post("/chat", json=payload)
        assert resp.status_code == 422
        # 422 响应体遵循 FastAPI 统一结构：{"detail": [ {loc, msg, type}, ... ]}
        detail = resp.json()["detail"]
        assert isinstance(detail, list) and detail
        assert "loc" in detail[0] and "msg" in detail[0]

    def test_agent_run_invalid_max_steps_422(self):
        resp = make_client().post(
            "/agent/run", json={"task": "做点事", "max_steps": 0}
        )
        assert resp.status_code == 422

    def test_invalid_json_body_422(self):
        resp = make_client().post(
            "/chat", content="{not json", headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 422


class TestAgentRun:
    def test_full_flow_with_tool_call(self):
        """完整流程：调 calculator -> 回填结果 -> 模型给出最终答案."""
        factory = scripted_agent_factory(
            [
                {"tool_calls": [{"name": "calculator", "arguments": {"expression": "2 + 3 * 4"}}]},
                {"content": "计算结果是 14"},
            ]
        )
        resp = make_client(agent_factory=factory).post(
            "/agent/run", json={"task": "计算 2 + 3 * 4"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "计算结果是 14"
        assert body["steps"] == 2  # 两轮 LLM 调用：决策 + 回答
        assert body["tool_calls"] == [
            {
                "name": "calculator",
                "arguments": {"expression": "2 + 3 * 4"},
                "result": "14",
                "success": True,
            }
        ]

    def test_direct_answer_without_tool_call(self):
        factory = scripted_agent_factory([{"content": "巴黎是法国首都"}])
        resp = make_client(agent_factory=factory).post(
            "/agent/run", json={"task": "法国首都是哪"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "巴黎是法国首都"
        assert body["tool_calls"] == []

    def test_max_steps_forwarded_to_agent(self):
        """请求里的 max_steps 透传给 Agent，超限任务按限停止."""
        # 脚本只给工具调用、永远不给最终答案，逼 Agent 走到步数上限
        factory = scripted_agent_factory(
            [{"tool_calls": [{"name": "calculator", "arguments": {"expression": "1+1"}}]}] * 5
        )
        resp = make_client(agent_factory=factory).post(
            "/agent/run", json={"task": "无限计算", "max_steps": 2}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "最大步数" in body["answer"]
        assert body["steps"] == 2

    def test_agent_exception_returns_500_error_response(self):
        """Agent 内部抛错 -> 全局异常处理器 -> 500 + ErrorResponse 结构."""

        def exploding_factory():
            raise RuntimeError("registry exploded")

        resp = make_client(agent_factory=exploding_factory).post(
            "/agent/run", json={"task": "会失败的任务"}
        )
        assert resp.status_code == 500
        body = resp.json()
        assert body["error_type"] == "RuntimeError"
        assert "registry exploded" in body["detail"]


class TestOpenAPIContract:
    """Pydantic 模型的第二重角色：自动生成 API 文档."""

    def test_openapi_schema_contains_endpoints_and_models(self):
        resp = make_client().get("/openapi.json")
        assert resp.status_code == 200
        spec = resp.json()
        assert set(spec["paths"]) >= {"/health", "/chat", "/agent/run"}
        # 请求/响应模型进入 components/schemas，供 /docs 与客户端代码生成使用
        schemas = spec["components"]["schemas"]
        for model in ("ChatRequest", "ChatResponse", "AgentRunRequest", "AgentRunResponse"):
            assert model in schemas


def _parse_sse_deltas(body: str) -> list[str]:
    """把 SSE 响应体解析为 delta 片段列表（忽略 [DONE] 哨兵）."""
    deltas: list[str] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            continue
        deltas.append(json.loads(payload)["delta"])
    return deltas


class TestChatStream:
    """day039：SSE 流式输出端点."""

    def test_stream_returns_sse_content_type(self):
        llm = MockLLM(responses=["你好，我是智研助手"])
        resp = make_client(llm=llm).post("/chat/stream", json={"message": "你好"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

    def test_stream_chunks_reassemble_to_full_reply(self):
        llm = MockLLM(responses=["你好，我是智研助手"])  # 9 字符，按 4 字切分
        resp = make_client(llm=llm).post("/chat/stream", json={"message": "你好"})
        deltas = _parse_sse_deltas(resp.text)
        assert len(deltas) > 1  # 确实分成了多片
        assert "".join(deltas) == "你好，我是智研助手"

    def test_stream_equivalent_to_one_shot(self):
        """流式与一次性在语义上等价：拼接结果 == /chat 的完整回复."""
        reply = "流式与一次性应等价"
        one_shot = make_client(llm=MockLLM(responses=[reply])).post(
            "/chat", json={"message": "hi"}
        )
        streamed = make_client(llm=MockLLM(responses=[reply])).post(
            "/chat/stream", json={"message": "hi"}
        )
        assert one_shot.json()["reply"] == reply
        assert "".join(_parse_sse_deltas(streamed.text)) == reply

    def test_stream_invalid_payload_422(self):
        resp = make_client().post("/chat/stream", json={"message": ""})
        assert resp.status_code == 422


class TestModelSwitch:
    """day039：``model`` 参数显式切换 + ``/models`` 枚举."""

    def test_list_models(self):
        client = make_client(models={"mock": MockLLM(), "fast": MockLLM()})
        resp = client.get("/models")
        assert resp.status_code == 200
        assert resp.json() == {"models": ["fast", "mock"]}

    def test_chat_with_explicit_model(self):
        client = make_client(models={"fast": MockLLM(default="快速模型的回复")})
        resp = client.post("/chat", json={"message": "hi", "model": "fast"})
        assert resp.status_code == 200
        assert resp.json() == {"reply": "快速模型的回复"}

    def test_stream_with_explicit_model(self):
        client = make_client(models={"fast": MockLLM(responses=["流式快速回复"])})
        resp = client.post("/chat/stream", json={"message": "hi", "model": "fast"})
        assert resp.status_code == 200
        assert "".join(_parse_sse_deltas(resp.text)) == "流式快速回复"

    def test_default_model_when_not_specified(self):
        """不传 model 时走注入的默认 llm，而非注册表里的模型."""
        llm = MockLLM(default="默认模型的回复")
        client = make_client(llm=llm, models={"fast": MockLLM(default="不应命中")})
        resp = client.post("/chat", json={"message": "hi"})
        assert resp.json() == {"reply": "默认模型的回复"}

    def test_unknown_model_returns_404(self):
        client = make_client(models={"fast": MockLLM()})
        resp = client.post("/chat", json={"message": "hi", "model": "nope"})
        assert resp.status_code == 404
        assert "未知模型" in resp.json()["detail"]
        assert "fast" in resp.json()["detail"]
