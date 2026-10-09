"""day038 编程题参考：FastAPI 服务端到端离线演示.

运行方式（项目根目录）：
    python scripts/api_demo.py
全程离线：用 fastapi.testclient.TestClient 在进程内调用 ASGI 应用，
LLM 由 MockLLM 脚本扮演，不监听端口、不访问网络。

真起服务的等价命令（教程第五章）：
    uvicorn smart_research_agent.api.app:create_app --factory
或：
    python -m smart_research_agent.api.app
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from smart_research_agent.agent.fc_agent import FunctionCallingAgent
from smart_research_agent.api.app import create_app
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry


def show(title: str, status: int, body: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"HTTP {status}")
    print(json.dumps(body, ensure_ascii=False, indent=2))


def main() -> None:
    # --- /health ---
    client = TestClient(create_app(llm=MockLLM(responses=["你好，我是智研助手"])))
    resp = client.get("/health")
    show("GET /health", resp.status_code, resp.json())
    assert resp.status_code == 200 and resp.json()["status"] == "ok"

    # --- /chat 正常 ---
    resp = client.post("/chat", json={"message": "你好"})
    show("POST /chat", resp.status_code, resp.json())
    assert resp.json()["reply"] == "你好，我是智研助手"

    # --- /chat 缺字段 -> 422 ---
    resp = client.post("/chat", json={})
    show("POST /chat（缺 message 字段）", resp.status_code, resp.json())
    assert resp.status_code == 422

    # --- /agent/run 完整工具调用流程 ---
    def agent_factory() -> FunctionCallingAgent:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        llm = MockLLM(
            tool_call_responses=[
                {"tool_calls": [{"name": "calculator", "arguments": {"expression": "1024 * 4"}}]},
                {"content": "1024 乘以 4 等于 4096。"},
            ]
        )
        return FunctionCallingAgent(llm, registry)

    client = TestClient(create_app(agent_factory=agent_factory))
    resp = client.post("/agent/run", json={"task": "帮我算 1024 乘以 4"})
    show("POST /agent/run（带工具调用）", resp.status_code, resp.json())
    assert resp.json()["answer"] == "1024 乘以 4 等于 4096。"
    assert resp.json()["tool_calls"][0]["result"] == "4096"

    # --- Agent 抛错 -> 500 ErrorResponse ---
    def exploding_factory():
        raise RuntimeError("工具注册表初始化失败")

    client = TestClient(
        create_app(agent_factory=exploding_factory), raise_server_exceptions=False
    )
    resp = client.post("/agent/run", json={"task": "会失败的任务"})
    show("POST /agent/run（Agent 抛错）", resp.status_code, resp.json())
    assert resp.status_code == 500 and resp.json()["error_type"] == "RuntimeError"

    print("\n全部断言通过：三个端点的正常路径、422 校验、500 错误契约均符合预期。")


if __name__ == "__main__":
    main()
