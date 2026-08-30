"""SmartResearch MCP Server：基于 FastMCP 的能力暴露服务.

提供一个 ``calculator`` 工具与 ``research://knowledge/{doc_id}`` 资源，
支持 stdio / sse 两种传输方式。SSE 模式下额外暴露 ``/health`` 健康检查路由，
供容器编排（Docker HEALTHCHECK / docker-compose healthcheck）探测存活状态。

用法::

    # 本地开发：stdio 传输（由 MCP Client 以子进程方式拉起）
    python -m smart_research_agent.mcp_server.server --transport stdio

    # 容器部署：SSE 传输，监听 8000 端口
    python -m smart_research_agent.mcp_server.server --transport sse
"""

from __future__ import annotations

import argparse
import os
from typing import Literal

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from smart_research_agent.mcp_server.protocol import ResourceDescriptor, ToolDescriptor
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 内置知识库：doc_id -> 文档内容。资源处理器从这里读取。
KNOWLEDGE_BASE: dict[str, str] = {
    "mcp-intro": "MCP（Model Context Protocol）是 Anthropic 提出的开放协议，"
    "用于标准化 LLM 应用与外部数据源、工具之间的连接。",
    "docker-layers": "Docker 镜像由只读层叠加而成，每条 Dockerfile 指令生成一层，"
    "层内容不变即可命中构建缓存。",
    "twelve-factor": "十二要素应用方法论要求：配置存环境变量、进程无状态、"
    "通过端口绑定对外提供服务。",
}

#: SSE 传输的默认监听端口，可用环境变量 MCP_PORT 覆盖。
DEFAULT_SSE_PORT = 8000

_calculator = CalculatorTool()


def create_server(
    host: str | None = None,
    port: int | None = None,
) -> FastMCP:
    """创建并配置 FastMCP Server 实例.

    host / port 仅对 SSE 传输有意义，默认读取环境变量 ``MCP_HOST`` / ``MCP_PORT``
    （容器内通过 ENV 或 docker-compose 的 environment 注入），
    缺省为 ``0.0.0.0:8000``——容器里必须监听 0.0.0.0 才能接受宿主机转发的请求。
    """
    server = FastMCP(
        name="smart-research-agent",
        instructions="智研 AI 助手的 MCP Server：提供计算工具与研究知识库资源。",
        host=host or os.environ.get("MCP_HOST", "0.0.0.0"),
        port=port or int(os.environ.get("MCP_PORT", str(DEFAULT_SSE_PORT))),
    )

    @server.tool()
    def calculator(expression: str) -> str:
        """计算数学表达式，支持 + - * / ** % 与括号，例如 '2 + 3 * (4 - 1)'."""
        logger.info("MCP tool 调用: calculator(%s)", expression)
        return _calculator.execute(expression=expression)

    @server.resource("research://knowledge/{doc_id}")
    def knowledge(doc_id: str) -> str:
        """按 doc_id 读取内置研究知识库文档."""
        logger.info("MCP resource 读取: research://knowledge/%s", doc_id)
        if doc_id not in KNOWLEDGE_BASE:
            raise ValueError(f"未知文档: {doc_id}（可用: {sorted(KNOWLEDGE_BASE)}）")
        return KNOWLEDGE_BASE[doc_id]

    @server.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> Response:
        """健康检查端点：仅 SSE/HTTP 传输下可达，供容器健康检查使用."""
        return JSONResponse({"status": "ok", "server": "smart-research-agent"})

    return server


def describe_capabilities() -> dict[str, list]:
    """用 protocol.py 的描述模型导出本 Server 的能力清单.

    这份清单与 FastMCP 自动生成的协议元数据对应，但使用项目自己的
    pydantic 模型表达，便于写文档、做协议级断言。
    """
    return {
        "tools": [
            ToolDescriptor(
                name="calculator",
                description=_calculator.description,
                input_schema=_calculator.parameters,
            ).model_dump()
        ],
        "resources": [
            ResourceDescriptor(
                uri="research://knowledge/{doc_id}",
                name="knowledge",
                description="内置研究知识库文档，按 doc_id 读取",
                mime_type="text/plain",
            ).model_dump()
        ],
        "prompts": [],
    }


def main(argv: list[str] | None = None) -> None:
    """命令行入口：解析 --transport 并启动 Server."""
    parser = argparse.ArgumentParser(description="SmartResearch MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default=os.environ.get("MCP_TRANSPORT", "stdio"),
        help="传输方式：stdio 用于本地子进程集成，sse 用于容器化网络部署",
    )
    args = parser.parse_args(argv)
    transport: Literal["stdio", "sse"] = args.transport

    server = create_server()
    logger.info("启动 MCP Server，传输方式: %s", transport)
    if transport == "sse":
        logger.info("SSE 监听地址: http://%s:%d", server.settings.host, server.settings.port)
    server.run(transport=transport)


if __name__ == "__main__":
    main()
