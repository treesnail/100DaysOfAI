"""完整的 FastMCP Server：智研 AI 助手的 MCP 服务端.

整合三类能力：

- **Resources**：``research://knowledge/{doc_id}`` 读取本地知识库文档；
- **Tools**：``calculate`` 包装 day004 的 CalculatorTool；
- **Prompts**：``research_report(topic)`` 调研报告提示词模板。

入口支持两种传输方式::

    python -m smart_research_agent.mcp_server.full_server --transport stdio
    python -m smart_research_agent.mcp_server.full_server --transport sse
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from smart_research_agent.mcp_server.protocol import (
    PromptDescriptor,
    ResourceDescriptor,
    ToolDescriptor,
)
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

SERVER_NAME = "smart-research-agent"
SERVER_INSTRUCTIONS = (
    "智研 AI 助手的 MCP 服务端。提供三类能力：\n"
    "1. Resources：通过 research://knowledge/{doc_id} 读取本地调研知识库；\n"
    "2. Tools：calculate 执行安全的算术计算；\n"
    "3. Prompts：research_report 生成结构化的调研报告提示词。"
)

_KNOWLEDGE_DIR = Path(__file__).resolve().parent / "data" / "knowledge"
_SAFE_DOC_ID = re.compile(r"^[a-z0-9_]+$")

_calculator = CalculatorTool()


def create_server() -> FastMCP:
    """创建并装配完整的 FastMCP Server 实例.

    工厂函数而非模块级单例：测试可以独立创建实例，互不污染。
    """
    app = FastMCP(
        name=SERVER_NAME,
        instructions=SERVER_INSTRUCTIONS,
    )

    @app.resource("research://knowledge/{doc_id}")
    def read_knowledge(doc_id: str) -> str:
        """按 doc_id 读取本地知识库中的 Markdown 文档."""
        if not _SAFE_DOC_ID.match(doc_id):
            raise ValueError(f"非法的 doc_id: {doc_id!r}，仅允许小写字母、数字与下划线")
        path = _KNOWLEDGE_DIR / f"{doc_id}.md"
        if not path.is_file():
            raise ValueError(f"知识库文档不存在: {doc_id}")
        return path.read_text(encoding="utf-8")

    @app.tool()
    def calculate(expression: str) -> str:
        """计算数学表达式，支持 + - * / ** % 与括号，例如 '2 + 3 * (4 - 1)'."""
        try:
            result = _calculator.execute(expression=expression)
        except Exception as exc:  # 兜底：任何意外异常都不能让 Server 崩溃
            logger.exception("calculate 工具内部错误")
            raise ToolError(f"计算工具内部错误: {exc}") from exc
        if result.startswith("计算失败"):
            # CalculatorTool 用字符串表达业务失败；转成 ToolError，
            # 让 MCP 层以 isError=true 的结构化错误返回给客户端。
            raise ToolError(result)
        return result

    @app.prompt()
    def research_report(topic: str) -> str:
        """生成一份针对指定主题的结构化调研报告提示词."""
        return (
            f"你是一名资深行业研究员。请围绕主题「{topic}」撰写一份调研报告，要求：\n"
            "1. 先给出 3~5 个调研维度的提纲；\n"
            "2. 逐维度展开分析，引用知识库资料（research://knowledge/...）；\n"
            "3. 涉及数字对比时用 calculate 工具核算；\n"
            "4. 最后给出结论与选型建议。"
        )

    return app


def capability_descriptors() -> dict[str, list]:
    """用 protocol.py 的描述符模型静态声明本 Server 的能力清单.

    与运行时 ``resources/list`` 等接口返回的内容对应，供文档生成与
    客户端预检使用。
    """
    return {
        "resources": [
            ResourceDescriptor(
                uri="research://knowledge/{doc_id}",
                name="read_knowledge",
                description="按 doc_id 读取本地知识库中的 Markdown 文档",
                mime_type="text/markdown",
            )
        ],
        "tools": [
            ToolDescriptor(
                name="calculate",
                description="计算数学表达式",
                input_schema=_calculator.parameters,
            )
        ],
        "prompts": [
            PromptDescriptor(
                name="research_report",
                description="生成一份针对指定主题的结构化调研报告提示词",
                arguments=[{"name": "topic", "description": "调研主题", "required": True}],
            )
        ],
    }


def main() -> int:  # pragma: no cover - 入口启动逻辑，由手动运行验证
    """解析命令行参数并启动 MCP Server."""
    parser = argparse.ArgumentParser(description="智研 AI 助手 MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="传输方式：stdio（本地子进程，默认）或 sse（HTTP 网络服务）",
    )
    args = parser.parse_args()

    app = create_server()
    if args.transport == "stdio":
        # stdio：由客户端以子进程方式拉起，通过标准输入输出收发 JSON-RPC。
        # stdout 是协议通道，任何日志都会污染它，必须把控制台日志让路到 stderr。
        for handler in logging.getLogger().handlers + logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stdout:
                handler.setStream(sys.stderr)
        logger.info("以 stdio 传输启动 MCP Server")
        app.run(transport="stdio")
    else:
        # sse：启动 HTTP 服务，默认监听 127.0.0.1:8000，端点 /sse 与 /messages/。
        # 供远程或多客户端场景使用；测试不真正启动 SSE 服务。
        logger.info("以 SSE 传输启动 MCP Server (http://127.0.0.1:8000/sse)")
        app.run(transport="sse")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
