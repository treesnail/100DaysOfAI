"""演示用 MCP Server：以 stdio 传输运行，供 McpClient 连接演示.

运行方式::

    python -m smart_research_agent.mcp_server.demo_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP


def build_demo_server() -> FastMCP:
    """构建演示 Server（测试中也复用此工厂，保证 Client/Server 契约一致）."""
    app = FastMCP("smart-research-demo")

    @app.tool()
    def search_docs(query: str) -> str:
        """在研究资料库中检索与 query 相关的文档摘要."""
        return f"[检索结果] 与「{query}」相关的文档共 2 篇：MCP 入门、ReAct 实战"

    @app.tool()
    def word_count(text: str) -> str:
        """统计文本的词数（按空白切分）."""
        return f"词数: {len(text.split())}"

    @app.tool()
    def unstable_tool(trigger: str) -> str:
        """演示用不稳定工具：总是抛出异常，用于验证异常回退."""
        raise RuntimeError(f"服务内部错误（trigger={trigger}）")

    @app.resource("docs://readme")
    def readme() -> str:
        return "# smart-research-agent 研究资料库"

    @app.prompt()
    def research_report(topic: str) -> str:
        """生成调研报告大纲模板."""
        return f"请围绕「{topic}」写一份调研报告：背景、现状、对比、结论。"

    return app


if __name__ == "__main__":
    build_demo_server().run(transport="stdio")
