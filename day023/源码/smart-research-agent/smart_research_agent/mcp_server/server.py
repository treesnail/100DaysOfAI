"""SmartResearch MCP Server：基于 FastMCP 的能力暴露.

向 MCP Client 暴露三类能力：

- Tool：``calculator``，复用 day003 的 AST 安全计算器；
- Resource：``research://knowledge/{doc_id}``，内置知识库文档；
- Prompt：``research_report``，调研报告提示词模板。

启动方式（stdio 传输，供 Client 以子进程方式连接）::

    python -m smart_research_agent.mcp_server.server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from smart_research_agent.tools.calculator import CalculatorTool

# 内置知识库：key 为 doc_id，value 为文档正文。
# 真实项目中这里会接向量库或文档存储，M2 阶段先用内存字典占位。
KNOWLEDGE_BASE: dict[str, str] = {
    "rag-intro": (
        "RAG（检索增强生成）通过先检索相关文档、再把文档作为上下文交给 LLM 生成，"
        "缓解大模型的知识陈旧与幻觉问题。"
    ),
    "mcp-intro": (
        "MCP（Model Context Protocol）是模型与外部能力之间的标准协议，"
        "把能力抽象为 Resources、Tools、Prompts 三类。"
    ),
    "agent-loop": (
        "ReAct 循环让 Agent 在 Thought → Action → Observation 的交替中逐步逼近答案，"
        "直到输出 Final Answer。"
    ),
}

_calculator = CalculatorTool()

app = FastMCP("smart-research-agent")


@app.tool(description="计算数学表达式，支持 + - * / ** % 与括号")
def calculator(expression: str) -> str:
    """计算数学表达式并返回结果字符串."""
    return _calculator.execute(expression=expression)


@app.resource("research://knowledge/{doc_id}", mime_type="text/plain")
def knowledge(doc_id: str) -> str:
    """按 doc_id 读取内置知识库文档."""
    if doc_id not in KNOWLEDGE_BASE:
        raise KeyError(f"知识库中不存在文档: {doc_id}")
    return KNOWLEDGE_BASE[doc_id]


@app.prompt(description="生成一份调研报告的撰写提示词")
def research_report(topic: str, style: str = "正式") -> str:
    """返回调研报告提示词模板，参数由 Client 填入."""
    return (
        f"请以{style}的风格撰写一份关于「{topic}」的调研报告，"
        "包含背景、现状分析、方案对比与结论建议四个部分。"
    )


def main() -> None:
    """以 stdio 传输启动 MCP Server（供 Client 以子进程方式连接）."""
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
