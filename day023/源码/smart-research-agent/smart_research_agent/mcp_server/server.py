"""SmartResearch MCP Server：基于 FastMCP 的能力暴露服务.

暴露的能力（day018 全集 + day022 部署化改造）：
  - Tool ``calculator``       ：包装 day004 的 CalculatorTool（复用其 AST 白名单求值）
  - Tool ``knowledge_search`` ：基于 day009 内存向量库的知识检索
  - Resource ``research://knowledge/{doc_id}``：内置知识库文档，按 doc_id 读取
  - Prompt ``research_report``：参数化调研报告提示词模板

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
import hashlib
import os
from typing import Literal

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from smart_research_agent.mcp_server.protocol import ResourceDescriptor, ToolDescriptor
from smart_research_agent.memory.vector_store import InMemoryVectorStore, VectorRecord
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 内置知识库：doc_id -> 文档内容。资源处理器从这里读取。
#: 真实项目中这里会接向量库或文档存储，现阶段先用内存字典占位。
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
    "docker-layers": (
        "Docker 镜像由只读层叠加而成，每条 Dockerfile 指令生成一层，"
        "层内容不变即可命中构建缓存。"
    ),
    "twelve-factor": (
        "十二要素应用方法论要求：配置存环境变量、进程无状态、"
        "通过端口绑定对外提供服务。"
    ),
}

#: SSE 传输的默认监听端口，可用环境变量 MCP_PORT 覆盖。
DEFAULT_SSE_PORT = 8000

_calculator = CalculatorTool()

# ---------------------------------------------------------------------------
# knowledge_search 工具依赖的内存向量库（day018 引入）
# ---------------------------------------------------------------------------

_EMBED_DIM = 128


def _toy_embed(text: str) -> list[float]:
    """字符二元组哈希向量：离线、确定性的教学版 embedding.

    共享字符二元组的文本会在相同维度上取值，余弦相似度因此能反映
    表层文本重叠。它没有真实语义（"开心"与"高兴"并不相近），生产环境
    应替换为真实 embedding 服务（day009 / M4-D8）。
    """
    vec = [0.0] * _EMBED_DIM
    grams = [text[i : i + 2] for i in range(len(text) - 1)] or [text]
    for gram in grams:
        idx = int.from_bytes(hashlib.md5(gram.encode("utf-8")).digest()[:2], "big") % _EMBED_DIM
        vec[idx] += 1.0
    return vec


_KNOWLEDGE_RECORDS: list[tuple[str, str]] = [
    (
        "kb-react",
        "ReAct 是一种 Agent 推理范式：Thought（思考）与 Action（行动）交替进行，"
        "每一步根据 Observation（观察结果）现场决策下一步动作。",
    ),
    (
        "kb-planner",
        "Plan-and-Execute 架构先用 Planner 把目标拆解为有序子任务列表，"
        "再把每个子任务交给执行层完成，适合长链路、结构可预知的任务。",
    ),
    (
        "kb-vector-store",
        "向量检索把文本编码为 embedding 向量，用余弦相似度度量语义距离，"
        "是长期记忆与 RAG 知识库的基础设施。",
    ),
    (
        "kb-mcp",
        "MCP（Model Context Protocol）把 Host 应用的能力标准化为三类原语："
        "Resources（数据）、Tools（函数）、Prompts（模板），"
        "Client 与 Server 之间通过 JSON-RPC 2.0 消息通信。",
    ),
    (
        "kb-memory",
        "短期记忆保存当前会话的消息列表并按策略截断；长期记忆把关键知识"
        "向量化存入向量库，执行前按需检索注入上下文。",
    ),
]


def _build_knowledge_store() -> InMemoryVectorStore:
    """把内置知识条目灌入内存向量库."""
    store = InMemoryVectorStore()
    for record_id, text in _KNOWLEDGE_RECORDS:
        store.add(VectorRecord(id=record_id, text=text, vector=_toy_embed(text)))
    return store


_knowledge_store = _build_knowledge_store()


# ---------------------------------------------------------------------------
# Prompt 模板：topic 必填，depth / style 可选
# ---------------------------------------------------------------------------

RESEARCH_REPORT_TEMPLATE = """你是一名资深行业研究员。请以{style}的风格，围绕主题「{topic}」撰写一份调研报告。

要求：
1. 深度：{depth}
2. 结构：背景与定义 → 现状与主流方案 → 对比分析 → 结论与建议
3. 每个论点尽量给出可查证的事实依据，避免空泛表述
4. 调用 knowledge_search 工具检索已有知识，调用 calculator 工具完成必要计算

请先列出写作提纲，再逐节展开。"""


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

    @server.tool(name=_calculator.name, description=_calculator.description)
    def calculator(expression: str) -> str:
        """计算数学表达式，支持 + - * / ** % 与括号，例如 '2 + 3 * (4 - 1)'."""
        result = _calculator.execute(expression=expression)
        logger.info("calculator(%r) = %s", expression, result)
        return result

    @server.tool(
        name="knowledge_search",
        description="在 SmartResearch Agent 内置知识库中做语义检索，返回与查询最相关的知识条目",
    )
    def knowledge_search(query: str, top_k: int = 3) -> str:
        """检索与 query 最相关的 top_k 条知识，按相似度降序返回."""
        hits = _knowledge_store.search(_toy_embed(query), top_k=top_k)
        if not hits:
            return "知识库为空"
        lines = [
            f"[{i}] (相似度 {score:.3f}) {record.text}"
            for i, (record, score) in enumerate(hits, 1)
        ]
        result = "\n".join(lines)
        logger.info("knowledge_search(%r) 命中 %d 条", query, len(hits))
        return result

    @server.resource("research://knowledge/{doc_id}", mime_type="text/plain")
    def knowledge(doc_id: str) -> str:
        """按 doc_id 读取内置研究知识库文档."""
        logger.info("MCP resource 读取: research://knowledge/%s", doc_id)
        if doc_id not in KNOWLEDGE_BASE:
            raise ValueError(
                f"未知文档: {doc_id}（知识库中不存在该文档，可用: {sorted(KNOWLEDGE_BASE)}）"
            )
        return KNOWLEDGE_BASE[doc_id]

    @server.prompt(description="生成一份调研报告的撰写提示词")
    def research_report(topic: str, depth: str = "详细", style: str = "正式") -> str:
        """调研报告写作提示词：topic / depth / style 由调用方动态填充.

        Args:
            topic: 调研主题，例如 "RAG 技术选型"
            depth: 调研深度，默认 "详细"，也可为 "简要"
            style: 行文风格，默认 "正式"，也可为 "学术" 等
        """
        return RESEARCH_REPORT_TEMPLATE.format(topic=topic, depth=depth, style=style)

    @server.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> Response:
        """健康检查端点：仅 SSE/HTTP 传输下可达，供容器健康检查使用."""
        return JSONResponse({"status": "ok", "server": "smart-research-agent"})

    return server


#: 模块级默认实例：内存会话测试与 stdio 子进程启动都直接使用它。
app = create_server()


def describe_capabilities() -> dict[str, list]:
    """用 protocol.py 的描述模型导出本 Server 的核心能力清单.

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
