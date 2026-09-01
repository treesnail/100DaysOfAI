"""MCP Server：用 FastMCP 把 SmartResearch Agent 的能力暴露为 MCP 原语.

暴露的能力：
  - Tool ``calculator``       ：包装 day004 的 CalculatorTool（复用其 AST 白名单求值）
  - Tool ``knowledge_search`` ：基于 day009 内存向量库的知识检索
  - Prompt ``research_report``：参数化调研报告提示词模板

运行方式（stdio 传输，由 MCP Client 以子进程启动）::

    python -m smart_research_agent.mcp_server.server
"""

from __future__ import annotations

import hashlib

from mcp.server.fastmcp import FastMCP

from smart_research_agent.memory.vector_store import InMemoryVectorStore, VectorRecord
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

app = FastMCP("smart-research")

# ---------------------------------------------------------------------------
# Tool 1: calculator —— 包装既有 CalculatorTool，复用而非重写计算逻辑
# ---------------------------------------------------------------------------

_calculator = CalculatorTool()


@app.tool(name=_calculator.name, description=_calculator.description)
def calculator(expression: str) -> str:
    """计算数学表达式，支持 + - * / ** % 与括号，例如 '2 + 3 * (4 - 1)'."""
    result = _calculator.execute(expression=expression)
    logger.info("calculator(%r) = %s", expression, result)
    return result


# ---------------------------------------------------------------------------
# Tool 2: knowledge_search —— 基于内存向量库的知识检索
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


_KNOWLEDGE_BASE: list[tuple[str, str]] = [
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
    """把内置知识库灌入内存向量库."""
    store = InMemoryVectorStore()
    for record_id, text in _KNOWLEDGE_BASE:
        store.add(VectorRecord(id=record_id, text=text, vector=_toy_embed(text)))
    return store


_knowledge_store = _build_knowledge_store()


@app.tool(
    name="knowledge_search",
    description="在 SmartResearch Agent 内置知识库中做语义检索，返回与查询最相关的知识条目",
)
def knowledge_search(query: str, top_k: int = 3) -> str:
    """检索与 query 最相关的 top_k 条知识，按相似度降序返回."""
    hits = _knowledge_store.search(_toy_embed(query), top_k=top_k)
    if not hits:
        return "知识库为空"
    lines = [f"[{i}] (相似度 {score:.3f}) {record.text}" for i, (record, score) in enumerate(hits, 1)]
    result = "\n".join(lines)
    logger.info("knowledge_search(%r) 命中 %d 条", query, len(hits))
    return result


# ---------------------------------------------------------------------------
# Prompt: research_report —— 参数化提示词模板，用户可控原语
# ---------------------------------------------------------------------------

RESEARCH_REPORT_TEMPLATE = """你是一名资深行业研究员。请围绕主题「{topic}」撰写一份调研报告。

要求：
1. 深度：{depth}
2. 结构：背景与定义 → 现状与主流方案 → 对比分析 → 结论与建议
3. 每个论点尽量给出可查证的事实依据，避免空泛表述
4. 调用 knowledge_search 工具检索已有知识，调用 calculator 工具完成必要计算

请先列出写作提纲，再逐节展开。"""


@app.prompt()
def research_report(topic: str, depth: str = "详细") -> str:
    """调研报告写作提示词：topic 与 depth 由调用方动态填充.

    Args:
        topic: 调研主题，例如 "RAG 技术选型"
        depth: 调研深度，默认 "详细"，也可为 "简要"
    """
    return RESEARCH_REPORT_TEMPLATE.format(topic=topic, depth=depth)


def main() -> None:  # pragma: no cover
    """以 stdio 传输启动 MCP Server（供 MCP Client 以子进程拉起）."""
    app.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
