"""SmartResearch MCP 资源服务器：用 FastMCP 暴露知识库文档与项目配置.

本模块是 day017 的核心产出：把项目的静态数据（data/knowledge/ 下的
Markdown 知识文档、Settings 中的白名单配置项）以 MCP Resources 原语
暴露给任意 MCP 客户端。

运行方式（stdio 传输，供 Claude Desktop 等宿主连接）::

    python -m smart_research_agent.mcp_server.resources_server

测试则通过 ``mcp.shared.memory.create_connected_server_and_client_session``
在进程内直连，无需启动子进程（见 tests/test_mcp_resources.py）。
"""

from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from smart_research_agent.config import settings
from smart_research_agent.mcp_server.protocol import ResourceDescriptor
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

# 项目根目录：本文件位于 smart_research_agent/mcp_server/resources_server.py，
# 向上三级即项目根（data/ 与 pyproject.toml 所在处）。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"

KNOWLEDGE_URI_TEMPLATE = "research://knowledge/{doc_id}"
CONFIG_URI_TEMPLATE = "research://config/{key}"

# 允许通过 research://config/{key} 暴露的配置键白名单。
# 关键安全决策：Settings 含 openai_api_key 等敏感字段，绝不整体暴露，
# 只放行明确无敏感的键。
CONFIG_WHITELIST = ("project_name", "default_model", "log_level", "debug")

app = FastMCP("smart-research")


class KnowledgeNotFoundError(ValueError):
    """请求的知识库文档不存在或 doc_id 非法."""


def load_knowledge_doc(doc_id: str) -> str:
    """从 data/knowledge/ 读取 Markdown 文档全文.

    Args:
        doc_id: 文档标识，对应 data/knowledge/{doc_id}.md。

    Raises:
        KnowledgeNotFoundError: doc_id 含路径分隔符（防目录穿越）或文件不存在。
    """
    if "/" in doc_id or "\\" in doc_id or doc_id.startswith("."):
        raise KnowledgeNotFoundError(f"非法的 doc_id: {doc_id!r}")
    path = KNOWLEDGE_DIR / f"{doc_id}.md"
    if not path.is_file():
        raise KnowledgeNotFoundError(f"知识库文档不存在: {doc_id!r}（应为 {path.name}）")
    logger.info("读取知识库文档: %s", path)
    return path.read_text(encoding="utf-8")


def load_config_item(key: str) -> str:
    """读取白名单内的配置项并转为字符串.

    Raises:
        KnowledgeNotFoundError: 键不在白名单（含不存在的键与被屏蔽的敏感键）。
    """
    if key not in CONFIG_WHITELIST:
        raise KnowledgeNotFoundError(f"配置项不可暴露或不存在: {key!r}")
    return str(getattr(settings, key))


@app.resource(
    KNOWLEDGE_URI_TEMPLATE,
    name="knowledge_doc",
    description="知识库 Markdown 文档，按 doc_id 读取 data/knowledge/{doc_id}.md",
    mime_type="text/markdown",
)
def knowledge_doc(doc_id: str) -> str:
    """research://knowledge/{doc_id} 资源的读取处理器."""
    return load_knowledge_doc(doc_id)


@app.resource(
    CONFIG_URI_TEMPLATE,
    name="config_item",
    description="项目配置项（白名单内），如 project_name / default_model",
    mime_type="text/plain",
)
def config_item(key: str) -> str:
    """research://config/{key} 资源的读取处理器."""
    return load_config_item(key)


# 项目内自描述清单：与上面注册的资源一一对应，
# 供能力文档生成（day023）与一致性测试消费。
RESOURCE_DESCRIPTORS = [
    ResourceDescriptor(
        uri=KNOWLEDGE_URI_TEMPLATE,
        name="knowledge_doc",
        description="知识库 Markdown 文档，按 doc_id 读取",
        mime_type="text/markdown",
    ),
    ResourceDescriptor(
        uri=CONFIG_URI_TEMPLATE,
        name="config_item",
        description="项目配置项（白名单内）",
        mime_type="text/plain",
    ),
]


def main() -> None:  # pragma: no cover
    """以 stdio 传输启动服务器（被宿主进程以子进程方式拉起）."""
    logger.info("启动 MCP 资源服务器: smart-research (stdio)")
    app.run(transport="stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
