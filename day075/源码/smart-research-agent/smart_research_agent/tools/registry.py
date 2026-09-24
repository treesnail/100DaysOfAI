"""工具注册表：工具的动态注册、校验与发现."""

from __future__ import annotations

from smart_research_agent.tools.base import BaseTool
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """集中管理 Agent 可用的工具."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if not isinstance(tool, BaseTool):
            raise TypeError("只能注册 BaseTool 的实例")
        if tool.name in self._tools:
            raise ValueError(f"工具重复注册: {tool.name}")
        self._tools[tool.name] = tool
        logger.info("注册工具: %s", tool.name)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    def describe(self) -> str:
        """生成给 LLM 阅读的工具清单文本."""
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())
