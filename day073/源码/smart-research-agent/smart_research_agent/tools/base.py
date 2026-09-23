"""工具抽象基类."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """所有 Agent 工具的抽象基类.

    契约：
      - name: 工具唯一标识，Agent 用它路由调用
      - description: 给 LLM 阅读的自然语言描述
      - parameters: JSON Schema 形式的参数说明
      - execute: 执行入口，返回字符串结果
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        ...

    def to_schema(self) -> dict[str, Any]:
        """导出工具 schema，用于拼入 LLM prompt 或 function calling."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
