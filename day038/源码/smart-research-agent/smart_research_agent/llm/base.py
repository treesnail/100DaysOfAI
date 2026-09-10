"""LLM 调用层抽象基类."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Message:
    """一条对话消息.

    role 取值: "system" | "user" | "assistant" | "tool"
    （"tool" 自 day037 起用于 function calling 的工具结果回填）
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class BaseLLM(ABC):
    """大模型调用的统一抽象.

    Agent 层只依赖 BaseLLM，不关心底层是哪家模型（依赖倒置原则）。
    """

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """发送对话消息，返回模型的文本回复."""

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """发送对话消息与可用工具规格，返回 OpenAI 兼容的结构化响应.

        返回 dict 形如 {"content": str | None, "tool_calls": [...]}：
        模型决定调用工具时填充 tool_calls，直接回答时填充 content。
        默认实现抛 NotImplementedError——不支持 function calling 的模型
        （或纯文本路线的旧实现）应继续使用 ReAct 文本协议（chat）。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 不支持 function calling，请改用 chat 走文本协议"
        )
