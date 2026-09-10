"""LLM 调用层抽象基类."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
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

    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """逐片段流式返回文本回复（day039）.

        默认实现把 ``chat`` 的完整回复作为**一个**片段 yield——对不支持真
        流式的实现而言，协议上仍是流式、内容不缺失。支持逐 token 增量返回
        的客户端（OpenAI 兼容、MockLLM）覆写本方法，把长回复切成一小串
        片段，让上层 SSE 可以边生成边下发，实现"首字秒出"的体验。

        子类覆写契约：yield 出的片段按顺序拼接，必须等于一次 chat 的完整
        回复（保证流式与一次性在语义上等价）。
        """
        yield self.chat(messages, temperature=temperature, max_tokens=max_tokens)

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