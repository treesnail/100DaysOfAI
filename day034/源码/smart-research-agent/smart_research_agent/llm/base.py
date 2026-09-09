"""LLM 调用层抽象基类."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Message:
    """一条对话消息.

    role 取值: "system" | "user" | "assistant"
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
