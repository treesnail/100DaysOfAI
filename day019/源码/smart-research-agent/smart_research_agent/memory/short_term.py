"""短期记忆：会话内消息列表与截断策略."""

from __future__ import annotations

from smart_research_agent.llm.base import Message


class ShortTermMemory:
    """保存当前会话的短期对话上下文.

    截断策略（超过 max_messages 时）：
      - 始终保留 system 消息
      - 其余消息按 FIFO 丢弃最老的
    """

    def __init__(self, max_messages: int = 20):
        if max_messages < 2:
            raise ValueError("max_messages 至少为 2")
        self.max_messages = max_messages
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        self._truncate()

    def get_all(self) -> list[Message]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def _truncate(self) -> None:
        if len(self._messages) <= self.max_messages:
            return
        system_msgs = [m for m in self._messages if m.role == "system"]
        others = [m for m in self._messages if m.role != "system"]
        keep = max(self.max_messages - len(system_msgs), 0)
        self._messages = system_msgs + others[-keep:] if keep else system_msgs[: self.max_messages]
