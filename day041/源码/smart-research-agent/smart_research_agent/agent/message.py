"""Agent 间通信的消息结构.

多 Agent 协作中，各角色之间通过 Message 传递任务与产出，
Orchestrator 把每一条 Message 记入 message_log，形成完整、可审计的协作轨迹。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Message:
    """一条 Agent 间的协作消息.

    sender / receiver 是角色名（如 "researcher"、"orchestrator"），
    content 是交接的内容（任务描述或上游产出），
    metadata 携带结构化附注（如所属阶段、话题），便于下游过滤与审计。
    """

    sender: str
    receiver: str
    content: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为 dict，便于日志输出与持久化."""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "metadata": dict(self.metadata),
        }
