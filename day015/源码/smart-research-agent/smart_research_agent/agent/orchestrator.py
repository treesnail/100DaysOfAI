"""多 Agent 编排器：研究 -> 分析 -> 撰稿 流水线.

Orchestrator 本身不调用 LLM，只做三件事：
1. 按固定顺序把任务交给各个角色；
2. 用 Message 在各角色之间传递任务与产出；
3. 把每一条 Message 记入 message_log，形成完整协作轨迹。
"""

from __future__ import annotations

from smart_research_agent.agent.message import Message
from smart_research_agent.agent.roles import AnalystAgent, ResearcherAgent, WriterAgent
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


class MultiAgentOrchestrator:
    """多 Agent 协作编排器：串联研究员、分析师、撰稿人."""

    def __init__(
        self,
        researcher: ResearcherAgent,
        analyst: AnalystAgent,
        writer: WriterAgent,
    ):
        self.researcher = researcher
        self.analyst = analyst
        self.writer = writer
        self.message_log: list[Message] = []

    def run(self, topic: str) -> dict:
        """按 研究 -> 分析 -> 撰稿 流水线完成一次协作，返回各阶段产出与消息轨迹."""
        logger.info("开始多 Agent 协作，话题: %s", topic)
        self.message_log.clear()

        # 阶段一：研究员检索资料
        self._record("orchestrator", self.researcher.name, topic, stage="dispatch_research")
        research = self.researcher.work(task=f"围绕话题收集资料：{topic}")
        self._record(self.researcher.name, self.analyst.name, research, stage="research")

        # 阶段二：分析师提炼要点
        analysis = self.analyst.work(task="从研究资料中提炼核心要点", context=research)
        self._record(self.analyst.name, self.writer.name, analysis, stage="analysis")

        # 阶段三：撰稿人成稿
        article = self.writer.work(task=f"根据要点撰写关于「{topic}」的短文", context=analysis)
        self._record(self.writer.name, "orchestrator", article, stage="article")

        logger.info("协作完成，共产生 %d 条消息", len(self.message_log))
        return {
            "topic": topic,
            "research": research,
            "analysis": analysis,
            "article": article,
            "message_log": [m.to_dict() for m in self.message_log],
        }

    def _record(self, sender: str, receiver: str, content: str, stage: str) -> None:
        """把一条协作消息记入 message_log."""
        self.message_log.append(
            Message(sender=sender, receiver=receiver, content=content, metadata={"stage": stage})
        )
