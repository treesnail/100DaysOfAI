"""ReAct Agent 骨架：定义 Agent 的核心组件与运行接口."""

from __future__ import annotations

from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


class ReactAgent:
    """基于 ReAct（Reasoning + Acting）范式的 Agent 骨架.

    核心组件（后续逐日接入）：
      - llm: 推理引擎（day005 接入）
      - tools: 行动能力（day004/day005 接入）
      - memory: 记忆系统（day008/day009 接入）
    """

    def __init__(self, llm=None, tools=None, max_steps: int = 10):
        self.llm = llm
        self.tools = tools or []
        self.max_steps = max_steps
        self.history: list[dict] = []

    def run(self, task: str) -> str:
        """执行任务的主入口（day006 实现完整 ReAct 循环）."""
        logger.info("接收到任务: %s", task)
        raise NotImplementedError("ReAct 循环将在 day006 实现")
