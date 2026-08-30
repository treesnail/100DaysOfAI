"""ReAct Agent：完整的推理-行动循环实现."""

from __future__ import annotations

from smart_research_agent.agent.parser import ReActStep, parse_react_output
from smart_research_agent.agent.prompts import REACT_SYSTEM_PROMPT
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.tools.registry import ToolRegistry
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 向后兼容别名：系统提示词已迁移至 agent/prompts.py 统一版本化管理
SYSTEM_PROMPT = REACT_SYSTEM_PROMPT


class ReactAgent:
    """ReAct 循环 Agent：Thought -> Action -> Observation 交替直到得出答案."""

    def __init__(self, llm: BaseLLM, registry: ToolRegistry, max_steps: int = 10):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.history: list[ReActStep] = []

    def run(self, task: str) -> str:
        logger.info("接收到任务: %s", task)
        messages = [
            Message(role="system", content=SYSTEM_PROMPT.format(tools=self.registry.describe())),
            Message(role="user", content=f"任务: {task}"),
        ]
        for step_no in range(1, self.max_steps + 1):
            raw = self.llm.chat(messages)
            logger.info("第 %d 步原始输出:\n%s", step_no, raw)
            step = parse_react_output(raw)
            self.history.append(step)
            if step.final_answer is not None:
                logger.info("得出最终答案")
                return step.final_answer
            observation = self._execute_tool(step)
            messages.append(Message(role="assistant", content=raw))
            messages.append(Message(role="user", content=f"Observation: {observation}"))
        return "达到最大步数限制，未能得出最终答案"

    def _execute_tool(self, step: ReActStep) -> str:
        tool = self.registry.get(step.action or "")
        if tool is None:
            return f"错误：不存在名为 {step.action} 的工具"
        required = tool.parameters.get("required") or []
        if not required:
            kwargs: dict = {}
        else:
            kwargs = {required[0]: step.action_input or ""}
        try:
            return tool.execute(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("工具执行异常: %s", exc)
            return f"工具执行失败: {exc}"
