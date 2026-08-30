"""ReAct Agent：完整的推理-行动循环实现."""

from __future__ import annotations

from smart_research_agent.agent.parser import ReActStep, parse_react_output
from smart_research_agent.evaluation.output_eval import OutputEvaluator
from smart_research_agent.evaluation.quality_logger import OutputQualityLogger
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.tools.registry import ToolRegistry
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """你是一个使用 ReAct 范式解决问题的智能助手。

严格按照以下格式交替输出：

Thought: 你对当前情况的分析与下一步打算
Action: 要调用的工具名（必须是可用工具之一）
Action Input: 工具的输入参数

当你已经有足够信息回答时，输出：
Thought: 总结性思考
Final Answer: 最终答案

可用工具：
{tools}
"""


class ReactAgent:
    """ReAct 循环 Agent：Thought -> Action -> Observation 交替直到得出答案."""

    def __init__(
        self,
        llm: BaseLLM,
        registry: ToolRegistry,
        max_steps: int = 10,
        output_evaluator: OutputEvaluator | None = None,
        quality_logger: OutputQualityLogger | None = None,
    ):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.output_evaluator = output_evaluator
        self.quality_logger = quality_logger
        self.history: list[ReActStep] = []

    def run(self, task: str) -> str:
        logger.info("接收到任务: %s", task)
        messages = [
            Message(role="system", content=SYSTEM_PROMPT.format(tools=self.registry.describe())),
            Message(role="user", content=f"任务: {task}"),
        ]
        observations: list[str] = []
        for step_no in range(1, self.max_steps + 1):
            raw = self.llm.chat(messages)
            logger.info("第 %d 步原始输出:\n%s", step_no, raw)
            step = parse_react_output(raw)
            self.history.append(step)
            if step.final_answer is not None:
                logger.info("得出最终答案")
                self._evaluate_output(task, step.final_answer, observations)
                return step.final_answer
            observation = self._execute_tool(step)
            observations.append(observation)
            messages.append(Message(role="assistant", content=raw))
            messages.append(Message(role="user", content=f"Observation: {observation}"))
        return "达到最大步数限制，未能得出最终答案"

    def _evaluate_output(self, task: str, answer: str, observations: list[str]) -> None:
        """最终答案产出时做质量评估并记录；评估失败绝不影响主流程."""
        if self.output_evaluator is None:
            return
        try:
            score = self.output_evaluator.evaluate(answer, task=task, observations=observations)
        except Exception as exc:  # noqa: BLE001
            logger.warning("输出质量评估异常（已跳过）: %s", exc)
            return
        logger.info(
            "输出质量: helpfulness=%.2f accuracy=%.2f format=%.2f overall=%.2f",
            score.helpfulness,
            score.accuracy,
            score.format_compliance,
            score.overall,
        )
        if self.quality_logger is not None:
            self.quality_logger.log(task, answer, score)

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
