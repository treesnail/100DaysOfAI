"""ReAct Agent：完整的推理-行动循环实现."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from smart_research_agent.agent.parser import ReActStep, parse_react_output
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.tools.registry import ToolRegistry
from smart_research_agent.utils.logger import get_logger

if TYPE_CHECKING:
    from smart_research_agent.security.audit import AuditLogger
    from smart_research_agent.security.injection_detector import PromptInjectionDetector
    from smart_research_agent.security.permissions import ToolPermissionPolicy

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
    """ReAct 循环 Agent：Thought -> Action -> Observation 交替直到得出答案.

    安全防护（全部可选，不传则行为与 day006 完全一致）：
      - injection_detector：run() 入口扫描任务文本，命中注入则直接拒绝执行；
      - permission_policy：每次工具调用前做权限检查，拒绝信息作为 Observation 反馈给 LLM；
      - audit_logger：记录每次工具调用的工具名/参数/结果/耗时/状态到 JSONL。
    """

    def __init__(
        self,
        llm: BaseLLM,
        registry: ToolRegistry,
        max_steps: int = 10,
        injection_detector: PromptInjectionDetector | None = None,
        permission_policy: ToolPermissionPolicy | None = None,
        audit_logger: AuditLogger | None = None,
    ):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.injection_detector = injection_detector
        self.permission_policy = permission_policy
        self.audit_logger = audit_logger
        self.history: list[ReActStep] = []

    def run(self, task: str) -> str:
        logger.info("接收到任务: %s", task)

        if self.injection_detector is not None:
            report = self.injection_detector.scan(task)
            if report.is_injection:
                logger.warning("检测到 Prompt 注入，命中模式: %s", report.matched_patterns)
                return "检测到潜在的 Prompt 注入攻击，任务已被拒绝执行。"

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
        tool_name = step.action or ""
        tool = self.registry.get(tool_name)
        if tool is None:
            return f"错误：不存在名为 {step.action} 的工具"
        required = tool.parameters.get("required") or []
        if not required:
            kwargs: dict = {}
        else:
            kwargs = {required[0]: step.action_input or ""}

        if self.permission_policy is not None:
            try:
                self.permission_policy.check(tool_name)
            except PermissionError as exc:
                logger.warning("工具调用被权限策略拒绝: %s", exc)
                if self.audit_logger is not None:
                    self.audit_logger.log(tool_name, kwargs, str(exc), 0.0, status="denied")
                return f"权限错误：{exc}"

        started = time.perf_counter()
        status = "success"
        try:
            result = tool.execute(**kwargs)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            logger.warning("工具执行异常: %s", exc)
            result = f"工具执行失败: {exc}"
        if self.audit_logger is not None:
            self.audit_logger.log(
                tool_name, kwargs, result, time.perf_counter() - started, status=status
            )
        return result
