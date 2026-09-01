"""ReAct Agent：完整的推理-行动循环实现，可选接入 Reflexion 反思重试与安全防护."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from smart_research_agent.agent.parser import ReActStep, parse_react_output
from smart_research_agent.agent.reflexion import Reflector
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

REFLECTION_HINT = """
注意：你之前的一次尝试失败了。反思模块复盘后给出的改进建议如下，本轮执行务必遵守：

{suggestion}
"""

# 判定「工具执行失败」的 Observation 前缀。
# 这是启发式约定：工具层目前用文本前缀传递错误，生产系统应改用结构化结果。
_ERROR_PREFIXES = ("错误：", "工具执行失败", "计算失败")


class ReactAgent:
    """ReAct 循环 Agent：Thought -> Action -> Observation 交替直到得出答案.

    可选参数 reflector：传入后启用 Reflexion 机制——某步工具执行失败或循环
    耗尽仍未产出答案时，先让 Reflector 复盘失败轨迹，把改进建议注入下一轮
    的 system/user prompt 再重试，最多重试 reflector.max_retries 次。
    不传 reflector 时行为与 day006 完全一致（单次尝试，错误 Observation 照常回喂）。

    安全防护（全部可选，不传则行为与基座完全一致）：
      - injection_detector：run() 入口扫描任务文本，命中注入则直接拒绝执行；
      - permission_policy：每次工具调用前做权限检查，拒绝信息作为 Observation 反馈给 LLM；
      - audit_logger：记录每次工具调用的工具名/参数/结果/耗时/状态到 JSONL。
    """

    def __init__(
        self,
        llm: BaseLLM,
        registry: ToolRegistry,
        max_steps: int = 10,
        reflector: Reflector | None = None,
        injection_detector: PromptInjectionDetector | None = None,
        permission_policy: ToolPermissionPolicy | None = None,
        audit_logger: AuditLogger | None = None,
    ):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.reflector = reflector
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

        max_attempts = 1 + (self.reflector.max_retries if self.reflector is not None else 0)
        suggestion = ""
        for attempt in range(1, max_attempts + 1):
            answer, failed, trajectory = self._attempt(task, suggestion)
            if not failed or attempt == max_attempts:
                return answer
            reflection = self.reflector.reflect(task, trajectory)
            suggestion = reflection.suggestion
            logger.info("第 %d 次尝试失败，注入反思建议后重试: %s", attempt, suggestion)
        return answer  # pragma: no cover - 循环内必有返回

    def _attempt(self, task: str, suggestion: str) -> tuple[str, bool, list[str]]:
        """跑一轮完整的 ReAct 循环.

        返回 (答案文本, 是否失败, 本轮轨迹)。
        失败判定（仅在启用 reflector 时生效）：某步工具执行失败立即中断本轮，
        或循环耗尽 max_steps 仍未给出 Final Answer。
        """
        system_content = SYSTEM_PROMPT.format(tools=self.registry.describe())
        user_content = f"任务: {task}"
        if suggestion:
            system_content += REFLECTION_HINT.format(suggestion=suggestion)
            user_content += f"\n\n上一次失败的教训（来自反思模块）：{suggestion}"
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content),
        ]
        trajectory: list[str] = []
        for step_no in range(1, self.max_steps + 1):
            raw = self.llm.chat(messages)
            logger.info("第 %d 步原始输出:\n%s", step_no, raw)
            step = parse_react_output(raw)
            self.history.append(step)
            if step.final_answer is not None:
                logger.info("得出最终答案")
                return step.final_answer, False, trajectory
            observation = self._execute_tool(step)
            trajectory.append(
                f"第{step_no}步 Thought: {step.thought} | "
                f"Action: {step.action}({step.action_input}) | Observation: {observation}"
            )
            if self.reflector is not None and observation.startswith(_ERROR_PREFIXES):
                logger.warning("检测到工具执行失败，中断本轮转入反思: %s", observation)
                return f"尝试失败：{observation}", True, trajectory
            messages.append(Message(role="assistant", content=raw))
            messages.append(Message(role="user", content=f"Observation: {observation}"))
        return "达到最大步数限制，未能得出最终答案", True, trajectory

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
