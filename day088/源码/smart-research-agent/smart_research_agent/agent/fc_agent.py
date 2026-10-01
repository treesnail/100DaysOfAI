"""FunctionCallingAgent：用 LLM 原生 function calling 走 ReAct 等价循环（day037）.

与 ReactAgent 的对照：
  - ReactAgent 走文本协议——模型按 Thought/Action/Action Input 格式输出自由文本，
    由 parser 正则解析，解析失败、工具名编造、参数格式漂移都要靠提示词约束兜底；
  - FunctionCallingAgent 走结构化协议——工具清单以 JSON Schema 随请求下发（tools 字段），
    模型直接返回 tool_calls 结构（工具名 + JSON 参数），解析由厂商侧完成，
    我方只需做 schema 校验与执行。

失败重试的反馈回路：
  - schema 校验失败 / 工具不存在：把错误详情作为 tool 消息回填，LLM 据此修正重试，
    连续失败超过 max_retries 次则放弃（防死循环烧 token）；
  - 工具执行异常：参数没错，错误是环境性的，作为观察结果回填但**不占重试预算**——
    重试模型无意义，应由模型决定换工具或放弃。
"""

from __future__ import annotations

import json

from smart_research_agent.agent.prompts import REACT_SYSTEM_PROMPT_VERSION
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.function_calling import (
    FunctionSpec,
    SchemaValidationError,
    ToolCall,
    parse_tool_calls,
    specs_from_tools,
    validate_arguments,
)
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: Function Calling 路线的系统提示词：沿用五要素解剖（角色/任务/格式/约束/上下文），
#: 但"输出格式"段从文本协议改为"优先调用函数"的决策规则——schema 已替我们把
#: 格式约束机器化了，提示词只需讲清"何时调、何时答"。
FC_SYSTEM_PROMPT = """你是 SmartResearch 智能研究助手，通过调用函数解决用户的研究与计算任务。

## 任务
接收用户任务后，判断是否需要调用函数获取信息或执行计算：
- 需要时，调用下方提供的函数（可一次调用多个）；
- 拿到函数返回结果后，若信息仍不足可继续调用其他函数；
- 信息足够时，直接用自然语言给出最终答案，不要再调用函数。

## 约束
1. 只能调用提供的函数，禁止编造函数名
2. 函数参数必须严格符合各自的 JSON Schema，必填参数不得缺失
3. 函数返回错误时，根据错误信息修正参数或改换函数后重试
4. 信息不足时优先调用函数获取，禁止凭记忆编造事实

（系统提示词版本基线：react_system@{version} 的 function calling 改写版）
"""


class FunctionCallingAgent:
    """原生 function calling 驱动的 Agent 循环.

    参数：
      - llm: 必须支持 chat_with_tools（MockLLM 或 OpenAI 兼容实现）；
      - registry: 工具注册表，工具元数据自动转为 FunctionSpec 下发；
      - max_steps: 最大 LLM 调用轮数；
      - max_retries: 连续 schema/unknown_tool 失败的最大容忍次数。
    """

    def __init__(
        self,
        llm: BaseLLM,
        registry,
        max_steps: int = 10,
        max_retries: int = 2,
    ):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps
        self.max_retries = max_retries
        #: 执行轨迹：(tool_call, 结果文本, 是否成功)，供测试与审计
        self.call_history: list[tuple[ToolCall, str, bool]] = []

    def run(self, task: str) -> str:
        specs = specs_from_tools(self.registry.list_tools())
        spec_map = {s.name: s for s in specs}
        tools_payload = [s.to_openai_tool() for s in specs]
        messages = [
            Message(
                role="system",
                content=FC_SYSTEM_PROMPT.format(version=REACT_SYSTEM_PROMPT_VERSION),
            ),
            Message(role="user", content=f"任务: {task}"),
        ]
        consecutive_failures = 0

        for step_no in range(1, self.max_steps + 1):
            response = self.llm.chat_with_tools(messages, tools_payload)
            logger.info("第 %d 步响应: %s", step_no, response)

            # 把 assistant 的决策记入对话历史（文本回复或 tool_calls 序列化）
            messages.append(
                Message(
                    role="assistant",
                    content=response.get("content")
                    or json.dumps(response.get("tool_calls"), ensure_ascii=False),
                )
            )

            try:
                calls = parse_tool_calls(response)
            except SchemaValidationError as exc:
                # 坏 JSON：连工具名/id 都不可信，无法回填 tool 消息，
                # 降级为 user 反馈让模型重新输出
                consecutive_failures += 1
                logger.warning("tool_calls 解析失败: %s", exc)
                if consecutive_failures > self.max_retries:
                    return f"函数调用连续失败 {consecutive_failures} 次，已放弃。最后的错误: {exc}"
                messages.append(
                    Message(
                        role="user",
                        content=f"你的函数调用格式有误：{exc}。请重新发起调用。",
                    )
                )
                continue

            if not calls:
                content = response.get("content") or ""
                if content:
                    logger.info("模型直接给出最终答案")
                    return content
                consecutive_failures += 1
                if consecutive_failures > self.max_retries:
                    return "模型既未调用函数也未给出答案，连续失败已达上限，已放弃。"
                messages.append(
                    Message(role="user", content="你既没有调用函数也没有回答问题，请重试。")
                )
                continue

            for call in calls:
                result, failed, counts_retry = self._execute(call, spec_map)
                self.call_history.append((call, result, not failed))
                if failed and counts_retry:
                    consecutive_failures += 1
                elif not failed:
                    consecutive_failures = 0
                messages.append(Message(role="tool", content=f"[{call.name}] {result}"))
                if consecutive_failures > self.max_retries:
                    return (
                        f"函数调用连续失败 {consecutive_failures} 次，已放弃。"
                        f"最后的错误: {result}"
                    )
        return "达到最大步数限制，未能得出最终答案"

    def _execute(
        self, call: ToolCall, spec_map: dict[str, FunctionSpec]
    ) -> tuple[str, bool, bool]:
        """执行一次函数调用，返回 (结果文本, 是否失败, 失败是否占重试预算)."""
        spec = spec_map.get(call.name)
        if spec is None:
            available = ", ".join(spec_map) or "（无）"
            return (
                f"错误：不存在名为 {call.name} 的函数，可用函数：{available}",
                True,
                True,
            )
        try:
            validate_arguments(spec, call.arguments)
        except SchemaValidationError as exc:
            return (f"参数校验失败：{exc}。请修正参数后重新调用。", True, True)
        tool = self.registry.get(call.name)
        try:
            return tool.execute(**call.arguments), False, False
        except Exception as exc:  # noqa: BLE001
            # 参数合法但执行异常：环境性失败，反馈观察结果但不占重试预算
            logger.warning("工具执行异常: %s", exc)
            return (f"工具执行失败: {exc}", True, False)
