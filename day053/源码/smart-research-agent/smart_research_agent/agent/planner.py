"""Planner：基于 LLM 的任务分解模块."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

PLAN_PROMPT = """你是一个任务规划器。请把用户目标拆解为有序、可执行的子任务列表。

要求：
1. 每个子任务用一句话描述，动词开头
2. 子任务数量 3~6 个
3. 严格输出 JSON 数组，不要输出其他内容，例如：
["子任务1", "子任务2", "子任务3"]

用户目标：{goal}
"""


@dataclass
class Plan:
    """一份拆解完成的计划."""

    goal: str
    steps: list[str] = field(default_factory=list)


class PlanParseError(ValueError):
    """计划输出解析失败."""


def parse_plan(text: str) -> list[str]:
    """从 LLM 输出中提取 JSON 字符串数组并校验."""
    cleaned = text.strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise PlanParseError("输出中未找到 JSON 数组")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise PlanParseError(f"JSON 解析失败: {exc}") from exc
    if not isinstance(data, list) or not all(isinstance(s, str) for s in data):
        raise PlanParseError("输出必须是字符串数组")
    if not data:
        raise PlanParseError("计划不能为空")
    return data


class Planner:
    """任务规划器：plan() 先行拆解，子任务交给执行层."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def plan(self, goal: str) -> Plan:
        logger.info("开始规划目标: %s", goal)
        raw = self.llm.chat([Message(role="user", content=PLAN_PROMPT.format(goal=goal))])
        steps = parse_plan(raw)
        logger.info("拆解出 %d 个子任务", len(steps))
        return Plan(goal=goal, steps=steps)
