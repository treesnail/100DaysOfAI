"""ReAct 输出解析器：把 LLM 文本输出解析为结构化步骤."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ReActStep:
    """一步 ReAct 推理的结构化结果."""

    thought: str
    action: str | None = None
    action_input: str | None = None
    final_answer: str | None = None


class ReActParseError(ValueError):
    """ReAct 输出解析失败."""


_PATTERN_THOUGHT = re.compile(r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer):|\Z)", re.S)
_PATTERN_ACTION = re.compile(r"Action:\s*(.+)")
_PATTERN_ACTION_INPUT = re.compile(
    r"Action Input:\s*(.+?)(?=\n(?:Thought|Observation|Action|Final Answer):|\Z)", re.S
)
_PATTERN_FINAL = re.compile(r"Final Answer:\s*(.+?)\s*\Z", re.S)


def parse_react_output(text: str) -> ReActStep:
    """解析单步 ReAct 输出.

    合法形态二选一：
      - Thought + Action + Action Input（继续行动）
      - Thought + Final Answer（终止循环）
    """
    thought_m = _PATTERN_THOUGHT.search(text)
    if not thought_m:
        raise ReActParseError("缺少 Thought 字段")
    step = ReActStep(thought=thought_m.group(1).strip())

    final_m = _PATTERN_FINAL.search(text)
    if final_m:
        step.final_answer = final_m.group(1).strip()
        return step

    action_m = _PATTERN_ACTION.search(text)
    input_m = _PATTERN_ACTION_INPUT.search(text)
    if not action_m or not input_m:
        raise ReActParseError("缺少 Action 或 Action Input 字段")
    step.action = action_m.group(1).strip()
    step.action_input = input_m.group(1).strip()
    return step
