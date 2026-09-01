"""Reflexion 反思模块：失败检测之后的原因总结与改进建议.

设计要点：
- Reflection 是一次复盘的结构化结果（是否失败 / 原因 / 建议）；
- Reflector 只在「已被判定为失败」的轨迹上工作，本身不做失败检测；
- 反思输出是「建议性」的，解析失败时软着陆兜底，绝不让反思本身成为新的故障点。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

REFLECT_PROMPT = """你是一名严格的事后复盘员。下面是 Agent 执行某任务的轨迹，该次执行已经失败。

请完成两件事：
1. 用一句话指出失败的根本原因（reason）
2. 给出一条可操作的改进建议，指导下一次尝试（suggestion）

严格输出 JSON 对象，不要输出其他内容，例如：
{{"reason": "失败原因", "suggestion": "改进建议"}}

任务：{task}

执行轨迹：
{trajectory}
"""


@dataclass
class Reflection:
    """一次反思的结构化结果.

    failed 恒为 True：Reflector 只在失败轨迹上被调用，这是调用方传入的事实，
    而不是 LLM 的判断。suggestion 为空表示 LLM 未给出可用建议。
    """

    failed: bool
    reason: str = ""
    suggestion: str = ""


def parse_reflection(text: str) -> tuple[str, str]:
    """解析 LLM 的反思输出，返回 (reason, suggestion).

    与 parse_plan 的「四道防线 + 抛异常」不同，这里采用软着陆：
    反思结果是建议性数据，解析不出来时用原文兜底，保证重试流程不被打断。
    """
    cleaned = text.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            reason = str(data.get("reason") or "").strip()
            suggestion = str(data.get("suggestion") or "").strip()
            if reason or suggestion:
                return reason, suggestion
    logger.warning("反思输出未按 JSON 格式返回，整段作为原因兜底: %s", cleaned[:80])
    return cleaned or "LLM 未给出有效复盘", ""


class Reflector:
    """反思器：对失败的执行轨迹做复盘，产出可注入下一轮的建议."""

    def __init__(self, llm: BaseLLM, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries
        self.reflections: list[Reflection] = []

    def reflect(self, task: str, trajectory: list[str]) -> Reflection:
        """对一次失败轨迹做反思.

        输入是原始任务与该次尝试的轨迹文本列表；输出是一次 Reflection。
        每次反思都会记录到 self.reflections，便于测试断言与运行期审计。
        """
        trajectory_text = "\n".join(trajectory) if trajectory else "(无可用轨迹)"
        prompt = REFLECT_PROMPT.format(task=task, trajectory=trajectory_text)
        raw = self.llm.chat([Message(role="user", content=prompt)])
        reason, suggestion = parse_reflection(raw)
        reflection = Reflection(failed=True, reason=reason, suggestion=suggestion)
        self.reflections.append(reflection)
        logger.info("反思完成: reason=%s | suggestion=%s", reason, suggestion)
        return reflection
