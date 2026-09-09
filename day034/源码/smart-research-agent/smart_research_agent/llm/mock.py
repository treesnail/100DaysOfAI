"""MockLLM：离线测试用的确定性 LLM 实现."""

from __future__ import annotations

from smart_research_agent.llm.base import BaseLLM, Message


def estimate_tokens(text: str) -> int:
    """粗略估算文本的 token 数（约 4 个字符 1 token，至少为 1）.

    真实的分词算法（BPE 等）在 day034 详解；这里只需要一个确定性的、
    与文本长度正相关的计数，供成本追踪测试使用。
    """
    return max(1, len(text) // 4)


class MockLLM(BaseLLM):
    """按预设脚本回复的 LLM，用于无网络、无 API Key 的测试环境.

    用法::

        llm = MockLLM(responses=["回答1", "回答2"])
        llm.chat([Message(role="user", content="hi")])  # -> "回答1"
        llm.chat([Message(role="user", content="hi")])  # -> "回答2"

    day032 起新增 token 用量统计（向后兼容，chat 返回值与旧行为完全一致）：
    每次调用把 prompt/completion 的估算 token 数记入 ``usage_log``，
    并累计到 ``total_prompt_tokens`` / ``total_completion_tokens``。

    day033 起新增 sampling 参数的确定性模拟（可选，不影响既有行为）：
    传入 ``candidates=["候选A", "候选B", ...]`` 后，当预设 responses 用尽，
    chat 按 temperature 从候选集中取回复——低温度（< 1.0）恒取第一个
    候选（贪心解码的近似），高温度（>= 1.0）在候选间轮换（模拟采样
    带来的多样性），全程确定性，可离线观察 temperature 的效果。
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        default: str = "mock response",
        candidates: list[str] | None = None,
    ):
        self._responses = list(responses or [])
        self._default = default
        self._candidates = list(candidates) if candidates else None
        self._candidate_cursor = 0
        self.calls: list[list[Message]] = []
        self.usage_log: list[dict[str, int]] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append(messages)
        if self._responses:
            reply = self._responses.pop(0)
        else:
            reply = self._pick_reply(temperature)
        prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        completion_tokens = min(estimate_tokens(reply), max_tokens)
        self.usage_log.append(
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        )
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        return reply

    def _pick_reply(self, temperature: float) -> str:
        """responses 用尽后的兜底：有候选集时按 temperature 模拟采样行为."""
        if not self._candidates:
            return self._default
        if temperature >= 1.0:
            # 高温度：候选间轮换，模拟采样带来的输出多样性（确定性实现）
            reply = self._candidates[self._candidate_cursor % len(self._candidates)]
            self._candidate_cursor += 1
            return reply
        # 低温度：恒取第一个候选，近似贪心解码（temperature -> 0 时总取概率最大者）
        return self._candidates[0]
