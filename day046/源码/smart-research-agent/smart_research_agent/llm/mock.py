"""MockLLM：离线测试用的确定性 LLM 实现."""

from __future__ import annotations

import json
from collections.abc import Iterator

from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.multimodal.image import IMAGE_TOKEN_BUDGET

#: 流式分块宽度：MockLLM.stream 按此宽度切分回复，模拟 token 增量到达的节奏
_CHUNK_SIZE = 4


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

    day037 起新增 function calling 模拟（``chat_with_tools``，可选）：
    传入 ``tool_call_responses=[...]`` 后，每次 chat_with_tools 弹出下一条
    脚本化响应——{"tool_calls": [{"name": ..., "arguments": {...}}]} 表示
    模型决定调工具，{"content": "..."} 表示直接回答。脚本用尽后兜底返回
    {"content": <_pick_reply 的结果>}。内部统一规整为 OpenAI 兼容格式
    （arguments 序列化为 JSON 字符串、补 id 与 type 字段），与线上协议对齐。

    day039 起新增流式（``stream``）：消费逻辑与 chat 完全一致，但按
    ``_CHUNK_SIZE`` 宽度切分回复逐片段 yield，供 SSE 端点离线测试。
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        default: str = "mock response",
        candidates: list[str] | None = None,
        tool_call_responses: list[dict] | None = None,
    ):
        self._responses = list(responses or [])
        self._default = default
        self._candidates = list(candidates) if candidates else None
        self._tool_call_responses = list(tool_call_responses or [])
        self._candidate_cursor = 0
        self.calls: list[list[Message]] = []
        self.tool_calls_log: list[dict] = []
        self.usage_log: list[dict[str, int]] = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _take_reply(self, temperature: float) -> str:
        """取出本条回复（消费脚本或走兜底），供 chat 与 stream 复用同一入口."""
        if self._responses:
            return self._responses.pop(0)
        return self._pick_reply(temperature)

    def _record_usage(
        self, messages: list[Message], reply: str, max_tokens: int
    ) -> None:
        """把一次调用（prompt/completion）的估算 token 记入 usage 统计."""
        prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        completion_tokens = min(estimate_tokens(reply), max_tokens)
        self.usage_log.append(
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        )
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append(messages)
        reply = self._take_reply(temperature)
        self._record_usage(messages, reply, max_tokens)
        return reply

    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """流式返回：消费逻辑与 chat 一致，但按固定宽度切分后逐片段 yield.

        分块宽度见模块常量 ``_CHUNK_SIZE``（4 个字符），模拟 token 增量到达；
        全程确定性，且 usage 统计与 chat 完全一致，保证"流式/一次性"
        在同一脚本下产生可比的成本口径。
        """
        self.calls.append(messages)
        reply = self._take_reply(temperature)
        self._record_usage(messages, reply, max_tokens)
        for i in range(0, len(reply), _CHUNK_SIZE):
            yield reply[i : i + _CHUNK_SIZE]

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict:
        """function calling 模拟：按 tool_call_responses 脚本返回结构化响应.

        脚本项的两种形态：
          - {"tool_calls": [{"name": "calculator", "arguments": {...} 或 "{...json...}"}]}
          - {"content": "直接回答的文本"}
        返回前统一规整为 OpenAI 兼容格式（arguments 序列化为 JSON 字符串、
        自动补 id/type）。脚本用尽后兜底 {"content": 按 temperature 选取的回复}。
        """
        self.calls.append(messages)
        if self._tool_call_responses:
            scripted = self._tool_call_responses.pop(0)
        else:
            scripted = {"content": self._pick_reply(temperature)}
        response = self._normalize_tool_response(scripted)
        self.tool_calls_log.append(
            {"messages": messages, "tools": tools, "response": response}
        )

        prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
        reply_text = response.get("content") or json.dumps(
            response.get("tool_calls") or [], ensure_ascii=False
        )
        completion_tokens = min(estimate_tokens(reply_text), max_tokens)
        self.usage_log.append(
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        )
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        return response

    @property
    def supports_vision(self) -> bool:
        """Mock 模拟一切能力：视觉也按脚本回复（day040）."""
        return True

    def chat_vision(
        self,
        messages: list[Message],
        image_data_url: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """视觉调用模拟（day040）：与 chat 同脚本同统计，图像按固定 token 计入成本.

        ``IMAGE_TOKEN_BUDGET``（multimodal/image.py）是 OpenAI 视觉计费的
        最低档近似：每张图按低细节计 85 token。离线统计由此贴近真实协议
        的成本口径——day043 的成本追踪不需要区分文本/视觉两种调用。
        """
        self.calls.append(messages)
        reply = self._take_reply(temperature)
        prompt_tokens = sum(estimate_tokens(m.content) for m in messages) + IMAGE_TOKEN_BUDGET
        completion_tokens = min(estimate_tokens(reply), max_tokens)
        self.usage_log.append(
            {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        )
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        return reply

    @staticmethod
    def _normalize_tool_response(scripted: dict) -> dict:
        """把简写脚本规整为 OpenAI 兼容响应（不改动调用方传入的原 dict）."""
        response: dict = {"content": scripted.get("content"), "tool_calls": []}
        for i, call in enumerate(scripted.get("tool_calls") or []):
            arguments = call.get("arguments", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            response["tool_calls"].append(
                {
                    "id": call.get("id", f"call_{i}"),
                    "type": "function",
                    "function": {"name": call["name"], "arguments": arguments},
                }
            )
        if not response["tool_calls"]:
            response.pop("tool_calls")
        return response

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