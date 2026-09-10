"""OpenAI 兼容 API 的 LLM 实现."""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from smart_research_agent.config import settings
from smart_research_agent.llm.base import BaseLLM, Message


class OpenAICompatibleLLM(BaseLLM):
    """通过 OpenAI SDK 调用任何 OpenAI 兼容端点."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self._client = OpenAI(
            api_key=api_key or settings.openai_api_key or "sk-missing",
            base_url=base_url or settings.openai_base_url,
        )
        self._model = model or settings.default_model

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[m.to_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """真流式：``stream=True`` 逐 chunk 产出增量文本（day039）.

        OpenAI 兼容流式响应的每个 chunk 结构与非流式相同，差别在增量语义：
        ``choices[0].delta`` 只携带**本轮新增**的片段——首个 chunk 通常只有
        ``role``，后续 chunk 才有 ``content``，最后一个可能是空 content 的
        结束标记。因此这里只 yield 非空的 content 增量，上层按顺序拼接即得
        完整回复（调用方还可据此做"上游推几片就走几片"的实时渲染）。
        """
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[m.to_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta