"""OpenAI 兼容 API 的 LLM 实现."""

from __future__ import annotations

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
