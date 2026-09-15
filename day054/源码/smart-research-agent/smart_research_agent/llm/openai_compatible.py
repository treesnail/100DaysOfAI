"""OpenAI 兼容 API 的 LLM 实现."""

from __future__ import annotations

from collections.abc import Iterator

from openai import OpenAI

from smart_research_agent.config import settings
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.multimodal.image import build_vision_payload


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

    @property
    def supports_vision(self) -> bool:
        """当前配置的模型是否支持视觉（day040）.

        这里固定返回 True，前提是 ``settings.default_model`` 指向视觉模型
        （gpt-4o-mini 等）。若配成纯文本模型（如 gpt-3.5-turbo），应改为
        按配置判断——能力声明是配置问题，不是写死的问题。
        """
        return True

    def chat_vision(
        self,
        messages: list[Message],
        image_data_url: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """视觉调用（day040）：把图像作为 content 部件数组发给视觉模型.

        OpenAI 兼容协议里，视觉请求的 user 消息 content 不是字符串，而是
        部件数组：[{"type": "text", ...}, {"type": "image_url", ...}]。
        图像以 data URL 内嵌，随请求自包含传输（组装见 build_vision_payload）。
        视觉部件挂在**最后一条 user 消息**上，其余消息（system 等）原样透传。
        """
        user_idx = None
        for i, m in enumerate(messages):
            if m.role == "user":
                user_idx = i
        if user_idx is None:
            raise ValueError("视觉消息至少需要一条 user 消息")
        built: list[dict] = []
        for i, msg in enumerate(messages):
            if i == user_idx:
                built.append(
                    {"role": "user", "content": build_vision_payload(msg.content, image_data_url)}
                )
            else:
                built.append(msg.to_dict())
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=built,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
