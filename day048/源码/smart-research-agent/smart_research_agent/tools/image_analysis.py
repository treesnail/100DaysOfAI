"""图像分析工具：把视觉-语言模型封装成 Agent 工具（day040）.

工具把「视觉能力」变成 Agent 可调用的一个动作：模型给出图片（本地路径、
base64 或 data URL）与问题，工具把图片处理成 VLM 需要的输入格式，交给
注入的视觉模型回答，返回文本描述。

视觉模型（``vision``）通过构造器注入：离线测试注入 MockLLM，生产默认
按 settings 构建 OpenAICompatibleLLM。工具本身不做网络传输——远程 URL
显式不支持（服务端组件发任意外网请求是 SSRF 边界，见 day044 铺垫）。
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Any

from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.multimodal.image import build_data_url, decode_data_url, validate_image
from smart_research_agent.tools.base import BaseTool


class ImageAnalysisTool(BaseTool):
    """分析图片并回答关于图片内容的问题."""

    def __init__(self, vision: BaseLLM | None = None):
        # 延迟导入 app 避免模块级循环依赖（app 会导入本工具）
        from smart_research_agent.api.app import default_llm

        self._vision = vision or default_llm()

    @property
    def name(self) -> str:
        return "image_analysis"

    @property
    def description(self) -> str:
        return (
            "分析图片并回答关于图片内容的问题；image 支持本地路径、base64 "
            "或 data URL（如 data:image/png;base64,...），不支持远程 URL"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "图片：本地路径 / base64 / data URL",
                },
                "question": {
                    "type": "string",
                    "description": "关于图片内容的问题，缺省为描述图片",
                },
            },
            "required": ["image"],
        }

    def _resolve_image_bytes(self, image: str) -> bytes:
        """把工具的 image 参数解析为图片字节.

        支持的形态（按优先级）：
          1. data URL（``data:image/...;base64,...``）；
          2. 本地文件路径（存在且可读）；
          3. 裸 base64 字符串（严格校验）。
        远程 URL 显式不支持，抛 ValueError 走业务失败文案。
        """
        if image.startswith("data:"):
            return decode_data_url(image)
        if image.startswith(("http://", "https://")):
            raise ValueError("不支持远程图片 URL，请使用本地路径、base64 或 data URL")
        path = Path(image)
        if path.is_file():
            return path.read_bytes()
        try:
            return base64.b64decode(image, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("无法解析图片参数：既不是文件路径、也不是合法的 base64") from exc

    def execute(self, image: str = "", question: str = "描述这张图片的内容", **kwargs: Any) -> str:
        """解析图片 → 校验编码 → 交给视觉模型回答.

        与 CalculatorTool 一致的错误契约：任何失败都以「图像分析失败: 原因」
        字符串返回（业务层可读），而不是抛异常让 Agent 循环崩溃。
        """
        try:
            data = self._resolve_image_bytes(image)
            mime = validate_image(data)
            data_url = build_data_url(data, mime)
            reply = self._vision.chat_vision(
                [Message(role="user", content=question)], image_data_url=data_url
            )
        except Exception as exc:  # noqa: BLE001 —— 与 CalculatorTool 同样的兜底
            return f"图像分析失败: {exc}"
        return reply
