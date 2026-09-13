"""day040 多模态测试：图像校验/编码/视觉消息组装 + ImageAnalysisTool 工具.

「图像字节 -> 协议消息」全链路离线可测：magic bytes 识别不依赖真实解码，
MockLLM 的 chat_vision 按脚本回复，因此工具与编码逻辑的每个分支都能
在无网络、无真实图片文件的环境里覆盖。
"""

from __future__ import annotations

import base64

import pytest

from smart_research_agent.llm.base import Message
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.multimodal.image import (
    IMAGE_TOKEN_BUDGET,
    MAX_IMAGE_BYTES,
    EmptyImageError,
    ImageTooLargeError,
    UnsupportedImageTypeError,
    build_data_url,
    build_vision_payload,
    decode_data_url,
    detect_mime,
    validate_image,
)
from smart_research_agent.tools.image_analysis import ImageAnalysisTool

# 只含 magic bytes 前缀的测试字节（detect_mime 按内容前缀识别，无需真实解码）
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 32
GIF_BYTES = b"GIF89a" + b"\x00" * 32
WEBP_BYTES = b"RIFF\x10\x00\x00\x00WEBPVP8 " + b"\x00" * 32


class TestDetectMime:
    def test_recognizes_png_jpeg_gif_webp(self):
        assert detect_mime(PNG_BYTES) == "image/png"
        assert detect_mime(JPEG_BYTES) == "image/jpeg"
        assert detect_mime(GIF_BYTES) == "image/gif"
        assert detect_mime(WEBP_BYTES) == "image/webp"

    def test_unknown_bytes_return_none(self):
        assert detect_mime(b"not an image at all") is None
        assert detect_mime(b"") is None


class TestValidateImage:
    def test_valid_image_returns_mime(self):
        assert validate_image(PNG_BYTES) == "image/png"
        assert validate_image(GIF_BYTES) == "image/gif"

    def test_empty_raises(self):
        with pytest.raises(EmptyImageError):
            validate_image(b"")

    def test_oversize_raises(self):
        with pytest.raises(ImageTooLargeError, match="超过上限"):
            validate_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_IMAGE_BYTES)

    def test_unsupported_type_raises(self):
        with pytest.raises(UnsupportedImageTypeError, match="无法识别"):
            validate_image(b"plain text file")

    def test_custom_limit_applies(self):
        with pytest.raises(ImageTooLargeError):
            validate_image(PNG_BYTES, max_bytes=8)


class TestDataUrl:
    def test_build_roundtrip(self):
        data_url = build_data_url(PNG_BYTES, "image/png")
        assert data_url.startswith("data:image/png;base64,")
        assert decode_data_url(data_url) == PNG_BYTES

    def test_build_auto_detects_mime(self):
        assert build_data_url(JPEG_BYTES).startswith("data:image/jpeg;base64,")

    def test_decode_rejects_non_image_prefix(self):
        with pytest.raises(ValueError, match="不是合法的图片 data URL"):
            decode_data_url("data:text/plain;base64,AA==")

    def test_decode_rejects_malformed_base64(self):
        with pytest.raises(ValueError, match="base64 段不合法"):
            decode_data_url("data:image/png;base64,!!!not-base64!!!")

    def test_base64_length_inflates_by_one_third(self):
        """3 字节编码为 4 字符（base64 的经典膨胀率），是体积上限的由来."""
        assert len(base64.b64encode(PNG_BYTES)) == (len(PNG_BYTES) + 2) // 3 * 4


class TestBuildVisionPayload:
    def test_payload_structure(self):
        payload = build_vision_payload("图里有什么", "data:image/png;base64,AAAA")
        assert payload == [
            {"type": "text", "text": "图里有什么"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]


class TestImageAnalysisTool:
    def test_analyzes_local_file_path(self, tmp_path):
        img = tmp_path / "pic.png"
        img.write_bytes(PNG_BYTES)
        vision = MockLLM(responses=["图片里有一只猫"])
        tool = ImageAnalysisTool(vision=vision)
        reply = tool.execute(image=str(img), question="图片里有什么")
        assert reply == "图片里有一只猫"
        # 视觉调用确实发生：MockLLM 记录到了消息
        assert vision.calls and vision.calls[-1][0].content == "图片里有什么"

    def test_analyzes_base64_input(self):
        vision = MockLLM(responses=["纯色图片"])
        tool = ImageAnalysisTool(vision=vision)
        b64 = base64.b64encode(PNG_BYTES).decode("ascii")
        assert tool.execute(image=b64) == "纯色图片"

    def test_analyzes_data_url_input(self):
        vision = MockLLM(responses=["描述内容"])
        tool = ImageAnalysisTool(vision=vision)
        url = build_data_url(PNG_BYTES, "image/png")
        assert tool.execute(image=url, question="描述它") == "描述内容"

    def test_remote_url_rejected(self):
        tool = ImageAnalysisTool(vision=MockLLM())
        reply = tool.execute(image="https://example.com/pic.png")
        assert "不支持远程图片 URL" in reply

    def test_garbage_input_returns_business_failure(self):
        """失败的形态是业务文案而非异常：Agent 循环不会因图片解析崩溃."""
        tool = ImageAnalysisTool(vision=MockLLM())
        reply = tool.execute(image="\x01\x02not a path or base64\x03")
        assert reply.startswith("图像分析失败:")

    def test_vision_token_cost_includes_image_budget(self):
        """MockLLM 的视觉调用按 IMAGE_TOKEN_BUDGET 计入 prompt 成本."""
        vision = MockLLM(responses=["回答"])
        vision.chat_vision(
            [Message(role="user", content="描述图片")], image_data_url="data:image/png;base64,AAAA"
        )
        assert vision.usage_log[-1]["prompt_tokens"] >= IMAGE_TOKEN_BUDGET
        assert vision.total_prompt_tokens >= IMAGE_TOKEN_BUDGET

    def test_default_vision_is_app_llm(self):
        """未注入视觉模型时，工具按默认 LLM 构建（离线为 MockLLM 占位）."""
        tool = ImageAnalysisTool()
        assert tool._vision is not None  # noqa: SLF001 —— 占位实现可离线创建
