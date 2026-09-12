"""多模态图像处理：校验、编码与视觉消息构造（day040）.

视觉-语言模型（VLM）的输入是「文本 + 图像」的混排消息。图像在进入模型
之前要过三道工序：

  1. **校验**：用 magic bytes 识别真实类型（扩展名不可信）、检查大小上限；
  2. **编码**：转成 base64，并包装成 data URL（``data:<mime>;base64,<...>``）；
  3. **组装**：按 OpenAI 兼容协议把文本与图像拼成 content 部件数组。

本模块只负责「图像字节 → 协议消息」这一段，不负责网络传输：图像以字节
进入、以 data URL 与内容部件数组出去，全程可离线、可测试。

错误类型细分（``EmptyImageError`` / ``ImageTooLargeError`` /
``UnsupportedImageTypeError``）让调用方可以按**类型**分支处理，而不是
解析错误文案——API 层据此把三类失败映射到 422 / 413 / 415（见 day040
教程第六章的错误码契约）。
"""

from __future__ import annotations

import base64
import binascii

#: 单张图片的体积上限（10 MB）。理由：base64 会把字节膨胀约 1/3（3 字节
#: 编码为 4 字符），且 VLM 对超高清大图按 tile 切块计费，体积越大 token
#: 成本越高（见 day040 教程第四章）。
MAX_IMAGE_BYTES = 10 * 1024 * 1024

#: 图像按「低细节」计费时的固定 token 预算（OpenAI 视觉计费的最低档，
#: 每张图约 85 token）。MockLLM 用它模拟视觉调用的成本口径，让离线测试
#: 的 token 统计贴近真实协议。
IMAGE_TOKEN_BUDGET = 85

#: 常见图像格式的 magic bytes 前缀（PNG / JPEG / GIF；WebP 特殊处理，见 detect_mime）
_IMAGE_MAGIC: tuple[tuple[str, bytes], ...] = (
    ("image/png", b"\x89PNG\r\n\x1a\n"),
    ("image/jpeg", b"\xff\xd8\xff"),
    ("image/gif", b"GIF87a"),
    ("image/gif", b"GIF89a"),
)


class EmptyImageError(ValueError):
    """图片字节为空."""


class ImageTooLargeError(ValueError):
    """图片超过体积上限."""


class UnsupportedImageTypeError(ValueError):
    """无法识别的图片格式（不是 PNG/JPEG/GIF/WebP）."""


def detect_mime(data: bytes) -> str | None:
    """用 magic bytes 识别图像真实 MIME 类型（不信任文件扩展名）.

    扩展名可以被随手伪造（改个后缀就骗过校验），magic bytes 是文件内容
    开头的几个字节，由格式规范定义，伪造成本高得多。识别不出返回 None。
    """
    for mime, magic in _IMAGE_MAGIC:
        if data.startswith(magic):
            return mime
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def validate_image(data: bytes, max_bytes: int | None = None) -> str:
    """校验图片字节并返回 MIME 类型；不合法时抛出细分错误类型.

    检查顺序：空文件 → 超限 → 类型。类型识别靠内容（magic bytes）而非
    调用方声称的类型——服务端校验永远以自己读到的字节为准。
    """
    if not data:
        raise EmptyImageError("图片为空")
    limit = max_bytes or MAX_IMAGE_BYTES
    if len(data) > limit:
        raise ImageTooLargeError(f"图片过大：{len(data)} 字节超过上限 {limit} 字节")
    mime = detect_mime(data)
    if mime is None:
        raise UnsupportedImageTypeError("无法识别的图片格式，仅支持 PNG/JPEG/GIF/WebP")
    return mime


def encode_image_base64(data: bytes) -> str:
    """把图片字节编码为 base64 字符串."""
    return base64.b64encode(data).decode("ascii")


def build_data_url(data: bytes, mime: str | None = None) -> str:
    """把图片字节打包为 data URL（``data:<mime>;base64,<...>``）.

    data URL 是 VLM 消息里图像的标准载体：图像内容内嵌在 URL 中，自包含、
    不需要额外的网络请求或图片托管服务——离线测试也因此不需要起一个
    文件服务。不传 mime 时自动校验并识别。
    """
    resolved = mime or validate_image(data)
    return f"data:{resolved};base64,{encode_image_base64(data)}"


def decode_data_url(data_url: str) -> bytes:
    """把 data URL 还原为图片字节（ImageAnalysisTool 的输入解析用）.

    只接受约定好的 ``data:image/<type>;base64,<...>`` 形态；base64 段
    校验严格（``validate=True``），畸形内容绝不静默解出错误字节。
    """
    if not data_url.startswith("data:image/"):
        raise ValueError("不是合法的图片 data URL")
    try:
        _, payload = data_url.split(",", 1)
        return base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("图片 data URL 的 base64 段不合法") from exc


def build_vision_payload(prompt: str, image_data_url: str) -> list[dict]:
    """按 OpenAI 兼容协议组装视觉消息的 content 部件数组.

    返回形如::

        [{"type": "text", "text": prompt},
         {"type": "image_url", "image_url": {"url": image_data_url}}]

    content 从「纯字符串」升级为「部件数组」是视觉-语言模型输入格式与
    纯文本模型的分水岭：数组里每个元素是一种模态（文本、图像……），
    模型按顺序理解它们的关系（见 day040 教程第二章）。
    """
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_data_url}},
    ]
