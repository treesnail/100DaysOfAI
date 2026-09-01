"""内容审核器：敏感词检测与 PII（个人身份信息）识别脱敏."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class ModerationResult:
    """一次内容审核的结果."""

    is_safe: bool
    sanitized_text: str
    flagged_words: list[str] = field(default_factory=list)
    pii_types: list[str] = field(default_factory=list)


# 默认敏感词表：教学用途的最小集合，生产环境应外置到配置文件。
DEFAULT_SENSITIVE_WORDS: list[str] = [
    "爆炸物制作",
    "自杀方法",
    "毒品配方",
]

# PII 正则：中国大陆手机号、身份证号、电子邮箱。
PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "手机号",
        re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
        "***手机号***",
    ),
    (
        "身份证号",
        re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
        "***身份证号***",
    ),
    (
        "邮箱",
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        "***邮箱***",
    ),
]


class ContentModerator:
    """输出侧内容审核器：检测敏感词与 PII，并对 PII 做脱敏替换."""

    def __init__(self, sensitive_words: list[str] | None = None) -> None:
        self._sensitive_words = list(
            sensitive_words if sensitive_words is not None else DEFAULT_SENSITIVE_WORDS
        )

    def moderate(self, text: str) -> ModerationResult:
        """审核文本：返回检测结果与脱敏后的文本."""
        flagged = [word for word in self._sensitive_words if word in text]

        sanitized = text
        pii_types: list[str] = []
        for pii_name, regex, replacement in PII_PATTERNS:
            if regex.search(sanitized):
                pii_types.append(pii_name)
                sanitized = regex.sub(replacement, sanitized)

        is_safe = not flagged and not pii_types
        return ModerationResult(
            is_safe=is_safe,
            sanitized_text=sanitized,
            flagged_words=flagged,
            pii_types=pii_types,
        )
