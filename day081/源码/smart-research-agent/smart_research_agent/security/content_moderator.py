"""内容审核器：敏感词检测与 PII（个人身份信息）识别脱敏.

day044 起扩展两类 PII：
  - 银行卡号：16~19 位数字，允许空格/连字符分隔；因银行卡号与订单号、
    时间戳同形（都是长数字串），纯正则脱敏会大面积误伤，故引入 Luhn
    校验——先正则取候选、再逐候选校验，通过才脱敏；
  - IPv4 地址：四段 0~255 的点分十进制，用「每段上界 255」的正则精确
    约束，避免把 999.999.999.999 之类非地址数字串误判。
"""

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

# PII 正则：中国大陆手机号、身份证号、电子邮箱、IPv4 地址（day044 新增后者）。
PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "手机号",
        # day031 红队加固：号段内部允许可选的 "-" 或空格分隔符，
        # 原规则 `(?<!\d)1[3-9]\d{9}(?!\d)` 要求 11 位连续数字，
        # 攻击者把号码写成 138-1234-5678 即可绕过脱敏（红队用例 pii-004 实证）
        re.compile(r"(?<!\d)1[3-9]\d[-\s]?\d{4}[-\s]?\d{4}(?!\d)"),
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
    (
        "IPv4 地址",
        # day044 新增：点分十进制，每段严格限制 0~255（25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d），
        # 前后 `(?<!\d)`/`(?!\d)` 拒绝更长数字串（如 1.2.3.45 只取合法段）。
        # 启发式局限：无法区分「真 IP」与「形如 1.2.3.4 的版本号」，生产可按
        # 上下文进一步过滤——教学版只演示"格式精确约束 + 脱敏"这一层。
        re.compile(
            r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\d)"
        ),
        "***IPv4地址***",
    ),
]


# 银行卡号候选：16~19 位数字，段间允许空格或连字符（如 4111 1111 1111 1111）。
BANK_CARD_PATTERN = re.compile(r"\b(?:\d[-\s]?){15,18}\d\b")


def _luhn_valid(digits: str) -> bool:
    """Luhn 校验：银行卡号/信用卡号的经典校验和算法.

    从右往左，偶数位（自右数第 2 位起）乘 2，超过 9 减 9；累加后若能被
    10 整除则有效。真实卡号（如 4111 1111 1111 1111）都满足该约束，而
    随机的长数字串（订单号、时间戳）大多不满足——这就是用 Luhn 过滤误报
    的数学依据。
    """
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _mask_bank_cards(text: str) -> tuple[str, bool]:
    """脱敏银行卡号：正则取候选 → 逐候选 Luhn 校验 → 通过才替换.

    返回 (脱敏后的文本, 是否命中了至少一个合法卡号)。Luhn 不过的候选
    原样保留——它们是订单号/时间戳之类的长数字串，不应被误伤。
    """
    hits = False

    def repl(match: re.Match[str]) -> str:
        nonlocal hits
        digits = re.sub(r"[-\s]", "", match.group(0))
        if _luhn_valid(digits):
            hits = True
            return "***银行卡号***"
        return match.group(0)

    return BANK_CARD_PATTERN.sub(repl, text), hits


class ContentModerator:
    """输出侧内容审核器：检测敏感词与 PII，并对 PII 做脱敏替换.

    pii_patterns 可注入（day031 新增）：与敏感词表同理，PII 规则也是业务策略，
    注入自定义规则便于复现历史版本行为（如红队评估中的加固前后对比）。
    """

    def __init__(
        self,
        sensitive_words: list[str] | None = None,
        pii_patterns: list[tuple[str, re.Pattern[str], str]] | None = None,
    ) -> None:
        self._sensitive_words = list(
            sensitive_words if sensitive_words is not None else DEFAULT_SENSITIVE_WORDS
        )
        self._pii_patterns = list(pii_patterns) if pii_patterns is not None else PII_PATTERNS

    def moderate(self, text: str) -> ModerationResult:
        """审核文本：返回检测结果与脱敏后的文本."""
        flagged = [word for word in self._sensitive_words if word in text]

        sanitized = text
        pii_types: list[str] = []

        # 银行卡号（day044）：先于纯正则 PII 处理。它是 16~19 位数字，
        # 与订单号/时间戳同形，必须经 Luhn 校验后才能确认是卡号。
        sanitized, bank_hits = _mask_bank_cards(sanitized)
        if bank_hits:
            pii_types.append("银行卡号")

        for pii_name, regex, replacement in self._pii_patterns:
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
