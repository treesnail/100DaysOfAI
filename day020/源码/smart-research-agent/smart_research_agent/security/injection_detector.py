"""Prompt 注入检测器：识别用户输入中试图劫持 Agent 指令的攻击."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class InjectionReport:
    """一次注入扫描的结果."""

    is_injection: bool
    matched_patterns: list[str] = field(default_factory=list)


# 注入攻击的常见话术模式（大小写不敏感）。
# 每条规则是一个 (正则模式, 人类可读名称) 二元组，
# 名称会进入 InjectionReport.matched_patterns，便于审计与调优。
INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"忽略(之前|以前|以上|上述|所有)的?(所有|一切)?(指令|指示|规则|提示|约束)", "忽略之前指令"),
    (r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", "ignore previous instructions"),
    (r"disregard\s+(all\s+)?(previous|prior)\s+(instructions?|prompts?)", "disregard previous instructions"),
    (r"(忘记|忘掉)(之前|以前|以上|上述)的?(指令|指示|规则|设定)", "忘记之前设定"),
    (r"forget\s+(everything|all)\s+(you\s+)?(were|know|learned)", "forget everything"),
    (r"你(现在|接下来)?(扮演|充当|变成|是)一个?(没有|不受)(限制|约束|审查)", "越狱角色扮演"),
    (r"(jailbreak|DAN\s+mode|do\s+anything\s+now)", "jailbreak/DAN 模式"),
    (r"(绕过|突破|解除)(安全|内容)?(限制|审查|过滤|防护)", "绕过安全限制"),
    (r"(泄露|输出|告诉|展示)我?你的?(系统|初始)(提示词|指令|prompt)", "套取系统提示词"),
    (r"(reveal|show|print|leak)\s+(your\s+)?(system\s+)?(prompt|instructions?)", "reveal system prompt"),
    (r"执行(以下|下面|这段)?(命令|代码|shell|脚本)", "诱导执行命令"),
]


class PromptInjectionDetector:
    """基于规则与正则的 Prompt 注入检测器.

    设计说明：注入检测无法做到 100% 准确，规则层的价值在于以零成本
    拦截最常见的公开攻击话术，并为后续接入分类模型预留接口。
    """

    def __init__(self, patterns: list[tuple[str, str]] | None = None) -> None:
        self._patterns: list[tuple[re.Pattern[str], str]] = [
            (re.compile(pattern, re.IGNORECASE), name)
            for pattern, name in (patterns or INJECTION_PATTERNS)
        ]

    def scan(self, text: str) -> InjectionReport:
        """扫描一段文本，返回注入检测报告（不修改原文，只报告）."""
        matched = [name for regex, name in self._patterns if regex.search(text)]
        return InjectionReport(is_injection=bool(matched), matched_patterns=matched)
