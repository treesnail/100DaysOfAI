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
    # day031 红队加固：原规则 `一个?` 的量词只作用于“个”，“一”是必需的，
    # 导致“你是没有审查机制的模型”这类不带“一个”的话术漏网（红队用例 jb-004 实证）
    (r"你(现在|接下来)?(扮演|充当|变成|是)(一个|一名|一位)?(没有|不受)(限制|约束|审查)", "越狱角色扮演"),
    (r"(jailbreak|DAN\s+mode|do\s+anything\s+now)", "jailbreak/DAN 模式"),
    (r"(绕过|突破|解除)(安全|内容)?(限制|审查|过滤|防护)", "绕过安全限制"),
    (r"(泄露|输出|告诉|展示)我?你的?(系统|初始)(提示词|指令|prompt)", "套取系统提示词"),
    (r"(reveal|show|print|leak)\s+(your\s+)?(system\s+)?(prompt|instructions?)", "reveal system prompt"),
    (r"执行(以下|下面|这段)?(命令|代码|shell|脚本)", "诱导执行命令"),
    # --- day031 红队加固：以下规则由红队评估（evaluation/redteam.py）发现的漏网用例驱动新增 ---
    # “无视/抛开”与“忽略”同义，但原规则只覆盖“忽略/忘记”，攻击者换个动词即可绕过
    (r"(无视|抛开|不管)(你(收到|得到)的|之前|以上|上述|所有|全部)的?(所有|全部|一切)?(指令|指示|规则|提示|约束|设定)", "无视既有指令"),
    # override/disobey/violate + directives/guidelines 是 ignore 的英文同义话术
    (r"(override|disobey|violate)\s+(your\s+|the\s+)*(previous|prior|above|safety|hidden)?\s*(directives?|instructions?|guidelines?|rules?|prompt)", "override 既有指令"),
    # 原“越狱角色扮演”规则要求“没有”后紧跟“限制/约束/审查”，
    # 攻击者插入“任何/道德”等修饰词即可绕过，这里补一个允许插入修饰词的变体规则
    (r"你(现在|接下来)?(扮演|充当|变成|是)(一个|一名|一位)?(没有|不受)(任何|所有|一切|道德|伦理|法律|内容)+(限制|约束|审查)", "越狱角色扮演变体"),
    # “开发者模式/开发者选项”话术：声称进入特殊模式以关闭安全过滤
    (r"(开发者模式|开发者选项|developer\s+mode|dev\s+mode)", "开发者模式越狱"),
    # 英文无限制角色扮演：you are an AI without any restrictions/filters
    (r"without\s+(any\s+)?(restrictions?|limitations?|limits?|filters?|censorship|constraints?)", "英文无限制声明"),
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
