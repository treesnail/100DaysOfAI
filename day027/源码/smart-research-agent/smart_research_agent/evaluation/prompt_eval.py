"""Prompt 质量评估：规则检查 + LLM-as-a-judge 加权合成.

设计思路：
- 规则侧（check_rules）用确定性检查覆盖"机器可判定"的问题：长度、结构段落、
  占位符闭合、矛盾指令、模糊措辞；
- 评委侧（LLM-as-a-judge）用结构化 rubric 让 LLM 按 1~5 打分并输出 JSON，
  覆盖"需要语义理解"的质量维度；
- 两侧按 RULE_WEIGHT / JUDGE_WEIGHT 加权合成最终 PromptScore，
  compare() 支持两个模板版本的对比回归。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from smart_research_agent.evaluation.harness import CaseResult, EvalCase, EvaluationHarness
from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 规则检查与 LLM 评委的加权系数（评委更懂语义，权重略高）
RULE_WEIGHT = 0.4
JUDGE_WEIGHT = 0.6

#: 提示词长度的合理区间（字符数）
MIN_PROMPT_LENGTH = 50
MAX_PROMPT_LENGTH = 4000

#: 平局判定阈值：两个版本 overall 分差不超过此值视为打平
TIE_THRESHOLD = 0.05

#: 矛盾指令词对：两者同时出现说明指令口径不一致
CONTRADICTION_PAIRS = [
    ("必须", "可以不"),
    ("严格", "随意"),
    ("只能", "也可以"),
    ("不要输出", "额外输出"),
]

#: 模糊措辞：无法被程序或模型判定的主观标准
VAGUE_WORDS = ["尽量", "大概", "差不多", "也许", "等等"]

#: LLM 评委的结构化 rubric：锚定每个维度的打分标准，强制 JSON 输出
JUDGE_PROMPT = """你是一位严格的提示词评审专家。请按以下 rubric 为提示词打分（1~5 的整数或小数）：

- clarity（清晰度）：角色、任务、输出格式是否一目了然。1 分=读完不知要做什么，5 分=无需任何猜测
- consistency（一致性）：指令之间是否自相矛盾、结构是否统一。1 分=多处矛盾，5 分=完全一致
- ambiguity_free（无歧义性）：是否存在模糊措辞或多种解读空间。1 分=歧义遍地，5 分=每种情况都有明确指引

严格输出 JSON，不要输出其他内容，格式：
{{"clarity": 4, "consistency": 5, "ambiguity_free": 3, "issues": ["问题1"], "suggestions": ["建议1"]}}

待评审的提示词：
{prompt}
"""

#: 评委输出无法解析时的中性兜底分
JUDGE_FALLBACK_SCORE = 3.0

_JUDGE_KEYS = ("clarity", "consistency", "ambiguity_free")


@dataclass
class PromptScore:
    """一份提示词的质量评分（各维度 1~5 分）."""

    clarity: float
    consistency: float
    ambiguity_free: float
    overall: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class RuleReport:
    """规则检查报告：各维度基础分 + 发现的问题与建议."""

    clarity: float
    consistency: float
    ambiguity_free: float
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """两个提示词版本的对比结果."""

    score_a: PromptScore
    score_b: PromptScore
    winner: str  # "a" | "b" | "tie"


def parse_judge_output(text: str) -> dict:
    """解析评委 LLM 的输出为打分字典.

    容错策略与 planner.parse_plan 相同：find/rfind 定位 JSON 边界，
    解析失败或缺键时返回中性兜底分并记录问题，绝不抛异常中断评估流程。
    """
    cleaned = text.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    data: dict = {}
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            data = {}
    try:
        scores = {key: _clamp(float(data[key])) for key in _JUDGE_KEYS}
    except (KeyError, TypeError, ValueError):
        logger.warning("评委输出无法解析，使用中性分兜底: %s", cleaned[:100])
        return {
            **{key: JUDGE_FALLBACK_SCORE for key in _JUDGE_KEYS},
            "issues": ["LLM 评审输出无法解析，该维度使用中性分兜底"],
            "suggestions": ["检查评委提示词或更换评审模型后重评"],
        }
    return {
        **scores,
        "issues": [str(i) for i in data.get("issues") or []],
        "suggestions": [str(s) for s in data.get("suggestions") or []],
    }


def _clamp(value: float, low: float = 1.0, high: float = 5.0) -> float:
    """把分数截断到 [low, high] 区间."""
    return max(low, min(high, value))


class PromptEvaluator:
    """提示词质量评估器：规则检查打底，LLM 评委加权合成."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def check_rules(self, prompt: str) -> RuleReport:
        """规则侧检查：确定性、离线、可精确测试."""
        issues: list[str] = []
        suggestions: list[str] = []
        clarity_deduct = 0
        consistency_deduct = 0
        ambiguity_deduct = 0

        # 1. 长度：过短信息不足，过长关键指令被稀释
        if len(prompt.strip()) < MIN_PROMPT_LENGTH:
            issues.append("提示词过短，信息不足以约束模型行为")
            suggestions.append("补充角色、任务、约束等关键段落")
            clarity_deduct += 1
        elif len(prompt) > MAX_PROMPT_LENGTH:
            issues.append("提示词过长，关键指令易被稀释")
            suggestions.append("精简提示词，只保留核心指令")
            clarity_deduct += 1

        # 2. 角色定义段落
        if not re.search(r"你是|扮演|作为", prompt):
            issues.append("缺少明确的角色定义")
            suggestions.append('以"你是……"开头定义助手的角色与能力边界')
            clarity_deduct += 1

        # 3. 任务 / 输出格式说明
        if not re.search(r"输出|格式|任务", prompt):
            issues.append("缺少明确的任务或输出格式说明")
            suggestions.append("说明要完成的任务以及期望的输出格式")
            clarity_deduct += 1

        # 4. 约束性指令
        if not re.search(r"必须|严格|要求|禁止|不要", prompt):
            issues.append("缺少约束性指令，模型的发挥空间过大")
            suggestions.append('补充"必须/禁止"类约束，收紧行为边界')
            ambiguity_deduct += 1

        # 5. 占位符闭合：{ 与 } 必须成对，且不允许空占位符 {}
        if prompt.count("{") != prompt.count("}") or re.search(r"\{\s*\}", prompt):
            issues.append("模板占位符未闭合或为空")
            suggestions.append("检查所有 {placeholder} 是否成对出现且命名非空")
            consistency_deduct += 1

        # 6. 矛盾指令检测
        for positive, negative in CONTRADICTION_PAIRS:
            if positive in prompt and negative in prompt:
                issues.append(f'检测到潜在矛盾指令："{positive}" 与 "{negative}" 并存')
                suggestions.append("统一指令口径，删除或调和矛盾表述")
                consistency_deduct += 1

        # 7. 模糊措辞
        vague = [word for word in VAGUE_WORDS if word in prompt]
        if vague:
            issues.append(f"包含模糊措辞：{'、'.join(vague)}")
            suggestions.append("用可判定的明确标准替换模糊措辞")
            ambiguity_deduct += 1

        return RuleReport(
            clarity=max(1.0, 5.0 - clarity_deduct),
            consistency=max(1.0, 5.0 - consistency_deduct),
            ambiguity_free=max(1.0, 5.0 - ambiguity_deduct),
            issues=issues,
            suggestions=suggestions,
        )

    def evaluate(self, prompt: str) -> PromptScore:
        """完整评估：规则侧 + 评委侧加权合成 PromptScore."""
        rule = self.check_rules(prompt)
        raw = self.llm.chat([Message(role="user", content=JUDGE_PROMPT.format(prompt=prompt))])
        judge = parse_judge_output(raw)

        clarity = self._blend(rule.clarity, judge["clarity"])
        consistency = self._blend(rule.consistency, judge["consistency"])
        ambiguity_free = self._blend(rule.ambiguity_free, judge["ambiguity_free"])
        overall = round((clarity + consistency + ambiguity_free) / 3, 2)
        logger.info("提示词评估完成: overall=%.2f", overall)
        return PromptScore(
            clarity=clarity,
            consistency=consistency,
            ambiguity_free=ambiguity_free,
            overall=overall,
            issues=rule.issues + judge["issues"],
            suggestions=rule.suggestions + judge["suggestions"],
        )

    def compare(self, prompt_a: str, prompt_b: str) -> ComparisonResult:
        """对比两个提示词版本，返回胜者（overall 分差 <= TIE_THRESHOLD 视为平局）."""
        score_a = self.evaluate(prompt_a)
        score_b = self.evaluate(prompt_b)
        diff = score_a.overall - score_b.overall
        if abs(diff) <= TIE_THRESHOLD:
            winner = "tie"
        else:
            winner = "a" if diff > 0 else "b"
        logger.info("版本对比: a=%.2f b=%.2f winner=%s", score_a.overall, score_b.overall, winner)
        return ComparisonResult(score_a=score_a, score_b=score_b, winner=winner)

    @staticmethod
    def _blend(rule_score: float, judge_score: float) -> float:
        """规则分与评委分加权合成."""
        return round(_clamp(RULE_WEIGHT * rule_score + JUDGE_WEIGHT * judge_score), 2)


class PromptEvalHarness(EvaluationHarness):
    """提示词回归评估执行器.

    EvalCase.input 为待评估的提示词文本，expected 为 overall 最低分数线
    （字符串形式的浮点数，缺省 4.0）。用于把"提示词改版不掉分"固化成回归用例。
    """

    def __init__(self, evaluator: PromptEvaluator, default_threshold: float = 4.0):
        self.evaluator = evaluator
        self.default_threshold = default_threshold

    def run_case(self, case: EvalCase) -> CaseResult:
        threshold = float(case.expected) if case.expected else self.default_threshold
        score = self.evaluator.evaluate(case.input)
        return CaseResult(
            case=case,
            output=f"overall={score.overall:.2f}",
            passed=score.overall >= threshold,
            detail="; ".join(score.issues),
        )
