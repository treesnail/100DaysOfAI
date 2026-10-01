"""输出质量评估器：规则校验 + 模型打分的双重评估.

三个评估维度（均为 0~1 分，越高越好）：
    - helpfulness（有用性）：答案是否真正回应了任务。只能靠模型侧 rubric 判断。
    - accuracy（准确性）：答案内容与事实依据是否一致。规则侧做数字一致性粗检，
      模型侧做语义层面的判断，两者平均。
    - format_compliance（格式合规）：输出是否满足约定的格式与长度约束。纯规则判定，
      不需要也不应该交给模型。
overall 为三个维度的加权和。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

RUBRIC_PROMPT = """你是一个输出质量评审员。请根据任务要求评审助手的最终答案。

任务：{task}
最终答案：{output}

从两个维度打分（1~5 的整数，5 最好）：
- helpfulness：答案是否完整、直接地回应了任务，对用户有实际帮助
- accuracy：答案中的论断是否合理、无明显事实错误

严格输出 JSON，不要输出其他内容：
{{"helpfulness": 4, "accuracy": 5, "issues": ["问题描述，没有则为空数组"]}}
"""

_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class OutputScore:
    """一次输出评估的结构化结果，各维度均为 0~1 分."""

    helpfulness: float
    accuracy: float
    format_compliance: float
    overall: float
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "helpfulness": round(self.helpfulness, 4),
            "accuracy": round(self.accuracy, 4),
            "format_compliance": round(self.format_compliance, 4),
            "overall": round(self.overall, 4),
            "issues": self.issues,
        }


class OutputEvaluator:
    """双重评估器：规则通道管格式与数字，模型通道管语义.

    参数：
        llm: 打分用 LLM（测试时注入 MockLLM）
        expected_format: 期望格式，None | "json" | "markdown"
        min_length / max_length: 输出长度约束（字符数）
        weights: (helpfulness, accuracy, format_compliance) 的权重，默认 (0.4, 0.4, 0.2)
    """

    def __init__(
        self,
        llm: BaseLLM,
        expected_format: str | None = None,
        min_length: int = 1,
        max_length: int = 10000,
        weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
    ):
        if expected_format not in (None, "json", "markdown"):
            raise ValueError(f"不支持的 expected_format: {expected_format}")
        self.llm = llm
        self.expected_format = expected_format
        self.min_length = min_length
        self.max_length = max_length
        self.weights = weights

    def evaluate(
        self,
        output: str,
        task: str = "",
        observations: list[str] | None = None,
    ) -> OutputScore:
        """对一次 Agent 输出做双重评估，返回 OutputScore."""
        observations = observations or []
        issues: list[str] = []

        format_score, format_issues = self._check_format(output)
        numeric_score, numeric_issues = self._check_numeric(output, observations)
        issues.extend(format_issues)
        issues.extend(numeric_issues)

        model_scores = self._model_score(output, task)
        if model_scores is not None:
            helpfulness = model_scores["helpfulness"] / 5
            accuracy = 0.5 * numeric_score + 0.5 * (model_scores["accuracy"] / 5)
            issues.extend(model_scores.get("issues", []))
        else:
            # 模型通道失败时降级为纯规则评估，保证评估器本身永远可用
            logger.warning("模型打分失败，降级为纯规则评估")
            helpfulness = 1.0 if len(output.strip()) >= self.min_length else 0.0
            accuracy = numeric_score

        w_h, w_a, w_f = self.weights
        overall = w_h * helpfulness + w_a * accuracy + w_f * format_score
        return OutputScore(
            helpfulness=helpfulness,
            accuracy=accuracy,
            format_compliance=format_score,
            overall=overall,
            issues=issues,
        )

    # ---------- 规则通道 ----------

    def _check_format(self, output: str) -> tuple[float, list[str]]:
        """格式合规校验：结构约定 + 长度约束，满分 1.0，每违反一项扣 0.5."""
        issues: list[str] = []
        score = 1.0

        if self.expected_format == "json" and not self._is_valid_json(output):
            score -= 0.5
            issues.append("输出不是合法 JSON，违反格式约定")
        elif self.expected_format == "markdown" and not self._looks_like_markdown(output):
            score -= 0.5
            issues.append("输出缺少 Markdown 结构（标题或列表），违反格式约定")

        length = len(output.strip())
        if length < self.min_length:
            score -= 0.5
            issues.append(f"输出过短（{length} 字符 < 下限 {self.min_length}）")
        elif length > self.max_length:
            score -= 0.5
            issues.append(f"输出过长（{length} 字符 > 上限 {self.max_length}）")

        return max(score, 0.0), issues

    @staticmethod
    def _is_valid_json(text: str) -> bool:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # 去掉 Markdown 代码块围栏
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.S).strip()
        try:
            json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            return False
        return True

    @staticmethod
    def _looks_like_markdown(text: str) -> bool:
        return bool(re.search(r"^#{1,6}\s|^\s*[-*]\s|^\s*\d+\.\s", text, re.M))

    @staticmethod
    def _extract_numbers(text: str) -> list[float]:
        return [float(m) for m in _NUMBER_PATTERN.findall(text)]

    def _check_numeric(self, output: str, observations: list[str]) -> tuple[float, list[str]]:
        """数字一致性粗检：输出中的数字应能在 Observation 里找到来源.

        没有 Observation 或输出中没有数字时无法判定，按满分处理（不误伤）。
        """
        if not observations:
            return 1.0, []
        output_numbers = self._extract_numbers(output)
        if not output_numbers:
            return 1.0, []
        obs_numbers = set()
        for obs in observations:
            obs_numbers.update(self._extract_numbers(obs))

        issues: list[str] = []
        matched = 0
        for num in output_numbers:
            if any(abs(num - ref) < 1e-6 for ref in obs_numbers):
                matched += 1
            else:
                issues.append(f"数字 {num:g} 未在任何 Observation 中出现，疑似编造")
        return matched / len(output_numbers), issues

    # ---------- 模型通道 ----------

    def _model_score(self, output: str, task: str) -> dict | None:
        """LLM-as-a-judge rubric 打分；输出无法解析时返回 None 触发降级."""
        raw = self.llm.chat(
            [Message(role="user", content=RUBRIC_PROMPT.format(task=task, output=output))],
            temperature=0.0,
        )
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            data = json.loads(raw[start : end + 1])
            helpfulness = float(data["helpfulness"])
            accuracy = float(data["accuracy"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
        if not (1 <= helpfulness <= 5 and 1 <= accuracy <= 5):
            return None
        issues = data.get("issues", [])
        return {
            "helpfulness": helpfulness,
            "accuracy": accuracy,
            "issues": [str(i) for i in issues] if isinstance(issues, list) else [],
        }
