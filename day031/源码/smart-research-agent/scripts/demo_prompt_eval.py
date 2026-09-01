"""离线演示：用 PromptEvaluator 评估并对比 ReAct 系统提示词 v1 / v2.

用法（在项目根目录）::

    python scripts/demo_prompt_eval.py

评委侧使用 MockLLM 脚本化回复，全程离线、结果确定。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smart_research_agent.agent.prompts import (  # noqa: E402
    REACT_SYSTEM_PROMPT_V1,
    REACT_SYSTEM_PROMPT_V2,
    REACT_SYSTEM_PROMPT_VERSION,
)
from smart_research_agent.evaluation.prompt_eval import PromptEvaluator  # noqa: E402
from smart_research_agent.llm.mock import MockLLM  # noqa: E402

# MockLLM 按顺序弹出：第一次 evaluate(v2) 用 JUDGE_V2，第二次 evaluate(v1) 用 JUDGE_V1
JUDGE_V2 = (
    '{"clarity": 5, "consistency": 5, "ambiguity_free": 4, '
    '"issues": [], "suggestions": ["可补充 few-shot 输出示例"]}'
)
JUDGE_V1 = (
    '{"clarity": 3, "consistency": 4, "ambiguity_free": 3, '
    '"issues": ["角色与任务混杂在同一段", "缺少独立的约束清单"], '
    '"suggestions": ["拆分为角色/任务/输出格式/约束四个段落"]}'
)


def main() -> int:
    evaluator = PromptEvaluator(llm=MockLLM(responses=[JUDGE_V2, JUDGE_V1]))
    result = evaluator.compare(REACT_SYSTEM_PROMPT_V2, REACT_SYSTEM_PROMPT_V1)

    print("=== ReAct 系统提示词版本对比 ===")
    print(f"v2 overall = {result.score_a.overall}  issues = {result.score_a.issues}")
    print(f"v1 overall = {result.score_b.overall}  issues = {result.score_b.issues}")
    print(f"winner = {result.winner}（当前生效版本: {REACT_SYSTEM_PROMPT_VERSION}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
