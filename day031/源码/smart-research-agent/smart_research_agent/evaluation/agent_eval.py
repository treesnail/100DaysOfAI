"""Agent 端到端评估：轨迹建模、三维指标与失败归因.

与"单轮输出评估"不同，Agent 评估的对象是一条**多步轨迹**：
Agent 调了哪些工具、用了多少步、最终有没有给出正确答案，都要被记录和度量。

三个核心指标：
  - completion_rate  任务完成率：成功轨迹占比（结果维度）
  - step_efficiency  步数效率：最小必要步数 / 实际步数（过程维度）
  - tool_accuracy    工具正确性：集合 precision/recall + LCS 顺序分（行为维度）

失败归因（attribution）：把每条失败/低效轨迹映射到一个组件维度——
  - parse_failure   解析失败：LLM 输出格式不合法，parser 抛异常
  - planning_error  规划错误：没有产出答案（达到最大步数）
  - wrong_tool      工具选错：产出了答案，但调了不该调的工具（precision < 1）
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from smart_research_agent.agent.react_agent import ReactAgent
from smart_research_agent.evaluation.harness import Harness, load_jsonl
from smart_research_agent.evaluation.metrics import mean
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

# 归一化判定：以下输出视为"规划/兜底答案"，既不算规划错误也不算完成
_NO_ANSWER_PREFIX = "达到最大步数限制"

# agent_factory 契约：给它一个任务的 MockLLM 响应脚本，它返回一个装配好的 ReactAgent
AgentFactory = Callable[[list[str]], ReactAgent]


@dataclass
class AgentRunTrace:
    """一次 Agent 运行的结构化轨迹（评估的最小数据单元）."""

    success: bool  # 任务是否完成（产出答案且内容符合预期）
    steps: int  # 实际推理步数（一次 LLM 调用 = 一步）
    tool_calls: list[str] = field(default_factory=list)  # 实际调用的工具名序列
    expected_tools: list[str] = field(default_factory=list)  # 数据集中标注的期望工具序列
    latency: float = 0.0  # 端到端耗时（秒）
    task_id: str = ""
    answer: str = ""  # 最终答案原文（供报告与人工复查）
    error: str | None = None  # 运行期异常信息（如解析失败）


@dataclass
class TaskResult:
    """单条评估任务的完整结果：轨迹 + 判定 + 归因."""

    task_id: str
    task: str
    trace: AgentRunTrace
    success: bool
    category: str  # "completed" / "planning_error" / "wrong_tool" / "parse_failure"


# ---------------------------------------------------------------------------
# 指标函数：输入轨迹列表，输出聚合分数
# ---------------------------------------------------------------------------


def completion_rate(traces: list[AgentRunTrace]) -> float:
    """任务完成率 = 成功轨迹数 / 总轨迹数."""
    if not traces:
        return 0.0
    return sum(1 for t in traces if t.success) / len(traces)


def step_efficiency(traces: list[AgentRunTrace]) -> float:
    """步数效率 = mean(最小必要步数 / 实际步数)，截断到 [0, 1].

    最小必要步数定义为 len(expected_tools) + 1：
    每个期望的工具调用占一步，最后输出 Final Answer 再占一步。
    实际步数小于最小必要步数时按比例截断到 1.0（不奖励"少做事"）。
    """
    ratios: list[float] = []
    for t in traces:
        min_steps = len(t.expected_tools) + 1
        if t.steps <= 0:
            ratios.append(0.0)
        else:
            ratios.append(min(1.0, min_steps / t.steps))
    return mean(ratios)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """最长公共子序列长度（经典动态规划，用于度量工具调用序列的顺序保持度）."""
    # dp[j] = 已处理的 a 前缀与 b[:j] 的 LCS 长度
    dp = [0] * (len(b) + 1)
    for item in a:
        prev = 0
        for j, other in enumerate(b, start=1):
            temp = dp[j]
            dp[j] = prev + 1 if item == other else max(dp[j], dp[j - 1])
            prev = temp
    return dp[-1]


def tool_accuracy(traces: list[AgentRunTrace]) -> dict[str, float]:
    """工具使用正确性：集合 precision/recall/F1 + LCS 顺序分.

    - precision: 实际调用的工具中，有多少是任务需要的（多调扣分）
    - recall:    任务需要的工具中，有多少真的被调用了（漏调扣分）
    - f1:        precision 与 recall 的调和平均
    - order:     LCS(actual, expected) / len(expected)，衡量调用顺序的保持度；
                 只统计成功轨迹（失败轨迹的序列本身不完整，顺序分没有意义）
    """
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    orders: list[float] = []
    for t in traces:
        actual_set = set(t.tool_calls)
        expected_set = set(t.expected_tools)
        hits = len(actual_set & expected_set)
        precision = hits / len(actual_set) if actual_set else (1.0 if not expected_set else 0.0)
        recall = hits / len(expected_set) if expected_set else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        if t.success:
            expected_len = len(t.expected_tools)
            order = _lcs_length(t.tool_calls, t.expected_tools) / expected_len if expected_len else 1.0
            orders.append(order)
    return {
        "precision": mean(precisions),
        "recall": mean(recalls),
        "f1": mean(f1s),
        "order": mean(orders),
    }


# ---------------------------------------------------------------------------
# 评估器：驱动 ReactAgent 跑数据集，产出指标与逐条结果
# ---------------------------------------------------------------------------


class AgentEvaluator:
    """Agent 端到端评估器：agent_factory 注入被测 Agent 的装配方式."""

    def __init__(self, agent_factory: AgentFactory):
        self.agent_factory = agent_factory

    def evaluate(self, tasks_path: str | Path) -> dict:
        """跑完数据集中的所有任务，返回 {"metrics": ..., "results": [...]}."""
        cases = load_jsonl(tasks_path)
        logger.info("加载评估任务 %d 条: %s", len(cases), tasks_path)

        harness = Harness()
        harness_result = harness.run(cases, self._run_case)
        results = [self._to_task_result(r) for r in harness_result.case_results]
        # Harness 层捕获的运行期异常统一归为解析失败类
        task_by_id = {str(c.get("id", "")): str(c.get("task", "")) for c in cases}
        for error in harness_result.errors:
            task_id, _, message = error.partition(": ")
            trace = AgentRunTrace(
                success=False, steps=0, latency=0.0, task_id=task_id, error=message
            )
            results.append(
                TaskResult(
                    task_id=task_id,
                    task=task_by_id.get(task_id, ""),
                    trace=trace,
                    success=False,
                    category="parse_failure",
                )
            )

        traces = [r.trace for r in results]
        metrics = {
            "total": len(results),
            "completion_rate": completion_rate(traces),
            "step_efficiency": step_efficiency(traces),
            "tool_accuracy": tool_accuracy(traces),
        }
        logger.info(
            "评估完成: 完成率 %.2f, 步数效率 %.2f, 工具 F1 %.2f",
            metrics["completion_rate"],
            metrics["step_efficiency"],
            metrics["tool_accuracy"]["f1"],
        )
        return {"metrics": metrics, "results": results}

    # -- 内部：单条任务的执行与判定 ----------------------------------------

    def _run_case(self, case: dict) -> dict:
        task_id = str(case.get("id", "unknown"))
        expected_tools = list(case.get("expected_tools", []))
        agent = self.agent_factory(list(case.get("mock_responses", [])))

        start = time.perf_counter()
        answer = agent.run(str(case["task"]))
        latency = time.perf_counter() - start

        tool_calls = [s.action for s in agent.history if s.action is not None]
        success = self._judge(case, answer)
        trace = AgentRunTrace(
            success=success,
            steps=len(agent.history),
            tool_calls=tool_calls,
            expected_tools=expected_tools,
            latency=latency,
            task_id=task_id,
            answer=answer,
        )
        return {
            "id": task_id,
            "task": str(case["task"]),
            "trace": trace,
            "success": success,
            "category": self._classify(trace),
        }

    @staticmethod
    def _judge(case: dict, answer: str) -> bool:
        """成功判定：有实质答案，且通过数据集声明的内容检查."""
        if answer.startswith(_NO_ANSWER_PREFIX):
            return False
        if "answer_contains" in case:
            return str(case["answer_contains"]) in answer
        if case.get("expect_answer_json") is True:
            try:
                json.loads(answer)
            except json.JSONDecodeError:
                return False
        return True

    @staticmethod
    def _classify(trace: AgentRunTrace) -> str:
        """失败归因：把轨迹映射到一个组件维度（成功则为 completed）."""
        if trace.error is not None:
            return "parse_failure"
        if trace.success:
            return "completed"
        if trace.answer.startswith(_NO_ANSWER_PREFIX) or not trace.answer:
            return "planning_error"
        # 产出了"答案"但没通过判定，且调了期望之外的工具 -> 工具选错
        if set(trace.tool_calls) - set(trace.expected_tools):
            return "wrong_tool"
        return "planning_error"

    @staticmethod
    def _to_task_result(case_result: dict) -> TaskResult:
        return TaskResult(
            task_id=case_result["id"],
            task=case_result["task"],
            trace=case_result["trace"],
            success=case_result["success"],
            category=case_result["category"],
        )
