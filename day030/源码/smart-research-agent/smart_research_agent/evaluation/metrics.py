"""通用评估指标函数（简版）.

这些函数操作的是"纯数据"（参考答案与预测结果的列表），
不关心 Agent、LLM 等任何上层概念——指标层与被测系统解耦，
同一套函数可以服务 Prompt 评估、输出评估和 Agent 评估。
"""

from __future__ import annotations


def accuracy(references: list[str], predictions: list[str]) -> float:
    """完全匹配准确率：预测与参考逐条相等（忽略首尾空白）的比例."""
    if len(references) != len(predictions):
        raise ValueError("references 与 predictions 长度必须一致")
    if not references:
        return 0.0
    hits = sum(1 for ref, pred in zip(references, predictions) if ref.strip() == pred.strip())
    return hits / len(references)


def contains_rate(references: list[str], predictions: list[str]) -> float:
    """包含率：参考答案作为子串出现在预测中的比例（宽松匹配）."""
    if len(references) != len(predictions):
        raise ValueError("references 与 predictions 长度必须一致")
    if not references:
        return 0.0
    hits = sum(1 for ref, pred in zip(references, predictions) if ref.strip() in pred)
    return hits / len(references)


def mean(values: list[float]) -> float:
    """算术平均值，空列表返回 0.0 而不是抛异常（评估时"没有数据"应得 0 分而非崩溃）."""
    if not values:
        return 0.0
    return sum(values) / len(values)
