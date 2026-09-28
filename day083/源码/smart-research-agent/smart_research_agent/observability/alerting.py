"""告警判定：评估指标低于阈值时产出告警，供 CI 门禁与报告标红使用.

设计原则：阈值判定与指标计算解耦——评估器只管算数（day030/031），
告警器只管比大小。这样阈值是纯粹的配置，改门禁不动评估代码。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Alert:
    """一条告警：哪个指标、实际值、阈值、中文描述."""

    metric: str
    value: float
    threshold: float
    message: str


def check_thresholds(
    metrics: dict[str, float], thresholds: dict[str, float]
) -> list[Alert]:
    """逐项比对指标与阈值，返回全部告警（指标缺失时跳过，不臆造告警）.

    Args:
        metrics: 指标名 -> 实际值，如 {"completion_rate": 0.75}。
        thresholds: 指标名 -> 最低可接受值，如 {"completion_rate": 0.8}。

    Returns:
        告警列表；为空表示全部达标。
    """
    alerts = []
    for metric, threshold in thresholds.items():
        if metric not in metrics:
            continue
        value = metrics[metric]
        if value < threshold:
            alerts.append(
                Alert(
                    metric=metric,
                    value=value,
                    threshold=threshold,
                    message=f"指标 {metric} = {value:.4f}，低于阈值 {threshold:.4f}",
                )
            )
    return alerts


def render_alerts(alerts: list[Alert]) -> str:
    """把告警渲染为 Markdown 段落（标红）；无告警返回空字符串."""
    if not alerts:
        return ""
    lines = ["", "## ⚠️ 告警", ""]
    for alert in alerts:
        lines.append(
            f'- <span style="color:red">🔴 {alert.message}</span>'
        )
    lines.append("")
    return "\n".join(lines)
