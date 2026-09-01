"""输出质量日志：把每次评估结果追加写入 jsonl，支持聚合看趋势.

jsonl（JSON Lines）每行一条独立 JSON 记录，追加写不需要读旧内容，
天然适合持续增长的运行期日志；聚合时再整体读取分析。
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from smart_research_agent.evaluation.output_eval import OutputScore
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


class OutputQualityLogger:
    """输出质量日志记录器.

    参数：
        log_path: jsonl 日志文件路径，默认 logs/output_quality.jsonl
        recent_window: aggregate() 中"近期"窗口的大小（条数）
    """

    def __init__(self, log_path: str | Path = "logs/output_quality.jsonl", recent_window: int = 10):
        self.log_path = Path(log_path)
        self.recent_window = recent_window

    def log(self, task: str, output: str, score: OutputScore) -> None:
        """追加一条评估记录（task / output 截断存储，避免日志膨胀）."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "task": task[:200],
            "output_preview": output[:200],
            **score.to_dict(),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("输出质量已记录: overall=%.2f", score.overall)

    def _read_all(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        records = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def aggregate(self) -> dict:
        """聚合全部记录：各维度均分、近期趋势、高频问题.

        趋势判定：近期窗口（最近 recent_window 条）的 overall 均值
        与之前记录的均值比较，差值超过 0.05 判为 improving / degrading。
        """
        records = self._read_all()
        if not records:
            return {"count": 0, "trend": "no_data"}

        def avg(key: str, rows: list[dict]) -> float:
            return sum(r[key] for r in rows) / len(rows)

        summary = {
            "count": len(records),
            "avg_helpfulness": round(avg("helpfulness", records), 4),
            "avg_accuracy": round(avg("accuracy", records), 4),
            "avg_format_compliance": round(avg("format_compliance", records), 4),
            "avg_overall": round(avg("overall", records), 4),
        }

        recent = records[-self.recent_window :]
        earlier = records[: -self.recent_window]
        summary["recent_overall"] = round(avg("overall", recent), 4)
        if earlier:
            diff = summary["recent_overall"] - avg("overall", earlier)
            summary["trend"] = "improving" if diff > 0.05 else "degrading" if diff < -0.05 else "stable"
        else:
            summary["trend"] = "insufficient_history"

        issue_counter: Counter[str] = Counter()
        for r in records:
            issue_counter.update(r.get("issues", []))
        summary["top_issues"] = issue_counter.most_common(5)
        return summary
