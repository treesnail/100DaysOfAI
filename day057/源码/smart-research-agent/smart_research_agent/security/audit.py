"""执行审计日志：把每次工具调用追加写入 JSONL 文件，支持查询."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AuditRecord:
    """一条工具调用审计记录."""

    tool_name: str
    arguments: dict
    result: str
    duration_seconds: float
    timestamp: float
    status: str = "success"  # success / denied / error


class AuditLogger:
    """审计日志器：追加写入 JSONL（每行一条 JSON），可从文件回放查询."""

    def __init__(self, log_path: str | Path) -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        tool_name: str,
        arguments: dict,
        result: str,
        duration_seconds: float,
        status: str = "success",
    ) -> AuditRecord:
        """记录一次工具调用，追加一行 JSON 到日志文件."""
        record = AuditRecord(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            duration_seconds=duration_seconds,
            timestamp=time.time(),
            status=status,
        )
        with self._path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return record

    def query(self, tool_name: str | None = None) -> list[AuditRecord]:
        """读取全部审计记录，可按工具名过滤."""
        if not self._path.exists():
            return []
        records: list[AuditRecord] = []
        with self._path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                record = AuditRecord(**json.loads(line))
                if tool_name is None or record.tool_name == tool_name:
                    records.append(record)
        return records
