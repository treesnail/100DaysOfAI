"""访问日志中间件（day044）：记录每个 HTTP 请求的方法/路径/状态码/耗时/来源 IP.

安全合规的第四条腿是"可审计"——光有审核还不够，出了问题要能查「谁在
什么时候、调了什么接口、结果如何」。``AccessLogMiddleware`` 挂在 ASGI
最外层，无论请求命中哪个路由（含 404/500），都会被记录到 ``AccessLogStore``。

与 ``AuditLogger``（工具调用审计，见 security/audit.py）的分工：
  - AuditLogger 记的是「Agent 内部调了什么工具」——业务语义层；
  - AccessLogStore 记的是「HTTP 层进来了什么请求」——网络接入层。
两者一内一外，共同构成完整的审计链路。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


@dataclass
class AccessLogRecord:
    """一条 HTTP 访问日志."""

    method: str
    path: str
    status_code: int
    duration_ms: float
    client_ip: str
    timestamp: float


class AccessLogStore:
    """访问日志存储：内存累积 + 可选追加写 JSONL 文件.

    内存记录让测试无需碰文件系统即可断言（query() 直接回放）；
    给了 ``log_path`` 才落盘，便于生产长期留存与离线分析。
    """

    def __init__(self, log_path: str | Path | None = None) -> None:
        self._path = Path(log_path) if log_path else None
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[AccessLogRecord] = []

    def record(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        client_ip: str,
    ) -> AccessLogRecord:
        """记录一条访问日志，追加到内存与（可选的）JSONL 文件."""
        record = AccessLogRecord(
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            client_ip=client_ip,
            timestamp=time.time(),
        )
        self._records.append(record)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as fp:
                fp.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return record

    def query(self) -> list[AccessLogRecord]:
        """返回全部已记录请求（按发生顺序）."""
        return list(self._records)


class AccessLogMiddleware(BaseHTTPMiddleware):
    """访问日志中间件：包住整个请求生命周期，测量耗时并记录结果.

    ``finally`` 保证无论请求正常返回还是抛异常，日志都会落一条——审计
    要覆盖「失败」而非只记「成功」。异常场景下 status_code 取兜底值 500，
    与 app.py 的全局异常处理器返回的 500 对齐。
    """

    def __init__(self, app, store: AccessLogStore) -> None:
        super().__init__(app)
        self._store = store

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        status_code = 500  # 兜底：异常未返回 response 时仍可记录
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            client_ip = request.client.host if request.client else "unknown"
            self._store.record(
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=duration_ms,
                client_ip=client_ip,
            )
