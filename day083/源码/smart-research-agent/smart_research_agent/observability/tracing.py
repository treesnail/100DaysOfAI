"""轻量 trace：每次 Agent 运行生成 trace_id，各步骤 span 落盘 jsonl，可查询回放.

trace 与 log 的区别（教程第三章详述）：log 是离散的文本事件，按时间流追加；
trace 是"一次请求"的结构化账本——所有 span 共享一个 trace_id，
span 之间有父子关系（parent_id），自带起止时间与状态。
log 回答"系统发生了什么"，trace 回答"这一次请求经历了什么"。
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Span:
    """一次操作的结构化记录：名称、起止时间、状态、父子关系与自定义属性."""

    name: str
    trace_id: str
    span_id: str
    parent_id: str | None = None
    start: float = field(default_factory=time.time)
    end: float | None = None
    status: str = "ok"  # "ok" | "error"
    attributes: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float | None:
        """耗时（毫秒）；span 未结束时为 None."""
        if self.end is None:
            return None
        return (self.end - self.start) * 1000

    def finish(self, status: str = "ok") -> None:
        self.end = time.time()
        self.status = status

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start": round(self.start, 6),
            "end": round(self.end, 6) if self.end is not None else None,
            "duration_ms": round(self.duration_ms, 3) if self.duration_ms is not None else None,
            "status": self.status,
            "attributes": self.attributes,
        }


class TraceContext:
    """一次 trace 的句柄：start_trace 返回，用 ``span()`` 开子 span."""

    def __init__(self, tracer: Tracer, trace_id: str, root: Span):
        self._tracer = tracer
        self.trace_id = trace_id
        self.root = root

    @contextmanager
    def span(self, name: str, parent: Span | None = None, **attributes) -> Iterator[Span]:
        """开一个子 span（默认挂在 root 下），异常时自动标记 status="error" 并继续抛出."""
        span = self._tracer._start_span(name, self.trace_id, parent or self.root, attributes)
        try:
            yield span
        except Exception:
            span.finish(status="error")
            self._tracer._write(span)
            raise
        span.finish(status="ok")
        self._tracer._write(span)

    def set_attribute(self, key: str, value) -> None:
        """给 root span 追加属性（如最终答案、步数），需在 trace 结束前调用."""
        self.root.attributes[key] = value


class Tracer:
    """追踪器：创建 trace、落盘 span、查询回放.

    Args:
        sink_path: span 落盘的 jsonl 路径；为 None 时只留在内存（spans 属性），
            便于测试。生产环境一般指向 logs/traces.jsonl。
    """

    def __init__(self, sink_path: str | Path | None = None):
        self.sink_path = Path(sink_path) if sink_path is not None else None
        self.spans: list[Span] = []

    @contextmanager
    def start_trace(self, name: str = "agent.run", **attributes) -> Iterator[TraceContext]:
        """开始一次 trace：生成 trace_id 与 root span，退出时收尾落盘."""
        trace_id = uuid.uuid4().hex[:16]
        root = Span(
            name=name,
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:8],
            attributes=dict(attributes),
        )
        ctx = TraceContext(self, trace_id, root)
        try:
            yield ctx
        except Exception:
            root.finish(status="error")
            self._write(root)
            raise
        root.finish(status="ok")
        self._write(root)

    def _start_span(
        self, name: str, trace_id: str, parent: Span, attributes: dict
    ) -> Span:
        return Span(
            name=name,
            trace_id=trace_id,
            span_id=uuid.uuid4().hex[:8],
            parent_id=parent.span_id,
            attributes=dict(attributes),
        )

    def _write(self, span: Span) -> None:
        self.spans.append(span)
        if self.sink_path is not None:
            self.sink_path.parent.mkdir(parents=True, exist_ok=True)
            with self.sink_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(span.to_dict(), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # 查询与回放
    # ------------------------------------------------------------------

    @staticmethod
    def load(path: str | Path) -> list[dict]:
        """从 jsonl 文件加载全部 span 记录（用于事后回放分析）."""
        records = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def get_trace(self, trace_id: str) -> list[Span]:
        """回放：根 span 永远排在最前，其余按开始时间返回某次 trace 的全部 span.

        根 span（parent_id 为 None）在窗口时间戳下 ``start`` 可能与子 span 相同，
        若只按 ``start`` 排序，根 span 会因最后落盘而排到末尾。这里先把根 span
        提到首位，再对其余 span 按开始时间排序，保证回放顺序稳定。
        """
        matched = [s for s in self.spans if s.trace_id == trace_id]
        return sorted(matched, key=lambda s: (s.parent_id is not None, s.start))

    def query(self, name: str | None = None, status: str | None = None) -> list[Span]:
        """按 span 名称 / 状态过滤（如找出所有 status="error" 的 llm.chat）."""
        return [
            s
            for s in self.spans
            if (name is None or s.name == name) and (status is None or s.status == status)
        ]


class _TimedLLMProxy:
    """包一层 LLM，把每次 chat 记为 trace 里的一个计时 span.

    只实现 chat，属性访问全部透传给被包对象——这样 Agent 侧零改造，
    MockLLM 的 calls/usage_log 等观测点照常工作。
    """

    def __init__(self, inner, ctx: TraceContext):
        self._inner = inner
        self._ctx = ctx

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def chat(self, messages, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        with self._ctx.span("llm.chat", messages=len(messages)) as span:
            reply = self._inner.chat(messages, temperature=temperature, max_tokens=max_tokens)
            span.attributes["reply_chars"] = len(reply)
            return reply


def run_with_trace(agent, task: str, tracer: Tracer) -> str:
    """带着 trace 跑一次 Agent：root span 计时整体，每次 LLM 调用各记一个子 span.

    通过临时替换 ``agent.llm`` 为计时代理实现插桩，Agent 代码零改造；
    结束后 root span 的 attributes 里带最终答案与历史步数。
    """
    with tracer.start_trace("agent.run", task=task) as ctx:
        original_llm = agent.llm
        agent.llm = _TimedLLMProxy(original_llm, ctx)
        try:
            answer = agent.run(task)
        finally:
            agent.llm = original_llm
        ctx.set_attribute("answer", answer)
        ctx.set_attribute("history_steps", len(agent.history))
    return answer
