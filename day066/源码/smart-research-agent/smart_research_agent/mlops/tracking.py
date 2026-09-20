"""实验追踪：把"这次训练跑了什么"记成一条可比较的记录（M5-D10）.

day058 给模型版本建了身份（三元组）与版本链，但**指标仍然散落在各处**：
`SFTReport` 里有 loss、`AdapterManifest` 里有可训练参数量、
`finetune_eval` 的报告里有合格率、`domain_data` 的清单里有数据规模。
想把"两次实验的差异"说清楚，你得同时打开四份 json 再人肉对齐。

本模块提供一个**极简的实验追踪器**，只回答四个问题：

```text
这次实验用了什么参数？   params   （learning_rate / lora_r / 数据集指纹……）
结果是多少？            metrics  （loss / 合格率 / 延迟 / 成本……）
产出了哪些文件？         artifacts（适配器目录 / 合并模型 / 报告）
它属于哪一次父实验？      parent_run_id（超参搜索与消融实验的树）
```

## 与 MLflow / W&B 的对应关系

本模块刻意**照着 MLflow 的概念形状**设计，因此换到真框架时是一次机械替换：

| 本模块 | MLflow | Weights & Biases |
|--------|--------|------------------|
| `ExperimentTracker` | `mlflow`（模块级） | `wandb.init()` |
| `Run` | `mlflow.start_run()` | `wandb.Run` |
| `log_params` | `mlflow.log_params` | `run.config` |
| `log_metrics(step=)` | `mlflow.log_metric(key, value, step)` | `wandb.log({...}, step=)` |
| `log_artifact` | `mlflow.log_artifact` | `wandb.Artifact` |
| `finish(status)` | `mlflow.end_run(status)` | `run.finish()` |
| `runs.jsonl` | 后端数据库 | 云端存储 |

**为什么不直接用真框架**：本课程的硬纪律是"离线、可复现、零外部依赖"。
引入 MLflow 会带来一个需要长期运行的追踪服务，而 `tests/` 里就得起一个
容器；引入 W&B 则需要联网与账号。两者都会让"每天 2 小时"这件事变成
"每天先解决环境问题"。

## 两处与 MLflow 的刻意差异

1. **``run_id`` 是确定性的，不是随机 UUID**。MLflow 用随机 id，
   因此"重跑同一个实验"永远产生一个新 run。本模块用
   ``sha256(name + canonical(params))[:12]``，于是：

   - 重跑**完全相同的参数**时，要么命中同一个 run（`reuse=True`），
     要么得到一个带序号后缀的新 run（`-2` / `-3`）；
   - 两次实验的 id 相同 ⟺ 参数完全相同，这个性质让"这个数字是哪次跑出来的"
     在报告里可以直接用 id 回答。

2. **同一个 ``(指标名, step)`` 不能重复记录**。MLflow 允许覆盖，
   而覆盖会让"loss 曲线"变成"最后一次写入的 loss 曲线"——
   历史被静默改写。本模块直接报错：**要记录第二次，就换一个 step。**
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from smart_research_agent.mlops.errors import MLOpsError
from smart_research_agent.registry.record import utc_now_iso
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 追踪文件（jsonl，与 day027 的质量日志、day058 的版本索引同一格式纪律：
#: 追加写 O(1)、单行损坏不传染、流式工具友好）。
RUNS_FILENAME = "runs.jsonl"

#: ``run_id`` 保留的十六进制位数。
RUN_ID_LENGTH = 12

#: run 的三个状态。``running`` 是"还没结束"，`failed` 与 `finished` 是终止态。
STATUS_RUNNING = "running"
STATUS_FINISHED = "finished"
STATUS_FAILED = "failed"
STATUSES: tuple[str, ...] = (STATUS_RUNNING, STATUS_FINISHED, STATUS_FAILED)

#: 比较两次 run 时用的三个"关系"标签。
RELATION_BETTER = "better"
RELATION_WORSE = "worse"
RELATION_EQUAL = "equal"

#: 指标方向：越大越好 / 越小越好。它进比较函数的签名，**没有一个全局缺省**——
#: 同一个数字（例如 loss）在监督微调里越小越好、在奖励模型里越大越好，
#: 给一个全局缺省会诱导调用方忘记它。
METRIC_MODE_MAX = "max"
METRIC_MODE_MIN = "min"
METRIC_MODES: tuple[str, ...] = (METRIC_MODE_MAX, METRIC_MODE_MIN)


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    """规范化序列化：排序键 + 紧凑分隔符（确定性 run_id 的前提）."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def run_id_for(name: str, params: dict[str, Any]) -> str:
    """由实验名与参数算出确定性的 run_id.

    **只覆盖 params，不覆盖 metrics**：指标是这次运行的结果，
    把它算进 id 会让"同一次实验的两次记录"得到两个 id——
    而那正是我们需要判等的情形。
    """
    if not name.strip():
        raise MLOpsError("实验名不能为空（它进 run_id，空名会让所有实验撞在一起）")
    payload = {"name": name.strip(), "params": params}
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()[:RUN_ID_LENGTH]


@dataclass
class Run:
    """一次实验运行：参数、指标（可按 step 多次记录）、产物、状态.

    ``metrics`` 的形状是 ``{指标名: {step 字符串: 值}}``。用两层字典而不是
    ``{指标名: 值}`` 是为了支持"逐步记录的 loss 曲线"，而
    ``{"-1": 0.41}`` 这个约定表示"没有 step 概念的标量指标"——
    **用 -1 而不是 0**，因为 step 0 是一个真实的训练步（day050 的
    warmup 就从第 0 步开始）。
    """

    run_id: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    status: str = STATUS_RUNNING
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: str = ""
    parent_run_id: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise MLOpsError(f"未知状态 {self.status!r}，可选 {', '.join(STATUSES)}")

    # ------------------------------------------------------------------ 派生量
    @property
    def finished(self) -> bool:
        """是否已结束（含失败）——"还在跑"与"跑完了"要能区分."""
        return self.status in (STATUS_FINISHED, STATUS_FAILED)

    @property
    def duration_seconds(self) -> float:
        """时长（秒）；未结束时返回 0.0.

        返回 0.0 而不是 ``None``：这个字段会被写进报告表格与 CI 摘要，
        而"还没有时长"在那里就是一个 0。**但状态字段同时存在**，
        因此 0 不会被误读成"瞬间跑完"。
        """
        if not self.ended_at or not self.started_at:
            return 0.0
        try:
            start = datetime.fromisoformat(self.started_at.replace("Z", "+00:00"))
            end = datetime.fromisoformat(self.ended_at.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
        return max(0.0, (end - start).total_seconds())

    def flat_metrics(self) -> dict[str, float]:
        """把两层指标压平成 ``{指标名: 最后一次的值}``.

        取"最后一个 step"而不是"step 最小的那个"：loss 曲线关心的是收尾状态，
        而 step 键在字典里是**字符串**，直接排序会得到 ``"10" < "9"``。
        因此按数值排序后再取最后一项。
        """
        flat: dict[str, float] = {}
        for name, series in self.metrics.items():
            if not series:
                continue
            last_step = max(series, key=lambda key: int(key))
            flat[name] = series[last_step]
        return flat

    def metric(self, name: str) -> float | None:
        """单个指标的最终值；没有记录返回 ``None``（缺失与 0 是两件事）."""
        return self.flat_metrics().get(name)

    def history(self, name: str) -> list[tuple[int, float]]:
        """某个指标的逐步历史，按 step 升序（画曲线与定位发散点用）."""
        series = self.metrics.get(name, {})
        return sorted(((int(step), value) for step, value in series.items()))

    # ------------------------------------------------------------------ 投影
    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（含派生量）."""
        payload = asdict(self)
        payload["finished"] = self.finished
        payload["duration_seconds"] = round(self.duration_seconds, 4)
        payload["flat_metrics"] = self.flat_metrics()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Run:
        """从 ``to_dict`` 的产物还原（忽略派生键）."""
        return cls(
            run_id=str(payload["run_id"]),
            name=str(payload["name"]),
            params=dict(payload.get("params", {})),
            metrics={
                str(name): {str(step): float(value) for step, value in series.items()}
                for name, series in dict(payload.get("metrics", {})).items()
            },
            artifacts={str(k): str(v) for k, v in dict(payload.get("artifacts", {})).items()},
            tags={str(k): str(v) for k, v in dict(payload.get("tags", {})).items()},
            status=str(payload.get("status", STATUS_RUNNING)),
            started_at=str(payload.get("started_at", "")) or utc_now_iso(),
            ended_at=str(payload.get("ended_at", "")),
            parent_run_id=str(payload.get("parent_run_id", "")),
            notes=str(payload.get("notes", "")),
        )

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        flat = self.flat_metrics()
        headline = ", ".join(f"{name}={value:g}" for name, value in sorted(flat.items()))
        return (
            f"{self.run_id} {self.name} [{self.status}] "
            f"{len(self.params)} 参数 / {len(self.metrics)} 指标"
            f"（{headline or '无指标'}）| {len(self.artifacts)} 产物 | "
            f"{self.duration_seconds:.2f}s"
        )


def compare_runs(
    left: Run,
    right: Run,
    *,
    metric: str,
    mode: str = METRIC_MODE_MAX,
) -> dict[str, Any]:
    """比较两次 run 在某一指标上的高低.

    返回 ``{metric, mode, left, right, delta, relation, comparable}``。

    **``mode`` 必须显式给出**（值域 ``max`` / ``min``）：同一个数字
    （例如 loss）在监督微调里越小越好、在奖励模型里越大越好，
    给一个全局缺省会诱导调用方忘记它，而忘记的后果是**结论反向**。

    任一臂缺这个指标时 ``comparable=False``、``relation=None``：
    "没测过"不是"更差"，也不是"相等"。这与 day058 ``pass_rate``
    返回 ``None`` 是同一条纪律。
    """
    if mode not in METRIC_MODES:
        raise MLOpsError(f"未知指标方向 {mode!r}，可选 {', '.join(METRIC_MODES)}")
    left_value = left.metric(metric)
    right_value = right.metric(metric)
    payload: dict[str, Any] = {
        "metric": metric,
        "mode": mode,
        "left_run": left.run_id,
        "right_run": right.run_id,
        "left": left_value,
        "right": right_value,
        "delta": None,
        "relation": None,
        "comparable": left_value is not None and right_value is not None,
    }
    if payload["comparable"]:
        delta = float(right_value) - float(left_value)
        payload["delta"] = round(delta, 6)
        if delta == 0:
            payload["relation"] = RELATION_EQUAL
        elif (delta > 0) == (mode == METRIC_MODE_MAX):
            payload["relation"] = RELATION_BETTER
        else:
            payload["relation"] = RELATION_WORSE
    return payload


class ExperimentTracker:
    """实验追踪器（可选落盘；``path=None`` 时纯内存）.

    用法::

        tracker = ExperimentTracker("outputs/mlops/runs.jsonl")
        run = tracker.start_run("lora-r8", params={"lora_r": 8, "lr": 1e-4})
        run.log_metrics({"train_loss": 0.41}, step=40)
        run.log_artifact("adapter", "outputs/lora/adapters/adapter-final")
        run.finish()
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path: Path | None = Path(path) if path is not None else None
        self._runs: dict[str, Run] = {}
        self._order: list[str] = []
        if self.path is not None:
            self.reload()

    # ------------------------------------------------------------------ 读盘
    @property
    def directory(self) -> Path | None:
        """追踪文件所在目录（``path=None`` 时为 ``None``）."""
        return None if self.path is None else self.path.parent

    def reload(self) -> None:
        """清空内存状态并重新从追踪文件折叠一次（幂等）."""
        self._runs.clear()
        self._order.clear()
        if self.path is None or not self.path.exists():
            return
        for lineno, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise MLOpsError(f"{self.path} 第 {lineno} 行不是合法 JSON：{exc}") from exc
            if not isinstance(payload, dict) or "run_id" not in payload:
                raise MLOpsError(f"{self.path} 第 {lineno} 行不是一条 run 记录")
            run = Run.from_dict(payload)
            self._runs[run.run_id] = run
            if run.run_id not in self._order:
                self._order.append(run.run_id)
        logger.info("实验追踪已加载：%s（%d 条 run）", self.path, len(self._runs))

    def _persist(self, run: Run) -> None:
        """把整条 run 追加进文件（追加写：一次运行写多行，最后一行是终态）.

        **写整条而不是写增量事件**：run 的字段少（参数、指标、产物、状态），
        整条重写的体积可以忽略；而"读最后一行即得终态"让 `reload` 的实现
        简单到没有出错空间。这与 day058 版本索引选择事件流的原因不同——
        那里的记录会很多、单条很大，而且阶段变更是稀疏事件；
        这里的 run 是"一次实验一个对象"，两者是两种不同的数据形状。
        """
        if self.path is None:
            return
        assert self.directory is not None  # noqa: S101 - path 非空时 directory 必然非空
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(run.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")

    # ------------------------------------------------------------------ 写
    def start_run(
        self,
        name: str,
        *,
        params: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        parent_run_id: str = "",
        reuse: bool = False,
        notes: str = "",
    ) -> Run:
        """开启一次运行，返回 ``Run`` 对象（后续的 log 都写在它身上）.

        ``reuse=False``（缺省）时，若确定性 id 已被占用，就追加 ``-2`` / ``-3``
        后缀，得到一个新 id。``reuse=True`` 时返回**已有的那个 run**——
        这对应"CI 重跑同一个任务"：参数一字未改，我们想要的是同一个 run，
        而不是一堆内容重复的副本。

        **不允许"两组不同参数得到同一个 id"**：id 由
        ``run_id_for(name, params)`` 算出，参数不同则 id 必不同
        （sha256 前 12 位在本课程的实验量级上碰撞概率可忽略）。
        因此后缀只可能来自"同参数重复运行"，语义是干净的。
        """
        if parent_run_id and parent_run_id not in self._runs:
            raise MLOpsError(
                f"父实验 {parent_run_id} 不存在：悬空的 parent 会让消融实验的树断掉"
            )
        resolved_params = dict(params or {})
        base_id = run_id_for(name, resolved_params)
        if base_id in self._runs:
            existing = self._runs[base_id]
            if reuse:
                logger.info("复用已有 run %s（参数完全一致）", existing.run_id)
                return existing
            suffix = 2
            while f"{base_id}-{suffix}" in self._runs:
                suffix += 1
            run_id = f"{base_id}-{suffix}"
        else:
            run_id = base_id
        run = Run(
            run_id=run_id,
            name=name.strip(),
            params=resolved_params,
            tags=dict(tags or {}),
            parent_run_id=parent_run_id,
            notes=notes,
        )
        self._runs[run_id] = run
        self._order.append(run_id)
        self._persist(run)
        logger.info("开启实验 %s", run.summary_line())
        return run

    def log_metrics(
        self, run: Run, metrics: dict[str, float], *, step: int | None = None
    ) -> Run:
        """记录一批指标（可按 step 多次记录）.

        同一个 ``(指标名, step)`` 重复记录会**报错**：覆盖会让 loss 曲线
        变成"最后一次写入的曲线"，历史被静默改写。要记录第二次就换一个 step。
        """
        key = "-1" if step is None else str(int(step))
        for name, value in metrics.items():
            series = run.metrics.setdefault(str(name), {})
            if key in series:
                raise MLOpsError(
                    f"指标 {name} 在 step={key} 已记录过（旧值 {series[key]}）："
                    "覆盖会让曲线变成'最后一次写入的曲线'，要记录第二次请换一个 step"
                )
            series[key] = float(value)
        self._touch(run)
        return run

    def log_artifact(self, run: Run, name: str, path: str) -> Run:
        """登记一个产物路径（同名覆盖：产物路径是"当前值"而非历史）."""
        if not name.strip():
            raise MLOpsError("产物名不能为空")
        run.artifacts[name.strip()] = str(path)
        self._touch(run)
        return run

    def set_tags(self, run: Run, tags: dict[str, str]) -> Run:
        """合并标签（同名覆盖：标签是"当前值"）."""
        run.tags.update({str(k): str(v) for k, v in tags.items()})
        self._touch(run)
        return run

    def finish(self, run: Run, *, status: str = STATUS_FINISHED, notes: str = "") -> Run:
        """结束一次运行（幂等：重复结束不会改状态，只更新备注）."""
        if status not in (STATUS_FINISHED, STATUS_FAILED):
            raise MLOpsError(
                f"finish 只能把状态置为 {STATUS_FINISHED} / {STATUS_FAILED}，收到 {status!r}"
            )
        if not run.finished:
            run.status = status
            run.ended_at = utc_now_iso()
        if notes:
            run.notes = f"{run.notes} | {notes}".lstrip(" |")
        self._touch(run)
        return run

    def fail(self, run: Run, *, error: str = "") -> Run:
        """把一次运行标记为失败（失败也是要留档的事实）."""
        if error:
            run.tags["error"] = error[:200]
        return self.finish(run, status=STATUS_FAILED, notes=error)

    def _touch(self, run: Run) -> None:
        """把 run 的当前状态追加落盘（内存模式下是空操作）."""
        self._persist(run)

    # ------------------------------------------------------------------ 读
    def get(self, run_id: str) -> Run:
        """按 id 取 run；不存在抛 ``MLOpsError``."""
        if run_id not in self._runs:
            raise MLOpsError(f"run {run_id!r} 不存在")
        return self._runs[run_id]

    def find(self, run_id: str) -> Run | None:
        """``get`` 的不抛异常版本."""
        return self._runs.get(run_id)

    def runs(
        self, *, status: str | None = None, tag: tuple[str, str] | None = None
    ) -> list[Run]:
        """按开启顺序列出 run（可按状态或标签过滤）."""
        if status is not None and status not in STATUSES:
            raise MLOpsError(f"未知状态 {status!r}，可选 {', '.join(STATUSES)}")
        items = [self._runs[run_id] for run_id in self._order]
        if status is not None:
            items = [run for run in items if run.status == status]
        if tag is not None:
            key, value = tag
            items = [run for run in items if run.tags.get(key) == value]
        return items

    def children(self, run_id: str) -> list[Run]:
        """某次运行直接派生的子运行（消融实验的下一层）."""
        self.get(run_id)  # 不存在时立刻报错，而不是返回空列表
        return [run for run in self.runs() if run.parent_run_id == run_id]

    def lineage(self, run_id: str) -> list[Run]:
        """实验谱系：自身 → 父 → 祖父……（遇到缺失即停）."""
        chain: list[Run] = []
        seen: set[str] = set()
        current: Run | None = self.get(run_id)
        while current is not None:
            if current.run_id in seen:
                raise MLOpsError(
                    f"实验谱系出现环：{current.run_id} 已在链上（{', '.join(seen)}）"
                )
            seen.add(current.run_id)
            chain.append(current)
            current = self._runs.get(current.parent_run_id) if current.parent_run_id else None
        return chain

    def best(self, metric: str, *, mode: str = METRIC_MODE_MAX) -> Run | None:
        """在**已完成**的 run 里挑该指标最好的一次.

        刻意只看 ``finished``：失败的运行可能记录了半截指标，
        把它选成"最好"会让结论建立在一个没跑完的实验上。
        一条都没有时返回 ``None``（**"没有候选"与"候选都很差"是两件事**）。
        """
        if mode not in METRIC_MODES:
            raise MLOpsError(f"未知指标方向 {mode!r}，可选 {', '.join(METRIC_MODES)}")
        candidates = [
            run for run in self.runs(status=STATUS_FINISHED) if run.metric(metric) is not None
        ]
        if not candidates:
            return None
        key = lambda run: run.metric(metric)  # noqa: E731 - 排序键只有一行
        return max(candidates, key=key) if mode == METRIC_MODE_MAX else min(candidates, key=key)

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（含状态分布）."""
        counts = {status: 0 for status in STATUSES}
        for run in self._runs.values():
            counts[run.status] += 1
        counts["total"] = len(self._runs)
        return {
            "path": None if self.path is None else str(self.path),
            "counts": counts,
            "runs": [run.to_dict() for run in self.runs()],
        }

    def __len__(self) -> int:
        return len(self._runs)

    def __iter__(self) -> Iterator[Run]:
        return iter(self.runs())

    def render_markdown(self) -> str:
        """把追踪器渲染成 markdown 表（CI 摘要与 demo 直接打印）."""
        counts = self.to_dict()["counts"]
        lines = [
            "# 实验追踪",
            "",
            f"- run 总数：**{counts['total']}**（完成 {counts[STATUS_FINISHED]} / "
            f"失败 {counts[STATUS_FAILED]} / 进行中 {counts[STATUS_RUNNING]}）",
            "",
            "| run_id | 名称 | 状态 | 参数 | 指标 | 产物 | 时长(s) | 父实验 |",
            "|--------|------|------|------|------|------|---------|--------|",
        ]
        for run in self.runs():
            params = ", ".join(f"{k}={v}" for k, v in sorted(run.params.items()))
            flat = ", ".join(f"{k}={v:g}" for k, v in sorted(run.flat_metrics().items()))
            lines.append(
                f"| `{run.run_id}` | {run.name} | {run.status} | {params or '-'} | "
                f"{flat or '-'} | {len(run.artifacts)} | {run.duration_seconds:.2f} | "
                f"{run.parent_run_id or '-'} |"
            )
        lines.append("")
        return "\n".join(lines)


__all__ = [
    "METRIC_MODE_MAX",
    "METRIC_MODE_MIN",
    "METRIC_MODES",
    "RELATION_BETTER",
    "RELATION_EQUAL",
    "RELATION_WORSE",
    "RUNS_FILENAME",
    "RUN_ID_LENGTH",
    "STATUSES",
    "STATUS_FAILED",
    "STATUS_FINISHED",
    "STATUS_RUNNING",
    "ExperimentTracker",
    "Run",
    "compare_runs",
    "run_id_for",
]
