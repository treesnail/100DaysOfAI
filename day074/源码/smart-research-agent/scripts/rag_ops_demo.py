"""离线演示：RAG 生产化运维的十个断面（day072 / M6-D10）.

跑法：

```bash
python scripts/rag_ops_demo.py                 # 跑完十节并写 outputs/rag_ops_demo.txt
python scripts/rag_ops_demo.py --write-compose # 顺带把 docker-compose.rag.yml 重新渲染一遍
```

全部离线：向量后端用 flat、编码器用默认的 char-ngram，**零网络、零 API Key**。
产物只写在 ``outputs/rag_ops_demo_work/`` 下（幂等，可随时删），
只有 ``--write-compose`` 会碰仓库根目录的那份 yml（它本来就该由规格渲染出来）。

十节里最值得看的是第 5、6、7、10 节：

```text
5   两级增量的差额：文件级改了 1 份，块级写了 2 块（同一件事的两个粒度）
6   换挡：文件级 67%（超阈值）→ 整库重建；而块级自己还会再判一次
7   弄坏它：语料目录不存在 → failed、水位不动、退避窗口里不再重试
10  探针：空库 503、同步后 200 —— curl -f 只看状态码，因此状态必须进状态码
```
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.indexing import (  # noqa: E402
    EmbeddingCache,
    IndexBackupStore,
    IndexBuilder,
    IndexingPipeline,
    IndexVersionStore,
    describe_embedding,
)
from smart_research_agent.llm.embedding import default_embedding  # noqa: E402
from smart_research_agent.rag_ops import (  # noqa: E402
    HEALTH_CHECKS,
    SERVICE_ROLES,
    VALIDATION_RULES,
    RagOpsPipeline,
    SchedulePolicy,
    SyncLedger,
    changed_lines,
    check_committed_compose,
    compose_summary_lines,
    default_spec,
    render_compose,
    scan_corpus,
    validate_spec,
    write_compose,
)
from smart_research_agent.rag_ops.types import to_iso  # noqa: E402
from smart_research_agent.vectorstore.registry import create_backend  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_NAME = "rag_ops_demo.txt"
WORK_DIR = OUTPUT_DIR / "rag_ops_demo_work"

CLOCK_START = datetime(2026, 10, 6, 3, 0, 0, tzinfo=timezone.utc)

CORPUS: dict[str, str] = {
    "rag.md": "# 检索增强生成\n\nRAG 把检索与生成结合起来。\n\n## 向量检索\n\n余弦相似度常用。\n",
    "ops.md": "# 运维\n\n索引需要定期增量更新。\n",
    "sched.md": "# 调度\n\n每天凌晨跑一次同步。\n",
}


class _Tee:
    """把 stdout 同时写终端与文件（与 scripts/indexing_demo.py 同一手法）."""

    def __init__(self, stream, handle) -> None:
        self._stream = stream
        self._handle = handle

    def write(self, text: str) -> int:
        """两边都写，返回写到终端的字符数."""
        self._handle.write(text)
        return self._stream.write(text)

    def flush(self) -> None:
        """两边都刷."""
        self._handle.flush()
        self._stream.flush()


class Clock:
    """可推动的固定时钟（让整场演示的每一行输出都可复现）."""

    def __init__(self, moment: datetime) -> None:
        self.moment = moment

    def __call__(self) -> datetime:
        """取当前时刻."""
        return self.moment

    def advance(self, **delta: float) -> None:
        """往前走一段."""
        self.moment = self.moment + timedelta(**delta)


def section(title: str) -> None:
    """打印一节的分隔标题."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def write_corpus(root: Path) -> Path:
    """把三份语料写进 ``root``（重复调用是幂等的）."""
    root.mkdir(parents=True, exist_ok=True)
    for name, body in CORPUS.items():
        (root / name).write_text(body, encoding="utf-8")
    return root


def build_pipeline(work: Path, corpus: Path, clock: Clock) -> RagOpsPipeline:
    """装配一条完全隔离的运维流水线（账本 / 报告 / 索引都写在演示目录下）."""
    backend = create_backend(backend="flat", metric="cosine", path="")
    embedding = default_embedding()
    identity = describe_embedding(embedding)
    versions = IndexVersionStore(path=str(work / "index"))
    backups = IndexBackupStore(path=str(work / "index" / "backups"))
    builder = IndexBuilder(
        backend,
        embedding,
        cache=EmbeddingCache(
            path="",
            provider=identity.provider,
            model=identity.model,
            dimension=identity.dimension,
        ),
        identity=identity,
        batch_size=8,
        versions=versions,
        backups=backups,
    )
    indexer = IndexingPipeline(builder, versions=versions, backups=backups)
    return RagOpsPipeline(
        indexer,
        store=backend,
        source_dir=str(corpus),
        policy=SchedulePolicy(
            interval_minutes=1440,
            jitter_seconds=300,
            max_lag_minutes=720,
            backoff_base_seconds=60,
            backoff_max_seconds=3600,
        ),
        ledger_path=str(work / "sync_ledger.json"),
        report_path=str(work / "ops_report.json"),
        persist_path=str(work / "index" / "vectors.json"),
        versions=versions,
        clock=clock,
    )


def show_report(title: str, report) -> None:
    """把一次运行的结论按固定顺序打印出来."""
    print(f"【{title}】")
    print(f"  状态行  {report.summary_line()}")
    print(f"  调度    {report.schedule.summary_line()}")
    print(f"          {report.schedule.reason}")
    if report.plan is not None:
        print(f"  差集    {report.plan.summary_line()}")
        print(f"  理由    {report.plan.reason}")
        for line in changed_lines(report.plan, limit=4):
            print(f"          {line}")
    if report.index is not None:
        print(
            f"  构建    mode={report.index['mode']} 写入 {report.index['written']} / "
            f"未变 {report.index['unchanged']} / 删除 {report.index['removed']} / "
            f"编码 {report.index['encoded']}（缓存命中 {report.index['cache_hits']}）"
        )
        print(f"          version={report.index['version_id']} ← {report.index['parent_version'] or '（首版）'}")
    if report.health is not None:
        print(f"  体检    {report.health.summary_line()}")
        for finding in report.health.findings:
            print(f"          {finding.summary_line()}")
    if report.metrics:
        print("  采样    " + " | ".join(item.summary_line() for item in report.metrics))
    for note in report.notes:
        print(f"  说明    {note}")
    if report.error:
        print(f"  错误    {report.error}")


def main() -> None:
    """十节演示."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    corpus = write_corpus(WORK_DIR / "corpus")
    clock = Clock(CLOCK_START)
    pipeline = build_pipeline(WORK_DIR, corpus, clock)

    # ------------------------------------------------------------------ 第 1 节
    section("第 1 节：语料与快照（扫盘一次，得到一份可 diff 的账）")
    snapshot, documents = scan_corpus(corpus, clock=clock())
    print(f"快照   {snapshot.summary_line()}")
    for entry in snapshot.entries:
        print(f"       {entry.summary_line()}")
    print(f"文档   {len(documents)} 份已被解析（同一次扫盘，文件只读一遍）")
    print("说明   快照号只由（路径，指纹）算出：不含时间、不含机器，因此可 diff")

    # ------------------------------------------------------------------ 第 2 节
    section("第 2 节：第一次运行（首次 → 全量）")
    first = pipeline.run_once()
    show_report("首次同步", first)

    # ------------------------------------------------------------------ 第 3 节
    section("第 3 节：同一天的第二次运行（未到点 → 什么都不做）")
    second = pipeline.run_once()
    show_report("未到点", second)
    print(f"  对照    落盘报告仍是上一次的：{pipeline.status()['last_report']['status']}")

    # ------------------------------------------------------------------ 第 4 节
    section("第 4 节：次日凌晨（到点，但语料一份都没变）")
    clock.advance(days=1, hours=1)
    third = pipeline.run_once()
    show_report("无变化", third)

    # ------------------------------------------------------------------ 第 5 节
    section("第 5 节：改一份文档（两级增量的差额）")
    (corpus / "ops.md").write_text(
        "# 运维\n\n索引需要定期增量更新。\n\n## 调度\n\n每天凌晨跑一次。\n",
        encoding="utf-8",
    )
    # 计划点带着 ≤300 秒的抖动，因此"整整一天"还差几秒：多走一小时再跑
    clock.advance(days=1, hours=1)
    fourth = pipeline.run_once()
    show_report("改一份", fourth)
    print("说明   文件级 sources_changed=1（谁变了），块级 chunks_written 见构建行")
    print("       两个粒度缺一不可：只有文件级会漏掉“要重算哪些块”，")
    print("       只有块级则每天都要把整个语料解析并切块一遍")

    # ------------------------------------------------------------------ 第 6 节
    section("第 6 节：再改两份（变更比例换挡 → 整库重建）")
    (corpus / "rag.md").write_text(
        "# 检索增强生成\n\nRAG 把检索与生成结合起来。\n\n## 向量检索\n\n余弦相似度最常用。\n",
        encoding="utf-8",
    )
    (corpus / "sched.md").write_text(
        "# 调度\n\n每天凌晨跑一次同步。\n\n## 退避\n\n失败之后指数退避。\n",
        encoding="utf-8",
    )
    clock.advance(days=1, hours=1)
    fifth = pipeline.run_once()
    show_report("两份都改", fifth)

    # ------------------------------------------------------------------ 第 7 节
    section("第 7 节：把它弄坏（语料目录不存在）")
    # 刻意换一个工作目录：账本与报告是**状态**，用同一份账本会让这一节
    # 直接落到"未到点"上，而它要演示的是"跑失败会怎样"。
    broken = build_pipeline(WORK_DIR / "broken", WORK_DIR / "gone", clock)
    failed = broken.run_once()
    show_report("目录不存在", failed)
    retry = broken.run_once()
    print()
    show_report("同一退避窗口内的第二次", retry)
    print("说明   失败**不动水位**：下一趟会重试同一批来源（重跑幂等，漏更新不幂等）")

    # ------------------------------------------------------------------ 第 8 节
    section("第 8 节：监控采样汇总（量到了什么，就报什么）")
    rows = pipeline.status()["metrics"]
    for row in rows:
        print(
            f"  {row['name']:<32} {row['latest']!s:<10} {row['unit']:<8} "
            f"{row['direction']:<14} （采样 {row['count']} 次，min {row['min']} / max {row['max']}）"
        )
    print("说明   quality_* 四个指标缺席：本进程没有跑过评估——")
    print("       “没量到”不会以 0 出现，否则看板会显示“幻觉率 0”，而那句话没有测量支撑")

    # ------------------------------------------------------------------ 第 9 节
    section("第 9 节：编排规格（三个服务 + 11 条静态判据）")
    spec = default_spec()
    problems = validate_spec(spec)
    print(f"角色    {' / '.join(SERVICE_ROLES)}")
    for line in compose_summary_lines(spec):
        print(f"        {line}")
    print(f"卷      {', '.join(spec.volumes)}")
    print(f"判据    {len(VALIDATION_RULES)} 条，结论：{'合规' if not problems else problems}")
    committed = Path("docker-compose.rag.yml")
    drift = check_committed_compose(spec, committed) if committed.exists() else "（文件不存在）"
    print(f"一致性  仓库里那份 yml：{drift or '与规格逐字节一致'}")
    print(f"渲染    同一份规格渲染两次是否相同：{render_compose(spec) == render_compose(spec)}")

    # ------------------------------------------------------------------ 第 10 节
    section("第 10 节：健康探针（空库 503、同步之后 200）")
    empty_pipeline = build_pipeline(WORK_DIR / "empty", corpus, clock)
    empty_health = empty_pipeline.health_report()
    print(f"空库    state={empty_health.state} exit_code={empty_health.exit_code}")
    for finding in empty_health.findings:
        print(f"        {finding.summary_line()}")
    healthy = pipeline.health_report()
    print()
    print(f"同步后  state={healthy.state} exit_code={healthy.exit_code}")
    for finding in healthy.findings:
        print(f"        {finding.summary_line()}")
    print(f"检查项  {', '.join(HEALTH_CHECKS)}")
    print("说明   探针只读账（库计数 / 清单 / 账本 / 已有评估结论），不跑评估：")
    print("       否则提供方一抖动，全部实例会被同一条探针一起摘掉")

    # ------------------------------------------------------------------ 收尾
    section("收尾：落盘产物与账本")
    ledger = SyncLedger.load(WORK_DIR / "sync_ledger.json")
    print(f"账本    {ledger.summary_line()}")
    print(f"报告    {pipeline.report_path}")
    print(f"索引    {WORK_DIR / 'index' / 'vectors.json'}")
    print(f"清单    {pipeline.current_manifest().summary_line()}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if "--write-compose" in sys.argv:
        write_compose(default_spec(), Path("docker-compose.rag.yml"))
        print("已重新渲染 docker-compose.rag.yml")
    with (OUTPUT_DIR / OUTPUT_NAME).open("w", encoding="utf-8") as handle:
        original = sys.stdout
        sys.stdout = _Tee(original, handle)
        try:
            main()
        finally:
            sys.stdout = original
    print(f"\n输出已写入 {OUTPUT_DIR / OUTPUT_NAME}")
