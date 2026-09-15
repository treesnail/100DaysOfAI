"""微调数据采集：把散落在仓库里的语料统一成 ``TrainingExample`` 列表（M5-D1）.

采集层要解决的不是"读文件"（那谁都会），而是**异构来源的语义对齐**：

- ``seed_examples.jsonl``：手写种子样本，直接就是 alpaca 格式，读进来即可；
- ``data/eval/agent_tasks.jsonl``：本课程 Agent 评估集，里面是"任务 + 期望
  工具 + 最终答案"的评测记录。它能变成训练数据——ReAct 工具调用轨迹正是
  我们想让模型学会的行为——但**必须先核对答案**：评估集只标注了"期望调用
  哪个工具"，其 ``Final Answer`` 并不保证正确（``task-04`` 把 7×8 答成 99）。
  所以带 ``answer_contains`` 且答案确实包含它 → 可用；不包含 → 打
  ``unverified`` 标签（随后被 ``default_rules`` 的 ``unverified_source``
  规则丢掉）；压根没有 ``answer_contains`` → 无法核对，直接跳过。
  这条"脏数据自动出局"的链路是刻意设计的：它证明了清洗规则真的在干活，
  而不是装饰。
- ``data/eval/redteam_cases.jsonl``：红队攻击用例。攻击 payload 本身不是
  答案，但"面对这类 payload 应当拒答"是**安全对齐**任务的正样本，所以
  把 payload 转成 ``instruction``、把统一的拒答文案作为 ``output``。

三个数据源实现同一个 ``DataSource`` 接口，``DataCollector`` 只负责"逐个
``load()``、拼成一个大列表、交给清洗器"——新增来源（例如从 HF datasets
拉取）只需再写一个子类，采集与清洗的其余部分零改动。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from smart_research_agent.config import settings
from smart_research_agent.finetune.cleaner import DatasetCleaner, FilterReport
from smart_research_agent.finetune.schema import (
    ALPACA,
    DEFAULT_FORMAT,
    DatasetFormatError,
    TrainingExample,
    parse_example,
)
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 安全对齐样本的固定拒答文案（模块常量，教程与测试都引用它）.
#: 刻意写死而不是"让模型自由发挥"：安全样本的 output 必须是稳定、可审计的
#: 标准动作；若由模型生成，不同批次会得到不同措辞，反而引入噪声。
SAFETY_REFUSAL_TEMPLATE = (
    "抱歉，这个请求涉及绕过安全限制或套取系统内部信息，我不能执行。"
    "如果你有正常的研究或学习需求，我很乐意在不违反安全规范的前提下提供帮助。"
)

#: 评估集任务的来源标识（同时用作 ``TrainingExample.source`` 与源名）
EVAL_TASK_SOURCE = "eval/agent_tasks"
#: 红队用例的来源标识
REDTEAM_SOURCE = "eval/redteam_cases"

#: 评估集轨迹样本的标签：``unverified`` 表示"最终答案没能通过 answer_contains 核对".
TAG_FROM_EVAL = "from:eval"
TAG_UNVERIFIABLE = "unverifiable"
TAG_REACT_TRACE = "react-trace"
TAG_UNVERIFIED = "unverified"
TAG_FROM_REDTEAM = "from:redteam"
TAG_SAFETY = "safety"


def iter_jsonl_records(path: str | Path, *, source_name: str) -> Iterator[dict]:
    """逐行读取 JSONL 并 yield 字典对象（坏行记 warning 后跳过）.

    容错是本函数的**职责**：线上采集遇到一行脏数据，正确的做法是记一笔
    警告继续跑完，而不是让整批数据全部作废。文件不存在同样只告警——
    "这个源还没准备好"不该让整条流水线崩掉。
    """
    target = Path(path)
    if not target.exists():
        logger.warning("数据源 %s：文件不存在，已跳过（%s）", source_name, target)
        return
    with target.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("%s：第 %d 行 JSON 解析失败，已跳过（%s）", source_name, lineno, exc)
                continue
            if not isinstance(record, dict):
                logger.warning("%s：第 %d 行不是 JSON 对象，已跳过", source_name, lineno)
                continue
            yield record


@dataclass
class SourceSpec:
    """一个 JSONL 数据源的描述：从哪读、按什么格式读、来源与许可证是什么.

    ``license`` 不是摆设：微调数据的许可证会沿链条传染到模型权重，
    "这条数据能不能商用"必须在采集那一刻就记下来，事后补是补不回来的。
    """

    name: str
    path: str | Path
    fmt: str = DEFAULT_FORMAT
    #: 来源类型（seed / eval / redteam / external），供治理报表分组
    kind: str = "jsonl"
    license: str = "unknown"


class DataSource(ABC):
    """数据源抽象：一个 ``load()`` 返回统一格式的样本列表.

    刻意保持最小接口（只有 ``load`` 与 ``name``）：数据源千差万别，
    但采集器只关心"给我一批 ``TrainingExample``"——接口越窄，
    新增来源的成本越低。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """源名（同时用作 ``DatasetBundle.source_kept`` 的键）."""

    @abstractmethod
    def load(self) -> list[TrainingExample]:
        """读取并解析该源的全部样本（坏行跳过，不整体失败）."""


class JSONLSource(DataSource):
    """通用 JSONL 源：按 ``SourceSpec.fmt`` 逐行 ``parse_example``.

    这是最常用的一类来源（种子样本、外部采购数据都走它）：坏行记 warning
    后跳过，单行失败不拖垮整批——采集层的容错策略与 ``load_jsonl``
    的严格策略形成互补。
    """

    def __init__(self, spec: SourceSpec):
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    def load(self) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        for record in iter_jsonl_records(self.spec.path, source_name=self.spec.name):
            try:
                example = parse_example(record, self.spec.fmt)
            except DatasetFormatError as exc:
                logger.warning("%s：样本解析失败，已跳过（%s）", self.spec.name, exc)
                continue
            if not example.source:
                # 样本自身没标来源时用 SourceSpec 补齐：来源可追溯是底线，
                # 缺了它后续无法按来源统计/回溯，也没法解释"这条数据哪来的"。
                example.source = self.spec.name
            if not example.license:
                example.license = self.spec.license
            examples.append(example)
        return examples


class EvalTaskSource(DataSource):
    """评估任务源：把 ``agent_tasks.jsonl`` 的 ReAct 轨迹转成训练样本.

    转换规则（对应"评估集轨迹为什么要打 unverified 标签"）：

    - 轨迹文本 = ``mock_responses`` 用 ``\\n`` 连接；最后一条含
      ``Final Answer:`` 的文本即最终答案；
    - 无 ``answer_contains`` → 该条无法核对答案（评估集只标注了"期望工具"），
      标记 ``("from:eval", "unverifiable")`` 并**跳过**，不进入数据集；
    - 有 ``answer_contains`` 且最终答案包含它 → 保留，标签
      ``("from:eval", "react-trace")``；
    - 有 ``answer_contains`` 但最终答案不包含 → 仍然解析出来，但打上
      ``unverified``，留给 ``default_rules`` 的 ``unverified_source``
      规则把它丢掉。**保留到这一步是刻意的**：清洗报告里出现
      ``unverified_source`` 计数，才能证明"评估集轨迹不能无脑当训练数据"
      这条结论在流水线里真的被执行了。

    统计口径（``kept``/``unverified``/``skipped``）在 ``load()`` 后可直接读取，
    也是教程引用的实测数字来源。
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        #: 通过 answer_contains 核对、可用的样本数（load 后有效）
        self.kept = 0
        #: 有 answer_contains 但答案对不上、被打 unverified 的样本数
        self.unverified = 0
        #: 没有 answer_contains、无法核对的样本数（已跳过）
        self.skipped = 0

    @property
    def name(self) -> str:
        return EVAL_TASK_SOURCE

    def load(self) -> list[TrainingExample]:
        self.kept = 0
        self.unverified = 0
        self.skipped = 0
        examples: list[TrainingExample] = []
        for record in iter_jsonl_records(self._path, source_name=self.name):
            task = str(record.get("task") or "")
            responses = record.get("mock_responses") or []
            trace = "\n".join(str(item) for item in responses)
            final_answer = next(
                (str(item) for item in reversed(responses) if "Final Answer:" in str(item)),
                "",
            )
            expected = record.get("answer_contains")
            if not task or not trace:
                logger.warning("%s：样本缺少 task 或 mock_responses，已跳过", self.name)
                continue
            if not expected:
                # 无法核对答案：评估集里的轨迹只标注了"期望工具"，
                # 没有 answer_contains 就没有判对错的标准，不进入训练集。
                self.skipped += 1
                continue
            verified = str(expected) in final_answer
            if verified:
                self.kept += 1
                tags = (TAG_FROM_EVAL, TAG_REACT_TRACE)
            else:
                self.unverified += 1
                tags = (TAG_FROM_EVAL, TAG_REACT_TRACE, TAG_UNVERIFIED)
            examples.append(
                TrainingExample(
                    instruction=task,
                    output=trace,
                    source=EVAL_TASK_SOURCE,
                    tags=tags,
                )
            )
        return examples


class RedTeamSource(DataSource):
    """红队用例源：把攻击 payload 转成"应当拒答"的安全对齐样本.

    为什么攻击样例能变成训练数据：对齐训练需要的是"面对这类请求，
    标准动作是什么"的示范。把 payload 当 ``instruction``、把统一拒答
    文案当 ``output``，就得到一条 (危险请求 → 拒绝) 的正样本。
    ``category`` 进 tags，方便按攻击类型统计覆盖率。
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)

    @property
    def name(self) -> str:
        return REDTEAM_SOURCE

    def load(self) -> list[TrainingExample]:
        examples: list[TrainingExample] = []
        for record in iter_jsonl_records(self._path, source_name=self.name):
            payload = str(record.get("payload") or "")
            if not payload:
                logger.warning("%s：样本缺少 payload，已跳过", self.name)
                continue
            category = str(record.get("category") or "unknown")
            examples.append(
                TrainingExample(
                    instruction=payload,
                    output=SAFETY_REFUSAL_TEMPLATE,
                    source=REDTEAM_SOURCE,
                    tags=(TAG_FROM_REDTEAM, TAG_SAFETY, category),
                )
            )
        return examples


@dataclass
class DatasetBundle:
    """一次采集 + 清洗的完整产物：样本、清洗报告、按源的保留数."""

    examples: list[TrainingExample] = field(default_factory=list)
    report: FilterReport = field(default_factory=lambda: FilterReport(0, 0, 0, 0))
    source_kept: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.examples)


class DataCollector:
    """数据采集器：逐个源 ``load()`` → 合并 → 交给清洗器 → 汇总归因.

    用法::

        collector = default_collector()
        bundle = collector.collect()
        bundle.source_kept    # {"seed": 14, "eval/agent_tasks": 5, ...}

    去重是**全局**的（不是每个源各自去重）：同一个问题从种子和评估集
    两条路进来时，保留首次出现的那条，避免跨源重复。
    """

    def __init__(
        self, sources: Sequence[DataSource], cleaner: DatasetCleaner | None = None
    ):
        self.sources = list(sources)
        self.cleaner = cleaner or DatasetCleaner()

    def collect(self) -> DatasetBundle:
        """采集全部数据源并清洗，返回数据集包（含按源的保留数）."""
        loaded: list[TrainingExample] = []
        source_names: list[str] = []
        for source in self.sources:
            source_names.append(source.name)
            items = source.load()
            logger.info("数据源 %s 载入 %d 条样本", source.name, len(items))
            loaded.extend(items)

        kept, report = self.cleaner.run(loaded)
        kept_by_source = Counter(example.source for example in kept)
        source_kept = {name: kept_by_source.get(name, 0) for name in source_names}
        # 样本自报的来源未在 sources 里出现时也补进统计，避免"账对不上"
        for source_name, count in kept_by_source.items():
            source_kept.setdefault(source_name, count)
        return DatasetBundle(examples=kept, report=report, source_kept=source_kept)

    def source_stats(self) -> dict[str, int]:
        """每个源在清洗后实际贡献的样本数（键为源名）.

        语义是"**保留数**"而非"载入数"：采集了 100 条但全被清洗掉，
        对数据集而言贡献就是 0——这个差值恰恰是数据质量的信号。
        """
        return dict(self.collect().source_kept)


def default_collector(data_dir: str | Path | None = None) -> DataCollector:
    """按配置装配默认采集器：种子样本 + 评估轨迹 + 红队安全样本.

    ``data_dir`` 缺省读 ``settings.finetune_data_dir``（``data/finetune``），
    评估集固定在其同级目录 ``data/eval/`` 下（``data`` 是项目的数据根）。
    三个源的清洗阈值全部来自 ``settings.finetune_*`` 配置项，
    测试传 ``tmp_path`` 即可完全离线地换一套数据。
    """
    if data_dir is None:
        base = Path(settings.finetune_data_dir)
        seed_path = Path(settings.finetune_seed_path)
    else:
        base = Path(data_dir)
        seed_path = base / "seed_examples.jsonl"
    eval_dir = base.parent / "eval"

    sources: list[DataSource] = [
        JSONLSource(
            SourceSpec(
                name="seed",
                path=seed_path,
                fmt=ALPACA,
                kind="seed",
                license="CC-BY-4.0",
            )
        ),
        EvalTaskSource(eval_dir / "agent_tasks.jsonl"),
        RedTeamSource(eval_dir / "redteam_cases.jsonl"),
    ]
    cleaner = DatasetCleaner(
        min_output_chars=settings.finetune_min_output_chars,
        max_output_chars=settings.finetune_max_output_chars,
        max_instruction_chars=settings.finetune_max_instruction_chars,
    )
    return DataCollector(sources, cleaner=cleaner)
