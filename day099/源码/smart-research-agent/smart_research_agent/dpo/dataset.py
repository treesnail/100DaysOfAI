"""领域偏好数据集的构造：把"金标准答案"变成 ``(prompt, chosen, rejected)``（M5-D7）.

day054 用的是 12 条手写种子偏好对——它足够讲清原理，但**不能说明偏好数据
从哪来**。本课要真的训一次 DPO，就必须先回答一个更基础的问题：

.. code-block:: text

    SmartResearch Agent 的偏好数据，从哪来？

公开的偏好数据集（UltraFeedback、HH-RLHF 等）都是通用域、英文为主，
与本项目的"引用规范 / 段落格式 / 拒答边界"三个维度不对齐。真实工程里有
两条可用的路，本模块把两条都实现出来，因为它们的取舍完全不同：

======================================  ==========================================================
**拒绝采样（rejection sampling）**        让模型对同一 prompt 采 K 个回答，用规则或裁判打分，
                                        取最高分为 chosen、最低分为 rejected。**贵**（K 倍推理），
                                        但数据分布与策略一致，是"在线"方法。
**规则劣化（degradation）**               以金标准答案为 chosen，用可复现的劣化算子把它"改坏"
                                        成 rejected。**便宜**、完全离线、可审计，但 rejected
                                        的分布是人工构造的，与模型真实错误不同。
======================================  ==========================================================

本模块用**两者的交集**：劣化算子负责**造出候选池**（离线、确定、可复现），
规则打分器负责**排序**（与拒绝采样完全同一段代码：``score_candidate``）。
于是"候选从哪来"可以换（真实训练时换成模型采样即可），而"谁更好"的判据
只有一份实现——**判据一分为二，"chosen 到底为什么更好"就答不出来了**。

## 四个劣化算子对应四个真实故障

每个算子都是"把答案改坏的一种方式"，且**与 day054 的五个对齐维度对齐**
（维度定义见 ``alignment.preference.ALIGNMENT_DIMENSIONS``）：

========================  ==================  ============================================
``drop_tail``             ``conciseness``?    删掉最后一句：答了一半，信息不完整
``vague_specifics``       ``honesty``          把具体数字/术语换成模糊说法：用"大概"把话说圆
``pad_verbose``           ``conciseness``      追加套话与铺垫：更长、更啰嗦、信息量不变
``add_overclaim``         ``honesty``          追加无依据的效果承诺：把结论说过头
========================  ==================  ============================================

``drop_tail`` 归到哪一维是一个真实的选择题：它同时像"信息不全"（更像
``citation``）与"答得太短"（``conciseness``）。本模块把它记为
``conciseness`` 会误导（它其实**更短**），记为 ``citation`` 又会与"给出处"
这个定义冲突。**最后按"缺信息"归到 ``honesty`` 的邻域**：本课程最终把
``drop_tail`` 的维度标为 ``honesty``，理由是它造成的失败是"答案不足以支撑
结论"——与"编造"共享同一类后果（用户被误导）。这类"归类靠人判断、必须在
数据里写清楚"的取舍，正是 day054 ``DIMENSION_GOALS`` 存在的意义。

## 一条不能省的体检：金标准必须真的是最好的

``build_preferences`` 在每条样本上都会检查"打分最高的候选是不是金标准
答案"。**如果某个劣化算子的产物得分反而更高，这一条就被丢弃并计数**
（``drop_reasons["gold_not_best"]``），而不是照常写进数据集。
理由与 day053 的"参考答案必须能通过它自己的用例"完全一致：一条
"chosen 并不更好"的偏好样本在 DPO 里贡献的梯度方向是**反的**，
而它在统计里仍然算一条合格数据。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smart_research_agent.alignment.preference import (
    ALIGNMENT_DIMENSIONS,
    PreferencePair,
    preference_fingerprint,
)
from smart_research_agent.dpo.errors import DPOError
from smart_research_agent.finetune.cleaner import DatasetCleaner
from smart_research_agent.finetune.schema import TrainingExample
from smart_research_agent.sft.template import load_training_examples
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 项目根目录（``data/`` 相对它定位，与 ``mcp_server`` 的定位方式一致）.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: SFT 种子样本：偏好的 chosen 侧直接来自它（**同一批数据的两种用法**）.
DEFAULT_SEED_EXAMPLES = PROJECT_ROOT / "data" / "finetune" / "seed_examples.jsonl"

#: 安全偏好数据：由 day031 的红队 payload 派生（见 ``read_safety_pairs``）.
DEFAULT_SAFETY_PAIRS = PROJECT_ROOT / "data" / "eval" / "safety_pairs.jsonl"

#: TRL ``DPOTrainer`` 认识的三个列名。**多一列都会被 json 直读路径带进去**，
#: 所以默认只输出这三列；需要溯源时用 ``include_meta=True``。
TRL_COLUMNS: tuple[str, ...] = ("prompt", "chosen", "rejected")

#: 劣化算子全集；改这里必须同步改 ``_OPS`` 与文档表格.
DEGRADATION_OPS: tuple[str, ...] = (
    "drop_tail",
    "vague_specifics",
    "pad_verbose",
    "add_overclaim",
)

#: 每个算子把答案改坏的方式（进文档与 API 响应）.
OP_REASONS: dict[str, str] = {
    "drop_tail": "删掉最后一句：答案里少了一部分依据，结论不再被完整支撑",
    "vague_specifics": "把具体数字与精确说法换成模糊表述：用“大概”把话说圆",
    "pad_verbose": "追加套话与铺垫：信息量不变但明显更长，读者要自己筛",
    "add_overclaim": "追加无依据的效果承诺：把可行结论说过头",
}

#: 每个算子造成的失败，语义上属于 day054 的哪一维.
OP_DIMENSION: dict[str, str] = {
    "drop_tail": "honesty",
    "vague_specifics": "honesty",
    "pad_verbose": "conciseness",
    "add_overclaim": "honesty",
}

#: 长于金标准的候选要扣分（每超出 1 倍扣 0.5 分）.
#:
#: 为什么长度必须进打分器而不是只进体检：``pad_verbose`` 与 ``add_overclaim``
#: 的**覆盖度与金标准完全相同**（它们只是追加），如果打分器只看覆盖度，
#: 这两个算子造出的候选会与金标准打成平手——"更啰嗦"这件事就白白丢掉了。
OVERLENGTH_PENALTY = 0.5

#: 关键短语的最短长度（字符）：低于它的片段多半是"是/的/与"这类虚词.
MIN_PHRASE_CHARS = 3

#: chosen 与 rejected 的得分差下限：低于它就丢弃这一对.
#:
#: 与 day054 ``near_duplicates`` 同一个理由，但检查的是**打分差**而不是
#: 文本相似度：得分差接近 0 的一对几乎不产生梯度（margin 接近 0），
#: 却照样占一条样本的位置。
MIN_SCORE_GAP = 0.05

#: rejected 的挑选方式.
#:
#: - ``worst``：取打分最低的候选。语义上最像"拒绝采样"，但**实测它会
#:   让四个算子退化成两个**（本课实测：``drop_tail`` 8 / ``pad_verbose`` 8，
#:   ``vague_specifics`` 与 ``add_overclaim`` 一条都没进数据）——因为
#:   "删掉半数句子"永远比"多说两句废话"错得更狠；
#: - ``rotate``：按样本序号在适用算子之间轮转。它牺牲了"rejected 一定是最差"
#:   这一点，换来**四个算子各占四分之一**，于是"模型对哪一类改坏方式的
#:   区分度最差"这个问题才有数据可答。
#:
#: 默认取 ``rotate``，理由是**覆盖度比单条样本的极端性更重要**：一份只含
#: 两种失败模式的数据，训出来的策略在另外两种失败模式上是没有信号的。
SELECTION_MODES: tuple[str, ...] = ("rotate", "worst")
DEFAULT_SELECTION = "rotate"

_SENTENCE_SPLIT = re.compile(r"[。！？；\n]+")
#: 句内片段分隔符：逗号、顿号、冒号、分号（中英文都收）
_CLAUSE_SPLIT = re.compile(r"[，、：,;]+")
#: 片段必须含有至少一个「中日韩字符或字母数字」，否则视为纯符号噪声
_MEANINGFUL = re.compile(r"[\u4e00-\u9fff\w]")
_DIGITS = re.compile(r"\d+(?:\.\d+)?")

#: 追加类算子使用的固定文本：**必须足够长**（否则长度比打不出差异），
#: 且不含任何"编造的事实"——它们是纯铺垫，不引入新的错误信息。
_VERBOSE_FILLER = (
    "总的来说，这个问题在工程实践中非常常见，需要结合具体场景综合考虑，"
    "建议在实际落地时多做几组对比实验，逐步调优，才能得到比较理想的效果。"
)
_OVERCLAIM_FILLER = (
    "按这个方案做，效果一定会非常好，几乎可以解决所有同类问题，"
    "业界已验证多年，直接照搬即可，不需要再做额外验证。"
)


@dataclass(frozen=True)
class CandidateScore:
    """一个候选回答的打分明细（四个量都必须交出来）."""

    op: str
    text: str
    coverage: float
    length_ratio: float
    penalty: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典（正文只留前 80 字，报告不塞长文本）."""
        payload = asdict(self)
        payload["text"] = self.text[:80]
        return payload


def key_phrases(text: str) -> tuple[str, ...]:
    """抽取"关键短语"：按句 → 按逗号顿号切分，保留有意义的片段.

    与 day053 的"事实点"是同一类东西，但**粒度更细**：事实点是人工标注的
    "必须说到的知识点"，这里是从金标准答案里机械切出来的片段。用它做覆盖率
    打分的好处是**完全确定、可复现、可手算**；坏处也要说清楚——它会
    惩罚同义改写（把"降低幻觉"写成"减少编造"就匹配不上）。真实训练里
    这一层应当换成语义匹配或模型裁判，本课的取舍是"先让机制可见"。
    """
    phrases: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(text):
        for clause in _CLAUSE_SPLIT.split(sentence):
            stripped = clause.strip().strip("“”\"'()（）[]【】。，、：；!?！？ ")
            if len(stripped) < MIN_PHRASE_CHARS or not _MEANINGFUL.search(stripped):
                continue
            if stripped not in phrases:
                phrases.append(stripped)
    return tuple(phrases)


def score_candidate(gold: str, candidate: str, *, op: str = "gold") -> CandidateScore:
    """给一个候选回答打分：**覆盖率 − 超长惩罚**.

    打分器只有两项，是刻意的：day054 的五个维度里，``format`` / ``refusal``
    需要结构化判定（不是字符级覆盖率能表达的），本课把它们留给手写种子数据
    与安全偏好集（``read_safety_pairs``）。**打分器覆盖不了的维度就不要假装
    能打分**——用一个字符级覆盖率冒充"格式合规"，比对未知的诚实还要糟。
    """
    if not gold.strip():
        raise DPOError("金标准答案不能为空：没有它就无法给候选打分")
    if not candidate.strip():
        raise DPOError("候选回答不能为空：空回答的打分无定义（不是 0 分）")
    phrases = key_phrases(gold)
    if not phrases:
        raise DPOError(
            f"金标准答案里抽不出任何关键短语（长度 {len(gold)} 字符）：无法打分"
        )
    covered = sum(1 for phrase in phrases if phrase in candidate)
    coverage = covered / len(phrases)
    length_ratio = len(candidate) / len(gold)
    penalty = OVERLENGTH_PENALTY * max(0.0, length_ratio - 1.0)
    return CandidateScore(
        op=op,
        text=candidate,
        coverage=round(coverage, 6),
        length_ratio=round(length_ratio, 6),
        penalty=round(penalty, 6),
        score=round(coverage - penalty, 6),
    )


def degrade(text: str, op: str) -> str | None:
    """按算子把答案改坏；**不适用时返回 ``None``（而不是原样返回）**.

    返回 ``None`` 而不是原文是关键：如果这里返回原文，``build_preferences``
    会得到一个"与 chosen 逐字相同"的 rejected 候选，而这类样本会在
    ``PreferencePair`` 构造期直接抛错（day054 的保护）。让"不适用"
    在算子内部就被表达出来，比让下游去发现"两者相同"更早、更明确。
    """
    if op not in DEGRADATION_OPS:
        raise DPOError(f"未知的劣化算子 {op!r}，可选：{', '.join(DEGRADATION_OPS)}")
    if op == "drop_tail":
        return _drop_tail(text)
    if op == "vague_specifics":
        return _vague_specifics(text)
    if op == "pad_verbose":
        return _append(text, _VERBOSE_FILLER)
    return _append(text, _OVERCLAIM_FILLER)


def _drop_tail(text: str) -> str | None:
    """删掉最后一句（只剩一句时视为不适用）."""
    sentences = [part for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    if len(sentences) < 2:
        return None
    kept = "。".join(sentence.strip() for sentence in sentences[:-1])
    return f"{kept}。" if kept.strip() else None


def _vague_specifics(text: str) -> str | None:
    """把数字改成模糊说法（没有数字时视为不适用）.

    只替换数字，**不替换术语**：把 "LoRA" 换成"某种方法"会让这一对变成
    "答错了知识点"而不是"答得不精确"，那是对齐到另一个维度去了。
    """
    if not _DIGITS.search(text):
        return None
    replaced = _DIGITS.sub("大约几个", text)
    return replaced if replaced != text else None


def _append(text: str, filler: str) -> str | None:
    """在末尾追加一段铺垫（文本为空时不适用）."""
    if not text.strip():
        return None
    return f"{text}{filler}"


def candidates_for(example: TrainingExample, *, ops: tuple[str, ...] = DEGRADATION_OPS) -> list[CandidateScore]:
    """一条样本的候选池：金标准 + 所有适用的劣化产物（按得分降序）.

    金标准永远在第一位（它的覆盖率是 1.0），排序只是为了**让调用方看得见
    差距**；真正决定 chosen / rejected 的是 ``build_preferences``。
    """
    scores = [score_candidate(example.output, example.output, op="gold")]
    for op in ops:
        degraded = degrade(example.output, op)
        if degraded is None:
            continue
        scores.append(score_candidate(example.output, degraded, op=op))
    return sorted(scores, key=lambda item: (-item.score, item.op))


@dataclass
class PreferenceBuildReport:
    """偏好数据构造报告：**每一次丢弃都要有名字**."""

    total_examples: int
    cleaned_examples: int
    built: int
    drop_reasons: dict[str, int] = field(default_factory=dict)
    by_op: dict[str, int] = field(default_factory=dict)
    cleaner: dict[str, Any] = field(default_factory=dict)

    @property
    def skip_rate(self) -> float:
        """丢弃率（空输入按 1.0 计：没有数据不等于"全都合格"）."""
        if self.total_examples == 0:
            return 1.0
        return 1.0 - self.built / self.total_examples

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        return {
            "total_examples": self.total_examples,
            "cleaned_examples": self.cleaned_examples,
            "built": self.built,
            "skip_rate": round(self.skip_rate, 4),
            "drop_reasons": dict(self.drop_reasons),
            "by_op": dict(self.by_op),
            "cleaner": self.cleaner,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        reasons = ", ".join(f"{key}={value}" for key, value in self.drop_reasons.items())
        return (
            f"偏好数据构造 | 输入 {self.total_examples} → 清洗后 {self.cleaned_examples} "
            f"→ 成对 {self.built} | 算子分布 {self.by_op} | 丢弃 {reasons or '无'}"
        )


def pick_rejected(
    pool: list[CandidateScore],
    *,
    index: int,
    ops: tuple[str, ...] = DEGRADATION_OPS,
    selection: str = DEFAULT_SELECTION,
) -> CandidateScore | None:
    """从候选池里挑一个作为 rejected（``None`` 表示这一条没有可用候选）.

    ``rotate`` 的轮转从 ``index % len(ops)`` 起步，遇到不适用的算子顺延；
    这样"哪一条样本用哪个算子"完全由序号决定，**不需要随机数**，
    同一份输入永远给出同一份数据。
    """
    if selection not in SELECTION_MODES:
        raise DPOError(
            f"未知的挑选方式 {selection!r}，可选：{', '.join(SELECTION_MODES)}"
        )
    degraded = [item for item in pool if item.op != "gold"]
    if not degraded:
        return None
    if selection == "worst":
        return degraded[-1]
    start = index % len(ops) if ops else 0
    for offset in range(len(ops)):
        op = ops[(start + offset) % len(ops)]
        match = next((item for item in degraded if item.op == op), None)
        if match is not None:
            return match
    return degraded[0]  # pragma: no cover - degraded 非空且 ops 覆盖全部算子时到不了


def build_preferences(
    examples: list[TrainingExample],
    *,
    cleaner: DatasetCleaner | None = None,
    ops: tuple[str, ...] = DEGRADATION_OPS,
    min_gap: float = MIN_SCORE_GAP,
    selection: str = DEFAULT_SELECTION,
) -> tuple[list[PreferencePair], PreferenceBuildReport]:
    """把 SFT 样本构造成偏好对，返回 ``(偏好对, 构造报告)``.

    流程（顺序不能换）：

    1. **清洗**：``DatasetCleaner`` 的三段管道（干净化 → 规则 → 去重）先跑，
       因为占位符答案（"TODO：待补充"）不能当 chosen——它是**金标准侧**
       的质量问题，而偏好数据对 chosen 的要求比对 SFT 输出更高；
    2. **造候选**：金标准 + 每个适用算子一份劣化产物；
    3. **自洽体检**：得分最高的必须是金标准，否则丢弃并计入
       ``gold_not_best``；
    4. **区分度体检**：``score(gold) − score(rejected) >= min_gap``，
       否则丢弃并计入 ``insufficient_gap``。
    """
    if not examples:
        raise DPOError("构造偏好数据至少需要一条 SFT 样本")
    if min_gap < 0:
        raise DPOError(f"min_gap 不能为负数，收到 {min_gap}")
    active_cleaner = cleaner if cleaner is not None else DatasetCleaner(min_output_chars=40)
    cleaned, filter_report = active_cleaner.run(examples)
    pairs: list[PreferencePair] = []
    drop_reasons: dict[str, int] = {}
    by_op: dict[str, int] = {op: 0 for op in ops}

    def _drop(reason: str) -> None:
        drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

    for index, example in enumerate(cleaned):
        pool = candidates_for(example, ops=ops)
        gold = next(item for item in pool if item.op == "gold")
        if pool[0].op != "gold":
            _drop("gold_not_best")
            continue
        rejected = pick_rejected(pool, index=index, ops=ops, selection=selection)
        if rejected is None:
            _drop("no_degradation")
            continue
        gap = round(gold.score - rejected.score, 6)
        if gap < min_gap:
            _drop("insufficient_gap")
            continue
        by_op[rejected.op] = by_op.get(rejected.op, 0) + 1
        pairs.append(
            PreferencePair(
                id=f"dpo-{index + 1:03d}",
                prompt=example.prompt_text,
                chosen=gold.text,
                rejected=rejected.text,
                dimension=OP_DIMENSION.get(rejected.op, "honesty"),
                metadata={
                    "op": rejected.op,
                    "score_gap": gap,
                    "chosen_score": gold.score,
                    "rejected_score": rejected.score,
                    "rejected_length_ratio": rejected.length_ratio,
                    "selection": selection,
                    "tags": list(example.tags),
                    "source": example.source,
                },
            )
        )
    report = PreferenceBuildReport(
        total_examples=len(examples),
        cleaned_examples=len(cleaned),
        built=len(pairs),
        drop_reasons=drop_reasons,
        by_op={name: count for name, count in by_op.items() if count},
        cleaner=filter_report.to_dict(),
    )
    logger.info("%s", report.summary_line())
    return pairs, report


def to_trl_rows(
    pairs: list[PreferencePair], *, include_meta: bool = False
) -> list[dict[str, Any]]:
    """投影成 TRL ``DPOTrainer`` 认识的列：``prompt`` / ``chosen`` / ``rejected``.

    列名不是随便定的：TRL 在拿到 ``datasets.Dataset`` 后会**按这三个名字取列**
    （``prompt``、``chosen``、``rejected``），换名字不会报"列缺失"这种
    友好错误，而是抛 ``KeyError`` 或把整段对话拼错。所以本模块把列名
    固化成 ``TRL_COLUMNS``，并由测试逐字断言。

    ``include_meta=True`` 时额外带上 id / 维度 / 算子：它们对训练无用，
    但让"这份 jsonl 里的每一行是怎么来的"可追溯——真实训练时这两列
    会被 ``DPOTrainer`` 忽略（多余列不会进模型），保留它们几乎零成本。
    """
    if not pairs:
        raise DPOError("不能把空的偏好数据集投影成 TRL 列")
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        row: dict[str, Any] = {
            "prompt": pair.prompt,
            "chosen": pair.chosen,
            "rejected": pair.rejected,
        }
        if include_meta:
            row["id"] = pair.id
            row["dimension"] = pair.dimension
            row["op"] = pair.metadata.get("op", "")
        rows.append(row)
    return rows


def write_trl_dataset(
    path: str | Path, pairs: list[PreferencePair], *, include_meta: bool = False
) -> Path:
    """写出一份可直接喂给 TRL 的 jsonl，返回落盘路径."""
    rows = to_trl_rows(pairs, include_meta=include_meta)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("TRL 偏好数据已写入 %s（%d 条）", target, len(rows))
    return target


def read_trl_dataset(path: str | Path) -> list[dict[str, Any]]:
    """读回 TRL jsonl 并校验三个必需列都存在.

    **校验放在读入侧**：真正让"列名写错"这件事变成硬错误的唯一位置就是这里。
    等到 ``DPOTrainer`` 里才炸，错误信息会指向 transformers 的内部栈。
    """
    source = Path(path)
    if not source.exists():
        raise DPOError(f"找不到 TRL 偏好数据文件：{source}")
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DPOError(f"TRL 数据第 {lineno} 行不是合法 JSON: {exc}") from exc
        missing = [column for column in TRL_COLUMNS if not str(payload.get(column, "")).strip()]
        if missing:
            raise DPOError(f"TRL 数据第 {lineno} 行缺列或为空：{', '.join(missing)}")
        rows.append(payload)
    if not rows:
        raise DPOError(f"TRL 偏好数据文件里没有任何样本：{source}")
    return rows


def read_safety_pairs(path: str | Path = DEFAULT_SAFETY_PAIRS) -> list[PreferencePair]:
    """读安全偏好数据（``data/eval/safety_pairs.jsonl``）.

    它与红队用例（``data/eval/redteam_cases.jsonl``，day031）的关系是
    **同题不同答**：payload 逐字复用（攻击话术不变），而每条补上两段范文——
    ``safe``（走拒答 + 说明边界）与 ``unsafe``（直接顺从）——于是安全能力
    第一次变成**可比较的偏好信号**，而不是"拦截率高不高"这一个数字。

    **内容红线（必须写清楚）**：两段范文都**不含任何可执行的有害步骤**。
    ``unsafe`` 侧只表达"顺从的立场"（例如"好的，我直接照做"），因为本课要
    训练/评估的是"**偏好拒答**"这件事本身，而不是"能不能造出有害内容"。
    这既是课程的安全边界，也是真实标注规范的做法：标注员写 rejected 时
    同样不应亲自编写可执行的有害内容。
    """
    source = Path(path)
    if not source.exists():
        raise DPOError(f"找不到安全偏好数据文件：{source}")
    pairs: list[PreferencePair] = []
    seen: set[str] = set()
    for lineno, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DPOError(f"安全偏好数据第 {lineno} 行不是合法 JSON: {exc}") from exc
        pair_id = str(payload.get("id", "")).strip()
        if pair_id in seen:
            raise DPOError(f"安全偏好数据第 {lineno} 行的 id 重复：{pair_id}")
        seen.add(pair_id)
        try:
            pairs.append(
                PreferencePair(
                    id=pair_id,
                    prompt=str(payload["prompt"]),
                    chosen=str(payload["safe"]),
                    rejected=str(payload["unsafe"]),
                    dimension="refusal",
                    metadata={"category": str(payload.get("category", ""))},
                )
            )
        except (KeyError, ValueError) as exc:
            raise DPOError(f"安全偏好数据第 {lineno} 行不合法: {exc}") from exc
    if not pairs:
        raise DPOError(f"安全偏好数据文件里没有任何样本：{source}")
    return pairs


def safety_category_table(pairs: list[PreferencePair]) -> dict[str, int]:
    """按攻击类别统计安全偏好样本数（空集合返回空字典，而不是抛异常）."""
    counts: dict[str, int] = {}
    for pair in pairs:
        category = str(pair.metadata.get("category", "")) or "unknown"
        counts[category] = counts.get(category, 0) + 1
    return counts


def dataset_report(pairs: list[PreferencePair]) -> dict[str, Any]:
    """偏好数据集的画像：维度分布、算子分布、得分差分布、长度比与指纹."""
    if not pairs:
        raise DPOError("不能对空的偏好数据集做画像")
    by_dimension = {name: 0 for name in ALIGNMENT_DIMENSIONS}
    by_op: dict[str, int] = {}
    gaps: list[float] = []
    ratios: list[float] = []
    for pair in pairs:
        by_dimension[pair.dimension] = by_dimension.get(pair.dimension, 0) + 1
        op = str(pair.metadata.get("op", ""))
        if op:
            by_op[op] = by_op.get(op, 0) + 1
        gap = pair.metadata.get("score_gap")
        if isinstance(gap, (int, float)):
            gaps.append(float(gap))
        ratios.append(pair.length_ratio)
    return {
        "total": len(pairs),
        "by_dimension": {name: count for name, count in by_dimension.items() if count},
        "by_op": by_op,
        "mean_score_gap": round(sum(gaps) / len(gaps), 6) if gaps else 0.0,
        "min_score_gap": round(min(gaps), 6) if gaps else 0.0,
        "mean_length_ratio": round(sum(ratios) / len(ratios), 4),
        "rejected_longer_fraction": round(
            sum(1 for pair in pairs if len(pair.rejected) > len(pair.chosen)) / len(pairs), 4
        ),
        "fingerprint": preference_fingerprint(pairs),
    }


def load_seed_examples(path: str | Path = DEFAULT_SEED_EXAMPLES) -> list[TrainingExample]:
    """读取 SFT 种子样本（复用 day050 的加载器，保证两课口径一致）."""
    return load_training_examples(path)


__all__ = [
    "DEFAULT_SAFETY_PAIRS",
    "DEFAULT_SEED_EXAMPLES",
    "DEFAULT_SELECTION",
    "DEGRADATION_OPS",
    "MIN_PHRASE_CHARS",
    "MIN_SCORE_GAP",
    "OP_DIMENSION",
    "OP_REASONS",
    "OVERLENGTH_PENALTY",
    "PROJECT_ROOT",
    "SELECTION_MODES",
    "TRL_COLUMNS",
    "CandidateScore",
    "PreferenceBuildReport",
    "build_preferences",
    "candidates_for",
    "dataset_report",
    "degrade",
    "key_phrases",
    "load_seed_examples",
    "pick_rejected",
    "read_safety_pairs",
    "read_trl_dataset",
    "safety_category_table",
    "score_candidate",
    "to_trl_rows",
    "write_trl_dataset",
]
