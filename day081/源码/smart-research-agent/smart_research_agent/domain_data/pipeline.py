"""领域数据流水线：把六个阶段串成一条可复现、可增量更新的链路（M5-D8）.

day048 的流水线是**一次性**的：采集 → 清洗 → 统计 → 切分 → 落盘。
它回答了"这一批数据能不能用"，但回答不了"下一批来了怎么办"。
本模块补上后半句，并沿用 day046 那条纪律——**顺序即策略**：

```text
clean → quality → near_dedup → mixing → augment → freeze
清洗     打分      两级去重     削峰      填谷      打指纹
```

每一步的顺序都有理由，换位就会出问题：

1. **clean 必须最先**：占位符样本、空指令必须先被硬规则拒掉，
   不能让它们带着分数进入后面的统计——`quality` 的分数分布会因此被拉偏，
   而"分数分布"是配比与增强的排序依据。
2. **quality 在去重之前**：先用便宜的五维打分把明显不合格的样本筛掉，
   再去跑昂贵的两两比较（``N2`` 次 64 维比较）。反过来也能跑，
   但会为一批注定被拒的样本付近重复检测的钱。
3. **near_dedup 在配比之前**：近重复会让某一组"看起来很多"。
   若不先去重，"配比"算的是重复项堆出来的占比，削掉的也是重复项——
   等于用一套虚拟的数量做了一轮真实的削减。
4. **mixing 在 augment 之前**：先削峰、后填谷。反过来（先增强再削峰），
   刚补进来的样本可能因为所属组超额而被立刻砍掉——白花一次增强。
5. **augment 在 freeze 之前**：增强是最后的"加法"，之后不再有任何
   会减少样本的步骤。这条约定让"最终数据集条数 = 前一步条数 + 新增"
   成为一个能一眼验算的等式。
6. **freeze 只做一件不可逆的事**：给最终数据集打内容指纹。
   指纹一旦算出，这个版本就被钉住了——它是下一次构建的 ``parent_fingerprint``。

**去重实际上被调用两次**：一次是上面的 ``near_dedup``（清理语料自身
"字面高度重叠"的重复采集），一次在 ``augment`` 内部（防止两条相近的
原样本被同一算子改成几乎一样的问法）。两次用的是同一个
``NearDuplicateIndex`` 类、同一个阈值，差别只在**索引里放了什么**：
第二次刻意**不放原样本**，否则增强样本会被自己的亲本判成重复
（本课实测会白白丢掉 5/13 次 ``prefix`` 产出）。第二次的账记在
``AugmentReport.dropped_duplicates`` 而不是某一份 ``DedupeReport`` 里
——报告的结构跟着**谁触发它**走，而不是跟着"概念上属于谁"走。

## 可持续更新：靠 ``merge_with_history`` 而不是重跑全量

``NearDuplicateIndex`` 支持增量登记，因此"历史数据集 + 新批次"的合并
不必把历史重新清洗、重新打分一遍：

```python
index = NearDuplicateIndex(threshold=0.8)
index.add_many(history)                  # 历史直接视为已接受
kept, report = index.filter(new_batch)   # 新批次跨批次去重
merged = list(history) + kept
```

代价是**顺序敏感**：新批次与历史撞车时丢弃的是新批次那一条。
这与"保留首次出现"是同一条规则——先入为主，且入的顺序被显式写进了
``DomainDataManifest.parent_fingerprint`` 的版本链里。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from smart_research_agent.config import settings
from smart_research_agent.domain_data.augment import (
    DEFAULT_MAX_PER_EXAMPLE,
    DEFAULT_OPS,
    AugmentOp,
    AugmentReport,
    augment_dataset,
    resolve_ops,
)
from smart_research_agent.domain_data.dedup import (
    DEFAULT_NEAR_DUP_THRESHOLD,
    DedupeReport,
    NearDuplicateIndex,
)
from smart_research_agent.domain_data.fingerprint import (
    DEFAULT_NUM_PERM,
    DEFAULT_SHINGLE_K,
    MINHASH_SEED,
    exact_key,
)
from smart_research_agent.domain_data.mixing import (
    DEFAULT_GROUP_BY,
    DEFAULT_MAX_GROUP_RATIO,
    MixDeficit,
    MixingReport,
    apply_mix_plan,
    deficit_report,
    group_counts,
    plan_mixing,
)
from smart_research_agent.domain_data.quality import (
    DEFAULT_QUALITY_THRESHOLD,
    QualityFilterResult,
    QualityWeights,
    default_weights,
    filter_by_quality,
    quality_scores,
    validate_threshold,
)
from smart_research_agent.finetune.cleaner import DatasetCleaner, FilterReport
from smart_research_agent.finetune.collector import DatasetBundle, default_collector
from smart_research_agent.finetune.dataset import dump_bundle
from smart_research_agent.finetune.schema import DEFAULT_FORMAT, TrainingExample

#: 六个阶段的固定顺序（与模块开头逐条对应）。测试会断言 ``run()`` 产出的
#: 阶段记录与这个元组逐项一致——"顺序即策略"如果只写在注释里，
#: 下一次重构就没人记得它为什么是这个顺序。
STAGE_ORDER: tuple[str, ...] = (
    "clean",
    "quality",
    "near_dedup",
    "mixing",
    "augment",
    "freeze",
)

#: 数据集清单文件名（落盘产物之一）
MANIFEST_FILENAME = "manifest.json"

#: 数据集内容指纹保留的十六进制字符数（与样本级精确指纹同一长度口径）
DATASET_FINGERPRINT_LENGTH = 16

#: 本模块首次出现的快照标签（沿用 ``sft.checkpoint.SNAPSHOT`` /
#: ``dpo.config.SNAPSHOT`` 的约定）。它进清单的 ``config``，因此一份
#: 落盘的数据集能自己说清"是哪一课的快照、按哪一版代码生成的"。
SNAPSHOT = "day057"


@dataclass
class StageRecord:
    """一个阶段的进/出计数与明细（沿用 day046 ``StageRecord`` 的记账形状）.

    三个派生量各有一个名字，**不共用一个"dropped"**：本流水线里
    ``augment`` 是唯一的加法阶段，"丢弃"算出负数（``-29``）读起来像错的，
    而评审看到负数时的第一反应是"这里的账错了"。所以把它拆成
    ``dropped``（只记减少）、``added``（只记增加）与 ``net_change``（净变化）：

    - ``dropped`` 与 ``added`` 恒非负，可以直接进报表；
    - ``net_change`` 是两者之差，也是"这个阶段让数据集变大了还是变小了"的答案；
    - 三者满足 ``kept - total_in == added - dropped == net_change``（测试钉住）。
    """

    name: str
    total_in: int
    kept: int
    detail: dict = field(default_factory=dict)

    @property
    def dropped(self) -> int:
        """本阶段**减少**的样本数（恒非负；增多的阶段为 0）."""
        return max(0, self.total_in - self.kept)

    @property
    def added(self) -> int:
        """本阶段**新增**的样本数（恒非负；只有 ``augment`` 阶段会为正）."""
        return max(0, self.kept - self.total_in)

    @property
    def net_change(self) -> int:
        """净变化量（有符号：减少为负、增加为正）."""
        return self.kept - self.total_in

    def to_dict(self) -> dict:
        """投影为可直接 json.dumps 的字典."""
        return {
            "name": self.name,
            "total_in": self.total_in,
            "kept": self.kept,
            "dropped": self.dropped,
            "added": self.added,
            "net_change": self.net_change,
            "detail": dict(self.detail),
        }


def dataset_fingerprint(examples: Sequence[TrainingExample]) -> str:
    """数据集内容指纹：逐条样本的 prompt 精确键按顺序拼接后的 SHA-1 前 16 位.

    三个刻意的选择：

    - **用 prompt 的精确键而不是整条样本的哈希**，与去重口径一致——
      数据集指纹要回答的是"这是不是同一批问题"，而不是"字节是否相同"；
    - **顺序敏感**：同样的样本换个顺序会得到不同指纹。这是刻意的：
      批次顺序会影响 day048 的可复现切分，也会影响训练时的样本先后，
      因此"顺序变了"就是"版本变了"，必须能被指纹察觉；
    - **只取 16 位十六进制**：指纹的用途是**版本链的人读标识**，
      不是密码学承诺。64 bit 在"同一个仓库的一百多个版本"这个量级上
      碰撞概率可以忽略，而短指纹能在报告里一行写完。
    """
    hasher = hashlib.sha1()
    for example in examples:
        hasher.update(exact_key(example.prompt_text).encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest()[:DATASET_FINGERPRINT_LENGTH]


@dataclass
class DomainDataManifest:
    """领域数据集清单：版本、指纹、阶段账、画像与配置快照.

    清单不是"日志"。日志描述过程，清单描述**结论**，因此它必须能被
    另一台机器在没有任何上下文的情况下读懂：这一版有多少条、按来源怎么分、
    质量分落在哪、上一版是谁、以及**所有影响结果的参数**（``config``）。
    缺了 ``config``，同一个指纹在别人机器上就复现不出来——
    而不可复现的指纹，只是一串好看的字符。
    """

    version: int
    fingerprint: str
    parent_fingerprint: str
    stages: list[StageRecord] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    raw_counts: dict[str, int] = field(default_factory=dict)
    groups: dict[str, int] = field(default_factory=dict)
    augment_by_op: dict[str, int] = field(default_factory=dict)
    quality: dict[str, float] = field(default_factory=dict)
    rejection_by_dimension: dict[str, int] = field(default_factory=dict)
    deficit: dict[str, int] = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        """最终数据集条数（= ``freeze`` 阶段的 kept）."""
        for stage in reversed(self.stages):
            if stage.name == "freeze":
                return stage.kept
        return 0

    def stage(self, name: str) -> StageRecord | None:
        """按名字取阶段记录（不存在返回 None，便于调用方兜底）."""
        return next((stage for stage in self.stages if stage.name == name), None)

    def group_ratios(self) -> dict[str, float]:
        """最终数据集按配比口径的占比（空集返回空字典）."""
        total = sum(self.groups.values())
        if not total:
            return {}
        return {name: count / total for name, count in self.groups.items()}

    def to_dict(self) -> dict:
        """投影为可直接 ``json.dumps`` 的字典（落盘的正是它）."""
        return {
            "version": self.version,
            "fingerprint": self.fingerprint,
            "parent_fingerprint": self.parent_fingerprint,
            "size": self.size,
            "stages": [stage.to_dict() for stage in self.stages],
            "counts": dict(self.counts),
            "raw_counts": dict(self.raw_counts),
            "groups": dict(self.groups),
            "group_ratios": {
                name: round(value, 4) for name, value in self.group_ratios().items()
            },
            "augment_by_op": dict(self.augment_by_op),
            "quality": dict(self.quality),
            "rejection_by_dimension": dict(self.rejection_by_dimension),
            "deficit": dict(self.deficit),
            "config": dict(self.config),
        }

    def render_markdown(self) -> str:
        """把清单渲染成 markdown（六阶段账 + 画像 + 配置）."""
        lines = [
            f"# 领域数据集 v{self.version} (`{self.fingerprint}`)",
            "",
            f"- 上一版：`{self.parent_fingerprint or '（首版）'}`",
            f"- 最终规模：**{self.size}** 条",
            f"- 来源分布：{self.counts}",
            f"- 配比（{self.config.get('group_by', 'source')}）：{self.groups}"
            f" → { {k: round(v, 4) for k, v in self.group_ratios().items()} }",
            f"- 质量分：{self.quality}",
            f"- 拒绝归因：{self.rejection_by_dimension or '无'}",
            f"- 增强算子分布：{self.augment_by_op or '无'}",
            f"- 配比缺口：{self.deficit or '无'}",
            "",
            "| 阶段 | 进 | 出 | 丢弃 | 新增 | 明细 |",
            "|------|----|----|------|------|------|",
        ]
        lines.extend(
            f"| `{stage.name}` | {stage.total_in} | {stage.kept} | {stage.dropped} | "
            f"{stage.added} | {json.dumps(stage.detail, ensure_ascii=False)} |"
            for stage in self.stages
        )
        lines.extend(["", "## 参数快照", "", "```json"])
        lines.append(json.dumps(self.config, ensure_ascii=False, indent=2, sort_keys=True))
        lines.extend(["```", ""])
        return "\n".join(lines)


@dataclass
class DomainDataRun:
    """一次流水线运行的完整产物：样本 + 清单 + 各阶段报告.

    所有中间报告都保留（而不是只留最终清单），因为"某条样本为什么没进
    最终数据集"是数据工程里最常被问到的问题，而回答它需要**逐阶段**的
    决策记录：它可能被清洗规则拒了、可能质量分不够、可能被判近重复、
    也可能被配比削掉了。只留一份总报告，这个问题就只能靠重跑复现。
    """

    examples: list[TrainingExample]
    manifest: DomainDataManifest
    clean_report: FilterReport
    quality_result: QualityFilterResult
    dedupe_report: DedupeReport
    mixing_report: MixingReport
    augment_report: AugmentReport

    def to_dict(self) -> dict:
        """投影为可直接 json.dumps 的摘要（不含样本正文）."""
        return {
            "manifest": self.manifest.to_dict(),
            "clean": self.clean_report.to_dict(),
            "quality": self.quality_result.to_dict(),
            "dedupe": self.dedupe_report.to_dict(),
            "mixing": self.mixing_report.to_dict(),
            "augment": self.augment_report.to_dict(),
            "size": len(self.examples),
        }

    def summary_lines(self) -> list[str]:
        """各阶段的一行摘要（demo 按顺序打印）."""
        return [
            f"采集：{self.clean_report.total} 条原始样本",
            f"清洗：保留 {self.clean_report.kept} / 丢弃 {self.clean_report.rejected}"
            f"（归因 {self.clean_report.drop_reasons or '无'}）",
            f"质量：{self.quality_result.summary_line()}",
            f"去重：{self.dedupe_report.summary_line()}",
            f"配比：{self.mixing_report.summary_line()}",
            f"增强：{self.augment_report.summary_line()}",
            f"冻结：v{self.manifest.version} `{self.manifest.fingerprint}` → "
            f"{len(self.examples)} 条",
        ]


class DomainDataPipeline:
    """领域数据流水线：一次 ``run`` 走完六个阶段并产出清单.

    用法::

        pipeline = default_pipeline()
        run = pipeline.run_from_sources()
        run.manifest.render_markdown()

    所有会影响结果的参数都在构造期固定（阈值、权重、算子、分组口径），
    因此"同一个 ``run`` 的输入 + 同一份参数 = 同一个指纹"这句话是有意义的。
    """

    def __init__(
        self,
        *,
        cleaner: DatasetCleaner | None = None,
        quality_weights: QualityWeights | None = None,
        quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
        near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
        shingle_k: int = DEFAULT_SHINGLE_K,
        num_perm: int = DEFAULT_NUM_PERM,
        seed: int = MINHASH_SEED,
        group_by: str = DEFAULT_GROUP_BY,
        max_group_ratio: float = DEFAULT_MAX_GROUP_RATIO,
        mixing: bool = True,
        augment: bool = True,
        augment_ops: Sequence[str] = DEFAULT_OPS,
        max_augment_per_example: int = DEFAULT_MAX_PER_EXAMPLE,
    ):
        self.cleaner = cleaner or DatasetCleaner()
        self.quality_weights = (
            quality_weights if quality_weights is not None else default_weights()
        )
        # 门槛在构造期就校验：超过 1 的门槛不会报错，只会把整批样本静默拒光。
        self.quality_threshold = validate_threshold(quality_threshold)
        self.near_dup_threshold = near_dup_threshold
        self.shingle_k = shingle_k
        self.num_perm = num_perm
        self.seed = seed
        self.group_by = group_by
        self.max_group_ratio = max_group_ratio
        self.mixing = mixing
        self.augment = augment
        self.max_augment_per_example = max_augment_per_example
        # 构造期解析算子：名字写错立刻报错，而不是在跑完前五个阶段之后
        # 才发现"增强这一步什么都没做"。
        self.ops: tuple[AugmentOp, ...] = resolve_ops(augment_ops)

    def config_snapshot(self) -> dict:
        """影响结果的参数快照（写进清单，指纹可复现的前提）."""
        return {
            "snapshot": SNAPSHOT,
            "min_output_chars": self.cleaner.min_output_chars,
            "max_output_chars": self.cleaner.max_output_chars,
            "max_instruction_chars": self.cleaner.max_instruction_chars,
            "dedupe_exact": self.cleaner.dedupe,
            "quality_weights": self.quality_weights.as_dict(),
            "quality_threshold": self.quality_threshold,
            "near_dup_threshold": self.near_dup_threshold,
            "shingle_k": self.shingle_k,
            "num_perm": self.num_perm,
            "seed": self.seed,
            "group_by": self.group_by,
            "max_group_ratio": self.max_group_ratio,
            "mixing_enabled": self.mixing,
            "augment_enabled": self.augment,
            "augment_ops": [op.name for op in self.ops],
            "max_augment_per_example": self.max_augment_per_example,
        }

    def new_index(self) -> NearDuplicateIndex:
        """按本流水线的参数新建一个近重复索引（三个阶段共用同一套参数）."""
        return NearDuplicateIndex(
            threshold=self.near_dup_threshold,
            k=self.shingle_k,
            num_perm=self.num_perm,
            seed=self.seed,
        )

    def run(
        self,
        examples: Sequence[TrainingExample],
        *,
        version: int = 1,
        parent_fingerprint: str = "",
        raw_counts: dict[str, int] | None = None,
        history: Sequence[TrainingExample] = (),
    ) -> DomainDataRun:
        """对一批**原始**样本执行六阶段流水线，返回完整产物.

        入参是原始样本（未清洗）：清洗是本流水线的第一站，
        而不是调用方的前置义务——否则"清洗阈值"就有两个来源，
        而 report 里的 ``drop_reasons`` 也就对不上。

        ``history`` 里的样本只进近重复索引（视为已经接受过的历史数据），
        **不进入本次产出**——它们已经在上一个版本里了。这是增量构建的
        最短路径：想要"历史 + 新批次"的合并结果，请用 ``merge_with_history``。
        """
        stages: list[StageRecord] = []

        # 阶段 1：清洗（硬规则 + 精确去重）
        cleaned, clean_report = self.cleaner.run(list(examples))
        stages.append(
            StageRecord(
                name="clean",
                total_in=len(examples),
                kept=len(cleaned),
                detail={
                    "drop_reasons": dict(clean_report.drop_reasons),
                    "exact_duplicates": clean_report.duplicates,
                    "keep_rate": round(clean_report.keep_rate, 4),
                },
            )
        )

        # 阶段 2：五维质量打分与门槛
        quality_result = filter_by_quality(
            cleaned, weights=self.quality_weights, threshold=self.quality_threshold
        )
        scored = quality_result.kept
        stages.append(
            StageRecord(
                name="quality",
                total_in=len(cleaned),
                kept=len(scored),
                detail={
                    "threshold": self.quality_threshold,
                    "distribution": quality_result.distribution(),
                    "rejection_by_dimension": quality_result.rejection_by_dimension(),
                },
            )
        )

        # 阶段 3：两级去重（历史样本先入索引 → 跨批次也生效）
        index = self.new_index()
        if history:
            index.add_many(list(history))
        deduped, dedupe_report = index.filter(scored)
        stages.append(
            StageRecord(
                name="near_dedup",
                total_in=len(scored),
                kept=len(deduped),
                detail={
                    "exact_duplicates": dedupe_report.exact_duplicates,
                    "near_duplicates": dedupe_report.near_duplicates,
                    "threshold": self.near_dup_threshold,
                    "history_size": len(history),
                },
            )
        )

        # 阶段 4：配比削峰（组内按质量分保留高分）
        scores = quality_scores(deduped, weights=self.quality_weights)
        if self.mixing:
            plan = plan_mixing(
                deduped, group_by=self.group_by, max_ratio=self.max_group_ratio
            )
            mixed, mixing_report = apply_mix_plan(deduped, plan, scores=scores)
        else:
            mixed, mixing_report = self._bypass_mixing(deduped)
        stages.append(
            StageRecord(
                name="mixing",
                total_in=len(deduped),
                kept=len(mixed),
                detail={
                    "enabled": self.mixing,
                    "iterations": mixing_report.iterations,
                    "feasible": mixing_report.feasible,
                    "dropped_by_group": mixing_report.dropped_by_group(),
                },
            )
        )

        # 阶段 5：增强填谷（内部再走一次去重，撞车的增强样本被丢弃）
        if self.augment and self.ops:
            augmented, augment_report = augment_dataset(
                mixed,
                ops=[op.name for op in self.ops],
                max_per_example=self.max_augment_per_example,
                dedupe=True,
                threshold=self.near_dup_threshold,
                k=self.shingle_k,
                num_perm=self.num_perm,
                seed=self.seed,
            )
        else:
            augmented, augment_report = self._bypass_augment(mixed)
        stages.append(
            StageRecord(
                name="augment",
                total_in=len(mixed),
                kept=len(augmented),
                detail={
                    "enabled": self.augment and bool(self.ops),
                    "generated": augment_report.generated,
                    "by_op": dict(augment_report.by_op),
                    "guard_rejected": dict(augment_report.guard_rejected),
                    "no_change": dict(augment_report.no_change),
                    "dropped_duplicates": augment_report.dropped_duplicates,
                },
            )
        )

        # 阶段 6：冻结（打内容指纹 + 汇总画像）
        fingerprint = dataset_fingerprint(augmented)
        stages.append(
            StageRecord(
                name="freeze",
                total_in=len(augmented),
                kept=len(augmented),
                detail={"fingerprint": fingerprint, "version": version},
            )
        )

        manifest = DomainDataManifest(
            version=version,
            fingerprint=fingerprint,
            parent_fingerprint=parent_fingerprint,
            stages=stages,
            counts=dict(group_counts(augmented, group_by="source")),
            raw_counts=dict(raw_counts or {}),
            groups=dict(group_counts(augmented, group_by=self.group_by)),
            augment_by_op=dict(augment_report.by_op),
            quality=quality_result.distribution(),
            rejection_by_dimension=quality_result.rejection_by_dimension(),
            deficit=self.deficit_of(augmented).deficits(),
            config=self.config_snapshot(),
        )
        return DomainDataRun(
            examples=list(augmented),
            manifest=manifest,
            clean_report=clean_report,
            quality_result=quality_result,
            dedupe_report=dedupe_report,
            mixing_report=mixing_report,
            augment_report=augment_report,
        )

    def _bypass_mixing(
        self, examples: list[TrainingExample]
    ) -> tuple[list[TrainingExample], MixingReport]:
        """配比被关闭时的等价产物：配额等于实际计数，一条不削.

        刻意返回**同一个 MixingReport 类型**而不是 ``None``：调用方（报告
        生成、API 序列化）因此不需要写"有没有配比"的分支，
        清单里的字段也永远是同一套形状。
        """
        counts = group_counts(examples, group_by=self.group_by)
        report = MixingReport(
            group_by=self.group_by,
            max_ratio=1.0,
            total_in=len(examples),
            total_out=len(examples),
            by_group_in=counts,
            by_group_out=dict(counts),
            iterations=0,
            feasible=True,
            warnings=["配比被显式关闭（max_ratio=1.0），未削减任何样本"],
        )
        return list(examples), report

    def _bypass_augment(
        self, examples: list[TrainingExample]
    ) -> tuple[list[TrainingExample], AugmentReport]:
        """增强被关闭（或算子表为空）时的等价产物."""
        report = AugmentReport(
            inputs=len(examples),
            applied=0,
            dropped_duplicates=0,
            threshold=self.near_dup_threshold,
        )
        return list(examples), report

    def deficit_of(
        self, examples: Sequence[TrainingExample], *, weights: dict[str, float] | None = None
    ) -> MixDeficit:
        """算当前数据集的配比缺口（等权目标；不裁剪任何样本）.

        等权是缺省目标而不是"随便取一个"：它对应一句能被说清的问题——
        "若我希望三个来源大致一样多，还差多少条？"。这个缺口数字就是
        ``augment`` 的靶子（本课的增强是无向的，定向补齐留给真实项目）。
        """
        counts = group_counts(examples, group_by=self.group_by)
        resolved = weights if weights is not None else {name: 1.0 for name in counts}
        return deficit_report(examples, group_by=self.group_by, weights=resolved)

    def merge_with_history(
        self,
        new_batch: Sequence[TrainingExample],
        history: Sequence[TrainingExample],
        *,
        version: int = 1,
        parent_fingerprint: str = "",
    ) -> tuple[list[TrainingExample], DedupeReport]:
        """跨批次增量合并：历史视为已接受，新批次与之做两级去重.

        返回 ``(history + 新批次中存活者, 去重报告)``。**历史在前**，
        因此"保留首次出现"的语义在跨批次场景下自然成立——同一个问题
        已经在历史里，就保留历史里那一条。

        这个方法刻意**不做**清洗、打分、配比与增强：它是"更新"，不是
        "重建"。要重建请把 ``history + new_batch`` 一起交给 ``run``
        （代价是全量重跑）。两条路都保留，是因为它们回答不同的问题：
        增量回答"新来的这批里有多少是真正新的"，重建回答"按今天的
        标准，整份数据集应该长什么样"。
        """
        index = self.new_index()
        index.add_many(list(history))
        kept, report = index.filter(list(new_batch))
        return list(history) + kept, report

    def load_raw_batch(
        self, data_dir: str | Path | None = None
    ) -> tuple[list[TrainingExample], dict[str, int]]:
        """用 day048 的三个数据源载入**原始**样本（不触发清洗）.

        刻意绕过 ``DataCollector.collect()``：那里面的清洗会把本流水线
        第一阶段该做的事提前做掉，导致 ``clean`` 阶段的账永远是 0 丢弃，
        报告里也看不到 ``unverified_source`` 这类真实归因。
        采集是采集，清洗是清洗——合并两件事，账就没法对。
        """
        collector = default_collector(data_dir)
        examples: list[TrainingExample] = []
        counts: dict[str, int] = {}
        for source in collector.sources:
            items = source.load()
            counts[source.name] = len(items)
            examples.extend(items)
        return examples, counts

    def run_from_sources(
        self, *, data_dir: str | Path | None = None, version: int = 1, parent_fingerprint: str = ""
    ) -> DomainDataRun:
        """便捷入口：从仓库内置的三个数据源载入原始样本并跑完整流水线."""
        examples, counts = self.load_raw_batch(data_dir)
        return self.run(
            examples, version=version, parent_fingerprint=parent_fingerprint, raw_counts=counts
        )

    def save(
        self,
        run: DomainDataRun,
        out_dir: str | Path,
        *,
        fmt: str = DEFAULT_FORMAT,
        eval_ratio: float | None = None,
        seed: int | None = None,
        write_split: bool = True,
    ) -> dict[str, Path]:
        """把运行产物落盘：切分后的 ``train.jsonl`` / ``eval.jsonl`` + ``manifest.json``.

        切分复用 day048 的 ``dump_bundle``（同一套可复现切分逻辑），
        本方法只多做一件事——把清单写出来。**清单必须随数据集一起落盘**：
        一份没有清单的 jsonl，三个月后没人说得清它是哪一版、按什么参数生成的。
        """
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        if write_split:
            paths = dict(
                dump_bundle(
                    DatasetBundle(examples=list(run.examples)),
                    target,
                    fmt=fmt,
                    eval_ratio=(
                        eval_ratio if eval_ratio is not None else settings.finetune_eval_ratio
                    ),
                    seed=seed if seed is not None else settings.finetune_split_seed,
                )
            )
        manifest_path = target / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(run.manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        paths["manifest"] = manifest_path
        return paths


def default_pipeline(data_dir: str | Path | None = None) -> DomainDataPipeline:
    """按配置装配默认流水线（阈值、权重、算子、配比全部来自 ``settings``）.

    ``data_dir`` 参数只为保持与 ``default_collector`` 一致的调用形状，
    流水线本身不持有数据目录（它只接收样本列表）——**把"数据在哪"与
    "怎么处理"分开**，测试因此可以用 ``tmp_path`` 或纯内存样本驱动整条链路。
    """
    del data_dir
    return DomainDataPipeline(
        quality_threshold=settings.domain_quality_threshold,
        near_dup_threshold=settings.domain_near_dup_threshold,
        shingle_k=settings.domain_shingle_k,
        num_perm=settings.domain_num_perm,
        group_by=settings.domain_group_by,
        max_group_ratio=settings.domain_max_group_ratio,
        augment=settings.domain_augment_enabled,
        max_augment_per_example=settings.domain_max_augment_per_example,
    )
