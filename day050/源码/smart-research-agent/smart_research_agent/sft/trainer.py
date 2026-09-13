"""SFT 训练器：渲染 → 编码 → mask → 批次 → 梯度累积 → 评估 → 落盘（M5-D2）.

本模块把 day048 产出的 ``train.jsonl`` / ``eval.jsonl`` 真正"喂进模型"，
并让 day049 预先算出的每一个数字（``micro-batches = 15`` /
``steps/epoch = 3`` / ``total_steps = 9`` / ``warmup_steps = 0``）在运行时
被逐一验证。如果算错了，第一步就会暴露。

训练循环的六个阶段与 day046 的流水线是同一种设计——**顺序即策略**：

    render  →  encode  →  mask        →  batch  →  accumulate  →  update
    模板渲染    token化    prompt 置 -100   补齐     梯度求和        SGD

其中 ``mask`` 是 SFT 与预训练唯一实质的区别，而它落在 ``labels`` 数组里
（prompt 段 ``-100``）——**一旦 mask 写错，后面的每一步都会忠实地执行
一个错误的目标**，且 loss 依然会下降（只是学的不是你想让它学的东西）。
所以 ``fit`` 的第一步不是训练，而是**先核对编码统计**。

关于步数口径：本训练器采用 ``drop_last``（每个 epoch 丢弃尾部不满一个
累积窗口的 micro-batch），这样"总步数"有纯算术定义，与 day049 的预算
一致。``plan_training`` 会同时给出 HF ``Trainer`` 的 flush 口径供对照，
并在两者不同时告警；``SFTReport.plan`` 里两个数字都在。
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from smart_research_agent.finetune.schema import TrainingExample
from smart_research_agent.sft import checkpoint as checkpoint_module
from smart_research_agent.sft.args import (
    LR_SOFT_RANGE_REFERENCE,
    SFTConfigError,
    SFTTrainingArgs,
    TrainingPlan,
    plan_training,
)
from smart_research_agent.sft.encoding import (
    CharTokenizer,
    EncodedSample,
    LengthSummary,
    SFTDataError,
    encode_supervised,
    iter_batches,
    length_summary,
    suggest_max_length,
)
from smart_research_agent.sft.loss import SFTLossError, perplexity
from smart_research_agent.sft.reference_model import (
    REFERENCE_LEARNING_RATE,
    ReferenceSFTModel,
)
from smart_research_agent.sft.template import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPLATE,
    render_supervised_list,
)
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class StepRecord:
    """一次**参数更新**的完整记录（不是一次前向）.

    ``micro_batches`` 与 ``supervised_tokens`` 一起出现，是为了让"这一步
    的 loss 是几个 micro-batch 上多少个监督 token 的平均"可核对——
    loss 没有分母就没有意义。
    """

    step: int
    epoch: int
    loss: float
    learning_rate: float
    supervised_tokens: int
    micro_batches: int

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典（写 ``train_log.jsonl``）."""
        return asdict(self)


@dataclass(frozen=True)
class EvalRecord:
    """一次评估的结果（按步号记录，用于观察过拟合的起点）."""

    step: int
    loss: float
    perplexity: float
    supervised_tokens: int

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        return asdict(self)


@dataclass(frozen=True)
class EncodingReport:
    """编码统计：**训练开始前必须核对的第一份数据**.

    它回答四个问题：多少条样本、多少条被截断、一共多少 token、
    其中多少是**真正参与 loss 的监督 token**。``ignored_tokens`` 是
    prompt + padding 的 token 数——**它们参与注意力计算但不产生梯度**，
    是"算力花在了不该花的地方"的直接度量。
    """

    total: int
    encoded: int
    truncated: int
    total_tokens: int
    supervised_tokens: int
    ignored_tokens: int
    max_length: int
    template: str

    @property
    def avg_tokens(self) -> float:
        """平均每条样本的 token 数（含 prompt，不含 batch 间 padding）."""
        return self.total_tokens / self.encoded if self.encoded else 0.0

    @property
    def supervised_ratio(self) -> float:
        """监督 token 占比——SFT 里这个比例越高，算力利用率越高."""
        return self.supervised_tokens / self.total_tokens if self.total_tokens else 0.0

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"{self.encoded}/{self.total} 条（截断 {self.truncated}）| "
            f"共 {self.total_tokens} token，其中监督 {self.supervised_tokens} "
            f"({self.supervised_ratio:.1%})、屏蔽 {self.ignored_tokens} | "
            f"平均 {self.avg_tokens:.1f} token/条"
        )

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典（附带两个派生比例）."""
        payload = asdict(self)
        payload["avg_tokens"] = round(self.avg_tokens, 2)
        payload["supervised_ratio"] = round(self.supervised_ratio, 4)
        return payload


@dataclass
class SFTReport:
    """一次 SFT 的完整汇总（``metrics.json`` 的主体）."""

    template: str
    train: EncodingReport
    eval: EncodingReport | None
    total_steps: int
    epochs_run: int
    initial_loss: float | None
    final_loss: float | None
    best_loss: float | None
    loss_drop_ratio: float | None
    eval_loss: float | None
    eval_perplexity: float | None
    supervised_tokens_seen: int
    optimizer_steps: int
    elapsed_seconds: float
    evals: list[EvalRecord] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)
    plan: TrainingPlan | None = None
    records: list[StepRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """投影为 ``metrics.json`` 的内容（逐步明细另存 ``train_log.jsonl``）."""
        return {
            "template": self.template,
            "train": self.train.to_dict(),
            "eval": self.eval.to_dict() if self.eval else None,
            "total_steps": self.total_steps,
            "epochs_run": self.epochs_run,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "loss_drop_ratio": self.loss_drop_ratio,
            "eval_loss": self.eval_loss,
            "eval_perplexity": self.eval_perplexity,
            "supervised_tokens_seen": self.supervised_tokens_seen,
            "optimizer_steps": self.optimizer_steps,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "evals": [record.to_dict() for record in self.evals],
            "checkpoints": list(self.checkpoints),
            "plan": self.plan.to_dict() if self.plan else None,
            "records_written": len(self.records),
        }


class SFTTrainer:
    """参考实现的最小 SFT 训练器（真实梯度下降，离线可跑）.

    用法::

        trainer = SFTTrainer(model, tokenizer, SFTTrainingArgs(output_dir=tmp))
        report = trainer.fit(train_examples, eval_examples)

    它刻意**不吞掉任何错误**：样本放不下 ``max_length``、批次切不出一个
    完整累积窗口、评估集没有监督位置，都会立刻抛出带修复建议的异常。
    训练类脚本最常见的失败模式是"跑完了但什么也没学到"，而所有这类失败
    在开头都有一个被忽略的信号。
    """

    def __init__(
        self,
        model: ReferenceSFTModel,
        tokenizer: CharTokenizer,
        args: SFTTrainingArgs,
        *,
        template: str = DEFAULT_TEMPLATE,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        include_turn_terminator: bool = True,
        mask_prompt: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.template = template
        self.system_prompt = system_prompt
        self.include_turn_terminator = include_turn_terminator
        #: 是否把 prompt 段 label 置 ``-100``。**标准 SFT 必须为 True**；
        #: 置 False 只用于"不屏蔽会怎样"的对照实验（见教程第六章）。
        self.mask_prompt = mask_prompt

    # ------------------------------------------------------------------ 渲染
    def render(self, examples: Sequence[TrainingExample]) -> list:
        """渲染（不编码），供长度统计与 API 预览复用."""
        return render_supervised_list(
            list(examples),
            template=self.template,
            system_prompt=self.system_prompt,
            include_turn_terminator=self.include_turn_terminator,
        )

    def lengths(self, examples: Sequence[TrainingExample]) -> LengthSummary:
        """渲染后的 token 长度分布——**``max_length`` 的唯一依据**.

        刻意先渲染再统计，而不是拿 day048 ``DatasetStats`` 的原始长度凑合：
        两者在本课程数据集上相差近三倍（84.24 token/条 vs 实测 248.7），
        照抄前者会让 ``max_length`` 定得远远不够。
        """
        return length_summary(self.render(examples), self.tokenizer)

    def suggest_max_length(
        self, examples: Sequence[TrainingExample], *, quantile: float = 0.95
    ) -> int:
        """按长度分位数给出 ``max_length`` 建议值（代理到 ``encoding``）."""
        return suggest_max_length(
            self.render(examples), self.tokenizer, quantile=quantile
        )

    # ------------------------------------------------------------------ 编码
    def encode(
        self, examples: Sequence[TrainingExample]
    ) -> tuple[list[EncodedSample], EncodingReport]:
        """把样本渲染 + 编码为监督样本，并给出编码统计.

        **不改动入参样本**，也不共享状态：编码是纯函数式的，同一批样本
        无论编码多少次结果都一样（与 day048 ``split_dataset`` 的"可复现"
        是同一条底线）。
        """
        rendered = self.render(examples)
        encoded: list[EncodedSample] = [
            encode_supervised(
                item,
                self.tokenizer,
                max_length=self.args.max_length,
                truncation=self.args.truncation,
                mask_prompt=self.mask_prompt,
            )
            for item in rendered
        ]
        total_tokens = sum(sample.total_tokens for sample in encoded)
        supervised = sum(sample.supervised_tokens for sample in encoded)
        report = EncodingReport(
            total=len(examples),
            encoded=len(encoded),
            truncated=sum(1 for sample in encoded if sample.truncated),
            total_tokens=total_tokens,
            supervised_tokens=supervised,
            ignored_tokens=total_tokens - supervised,
            max_length=self.args.max_length,
            template=self.template,
        )
        return encoded, report

    # ------------------------------------------------------------------ 训练
    def fit(
        self,
        train_examples: Sequence[TrainingExample],
        eval_examples: Sequence[TrainingExample] = (),
        *,
        save_at_end: bool = True,
    ) -> SFTReport:
        """跑完一次 SFT 训练循环，返回完整报告.

        执行顺序（每一步都先核对、再继续）：

        1. **校验超参**（``args.validate()``）——非法值不留到循环里；
        2. **编码训练集与评估集**，打印编码统计（监督 token 占比）；
        3. **计划**（``plan_training``）——算出步数并把风险写成 warnings；
        4. **切批次**（``drop_last=True``）——切不出一个完整累积窗口直接报错；
        5. **训练循环**——梯度累积 + 学习率调度 + 按间隔评估/保存；
        6. **收尾**——末次评估、末次落盘、汇总报告。
        """
        self.args.validate()
        if not train_examples:
            raise SFTDataError("训练集为空：没有样本就没有训练")

        # 长度分布必须**先于编码**算：编码会被 max_length 截断，截断后的
        # 分布无法回答"max_length 该取多少"这个反问题。
        length_stats = self.lengths(list(train_examples) + list(eval_examples))
        logger.info("渲染后长度分布：%s", length_stats.summary_line())

        train_encoded, train_report = self.encode(train_examples)
        eval_encoded, eval_report = (
            self.encode(eval_examples) if len(eval_examples) > 0 else ([], None)
        )
        logger.info("训练集编码：%s", train_report.summary_line())
        if eval_report is not None:
            logger.info("评估集编码：%s", eval_report.summary_line())

        plan = plan_training(
            self.args,
            train_size=len(train_encoded),
            eval_size=len(eval_encoded),
            dataset_avg_tokens=length_stats.mean,
            dataset_max_tokens=length_stats.maximum,
            lr_range=LR_SOFT_RANGE_REFERENCE,
        )
        for message in plan.warnings:
            logger.warning("SFT 计划告警：%s", message)

        batches = iter_batches(
            train_encoded,
            batch_size=self.args.per_device_train_batch_size,
            drop_last=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if not batches:
            raise SFTDataError(
                f"训练集切不出任何完整批次：{len(train_encoded)} 条样本 / "
                f"per_device_train_batch_size={self.args.per_device_train_batch_size}"
            )
        if plan.steps_per_epoch == 0:
            raise SFTConfigError(
                f"drop_last 口径下每个 epoch 有 0 次参数更新：{len(batches)} 个 micro-batch "
                f"不足一个累积窗口（gradient_accumulation_steps="
                f"{self.args.gradient_accumulation_steps}）。请调小累积步数、调小批大小，"
                "或补充训练数据。"
            )

        eval_batches = (
            iter_batches(
                eval_encoded,
                batch_size=self.args.per_device_eval_batch_size,
                drop_last=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            if eval_encoded
            else []
        )

        total_steps = plan.total_steps
        records: list[StepRecord] = []
        evals: list[EvalRecord] = []
        checkpoints: list[str] = []
        started = time.perf_counter()

        rng = random.Random(self.args.data_seed)
        window_loss = 0.0
        window_tokens = 0
        window_micro = 0
        global_step = 0
        epochs_run = 0

        while global_step < total_steps:
            epochs_run += 1
            order = list(batches)
            if self.args.shuffle:
                # 每轮重新打乱：同一份数据在不同 epoch 看到不同的批组合。
                # 用局部 ``Random(seed)`` 而不是全局 ``random``——测试与
                # 并发调用彼此隔离（与 day048 ``split_dataset`` 同一条纪律）。
                rng.shuffle(order)
            for batch in order:
                loss_sum, count = self.model.accumulate(batch)
                window_loss += loss_sum
                window_tokens += count
                window_micro += 1
                if window_micro < self.args.gradient_accumulation_steps:
                    continue

                learning_rate = self.args.lr_at(global_step, total_steps)
                if learning_rate > 0:
                    self.model.apply_update(learning_rate)
                else:
                    # warmup 第 0 步的学习率是 0：更新无效果，但梯度必须清零，
                    # 否则它会**漏进下一个窗口**，把两次更新混成一次。
                    self.model.zero_grad()
                global_step += 1
                records.append(
                    StepRecord(
                        step=global_step,
                        epoch=epochs_run,
                        loss=window_loss / window_tokens,
                        learning_rate=learning_rate,
                        supervised_tokens=window_tokens,
                        micro_batches=window_micro,
                    )
                )
                window_loss = 0.0
                window_tokens = 0
                window_micro = 0

                if eval_batches and self.args.eval_strategy == "steps" and (
                    global_step % self.args.eval_steps == 0
                ):
                    evals.append(self._evaluate(eval_batches, global_step))
                if self.args.save_strategy == "steps" and (
                    global_step % self.args.save_steps == 0
                ):
                    checkpoints.append(self._save(global_step, records, evals, plan))
                if global_step >= total_steps:
                    break
            # drop_last：epoch 末尾不满一个累积窗口的 micro-batch 直接丢弃，
            # 步数因此有纯算术定义（= micro_batches // accum × epochs）。
            window_loss = 0.0
            window_tokens = 0
            window_micro = 0

        # 末次评估：无论 eval_strategy 是什么，训练结束后都要给一个评估数字
        if eval_batches:
            evals.append(self._evaluate(eval_batches, global_step))

        elapsed = time.perf_counter() - started
        report = self._build_report(
            train_report, eval_report, plan, records, evals, checkpoints, epochs_run, elapsed
        )
        if save_at_end:
            report.checkpoints.append(self._save(global_step, records, evals, plan, final=True))
        return report

    # ------------------------------------------------------------------ 内部
    def _evaluate(self, eval_batches: Sequence[Any], step: int) -> EvalRecord:
        """评估一次（只前向，不更新参数）."""
        loss, count = self.model.evaluate(eval_batches)
        return EvalRecord(
            step=step, loss=loss, perplexity=perplexity(loss), supervised_tokens=count
        )

    def _save(
        self,
        step: int,
        records: Sequence[StepRecord],
        evals: Sequence[EvalRecord],
        plan: TrainingPlan,
        *,
        final: bool = False,
    ) -> str:
        """落盘一个检查点，返回目录路径.

        末次检查点写在 ``output_dir/final``，中间检查点写在
        ``output_dir/checkpoint-{step}``（与 HF 的命名习惯一致），并按
        ``save_total_limit`` 删除最旧的中间检查点——**磁盘不是无限的，
        而"训练跑了两天因为磁盘满而崩"是最令人沮丧的失败方式**。
        """
        name = "final" if final else f"checkpoint-{step}"
        target = Path(self.args.output_dir) / name
        checkpoint_module.save_checkpoint(
            target,
            args=self.args,
            model=self.model,
            tokenizer=self.tokenizer,
            report=_CheckpointView(
                train=self._encoding_placeholder,
                eval=None,
                total_steps=plan.total_steps,
                epochs_run=0,
                initial_loss=records[0].loss if records else None,
                final_loss=records[-1].loss if records else None,
                best_loss=min((r.loss for r in records), default=None),
                loss_drop_ratio=None,
                eval_loss=evals[-1].loss if evals else None,
                eval_perplexity=evals[-1].perplexity if evals else None,
                supervised_tokens_seen=sum(r.supervised_tokens for r in records),
                optimizer_steps=step,
                elapsed_seconds=0.0,
                evals=list(evals),
            ),
            records=records,
        )
        self._prune_checkpoints()
        logger.info("已保存检查点 %s（步数 %d）", target, step)
        return str(target)

    @property
    def _encoding_placeholder(self) -> EncodingReport:
        """中间检查点的占位编码统计（真正的统计在最终 ``metrics.json`` 里）.

        中间检查点要回答的是"权重到哪一步了"，编码统计属于实验级信息，
        只在末次落盘时给全——**避免每存一次就把整个数据集的统计重算一遍**。
        """
        return EncodingReport(
            total=0,
            encoded=0,
            truncated=0,
            total_tokens=0,
            supervised_tokens=0,
            ignored_tokens=0,
            max_length=self.args.max_length,
            template=self.template,
        )

    def _prune_checkpoints(self) -> None:
        """按 ``save_total_limit`` 删除最旧的中间检查点.

        ``"final"`` 永远不删——它是这次实验的交付物。
        """
        limit = self.args.save_total_limit
        output_dir = Path(self.args.output_dir)
        if limit is None or not output_dir.is_dir():
            return
        intermediates = sorted(
            (path for path in output_dir.glob("checkpoint-*") if path.is_dir()),
            key=lambda path: int(path.name.split("-", 1)[1]),
        )
        for stale in intermediates[: max(0, len(intermediates) - limit)]:
            for child in stale.iterdir():
                child.unlink()
            stale.rmdir()

    def _build_report(
        self,
        train_report: EncodingReport,
        eval_report: EncodingReport | None,
        plan: TrainingPlan,
        records: Sequence[StepRecord],
        evals: Sequence[EvalRecord],
        checkpoints: Sequence[str],
        epochs_run: int,
        elapsed: float,
    ) -> SFTReport:
        """汇总报告（含 loss 降幅——比"loss 是多少"更有信息量的数字）.

        ``supervised_tokens_seen`` 取 **``records`` 的求和**，而不是"曾经
        累加进梯度缓冲的总量"：``drop_last`` 会丢弃每个 epoch 尾部未满的
        累积窗口，那部分 token **没有参与任何一次参数更新**。用后者会让
        报告里的数字比实际大（实测差 ~13%），属于典型的"账对不上"。
        """
        initial, final, drop_ratio = summarize_losses(records)
        return SFTReport(
            template=self.template,
            train=train_report,
            eval=eval_report,
            total_steps=plan.total_steps,
            epochs_run=epochs_run,
            initial_loss=initial,
            final_loss=final,
            best_loss=min((record.loss for record in records), default=None),
            loss_drop_ratio=drop_ratio,
            eval_loss=evals[-1].loss if evals else None,
            eval_perplexity=evals[-1].perplexity if evals else None,
            supervised_tokens_seen=sum(record.supervised_tokens for record in records),
            optimizer_steps=len(records),
            elapsed_seconds=elapsed,
            evals=list(evals),
            checkpoints=list(checkpoints),
            plan=plan,
            records=list(records),
        )


@dataclass
class _CheckpointView:
    """给 ``save_checkpoint`` 用的轻量报告视图（只提供它需要的字段）.

    中间检查点的 ``metrics.json`` 不需要完整报告：它只需要"到哪一步了"。
    用一个窄的适配对象而不是把完整 ``SFTReport`` 传下去，避免为了存一次
    中间检查点而重算整份报告。
    """

    train: EncodingReport
    eval: EncodingReport | None
    total_steps: int
    epochs_run: int
    initial_loss: float | None
    final_loss: float | None
    best_loss: float | None
    loss_drop_ratio: float | None
    eval_loss: float | None
    eval_perplexity: float | None
    supervised_tokens_seen: int
    optimizer_steps: int
    elapsed_seconds: float
    evals: list[EvalRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """与 ``SFTReport.to_dict`` 的键保持一致（便于统一读取）."""
        return {
            "template": self.train.template,
            "train": self.train.to_dict(),
            "eval": self.eval.to_dict() if self.eval else None,
            "total_steps": self.total_steps,
            "epochs_run": self.epochs_run,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "loss_drop_ratio": self.loss_drop_ratio,
            "eval_loss": self.eval_loss,
            "eval_perplexity": self.eval_perplexity,
            "supervised_tokens_seen": self.supervised_tokens_seen,
            "optimizer_steps": self.optimizer_steps,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "evals": [record.to_dict() for record in self.evals],
            "checkpoints": [],
            "plan": None,
            "records_written": 0,
        }


def summarize_losses(
    records: Sequence[StepRecord],
) -> tuple[float | None, float | None, float | None]:
    """从逐步记录里汇总 ``(initial, final, drop_ratio)``.

    独立成一个**纯函数**而不是写在 ``_build_report`` 内部，是为了让它
    可以被单独测试——包括两条平时走不到的路径：空记录（返回三个
    ``None``）与 ``initial <= 0``（降幅无定义）。把这类判断埋在私有方法
    里，"要么永远测不到、要么只能靠 pragma 忽略"，两种做法都会让覆盖率
    这个指标失去意义。

    ``drop_ratio`` 用**相对降幅**而不是绝对差值：``6.32 → 5.48`` 与
    ``2.00 → 1.16`` 的绝对差都是 0.84，但前者只降了 13%、后者降了 42%。
    相对值才能跨数据集比较。
    """
    if not records:
        return None, None, None
    initial = records[0].loss
    final = records[-1].loss
    ratio = (initial - final) / initial if initial > 0 else None
    return initial, final, ratio


def train_reference(
    train_examples: Sequence[TrainingExample],
    eval_examples: Sequence[TrainingExample] = (),
    *,
    args: SFTTrainingArgs | None = None,
    template: str = DEFAULT_TEMPLATE,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    tokenizer: CharTokenizer | None = None,
    seed: int = 42,
) -> tuple[SFTReport, ReferenceSFTModel, CharTokenizer]:
    """端到端便捷入口：建词表 → 建模型 → 训练 → 返回 ``(报告, 模型, 分词器)``.

    词表从**训练集与评估集的全部文本**构建（必须包含评估集：否则评估集
    里的字符会全部落到 ``<unk>``，评估就失去了意义——这是真实项目里很
    常见的一个错误）。

    ``args`` 的缺省值刻意与"7B 全参 SFT"不同，两个字段都写了理由：

    - ``learning_rate = REFERENCE_LEARNING_RATE``（8.0）：参考模型只有
      557×557 个参数、用纯 SGD，学习率比 7B 全参 + AdamW 大四个数量级；
      **学习率没有绝对尺度**，照抄别人的值只会得到一条水平的 loss 曲线；
    - ``gradient_accumulation_steps = 1``：不做梯度累积。这样即使只有
      4 条样本也能切出完整的更新窗口——**缺省值必须对最小输入也成立**；
      累积机制由 ``SFTTrainer`` 的循环与专门用例覆盖，不必写进缺省值；
    - ``num_train_epochs = 2.0``：与批大小一起把步数控制在几十步量级，
      让演示既能看到 loss 下降，又不至于等上一分钟。
    """
    effective_args = args or SFTTrainingArgs(
        learning_rate=REFERENCE_LEARNING_RATE,
        gradient_accumulation_steps=1,
        num_train_epochs=2.0,
    )
    rendered = render_supervised_list(
        list(train_examples) + list(eval_examples),
        template=template,
        system_prompt=system_prompt,
    )
    effective_tokenizer = tokenizer or CharTokenizer.from_texts([item.text for item in rendered])
    model = ReferenceSFTModel(effective_tokenizer.vocab_size, seed=seed)
    trainer = SFTTrainer(
        model,
        effective_tokenizer,
        effective_args,
        template=template,
        system_prompt=system_prompt,
    )
    report = trainer.fit(train_examples, eval_examples)
    return report, model, effective_tokenizer


__all__ = [
    "EncodingReport",
    "EvalRecord",
    "SFTLossError",
    "SFTReport",
    "SFTTrainer",
    "StepRecord",
    "summarize_losses",
    "train_reference",
]
