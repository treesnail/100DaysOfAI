"""SFT 训练器与检查点测试（day050）：训练循环、梯度累积、评估、落盘.

两个测试在这里承担"证明结论"的角色：

1. :meth:`TestTrainingActuallyLearns.test_loss_decreases` —— **训练真的在学**：
   loss 从 ``ln(V)`` 附近（≈ 均匀分布）下降到明显更低；
2. :meth:`TestPromptMaskingAblation.test_masked_prompt_beats_unmasked` ——
   **屏蔽 prompt 确实更好**：两种训练法都用"同一把尺子"（评估集一律按
   屏蔽口径编码）测量答案 token 上的 loss。这条是本课核心机制的唯一
   实证，也是把它写成测试的理由。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from smart_research_agent.finetune.schema import TrainingExample
from smart_research_agent.sft.args import SFTConfigError, SFTTrainingArgs
from smart_research_agent.sft.checkpoint import (
    CHECKPOINT_FILES,
    CheckpointError,
    checkpoint_summary,
    load_checkpoint,
    save_checkpoint,
)
from smart_research_agent.sft.encoding import (
    IGNORE_INDEX,
    Batch,
    SFTDataError,
    iter_batches,
)
from smart_research_agent.sft.reference_model import (
    REFERENCE_LEARNING_RATE,
    ModelState,
    ReferenceSFTModel,
    top_k_next,
)
from smart_research_agent.sft.template import (
    CHATML,
    PLAIN,
    SFTTemplateError,
    render_supervised,
)
from smart_research_agent.sft.trainer import (
    EncodingReport,
    EvalRecord,
    SFTReport,
    SFTTrainer,
    StepRecord,
    summarize_losses,
    train_reference,
)

TEMPLATE = PLAIN
#: 参考模型的学习率（标定值）；必须显式传，因为 dataclass 缺省是 7B 量级的 2e-4
REF_LR = REFERENCE_LEARNING_RATE


def make_examples(count: int = 8) -> list[TrainingExample]:
    """合成一批小而规整的样本（带重复结构，便于观察 loss 下降）."""
    return [
        TrainingExample(
            instruction=f"请解释第 {index} 个概念",
            output=f"第 {index} 个概念：它把复杂问题拆成可以逐步核对的步骤，因此更容易验证。",
        )
        for index in range(count)
    ]


def ref_args(tmp_path: Path, **overrides) -> SFTTrainingArgs:
    """一套跑得快、且能让 loss 明显下降的参考配置."""
    base = SFTTrainingArgs(
        output_dir=str(tmp_path / "out"),
        learning_rate=REF_LR,
        num_train_epochs=6.0,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=4,
        max_length=96,
        save_strategy="no",
        eval_strategy="no",
    )
    return base.with_overrides(**overrides) if overrides else base


@pytest.fixture
def data():
    train = make_examples(8)
    eval_set = make_examples(8)[:3]
    from smart_research_agent.sft.encoding import CharTokenizer

    texts = [
        render_supervised(example, template=TEMPLATE, system_prompt=None).text
        for example in list(train) + list(eval_set)
    ]
    tokenizer = CharTokenizer.from_texts(texts)
    return train, eval_set, tokenizer


def build_trainer(tmp_path, data, **overrides) -> tuple[SFTTrainer, ReferenceSFTModel]:
    _, _, tokenizer = data
    args = ref_args(tmp_path, **overrides)
    model = ReferenceSFTModel(tokenizer.vocab_size, seed=42)
    return (
        SFTTrainer(model, tokenizer, args, template=TEMPLATE, system_prompt=None),
        model,
    )


class TestEncodeDataset:
    """编码统计：训练开始前必须先核对的第一份数据."""

    def test_encode_report_numbers(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        encoded, report = trainer.encode(data[0])
        assert report.total == 8
        assert report.encoded == 8
        assert report.truncated == 0
        assert report.max_length == 96
        assert report.template == PLAIN
        assert report.total_tokens > report.supervised_tokens > 0
        assert report.ignored_tokens == report.total_tokens - report.supervised_tokens
        assert len(encoded) == 8

    def test_supervised_ratio_between_zero_and_one(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        _, report = trainer.encode(data[0])
        assert 0.0 < report.supervised_ratio < 1.0

    def test_summary_line_mentions_supervision(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        _, report = trainer.encode(data[0])
        assert "监督" in report.summary_line()

    def test_encoding_does_not_mutate_examples(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        before = [(example.instruction, example.output) for example in data[0]]
        trainer.encode(data[0])
        after = [(example.instruction, example.output) for example in data[0]]
        assert before == after

    def test_encoding_is_reproducible(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        first, _ = trainer.encode(data[0])
        second, _ = trainer.encode(data[0])
        assert [sample.input_ids for sample in first] == [sample.input_ids for sample in second]

    def test_length_distribution_and_suggestion(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        summary = trainer.lengths(data[0])
        assert summary.count == 8
        assert summary.maximum >= summary.minimum
        suggested = trainer.suggest_max_length(data[0])
        assert suggested >= summary.p95
        assert suggested % 32 == 0


class TestTrainingActuallyLearns:
    """参考模型必须**真的在学**——否则训练循环只是一段长时间不动的打印."""

    def test_loss_decreases(self, data, tmp_path):
        trainer, model = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=False)
        assert report.initial_loss is not None and report.final_loss is not None
        assert report.final_loss < report.initial_loss
        assert report.loss_drop_ratio is not None and report.loss_drop_ratio > 0
        # 起点必须接近均匀分布（ln V），否则说明初始化或 mask 有问题
        assert report.initial_loss == pytest.approx(model.uniform_loss, abs=0.05)

    def test_step_count_matches_plan(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=False)
        # micro = 8 // 2 = 4，steps/epoch = 4 // 2 = 2，total = 2 * 6 = 12
        assert report.total_steps == 12
        assert report.optimizer_steps == 12
        assert len(report.records) == 12
        assert report.plan is not None and report.plan.total_steps == 12

    def test_supervised_tokens_seen_sums_records(self, data, tmp_path):
        """报告里的 token 账必须等于**实际用于更新的** token 之和."""
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=False)
        assert report.supervised_tokens_seen == sum(
            record.supervised_tokens for record in report.records
        )
        assert report.supervised_tokens_seen > 0

    def test_micro_batches_per_step(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=False)
        assert {record.micro_batches for record in report.records} == {2}

    def test_lr_decays_across_steps(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=False)
        rates = [record.learning_rate for record in report.records]
        assert rates[0] > rates[-1]

    def test_epoch_number_increases(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=False)
        epochs = [record.epoch for record in report.records]
        assert epochs == sorted(epochs)
        assert report.epochs_run == max(epochs)

    def test_probability_concentrates_after_training(self, data, tmp_path):
        """训练后最高概率明显高于均匀分布——"学到东西"的可视化证据."""
        trainer, model = build_trainer(tmp_path, data)
        trainer.fit(data[0], data[1], save_at_end=False)
        top = top_k_next(model, context_token=5, k=1)
        assert top[0][1] * model.vocab_size > 1.5


class TestDeterminism:
    """可复现性是硬要求：同 seed 两次训练必须得到逐位相同的 loss 序列."""

    def test_same_seed_same_loss_sequence(self, data, tmp_path):
        trainer_a, _ = build_trainer(tmp_path / "a", data)
        trainer_b, _ = build_trainer(tmp_path / "b", data)
        report_a = trainer_a.fit(data[0], data[1], save_at_end=False)
        report_b = trainer_b.fit(data[0], data[1], save_at_end=False)
        assert [r.loss for r in report_a.records] == [r.loss for r in report_b.records]

    def test_different_data_seed_changes_order_but_not_step_count(self, data, tmp_path):
        trainer_a, _ = build_trainer(tmp_path / "a", data, data_seed=1)
        trainer_b, _ = build_trainer(tmp_path / "b", data, data_seed=2)
        report_a = trainer_a.fit(data[0], data[1], save_at_end=False)
        report_b = trainer_b.fit(data[0], data[1], save_at_end=False)
        assert report_a.optimizer_steps == report_b.optimizer_steps
        assert [r.loss for r in report_a.records] != [r.loss for r in report_b.records]

    def test_shuffle_disabled_keeps_batch_order(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data, shuffle=False)
        report = trainer.fit(data[0], data[1], save_at_end=False)
        assert report.optimizer_steps == 12


class TestPromptMaskingAblation:
    """核心机制的实证：屏蔽 prompt 的答案 loss 更低."""

    def test_masked_prompt_beats_unmasked(self, data, tmp_path):
        train, eval_set, tokenizer = data
        # 统一的尺子：评估集一律按**屏蔽**口径编码，衡量答案 token 的预测能力
        args = ref_args(tmp_path)
        ruler_trainer = SFTTrainer(
            ReferenceSFTModel(tokenizer.vocab_size, seed=42),
            tokenizer,
            args,
            template=TEMPLATE,
            system_prompt=None,
        )
        ruler_batches = iter_batches(
            ruler_trainer.encode(eval_set)[0], batch_size=args.per_device_eval_batch_size
        )

        results = {}
        for masked in (True, False):
            model = ReferenceSFTModel(tokenizer.vocab_size, seed=42)
            trainer = SFTTrainer(
                model,
                tokenizer,
                args,
                template=TEMPLATE,
                system_prompt=None,
                mask_prompt=masked,
            )
            trainer.fit(train, eval_set, save_at_end=False)
            loss, tokens = model.evaluate(ruler_batches)
            results[masked] = (loss, tokens)

        assert results[True][1] == results[False][1]  # 同一把尺子，token 数相同
        assert results[True][0] < results[False][0]

    def test_unmasked_reports_full_supervision(self, data, tmp_path):
        train, _, tokenizer = data
        args = ref_args(tmp_path)
        trainer = SFTTrainer(
            ReferenceSFTModel(tokenizer.vocab_size, seed=42),
            tokenizer,
            args,
            template=TEMPLATE,
            system_prompt=None,
            mask_prompt=False,
        )
        _, report = trainer.encode(train)
        assert report.supervised_ratio == 1.0
        assert report.ignored_tokens == 0


class TestEvalAndCheckpoints:
    """评估间隔与检查点落盘."""

    def test_final_eval_always_runs(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=False)
        assert report.eval_loss is not None
        assert report.eval_perplexity is not None
        assert report.eval_perplexity == pytest.approx(math.exp(report.eval_loss))
        assert len(report.evals) >= 1
        assert report.evals[-1].step == report.optimizer_steps

    def test_eval_at_steps(self, data, tmp_path):
        trainer, _ = build_trainer(
            tmp_path, data, eval_strategy="steps", eval_steps=4
        )
        report = trainer.fit(data[0], data[1], save_at_end=False)
        steps = [record.step for record in report.evals]
        assert 4 in steps and 8 in steps
        assert steps[-1] == 12  # 末次评估始终执行

    def test_no_eval_set_skips_eval(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], save_at_end=False)
        assert report.eval_loss is None
        assert report.eval is None
        assert report.evals == []

    def test_save_at_end_creates_final(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=True)
        assert report.checkpoints[-1].endswith("final")
        assert (Path(report.checkpoints[-1]) / "metrics.json").exists()

    def test_no_save_at_end_creates_nothing(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=False)
        assert report.checkpoints == []
        assert not (tmp_path / "out").exists()

    def test_intermediate_checkpoints_and_pruning(self, data, tmp_path):
        trainer, _ = build_trainer(
            tmp_path, data, save_strategy="steps", save_steps=4, save_total_limit=1
        )
        report = trainer.fit(data[0], data[1], save_at_end=True)
        saved = [Path(path).name for path in report.checkpoints]
        # 步数 4 / 8 / 12 各存一次（12 恰好等于 total_steps，与末次并存）
        assert saved == ["checkpoint-4", "checkpoint-8", "checkpoint-12", "final"]
        remaining = sorted(path.name for path in (tmp_path / "out").iterdir())
        # save_total_limit=1 只保留最新的一个中间检查点，final 永不删除
        assert remaining == ["checkpoint-12", "final"]

    def test_all_checkpoint_files_written(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=True)
        target = Path(report.checkpoints[-1])
        assert sorted(path.name for path in target.iterdir()) == sorted(CHECKPOINT_FILES)

    def test_checkpoint_roundtrip(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=True)
        loaded = load_checkpoint(report.checkpoints[-1])
        assert loaded["args"].max_length == 96
        assert loaded["tokenizer"].vocab_size == data[2].vocab_size
        assert loaded["state"].updates == report.optimizer_steps
        assert len(loaded["records"]) == report.optimizer_steps
        assert loaded["metrics"]["total_steps"] == report.total_steps

    def test_resume_from_checkpoint_state(self, data, tmp_path):
        """检查点里的权重可以载回一个新模型（继续训练的入口）."""
        trainer, model = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=True)
        loaded = load_checkpoint(report.checkpoints[-1])
        restored = ReferenceSFTModel.from_state(loaded["state"])
        assert restored.state_dict() == model.state_dict()

    def test_load_state_rejects_vocab_mismatch(self, data, tmp_path):
        model = ReferenceSFTModel(10)
        with pytest.raises(Exception, match="词表大小不一致"):
            model.load_state(
                ModelState(vocab_size=11, weights=tuple([(0.0,)] * 11), bias=(0.0,) * 11, updates=0)
            )

    def test_checkpoint_summary(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=True)
        line = checkpoint_summary(report.checkpoints[-1])
        assert "检查点" in line and "步数" in line

    def test_checkpoint_summary_missing_metrics(self, tmp_path):
        with pytest.raises(CheckpointError, match="找不到 metrics.json"):
            checkpoint_summary(tmp_path / "nope")


class TestCheckpointErrors:
    """检查点的失败路径必须显式：产物不完整时宁可报错."""

    def test_missing_directory(self, tmp_path):
        with pytest.raises(CheckpointError, match="目录不存在"):
            load_checkpoint(tmp_path / "nope")

    def test_incomplete_checkpoint(self, data, tmp_path):
        target = tmp_path / "partial"
        target.mkdir()
        (target / "training_args.json").write_text("{}", encoding="utf-8")
        with pytest.raises(CheckpointError, match="检查点不完整"):
            load_checkpoint(target)

    def test_corrupt_json(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=True)
        target = Path(report.checkpoints[-1])
        (target / "tokenizer.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(CheckpointError, match="不是合法 JSON"):
            load_checkpoint(target)

    def test_corrupt_train_log_line(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=True)
        target = Path(report.checkpoints[-1])
        (target / "train_log.jsonl").write_text("{oops\n", encoding="utf-8")
        with pytest.raises(CheckpointError, match="第 1 行解析失败"):
            load_checkpoint(target)

    def test_save_creates_directory(self, data, tmp_path):
        _, _, tokenizer = data
        model = ReferenceSFTModel(tokenizer.vocab_size, seed=1)
        report = SFTReport(
            template=TEMPLATE,
            train=EncodingReport(1, 1, 0, 10, 5, 5, 96, TEMPLATE),
            eval=None,
            total_steps=1,
            epochs_run=1,
            initial_loss=1.0,
            final_loss=0.5,
            best_loss=0.5,
            loss_drop_ratio=0.5,
            eval_loss=None,
            eval_perplexity=None,
            supervised_tokens_seen=5,
            optimizer_steps=1,
            elapsed_seconds=0.1,
        )
        paths = save_checkpoint(
            tmp_path / "deep" / "nested",
            args=ref_args(tmp_path),
            model=model,
            tokenizer=tokenizer,
            report=report,
            records=[StepRecord(1, 1, 0.5, REF_LR, 5, 1)],
        )
        assert set(paths) == set(CHECKPOINT_FILES)
        payload = json.loads((tmp_path / "deep" / "nested" / "metrics.json").read_text("utf-8"))
        assert payload["_meta"]["snapshot"] == "day050"


class TestFitErrors:
    """训练前的护栏：每一条错误都要给出可执行的修复方向."""

    def test_empty_train_set(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        with pytest.raises(SFTDataError, match="训练集为空"):
            trainer.fit([], data[1], save_at_end=False)

    def test_no_full_batch_possible(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data, per_device_train_batch_size=100)
        with pytest.raises(SFTDataError, match="切不出任何完整批次"):
            trainer.fit(data[0], data[1], save_at_end=False)

    def test_zero_updates_per_epoch(self, data, tmp_path):
        """drop_last 下每个 epoch 0 次更新——必须显式失败而不是"跑完了"."""
        trainer, _ = build_trainer(
            tmp_path, data, per_device_train_batch_size=8, gradient_accumulation_steps=4
        )
        with pytest.raises(SFTConfigError, match="0 次参数更新"):
            trainer.fit(data[0], data[1], save_at_end=False)

    def test_validate_runs_first(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data, learning_rate=0.0)
        with pytest.raises(SFTConfigError, match="learning_rate 必须为正数"):
            trainer.fit(data[0], data[1], save_at_end=False)

    def test_answer_does_not_fit_max_length(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data, max_length=20)
        with pytest.raises(SFTDataError, match="已超过"):
            trainer.fit(data[0], data[1], save_at_end=False)

    def test_unknown_template(self, data, tmp_path):
        _, _, tokenizer = data
        args = ref_args(tmp_path)
        trainer = SFTTrainer(
            ReferenceSFTModel(tokenizer.vocab_size),
            tokenizer,
            args,
            template="nope",
            system_prompt=None,
        )
        with pytest.raises(SFTTemplateError, match="不支持的模板"):
            trainer.fit(data[0], save_at_end=False)

    def test_empty_output_example_rejected(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        bad = [*data[0], TrainingExample(instruction="问题", output="   ")]
        with pytest.raises(SFTTemplateError, match="output 为空"):
            trainer.fit(bad, save_at_end=False)


class TestGradientAccumulation:
    """梯度累积的会计：累积窗口内各 micro-batch 的监督 token 数被正确汇总."""

    def test_window_tokens_equal_sum_of_micro_batches(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=False)
        encoded, _ = trainer.encode(data[0])
        batches = iter_batches(encoded, batch_size=2, drop_last=True)
        expected_first = sum(batch.supervised_tokens for batch in batches[:2])
        assert report.records[0].supervised_tokens == expected_first

    def test_apply_update_requires_pending_gradients(self, data, tmp_path):
        _, _, tokenizer = data
        model = ReferenceSFTModel(tokenizer.vocab_size)
        with pytest.raises(Exception, match="没有待应用的梯度"):
            model.apply_update(1.0)

    def test_apply_update_rejects_non_positive_lr(self, data, tmp_path):
        _, _, tokenizer = data
        model = ReferenceSFTModel(tokenizer.vocab_size)
        trainer = SFTTrainer(
            model, tokenizer, ref_args(tmp_path), template=TEMPLATE, system_prompt=None
        )
        batches = iter_batches(trainer.encode(data[0])[0], batch_size=2)
        model.accumulate(batches[0])
        with pytest.raises(Exception, match="learning_rate 必须为正数"):
            model.apply_update(0.0)

    def test_zero_lr_step_clears_gradients(self, data, tmp_path):
        """warmup 比例为 1 时第 0 步学习率为 0：更新无效果，但梯度必须清零."""
        trainer, model = build_trainer(
            tmp_path, data, warmup_ratio=0.5, num_train_epochs=4.0
        )
        report = trainer.fit(data[0], data[1], save_at_end=False)
        assert report.records[0].learning_rate == 0.0
        assert model.updates == report.optimizer_steps - 1  # 第 0 步没有真的更新

    def test_evaluate_does_not_pollute_gradients(self, data, tmp_path):
        """评估必须不污染梯度缓冲——否则"训练-评估-继续训练"会串味."""
        _, _, tokenizer = data
        model = ReferenceSFTModel(tokenizer.vocab_size, seed=3)
        trainer = SFTTrainer(
            model, tokenizer, ref_args(tmp_path), template=TEMPLATE, system_prompt=None
        )
        batches = iter_batches(trainer.encode(data[0])[0], batch_size=2)
        model.accumulate(batches[0])
        before = [row[:] for row in model.state_dict().weights]
        model.evaluate(batches)
        assert [row[:] for row in model.state_dict().weights] == before
        model.apply_update(1.0)  # 缓冲仍在，可以正常更新


class TestTrainReference:
    """端到端便捷入口：建词表 → 建模型 → 训练."""

    def test_default_args_are_reference_scaled(self):
        train = make_examples(4)
        report, model, tokenizer = train_reference(train, train[:2])
        # 缺省 lr 必须是参考模型量级（8.0），而不是 7B 的 2e-4
        assert report.records[0].learning_rate > 1.0
        assert tokenizer.vocab_size == model.vocab_size

    def test_default_args_survive_tiny_dataset(self):
        """缺省配置必须对最小输入也成立：4 条样本也要切得出更新窗口."""
        train = make_examples(2)
        report, _, _ = train_reference(train, train[:1])
        assert report.optimizer_steps >= 1

    def test_vocab_includes_eval_set_chars(self):
        train = [
            TrainingExample(instruction="问题A", output="答案甲甲甲甲甲甲甲甲"),
            TrainingExample(instruction="问题B", output="答案甲甲甲甲甲甲甲甲"),
        ]
        eval_set = [TrainingExample(instruction="问题B", output="答案乙乙乙乙乙乙乙乙")]
        _, _, tokenizer = train_reference(train, eval_set)
        rendered_eval = render_supervised(eval_set[0], template=CHATML).text
        # 评估集里出现的每个字符都必须在词表内（否则会被 <unk> 吞掉）
        assert all(char in tokenizer.tokens for char in rendered_eval)

    def test_explicit_tokenizer_is_reused(self):
        from smart_research_agent.sft.encoding import CharTokenizer

        train = make_examples(4)
        tokenizer = CharTokenizer.from_texts(
            [render_supervised(ex, template=PLAIN, system_prompt=None).text for ex in train]
        )
        _, model, returned = train_reference(train, train[:2], tokenizer=tokenizer)
        assert returned is tokenizer
        assert model.vocab_size == tokenizer.vocab_size

    def test_report_to_dict_is_json_ready(self):
        train = make_examples(4)
        report, _, _ = train_reference(train, train[:2])
        payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
        assert payload["optimizer_steps"] == report.optimizer_steps
        assert payload["train"]["supervised_tokens"] > 0


class TestReportSerialization:
    """报告与记录的序列化：``metrics.json`` / ``train_log.jsonl`` 的内容."""

    def test_step_record_to_dict(self):
        assert StepRecord(3, 2, 1.5, 0.1, 42, 4).to_dict() == {
            "step": 3,
            "epoch": 2,
            "loss": 1.5,
            "learning_rate": 0.1,
            "supervised_tokens": 42,
            "micro_batches": 4,
        }

    def test_eval_record_to_dict(self):
        assert EvalRecord(5, 2.0, math.exp(2.0), 10).to_dict()["step"] == 5

    def test_encoding_report_to_dict(self):
        payload = EncodingReport(2, 2, 1, 20, 8, 12, 96, CHATML).to_dict()
        assert payload["avg_tokens"] == 10.0
        assert payload["supervised_ratio"] == 0.4

    def test_encoding_report_empty(self):
        report = EncodingReport(0, 0, 0, 0, 0, 0, 96, CHATML)
        assert report.avg_tokens == 0.0
        assert report.supervised_ratio == 0.0

    def test_training_plan_is_embedded_in_report(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=False)
        payload = report.to_dict()
        assert payload["plan"]["total_steps"] == 12
        assert payload["records_written"] == 12


class TestSummarizeLosses:
    """``summarize_losses``：把"降了多少"变成可跨数据集比较的相对值."""

    def test_empty_records(self):
        assert summarize_losses([]) == (None, None, None)

    def test_relative_drop(self):
        records = [StepRecord(1, 1, 2.0, 0.1, 10, 1), StepRecord(2, 1, 1.0, 0.1, 10, 1)]
        assert summarize_losses(records) == (2.0, 1.0, 0.5)

    def test_loss_increased(self):
        records = [StepRecord(1, 1, 1.0, 0.1, 10, 1), StepRecord(2, 1, 1.5, 0.1, 10, 1)]
        assert summarize_losses(records) == (1.0, 1.5, -0.5)

    def test_zero_initial_loss_gives_no_ratio(self):
        """初始 loss 为 0 时相对降幅无定义，返回 None 而不是除零崩溃."""
        records = [StepRecord(1, 1, 0.0, 0.1, 10, 1), StepRecord(2, 1, 0.0, 0.1, 10, 1)]
        assert summarize_losses(records) == (0.0, 0.0, None)


class TestReferenceModelEdgeCases:
    """参考模型的边界与便捷入口：这些路径平时训练跑不到，但必须是对的."""

    def test_vocab_size_floor(self):
        with pytest.raises(ValueError, match="vocab_size 至少为 2"):
            ReferenceSFTModel(1)

    def test_init_scale_must_be_positive(self):
        with pytest.raises(ValueError, match="init_scale 必须为正数"):
            ReferenceSFTModel(8, init_scale=0.0)

    def test_uniform_loss_matches_ln_vocab(self):
        model = ReferenceSFTModel(16)
        assert model.uniform_loss == pytest.approx(math.log(16))

    def test_logits_out_of_range(self):
        model = ReferenceSFTModel(8)
        with pytest.raises(Exception, match="上下文 token"):
            model.logits(8)
        with pytest.raises(Exception, match="上下文 token"):
            model.logits(-1)

    def test_accumulate_rejects_label_out_of_vocab(self):
        model = ReferenceSFTModel(8)
        batch = Batch(
            input_ids=((1, 2),),
            labels=((IGNORE_INDEX, 99),),
            attention_mask=((1, 1),),
            pad_token_id=0,
        )
        with pytest.raises(Exception, match="标签 99 落在词表范围"):
            model.accumulate(batch)

    def test_accumulate_rejects_batch_without_supervision(self):
        model = ReferenceSFTModel(8)
        batch = Batch(
            input_ids=((1, 2, 3),),
            labels=((IGNORE_INDEX,) * 3,),
            attention_mask=((1, 1, 1),),
            pad_token_id=0,
        )
        with pytest.raises(Exception, match="本批没有任何监督位置"):
            model.accumulate(batch)

    def test_step_combines_accumulate_and_update(self):
        model = ReferenceSFTModel(8, seed=1)
        batch = Batch(
            input_ids=((1, 2, 3),),
            labels=((IGNORE_INDEX, 2, 3),),
            attention_mask=((1, 1, 1),),
            pad_token_id=0,
        )
        loss_sum, count = model.step(batch, learning_rate=1.0)
        assert count == 2
        assert loss_sum / count == pytest.approx(model.uniform_loss, abs=0.2)
        assert model.updates == 1

    def test_evaluate_rejects_empty_batch_list(self):
        """评估集切不出任何批次时，loss 无定义——必须报错而不是零除.

        注意：单批全屏蔽的情况由 ``accumulate`` 先拦下（报"本批没有任何
        监督位置"），这里守的是**批次列表本身为空**这条更外围的路径。
        """
        model = ReferenceSFTModel(8)
        with pytest.raises(Exception, match="评估集没有任何监督位置"):
            model.evaluate([])

    def test_evaluate_leaves_gradients_untouched_even_when_it_raises(self):
        model = ReferenceSFTModel(8)
        with pytest.raises(Exception, match="评估集没有任何监督位置"):
            model.evaluate([])
        assert model.updates == 0

    def test_predict_next_sums_to_one(self):
        model = ReferenceSFTModel(8)
        assert sum(model.predict_next(1)) == pytest.approx(1.0)

    def test_top_k_next_rejects_non_positive_k(self):
        model = ReferenceSFTModel(8)
        with pytest.raises(Exception, match="k 必须为正整数"):
            top_k_next(model, 1, k=0)

    def test_top_k_next_returns_sorted_candidates(self):
        model = ReferenceSFTModel(8)
        top = top_k_next(model, 1, k=3)
        assert len(top) == 3
        assert [probability for _, probability in top] == sorted(
            (probability for _, probability in top), reverse=True
        )

    def test_token_frequencies(self):
        from smart_research_agent.sft.reference_model import token_frequencies

        assert token_frequencies([[1, 2, 1], [2, 3]]) == {1: 2, 2: 2, 3: 1}
        assert token_frequencies([]) == {}


class TestCheckpointEdgeCases:
    """检查点的边界：空白行容错与"不限量"时不做清理."""

    def test_blank_lines_in_train_log_are_skipped(self, data, tmp_path):
        trainer, _ = build_trainer(tmp_path, data)
        report = trainer.fit(data[0], data[1], save_at_end=True)
        target = Path(report.checkpoints[-1]) / "train_log.jsonl"
        content = target.read_text(encoding="utf-8")
        target.write_text(content + "\n  \n", encoding="utf-8")
        loaded = load_checkpoint(report.checkpoints[-1])
        assert len(loaded["records"]) == report.optimizer_steps

    def test_unlimited_checkpoints_are_kept(self, data, tmp_path):
        trainer, _ = build_trainer(
            tmp_path, data, save_strategy="steps", save_steps=6, save_total_limit=None
        )
        report = trainer.fit(data[0], data[1], save_at_end=True)
        remaining = sorted(path.name for path in (tmp_path / "out").iterdir())
        assert remaining == ["checkpoint-12", "checkpoint-6", "final"]
        assert len(report.checkpoints) == 3

    def test_prune_is_noop_without_output_dir(self, data, tmp_path):
        """检查点目录不存在时清理逻辑必须安全返回（它可能被先删掉了）."""
        trainer, _ = build_trainer(tmp_path, data)
        assert not (tmp_path / "out").exists()
        trainer._prune_checkpoints()  # noqa: SLF001 - 直接验证防御式路径
