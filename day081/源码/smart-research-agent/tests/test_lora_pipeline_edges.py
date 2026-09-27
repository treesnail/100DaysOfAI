"""day052 流水线的边界与"门禁能不能真的拦住"（M5-D4）.

本文件补的是四类**平时走不到、但必须走得到**的路径：

1. ``LoRATrainingReport.adapter_saving_factor`` 在"适配器体积为 0"时返回 0
   而不是抛除零错误（报告要能在异常情况下被打印出来）；
2. ``LoRATrainer`` 显式传 ``template`` / ``system_prompt`` 的转发路径；
3. ``LoRATrainer.fit(save_at_end=False)`` —— 只训练、不落盘的路径；
4. **合并门禁的失败路径**：``verify_merge`` 在"合并结果与适配器模型不一致"时
   必须判定不通过。这一条尤其重要——**一个永远不会失败的门禁不是门禁**，
   而本课的实现里合并恰好是逐位一致的，所以这条路径只能靠构造偏差来触发。
"""

from __future__ import annotations

from smart_research_agent.finetune.schema import TrainingExample
from smart_research_agent.peft.deploy import verify_merge
from smart_research_agent.peft.models import LoRAReferenceModel, default_reference_lora_config
from smart_research_agent.peft.trainer import LoRATrainer, LoRATrainingReport
from smart_research_agent.sft import (
    IGNORE_INDEX,
    PLAIN,
    REFERENCE_LEARNING_RATE,
    Batch,
    CharTokenizer,
    ReferenceSFTModel,
    SFTTrainingArgs,
)

VOCAB = 557


def build_tokenizer() -> CharTokenizer:
    """构造一个足够大的字符级词表（让适配器体积与全参体积可比）."""
    return CharTokenizer.from_texts(["".join(chr(0x4E00 + i) for i in range(VOCAB))])


def make_examples(count: int) -> list[TrainingExample]:
    """造几条短样本（模板用 ``plain``，system 置 None，便于断言）."""
    return [
        TrainingExample(
            instruction=f"问题 {index}：什么是 LoRA？",
            output=f"回答 {index}：低秩适配。",
        )
        for index in range(count)
    ]


def make_batch(*, rows: int = 2) -> Batch:
    """一个带重复结构的小批次（可学、秒级完成）."""
    pattern = (3, 5, 7, 3, 5)
    return Batch(
        input_ids=tuple(pattern for _ in range(rows)),
        labels=tuple((IGNORE_INDEX, *pattern[1:]) for _ in range(rows)),
        attention_mask=tuple(tuple([1] * len(pattern)) for _ in range(rows)),
        pad_token_id=0,
    )


def tiny_args(tmp_path) -> SFTTrainingArgs:
    """最小可跑的 SFT 超参（1 个 epoch、不做累积）."""
    return SFTTrainingArgs(
        output_dir=str(tmp_path / "out"),
        learning_rate=REFERENCE_LEARNING_RATE,
        num_train_epochs=1.0,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        max_length=64,
        save_strategy="no",
        eval_strategy="no",
    )


class TestReportEdgeCases:
    """报告本身要能在"数据不完整"时被打印出来，而不是抛异常."""

    def test_saving_factor_is_zero_without_adapter_bytes(self):
        """适配器体积为 0（例如没落盘）时节省倍数是 0，不是除零错误."""
        report = LoRATrainingReport(
            lora={}, reference={}, sft={}, adapter_bytes_final=0, full_finetune_bytes=1024
        )
        assert report.adapter_saving_factor == 0.0
        assert report.summary_line()  # 摘要仍然可打印

    def test_saving_factor_uses_full_finetune_bytes(self):
        report = LoRATrainingReport(
            lora={}, reference={}, sft={}, adapter_bytes_final=256, full_finetune_bytes=1024
        )
        assert report.adapter_saving_factor == 4.0


class TestTrainerEdges:
    """``LoRATrainer`` 的两条转发路径."""

    def _trainer(self, tmp_path, **kwargs) -> LoRATrainer:
        tokenizer = build_tokenizer()
        model = LoRAReferenceModel(
            ReferenceSFTModel(tokenizer.vocab_size, seed=42),
            default_reference_lora_config(r=4, lora_alpha=8),
            seed=42,
        )
        return LoRATrainer(model, tokenizer, tiny_args(tmp_path), **kwargs)

    def test_explicit_template_and_system_prompt_are_forwarded(self, tmp_path):
        """显式传 ``template`` / ``system_prompt`` 时缺省被覆盖（两条赋值都走到）."""
        trainer = self._trainer(
            tmp_path, template=PLAIN, system_prompt=None, adapter_save_steps=0
        )
        assert trainer._sft_defaults["template"] == PLAIN
        assert trainer._sft_defaults["system_prompt"] is None

    def test_fit_without_final_save_keeps_no_checkpoint(self, tmp_path):
        """``save_at_end=False``：训练照常、但不落盘（返回空清单）."""
        trainer = self._trainer(tmp_path, template=PLAIN, system_prompt=None)
        examples = make_examples(4)
        report = trainer.fit(examples, examples[:2], save_at_end=False)
        assert report.adapter_checkpoints == []
        assert report.adapter_bytes_final == 0
        assert report.adapter_saving_factor == 0.0
        assert report.sft["optimizer_steps"] > 0

    def test_fit_with_final_save_records_one_checkpoint(self, tmp_path):
        trainer = self._trainer(tmp_path, template=PLAIN, system_prompt=None)
        examples = make_examples(4)
        report = trainer.fit(examples, examples[:2])
        assert [item.boundary for item in report.adapter_checkpoints] == ["adapter-final"]
        assert report.adapter_bytes_final > 0
        assert report.adapter_saving_factor > 1.0


class TestMergeGate:
    """合并门禁必须能判定"不通过"——否则它就不是门禁."""

    def test_gate_passes_on_a_real_merge(self):
        tokenizer = build_tokenizer()
        model = LoRAReferenceModel(
            ReferenceSFTModel(tokenizer.vocab_size, seed=42),
            default_reference_lora_config(r=4, lora_alpha=8),
            seed=42,
        )
        model.accumulate(make_batch())
        model.apply_update(REFERENCE_LEARNING_RATE)
        verification = verify_merge(model, [make_batch()])
        assert verification.bitwise_identical is True
        assert verification.max_logit_difference == 0.0
        assert verification.passed is True

    def test_gate_fails_when_merge_drifts(self, monkeypatch):
        """把 ``merge()`` 换成一个"差一点点"的结果：门禁必须判定不通过.

        这条路径在正常实现里永远走不到（合并是逐位一致的），所以只能构造偏差。
        它守的是一条纪律：**部署门禁必须有一条"失败时会怎样"的用例**——
        否则将来有人把 ``passed`` 写成恒真，所有测试仍然全绿。
        """
        tokenizer = build_tokenizer()
        model = LoRAReferenceModel(
            ReferenceSFTModel(tokenizer.vocab_size, seed=42),
            default_reference_lora_config(r=4, lora_alpha=8),
            seed=42,
        )
        model.accumulate(make_batch())
        model.apply_update(REFERENCE_LEARNING_RATE)
        honest_merge = model.merge

        def drifting_merge():
            """真实合并 + 在 logits 上制造 0.5 的偏差（模拟行/列约定用错）."""
            merged = honest_merge()

            class _Drifted:
                vocab_size = getattr(merged, "vocab_size", VOCAB)

                def logits(self, context_token: int) -> list[float]:
                    return [value + 0.5 for value in merged.logits(context_token)]

                def evaluate(self, batches):
                    return merged.evaluate(batches)

            return _Drifted()

        monkeypatch.setattr(model, "merge", drifting_merge)
        verification = verify_merge(model, [make_batch()])
        assert verification.bitwise_identical is False
        assert verification.max_logit_difference == 0.5
        assert verification.passed is False
        # tolerance 放宽到 0.5 时同一个结果应当被判为通过——容差是显式的
        relaxed = verify_merge(model, [make_batch()], tolerance=0.5)
        assert relaxed.passed is True
