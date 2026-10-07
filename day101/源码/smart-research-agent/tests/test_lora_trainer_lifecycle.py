"""LoRA 训练器与适配器生命周期测试（day052）：落盘、恢复、清理.

本文件里承担"证明结论"角色的四条用例：

1. :meth:`TestSaveAdapter.test_hash_ignores_training_state` —— **内容哈希只覆盖
   配置与权重**。同一个适配器续训一步之后权重没变、但"训到哪了"变了；哈希要
   回答的是"权重是不是同一份"，把步数/时间戳混进去会让这个答案随无关量漂移。
2. :meth:`TestLoRATrainerFit.test_intermediate_and_final_checkpoints` ——
   ``adapter_save_steps=2`` 时**中间适配器与 ``adapter-final`` 都在**，且中间
   步号是 ``save_steps`` 的倍数；循环里落盘靠 ``on_step`` 回调，训练逻辑不复制。
3. :meth:`TestLoRATrainerFit.test_report_serialization_and_saving_factor` ——
   ``adapter_saving_factor > 1``：适配器（几十 KiB）比全参（``V×V`` 权重）小，
   这是"LoRA 能交付"这条结论的落盘版本。
4. :meth:`TestRestore.test_roundtrip_restores_matrices_and_hash` —— 把
   ``adapter-final`` 灌回一个**新的**同配置模型后，两者的
   ``layer.snapshot_matrices()`` 逐位相等，且返回的 ``sha256`` 与保存时相同。

词表说明：适配器的 ``adapter_model.json`` 是**逐元素换行**序列化的，词表只有
几十个字符时适配器反而比全参权重还大（实测 23 KB vs 8 KB），"适配器远小于全参"
这条对照就失去意义。所以本文件建词表时并入一段填充文本，把字符表撑到接近真实
语料的规模（实测 211 个 token）——**这不是为了让断言通过，而是为了让对照成立**。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smart_research_agent.finetune.schema import TrainingExample
from smart_research_agent.peft.config import PEFTConfigError
from smart_research_agent.peft.models import LoRAReferenceModel, default_reference_lora_config
from smart_research_agent.peft.trainer import (
    ADAPTER_FILES,
    FINAL_ADAPTER_NAME,
    INTERMEDIATE_PREFIX,
    AdapterCheckpoint,
    AdapterCheckpointError,
    LoRATrainer,
    LoRATrainingReport,
    adapter_summary,
    load_adapter,
    prune_adapters,
    save_adapter,
)
from smart_research_agent.sft import (
    IGNORE_INDEX,
    PLAIN,
    REFERENCE_LEARNING_RATE,
    Batch,
    CharTokenizer,
    ReferenceSFTModel,
    SFTTrainingArgs,
    render_supervised,
)
from smart_research_agent.sft.trainer import StepRecord

#: 训练用的学习率：与 day050 参考模型同一条标定（557 维 bigram + 纯 SGD）
REFERENCE_LR = REFERENCE_LEARNING_RATE

#: 填充文本：只用来把字符词表撑到真实语料的规模（它不参与任何训练样本）
VOCAB_FILLER = (
    "零零壹贰叁肆伍陆柒捌玖拾佰仟万亿兆京垓秭穰沟涧正载"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳午未申酉戌亥"
    "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏闰余成岁律吕调阳"
    "云腾致雨露结为霜金生丽水玉出昆冈剑号巨阙珠称夜光果珍李柰菜重芥姜"
)

#: 一次小规模训练里的样本条数（4 条 / 批大小 2 / 累积 1 / 2 个 epoch = 4 步）
TRAIN_COUNT = 4

#: 中间适配器的落盘间隔（4 步训练 → 第 2、4 步各落一份中间适配器）
SAVE_STEPS = 2


def make_examples(count: int = TRAIN_COUNT) -> list[TrainingExample]:
    """合成一批小而重复的样本（与 day050 / day051 测试同一套做法）."""
    return [
        TrainingExample(
            instruction=f"请解释第 {index} 个概念",
            output="它把复杂问题拆成可以逐步核对的步骤，因此更容易验证。",
        )
        for index in range(count)
    ]


def make_batch(*, rows: int = 4) -> Batch:
    """构造一个带重复结构的小批次（上下文 3 → 5 → 7 → 3，一份可学的大明文法）."""
    pattern = (3, 5, 7, 3, 5, 7, 3, 5, 7)
    return Batch(
        input_ids=tuple(pattern for _ in range(rows)),
        labels=tuple((IGNORE_INDEX, *pattern[1:]) for _ in range(rows)),
        attention_mask=tuple(tuple([1] * len(pattern)) for _ in range(rows)),
        pad_token_id=0,
    )


def build_tokenizer(examples: list[TrainingExample]) -> CharTokenizer:
    """用样本渲染后的文本建字符词表，并并入填充文本撑大字符表（见模块 docstring）."""
    texts = [
        render_supervised(example, template=PLAIN, system_prompt=None).text
        for example in examples
    ]
    return CharTokenizer.from_texts([*texts, VOCAB_FILLER])


def build_model(base: ReferenceSFTModel, **overrides) -> LoRAReferenceModel:
    """在基座上挂一个默认的参考适配器（``dropout=0`` / ``targets=bigram``）."""
    return LoRAReferenceModel(base, default_reference_lora_config(**overrides), seed=42)


def ref_args(tmp_path: Path, **overrides) -> SFTTrainingArgs:
    """一套跑得快、步数可预期的参考配置（4 步：4 条 / 批 2 / 累积 1 / 2 epoch）."""
    base = SFTTrainingArgs(
        output_dir=str(tmp_path / "out"),
        learning_rate=REFERENCE_LR,
        num_train_epochs=2.0,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=2,
        max_length=128,
        save_strategy="no",
        eval_strategy="no",
    )
    return base.with_overrides(**overrides) if overrides else base


def fit_lora(
    tmp_path: Path, tokenizer: CharTokenizer, *, adapter_save_steps: int
) -> tuple[LoRATrainingReport, LoRATrainer]:
    """跑一次最小规模的 LoRA 训练，返回 ``(报告, 训练器)``（``save_at_end`` 保持缺省 True）."""
    base = ReferenceSFTModel(tokenizer.vocab_size, seed=42)
    trainer = LoRATrainer(
        build_model(base),
        tokenizer,
        ref_args(tmp_path),
        adapter_save_steps=adapter_save_steps,
        template=PLAIN,
        system_prompt=None,
    )
    return trainer.fit(make_examples()), trainer


@pytest.fixture
def tokenizer() -> CharTokenizer:
    """词表被撑大的字符分词器（模型词表大小与它一致，训练才自洽）."""
    return build_tokenizer(make_examples())


@pytest.fixture
def base(tokenizer: CharTokenizer) -> ReferenceSFTModel:
    """一个 ``V = tokenizer.vocab_size`` 的参考基座（每个用例新建一份）."""
    return ReferenceSFTModel(tokenizer.vocab_size, seed=42)


@pytest.fixture
def model(base: ReferenceSFTModel) -> LoRAReferenceModel:
    """挂上默认适配器的参考模型（``ΔW = 0``）."""
    return build_model(base)


@pytest.fixture
def args(tmp_path: Path) -> SFTTrainingArgs:
    """一套指向 ``tmp_path`` 的参考超参（绝不写进仓库目录）."""
    return ref_args(tmp_path)


@pytest.fixture
def batches() -> list[Batch]:
    """两个小批次（4 条与 2 条），供适配器往返用例做一次真实更新."""
    return [make_batch(rows=4), make_batch(rows=2)]


class TestSaveAdapter:
    """``save_adapter``：三个文件、一份记录、一个只覆盖配置与权重的内容哈希."""

    def test_writes_three_files(self, tmp_path: Path, model: LoRAReferenceModel, args):
        target = tmp_path / "adapter"
        checkpoint = save_adapter(target, model=model, args=args, step=1, boundary="final")
        assert sorted(path.name for path in target.iterdir()) == sorted(ADAPTER_FILES)
        assert checkpoint.files == ADAPTER_FILES

    def test_checkpoint_fields_and_short_hash(self, tmp_path: Path, model, args):
        target = tmp_path / "adapter"
        checkpoint = save_adapter(target, model=model, args=args, step=5, boundary="step-5")
        assert isinstance(checkpoint, AdapterCheckpoint)
        assert checkpoint.directory == str(target)
        assert (checkpoint.boundary, checkpoint.step) == ("step-5", 5)
        assert checkpoint.trainable_parameters == model.trainable_parameters
        expected_bytes = sum((target / name).stat().st_size for name in ADAPTER_FILES)
        assert checkpoint.adapter_bytes == expected_bytes
        assert len(checkpoint.content_sha256) == 64
        assert checkpoint.short_hash == checkpoint.content_sha256[:12]
        # 没有 records 时训练状态只能取超参里的学习率，loss 无值
        assert checkpoint.train_loss is None
        assert checkpoint.learning_rate == args.learning_rate
        assert checkpoint.to_dict()["files"] == list(ADAPTER_FILES)
        assert checkpoint.to_dict()["short_hash"] == checkpoint.short_hash

    def test_negative_step_rejected(self, tmp_path: Path, model, args):
        with pytest.raises(PEFTConfigError, match="step 不能为负数"):
            save_adapter(tmp_path / "adapter", model=model, args=args, step=-1, boundary="final")
        assert not (tmp_path / "adapter").exists()

    def test_empty_boundary_rejected(self, tmp_path: Path, model, args):
        with pytest.raises(PEFTConfigError, match="boundary 不能为空"):
            save_adapter(tmp_path / "adapter", model=model, args=args, step=1, boundary="   ")
        assert not (tmp_path / "adapter").exists()

    def test_same_seed_same_weights_gives_same_hash(self, tmp_path: Path, base, args):
        first = save_adapter(
            tmp_path / "a", model=build_model(base), args=args, step=1, boundary="final"
        )
        second = save_adapter(
            tmp_path / "b", model=build_model(base), args=args, step=1, boundary="final"
        )
        assert first.content_sha256 == second.content_sha256

    def test_different_lora_seed_changes_hash(self, tmp_path: Path, base, args):
        """换一个适配器初始化种子必须换一个哈希——否则哈希什么也没覆盖."""
        other = LoRAReferenceModel(base, default_reference_lora_config(), seed=7)
        same = save_adapter(
            tmp_path / "a", model=build_model(base), args=args, step=1, boundary="final"
        )
        different = save_adapter(
            tmp_path / "b", model=other, args=args, step=1, boundary="final"
        )
        assert same.content_sha256 != different.content_sha256

    def test_hash_ignores_training_state(self, tmp_path: Path, model, args):
        """步数与训练记录改变不了哈希：哈希只回答"权重是不是同一份"."""
        record = StepRecord(
            step=3, epoch=1, loss=1.25, learning_rate=0.5, supervised_tokens=42, micro_batches=2
        )
        plain = save_adapter(
            tmp_path / "a", model=model, args=args, step=1, boundary="final", records=()
        )
        recorded = save_adapter(
            tmp_path / "b",
            model=model,
            args=args,
            step=3,
            boundary="step-3",
            records=[record],
        )
        assert (plain.step, recorded.step) == (1, 3)
        assert (plain.train_loss, recorded.train_loss) == (None, 1.25)
        assert plain.content_sha256 == recorded.content_sha256
        assert load_adapter(tmp_path / "b")["state"]["step"] == 3


class TestLoadAdapter:
    """``load_adapter``：往返逐位一致，缺任何一件就报错（不做"部分可用"的降级）."""

    def test_roundtrip_restores_matrices_and_hash(self, tmp_path: Path, model, args, batches):
        for batch in batches:
            model.accumulate(batch)
            model.apply_update(REFERENCE_LR)
        checkpoint = save_adapter(
            tmp_path / "adapter", model=model, args=args, step=2, boundary="final"
        )
        payload = load_adapter(tmp_path / "adapter")
        a_matrix, b_matrix = model.layer.snapshot_matrices()
        assert payload["model"]["a"] == a_matrix
        assert payload["model"]["b"] == b_matrix
        assert payload["sha256"] == checkpoint.content_sha256
        assert payload["directory"] == str(tmp_path / "adapter")
        assert payload["state"]["step"] == 2

    def test_missing_directory_rejected(self, tmp_path: Path):
        with pytest.raises(AdapterCheckpointError, match="适配器目录不存在"):
            load_adapter(tmp_path / "nope")

    def test_incomplete_directory_rejected(self, tmp_path: Path):
        target = tmp_path / "partial"
        target.mkdir()
        (target / "adapter_config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(AdapterCheckpointError, match="适配器不完整"):
            load_adapter(target)

    def test_corrupt_json_rejected(self, tmp_path: Path, model, args):
        target = tmp_path / "broken"
        save_adapter(target, model=model, args=args, step=1, boundary="final")
        (target / "adapter_model.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(AdapterCheckpointError, match="不是合法 JSON"):
            load_adapter(target)


class TestAdapterSummary:
    """``adapter_summary``：巡检不需要加载权重，只读 ``training_state.json``."""

    def test_summary_mentions_state(self, tmp_path: Path, model, args):
        record = StepRecord(
            step=4, epoch=1, loss=0.5, learning_rate=0.25, supervised_tokens=8, micro_batches=1
        )
        save_adapter(
            tmp_path / "adapter",
            model=model,
            args=args,
            step=4,
            boundary="step-4",
            records=[record],
        )
        line = adapter_summary(tmp_path / "adapter")
        assert "适配器" in line
        assert "step=4" in line
        assert str(model.trainable_parameters) in line

    def test_missing_state_file_rejected(self, tmp_path: Path):
        with pytest.raises(AdapterCheckpointError, match="找不到 training_state.json"):
            adapter_summary(tmp_path / "nope")


class TestPruneAdapters:
    """``prune_adapters``：只清最旧的中间适配器，``adapter-final`` 永不删除."""

    @pytest.fixture
    def adapter_dir(self, tmp_path: Path, model, args) -> Path:
        """3 个中间适配器（步 2 / 4 / 6）+ 1 个 ``adapter-final``."""
        root = tmp_path / "adapters"
        for step in (2, 4, 6):
            boundary = f"{INTERMEDIATE_PREFIX}{step}"
            save_adapter(root / boundary, model=model, args=args, step=step, boundary=boundary)
        save_adapter(
            root / FINAL_ADAPTER_NAME,
            model=model,
            args=args,
            step=6,
            boundary=FINAL_ADAPTER_NAME,
        )
        return root

    def test_keeps_newest_and_never_deletes_final(self, adapter_dir: Path):
        removed = prune_adapters(adapter_dir, limit=2)
        assert removed == [f"{INTERMEDIATE_PREFIX}2"]
        remaining = sorted(path.name for path in adapter_dir.iterdir())
        assert remaining == sorted(
            [f"{INTERMEDIATE_PREFIX}4", f"{INTERMEDIATE_PREFIX}6", FINAL_ADAPTER_NAME]
        )
        assert (adapter_dir / FINAL_ADAPTER_NAME).is_dir()

    def test_non_positive_limit_rejected(self, adapter_dir: Path):
        with pytest.raises(PEFTConfigError, match="limit 必须为正整数"):
            prune_adapters(adapter_dir, limit=0)

    def test_missing_directory_returns_empty(self, tmp_path: Path):
        assert prune_adapters(tmp_path / "nope", limit=2) == []


class TestLoRATrainerFit:
    """``LoRATrainer.fit``：按间隔落中间适配器，收尾落 ``adapter-final``."""

    def test_negative_save_steps_rejected(self, tmp_path: Path, tokenizer, base):
        with pytest.raises(PEFTConfigError, match="adapter_save_steps 不能为负数"):
            LoRATrainer(
                build_model(base), tokenizer, ref_args(tmp_path), adapter_save_steps=-1
            )

    def test_default_adapter_output_dir(self, tmp_path: Path, tokenizer, base, args):
        trainer = LoRATrainer(build_model(base), tokenizer, args)
        assert trainer.adapter_output_dir == f"{args.output_dir}/adapters"
        assert Path(trainer.adapter_output_dir) == tmp_path / "out" / "adapters"

    def test_intermediate_and_final_checkpoints(self, tmp_path: Path, tokenizer):
        report, trainer = fit_lora(tmp_path, tokenizer, adapter_save_steps=SAVE_STEPS)
        assert report.sft["optimizer_steps"] == 4
        assert [item.boundary for item in report.adapter_checkpoints] == [
            f"{INTERMEDIATE_PREFIX}2",
            f"{INTERMEDIATE_PREFIX}4",
            FINAL_ADAPTER_NAME,
        ]
        on_disk = sorted(path.name for path in Path(trainer.adapter_output_dir).iterdir())
        assert on_disk == sorted(
            [f"{INTERMEDIATE_PREFIX}2", f"{INTERMEDIATE_PREFIX}4", FINAL_ADAPTER_NAME]
        )
        intermediates = [
            item
            for item in report.adapter_checkpoints
            if item.boundary.startswith(INTERMEDIATE_PREFIX)
        ]
        assert intermediates
        assert [item.step for item in intermediates] == [2, 4]
        assert all(item.step % SAVE_STEPS == 0 for item in intermediates)
        # 回调拿到的是完整快照：final 的步号 == 报告里的优化器步数
        final = report.adapter_checkpoints[-1]
        assert final.boundary == FINAL_ADAPTER_NAME
        assert final.step == report.sft["optimizer_steps"]

    def test_zero_save_steps_keeps_only_final(self, tmp_path: Path, tokenizer):
        report, trainer = fit_lora(tmp_path, tokenizer, adapter_save_steps=0)
        assert report.adapter_save_steps == 0
        assert [item.boundary for item in report.adapter_checkpoints] == [FINAL_ADAPTER_NAME]
        assert report.adapter_checkpoints[-1].step == report.sft["optimizer_steps"]
        on_disk = sorted(path.name for path in Path(trainer.adapter_output_dir).iterdir())
        assert on_disk == [FINAL_ADAPTER_NAME]

    def test_report_serialization_and_saving_factor(self, tmp_path: Path, tokenizer):
        report, _ = fit_lora(tmp_path, tokenizer, adapter_save_steps=SAVE_STEPS)
        payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
        assert payload["adapter_save_steps"] == SAVE_STEPS
        assert len(payload["adapter_checkpoints"]) == 3
        assert payload["adapter_bytes_final"] == report.adapter_bytes_final
        assert payload["adapter_checkpoints"][-1]["short_hash"] == (
            report.adapter_checkpoints[-1].short_hash
        )
        # 适配器比全参小得多：这是"LoRA 能交付"的落盘版本
        vocab = tokenizer.vocab_size
        assert report.full_finetune_bytes == (vocab**2 + vocab) * 4
        assert report.adapter_saving_factor > 1
        assert payload["adapter_saving_factor"] == round(report.adapter_saving_factor, 4)
        assert "适配器" in report.summary_line()


class TestRestore:
    """``LoRATrainer.restore``：把适配器灌回模型（不加载基座权重）."""

    def test_roundtrip_restores_matrices_and_hash(self, tmp_path: Path, tokenizer):
        report, trainer = fit_lora(tmp_path, tokenizer, adapter_save_steps=0)
        final = report.adapter_checkpoints[-1]
        assert final.boundary == FINAL_ADAPTER_NAME
        assert final.step == report.sft["optimizer_steps"]

        fresh = build_model(ReferenceSFTModel(tokenizer.vocab_size, seed=42))
        restorer = LoRATrainer(fresh, tokenizer, ref_args(tmp_path))
        adapter_dir = Path(trainer.adapter_output_dir) / FINAL_ADAPTER_NAME
        payload = restorer.restore(adapter_dir)

        assert fresh.layer.snapshot_matrices() == trainer.model.layer.snapshot_matrices()
        assert fresh.updates == final.step
        assert payload["sha256"] == final.content_sha256

    def test_rank_mismatch_rejected(self, tmp_path: Path, tokenizer, model, args):
        adapter_dir = tmp_path / "adapters" / FINAL_ADAPTER_NAME
        save_adapter(
            adapter_dir, model=model, args=args, step=1, boundary=FINAL_ADAPTER_NAME
        )
        mismatched = build_model(
            ReferenceSFTModel(tokenizer.vocab_size, seed=42), r=4, lora_alpha=8
        )
        with pytest.raises(AdapterCheckpointError, match="适配器配置与当前模型不一致"):
            LoRATrainer(mismatched, tokenizer, ref_args(tmp_path)).restore(adapter_dir)

    def test_alpha_mismatch_rejected(self, tmp_path: Path, tokenizer, model, args):
        adapter_dir = tmp_path / "adapters" / FINAL_ADAPTER_NAME
        save_adapter(
            adapter_dir, model=model, args=args, step=1, boundary=FINAL_ADAPTER_NAME
        )
        mismatched = build_model(
            ReferenceSFTModel(tokenizer.vocab_size, seed=42), r=8, lora_alpha=32
        )
        with pytest.raises(AdapterCheckpointError, match="适配器配置与当前模型不一致"):
            LoRATrainer(mismatched, tokenizer, ref_args(tmp_path)).restore(adapter_dir)
