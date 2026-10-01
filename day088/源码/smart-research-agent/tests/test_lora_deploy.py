"""合并与部署测试（M5-D4）：把适配器变成"可以上线的模型"，每一步都可复核.

本文件里承担"证明结论"角色的两条用例：

1. :meth:`TestVerifyMerge.test_trained_model_still_merges_exactly` —— 训练过
   （``ΔW ≠ 0``）的适配器合并之后，与带适配器的模型在**全部** 557 个上下文上
   输出逐位相同、两次 loss 完全相等（``max_logit_difference == 0.0``）。合并是
   一个"把小矩阵乘进大矩阵"的不可逆动作，行/列约定用错时它**不会报错**，只会
   让输出全错——所以这条断言是部署流水线的硬门禁；
2. :meth:`TestAdapterPersistence.test_save_writes_exactly_three_files` /
   :meth:`TestRepackageAdapter.test_publishes_exactly_four_files` —— 一个适配器
   目录"交付时应当恰好有几个文件"是**磁盘上的事实**，它决定别人拿到手的目录
   能不能被加载（多一个临时文件是脏，少一个必需文件是不可用）。

适配器的内容哈希要回答的是"线上跑的是哪一份"，因此它**只覆盖配置与权重**、
不覆盖训练状态（步数/时间戳会让答案随无关量漂移）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from smart_research_agent.peft.config import LoRAConfig, PEFTConfigError
from smart_research_agent.peft.deploy import (
    MANIFEST_FILE,
    AdapterManifest,
    MergeVerification,
    adapter_registry,
    build_manifest,
    deployment_report,
    read_manifest,
    render_inference_script,
    render_merge_script,
    repackage_adapter,
    verify_merge,
    write_manifest,
)
from smart_research_agent.peft.models import LoRAReferenceModel, default_reference_lora_config
from smart_research_agent.peft.targets import MODEL_SPECS
from smart_research_agent.peft.trainer import (
    ADAPTER_FILES,
    AdapterCheckpointError,
    adapter_content_hash,
    adapter_summary,
    load_adapter,
    save_adapter,
)
from smart_research_agent.sft import (
    IGNORE_INDEX,
    Batch,
    ReferenceSFTModel,
)
from smart_research_agent.sft.args import SFTTrainingArgs
from smart_research_agent.sft.hf_script import DEFAULT_BASE_MODEL

#: 参考模型的词表大小（课程数据集实测值）；r=8 时可训练 8 912 个参数
REFERENCE_VOCAB = 557

#: 参考模型的步长：557 维 bigram + 纯 SGD 的标定量级（见 test_lora_models.py）
REFERENCE_LR = 2.0

#: 本文件统一用第 5 步落盘（"训到哪了"必须能从产物里读出来）
ADAPTER_STEP = 5

#: 一个 ``rows × 9`` 批次的监督位置数：每行 8 个（首列被 prompt 屏蔽）
TOKENS_PER_ROW = 8


def build_base(vocab_size: int = REFERENCE_VOCAB, *, seed: int = 42) -> ReferenceSFTModel:
    """构造基座（``V × V`` 的 bigram 参考模型）."""
    return ReferenceSFTModel(vocab_size, seed=seed)


def build_model(base: ReferenceSFTModel, **overrides) -> LoRAReferenceModel:
    """在基座上挂一个默认的参考适配器（``dropout=0`` / ``targets=bigram``）."""
    return LoRAReferenceModel(base, default_reference_lora_config(**overrides), seed=42)


def make_batch(*, rows: int = 4) -> Batch:
    """构造一个带重复结构的小批次（上下文 3→5→7→3… 的 bigram 数据）."""
    pattern = (3, 5, 7, 3, 5, 7, 3, 5, 9)
    input_ids = tuple(pattern for _ in range(rows))
    labels = tuple((IGNORE_INDEX, *pattern[1:]) for _ in range(rows))
    masks = tuple(tuple([1] * len(pattern)) for _ in range(rows))
    return Batch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=masks,
        pad_token_id=0,
    )


def train_in_place(
    model: LoRAReferenceModel, batches, *, learning_rate: float = REFERENCE_LR
) -> None:
    """在给定批次上就地训练（每个批次一次 ``accumulate + apply_update``）."""
    for batch in batches:
        model.accumulate(batch)
        model.apply_update(learning_rate)


@pytest.fixture
def base() -> ReferenceSFTModel:
    """一个 ``V=557`` 的参考基座（每个用例新建一份，避免状态泄漏）."""
    return build_base()


@pytest.fixture
def model(base: ReferenceSFTModel) -> LoRAReferenceModel:
    """挂上默认适配器的参考模型（此时 ``ΔW`` 恒为 0）."""
    return build_model(base)


@pytest.fixture
def batches() -> list[Batch]:
    """训练与验证共用的一批小批次（4 条 + 2 条 = 48 个监督位置）."""
    return [make_batch(rows=4), make_batch(rows=2)]


@pytest.fixture
def adapter_dir(tmp_path, model) -> Path:
    """一个已落盘的适配器目录（三个文件齐全，``step=5``）."""
    directory = tmp_path / "adapter-final"
    save_adapter(
        directory,
        model=model,
        args=SFTTrainingArgs(),
        step=ADAPTER_STEP,
        boundary="final",
    )
    return directory


@pytest.fixture
def manifest(adapter_dir) -> AdapterManifest:
    """``adapter_dir`` 对应的清单（带一组标签，便于核对 tags 的往返）."""
    return build_manifest(adapter_dir, tags={"squad": "day052"})


def save_model_to(directory, model, *, step: int = ADAPTER_STEP, boundary: str = "final"):
    """把模型落盘成一个适配器（本文件的落盘入口，避免每个用例重复参数）."""
    return save_adapter(
        directory,
        model=model,
        args=SFTTrainingArgs(),
        step=step,
        boundary=boundary,
    )


class TestVerifyMerge:
    """``verify_merge``：合并前后的模型必须是**同一个模型**（逐位一致）."""

    def test_untrained_model_is_bitwise_identical(self, model, batches):
        """``ΔW = 0`` 时合并是恒等变换：全部上下文、两次 loss 都必须完全一致."""
        verification = verify_merge(model, batches)
        assert verification.contexts_checked == REFERENCE_VOCAB
        assert verification.bitwise_identical is True
        assert verification.max_logit_difference == 0.0
        assert verification.passed is True
        assert verification.adapter_loss == verification.merged_loss
        assert verification.supervised_tokens == 6 * TOKENS_PER_ROW == 48

    def test_trained_model_still_merges_exactly(self, model, batches):
        """训练的适配器合并后仍然逐位一致——这条用例守的是行/列约定没有混用."""
        train_in_place(model, batches)
        assert model.delta_is_zero() is False
        verification = verify_merge(model, batches)
        assert verification.bitwise_identical is True
        assert verification.max_logit_difference == 0.0
        assert verification.passed is True
        assert verification.adapter_loss == verification.merged_loss
        assert verification.loss_difference == 0.0

    def test_tolerance_is_recorded(self, model, batches):
        verification = verify_merge(model, batches, tolerance=1e-6)
        assert verification.tolerance == pytest.approx(1e-6)

    def test_negative_tolerance_rejected(self, model, batches):
        with pytest.raises(PEFTConfigError, match="tolerance 不能为负数"):
            verify_merge(model, batches, tolerance=-1e-6)

    def test_to_dict_is_json_serializable_with_derived_flags(self, model, batches):
        payload = verify_merge(model, batches).to_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert payload["passed"] is True
        assert payload["loss_difference"] == 0.0
        assert set(payload) == {
            "contexts_checked",
            "bitwise_identical",
            "max_logit_difference",
            "adapter_loss",
            "merged_loss",
            "supervised_tokens",
            "tolerance",
            "passed",
            "loss_difference",
        }

    def test_summary_line_reports_the_gate(self, model, batches):
        line = verify_merge(model, batches).summary_line()
        assert f"检查 {REFERENCE_VOCAB} 个上下文" in line
        assert "逐位一致 True" in line
        assert "最大 logits 差 0.000e+00" in line
        assert "通过 True" in line


class TestMergeVerificationGate:
    """门禁本身的语义：逐位一致**或**差异在容差内才算通过."""

    def _verification(self, *, tolerance: float, difference: float) -> MergeVerification:
        return MergeVerification(
            contexts_checked=10,
            bitwise_identical=False,
            max_logit_difference=difference,
            adapter_loss=1.0,
            merged_loss=1.25,
            supervised_tokens=5,
            tolerance=tolerance,
        )

    def test_bitwise_identity_passes_regardless_of_tolerance(self):
        """逐位一致是更强的结论：容差再小也通过."""
        verification = MergeVerification(
            contexts_checked=1,
            bitwise_identical=True,
            max_logit_difference=1e9,
            adapter_loss=1.0,
            merged_loss=1.0,
            supervised_tokens=1,
            tolerance=0.0,
        )
        assert verification.passed is True

    def test_difference_within_tolerance_passes(self):
        verification = self._verification(tolerance=0.5, difference=0.25)
        assert verification.passed is True
        assert verification.loss_difference == pytest.approx(0.25)

    def test_difference_beyond_tolerance_fails(self):
        assert self._verification(tolerance=0.1, difference=0.25).passed is False


class TestAdapterPersistence:
    """``save_adapter`` / ``load_adapter`` / ``adapter_summary``：只有三个文件."""

    def test_save_writes_exactly_three_files(self, adapter_dir):
        names = sorted(path.name for path in adapter_dir.iterdir())
        assert names == sorted(ADAPTER_FILES)

    def test_checkpoint_metadata(self, tmp_path, model):
        checkpoint = save_model_to(tmp_path / "adapter-final", model)
        assert checkpoint.files == ADAPTER_FILES
        assert checkpoint.step == ADAPTER_STEP
        assert checkpoint.boundary == "final"
        assert checkpoint.trainable_parameters == model.trainable_parameters
        assert checkpoint.trainable_parameters == 2 * 8 * REFERENCE_VOCAB
        assert checkpoint.learning_rate == pytest.approx(SFTTrainingArgs().learning_rate)
        assert checkpoint.train_loss is None
        assert checkpoint.short_hash == checkpoint.content_sha256[:12]

    def test_load_returns_documented_keys(self, adapter_dir):
        payload = load_adapter(adapter_dir)
        assert set(payload) == {"config", "model", "state", "sha256", "directory"}
        assert payload["directory"] == str(adapter_dir)
        assert payload["config"]["r"] == 8
        assert payload["state"]["step"] == ADAPTER_STEP
        assert payload["state"]["boundary"] == "final"

    def test_load_reports_the_same_hash_as_the_checkpoint(self, adapter_dir):
        payload = load_adapter(adapter_dir)
        expected = adapter_content_hash(config=payload["config"], model=payload["model"])
        assert payload["sha256"] == expected

    def test_missing_file_rejected(self, adapter_dir):
        (adapter_dir / "adapter_config.json").unlink()
        with pytest.raises(AdapterCheckpointError, match="适配器不完整"):
            load_adapter(adapter_dir)

    def test_corrupt_json_rejected(self, adapter_dir):
        """不做"部分可用"的降级：JSON 坏了就必须拒绝，而不是返回半个对象."""
        (adapter_dir / "adapter_model.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(AdapterCheckpointError, match="不是合法 JSON"):
            load_adapter(adapter_dir)

    def test_missing_directory_rejected(self, tmp_path):
        with pytest.raises(AdapterCheckpointError, match="适配器目录不存在"):
            load_adapter(tmp_path / "no-such-adapter")

    def test_adapter_summary_reads_only_the_state(self, adapter_dir):
        line = adapter_summary(adapter_dir)
        assert f"适配器 {adapter_dir}" in line
        assert "final step=5" in line
        assert "可训练 8912" in line

    def test_adapter_summary_missing_state_rejected(self, tmp_path):
        with pytest.raises(AdapterCheckpointError, match="找不到 training_state.json"):
            adapter_summary(tmp_path / "no-such-adapter")


class TestAdapterContentHash:
    """内容哈希：只认"权重是不是同一份"，不认"训到哪了"."""

    def test_stable_for_the_same_weights(self, tmp_path, base):
        first = save_model_to(tmp_path / "a", build_model(base))
        second = save_model_to(tmp_path / "b", build_model(base))
        assert first.content_sha256 == second.content_sha256

    def test_changes_when_weights_change(self, tmp_path, base, batches):
        before = save_model_to(tmp_path / "a", build_model(base))
        trained = build_model(base)
        train_in_place(trained, batches)
        after = save_model_to(tmp_path / "b", trained, step=ADAPTER_STEP + 1)
        assert before.content_sha256 != after.content_sha256

    def test_hash_is_independent_of_key_order(self):
        """规范化序列化（排序键）是"哈希能回答问题"的前提."""
        assert adapter_content_hash(config={"r": 8, "alpha": 16}, model={"a": []}) == (
            adapter_content_hash(config={"alpha": 16, "r": 8}, model={"a": []})
        )

    def test_hash_changes_with_config(self):
        assert adapter_content_hash(config={"r": 8}, model={"a": []}) != (
            adapter_content_hash(config={"r": 16}, model={"a": []})
        )


class TestManifest:
    """适配器清单：产物要自描述（哪个基座、哪个 r、训到第几步）."""

    def test_build_manifest_fields(self, adapter_dir):
        manifest = build_manifest(adapter_dir, tags={"env": "test"})
        assert manifest.adapter_dir == str(adapter_dir)
        assert manifest.base_model == DEFAULT_BASE_MODEL
        assert manifest.lora["r"] == 8
        assert manifest.lora["lora_alpha"] == 16
        assert manifest.lora["resolved_targets"] == ["weight"]
        assert manifest.trainable_parameters == 8912
        assert manifest.adapter_bytes == sum(
            (adapter_dir / name).stat().st_size for name in ADAPTER_FILES
        )
        assert manifest.content_sha256 == load_adapter(adapter_dir)["sha256"]
        assert manifest.step == ADAPTER_STEP
        assert manifest.train_loss is None
        assert manifest.learning_rate == pytest.approx(SFTTrainingArgs().learning_rate)
        assert manifest.merged is False
        assert manifest.tags == {"env": "test"}

    def test_to_dict_adds_short_hash(self, manifest):
        payload = manifest.to_dict()
        assert payload["short_hash"] == manifest.content_sha256[:12]
        assert json.loads(json.dumps(payload)) == payload

    def test_from_dict_ignores_unknown_keys(self):
        """``short_hash`` 是 ``to_dict`` 的派生量，不在字段表里，往返时被丢弃."""
        payload = {
            "adapter_dir": "outputs/lora/adapters/adapter-final",
            "base_model": "some/base",
            "lora": {"r": 8},
            "trainable_parameters": 8912,
            "adapter_bytes": 123,
            "content_sha256": "abc123def456",
            "step": ADAPTER_STEP,
            "train_loss": None,
            "learning_rate": 2e-4,
            "short_hash": "abc123def456",
            "future_field": 1,
        }
        manifest = AdapterManifest.from_dict(payload)
        assert manifest.adapter_dir == payload["adapter_dir"]
        assert manifest.base_model == "some/base"
        assert manifest.lora == {"r": 8}
        assert manifest.trainable_parameters == 8912
        assert manifest.step == ADAPTER_STEP
        assert manifest.train_loss is None
        assert manifest.merged is False
        assert manifest.tags == {}
        assert "future_field" not in manifest.to_dict()
        assert AdapterManifest.from_dict(manifest.to_dict()) == manifest

    def test_write_then_read_roundtrip(self, adapter_dir):
        written = build_manifest(adapter_dir, merged=True, tags={"team": "m5"})
        path = write_manifest(adapter_dir, written)
        assert path == adapter_dir / MANIFEST_FILE
        assert path.exists()
        assert read_manifest(adapter_dir) == written

    def test_missing_manifest_rejected(self, tmp_path):
        with pytest.raises(AdapterCheckpointError, match="找不到适配器清单"):
            read_manifest(tmp_path)

    def test_summary_line_is_self_describing(self, manifest):
        line = manifest.summary_line()
        assert line.startswith("adapter-final | ")
        assert "r=8 alpha=16" in line
        assert f"step {ADAPTER_STEP}" in line
        assert f"sha256:{manifest.content_sha256[:12]}" in line
        assert "已合并 False" in line


class TestDeploymentReport:
    """两种部署形态的体积对照：一个基座 + N 个适配器 vs N 份合并模型."""

    def test_base_model_bytes_follow_the_bit_width(self, manifest):
        spec_parameters = MODEL_SPECS["llama-2-7b"].total_parameters()
        report = deployment_report(spec_parameters=spec_parameters, manifest=manifest)
        assert spec_parameters == 6_738_415_616
        assert report["base_model_bytes"] == 6_738_415_616 * 16 // 8
        assert report["base_model_bytes"] == 13_476_831_232

    def test_adapter_is_negligible_next_to_the_base(self, manifest):
        report = deployment_report(spec_parameters=6_738_415_616, manifest=manifest)
        assert report["adapter_bytes"] == manifest.adapter_bytes
        assert report["adapter_ratio"] < 1e-4
        assert report["adapter_mebibytes"] < 1.0

    def test_multi_adapter_saving_factor_is_near_ten(self, manifest):
        report = deployment_report(spec_parameters=6_738_415_616, manifest=manifest)
        assert report["multi_adapter_saving_factor"] == pytest.approx(10.0, abs=0.01)
        assert report["merged_model_bytes"] == report["base_model_bytes"]
        assert report["ten_merged_models_bytes"] == report["base_model_bytes"] * 10
        assert report["one_base_plus_ten_adapters_bytes"] == (
            report["base_model_bytes"] + report["adapter_bytes"] * 10
        )

    @pytest.mark.parametrize("spec_parameters", [0, -1])
    def test_non_positive_parameters_rejected(self, manifest, spec_parameters):
        with pytest.raises(PEFTConfigError, match="spec_parameters 必须为正整数"):
            deployment_report(spec_parameters=spec_parameters, manifest=manifest)

    @pytest.mark.parametrize("bits", [0, -16.0])
    def test_non_positive_bit_width_rejected(self, manifest, bits):
        with pytest.raises(PEFTConfigError, match="base_bits_per_parameter 必须为正数"):
            deployment_report(
                spec_parameters=6_738_415_616, manifest=manifest, base_bits_per_parameter=bits
            )


class TestAdapterRegistry:
    """注册表：一个基座 + N 个适配器的运维视图（按名字与步号排序）."""

    @staticmethod
    def _manifest(root, name: str, step: int, model) -> AdapterManifest:
        checkpoint = save_model_to(root / name, model, step=step, boundary=name)
        return build_manifest(checkpoint.directory)

    def test_rows_are_sorted_by_adapter_name_then_step(self, tmp_path, base):
        model = build_model(base)
        manifests = [
            self._manifest(tmp_path, "adapter-step-20", 20, model),
            self._manifest(tmp_path, "adapter-final", 30, model),
            self._manifest(tmp_path, "adapter-step-10", 10, model),
        ]
        rows = adapter_registry(manifests)
        assert [row["adapter"] for row in rows] == [
            "adapter-final",
            "adapter-step-10",
            "adapter-step-20",
        ]

    def test_row_fields(self, tmp_path, base):
        manifest = self._manifest(tmp_path, "adapter-final", ADAPTER_STEP, build_model(base))
        row = adapter_registry([manifest])[0]
        assert row == {
            "adapter": "adapter-final",
            "base_model": DEFAULT_BASE_MODEL,
            "step": ADAPTER_STEP,
            "train_loss": None,
            "trainable_parameters": 8912,
            "adapter_bytes": manifest.adapter_bytes,
            "short_hash": manifest.content_sha256[:12],
            "merged": False,
            "tags": {},
        }

    def test_empty_registry(self):
        assert adapter_registry([]) == []


class TestScripts:
    """两份生成脚本：能被 ``compile`` 编译，且核心片段与配置同源."""

    def test_merge_script_compiles(self):
        compile(render_merge_script(LoRAConfig()), "merge_adapter.py", "exec")

    def test_inference_script_compiles(self):
        compile(render_inference_script(), "infer_adapter.py", "exec")

    def test_merge_script_contains_the_core_calls(self):
        text = render_merge_script(LoRAConfig())
        assert "PeftModel.from_pretrained" in text
        assert "merge_and_unload" in text
        assert "save_pretrained" in text
        assert "adapter_manifest.json" in text

    def test_inference_script_contains_the_core_calls(self):
        text = render_inference_script()
        assert "PeftModel.from_pretrained" in text
        assert "apply_chat_template" in text
        assert "adapter_manifest.json" in text

    @pytest.mark.parametrize(
        "lora",
        [
            LoRAConfig(),
            LoRAConfig(r=16, lora_alpha=32, target_modules="mlp", lora_dropout=0.0),
            LoRAConfig(r=4, lora_alpha=8, target_modules=("q_proj", "v_proj"), use_rslora=True),
        ],
    )
    def test_merge_script_embeds_the_same_lora_config(self, lora):
        """脚本里内嵌的 ``LORA_CONFIG`` 必须与 ``to_peft_dict()`` 同源（不是手抄）."""
        match = re.search(
            r'LORA_CONFIG = json\.loads\(\s*"""(.*?)"""\s*\)',
            render_merge_script(lora),
            re.S,
        )
        assert match is not None
        assert json.loads(match.group(1)) == lora.to_peft_dict()

    def test_merge_script_uses_the_requested_paths(self):
        text = render_merge_script(
            LoRAConfig(), base_model="meta-llama/Llama-2-7b-hf", output_dir="out/merged"
        )
        assert "meta-llama/Llama-2-7b-hf" in text
        assert "out/merged" in text

    def test_inference_script_uses_the_requested_paths(self):
        text = render_inference_script(base_model="base/x", adapter_dir="adapters/a")
        assert "base/x" in text
        assert "adapters/a" in text

    def test_merge_script_rejects_invalid_config(self):
        with pytest.raises(PEFTConfigError, match="r 必须为正整数"):
            render_merge_script(LoRAConfig(r=0))


class TestRepackageAdapter:
    """``repackage_adapter``：复制的是三个文件，清单**重新生成**."""

    def test_publishes_exactly_four_files(self, adapter_dir, tmp_path):
        target = tmp_path / "published"
        repackage_adapter(adapter_dir, target)
        names = sorted(path.name for path in target.iterdir())
        assert names == sorted([*ADAPTER_FILES, MANIFEST_FILE])

    def test_temporary_files_are_not_copied(self, adapter_dir, tmp_path):
        """发布目录里不能留下训练过程中的临时文件（这是"脏目录"的来源）."""
        (adapter_dir / "optimizer_state.bin").write_text("junk", encoding="utf-8")
        (adapter_dir / "debug.tmp").write_text("scratch", encoding="utf-8")
        target = tmp_path / "published"
        repackage_adapter(adapter_dir, target)
        assert {path.name for path in target.iterdir()} == {*ADAPTER_FILES, MANIFEST_FILE}

    def test_manifest_is_regenerated_for_the_new_directory(self, adapter_dir, tmp_path):
        source = build_manifest(adapter_dir)
        target = tmp_path / "published"
        manifest = repackage_adapter(adapter_dir, target, tags={"release": "v1"})
        assert manifest.adapter_dir == str(target)
        assert manifest.base_model == DEFAULT_BASE_MODEL
        assert manifest.merged is False
        assert manifest.tags == {"release": "v1"}
        assert manifest.content_sha256 == source.content_sha256
        assert manifest.trainable_parameters == source.trainable_parameters
        assert manifest.step == source.step
        assert read_manifest(target) == manifest

    def test_bytes_match_the_source_adapter(self, adapter_dir, tmp_path):
        target = tmp_path / "published"
        manifest = repackage_adapter(adapter_dir, target)
        assert manifest.adapter_bytes == sum(
            (target / name).stat().st_size for name in ADAPTER_FILES
        )
        assert manifest.adapter_bytes == build_manifest(adapter_dir).adapter_bytes

    def test_target_directory_is_created_on_demand(self, adapter_dir, tmp_path):
        target = tmp_path / "nested" / "release" / "v1"
        assert not target.exists()
        repackage_adapter(adapter_dir, target)
        assert target.is_dir()
