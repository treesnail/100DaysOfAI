"""day060 ``serving.spec`` 的单元测试：部署形态与显存算术."""

from __future__ import annotations

import pytest

from smart_research_agent.serving.errors import ServingError
from smart_research_agent.serving.spec import (
    ARCH_QWEN3_8B,
    DEFAULT_KV_BYTES_PER_VALUE,
    DEFAULT_NUM_PARALLEL,
    KINDS,
    KIND_ADAPTER,
    KIND_BASE,
    KIND_DESCRIPTIONS,
    KIND_MERGED,
    KIND_REQUIRED_FIELDS,
    SERVING_LIMITATIONS,
    SERVING_OUT_OF_SCOPE,
    SUPPORTED_QUANTIZATION_BITS,
    ModelArchitecture,
    ServingSpec,
    memory_breakdown,
    recommend_kind,
    spec_table,
    weights_bytes,
)


# --------------------------------------------------------------------------- #
# 形态声明与必填字段
# --------------------------------------------------------------------------- #


def test_kind_table_is_the_single_source_of_truth() -> None:
    """形态 → 必填字段的映射必须是表驱动的，且三种形态各有一行."""
    assert KINDS == (KIND_BASE, KIND_ADAPTER, KIND_MERGED)
    assert set(KIND_REQUIRED_FIELDS) == set(KINDS)
    assert KIND_REQUIRED_FIELDS[KIND_ADAPTER] == ("adapter_dir",)
    assert KIND_REQUIRED_FIELDS[KIND_MERGED] == ("merged_dir",)
    assert KIND_REQUIRED_FIELDS[KIND_BASE] == ()
    assert set(KIND_DESCRIPTIONS) == set(KINDS)


def test_base_spec_has_no_artifact_path() -> None:
    """``base`` 形态不需要任何产物路径——它就是未微调的基座."""
    spec = ServingSpec(name="smart-research-base", kind=KIND_BASE)
    assert spec.artifact_path == ""
    assert spec.uses_adapter is False
    assert spec.is_multi_tenant is False
    assert "未微调基座" in spec.summary_line()


def test_adapter_spec_without_adapter_dir_is_rejected_at_construction() -> None:
    """**本模块存在的理由**：声明成 adapter 却没给路径必须在构造期失败.

    这类部署不会报错——它只会安静地服务基座，而"专属模型上线了却没生效"
    正是这样发生的。
    """
    with pytest.raises(ServingError, match="缺少必填字段 adapter_dir"):
        ServingSpec(name="x", kind=KIND_ADAPTER)


def test_merged_spec_without_merged_dir_is_rejected() -> None:
    with pytest.raises(ServingError, match="缺少必填字段 merged_dir"):
        ServingSpec(name="x", kind=KIND_MERGED)


def test_blank_artifact_path_counts_as_missing() -> None:
    """只有空格的路径与空串同等处理（``strip`` 之后再判）."""
    with pytest.raises(ServingError, match="缺少必填字段 adapter_dir"):
        ServingSpec(name="x", kind=KIND_ADAPTER, adapter_dir="   ")


def test_adapter_spec_records_the_path_it_loads() -> None:
    spec = ServingSpec(
        name="smart-research-qwen3-8b",
        kind=KIND_ADAPTER,
        adapter_dir="outputs/lora/adapters/adapter-final",
    )
    assert spec.artifact_path == "outputs/lora/adapters/adapter-final"
    assert spec.uses_adapter is True
    assert "adapter-final" in spec.summary_line()


def test_merged_spec_uses_the_merged_directory() -> None:
    spec = ServingSpec(name="x", kind=KIND_MERGED, merged_dir="outputs/lora-merged")
    assert spec.artifact_path == "outputs/lora-merged"
    assert spec.uses_adapter is False


def test_adapter_on_vllm_is_flagged_as_multi_tenant() -> None:
    """一个基座 + N 个适配器在 vLLM 上需要额外配置，因此单独标记出来."""
    spec = ServingSpec(
        name="x", kind=KIND_ADAPTER, backend="vllm", adapter_dir="a"
    )
    assert spec.is_multi_tenant is True
    ollama_spec = ServingSpec(name="x", kind=KIND_ADAPTER, adapter_dir="a")
    assert ollama_spec.is_multi_tenant is False


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ServingError, match="未知部署形态"):
        ServingSpec(name="x", kind="quantized")


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ServingError, match="未知推理后端"):
        ServingSpec(name="x", kind=KIND_BASE, backend="tgi")


def test_blank_name_is_rejected() -> None:
    with pytest.raises(ServingError, match="部署单元名不能为空"):
        ServingSpec(name="")


def test_blank_base_model_is_rejected() -> None:
    """基座是三元组的第一项，空基座会让"基座不匹配"这类检查无从做起."""
    with pytest.raises(ServingError, match="base_model 不能为空"):
        ServingSpec(name="x", kind=KIND_BASE, base_model="")


def test_context_length_and_parallel_must_be_positive() -> None:
    with pytest.raises(ServingError, match="context_length 必须为正整数"):
        ServingSpec(name="x", kind=KIND_BASE, context_length=0)
    with pytest.raises(ServingError, match="max_parallel 至少为 1"):
        ServingSpec(name="x", kind=KIND_BASE, max_parallel=0)


def test_unsupported_quantization_is_rejected() -> None:
    with pytest.raises(ServingError, match="不支持的量化位宽"):
        ServingSpec(name="x", kind=KIND_BASE, bits_per_parameter=6.0)


def test_spec_to_dict_carries_derived_fields() -> None:
    spec = ServingSpec(
        name="x", kind=KIND_ADAPTER, adapter_dir="a", context_length=8192, max_parallel=2
    )
    payload = spec.to_dict()
    assert payload["uses_adapter"] is True
    assert payload["artifact_path"] == "a"
    assert payload["description"] == KIND_DESCRIPTIONS[KIND_ADAPTER]
    assert payload["context_length"] == 8192
    assert payload["max_parallel"] == 2


# --------------------------------------------------------------------------- #
# 结构参数与显存算术
# --------------------------------------------------------------------------- #


def test_architecture_rejects_invalid_numbers() -> None:
    with pytest.raises(ServingError, match="模型结构需要名字"):
        ModelArchitecture(name="", layers=1, kv_heads=1, head_dim=1, parameters=1)
    with pytest.raises(ServingError, match="layers 必须为正整数"):
        ModelArchitecture(name="a", layers=0, kv_heads=1, head_dim=1, parameters=1)
    with pytest.raises(ServingError, match="kv_heads 必须为正整数"):
        ModelArchitecture(name="a", layers=1, kv_heads=0, head_dim=1, parameters=1)
    with pytest.raises(ServingError, match="head_dim 必须为正整数"):
        ModelArchitecture(name="a", layers=1, kv_heads=1, head_dim=0, parameters=1)
    with pytest.raises(ServingError, match="parameters 必须为正整数"):
        ModelArchitecture(name="a", layers=1, kv_heads=1, head_dim=1, parameters=0)


def test_kv_cache_per_token_matches_the_published_architecture() -> None:
    """实测值：Qwen3-8B（36 层 / 8 KV 头 / head_dim 128，bf16）每 token 144 KiB.

    算式：``2 × 36 × 8 × 128 × 2 = 147456`` 字节。**K 与 V 各一份，
    因此第一个乘数是 2**——这是最常被漏掉的一项，漏掉它会让估算正好少一半。
    """
    assert DEFAULT_KV_BYTES_PER_VALUE == 2
    assert ARCH_QWEN3_8B.kv_bytes_per_token == 147456
    assert ARCH_QWEN3_8B.kv_bytes_per_token / 1024 == 144.0


def test_kv_cache_grows_with_context_and_parallelism() -> None:
    """4096 上下文：单并发 576 MiB，四并发 2304 MiB（**多要 1.69 GiB**）."""
    assert ARCH_QWEN3_8B.kv_cache_bytes(4096, 1) == 603_979_776
    assert ARCH_QWEN3_8B.kv_cache_bytes(4096, 1) / 1024**2 == 576.0
    assert ARCH_QWEN3_8B.kv_cache_bytes(4096, 4) == 2_415_919_104
    assert ARCH_QWEN3_8B.kv_cache_bytes(4096, 4) / 1024**2 == 2304.0


def test_kv_cache_rejects_invalid_arguments() -> None:
    with pytest.raises(ServingError, match="context_length 必须为正整数"):
        ARCH_QWEN3_8B.kv_cache_bytes(0)
    with pytest.raises(ServingError, match="max_parallel 至少为 1"):
        ARCH_QWEN3_8B.kv_cache_bytes(4096, 0)


def test_weights_bytes_by_quantization() -> None:
    assert weights_bytes(8_200_000_000, 16.0) == 16_400_000_000
    assert weights_bytes(8_200_000_000, 4.0) == 4_100_000_000
    assert weights_bytes(1_000, 32.0) == 4_000
    assert weights_bytes(1_000, 8.0) == 1_000


def test_weights_bytes_rejects_bad_input() -> None:
    with pytest.raises(ServingError, match="parameters 必须为正整数"):
        weights_bytes(0, 16.0)
    with pytest.raises(ServingError, match="不支持的量化位宽"):
        weights_bytes(1000, 12.0)


def test_quantization_options_are_a_closed_set() -> None:
    """只支持四档：给一个任意浮点数会让"按几位算的"变得不可核对."""
    assert SUPPORTED_QUANTIZATION_BITS == (4.0, 8.0, 16.0, 32.0)


def test_memory_breakdown_splits_weights_kv_and_adapter() -> None:
    """三块分开列的理由：它们由**不同的决策**决定，合成一个数就没法回答该改哪个."""
    payload = memory_breakdown(
        ARCH_QWEN3_8B, context_length=4096, max_parallel=2, adapter_bytes=1024
    )
    assert payload["weights_bytes"] == 16_400_000_000
    assert payload["weights_gib"] == pytest.approx(15.2737, abs=1e-4)
    assert payload["kv_cache_bytes"] == 1_207_959_552
    assert payload["kv_cache_mib"] == 1152.0
    assert payload["adapter_bytes"] == 1024
    assert (
        payload["total_bytes"]
        == payload["weights_bytes"] + payload["kv_cache_bytes"] + 1024
    )
    assert payload["kv_kib_per_token"] == 144.0
    assert payload["weights_is_lower_bound"] is False


def test_memory_breakdown_flags_quantized_weights_as_a_lower_bound() -> None:
    """4-bit 的真实占用不等于「参数数 / 2」（还有量化常数与未量化层）.

    因此本模块给的是**下界**，而报告必须说出来——"下界"与"以为精确"是两件事。
    """
    payload = memory_breakdown(ARCH_QWEN3_8B, bits_per_parameter=4.0)
    assert payload["weights_bytes"] == 4_100_000_000
    assert payload["weights_is_lower_bound"] is True


def test_memory_breakdown_rejects_negative_adapter() -> None:
    with pytest.raises(ServingError, match="adapter_bytes 不能为负数"):
        memory_breakdown(ARCH_QWEN3_8B, adapter_bytes=-1)


def test_architecture_to_dict_carries_per_token_derivations() -> None:
    payload = ARCH_QWEN3_8B.to_dict()
    assert payload["name"] == "Qwen3-8B"
    assert payload["layers"] == 36
    assert payload["kv_heads"] == 8
    assert payload["kv_bytes_per_token"] == 147456
    assert payload["kv_kib_per_token"] == 144.0


# --------------------------------------------------------------------------- #
# 形态建议与对照表
# --------------------------------------------------------------------------- #


def test_recommend_merged_for_a_single_variant() -> None:
    kind, reason = recommend_kind(variants=1)
    assert kind == KIND_MERGED
    assert "部署链路与普通模型一致" in reason


def test_recommend_adapter_for_multiple_variants() -> None:
    kind, reason = recommend_kind(variants=3)
    assert kind == KIND_ADAPTER
    assert "省下 2 份完整权重" in reason


def test_hot_swap_overrides_the_single_variant_rule() -> None:
    """只有一个业务、但要毫秒级切回基座做对照 → 仍然是 adapter 形态."""
    kind, reason = recommend_kind(variants=1, needs_hot_swap=True)
    assert kind == KIND_ADAPTER
    assert "热插拔" in reason


def test_recommend_kind_rejects_negative_variants() -> None:
    """零个业务是合法的（还没上业务），负数不是."""
    assert recommend_kind(variants=0)[0] == KIND_MERGED
    with pytest.raises(ServingError, match="variants 不能为负数"):
        recommend_kind(variants=-1)


def test_spec_table_lists_required_fields_and_costs() -> None:
    rows = spec_table()
    assert [row["kind"] for row in rows] == list(KINDS)
    adapter_row = next(row for row in rows if row["kind"] == KIND_ADAPTER)
    assert adapter_row["required_fields"] == ["adapter_dir"]
    assert adapter_row["required_fields_text"] == "adapter_dir"
    assert "peft" in adapter_row["runtime_dependency"]
    assert "十几 MB" in adapter_row["switch_cost"]
    base_row = next(row for row in rows if row["kind"] == KIND_BASE)
    assert base_row["required_fields_text"] == "（无）"
    assert base_row["runtime_dependency"] == "不需要额外运行时"


def test_default_parallel_matches_the_ollama_faq() -> None:
    """Ollama 官方 FAQ：``OLLAMA_NUM_PARALLEL`` 默认 1。并发 1 与"没配置"同义."""
    assert DEFAULT_NUM_PARALLEL == 1
    assert ServingSpec(name="x", kind=KIND_BASE).max_parallel == 1


def test_limitations_each_carry_a_checkable_number() -> None:
    """四条限制各带一个可核对的数字——**无法否掉结论的限制等于免责声明**."""
    assert len(SERVING_LIMITATIONS) == 4
    assert any("$1.006" in item for item in SERVING_LIMITATIONS)
    assert any("5%" in item for item in SERVING_LIMITATIONS)
    assert any("must_contain" in item for item in SERVING_LIMITATIONS)
    assert any("31 万参数" in item for item in SERVING_LIMITATIONS)


def test_out_of_scope_names_three_concrete_usages() -> None:
    assert len(SERVING_OUT_OF_SCOPE) == 3
    assert any("采购决策" in item for item in SERVING_OUT_OF_SCOPE)
    assert any("A/B" in item for item in SERVING_OUT_OF_SCOPE)
    assert any("摘掉" in item for item in SERVING_OUT_OF_SCOPE)
