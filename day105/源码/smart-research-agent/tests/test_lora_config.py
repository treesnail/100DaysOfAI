"""LoRA / QLoRA 配置测试（M5-D3）：默认值、派生量、校验与序列化.

本文件守的是 ``peft/config.py`` 里那些**只影响参数量与显存、不影响训练循环**
的算术关系：``scaling = alpha/r``（rsLoRA 时是 ``alpha/sqrt(r)``）、单层新增
参数 ``r·(in+out)``、每参数位数 ``4 + 常数开销``。这些数字算错时训练会照常
跑、loss 会照常降，只是训出来的东西与以为的不是同一回事——所以它们必须被
断言钉死，而不是留在注释里。
"""

from __future__ import annotations

import pytest

from smart_research_agent.peft.config import (
    BASELINE_ONLY_QUANT_TYPES,
    CODE_BITS,
    DTYPE_BYTES,
    LORA_TARGET_PRESETS,
    RUNTIME_ONLY_INIT_MODES,
    SUPPORTED_INIT_MODES,
    SUPPORTED_LORA_BIASES,
    SUPPORTED_QUANT_TYPES,
    LoRAConfig,
    PEFTConfigError,
    QLoRAConfig,
    default_peft_lr_range,
    quantization_table,
)


class TestConstants:
    """预设表与支持列表：配置合法性的单一事实来源."""

    def test_target_presets_expand_to_expected_modules(self):
        """预设名必须展开成"到底适配了什么"那句可核对的话."""
        assert LORA_TARGET_PRESETS["attention"] == ("q_proj", "v_proj")
        assert LORA_TARGET_PRESETS["all_linear"] == (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
        assert LORA_TARGET_PRESETS["bigram"] == ("weight",)

    def test_bias_values_match_peft(self):
        """``bias`` 的取值与 peft ``LoraConfig.bias`` 一致."""
        assert SUPPORTED_LORA_BIASES == ("none", "all", "lora_only")

    def test_runtime_only_init_modes_are_strict_subset(self):
        """需要运行时才能执行的初始化是本课支持模式的真子集（会被显式拒绝）."""
        assert set(RUNTIME_ONLY_INIT_MODES) < set(SUPPORTED_INIT_MODES)
        assert set(RUNTIME_ONLY_INIT_MODES) == {"loftq", "eva", "olora", "orthogonal"}

    def test_int4_is_baseline_only_but_a_supported_quant_type(self):
        """``int4`` 是对照基线：validate() 认它，BitsAndBytesConfig 不认它."""
        assert BASELINE_ONLY_QUANT_TYPES == ("int4",)
        assert "int4" in SUPPORTED_QUANT_TYPES

    def test_code_bits_and_dtype_bytes(self):
        assert CODE_BITS == 4
        assert DTYPE_BYTES["bfloat16"] == 2
        assert DTYPE_BYTES["float32"] == 4
        assert DTYPE_BYTES["uint8"] == 1


class TestLoRAConfigDefaults:
    """缺省值与 peft 官方文档的常见组合逐字一致."""

    def test_field_defaults(self):
        config = LoRAConfig()
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == pytest.approx(0.05)
        assert config.target_modules == "attention"
        assert config.bias == "none"
        assert config.use_rslora is False
        assert config.use_dora is False
        assert config.init_lora_weights == "gaussian"
        assert config.task_type == "CAUSAL_LM"
        assert config.fan_in_fan_out is False

    def test_defaults_are_valid(self):
        LoRAConfig().validate()

    def test_default_preset_resolves_to_q_and_v(self):
        config = LoRAConfig()
        assert config.resolved_targets == ("q_proj", "v_proj")
        assert config.target_preset_name == "attention"

    def test_custom_targets_are_normalized_to_tuple(self):
        """直接传列表也要在构造处变成元组，后续派生量才看到同一个集合."""
        config = LoRAConfig(target_modules=["q_proj", "v_proj"])
        assert config.target_modules == ("q_proj", "v_proj")
        assert config.resolved_targets == ("q_proj", "v_proj")
        assert config.target_preset_name == "custom"


class TestLoRAConfigDerived:
    """派生量算术：scaling、单层参数、秩上界与可训练占比."""

    def test_scaling_is_alpha_over_r(self):
        assert LoRAConfig(r=8, lora_alpha=16).scaling == pytest.approx(2.0)
        assert LoRAConfig(r=16, lora_alpha=16).scaling == pytest.approx(1.0)

    def test_scaling_ignores_absolute_alpha_at_fixed_ratio(self):
        """alpha = 2r 的惯例让"r 翻倍"时缩放保持不变."""
        assert LoRAConfig(r=4, lora_alpha=8).scaling == pytest.approx(
            LoRAConfig(r=8, lora_alpha=16).scaling
        )

    def test_rslora_scaling_uses_sqrt(self):
        config = LoRAConfig(r=4, lora_alpha=16, use_rslora=True)
        assert config.scaling == pytest.approx(16 / 2.0)

    def test_scaling_formula_text(self):
        assert LoRAConfig(r=8, lora_alpha=16).scaling_formula == "alpha/r = 16/8 = 2.000000"
        assert (
            LoRAConfig(r=4, lora_alpha=16, use_rslora=True).scaling_formula
            == "alpha/sqrt(r) = 16/sqrt(4) = 8.000000"
        )

    def test_scaling_rejects_non_positive_r(self):
        """``r = 0`` 时缩放无定义——在取属性处就拒绝，而不是抛 ZeroDivisionError."""
        with pytest.raises(PEFTConfigError, match="r 必须为正整数"):
            LoRAConfig(r=0).scaling

    def test_requires_peft_runtime_flag(self):
        assert LoRAConfig().requires_peft_runtime is False
        assert LoRAConfig(init_lora_weights="loftq").requires_peft_runtime is True

    def test_adapter_parameter_count_is_linear_in_r(self):
        """``4096×4096`` 的投影层：r=8 只新增 65 536 个参数（原参数 16 777 216）."""
        config = LoRAConfig()
        assert config.adapter_parameter_count(4096, 4096) == 65_536
        assert config.adapter_parameter_count(4096, 4096) == 8 * (4096 + 4096)

    def test_adapter_parameter_count_adds_bias(self):
        lora_only = LoRAConfig(r=8, bias="lora_only").adapter_parameter_count(4096, 4096)
        all_bias = LoRAConfig(r=8, bias="all").adapter_parameter_count(4096, 4096)
        assert lora_only == 65_536 + 4096
        assert all_bias == 65_536 + 8192
        assert LoRAConfig(r=8, bias="none").adapter_parameter_count(4096, 4096) == 65_536

    def test_adapter_parameter_count_rejects_bad_shape(self):
        with pytest.raises(PEFTConfigError, match="必须为正整数"):
            LoRAConfig().adapter_parameter_count(0, 4096)

    def test_rank_upper_bound_is_min_of_three(self):
        config = LoRAConfig(r=64)
        assert config.rank_upper_bound(4096, 4096) == 64
        assert config.rank_upper_bound(4, 4096) == 4
        assert config.rank_upper_bound(32, 8) == 8

    def test_trainable_ratio_denominator_includes_adapter(self):
        """分母是"模型总参数"（含适配器），与 peft 打印的口径一致."""
        assert LoRAConfig().trainable_ratio(1000, 100) == pytest.approx(100 / 1100)

    def test_trainable_ratio_rejects_zero_total(self):
        with pytest.raises(PEFTConfigError, match="总参数量必须为正数"):
            LoRAConfig().trainable_ratio(0, 0)


class TestLoRAConfigValidation:
    """``validate()`` 的每一个非法分支各有一条用例."""

    @pytest.mark.parametrize(
        "changes, match",
        [
            ({"r": 0}, "r 必须为正整数"),
            ({"lora_alpha": 0}, "lora_alpha 必须为正数"),
            ({"lora_dropout": 1.0}, r"lora_dropout 必须落在 \[0, 1\)"),
            ({"lora_dropout": -0.1}, r"lora_dropout 必须落在 \[0, 1\)"),
            ({"bias": "both"}, "未知的 bias"),
            ({"init_lora_weights": "magic"}, "未知的 init_lora_weights"),
            ({"target_modules": ()}, "target_modules 不能为空"),
            ({"target_modules": ("q_proj", "q_proj")}, "target_modules 存在重复项"),
            ({"target_modules": "no_such_preset"}, "未知的 target_modules 预设"),
            ({"use_dora": True, "use_rslora": True}, "use_dora 与 use_rslora 不能同时启用"),
            ({"use_dora": True, "init_lora_weights": "random"}, "不能为 random"),
            ({"init_lora_weights": "loftq", "fan_in_fan_out": True}, "不支持 fan_in_fan_out"),
        ],
    )
    def test_invalid_configs_rejected(self, changes, match):
        with pytest.raises(PEFTConfigError, match=match):
            LoRAConfig(**changes).validate()

    def test_zero_dropout_and_lora_only_bias_are_valid(self):
        LoRAConfig(lora_dropout=0.0, bias="lora_only").validate()

    def test_runtime_only_init_modes_pass_validate(self):
        """它们只是"本课参考实现跑不了"，不是配置非法."""
        for mode in RUNTIME_ONLY_INIT_MODES:
            LoRAConfig(init_lora_weights=mode).validate()

    def test_error_type_is_value_error(self):
        """``PEFTConfigError`` 继承 ``ValueError``，便于按类型捕获."""
        assert issubclass(PEFTConfigError, ValueError)


class TestLoRAConfigSerialization:
    """投影与序列化：字段名逐字对齐、往返一致、不改原对象."""

    def test_to_peft_dict_keys_are_field_aligned(self):
        payload = LoRAConfig().to_peft_dict()
        assert set(payload) == {
            "r",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "bias",
            "use_rslora",
            "use_dora",
            "init_lora_weights",
            "task_type",
            "fan_in_fan_out",
        }
        assert payload["target_modules"] == ["q_proj", "v_proj"]
        assert payload["r"] == 8 and payload["lora_alpha"] == 16
        assert payload["bias"] == "none" and payload["task_type"] == "CAUSAL_LM"

    def test_to_dict_adds_derived_values(self):
        config = LoRAConfig()
        payload = config.to_dict()
        # to_dict 保留原始写法（预设名），展开后的集合另用 resolved_targets 给出
        assert payload["target_modules"] == "attention"
        assert payload["resolved_targets"] == ["q_proj", "v_proj"]
        assert payload["target_preset_name"] == "attention"
        assert payload["scaling"] == pytest.approx(config.scaling)
        assert payload["scaling_formula"] == config.scaling_formula
        assert payload["requires_peft_runtime"] is False

    def test_from_dict_roundtrip(self):
        config = LoRAConfig(
            r=16, lora_alpha=32, target_modules=("q_proj", "v_proj"), lora_dropout=0.1
        )
        assert LoRAConfig.from_dict(config.to_dict()) == config

    def test_from_dict_preset_string_is_preserved(self):
        """``to_dict`` 保留预设名，因此往返之后预设名不会丢失."""
        restored = LoRAConfig.from_dict(LoRAConfig(target_modules="mlp").to_dict())
        assert restored.target_modules == "mlp"
        assert restored.target_preset_name == "mlp"
        assert restored.resolved_targets == ("gate_proj", "up_proj", "down_proj")

    def test_from_dict_ignores_unknown_and_derived_keys(self):
        restored = LoRAConfig.from_dict({"r": 4, "future_field": 1, "scaling": 99.0})
        assert restored.r == 4
        assert restored.lora_alpha == 16

    def test_from_dict_converts_target_list_to_tuple(self):
        restored = LoRAConfig.from_dict({"target_modules": ["q_proj", "v_proj"]})
        assert restored.target_modules == ("q_proj", "v_proj")

    def test_with_overrides_does_not_mutate(self):
        config = LoRAConfig()
        changed = config.with_overrides(r=32)
        assert config.r == 8
        assert changed.r == 32
        assert changed.lora_alpha == config.lora_alpha

    def test_describe_line(self):
        expected = (
            "LoRA r=8 alpha=16 dropout=0.05 targets=attention['q_proj', 'v_proj'] "
            "bias=none scaling[alpha/r = 16/8 = 2.000000]"
        )
        assert LoRAConfig().describe() == expected


class TestQLoRAConfigDefaults:
    """缺省值与 ``BitsAndBytesConfig``（nf4 + 二级量化 + block=64）一致."""

    def test_field_defaults(self):
        config = QLoRAConfig()
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_compute_dtype == "bfloat16"
        assert config.bnb_4bit_use_double_quant is True
        assert config.bnb_4bit_quant_storage == "uint8"
        assert config.block_size == 64
        assert config.double_quant_block_size == 256

    def test_defaults_are_valid(self):
        QLoRAConfig().validate()

    def test_code_bits(self):
        assert QLoRAConfig().code_bits == CODE_BITS == 4

    def test_compute_dtype_bytes(self):
        assert QLoRAConfig().compute_dtype_bytes == 2
        assert QLoRAConfig(bnb_4bit_compute_dtype="float32").compute_dtype_bytes == 4


class TestQLoRAConfigDerived:
    """每参数存储的三段算术：码点 4 bit + 一级常数 + 二级常数."""

    def test_constant_bits_with_double_quant(self):
        config = QLoRAConfig()
        assert config.constant_bits_per_parameter == pytest.approx(8 / 64 + 32 / (64 * 256))
        assert config.constant_bits_per_parameter == pytest.approx(0.126953125)

    def test_constant_bits_single_quant_is_fp32_per_block(self):
        config = QLoRAConfig(bnb_4bit_use_double_quant=False)
        assert config.constant_bits_per_parameter == pytest.approx(32 / 64)
        assert config.constant_bits_per_parameter == pytest.approx(0.5)

    def test_bits_per_parameter(self):
        assert QLoRAConfig().bits_per_parameter == pytest.approx(4.126953125)

    def test_bytes_per_parameter_with_double_quant(self):
        """论文口径：nf4 + 二级量化 + block=64 是 0.515869 字节/参数."""
        assert QLoRAConfig(block_size=64).bytes_per_parameter == pytest.approx(0.515869)

    def test_bytes_per_parameter_without_double_quant(self):
        config = QLoRAConfig(bnb_4bit_use_double_quant=False)
        assert config.bytes_per_parameter == pytest.approx(0.5625)

    def test_larger_block_size_saves_constant_overhead(self):
        """块越大，每块共享的 absmax 越省——"4-bit 并不等于 0.5 字节"是常数开销造成的."""
        fine = QLoRAConfig(block_size=64).bytes_per_parameter
        coarse = QLoRAConfig(block_size=256).bytes_per_parameter
        assert coarse == pytest.approx(0.503967, abs=1e-6)
        assert coarse < fine

    def test_base_weight_bytes_rounds_to_int(self):
        assert QLoRAConfig().base_weight_bytes(1_000_000) == 515_869

    def test_base_weight_bytes_rejects_non_positive(self):
        with pytest.raises(PEFTConfigError, match="base_parameters 必须为正整数"):
            QLoRAConfig().base_weight_bytes(0)


class TestQLoRAConfigValidation:
    """``validate()`` 的每一个非法分支各有一条用例."""

    @pytest.mark.parametrize(
        "changes, match",
        [
            ({"bnb_4bit_quant_type": "int5"}, "未知的量化类型"),
            ({"bnb_4bit_compute_dtype": "float64"}, "未知的 compute_dtype"),
            ({"bnb_4bit_quant_storage": "int8"}, "未知的 quant_storage"),
            ({"block_size": 100}, "未知的 block_size"),
            ({"double_quant_block_size": 100}, "未知的 double_quant_block_size"),
        ],
    )
    def test_invalid_configs_rejected(self, changes, match):
        with pytest.raises(PEFTConfigError, match=match):
            QLoRAConfig(**changes).validate()

    def test_all_supported_block_sizes_are_valid(self):
        for size in (64, 128, 256, 512):
            QLoRAConfig(block_size=size, double_quant_block_size=size).validate()

    def test_error_type_is_value_error(self):
        assert issubclass(PEFTConfigError, ValueError)


class TestQLoRAConfigSerialization:
    """``to_bnb_dict`` / ``to_dict`` / ``from_dict`` / ``with_overrides``."""

    def test_to_bnb_dict_keys(self):
        payload = QLoRAConfig().to_bnb_dict()
        assert set(payload) == {
            "load_in_4bit",
            "bnb_4bit_quant_type",
            "bnb_4bit_compute_dtype",
            "bnb_4bit_use_double_quant",
            "bnb_4bit_quant_storage",
        }
        assert payload["load_in_4bit"] is True
        assert payload["bnb_4bit_quant_type"] == "nf4"

    def test_to_bnb_dict_rejects_baseline_quant_type(self):
        """``int4`` 是对照基线：与其让下游报一个看不懂的错，不如在这里说清楚."""
        with pytest.raises(PEFTConfigError, match="量化对照基线"):
            QLoRAConfig(bnb_4bit_quant_type="int4").to_bnb_dict()

    def test_int4_passes_validate_but_fails_bnb_projection(self):
        config = QLoRAConfig(bnb_4bit_quant_type="int4")
        config.validate()
        with pytest.raises(PEFTConfigError):
            config.to_bnb_dict()

    def test_to_dict_rounds_derived_values(self):
        payload = QLoRAConfig().to_dict()
        assert payload["bits_per_parameter"] == 4.126953
        assert payload["bytes_per_parameter"] == 0.515869
        assert payload["constant_bits_per_parameter"] == 0.126953

    def test_from_dict_roundtrip(self):
        config = QLoRAConfig(block_size=128, bnb_4bit_compute_dtype="float16")
        assert QLoRAConfig.from_dict(config.to_dict()) == config

    def test_from_dict_ignores_unknown_keys(self):
        restored = QLoRAConfig.from_dict({"block_size": 512, "nope": 1, "as_gib": 3.0})
        assert restored.block_size == 512
        assert restored.bnb_4bit_quant_type == "nf4"

    def test_with_overrides_does_not_mutate(self):
        config = QLoRAConfig()
        changed = config.with_overrides(block_size=512)
        assert config.block_size == 64
        assert changed.block_size == 512

    def test_describe_line(self):
        expected = (
            "QLoRA nf4 block=64 double_quant=True compute=bfloat16 "
            "4.126953 bit/参数（0.515869 B）"
        )
        assert QLoRAConfig().describe() == expected


class TestQuantizationTable:
    """``quantization_table``：块大小 → 每参数存储的对照表."""

    def test_rows_follow_input_order(self):
        rows = quantization_table((64, 128, 256, 512))
        assert [row["block_size"] for row in rows] == [64, 128, 256, 512]

    def test_default_argument_covers_four_block_sizes(self):
        assert len(quantization_table()) == 4

    def test_block_size_64_row_matches_formula(self):
        row = quantization_table((64,))[0]
        assert row["bytes_per_parameter"] == pytest.approx(0.515869, abs=1e-6)
        assert row["bytes_per_parameter_single_quant"] == pytest.approx(0.5625)
        assert row["bits_per_parameter"] == pytest.approx(4.126953, abs=1e-6)
        assert row["constant_bits_per_parameter"] == pytest.approx(0.126953, abs=1e-6)

    def test_larger_block_saves_storage(self):
        rows = quantization_table()
        assert rows[-1]["bytes_per_parameter"] < rows[0]["bytes_per_parameter"]
        assert rows[-1]["constant_bits_per_parameter"] < rows[0]["constant_bits_per_parameter"]


class TestDefaultPeftLrRange:
    """LoRA 的学习率区间与全参微调不同，必须从本模块取."""

    def test_range_matches_sft_args_value(self):
        low, high = default_peft_lr_range()
        assert (low, high) == (1e-4, 5e-4)
        assert low < high
