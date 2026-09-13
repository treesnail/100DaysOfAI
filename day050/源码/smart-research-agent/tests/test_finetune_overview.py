"""微调方法总览与选型建议测试（day048）：规则分支、可训练比例与表格渲染."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from smart_research_agent.finetune.overview import (
    DPO,
    LORA,
    METHODS,
    QLORA,
    RLHF_PPO,
    SFT,
    MethodProfile,
    lora_trainable_ratio,
    recommend_method,
    render_methods_table,
)

#: 方法家族只允许这三类（画像里的 family 字段是文档与 API 的公共契约）
VALID_FAMILIES = {"parameter-efficient", "full-parameter", "alignment"}


class TestMethods:
    """METHODS 知识常量：五个 key、画像字段完整."""

    def test_covers_five_keys(self):
        assert set(METHODS) == {SFT, LORA, QLORA, DPO, RLHF_PPO}

    def test_key_field_matches_dict_key(self):
        for key, profile in METHODS.items():
            assert profile.key == key
            assert isinstance(profile, MethodProfile)

    def test_families_are_known(self):
        assert {profile.family for profile in METHODS.values()} <= VALID_FAMILIES

    def test_profiles_are_frozen(self):
        with pytest.raises(FrozenInstanceError):
            METHODS[SFT].family = "changed"  # type: ignore[misc]

    def test_artifacts_are_non_empty_tuples(self):
        for profile in METHODS.values():
            assert isinstance(profile.artifacts, tuple)
            assert profile.artifacts


class TestRecommendMethod:
    """确定性选型规则：四条主分支 + 两条告警."""

    def test_preference_pairs_prefers_dpo(self):
        advice = recommend_method(examples=5000, gpu_memory_gb=24, has_preference_pairs=True)
        assert advice.method == DPO
        assert advice.warnings == []
        assert any("偏好对" in reason for reason in advice.reasons)

    def test_small_preference_data_warns_about_rlhf_ppo(self):
        advice = recommend_method(examples=800, gpu_memory_gb=24, has_preference_pairs=True)
        assert advice.method == DPO
        assert len(advice.warnings) == 1
        assert "RLHF-PPO" in advice.warnings[0]

    def test_full_capability_with_80gb_prefers_sft(self):
        advice = recommend_method(
            examples=20000,
            gpu_memory_gb=80,
            need_full_capability=True,
        )
        assert advice.method == SFT
        assert advice.warnings == []

    def test_full_capability_below_threshold_does_not_pick_sft(self):
        advice = recommend_method(examples=20000, gpu_memory_gb=79, need_full_capability=True)
        assert advice.method == LORA

    def test_low_memory_prefers_qlora(self):
        advice = recommend_method(examples=2000, gpu_memory_gb=8)
        assert advice.method == QLORA
        assert any("4-bit" in reason for reason in advice.reasons)

    def test_default_is_lora(self):
        advice = recommend_method(examples=2000, gpu_memory_gb=24)
        assert advice.method == LORA
        assert advice.warnings == []

    def test_small_dataset_always_warns(self):
        for kwargs in (
            {"examples": 100, "gpu_memory_gb": 24},
            {"examples": 100, "gpu_memory_gb": 8},
            {"examples": 100, "gpu_memory_gb": 80, "need_full_capability": True},
        ):
            advice = recommend_method(**kwargs)
            assert any("样本量偏少" in warning for warning in advice.warnings)

    def test_preference_branch_can_carry_two_warnings(self):
        advice = recommend_method(
            examples=100, gpu_memory_gb=80, has_preference_pairs=True, need_full_capability=True
        )
        assert advice.method == DPO
        assert len(advice.warnings) == 2

    def test_boundary_exactly_500_has_no_volume_warning(self):
        advice = recommend_method(examples=500, gpu_memory_gb=24)
        assert advice.warnings == []

    def test_reasons_are_always_present(self):
        for examples in (10, 5000):
            advice = recommend_method(examples=examples, gpu_memory_gb=40)
            assert advice.reasons


class TestLoraTrainableRatio:
    """LoRA 可训练比例：精确公式与参数校验."""

    def test_exact_value_for_4096(self):
        expected = 8 * (4096 + 4096) / (4096 * 4096)
        assert lora_trainable_ratio(4096, 4096, 8) == expected
        assert lora_trainable_ratio(4096, 4096, 8) == 16 / 4096
        assert lora_trainable_ratio(4096, 4096, 8) == pytest.approx(0.00390625)

    def test_non_square_layer(self):
        expected = 4 * (1024 + 4096) / (1024 * 4096)
        assert lora_trainable_ratio(1024, 4096, 4) == expected

    def test_rank_scales_linearly(self):
        assert lora_trainable_ratio(4096, 4096, 16) == 2 * lora_trainable_ratio(4096, 4096, 8)

    @pytest.mark.parametrize(
        "in_features,out_features,rank",
        [(0, 4096, 8), (4096, 0, 8), (4096, 4096, 0), (-1, 4096, 8), (4096, -1, 8)],
    )
    def test_invalid_params_raise(self, in_features, out_features, rank):
        with pytest.raises(ValueError, match="必须为正整数"):
            lora_trainable_ratio(in_features, out_features, rank)


class TestRenderMethodsTable:
    """Markdown 表格：表头、分隔行与五个方法都在."""

    def test_contains_header_and_separator(self):
        table = render_methods_table()
        lines = table.splitlines()
        assert lines[0].startswith("| 方法 |")
        assert set(lines[1].replace("|", "").replace(" ", "")) == {"-"}
        assert "可训练比例" in lines[0]

    def test_every_method_has_a_row(self):
        table = render_methods_table()
        for key in METHODS:
            assert f"`{key}`" in table

    def test_row_count(self):
        assert len(render_methods_table().splitlines()) == 2 + len(METHODS)
