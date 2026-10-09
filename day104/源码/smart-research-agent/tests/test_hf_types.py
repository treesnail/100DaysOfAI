"""``hf_source.types``：画像、口径表与源码阅读笔记（day085 / M7-D9）."""

from __future__ import annotations

import pytest

from smart_research_agent.hf_source import types
from smart_research_agent.hf_source.errors import ParameterError


def test_two_profiles_are_registered() -> None:
    """画像表里只有两个模型，而且名字与键一致."""
    assert tuple(types.PROFILES) == ("gpt2", "bert")
    for name, profile in types.PROFILES.items():
        assert profile.name == name
        assert profile.source_file == types.SOURCE_FILES[name]


@pytest.mark.parametrize(
    ("name", "placement", "epsilon", "activation", "fused", "normalize", "token_type"),
    [
        ("gpt2", "pre", 1e-5, "gelu_new", True, False, False),
        ("bert", "post", 1e-12, "gelu", False, True, True),
    ],
)
def test_profile_fields_are_the_config_defaults(
    name: str,
    placement: str,
    epsilon: float,
    activation: str,
    fused: bool,
    normalize: bool,
    token_type: bool,
) -> None:
    """每一条画像字段都对应配置文件里的一个默认值（**四个默认值的来源**）."""
    profile = types.PROFILES[name]
    assert profile.norm_placement == placement
    assert profile.ln_eps == epsilon
    assert profile.activation == activation
    assert profile.fused_qkv is fused
    assert profile.normalize_embeddings is normalize
    assert profile.uses_token_type is token_type


def test_three_layer_norm_defaults_are_recorded() -> None:
    """三个 eps 默认值（1e-5 / 1e-12 / 1e-6）都在表里，且与常量逐位相同."""
    assert types.LN_EPS_DEFAULTS == {"gpt2": 1e-5, "bert": 1e-12, "t5": 1e-6}
    assert types.LN_EPS_DEFAULTS["gpt2"] == types.LN_EPS_GPT2
    assert types.LN_EPS_DEFAULTS["bert"] == types.LN_EPS_BERT
    assert types.LN_EPS_DEFAULTS["t5"] == types.LN_EPS_T5


def test_profile_of_rejects_unknown_names() -> None:
    """未知模型名当场拒绝（**绝不回退到某个默认画像**）."""
    with pytest.raises(ParameterError):
        types.profile_of("t5")


def test_profile_to_dict_carries_every_field() -> None:
    """画像摊平之后的键与字段一一对应."""
    flattened = types.PROFILES["gpt2"].to_dict()
    assert set(flattened) == {
        "name",
        "norm_placement",
        "ln_eps",
        "activation",
        "fused_qkv",
        "normalize_embeddings",
        "uses_token_type",
        "source_file",
    }


def test_source_shape_derived_quantities() -> None:
    """两个派生量：``head_dim = hidden / heads``、``fused_width = 3·hidden``."""
    shape = types.SourceShape(tokens=4, hidden=6, heads=2, vocab=9)
    assert shape.head_dim == 3
    assert shape.fused_width == 18
    assert shape.causal_default is False
    assert shape.to_dict()["fused_width"] == 18


def test_attention_stage_tables_are_closed() -> None:
    """十个阶段的名单、说明、形状三张表**逐键对齐**."""
    assert len(types.ATTENTION_STAGES) == 10
    assert set(types.ATTENTION_STAGE_DESCRIPTIONS) == set(types.ATTENTION_STAGES)
    assert set(types.ATTENTION_STAGE_SHAPES) == set(types.ATTENTION_STAGES)


def test_generation_strategy_tables_are_closed() -> None:
    """五种策略的名单与说明逐键对齐."""
    assert len(types.GENERATION_STRATEGIES) == 5
    assert set(types.GENERATION_STRATEGY_DESCRIPTIONS) == set(types.GENERATION_STRATEGIES)


def test_activation_and_placement_tables_are_closed() -> None:
    """三种激活与两种摆放的名单和说明逐键对齐."""
    assert set(types.ACTIVATIONS) == {"gelu", "gelu_new", "relu"}
    assert set(types.ACTIVATION_DESCRIPTIONS) == set(types.ACTIVATIONS)
    assert set(types.NORM_PLACEMENTS) == {"pre", "post"}
    assert set(types.NORM_PLACEMENT_DESCRIPTIONS) == set(types.NORM_PLACEMENTS)


def test_seven_properties_have_descriptions_and_failures() -> None:
    """七条性质：名单、说明、"失败意味着什么"三张表逐键对齐."""
    assert len(types.SOURCE_PROPERTIES) == 7
    assert set(types.PROPERTY_DESCRIPTIONS) == set(types.SOURCE_PROPERTIES)
    assert set(types.PROPERTY_FAILURE) == set(types.SOURCE_PROPERTIES)


def test_source_notes_are_twelve_and_ordered() -> None:
    """十二条源码阅读笔记，且顺序表与键集合一致."""
    assert len(types.SOURCE_NOTES) == 12
    assert types.SOURCE_NOTES_ORDER == tuple(types.SOURCE_NOTES)
    assert all(value for value in types.SOURCE_NOTES.values())


def test_source_boundaries_are_listed() -> None:
    """边界是一份非空清单（**这一课明确不承诺的事**）."""
    assert len(types.SOURCE_BOUNDARIES) >= 4
    assert all(value for value in types.SOURCE_BOUNDARIES)


def test_warper_order_is_four_steps() -> None:
    """四个 warper 的顺序是一个常量（顺序即语义）."""
    assert types.WARPER_ORDER == ("repetition_penalty", "temperature", "top_k", "top_p")
    assert "top_k 与 top_p 之间" in types.WARPER_ORDER_NOTE


def test_source_version_constants() -> None:
    """版本号写进常量（"当时读的是哪一版"不是传说）."""
    assert types.SOURCE_LIBRARY == "huggingface/transformers"
    assert types.SOURCE_VERSION.startswith("5")
    assert "v5." in types.SOURCE_VERSION_SAMPLE


def test_generation_settings_to_dict() -> None:
    """生成的配置摊平之后含全部字段（与 ``GenerationConfig`` 对齐）."""
    settings = types.GenerationSettings(max_new_tokens=3, top_k=2, seed=11)
    flattened = settings.to_dict()
    assert flattened["max_new_tokens"] == 3
    assert flattened["top_k"] == 2
    assert flattened["seed"] == 11
    assert "eos_token" in flattened


def test_generation_step_and_result_to_dict() -> None:
    """一步与一次生成的账：**生成的片段不含 prompt**."""
    step = types.GenerationStep(
        step=0, chosen=3, kept=5, top_probability=0.5, entropy=1.0, strategy="greedy"
    )
    assert step.to_dict()["chosen"] == 3
    result = types.GenerationResult(
        token_ids=(1, 2, 3, 4),
        prompt_length=2,
        steps=(step,),
        strategy="greedy",
        stopped_early=True,
        notes=("a", "b"),
    )
    assert result.generated == (3, 4)
    assert result.to_dict()["prompt_length"] == 2
    assert result.to_dict()["stopped_early"] is True
