"""``hf_integration.config`` 与 ``types.ModelCard``：那份配置就是模型的形状（day086）."""

from __future__ import annotations

import pytest

from smart_research_agent.hf_integration import config, errors, types
from smart_research_agent.hf_integration.config import (
    DEFAULT_ACTIVATIONS,
    DEFAULT_LN_EPS,
    KNOWN_ACTIVATIONS,
    architecture_of,
    card_from_snapshot,
    check_snapshot_is_a_model,
    describe,
    ffn_of,
    missing_keys,
    parameter_count,
    parse_config,
    profile_agreement,
    reference_cards,
    tie_delta,
    tiny_card,
    untied_card,
)
from smart_research_agent.hf_integration.types import (
    ARCHITECTURE_BERT,
    ARCHITECTURE_GPT2,
    ARCHITECTURES,
    BERT_BASE_PARAMETERS,
    BERT_DEFAULT_INTERMEDIATE,
    CONFIG_FILE,
    FFN_RATIO,
    GPT2_SMALL_PARAMETERS,
    REQUIRED_CONFIG_KEYS,
    SIZE_KEYS,
    ModelCard,
)

from tests.hf_integration_samples import build_case, config_payload

GPT2_TINY = config_payload(ARCHITECTURE_GPT2, 277)
BERT_TINY = config_payload(ARCHITECTURE_BERT, 277)


# --------------------------------------------------------------------------- 键名映射


def test_size_keys_cover_the_same_seven_quantities() -> None:
    """两个架构给的是**同一个量的两个键名**：七项逐一对应（含前馈中间维）."""
    assert set(SIZE_KEYS[ARCHITECTURE_GPT2]) == set(SIZE_KEYS[ARCHITECTURE_BERT])
    assert len(SIZE_KEYS[ARCHITECTURE_GPT2]) == 7
    assert SIZE_KEYS[ARCHITECTURE_GPT2]["hidden"] == "n_embd"
    assert SIZE_KEYS[ARCHITECTURE_BERT]["hidden"] == "hidden_size"
    assert SIZE_KEYS[ARCHITECTURE_GPT2]["layers"] == "n_layer"
    assert SIZE_KEYS[ARCHITECTURE_BERT]["layers"] == "num_hidden_layers"
    assert SIZE_KEYS[ARCHITECTURE_GPT2]["positions"] == "n_positions"
    assert SIZE_KEYS[ARCHITECTURE_BERT]["positions"] == "max_position_embeddings"
    assert SIZE_KEYS[ARCHITECTURE_GPT2]["ffn"] == "n_inner"
    assert SIZE_KEYS[ARCHITECTURE_BERT]["ffn"] == "intermediate_size"


def test_required_keys_are_registered_for_both_architectures() -> None:
    """"必需要的键"是一张有内容的表（缺一个就装不起来）."""
    for model_type in ARCHITECTURES:
        assert "model_type" in REQUIRED_CONFIG_KEYS[model_type]
        assert "vocab_size" in REQUIRED_CONFIG_KEYS[model_type]
        assert len(REQUIRED_CONFIG_KEYS[model_type]) >= 6


def test_missing_keys_follows_the_table_order() -> None:
    """缺键清单的顺序与常量表一致（因此报告里那几行是可复现的）."""
    assert missing_keys({}, model_type=ARCHITECTURE_GPT2) == REQUIRED_CONFIG_KEYS[
        ARCHITECTURE_GPT2
    ]
    assert missing_keys(GPT2_TINY, model_type=ARCHITECTURE_GPT2) == ()
    with pytest.raises(errors.ConfigError):
        missing_keys(GPT2_TINY, model_type="t5")


# --------------------------------------------------------------------------- 架构


def test_architecture_of_reads_model_type() -> None:
    """``model_type`` 是决定"用哪条公式"的那一个字段."""
    assert architecture_of({"model_type": "gpt2"}) == ARCHITECTURE_GPT2
    assert architecture_of({"model_type": "bert"}) == ARCHITECTURE_BERT


def test_architecture_of_rejects_missing_or_unknown() -> None:
    """缺键与不认识的名字分别报，且**绝不回退**到某个默认架构."""
    with pytest.raises(errors.ConfigError, match="model_type"):
        architecture_of({"vocab_size": 10})
    with pytest.raises(errors.ConfigError, match="字符串"):
        architecture_of({"model_type": 7})
    with pytest.raises(errors.ConfigError, match="只给了两个架构"):
        architecture_of({"model_type": "t5"})


# --------------------------------------------------------------------------- 两个解析


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        (ARCHITECTURE_GPT2, {
            "hidden": 16, "heads": 2, "layers": 2, "vocab": 277, "positions": 32,
            "activation": "gelu_new", "ln_eps": 1e-5, "tie": True,
            "token_type": False, "causal": True, "fused": True, "type_vocab": 0,
        }),
        (ARCHITECTURE_BERT, {
            "hidden": 16, "heads": 2, "layers": 2, "vocab": 277, "positions": 32,
            "activation": "gelu", "ln_eps": 1e-12, "tie": True,
            "token_type": True, "causal": False, "fused": False, "type_vocab": 2,
        }),
    ],
)
def test_parse_config_fills_every_field(model_type: str, expected: dict[str, object]) -> None:
    """两个架构走**同一条解析路径**，字段逐个对上（含四个派生量）."""
    payload = config_payload(model_type, 277)
    card = parse_config(payload, name="tiny")
    assert card.name == "tiny"
    assert card.model_type == model_type
    assert card.hidden == expected["hidden"]
    assert card.heads == expected["heads"]
    assert card.layers == expected["layers"]
    assert card.vocab == expected["vocab"]
    assert card.positions == expected["positions"]
    assert card.activation == expected["activation"]
    assert card.ln_eps == expected["ln_eps"]
    assert card.tie_word_embeddings is expected["tie"]
    assert card.uses_token_type is expected["token_type"]
    assert card.causal is expected["causal"]
    assert card.fused_qkv is expected["fused"]
    assert card.type_vocab == expected["type_vocab"]
    assert card.head_dim == 8
    assert card.ffn == FFN_RATIO * 16
    assert card.architectures == tuple(payload["architectures"])  # type: ignore[arg-type]


def test_ffn_comes_from_the_config_when_it_is_there() -> None:
    """前馈中间维**配置优先**，两个架构的缺省规矩不同（这是与真库对账读出来的）.

    ```text
    GPT-2  n_inner 缺省     ⇒ 4·hidden（写 100 就用 100）
    BERT   intermediate_size 缺省 ⇒ 3072（与 hidden 无关）
    ```
    """
    gpt2 = parse_config(config_payload(ARCHITECTURE_GPT2, 277))
    assert gpt2.intermediate is None
    assert gpt2.ffn == 4 * gpt2.hidden == FFN_RATIO * 16
    gpt2_explicit = parse_config(config_payload(ARCHITECTURE_GPT2, 277, n_inner=100))
    assert gpt2_explicit.ffn == 100
    bert = parse_config(config_payload(ARCHITECTURE_BERT, 277))
    assert bert.intermediate == 64
    assert bert.ffn == 64
    payload = config_payload(ARCHITECTURE_BERT, 277)
    del payload["intermediate_size"]
    assert parse_config(payload).ffn == BERT_DEFAULT_INTERMEDIATE == 3072
    assert ffn_of(bert) == bert.ffn


def test_card_name_falls_back_to_name_or_path() -> None:
    """没给 ``name`` 时用配置里的 ``_name_or_path``，再不行用架构名."""
    payload = config_payload(ARCHITECTURE_GPT2, 277)
    assert parse_config(payload).name == "org/tiny"
    payload.pop("_name_or_path")
    assert parse_config(payload).name == ARCHITECTURE_GPT2


def test_defaults_are_used_when_optional_keys_are_absent() -> None:
    """可选键缺席时取**配置文件里的默认值**（而不是随便一个值）."""
    payload = config_payload(ARCHITECTURE_GPT2, 277)
    del payload["activation_function"]
    del payload["layer_norm_epsilon"]
    del payload["tie_word_embeddings"]
    card = parse_config(payload)
    assert card.activation == DEFAULT_ACTIVATIONS[ARCHITECTURE_GPT2]
    assert card.ln_eps == DEFAULT_LN_EPS[ARCHITECTURE_GPT2]
    assert card.tie_word_embeddings is True


def test_bert_type_vocab_override_is_read() -> None:
    """BERT 的 ``type_vocab_size`` 会被读进来（它决定多一张多大的表）."""
    payload = config_payload(ARCHITECTURE_BERT, 277)
    payload["type_vocab_size"] = 4
    assert parse_config(payload).type_vocab == 4


def test_head_count_must_divide_hidden() -> None:
    """头数不能整除隐藏维 ⇒ **这份配置不成立**（不是"参数越界"）.

    同一个条件在 ``hf_source.split_heads`` 里报 ``ShapeError``——
    因为那里要改的是**这次调用的 heads**，这里要改的是**那份配置文件**。
    """
    payload = config_payload(ARCHITECTURE_GPT2, 277)
    payload["n_head"] = 5
    with pytest.raises(errors.ConfigError, match="整除"):
        parse_config(payload)
    with pytest.raises(errors.ParameterError):
        parse_config(payload)


@pytest.mark.parametrize("bad", ["16", 16.0, True, None])
def test_size_keys_must_be_integers(bad: object) -> None:
    """JSON 里的 16.0 与 '16' 都不是一个合法的维度."""
    payload = config_payload(ARCHITECTURE_GPT2, 277)
    payload["n_embd"] = bad
    with pytest.raises(errors.ConfigError, match="整数"):
        parse_config(payload)


def test_size_keys_must_be_positive() -> None:
    """零宽与负宽都要被拒绝（一个零宽的向量上没有点积）."""
    payload = config_payload(ARCHITECTURE_GPT2, 277)
    payload["n_layer"] = 0
    with pytest.raises(errors.ConfigError, match=">= 1"):
        parse_config(payload)


def test_missing_size_key_is_reported_with_its_name() -> None:
    """缺哪一个键就报哪一个（消息里带上那个键的真名）."""
    payload = config_payload(ARCHITECTURE_BERT, 277)
    del payload["num_attention_heads"]
    with pytest.raises(errors.ConfigError, match="num_attention_heads"):
        parse_config(payload)


def test_unknown_activation_is_rejected() -> None:
    """不认识的激活当场拒绝——回退到 relu 会让读数印错模型."""
    payload = config_payload(ARCHITECTURE_GPT2, 277)
    payload["activation_function"] = "swish"
    with pytest.raises(errors.ConfigError, match="不认"):
        parse_config(payload)
    payload["activation_function"] = 7
    with pytest.raises(errors.ConfigError, match="字符串"):
        parse_config(payload)
    assert KNOWN_ACTIVATIONS[ARCHITECTURE_GPT2][0] == "gelu_new"


def test_ln_eps_must_be_a_positive_number() -> None:
    """eps 必须是一个正的数（它是分母上的那一项）."""
    payload = config_payload(ARCHITECTURE_GPT2, 277)
    payload["layer_norm_epsilon"] = "1e-5"
    with pytest.raises(errors.ConfigError, match="必须是数"):
        parse_config(payload)
    payload["layer_norm_epsilon"] = 0.0
    with pytest.raises(errors.ConfigError, match="为正"):
        parse_config(payload)


def test_tie_flag_must_be_boolean() -> None:
    """``tie_word_embeddings`` 必须显式是布尔（字符串 "true" 不算）."""
    payload = config_payload(ARCHITECTURE_GPT2, 277)
    payload["tie_word_embeddings"] = "true"
    with pytest.raises(errors.ConfigError, match="布尔"):
        parse_config(payload)


def test_architectures_must_be_a_list() -> None:
    """``architectures`` 必须是列表（它是一份可选的元数据）."""
    payload = config_payload(ARCHITECTURE_GPT2, 277)
    payload["architectures"] = "GPT2LMHeadModel"
    with pytest.raises(errors.ConfigError, match="列表"):
        parse_config(payload)


# --------------------------------------------------------------------------- 参数量


def test_reference_cards_match_the_library_readings() -> None:
    """两张真实卡片的参数量与**库读数**整数相等（两个都是可复核的整数）."""
    cards = reference_cards()
    assert cards["gpt2"].parameter_count == GPT2_SMALL_PARAMETERS == 124_439_808
    assert cards["bert-base-uncased"].parameter_count == BERT_BASE_PARAMETERS == 109_482_240
    assert parameter_count(cards["gpt2"]) == GPT2_SMALL_PARAMETERS


def test_tiny_cards_have_hand_checkable_counts() -> None:
    """玩具卡片的参数量可以手算：``word + position + L·(12h²+13h) + ln_f``."""
    card = parse_config(GPT2_TINY)
    hidden, layers, vocab, positions = 16, 2, 277, 32
    per_layer = 12 * hidden * hidden + 13 * hidden
    assert card.parameter_count == vocab * hidden + positions * hidden + layers * per_layer + 2 * hidden
    assert card.parameter_count == 11536


def test_tying_saves_exactly_one_embedding_table() -> None:
    """共享与不共享之差**恰好**是 ``vocab × hidden``（一个整数，不是"差不多"）."""
    tied = parse_config(GPT2_TINY)
    untied = untied_card(tied)
    assert tied.tie_word_embeddings is True
    assert untied.tie_word_embeddings is False
    assert untied.parameter_count - tied.parameter_count == tie_delta(tied) == 277 * 16


def test_unknown_architecture_in_a_hand_built_card_is_rejected() -> None:
    """手工造一张未知架构的卡片时，参数量那里会当场拒绝."""
    card = ModelCard(
        name="x",
        model_type="t5",
        hidden=16,
        heads=2,
        layers=2,
        vocab=10,
        positions=8,
        activation="relu",
        ln_eps=1e-6,
    )
    with pytest.raises(errors.ConfigError, match="未知的架构"):
        _ = card.parameter_count


# --------------------------------------------------------------------------- 工厂与读数


def test_tiny_card_factory_both_architectures() -> None:
    """工厂给出的默认值本身是合法的（而且 ``hidden`` 能被 ``heads`` 整除）."""
    for model_type in ARCHITECTURES:
        card = tiny_card(model_type, vocab=64, hidden=8, heads=4, layers=1, positions=16)
        assert card.model_type == model_type
        assert card.hidden % card.heads == 0
        assert card.name == f"{model_type}-tiny"
    with pytest.raises(errors.ConfigError, match="配置模板"):
        tiny_card("t5")


def test_tiny_card_overrides_flow_into_the_payload() -> None:
    """工厂的 ``overrides`` 直接改那份配置；改成一个不成立的组合时会当场拒绝."""
    card = tiny_card(ARCHITECTURE_GPT2, heads=4, hidden=16)
    assert card.heads == 4
    assert card.head_dim == 4
    with pytest.raises(errors.ConfigError, match="整除"):
        tiny_card(ARCHITECTURE_GPT2, heads=3, hidden=16)


def test_describe_and_ffn_of() -> None:
    """两个一行读数（报告与演示脚本共用）."""
    card = tiny_card(ARCHITECTURE_GPT2, vocab=64, hidden=8, heads=2, layers=1, positions=16)
    text = describe(card)
    assert "gpt2-tiny" in text
    assert "causal" in text
    assert "tie=True" in text
    assert ffn_of(card) == 4 * 8


def test_profile_agreement_with_day085_source_profile() -> None:
    """跨天对账：配置里的默认值与 day085 从源码读出的画像**一一对上**."""
    for name, card in reference_cards().items():
        agreement = profile_agreement(card)
        assert agreement["agreed"] is True, name
        assert agreement["activation"]["config"] == agreement["activation"]["source"]
        assert agreement["ln_eps"]["config"] == agreement["ln_eps"]["source"]


def test_card_from_snapshot_and_self_check() -> None:
    """从快照读配置的两条入口给出同一张卡片."""
    case = build_case("bert")
    card = card_from_snapshot(case.resolver, case.snapshot, name="from-snapshot")
    assert card.name == "from-snapshot"
    assert card.model_type == ARCHITECTURE_BERT
    assert check_snapshot_is_a_model(case.resolver, case.snapshot).model_type == ARCHITECTURE_BERT


def test_card_to_dict_carries_every_derived_quantity() -> None:
    """摊平之后的键与那五个派生量都在（报告里读它）."""
    card = tiny_card(ARCHITECTURE_BERT, vocab=64, hidden=8, heads=2, layers=1, positions=16)
    flattened = card.to_dict()
    assert set(flattened) == {
        "name", "model_type", "hidden", "heads", "head_dim", "layers", "ffn", "ffn_ratio",
        "vocab", "positions", "activation", "ln_eps", "tie_word_embeddings", "uses_token_type",
        "type_vocab", "causal", "fused_qkv", "parameters", "architectures",
    }
    assert flattened["head_dim"] == 4
    assert flattened["parameters"] == card.parameter_count


def test_config_file_constant_matches_types() -> None:
    """配置文件的名字是一个常量（两处引用同一个）。"""
    assert CONFIG_FILE == "config.json"
    assert config.CONFIG_FILE == types.CONFIG_FILE
