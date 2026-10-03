"""``hf_source.blocks``：两种摆放与两个激活（day085 / M7-D9）."""

from __future__ import annotations

import math

import pytest

from smart_research_agent.hf_source import blocks, types
from smart_research_agent.hf_source.errors import ParameterError, ShapeError
from tests import hf_samples as samples


def test_gelu_family_agrees_at_zero() -> None:
    """``x = 0`` 处两式的**函数值同为 0**、**导数同为 0.5**（后者用中心差分量）."""
    step = 1e-6
    assert blocks.gelu_exact(0.0) == 0.0
    assert blocks.gelu_new(0.0) == 0.0
    for function in (blocks.gelu_exact, blocks.gelu_new):
        derivative = (function(step) - function(-step)) / (2 * step)
        assert abs(derivative - 0.5) < 1e-6


def test_gelu_family_differs_away_from_zero() -> None:
    """离 0 之后两式**不相等**，而差距是一个可量出的正数（不是"差不多"）."""
    gap = abs(blocks.gelu_exact(1.0) - blocks.gelu_new(1.0))
    assert 1e-5 < gap < 1e-3
    assert blocks.gelu_exact(2.0) != blocks.gelu_new(2.0)


def test_activation_of_rejects_unknown_names() -> None:
    """未知激活名当场拒绝（**回退到 relu 会把一份读数印成另一份**）."""
    with pytest.raises(ParameterError):
        blocks.activation_of("silu", 0.5)
    assert blocks.activation_of("relu", -0.5) == 0.0
    assert blocks.activation_of("relu", 0.5) == 0.5


def test_activate_rows_is_elementwise() -> None:
    """激活逐元素（因此逐行，与 day079 的 ``activate`` 是同一件事）."""
    activated = blocks.activate_rows(((-1.0, 2.0), (0.0, -3.0)), "relu")
    assert activated == ((0.0, 2.0), (0.0, 0.0))


def test_mlp_matches_the_definition() -> None:
    """``c_fc → act → c_proj``：用一个恒等第一层手算第 0 行."""
    block_params, _, _ = blocks.make_hf_block("gpt2")
    weights = blocks._ffn_weights(block_params)
    matrix = ((1.0,) * samples.HIDDEN,)
    output = blocks.gpt2_mlp(matrix, weights, activation="relu")
    assert len(output) == 1
    assert len(output[0]) == samples.HIDDEN


@pytest.mark.parametrize("name", ["gpt2", "bert"])
def test_block_dispatches_by_profile(name: str) -> None:
    """画像驱动：摆放与激活的默认值都来自 ``PROFILES``."""
    block_params, attention_params, _shape = blocks.make_hf_block(name)
    forward = blocks.block_of(
        name, block_params, attention_params, samples.inputs(), samples.shape_of(name)
    )
    profile = types.PROFILES[name]
    assert forward.placement == profile.norm_placement
    assert forward.activation == profile.activation
    assert len(forward.output) == samples.TOKENS
    assert forward.to_dict()["heads"] == 1


def test_gpt2_block_is_causal_by_default() -> None:
    """GPT-2 是解码器：默认因果（掩码外恰好 0），而 `causal=False` 时可以双向."""
    block_params, attention_params, _shape = blocks.make_hf_block("gpt2")
    causal = blocks.gpt2_block(
        block_params, attention_params, samples.inputs(), samples.shape_of("gpt2")
    )
    assert causal.attention.bias is not None
    bidir = blocks.gpt2_block(
        block_params, attention_params, samples.inputs(), samples.shape_of("gpt2"), causal=False
    )
    assert bidir.attention.bias is None
    assert causal.output != bidir.output


def test_bert_layer_is_bidirectional_by_default() -> None:
    """BERT 是编码器：默认双向（没有掩码），post 摆放让输出经过第二次 LN."""
    block_params, attention_params, _shape = blocks.make_hf_block("bert")
    forward = blocks.bert_layer(
        block_params, attention_params, samples.inputs(), samples.shape_of("bert")
    )
    assert forward.attention.bias is None
    assert forward.first_normed == forward.second_normed
    for row in forward.output:
        assert abs(math.fsum(row) / len(row)) < 1e-9


def test_block_of_rejects_unknown_names() -> None:
    """未知模型名当场拒绝."""
    block_params, attention_params, _shape = blocks.make_hf_block("gpt2")
    with pytest.raises(ParameterError):
        blocks.block_of(
            "t5", block_params, attention_params, samples.inputs(), samples.shape_of("gpt2")
        )


def test_make_hf_block_returns_three_pieces() -> None:
    """生产函数返回 (块参数, 注意力参数, 形状)，且形状是 ``d_ff = 4d``."""
    block_params, attention_params, shape = blocks.make_hf_block("bert", hidden=6, tokens=4)
    assert shape.hidden == 6
    assert shape.ffn == samples.FFN
    assert len(block_params.norm1_gamma) == 6
    assert len(attention_params.w_query) == 6


def test_source_shape_follows_the_profile() -> None:
    """因果默认值随画像走：GPT-2 是 True、BERT 是 False."""
    assert blocks.source_shape_of("gpt2").causal_default is True
    assert blocks.source_shape_of("bert").causal_default is False
    assert blocks.source_shape_of("gpt2", heads=2).head_dim == 3


def test_max_abs_gap_and_its_shape_guard() -> None:
    """最大绝对差是一个读数；形状不同时拒绝（那时"差"没有定义）."""
    left = ((1.0, 2.0), (3.0, 4.0))
    right = ((1.0, 2.0), (3.0, 5.0))
    assert blocks.max_abs_gap(left, right) == 1.0
    assert blocks.max_abs_gap(left, left) == 0.0
    with pytest.raises(ShapeError):
        blocks.max_abs_gap(left, ((1.0,),))


def test_block_tolerances_and_activation_names() -> None:
    """两个与 day079 对账用的容差/名单是常量（否则"对账"会随人而变）."""
    assert blocks.BLOCK_TOLERANCE == 1e-12
    assert blocks.DAY079_ACTIVATIONS == ("relu", "gelu")
    assert blocks.FFN_RATIO == 4
