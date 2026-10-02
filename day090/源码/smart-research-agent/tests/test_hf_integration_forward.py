"""``hf_integration.forward`` 与 ``features``：一次真前向与一次池化（day086）."""

from __future__ import annotations

import pytest

from smart_research_agent.hf_integration import errors, features, forward
from smart_research_agent.hf_integration.forward import (
    BIAS_ITEMS_PER_LAYER,
    LAYER_CONFIG_SCALE,
    LAYER_WEIGHTS_SCALE,
    POLICIES,
    POLICY_NAIVE,
    POLICY_SAFE,
    bias_breakdown_line,
    embedding_norm_line,
    hidden_states,
    logits,
    make_logits_fn,
    make_weights,
    pooler_output,
)
from smart_research_agent.hf_integration.config import tiny_card
from smart_research_agent.hf_integration.types import (
    ARCHITECTURE_BERT,
    ARCHITECTURE_GPT2,
    POOLING_LAST_TOKEN,
    POOLING_MAX,
    POOLING_MEAN,
    POOLING_STRATEGIES,
)

from tests.hf_integration_samples import build_case


# --------------------------------------------------------------------------- 权重


def test_make_weights_shapes_and_counters() -> None:
    """权重的每一块形状与配置一致，且两个参数量口径都印得出来."""
    case = build_case("gpt2")
    weights = case.weights
    assert len(weights.word) == case.card.vocab
    assert len(weights.position) == case.card.positions
    assert weights.token_type is None
    assert weights.pooler_w is None
    assert len(weights.layers) == case.card.layers
    assert weights.to_dict()["gap_is_explained"] is True
    assert weights.to_dict()["tied"] is True
    assert weights.hidden == case.card.hidden


def test_tying_shares_the_very_same_object() -> None:
    """共享不是"再复制一份"：``head`` 与 ``word`` 是**同一个对象**."""
    case = build_case("gpt2")
    assert case.weights.tied is True
    assert case.weights.head is case.weights.word


def test_untied_head_is_a_separate_matrix() -> None:
    """不共享时另有一份输出投影（`.card.tie_word_embeddings` 是唯一的开关）."""
    case = build_case("gpt2", tie_word_embeddings=False)
    assert case.weights.tied is False
    assert case.weights.head is not case.weights.word
    assert len(case.weights.head) == case.card.vocab


def test_bert_weights_carry_three_extra_pieces() -> None:
    """BERT 侧多三样：``token_type`` 表、嵌入层那次 LN 的 γ/β、pooler."""
    case = build_case("bert")
    assert case.weights.token_type is not None
    assert len(case.weights.token_type) == case.card.type_vocab
    assert case.weights.embed_gamma is not None
    assert case.weights.embed_beta is not None
    assert case.weights.pooler_w is not None
    assert case.weights.pooler_b is not None
    assert embedding_norm_line(case.card, case.weights).startswith(ARCHITECTURE_BERT)


def test_layers_are_not_copies_of_each_other() -> None:
    """层与层必须不同（"所有层完全一样"这种错误会让读数看起来正常）."""
    case = build_case("gpt2", n_layer=3)
    assert case.card.layers == 3
    first, _ = case.weights.layer_of(0)
    second, _ = case.weights.layer_of(1)
    assert first.ffn_w_in != second.ffn_w_in


def test_layer_lookup_rejects_out_of_range() -> None:
    """层号越界当场拒绝（消息里带上配置里写了多少层）."""
    case = build_case("gpt2")
    with pytest.raises(errors.ShapeError, match="越界"):
        case.weights.layer_of(99)


def test_same_seed_gives_the_same_weights() -> None:
    """同一个种子给出**逐位相同**的权重（"可复现"是一条能被断言的结论）."""
    card = tiny_card(ARCHITECTURE_GPT2, vocab=64, hidden=8, heads=2, layers=1, positions=16)
    assert make_weights(card, seed=11).word == make_weights(card, seed=11).word
    assert make_weights(card, seed=11).word != make_weights(card, seed=12).word


def test_matrix_factory_rejects_bad_shapes() -> None:
    """矩阵工厂的两条护栏：形状必须为正、幅度必须是正的有限数."""
    with pytest.raises(errors.ShapeError, match="形状必须为正"):
        forward._matrix(0, 4, seed=1)
    with pytest.raises(errors.ShapeError, match="scale"):
        forward._matrix(2, 2, seed=1, scale=0.0)
    with pytest.raises(errors.ShapeError, match="scale"):
        forward._matrix(2, 2, seed=1, scale=float("nan"))


# --------------------------------------------------------------------------- 参数量差


@pytest.mark.parametrize("model_type", [ARCHITECTURE_GPT2, ARCHITECTURE_BERT])
def test_parameter_gap_is_exactly_the_attention_bias(model_type: str) -> None:
    """配置口径与权重之差**恰好**是 ``4·hidden·layers``（一个能被拆开的整数）."""
    case = build_case(model_type)
    weights, card = case.weights, case.card
    assert card.parameter_count - weights.parameter_count == weights.expected_gap
    assert weights.expected_gap == BIAS_ITEMS_PER_LAYER * card.hidden * card.layers
    assert weights.gap_is_explained is True
    line = bias_breakdown_line(card, weights)
    assert "成立：True" in line


def test_layer_scale_constants_are_recorded() -> None:
    """两个口径的每层系数写进常量（``12h²+13h`` 与 ``12h²+9h``），而且能对上公式."""
    hidden, layers = 16, 2
    config_scale, config_linear = LAYER_CONFIG_SCALE
    weights_scale, weights_linear = LAYER_WEIGHTS_SCALE
    assert config_scale == weights_scale == 12
    assert config_linear - weights_linear == BIAS_ITEMS_PER_LAYER
    assert (config_scale - weights_scale) * hidden * hidden + (
        config_linear - weights_linear
    ) * hidden == 4 * hidden
    assert BIAS_ITEMS_PER_LAYER * hidden * layers == 128


def test_bias_breakdown_differs_between_the_two_architectures() -> None:
    """同一个 ``4h`` 的**来源**在两个架构里不同（一个融合投影分成两段，一个四次独立）."""
    assert "c_attn" in forward.BIAS_BREAKDOWN[ARCHITECTURE_GPT2]
    assert "query" in forward.BIAS_BREAKDOWN[ARCHITECTURE_BERT]
    assert forward.BIAS_BREAKDOWN[ARCHITECTURE_GPT2] != forward.BIAS_BREAKDOWN[ARCHITECTURE_BERT]


# --------------------------------------------------------------------------- 前向


def test_hidden_states_keeps_only_real_rows() -> None:
    """每条样本只保留**真实长度**的那几行（填充行存在语义问题）."""
    case = build_case("gpt2")
    ids = [case.tokenizer.encode(text) for text in case.texts]
    batch = hidden_states(case.card, case.weights, tuple(tuple(row) for row in ids))
    assert batch.lengths() == tuple(len(row) for row in ids)
    assert batch.batch_size == len(case.texts)
    assert batch.hidden == case.card.hidden
    assert len(batch.padded()) == sum(len(row) for row in ids)
    assert batch.to_dict()["policy"] == POLICY_SAFE


def test_causal_padding_is_inert_but_sample_borders_are_not() -> None:
    """因果侧：``safe`` 与逐条**逐位相同**；而 naive 会在**样本边界**上串味.

    这是本课最值钱的一条工程结论：批量之后"因果"不再是一张 ``(n, n)`` 的下三角，
    而必须升级成**段内下三角**（:func:`block_causal_mask`）。
    漏掉那一步的后果不是报错，而是第 2 条样本的输出里混进了第 1 条的内容。
    """
    case = build_case("gpt2")
    ids = [case.tokenizer.encode(text) for text in case.texts]
    width = max(len(row) for row in ids)
    rows = tuple(tuple(row) + (0,) * (width - len(row)) for row in ids)
    masks = tuple((1,) * len(row) + (0,) * (width - len(row)) for row in ids)
    batch = hidden_states(case.card, case.weights, rows, masks)
    singles = tuple(
        hidden_states(case.card, case.weights, (row,), (mask,)).rows[0]
        for row, mask in zip(rows, masks, strict=True)
    )
    assert batch.rows == singles
    naive = hidden_states(case.card, case.weights, rows, masks, policy=POLICY_NAIVE)
    assert naive.padded() != batch.padded()
    assert features.max_gap(batch.padded(), naive.padded()) > 0.0
    # 第 1 条（段内第一条）在 naive 下**碰巧是对的**——这正是它难被发现的原因
    assert naive.rows[0] == batch.rows[0]
    assert naive.rows[1] != batch.rows[1]


def test_block_causal_mask_structure() -> None:
    """段内下三角：三条各 2 长的样本，允许集合是一张 6×6 的块对角下三角."""
    mask = forward.block_causal_mask((2, 2, 2))
    assert len(mask) == 6
    assert all(len(row) == 6 for row in mask)
    assert mask[1] == (True, True, False, False, False, False)
    assert mask[2] == (False, False, True, False, False, False)
    assert mask[5] == (False, False, False, False, True, True)
    for index, row in enumerate(mask):
        assert sum(row) <= 2
        assert row[index] is True


def test_block_causal_mask_rejects_bad_lengths() -> None:
    """空长度表与零长度的段都要拒绝（零长度的段没有掩码可描述）."""
    with pytest.raises(errors.ShapeError, match="至少一条"):
        forward.block_causal_mask(())
    with pytest.raises(errors.ShapeError, match=">= 1"):
        forward.block_causal_mask((2, 0))


def test_block_matches_day085_for_a_single_causal_sample() -> None:
    """单条样本、``L=1``、``heads=1``：本包的一层与 day085 的 ``gpt2_block`` **逐位**相同.

    两条路径的差别只有一处：day085 造一张加性偏置，本包给一张显式掩码。
    它们必须在**每一个浮点数**上一致——这才是"接线没有走散"的证据。
    """
    from smart_research_agent.hf_integration.forward import embed_tokens, run_one_block
    from smart_research_agent.hf_source.blocks import gpt2_block
    from smart_research_agent.hf_source.types import SourceShape

    case = build_case("gpt2", n_layer=1, n_head=1)
    ids = tuple(case.tokenizer.encode(case.prompt))
    states = embed_tokens(case.card, case.weights, (ids,))
    mine = run_one_block(case.card, case.weights, 0, states, causal=True)
    block, attention = case.weights.layer_of(0)
    shape = SourceShape(
        tokens=len(ids),
        hidden=case.card.hidden,
        heads=1,
        vocab=case.card.vocab,
        causal_default=True,
    )
    reference = gpt2_block(
        block, attention, states, shape, activation=case.card.activation, causal=True
    ).output
    assert mine == reference


def test_bidirectional_padding_is_not_inert() -> None:
    """双向侧：``naive`` 与 ``safe`` 的差**不为 0**——填充真的改写了真实位置那一行."""
    case = build_case("bert")
    ids = [case.tokenizer.encode(text) for text in case.texts]
    width = max(len(row) for row in ids)
    rows = tuple(tuple(row) + (0,) * (width - len(row)) for row in ids)
    masks = tuple((1,) * len(row) + (0,) * (width - len(row)) for row in ids)
    safe = hidden_states(case.card, case.weights, rows, masks)
    naive = hidden_states(case.card, case.weights, rows, masks, policy=POLICY_NAIVE)
    assert safe.rows != naive.rows
    assert features.max_gap(safe.padded(), naive.padded()) > 0.0
    assert safe.batched is False
    assert naive.batched is True

def test_unknown_policy_is_rejected() -> None:
    """未知的填充策略当场拒绝（可选值就两个）."""
    case = build_case("gpt2")
    with pytest.raises(errors.ShapeError, match="填充策略"):
        hidden_states(case.card, case.weights, ((1, 2),), policy="fast")
    assert POLICIES == (POLICY_SAFE, POLICY_NAIVE)


@pytest.mark.parametrize(
    ("ids", "masks", "match"),
    [
        ((), None, "至少要有一条样本"),
        (((),), None, "是空的"),
        (((1, 2),), ((1,),), "掩码长度"),
        (((1, 2),), ((1, 1), (1, 1)), "行数不一致"),
        (((1, 2),), ((0, 0),), "全是 0"),
        (((1, 0, 0),), ((0, 1, 0),), "空洞"),
    ],
)
def test_bad_batches_are_rejected(
    ids: object, masks: object, match: str
) -> None:
    """六类坏批次：空批、空行、掩码长度、行数、全 0、空洞."""
    case = build_case("gpt2")
    with pytest.raises((errors.ShapeError, errors.NumericError), match=match):
        hidden_states(case.card, case.weights, ids, masks)  # type: ignore[arg-type]


def test_id_and_position_bounds_are_enforced() -> None:
    """两条硬上界：id 必须在词表里、长度不能超过位置表."""
    case = build_case("gpt2")
    with pytest.raises(errors.TokenError, match="不在词表里"):
        hidden_states(case.card, case.weights, ((case.card.vocab + 1,),))
    with pytest.raises(errors.TokenError, match="不在词表里"):
        hidden_states(case.card, case.weights, ((-1,),))
    too_long = tuple(range(case.card.positions + 1))
    with pytest.raises(errors.ShapeError, match="位置表"):
        hidden_states(case.card, case.weights, (too_long,))


def test_mask_is_optional() -> None:
    """不给掩码时按行长推出"全是真实位"（长度相同的那一批不需要掩码）."""
    case = build_case("gpt2")
    batch = hidden_states(case.card, case.weights, ((1, 2), (3, 4)))
    assert batch.lengths() == (2, 2)


def test_forward_is_deterministic() -> None:
    """同样输入 + 同样权重 ⇒ 同样输出（前向里没有随机性）."""
    case = build_case("bert")
    first = hidden_states(case.card, case.weights, ((1, 2, 3),))
    second = hidden_states(case.card, case.weights, ((1, 2, 3),))
    assert first.rows == second.rows


def test_logits_have_vocab_width() -> None:
    """logits 的宽度恰好是词表大小，而且与末位 hidden state 的点积一致."""
    case = build_case("gpt2")
    values = logits(case.card, case.weights, (1, 2, 3))
    assert len(values) == case.card.vocab
    batch = hidden_states(case.card, case.weights, ((1, 2, 3),))
    last = batch.rows[0][-1]
    import math

    expected = math.fsum(a * b for a, b in zip(last, case.weights.head[0], strict=True))
    assert values[0] == expected


def test_logits_requires_at_least_one_token() -> None:
    """空序列没有末位（logits 需要至少一个 token）."""
    case = build_case("gpt2")
    with pytest.raises(errors.ShapeError, match="至少一个 token"):
        logits(case.card, case.weights, ())


def test_logits_fn_is_just_a_wrapper() -> None:
    """那个"模型"就是一个闭包：同样的输入给同样的 logits."""
    case = build_case("gpt2")
    logits_fn = make_logits_fn(case.card, case.weights)
    assert logits_fn((1, 2)) == logits(case.card, case.weights, (1, 2))


def test_pooler_output_only_exists_on_bert() -> None:
    """``pooler_output`` 只在 BERT 侧存在（GPT-2 没有 pooler）."""
    bert = build_case("bert")
    batch = hidden_states(bert.card, bert.weights, ((1, 2, 3),))
    pooled = pooler_output(bert.weights, batch)
    assert len(pooled) == 1
    assert len(pooled[0]) == bert.card.hidden
    assert all(-1.0 < value < 1.0 for value in pooled[0])
    gpt2 = build_case("gpt2")
    with pytest.raises(errors.TokenError, match="pooler"):
        pooler_output(
            gpt2.weights, hidden_states(gpt2.card, gpt2.weights, ((1, 2, 3),))
        )


# --------------------------------------------------------------------------- 池化


def test_pool_rows_three_strategies_by_hand() -> None:
    """三法各自取什么（用一张能手算的小矩阵核对）."""
    rows = (
        ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)),
        ((0.0, 0.0), (2.0, 2.0)),
    )
    assert features.pool_rows(rows, POOLING_MEAN) == ((3.0, 4.0), (1.0, 1.0))
    assert features.pool_rows(rows, POOLING_MAX) == ((5.0, 6.0), (2.0, 2.0))
    assert features.pool_rows(rows, POOLING_LAST_TOKEN) == ((5.0, 6.0), (2.0, 2.0))


def test_pool_rows_rejects_bad_inputs() -> None:
    """五类坏输入：未知策略、零行、零宽、空行、宽度不齐、非有限数."""
    with pytest.raises(errors.ParameterError, match="池化策略"):
        features.pool_rows((((1.0,),),), "median")
    with pytest.raises(errors.ShapeError, match="0 行"):
        features.pool_rows((), POOLING_MEAN)
    with pytest.raises(errors.ShapeError, match="空向量"):
        features.pool_rows(((),), POOLING_MEAN)
    with pytest.raises(errors.NumericError, match="没有位置"):
        features.pool_rows((((1.0,),), ()), POOLING_MEAN)
    with pytest.raises(errors.ShapeError, match="宽度"):
        features.pool_rows((((1.0, 2.0),), ((1.0,),)), POOLING_MEAN)
    with pytest.raises(errors.NumericError, match="非有限数"):
        features.pool_rows((((1.0, float("inf")),),), POOLING_MEAN)


def test_pool_forward_returns_a_pooled_batch() -> None:
    """推荐入口：给一次前向的结果，拿一组向量."""
    case = build_case("bert")
    batch = hidden_states(case.card, case.weights, ((1, 2, 3), (4, 5)))
    pooled = features.pool_forward(batch, POOLING_MEAN)
    assert pooled.batch_size == 2
    assert pooled.dim == case.card.hidden
    assert pooled.strategy == POOLING_MEAN
    assert pooled.to_dict()["norms"][0] > 0.0
    normalized = features.pool_forward(batch, POOLING_MEAN, normalize_vectors=True)
    assert normalized.normalized is True
    import math

    assert math.isclose(
        math.sqrt(math.fsum(v * v for v in normalized.vectors[0])), 1.0, rel_tol=1e-12
    )


def test_pool_masked_correct_versus_buggy() -> None:
    """同一个分母之差：正确版只对真实位置求和，违反版把填充也算进去."""
    states = ((1.0, 1.0), (2.0, 2.0), (0.0, 0.0), (0.0, 0.0))
    mask = (1, 1, 0, 0)
    correct, evidence = features.pool_masked(states, mask, POOLING_MEAN, width=4)
    buggy, _ = features.pool_masked(states, mask, POOLING_MEAN, width=4, ignore_mask=True)
    assert correct == ((1.5, 1.5),)
    assert buggy == ((0.75, 0.75),)
    assert correct != buggy
    assert "填充 2 格" in evidence[0]


def test_pool_masked_rejects_bad_shapes() -> None:
    """四类坏输入：非正宽度、除不尽、掩码长度不符、全 0 掩码."""
    states = ((1.0,), (2.0,))
    with pytest.raises(errors.ParameterError, match="width"):
        features.pool_masked(states, (1, 1), POOLING_MEAN, width=0)
    with pytest.raises(errors.ShapeError, match="整除"):
        features.pool_masked(states, (1, 1), POOLING_MEAN, width=3)
    with pytest.raises(errors.ShapeError, match="逐行对齐"):
        features.pool_masked(states, (), POOLING_MEAN, width=1)
    with pytest.raises(errors.NumericError, match="全是 0"):
        features.pool_masked(((1.0,),), (0,), POOLING_MEAN, width=1)


def test_compare_pooling_reports_the_cost() -> None:
    """对照记录：填充占比、违反 mask 的代价、两版是否相同.

    填充那一行刻意取一个**比真实行都大**的值：这样三种策略都会真的被它带偏
    （若取 0，``max`` 恰好不受影响——那会让"这一族的代价"在一种策略上变成 0，
    而"它没问题"与"这次样本碰巧没暴露它"读起来一样）。
    """
    states = ((1.0, 1.0), (3.0, 3.0), (5.0, 5.0), (9.0, 9.0))
    mask = (1, 1, 1, 0)
    for strategy in POOLING_STRATEGIES:
        comparison = features.compare_pooling(states, mask, strategy, width=4)
        assert comparison.padding_waste == 0.25
        assert comparison.identical is False
        assert comparison.gap > 0.0
        assert "违反 mask 的代价" in comparison.line()


def test_compare_pooling_identical_without_padding() -> None:
    """**没有填充时两版逐位相同**——这正是"这个 bug 只在有填充时出现"的证据."""
    states = ((1.0, 1.0), (3.0, 3.0))
    mask = (1, 1)
    comparison = features.compare_pooling(states, mask, POOLING_MEAN, width=2)
    assert comparison.padding_waste == 0.0
    assert comparison.identical is True
    assert comparison.gap == 0.0


def test_l2_normalize_and_cosine() -> None:
    """归一化与余弦相似度：自己与自己为 1，正交为 0，零向量当场拒绝."""
    normalized = features.l2_normalize(((3.0, 4.0),))
    assert normalized == ((0.6, 0.8),)
    assert features.cosine_similarity((1.0, 0.0), (1.0, 0.0)) == 1.0
    assert features.cosine_similarity((1.0, 0.0), (0.0, 1.0)) == 0.0
    with pytest.raises(errors.NumericError, match="全零"):
        features.l2_normalize(((0.0, 0.0),))
    with pytest.raises(errors.NumericError, match="全零"):
        features.cosine_similarity((0.0, 0.0), (1.0, 0.0))
    with pytest.raises(errors.ShapeError, match="长度不同"):
        features.cosine_similarity((1.0,), (1.0, 0.0))


def test_max_gap_checks_shapes() -> None:
    """最大绝对差的两条形状护栏（行数、列数各一条）."""
    assert features.max_gap(((1.0,),), ((1.5,),)) == 0.5
    with pytest.raises(errors.ShapeError, match="行数"):
        features.max_gap(((1.0,),), ((1.0,), (2.0,)))
    with pytest.raises(errors.ShapeError, match="列数"):
        features.max_gap(((1.0,),), ((1.0, 2.0),))


def test_pooling_line_and_tolerance() -> None:
    """两条读数入口：一句话说明规矩、一个容差."""
    for strategy in POOLING_STRATEGIES:
        assert strategy in features.pooling_line(strategy)
        assert features.tolerance_of(strategy) == features.POOLING_TOLERANCE
    with pytest.raises(errors.ParameterError):
        features.pooling_line("median")
    assert set(features.POOLING_RULES) == set(POOLING_STRATEGIES)
