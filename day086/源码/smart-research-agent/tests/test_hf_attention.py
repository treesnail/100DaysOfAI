"""``hf_source.attention``：投影内部分头与加性掩码（day085 / M7-D9）."""

from __future__ import annotations

import math

import pytest

from smart_research_agent.hf_source import attention, types
from smart_research_agent.hf_source.errors import AssemblyError, ParameterError, ShapeError
from smart_research_agent.math_foundations.linalg import matmul, transpose
from tests import hf_samples as samples


def params():
    """一份写死的注意力参数（委托 day075 的生产函数）."""
    return samples.attention_parameters()


def shape(heads: int = 1) -> types.SourceShape:
    """写死的形状（``tokens=4、hidden=6``）."""
    return types.SourceShape(tokens=samples.TOKENS, hidden=samples.HIDDEN, heads=heads, vocab=8)


def test_scale_is_the_reciprocal_square_root() -> None:
    """``1/√head_dim``：``head_dim=4`` 时恰好是 0.5（**可手算**）."""
    assert attention.hf_scale(4) == 0.5
    assert attention.hf_scale(1) == 1.0
    assert math.isclose(attention.hf_scale(3), 1.0 / math.sqrt(3.0), rel_tol=0.0)


def test_scale_rejects_non_positive() -> None:
    """``head_dim`` 必须为正（否则 1/0 会变成 inf 并一路污染）。"""
    for value in (0, -2):
        with pytest.raises(ParameterError):
            attention.hf_scale(value)


def test_resolve_heads_rejects_zero_and_indivisible() -> None:
    """头数为 0 是参数错、不能整除是形状错——**两族的修法不同**."""
    with pytest.raises(ParameterError):
        attention.resolve_heads(shape(heads=0))
    with pytest.raises(ShapeError):
        attention.resolve_heads(shape(heads=4))
    assert attention.resolve_heads(shape(heads=2)) == 2
    assert attention.resolve_heads(shape(heads=2)) == 2


def test_fused_weight_concatenates_rows_in_order() -> None:
    """融合投影就是 q/k/v 三块**按行拼接**（拼的顺序就是切开的顺序）."""
    current = params()
    fused = attention.fused_weight(current.w_query, current.w_key, current.w_value)
    assert len(fused) == 3 * samples.HIDDEN
    assert attention.split_fused(fused, samples.HIDDEN) == (
        current.w_query,
        current.w_key,
        current.w_value,
    )


def test_fused_weight_rejects_mismatched_shapes() -> None:
    """三块的行数与列数都必须相同（否则拼起来的两半来自两个空间）."""
    current = params()
    with pytest.raises(AssemblyError):
        attention.fused_weight(current.w_query[:-1], current.w_key, current.w_value)
    skinny = tuple(row[:-1] for row in current.w_key)
    with pytest.raises(AssemblyError):
        attention.fused_weight(current.w_query, skinny, current.w_value)


def test_split_fused_rejects_wrong_total_width() -> None:
    """总行数必须**恰好**是 ``3·hidden``（多了少了都不接受）."""
    current = params()
    fused = attention.fused_weight(current.w_query, current.w_key, current.w_value)
    with pytest.raises(ShapeError):
        attention.split_fused(fused[:-1], samples.HIDDEN)
    with pytest.raises(ParameterError):
        attention.split_fused(fused, 0)


def test_project_matches_the_definition() -> None:
    """投影就是 ``x·Wᵀ``（这条口径从 day075 起只有一条）."""
    matrix = samples.inputs()
    weight = params().w_query
    assert attention.project(matrix, weight) == matmul(matrix, transpose(weight))


def test_add_bias_is_identity_without_bias() -> None:
    """``bias=None`` 时是恒等；给错长度当场拒绝."""
    matrix = samples.inputs()
    assert attention.add_bias(matrix, None) == matrix
    added = attention.add_bias(matrix, tuple([1.0] * samples.HIDDEN))
    assert added[0][0] == matrix[0][0] + 1.0
    with pytest.raises(ShapeError):
        attention.add_bias(matrix, (1.0, 2.0))


def test_split_heads_cuts_columns_and_merges_back() -> None:
    """分头按 ``head_dim`` 切列，拼回必须**逐位**还原（分头只是记账）."""
    matrix = samples.inputs()
    heads = attention.split_heads(matrix, 3)
    assert len(heads) == 3
    assert all(len(head) == samples.TOKENS and len(head[0]) == 2 for head in heads)
    assert attention.merge_heads(heads) == matrix
    assert heads[0][0] == matrix[0][0:2]
    assert heads[1][0] == matrix[0][2:4]


def test_split_heads_rejects_bad_head_counts() -> None:
    """宽度不能被头数整除时在入口拒绝（HF 会在 reshape 时才炸）."""
    with pytest.raises(ParameterError):
        attention.split_heads(samples.inputs(), 0)
    with pytest.raises(ShapeError):
        attention.split_heads(samples.inputs(), 4)


def test_merge_heads_rejects_empty_and_mismatched_groups() -> None:
    """空组、空头、形状不一致三种都拒绝（形状不一致会被 zip 静默截断）."""
    with pytest.raises(ShapeError):
        attention.merge_heads(())
    with pytest.raises(ShapeError):
        attention.merge_heads((((),),))
    wide = tuple(tuple([0.0] * (samples.HIDDEN)) for _ in range(samples.TOKENS))
    narrow = tuple(tuple([0.0] * 2) for _ in range(samples.TOKENS))
    with pytest.raises(ShapeError):
        attention.merge_heads((narrow, wide))


def test_additive_bias_translates_a_mask() -> None:
    """显式掩码翻成 ``0.0 / -floor``，且**对角线永远允许**（用 ``causal_bias`` 造一张）."""
    bias = attention.additive_bias_of(
        ((True, False), (True, True)),
    )
    assert bias == ((0.0, -attention.BIAS_FLOOR), (0.0, 0.0))


def test_additive_bias_rejects_an_empty_row_set() -> None:
    """整张掩码全 False 时拒绝（softmax 的分母会是 0）."""
    with pytest.raises(AssemblyError):
        attention.additive_bias_of(((False, False), (False, False)))


def test_causal_bias_is_upper_triangular_negative() -> None:
    """因果加性偏置：上三角是 ``-floor``、其余是 0（行列搞反时形状完全合法）."""
    bias = attention.causal_bias(3)
    assert bias[0] == (0.0, -attention.BIAS_FLOOR, -attention.BIAS_FLOOR)
    assert bias[2] == (0.0, 0.0, 0.0)
    with pytest.raises(ParameterError):
        attention.causal_bias(0)


def test_row_softmax_matches_the_masked_reference() -> None:
    """带加性偏置的 softmax 与 day073 的 ``masked_softmax_rows`` **逐位相同**.

    这是"``-1e9`` 与 ``-inf`` 在这一步等价"的判据：被挡格子的 ``exp`` 直接下溢到 0。
    """
    from smart_research_agent.math_foundations.attention import masked_softmax_rows

    scores = samples.inputs()
    mask = tuple(
        tuple(column <= row for column in range(samples.HIDDEN)) for row in range(samples.TOKENS)
    )
    bias = attention.additive_bias_of(mask)
    assert attention.row_softmax(scores, bias) == masked_softmax_rows(scores, mask)


def test_row_softmax_rejects_degenerate_rows() -> None:
    """全 ``-inf`` 的行与偏置宽度不符的行都拒绝（HF 会在那里给出 nan 或静默截断）."""
    with pytest.raises(AssemblyError):
        attention.row_softmax(((float("-inf"), float("-inf")),))
    with pytest.raises(ShapeError):
        attention.row_softmax(samples.inputs(), ((0.0, 0.0),) * samples.TOKENS)


def test_score_magnitude_guard_rejects_non_finite_and_huge() -> None:
    """"加性掩码 = 显式掩码"有前提：打分必须远离 ``-floor``."""
    with pytest.raises(AssemblyError):
        attention.check_score_magnitude(((float("nan"), 0.0),))
    with pytest.raises(AssemblyError):
        attention.check_score_magnitude(((attention.BIAS_FLOOR, 0.0),))
    attention.check_score_magnitude(((1.0, -1.0),))


def test_attention_rejects_causal_plus_explicit_mask() -> None:
    """``causal=True`` 与显式掩码**不能同时给**（谁生效取决于实现顺序）."""
    mask = ((True,) * samples.TOKENS,) * samples.TOKENS
    with pytest.raises(AssemblyError):
        attention.hf_attention(params(), samples.inputs(), shape(), causal=True, mask=mask)


def test_attention_rejects_shape_mismatch() -> None:
    """输入的行数与宽度必须与形状一致."""
    with pytest.raises(ShapeError):
        attention.hf_attention(params(), samples.inputs(tokens=3), shape())
    narrow = tuple(row[:-1] for row in samples.inputs())
    with pytest.raises(ShapeError):
        attention.hf_attention(params(), narrow, shape())


def test_attention_masked_entries_are_exactly_zero() -> None:
    """加性掩码挡掉的格子**逐位是 0.0**（不是"很小"）."""
    forward = attention.hf_attention(params(), samples.inputs(), shape(), causal=True)
    blocked = 0
    for head in forward.weights:
        for index, row in enumerate(head):
            for column, value in enumerate(row):
                if column > index:
                    blocked += 1
                    assert value == 0.0
    assert blocked == 6


def test_attention_rows_are_distributions() -> None:
    """每一行的和是 1（4 行 × 1 头）."""
    forward = attention.hf_attention(params(), samples.inputs(), shape())
    assert forward.bias is None
    for row in forward.weights[0]:
        assert abs(math.fsum(row) - 1.0) < 1e-12


def test_attention_heads_change_the_output() -> None:
    """``heads=1`` 与 ``heads=2`` 的输出不同（分头不是记账，它改变了投影划分）."""
    single = attention.hf_attention(params(), samples.inputs(), shape(heads=1))
    double = attention.hf_attention(params(), samples.inputs(), shape(heads=2))
    assert single.output != double.output
    assert len(double.head_queries) == 2


def test_average_weights_and_to_dict() -> None:
    """两头的平均权重仍是一张 (n, n) 的行随机表；摊平字段含派生量."""
    forward = attention.hf_attention(params(), samples.inputs(), shape(heads=2))
    averaged = forward.average_weights()
    assert len(averaged) == samples.TOKENS
    for row in averaged:
        assert abs(math.fsum(row) - 1.0) < 1e-12
    flattened = forward.to_dict()
    assert flattened["heads"] == 2
    assert flattened["head_dim"] == 3
    assert forward.depth == 2


@pytest.mark.parametrize(
    "mask",
    [
        "not-a-mask",
        (),
        ((), ()),
        ((True, False), (True,)),
        ((True, 0), (True, True)),
    ],
)
def test_checked_mask_rejects_malformed_masks(mask: object) -> None:
    """掩码的四种坏形状都在入口拒绝（**长度不齐会被 zip 静默截断**）."""
    with pytest.raises(ShapeError):
        attention.checked_mask(mask)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "scores",
    [
        42,
        (),
        ((),),
        ((1.0, 2.0), (3.0,)),
        ((1.0, True), (3.0, 4.0)),
        ((float("nan"), 2.0), (3.0, 4.0)),
    ],
)
def test_checked_scores_rejects_malformed_scores(scores: object) -> None:
    """打分的六种坏输入都在入口拒绝（**本包的族**，不是上游校验器的族）."""
    with pytest.raises((ShapeError, AssemblyError)):
        attention.checked_scores(scores)


def test_checked_scores_rejects_nan_as_assembly_failure() -> None:
    """非有限数归 **AssemblyError**：它是"这次拼出来的账不可用"，不是形状问题."""
    with pytest.raises(AssemblyError):
        attention.checked_scores(((float("inf"), 1.0),))
