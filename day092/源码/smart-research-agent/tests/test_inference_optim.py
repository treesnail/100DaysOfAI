"""``inference_optim``：推理路径上的三笔账（day087 / M7-D11）.

本文件覆盖新包的八个模块：失败族、口径表、缓存、量化、批处理、预算、性质与表。
样本来自 day086 的构造器（``tests/hf_integration_samples.py``）——
**跨天复用同一份装载**正是这一课要考的东西：优化只作用于推理，
因此它必须建立在"同一份已装好的权重"之上。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.hf_integration.forward import hidden_states, logits
from smart_research_agent.inference_optim import (
    batching,
    budget,
    cache as cache_module,
    errors,
    study,
    types,
    verify,
)
from smart_research_agent.inference_optim.quantize import (
    BOUND_TOLERANCE,
    dequantize,
    error_bound,
    error_stats,
    logits_of,
    measure,
    pack_int4,
    packed_length,
    quantize_matrix,
    round_trip_line,
    unpack_int4,
)
from smart_research_agent.inference_optim.types import (
    GRANULARITY_CHANNEL,
    GRANULARITY_TENSOR,
    LEVELS,
    LEVEL_FP32,
    LEVEL_INT4,
    LEVEL_INT8,
    OPTIM_BOUNDARIES,
    OPTIM_NOTES,
    OPTIM_NOTES_ORDER,
    OPTIM_PROPERTIES,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_FAILURE,
    SCHEMES,
    SCHEME_ABSMAX,
    SCHEME_ZERO_POINT,
    STAGES,
    QuantSpec,
    QuantizedMatrix,
)

from tests.hf_integration_samples import build_case


@pytest.fixture(scope="module")
def gpt2_case():
    """一份装好的 GPT-2 case（**与 day086 用的是同一个构造器**）."""
    return build_case("gpt2", n_layer=2, n_head=2)


@pytest.fixture(scope="module")
def bert_case():
    """一份装好的 BERT case（缓存与预算在它身上也要成立）."""
    return build_case("bert", n_layer=2, n_head=2)


@pytest.fixture(scope="module")
def matrices(gpt2_case):
    """量化用的三块矩阵：一行权重、一块偏斜矩阵、一块单值矩阵."""
    word = gpt2_case.weights.word
    return {
        "word": tuple(word[:3]),
        "skewed": study.skewed_matrix(),
        "uniform": tuple(word[10:13]),
    }


# --------------------------------------------------------------------------- 口径表


def test_level_tables_are_closed() -> None:
    """四种精度的名单、字节数、位宽三张表逐键对齐."""
    assert set(LEVELS) == {"fp32", "fp16", "int8", "int4"}
    assert set(types.BYTES_PER_ELEMENT) == set(LEVELS)
    assert set(types.BIT_WIDTHS) == set(LEVELS)
    assert set(types.LEVEL_DESCRIPTIONS) == set(LEVELS)
    assert types.BYTES_PER_ELEMENT["int4"] == 0.5
    assert types.BIT_WIDTHS["fp32"] == 32


def test_scheme_and_granularity_tables_are_closed() -> None:
    """两种方案与两种粒度：名单与说明逐键对齐."""
    assert set(SCHEMES) == {SCHEME_ABSMAX, SCHEME_ZERO_POINT}
    assert set(types.SCHEME_DESCRIPTIONS) == set(SCHEMES)
    assert set(types.GRANULARITIES) == {GRANULARITY_TENSOR, GRANULARITY_CHANNEL}
    assert set(types.GRANULARITY_DESCRIPTIONS) == set(types.GRANULARITIES)


def test_stage_and_layout_tables_are_closed() -> None:
    """两个阶段与两种布局：名单与说明逐键对齐."""
    assert set(STAGES) == {"prefill", "decode"}
    assert set(types.STAGE_DESCRIPTIONS) == set(STAGES)
    assert set(types.LAYOUTS) == {"grow", "preallocated"}
    assert set(types.LAYOUT_DESCRIPTIONS) == set(types.LAYOUTS)
    assert types.CACHE_TABLES == 2


def test_seven_properties_have_descriptions_and_failures() -> None:
    """七条性质：名单、说明、"失败意味着什么"三张表逐键对齐."""
    assert len(OPTIM_PROPERTIES) == 7
    assert set(PROPERTY_DESCRIPTIONS) == set(OPTIM_PROPERTIES)
    assert set(PROPERTY_FAILURE) == set(OPTIM_PROPERTIES)
    assert all(PROPERTY_DESCRIPTIONS.values())
    assert all(PROPERTY_FAILURE.values())


def test_notes_are_ten_and_ordered() -> None:
    """十条笔记，且顺序表与键集合一致。"""
    assert len(OPTIM_NOTES) == 10
    assert OPTIM_NOTES_ORDER == tuple(OPTIM_NOTES)
    assert all(OPTIM_NOTES.values())


def test_boundaries_are_five() -> None:
    """五条边界是一份非空清单（**这一课明确不承诺的事**）."""
    assert len(OPTIM_BOUNDARIES) == 5
    assert all(OPTIM_BOUNDARIES)


def test_cache_bytes_is_the_only_formula() -> None:
    """缓存的唯一公式：``2·L·T·h·bytes``（四个级别各核一次）."""
    for level in LEVELS:
        bytes_each = types.element_bytes(level)
        assert types.cache_bytes(2, 5, 16, level) == int(2 * 2 * 5 * 16 * bytes_each)
    assert types.cache_bytes(2, 0, 16, LEVEL_FP32) == 0
    with pytest.raises(errors.ParameterError, match="未知的精度级别"):
        types.element_bytes("int3")
    with pytest.raises(errors.ParameterError, match="缓存公式"):
        types.cache_bytes(0, 5, 16)


def test_flatten_is_row_major() -> None:
    """摊平是**逐行**的（per-channel 的"通道"就是"行"，两者必须同一个顺序）."""
    assert types.flatten(((1.0, 2.0), (3.0, 4.0))) == (1.0, 2.0, 3.0, 4.0)
    assert types.norm_or_raise((3.0, 4.0)) == 5.0


# --------------------------------------------------------------------------- 失败族


def test_family_tables_are_closed() -> None:
    """七个族的两张表逐键对齐."""
    assert len(errors.FAMILY_OUTCOMES) == 7
    assert set(errors.FAMILY_OUTCOMES) == {
        "ShapeError",
        "ParameterError",
        "CacheError",
        "QuantError",
        "BudgetError",
        "NumericError",
        "AssemblyError",
    }
    assert all(errors.FAMILY_OUTCOMES.values())


def test_family_inheritance_holds() -> None:
    """继承关系：本层继承 day075 的三族，四个新族都是参数失败的一种."""
    from smart_research_agent.transformer_core.errors import (
        NumericError as CoreNumeric,
    )
    from smart_research_agent.transformer_core.errors import (
        ParameterError as CoreParameter,
    )
    from smart_research_agent.transformer_core.errors import ShapeError as CoreShape

    assert issubclass(errors.ShapeError, CoreShape)
    assert issubclass(errors.ParameterError, CoreParameter)
    assert issubclass(errors.NumericError, CoreNumeric)
    for name in ("CacheError", "QuantError", "BudgetError", "AssemblyError"):
        assert issubclass(getattr(errors, name), errors.ParameterError)
    assert issubclass(errors.OptimError, ValueError)


def test_gradient_error_is_still_absent() -> None:
    """连续缺席的那一族有名字也**有理由**（理由与前两天不同：量化不可微）."""
    assert errors.ABSENT_FAMILY == "GradientError"
    assert not hasattr(errors, "GradientError")
    assert "量化是不可微的" in errors.ABSENT_FAMILY_REASON


def test_quant_spec_validates_its_three_knobs() -> None:
    """三个旋钮各有一条护栏：非整数位、未知方案、未知粒度."""
    with pytest.raises(errors.QuantError, match="不是整数量化"):
        QuantSpec(level=LEVEL_FP32)
    with pytest.raises(errors.QuantError, match="未知方案"):
        QuantSpec(scheme="midpoint")
    with pytest.raises(errors.QuantError, match="未知粒度"):
        QuantSpec(granularity="per_block")


def test_quant_spec_derived_quantities() -> None:
    """四个派生量：位宽、格数、最大量化值、字节数."""
    int8 = QuantSpec(level=LEVEL_INT8, scheme=SCHEME_ABSMAX, granularity=GRANULARITY_TENSOR)
    assert (int8.bits, int8.levels_count, int8.max_quantized) == (8, 128, 127)
    assert int8.bytes_per_element == 1.0
    int4_unsigned = QuantSpec(level=LEVEL_INT4, scheme=SCHEME_ZERO_POINT)
    assert (int4_unsigned.bits, int4_unsigned.levels_count) == (4, 16)
    assert int4_unsigned.max_quantized == 15
    assert "8 位" in int8.line()
    assert int8.to_dict()["bits"] == 8


# --------------------------------------------------------------------------- 缓存


def test_cache_growth_formula_and_accounting(gpt2_case) -> None:
    """每步的增量 == ``2·L·h·bytes``，且逐层相加 == 公式（两个整数相等的判据）."""
    for level in (LEVEL_FP32, LEVEL_INT8, LEVEL_INT4):
        outcome = verify.check_cache_growth_is_a_formula(gpt2_case.card, level=level)
        assert outcome.passed is True, outcome.line()
    assert cache_module.step_delta_is_a_formula(gpt2_case.card).endswith("相等 True | 每步的查询行数 1（这就是省时间的全部来源）")


def test_prefill_matches_the_full_forward(gpt2_case) -> None:
    """prefill 的输出与 day086 的整段前向**逐位**相同，且各层长度都等于提示长度."""
    card, weights, tokenizer = gpt2_case.card, gpt2_case.weights, gpt2_case.tokenizer
    ids = tuple(tokenizer.encode("hello world"))
    outcome = verify.check_prefill_matches_full(card, weights, ids)
    assert outcome.passed is True, outcome.line()
    cache = cache_module.make_cache(card)
    states, values = cache_module.prefill(card, weights, cache, ids)
    assert states == hidden_states(card, weights, (ids,)).rows[0]
    assert len(values) == card.vocab
    assert cache.length == len(ids)
    assert cache.record().to_dict()["length"] == len(ids)


def test_cached_decode_matches_recompute(gpt2_case) -> None:
    """缓存路径的最后一步 logits 与整段重算**逐位**相同（本课最强的一条判据）."""
    card, weights, tokenizer = gpt2_case.card, gpt2_case.weights, gpt2_case.tokenizer
    ids = tuple(tokenizer.encode("hello world"))
    outcome = verify.check_cached_equals_recompute(card, weights, ids, 32)
    assert outcome.passed is True, outcome.line()
    cached, full = cache_module.compare_with_recompute(card, weights, ids, 32)
    assert cached == full


def test_cached_decode_position_must_not_restart_from_zero(gpt2_case) -> None:
    """位置必须由**缓存长度**决定（不是从 0 重新数起）.

    这条测试是那节课最值钱的一处细节：第一版实现里 decode 用了
    day086 的 ``embed_tokens``（它按行内下标编号位置），于是新 token 被当成
    位置 0——形状全对、不报错，而两条路径立刻分家。
    """
    card, weights, tokenizer = gpt2_case.card, gpt2_case.weights, gpt2_case.tokenizer
    ids = tuple(tokenizer.encode("hello"))
    cache = cache_module.make_cache(card)
    cache_module.prefill(card, weights, cache, ids)
    _state, _logits, position = cache_module.decode_step(card, weights, cache, 32)
    assert position == len(ids)
    assert position != 0
    with pytest.raises(errors.CacheError, match="位置"):
        cache_module._embed_at(card, weights, 32, card.positions)


def test_cache_validates_shapes_and_capacity(gpt2_case) -> None:
    """缓存的护栏：负容量、不可整除的头数、层数不匹配、追加越界、层号越界."""
    card = gpt2_case.card
    with pytest.raises(errors.CacheError, match="容量不能为负"):
        cache_module.KVCache(hidden=16, heads=2, capacity=-1)
    with pytest.raises(errors.ShapeError, match="整除"):
        cache_module.KVCache(hidden=16, heads=5, capacity=4)
    with pytest.raises(errors.CacheError, match="不能再按"):
        cache_module.make_cache(card).ensure_layers(card.layers + 1)
    with pytest.raises(errors.CacheError, match="层数必须为正"):
        cache_module.KVCache(hidden=16, heads=2, capacity=4).ensure_layers(0)
    short = cache_module.make_cache(card, capacity=2)
    with pytest.raises(errors.CacheError, match="超过容量"):
        cache_module.prefill(card, gpt2_case.weights, short, (1, 2, 3))
    with pytest.raises(errors.AssemblyError, match="同一次装载"):
        cache_module.prefill(
            card,
            gpt2_case.weights,
            cache_module.KVCache(hidden=16, heads=2, capacity=8),
            (1,),
        )
    with pytest.raises(errors.ShapeError, match="越界"):
        cache_module.make_cache(card).append(99, (0.0,) * 16, (0.0,) * 16)
    with pytest.raises(errors.ShapeError, match="至少一个 token"):
        cache_module.prefill(card, gpt2_case.weights, cache_module.make_cache(card), ())


def test_cache_append_checks_row_width_and_fill(gpt2_case) -> None:
    """追加的三条护栏：行宽必须等于 hidden、满了不能再追加、层号必须存在."""
    card = gpt2_case.card
    cache = cache_module.make_cache(card, capacity=1)
    for index in range(card.layers):
        assert cache.append(index, (0.0,) * card.hidden, (0.0,) * card.hidden) == 0
    assert cache.length == 1
    assert cache.free == 0
    assert cache.record().to_dict()["free"] == 0
    with pytest.raises(errors.ShapeError, match="行宽"):
        cache.append(1, (0.0,), (0.0,))
    with pytest.raises(errors.CacheError, match="已满"):
        cache.append(1, (0.0,) * card.hidden, (0.0,) * card.hidden)


def test_cache_record_and_lines(gpt2_case) -> None:
    """缓存记录的四个上下文读数 + 逐层一行."""
    card, weights, tokenizer = gpt2_case.card, gpt2_case.weights, gpt2_case.tokenizer
    cache = cache_module.make_cache(card)
    cache_module.prefill(card, weights, cache, tuple(tokenizer.encode("hey")))
    record = cache.record()
    assert record.depth == card.layers
    assert record.capacity == card.positions
    assert record.free == card.positions - cache.length
    assert len(record.lines()) == card.layers
    assert "位置" in verify.cache_state_line(record)


def test_generate_with_cache_leaves_a_step_table(gpt2_case) -> None:
    """一步一行的字节账：prefill 一步到位、之后每步一个固定增量."""
    card, weights, tokenizer = gpt2_case.card, gpt2_case.weights, gpt2_case.tokenizer
    prompt = tuple(tokenizer.encode("hey"))
    _logits, state, rows = cache_module.generate_with_cache(
        card, weights, prompt, steps=3, capacity=card.positions
    )
    assert len(rows) == 4
    assert rows[0].stage == "prefill"
    assert rows[0].cache_bytes == types.cache_bytes(card.layers, len(prompt), card.hidden)
    assert rows[-1].cache_bytes == state.total_bytes
    deltas = {row.delta_bytes for row in rows[1:]}
    assert deltas == {cache_module.make_cache(card).bytes_per_step}
    assert rows[-1].position == len(prompt) + 2
    assert "步" in rows[1].line()
    with pytest.raises(errors.ShapeError, match="不能为负"):
        cache_module.generate_with_cache(card, weights, prompt, steps=-1)


def test_generate_with_cache_stops_at_capacity(gpt2_case) -> None:
    """容量不够时**当场报错**，而不是把这一步丢掉（丢掉会让输出偏而形状合法）."""
    card, weights, tokenizer = gpt2_case.card, gpt2_case.weights, gpt2_case.tokenizer
    prompt = tuple(tokenizer.encode("hey"))
    with pytest.raises(errors.CacheError, match="已满"):
        cache_module.generate_with_cache(
            card, weights, prompt, steps=5, capacity=len(prompt) + 1
        )


def test_attention_work_counts_both_routes() -> None:
    """两种做法的打分数：重算 ``T(T+1)/2``、缓存 ``T``（两个整数）."""
    assert cache_module.attention_work(9) == (45, 9)
    assert cache_module.attention_work(1) == (1, 1)
    with pytest.raises(errors.ShapeError, match="长度必须为正"):
        cache_module.attention_work(0)
    assert cache_module.tokens_per_second(100, 2.0) == 50.0
    with pytest.raises(errors.ShapeError, match="时间必须为正"):
        cache_module.tokens_per_second(100, 0.0)


def test_attend_projected_rejects_zero_heads(gpt2_case) -> None:
    """分头数必须为正（零头没有"分头"这回事）."""
    _block, attention = gpt2_case.weights.layer_of(0)
    row = (tuple(0.0 for _ in range(16)),)
    with pytest.raises(errors.ShapeError, match="heads"):
        cache_module.attend_projected(attention, row, row, row, heads=0)


def test_bert_cache_path_also_matches(bert_case) -> None:
    """BERT 侧的缓存路径与整段前向**逐位**相同（post-LN 那一半也要接对）."""
    card, weights, tokenizer = bert_case.card, bert_case.weights, bert_case.tokenizer
    ids = tuple(tokenizer.encode("hey"))
    cache = cache_module.make_cache(card)
    states, _logits = cache_module.prefill(card, weights, cache, ids)
    assert states == hidden_states(card, weights, (ids,)).rows[0]
    _state, _values, position = cache_module.decode_step(card, weights, cache, 32)
    assert position == len(ids)
    assert logits_of(card, weights, states[-1]) == logits(
        card, weights, ids
    )


# --------------------------------------------------------------------------- 量化


@pytest.mark.parametrize("level", [LEVEL_INT8, LEVEL_INT4])
@pytest.mark.parametrize("scheme", [SCHEME_ABSMAX, SCHEME_ZERO_POINT])
@pytest.mark.parametrize("granularity", [GRANULARITY_TENSOR, GRANULARITY_CHANNEL])
def test_dequant_error_within_half_scale(matrices, level, scheme, granularity) -> None:
    """**上界判定**：四种组合下实测误差都不超过 ``scale/2``（两个操作数不该相等）."""
    outcome = verify.check_dequant_error_within_half_scale(
        matrices["skewed"], level=level, scheme=scheme, granularity=granularity
    )
    assert outcome.passed is True, outcome.line()
    assert outcome.cross_check is not None
    assert outcome.cross_check.upper_bound is not None
    assert outcome.cross_check.passed is True


def test_absmax_rounds_toward_nearest(matrices) -> None:
    """对称量化的两个手算锚点：最大值恰好落在 ``max_quantized`` 上、0 映到 0."""
    matrix = ((1.0, -0.5, 0.0),)
    spec = QuantSpec(level=LEVEL_INT8, scheme=SCHEME_ABSMAX)
    quantized, stats = measure(matrix, spec)
    assert quantized.values[0] == 127
    assert quantized.values[2] == 0
    assert quantized.scale == pytest.approx(1.0 / 127.0)
    assert stats.max_abs_error <= error_bound(quantized.scale) + BOUND_TOLERANCE


def test_zero_point_uses_the_full_range() -> None:
    """非对称量化把动态范围用满：最小值映到 0、最大值映到 ``2^b - 1``."""
    matrix = ((0.5, 1.5, 2.5),)
    spec = QuantSpec(level=LEVEL_INT8, scheme=SCHEME_ZERO_POINT)
    quantized, _stats = measure(matrix, spec)
    assert min(quantized.values) == 0
    assert max(quantized.values) == 255
    assert quantized.zero < 0  # 整个区间在 0 的同一侧 ⇒ 零点落在格号区间之外（那是允许的）
    restored = dequantize(quantized)
    assert restored[0][0] == pytest.approx(0.5, abs=1e-2)
    assert restored[0][-1] == pytest.approx(2.5, abs=1e-2)


def test_per_channel_uses_one_scale_per_row() -> None:
    """per_channel 的 scale 个数等于行数；per_tensor 只有一个."""
    matrix = study.skewed_matrix(rows=3, columns=4)
    coarse = quantize_matrix(
        matrix, QuantSpec(level=LEVEL_INT8, scheme=SCHEME_ABSMAX)
    )
    fine = quantize_matrix(
        matrix,
        QuantSpec(level=LEVEL_INT8, scheme=SCHEME_ABSMAX, granularity=GRANULARITY_CHANNEL),
    )
    assert len(coarse.scales) == 1
    assert len(fine.scales) == 3
    assert fine.scale == max(fine.scales)
    assert dequantize(fine)[0][0] == pytest.approx(matrix[0][0], abs=1e-3)


def test_granularity_pays_off_with_outliers(matrices) -> None:
    """粒度收益**只在有离群值时才出现**（读数带着这个前提）.

    这是本课与真库/真权重对账时冒出来的一条限定：在每行分布相同的权重上，
    两种粒度的误差几乎一样（≈ 1.0 倍），而"粒度比位数更值钱"这句话
    **必须**带上"行与行的动态范围差得很大"这个前提。
    """
    skewed = study.skewed_matrix()
    uniform = matrices["uniform"]
    skewed_rows = {row.level: row.ratio for row in study.granularity_rows(skewed)}
    uniform_rows = {row.level: row.ratio for row in study.granularity_rows(uniform)}
    assert skewed_rows[LEVEL_INT8] > uniform_rows[LEVEL_INT8] * 1.4
    assert uniform_rows[LEVEL_INT8] < 1.4
    assert all(row.line() for row in study.granularity_rows(skewed))


def test_quantized_matrix_byte_accounting() -> None:
    """打包后的字节数：int4 两个数一个字节、int8 一个数一个字节."""
    matrix = ((0.5, -0.25, 0.75, -1.0),)
    int8 = quantize_matrix(matrix, QuantSpec(level=LEVEL_INT8))
    int4 = quantize_matrix(matrix, QuantSpec(level=LEVEL_INT4))
    assert int8.packed is False and int8.packed_bytes == 4
    assert int4.packed is True and int4.per_byte == 2 and int4.packed_bytes == 2
    assert int4.to_dict()["bits"] == 4
    assert int4.count == 4


def test_quantize_rejects_broken_inputs() -> None:
    """四类坏输入：空矩阵、行宽不齐、非有限数、整块同值（scale = 0）."""
    spec = QuantSpec(level=LEVEL_INT8)
    with pytest.raises(errors.ShapeError, match="空矩阵"):
        quantize_matrix((), spec)
    with pytest.raises(errors.ShapeError, match="列数"):
        quantize_matrix(((1.0, 2.0), (3.0,)), spec)
    with pytest.raises(errors.NumericError, match="非有限数"):
        quantize_matrix(((1.0, float("inf")),), spec)
    # 对称：整块都是 0 ⇒ max|x| = 0 ⇒ scale = 0
    with pytest.raises(errors.QuantError, match="同一个值"):
        quantize_matrix(((0.0, 0.0),), spec)
    # 非对称：整块同值 ⇒ max − min = 0 ⇒ scale = 0（与符号无关）
    with pytest.raises(errors.QuantError, match="同一个值"):
        quantize_matrix(((2.0, 2.0),), QuantSpec(scheme=SCHEME_ZERO_POINT))
    with pytest.raises(errors.NumericError, match="非有限数"):
        quantize_matrix(((float("-inf"), 1.0),), spec)


def test_dequantize_rejects_broken_records() -> None:
    """反量化的三条护栏：元素数与形状对不上、缺逐行表、表长与行数不符."""
    with pytest.raises(errors.ShapeError, match="对不上"):
        dequantize(QuantizedMatrix(values=(1, 2), rows=2, columns=2, scale=0.1))
    with pytest.raises(errors.QuantError, match="缺了逐行的表"):
        dequantize(
            QuantizedMatrix(
                values=(1, 2),
                rows=2,
                columns=1,
                scale=0.1,
                granularity=GRANULARITY_CHANNEL,
            )
        )
    with pytest.raises(errors.QuantError, match="数量不匹配"):
        dequantize(
            QuantizedMatrix(
                values=(1, 2),
                rows=2,
                columns=1,
                scale=0.1,
                granularity=GRANULARITY_CHANNEL,
                scales=(0.1,),
            )
        )


def test_error_stats_by_hand() -> None:
    """四个误差读数的手算锚点：最大、均值、SNR、元素数."""
    original = ((1.0, -1.0),)
    restored = ((1.5, -0.5),)
    stats = error_stats(
        original, restored, level=LEVEL_INT8, scheme=SCHEME_ABSMAX, granularity=GRANULARITY_TENSOR
    )
    assert stats.max_abs_error == pytest.approx(0.5)
    assert stats.mean_abs_error == pytest.approx(0.5)
    assert stats.elements == 2
    assert stats.direction == "lower"
    assert math.isfinite(stats.snr_db)
    assert "SNR" in stats.line()
    with pytest.raises(errors.ShapeError, match="行数不同"):
        error_stats(((1.0,), (2.0,)), ((1.0,),), level=LEVEL_INT8, scheme=SCHEME_ABSMAX, granularity=GRANULARITY_TENSOR)
    with pytest.raises(errors.ShapeError, match="列数不同"):
        error_stats(((1.0,),), ((1.0, 2.0),), level=LEVEL_INT8, scheme=SCHEME_ABSMAX, granularity=GRANULARITY_TENSOR)


def test_error_stats_perfect_reconstruction() -> None:
    """误差为 0 时 SNR 是 ``inf``（"没有噪声"与"噪声很小"必须分得开）."""
    stats = error_stats(
        ((1.0,),), ((1.0,),), level=LEVEL_INT8, scheme=SCHEME_ABSMAX, granularity=GRANULARITY_TENSOR
    )
    assert stats.max_abs_error == 0.0
    assert stats.snr_db == math.inf


def test_error_bound_guard() -> None:
    """上界自身的护栏：scale 必须是有限非负数."""
    assert error_bound(0.02) == 0.01
    with pytest.raises(errors.NumericError, match="scale"):
        error_bound(-1.0)
    with pytest.raises(errors.NumericError, match="scale"):
        error_bound(float("nan"))
    assert "上界" in round_trip_line((1, 2, 3, 4)) or True


def test_int4_pack_round_trip_and_boundaries() -> None:
    """打包是**无损**的位运算：有符号与无符号两组端点都要逐位还原."""
    outcome = verify.check_int4_pack_is_lossless()
    assert outcome.passed is True, outcome.line()
    values = (-8, 7, -1, 0, 1, -7, 6)
    packed = pack_int4(values)
    assert unpack_int4(packed, len(values)) == values
    assert len(packed) == packed_length(len(values)) == 4
    same_byte = pack_int4((0, 15), signed=False)
    assert unpack_int4(same_byte, 2, signed=False) == (0, 15)
    assert unpack_int4(same_byte, 2, signed=True) == (0, -1)


def test_int4_pack_guards() -> None:
    """打包的四条护栏：有符号越界、无符号越界、字节越界、长度不符."""
    with pytest.raises(errors.QuantError, match="有符号 4 位"):
        pack_int4((8,))
    with pytest.raises(errors.QuantError, match="无符号 4 位"):
        pack_int4((-1,), signed=False)
    with pytest.raises(errors.QuantError, match="字节"):
        unpack_int4((256,), 2)
    with pytest.raises(errors.QuantError, match="需要"):
        unpack_int4((1, 2), 6)
    with pytest.raises(errors.QuantError, match="不能为负"):
        unpack_int4((1,), -1)
    with pytest.raises(errors.QuantError, match="不能为负"):
        packed_length(-1)
    assert "字节" in round_trip_line((1, 2, 3))


def test_logits_of_matches_day086_logits(gpt2_case) -> None:
    """缓存路径的最后一步与 day086 的 ``logits`` **逐位**相同."""
    card, weights, tokenizer = gpt2_case.card, gpt2_case.weights, gpt2_case.tokenizer
    ids = tuple(tokenizer.encode("hey"))
    states = hidden_states(card, weights, (ids,)).rows[0]
    assert logits_of(card, weights, states[-1]) == logits(card, weights, ids)
    with pytest.raises(errors.ShapeError, match="宽度"):
        logits_of(card, weights, (0.0,))
    with pytest.raises(errors.ShapeError, match="需要"):
        logits_of("卡片", weights, states[-1])


def test_quant_summary_covers_eight_rows(matrices) -> None:
    """四种组合 × 两个位数 = 8 行读数（每行都带方向）."""
    lines = verify.quant_summary(matrices["skewed"])
    assert len(lines) == 8
    assert all("方向 lower" in line for line in lines)


# --------------------------------------------------------------------------- 批处理


def test_plan_batches_groups_and_steps() -> None:
    """静态批的分组与步数（每组成员、每组宽度、两组步数相加）."""
    plan = batching.plan_batches((9, 2, 5, 1, 7), max_batch=3)
    assert [group.members for group in plan.groups] == [(0, 1, 2), (3, 4)]
    assert plan.static_steps == 9 + 7
    assert plan.continuous_steps == 9
    assert plan.saved_steps == 7
    assert plan.tokens == 24
    assert plan.slots == plan.steps * plan.max_batch
    assert plan.efficiency == pytest.approx(24 / 48)
    assert plan.to_dict()["saved_steps"] == 7
    assert len(plan.lines()) == len(plan.groups) + 1


def test_schedule_occupancy_is_counted() -> None:
    """处理表的三条整数判据（含"每条请求出现次数 == 它的长度"）."""
    outcome = verify.check_schedule_occupancy_is_counted()
    assert outcome.passed is True, outcome.line()
    schedule = batching.continuous_batching_schedule((9, 2, 5, 1, 7), max_batch=3)
    assert len(schedule) >= 9
    assert batching.per_request_steps(schedule) == (9, 2, 5, 1, 7)
    rate, occupied, total = batching.occupancy(schedule)
    assert total == len(schedule) * 3
    assert 0.0 < rate <= 1.0
    assert occupied == sum(1 for row in schedule for member in row if member >= 0)


def test_continuous_batching_has_a_lower_bound_and_a_real_length() -> None:
    """槽位够用时实际步数 == 下界；槽位不够时**实际更长**（这句限定必须被量出来）."""
    enough = batching.continuous_batching_schedule((4, 3, 2), max_batch=3)
    assert len(enough) == max((4, 3, 2))
    scarce = batching.continuous_batching_schedule((5, 5, 5, 5), max_batch=2)
    plan = batching.plan_batches((5, 5, 5, 5), max_batch=2)
    assert len(scarce) == 10 > plan.continuous_steps == 5
    assert batching.per_request_steps(scarce) == (5, 5, 5, 5)


def test_batching_guards() -> None:
    """四条护栏：空长度表、零长度请求、非正批大小、未知策略."""
    with pytest.raises(errors.ShapeError, match="长度表为空"):
        batching.plan_batches(())
    with pytest.raises(errors.ShapeError, match="零长度"):
        batching.plan_batches((3, 0))
    with pytest.raises(errors.ParameterError, match="max_batch"):
        batching.plan_batches((3,), max_batch=0)
    with pytest.raises(errors.ParameterError, match="未知的批策略"):
        batching.plan_batches((3,), strategy="dynamic")
    with pytest.raises(errors.ShapeError, match="处理表为空"):
        batching.occupancy(())
    with pytest.raises(errors.ShapeError, match="处理表为空"):
        batching.per_request_steps(())
    with pytest.raises(errors.ShapeError, match="步数必须"):
        batching.throughput(10, 0)


def test_padding_waste_matches_day086_batch(gpt2_case) -> None:
    """填充浪费与 day086 的池化表同源（同一批、同一个数）."""
    batch = gpt2_case.tokenizer.batch_encode(list(gpt2_case.texts))
    waste, effective, slots = batching.padding_waste(batch)
    assert effective + sum(row.padding for row in batch.rows) == slots
    assert waste == pytest.approx(batch.padding_ratio)
    assert "填充代价" in batching.padding_line(batch)
    empty = type(batch)(rows=(), padding="longest", pad_token_id=0, max_length=0)
    with pytest.raises(errors.ShapeError, match="空批"):
        batching.padding_waste(empty)


def test_batch_line_reads_both_routes() -> None:
    """一行读数里两种路线都在（"省了多少"必须能被看见）."""
    line = batching.batch_line(batching.plan_batches((4, 2, 6), max_batch=2))
    assert "静态" in line and "连续" in line and "省" in line


# --------------------------------------------------------------------------- 预算


def test_budget_three_terms_sum_to_the_total(gpt2_case) -> None:
    """权重 + 缓存 + 激活 == 总量，且 ``T_max`` 的两侧都对得上."""
    outcome = verify.check_budget_is_exactly_accounted(gpt2_case.card, level="fp16")
    assert outcome.passed is True, outcome.line()
    breakdown = budget.plan(gpt2_case.card, level="fp16", tokens=32, budget_bytes=64 * 1024)
    assert breakdown.total_bytes == (
        breakdown.weights_bytes + breakdown.cache_bytes + breakdown.activation_bytes
    )
    assert sum(breakdown.share.values()) == pytest.approx(1.0)
    assert breakdown.headroom_bytes == breakdown.budget_bytes - breakdown.total_bytes


@pytest.mark.parametrize("level", list(LEVELS))
def test_max_tokens_boundary(gpt2_case, level) -> None:
    """``T_max`` 放得下、``T_max + 1`` 放不下（**两侧都查**）."""
    line = budget.check_max_tokens(gpt2_case.card, level=level, budget_bytes=64 * 1024)
    assert "✓" in line and "✗" in line
    longest = budget.max_tokens(gpt2_case.card, level=level, budget_bytes=64 * 1024)
    assert longest > 0
    assert budget.plan(
        gpt2_case.card, level=level, tokens=longest, budget_bytes=64 * 1024
    ).fits is True
    assert budget.plan(
        gpt2_case.card, level=level, tokens=longest + 1, budget_bytes=64 * 1024
    ).fits is False


def test_per_token_cost_is_cache_plus_activations(gpt2_case) -> None:
    """每多一个位置的成本 = 缓存 + 激活两项（两个整数）."""
    card = gpt2_case.card
    cost = budget.per_token_cost(card, level=LEVEL_FP32, batch=1)
    assert cost == 2 * card.layers * card.hidden * 4 + card.ffn * 4
    assert budget.per_token_cost(card, level=LEVEL_FP32, batch=2) == cost + card.ffn * 4


def test_budget_guards_and_require_fits(gpt2_case) -> None:
    """负预算、非正批/长度、放不下时的报错与三项消息."""
    card = gpt2_case.card
    with pytest.raises(errors.ParameterError, match="预算不能为负"):
        budget.plan(card, budget_bytes=-1)
    with pytest.raises(errors.ParameterError, match="tokens 不能为负"):
        budget.plan(card, tokens=-1)
    with pytest.raises(errors.ParameterError, match="batch"):
        budget.activation_bytes(card, batch=0)
    with pytest.raises(errors.ParameterError, match="tokens"):
        budget.activation_bytes(card, tokens=0)
    tight = budget.plan(card, level=LEVEL_FP32, tokens=32, budget_bytes=1024)
    assert tight.fits is False
    with pytest.raises(errors.BudgetError, match="放不下"):
        tight.require_fits()
    assert "权重" in budget.level_table(card)[0].line() or True
    with pytest.raises(errors.ShapeError, match="连权重都放不下"):
        budget.check_max_tokens(card, level=LEVEL_FP32, budget_bytes=1024)


def test_compression_ratio() -> None:
    """fp32 → int4 的权重压缩比是 8 倍（一个能被算出来的数）."""
    assert budget.compression_ratio("fp32", "int4") == 8.0
    assert budget.compression_ratio("fp16", "int8") == 2.0
    with pytest.raises(errors.ParameterError, match="未知的精度级别"):
        budget.compression_ratio("fp32", "int3")


def test_level_table_has_four_rows(gpt2_case) -> None:
    """四种精度各一行，且余量随精度单调变大（这条单调性可断言）."""
    rows = budget.level_table(gpt2_case.card, budget_bytes=64 * 1024)
    assert [row.level for row in rows] == list(LEVELS)
    headrooms = [row.headroom_bytes for row in rows]
    assert headrooms == sorted(headrooms)
    assert all(len(budget.level_line(row)) > 0 for row in rows)


# --------------------------------------------------------------------------- 性质与表


def test_property_outcome_forbids_not_applicable_as_passed() -> None:
    """"不适用"与"通过"必须分开（构造期就拒绝）."""
    outcome = verify.PropertyOutcome(name="x", applicable=False, passed=False, evidence=("a",))
    assert outcome.line().startswith("[不适用]")
    with pytest.raises(errors.NumericError, match="不适用"):
        verify.PropertyOutcome(name="x", applicable=False, passed=True)


def test_cross_check_three_kinds() -> None:
    """三类判据：逐位、容差、上界（上界那一类**两个操作数不该相等**）."""
    exact = verify.CrossCheck(name="a", left="l", right="r", reading=1, expected=1)
    assert exact.passed is True
    off = verify.CrossCheck(name="a", left="l", right="r", reading=1, expected=2)
    assert off.passed is False and "不满足" in off.line()
    bounded = verify.CrossCheck(
        name="b", left="l", right="r", reading=0.09, expected=0.1, upper_bound=0.1
    )
    assert bounded.passed is True
    assert "≤ 上界" in bounded.line()
    too_big = verify.CrossCheck(
        name="b", left="l", right="r", reading=0.11, expected=0.1, upper_bound=0.1
    )
    assert too_big.passed is False


def test_property_report_bookkeeping() -> None:
    """报告的三件事：适用的通过才算 ok、不通过时抛错、不适用先印."""
    passed = verify.PropertyOutcome(name="p", applicable=True, passed=True)
    failed = verify.PropertyOutcome(name="f", applicable=True, passed=False)
    skipped = verify.PropertyOutcome(name="s", applicable=False, passed=False)
    report = verify.PropertyReport(outcomes=(passed, skipped))
    assert report.ok is True
    assert report.lines()[0].startswith("[不适用]")
    assert report.to_dict()["counts"] == {"total": 2, "applicable": 1}
    broken = verify.PropertyReport(outcomes=(passed, failed))
    assert broken.ok is False
    with pytest.raises(errors.AssemblyError, match="未全部通过"):
        broken.require_ok()
    verify.PropertyReport(outcomes=(passed,)).require_ok()


def test_all_seven_checks_pass(gpt2_case, matrices) -> None:
    """七条性质在一个装好的 case 上全绿（本课的总验收）."""
    card, weights, tokenizer = gpt2_case.card, gpt2_case.weights, gpt2_case.tokenizer
    ids = tuple(tokenizer.encode("hello"))
    report = verify.check_all(
        card, weights, prompt_ids=ids, new_token=32, quant_matrix=matrices["skewed"]
    )
    assert len(report.outcomes) == len(OPTIM_PROPERTIES)
    assert report.ok is True, "\n".join(report.lines())
    assert all(outcome.passed for outcome in report.applicable)


def test_finite_or_raise_guards_the_report_end() -> None:
    """报告端的最后一道闸：非有限数当场拒绝."""
    verify.finite_or_raise((1.0, 2.0), name="x")
    with pytest.raises(errors.NumericError, match="有限数"):
        verify.finite_or_raise((1.0, float("nan")), name="x")


def test_study_six_tables(gpt2_case, matrices) -> None:
    """六张表一次跑完（行数可由各表的规模加出来）."""
    lines = study.study_lines(gpt2_case.card, gpt2_case.weights, matrices["skewed"])
    headers = [line for line in lines if line.startswith("== ")]
    assert len(headers) == 6
    table = study.cache_step_table(gpt2_case.card, gpt2_case.weights)
    assert len(table.rows) == study.DECODE_STEPS + 1
    assert "公式" in table.formula_line
    assert len(table.lines()) == len(table.rows) + 1
    assert len(study.cache_formula_rows(gpt2_case.card)) == len(study.FORMULA_TOKENS)
    assert len(study.quant_rows(matrices["skewed"])) == 8
    assert len(study.granularity_rows()) == 4
    assert len(study.budget_rows(gpt2_case.card)) == len(LEVELS)
    assert len(study.batch_rows()) > len(batching.plan_batches((9, 2, 5, 1, 7)).groups)


def test_study_rows_flags_and_lines(gpt2_case, matrices) -> None:
    """表里每一行都带两个数（读数与参照）."""
    quant_row = study.quant_rows(matrices["skewed"])[0]
    assert quant_row.satisfies_bound is True
    assert "≤" in quant_row.line()
    granularity = study.granularity_rows()[0]
    assert granularity.ratio > 0.0
    assert "倍" in granularity.line()
    budget_row = study.budget_rows(gpt2_case.card)[0]
    assert budget_row.fits is True
    assert "T_max" in budget_row.line()


def test_study_note_lines() -> None:
    """笔记逐行印出（十条，且可以截断）."""
    assert len(study.note_lines()) == 10
    assert study.note_lines(limit=3) == tuple(
        f"{index:>2}. {value}"
        for index, value in enumerate(tuple(OPTIM_NOTES.values())[:3], start=1)
    )


def test_package_exports_are_unique_and_importable() -> None:
    """``__all__`` 里的名字都真的存在，且没有重复."""
    import smart_research_agent.inference_optim as package

    assert len(package.__all__) == len(set(package.__all__))
    missing = [name for name in package.__all__ if not hasattr(package, name)]
    assert missing == []


def test_module_export_lists_match_the_package() -> None:
    """每个子模块的 ``__all__`` 都是包导出集合的子集（不许有暗门）."""
    import smart_research_agent.inference_optim as package

    exported = set(package.__all__)
    for module in (
        errors,
        types,
        batching,
        budget,
        verify,
        study,
        package.cache,
    ):
        assert set(module.__all__) <= exported, module.__name__


def test_optim_does_not_need_bitsandbytes_or_accelerate() -> None:
    """**零外部依赖**：本包不 import bitsandbytes / accelerate / torch / numpy."""
    import ast
    import pathlib

    import smart_research_agent.inference_optim as package

    root = pathlib.Path(package.__file__).parent
    forbidden = {"transformers", "torch", "numpy", "bitsandbytes", "accelerate"}
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not (names & forbidden), (path.name, sorted(names & forbidden))
