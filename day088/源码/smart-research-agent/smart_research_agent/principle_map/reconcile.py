"""``reconcile``：探针——**真的 import 并调用既有包**，把读数算出来（day088 / M7-D12）.

本模块是今天唯一"动手"的地方。十二条命题各有一个探针，每个探针只做一件事：

```text
probe_attention_is_differentiable_retrieval       输出 == weights · V（"可微混合"的字面含义）
probe_cosine_is_normalized_dot                    cos(a,b) == a·b/(‖a‖‖b‖)，且放大后不变
probe_attention_rows_are_distributions            每一行的和与 1 的最大偏差
probe_causal_mask_blocks_future                   严格上三角的权重最大值（应为 0.0）
probe_multi_head_splits_inside_projection         merge(split(x)) 与 x 逐位比较
probe_position_encoding_breaks_permutation        置换缺口的两个读数（无编码 vs 有编码）
probe_embedding_similarity_is_direction           5a 与 -3a 的余弦（应为 1.0 与 -1.0）
probe_rank_ordering_matches_attention_peaks       秩相关 / 峰值一致 / top-k 重合
probe_cache_reuse_is_bitwise_exact                缓存路径与整段重算的最后一行 logits
probe_cache_bytes_is_a_formula                    五个长度上 2·L·T·h·bytes 的核对
probe_quantization_error_bounded_by_half_step     实测误差 <= scale/2（上界判定）
probe_generation_respects_budget_breakdown        三项之和与总量、以及 T_max 的两侧
```

## 两条纪律

1. **输入全部写死**（必要时用确定性 LCG 造数）：同一个探针连续两次调用必须**逐位相同**。
   这条纪律由 :func:`verify.check_evidence_is_reproducible` 当场检查。
2. **禁止写死未被真实验证的数字**：本模块里出现的每一个数都来自一次真实的函数调用，
   没有任何一个是从别处抄来的常数。

## 一处刻意的构造：秩相关那条探针

"注意力的排序与检索的排序一致"这句话**不是恒真的**——它只在
"投影是恒等映射、且行已归一化"时成立：那时打分 ``q·k`` 就等于余弦，
而 softmax 是单调的。探针因此把这两个前提交代清楚，而不是假装它对所有参数都成立。
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable

from smart_research_agent.math_foundations.attention import scaled_dot_product_attention
from smart_research_agent.math_foundations.linalg import cosine, dot, norm, normalize
from smart_research_agent.math_foundations.probability import uniforms
from smart_research_agent.principle_map.errors import ClaimError
from smart_research_agent.inference_optim.types import (
    GRANULARITY_TENSOR,
    LEVEL_FP32,
    LEVEL_INT8,
)
from smart_research_agent.principle_map.types import (
    PRINCIPLE_ATTENTION_IS_DIFFERENTIABLE_RETRIEVAL,
    PRINCIPLE_ATTENTION_ROWS_ARE_DISTRIBUTIONS,
    PRINCIPLE_CACHE_BYTES_IS_A_FORMULA,
    PRINCIPLE_CACHE_REUSE_IS_BITWISE_EXACT,
    PRINCIPLE_CAUSAL_MASK_BLOCKS_FUTURE,
    PRINCIPLE_COSINE_IS_NORMALIZED_DOT,
    PRINCIPLE_EMBEDDING_SIMILARITY_IS_DIRECTION,
    PRINCIPLE_GENERATION_RESPECTS_BUDGET_BREAKDOWN,
    PRINCIPLE_MULTI_HEAD_SPLITS_INSIDE_PROJECTION,
    PRINCIPLE_POSITION_ENCODING_BREAKS_PERMUTATION,
    PRINCIPLE_QUANTIZATION_ERROR_BOUNDED_BY_HALF_STEP,
    PRINCIPLE_RANK_ORDERING_MATCHES_ATTENTION_PEAKS,
    PRINCIPLES,
    Evidence,
    require_principle_id,
)

#: 浮点容差（"两条独立求和的路径"用它；"逐位"那一类用 ``==``）.
EVIDENCE_TOLERANCE = 1e-12

#: 今天的探针统一用这一份提示（写死，因此可复现）.
PROBE_PROMPT: tuple[int, ...] = (1, 2, 3)

#: 今天的探针统一用这个新 token（写死）.
PROBE_NEW_TOKEN = 32


# ---------------------------------------------------------------------------
# 确定性造数（LCG，不是密码学安全的）
# ---------------------------------------------------------------------------


def _lcg_matrix(rows: int, columns: int, *, seed: int) -> tuple[tuple[float, ...], ...]:
    """确定性矩阵（``uniforms`` 的 [0,1) 映射到 [-1,1)）——同一个种子永远同一批数."""
    raw = uniforms(rows * columns, seed=seed)
    values = [value * 2.0 - 1.0 for value in raw]
    return tuple(
        tuple(values[row * columns : (row + 1) * columns]) for row in range(rows)
    )


def _normalized_rows(matrix: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
    """把每一行归一到单位长度（**这正是 day064 的写入侧要求**）."""
    return tuple(normalize(row) for row in matrix)


def _permute(matrix: tuple[tuple[float, ...], ...], order: tuple[int, ...]) -> tuple[tuple[float, ...], ...]:
    """按 ``order`` 重排行（``order[i]`` 是新的第 i 行取自旧矩阵的哪一行）."""
    return tuple(matrix[index] for index in order)


def _max_abs_diff(left: tuple[tuple[float, ...], ...], right: tuple[tuple[float, ...], ...]) -> float:
    """两个矩阵逐元素最大绝对差（形状不一致时抛 ``ShapeError``）."""
    from smart_research_agent.principle_map.errors import ShapeError

    if len(left) != len(right):
        raise ShapeError(f"行数不同：{len(left)} 与 {len(right)}。")
    worst = 0.0
    for row_left, row_right in zip(left, right, strict=True):
        if len(row_left) != len(row_right):
            raise ShapeError("列数不同：两份矩阵形状不一致。")
        for a, b in zip(row_left, row_right, strict=True):
            worst = max(worst, abs(a - b))
    return worst


def _identity(size: int) -> tuple[tuple[float, ...], ...]:
    """``size × size`` 单位矩阵."""
    return tuple(
        tuple(1.0 if row == column else 0.0 for column in range(size))
        for row in range(size)
    )


# ---------------------------------------------------------------------------
# 一份确定性的 in-memory 模型（GPT-2 玩具卡片 + LCG 权重）
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _case() -> tuple[object, object]:
    """装一份 GPT-2 玩具 case（**内存假仓库、零网络、零文件**）.

    它镜像 day087 演示脚本里的 ``load_case``：同一个 commit、同一个种子，
    因此"缓存逐位相同"那条探针与 day087 的演示读的是同一批数。
    """
    import json

    from smart_research_agent.hf_integration import config, forward, hub
    from smart_research_agent.hf_integration.tokenizer import build_tiny_tokenizer

    built = build_tiny_tokenizer()
    tiny = {
        "model_type": "gpt2",
        "vocab_size": len(built.vocab),
        "n_embd": 16,
        "n_head": 2,
        "n_layer": 2,
        "n_positions": 32,
        "activation_function": "gelu_new",
        "layer_norm_epsilon": 1e-5,
        "tie_word_embeddings": True,
    }
    commit = "1f2e3d4c5b6a7988071625344352617069859a0b"
    repo_id = "org/tiny-gpt2"
    files = {
        "config.json": json.dumps(tiny).encode("utf-8"),
        "vocab.json": json.dumps(built.vocab).encode("utf-8"),
        "merges.txt": (
            "#version: 0.2\n" + "\n".join(f"{a} {b}" for a, b in built.merges) + "\n"
        ).encode("utf-8"),
        "model.safetensors": b"\x00\x01",
    }
    fake = hub.InMemoryHub(commits={repo_id: {"main": commit}}, trees={repo_id: {commit: files}})
    resolver = hub.HubResolver(hub=fake, cache_dir="C:/cache/hub")
    snapshot = resolver.resolve(repo_id, local_files_only=False)
    card = config.parse_config(tiny, name=repo_id)
    weights = forward.make_weights(card, seed=7)
    del snapshot
    return card, weights


# ---------------------------------------------------------------------------
# 十二个探针（每个只做一件事）
# ---------------------------------------------------------------------------


def probe_attention_is_differentiable_retrieval() -> Evidence:
    """输出 == ``weights · V``（可微混合的字面含义），且权重是一组分布."""
    queries = ((1.0, 0.0), (0.0, 1.0))
    keys = ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5))
    values = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
    report = scaled_dot_product_attention(queries, keys, values)
    recomputed = tuple(
        tuple(
            math.fsum(report.weights[row][column] * values[column][col] for column in range(len(values)))
            for col in range(len(values[0]))
        )
        for row in range(len(queries))
    )
    gap = _max_abs_diff(report.output, recomputed)
    row_gap = max(abs(sum(row) - 1.0) for row in report.weights)
    negatives = sum(1 for row in report.weights for weight in row if weight < 0.0)
    return Evidence(
        principle=PRINCIPLE_ATTENTION_IS_DIFFERENTIABLE_RETRIEVAL,
        reading=gap,
        expected=0.0,
        exact=False,
        source="math_foundations.attention.scaled_dot_product_attention",
        note=(
            f"|output - weights·V| = {gap:.3e}；行和与 1 的最大偏差 {row_gap:.3e}；"
            f"负权重个数 {negatives}；峰值 {[round(value, 4) for value in report.peak_weights]}"
        ),
    )


def probe_cosine_is_normalized_dot() -> Evidence:
    """``cos(a,b) == a·b/(‖a‖‖b‖)``，且把任一侧**同向放大**后的余弦不变."""
    left = (3.0, 4.0)
    right = (4.0, 3.0)
    observed = cosine(left, right)
    manual = dot(left, right) / (norm(left) * norm(right))
    scaled = cosine((6.0, 8.0), right)  # 2·left 与 right ⇒ 应为同一个余弦
    reading = max(abs(observed - manual), abs(scaled - observed))
    return Evidence(
        principle=PRINCIPLE_COSINE_IS_NORMALIZED_DOT,
        reading=reading,
        expected=0.0,
        exact=False,
        source="math_foundations.linalg.cosine",
        note=(
            f"cos((3,4),(4,3)) = {observed:.6f}（手算 24/25 = {manual:.6f}）；"
            f"cos(2a,b) = {scaled:.6f}（与 cos(a,b) 同向放大应不变）"
        ),
    )


def probe_attention_rows_are_distributions() -> Evidence:
    """每一行的和与 1 的最大偏差（**跨天对账**：调 ``transformer_core``）."""
    from smart_research_agent.transformer_core.layers import self_attention
    from smart_research_agent.transformer_core.train import default_parameters
    from smart_research_agent.transformer_core.types import (
        DEFAULT_SUM_TOLERANCE,
        assert_rows_are_distributions,
    )

    params = default_parameters(4, seed=7)
    inputs = _lcg_matrix(5, 4, seed=21)
    forward = self_attention(params, inputs)
    sums = assert_rows_are_distributions(forward.weights)
    reading = max(abs(value - 1.0) for value in sums)
    return Evidence(
        principle=PRINCIPLE_ATTENTION_ROWS_ARE_DISTRIBUTIONS,
        reading=reading,
        expected=0.0,
        exact=False,
        upper_bound=DEFAULT_SUM_TOLERANCE,
        source="transformer_core.types.assert_rows_are_distributions",
        note=(
            f"{len(forward.weights)} 行、最大行和偏差 {reading:.3e}"
            f"（容差 {DEFAULT_SUM_TOLERANCE:.0e}）；"
            f"最小权重 {min(weight for row in forward.weights for weight in row):.3e}"
        ),
    )


def probe_causal_mask_blocks_future() -> Evidence:
    """严格上三角的权重最大值（被因果掩码挡住的位置应当**恰好是 0.0**）."""
    from smart_research_agent.transformer_core.layers import resolve_mask, self_attention
    from smart_research_agent.transformer_core.train import default_parameters

    params = default_parameters(4, seed=7)
    inputs = _lcg_matrix(5, 4, seed=21)
    forward = self_attention(params, inputs, causal=True)
    mask = resolve_mask(5, causal=True)
    mask_ok = all(mask[row][column] == (column <= row) for row in range(5) for column in range(5))
    worst = max(
        (abs(forward.weights[row][column]) for row in range(5) for column in range(5) if column > row),
        default=0.0,
    )
    return Evidence(
        principle=PRINCIPLE_CAUSAL_MASK_BLOCKS_FUTURE,
        reading=worst,
        expected=0.0,
        exact=True,
        source="transformer_core.layers.resolve_mask + self_attention",
        note=(
            f"严格上三角的最大权重 {worst!r}（应为 0.0）；"
            f"掩码表 == (j <= i)：{mask_ok}；可见位置总数 "
            f"{sum(1 for row in mask for cell in row if cell)}"
        ),
    )


def probe_multi_head_splits_inside_projection() -> Evidence:
    """``merge_heads(split_heads(x)) == x``（**逐位**，含每头形状）."""
    from smart_research_agent.hf_source.attention import merge_heads, split_heads

    matrix = _lcg_matrix(4, 4, seed=33)
    heads = 2
    parts = split_heads(matrix, heads)
    merged = merge_heads(parts)
    reading = 0.0 if merged == matrix else 1.0
    return Evidence(
        principle=PRINCIPLE_MULTI_HEAD_SPLITS_INSIDE_PROJECTION,
        reading=reading,
        expected=0.0,
        exact=True,
        source="hf_source.attention.split_heads / merge_heads",
        note=(
            f"{len(parts)} 个头，每头 {len(parts[0])}×{len(parts[0][0])}；"
            f"往返逐位还原 {merged == matrix}"
        ),
    )


def probe_position_encoding_breaks_permutation() -> Evidence:
    """置换缺口：无编码时 0.0，注入位置表之后**大于零**（这才是"知道顺序"的证据）."""
    from smart_research_agent.positional_encoding.layers import inject, sinusoidal_table
    from smart_research_agent.positional_encoding.types import PAIRING_ALIGNED
    from smart_research_agent.transformer_core.layers import self_attention
    from smart_research_agent.transformer_core.train import default_parameters

    size = 4
    rows = 5
    params = default_parameters(size, seed=7)
    inputs = _lcg_matrix(rows, size, seed=44)
    table = sinusoidal_table(rows, size, pairing=PAIRING_ALIGNED)
    order = (0, 2, 1, 3, 4)
    permuted = _permute(inputs, order)

    base = self_attention(params, inputs).output
    base_permuted = self_attention(params, permuted).output
    gap_plain = _max_abs_diff(_permute(base, order), base_permuted)

    _taken, injected = inject(inputs, table)
    _taken_two, injected_permuted = inject(permuted, table)
    encoded = self_attention(params, injected).output
    encoded_permuted = self_attention(params, injected_permuted).output
    gap_encoded = _max_abs_diff(_permute(encoded, order), encoded_permuted)

    holds = gap_plain <= EVIDENCE_TOLERANCE and gap_encoded > 1e-6
    return Evidence(
        principle=PRINCIPLE_POSITION_ENCODING_BREAKS_PERMUTATION,
        reading=0.0 if holds else 1.0,
        expected=0.0,
        exact=True,
        source="positional_encoding.layers.inject + transformer_core.self_attention",
        note=(
            f"无编码的置换缺口 {gap_plain:.3e}（应为 0）；"
            f"注入位置表后的缺口 {gap_encoded:.3e}（应 > 0）"
        ),
    )


def probe_embedding_similarity_is_direction() -> Evidence:
    """方向相似：``cos(5a, a) == 1.0``、``cos(-a, a) == -1.0``，并与 day073 逐位一致."""
    from smart_research_agent.vectorstore.metrics import cosine_similarity

    base = (1.0, 2.0, 3.0)
    scaled = (5.0, 10.0, 15.0)
    negated = (-1.0, -2.0, -3.0)
    forward_cos = cosine_similarity(scaled, base)
    backward_cos = cosine_similarity(negated, base)
    bridge_cos = cosine(base, scaled)
    reading = max(abs(forward_cos - 1.0), abs(backward_cos + 1.0), abs(forward_cos - bridge_cos))
    return Evidence(
        principle=PRINCIPLE_EMBEDDING_SIMILARITY_IS_DIRECTION,
        reading=reading,
        expected=0.0,
        exact=False,
        source="vectorstore.metrics.cosine_similarity / math_foundations.linalg.cosine",
        note=(
            f"cos(5a,a) = {forward_cos:.6f}；cos(-a,a) = {backward_cos:.6f}；"
            f"与 day073 的差 {abs(forward_cos - bridge_cos):.3e}"
        ),
    )


def probe_rank_ordering_matches_attention_peaks() -> Evidence:
    """恒等投影 + 单位行时，注意力排序与余弦检索排序**完全一致**（秩相关 1.0）.

    这是一处刻意的构造：它把"两个排序一致"的前提（恒等投影、行已归一化）交代清楚，
    而不是假装它对所有参数都成立。
    """
    from smart_research_agent.transformer_core.layers import self_attention
    from smart_research_agent.transformer_core.types import AttentionParams
    from smart_research_agent.transformer_core.verify import compare_with_retrieval

    size = 4
    rows = 5
    identity = _identity(size)
    params = AttentionParams(
        w_query=identity, w_key=identity, w_value=identity, w_output=identity
    )
    inputs = _normalized_rows(_lcg_matrix(rows, size, seed=55))
    forward = self_attention(params, inputs)
    comparison = compare_with_retrieval(forward, top_k=2)
    reading = min(comparison.peak_ratio, comparison.overlap_ratio, comparison.rank_correlation)
    return Evidence(
        principle=PRINCIPLE_RANK_ORDERING_MATCHES_ATTENTION_PEAKS,
        reading=reading,
        expected=1.0,
        exact=False,
        source="transformer_core.verify.compare_with_retrieval",
        note=(
            f"峰值一致 {comparison.peak_agreement}/{comparison.rows}；"
            f"top-{comparison.top_k} 重合 {comparison.overlap}/{comparison.rows * comparison.top_k}；"
            f"秩相关 {comparison.rank_correlation:+.6f}"
        ),
    )


def probe_cache_reuse_is_bitwise_exact() -> Evidence:
    """缓存路径的最后一步 logits 与整段重算**逐位**相同（**跨天对账**）."""
    from smart_research_agent.inference_optim.cache import compare_with_recompute

    card, weights = _case()
    cached, full = compare_with_recompute(card, weights, PROBE_PROMPT, PROBE_NEW_TOKEN)
    reading = 0.0 if cached == full else 1.0
    return Evidence(
        principle=PRINCIPLE_CACHE_REUSE_IS_BITWISE_EXACT,
        reading=reading,
        expected=0.0,
        exact=True,
        source="inference_optim.cache.compare_with_recompute",
        note=(
            f"提示 {len(PROBE_PROMPT)} 个 token + 新 token {PROBE_NEW_TOKEN}；"
            f"两条路径宽度 {len(cached)}（= 词表）；逐位相同 {cached == full}"
        ),
    )


def probe_cache_bytes_is_a_formula() -> Evidence:
    """五个长度上核对 ``cache_bytes == 2 · L · T · h · bytes``（**整数相等**）."""
    from smart_research_agent.inference_optim.types import cache_bytes

    card, _weights = _case()
    lengths = (0, 1, 5, 16, 32)
    mismatches = 0
    per_step = int(2 * card.layers * card.hidden * 4)
    for tokens in lengths:
        formula = cache_bytes(card.layers, tokens, card.hidden, LEVEL_FP32)
        expected = per_step * tokens if tokens else 0
        mismatches += int(formula != expected)
    return Evidence(
        principle=PRINCIPLE_CACHE_BYTES_IS_A_FORMULA,
        reading=float(mismatches),
        expected=0.0,
        exact=True,
        source="inference_optim.types.cache_bytes",
        note=(
            f"L={card.layers}、h={card.hidden}、每步 {per_step} 字节；"
            f"核对长度 {list(lengths)}；不等个数 {mismatches}"
        ),
    )


def probe_quantization_error_bounded_by_half_step() -> Evidence:
    """实测最大误差 <= ``scale/2``（**上界判定**：两个操作数不该相等）."""
    from smart_research_agent.inference_optim.quantize import error_bound, measure
    from smart_research_agent.inference_optim.study import skewed_matrix
    from smart_research_agent.inference_optim.types import QuantSpec

    matrix = skewed_matrix()
    spec = QuantSpec(level=LEVEL_INT8, granularity=GRANULARITY_TENSOR)
    quantized, stats = measure(matrix, spec)
    bound = error_bound(quantized.scale)
    return Evidence(
        principle=PRINCIPLE_QUANTIZATION_ERROR_BOUNDED_BY_HALF_STEP,
        reading=stats.max_abs_error,
        expected=bound,
        exact=False,
        upper_bound=bound,
        source="inference_optim.quantize.error_bound",
        note=(
            f"{spec.line()}；scale={quantized.scale:.6e}；"
            f"最大误差 {stats.max_abs_error:.3e} <= 上界 {bound:.3e}；SNR {stats.snr_db:.2f} dB"
        ),
    )


def probe_generation_respects_budget_breakdown() -> Evidence:
    """三项之和 == 总量，且 ``T_max`` 放得下、``T_max + 1`` 放不下（**跨天对账**）."""
    from smart_research_agent.inference_optim.budget import max_tokens, plan

    card, _weights = _case()
    budget_bytes = 64 * 1024
    breakdown = plan(card, level=LEVEL_FP32, tokens=32, batch=1, budget_bytes=budget_bytes)
    summed = breakdown.weights_bytes + breakdown.cache_bytes + breakdown.activation_bytes
    longest = max_tokens(card, level=LEVEL_FP32, batch=1, budget_bytes=budget_bytes)
    inside = plan(card, level=LEVEL_FP32, tokens=longest, batch=1, budget_bytes=budget_bytes).fits
    outside = plan(
        card, level=LEVEL_FP32, tokens=longest + 1, batch=1, budget_bytes=budget_bytes
    ).fits
    reading = float(breakdown.total_bytes) if (inside and not outside) else -1.0
    return Evidence(
        principle=PRINCIPLE_GENERATION_RESPECTS_BUDGET_BREAKDOWN,
        reading=reading,
        expected=float(summed),
        exact=True,
        source="inference_optim.budget.plan / max_tokens",
        note=(
            f"权重 {breakdown.weights_bytes} + 缓存 {breakdown.cache_bytes} + "
            f"激活 {breakdown.activation_bytes} = {breakdown.total_bytes}；T_max={longest}、"
            f"放得下 {inside}、T_max+1 放得下 {outside}"
        ),
    )


#: 十二个探针的注册表（**键与 types.PRINCIPLES 逐键对齐**）.
PROBES: dict[str, Callable[[], Evidence]] = {
    PRINCIPLE_ATTENTION_IS_DIFFERENTIABLE_RETRIEVAL: probe_attention_is_differentiable_retrieval,
    PRINCIPLE_COSINE_IS_NORMALIZED_DOT: probe_cosine_is_normalized_dot,
    PRINCIPLE_ATTENTION_ROWS_ARE_DISTRIBUTIONS: probe_attention_rows_are_distributions,
    PRINCIPLE_CAUSAL_MASK_BLOCKS_FUTURE: probe_causal_mask_blocks_future,
    PRINCIPLE_MULTI_HEAD_SPLITS_INSIDE_PROJECTION: probe_multi_head_splits_inside_projection,
    PRINCIPLE_POSITION_ENCODING_BREAKS_PERMUTATION: probe_position_encoding_breaks_permutation,
    PRINCIPLE_EMBEDDING_SIMILARITY_IS_DIRECTION: probe_embedding_similarity_is_direction,
    PRINCIPLE_RANK_ORDERING_MATCHES_ATTENTION_PEAKS: probe_rank_ordering_matches_attention_peaks,
    PRINCIPLE_CACHE_REUSE_IS_BITWISE_EXACT: probe_cache_reuse_is_bitwise_exact,
    PRINCIPLE_CACHE_BYTES_IS_A_FORMULA: probe_cache_bytes_is_a_formula,
    PRINCIPLE_QUANTIZATION_ERROR_BOUNDED_BY_HALF_STEP: probe_quantization_error_bounded_by_half_step,
    PRINCIPLE_GENERATION_RESPECTS_BUDGET_BREAKDOWN: probe_generation_respects_budget_breakdown,
}

if set(PROBES) != set(PRINCIPLES):  # pragma: no cover - 只在有人改表时触发
    raise ClaimError(
        "探针注册表的键与 types.PRINCIPLES 不一致："
        f"多 {sorted(set(PROBES) - set(PRINCIPLES))}、缺 {sorted(set(PRINCIPLES) - set(PROBES))}——"
        "少一个探针的那条命题在图上会只剩一个'没有证据'的箭头。"
    )


def probe_all() -> tuple[Evidence, ...]:
    """按 ``PRINCIPLES`` 的顺序跑完全部探针（**单次计算、可重复**）."""
    return tuple(PROBES[principle_id]() for principle_id in PRINCIPLES)


def evidence_for(principle_id: str) -> Evidence:
    """某条命题的现场读数（未知 id 抛 ``ReferenceError``）."""
    require_principle_id(principle_id)
    return PROBES[principle_id]()


def probe_source(principle_id: str) -> str:
    """某条命题的探针来源串（与 :attr:`types.Evidence.source` 同源）."""
    return evidence_for(principle_id).source


def evidence_sources() -> dict[str, str]:
    """十二条命题各自的来源（报告里读它）."""
    from smart_research_agent.principle_map import claims

    return {principle.id: claims.principle(principle.id).artifact for principle in claims.principles()}


__all__ = [
    "EVIDENCE_TOLERANCE",
    "PROBES",
    "PROBE_NEW_TOKEN",
    "PROBE_PROMPT",
    "evidence_for",
    "evidence_sources",
    "probe_attention_is_differentiable_retrieval",
    "probe_attention_rows_are_distributions",
    "probe_cache_bytes_is_a_formula",
    "probe_cache_reuse_is_bitwise_exact",
    "probe_causal_mask_blocks_future",
    "probe_all",
    "probe_cosine_is_normalized_dot",
    "probe_embedding_similarity_is_direction",
    "probe_generation_respects_budget_breakdown",
    "probe_multi_head_splits_inside_projection",
    "probe_position_encoding_breaks_permutation",
    "probe_quantization_error_bounded_by_half_step",
    "probe_rank_ordering_matches_attention_peaks",
    "probe_source",
]
