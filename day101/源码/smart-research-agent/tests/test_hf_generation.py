"""``hf_source.generate``：四个 warper 与两种搜索（day085 / M7-D9）."""

from __future__ import annotations

import math

import pytest

from smart_research_agent.hf_source import generation as generate
from smart_research_agent.hf_source import types
from smart_research_agent.hf_source.errors import (
    GenerationError,
    NumericError,
    ParameterError,
    ShapeError,
)
from tests import hf_samples as samples

LOGITS = generate.FILTERED_LOGIT


def settings(**overrides: object) -> types.GenerationSettings:
    """一份默认配置（每个用例只改一个旋钮——day081 的纪律在本包继续生效）."""
    return types.GenerationSettings(**overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_new_tokens": 0},
        {"temperature": 0.0},
        {"temperature": float("inf")},
        {"top_k": -1},
        {"top_p": 0.0},
        {"top_p": 1.5},
        {"repetition_penalty": 0.0},
        {"num_beams": 0},
        {"eos_token": 99},
    ],
)
def test_validate_settings_rejects_bad_values(overrides: dict[str, object]) -> None:
    """每一个非法开关都在入口被拒绝，并指出该改什么."""
    with pytest.raises((ParameterError, ShapeError)):
        generate.validate_settings(settings(**overrides), samples.VOCAB)


def test_validate_settings_rejects_bad_vocab() -> None:
    """词表大小必须为正."""
    with pytest.raises(ParameterError):
        generate.validate_settings(settings(), 0)


def test_validate_settings_strategy_failures_are_generation_errors() -> None:
    """策略层面的三种失败归 **GenerationError**（要改的是"这次怎么生成"）."""
    with pytest.raises(GenerationError):
        generate.validate_settings(settings(top_k=samples.VOCAB + 1), samples.VOCAB)
    with pytest.raises(GenerationError):
        generate.validate_settings(settings(num_beams=samples.VOCAB + 1), samples.VOCAB)
    with pytest.raises(GenerationError):
        generate.validate_settings(settings(length_penalty=-1.0), samples.VOCAB)
    with pytest.raises(GenerationError):
        generate.validate_settings(settings(num_beams=2, do_sample=True), samples.VOCAB)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "greedy"),
        ({"num_beams": 2}, "beam"),
        ({"do_sample": True}, "sample"),
        ({"do_sample": True, "top_k": 3}, "top_k"),
        ({"do_sample": True, "top_p": 0.9}, "top_p"),
    ],
)
def test_strategy_of_maps_each_configuration(overrides: dict[str, object], expected: str) -> None:
    """五种策略各有**一个**配置组合落在它上面（策略判定只有一个答案）."""
    assert generate.strategy_of(settings(**overrides)) == expected


def test_softmax_of_is_stable_and_rejects_degenerate_rows() -> None:
    """先减最大值：``z = [1000, 1001]`` 也得到 [0.269, 0.731] 而不是 nan."""
    probabilities = generate.softmax_of((1000.0, 1001.0))
    assert abs(math.fsum(probabilities) - 1.0) < 1e-12
    assert abs(probabilities[1] - 0.7310585786300049) < 1e-12
    with pytest.raises(NumericError):
        generate.softmax_of((float("-inf"), float("-inf")))


def test_temperature_is_a_division() -> None:
    """温度是**除**（HF 的原式），因此 T<1 更尖、T>1 更平."""
    assert generate.apply_temperature((2.0, -2.0), 2.0) == (1.0, -1.0)
    with pytest.raises(ParameterError):
        generate.apply_temperature((1.0,), 0.0)


def test_repetition_penalty_is_a_division_for_positive_logits() -> None:
    """正 logits 除以惩罚、负 logits 乘以惩罚（**写成减法会在负值上反向**）."""
    assert generate.apply_repetition_penalty((1.0, -1.0, 0.5), (0, 1), 2.0) == (0.5, -2.0, 0.5)
    assert generate.apply_repetition_penalty((1.0, -1.0), (0, 1), 1.0) == (1.0, -1.0)
    with pytest.raises(ParameterError):
        generate.apply_repetition_penalty((1.0,), (0,), 0.0)


def test_top_k_filter_keeps_the_largest_and_counts_them() -> None:
    """top-k 只留最大的 k 个（并列时按下标），保留个数是**常数**."""
    filtered, kept = generate.top_k_filter(generate.FILTER_LOGITS, 2)
    assert kept == 2
    assert filtered[0] == 2.0
    assert filtered[1] == 1.0
    assert all(value == LOGITS for value in filtered[2:])
    untouched, everything = generate.top_k_filter(generate.FILTER_LOGITS, 0)
    assert untouched == generate.FILTER_LOGITS
    assert everything == len(generate.FILTER_LOGITS)
    with pytest.raises(ParameterError):
        generate.top_k_filter(generate.FILTER_LOGITS, -1)


def test_top_k_ties_break_by_index() -> None:
    """并列时下标小的优先（与 ``torch.topk`` 的稳定序一致）."""
    filtered, kept = generate.top_k_filter((1.0, 1.0, 0.0), 1)
    assert kept == 1
    assert filtered[0] == 1.0
    assert filtered[1] == LOGITS


def test_top_p_filter_is_the_nucleus_of_hf() -> None:
    """top-p 留的是"累积概率刚好超过 p 的最小集合"——保留个数**由数据决定**."""
    _, tight = generate.top_p_filter(generate.FILTER_LOGITS, 0.5)
    _, loose = generate.top_p_filter(generate.FILTER_LOGITS, 0.9)
    assert tight == 1
    assert loose == 4
    assert tight < loose
    with pytest.raises(ParameterError):
        generate.top_p_filter(generate.FILTER_LOGITS, 0.0)


def test_top_p_always_keeps_at_least_one() -> None:
    """"至少留一个"是一条**恒等式**：极小的 p 也留一个（这是 HF 那行赋 0 的作用）."""
    for value in (1e-9, 0.01, 0.2):
        _, kept = generate.top_p_filter(generate.FILTER_LOGITS, value)
        assert kept >= generate.MIN_TOKENS_TO_KEEP


def test_transform_logits_chains_the_warpers_in_order() -> None:
    """四个 warper 按固定顺序串联：温度先缩放，top_k / top_p 再截断."""
    transformed, kept = generate.transform_logits(
        generate.FILTER_LOGITS, settings(do_sample=True, temperature=0.5), ()
    )
    assert kept == len(generate.FILTER_LOGITS)
    assert transformed[0] == 4.0
    _, both = generate.transform_logits(
        generate.FILTER_LOGITS, settings(do_sample=True, top_k=3, top_p=0.9), ()
    )
    assert both == 3


def test_distribution_entropy_anchors() -> None:
    """均匀分布的熵是 ``ln k``；退化分布是 0（**零项不靠 -inf 的极限**）."""
    assert abs(generate.distribution_entropy((0.5, 0.5)) - math.log(2.0)) < 1e-15
    assert generate.distribution_entropy((1.0, 0.0)) == 0.0


def test_greedy_index_and_sample_index() -> None:
    """贪心取最大值的**最小下标**；采样是逆变换（只依赖传入的均匀数）."""
    assert generate.greedy_index((1.0, 3.0, 3.0)) == 1
    assert generate.greedy_index((5.0, 1.0)) == 0
    assert generate.sample_index((0.5, 0.5), 0.0) == 0
    assert generate.sample_index((0.5, 0.5), 0.75) == 1
    with pytest.raises(ParameterError):
        generate.sample_index((0.5, 0.5), 1.0)


def test_log_softmax_is_the_log_of_softmax() -> None:
    """``log_softmax`` 逐元素等于 ``log(softmax)``（beam 用的是累加而不是连乘）."""
    log_probs = generate.log_softmax_of((1.0, 2.0))
    probabilities = generate.softmax_of((1.0, 2.0))
    assert all(abs(math.exp(a) - b) < 1e-15 for a, b in zip(log_probs, probabilities))


def test_generate_greedy_is_deterministic() -> None:
    """贪心完全确定，且每一步的保留个数等于词表大小（没有截断）。"""
    first = generate.generate(
        generate_module_logits,
        (1, 3),
        settings(max_new_tokens=4),
        vocab=samples.VOCAB,
    )
    second = generate.generate(
        generate_module_logits,
        (1, 3),
        settings(max_new_tokens=4),
        vocab=samples.VOCAB,
    )
    assert first.generated == second.generated
    assert first.strategy == "greedy"
    assert all(step.kept == samples.VOCAB for step in first.steps)


def test_generate_sampling_repeats_with_the_same_seed() -> None:
    """同一个种子给出同一串 token（**采样在"u 的来源"之外是纯函数**）."""
    config = settings(max_new_tokens=4, do_sample=True, seed=13)
    first = generate.generate(generate_module_logits, (1, 3), config, vocab=samples.VOCAB)
    second = generate.generate(generate_module_logits, (1, 3), config, vocab=samples.VOCAB)
    assert first.generated == second.generated
    other = generate.generate(
        generate_module_logits,
        (1, 3),
        settings(max_new_tokens=4, do_sample=True, seed=14),
        vocab=samples.VOCAB,
    )
    assert other.generated != first.generated


def test_generate_stops_at_eos() -> None:
    """命中 eos 时提前结束，并把原因写进 notes（"为什么少了几个 token"必须可答）."""
    result = generate.generate(
        generate_module_logits,
        (1, 3),
        settings(max_new_tokens=6, eos_token=7),
        vocab=samples.VOCAB,
    )
    assert result.stopped_early is True
    assert len(result.generated) == 1
    assert any("eos_token" in note for note in result.notes)


def test_generate_rejects_empty_prompt_and_bad_logits() -> None:
    """空 prompt 与长度不符的 logits 都拒绝（生成需要至少一个起点）。"""
    with pytest.raises(ShapeError):
        generate.generate(
            generate_module_logits, (), settings(max_new_tokens=2), vocab=samples.VOCAB
        )
    with pytest.raises(ShapeError):
        generate.generate(
            lambda tokens: (1.0, 2.0), (1,), settings(max_new_tokens=2), vocab=samples.VOCAB
        )


def test_generate_dispatches_to_beam() -> None:
    """``num_beams > 1`` 时自动走 beam 分支（策略由配置决定，不由调用方决定）。"""
    result = generate.generate(
        generate_module_logits,
        (2, 5),
        settings(max_new_tokens=4, num_beams=2),
        vocab=samples.VOCAB,
    )
    assert result.strategy == "beam"
    assert len(result.generated) == 4
    assert any("beam 宽度 2" in note for note in result.notes)


def test_beam_search_length_penalty_changes_the_normalised_score() -> None:
    """长度惩罚是**打分公式的一部分**：它改变归一化分数（而不是改变模型）."""
    short = generate.beam_search(
        generate_module_logits,
        (2, 5),
        settings(max_new_tokens=4, num_beams=2, length_penalty=0.5),
        vocab=samples.VOCAB,
    )
    long = generate.beam_search(
        generate_module_logits,
        (2, 5),
        settings(max_new_tokens=4, num_beams=2, length_penalty=2.0),
        vocab=samples.VOCAB,
    )
    assert short.notes[-1] != long.notes[-1]
    assert short.generated == long.generated  # 同一批候选，只是分数被重新加权


def test_beam_search_rejects_bad_inputs() -> None:
    """空 prompt 拒绝；``num_beams > 词表`` 由校验拒绝."""
    with pytest.raises(ShapeError):
        generate.beam_search(
            generate_module_logits, (), settings(max_new_tokens=2, num_beams=2),
            vocab=samples.VOCAB,
        )
    with pytest.raises(GenerationError):
        generate.beam_search(
            generate_module_logits, (1,), settings(max_new_tokens=2, num_beams=99),
            vocab=samples.VOCAB,
        )


@pytest.mark.parametrize(
    "logits",
    [
        "nope",
        (),
        (1.0, None),
        (1.0, True),
        (1.0, float("nan")),
        (1.0, float("-inf")),
    ],
)
def test_checked_logits_rejects_malformed_input(logits: object) -> None:
    """logits 的六种坏输入都在入口拒绝（**本包自己的族**，理由写在函数里）."""
    with pytest.raises((ShapeError, NumericError)):
        generate.checked_logits(logits)


def test_checked_logits_returns_floats() -> None:
    """整数 logits 被归一成浮点（整数与浮点混在一起会让"逐位"这件事变复杂）."""
    assert generate.checked_logits((1, 2)) == (1.0, 2.0)


def generate_module_logits(tokens: tuple[int, ...]) -> tuple[float, ...]:
    """测试用的"模型"：转发 :func:`hf_source.study.toy_logits`（**确定性**）."""
    from smart_research_agent.hf_source.study import toy_logits

    return toy_logits(tokens, vocab=samples.VOCAB)
