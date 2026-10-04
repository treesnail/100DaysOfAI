"""``hf_source.verify``：七条性质与两类比法（day085 / M7-D9）."""

from __future__ import annotations

import dataclasses

import pytest

from smart_research_agent.hf_source import attention, verify
from smart_research_agent.hf_source.errors import AssemblyError, NumericError, ShapeError
from smart_research_agent.hf_source.types import PROPERTY_PRE_NORM_BLOCK_MATCHES_DAY079
from tests import hf_samples as samples


def forward(heads: int = 1, causal: bool = True):
    """一次写死的前向（``tokens=4、hidden=6``）."""
    shape = samples.shape_of("gpt2" if causal else "bert", heads=heads)
    return attention.hf_attention(samples.attention_parameters(), samples.inputs(), shape, causal=causal)


def test_rows_are_distributions_passes_on_a_real_forward() -> None:
    """真前向：行和误差恰好 0（``fsum`` 的求和与归一化同源）."""
    outcome = verify.check_rows_are_distributions(forward())
    assert outcome.passed is True
    assert outcome.applicable is True
    assert "4 行" in outcome.evidence[0]


@pytest.mark.parametrize("row", [(-0.1, 0.5, 0.6), (float("nan"), 0.5, 0.5)])
def test_rows_are_distributions_rejects_bad_rows(row: tuple[float, ...]) -> None:
    """负数与非有限数都**当场抛**（不是"记一条失败"）——它们说明归一化之前动过手."""
    broken = dataclasses.replace(forward(), weights=((row,) * samples.TOKENS,))
    with pytest.raises(NumericError):
        verify.check_rows_are_distributions(broken)


def test_rows_are_distributions_can_fail() -> None:
    """行和不是 1 时是"失败"而不是"抛错"（两类结论必须分得开）."""
    broken = dataclasses.replace(forward(), weights=(((0.2, 0.2, 0.2),) * samples.TOKENS,))
    outcome = verify.check_rows_are_distributions(broken)
    assert outcome.passed is False


def test_masked_entries_are_exact_zero() -> None:
    """因果掩码下 6 个格子恰好是 0；双向时**不适用**（并说明没有东西可查）."""
    causal = verify.check_masked_entries_are_exact_zero(forward(causal=True))
    assert causal.passed is True
    assert "6 个格子" in causal.evidence[0]
    bidirectional = verify.check_masked_entries_are_exact_zero(forward(causal=False))
    assert bidirectional.applicable is False
    assert "没有东西可查" in bidirectional.evidence[0]


def test_masked_entries_can_fail() -> None:
    """把被挡格子填成 0.001 ⇒ 这条性质必须亮红（否则它没有分辨力）."""
    rows = []
    for index in range(samples.TOKENS):
        rows.append(tuple(0.001 if column > index else 0.25 for column in range(samples.TOKENS)))
    broken = dataclasses.replace(forward(), weights=(tuple(rows),))
    assert verify.check_masked_entries_are_exact_zero(broken).passed is False


def test_causal_prefix_is_stable() -> None:
    """扰动最后一行：前 3 行逐位不变，被扰动那一行确实变了."""
    outcome = verify.check_causal_prefix_is_stable(
        samples.attention_parameters(), samples.inputs(), samples.shape_of("gpt2")
    )
    assert outcome.passed is True
    assert "3/3" in outcome.evidence[1]
    bidirectional = verify.check_causal_prefix_is_stable(
        samples.attention_parameters(), samples.inputs(), samples.shape_of("bert")
    )
    assert bidirectional.applicable is False


def test_split_merge_round_trip() -> None:
    """分头再拼回逐位还原（heads=2 与 heads=3 都查一遍）."""
    for heads in (2, 3):
        outcome = verify.check_split_merge_round_trip(tuple(samples.inputs()), heads)
        assert outcome.passed is True


def test_fused_matches_separate() -> None:
    """融合投影切回去与三次独立投影逐位相同（**三条路径的最大差 0**）."""
    outcome = verify.check_fused_matches_separate(samples.attention_parameters(), samples.inputs())
    assert outcome.passed is True
    assert outcome.evidence[1].endswith("True")
    assert outcome.evidence[2].startswith("三条投影路径的最大差 0")


def test_single_head_matches_multi_head_is_bit_exact_in_this_sample() -> None:
    """跨天对账实测**逐位相同**（判据仍是容差——不把"偶然逐位"写成承诺）."""
    outcome = verify.check_single_head_matches_multi_head(
        samples.attention_parameters(), samples.inputs(), samples.shape_of("gpt2"), causal=False
    )
    assert outcome.passed is True
    assert outcome.cross_check is not None
    assert outcome.cross_check.max_gap <= verify.SINGLE_HEAD_TOLERANCE
    assert "multi_head.layers" in outcome.cross_check.line()
    multi = verify.check_single_head_matches_multi_head(
        samples.attention_parameters(), samples.inputs(), samples.shape_of("gpt2", heads=2)
    )
    assert multi.applicable is False


def test_pre_norm_block_matches_day079() -> None:
    """pre 摆放的本包块与 day079 的 encoder_block 一致（同一份参数、逐位对账）."""
    outcome = verify.check_pre_norm_block_matches_day079(
        samples.block_parameters(),
        samples.attention_parameters(),
        samples.inputs(),
        samples.shape_of("gpt2"),
    )
    assert outcome.passed is True
    assert outcome.cross_check is not None
    assert "encoder_decoder.layers" in outcome.cross_check.line()
    multi = verify.check_pre_norm_block_matches_day079(
        samples.block_parameters(),
        samples.attention_parameters(),
        samples.inputs(),
        samples.shape_of("gpt2", heads=2),
    )
    assert multi.applicable is False


def test_property_outcome_forbids_inapplicable_but_passed() -> None:
    """``applicable=False`` 与 ``passed=True`` 不许同时出现（构造期就拒绝）."""
    with pytest.raises(NumericError):
        verify.PropertyOutcome(name="x", applicable=False, passed=True)


def test_property_outcome_lines() -> None:
    """两行文本各自带上自己的结论（不适用也要印出来）."""
    passed = verify.PropertyOutcome(name="a", applicable=True, passed=True, evidence=("e",))
    assert passed.line().startswith("[通过] a")
    skipped = verify.PropertyOutcome(name="b", applicable=False, passed=False, evidence=("没有",))
    assert skipped.line().startswith("[不适用] b")


def test_property_report_ok_and_require_ok() -> None:
    """``ok`` 只看适用的那些；失败时 ``require_ok`` 抛错并带上原文."""
    good = verify.PropertyReport(
        outcomes=(
            verify.PropertyOutcome(name="a", applicable=True, passed=True),
            verify.PropertyOutcome(name="b", applicable=False, passed=False),
        )
    )
    assert good.ok is True
    good.require_ok()
    assert good.to_dict()["counts"] == {"total": 2, "applicable": 1}
    bad = verify.PropertyReport(
        outcomes=(verify.PropertyOutcome(name="a", applicable=True, passed=False, evidence=("坏了",)),)
    )
    assert bad.ok is False
    with pytest.raises(AssemblyError):
        bad.require_ok()


def test_property_report_lines_put_inapplicable_first() -> None:
    """报告里"不适用"排在前面（让"这一次没查它"一眼可见）."""
    report = verify.PropertyReport(
        outcomes=(
            verify.PropertyOutcome(name="a", applicable=True, passed=True),
            verify.PropertyOutcome(name="b", applicable=False, passed=False, evidence=("无",)),
        )
    )
    assert report.lines()[0].startswith("[不适用]")


def test_cross_check_passed_and_line() -> None:
    """两条实现之间的读数：差与容差都在行里（否则这个数无法被追溯）."""
    check = verify.CrossCheck(name="n", left="l", right="r", max_gap=1e-15, tolerance=1e-12)
    assert check.passed is True
    assert "一致" in check.line()
    assert "最大差 1.000e-15" in check.line()
    over = verify.CrossCheck(name="n", left="l", right="r", max_gap=1.0, tolerance=1e-12)
    assert over.passed is False
    assert "不一致" in over.line()


def test_check_all_runs_the_declared_list() -> None:
    """给块参数时跑满七条；不给时恰好少那一条跨天对账."""
    full = verify.check_all(
        samples.attention_parameters(),
        samples.inputs(),
        samples.shape_of("gpt2"),
        block_params=samples.block_parameters(),
    )
    assert full.ok is True
    assert len(full.outcomes) == 7
    partial = verify.check_all(
        samples.attention_parameters(), samples.inputs(), samples.shape_of("gpt2")
    )
    assert len(partial.outcomes) == 6
    assert PROPERTY_PRE_NORM_BLOCK_MATCHES_DAY079 not in {
        outcome.name for outcome in partial.outcomes
    }


def test_check_all_detects_a_broken_property_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """名单与报告对不上时当场拒绝（**报告里安静地少一行是最坏的失败**）."""
    monkeypatch.setattr(verify, "SOURCE_PROPERTIES", ("only_one_name",))
    with pytest.raises(ShapeError):
        verify.check_all(
            samples.attention_parameters(), samples.inputs(), samples.shape_of("gpt2")
        )


def test_default_inputs_rejects_non_positive() -> None:
    """样本的构造参数必须为正."""
    with pytest.raises(ShapeError):
        verify.default_inputs(0, 6)
    with pytest.raises(ShapeError):
        verify.default_inputs(4, 0)


def test_scale_agreement_line_is_bit_exact() -> None:
    """三处缩放系数由同一个 ``1/√head_dim`` 派生，因此必须逐位相同."""
    line = verify.scale_agreement_line(3)
    assert "逐位相同：True" in line
    assert "hf_scale" in line and "head_scale" in line and "softmax_shape_scale" in line
