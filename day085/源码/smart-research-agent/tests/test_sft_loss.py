"""SFT 损失函数测试（day050）：交叉熵、屏蔽位与困惑度的精确值.

断言全部用**手算值**：``ln 2``、``ln 3``、``2.0``。损失函数是最不该
"看起来差不多"的地方——分母写错时，loss 依然是一个漂亮的数字。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.sft.encoding import IGNORE_INDEX
from smart_research_agent.sft.loss import (
    SFTLossError,
    cross_entropy,
    log_softmax,
    masked_cross_entropy,
    masked_token_accuracy,
    mean,
    perplexity,
    softmax,
)

LN2 = math.log(2)
LN3 = math.log(3)


class TestSoftmax:
    """softmax：和为一、平移不变、数值稳定."""

    def test_uniform_logits(self):
        probs = softmax([0.0, 0.0, 0.0, 0.0])
        assert probs == pytest.approx([0.25] * 4)

    def test_sums_to_one(self):
        assert sum(softmax([1.0, 2.0, 3.0])) == pytest.approx(1.0)

    def test_shift_invariance(self):
        assert softmax([1.0, 2.0]) == pytest.approx(softmax([101.0, 102.0]))

    def test_large_logits_do_not_overflow(self):
        """不减最大值时 ``exp(1000)`` 会溢出为 inf，softmax 变成 nan."""
        probs = softmax([1000.0, 1000.0])
        assert probs == pytest.approx([0.5, 0.5])
        assert not any(math.isnan(value) for value in probs)

    def test_empty_rejected(self):
        with pytest.raises(SFTLossError, match="不能为空"):
            softmax([])


class TestLogSoftmax:
    """log-softmax：与 ``log(softmax)`` 数学等价，但数值上更安全."""

    def test_matches_log_of_softmax(self):
        logits = [0.5, -1.0, 2.0]
        assert log_softmax(logits) == pytest.approx([math.log(p) for p in softmax(logits)])

    def test_exp_sums_to_one(self):
        assert sum(math.exp(v) for v in log_softmax([3.0, 1.0, -2.0])) == pytest.approx(1.0)

    def test_small_probability_stays_finite(self):
        """概率下溢为 0 时 ``log`` 会得到 -inf，而 log_softmax 仍是有限值."""
        result = log_softmax([0.0, -1000.0])
        assert all(math.isfinite(value) for value in result)

    def test_empty_rejected(self):
        with pytest.raises(SFTLossError, match="不能为空"):
            log_softmax([])


class TestCrossEntropy:
    """单点交叉熵：``-log p(target)``."""

    def test_uniform_two_way(self):
        assert cross_entropy([0.0, 0.0], 0) == pytest.approx(LN2)

    def test_uniform_three_way(self):
        assert cross_entropy([0.0, 0.0, 0.0], 2) == pytest.approx(LN3)

    def test_confident_correct_is_near_zero(self):
        assert cross_entropy([100.0, 0.0], 0) < 1e-6

    def test_confident_wrong_is_large(self):
        assert cross_entropy([100.0, 0.0], 1) >= 100.0

    def test_target_out_of_range_rejected(self):
        with pytest.raises(SFTLossError, match="落在 logits 范围"):
            cross_entropy([0.0, 0.0], 2)


class TestMaskedCrossEntropy:
    """带屏蔽位的平均交叉熵——**分母只数监督 token**."""

    def test_mean_over_supervised_positions_only(self):
        logits = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        labels = [0, IGNORE_INDEX, 1]
        loss, count = masked_cross_entropy(logits, labels)
        assert count == 2
        assert loss == pytest.approx(LN2)

    def test_denominator_is_not_sequence_length(self):
        """若分母误用序列长度，loss 会被系统性压低——这里把它钉死."""
        logits = [[0.0, 0.0]] * 4
        labels = [0, 0, IGNORE_INDEX, IGNORE_INDEX]
        loss, count = masked_cross_entropy(logits, labels)
        assert count == 2
        assert loss == pytest.approx(LN2)  # 而不是 LN2 * 2 / 4

    def test_all_masked_rejected(self):
        with pytest.raises(SFTLossError, match="没有任何监督位置"):
            masked_cross_entropy([[0.0, 0.0]], [IGNORE_INDEX])

    def test_length_mismatch_rejected(self):
        with pytest.raises(SFTLossError, match="长度不一致"):
            masked_cross_entropy([[0.0, 0.0]], [0, 0])

    def test_mixed_difficulty_averages(self):
        """一条很确定 + 一条完全不确定 → 平均值介于两者之间."""
        logits = [[100.0, 0.0], [0.0, 0.0]]
        labels = [0, 0]
        loss, count = masked_cross_entropy(logits, labels)
        assert count == 2
        assert loss == pytest.approx(LN2 / 2, abs=1e-6)


class TestMaskedTokenAccuracy:
    """监督位置上的下一 token 命中率."""

    def test_hit_and_miss(self):
        accuracy, count = masked_token_accuracy(
            [[1.0, 0.0], [1.0, 0.0]], [0, 1]
        )
        assert count == 2
        assert accuracy == pytest.approx(0.5)

    def test_all_hits(self):
        accuracy, count = masked_token_accuracy([[1.0, 0.0], [0.0, 5.0]], [0, 1])
        assert (accuracy, count) == (1.0, 2)

    def test_ignored_positions_not_counted(self):
        accuracy, count = masked_token_accuracy(
            [[0.0, 1.0], [1.0, 0.0]], [IGNORE_INDEX, 0]
        )
        assert (accuracy, count) == (1.0, 1)

    def test_tie_picks_first_index(self):
        """并列时取第一个下标（``max`` 的稳定性），这条行为必须是确定的."""
        accuracy, _ = masked_token_accuracy([[1.0, 1.0]], [0])
        assert accuracy == 1.0

    def test_all_masked_rejected(self):
        with pytest.raises(SFTLossError, match="没有任何监督位置"):
            masked_token_accuracy([[0.0, 0.0]], [IGNORE_INDEX])

    def test_length_mismatch_rejected(self):
        with pytest.raises(SFTLossError, match="长度不一致"):
            masked_token_accuracy([[0.0, 0.0]], [])


class TestPerplexity:
    """困惑度：``exp(loss)``，单位是"候选个数"，比 nats 好解释."""

    def test_exponential(self):
        assert perplexity(LN2) == pytest.approx(2.0)

    def test_zero_loss_is_one(self):
        assert perplexity(0.0) == 1.0

    def test_uniform_over_vocab(self):
        """均匀分布时困惑度等于词表大小."""
        assert perplexity(math.log(557)) == pytest.approx(557.0)

    def test_negative_rejected(self):
        with pytest.raises(SFTLossError, match="不能为负数"):
            perplexity(-0.1)

    def test_huge_loss_saturates(self):  # pragma: no cover - 防御式分支
        assert perplexity(1e6) == math.inf


class TestMean:
    """安全求均值：空序列必须失败而不是返回 0."""

    def test_average(self):
        assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_single(self):
        assert mean([7.0]) == 7.0

    def test_empty_rejected(self):
        with pytest.raises(SFTLossError, match="不能对空序列求均值"):
            mean([])
