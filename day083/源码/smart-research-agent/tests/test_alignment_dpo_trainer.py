"""DPO 训练器测试（day054 C 组）：``dpo_step`` 与 ``train_dpo`` 的逐步历史.

本文件钉住的是"**在参考模型上真的走几步 DPO**"这条链路上的四件事：

- **起点可自检**：策略 = 参考模型的副本时，``loss`` 必然等于 ``ln 2``
  （margin 恰好为 0）。这条断言在"梯度符号写反"的事故里也能通过
  （那时方向还没显形），所以它只是一个起点，不是全部；
- **梯度方向**：更新一次之后，策略对 chosen 的序列对数概率必须**上升**、
  对 rejected 必须**下降**，于是 margin 由 0 变成正数。源码注释里记着
  符号写反的事故（loss 从 ``ln 2`` 缓慢涨到几十而没有任何地方报错），
  本文件用 ``sequence_logprob`` 前后各量一次，把方向钉成回归测试；
- **分母交出来**：``positions`` 是"参与计分的位置数之和"，必须等于
  ``Σ(len(ids) − 1)``（实测 763）。用批数或序列长度做分母会让有效学习率
  被悄悄缩放，而日志里的 ``learning_rate`` 看不出任何异常；
- **历史比最终值重要**：``train_dpo`` 返回逐步历史（训练 margin / loss /
  留出准确率 / 留出 margin / KL）。默认训练 42 步把留出准确率从 0.0 抬到
  0.8；而过训练对照（lr=2.0、140 步）把 KL 抬到 2.43（54 倍）、留出 margin
  抬到 0.91，**留出准确率一点没涨**——"继续训练买到了什么"的答案在历史里。

实测基线（缺省配置 β=0.1 / lr=0.5 / 6 轮 / seed=42，词表 460、训练 7 条 /
留出 5 条）：``initial_loss = ln 2``、``final_loss = 0.6715892805253807``、
``best_valid_accuracy = 0.8``、``best_step = 2``。这些数字写死在各用例的
docstring 里，改动实现会立刻在这里显形。
"""

from __future__ import annotations

import dataclasses
import json
import math

import pytest

from smart_research_agent.alignment import SEED_PAIRS, split_preferences
from smart_research_agent.alignment.dpo_trainer import (
    DPOStepRecord,
    DPOReport,
    dpo_step,
    train_dpo,
)
from smart_research_agent.finetune_eval.metrics import FinetuneEvalError
from smart_research_agent.finetune_eval.model_probe import sequence_logprob
from smart_research_agent.sft.encoding import CharTokenizer
from smart_research_agent.sft.reference_model import ReferenceSFTModel

PAIRS = list(SEED_PAIRS)
TRAIN, VALID = split_preferences(PAIRS)
TOKENIZER = CharTokenizer.from_texts([p.prompt + p.chosen + p.rejected for p in PAIRS])
REFERENCE = ReferenceSFTModel(TOKENIZER.vocab_size, seed=42)


def fresh_policy() -> ReferenceSFTModel:
    """初始策略 = 参考模型的副本（DPO 的起点）。"""
    return ReferenceSFTModel.from_state(REFERENCE.state_dict())


#: margin 为 0 时的 DPO loss：``−log σ(0) = ln 2``（起点自检的参照值）.
LN2 = math.log(2.0)

#: 默认配置与其实测读数（写死在断言里，改动实现会立刻显形）.
DEFAULT_BETA = 0.1
DEFAULT_LEARNING_RATE = 0.5
DEFAULT_EPOCHS = 6
DEFAULT_STEPS = 42
DEFAULT_FINAL_LOSS = 0.6715892805253807
DEFAULT_LOSS_DROP = 0.021557900034564592
DEFAULT_LAST_MARGIN = 0.43590801967513926
DEFAULT_LAST_VALID_MARGIN = 0.07940482870061487
DEFAULT_LAST_KL = 0.044928325994291536
DEFAULT_FIRST_ACCURACY = 0.6
BEST_VALID_ACCURACY = 0.8
BEST_STEP = 2

#: 一个批（全体训练对）参与计分的位置数之和（实测值）.
TOTAL_POSITIONS = 763

#: 过训练对照的配置与实测读数（kl 涨 54 倍，留出准确率一步没涨）.
OVER_LEARNING_RATE = 2.0
OVER_EPOCHS = 20
OVER_STEPS = 140
OVER_LAST_KL = 2.427636334873461
OVER_LAST_VALID_MARGIN = 0.908943477970405
OVER_LAST_TRAIN_MARGIN = 5.463145056162801


def make_record(
    step: int,
    *,
    mean_margin: float = 0.0,
    mean_loss: float = LN2,
    valid_accuracy: float = 0.0,
    kl: float = 0.0,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    valid_margin: float = 0.0,
) -> DPOStepRecord:
    """构造一条 ``DPOStepRecord``（只给关心的字段，其余取中性值）."""
    return DPOStepRecord(
        step=step,
        mean_margin=mean_margin,
        mean_loss=mean_loss,
        valid_accuracy=valid_accuracy,
        kl=kl,
        learning_rate=learning_rate,
        valid_margin=valid_margin,
    )


@pytest.fixture(scope="module")
def default_report() -> DPOReport:
    """默认配置（β 0.1 / lr 0.5 / 6 轮 / seed 42）的完整历史：42 步.

    模块内共享一次——每个训练步骤都要在 5 条留出对上算准确率、在 7 条训练对
    上算 KL，重跑一遍的代价是秒钟级。
    """
    return train_dpo(
        fresh_policy(),
        REFERENCE,
        TOKENIZER,
        TRAIN,
        valid_pairs=VALID,
        beta=DEFAULT_BETA,
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_EPOCHS,
        seed=42,
    )


@pytest.fixture(scope="module")
def over_trained_report() -> DPOReport:
    """过训练对照（β 0.1 / lr 2.0 / 20 轮）的完整历史：140 步."""
    return train_dpo(
        fresh_policy(),
        REFERENCE,
        TOKENIZER,
        TRAIN,
        valid_pairs=VALID,
        beta=DEFAULT_BETA,
        learning_rate=OVER_LEARNING_RATE,
        epochs=OVER_EPOCHS,
        seed=42,
    )


class TestDpoStepValidation:
    """``dpo_step`` 的参数校验：空批与负学习率立刻报错，``lr=0`` 才是合法的"只测量"."""

    def test_empty_pairs_raises(self):
        """空批抛错：没有偏好对就没有 margin，DPO 的 loss 无从定义."""
        with pytest.raises(FinetuneEvalError, match="至少要有一条偏好对"):
            dpo_step(
                fresh_policy(),
                REFERENCE,
                TOKENIZER,
                [],
                beta=DEFAULT_BETA,
                learning_rate=0.0,
            )

    @pytest.mark.parametrize("learning_rate", [-1.0, -1e-6, -0.5, -100.0])
    def test_negative_learning_rate_raises(self, learning_rate):
        """负学习率含义不明确（"反向更新"还是"写错了"），直接拒绝."""
        with pytest.raises(FinetuneEvalError, match="不能为负数"):
            dpo_step(
                fresh_policy(),
                REFERENCE,
                TOKENIZER,
                TRAIN,
                beta=DEFAULT_BETA,
                learning_rate=learning_rate,
            )

    def test_zero_learning_rate_measures_without_updating(self):
        """``lr=0`` 是**允许**的：语义是"只测量、不更新"（起点检查靠它）."""
        policy = fresh_policy()
        before = policy.state_dict()
        result = dpo_step(
            policy, REFERENCE, TOKENIZER, TRAIN, beta=DEFAULT_BETA, learning_rate=0.0
        )
        assert result["updated"] is False
        assert result["updates"] == 0
        assert policy.state_dict() == before

    def test_second_measurement_keeps_updates_at_zero(self):
        """再测一次仍然是 0：**"测了一次"不是"更新了一次"**.

        把测量算成更新会让"这个模型训了多少步"这个数字失去意义。
        """
        policy = fresh_policy()
        first = dpo_step(
            policy, REFERENCE, TOKENIZER, TRAIN, beta=DEFAULT_BETA, learning_rate=0.0
        )
        second = dpo_step(
            policy, REFERENCE, TOKENIZER, TRAIN, beta=DEFAULT_BETA, learning_rate=0.0
        )
        assert first["updates"] == 0
        assert second["updates"] == 0
        assert second["updated"] is False
        assert second["mean_loss"] == first["mean_loss"]
        assert policy.state_dict().updates == 0

    @pytest.mark.parametrize("learning_rate", [1e-9, 0.5, 1.0, 5.0])
    def test_positive_learning_rate_counts_one_update(self, learning_rate):
        """``lr > 0`` 时 ``updates`` 恰好递增 1（学习率多小都算真的更新过）."""
        policy = fresh_policy()
        result = dpo_step(
            policy,
            REFERENCE,
            TOKENIZER,
            TRAIN,
            beta=DEFAULT_BETA,
            learning_rate=learning_rate,
        )
        assert result["updated"] is True
        assert result["updates"] == 1
        assert policy.state_dict().updates == 1

    def test_zero_learning_rate_leaves_parameters_bitwise_identical(self):
        """``lr=0`` 连一个浮点位都不动：测量与训练走同一条路径，但只测量就不写回."""
        policy = fresh_policy()
        before = policy.state_dict()
        dpo_step(policy, REFERENCE, TOKENIZER, TRAIN, beta=DEFAULT_BETA, learning_rate=0.0)
        after = policy.state_dict()
        assert after.weights == before.weights
        assert after.bias == before.bias


class TestDpoStepReadings:
    """``dpo_step`` 的读数：起点 loss 是 ``ln 2``、分母 ``positions`` 交出来."""

    def test_result_keys_are_complete(self):
        """返回字典恰好六个键（读数全交出来，调用方不必再猜）."""
        result = dpo_step(
            fresh_policy(), REFERENCE, TOKENIZER, TRAIN, beta=DEFAULT_BETA, learning_rate=0.0
        )
        assert set(result) == {
            "mean_loss",
            "mean_margin",
            "mean_gradient",
            "positions",
            "updates",
            "updated",
        }

    @pytest.mark.parametrize("beta", [0.05, 0.1, 0.5, 1.0, 2.0])
    def test_initial_loss_is_ln_two_for_every_beta(self, beta):
        """起点语义：策略与参考模型逐位相同 ⇒ margin 为 0 ⇒ loss 恒为 ``ln 2``.

        **换 β 不改变这个结论**——β 只是把 0 乘了一下，这正是"起点检查"
        能抓住"ref 或 β 接错了"的原因。
        """
        result = dpo_step(
            fresh_policy(), REFERENCE, TOKENIZER, TRAIN, beta=beta, learning_rate=0.0
        )
        assert result["mean_loss"] == pytest.approx(LN2, abs=1e-12)

    def test_initial_margin_is_zero(self):
        """起点 margin 严格为 0（四个对数概率两两相同，差为 0）."""
        result = dpo_step(
            fresh_policy(), REFERENCE, TOKENIZER, TRAIN, beta=DEFAULT_BETA, learning_rate=0.0
        )
        assert result["mean_margin"] == 0.0

    @pytest.mark.parametrize("beta", [0.05, 0.1, 0.5, 1.0, 2.0])
    def test_mean_gradient_is_minus_half_beta(self, beta):
        """``∂L/∂margin = −β·σ(−β·margin)``，margin=0 时等于 ``−β/2``.

        实测 β=0.1 时是 ``-0.049999999999999996``——这条断言同时也是
        "梯度算的是对数概率而不是交叉熵"的间接证据（符号写反时这里是 +β/2）。
        """
        result = dpo_step(
            fresh_policy(), REFERENCE, TOKENIZER, TRAIN, beta=beta, learning_rate=0.0
        )
        assert result["mean_gradient"] == pytest.approx(-beta / 2, abs=1e-12)

    def test_positions_counts_every_scored_position(self):
        """``positions == Σ(len(ids) − 1)``（chosen 与 rejected 一起数）.

        实测 763。它是**分母**：用批数或序列长度做分母会让有效学习率被
        悄悄缩放，而日志里的 ``learning_rate`` 看不出任何异常。
        """
        expected = sum(
            len(TOKENIZER.encode(pair.chosen)) - 1 + len(TOKENIZER.encode(pair.rejected)) - 1
            for pair in TRAIN
        )
        result = dpo_step(
            fresh_policy(), REFERENCE, TOKENIZER, TRAIN, beta=DEFAULT_BETA, learning_rate=0.0
        )
        assert expected == TOTAL_POSITIONS == 763
        assert result["positions"] == expected

    def test_positions_cross_checked_by_sequence_logprob(self):
        """用探针的 ``sequence_logprob`` 复核分母（两个模块的口径必须一致）."""
        expected = sum(
            sequence_logprob(REFERENCE, TOKENIZER.encode(pair.chosen))[1]
            + sequence_logprob(REFERENCE, TOKENIZER.encode(pair.rejected))[1]
            for pair in TRAIN
        )
        assert expected == TOTAL_POSITIONS

    def test_positions_is_independent_of_learning_rate(self):
        """分母与学习率无关：``lr=0`` 与 ``lr=1`` 给出同一个 ``positions``."""
        measured = dpo_step(
            fresh_policy(), REFERENCE, TOKENIZER, TRAIN, beta=DEFAULT_BETA, learning_rate=0.0
        )
        trained = dpo_step(
            fresh_policy(), REFERENCE, TOKENIZER, TRAIN, beta=DEFAULT_BETA, learning_rate=1.0
        )
        assert measured["positions"] == trained["positions"] == TOTAL_POSITIONS

    def test_updates_reads_the_model_not_the_call(self):
        """``updates`` 读的是模型状态：连续两次更新之后应当读到 2."""
        policy = fresh_policy()
        dpo_step(policy, REFERENCE, TOKENIZER, TRAIN[:1], beta=DEFAULT_BETA, learning_rate=1.0)
        second = dpo_step(
            policy, REFERENCE, TOKENIZER, TRAIN[:1], beta=DEFAULT_BETA, learning_rate=1.0
        )
        assert second["updates"] == 2
        assert policy.state_dict().updates == 2


class TestGradientDirection:
    """梯度方向（本课最重要的一条回归测试）：chosen 上升、rejected 下降.

    源码注释里记着一次真实事故：把 chosen 的梯度按"交叉熵方向"累加，会让
    p(chosen) 一路**下降**、p(rejected) 一路上升，loss 从 ``ln 2`` 单调涨到
    几十，而代码没有任何地方报错。**起点 loss = ln 2 在两种符号下都成立**
    （那时 margin 恰好为 0），只有走一步才能看出来。
    """

    def test_chosen_logprob_rises_after_one_update(self):
        """更新一次之后，策略对 chosen 的序列对数概率必须**上升**."""
        policy = fresh_policy()
        pair = TRAIN[0]
        before = sequence_logprob(policy, TOKENIZER.encode(pair.chosen))[0]
        dpo_step(policy, REFERENCE, TOKENIZER, [pair], beta=DEFAULT_BETA, learning_rate=1.0)
        after = sequence_logprob(policy, TOKENIZER.encode(pair.chosen))[0]
        assert after > before

    def test_rejected_logprob_falls_after_one_update(self):
        """同一次更新里，策略对 rejected 的序列对数概率必须**下降**."""
        policy = fresh_policy()
        pair = TRAIN[0]
        before = sequence_logprob(policy, TOKENIZER.encode(pair.rejected))[0]
        dpo_step(policy, REFERENCE, TOKENIZER, [pair], beta=DEFAULT_BETA, learning_rate=1.0)
        after = sequence_logprob(policy, TOKENIZER.encode(pair.rejected))[0]
        assert after < before

    def test_margin_turns_positive_after_one_update(self):
        """margin 从 0 变成**正数**（这一对现在被判对了）."""
        policy = fresh_policy()
        pair = TRAIN[0]
        first = dpo_step(
            policy, REFERENCE, TOKENIZER, [pair], beta=DEFAULT_BETA, learning_rate=1.0
        )
        second = dpo_step(
            policy, REFERENCE, TOKENIZER, [pair], beta=DEFAULT_BETA, learning_rate=0.0
        )
        assert first["mean_margin"] == 0.0
        assert second["mean_margin"] > 0.0

    def test_loss_drops_below_ln_two_after_one_update(self):
        """方向正确时 loss 低于 ``ln 2``（写反时它会**高于** ``ln 2``）."""
        policy = fresh_policy()
        pair = TRAIN[0]
        dpo_step(policy, REFERENCE, TOKENIZER, [pair], beta=DEFAULT_BETA, learning_rate=1.0)
        measured = dpo_step(
            policy, REFERENCE, TOKENIZER, [pair], beta=DEFAULT_BETA, learning_rate=0.0
        )
        assert measured["mean_loss"] < LN2

    @pytest.mark.parametrize("index", [0, 1, 2, 3, 4, 5, 6])
    def test_direction_holds_for_every_training_pair(self, index):
        """七条训练对上逐条验证方向（不是只有第一条对）."""
        policy = fresh_policy()
        pair = TRAIN[index]
        chosen_before = sequence_logprob(policy, TOKENIZER.encode(pair.chosen))[0]
        rejected_before = sequence_logprob(policy, TOKENIZER.encode(pair.rejected))[0]
        dpo_step(policy, REFERENCE, TOKENIZER, [pair], beta=DEFAULT_BETA, learning_rate=1.0)
        chosen_after = sequence_logprob(policy, TOKENIZER.encode(pair.chosen))[0]
        rejected_after = sequence_logprob(policy, TOKENIZER.encode(pair.rejected))[0]
        assert chosen_after > chosen_before
        assert rejected_after < rejected_before
        assert chosen_after - rejected_after > chosen_before - rejected_before


class TestTrainDpoValidation:
    """``train_dpo`` 的参数校验：空训练集 / **空验证集** / 非正轮数都要报错."""

    def test_empty_train_pairs_raises(self):
        """空训练集抛错（没有可优化的样本）."""
        with pytest.raises(FinetuneEvalError, match="训练至少需要一条训练偏好对"):
            train_dpo(
                fresh_policy(),
                REFERENCE,
                TOKENIZER,
                [],
                valid_pairs=VALID,
                beta=DEFAULT_BETA,
                learning_rate=DEFAULT_LEARNING_RATE,
            )

    def test_empty_valid_pairs_raises(self):
        """**空验证集必须抛错**：没有留出数据就无法判断何时停止.

        偏好训练的过优化曲线（验证准确率先升后降）只能在留出数据上量；
        一个空验证集会让人用训练 margin 当早停判据——那等价于"训练 loss
        降到 0 就停"，是 DPO 最常见的误用方式。
        """
        with pytest.raises(FinetuneEvalError, match="留出偏好对"):
            train_dpo(
                fresh_policy(),
                REFERENCE,
                TOKENIZER,
                TRAIN,
                valid_pairs=[],
                beta=DEFAULT_BETA,
                learning_rate=DEFAULT_LEARNING_RATE,
            )

    @pytest.mark.parametrize("epochs", [0, -1, -6])
    def test_epochs_below_one_raises(self, epochs):
        """``epochs < 1`` 抛错（0 轮不是"不训练"，是参数写错了）."""
        with pytest.raises(FinetuneEvalError, match="epochs 必须为正整数"):
            train_dpo(
                fresh_policy(),
                REFERENCE,
                TOKENIZER,
                TRAIN,
                valid_pairs=VALID,
                beta=DEFAULT_BETA,
                learning_rate=DEFAULT_LEARNING_RATE,
                epochs=epochs,
            )

    def test_one_epoch_is_allowed(self):
        """``epochs=1`` 合法：步数恰好等于训练对条数（每轮每条一对走一步）."""
        report = train_dpo(
            fresh_policy(),
            REFERENCE,
            TOKENIZER,
            TRAIN,
            valid_pairs=VALID,
            beta=DEFAULT_BETA,
            learning_rate=DEFAULT_LEARNING_RATE,
            epochs=1,
            seed=42,
        )
        assert report.steps == len(TRAIN) == 7
        assert report.epochs == 1


class TestDefaultTrainingHistory:
    """默认训练的逐步历史：42 步、起点 ``ln 2``、最优留出准确率 0.8@step 2."""

    def test_steps_is_42(self, default_report):
        """6 轮 × 7 条训练对 = 42 次参数更新."""
        assert default_report.steps == DEFAULT_STEPS == 42
        assert len(default_report.records) == 42
        assert default_report.epochs == DEFAULT_EPOCHS
        assert default_report.train_pairs == len(TRAIN) == 7
        assert default_report.valid_pairs == len(VALID) == 5

    def test_initial_loss_is_ln_two(self, default_report):
        """``initial_loss`` 是**更新之前**那一次的平均 loss，必然等于 ``ln 2``."""
        assert default_report.initial_loss == pytest.approx(LN2, abs=1e-12)
        assert default_report.zero_margin_loss == pytest.approx(LN2, abs=1e-12)

    def test_final_loss_matches_the_measurement(self, default_report):
        """末步平均 loss 实测 0.6715892805253807（约 3% 的降幅）."""
        assert default_report.final_loss == pytest.approx(DEFAULT_FINAL_LOSS, rel=1e-6)
        assert default_report.final_loss < default_report.initial_loss

    def test_loss_drop_matches_the_measurement(self, default_report):
        """``loss_drop == initial_loss − final_loss``（实测 +0.0215579）."""
        assert default_report.loss_drop == pytest.approx(DEFAULT_LOSS_DROP, rel=1e-6)
        assert default_report.loss_drop == pytest.approx(
            default_report.initial_loss - default_report.final_loss, rel=1e-12
        )

    def test_best_valid_accuracy_and_step(self, default_report):
        """留出准确率 5 条 → 0.2 一格；最佳 0.8 出现在第 2 步.

        ``best_step`` 是"该在哪停"的候选：第 2 步之后 40 步的训练没有买到
        任何留出收益（KL 却涨了 130 倍）。
        """
        assert default_report.best_valid_accuracy == BEST_VALID_ACCURACY == 0.8
        assert default_report.best_step == BEST_STEP == 2

    def test_first_record_reads_the_untouched_start_point(self, default_report):
        """第一条记录：训练 margin 为 0、loss 为 ``ln 2``、准确率 0.6.

        准确率在"模型还没动"时就已经不是 0——**偏好准确率衡量的是"策略是否
        比参考模型更偏好 chosen"的排序信号，起点上 5 条里有 3 条恰好判对**。
        """
        first = default_report.records[0]
        assert first.step == 1
        assert first.mean_margin == 0.0
        assert first.mean_loss == pytest.approx(LN2, abs=1e-12)
        assert first.valid_accuracy == DEFAULT_FIRST_ACCURACY == 0.6

    def test_record_steps_are_consecutive(self, default_report):
        """``step`` 从 1 连续到 ``steps``（历史可以被逐行对比）."""
        assert [record.step for record in default_report.records] == list(
            range(1, DEFAULT_STEPS + 1)
        )

    def test_records_echo_the_configuration(self, default_report):
        """每条记录都带着这次训练的学习率（报告不必回头去猜配置）."""
        assert {record.learning_rate for record in default_report.records} == {
            DEFAULT_LEARNING_RATE
        }

    def test_valid_accuracy_takes_multiples_of_one_fifth(self, default_report):
        """5 条留出对 ⇒ 准确率只可能取 0.2 的倍数（粗但不可骗）."""
        values = {record.valid_accuracy for record in default_report.records}
        assert values <= {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
        assert max(values) == BEST_VALID_ACCURACY

    def test_kl_ends_higher_than_it_starts(self, default_report):
        """KL 的末值高于首值（策略确实偏离了参考模型）.

        实测 0.000332 → 0.044928（约 135 倍）。**这个序列不是单调的**
        （见下一条），所以这里只断言"末值 > 首值"。
        """
        kls = [record.kl for record in default_report.records]
        assert kls[-1] > kls[0]
        assert kls[-1] == pytest.approx(DEFAULT_LAST_KL, rel=1e-6)

    def test_kl_is_not_monotone(self, default_report):
        """KL 在早期出现回落（step 3 → step 4）：**别拿 KL 当早停判据**.

        实测 step 3 的 KL 0.000444 大于 step 4 的 0.000437。
        """
        kls = [record.kl for record in default_report.records]
        assert kls[3] < kls[2]
        assert max(kls) == kls[-1]

    def test_valid_margin_peaks_at_the_last_step(self, default_report):
        """留出 margin 在末步仍是最高的：42 步时**没有**出现过优化.

        它与训练 margin 的方向一致，所以三条过优化判据里只有
        ``accuracy_saturated`` 会触发（见 ``pipeline`` 的用例）。
        """
        margins = [record.valid_margin for record in default_report.records]
        assert max(margins) == margins[-1]
        assert margins[-1] == pytest.approx(DEFAULT_LAST_VALID_MARGIN, rel=1e-6)
        assert all(later >= earlier for earlier, later in zip(margins, margins[1:]))

    def test_training_margin_can_be_negative_on_a_single_pair(self, default_report):
        """训练 margin 出现负值：每一步只走一条偏好对，方向不必处处为正.

        这正是"不能拿训练 margin 当早停判据"的直接证据——它连单调都不是。
        """
        margins = [record.mean_margin for record in default_report.records]
        assert margins[0] == 0.0
        assert any(margin < 0 for margin in margins)
        assert margins[-1] == pytest.approx(DEFAULT_LAST_MARGIN, rel=1e-6)

    def test_mean_loss_is_not_monotone_either(self, default_report):
        """平均 loss 也不是单调下降：最小的一步出现在中途（第 36 步）.

        "只看最终 loss 的报告"因此会给出一个含糊的结论——本课把整条历史
        交出来，就是为了让这件事看得见。
        """
        losses = [record.mean_loss for record in default_report.records]
        assert min(losses) < default_report.final_loss
        assert losses.index(min(losses)) + 1 < DEFAULT_STEPS


class TestOverTrainingControl:
    """过训练对照（lr=2.0 / 20 轮 / 140 步）：只买到 KL，没买到留出收益."""

    def test_steps_is_140(self, over_trained_report):
        """20 轮 × 7 条 = 140 步（是默认配置的 3.3 倍）."""
        assert over_trained_report.steps == OVER_STEPS == 140

    def test_final_kl_exceeds_two(self, over_trained_report):
        """末步 KL 涨到 2.43（默认配置的 54 倍，超 0.5 的预算 4.9 倍）."""
        assert over_trained_report.records[-1].kl == pytest.approx(OVER_LAST_KL, rel=1e-6)
        assert over_trained_report.records[-1].kl > 2.0

    def test_valid_margin_exceeds_half(self, over_trained_report):
        """留出 margin 涨到 0.91（默认配置末步的 11.4 倍）——**它并没有回落**."""
        assert over_trained_report.records[-1].valid_margin == pytest.approx(
            OVER_LAST_VALID_MARGIN, rel=1e-6
        )
        assert over_trained_report.records[-1].valid_margin > 0.5

    def test_more_training_buys_no_extra_accuracy(self, over_trained_report):
        """多训 98 步，留出准确率一点没涨：仍然是 0.8、峰值仍在第 2 步.

        "继续训练买到了什么"的答案是：**只买到了 KL**。
        """
        assert over_trained_report.best_valid_accuracy == BEST_VALID_ACCURACY
        assert over_trained_report.best_step == BEST_STEP

    def test_training_margin_outruns_the_held_out_margin(self, over_trained_report):
        """训练 margin（5.46）远跑在留出 margin（0.91）前面——它不会告诉你该停."""
        last = over_trained_report.records[-1]
        assert last.mean_margin == pytest.approx(OVER_LAST_TRAIN_MARGIN, rel=1e-6)
        assert last.mean_margin > 5 * last.valid_margin

    def test_loss_keeps_dropping_while_accuracy_stands_still(self, over_trained_report):
        """loss 降到 0.4568（降幅是默认配置的 11 倍）而准确率没动.

        **这就是 DPO 最常见的失败模式**："loss 一路降、模型一路坏"。
        """
        assert over_trained_report.final_loss < over_trained_report.initial_loss
        assert over_trained_report.final_loss < DEFAULT_FINAL_LOSS
        assert over_trained_report.loss_drop > 10 * DEFAULT_LOSS_DROP


class TestReproducibility:
    """可复现性：同一个 ``seed`` 的两次训练逐位相同（用 ``==``，不用 approx）."""

    def test_same_seed_repeats_the_history_bitwise(self):
        """两次 ``train_dpo``（同 seed 同配置）的 steps / final_loss / best_step 逐位相同.

        用子集（前 2 条训练对、前 1 条留出对）与 6 轮，是为了让这条用例便宜：
        它验的是"切分顺序与训练顺序都由 seed 决定"，不是收敛到某个值。
        """
        subset = TRAIN[:2]
        held_out = VALID[:1]

        def run() -> DPOReport:
            return train_dpo(
                fresh_policy(),
                REFERENCE,
                TOKENIZER,
                subset,
                valid_pairs=held_out,
                beta=DEFAULT_BETA,
                learning_rate=DEFAULT_LEARNING_RATE,
                epochs=DEFAULT_EPOCHS,
                seed=42,
            )

        first = run()
        second = run()
        assert first.steps == second.steps
        assert first.final_loss == second.final_loss
        assert first.best_step == second.best_step
        assert first.best_valid_accuracy == second.best_valid_accuracy
        assert [record.mean_margin for record in first.records] == [
            record.mean_margin for record in second.records
        ]

    def test_history_is_reproducible_for_a_different_configuration(self):
        """换配置（单轮、3 条训练对）同样是逐位可复现的."""
        subset = TRAIN[:3]

        def run() -> DPOReport:
            return train_dpo(
                fresh_policy(),
                REFERENCE,
                TOKENIZER,
                subset,
                valid_pairs=VALID[:2],
                beta=0.5,
                learning_rate=1.0,
                epochs=1,
                seed=2024,
            )

        first = run()
        second = run()
        assert first.steps == 3
        assert first.final_loss == second.final_loss
        assert [record.kl for record in first.records] == [
            record.kl for record in second.records
        ]


class TestStepRecordDataclass:
    """``DPOStepRecord``：投影、打印与不可变."""

    def test_to_dict_has_every_key(self):
        """``to_dict`` 七个键齐全（含 ``valid_margin``——它是早停的主判据）."""
        payload = make_record(3, valid_margin=0.42).to_dict()
        assert set(payload) == {
            "step",
            "mean_margin",
            "mean_loss",
            "valid_accuracy",
            "valid_margin",
            "kl",
            "learning_rate",
        }
        assert payload["valid_margin"] == 0.42
        assert payload["step"] == 3

    def test_to_dict_is_json_serializable(self):
        """记录最终要落成 JSON，所以它必须可以 ``json.dumps``."""
        payload = make_record(1, kl=0.5, valid_margin=-0.25).to_dict()
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload

    def test_default_valid_margin_is_zero(self):
        """``valid_margin`` 的缺省值是 0.0（老报告缺这个字段时不会炸）."""
        record = DPOStepRecord(
            step=1,
            mean_margin=0.1,
            mean_loss=0.6,
            valid_accuracy=0.5,
            kl=0.01,
            learning_rate=0.5,
        )
        assert record.valid_margin == 0.0

    def test_summary_line_prints_every_signal(self):
        """一行摘要同时给出步数、两种 margin、loss、准确率与 KL."""
        line = make_record(7, mean_margin=0.25, kl=0.125, valid_margin=0.5).summary_line()
        assert "step   7" in line
        assert "+0.250000" in line
        assert "+0.500000" in line
        assert "KL 0.125000" in line
        assert "0.5000" in line

    def test_record_is_immutable(self):
        """记录是 frozen 的：历史一旦写下就不该被就地改写."""
        record = make_record(1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.kl = 9.9  # type: ignore[misc]

    def test_equal_records_compare_equal(self):
        """逐字段相等的两条记录相等（历史可以整体比较，不必逐键核对）."""
        assert make_record(2, kl=0.1) == make_record(2, kl=0.1)
        assert make_record(2, kl=0.1) != make_record(3, kl=0.1)


class TestReportDataclass:
    """``DPOReport``：历史派生量、投影与两个边界（空历史 / 平局）."""

    def test_empty_history_has_zero_readings(self):
        """空历史时 ``final_loss == 0.0``、``best_step == 0``、``best_valid_accuracy == 0.0``.

        报告要能打印半成品（训练还没跑或全部记录被过滤掉时），所以这些
        派生量在空列表上必须有确定答案，而不是抛异常。
        """
        report = DPOReport(
            beta=DEFAULT_BETA,
            learning_rate=DEFAULT_LEARNING_RATE,
            epochs=DEFAULT_EPOCHS,
            train_pairs=7,
            valid_pairs=5,
        )
        assert report.steps == 0
        assert report.final_loss == 0.0
        assert report.best_step == 0
        assert report.best_valid_accuracy == 0.0
        assert report.loss_drop == report.initial_loss

    def test_loss_drop_is_initial_minus_final(self):
        """``loss_drop`` 的定义式：``initial_loss − final_loss``（负值表示涨了）."""
        report = DPOReport(
            beta=DEFAULT_BETA,
            learning_rate=DEFAULT_LEARNING_RATE,
            epochs=1,
            train_pairs=1,
            valid_pairs=1,
            records=[make_record(1, mean_loss=0.5)],
            initial_loss=LN2,
        )
        assert report.loss_drop == pytest.approx(LN2 - 0.5)
        assert report.steps == 1

    def test_best_step_takes_the_earliest_tie(self):
        """准确率平局时取**最早**的那一步（早停点应当尽量早）."""
        report = DPOReport(
            beta=DEFAULT_BETA,
            learning_rate=DEFAULT_LEARNING_RATE,
            epochs=1,
            train_pairs=3,
            valid_pairs=3,
            records=[
                make_record(1, valid_accuracy=0.8),
                make_record(5, valid_accuracy=0.8),
                make_record(9, valid_accuracy=0.6),
            ],
        )
        assert report.best_step == 1
        assert report.best_valid_accuracy == 0.8

    def test_best_valid_accuracy_is_the_maximum(self):
        """``best_valid_accuracy`` 取历史最大值，不取末值."""
        report = DPOReport(
            beta=DEFAULT_BETA,
            learning_rate=DEFAULT_LEARNING_RATE,
            epochs=1,
            train_pairs=2,
            valid_pairs=2,
            records=[
                make_record(1, valid_accuracy=0.4),
                make_record(2, valid_accuracy=1.0),
                make_record(3, valid_accuracy=0.2),
            ],
        )
        assert report.best_valid_accuracy == 1.0
        assert report.best_step == 2
        assert report.final_loss == LN2

    def test_to_dict_has_every_key(self):
        """``to_dict`` 十三个键齐全（含拍平的 ``steps`` / ``loss_drop`` / ``records``）."""
        report = DPOReport(
            beta=DEFAULT_BETA,
            learning_rate=DEFAULT_LEARNING_RATE,
            epochs=2,
            train_pairs=7,
            valid_pairs=5,
            records=[make_record(1, kl=0.01), make_record(2, kl=0.02)],
            initial_loss=LN2,
            zero_margin_loss=LN2,
        )
        payload = report.to_dict()
        assert set(payload) == {
            "beta",
            "learning_rate",
            "epochs",
            "train_pairs",
            "valid_pairs",
            "steps",
            "initial_loss",
            "final_loss",
            "loss_drop",
            "zero_margin_loss",
            "best_valid_accuracy",
            "best_step",
            "records",
        }
        assert payload["steps"] == 2
        assert len(payload["records"]) == 2
        assert set(payload["records"][0]) == set(make_record(1).to_dict())

    def test_to_dict_is_json_serializable(self):
        """报告可以整体 ``json.dumps``（它最终要落盘留档）."""
        payload = DPOReport(
            beta=DEFAULT_BETA,
            learning_rate=DEFAULT_LEARNING_RATE,
            epochs=1,
            train_pairs=1,
            valid_pairs=1,
            records=[make_record(1, kl=0.3)],
            initial_loss=LN2,
        ).to_dict()
        assert json.loads(json.dumps(payload, ensure_ascii=False)) == payload

    def test_summary_line_is_printable(self):
        """一行摘要里同时出现步数、loss 的起点与终点、最佳准确率与它的步数."""
        line = (
            DPOReport(
                beta=DEFAULT_BETA,
                learning_rate=DEFAULT_LEARNING_RATE,
                epochs=1,
                train_pairs=1,
                valid_pairs=1,
                records=[make_record(1, mean_loss=0.5, valid_accuracy=0.4)],
                initial_loss=LN2,
            ).summary_line()
        )
        assert "1 步" in line
        assert "0.693147 → 0.500000" in line
        assert "最佳验证准确率 0.4000@step 1" in line

    def test_default_report_summary_line_matches_the_measurement(self, default_report):
        """默认训练的一行摘要：42 步、0.693147 → 0.671589、最佳 0.8@step 2."""
        line = default_report.summary_line()
        assert "42 步" in line
        assert "loss 0.693147 → 0.671589（+0.021558）" in line
        assert "最佳验证准确率 0.8000@step 2" in line
