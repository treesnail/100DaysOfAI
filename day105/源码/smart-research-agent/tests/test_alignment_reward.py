"""离线对齐信号的测试（M5-D6）：偏好准确率、policy KL、GAE 优势、过优化体检.

``alignment/reward.py`` 与前一天的 ``model_probe`` 一样是"白盒读数"：它的数字
必须**能被手算复现**。本文件的期望值分两类：

1. **纯函数**（``reward_normalize`` / ``gae_advantages`` / ``advantage_report`` /
   ``optimization_summary``）——期望值全部手算，不拿被测函数的中间量当基准；
2. **真实模型端到端**（``pair_reward`` / ``reward_table`` / ``preference_accuracy``
   / ``policy_kl``）——用与课程 demo 完全相同的构造（种子 42 的 bigram 参考模型 +
   字符级分词器 + 12 条种子偏好对，切分为 7 训练 / 5 留出），并只用
   **"策略是参考模型的逐位副本"** 这条性质推出期望值：

.. code-block:: text

    policy == reference  ⇒  四个对数概率两两相同  ⇒  margin == 0
                        ⇒  reward_margin == 0    ⇒  correct is False
                        ⇒  policy_kl == 0.0      ⇒  accuracy == 0.0

四条承担"证明结论"角色的用例：

1. :meth:`TestPairRewardEndToEnd.test_margin_is_zero_at_start` —— 起点 margin
   必须**恰好**为 0（``approx(abs=1e-12)``）。它同时是"DPO loss = ln 2"与
   "policy KL = 0"两条结论的共同前提；
2. :meth:`TestPreferenceAccuracy.test_by_dimension_covers_five_dimensions` ——
   留出集必须是"五个维度各 1 条"，逐维度 ``total == 1``。判据粗糙（5 值量）
   本身就是要看见的事：**0.2 一格**意味着别拿它当细分判据；
3. :meth:`TestPolicyKl.test_positive_after_one_step` —— 初始 KL 为 0、走一步
   DPO 之后必须 > 0。这两条一起才能证明"KL 是模型侧量出来的"，而不是
   "常量 0"或"恒定小数"；
4. :meth:`TestGaeAdvantages.test_last_step_next_value_is_zero` —— 最后一步的
   ``V(s_T)`` 取 0：用一个 ``values`` 非零的例子核对（``A_T = r_T − V(s_T)``），
   否则"回合结束"这个约定会被"多算一段未来"悄悄替换掉。

另外两条刻意记录的**实现口径**（测试按实现断言，不改源码）：

- ``reward_normalize`` 用**总体标准差**（除以 ``N`` 而不是 ``N − 1``），
  且常数序列返回全 0（不抛异常）——见 :meth:`TestRewardNormalize.test_std_is_population_formula`；
- ``optimization_summary`` 的 ``kl_exceeded`` **只看最终 KL**（中途超预算、
  最后回落的历史不会触发），而 ``over_optimized`` 要求"最终严格低于峰值"
  （回到峰值不算过优化）——见 :meth:`TestOptimizationSummary.test_kl_exceeded_uses_final_value_only`
  与 :meth:`TestOptimizationSummary.test_return_to_peak_is_not_over_optimization`。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest

from smart_research_agent.alignment import SEED_PAIRS, dpo_step, split_preferences
from smart_research_agent.alignment.objectives import (
    dpo_margin,
    implicit_reward_margin,
    mean_kl,
)
from smart_research_agent.alignment.preference import PreferencePair
from smart_research_agent.alignment.reward import (
    DEFAULT_GAMMA,
    DEFAULT_KL_BUDGET,
    DEFAULT_LAMBDA,
    PairReward,
    advantage_report,
    gae_advantages,
    optimization_summary,
    over_optimization_flags,
    pair_reward,
    policy_kl,
    preference_accuracy,
    reward_normalize,
    reward_table,
)
from smart_research_agent.finetune_eval.metrics import FinetuneEvalError
from smart_research_agent.finetune_eval.model_probe import sequence_logprob
from smart_research_agent.sft.encoding import CharTokenizer
from smart_research_agent.sft.reference_model import ReferenceSFTModel

#: 课程 demo 用的 β（``scripts/alignment_demo.py`` 同参）.
DEMO_BETA = 0.1

#: 课程 demo 用的学习率：一步就该让 KL 离开 0.
DEMO_LEARNING_RATE = 0.5

#: 过优化体检的默认 KL 预算（定义在本模块而不是训练器，理由见源码注释）.
KL_BUDGET = 0.5


# ---------------------------------------------------------------------------
# 测试装置：与模块文档逐字一致的构造顺序（种子 42，可复现）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignmentEngine:
    """一整套离线对齐装置：分词器 + 冻结的参考模型 + 训练 / 留出偏好对."""

    tokenizer: CharTokenizer
    reference: ReferenceSFTModel
    train: list[PreferencePair]
    valid: list[PreferencePair]


@pytest.fixture(scope="module")
def engine() -> AlignmentEngine:
    """构造 12 条种子偏好对、按维度分层切分，并建一个种子 42 的参考模型.

    参考模型是**冻结**的（没有任何被测函数会写它），因此整个模块共用一份；
    策略则每个测试各建一份（``policy`` fixture），因为 ``dpo_step`` 会改它。
    """
    pairs = list(SEED_PAIRS)
    train, valid = split_preferences(pairs)
    tokenizer = CharTokenizer.from_texts(
        [pair.prompt + pair.chosen + pair.rejected for pair in pairs]
    )
    reference = ReferenceSFTModel(tokenizer.vocab_size, seed=42)
    return AlignmentEngine(
        tokenizer=tokenizer, reference=reference, train=train, valid=valid
    )


@pytest.fixture
def policy(engine: AlignmentEngine) -> ReferenceSFTModel:
    """策略：参考模型的**逐位副本**（"起点读数"全部由这条性质推出）."""
    return ReferenceSFTModel.from_state(engine.reference.state_dict())


def dpo_tune(
    policy: ReferenceSFTModel,
    engine: AlignmentEngine,
    *,
    beta: float = DEMO_BETA,
    learning_rate: float = DEMO_LEARNING_RATE,
) -> dict[str, Any]:
    """在全部训练对上走一步 DPO（默认与 demo 同参：``β=0.1``、``lr=0.5``）."""
    return dpo_step(
        policy,
        engine.reference,
        engine.tokenizer,
        engine.train,
        beta=beta,
        learning_rate=learning_rate,
    )


def first_valid_reward(
    policy: ReferenceSFTModel,
    engine: AlignmentEngine,
    *,
    beta: float = DEMO_BETA,
) -> PairReward:
    """留出集第一条偏好对的读数（"起点读数"这一组断言的统一入口）."""
    return pair_reward(
        policy, engine.reference, engine.tokenizer, engine.valid[0], beta=beta
    )


def synthetic_reward(**overrides: Any) -> PairReward:
    """自造一条 ``PairReward``（四个对数概率直接写，便于手算三个派生量）.

    缺省值故意让 ``margin = (−1 + 2) − (−1.5 + 2) = 0.5``、``β = 1.0``，
    于是 ``reward_margin = 0.5 > 0``、``correct is True``——三个分支里的
    "正分支"是缺省，另外两个分支由测试显式改写。
    """
    payload: dict[str, Any] = {
        "pair_id": "synthetic-01",
        "dimension": "citation",
        "policy_chosen": -1.0,
        "policy_rejected": -2.0,
        "reference_chosen": -1.5,
        "reference_rejected": -2.0,
        "chosen_tokens": 4,
        "rejected_tokens": 3,
        "beta": 1.0,
    }
    payload.update(overrides)
    return PairReward(**payload)


# ---------------------------------------------------------------------------
# 训练历史样例（过优化体检的输入）
# ---------------------------------------------------------------------------

#: 健康历史：留出 margin 一路涨、准确率一路创新高、最终 KL 不超预算。
HEALTHY_HISTORY: list[dict[str, Any]] = [
    {"step": 1, "valid_margin": 0.10, "valid_accuracy": 0.2, "mean_margin": 0.20, "kl": 0.05},
    {"step": 2, "valid_margin": 0.25, "valid_accuracy": 0.4, "mean_margin": 0.45, "kl": 0.10},
    {"step": 3, "valid_margin": 0.40, "valid_accuracy": 0.8, "mean_margin": 0.70, "kl": 0.20},
]

#: 过优化历史：留出 margin 在第 2 步见顶后回落，而训练 margin 仍在涨（0.20 → 0.95）。
OVER_OPTIMIZED_HISTORY: list[dict[str, Any]] = [
    {"step": 10, "valid_margin": 0.10, "valid_accuracy": 0.2, "mean_margin": 0.20, "kl": 0.05},
    {"step": 20, "valid_margin": 0.60, "valid_accuracy": 0.4, "mean_margin": 0.50, "kl": 0.10},
    {"step": 30, "valid_margin": 0.30, "valid_accuracy": 0.8, "mean_margin": 0.95, "kl": 0.20},
]

#: KL 超预算历史：最终 KL 0.90 是预算 0.5 的 1.8 倍。
KL_EXCEEDED_HISTORY: list[dict[str, Any]] = [
    {"step": 1, "valid_margin": 0.10, "valid_accuracy": 0.2, "mean_margin": 0.20, "kl": 0.05},
    {"step": 2, "valid_margin": 0.30, "valid_accuracy": 0.4, "mean_margin": 0.50, "kl": 0.30},
    {"step": 3, "valid_margin": 0.50, "valid_accuracy": 0.8, "mean_margin": 0.80, "kl": 0.90},
]

#: 准确率饱和历史：0.8 在第 2 步出现后不再创新高。
SATURATED_HISTORY: list[dict[str, Any]] = [
    {"step": 1, "valid_margin": 0.10, "valid_accuracy": 0.2, "mean_margin": 0.20, "kl": 0.05},
    {"step": 2, "valid_margin": 0.30, "valid_accuracy": 0.8, "mean_margin": 0.50, "kl": 0.10},
    {"step": 3, "valid_margin": 0.50, "valid_accuracy": 0.8, "mean_margin": 0.80, "kl": 0.20},
]

#: 三条判据同时触发的历史：过优化 + KL 超预算 + 准确率饱和。
ALL_BAD_HISTORY: list[dict[str, Any]] = [
    {"step": 1, "valid_margin": 0.10, "valid_accuracy": 0.2, "mean_margin": 0.20, "kl": 0.05},
    {"step": 2, "valid_margin": 0.60, "valid_accuracy": 0.6, "mean_margin": 0.50, "kl": 0.30},
    {"step": 3, "valid_margin": 0.30, "valid_accuracy": 0.6, "mean_margin": 0.95, "kl": 0.90},
]

#: "回到峰值"的历史：margin 第 1 步就是峰值、最后一步**恰好回到**该峰值。
FLAT_PEAK_HISTORY: list[dict[str, Any]] = [
    {"step": 1, "valid_margin": 0.50, "valid_accuracy": 0.2, "mean_margin": 0.20, "kl": 0.05},
    {"step": 2, "valid_margin": 0.20, "valid_accuracy": 0.4, "mean_margin": 0.50, "kl": 0.10},
    {"step": 3, "valid_margin": 0.50, "valid_accuracy": 0.8, "mean_margin": 0.80, "kl": 0.20},
]

#: KL 中途冲高、最后回落：``kl_exceeded`` **只看最终值**，因此不触发。
KL_SPIKE_HISTORY: list[dict[str, Any]] = [
    {"step": 1, "valid_margin": 0.10, "valid_accuracy": 0.2, "mean_margin": 0.20, "kl": 0.90},
    {"step": 2, "valid_margin": 0.30, "valid_accuracy": 0.4, "mean_margin": 0.50, "kl": 0.20},
    {"step": 3, "valid_margin": 0.50, "valid_accuracy": 0.8, "mean_margin": 0.80, "kl": 0.30},
]

#: 只有 step 的历史：其余字段走 ``.get(key, 0.0)`` 的缺省值。
MISSING_KEYS_HISTORY: list[dict[str, Any]] = [{"step": 7}, {"step": 8}]

#: 用来给 ``optimization_summary`` / ``over_optimization_flags`` 对账的几份历史。
ALL_HISTORIES = {
    "healthy": HEALTHY_HISTORY,
    "over_optimized": OVER_OPTIMIZED_HISTORY,
    "kl_exceeded": KL_EXCEEDED_HISTORY,
    "saturated": SATURATED_HISTORY,
    "all_bad": ALL_BAD_HISTORY,
    "flat_peak": FLAT_PEAK_HISTORY,
    "kl_spike": KL_SPIKE_HISTORY,
}


# ---------------------------------------------------------------------------
# PairReward：一条偏好对的读数
# ---------------------------------------------------------------------------


class TestPairRewardDataclass:
    """``PairReward`` 的三个派生量全部由四个对数概率与 β 算出，可逐个手算."""

    def test_margin_matches_dpo_margin(self):
        """``margin`` 必须就是 ``objectives.dpo_margin`` 的结果（同一个口径）."""
        item = synthetic_reward()
        assert item.margin == dpo_margin(-1.0, -2.0, -1.5, -2.0)
        assert item.margin == pytest.approx(0.5, rel=1e-15)

    @pytest.mark.parametrize("beta", [0.1, 0.5, 1.0, 2.0])
    def test_reward_margin_is_beta_times_margin(self, beta):
        """``reward_margin = β · margin``：换 β 不改变 margin，只缩放奖励差."""
        item = synthetic_reward(beta=beta)
        assert item.reward_margin == pytest.approx(beta * item.margin, rel=1e-15)

    def test_reward_margin_matches_implicit_reward_margin(self):
        """与 ``objectives.implicit_reward_margin`` 逐位一致（同一个公式）."""
        item = synthetic_reward(beta=0.5)
        assert item.reward_margin == implicit_reward_margin(
            policy_chosen=item.policy_chosen,
            policy_rejected=item.policy_rejected,
            reference_chosen=item.reference_chosen,
            reference_rejected=item.reference_rejected,
            beta=item.beta,
        )

    def test_margin_ignores_beta(self):
        """margin 是"纯数据"的量：换 β 不该动它（β 只留在 loss 里乘）."""
        assert synthetic_reward(beta=0.1).margin == synthetic_reward(beta=5.0).margin

    def test_correct_true_when_reward_margin_positive(self):
        """奖励差为正 → 判对（判据是"隐式奖励之差 > 0"，不是准确率）."""
        item = synthetic_reward()
        assert item.reward_margin > 0.0
        assert item.correct is True

    @pytest.mark.parametrize(
        ("policy_chosen", "policy_rejected"),
        [(-1.0, -1.0), (-2.5, -2.5)],
    )
    def test_correct_false_when_reward_margin_zero(self, policy_chosen, policy_rejected):
        """奖励差恰好为 0 时**判错**（``> 0`` 是严格大于，起点必须算错）."""
        item = synthetic_reward(
            policy_chosen=policy_chosen,
            policy_rejected=policy_rejected,
            reference_chosen=policy_chosen,
            reference_rejected=policy_rejected,
        )
        assert item.reward_margin == 0.0
        assert item.correct is False

    def test_correct_false_when_reward_margin_negative(self):
        """奖励差为负 → 判错（这一条让"策略学会了反向偏好"可见）."""
        item = synthetic_reward(policy_chosen=-2.0, policy_rejected=-1.0)
        assert item.reward_margin < 0.0
        assert item.correct is False

    def test_to_dict_keys(self):
        """十二个键齐全：四个对数概率 + 两个分母 + 三个派生量 + 身份 + β."""
        assert set(synthetic_reward().to_dict()) == {
            "pair_id",
            "dimension",
            "policy_chosen",
            "policy_rejected",
            "reference_chosen",
            "reference_rejected",
            "chosen_tokens",
            "rejected_tokens",
            "beta",
            "margin",
            "reward_margin",
            "correct",
        }

    def test_to_dict_copies_the_derived_values(self):
        """``to_dict`` 的三个派生量必须来自属性本身（不是另算一遍的副本）."""
        item = synthetic_reward(beta=0.25)
        payload = item.to_dict()
        assert payload["margin"] == item.margin
        assert payload["reward_margin"] == item.reward_margin
        assert payload["correct"] == item.correct

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({}, "判对"),
            ({"policy_chosen": -1.0, "policy_rejected": -1.0}, "判错"),
            ({"policy_chosen": -2.0, "policy_rejected": -1.0}, "判错"),
        ],
    )
    def test_summary_line_marks_the_state(self, overrides, expected):
        """摘要行里的"判对 / 判错"必须与 ``correct`` 一致（报告逐行看它）."""
        item = synthetic_reward(**overrides)
        assert expected in item.summary_line()

    def test_summary_line_contains_identity_and_numbers(self):
        """摘要行要能指回是哪一条偏好对（id + 维度），并带上两个 margin."""
        line = synthetic_reward(pair_id="pref-x", dimension="honesty").summary_line()
        assert "pref-x（honesty）" in line
        assert "margin +0.500000" in line
        assert "隐式奖励差 +0.500000" in line

    def test_is_frozen(self):
        """``PairReward`` 是冻结数据类：读数算完就不该被就地修改."""
        item = synthetic_reward()
        with pytest.raises(AttributeError):
            item.beta = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# pair_reward：真实模型的端到端读数
# ---------------------------------------------------------------------------


class TestPairRewardEndToEnd:
    """用真实参考模型核对"策略 = 参考模型"时的起点读数."""

    def test_margin_is_zero_at_start(self, engine: AlignmentEngine, policy: ReferenceSFTModel):
        """**起点检查**：策略逐位等于参考模型 → margin 恰好为 0、判错.

        ``approx(abs=1e-12)`` 而不是"小于某个阈值"：这一步的 margin 必须是
        **精确的 0**（四个对数概率两两相同），否则"loss = ln 2"这条结论
        就只是一个近似。
        """
        item = first_valid_reward(policy, engine)
        assert item.margin == pytest.approx(0.0, abs=1e-12)
        assert item.reward_margin == pytest.approx(0.0, abs=1e-12)
        assert item.correct is False

    def test_policy_and_reference_logprobs_are_identical(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel
    ):
        """未训练时四个对数概率**逐位相同**（不是"很接近"）."""
        item = first_valid_reward(policy, engine)
        assert item.policy_chosen == item.reference_chosen
        assert item.policy_rejected == item.reference_rejected
        assert item.policy_chosen != item.policy_rejected

    def test_token_counts_are_positive(self, engine: AlignmentEngine, policy: ReferenceSFTModel):
        """两个分母都 ≥ 1（分母交出来才能判断"这条样本到底贡献了多少位置"）."""
        item = first_valid_reward(policy, engine)
        assert item.chosen_tokens >= 1
        assert item.rejected_tokens >= 1

    def test_token_counts_skip_the_first_position(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel
    ):
        """分母 = 编码长度 − 1：位置 0 没有前文，不计分（探针口径）."""
        pair = engine.valid[0]
        item = pair_reward(policy, engine.reference, engine.tokenizer, pair, beta=DEMO_BETA)
        assert item.chosen_tokens == len(engine.tokenizer.encode(pair.chosen)) - 1
        assert item.rejected_tokens == len(engine.tokenizer.encode(pair.rejected)) - 1

    def test_carries_pair_identity(self, engine: AlignmentEngine, policy: ReferenceSFTModel):
        """id / 维度 / β 必须从入参与调用参数带出来（报告靠它们逐行对齐）."""
        pair = engine.valid[0]
        item = pair_reward(policy, engine.reference, engine.tokenizer, pair, beta=0.37)
        assert item.pair_id == pair.id
        assert item.dimension == pair.dimension
        assert item.beta == 0.37

    def test_to_dict_and_summary_are_usable(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel
    ):
        """两个投影接口在真实读数上都能用，且互相一致."""
        pair = engine.valid[0]
        item = pair_reward(policy, engine.reference, engine.tokenizer, pair, beta=DEMO_BETA)
        payload = item.to_dict()
        assert payload["pair_id"] == pair.id
        assert payload["correct"] is False
        assert pair.id in item.summary_line()

    @pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
    def test_every_valid_pair_starts_at_zero_margin(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel, index: int
    ):
        """留出集里**每一条**的起点 margin 都是 0（不是只有第一条）."""
        item = pair_reward(
            policy, engine.reference, engine.tokenizer, engine.valid[index], beta=DEMO_BETA
        )
        assert item.margin == pytest.approx(0.0, abs=1e-12)
        assert item.correct is False

    @pytest.mark.parametrize("beta", [0.05, 0.1, 1.0, 2.0])
    def test_margin_is_zero_for_any_beta(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel, beta: float
    ):
        """起点 margin 与 β 无关（β 只乘在 loss 上），奖励差则恒为 0."""
        item = pair_reward(policy, engine.reference, engine.tokenizer, engine.valid[0], beta=beta)
        assert item.margin == 0.0
        assert item.reward_margin == 0.0


# ---------------------------------------------------------------------------
# reward_table
# ---------------------------------------------------------------------------


class TestRewardTable:
    """批量算奖励：顺序与条数必须与输入一一对应（报告要逐行比对）."""

    def test_empty_raises(self, engine: AlignmentEngine, policy: ReferenceSFTModel):
        """空偏好数据集报错，而不是返回空列表（"没量到"不能伪装成"量到 0 条"）."""
        with pytest.raises(FinetuneEvalError):
            reward_table(policy, engine.reference, engine.tokenizer, [], beta=DEMO_BETA)

    @pytest.mark.parametrize("count", [1, 2, 3, 5])
    def test_length_matches_input(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel, count: int
    ):
        """条数等于输入条数（少了会静默改变分母）."""
        pairs = engine.valid[:count]
        table = reward_table(policy, engine.reference, engine.tokenizer, pairs, beta=DEMO_BETA)
        assert len(table) == count

    def test_preserves_input_order(self, engine: AlignmentEngine, policy: ReferenceSFTModel):
        """顺序与输入一致：报告按行比对的前提."""
        table = reward_table(
            policy, engine.reference, engine.tokenizer, engine.valid, beta=DEMO_BETA
        )
        assert [item.pair_id for item in table] == [pair.id for pair in engine.valid]

    def test_entries_are_pair_rewards(self, engine: AlignmentEngine, policy: ReferenceSFTModel):
        """每个元素都是 ``PairReward``，且三个派生量可用."""
        table = reward_table(
            policy, engine.reference, engine.tokenizer, engine.train, beta=DEMO_BETA
        )
        assert all(isinstance(item, PairReward) for item in table)
        assert all(item.margin == 0.0 for item in table)

    def test_matches_single_pair_reward(self, engine: AlignmentEngine, policy: ReferenceSFTModel):
        """批量结果必须与逐条调 ``pair_reward`` 的结果一致（同一条计算路径）."""
        table = reward_table(
            policy, engine.reference, engine.tokenizer, engine.valid, beta=0.25
        )
        singles = [
            pair_reward(policy, engine.reference, engine.tokenizer, pair, beta=0.25)
            for pair in engine.valid
        ]
        assert table == singles

    def test_accepts_tuple_input(self, engine: AlignmentEngine, policy: ReferenceSFTModel):
        """接受任意 ``Sequence``（调用方常传元组）."""
        as_tuple = tuple(engine.valid)
        assert len(
            reward_table(policy, engine.reference, engine.tokenizer, as_tuple, beta=DEMO_BETA)
        ) == len(as_tuple)


# ---------------------------------------------------------------------------
# preference_accuracy
# ---------------------------------------------------------------------------


class TestPreferenceAccuracy:
    """留出偏好准确率是"对齐训练唯一的离线判据"，它的分母必须交出来."""

    def test_empty_raises(self, engine: AlignmentEngine, policy: ReferenceSFTModel):
        """空偏好数据集报错（准确率的分母不能是 0）."""
        with pytest.raises(FinetuneEvalError):
            preference_accuracy(policy, engine.reference, engine.tokenizer, [], beta=DEMO_BETA)

    def test_initial_accuracy_is_zero(self, engine: AlignmentEngine, policy: ReferenceSFTModel):
        """未训练时准确率恰好 0.0、判对 0 条（奖励差全为 0，严格大于不成立）."""
        report = preference_accuracy(
            policy, engine.reference, engine.tokenizer, engine.valid, beta=DEMO_BETA
        )
        assert report["accuracy"] == 0.0
        assert report["correct"] == 0
        assert report["total"] == len(engine.valid)

    def test_initial_mean_margins_are_zero(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel
    ):
        """两个均值（margin 与奖励 margin）都是 0.0——不是"很小"，是 0."""
        report = preference_accuracy(
            policy, engine.reference, engine.tokenizer, engine.valid, beta=DEMO_BETA
        )
        assert report["mean_margin"] == 0.0
        assert report["mean_reward_margin"] == 0.0

    def test_by_dimension_covers_five_dimensions(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel
    ):
        """逐维度分解必须覆盖留出集的 5 个维度，且每维**恰好 1 条**.

        ``total == 1`` 这件事要写在测试里，因为它是"准确率只有 5 个取值
        （0.2 一格）"的根源：判据粗，就必须配连续量（``mean_margin``）一起看。
        """
        report = preference_accuracy(
            policy, engine.reference, engine.tokenizer, engine.valid, beta=DEMO_BETA
        )
        by_dimension = report["by_dimension"]
        assert sorted(by_dimension) == sorted({pair.dimension for pair in engine.valid})
        assert len(by_dimension) == 5
        for stats in by_dimension.values():
            assert stats["total"] == 1
            assert stats["correct"] == 0
            assert stats["accuracy"] == 0.0

    def test_pairs_payload_lists_every_pair(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel
    ):
        """``pairs`` 明细逐条列出（``total`` 与明细长度必须一致）."""
        report = preference_accuracy(
            policy, engine.reference, engine.tokenizer, engine.valid, beta=DEMO_BETA
        )
        assert len(report["pairs"]) == report["total"]
        assert [item["pair_id"] for item in report["pairs"]] == [pair.id for pair in engine.valid]

    def test_report_keys_are_complete(self, engine: AlignmentEngine, policy: ReferenceSFTModel):
        """七个键齐全（少一个都会让报告缺一列或无法追溯）."""
        report = preference_accuracy(
            policy, engine.reference, engine.tokenizer, engine.valid, beta=DEMO_BETA
        )
        assert set(report) == {
            "total",
            "correct",
            "accuracy",
            "mean_margin",
            "mean_reward_margin",
            "by_dimension",
            "pairs",
        }

    def test_accuracy_rises_after_one_step(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel
    ):
        """走一步 DPO 之后准确率从 0.0 升到 0.8（4/5 条被判对，本课实测值）."""
        update = dpo_tune(policy, engine)
        report = preference_accuracy(
            policy, engine.reference, engine.tokenizer, engine.valid, beta=DEMO_BETA
        )
        assert update["updated"] is True
        assert report["accuracy"] == pytest.approx(0.8, abs=1e-12)
        assert report["correct"] == 4

    def test_accuracy_is_correct_over_total_after_step(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel
    ):
        """``accuracy == correct / total``：分子分母都交出来之后要能自洽."""
        dpo_tune(policy, engine)
        report = preference_accuracy(
            policy, engine.reference, engine.tokenizer, engine.valid, beta=DEMO_BETA
        )
        assert report["accuracy"] == pytest.approx(
            report["correct"] / report["total"], rel=1e-12
        )

    def test_mean_margin_is_the_average_of_pair_margins(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel
    ):
        """``mean_margin`` 是逐条 margin 的算术平均（不是奖励 margin 的平均）."""
        dpo_tune(policy, engine)
        report = preference_accuracy(
            policy, engine.reference, engine.tokenizer, engine.valid, beta=DEMO_BETA
        )
        margins = [item["margin"] for item in report["pairs"]]
        assert report["mean_margin"] == pytest.approx(sum(margins) / len(margins), rel=1e-12)

    def test_reward_margin_is_beta_times_mean_margin(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel
    ):
        """``mean_reward_margin == β · mean_margin``（β 只缩放、不改变符号）."""
        dpo_tune(policy, engine)
        report = preference_accuracy(
            policy, engine.reference, engine.tokenizer, engine.valid, beta=DEMO_BETA
        )
        assert report["mean_reward_margin"] == pytest.approx(
            DEMO_BETA * report["mean_margin"], rel=1e-12
        )

    def test_dimension_accuracy_after_step_is_not_uniform(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel
    ):
        """走一步后**不是**五个维度全对：逐维度分解必须能分辨出"还差哪一维"."""
        dpo_tune(policy, engine)
        report = preference_accuracy(
            policy, engine.reference, engine.tokenizer, engine.valid, beta=DEMO_BETA
        )
        accuracies = [stats["accuracy"] for stats in report["by_dimension"].values()]
        assert sorted(accuracies) == [0.0, 1.0, 1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# policy_kl
# ---------------------------------------------------------------------------


class TestPolicyKl:
    """``policy_kl`` 量的是"对齐把模型推离了多远"：起点必须是 0."""

    def test_zero_at_start(self, engine: AlignmentEngine, policy: ReferenceSFTModel):
        """策略与参考模型逐位相同 → KL **恰好** 0.0（三个估计器都退化为 0）."""
        assert policy_kl(policy, engine.reference, engine.tokenizer, engine.train) == 0.0

    @pytest.mark.parametrize("estimator", ["k1", "k2", "k3"])
    def test_all_estimators_zero_at_start(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel, estimator: str
    ):
        """三种估计器都必须是 0.0（不是"只有 k3 是 0"）."""
        assert (
            policy_kl(policy, engine.reference, engine.tokenizer, engine.train, estimator=estimator)
            == 0.0
        )

    @pytest.mark.parametrize("estimator", ["", "k0", "k4", "K3"])
    def test_unknown_estimator_raises(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel, estimator: str
    ):
        """未知估计器报错（透传到 ``mean_kl`` 的白名单校验）."""
        with pytest.raises(FinetuneEvalError):
            policy_kl(
                policy, engine.reference, engine.tokenizer, engine.train, estimator=estimator
            )

    def test_empty_pairs_raises(self, engine: AlignmentEngine, policy: ReferenceSFTModel):
        """空偏好数据集报错：一批都没有文本时 KL 没有分母."""
        with pytest.raises(FinetuneEvalError):
            policy_kl(policy, engine.reference, engine.tokenizer, [])

    def test_positive_after_one_step(self, engine: AlignmentEngine, policy: ReferenceSFTModel):
        """走一步 DPO 之后 KL 必须 > 0——否则它就不是"模型侧量出来的".

        实测一步之后约 ``3.5e-05``：很小，但**方向正确**且可复现。
        """
        dpo_tune(policy, engine)
        assert policy_kl(policy, engine.reference, engine.tokenizer, engine.train) > 0.0

    def test_matches_manual_mean_kl_of_both_sides(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel
    ):
        """与"手工把 chosen / rejected 两侧的对数概率都收齐再取均值"逐位一致.

        这条测试是**独立重算**：它不用 ``policy_kl`` 的中间量，只借
        ``sequence_logprob`` 与 ``mean_kl``（两者各自有独立测试）。
        """
        dpo_tune(policy, engine)
        logprobs: list[float] = []
        references: list[float] = []
        for pair in engine.train:
            for text in (pair.chosen, pair.rejected):
                ids = engine.tokenizer.encode(text)
                logprobs.append(sequence_logprob(policy, ids)[0])
                references.append(sequence_logprob(engine.reference, ids)[0])
        assert policy_kl(policy, engine.reference, engine.tokenizer, engine.train) == mean_kl(
            logprobs, references, estimator="k3"
        )

    def test_single_pair_uses_both_sides_of_that_pair(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel
    ):
        """单条偏好对也统计 chosen 与 rejected **两侧**（分母是 2 条文本）."""
        dpo_tune(policy, engine)
        pair = engine.train[0]
        logprobs: list[float] = []
        references: list[float] = []
        for text in (pair.chosen, pair.rejected):
            ids = engine.tokenizer.encode(text)
            logprobs.append(sequence_logprob(policy, ids)[0])
            references.append(sequence_logprob(engine.reference, ids)[0])
        assert policy_kl(
            policy, engine.reference, engine.tokenizer, [pair], estimator="k1"
        ) == mean_kl(logprobs, references, estimator="k1")

    def test_grows_with_a_second_step(self, engine: AlignmentEngine, policy: ReferenceSFTModel):
        """第二步之后 KL 比第一步更大（"偏离是累积的"，不会自己回收）."""
        dpo_tune(policy, engine)
        after_one = policy_kl(policy, engine.reference, engine.tokenizer, engine.train)
        dpo_tune(policy, engine)
        after_two = policy_kl(policy, engine.reference, engine.tokenizer, engine.train)
        assert after_two > after_one > 0.0

    def test_matches_zero_when_policy_is_untouched_by_a_measure_only_step(
        self, engine: AlignmentEngine, policy: ReferenceSFTModel
    ):
        """``learning_rate = 0`` 只测量不更新：KL 仍然必须是 0.0."""
        dpo_step(
            policy,
            engine.reference,
            engine.tokenizer,
            engine.train,
            beta=DEMO_BETA,
            learning_rate=0.0,
        )
        assert policy_kl(policy, engine.reference, engine.tokenizer, engine.train) == 0.0


# ---------------------------------------------------------------------------
# reward_normalize
# ---------------------------------------------------------------------------


class TestRewardNormalize:
    """奖励白化：``(r − μ) / σ``，返回 ``(标准化值, μ, σ)``."""

    def test_empty_raises(self):
        """空序列报错（μ 与 σ 都没有定义）."""
        with pytest.raises(FinetuneEvalError):
            reward_normalize([])

    @pytest.mark.parametrize("value", [0.0, 2.5, -7.0])
    def test_constant_sequence_gives_zeros(self, value):
        """常数序列 → 全 0 且 ``σ == 0.0``（不抛异常、也不除以 0）.

        语义是"这一批样本没有区分度"：让上层发现"优势全为 0、梯度也就全为 0"。
        """
        normalized, mean, std = reward_normalize([value] * 4)
        assert normalized == [0.0] * 4
        assert mean == value
        assert std == 0.0

    def test_hand_computed_one_two_three(self):
        """``[1, 2, 3]``：``μ = 2``、``σ = sqrt(2/3) ≈ 0.8165``、三值为 ∓1.2247 与 0."""
        normalized, mean, std = reward_normalize([1.0, 2.0, 3.0])
        assert mean == pytest.approx(2.0, rel=1e-15)
        assert std == pytest.approx(math.sqrt(2.0 / 3.0), rel=1e-15)
        assert normalized[0] == pytest.approx(-math.sqrt(1.5), rel=1e-15)
        assert normalized[1] == pytest.approx(0.0, abs=1e-15)
        assert normalized[2] == pytest.approx(math.sqrt(1.5), rel=1e-15)

    def test_normalized_values_sum_to_zero(self):
        """标准化后的和约为 0（``approx(abs=1e-9)``：浮点求和必然有残差）."""
        for values in ([1.0, 2.0, 3.0], [0.5, -1.5, 4.0, 2.0], [-3.0, -1.0, 7.0]):
            normalized, _, _ = reward_normalize(values)
            assert sum(normalized) == pytest.approx(0.0, abs=1e-9)

    def test_single_value_is_zero(self):
        """只有一个样本 → 无区分度 → 全 0、``σ == 0.0``."""
        assert reward_normalize([3.5]) == ([0.0], 3.5, 0.0)

    def test_preserves_length(self):
        """标准化不改变长度（优势与奖励必须一一对应）."""
        normalized, _, _ = reward_normalize([1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(normalized) == 5

    @pytest.mark.parametrize("scale", [0.1, 2.0, 100.0])
    def test_scale_invariance(self, scale: float):
        """整体缩放不改变标准化结果（σ 与值一起缩放，比值不变）."""
        normalized, _, _ = reward_normalize([1.0, 2.0, 3.0])
        scaled, _, _ = reward_normalize([value * scale for value in (1.0, 2.0, 3.0)])
        assert scaled == pytest.approx(normalized, rel=1e-12)

    def test_translation_only_moves_the_mean(self):
        """整体平移只改 μ，不改标准化值（白化对基线不敏感）."""
        normalized, mean, _ = reward_normalize([11.0, 12.0, 13.0])
        baseline, _, _ = reward_normalize([1.0, 2.0, 3.0])
        assert normalized == pytest.approx(baseline, rel=1e-12)
        assert mean == pytest.approx(12.0, rel=1e-15)

    @pytest.mark.parametrize(
        "values", [[1.0, 2.0], [1.0, 2.0, 3.0], [-5.0, 0.0, 5.0, 10.0], [0.25, 0.75]]
    )
    def test_std_is_population_formula(self, values):
        """**实现口径**：``σ`` 是**总体**标准差（除以 ``N``，不是 ``N − 1``）."""
        _, mean, std = reward_normalize(values)
        expected_mean = sum(values) / len(values)
        expected_std = math.sqrt(
            sum((value - expected_mean) ** 2 for value in values) / len(values)
        )
        assert mean == pytest.approx(expected_mean, rel=1e-15)
        assert std == pytest.approx(expected_std, rel=1e-15)

    def test_returns_a_three_tuple(self):
        """返回值结构固定为 ``(列表, 浮点, 浮点)``（报告要分别取用）."""
        result = reward_normalize([1.0, 2.0, 3.0])
        assert isinstance(result, tuple) and len(result) == 3
        assert isinstance(result[0], list)
        assert isinstance(result[1], float) and isinstance(result[2], float)


# ---------------------------------------------------------------------------
# gae_advantages
# ---------------------------------------------------------------------------


class TestGaeAdvantages:
    """``δ_t = r_t + γV(s_{t+1}) − V(s_t)``、``A_t = δ_t + γλ·A_{t+1}``."""

    def test_length_mismatch_raises(self):
        """长度不一致直接报错：错位之后算出来的优势**看起来完全正常**."""
        with pytest.raises(FinetuneEvalError):
            gae_advantages([1.0, 2.0], [0.0])

    def test_length_mismatch_raises_for_extra_values(self):
        """反方向同样报错（不能只校验一侧）."""
        with pytest.raises(FinetuneEvalError):
            gae_advantages([1.0], [0.0, 0.5])

    def test_empty_raises(self):
        """空序列报错（"没有时间步"与"优势全为 0"必须分开）."""
        with pytest.raises(FinetuneEvalError):
            gae_advantages([], [])

    @pytest.mark.parametrize("gamma", [-0.1, 1.1, 2.0, float("nan"), float("inf")])
    def test_gamma_out_of_range_raises(self, gamma: float):
        """``γ`` 必须落在 ``[0, 1]``（``nan`` 也会被链式比较拦下）."""
        with pytest.raises(FinetuneEvalError):
            gae_advantages([1.0], [0.0], gamma=gamma)

    @pytest.mark.parametrize("lam", [-0.1, 1.1, 2.0, float("nan"), float("inf")])
    def test_lambda_out_of_range_raises(self, lam: float):
        """``λ`` 必须落在 ``[0, 1]``."""
        with pytest.raises(FinetuneEvalError):
            gae_advantages([1.0], [0.0], lam=lam)

    @pytest.mark.parametrize(("gamma", "lam"), [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)])
    def test_boundaries_are_allowed(self, gamma: float, lam: float):
        """两个端点是合法的（``γ = λ = 1`` 退化、``= 0`` 只看即时奖励）."""
        assert len(gae_advantages([1.0, 2.0], [0.0, 0.0], gamma=gamma, lam=lam)) == 2

    def test_undiscounted_equals_return_minus_baseline(self):
        """``γ = λ = 1``、``rewards = [1, 1, 1]``、``values = [0, 0, 0]`` → ``[3, 2, 1]``.

        手算：``A_t = Σ_{k≥t} r_k − V(s_t)``（未来折扣全为 1，基线全为 0）。
        """
        assert gae_advantages([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], gamma=1.0, lam=1.0) == [
            3.0,
            2.0,
            1.0,
        ]

    def test_zero_discount_uses_immediate_delta(self):
        """``γ = λ = 0``：``A_t = r_t − V(s_t)``（只看即时奖励，不折未来）."""
        advantages = gae_advantages([1.0, 2.0, 3.0], [0.5, 0.5, 0.5], gamma=0.0, lam=0.0)
        assert advantages == pytest.approx([0.5, 1.5, 2.5], rel=1e-15)

    def test_last_step_next_value_is_zero(self):
        """**实现口径**：最后一步的 ``V(s_T)`` 取 0，于是 ``A_T = r_T − V(s_T)``.

        用一个 ``values`` 非零的例子核对（``r = [1, 2]``、``V = [0.5, 0.5]``、
        ``γ = λ = 1``）::

            t=1: δ = 2 + 1·0   − 0.5 = 1.5   A_1 = 1.5     ← 下一步的价值取 0
            t=0: δ = 1 + 1·0.5 − 0.5 = 1.0   A_0 = 1.0 + 1.5 = 2.5

        如果实现把"最后一步之后"错写成复用 ``values[-1]``，``A_1`` 会变成
        ``2.0``——这条断言就是为那种写法准备的。
        """
        advantages = gae_advantages([1.0, 2.0], [0.5, 0.5], gamma=1.0, lam=1.0)
        assert advantages[-1] == pytest.approx(2.0 - 0.5, rel=1e-15)
        assert advantages == pytest.approx([2.5, 1.5], rel=1e-15)

    def test_bootstraps_backwards(self):
        """``γ = λ = 0.5`` 的三步手算：``A_2 = 1.5``、``A_1 = 2.125``、``A_0 = 1.53125``.

        手算过程（``r = [1, 2, 3]``、``V = [0.5, 1.0, 1.5]``）::

            t=2: δ = 3 + 0.5·0 − 1.5 = 1.5      A_2 = 1.5
            t=1: δ = 2 + 0.5·1.5 − 1.0 = 1.75   A_1 = 1.75 + 0.25·1.5 = 2.125
            t=0: δ = 1 + 0.5·1.0 − 0.5 = 1.0    A_0 = 1.0 + 0.25·2.125 = 1.53125
        """
        advantages = gae_advantages([1.0, 2.0, 3.0], [0.5, 1.0, 1.5], gamma=0.5, lam=0.5)
        assert advantages == pytest.approx([1.53125, 2.125, 1.5], rel=1e-15)

    @pytest.mark.parametrize(
        ("rewards", "values", "expected"),
        [
            ([1.0], [0.0], [1.0]),
            ([1.0], [0.25], [0.75]),
            ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0]),
            ([2.0, 2.0], [1.0, 1.0], [3.0, 1.0]),
        ],
    )
    def test_small_hand_computed_cases(self, rewards, values, expected):
        """几组小例子逐一手算（``γ = λ = 1``，即回报减基线）."""
        assert gae_advantages(rewards, values, gamma=1.0, lam=1.0) == pytest.approx(
            expected, rel=1e-15
        )

    def test_output_length_matches_input(self):
        """输出与输入等长（优势要能逐位置乘回梯度）."""
        rewards = [0.5, -1.0, 2.0, 0.0]
        values = [0.0, 0.5, 1.0, 1.5]
        assert len(gae_advantages(rewards, values)) == 4

    def test_defaults_match_the_declared_constants(self):
        """缺省 ``γ = 1.0``、``λ = 0.95``（报告里引用的是这两个常量）."""
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        expected = gae_advantages(
            rewards, values, gamma=DEFAULT_GAMMA, lam=DEFAULT_LAMBDA
        )
        assert gae_advantages(rewards, values) == expected
        assert (DEFAULT_GAMMA, DEFAULT_LAMBDA) == (1.0, 0.95)

    def test_larger_lambda_keeps_earlier_advantages_larger(self):
        """``λ`` 越大，"奖励归因到更远"——``γ = 1`` 时早期优势随 ``λ`` 增大."""
        small = gae_advantages([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], gamma=1.0, lam=0.0)
        large = gae_advantages([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], gamma=1.0, lam=1.0)
        assert large[0] > small[0]
        assert large[-1] == small[-1] == 1.0

    def test_advantages_are_floats(self):
        """返回值是浮点列表（不是 int，便于后续加权求和）."""
        advantages = gae_advantages([1.0, 1.0], [0.0, 0.0], gamma=1.0, lam=1.0)
        assert all(isinstance(value, float) for value in advantages)


# ---------------------------------------------------------------------------
# advantage_report
# ---------------------------------------------------------------------------


class TestAdvantageReport:
    """优势的汇总：均值必须与"正负比例"一起看（分辨"偏向"与"两极分化"）."""

    def test_empty_raises(self):
        """空序列报错（``min`` / ``max`` 都没有定义）."""
        with pytest.raises(FinetuneEvalError):
            advantage_report([])

    def test_hand_computed_three_two_one(self):
        """``[3, 2, 1]``：``μ = 2``、``σ = sqrt(2/3)``、``min = 1``、``max = 3``、正比例 1.0."""
        report = advantage_report([3.0, 2.0, 1.0])
        assert report["steps"] == 3
        assert report["mean"] == pytest.approx(2.0, rel=1e-15)
        assert report["std"] == pytest.approx(math.sqrt(2.0 / 3.0), rel=1e-15)
        assert report["min"] == 1.0
        assert report["max"] == 3.0
        assert report["positive_fraction"] == 1.0

    def test_zero_is_not_counted_as_positive(self):
        """``0`` **不算**正（``value > 0`` 严格大于）：``[-1, 0, 1]`` 的正比例是 1/3."""
        report = advantage_report([-1.0, 0.0, 1.0])
        assert report["positive_fraction"] == pytest.approx(1.0 / 3.0, rel=1e-15)
        assert report["mean"] == 0.0

    def test_all_negative(self):
        """全负时正比例为 0.0（"整体偏好反了"的读数）."""
        report = advantage_report([-1.0, -2.0, -3.0])
        assert report["positive_fraction"] == 0.0
        assert report["max"] == -1.0

    @pytest.mark.parametrize("length", [1, 2, 5, 8])
    def test_steps_matches_length(self, length: int):
        """``steps`` 是分母，必须等于输入长度."""
        report = advantage_report([1.0] * length)
        assert report["steps"] == length

    def test_keys_are_complete(self):
        """六个键齐全（报告逐列取用）."""
        assert set(advantage_report([1.0])) == {
            "steps",
            "mean",
            "std",
            "min",
            "max",
            "positive_fraction",
        }

    def test_std_is_population_formula(self):
        """``std`` 与 ``reward_normalize`` 同口径：总体标准差（除以 N）."""
        values = [0.5, -1.5, 4.0]
        report = advantage_report(values)
        mean = sum(values) / len(values)
        expected = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        assert report["std"] == pytest.approx(expected, rel=1e-15)

    def test_matches_gae_output(self):
        """端到端组合：``advantage_report(gae_advantages(...))`` 的均值＝算术平均."""
        advantages = gae_advantages([1.0, 2.0, 3.0], [0.5, 1.0, 1.5], gamma=0.5, lam=0.5)
        report = advantage_report(advantages)
        assert report["steps"] == 3
        assert report["mean"] == pytest.approx(sum(advantages) / 3, rel=1e-15)
        assert report["max"] == 2.125
        assert report["min"] == 1.5

    def test_spread_is_visible_in_std(self):
        """两极分化的样本 std 更大（这正是"要跟均值一起看"的理由）."""
        tight = advantage_report([1.0, 1.1, 0.9])
        wide = advantage_report([5.0, -5.0, 0.0])
        assert wide["std"] > tight["std"]
        assert wide["positive_fraction"] == pytest.approx(1.0 / 3.0, rel=1e-15)


# ---------------------------------------------------------------------------
# 过优化体检
# ---------------------------------------------------------------------------


class TestOptimizationSummary:
    """三条判据各自对应一种可观测的故障，**空清单是有歧义的**."""

    def test_empty_history_all_false(self):
        """空历史：三个布尔值全 False、``steps == 0``、``flags == []``."""
        summary = optimization_summary([])
        assert summary["steps"] == 0
        assert summary["over_optimized"] is False
        assert summary["kl_exceeded"] is False
        assert summary["accuracy_saturated"] is False
        assert summary["flags"] == []

    def test_empty_history_has_all_keys(self):
        """空历史也必须交齐 12 个键（"无数据"与"字段缺失"要能分辨）."""
        assert set(optimization_summary([])) == {
            "steps",
            "valid_margin_peak",
            "valid_margin_peak_step",
            "valid_margin_final",
            "valid_accuracy_best",
            "valid_accuracy_best_step",
            "valid_accuracy_final",
            "kl_final",
            "over_optimized",
            "kl_exceeded",
            "accuracy_saturated",
            "flags",
        }

    def test_keys_stable_between_empty_and_non_empty(self):
        """空历史与非空历史的键集合必须完全一致（报告模板才能固定）."""
        assert set(optimization_summary([])) == set(optimization_summary(HEALTHY_HISTORY))

    def test_healthy_history_has_no_flags(self):
        """健康历史：留出 margin 一直涨、KL 未超、准确率一直在创新高 → 三个布尔全 False."""
        summary = optimization_summary(HEALTHY_HISTORY)
        assert summary["over_optimized"] is False
        assert summary["kl_exceeded"] is False
        assert summary["accuracy_saturated"] is False
        assert summary["flags"] == []

    def test_over_optimized_flag_text_and_peak_fields(self):
        """过优化：峰值与回落点都要写进 flag，且要指出"训练 margin 不会告诉你这件事"."""
        summary = optimization_summary(OVER_OPTIMIZED_HISTORY)
        assert summary["over_optimized"] is True
        assert summary["valid_margin_peak"] == pytest.approx(0.60, rel=1e-15)
        assert summary["valid_margin_peak_step"] == 20
        assert summary["valid_margin_final"] == pytest.approx(0.30, rel=1e-15)
        assert summary["kl_exceeded"] is False
        assert summary["accuracy_saturated"] is False
        assert len(summary["flags"]) == 1
        assert "过优化" in summary["flags"][0]
        assert "峰值" in summary["flags"][0]
        assert "第 20 步" in summary["flags"][0]
        assert "0.950000" in summary["flags"][0]

    def test_kl_exceeded_flag_text(self):
        """KL 超预算：最终 KL 与预算都要出现在 flag 里，并点名后果."""
        summary = optimization_summary(KL_EXCEEDED_HISTORY)
        assert summary["kl_exceeded"] is True
        assert summary["kl_final"] == pytest.approx(0.90, rel=1e-15)
        assert len(summary["flags"]) == 1
        assert "超出预算" in summary["flags"][0]
        assert "0.900000" in summary["flags"][0]
        assert "参考模型" in summary["flags"][0]

    @pytest.mark.parametrize(
        ("budget", "expected"),
        [(0.0, True), (0.5, True), (0.8, True), (0.9, False), (1.0, False)],
    )
    def test_kl_budget_boundary(self, budget: float, expected: bool):
        """判据是**严格大于**预算：``kl_final == budget`` 不触发（边界值 0.9）."""
        final_kl = KL_EXCEEDED_HISTORY[-1]["kl"]
        assert final_kl == 0.9
        summary = optimization_summary(KL_EXCEEDED_HISTORY, kl_budget=budget)
        assert summary["kl_exceeded"] is expected

    def test_kl_exceeded_uses_final_value_only(self):
        """**实现口径**：只看**最终** KL——中途冲到 0.9 但最后回到 0.3 不触发.

        这条口径值得写在测试里：它意味着"KL 冲高后又回落"不会被记账，
        调用方要自己看历史（``kl_final`` 与历史里的 ``kl`` 是两件事）。
        """
        summary = optimization_summary(KL_SPIKE_HISTORY)
        assert summary["kl_final"] == pytest.approx(0.30, rel=1e-15)
        assert summary["kl_exceeded"] is False
        assert max(entry["kl"] for entry in KL_SPIKE_HISTORY) == 0.9

    def test_accuracy_saturated_flag_text(self):
        """准确率饱和：要指出"自第几步起不再创新高"与"KL 已从多少涨到多少"."""
        summary = optimization_summary(SATURATED_HISTORY)
        assert summary["accuracy_saturated"] is True
        assert summary["valid_accuracy_best"] == pytest.approx(0.8, rel=1e-15)
        assert summary["valid_accuracy_best_step"] == 2
        assert summary["valid_accuracy_final"] == pytest.approx(0.8, rel=1e-15)
        assert len(summary["flags"]) == 1
        assert "不再创新高" in summary["flags"][0]
        assert "第 2 步" in summary["flags"][0]

    def test_return_to_peak_is_not_over_optimization(self):
        """**边界**：margin 最终**回到**峰值不算过优化（判据要求严格低于峰值）.

        这条边界很实用：留出 margin 出现"U 形"时不该被误判成过优化——
        真正的过优化是"回不去"。
        """
        summary = optimization_summary(FLAT_PEAK_HISTORY)
        assert summary["valid_margin_peak"] == pytest.approx(0.50, rel=1e-15)
        assert summary["valid_margin_peak_step"] == 1
        assert summary["valid_margin_final"] == pytest.approx(0.50, rel=1e-15)
        assert summary["over_optimized"] is False
        assert summary["flags"] == []

    def test_missing_keys_default_to_zero(self):
        """缺字段的历史走 ``.get(..., 0.0)`` 缺省：不会 KeyError，但会露出问题.

        只有 ``step`` 的历史里，准确率恒为 0 → "自第 7 步起不再创新高"
        （峰值取**最早**的最大值下标）。这类历史通常意味着写日志的代码漏了字段。
        """
        summary = optimization_summary(MISSING_KEYS_HISTORY)
        assert summary["steps"] == 2
        assert summary["valid_margin_peak"] == 0.0
        assert summary["valid_margin_peak_step"] == 7
        assert summary["kl_final"] == 0.0
        assert summary["accuracy_saturated"] is True
        assert summary["over_optimized"] is False
        assert "第 7 步" in summary["flags"][0]

    def test_all_three_flags_order(self):
        """三条判据同时触发时，flag 顺序固定：过优化 → KL → 准确率."""
        flags = optimization_summary(ALL_BAD_HISTORY)["flags"]
        assert len(flags) == 3
        assert "过优化" in flags[0]
        assert "超出预算" in flags[1]
        assert "不再创新高" in flags[2]

    def test_flags_match_summary_subkey(self):
        """``over_optimization_flags`` 必须就是 summary 的 ``flags``（同一个函数）."""
        for name, history in ALL_HISTORIES.items():
            assert over_optimization_flags(history) == optimization_summary(history)["flags"], name

    def test_flags_are_human_readable_strings(self):
        """flag 是非空字符串（它会直接进报告，不能是元组或对象）."""
        for flag in optimization_summary(ALL_BAD_HISTORY)["flags"]:
            assert isinstance(flag, str)
            assert flag.strip()

    def test_custom_budget_is_forwarded(self):
        """``kl_budget`` 参数要透传（``over_optimization_flags`` 与 summary 一致）."""
        assert over_optimization_flags(KL_EXCEEDED_HISTORY, kl_budget=0.8) == (
            optimization_summary(KL_EXCEEDED_HISTORY, kl_budget=0.8)["flags"]
        )
        assert len(over_optimization_flags(KL_EXCEEDED_HISTORY, kl_budget=0.8)) == 1
        assert over_optimization_flags(KL_EXCEEDED_HISTORY, kl_budget=1.0) == []

    def test_default_budget_is_the_module_constant(self):
        """缺省预算必须是模块常量 ``DEFAULT_KL_BUDGET``（不是训练器里的副本）."""
        assert DEFAULT_KL_BUDGET == KL_BUDGET == 0.5
        assert optimization_summary(KL_EXCEEDED_HISTORY) == optimization_summary(
            KL_EXCEEDED_HISTORY, kl_budget=DEFAULT_KL_BUDGET
        )

    def test_summary_reports_best_and_final_accuracy(self):
        """最优 / 最终准确率与它们的步号都要交出来（早停点是这两者的比较）."""
        summary = optimization_summary(HEALTHY_HISTORY)
        assert summary["valid_accuracy_best"] == pytest.approx(0.8, rel=1e-15)
        assert summary["valid_accuracy_best_step"] == 3
        assert summary["valid_accuracy_final"] == pytest.approx(0.8, rel=1e-15)
        assert summary["steps"] == 3
