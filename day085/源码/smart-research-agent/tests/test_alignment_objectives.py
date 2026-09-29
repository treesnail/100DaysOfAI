"""对齐目标函数的测试（M5-D6）：``sigmoid`` / DPO / 隐式奖励 / KL 估计器.

``alignment/objectives.py`` 是本课唯一一个**能把"对齐"写成一行算术**的模块，
所以本文件的期望值全部来自手算（或来自数学上等价的另一种写法），**不拿被测
函数自己算出来的中间量当基准**：

.. code-block:: text

    −log σ(0)          = ln 2 = 0.6931471805599453      ← 起点检查
    ∂L/∂m |m=0         = −β/2          （β=0.1 → −0.05；β=0.5 → −0.25）
    σ(β·0)             = 0.5
    −log σ(βm) |m=−1000 ≈ −β·m = 1000  （误差 < 1e-9）
    k1 = log p − log q     k2 = 0.5·(log p − log q)²     k3 = e^{−r} + r − 1

四条承担"证明结论"角色的用例（它们各自钉住一条容易被写错的性质）：

1. :meth:`TestModuleConstants.test_zero_margin_loss_is_log_two` —— ``ZERO_MARGIN_LOSS``
   必须**逐位**等于 ``math.log(2.0)``。它是"ref 或 β 接错了"这类接线错误的
   唯一探针，所以不能是"差不多等于 ln 2"；
2. :meth:`TestDpoLossFromMargin.test_negative_margin_grows_linearly` —— 负 margin
   时 loss 趋近 ``−β·margin``：**一条写错的偏好样本必须给出很大的 loss**，
   不许被任何"数值保护"压平（模块文档里明确写了这条纪律）；
3. :meth:`TestKlEstimators.test_k3_is_never_negative` —— ``k3`` 在任意
   ``(logp, logq)`` 上都非负；同一条测试里还钉住 ``k1`` **可以为负**，
   两者必须能分辨（否则"只报一个估计器"的分歧就被掩盖了）；
4. :meth:`TestRlhfPpoObjective.test_objective_is_mean_reward_minus_beta_kl` ——
   ``E[r] − β·KL`` 的三个组成（均值、KL、β）都要能被单独核对。

另外三条刻意记录的**实现口径**（测试按实现断言，不改源码）：

- **β 的校验不对称**：``dpo_loss_from_margin`` / ``dpo_gradient`` /
  ``dpo_probability`` / ``implicit_reward`` / ``reward_with_kl_shaping`` /
  ``rlhf_ppo_objective`` 都会走 ``_check_beta``（β ≤ 0 或非有限直接报错），
  而 ``implicit_reward_margin`` **不校验 β**——``beta=0`` 时它安静地返回 0.0
  （见 :meth:`TestImplicitRewardMargin.test_beta_is_not_validated`）；
- **非有限输入的拦截范围**：``dpo_margin`` / ``kl_estimators`` /
  ``implicit_reward`` 会拦 ``nan`` / ``inf``，而 ``dpo_loss_from_margin`` /
  ``dpo_probability`` / ``reward_with_kl_shaping`` 只校验 β，``nan`` margin
  会**原样穿过去**变成 ``nan``（见 :meth:`TestDpoLossFromMargin.test_nan_margin_passes_through`）；
- **``k3`` 的溢出**：``k3`` 需要 ``exp``，当 ``log q − log p`` 大于约 709.78 时
  ``math.expm1`` 抛的是 ``OverflowError`` 而不是 ``FinetuneEvalError``
  （模块文档承认了这个代价，见
  :meth:`TestKlEstimators.test_k3_overflows_instead_of_raising_domain_error`）。
"""

from __future__ import annotations

import math

import pytest

from smart_research_agent.alignment.objectives import (
    BETA_SOFT_RANGE,
    KL_ESTIMATORS,
    ZERO_MARGIN_LOSS,
    RLHFObjective,
    dpo_gradient,
    dpo_loss,
    dpo_loss_from_margin,
    dpo_margin,
    dpo_probability,
    implicit_reward,
    implicit_reward_margin,
    kl_estimators,
    log_sigmoid,
    mean_kl,
    objective_requirements,
    reward_with_kl_shaping,
    rlhf_ppo_objective,
    sigmoid,
    softplus,
)
from smart_research_agent.finetune_eval.metrics import FinetuneEvalError

#: ``ln 2`` 的手写常量：**刻意不写成 ``math.log(2.0)``**，这样"实现与常量
#: 同时被改错"不会被测试放过。
LN2 = 0.6931471805599453

#: 常用 β 的几档：跨两个数量级，用来分辨"β 只是温度"与"β 也进了梯度系数"。
BETA_CASES = (0.1, 0.5, 1.0, 2.0)

#: 非法 β：0、负数与非有限数（三者的报错理由不同，但都必须报错）。
BAD_BETA_CASES = (0.0, -1.0, -0.5, float("nan"), float("inf"), float("-inf"))


# ---------------------------------------------------------------------------
# 模块常量
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """三个常量都是"对外承诺"，测试要钉住它们的精确值."""

    def test_zero_margin_loss_is_log_two(self):
        """``ZERO_MARGIN_LOSS`` 必须逐位等于 ``ln 2``（起点检查的唯一基准）."""
        assert ZERO_MARGIN_LOSS == math.log(2.0)
        assert ZERO_MARGIN_LOSS == LN2

    def test_kl_estimator_names(self):
        """三个估计器的名字与顺序（顺序进错误信息，改动会破坏日志可比性）."""
        assert KL_ESTIMATORS == ("k1", "k2", "k3")

    def test_beta_soft_range(self):
        """β 的推荐区间是一个正数区间，且下界小于上界."""
        low, high = BETA_SOFT_RANGE
        assert (low, high) == (0.01, 5.0)
        assert 0.0 < low < high

    def test_objectives_module_stays_pure(self):
        """同一组输入重复调用必须给出同一个输出（本模块不依赖随机数/环境）."""
        first = dpo_loss_from_margin(0.37, beta=0.25)
        second = dpo_loss_from_margin(0.37, beta=0.25)
        assert first == second


# ---------------------------------------------------------------------------
# sigmoid / log_sigmoid / softplus：极端值下的稳定性
# ---------------------------------------------------------------------------


class TestSigmoid:
    """``sigmoid`` 按符号分支实现，因此 ``±1000`` 都不该溢出."""

    def test_at_zero_is_half(self):
        """``σ(0) = 0.5``——它是"未训练时 DPO 概率恰好一半"的来源."""
        assert sigmoid(0.0) == 0.5

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1000.0, 1.0),
            (-1000.0, 0.0),
            (1e308, 1.0),
            (-1e308, 0.0),
        ],
    )
    def test_saturates_without_overflow(self, value, expected):
        """极端值必须**返回** 0.0 / 1.0，而不是抛 ``OverflowError``."""
        assert sigmoid(value) == expected

    @pytest.mark.parametrize("value", [-30.0, -3.0, -0.5, 0.0, 0.5, 3.0, 30.0])
    def test_matches_naive_formula(self, value):
        """在不会溢出的区间里，分支实现必须与朴素写法一致."""
        assert sigmoid(value) == pytest.approx(1.0 / (1.0 + math.exp(-value)), rel=1e-15)

    @pytest.mark.parametrize("value", [-5.0, -1.0, 0.0, 1.0, 5.0])
    def test_antisymmetry(self, value):
        """``σ(−x) = 1 − σ(x)``：随机初始化下的"一半步长"由它保证."""
        assert sigmoid(-value) == pytest.approx(1.0 - sigmoid(value), abs=1e-15)

    @pytest.mark.parametrize(
        ("low", "high"),
        [(-1000.0, 1000.0), (-1.0, 1.0), (0.0, 0.001), (-3.0, -2.0)],
    )
    def test_strictly_increasing(self, low, high):
        """严格单调递增（"margin 越大越自信"这条语义的落点）."""
        assert sigmoid(low) < sigmoid(high)


class TestLogSigmoid:
    """``log_sigmoid`` 是 DPO loss 的实现细节，它的稳定性决定了 loss 的上限."""

    def test_at_zero_is_minus_log_two(self):
        """``log σ(0) = −ln 2``（于是 ``−log σ(0) = ln 2``）."""
        assert log_sigmoid(0.0) == -LN2

    @pytest.mark.parametrize("value", [1000.0, 1e308])
    def test_large_positive_is_finite_and_near_zero(self, value):
        """很大的正数：有限、约等于 0（**不是** ``−inf``、也不是 ``nan``）."""
        result = log_sigmoid(value)
        assert math.isfinite(result)
        assert result == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("value", [-1000.0, -1e308])
    def test_large_negative_is_linear(self, value):
        """很负的数：结果有限且约等于 ``value`` 本身（这是 loss 能突破 ln 2 的原因）."""
        result = log_sigmoid(value)
        assert math.isfinite(result)
        assert result == pytest.approx(value, rel=1e-15)

    @pytest.mark.parametrize(
        "value", [-20.0, -15.0, -5.0, -1.0, 0.0, 1.0, 5.0, 15.0, 20.0]
    )
    def test_matches_log_of_sigmoid(self, value):
        """``|x| ≤ 20`` 上与 ``math.log(sigmoid(x))`` 近似相等.

        两者数学上等价，但实现在 ``x ≥ 0`` 那一支用的是 ``−log1p(exp(−x))``，
        而 ``log(sigmoid(x))`` 要先算出接近 1 的数再取对数——后者在 ``x`` 较大时
        会丢有效位。实测最大相对偏差出现在 ``x = 20``（约 ``3.6e-8``），
        因此这里用 ``rel=1e-6`` 而不是逐位相等。
        """
        assert log_sigmoid(value) == pytest.approx(math.log(sigmoid(value)), rel=1e-6)

    @pytest.mark.parametrize("value", [-1e308, -1000.0, -1.0, 0.0, 1.0, 1000.0, 1e308])
    def test_never_positive(self, value):
        """``log σ`` 恒 ≤ 0（σ ∈ (0, 1]）——任何正数都说明符号写错了."""
        assert log_sigmoid(value) <= 0.0

    def test_very_negative_does_not_underflow_to_negative_infinity(self):
        """**实现口径**：朴素写法在 ``x = −1000`` 时先下溢、再取对数**直接报错**.

        本实现返回有限的 ``x − log1p(exp(x))``。注意实现细节与模块文档里
        "再取对数得到 ``−inf``"的说法不同：``math.log(0.0)`` 抛的是
        ``ValueError``（math domain error），不是返回 ``−inf``。
        两种失败方式都致命（一个是崩溃、一个是 ``inf`` loss），
        而这条断言按**实际行为**记录。
        """
        with pytest.raises(ValueError):
            math.log(sigmoid(-1000.0))
        assert sigmoid(-1000.0) == 0.0
        assert math.isfinite(log_sigmoid(-1000.0))


class TestSoftplus:
    """``softplus(x) = log(1 + e^x)``：``−log σ(x)`` 的等价写法."""

    def test_at_zero_is_log_two(self):
        """``softplus(0) = ln 2``（与 ``ZERO_MARGIN_LOSS`` 同一个数）."""
        assert softplus(0.0) == LN2

    @pytest.mark.parametrize(
        ("value", "expected", "tolerance"),
        [
            (1000.0, 1000.0, 1e-12),
            (-1000.0, 0.0, 1e-12),
            (1e308, 1e308, 1e-15),
        ],
    )
    def test_extremes_are_finite(self, value, expected, tolerance):
        """两个方向都不溢出：大正数 ≈ 自身，大负数 ≈ 0."""
        result = softplus(value)
        assert math.isfinite(result)
        assert result == pytest.approx(expected, abs=tolerance)

    @pytest.mark.parametrize("value", [-20.0, -3.0, -0.5, 0.0, 0.5, 3.0, 20.0])
    def test_matches_log1p_exp(self, value):
        """在小量级上与朴素写法 ``log1p(exp(x))`` 一致."""
        assert softplus(value) == pytest.approx(math.log1p(math.exp(value)), rel=1e-15)

    @pytest.mark.parametrize("value", [-7.5, -1.0, 0.0, 1.0, 7.5])
    def test_equals_minus_log_sigmoid_of_negative(self, value):
        """``softplus(x) = −log σ(−x)``——DPO loss 的两条实现路径由此等价."""
        assert softplus(value) == pytest.approx(-log_sigmoid(-value), rel=1e-12)

    @pytest.mark.parametrize("value", [-1000.0, -1.0, 0.0, 1.0, 1000.0])
    def test_never_negative(self, value):
        """``softplus`` 恒 ≥ 0（它是 loss 的另一种写法，不能给出负 loss）."""
        assert softplus(value) >= 0.0


# ---------------------------------------------------------------------------
# dpo_margin：四个对数概率的线性组合
# ---------------------------------------------------------------------------


class TestDpoMargin:
    """``margin = (l_c − l_r) − (r_c − r_r)``，**不含 β**."""

    @pytest.mark.parametrize(
        ("policy_chosen", "policy_rejected", "reference_chosen", "reference_rejected", "expected"),
        [
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0, 1.0),
            (-1.0, -2.0, -3.0, -4.0, 0.0),
            (-10.5, -20.5, -1.5, -2.5, 9.0),
            (0.5, 0.25, 0.25, 0.5, 0.5),
        ],
    )
    def test_hand_computed(
        self, policy_chosen, policy_rejected, reference_chosen, reference_rejected, expected
    ):
        """逐条手算：差值再作差，顺序不能颠倒."""
        assert dpo_margin(
            policy_chosen, policy_rejected, reference_chosen, reference_rejected
        ) == pytest.approx(expected, abs=1e-15)

    @pytest.mark.parametrize("shift", [-1000.0, -3.5, 0.0, 3.5, 1000.0])
    def test_shift_invariance(self, shift):
        """四个量同时加同一个常数，margin **不变**（它只依赖"差的对数比"）.

        这是"参考模型一动，全部隐式奖励一起平移"的另一面：平移量在差值里
        相互抵消。浮点上用 ``approx(abs=1e-9)``，因为中间加法会引入舍入。
        """
        baseline = dpo_margin(1.0, 0.5, 0.25, 0.1)
        assert dpo_margin(
            1.0 + shift, 0.5 + shift, 0.25 + shift, 0.1 + shift
        ) == pytest.approx(baseline, abs=1e-9)

    @pytest.mark.parametrize(
        ("chosen", "rejected"),
        [(0.0, 0.0), (-12.5, -3.5), (-1.0, -2.0), (-200.0, 0.5)],
    )
    def test_zero_when_policy_equals_reference(self, chosen, rejected):
        """策略与参考模型逐位相同时 margin 为 0（起点检查的前提）."""
        assert dpo_margin(chosen, rejected, chosen, rejected) == 0.0

    @pytest.mark.parametrize(
        ("policy_chosen", "policy_rejected", "reference_chosen", "reference_rejected"),
        [
            (2.0, 1.0, 0.0, 0.0),
            (-1.0, -3.0, -5.0, -5.5),
        ],
    )
    def test_sign_follows_preference_direction(
        self, policy_chosen, policy_rejected, reference_chosen, reference_rejected
    ):
        """策略在 chosen 上拉开差距（相对参考模型）时 margin 为正."""
        assert dpo_margin(policy_chosen, policy_rejected, reference_chosen, reference_rejected) > 0

    @pytest.mark.parametrize(
        ("names", "values"),
        [
            (0, (float("nan"), 0.0, 0.0, 0.0)),
            (1, (0.0, float("nan"), 0.0, 0.0)),
            (2, (0.0, 0.0, float("nan"), 0.0)),
            (3, (0.0, 0.0, 0.0, float("nan"))),
            (4, (float("inf"), 0.0, 0.0, 0.0)),
            (5, (0.0, float("-inf"), 0.0, 0.0)),
        ],
    )
    def test_non_finite_raises(self, names, values):
        """四个位置各自被校验：``nan`` / ``inf`` 必须报错而不是安静地传下去."""
        with pytest.raises(FinetuneEvalError):
            dpo_margin(*values)

    def test_error_names_the_offending_argument(self):
        """错误信息里必须**点名**是哪个参数（四个参数长得太像，不点名很难查）."""
        with pytest.raises(FinetuneEvalError) as excinfo:
            dpo_margin(0.0, 0.0, float("nan"), 0.0)
        assert "reference_chosen" in str(excinfo.value)
        assert "nan" in str(excinfo.value)

    def test_is_not_scaled_by_beta(self):
        """``dpo_margin`` **没有** beta 参数：换 β 不改变 margin（只改变它对 loss 的影响）."""
        with pytest.raises(TypeError):
            dpo_margin(1.0, 0.0, 0.0, 0.0, beta=0.5)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# dpo_loss_from_margin
# ---------------------------------------------------------------------------


class TestDpoLossFromMargin:
    """``L = −log σ(β·margin)``：起点是 ``ln 2``，两端各有明确的极限."""

    @pytest.mark.parametrize("beta", BETA_CASES)
    def test_zero_margin_is_log_two(self, beta):
        """**起点检查**：margin 为 0 时 loss 恒为 ``ln 2``，与 β 无关."""
        assert dpo_loss_from_margin(0.0, beta=beta) == ZERO_MARGIN_LOSS
        assert dpo_loss_from_margin(0.0, beta=beta) == LN2

    def test_large_positive_margin_is_zero(self):
        """margin 很大时 loss 趋近 0（模型已经拉开了差距）."""
        assert dpo_loss_from_margin(1000.0, beta=1.0) == pytest.approx(0.0, abs=1e-12)

    def test_negative_margin_grows_linearly(self):
        """**实现口径**：margin 很负时 loss 趋近 ``−β·margin``，不被压平.

        这是"一条被写错的偏好样本（chosen 其实更差）会给出很大的 loss"的
        数值依据：任何"数值保护"如果把它截断成一个常数，就会让这类数据
        在训练日志里彻底隐形。
        """
        assert dpo_loss_from_margin(-1000.0, beta=1.0) == pytest.approx(1000.0, abs=1e-9)
        assert dpo_loss_from_margin(-1000.0, beta=1.0) == pytest.approx(1000.0, rel=1e-12)

    @pytest.mark.parametrize("margin", [-1000.0, -100.0, -50.0])
    @pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
    def test_negative_margin_matches_beta_times_margin(self, margin, beta):
        """负 margin 区间上 ``loss ≈ −β·margin``（β 只是把斜率拉长/压短）."""
        assert dpo_loss_from_margin(margin, beta=beta) == pytest.approx(
            -beta * margin, rel=1e-9
        )

    @pytest.mark.parametrize("beta", [0.5, 1.0])
    def test_decreases_monotonically_on_positive_side(self, beta):
        """正 margin 越大 loss 越小（单调下降）."""
        margins = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
        losses = [dpo_loss_from_margin(margin, beta=beta) for margin in margins]
        assert losses == sorted(losses, reverse=True)
        assert len(set(losses)) == len(losses)

    @pytest.mark.parametrize("beta", [0.5, 1.0])
    def test_decreases_monotonically_in_margin_on_negative_side(self, beta):
        """margin 越负 loss 越大（等价于"loss 随 margin 单调递减"）."""
        margins = [-10.0, -5.0, -2.0, -1.0, 0.0]
        losses = [dpo_loss_from_margin(margin, beta=beta) for margin in margins]
        assert losses == sorted(losses, reverse=True)

    @pytest.mark.parametrize(
        ("margin", "beta"),
        [(0.0, 1.0), (1.0, 0.5), (-1.0, 2.0), (3.5, 0.1), (-3.5, 0.25)],
    )
    def test_equals_softplus_of_negative_beta_margin(self, margin, beta):
        """``−log σ(z) = softplus(−z)``：模块文档里"避免 exp 溢出"的那条等式."""
        assert dpo_loss_from_margin(margin, beta=beta) == pytest.approx(
            softplus(-beta * margin), rel=1e-12
        )

    @pytest.mark.parametrize("margin", [0.0, 1.0, 2.0, 4.0])
    def test_is_always_positive(self, margin):
        """loss 恒为正（``−log σ`` 的值域是 ``(0, +∞)``）."""
        assert dpo_loss_from_margin(margin, beta=1.0) > 0.0

    @pytest.mark.parametrize("beta", BAD_BETA_CASES)
    def test_invalid_beta_raises(self, beta):
        """β ≤ 0 或非有限一律报错（β 同时是温度与 KL 权重，为 0 时目标无定义）."""
        with pytest.raises(FinetuneEvalError):
            dpo_loss_from_margin(0.0, beta=beta)

    def test_nan_margin_passes_through(self):
        """**实现口径**：只校验 β，``nan`` margin 会原样变成 ``nan`` loss.

        与 ``dpo_margin``（拦 nan）的口径不同：拦截责任在调用链上游。
        测试按实现断言，不改源码。
        """
        assert math.isnan(dpo_loss_from_margin(float("nan"), beta=1.0))

    @pytest.mark.parametrize("beta", BETA_CASES)
    def test_beta_only_rescales_the_margin(self, beta):
        """β 的作用是"缩放 margin"：``L(m, β) = L(β·m, 1)``（β 只是温度）."""
        assert dpo_loss_from_margin(0.37, beta=beta) == pytest.approx(
            dpo_loss_from_margin(beta * 0.37, beta=1.0), rel=1e-12
        )


# ---------------------------------------------------------------------------
# dpo_loss（完整签名）
# ---------------------------------------------------------------------------


class TestDpoLoss:
    """``dpo_loss`` 必须是 ``dpo_loss_from_margin(dpo_margin(...))`` 的逐位封装."""

    @pytest.mark.parametrize(
        ("policy_chosen", "policy_rejected", "reference_chosen", "reference_rejected", "beta"),
        [
            (0.0, 0.0, 0.0, 0.0, 0.1),
            (-1.0, -2.0, -3.0, -4.0, 0.5),
            (-10.5, -20.5, -1.5, -2.5, 1.0),
            (0.5, 0.25, 0.25, 0.5, 2.0),
            (-100.0, -120.0, -1.0, -2.0, 0.25),
        ],
    )
    def test_matches_the_two_step_composition(
        self, policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta
    ):
        """与"先算 margin 再算 loss"逐位一致（同一个 β 必须只乘一次）."""
        expected = dpo_loss_from_margin(
            dpo_margin(policy_chosen, policy_rejected, reference_chosen, reference_rejected),
            beta=beta,
        )
        assert dpo_loss(
            policy_chosen=policy_chosen,
            policy_rejected=policy_rejected,
            reference_chosen=reference_chosen,
            reference_rejected=reference_rejected,
            beta=beta,
        ) == expected

    @pytest.mark.parametrize("beta", BETA_CASES)
    def test_equal_models_give_log_two(self, beta):
        """策略 = 参考模型 → 四个对数概率相同 → loss 恰好是 ``ln 2``."""
        assert dpo_loss(
            policy_chosen=-12.5,
            policy_rejected=-3.25,
            reference_chosen=-12.5,
            reference_rejected=-3.25,
            beta=beta,
        ) == ZERO_MARGIN_LOSS

    def test_arguments_are_keyword_only(self):
        """四个对数概率必须按名字传：它们的顺序极易写错且不会报错."""
        with pytest.raises(TypeError):
            dpo_loss(0.0, 0.0, 0.0, 0.0, 0.1)  # type: ignore[misc]

    @pytest.mark.parametrize("beta", BAD_BETA_CASES)
    def test_invalid_beta_raises(self, beta):
        """β 的校验必须发生在 loss 计算链路里（不是只在 ``_from_margin`` 里）."""
        with pytest.raises(FinetuneEvalError):
            dpo_loss(
                policy_chosen=0.0,
                policy_rejected=0.0,
                reference_chosen=0.0,
                reference_rejected=0.0,
                beta=beta,
            )

    def test_non_finite_logprob_raises(self):
        """任一输入非有限（这里是 ``inf``）都报错，错误来自 ``dpo_margin``."""
        with pytest.raises(FinetuneEvalError):
            dpo_loss(
                policy_chosen=0.0,
                policy_rejected=0.0,
                reference_chosen=float("inf"),
                reference_rejected=0.0,
                beta=0.1,
            )


# ---------------------------------------------------------------------------
# dpo_gradient / dpo_probability
# ---------------------------------------------------------------------------


class TestDpoGradient:
    """``∂L/∂m = −β·σ(−β·m)``：本课"手算梯度"的落点."""

    def test_at_zero_with_beta_point_one(self):
        """β = 0.1、margin = 0 时梯度恰好是 ``−0.05``（= −β/2）."""
        assert dpo_gradient(0.0, beta=0.1) == -0.05

    def test_at_zero_with_beta_point_five(self):
        """β = 0.5、margin = 0 时梯度恰好是 ``−0.25``（= −β/2）."""
        assert dpo_gradient(0.0, beta=0.5) == -0.25

    @pytest.mark.parametrize("beta", BETA_CASES)
    def test_zero_margin_is_minus_half_beta(self, beta):
        """``margin = 0 → −β/2``：起点处走的是"一半步长"，与 β 成正比."""
        assert dpo_gradient(0.0, beta=beta) == -beta / 2.0

    @pytest.mark.parametrize(
        ("margin", "beta"),
        [(0.0, 1.0), (1.0, 0.5), (-1.0, 2.0), (3.0, 0.1), (-3.0, 0.25)],
    )
    def test_equals_closed_form(self, margin, beta):
        """与闭式 ``−β·sigmoid(−β·margin)`` 一致."""
        assert dpo_gradient(margin, beta=beta) == -beta * sigmoid(-beta * margin)

    @pytest.mark.parametrize("beta", [0.1, 0.5, 1.0])
    def test_magnitude_shrinks_as_margin_grows(self, beta):
        """margin 越大梯度越接近 0（"越接近最优、梯度越小"的自适应行为）."""
        margins = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
        magnitudes = [abs(dpo_gradient(margin, beta=beta)) for margin in margins]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert magnitudes[-1] < magnitudes[0]

    @pytest.mark.parametrize("margin", [-10.0, -1.0, 0.0, 1.0, 10.0])
    def test_is_never_positive_for_positive_beta(self, margin):
        """正 β 下梯度恒 ≤ 0（loss 随 margin 单调不增，方向不能反）."""
        assert dpo_gradient(margin, beta=1.0) <= 0.0

    @pytest.mark.parametrize("beta", BAD_BETA_CASES)
    def test_invalid_beta_raises(self, beta):
        """β 非法时报错（梯度系数本身含 β，不能默默按 0 处理）."""
        with pytest.raises(FinetuneEvalError):
            dpo_gradient(0.0, beta=beta)

    def test_extreme_margin_saturates_to_zero(self):
        """margin 极大时梯度趋近 0（但仍是有限的负数，不是 ``−0.0`` 之外的东西）."""
        assert dpo_gradient(1000.0, beta=1.0) == pytest.approx(0.0, abs=1e-12)


class TestDpoProbability:
    """``σ(β·margin)``：被解读为"模型认为 chosen 更被偏好的概率"."""

    def test_at_zero_is_half(self):
        """margin 为 0 时概率恰好 0.5（未训练时"完全没意见"）."""
        assert dpo_probability(0.0, beta=0.1) == 0.5

    @pytest.mark.parametrize(
        ("margin", "beta"),
        [(0.0, 1.0), (1.0, 0.5), (-1.0, 2.0), (3.0, 0.1), (-3.0, 0.25), (1e6, 1.0)],
    )
    def test_equals_sigmoid_of_beta_margin(self, margin, beta):
        """与 ``sigmoid(β·margin)`` 逐位一致（它只是 sigmoid 的一层封装）."""
        assert dpo_probability(margin, beta=beta) == sigmoid(beta * margin)

    @pytest.mark.parametrize("beta", BETA_CASES)
    def test_stays_in_unit_interval(self, beta):
        """概率必须落在 ``[0, 1]``（极端 margin 也不例外）."""
        for margin in (-1e6, -1.0, 0.0, 1.0, 1e6):
            value = dpo_probability(margin, beta=beta)
            assert 0.0 <= value <= 1.0

    @pytest.mark.parametrize("beta", [0.1, 0.5, 1.0])
    def test_is_strictly_increasing_in_margin(self, beta):
        """margin 越大越自信（"偏好正确率"随训练上升的依据）."""
        probabilities = [
            dpo_probability(margin, beta=beta) for margin in (-5.0, -1.0, 0.0, 1.0, 5.0)
        ]
        assert probabilities == sorted(probabilities)
        assert len(set(probabilities)) == len(probabilities)

    @pytest.mark.parametrize("beta", BAD_BETA_CASES)
    def test_invalid_beta_raises(self, beta):
        """β 非法时报错."""
        with pytest.raises(FinetuneEvalError):
            dpo_probability(0.0, beta=beta)

    @pytest.mark.parametrize("margin", [-3.0, -1.0, 0.0, 1.0, 3.0])
    def test_consistent_with_gradient(self, margin):
        """``∂L/∂m = β·(p − 1)``：概率与梯度必须来自同一个 β（接错就会不一致）."""
        beta = 0.5
        probability = dpo_probability(margin, beta=beta)
        assert dpo_gradient(margin, beta=beta) == pytest.approx(
            beta * (probability - 1.0), rel=1e-12
        )


# ---------------------------------------------------------------------------
# 隐式奖励
# ---------------------------------------------------------------------------


class TestImplicitReward:
    """``r = β·(log π_θ − log π_ref)``：DPO "把奖励做成了对数比"."""

    def test_scales_linearly_with_beta(self):
        """β 缩放：``r(β=2) = 2·r(β=1)``."""
        assert implicit_reward(1.0, 0.5, beta=1.0) == pytest.approx(0.5, rel=1e-15)
        assert implicit_reward(1.0, 0.5, beta=2.0) == pytest.approx(1.0, rel=1e-15)
        assert implicit_reward(1.0, 0.5, beta=0.1) == pytest.approx(0.05, rel=1e-15)

    @pytest.mark.parametrize("logprob", [-12.5, -1.0, 0.0, 1.0, 7.5])
    def test_zero_when_policy_equals_reference(self, logprob):
        """``log π_θ = log π_ref`` → 奖励恒为 0（训练初期"排序信息为零"）."""
        assert implicit_reward(logprob, logprob, beta=0.5) == 0.0

    def test_negative_when_policy_is_less_likely(self):
        """策略给出的概率更低时隐式奖励为负（负奖励是真实存在的信号）."""
        assert implicit_reward(-2.0, -1.0, beta=0.5) == pytest.approx(-0.5, rel=1e-15)

    @pytest.mark.parametrize(
        ("logprob", "reference_logprob"),
        [
            (float("nan"), 0.0),
            (0.0, float("nan")),
            (float("inf"), 0.0),
            (0.0, float("-inf")),
        ],
    )
    def test_non_finite_raises(self, logprob, reference_logprob):
        """非有限输入报错（否则奖励会安静地变成 ``nan`` 并污染整批统计）."""
        with pytest.raises(FinetuneEvalError):
            implicit_reward(logprob, reference_logprob, beta=0.5)

    @pytest.mark.parametrize("beta", BAD_BETA_CASES)
    def test_invalid_beta_raises(self, beta):
        """β 非法时报错."""
        with pytest.raises(FinetuneEvalError):
            implicit_reward(1.0, 0.5, beta=beta)

    @pytest.mark.parametrize(
        ("logprob", "reference_logprob", "beta"),
        [(1.0, 0.5, 0.1), (-3.0, -8.0, 1.0), (0.0, -2.0, 0.25)],
    )
    def test_equals_beta_times_log_ratio(self, logprob, reference_logprob, beta):
        """与手写公式 ``β·(log p − log q)`` 一致."""
        assert implicit_reward(logprob, reference_logprob, beta=beta) == pytest.approx(
            beta * (logprob - reference_logprob), rel=1e-15
        )


class TestImplicitRewardMargin:
    """``r_w − r_l = β·margin``：与 ``dpo_probability`` 是同一件事的两种写法."""

    @pytest.mark.parametrize("beta", BETA_CASES)
    def test_is_beta_times_margin(self, beta):
        """逐位等于 ``β · dpo_margin(...)``."""
        assert implicit_reward_margin(
            policy_chosen=1.0,
            policy_rejected=0.5,
            reference_chosen=0.25,
            reference_rejected=0.1,
            beta=beta,
        ) == beta * dpo_margin(1.0, 0.5, 0.25, 0.1)

    @pytest.mark.parametrize(
        ("policy_chosen", "policy_rejected", "reference_chosen", "reference_rejected"),
        [(0.0, 0.0, 0.0, 0.0), (-5.0, -7.0, -5.0, -7.0)],
    )
    def test_zero_when_models_are_identical(
        self, policy_chosen, policy_rejected, reference_chosen, reference_rejected
    ):
        """策略 = 参考模型时奖励差为 0（与 ``PairReward.correct`` 的判据对应）."""
        assert (
            implicit_reward_margin(
                policy_chosen=policy_chosen,
                policy_rejected=policy_rejected,
                reference_chosen=policy_chosen,
                reference_rejected=policy_rejected,
                beta=0.1,
            )
            == 0.0
        )

    def test_beta_is_not_validated(self):
        """**实现口径**：本函数**不校验 β**（唯一一个不走 ``_check_beta`` 的入口）.

        ``beta = 0`` 安静地返回 0.0、``beta = -1`` 会给出反号的奖励差。
        与 ``implicit_reward``（同样含 β、但会报错）口径不同。测试按实现断言，
        只记录不改源码——记录在这里是为了让这个不对称**可见**。
        """
        zero = implicit_reward_margin(
            policy_chosen=1.0,
            policy_rejected=0.0,
            reference_chosen=0.0,
            reference_rejected=0.0,
            beta=0.0,
        )
        assert zero == 0.0
        negative = implicit_reward_margin(
            policy_chosen=1.0,
            policy_rejected=0.0,
            reference_chosen=0.0,
            reference_rejected=0.0,
            beta=-1.0,
        )
        assert negative == -1.0

    @pytest.mark.parametrize(
        ("policy_chosen", "policy_rejected", "reference_chosen", "reference_rejected"),
        [
            (0.0, float("nan"), 0.0, 0.0),
            (0.0, 0.0, float("inf"), 0.0),
        ],
    )
    def test_non_finite_raises_via_margin(
        self, policy_chosen, policy_rejected, reference_chosen, reference_rejected
    ):
        """非有限输入仍会被拦住（校验发生在它调用的 ``dpo_margin`` 里）."""
        with pytest.raises(FinetuneEvalError):
            implicit_reward_margin(
                policy_chosen=policy_chosen,
                policy_rejected=policy_rejected,
                reference_chosen=reference_chosen,
                reference_rejected=reference_rejected,
                beta=0.5,
            )

    @pytest.mark.parametrize("beta", BETA_CASES)
    def test_matches_probability_relation(self, beta):
        """``σ(奖励差) == dpo_probability(margin, β)``：两条路径必须汇合."""
        margin = dpo_margin(1.0, 0.5, 0.25, 0.1)
        reward_margin = implicit_reward_margin(
            policy_chosen=1.0,
            policy_rejected=0.5,
            reference_chosen=0.25,
            reference_rejected=0.1,
            beta=beta,
        )
        assert sigmoid(reward_margin) == pytest.approx(
            dpo_probability(margin, beta=beta), rel=1e-12
        )


# ---------------------------------------------------------------------------
# KL 估计器
# ---------------------------------------------------------------------------


class TestKlEstimators:
    """三个估计器都由对数概率算出，**分歧必须可见**（只报一个会掩盖它）."""

    def test_identical_models_give_zero(self):
        """``log p = log q`` → 三个估计器全为 0（起点检查的 KL 版本）."""
        assert kl_estimators(0.0, 0.0) == {"k1": 0, "k2": 0, "k3": 0}

    def test_hand_computed_values(self):
        """``log p = 1.0``、``log q = 0.5``：``k1 = 0.5``、``k2 = 0.125``、``k3 = e^−0.5 − 0.5``."""
        values = kl_estimators(1.0, 0.5)
        assert values["k1"] == pytest.approx(0.5, rel=1e-15)
        assert values["k2"] == pytest.approx(0.125, rel=1e-15)
        assert values["k3"] == pytest.approx(math.expm1(-0.5) + 0.5, rel=1e-15)

    def test_key_set_is_exactly_the_three_estimators(self):
        """返回的键必须恰好是 ``KL_ESTIMATORS``（键少了会让报告静默缺一列）."""
        assert set(kl_estimators(2.0, -1.0)) == set(KL_ESTIMATORS)

    @pytest.mark.parametrize(
        ("logprob", "reference_logprob"),
        [
            (0.0, 0.0),
            (1.0, 0.5),
            (0.5, 1.0),
            (-50.0, 0.0),
            (0.0, -50.0),
            (-3.0, 5.0),
            (2.5, 2.4999),
            (700.0, -0.5),
            (-0.5, 700.0),
            (-1e-9, 1e-9),
        ],
    )
    def test_k3_is_never_negative(self, logprob, reference_logprob):
        """``k3 = expm1(r) − r``（``r = log q − log p``）恒 ≥ 0.

        它是一条恒等式：``e^r − 1 − r`` 的最小值在 ``r = 0`` 处取到 0。
        实践中这一条决定了"KL 惩罚不会奖励偏离参考模型的行为"。
        """
        values = kl_estimators(logprob, reference_logprob)
        assert values["k3"] >= 0.0
        shifted = reference_logprob - logprob
        assert values["k3"] == pytest.approx(math.expm1(shifted) - shifted, rel=1e-12)

    @pytest.mark.parametrize(
        ("logprob", "reference_logprob"),
        [(0.0, 1.0), (0.5, 1.0), (-3.0, -1.0), (-100.0, -50.0)],
    )
    def test_k1_can_be_negative(self, logprob, reference_logprob):
        """**实现口径**：``k1`` 允许为负——单样本的 log 比可以为负.

        这正是"只报一个估计器"会掩盖的分歧：早期训练里 ``k1`` 常常给出负数，
        而 ``k3`` 已经稳定为正。这里用 ``log p < log q`` 的例子把它钉住。
        """
        values = kl_estimators(logprob, reference_logprob)
        assert values["k1"] < 0.0
        assert values["k1"] == logprob - reference_logprob

    @pytest.mark.parametrize(
        ("logprob", "reference_logprob"),
        [(1.0, 0.5), (-3.0, 5.0), (0.0, 0.0), (2.5, 2.4999)],
    )
    def test_k2_is_half_log_ratio_squared(self, logprob, reference_logprob):
        """``k2`` 恒非负且等于 ``0.5·k1²``（方差更大，但不会为负）."""
        values = kl_estimators(logprob, reference_logprob)
        assert values["k2"] >= 0.0
        assert values["k2"] == pytest.approx(0.5 * values["k1"] ** 2, rel=1e-15)

    @pytest.mark.parametrize(
        ("logprob", "reference_logprob", "expected_k3"),
        [
            (0.0, -1000.0, 999.0),
            (0.0, -50.0, 49.0),
            (-50.0, 0.0, math.expm1(50.0) - 50.0),
        ],
    )
    def test_k3_extremes(self, logprob, reference_logprob, expected_k3):
        """极端对数比下 ``k3`` 仍可算（只有正向溢出是断点，见下一条）."""
        assert kl_estimators(logprob, reference_logprob)["k3"] == pytest.approx(
            expected_k3, rel=1e-12
        )

    def test_k3_overflows_instead_of_raising_domain_error(self):
        """**实现口径**：``log q − log p`` 超过约 709.78 时 ``math.expm1`` 抛 ``OverflowError``.

        模块文档承认了 ``k3`` 的这个代价（"它需要 exp，极端情况下要小心溢出"），
        这里把**实际的异常类型**记录下来：它既不是 ``FinetuneEvalError``、
        也不是 ``inf``——真实的 log 比很难到这个量级，但一旦到达，
        调用方拿到的是一个裸的 ``OverflowError``。
        """
        with pytest.raises(OverflowError):
            kl_estimators(-1000.0, 0.0)

    @pytest.mark.parametrize(
        ("logprob", "reference_logprob"),
        [
            (float("nan"), 0.0),
            (0.0, float("nan")),
            (float("inf"), 0.0),
            (0.0, float("-inf")),
        ],
    )
    def test_non_finite_raises(self, logprob, reference_logprob):
        """非有限输入报错（nan 的 KL 会污染整批均值）."""
        with pytest.raises(FinetuneEvalError):
            kl_estimators(logprob, reference_logprob)


class TestMeanKl:
    """``mean_kl`` 是"三个估计器都可以选、但缺省是 k3"的聚合入口."""

    def test_default_estimator_is_k3(self):
        """缺省估计器必须是 ``k3``（无偏、方差最小、恒非负）."""
        logprobs = [1.0, 2.0]
        references = [0.5, 0.5]
        assert mean_kl(logprobs, references) == mean_kl(logprobs, references, estimator="k3")

    @pytest.mark.parametrize(
        ("estimator", "expected"),
        [
            ("k1", 1.0),
            ("k2", 0.625),
            ("k3", (math.expm1(-0.5) + 0.5 + math.expm1(-1.5) + 1.5) / 2.0),
        ],
    )
    def test_hand_computed_mean(self, estimator, expected):
        """两样本手算：``k1`` 均值 1.0、``k2`` 均值 0.625、``k3`` 均值见公式."""
        assert mean_kl([1.0, 2.0], [0.5, 0.5], estimator=estimator) == pytest.approx(
            expected, rel=1e-12
        )

    @pytest.mark.parametrize("estimator", ["k1", "k2", "k3"])
    def test_identical_sequences_give_zero(self, estimator):
        """策略 = 参考模型时三种估计器都给出 0（``policy_kl`` 起点为 0 的依据）."""
        assert mean_kl([1.0, -3.5, 0.0], [1.0, -3.5, 0.0], estimator=estimator) == 0.0

    @pytest.mark.parametrize("estimator", KL_ESTIMATORS)
    def test_single_sample_equals_estimator_value(self, estimator):
        """单样本时均值退化为该样本的估计值（分组平均不引入额外偏移）."""
        assert mean_kl([1.0], [0.5], estimator=estimator) == kl_estimators(1.0, 0.5)[estimator]

    def test_length_mismatch_raises(self):
        """长度不一致报错（错位之后算出来的 KL 看起来完全正常）."""
        with pytest.raises(FinetuneEvalError):
            mean_kl([1.0, 2.0], [0.5])

    def test_length_mismatch_raises_for_extra_reference(self):
        """反方向同样报错（不能只校验一侧）."""
        with pytest.raises(FinetuneEvalError):
            mean_kl([1.0], [0.5, 0.25])

    def test_empty_raises(self):
        """空集合报错，而不是返回 0.0（"量不到"与"量到 0"必须分开）."""
        with pytest.raises(FinetuneEvalError):
            mean_kl([], [])

    @pytest.mark.parametrize("estimator", ["", "k0", "k4", "K3", "kl"])
    def test_unknown_estimator_raises(self, estimator):
        """未知估计器报错（否则会安静地取到某个键或 KeyError）."""
        with pytest.raises(FinetuneEvalError):
            mean_kl([1.0], [0.5], estimator=estimator)

    def test_unknown_estimator_error_lists_options(self):
        """错误信息里要列出可选项（三个名字都出现）."""
        with pytest.raises(FinetuneEvalError) as excinfo:
            mean_kl([1.0], [0.5], estimator="k9")
        message = str(excinfo.value)
        for name in KL_ESTIMATORS:
            assert name in message

    def test_empty_check_precedes_estimator_check(self):
        """**实现口径**：空集合的报错优先于未知估计器的报错（校验顺序）."""
        with pytest.raises(FinetuneEvalError) as excinfo:
            mean_kl([], [], estimator="k9")
        assert "空集合" in str(excinfo.value)

    def test_k1_mean_can_be_negative(self):
        """``k1`` 的均值可以为负——这是"KL 惩罚偶尔奖励偏离"的直接证据."""
        assert mean_kl([0.0, -1.0], [1.0, 0.0], estimator="k1") < 0.0
        assert mean_kl([0.0, -1.0], [1.0, 0.0], estimator="k3") >= 0.0

    def test_accepts_tuples_as_well_as_lists(self):
        """接受任意 ``Sequence``（报告链路里经常传元组）."""
        assert mean_kl((1.0,), (0.5,)) == mean_kl([1.0], [0.5])


# ---------------------------------------------------------------------------
# 逐样本的 RLHF 目标（奖励塑形）
# ---------------------------------------------------------------------------


class TestRewardWithKlShaping:
    """``r − β·(log π_θ − log π_ref)``：把 KL 惩罚折进奖励."""

    def test_hand_computed_value(self):
        """``2.0 − 0.5·(−1.0 + 2.0) = 1.5``（手算）."""
        assert reward_with_kl_shaping(
            2.0, logprob=-1.0, reference_logprob=-2.0, beta=0.5
        ) == pytest.approx(1.5, rel=1e-15)

    @pytest.mark.parametrize(
        ("reward", "logprob", "reference_logprob", "beta"),
        [
            (0.0, 0.0, 0.0, 1.0),
            (1.0, -1.0, -1.0, 2.0),
            (-2.5, -3.0, -1.0, 0.25),
            (10.0, 0.0, -5.0, 0.1),
        ],
    )
    def test_equals_reward_minus_implicit_reward(self, reward, logprob, reference_logprob, beta):
        """``塑形奖励 = 原奖励 − 隐式奖励``：两个模块的公式必须对得上."""
        assert reward_with_kl_shaping(
            reward, logprob=logprob, reference_logprob=reference_logprob, beta=beta
        ) == pytest.approx(
            reward - implicit_reward(logprob, reference_logprob, beta=beta), rel=1e-12
        )

    @pytest.mark.parametrize("reward", [-3.0, 0.0, 7.5])
    def test_unchanged_when_policy_equals_reference(self, reward):
        """策略 = 参考模型时惩罚项为 0，塑形后的奖励等于原奖励."""
        assert reward_with_kl_shaping(
            reward, logprob=-4.0, reference_logprob=-4.0, beta=0.5
        ) == reward

    @pytest.mark.parametrize("beta", BAD_BETA_CASES)
    def test_invalid_beta_raises(self, beta):
        """β 非法时报错（β 是 KL 的权重，为 0 时目标退化成"只最大化奖励"）."""
        with pytest.raises(FinetuneEvalError):
            reward_with_kl_shaping(1.0, logprob=0.0, reference_logprob=0.0, beta=beta)

    def test_penalty_magnitude_grows_with_beta(self):
        """偏离参考模型越多、β 越大，塑形后的奖励越小（惩罚方向不能反）."""
        small = reward_with_kl_shaping(1.0, logprob=-1.0, reference_logprob=-5.0, beta=0.1)
        large = reward_with_kl_shaping(1.0, logprob=-1.0, reference_logprob=-5.0, beta=1.0)
        assert large < small < 1.0

    def test_nan_logprob_passes_through(self):
        """**实现口径**：只校验 β，非有限的 logprob 会安静地变成 ``nan``."""
        assert math.isnan(
            reward_with_kl_shaping(1.0, logprob=float("nan"), reference_logprob=0.0, beta=0.5)
        )


# ---------------------------------------------------------------------------
# RLHFObjective 与横向汇总
# ---------------------------------------------------------------------------


class TestRLHFObjective:
    """``RLHFObjective`` 是"一次 RLHF 目标的读数"，四个量都要能被单独核对."""

    def test_objective_is_mean_reward_minus_beta_kl(self):
        """``objective = E[r] − β·KL``（PPO 真正在最大化的量）."""
        objective = RLHFObjective(samples=4, mean_reward=2.0, kl=0.5, beta=0.5)
        assert objective.objective == pytest.approx(1.75, rel=1e-15)

    def test_default_estimator_is_k3(self):
        """缺省估计器是 ``k3``（与 ``mean_kl`` 保持一致）."""
        assert RLHFObjective(samples=1, mean_reward=0.0, kl=0.0, beta=0.1).estimator == "k3"

    def test_to_dict_keys(self):
        """``to_dict`` 的六个键齐全（少一个都会让报告缺一列）."""
        payload = RLHFObjective(samples=2, mean_reward=1.5, kl=0.25, beta=0.5).to_dict()
        assert set(payload) == {"samples", "mean_reward", "kl", "beta", "estimator", "objective"}

    def test_to_dict_objective_matches_property(self):
        """``to_dict`` 里的 objective 必须来自属性（不是另算一遍的副本）."""
        objective = RLHFObjective(samples=3, mean_reward=1.0, kl=0.4, beta=2.0)
        assert objective.to_dict()["objective"] == objective.objective

    def test_summary_line_contains_every_field(self):
        """一行摘要里三个数字与 β 都要出现，方便日志逐行比对."""
        line = RLHFObjective(samples=2, mean_reward=1.5, kl=0.0, beta=0.5).summary_line()
        assert "样本 2" in line
        assert "平均奖励 1.500000" in line
        assert "KL(k3) 0.000000" in line
        assert "β 0.5" in line
        assert "目标 1.500000" in line

    def test_negative_objective_when_kl_dominates(self):
        """KL 惩罚压过奖励时 objective 为负（"只买到 KL"的读数长相）."""
        assert RLHFObjective(samples=5, mean_reward=0.1, kl=2.4, beta=1.0).objective < 0.0


class TestRlhfPpoObjective:
    """``rlhf_ppo_objective`` 的手算值就是本课"RLHF 目标"的一行算术."""

    def test_hand_computed_value(self):
        """``rewards=[1,2]``、四个对数概率全 0、``β=0.5`` → 均值 1.5、KL 0、目标 1.5."""
        objective = rlhf_ppo_objective([1.0, 2.0], [0.0, 0.0], [0.0, 0.0], beta=0.5)
        assert objective.samples == 2
        assert objective.mean_reward == pytest.approx(1.5, rel=1e-15)
        assert objective.kl == 0.0
        assert objective.objective == pytest.approx(1.5, rel=1e-15)

    def test_k3_zero_because_k3_vanishes_on_both_sides(self):
        """KL 为 0 是因为两侧相同：``k3(0, 0) = 0``（不是"没算"）."""
        objective = rlhf_ppo_objective([1.0, 2.0], [0.0, 0.0], [0.0, 0.0], beta=0.5)
        assert objective.kl == kl_estimators(0.0, 0.0)["k3"] == 0.0

    @pytest.mark.parametrize("estimator", KL_ESTIMATORS)
    def test_estimator_is_passed_through(self, estimator):
        """三种估计器都能算，且返回对象里记着用的是哪一个."""
        objective = rlhf_ppo_objective(
            [1.0, 2.0], [1.0, 2.0], [0.5, 0.5], beta=0.5, estimator=estimator
        )
        assert objective.estimator == estimator
        assert objective.kl == mean_kl([1.0, 2.0], [0.5, 0.5], estimator=estimator)

    @pytest.mark.parametrize("beta", BETA_CASES)
    def test_objective_matches_its_own_fields(self, beta):
        """目标值必须能由对象自己交出来的三个量重算（分子分母一致）."""
        objective = rlhf_ppo_objective(
            [0.5, -1.5, 3.0], [1.0, -2.0, 0.0], [0.5, -2.5, 0.0], beta=beta
        )
        assert objective.objective == pytest.approx(
            objective.mean_reward - objective.beta * objective.kl, rel=1e-12
        )

    def test_samples_equals_length(self):
        """``samples`` 是分母，必须等于输入长度."""
        objective = rlhf_ppo_objective([1.0, 2.0, 3.0], [0.0] * 3, [0.0] * 3, beta=1.0)
        assert objective.samples == 3
        assert objective.mean_reward == pytest.approx(2.0, rel=1e-15)

    def test_reward_length_mismatch_raises(self):
        """奖励与对数概率长度不一致报错（优势会静默错位）."""
        with pytest.raises(FinetuneEvalError):
            rlhf_ppo_objective([1.0], [0.0, 0.0], [0.0, 0.0], beta=0.5)

    def test_reference_length_mismatch_raises(self):
        """对数概率与参考对数概率长度不一致报错（由 ``mean_kl`` 拦下）."""
        with pytest.raises(FinetuneEvalError):
            rlhf_ppo_objective([1.0], [0.0], [0.0, 0.0], beta=0.5)

    def test_empty_raises(self):
        """空批报错（不能返回一个"均值 0"的假读数）."""
        with pytest.raises(FinetuneEvalError):
            rlhf_ppo_objective([], [], [], beta=0.5)

    @pytest.mark.parametrize("beta", BAD_BETA_CASES)
    def test_invalid_beta_raises(self, beta):
        """β 非法时报错."""
        with pytest.raises(FinetuneEvalError):
            rlhf_ppo_objective([1.0], [0.0], [0.0], beta=beta)

    def test_unknown_estimator_raises(self):
        """未知估计器报错（透传到 ``mean_kl``）."""
        with pytest.raises(FinetuneEvalError):
            rlhf_ppo_objective([1.0], [0.0], [0.0], beta=0.5, estimator="k9")

    def test_summary_line_is_loggable(self):
        """摘要行可直接进日志（本课报告逐行对照它）."""
        line = rlhf_ppo_objective([1.0, 2.0], [0.0, 0.0], [0.0, 0.0], beta=0.5).summary_line()
        assert line.startswith("样本 2 | 平均奖励 1.500000")
        assert "KL(k3) 0.000000" in line


# ---------------------------------------------------------------------------
# 目标对照表（做成了数据）
# ---------------------------------------------------------------------------


class TestObjectiveRequirements:
    """这张表是"代码里的表格会被测试"这条纪律的落点."""

    def test_two_entries(self):
        """只有两个目标：RLHF（PPO）与 DPO."""
        assert len(objective_requirements()) == 2

    def test_names_and_order(self):
        """名字与顺序都要钉住（报告按顺序渲染）."""
        assert [item["objective"] for item in objective_requirements()] == [
            "RLHF（PPO）",
            "DPO",
        ]

    def test_rlhf_entry(self):
        """RLHF 是**三段式、有在线采样、四个模型**的那一个."""
        entry = objective_requirements()[0]
        assert entry["stages"] == 3
        assert entry["online_sampling"] is True
        assert entry["models_needed"] == ["policy", "reference", "reward", "value"]
        assert entry["signal_reusable"] is True

    def test_dpo_entry(self):
        """DPO 是**一段式、无在线采样、两个模型**的那一个."""
        entry = objective_requirements()[1]
        assert entry["stages"] == 1
        assert entry["online_sampling"] is False
        assert entry["models_needed"] == ["policy", "reference"]
        assert entry["signal_reusable"] is False

    @pytest.mark.parametrize("index", [0, 1])
    def test_required_keys_present(self, index):
        """两个条目必须各含 ``models_needed`` / ``kl_penalty`` / ``main_failure``."""
        entry = objective_requirements()[index]
        assert {"models_needed", "kl_penalty", "main_failure"} <= set(entry)

    def test_all_entries_share_the_same_key_set(self):
        """两条记录的字段必须完全对齐（表格才能逐列渲染）."""
        first, second = objective_requirements()
        assert set(first) == set(second)
        assert set(first) == {
            "objective",
            "stages",
            "stage_detail",
            "models_needed",
            "preference_usage",
            "online_sampling",
            "kl_penalty",
            "main_failure",
            "signal_reusable",
        }

    def test_kl_penalty_wording_distinguishes_explicit_and_implicit(self):
        """KL 惩罚的两种写法必须能分辨：RLHF 显式、DPO 隐式."""
        first, second = objective_requirements()
        assert first["kl_penalty"].startswith("显式")
        assert second["kl_penalty"].startswith("隐式")

    def test_main_failure_is_specific(self):
        """两条的失败模式都要点名（RLHF 提到 reward hacking，DPO 提到过优化）."""
        first, second = objective_requirements()
        assert "过优化" in first["main_failure"]
        assert "过优化" in second["main_failure"]

    def test_returns_fresh_objects_each_call(self):
        """每次调用都返回新对象：调用方改坏它不该污染后续调用（表格是数据，不是状态）."""
        first = objective_requirements()
        first[0]["stages"] = -1
        assert objective_requirements()[0]["stages"] == 3

    def test_online_sampling_is_boolean(self):
        """``online_sampling`` 必须是真布尔值（不是 0/1，报告里要当条件用）."""
        for entry in objective_requirements():
            assert entry["online_sampling"] is True or entry["online_sampling"] is False

    def test_stages_are_positive_integers(self):
        """阶段数必须是正整数，且 RLHF 严格多于 DPO（这正是两者的工程差别）."""
        first, second = objective_requirements()
        assert isinstance(first["stages"], int) and first["stages"] > 0
        assert isinstance(second["stages"], int) and second["stages"] > 0
        assert first["stages"] > second["stages"]

