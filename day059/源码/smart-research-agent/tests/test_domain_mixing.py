"""day057 配比控制测试（M5-D8）：分组口径、不动点迭代、削减规则与缺口报告.

本文件钉住四件事：

1. **三种分组口径**（``source``/``origin``/``safety``）各自的分组键，以及
   未知口径必须抛错（分组错了，后面所有占比都是错的）；
2. **不动点迭代**：给"任一组不超过 X%"这条约束，砍完一组后总数变小、
   其余组占比反而变大，必须重算。本课数据集 16:5:16 收 0.4 的实测结果是
   迭代 **6** 轮后收敛到 10:5:10（丢 12 条）；换成 ``safety`` 口径
   21:16 收 0.4 则**收敛但不可达**（9 轮后只剩 2 条，两组各占 50%）；
3. **削减时该砍谁**：给了质量分就砍组内最低分的，没给就按原顺序；
4. **缺口报告**：``sum(targets) == sum(counts)`` 必须恒成立——
   否则"缺口"就变成了"凭空要变出样本"。

全部断言只用内存样本，不联网、不用随机数、不依赖当前工作目录。
"""

from __future__ import annotations

import json

import pytest

from smart_research_agent.domain_data.augment import OPS as AUG_OPS
from smart_research_agent.domain_data.augment import augment_example
from smart_research_agent.domain_data.errors import DomainDataError
from smart_research_agent.domain_data.mixing import (
    DEFAULT_GROUP_BY,
    DEFAULT_MAX_GROUP_RATIO,
    GROUP_BY_CHOICES,
    GROUP_NORMAL,
    GROUP_SAFETY,
    GROUP_UNKNOWN,
    MIN_GROUP_KEEP,
    ORIGIN_AUGMENTED,
    ORIGIN_ORIGINAL,
    apply_mix_plan,
    deficit_report,
    group_counts,
    group_key,
    mixing_scores,
    plan_mixing,
)
from smart_research_agent.domain_data.quality import quality_scores
from smart_research_agent.finetune.schema import TrainingExample

#: 与仓库真实数据同名的三个来源：16 : 5 : 16 正是本课清洗后的来源分布
SEED = "seed"
AGENT = "eval/agent_tasks"
REDTEAM = "eval/redteam_cases"

#: 真实语料的清洗后规模（用来对齐"实测数字"，见模块 docstring）
REAL_TOTAL = 37


def make(source: str, index: int, *, tags: tuple[str, ...] = ()) -> TrainingExample:
    """构造一条只用于配比统计的样本（内容足够长，不会被误当成异常输入）."""
    return TrainingExample(
        instruction=f"{source} 的第 {index} 条问题：如何评估检索质量？",
        output="这是一条用于配比计算的样本答案，内容足够长以避免触发清洗规则。",
        source=source,
        tags=tags,
        license="CC-BY-4.0",
    )


def batch(counts: dict[str, int], *, tags: tuple[str, ...] = ()) -> list[TrainingExample]:
    """按"来源 → 条数"构造批次，顺序固定（配比结果与顺序无关，但报告要可复现）."""
    return [
        make(source, index, tags=tags) for source, count in counts.items() for index in range(count)
    ]


def real_batch() -> list[TrainingExample]:
    """本课清洗后的来源分布：16（seed）: 5（评估轨迹）: 16（红队安全）."""
    return batch({SEED: 16, AGENT: 5, REDTEAM: 16})


def safety_batch() -> list[TrainingExample]:
    """``safety`` 口径下的 21 : 16 —— 21 条普通样本 + 16 条安全样本."""
    return batch({SEED: 21}) + batch({REDTEAM: 16}, tags=("safety",))


class TestGroupKey:
    """三种分组口径：都用**单个**分组，因此"占比之和为 1"总是成立."""

    def test_group_by_choices_and_default(self):
        """口径全集与缺省值（缺省按来源：来源是一定存在且一定单一的治理字段）."""
        assert GROUP_BY_CHOICES == ("source", "origin", "safety")
        assert DEFAULT_GROUP_BY == "source"
        assert DEFAULT_MAX_GROUP_RATIO == 0.5
        assert MIN_GROUP_KEEP == 1

    def test_source_view_uses_the_source_field(self):
        """``source`` 口径直接取 ``source`` 字段；空来源落到显式的 ``(unknown)``.

        空来源**不能丢弃**也不能静默并进别的组：把它归到一个显式命名的组里，
        "数据治理有问题"这件事才会出现在报表上。
        """
        assert group_key(make(SEED, 0)) == SEED
        assert group_key(make(AGENT, 0)) == AGENT
        empty = TrainingExample(instruction="没有来源的问题", output="答案内容足够长。")
        assert group_key(empty) == GROUP_UNKNOWN
        assert GROUP_UNKNOWN == "(unknown)"

    def test_origin_view_separates_original_from_augmented(self):
        """``origin`` 口径按 ``aug:*`` 标签区分：增强样本不能无声地把配比撑歪."""
        original = make(SEED, 0)
        assert group_key(original, group_by="origin") == ORIGIN_ORIGINAL

        augmented = augment_example(original, AUG_OPS["prefix"]).example
        assert augmented is not None
        assert group_key(augmented, group_by="origin") == ORIGIN_AUGMENTED
        # 三种口径互不干扰：同一条增强样本按来源看仍然是 seed
        assert group_key(augmented, group_by="source") == SEED

    def test_safety_view_uses_is_safety_example(self):
        """``safety`` 口径与 ``TrainingExample.is_safety_example()`` 同源."""
        safe = make(REDTEAM, 0, tags=("from:redteam", "safety", "jailbreak"))
        normal = make(SEED, 0, tags=("rag",))
        assert safe.is_safety_example() is True
        assert group_key(safe, group_by="safety") == GROUP_SAFETY
        assert group_key(normal, group_by="safety") == GROUP_NORMAL

    def test_unknown_group_by_raises(self):
        """未知口径必须抛 ``DomainDataError``，而不是退化成"不分组"."""
        with pytest.raises(DomainDataError, match="未知的分组口径"):
            group_key(make(SEED, 0), group_by="nope")
        with pytest.raises(DomainDataError, match="未知的分组口径"):
            group_counts([make(SEED, 0)], group_by="by_month")

    def test_group_counts_keeps_first_appearance_order(self):
        """计数键顺序 = 首次出现顺序（同一份数据两次跑出来的报告逐键一致）."""
        examples = [make(REDTEAM, 0), make(SEED, 0), make(REDTEAM, 1), make(AGENT, 0)]
        counts = group_counts(examples)
        assert counts == {REDTEAM: 2, SEED: 1, AGENT: 1}
        assert list(counts) == [REDTEAM, SEED, AGENT]


class TestPlanMixing:
    """上限口径的不动点迭代：实测数字逐条断言（它们是"硬删太贵"的定量答案）."""

    def test_fixed_point_on_the_16_5_16_distribution(self):
        """16:5:16 收 0.4 → 10:5:10，迭代 6 轮，丢 12 条，且最终可行.

        迭代过程（模块 docstring 里的实测表）：
        37(cap14) → 33(cap13) → 31(cap12) → 29(cap11) → 27(cap10) → 25(cap10) 收敛。
        砍掉一组后总数变小、其余组占比变大，所以必须重算——这正是
        "把超过的组砍到 40%" 这个直觉答案的错误之处。
        """
        plan = plan_mixing(real_batch(), max_ratio=0.4)
        assert plan.counts == {SEED: 16, AGENT: 5, REDTEAM: 16}
        assert plan.quotas == {SEED: 10, AGENT: 5, REDTEAM: 10}
        assert plan.total_in == REAL_TOTAL
        assert plan.total_out == 25
        assert plan.dropped == 12
        assert plan.iterations == 6
        assert plan.feasible is True
        assert plan.warnings == []
        assert plan.final_ratios() == {SEED: 0.4, AGENT: 0.2, REDTEAM: 0.4}
        assert plan.final_ratios()[SEED] <= 0.4

    def test_cap_0_5_is_a_safety_net_that_drops_nothing(self):
        """同一份数据收 0.5 一条都不削：缺省值是"安全网"，不是"日常手段".

        16/37 = 43.2% 落在 0.5 以内（cap = floor(0.5 × 37) = 18），
        所以第 1 轮就发现"没有超额组"并退出——这就是 day049 那条
        "该调权或分阶段训练，而不是硬删样本"在缺省参数下的样子。
        """
        plan = plan_mixing(real_batch(), max_ratio=0.5)
        assert plan.quotas == plan.counts
        assert plan.dropped == 0
        assert plan.total_out == REAL_TOTAL
        assert plan.iterations == 1
        assert plan.feasible is True

    def test_safety_view_at_0_4_converges_but_is_infeasible(self):
        """21:16 按 safety 口径收 0.4：迭代 9 轮后只剩 2 条，且**不可达**.

        "收敛不等于可达"：两组各占一半时，必然有一组 >= 50%，约束根本无法满足。
        此时正确的做法不是假装成功，而是 ``feasible=False`` 加一条告警——
        一个悄悄违反自己约束的护栏，比没有护栏更糟。
        """
        plan = plan_mixing(safety_batch(), group_by="safety", max_ratio=0.4)
        assert plan.counts == {GROUP_NORMAL: 21, GROUP_SAFETY: 16}
        assert plan.quotas == {GROUP_NORMAL: MIN_GROUP_KEEP, GROUP_SAFETY: MIN_GROUP_KEEP}
        assert plan.total_out == 2
        assert plan.iterations == 9
        assert plan.feasible is False
        assert plan.warnings  # 告警必须非空：这个结论本身就要被输出
        assert "无法全部落在 0.4 以内" in plan.warnings[-1]
        assert plan.final_ratios() == {GROUP_NORMAL: 0.5, GROUP_SAFETY: 0.5}

    def test_max_ratio_must_be_a_probability(self):
        """``max_ratio`` 必须落在 (0, 1]：0 会清空整批，>1 等于约束静默失效."""
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(DomainDataError, match=r"配比上限必须落在 \(0, 1\] 区间"):
                plan_mixing(real_batch(), max_ratio=bad)

    def test_single_group_warns_and_trims_nothing(self):
        """只有一个分组时告警且**不做无意义的迭代**（一组占 100%，上限无意义）."""
        single = batch({SEED: 5})
        plan = plan_mixing(single, max_ratio=0.1)
        assert plan.counts == {SEED: 5}
        assert plan.quotas == {SEED: 5}
        assert plan.total_out == 5
        assert plan.dropped == 0
        assert plan.iterations == 0
        assert plan.warnings and "只落在一个分组里" in plan.warnings[0]

    def test_empty_dataset_does_not_divide_by_zero(self):
        """空数据集：配额为空、总数为 0、占比是空字典（空集是**合法**的中间状态）.

        这条断言把"没有数据"与"数据全被削掉"区分开：两者都不抛错，
        但都必须能在报告里被读出来——占比是一个"0/0"式的除法，
        它只能在空集分支里被安全地短路掉。
        """
        plan = plan_mixing([])
        assert plan.counts == {}
        assert plan.quotas == {}
        assert plan.total_in == plan.total_out == plan.dropped == 0
        assert plan.final_ratios() == {}
        assert plan.feasible is True
        assert plan.warnings  # 一个分组都没有 → 告警

        kept, report = apply_mix_plan([], plan)
        assert kept == []
        assert report.total_in == report.total_out == 0
        assert report.final_ratios() == {}
        assert report.dropped_by_group() == {}
        assert json.loads(json.dumps(report.to_dict(), ensure_ascii=False))["total_out"] == 0

    def test_plan_projection_is_json_serialisable(self):
        """``to_dict`` 要能直接 ``json.dumps``（清单与报告都要落盘）."""
        plan = plan_mixing(real_batch(), max_ratio=0.4)
        payload = json.loads(json.dumps(plan.to_dict(), ensure_ascii=False))
        assert payload["quotas"] == plan.quotas
        assert payload["total_out"] == plan.total_out
        assert payload["dropped"] == 12
        assert payload["feasible"] is True
        assert payload["final_ratios"][SEED] == 0.4
        assert "配比" in plan.summary_line()


class TestApplyMixPlan:
    """按方案削减：组内保留谁由"有没有分数"决定，输出保持入参相对顺序."""

    def test_highest_scores_are_kept_within_a_group(self):
        """给了 ``scores`` 就砍组内**最低分**的那些（分数 = 下标，便于肉眼核对）.

        为什么不是"砍排在最后的"：排在最后可能只是采集顺序靠后，
        而质量分是唯一能回答"这条更该留"的依据。
        """
        examples = real_batch()
        plan = plan_mixing(examples, max_ratio=0.4)
        scores = [float(index) for index in range(len(examples))]

        kept, report = apply_mix_plan(examples, plan, scores=scores)
        assert report.total_in == REAL_TOTAL
        assert report.total_out == 25
        assert report.by_group_out == plan.quotas
        assert report.dropped_by_group() == {SEED: 6, REDTEAM: 6}

        # seed 组在下标 0~15：保留分最高的 10 条 → 下标 6~15
        seed_kept = [example for example in kept if example.source == SEED]
        assert seed_kept == [examples[index] for index in range(6, 16)]
        # redteam 组在下标 21~36：保留分最高的 10 条 → 下标 27~36
        redteam_kept = [example for example in kept if example.source == REDTEAM]
        assert redteam_kept == [examples[index] for index in range(27, 37)]
        # agent_tasks 组只有 5 条、配额正好 5 条，一条不削
        assert len([e for e in kept if e.source == AGENT]) == 5

    def test_without_scores_the_original_order_decides(self):
        """没给分数时按原顺序保留前 ``quota`` 条（让"只想看配比怎么变"的调用方省一步打分）."""
        examples = real_batch()
        plan = plan_mixing(examples, max_ratio=0.4)

        kept, report = apply_mix_plan(examples, plan)
        assert report.by_group_out == plan.quotas
        assert [example for example in kept if example.source == SEED] == [
            examples[index] for index in range(10)
        ]

    def test_output_preserves_input_relative_order(self):
        """削减后的数据集仍然"看起来像原来那份的子集"（逐条对照不需要在脑子里重排）."""
        examples = real_batch()
        plan = plan_mixing(examples, max_ratio=0.4)
        kept, _ = apply_mix_plan(examples, plan, scores=[float(i) for i in range(len(examples))])

        positions = [examples.index(example) for example in kept]
        assert positions == sorted(positions)
        assert len(set(positions)) == len(positions)

    def test_score_length_must_match_examples(self):
        """``scores`` 与 ``examples`` 必须等长（错位的分数会让削减砍错人）."""
        examples = real_batch()
        plan = plan_mixing(examples, max_ratio=0.4)
        with pytest.raises(DomainDataError, match="scores 长度"):
            apply_mix_plan(examples, plan, scores=[1.0])

    def test_report_carries_infeasibility_to_the_caller(self):
        """不可达这个结论必须随报告传到调用方（``feasible=False`` + 告警 + 占比）.

        混配比执行报告的字段与清单里的 ``mixing`` 阶段明细同一套形状，
        因此"配比到底达没达标"永远能在最终产物里被复述一遍。
        """
        examples = safety_batch()
        plan = plan_mixing(examples, group_by="safety", max_ratio=0.4)
        kept, report = apply_mix_plan(examples, plan)
        assert len(kept) == report.total_out == 2
        assert report.group_by == "safety"
        assert report.max_ratio == 0.4
        assert report.iterations == plan.iterations
        assert report.feasible is False
        assert report.warnings == list(plan.warnings)

        payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
        assert payload["total_out"] == report.total_out
        assert payload["feasible"] is False
        assert payload["warnings"]
        assert payload["by_group_out"] == report.by_group_out
        assert payload["final_ratios"] == {
            name: round(value, 4) for name, value in report.final_ratios().items()
        }
        assert "配比执行" in report.summary_line()


class TestDeficitReport:
    """目标口径：只回答"要真正补齐配比，还差多少条"，**不裁剪任何样本**."""

    def test_targets_always_sum_to_the_total(self):
        """``sum(targets) == sum(counts)`` 恒成立（最大余数法把取整损失补回来）.

        等权 3 组、总数 37：每组 ``floor(37/3) = 12``，余 1 条按"小数部分从大到小"
        发放；三组小数部分并列 → 按组名升序 → ``eval/agent_tasks`` 拿到那 1 条。
        这个 13 不是随便来的，是"最大余数法 + 组名升序"两条规则一起算出来的。
        """
        report = deficit_report(real_batch(), weights={SEED: 1.0, AGENT: 1.0, REDTEAM: 1.0})
        assert report.counts == {SEED: 16, AGENT: 5, REDTEAM: 16}
        assert report.targets == {SEED: 12, AGENT: 13, REDTEAM: 12}
        assert sum(report.targets.values()) == sum(report.counts.values()) == REAL_TOTAL
        assert report.total == REAL_TOTAL

        # 缺口 = 目标 − 实际：正数不足、负数超额
        assert report.deficits() == {SEED: -4, AGENT: 8, REDTEAM: -4}
        assert report.under() == {AGENT: 8}
        assert report.over() == {SEED: 4, REDTEAM: 4}

    def test_equal_weights_on_sixty_one_samples(self):
        """等权 3 组、总数 61：目标 21:20:20，缺口正好是 10 条（补最小那组）.

        ``floor(61/3) = 20`` 三组各 20 → 余 1 条按组名升序发给 ``g0``。
        所以"等权"并不意味着"目标 = 组大小"，它意味着每组拿到总数的三等份。
        """
        examples = batch({"g0": 31, "g1": 20, "g2": 10})
        report = deficit_report(examples, weights={"g0": 1.0, "g1": 1.0, "g2": 1.0})
        assert report.counts == {"g0": 31, "g1": 20, "g2": 10}
        assert report.targets == {"g0": 21, "g1": 20, "g2": 20}
        assert sum(report.targets.values()) == 61
        assert report.under() == {"g2": 10}
        assert report.over() == {"g0": 10}
        assert report.deficits()["g1"] == 0

    def test_weights_must_cover_every_group(self):
        """权重表漏掉一个组 = 悄悄宣称"这组目标为 0"，必须直接拒绝."""
        with pytest.raises(DomainDataError, match="权重表缺少分组"):
            deficit_report(real_batch(), weights={SEED: 1.0})

    def test_weights_must_be_positive(self):
        """权重为 0 的组无法被表达（要表达"不要这组"请直接从权重表里删掉它）."""
        with pytest.raises(DomainDataError, match="目标权重必须为正数"):
            deficit_report(real_batch(), weights={SEED: 1.0, AGENT: 0.0, REDTEAM: 1.0})
        with pytest.raises(DomainDataError, match="目标权重必须为正数"):
            deficit_report(real_batch(), weights={SEED: -1.0, AGENT: 1.0, REDTEAM: 1.0})

    def test_under_and_over_only_list_positive_numbers(self):
        """``under()``/``over()`` 只列正数，且两者不会同时包含同一组."""
        report = deficit_report(real_batch(), weights={SEED: 1.0, AGENT: 1.0, REDTEAM: 1.0})
        assert all(gap > 0 for gap in report.under().values())
        assert all(gap > 0 for gap in report.over().values())
        assert not set(report.under()) & set(report.over())
        assert set(report.under()) | set(report.over()) <= set(report.counts)
        assert report.deficits() == {SEED: -4, AGENT: 8, REDTEAM: -4}

    def test_deficit_projection_is_json_serialisable(self):
        """``to_dict`` 可 json.dumps，且逐键与属性方法一致."""
        report = deficit_report(real_batch(), weights={SEED: 1.0, AGENT: 1.0, REDTEAM: 1.0})
        payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
        assert payload["targets"] == report.targets
        assert payload["counts"] == report.counts
        assert payload["deficits"] == report.deficits()
        assert payload["under"] == report.under()
        assert payload["over"] == report.over()
        assert "配比缺口" in report.summary_line()

    def test_weights_are_normalised_to_the_active_groups(self):
        """只有实际出现的组进入权重计算（多传的组不会稀释目标）."""
        report = deficit_report(
            real_batch(), weights={SEED: 2.0, AGENT: 2.0, REDTEAM: 2.0, "other": 9.0}
        )
        assert set(report.weights) == {SEED, AGENT, REDTEAM}
        assert sum(report.targets.values()) == REAL_TOTAL


class TestMixingScores:
    """``mixing_scores`` 是 ``quality_scores`` 的薄别名（配比与质量唯一的接口）."""

    def test_mixing_scores_equals_quality_scores(self):
        """两者必须给出完全相同的排序键：不同就说明配比砍错了人."""
        examples = real_batch()
        assert mixing_scores(examples) == quality_scores(examples)
        scores = mixing_scores(examples)
        assert len(scores) == len(examples)
        assert all(isinstance(value, float) for value in scores)
        assert all(0.0 <= value <= 1.0 for value in scores)
