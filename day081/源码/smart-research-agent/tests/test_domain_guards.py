"""day057 构造期护栏测试：把"静默失效"的配置挡在流水线之外（全部离线）.

本文件不按模块划分，而按**主题**划分：它收集 `domain_data` 包里所有
"参数写错了会怎样"的判定。这样归类是有意的——这些校验散落在五个模块里，
只有放在一起才看得出它们守的是同一条纪律：

> **一个"看起来在跑、其实什么也没做（或什么都拒了）"的配置，
> 比一个抛异常的错误危险得多。**

三条护栏各自对应的真实故障：

- ``NearDuplicateIndex(k=0)`` / ``num_perm=0``：延迟到第一次 ``add`` 才报错，
  堆栈里出现的是 ``shingles``，一个"参数写错"的问题被伪装成"运行时出错"；
- ``threshold=0``：任何两条样本都算重复，整批样本被清空；
- ``threshold>1``：没有任何样本能满足，近重复检测**静默失效**；
- ``quality_threshold>1``：整批被质量门槛拒光，报告里只剩一行 ``kept: 0``。
"""

from __future__ import annotations

import pytest

from smart_research_agent.domain_data import (
    DEFAULT_QUALITY_THRESHOLD,
    DomainDataError,
    DomainDataPipeline,
    NearDuplicateIndex,
    QualityWeights,
    filter_by_quality,
    validate_threshold,
)
from smart_research_agent.finetune.schema import TrainingExample


def sample(index: int = 0) -> TrainingExample:
    """一条最小可用样本（现场构造，不读磁盘）."""
    return TrainingExample(
        instruction=f"什么是检索增强生成？第 {index} 问",
        output="检索增强生成是把检索到的外部知识作为上下文交给模型再生成答案的方法。",
        source="probe",
    )


class TestNearDuplicateIndexGuards:
    @pytest.mark.parametrize("bad_k", [0, -1])
    def test_shingle_window_validated_at_construction(self, bad_k):
        """``k`` 必须在构造期校验：否则错误会晚一步、且伪装成运行时异常。"""
        with pytest.raises(DomainDataError, match="shingle 窗口 k"):
            NearDuplicateIndex(k=bad_k)

    @pytest.mark.parametrize("bad_perm", [0, -3])
    def test_num_perm_validated_at_construction(self, bad_perm):
        """``num_perm`` 同理由构造期负责。"""
        with pytest.raises(DomainDataError, match="num_perm"):
            NearDuplicateIndex(num_perm=bad_perm)

    @pytest.mark.parametrize("bad_threshold", [0.0, -0.1, 1.5])
    def test_threshold_range_enforced(self, bad_threshold):
        """阈值必须落在 ``(0, 1]``：0 会清空整批，>1 会让检测静默失效。"""
        with pytest.raises(ValueError, match="近重复阈值"):
            NearDuplicateIndex(threshold=bad_threshold)

    @pytest.mark.parametrize("good_threshold", [0.5, 1.0])
    def test_legal_thresholds_pass(self, good_threshold):
        """合法阈值必须能构造——护栏不能把正常配置也挡掉。"""
        assert NearDuplicateIndex(threshold=good_threshold).threshold == good_threshold


class TestQualityThresholdGuard:
    @pytest.mark.parametrize("bad_threshold", [1.5, -0.01, 2.0])
    def test_out_of_range_threshold_rejected(self, bad_threshold):
        """``>1`` 的门槛不会报错，只会把整批样本静默拒光——所以必须提前拦。"""
        with pytest.raises(DomainDataError, match="质量门槛"):
            validate_threshold(bad_threshold)

    def test_filter_by_quality_validates_its_argument(self):
        """走公开过滤器时同样要拦（调用方不见得会先调 ``validate_threshold``）。"""
        with pytest.raises(DomainDataError, match="质量门槛"):
            filter_by_quality([sample()], threshold=1.2)

    @pytest.mark.parametrize("good_threshold", [0.0, DEFAULT_QUALITY_THRESHOLD, 1.0])
    def test_legal_thresholds_pass(self, good_threshold):
        """边界值 0.0 与 1.0 都合法：前者全放行、后者只放行满分样本。"""
        assert validate_threshold(good_threshold) == good_threshold

    def test_pipeline_validates_at_construction(self):
        """流水线构造期就拦下坏门槛——这是本包"校验前置"纪律的落点。"""
        with pytest.raises(DomainDataError, match="质量门槛"):
            DomainDataPipeline(quality_threshold=1.01)

    def test_extreme_threshold_empties_the_batch(self):
        """1.0 的门槛在真实样本上会拒掉不满分的样本：静默失效的"前一步"。"""
        result = filter_by_quality([sample()], threshold=1.0)
        assert result.total == 1
        assert result.kept == [] or all(score.total == 1.0 for score in result.scores)


class TestWeightGuards:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"length": 0.5},
            {"length": 0.3, "repetition": 0.3, "structure": 0.3},
        ],
    )
    def test_weight_sum_must_be_one(self, kwargs):
        """权重之和必须恰好是 1.0：否则不同批次的 ``total`` 不在同一尺度上。"""
        with pytest.raises(DomainDataError, match="权重之和"):
            QualityWeights(**kwargs)

    def test_negative_weight_rejected_before_the_sum_check(self):
        """负权重先被拦：报错信息要指向真正的错处，而不是"和不为 1"这个副作用.

        这是一条关于**报错顺序**的断言：负权重同样会让和为 0.9 != 1.0，
        但告诉调用方"维度 length 的权重不能为负"远比"权重之和不为 1"有用。
        """
        with pytest.raises(DomainDataError, match="不能为负"):
            QualityWeights(length=-0.1, repetition=0.65)
