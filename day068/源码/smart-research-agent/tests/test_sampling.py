"""day033 采样参数测试：SamplingParams 校验、temperature/top_p 数学、MockLLM 采样模拟."""

from __future__ import annotations

import pytest

from smart_research_agent.llm.base import Message
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.llm.sampling import (
    SamplingParams,
    nucleus_filter,
    softmax_with_temperature,
)


class TestSamplingParams:
    def test_defaults_valid(self):
        params = SamplingParams()
        assert params.temperature == 0.7
        assert params.top_p == 1.0
        assert params.max_tokens == 1024

    def test_temperature_bounds(self):
        SamplingParams(temperature=0.0)  # 贪心解码，合法
        SamplingParams(temperature=2.0)
        with pytest.raises(ValueError, match="temperature"):
            SamplingParams(temperature=-0.1)
        with pytest.raises(ValueError, match="temperature"):
            SamplingParams(temperature=2.1)

    def test_top_p_bounds(self):
        SamplingParams(top_p=1.0)
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=0.0)  # 0 会截掉所有候选，非法
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=1.01)

    def test_max_tokens_bounds(self):
        SamplingParams(max_tokens=1)
        with pytest.raises(ValueError, match="max_tokens"):
            SamplingParams(max_tokens=0)

    def test_frozen(self):
        params = SamplingParams()
        with pytest.raises(AttributeError):
            params.temperature = 1.5  # type: ignore[misc]


class TestSoftmaxWithTemperature:
    def test_temperature_one_is_plain_softmax(self):
        probs = softmax_with_temperature([1.0, 2.0, 3.0], temperature=1.0)
        assert sum(probs) == pytest.approx(1.0)
        assert probs[2] > probs[1] > probs[0]

    def test_low_temperature_sharpens(self):
        """T -> 0 时分布趋于 one-hot（贪心）."""
        probs = softmax_with_temperature([1.0, 2.0, 3.0], temperature=0.1)
        assert probs[2] > 0.99

    def test_high_temperature_flattens(self):
        """T 很大时分布趋于均匀."""
        probs = softmax_with_temperature([1.0, 2.0, 3.0], temperature=100.0)
        assert max(probs) - min(probs) < 0.01

    def test_monotonic_in_temperature(self):
        """同一 logits 下，温度越高最大概率越低（越平坦）."""
        logits = [0.0, 1.0, 4.0]
        low = softmax_with_temperature(logits, temperature=0.5)
        mid = softmax_with_temperature(logits, temperature=1.0)
        high = softmax_with_temperature(logits, temperature=2.0)
        assert low[2] > mid[2] > high[2]

    def test_numerically_stable_with_large_logits(self):
        probs = softmax_with_temperature([1000.0, 1001.0, 1002.0])
        assert sum(probs) == pytest.approx(1.0)

    def test_zero_temperature_rejected(self):
        """T=0 的贪心语义用 argmax 表达，不属于 softmax 函数."""
        with pytest.raises(ValueError, match="argmax"):
            softmax_with_temperature([1.0, 2.0], temperature=0.0)


class TestNucleusFilter:
    def test_top_p_one_is_identity(self):
        probs = [0.5, 0.3, 0.2]
        assert nucleus_filter(probs, top_p=1.0) == pytest.approx(probs)

    def test_truncates_tail(self):
        """top_p=0.5 时只留最大者并归一化为 1."""
        result = nucleus_filter([0.5, 0.3, 0.2], top_p=0.5)
        assert result == pytest.approx([1.0, 0.0, 0.0])

    def test_keeps_token_crossing_threshold(self):
        """使累积和越过 top_p 的那个 token 保留在集合内."""
        result = nucleus_filter([0.4, 0.35, 0.25], top_p=0.5)
        # 0.4 不够 0.5，加上 0.35 后 0.75 >= 0.5 截断
        assert result[0] > 0 and result[1] > 0 and result[2] == 0.0
        assert sum(result) == pytest.approx(1.0)

    def test_candidate_set_adapts_to_shape(self):
        """分布越平，核内候选越多——核采样相对 top-k 的自适应优势."""
        sharp = nucleus_filter([0.9, 0.05, 0.05], top_p=0.9)
        flat = nucleus_filter([0.34, 0.33, 0.33], top_p=0.9)
        assert sum(1 for p in sharp if p > 0) == 1
        assert sum(1 for p in flat if p > 0) == 3

    def test_invalid_top_p(self):
        with pytest.raises(ValueError, match="top_p"):
            nucleus_filter([0.5, 0.5], top_p=0.0)


class TestMockLLMSampling:
    def test_low_temperature_always_first_candidate(self):
        """低温度近似贪心：恒取第一个候选."""
        llm = MockLLM(candidates=["候选A", "候选B", "候选C"])
        replies = [
            llm.chat([Message(role="user", content="hi")], temperature=0.0)
            for _ in range(3)
        ]
        assert replies == ["候选A", "候选A", "候选A"]

    def test_high_temperature_rotates_candidates(self):
        """高温度模拟采样多样性：候选间确定性轮换."""
        llm = MockLLM(candidates=["候选A", "候选B", "候选C"])
        replies = [
            llm.chat([Message(role="user", content="hi")], temperature=1.5)
            for _ in range(4)
        ]
        assert replies == ["候选A", "候选B", "候选C", "候选A"]

    def test_responses_take_priority_over_candidates(self):
        """预设脚本优先，用尽后才进入采样模拟."""
        llm = MockLLM(responses=["脚本回复"], candidates=["候选A"])
        assert llm.chat([Message(role="user", content="hi")], temperature=1.5) == "脚本回复"
        assert llm.chat([Message(role="user", content="hi")], temperature=1.5) == "候选A"

    def test_no_candidates_falls_back_to_default(self):
        llm = MockLLM(default="兜底")
        assert llm.chat([Message(role="user", content="hi")], temperature=1.5) == "兜底"

    def test_usage_log_still_recorded(self):
        llm = MockLLM(candidates=["候选A"])
        llm.chat([Message(role="user", content="1234")], temperature=1.5)
        assert llm.usage_log[0]["prompt_tokens"] == 1
        assert llm.total_completion_tokens == 1
