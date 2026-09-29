"""tokenizer 模块的单元测试（day034）.

覆盖两条计数路径：tiktoken 精确计数（本机词表已缓存，离线可用）
与字符级估算兜底，以及成本预估与 day032 价格表的衔接。
"""

from __future__ import annotations

import pytest

from smart_research_agent.llm.tokenizer import (
    TokenCounter,
    estimate_tokens_chars,
)
from smart_research_agent.observability.cost_tracker import DEFAULT_PRICE_TABLE


class TestCharEstimation:
    """字符级兜底估算（不依赖 tiktoken，任何环境确定性成立）."""

    def test_empty_text_is_zero(self):
        assert estimate_tokens_chars("") == 0

    def test_cjk_chars_counted_one_per_char(self):
        assert estimate_tokens_chars("你好世界") == 4

    def test_ascii_roughly_four_chars_per_token(self):
        assert estimate_tokens_chars("abcdefghijklmnop") == 4  # 16 / 4

    def test_mixed_cjk_and_ascii(self):
        # 4 个汉字 = 4 token，"abcd" = 1 token
        assert estimate_tokens_chars("你好世界abcd") == 5

    def test_nonzero_for_short_text(self):
        assert estimate_tokens_chars("a") == 1


@pytest.mark.skipif(
    TokenCounter().backend != "tiktoken",
    reason="本机 tiktoken 词表缓存不可用，跳过精确计数断言",
)
class TestTiktokenBackend:
    """tiktoken 路径：encoding 名映射与精确计数."""

    def test_backend_is_tiktoken_by_default(self):
        assert TokenCounter().backend == "tiktoken"

    def test_english_count_matches_cl100k(self):
        # cl100k_base 下 "hello world" 恰为 2 个 token
        assert TokenCounter().count_tokens("hello world", "gpt-3.5-turbo") == 2

    def test_chinese_more_tokens_per_char_than_english(self):
        counter = TokenCounter()
        zh = counter.count_tokens("你好世界你好世界你好世界你好世界", "gpt-3.5-turbo")
        en = counter.count_tokens("abcdefghijklmnop", "gpt-3.5-turbo")
        # 16 个汉字的 token 数显著多于 16 个 ASCII 字符（中英文 token 效率差异）
        assert zh > en

    def test_model_name_selects_encoding(self):
        counter = TokenCounter()
        assert counter.encoding_name_for("gpt-4o") == "o200k_base"
        assert counter.encoding_name_for("gpt-4") == "cl100k_base"
        # 未知模型回退默认 encoding，而不是报错
        assert counter.encoding_name_for("some-future-model") == "cl100k_base"


class TestFallbackBackend:
    """强制关闭 tiktoken 后，计数走字符估算路径."""

    def test_backend_switches_to_fallback(self):
        assert TokenCounter(prefer_tiktoken=False).backend == "fallback"

    def test_count_matches_char_estimation(self):
        counter = TokenCounter(prefer_tiktoken=False)
        assert counter.count_tokens("你好世界") == estimate_tokens_chars("你好世界")

    def test_fallback_never_raises_on_unknown_model(self):
        counter = TokenCounter(prefer_tiktoken=False)
        assert counter.count_tokens("hi", "no-such-model") >= 1


class TestCountMessages:
    def test_includes_per_message_overhead(self):
        counter = TokenCounter(prefer_tiktoken=False)
        messages = [{"role": "user", "content": "你好世界"}]
        # 内容 4 token + 每条消息 4 token 元数据开销
        assert counter.count_messages(messages) == 8

    def test_empty_message_list(self):
        assert TokenCounter().count_messages([]) == 0


class TestEstimateCost:
    def test_cost_follows_price_table(self):
        counter = TokenCounter(prefer_tiktoken=False)  # 用估算路径保证数字确定
        result = counter.estimate_cost("a" * 4000, "gpt-4o-mini", expected_completion_tokens=500)
        assert result["prompt_tokens"] == 1000
        price = DEFAULT_PRICE_TABLE["gpt-4o-mini"]
        expected_input = 1000 * price["input"] / 1000.0
        expected_output = 500 * price["output"] / 1000.0
        assert result["input_cost_usd"] == pytest.approx(expected_input)
        assert result["output_cost_usd"] == pytest.approx(expected_output)
        assert result["total_cost_usd"] == pytest.approx(expected_input + expected_output)

    def test_unknown_model_raises_key_error(self):
        # 与 day032 CostTracker.record 同款策略：宁可报错不可漏算
        with pytest.raises(KeyError):
            TokenCounter().estimate_cost("hi", "no-such-model")

    def test_custom_price_table_accepted(self):
        counter = TokenCounter(prefer_tiktoken=False)
        table = {"my-model": {"input": 0.001, "output": 0.002}}
        result = counter.estimate_cost("a" * 4000, "my-model", table, 1000)
        # 输入 1000 token * 0.001/1K + 输出 1000 token * 0.002/1K = 0.003
        assert result["total_cost_usd"] == pytest.approx(0.003)
