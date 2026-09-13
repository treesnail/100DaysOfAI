"""SFT 编码测试（day050）：token 化、label mask、截断、批次与长度分位.

本文件是全课最"较真"的测试：**每一个 token 都要能数出来、指出来**。
所以断言里大量出现精确的数字（17 / 2 / 8 / 10），而不是"大于 0"这类
宽松条件——mask 写错时，宽松条件几乎一定能通过。
"""

from __future__ import annotations

import pytest

from smart_research_agent.finetune.schema import TrainingExample
from smart_research_agent.sft.encoding import (
    IGNORE_INDEX,
    PAD_TOKEN,
    PAD_TOKEN_ID,
    SPECIAL_TOKENS,
    TRUNCATION_HEAD,
    TRUNCATION_KEEP_ANSWER,
    UNK_TOKEN,
    UNK_TOKEN_ID,
    Batch,
    CharTokenizer,
    EncodedSample,
    LengthSummary,
    SFTDataError,
    collate,
    encode_supervised,
    iter_batches,
    length_summary,
    suggest_max_length,
)
from smart_research_agent.sft.template import PLAIN, RenderedSample, render_supervised


def tiny_rendered():
    """一个字符数可手算的最小渲染样本（PLAIN 模板、无 system）.

    前缀 ``"### 指令\\n问\\n\\n### 回答\\n"`` 共 17 字符，
    监督区间 ``"答\\n"`` 共 2 字符（答案 + 换行终止符），合计 19。
    """
    return render_supervised(
        TrainingExample(instruction="问", output="答"), template=PLAIN, system_prompt=None
    )


@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer.from_texts([tiny_rendered().text])


class TestConstants:
    """常量必须稳定：它们是与 PyTorch / HF 生态对齐的接口."""

    def test_ignore_index_is_minus_100(self):
        assert IGNORE_INDEX == -100

    def test_pad_token_id_is_zero(self):
        assert PAD_TOKEN_ID == 0

    def test_special_tokens_order(self):
        assert SPECIAL_TOKENS == (PAD_TOKEN, UNK_TOKEN)
        assert UNK_TOKEN_ID == 1

    def test_truncation_names(self):
        assert (TRUNCATION_KEEP_ANSWER, TRUNCATION_HEAD) == ("keep-answer", "head")


class TestCharTokenizer:
    """字符级分词器：离线、确定、可逆."""

    def test_specials_are_prefixed_and_indexable(self, tokenizer):
        assert tokenizer.tokens[:2] == SPECIAL_TOKENS
        assert tokenizer.tokens[PAD_TOKEN_ID] == PAD_TOKEN
        assert tokenizer.tokens[UNK_TOKEN_ID] == UNK_TOKEN
        assert tokenizer.pad_token_id == PAD_TOKEN_ID
        assert tokenizer.ignore_index == IGNORE_INDEX

    def test_vocab_is_specials_plus_chars(self, tokenizer):
        # 19 个字符里有重复（"#"、" "、"答" 之外的换行等），去重后 + 2 个特殊 token
        unique_chars = len(set(tiny_rendered().text))
        assert tokenizer.vocab_size == unique_chars + len(SPECIAL_TOKENS)

    def test_encode_decode_roundtrip(self, tokenizer):
        text = tiny_rendered().text
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_unknown_char_maps_to_unk(self, tokenizer):
        unknown = "§"
        assert unknown not in tokenizer.tokens[2:]
        assert tokenizer.encode(unknown) == [UNK_TOKEN_ID]

    def test_decode_skips_specials_and_out_of_range(self, tokenizer):
        assert tokenizer.decode([PAD_TOKEN_ID, UNK_TOKEN_ID]) == ""
        assert tokenizer.decode([0, 999999]) == ""

    def test_from_texts_dedupes_and_keeps_order(self):
        tokenizer = CharTokenizer.from_texts(["ba", "ab"])
        assert tokenizer.tokens[2:] == ("b", "a")

    def test_dict_roundtrip(self, tokenizer):
        restored = CharTokenizer.from_dict(tokenizer.to_dict())
        assert restored.tokens == tokenizer.tokens
        assert restored.vocab_size == tokenizer.vocab_size

    def test_dict_roundtrip_without_specials_prefix(self):
        """``to_dict`` 的产物若被剥掉了特殊 token 前缀，也要能正确还原."""
        restored = CharTokenizer.from_dict({"tokens": ["a", "b"]})
        assert restored.tokens == (*SPECIAL_TOKENS, "a", "b")

    def test_len(self, tokenizer):
        assert len(tokenizer) == tokenizer.vocab_size

    def test_empty_payload(self):
        restored = CharTokenizer.from_dict({})
        assert restored.tokens == SPECIAL_TOKENS


class TestEncodedSampleInvariants:
    """构造期不变式：送错数据必须在**构造处**失败."""

    def test_lengths_must_match(self):
        with pytest.raises(SFTDataError, match="三个序列必须等长"):
            EncodedSample(
                input_ids=(1, 2),
                labels=(1,),
                attention_mask=(1, 1),
                prompt_tokens=1,
                supervised_tokens=1,
                masked_prompt=True,
                truncated=False,
            )

    def test_zero_supervised_rejected(self):
        with pytest.raises(SFTDataError, match="监督 token 数为 0"):
            EncodedSample(
                input_ids=(1, 2),
                labels=(IGNORE_INDEX, IGNORE_INDEX),
                attention_mask=(1, 1),
                prompt_tokens=2,
                supervised_tokens=0,
                masked_prompt=True,
                truncated=False,
            )

    def test_zero_prompt_rejected(self):
        with pytest.raises(SFTDataError, match="prompt token 数为 0"):
            EncodedSample(
                input_ids=(1,),
                labels=(1,),
                attention_mask=(1,),
                prompt_tokens=0,
                supervised_tokens=1,
                masked_prompt=True,
                truncated=False,
            )

    def test_masked_supervised_count_must_match_span(self):
        """屏蔽 prompt 时，监督 token 数必须等于监督区间长度（不一致即为记账错误）."""
        with pytest.raises(SFTDataError, match="监督 token 数必须等于监督区间的长度"):
            EncodedSample(
                input_ids=(1, 2, 3),
                labels=(IGNORE_INDEX, 2, 3),
                attention_mask=(1, 1, 1),
                prompt_tokens=1,
                supervised_tokens=1,
                masked_prompt=True,
                truncated=False,
            )

    def test_unmasked_prompt_allows_full_span(self):
        sample = EncodedSample(
            input_ids=(1, 2, 3),
            labels=(1, 2, 3),
            attention_mask=(1, 1, 1),
            prompt_tokens=1,
            supervised_tokens=3,
            masked_prompt=False,
            truncated=False,
        )
        assert sample.masked_tokens == 0
        assert sample.supervised_ratio == 1.0

    def test_derived_properties(self):
        sample = EncodedSample(
            input_ids=(1, 2, 3, 4),
            labels=(IGNORE_INDEX, IGNORE_INDEX, 3, 4),
            attention_mask=(1, 1, 1, 1),
            prompt_tokens=2,
            supervised_tokens=2,
            masked_prompt=True,
            truncated=False,
        )
        assert sample.total_tokens == 4
        assert sample.masked_tokens == 2
        assert sample.supervised_ratio == 0.5

    def test_supervised_ratio_of_empty_is_zero(self):  # pragma: no cover - 防御式
        sample = EncodedSample(
            input_ids=(1,),
            labels=(1,),
            attention_mask=(1,),
            prompt_tokens=1,
            supervised_tokens=1,
            masked_prompt=False,
            truncated=False,
        )
        assert sample.supervised_ratio == 1.0


class TestEncodeSupervised:
    """``encode_supervised``：本课最核心的函数."""

    def test_exact_token_split(self, tokenizer):
        sample = encode_supervised(tiny_rendered(), tokenizer, max_length=64)
        assert sample.total_tokens == 19
        assert sample.prompt_tokens == 17
        assert sample.supervised_tokens == 2
        assert sample.truncated is False
        assert sample.masked_prompt is True

    def test_prompt_labels_are_all_ignore_index(self, tokenizer):
        sample = encode_supervised(tiny_rendered(), tokenizer, max_length=64)
        assert set(sample.labels[:17]) == {IGNORE_INDEX}
        assert all(value != IGNORE_INDEX for value in sample.labels[17:])

    def test_supervised_labels_equal_input_ids(self, tokenizer):
        """监督区间的 label 必须等于对应位置的 input_id（下一 token 预测）."""
        sample = encode_supervised(tiny_rendered(), tokenizer, max_length=64)
        assert sample.labels[17:] == sample.input_ids[17:]

    def test_attention_mask_all_ones(self, tokenizer):
        sample = encode_supervised(tiny_rendered(), tokenizer, max_length=64)
        assert sample.attention_mask == (1,) * 19

    def test_max_length_too_small_rejected(self, tokenizer):
        with pytest.raises(SFTDataError, match="max_length 至少为"):
            encode_supervised(tiny_rendered(), tokenizer, max_length=1)

    def test_unknown_truncation_rejected(self, tokenizer):
        with pytest.raises(SFTDataError, match="不支持的截断策略"):
            encode_supervised(tiny_rendered(), tokenizer, max_length=64, truncation="middle")

    def test_keep_answer_truncates_prompt_from_left(self, tokenizer):
        """keep-answer 从**左侧**裁 prompt：靠近答案的指令保留下来."""
        rendered = tiny_rendered()
        sample = encode_supervised(rendered, tokenizer, max_length=10)
        assert sample.total_tokens == 10
        assert sample.prompt_tokens == 8
        assert sample.supervised_tokens == 2
        assert sample.truncated is True
        # 保留的 prompt 是原文的**后** 8 个字符
        kept_prompt_text = tokenizer.decode(sample.input_ids[:8])
        assert kept_prompt_text == rendered.prompt_text[-8:]

    def test_keep_answer_requires_room_for_answer(self, tokenizer):
        with pytest.raises(SFTDataError, match="已超过"):
            encode_supervised(tiny_rendered(), tokenizer, max_length=2)

    def test_keep_answer_no_truncation_when_it_fits_exactly(self, tokenizer):
        sample = encode_supervised(tiny_rendered(), tokenizer, max_length=19)
        assert sample.truncated is False
        assert sample.total_tokens == 19

    def test_head_truncation_keeps_first_tokens(self, tokenizer):
        rendered = tiny_rendered()
        sample = encode_supervised(
            rendered, tokenizer, max_length=18, truncation=TRUNCATION_HEAD
        )
        assert sample.total_tokens == 18
        assert sample.prompt_tokens == 17
        assert sample.supervised_tokens == 1
        assert sample.truncated is True

    def test_head_truncation_rejects_zero_supervision(self, tokenizer):
        with pytest.raises(SFTDataError, match="head 截断后没有任何监督 token"):
            encode_supervised(
                tiny_rendered(), tokenizer, max_length=10, truncation=TRUNCATION_HEAD
            )

    def test_unmasked_prompt_turns_everything_into_supervision(self, tokenizer):
        """``mask_prompt=False`` 只用于对照实验：监督范围扩展到整条序列."""
        sample = encode_supervised(
            tiny_rendered(), tokenizer, max_length=64, mask_prompt=False
        )
        assert sample.masked_prompt is False
        assert sample.supervised_tokens == sample.total_tokens == 19
        assert sample.masked_tokens == 0
        assert sample.prompt_tokens == 17  # 前缀长度与是否屏蔽无关

    def test_zero_length_prompt_rejected(self):
        """``prompt_chars=0`` 的渲染结果（防御式路径）：必须显式报错.

        正常渲染绝不会产出零长前缀（ChatML/Llama3/Plain 都带角色标记），
        但 ``RenderedSample`` 允许这种构造；说清"为什么不能编码"比抛一个
        下标越界友好得多。
        """
        rendered = RenderedSample(
            text="abc", prompt_chars=0, supervised_chars=3, template=PLAIN
        )
        tokenizer = CharTokenizer.from_texts(["abc"])
        with pytest.raises(SFTDataError, match="前缀编码后为空"):
            encode_supervised(rendered, tokenizer, max_length=8)

    def test_empty_prompt_after_encoding_rejected(self):
        """词表里没有的字符会变成 <unk>，但仍占位；这里构造真正空的前缀."""
        tokenizer = CharTokenizer.from_texts(["答"])
        rendered = render_supervised(
            TrainingExample(instruction="问", output="答"), template=PLAIN, system_prompt=None
        )
        # 用只含答案字符的词表不影响前缀长度（<unk> 仍占一个 token）
        sample = encode_supervised(rendered, tokenizer, max_length=64)
        assert sample.prompt_tokens == 17


class TestCollate:
    """批次组装：padding 的三件套必须一起补."""

    def _samples(self, tokenizer):
        rendered_a = tiny_rendered()
        rendered_b = render_supervised(
            TrainingExample(instruction="问两个问题", output="答得更长一些"),
            template=PLAIN,
            system_prompt=None,
        )
        return [
            encode_supervised(rendered_a, tokenizer, max_length=64),
            encode_supervised(rendered_b, tokenizer, max_length=64),
        ]

    def test_padding_trio(self, tokenizer):
        samples = self._samples(tokenizer)
        batch = collate(samples)
        assert batch.batch_size == 2
        assert batch.max_length == max(s.total_tokens for s in samples)
        # 短的那条的 padding 位置三件套
        padding = batch.max_length - samples[0].total_tokens
        assert batch.input_ids[0][-padding:] == (PAD_TOKEN_ID,) * padding
        assert batch.labels[0][-padding:] == (IGNORE_INDEX,) * padding
        assert batch.attention_mask[0][-padding:] == (0,) * padding

    def test_lengths_and_padding_tokens(self, tokenizer):
        samples = self._samples(tokenizer)
        batch = collate(samples)
        assert batch.lengths == tuple(s.total_tokens for s in samples)
        assert batch.padding_tokens == batch.batch_size * batch.max_length - sum(batch.lengths)

    def test_supervised_tokens_counts_only_unmasked(self, tokenizer):
        samples = self._samples(tokenizer)
        batch = collate(samples)
        assert batch.supervised_tokens == sum(s.supervised_tokens for s in samples)

    def test_pad_to_fixed_length(self, tokenizer):
        batch = collate(self._samples(tokenizer), pad_to=64)
        assert batch.max_length == 64
        assert all(len(row) == 64 for row in batch.input_ids)

    def test_pad_to_smaller_than_longest_rejected(self, tokenizer):
        with pytest.raises(SFTDataError, match="小于批内最长样本"):
            collate(self._samples(tokenizer), pad_to=1)

    def test_empty_batch_rejected(self):
        with pytest.raises(SFTDataError, match="空批次"):
            collate([])

    def test_empty_batch_properties(self):
        """空 Batch 的属性不应崩溃（防御式，供日志层安全调用）."""
        batch = Batch(input_ids=(), labels=(), attention_mask=(), pad_token_id=PAD_TOKEN_ID)
        assert batch.batch_size == 0
        assert batch.max_length == 0
        assert batch.lengths == ()
        assert batch.supervised_tokens == 0
        assert batch.padding_tokens == 0


class TestIterBatches:
    """批次切分：drop_last 的两种行为."""

    def _samples(self, tokenizer, count: int):
        return [
            encode_supervised(
                render_supervised(
                    TrainingExample(instruction=f"问题{i}", output=f"答案{i}"),
                    template=PLAIN,
                    system_prompt=None,
                ),
                tokenizer,
                max_length=64,
            )
            for i in range(count)
        ]

    def test_default_keeps_tail(self, tokenizer):
        batches = iter_batches(self._samples(tokenizer, 5), batch_size=2)
        assert [b.batch_size for b in batches] == [2, 2, 1]

    def test_drop_last_removes_tail(self, tokenizer):
        batches = iter_batches(self._samples(tokenizer, 5), batch_size=2, drop_last=True)
        assert [b.batch_size for b in batches] == [2, 2]

    def test_batch_size_must_be_positive(self, tokenizer):
        with pytest.raises(SFTDataError, match="batch_size 必须为正整数"):
            iter_batches(self._samples(tokenizer, 2), batch_size=0)

    def test_empty_input(self):
        assert iter_batches([], batch_size=2) == []


class TestLengthSummary:
    """长度分位——``max_length`` 的唯一依据."""

    def test_summary_values_on_small_set(self, tokenizer):
        rendered = [
            render_supervised(
                TrainingExample(instruction="问" * i, output="答"),
                template=PLAIN,
                system_prompt=None,
            )
            for i in range(1, 6)
        ]
        summary = length_summary(rendered, tokenizer)
        assert summary.count == 5
        assert summary.minimum <= summary.p50 <= summary.p90 <= summary.p95 <= summary.maximum
        assert summary.quantile(0.0) == summary.minimum
        assert summary.quantile(1.0) == summary.maximum
        assert summary.mean > 0

    def test_quantile_out_of_range_rejected(self, tokenizer):
        summary = length_summary([tiny_rendered()], tokenizer)
        with pytest.raises(SFTDataError, match="分位比例必须落在"):
            summary.quantile(1.5)

    def test_quantile_without_lengths_rejected(self):
        summary = LengthSummary(
            count=1,
            minimum=1,
            maximum=1,
            mean=1.0,
            p50=1,
            p90=1,
            p95=1,
            prompt_mean=1.0,
            supervised_mean=1.0,
        )
        with pytest.raises(SFTDataError, match="缺少 lengths"):
            summary.quantile(0.5)

    def test_summary_line_and_dict(self, tokenizer):
        summary = length_summary([tiny_rendered()], tokenizer)
        assert "1 条" in summary.summary_line()
        payload = summary.to_dict()
        assert payload["count"] == 1
        assert "lengths" not in payload
        assert payload["min"] == payload["max"] == 19

    def test_empty_input_rejected(self, tokenizer):
        with pytest.raises(SFTDataError, match="长度分布需要至少一条渲染样本"):
            length_summary([], tokenizer)


class TestSuggestMaxLength:
    """``suggest_max_length``：按分位取整到 granularity 的倍数."""

    def _rendered(self):
        return [
            render_supervised(
                TrainingExample(instruction="问" * i, output="答"),
                template=PLAIN,
                system_prompt=None,
            )
            for i in range(1, 11)
        ]

    def test_rounds_up_to_granularity(self, tokenizer):
        rendered = self._rendered()
        needed = length_summary(rendered, tokenizer).quantile(0.95)
        suggested = suggest_max_length(
            rendered, tokenizer, quantile=0.95, granularity=32, minimum=8
        )
        assert suggested % 32 == 0
        assert suggested >= needed
        assert suggested - 32 < needed

    def test_quantile_one_covers_maximum(self, tokenizer):
        rendered = self._rendered()
        maximum = length_summary(rendered, tokenizer).maximum
        assert suggest_max_length(rendered, tokenizer, quantile=1.0) >= maximum

    def test_minimum_floor_applies(self, tokenizer):
        rendered = [tiny_rendered()]
        assert suggest_max_length(rendered, tokenizer, quantile=0.5, minimum=64) == 64

    def test_granularity_must_be_positive(self, tokenizer):
        with pytest.raises(SFTDataError, match="granularity 必须为正整数"):
            suggest_max_length([tiny_rendered()], tokenizer, granularity=0)

    def test_custom_granularity(self, tokenizer):
        rendered = self._rendered()
        suggested = suggest_max_length(rendered, tokenizer, quantile=1.0, granularity=8)
        assert suggested % 8 == 0
