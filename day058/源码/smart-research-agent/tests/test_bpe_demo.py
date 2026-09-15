"""bpe_demo 迷你 BPE 训练器的单元测试（day034）.

迷你 BPE 的全部价值在于"过程可断言"：每一步合并的符号对、顺序、
最终词表都是确定性的，这些测试就是 BPE 核心循环的正确性证明。
"""

from __future__ import annotations

from smart_research_agent.llm.bpe_demo import (
    END_OF_WORD,
    BPEMerge,
    count_pairs,
    encode,
    merge_pair,
    split_word,
    train_bpe,
)

#: 经典教学语料（与 Sennrich 等人 BPE 论文的演示语料一致）.
CORPUS = {"low": 5, "lower": 2, "newest": 6, "widest": 3}


class TestSplitWord:
    def test_chars_plus_end_marker(self):
        assert split_word("low") == ("l", "o", "w", END_OF_WORD)

    def test_single_char(self):
        assert split_word("a") == ("a", END_OF_WORD)


class TestCountPairs:
    def test_weighted_by_frequency(self):
        vocab = {("l", "o", "w", END_OF_WORD): 5, ("l", "o", "w", "e", "r", END_OF_WORD): 2}
        counts = count_pairs(vocab)
        # ("l", "o") 出现在两个词中：5 + 2 = 7
        assert counts[("l", "o")] == 7
        # ("w", "e") 只在第二个词中
        assert counts[("w", "e")] == 2

    def test_single_symbol_word_has_no_pairs(self):
        assert count_pairs({("a",): 3}) == {}


class TestMergePair:
    def test_merges_all_occurrences(self):
        vocab = {("l", "o", "w", END_OF_WORD): 5}
        merged = merge_pair(vocab, ("l", "o"))
        assert merged == {("lo", "w", END_OF_WORD): 5}

    def test_left_to_right_greedy(self):
        # ("a", "a") 在 "aaa" 中从左到右贪心合并：("aa", "a")，而非 ("a", "aa")
        vocab = {("a", "a", "a"): 1}
        merged = merge_pair(vocab, ("a", "a"))
        assert merged == {("aa", "a"): 1}

    def test_frequency_preserved(self):
        vocab = {("e", "s", "t", END_OF_WORD): 6}
        merged = merge_pair(vocab, ("e", "s"))
        assert merged[("es", "t", END_OF_WORD)] == 6


class TestTrainBPE:
    def test_first_merges_are_deterministic(self):
        merges, _ = train_bpe(CORPUS, 4)
        # 与论文演示完全一致的前四步：
        # ("e","s") 频率 9 最高 → ("es","t") → ("est","</w>") → ("l","o")
        assert merges[:4] == [
            BPEMerge(pair=("e", "s"), merged="es"),
            BPEMerge(pair=("es", "t"), merged="est"),
            BPEMerge(pair=("est", END_OF_WORD), merged="est" + END_OF_WORD),
            BPEMerge(pair=("l", "o"), merged="lo"),
        ]

    def test_vocab_shrinks_as_merges_accumulate(self):
        _, vocab = train_bpe(CORPUS, 10)
        # 10 轮后 "low" 与 "newest" 都已合并成单符号词
        assert ("low" + END_OF_WORD,) in vocab
        assert ("newest" + END_OF_WORD,) in vocab

    def test_zero_merges_returns_initial_vocab(self):
        merges, vocab = train_bpe(CORPUS, 0)
        assert merges == []
        assert vocab["l", "o", "w", END_OF_WORD] == 5

    def test_stops_early_when_no_pairs(self):
        # "ab" 两轮后合并成单符号 ("ab</w>",)，第三轮无对可合并，提前结束
        merges, _ = train_bpe({"ab": 1}, 10)
        assert len(merges) == 2

    def test_tie_breaking_is_lexicographic(self):
        # 两个对频率相同（各 2 次），选字典序小者 ("a","b") 而非 ("c","d")
        merges, _ = train_bpe({"ab": 2, "cd": 2}, 1)
        assert merges[0].pair == ("a", "b")


class TestEncode:
    def test_unseen_word_decomposes_into_known_subwords(self):
        """OOV 演示：语料里没有 "lowest"，但学过 "low" 和 "est</w>"."""
        merges, _ = train_bpe(CORPUS, 10)
        assert encode("lowest", merges) == ["low", "est" + END_OF_WORD]

    def test_seen_word_collapses_to_single_token(self):
        merges, _ = train_bpe(CORPUS, 10)
        assert encode("low", merges) == ["low" + END_OF_WORD]

    def test_empty_merges_returns_characters(self):
        symbols = encode("hi", [])
        assert symbols == ["h", "i", END_OF_WORD]
