"""迷你 BPE 训练器：几十行代码讲透字节对编码的核心循环（day034 教学教具）.

工业级 BPE（tiktoken、SentencePiece 的 BPE 模式）在字节级别、用正则预切分、
有上千万词的语料；但核心循环只有三步，几十行代码就能完整呈现：

1. **统计**：数一遍当前所有相邻符号对的出现频率；
2. **选择**：挑频率最高的那对符号；
3. **合并**：把这对符号在所有出现处替换成一个新符号，并记下这条合并规则。

重复 num_merges 轮，得到的合并规则表（merges）就是"学到的词表"。
编码新文本时，按合并规则的产生顺序依次套用即可——规则顺序即优先级。

为了直观，本模块以字符为初始符号、以 ``</w>`` 标记词尾（经典 GPT-2 之前的
教学口径），语料用 ``{词: 词频}`` 的字典表示。全部过程确定性：
频率相同时按符号对的字典序选最小者，测试可以精确断言每一步。
"""

from __future__ import annotations

from dataclasses import dataclass

#: 词尾标记：让 "est" 与词尾的 "est</w>" 成为不同子词，保留词边界信息。
END_OF_WORD = "</w>"


def split_word(word: str) -> tuple[str, ...]:
    """把词切为初始符号序列：每个字符一个符号，末尾追加词尾标记."""
    return tuple(word) + (END_OF_WORD,)


def count_pairs(vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
    """统计词表中所有相邻符号对的总频率（按词频加权）."""
    counts: dict[tuple[str, str], int] = {}
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            counts[pair] = counts.get(pair, 0) + freq
    return counts


def merge_pair(
    vocab: dict[tuple[str, ...], int], pair: tuple[str, str]
) -> dict[tuple[str, ...], int]:
    """把词表中所有与 pair 相邻相等的符号序列合并为一个新符号.

    新符号是 pair 两元素的直接拼接（如 ("l", "o") -> "lo"）。
    从左到右贪心扫描，已合并的符号不再参与本轮后续匹配。
    """
    merged_symbol = "".join(pair)
    new_vocab: dict[tuple[str, ...], int] = {}
    for symbols, freq in vocab.items():
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(merged_symbol)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab


@dataclass(frozen=True)
class BPEMerge:
    """一条合并规则：pair 两符号按序出现时替换为 merged."""

    pair: tuple[str, str]
    merged: str


def train_bpe(corpus: dict[str, int], num_merges: int) -> tuple[list[BPEMerge], dict[tuple[str, ...], int]]:
    """在 ``{词: 词频}`` 语料上训练 BPE，返回 (合并规则表, 最终词表).

    每轮：统计相邻对频率 → 选频率最高者（并列时取字典序最小，保证确定性）
    → 全局合并并记录规则。语料中没有任何可合并对时提前结束。
    """
    vocab = {split_word(word): freq for word, freq in corpus.items()}
    merges: list[BPEMerge] = []
    for _ in range(num_merges):
        counts = count_pairs(vocab)
        if not counts:
            break
        # 频率最高者优先；并列时取符号对字典序最小者，
        # 使结果与字典遍历顺序无关（确定性，测试可精确断言）。
        best = sorted(counts, key=lambda p: (-counts[p], p))[0]
        vocab = merge_pair(vocab, best)
        merges.append(BPEMerge(pair=best, merged="".join(best)))
    return merges, vocab


def encode(word: str, merges: list[BPEMerge]) -> list[str]:
    """用训练得到的合并规则表编码一个新词（规则顺序即套用优先级）.

    未知词也能编——这就是 subword 分词解决 OOV 的方式：哪怕从没见过
    "lower"，只要见过 "low" 和 "er"</w>"，就能拼出合理的切分。
    """
    symbols = list(split_word(word))
    for merge in merges:
        merged_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == merge.pair:
                merged_symbols.append(merge.merged)
                i += 2
            else:
                merged_symbols.append(symbols[i])
                i += 1
        symbols = merged_symbols
    return symbols
