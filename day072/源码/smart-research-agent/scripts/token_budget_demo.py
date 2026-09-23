"""day034 习题第 10 题参考解答：迷你 BPE 训练 + token 预算对比实验.

运行方式（项目根目录）::

    python scripts/token_budget_demo.py

全程离线：tiktoken 词表已在本地缓存；不可用时会自动走字符估算路径并打印说明。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smart_research_agent.llm.bpe_demo import encode, train_bpe
from smart_research_agent.llm.tokenizer import TokenCounter


def demo_bpe() -> None:
    """第一部分：在演示语料上训练迷你 BPE，验证合并过程与 OOV 行为."""
    corpus = {"deep": 4, "deeper": 3, "deepest": 5, "learning": 6}
    merges, vocab = train_bpe(corpus, 16)
    print("== 迷你 BPE 合并规则表（前 16 轮）==")
    for i, m in enumerate(merges, 1):
        print(f"  {i:>2}. {m.pair} -> {m.merged!r}")

    # (a) 第一条规则：("d","e") 与 ("e","e") 频率并列 12，字典序小者胜出
    assert merges[0].merged == "de", merges[0]
    # (b) 最高频整词 deepest（词频 5）进入词表，编码为单个符号
    assert encode("deepest", merges) == ["deepest</w>"]
    # (c) 未见词 "deeply" 被拆成已知片段 deep + 字符兜底 —— OOV 的现场演示
    assert encode("deeply", merges) == ["deep", "l", "y", "</w>"]
    print(f"最终词表: {vocab}")
    print(f"encode('deepest') = {encode('deepest', merges)}  (高频整词 -> 单 token)")
    print(f"encode('deeply')  = {encode('deeply', merges)}  (未见词 -> 已知片段 + 字符兜底)")


def demo_token_count() -> None:
    """第二部分：tiktoken 精确路径 vs 字符估算路径的对照计数."""
    samples = [
        "请用三句话总结 Transformer 的核心思想。",
        "帮我比较一下 BPE 和 WordPiece 两种分词算法的差异。",
        "Summarize the core idea of the Transformer architecture in three sentences.",
        "Explain the difference between BPE and WordPiece tokenization.",
    ]
    precise = TokenCounter()
    fallback = TokenCounter(prefer_tiktoken=False)
    print(f"\n== token 计数对照（精确路径 backend={precise.backend}）==")
    for text in samples:
        n_precise = precise.count_tokens(text, "gpt-4o")
        n_fallback = fallback.count_tokens(text)
        print(f"  精确 {n_precise:>3} | 估算 {n_fallback:>3} | {text[:24]}…")
        assert n_precise > 0 and n_fallback > 0


def demo_cost() -> None:
    """第三部分：同一条中文提示词按两档模型预估费用."""
    prompt = "请详细分析 RAG 与微调各自的适用场景，并给出选型建议。"
    counter = TokenCounter()
    mini = counter.estimate_cost(prompt, "gpt-4o-mini", expected_completion_tokens=200)
    full = counter.estimate_cost(prompt, "gpt-4o", expected_completion_tokens=200)
    print("\n== 成本预估（同一提示词，预计输出 200 token）==")
    print(f"  gpt-4o-mini: ${mini['total_cost_usd']:.6f}  (输入 {mini['prompt_tokens']} token)")
    print(f"  gpt-4o     : ${full['total_cost_usd']:.6f}  (输入 {full['prompt_tokens']} token)")
    assert full["total_cost_usd"] > mini["total_cost_usd"]


def main() -> int:
    print(f"本机计数路径: {TokenCounter().backend}")
    demo_bpe()
    demo_token_count()
    demo_cost()
    print("\n全部断言通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
