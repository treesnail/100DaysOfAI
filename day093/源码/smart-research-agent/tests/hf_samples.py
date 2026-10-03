"""``hf_source`` 的共享样本：形状、参数与比较工具（day085 / M7-D9）.

三条纪律（与前十天逐字相同）：

```text
① 参数**委托生产函数**造（day079 的 make_block_parameters、day075 的 default_parameters），
   因此"同样的一次前向"是一条可以被复核的结论，而不是"上次跑出来的值"。
② 输入**写死**且每一行有非零方差（全零行会让 LN 与 softmax 同时退化，
   而那种"看起来正常"的读数是假象）。
③ 比较分两种：``approx`` 用于"两份独立实现之间的读数"，
   逐位比较直接写 ``==``——**不把它们混成一个函数**。
"""

from __future__ import annotations

from smart_research_agent.encoder_decoder.depth import make_block_parameters
from smart_research_agent.encoder_decoder.types import BlockParameters, BlockShape
from smart_research_agent.hf_source.blocks import source_shape_of
from smart_research_agent.hf_source.types import SourceShape
from smart_research_agent.math_foundations.types import Matrix
from smart_research_agent.transformer_core.train import default_parameters
from smart_research_agent.transformer_core.types import AttentionParams

#: 隐藏维 ``d``（与 day075 ~ day083 的小样本一致）.
HIDDEN = 6

#: 前馈的中间维 ``d_ff = 4d``.
FFN = 24

#: 一个序列的 token 数。
TOKENS = 4

#: 词表大小（生成实验用）。
VOCAB = 8

#: 参数种子（所有生产函数共用它）。
SEED = 7

#: 与 day079 / day076 对账时用的容差（两份独立实现之间）。
TOLERANCE = 1e-12


def block_shape(tokens: int = TOKENS, hidden: int = HIDDEN) -> BlockShape:
    """写死的块形状：``hidden=6、ffn=24、tokens=4``."""
    return BlockShape(hidden=hidden, ffn=4 * hidden, tokens=tokens)


def block_parameters(tokens: int = TOKENS, hidden: int = HIDDEN) -> BlockParameters:
    """一个块的初始参数（**委托 day079 的生产函数**，种子与幅度钉住）."""
    return make_block_parameters(block_shape(tokens, hidden), seed=SEED)


def attention_parameters(size: int = HIDDEN) -> AttentionParams:
    """确定性的小随机注意力参数（**委托 day075 的生产函数**）."""
    return default_parameters(size, seed=SEED)


def inputs(tokens: int = TOKENS, hidden: int = HIDDEN) -> Matrix:
    """写死的斜坡输入：``x[i][j] = 0.2·(i+1) + 0.1·(j+1)``（**每行非零方差**）."""
    return tuple(
        tuple(0.2 * (row + 1) + 0.1 * (column + 1) for column in range(hidden))
        for row in range(tokens)
    )


def shape_of(name: str = "gpt2", *, heads: int = 1, tokens: int = TOKENS) -> SourceShape:
    """一个模型的形状（**画像驱动**：因果默认值随模型走）."""
    return source_shape_of(name, hidden=HIDDEN, tokens=tokens, heads=heads, vocab=VOCAB)


def word_table(vocab: int = VOCAB, hidden: int = HIDDEN) -> Matrix:
    """一张写死的词表（第 ``i`` 行是 ``i/10`` 的常数行）——**可手算**."""
    return tuple(tuple((index + 1) / 10.0 for _ in range(hidden)) for index in range(vocab))


def position_table(positions: int = 8, hidden: int = HIDDEN) -> Matrix:
    """一张写死的位置表（第 ``p`` 行是 ``p`` 的常数行）——**可手算**."""
    return tuple(tuple(float(row) for _ in range(hidden)) for row in range(positions))


def type_table(types: int = 2, hidden: int = HIDDEN) -> Matrix:
    """一张写死的 token_type 表：句子 A 是**全 0 行**、句子 B 是**全 1 行**.

    第 0 行取全 0 是刻意的：于是"没有类型表"与"类型表取第 0 行"**逐位相同**，
    而这一条正是 HF 在 ``token_type_ids=None`` 时的行为（整条序列都是句子 A）。
    """
    return tuple(
        tuple(float(index) for _ in range(hidden)) for index in range(types)
    )


def approx(left: float, right: float, *, tolerance: float = TOLERANCE) -> bool:
    """两份实现之间的比较（相对 + 绝对混合口径）."""
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def approx_matrix(
    left: Matrix,
    right: Matrix,
    *,
    tolerance: float = TOLERANCE,
) -> bool:
    """逐元素比较两个矩阵（形状不同直接 ``False``）."""
    if len(left) != len(right) or len(left[0]) != len(right[0]):
        return False
    return all(
        approx(a, b, tolerance=tolerance)
        for row_a, row_b in zip(left, right)
        for a, b in zip(row_a, row_b)
    )


def row_norm(matrix: Matrix) -> float:
    """整个矩阵的 Frobenius 范数（读数的辅助量）."""
    return sum(value * value for row in matrix for value in row) ** 0.5
