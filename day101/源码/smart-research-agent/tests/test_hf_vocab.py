"""``hf_source.vocab``：词表与嵌入层的第一个分岔（day085 / M7-D9）."""

from __future__ import annotations

import math

import pytest

from smart_research_agent.hf_source import vocab
from smart_research_agent.hf_source.errors import ShapeError
from tests import hf_samples as samples


def tables(with_types: bool = True) -> vocab.EmbeddingTables:
    """三张写死的表（词表 ``i/10``、位置表 ``p``、类型表 ``0/1``）."""
    return vocab.EmbeddingTables(
        word=samples.word_table(),
        position=samples.position_table(),
        token_type=samples.type_table() if with_types else None,
    )


def test_gather_rows_picks_the_requested_rows() -> None:
    """按行号取行：第 ``i`` 行就是 ``(i+1)/10`` 的常数行."""
    rows = vocab.gather_rows(samples.word_table(), (0, 2))
    assert rows[0] == tuple([0.1] * samples.HIDDEN)
    assert rows[1] == tuple([0.3] * samples.HIDDEN)


@pytest.mark.parametrize("indices", [(-1,), (samples.VOCAB,), (0, 99)])
def test_gather_rows_rejects_out_of_range(indices: tuple[int, ...]) -> None:
    """越界当场拒绝（**回绕会让第 1025 个 token 拿到第 1 个位置的编码**）."""
    with pytest.raises(ShapeError):
        vocab.gather_rows(samples.word_table(), indices)


def test_gather_rows_rejects_empty_and_non_integer() -> None:
    """空行号与浮点行号都拒绝（形状合法但语义不明）."""
    with pytest.raises(ShapeError):
        vocab.gather_rows(samples.word_table(), ())
    with pytest.raises(ShapeError):
        vocab.gather_rows(samples.word_table(), (1.5,))  # type: ignore[arg-type]


def test_position_ids_and_default_token_types() -> None:
    """位置序列是 ``0..n-1``；BERT 的默认类型全是句子 A（0）."""
    assert vocab.position_ids(3) == (0, 1, 2)
    assert vocab.default_token_types(3) == (0, 0, 0)
    assert vocab.default_token_types(2) == (vocab.DEFAULT_TOKEN_TYPE,) * 2


@pytest.mark.parametrize("count", [0, -1])
def test_position_ids_rejects_non_positive(count: int) -> None:
    """长度必须为正：空序列在注意力里没有定义."""
    with pytest.raises(ShapeError):
        vocab.position_ids(count)
    with pytest.raises(ShapeError):
        vocab.default_token_types(count)


def test_gpt2_embed_is_a_plain_sum() -> None:
    """GPT-2 的嵌入是**两行之和**：``wte[2] + wpe[0]`` 与 ``wte[3] + wpe[1]``."""
    embedded = vocab.gpt2_embed(tables(), (2, 3))
    assert embedded[0] == tuple([0.3 + 0.0] * samples.HIDDEN)
    assert embedded[1] == tuple([0.4 + 1.0] * samples.HIDDEN)


def test_gpt2_embed_offset_shifts_positions() -> None:
    """``offset`` 决定第一个 token 拿第几个位置（增量解码时的起点）."""
    embedded = vocab.gpt2_embed(tables(), (2, 3), offset=2)
    assert embedded[0] == tuple([0.3 + 2.0] * samples.HIDDEN)
    assert embedded[1] == tuple([0.4 + 3.0] * samples.HIDDEN)


def test_bert_embed_normalises_every_row() -> None:
    """BERT 的嵌入**多一次 LayerNorm**：每一行的均值必须是 0（day079 的口径）."""
    embedded = vocab.bert_embed(tables(), (2, 3), (0, 1))
    for row in embedded:
        assert abs(math.fsum(row) / len(row)) < 1e-12


def test_bert_embed_without_type_table_degrades_gracefully() -> None:
    """没有类型表时按"整条序列都是句子 A"处理（**不是跳过这一项加法**）."""
    without = vocab.bert_embed(tables(False), (2, 3))
    with_type = vocab.bert_embed(tables(), (2, 3), (0, 0))
    assert without == with_type


def test_bert_embed_rejects_type_length_mismatch() -> None:
    """类型个数与 token 个数必须一致（广播出来的结果看起来正常）."""
    with pytest.raises(ShapeError):
        vocab.bert_embed(tables(), (2, 3), (0,))


def test_embedding_of_dispatches_by_profile() -> None:
    """按画像分派：GPT-2 不做 LN、BERT 做 LN——两者的第一行**不相等**."""
    gpt2 = vocab.embedding_of("gpt2", tables(), (2, 3))
    bert = vocab.embedding_of("bert", tables(), (2, 3))
    assert gpt2[0] != bert[0]


def test_embedding_of_rejects_token_types_for_gpt2() -> None:
    """GPT-2 没有"句子 A / 句子 B"这一维（静默丢掉会让两个句子看起来一样）."""
    with pytest.raises(ShapeError):
        vocab.embedding_of("gpt2", tables(), (2, 3), (0, 1))


def test_bert_embed_uses_the_bert_epsilon() -> None:
    """画像里的 eps 被真的传下去（1e-12 而不是 1e-5）."""
    with_profile = vocab.embedding_of("bert", tables(), (2, 3))
    explicit = vocab.bert_embed(tables(), (2, 3), epsilon=1e-12)
    assert with_profile == explicit


def test_embedding_tables_to_dict() -> None:
    """三张表的形状摊平成三个数."""
    flattened = tables().to_dict()
    assert flattened == {"vocab": samples.VOCAB, "positions": 8, "has_token_type": True}
    assert tables(False).to_dict()["has_token_type"] is False


def test_embedding_note_lists_the_differences() -> None:
    """画像对照文本里必须出现两个模型各自的关键值（这张文本是报告的一部分）."""
    note = vocab.embedding_note("gpt2")
    assert "pre" in note
    assert "1e-05" in note
    assert "bert" not in note.split("\n")[0]


def test_position_table_lengths_are_recorded() -> None:
    """两张位置表的长度作为常量记下来（越界演示要用到）."""
    assert vocab.GPT2_N_POSITIONS == 1024
    assert vocab.BERT_MAX_POSITION_EMBEDDINGS == 512
