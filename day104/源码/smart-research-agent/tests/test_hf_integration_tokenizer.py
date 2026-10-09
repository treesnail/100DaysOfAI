"""``hf_integration.tokenizer``：两个文件加一段算术（day086 / M7-D10）."""

from __future__ import annotations

import pytest

from smart_research_agent.hf_integration import errors
from smart_research_agent.hf_integration.tokenizer import (
    PADDING_LONGEST,
    PADDING_MAX_LENGTH,
    ByteBPETokenizer,
    build_tiny_tokenizer,
    bytes_to_unicode,
    get_pairs,
    merge_ranks,
    parse_merges,
    pretokenize,
    split_on_specials,
)

from tests.hf_integration_samples import build_case, merges_text


# --------------------------------------------------------------------------- 字节映射


def test_bytes_to_unicode_is_a_bijection() -> None:
    """256 个字节与 256 个字符一一对应（这是"字节级无损"的全部理由）."""
    table = bytes_to_unicode()
    assert len(table) == 256
    assert set(table) == set(range(256))
    assert len(set(table.values())) == 256


def test_bytes_to_unicode_maps_readable_ascii_to_itself() -> None:
    """可见 ASCII 原地映到自身；不可打印的那些落在 ``chr(256+n)`` 那一段里.

    不可打印的字节按**字节序**依次编号：第 0 个缺失字节（``0x00``）映到 ``chr(256)``，
    紧随其后的 33 个（``0x00``~``0x20``）之后轮到 ``0x7f``，它映到 ``chr(256+33)``。
    """
    table = bytes_to_unicode()
    assert table[ord("a")] == "a"
    assert table[ord("!")] == "!"
    assert table[0] == chr(256)
    assert table[10] == chr(266)
    assert table[127] == chr(256 + 33)
    assert table[32] not in (" ", "\n")


def test_get_pairs_returns_adjacent_pairs() -> None:
    """相邻对是一个集合（重复的只算一次）."""
    assert get_pairs(("a", "b", "c")) == {("a", "b"), ("b", "c")}
    assert get_pairs(("a", "a", "a")) == {("a", "a")}
    assert get_pairs(("a",)) == set()


# --------------------------------------------------------------------------- 预分词


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello world", ("hello", " world")),
        ("hello   world", ("hello", "   ", "world")),
        ("don't stop", ("don", "'t", " stop")),
        ("a1b", ("a", "1", "b")),
        ("hello,world!", ("hello", ",", "world", "!")),
        ("  leading", ("  ", "leading")),
        ("trailing  ", ("trailing", "  ")),
        ("", ()),
        ("\t", ("\t",)),
        ("你好 world", ("你好", " world")),
    ],
)
def test_pretokenize_follows_the_four_branches(text: str, expected: tuple[str, ...]) -> None:
    """四条分支的顺序即优先级：**词首空格与词绑在一起**，不是一个独立的 token."""
    assert pretokenize(text) == expected


def test_leading_space_is_sticky_rather_than_separate() -> None:
    """这一条设计有一个立刻可见的后果：生成文本里的空格不会丢."""
    pieces = pretokenize("a b")
    assert pieces == ("a", " b")
    assert " " not in pieces


def test_contractions_are_case_sensitive() -> None:
    """缩写识别**区分大小写**（真实实现的模式里没有 ``IGNORECASE``）.

    ``"don't"`` 的小写后缀命中第一条分支；``"I'M"`` 的大写后缀不命中，
    于是它落到"其他非空白"那条分支上切成 ``("I", "'", "M")``。
    """
    assert pretokenize("I'M here") == ("I", "'", "M", " here")
    assert pretokenize("i'm here") == ("i", "'m", " here")


def test_contraction_at_the_end_of_text() -> None:
    """文本以缩写结尾时也要能切（长度不足的候选会被跳过）."""
    assert pretokenize("don't") == ("don", "'t")


# --------------------------------------------------------------------------- 合并表


def test_parse_merges_skips_header_and_blank_lines() -> None:
    """注释与空行跳过，其余每行必须是两个片段."""
    assert parse_merges("#version: 0.2\n\na b\nc d\n") == (("a", "b"), ("c", "d"))


def test_broken_merges_line_is_a_token_error() -> None:
    """行数不对 ⇒ TokenError（静默跳过会得到一份被截短的合并表）."""
    with pytest.raises(errors.TokenError, match="两个"):
        parse_merges("a b c\n")
    with pytest.raises(errors.TokenError, match="两个"):
        parse_merges("a\n")
    with pytest.raises(errors.TokenError, match="空的"):
        parse_merges("#version: 0.2\n\n")


def test_merge_ranks_let_the_last_duplicate_win() -> None:
    """**重复的合并对由最后一次出现决定名次**（复刻 ``dict(zip(...))`` 的行为）.

    这是一条容易被直觉读反的事实：看着像"第一次生效"，而真实实现里是反的。
    """
    ranks = merge_ranks((("a", "b"), ("c", "d"), ("a", "b")))
    assert ranks[("a", "b")] == 2
    assert ranks[("c", "d")] == 1


# --------------------------------------------------------------------------- 构造


def test_tiny_tokenizer_shape() -> None:
    """玩具词表：256 个基元 + 合并 + 一个特殊 token."""
    tokenizer = build_tiny_tokenizer()
    assert tokenizer.base_size == 256
    assert tokenizer.vocab_size == 256 + len(tokenizer.merges) + 1
    assert tokenizer.eos_id == tokenizer.vocab_size - 1
    assert tokenizer.pad_id == tokenizer.eos_id
    assert tokenizer.pad_token == "<|endoftext|>"


def test_tiny_tokenizer_without_end_of_text() -> None:
    """``end_of_text=False`` 时没有 eos，``pad_id`` 退回 0（**并且是查得到的那个 0**）."""
    tokenizer = build_tiny_tokenizer(end_of_text=False)
    assert tokenizer.eos_id is None
    assert tokenizer.pad_id == 0
    assert tokenizer.pad_token == tokenizer.id_to_token[0]


def test_extras_land_in_the_vocabulary_but_merging_is_what_shrinks_a_word() -> None:
    """``extras`` 把一个 token 放进**词表**；而"一个词变成一个 token"靠的是**合并链**.

    这两件事常被混为一谈：往词表里塞一个 ``"world"`` 不会让 ``encode("world")`` 变短，
    因为 BPE 只能沿合并表往前走。真正的短来自 ``hello`` 那条链
    （``h+e → he``、``l+l → ll``、``he+ll → hell``、``hell+o → hello``）。
    """
    built = build_tiny_tokenizer()
    with_extra = build_tiny_tokenizer(extras=("world",))
    mapped = "".join(built.byte_encoder[byte] for byte in "world".encode("utf-8"))
    assert mapped not in built.token_to_id
    assert mapped in with_extra.token_to_id
    assert len(built.encode("hello")) == 1
    assert len(built.encode("world")) > 1


def test_from_files_accepts_json_text_and_dict() -> None:
    """两个入口给出同一个分词器（``vocab`` 可以是文本也可以是已解析的字典）.

    第二行同时说明一件事：真实 GPT-2 的 ``vocab.json`` 里**本来就有**
    ``"<|endoftext|>"`` 这个键，但"它在词表里"与"它被当成特殊 token"是两回事
    ——不声明它是特殊 token 时，它会被**当普通文本切**。
    """
    built = build_tiny_tokenizer()
    text = ByteBPETokenizer.from_files(
        built.vocab, merges_text(built), name="from-json", specials=built.specials
    )
    table = ByteBPETokenizer.from_files(built.vocab, merges_text(built))
    assert text.vocab_size == table.vocab_size == built.vocab_size
    assert text.name == "from-json"
    assert text.eos_id == built.eos_id
    assert table.eos_id is None
    assert len(table.encode("hey<|endoftext|>hey")) > len(text.encode("hey<|endoftext|>hey"))


def test_from_files_rejects_broken_inputs() -> None:
    """四种坏输入分别报（JSON 坏、顶层不是对象、id 不是整数、id 是负数）."""
    built = build_tiny_tokenizer()
    merges = merges_text(built)
    with pytest.raises(errors.TokenError, match="JSON"):
        ByteBPETokenizer.from_files("{", merges)
    with pytest.raises(errors.TokenError, match="非空对象"):
        ByteBPETokenizer.from_files("[]", merges)
    with pytest.raises(errors.TokenError, match="整数"):
        ByteBPETokenizer.from_files('{"a": "1"}', merges)
    with pytest.raises(errors.TokenError, match="负数"):
        ByteBPETokenizer.from_files('{"a": -1}', merges)


def test_constructor_rejects_broken_tables() -> None:
    """四类不成立的组合：空词表、空合并表、特殊 token 的 id 与词表不一致、id 不唯一."""
    built = build_tiny_tokenizer()
    with pytest.raises(errors.TokenError, match="词表为空"):
        ByteBPETokenizer(vocab={}, merges=built.merges)
    with pytest.raises(errors.TokenError, match="合并表为空"):
        ByteBPETokenizer(vocab=built.vocab, merges=())
    with pytest.raises(errors.TokenError, match="id 不同"):
        ByteBPETokenizer(
            vocab=built.vocab, merges=built.merges, specials={"<|endoftext|>": 0}
        )
    duplicated = dict(built.vocab)
    duplicated[built.id_to_token[0]] = 1
    with pytest.raises(errors.TokenError, match="id"):
        ByteBPETokenizer(vocab=duplicated, merges=built.merges)


def test_merges_must_be_closed_over_the_vocab() -> None:
    """合并的**结果**必须在词表里，否则切出来会得到一个查不到的 token."""
    built = build_tiny_tokenizer()
    trimmed = dict(built.vocab)
    merged_token = built.merges[0][0] + built.merges[0][1]
    del trimmed[merged_token]
    with pytest.raises(errors.TokenError, match="同源"):
        ByteBPETokenizer(vocab=trimmed, merges=built.merges)


def test_from_snapshot_builds_the_same_tokenizer() -> None:
    """从快照里读两个文件与直接构造等价（这是"装载"的最后一步）."""
    case = build_case("gpt2")
    tokenizer = ByteBPETokenizer.from_snapshot(case.resolver, case.snapshot)
    assert tokenizer.vocab_size == build_tiny_tokenizer().vocab_size
    assert tokenizer.name == case.snapshot.repo_id


# --------------------------------------------------------------------------- bpe 与编解码


def test_bpe_merges_by_rank_order() -> None:
    """合并按**名次**来：名次小的先合（顺序即优先级）."""
    tokenizer = build_tiny_tokenizer()
    table = tokenizer.byte_encoder
    mapped = "".join(table[byte] for byte in "hello".encode("utf-8"))
    assert tokenizer.bpe(mapped) == ("hello",)
    assert tokenizer.bpe(table[ord("x")]) == (table[ord("x")],)


def test_bpe_caches_pieces() -> None:
    """同一段片段只算一次（缓存是可见的：再问一次走的是同一个结果）."""
    tokenizer = build_tiny_tokenizer()
    table = tokenizer.byte_encoder
    mapped = "".join(table[byte] for byte in "world".encode("utf-8"))
    first = tokenizer.bpe(mapped)
    assert mapped in tokenizer.cache
    assert tokenizer.bpe(mapped) == first


def test_encode_decode_round_trip_on_many_texts() -> None:
    """**逐位**往返：字节级映射是可逆的，因此这条判据没有容差."""
    tokenizer = build_tiny_tokenizer()
    samples = (
        "hello world",
        "don't stop",
        "你好，世界",
        "emoji 🙂 ok",
        "a1b2c3",
        "  leading",
        "trailing  ",
        "tab\tand\nnewline",
        "?!@#$%^&*()",
    )
    for text in samples:
        assert tokenizer.round_trip(text) is True, text


def test_chinese_costs_three_tokens_under_this_vocabulary() -> None:
    """一个汉字在这份词表下**恰好 3 个 token**（UTF-8 三字节）.

    这个数是**词表的性质**，不是分词的铁律：真实 GPT-2 的 50257 词表里
    常见汉字常被合并成 1~2 个 token——而两边都没错。
    """
    tokenizer = build_tiny_tokenizer()
    assert len(tokenizer.encode("好")) == 3
    assert tokenizer.decode(tokenizer.encode("好")) == "好"


def test_encode_empty_text_is_empty() -> None:
    """空文本编出 0 个 token（不是"一个 padding"）."""
    assert build_tiny_tokenizer().encode("") == []


def test_encode_rejects_non_strings() -> None:
    """``encode`` 只吃字符串."""
    with pytest.raises(errors.ParameterError, match="字符串"):
        build_tiny_tokenizer().encode(123)  # type: ignore[arg-type]


def test_encode_handles_special_tokens() -> None:
    """特殊 token 先被切出来，再作为**整块**映射成一个 id."""
    tokenizer = build_tiny_tokenizer()
    ids = tokenizer.encode("hey<|endoftext|>hey")
    assert tokenizer.eos_id in ids
    assert tokenizer.decode(ids) == "heyhey"
    assert tokenizer.decode(ids, skip_special_tokens=False) == "hey<|endoftext|>hey"
    assert tokenizer.encode("hey<|endoftext|>hey", allow_special=False) != ids


def test_encode_reports_a_broken_vocabulary() -> None:
    """词表缺了某个字节字符时，报的是"词表坏了"，而不是"这段文本有问题"."""
    built = build_tiny_tokenizer()
    trimmed = dict(built.vocab)
    del trimmed[built.byte_encoder[ord("z")]]
    tokenizer = ByteBPETokenizer(vocab=trimmed, merges=built.merges)
    assert tokenizer.round_trip("hello") is True
    with pytest.raises(errors.TokenError, match="不在词表里"):
        tokenizer.encode("z")


def test_decode_rejects_bad_ids() -> None:
    """非整数 id 与越界 id 分别报（越界 id 意味着"它来自另一个分词器"）."""
    tokenizer = build_tiny_tokenizer()
    with pytest.raises(errors.ParameterError, match="整数"):
        tokenizer.decode(["a"])  # type: ignore[list-item]
    with pytest.raises(errors.TokenError, match="不在词表里"):
        tokenizer.decode([tokenizer.vocab_size + 5])


def test_encode_with_mask_has_no_padding() -> None:
    """单条编码不带填充：掩码全是 1."""
    encoding = build_tiny_tokenizer().encode_with_mask("hello")
    assert encoding.real_length == encoding.length
    assert encoding.padding == 0
    assert encoding.truncated is False
    assert set(encoding.attention_mask) == {1}


# --------------------------------------------------------------------------- 批量


def test_batch_encode_pads_to_the_longest() -> None:
    """``padding="longest"``：填到这一批最长的那个，掩码区分真实与填充."""
    tokenizer = build_tiny_tokenizer()
    batch = tokenizer.batch_encode(["hey", "hello world"])
    assert batch.padding == PADDING_LONGEST
    assert batch.width == batch.rows[1].length
    assert batch.rows[0].padding == batch.width - batch.rows[0].real_length
    assert batch.rows[0].attention_mask[:2] == (1, 1)
    assert set(batch.rows[0].attention_mask[2:]) == {0}
    assert 0.0 < batch.padding_ratio < 1.0
    assert batch.real_width == batch.width


def test_batch_encode_max_length_padding_is_stable() -> None:
    """``padding="max_length"``：每一批的宽度都由参数决定（不依赖同批的别人）."""
    tokenizer = build_tiny_tokenizer()
    batch = tokenizer.batch_encode(["hey"], padding=PADDING_MAX_LENGTH, max_length=9)
    assert batch.width == 9
    assert batch.max_length == 9
    assert batch.rows[0].length == 9
    assert batch.rows[0].real_length == len(tokenizer.encode("hey"))


def test_batch_encode_truncates_from_the_right() -> None:
    """截断**从右边**（与真实库的默认方向一致），并留下一个可查的标记."""
    tokenizer = build_tiny_tokenizer()
    full = tokenizer.encode("hello world")
    batch = tokenizer.batch_encode(["hello world"], max_length=3)
    assert batch.rows[0].input_ids == tuple(full[:3])
    assert batch.rows[0].truncated is True


def test_batch_encode_refuses_to_truncate_silently() -> None:
    """``truncation=False`` 时超过预算**当场报错**（不替调用方砍）."""
    with pytest.raises(errors.ShapeError, match="超过 max_length"):
        build_tiny_tokenizer().batch_encode(["hello world"], max_length=2, truncation=False)


def test_batch_encode_rejects_bad_arguments() -> None:
    """四类坏参数：空批次、非正 max_length、未知策略、``padding=False``."""
    tokenizer = build_tiny_tokenizer()
    with pytest.raises(errors.ShapeError, match="空的一批"):
        tokenizer.batch_encode([])
    with pytest.raises(errors.ParameterError, match="max_length"):
        tokenizer.batch_encode(["hey"], max_length=0)
    with pytest.raises(errors.ParameterError, match="填充策略"):
        tokenizer.batch_encode(["hey"], padding="left-ish")
    with pytest.raises(errors.ParameterError, match="padding=False"):
        tokenizer.batch_encode(["hey"], padding=False)
    with pytest.raises(errors.ParameterError, match="bool"):
        tokenizer.batch_encode(["hey"], padding=3)  # type: ignore[arg-type]


def test_max_length_padding_requires_the_budget() -> None:
    """``padding='max_length'`` 必须同时给 ``max_length``（否则宽度会随批次变）."""
    with pytest.raises(errors.ParameterError, match="max_length"):
        build_tiny_tokenizer().batch_encode(["hey"], padding=PADDING_MAX_LENGTH)


def test_pad_row_refuses_a_narrower_target() -> None:
    """目标宽度比这一行还窄时当场拒绝（而不是截掉它的尾巴）."""
    tokenizer = build_tiny_tokenizer()
    encoding = tokenizer.encode_with_mask("hello world")
    assert encoding.length > 2
    with pytest.raises(errors.ShapeError, match="目标宽度"):
        tokenizer._pad_row(encoding, 1, tokenizer.pad_id)


def test_batch_encode_keeps_the_real_tokens() -> None:
    """填充只加在右边：真实那一段**逐位不变**（这是池化能对的前提）."""
    tokenizer = build_tiny_tokenizer()
    batch = tokenizer.batch_encode(["hey", "hello world"])
    assert batch.rows[0].input_ids[: batch.rows[0].real_length] == tuple(tokenizer.encode("hey"))


# --------------------------------------------------------------------------- 特殊 token 切分


def test_split_on_specials_prefers_the_longest_match() -> None:
    """长的优先（否则 ``<|ab|>`` 会被 ``<|a`` 先咬住）."""
    assert split_on_specials("x<|ab|>y<|a|>z", ("<|a|>", "<|ab|>")) == (
        "x",
        "<|ab|>",
        "y",
        "<|a|>",
        "z",
    )


def test_split_on_specials_edge_cases() -> None:
    """空表、空文本、以特殊 token 开头都要能处理."""
    assert split_on_specials("abc", ()) == ("abc",)
    assert split_on_specials("", ("<|e|>",)) == ()
    assert split_on_specials("<|e|>abc", ("<|e|>",)) == ("<|e|>", "abc")
    assert split_on_specials("abc", ("<|e|>",)) == ("abc",)


def test_tokenize_returns_the_pieces() -> None:
    """``tokenize`` 与 ``encode`` 逐位对应（一个给字符串、一个给 id）."""
    tokenizer = build_tiny_tokenizer()
    pieces = tokenizer.tokenize("hey")
    assert len(pieces) == len(tokenizer.encode("hey"))
    assert all(piece in tokenizer.token_to_id for piece in pieces)
    assert len(tokenizer) == tokenizer.vocab_size
