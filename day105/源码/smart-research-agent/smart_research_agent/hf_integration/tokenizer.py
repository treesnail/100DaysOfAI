"""``tokenizer``：两个文件加一段算术——**byte-level BPE**（day086 / M7-D10）.

GPT-2 的分词器不是"一个模型"，而是**两份数据 + 一循环**：

```text
vocab.json      token 字符串 → id      （含 256 个"字节字符"作为基元）
merges.txt      合并规则，**顺序即名次**（先出现的先被用）
一个循环        在每一对相邻片段里挑名次最小的那一对合并，直到没有可合并的对
```

## 三件必须先说清的事

```text
① "字节级"是什么意思
   先把文本按 UTF-8 编成字节，再把每个字节映射成一个**可打印字符**
   （bytes_to_unicode：256 个字节与 256 个字符一一对应）。
   于是"任何字节串都能被切出来"——这一族里**没有 unknown token 这件事**。

② 一个汉字要几个 token 是**词表的性质**，不是分词的铁律
   本包的玩具词表只有 256 个基元 + 少量合并，因此一个汉字恰好 3 个 token
   （UTF-8 三字节）。真实 GPT-2 的 50257 词表里，常见汉字常被合并成 1~2 个 token。
   同一条文本在两份词表下的 token 数可以差三倍——而**两边都没错**。

③ 预分词是四条分支，不是"按空格切"
   GPT-2 在 BPE 之前先按四条规则切：缩写（'s / 't / 're / …）、
   "可选空格 + 字母串"、"可选空格 + 数字串"、"可选空格 + 其他非空白串"、
   以及一段尾随空白。本包用一段**手写扫描器**复刻这四条分支
   （真实实现用 ``regex`` 库的 ``\\p{L}`` / ``\\p{N}``，本包不用额外依赖）。
```

## 一个被复刻的怪癖（值得写下来）

真实的 ``Encoder.__init__`` 里是 ``dict(zip(bpe_merges, range(len(bpe_merges))))``。
``dict`` 的构造会让**后出现的重复键覆盖先出现的**——因此一份含重复合并对的
``merges.txt`` 里，是**最后一次**出现决定名次，而不是第一次。
本包逐字复刻这条行为，并给它配一条测试：一个看着像"第一次生效"的直觉，
在真实实现里是反的。
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field

from smart_research_agent.hf_integration.errors import ParameterError, ShapeError, TokenError
from smart_research_agent.hf_integration.hub import HubResolver
from smart_research_agent.hf_integration.types import (
    MERGES_FILE,
    VOCAB_FILE,
    Encoding,
    Snapshot,
    TextBatch,
)

#: 词表文件的版本注释行（真实文件的第一行就是它）。
MERGES_HEADER = "#version"

#: 缩写类后缀（GPT-2 的第一条分支）。**注意它是大小写敏感的**：
#: 真实实现的模式是 ``'s|'t|'re|'ve|'m|'ll|'d``，没有 ``IGNORECASE``——
#: 因此 ``"I'M"`` 会被切成 ``("I", "'", "M")`` 而不是 ``("I", "'M")``。
CONTRACTION_SUFFIXES: tuple[str, ...] = ("s", "t", "re", "ve", "m", "ll", "d")

#: 三种填充策略（``True`` 等价于 ``"longest"``，与真实库一致）。
PADDING_LONGEST = "longest"
PADDING_MAX_LENGTH = "max_length"
PADDING_STRATEGIES: tuple[str, ...] = (PADDING_LONGEST, PADDING_MAX_LENGTH)


def bytes_to_unicode() -> dict[int, str]:
    """256 个字节 → 256 个可打印字符（GPT-2 的 ``bytes_to_unicode``）.

    映射规则分两段：先把可见的 ASCII 与 Latin-1 段**原地**映到自身，
    再把剩下的字节（控制字符与 128~160 之间的空白类）映到 ``256+n`` 上。
    因此这个映射是一一对应的、可逆的——"字节级"能做到无损的原因就在这里。
    """
    direct = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes = list(direct)
    extra = 0
    for byte in range(2**8):
        if byte not in direct:
            direct.append(byte)
            codes.append(2**8 + extra)
            extra += 1
    return dict(zip(direct, (chr(code) for code in codes), strict=True))


def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """一段片段里所有**相邻**的对（集合，因此重复的只算一次）."""
    return {(word[index], word[index + 1]) for index in range(len(word) - 1)}


def _is_letter(char: str) -> bool:
    """是不是"字母类"（对应真实实现的 ``\\p{L}``：含汉字与日文假名）."""
    return unicodedata.category(char).startswith("L")


def _is_digit(char: str) -> bool:
    """是不是"数字类"（对应 ``\\p{N}``）."""
    return unicodedata.category(char).startswith("N")


def _is_space(char: str) -> bool:
    """是不是空白（``\\s`` 的 Python 口径）."""
    return char.isspace()


def pretokenize(text: str) -> tuple[str, ...]:
    """GPT-2 的四条预分词分支（手写扫描器版本，顺序即优先级）.

    ```text
    ① 缩写        's  't  're  've  'm  'll  'd     （**不吃**前面的空格）
    ② 字母串      可选空格 + 连续的字母类字符
    ③ 数字串      可选空格 + 连续的数字类字符
    ④ 其他非空白  可选空格 + 连续的非空白、非字母、非数字字符
    ⑤ 空白串      连续的空白（真实实现里带一个 (?!\\S) 的尾巴规则）
    ```

    分支②③④都**允许带一个前导空格**，这条设计有一个立刻可见的后果：
    ``"hello world"`` 会切成 ``("hello", " world")`` 而不是 ``("hello", " ", "world")``
    ——因此"词首空格"是与词**绑在一起**的一个 token，而不是一个独立的空格 token。
    这就是为什么 GPT-2 的生成文本里空格从不丢失。
    """
    pieces: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        # ① 缩写：'s / 't / 're / 've / 'm / 'll / 'd（**不吃**前面的空格）
        if char == "'":
            matched: str | None = None
            for suffix in sorted(CONTRACTION_SUFFIXES, key=len, reverse=True):
                chunk = text[index + 1 : index + 1 + len(suffix)]
                if chunk == suffix:
                    matched = suffix
                    break
            if matched is not None:
                pieces.append(text[index : index + 1 + len(matched)])
                index += 1 + len(matched)
                continue
        # ②③④ 可选空格 + 一段同类字符。**这三条必须排在空白串之前**：
        # 否则 " world" 会先被 ⑤ 咬走那个空格，于是每个词首都多出一个空格 token。
        start = index
        head_index = index
        if char == " " and index + 1 < length and not _is_space(text[index + 1]):
            head_index = index + 1
        head = text[head_index]
        if _is_letter(head):
            index = head_index
            while index < length and _is_letter(text[index]):
                index += 1
        elif _is_digit(head):
            index = head_index
            while index < length and _is_digit(text[index]):
                index += 1
        elif not _is_space(head):
            index = head_index
            while (
                index < length
                and not _is_space(text[index])
                and not _is_letter(text[index])
                and not _is_digit(text[index])
            ):
                index += 1
        else:
            # ⑤ 空白串（真实实现里带一个 (?!\S) 的尾巴规则；本包按"整段空白"处理）
            while index < length and _is_space(text[index]):
                index += 1
        pieces.append(text[start:index])
    return tuple(piece for piece in pieces if piece)


def parse_merges(text: str) -> tuple[tuple[str, str], ...]:
    """解析 ``merges.txt``：跳过注释与空行，其余每行必须是**两个**片段.

    行数不对时按"这份文件坏了"报（:class:`TokenError`），而不是静默跳过——
    静默跳过的后果是一份被截短的合并表，它照样能切出 token，
    只是切法与训练时**不一样**。
    """
    merges: list[tuple[str, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(MERGES_HEADER):
            continue
        parts = stripped.split(" ")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise TokenError(
                f"merges.txt 第 {number} 行不是两个以空格分隔的片段：{line!r}。"
                "这份文件坏了——静默跳过它的后果是一份被截短的合并表，"
                "它照样能切出 token，只是切法与训练时不一样。"
            )
        merges.append((parts[0], parts[1]))
    if not merges:
        raise TokenError("merges.txt 里一条合并规则都没有：这份文件是空的或只有注释。")
    return tuple(merges)


def merge_ranks(merges: tuple[tuple[str, str], ...]) -> dict[tuple[str, str], int]:
    """合并对 → 名次。**重复的对由最后一次出现决定**（复刻 ``dict(zip(...))`` 的行为）."""
    return dict(zip(merges, range(len(merges)), strict=True))


@dataclass
class ByteBPETokenizer:
    """一份 byte-level BPE 分词器（**只有两个数据源**：词表与合并表）."""

    vocab: dict[str, int]
    merges: tuple[tuple[str, str], ...]
    name: str = "byte-bpe"
    specials: dict[str, int] = field(default_factory=dict)
    byte_encoder: dict[int, str] = field(default_factory=bytes_to_unicode)
    cache: dict[str, tuple[str, ...]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.vocab:
            raise TokenError("词表为空：一个 token 都没有的分词器切不出任何东西。")
        if not self.merges:
            raise TokenError("合并表为空：那样每个字节都会单独成为一个 token。")
        # 特殊 token **可以**同时在普通词表里（真实 GPT-2 的 vocab.json 里就有
        # ``"<|endoftext|>": 50256``），但两处的 id 必须一致——不一致时那段文本
        # 会按两套 id 里的哪一套走，取决于实现顺序。
        for token, token_id in self.specials.items():
            existing = self.vocab.get(token)
            if existing is not None and existing != token_id:
                raise TokenError(
                    f"特殊 token {token!r} 与普通 token 撞名且 id 不同"
                    f"（{existing} vs {token_id}）：两套 id 并存时，"
                    "一段恰好等于该字符串的文本会被按哪一套切，取决于实现顺序。"
                )
        self.byte_decoder = {char: byte for byte, char in self.byte_encoder.items()}
        self.token_to_id = dict(self.vocab)
        self.token_to_id.update(self.specials)
        self.id_to_token = {token_id: token for token, token_id in self.token_to_id.items()}
        if len(self.id_to_token) != len(self.token_to_id):
            raise TokenError("词表里有两个 token 共用一个 id：id 必须唯一。")
        self.ranks = merge_ranks(self.merges)
        self._check_merges_are_closed()

    # -- 自检 ---------------------------------------------------------------------

    def _check_merges_are_closed(self) -> None:
        """每一条合并的**结果**都必须在词表里（否则切出来会得到一个不存在的 token）.

        这就是"词表与合并表必须同源"那条边界在代码里的样子：
        两个来源的文件能拼出一个**可运行**的分词器，而它在某一段文本上
        会抛出一个不存在的 token——那时模型不报错，只是答得不对。
        """
        for first, second in self.merges:
            if first + second not in self.vocab:
                raise TokenError(
                    f"merges.txt 里有一条合并 {first!r} + {second!r} = "
                    f"{first + second!r}，而这个词表里没有它："
                    "vocab.json 与 merges.txt 必须同源，否则切出来的 token 无处可查。"
                )

    @property
    def vocab_size(self) -> int:
        """词表大小（含特殊 token）."""
        return len(self.token_to_id)

    @property
    def base_size(self) -> int:
        """基元个数（"一个字节能被表示"的那些字符）——它必须恰好是 256."""
        return sum(1 for token in self.vocab if len(token) == 1 and token in self.byte_decoder)

    def __len__(self) -> int:
        return self.vocab_size

    # -- 构造 ---------------------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        vocab: dict[str, int] | str,
        merges: str,
        *,
        name: str = "byte-bpe",
        specials: dict[str, int] | None = None,
    ) -> ByteBPETokenizer:
        """从两个文件的内容构造（``vocab`` 可以是已解析的字典，也可以是 JSON 文本）."""
        if isinstance(vocab, str):
            try:
                payload = json.loads(vocab)
            except json.JSONDecodeError as exc:
                raise TokenError(
                    f"vocab.json 不是合法 JSON（{exc.msg}，第 {exc.lineno} 行）。"
                ) from exc
        else:
            payload = dict(vocab)
        if not isinstance(payload, dict) or not payload:
            raise TokenError("vocab.json 的顶层必须是一个非空对象（token → id）。")
        table: dict[str, int] = {}
        for token, token_id in payload.items():
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TokenError(f"词表里 {token!r} 的 id 是 {token_id!r}，不是整数。")
            if token_id < 0:
                raise TokenError(f"词表里 {token!r} 的 id 是负数 {token_id}。")
            table[str(token)] = token_id
        return cls(
            vocab=table,
            merges=parse_merges(merges),
            name=name,
            specials=dict(specials or {}),
        )

    @classmethod
    def from_snapshot(
        cls,
        resolver: HubResolver,
        snapshot: Snapshot,
        *,
        vocab_file: str = VOCAB_FILE,
        merges_file: str = MERGES_FILE,
        specials: dict[str, int] | None = None,
        name: str | None = None,
    ) -> ByteBPETokenizer:
        """从一份快照里读出两个文件并构造（**这是"装载"的最后一步**）."""
        return cls.from_files(
            resolver.read_text(snapshot, vocab_file),
            resolver.read_text(snapshot, merges_file),
            name=name or snapshot.repo_id,
            specials=specials,
        )

    # -- 核心算法 -----------------------------------------------------------------

    def bpe(self, token: str) -> tuple[str, ...]:
        """在一段（**已经映射成字节字符**的）片段上跑合并循环.

        循环每一步挑"名次最小"的那一对合并——因此**顺序即优先级**。
        名次最小的对不存在时循环结束。缓存按片段记，因为同一段片段
        在一次批量编码里会被反复遇到。
        """
        cached = self.cache.get(token)
        if cached is not None:
            return cached
        word = tuple(token)
        if len(word) < 2:
            self.cache[token] = word
            return word
        pairs = get_pairs(word)
        while True:
            best: tuple[str, str] | None = None
            best_rank = None
            for pair in pairs:
                rank = self.ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best, best_rank = pair, rank
            if best is None:
                break
            first, second = best
            merged: list[str] = []
            index = 0
            while index < len(word):
                if word[index] == first and index + 1 < len(word) and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        self.cache[token] = word
        return word

    def to_pieces(self, text: str) -> tuple[str, ...]:
        """文本 → 字节字符片段串（**不查词表**，因此它只反映切法）."""
        pieces: list[str] = []
        for chunk in pretokenize(text):
            mapped = "".join(self.byte_encoder[byte] for byte in chunk.encode("utf-8"))
            pieces.extend(self.bpe(mapped))
        return tuple(pieces)

    def tokenize(self, text: str) -> tuple[str, ...]:
        """文本 → token 字符串串（含特殊 token 的切分）."""
        return tuple(self.id_to_token[token_id] for token_id in self.encode(text))

    def encode(self, text: str, *, allow_special: bool = True) -> list[int]:
        """文本 → id 串（``allow_special=True`` 时先按特殊 token 切开）.

        任何字节都是合法的基元，因此这一族里**没有 unknown token**；
        只有当词表本身缺了某个字节字符时才会抛 :class:`TokenError`——
        那不是"这段文本的问题"，而是"这份词表坏了"。
        """
        if not isinstance(text, str):
            raise ParameterError(f"encode 需要字符串，收到 {type(text).__name__}。")
        if not text:
            return []
        pieces: list[str] = []
        if allow_special and self.specials:
            for chunk in split_on_specials(text, tuple(self.specials)):
                if chunk in self.specials:
                    pieces.append(chunk)
                else:
                    pieces.extend(self.to_pieces(chunk))
        else:
            pieces.extend(self.to_pieces(text))
        ids: list[int] = []
        for piece in pieces:
            token_id = self.token_to_id.get(piece)
            if token_id is None:
                raise TokenError(
                    f"片段 {piece!r} 不在词表里：它来自某个字节的映射，"
                    "因此这说明词表缺了这个字节字符（词表坏了，不是文本坏了）。"
                )
            ids.append(token_id)
        return ids

    def decode(self, ids: tuple[int, ...] | list[int], *, skip_special_tokens: bool = True) -> str:
        """id 串 → 文本（**逐位还原**：字节级映射是可逆的）."""
        chunks: list[bytes] = []
        for token_id in ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise ParameterError(f"decode 需要整数 id，收到 {token_id!r}。")
            piece = self.id_to_token.get(token_id)
            if piece is None:
                raise TokenError(
                    f"id {token_id} 不在词表里（词表大小 {self.vocab_size}）："
                    "它可能来自另一个分词器——那时解码出来的文本是**另一套**字节。"
                )
            if piece in self.specials:
                if skip_special_tokens:
                    continue
                chunks.append(piece.encode("utf-8"))
                continue
            try:
                chunks.append(bytes(self.byte_decoder[char] for char in piece))
            except KeyError as exc:  # pragma: no cover - 词表里有非字节字符才触发
                raise TokenError(
                    f"token {piece!r} 里有不在字节映射里的字符 {exc.args[0]!r}："
                    "字节级 BPE 的每一个 token 都必须由那 256 个字节字符组成。"
                ) from exc
        return b"".join(chunks).decode("utf-8", errors="replace")

    def round_trip(self, text: str) -> bool:
        """``decode(encode(text)) == text``（对任意字符串都必须成立）."""
        return self.decode(self.encode(text)) == text

    def encode_with_mask(self, text: str) -> Encoding:
        """编成一条 :class:`Encoding`（**不填充**：长度就是真实长度）."""
        ids = self.encode(text)
        tokens = tuple(self.id_to_token[token_id] for token_id in ids)
        return Encoding(
            tokens=tokens,
            input_ids=tuple(ids),
            attention_mask=tuple(1 for _ in ids),
        )

    def batch_encode(
        self,
        texts: tuple[str, ...] | list[str],
        *,
        max_length: int | None = None,
        padding: bool | str = True,
        truncation: bool = True,
        pad_token_id: int | None = None,
    ) -> TextBatch:
        """一批文本 → 对齐好的 :class:`TextBatch`（填充 + 截断 + 掩码）.

        三个决定都写在签名里，因为它们**都会改变读数**：

        ```text
        padding="longest"     填到这一批最长的那个（省算力，但同一批的宽度依赖别人）
        padding="max_length"  填到 max_length（宽度稳定，代价是可能填很多）
        truncation=True       超过 max_length 的**从右边**砍掉（默认方向与真实库一致）
        ```
        """
        if not texts:
            raise ShapeError("batch_encode 收到空的一批：没有文本就没有宽度。")
        if max_length is not None and max_length < 1:
            raise ParameterError(f"max_length 必须 >= 1，收到 {max_length}。")
        if isinstance(padding, str):
            if padding not in PADDING_STRATEGIES:
                raise ParameterError(
                    f"未知的填充策略 {padding!r}：可选 {list(PADDING_STRATEGIES)}"
                    "（True 等价于 'longest'）。"
                )
            strategy = padding
        elif padding is True:
            strategy = PADDING_LONGEST
        elif padding is False:
            raise ParameterError(
                "padding=False 与 batch_encode 的契约不符：一批宽度不齐的张量"
                "喂不进一次前向。要逐条编码请直接用 encode_with_mask。"
            )
        else:  # pragma: no cover - 只有传了奇怪的对象才会到这里
            raise ParameterError(f"padding 必须是 bool 或字符串，收到 {padding!r}。")
        length = max_length
        if strategy == PADDING_MAX_LENGTH and length is None:
            raise ParameterError(
                "padding='max_length' 必须同时给 max_length："
                "否则'填到多长'这个问题的答案取决于当前这一批，"
                "而它会在下一批上悄悄变掉。"
            )
        rows: list[Encoding] = []
        for text in texts:
            ids = self.encode(text)
            truncated = False
            if length is not None and len(ids) > length:
                if not truncation:
                    raise ShapeError(
                        f"一条文本编出 {len(ids)} 个 token，超过 max_length={length}，"
                        "而 truncation=False：本包不会替调用方砍掉它。"
                    )
                ids = ids[:length]
                truncated = True
            rows.append(
                Encoding(
                    tokens=tuple(self.id_to_token[token_id] for token_id in ids),
                    input_ids=tuple(ids),
                    attention_mask=tuple(1 for _ in ids),
                    truncated=truncated,
                )
            )
        target = max((row.length for row in rows), default=0) if length is None else length
        if length is not None and strategy == PADDING_LONGEST:
            target = max(row.length for row in rows)
        pad = self.pad_id if pad_token_id is None else pad_token_id
        padded = tuple(self._pad_row(row, target, pad) for row in rows)
        return TextBatch(
            rows=padded,
            padding=strategy,
            pad_token_id=pad,
            max_length=target if length is None else length,
        )

    @property
    def eos_id(self) -> int | None:
        """结束符的 id（``<|endoftext|>`` 这一类的特殊 token；没有就给 ``None``）.

        它同时是"句子的结束"与（在 GPT-2 里）"填充用什么"——
        因此 :attr:`pad_id` 优先取它。**没有 eos 的分词器不会报错**，
        它只会让生成跑满预算（那看起来像"模型话多"）。
        """
        for token, token_id in self.specials.items():
            if token.startswith("<|") and token.endswith("|>"):
                return token_id
        return None

    @property
    def pad_id(self) -> int:
        """填充用的 id：优先取 :attr:`eos_id`，否则退回 id 0.

        真实库里 GPT-2 的 ``pad_token`` 默认是 ``None``，于是"没有 pad token"
        会让批量编码在**没有 padding 的情况下**直接抛错——本包改成"退回 id 0"
        并把它**印出来**（``TextBatch.pad_token_id``），因为一个静默的 0 与
        一个写下来的 0 在报告里必须长得不一样。
        """
        eos = self.eos_id
        return 0 if eos is None else eos

    @property
    def pad_token(self) -> str:
        """填充用的 token **字符串**（词表里查得到就用真的，查不到就标出来）."""
        token_id = self.pad_id
        return self.id_to_token.get(token_id, f"<pad:{token_id}>")

    def _pad_row(self, row: Encoding, target: int, pad_token_id: int) -> Encoding:
        """把一行填到 ``target``（**从右边填**，与真实库的默认方向一致）.

        填充位的 ``attention_mask`` 是 0，而**真实位的掩码全是 1**——
        池化用的就是这张表，因此"填了几个"这件事在数据里是可查的，
        而不是靠"我记得它有多长"。
        """
        if row.length > target:
            raise ShapeError(f"目标宽度 {target} 小于这一行的长度 {row.length}。")
        if row.length == target:
            return row
        label = self.pad_token if pad_token_id == self.pad_id else f"<pad:{pad_token_id}>"
        extra = target - row.length
        return Encoding(
            tokens=row.tokens + tuple(label for _ in range(extra)),
            input_ids=row.input_ids + tuple(pad_token_id for _ in range(extra)),
            attention_mask=row.attention_mask + tuple(0 for _ in range(extra)),
            truncated=row.truncated,
        )


def split_on_specials(text: str, specials: tuple[str, ...]) -> tuple[str, ...]:
    """按特殊 token 把文本切开（**长的优先**，否则 ``<|e|>`` 会被 ``<|`` 先咬住）."""
    if not specials:
        return (text,) if text else ()
    ordered = tuple(sorted(specials, key=len, reverse=True))
    chunks: list[str] = []
    rest = text
    while rest:
        hit = None
        position = len(rest)
        for token in ordered:
            found = rest.find(token)
            if found != -1 and found < position:
                hit, position = token, found
        if hit is None:
            chunks.append(rest)
            break
        if position:
            chunks.append(rest[:position])
        chunks.append(hit)
        rest = rest[position + len(hit) :]
    return tuple(chunk for chunk in chunks if chunk)


def build_tiny_tokenizer(
    *,
    extras: tuple[str, ...] = (),
    end_of_text: bool = True,
) -> ByteBPETokenizer:
    """造一份**确定性**的玩具词表（256 个基元 + 若干合并），测试与演示都从它出发.

    基元是 256 个"字节字符"，因此任何 UTF-8 文本都编码得出（这一族里没有 unknown）。
    合并表刻意包含几条**跨字节**的规则，好让"一个汉字几个 token"这件事
    在两份词表下有不同的答案（见本模块文档第 ② 条）。
    """
    encoder = bytes_to_unicode()
    vocab: dict[str, int] = {}
    for byte in range(256):
        vocab[encoder[byte]] = byte
    merges: list[tuple[str, str]] = []

    def mapped(text: str) -> str:
        """把一段可读文本映成"字节字符"形态（**合并表里的片段就是这个形态**）."""
        return "".join(encoder[byte] for byte in text.encode("utf-8"))

    def add_text(first: str, second: str) -> None:
        """加一条合并（两边都必须已经在词表里，否则这条规则永远不会生效）."""
        left, right = mapped(first), mapped(second)
        if left not in vocab or right not in vocab:
            raise TokenError(
                f"合并规则 {first!r}+{second!r} 的左侧或右侧不在词表里："
                "一条永不生效的规则会让'我写了这条合并'与'它被用上了'看起来一样。"
            )
        merged = left + right
        if merged in vocab:
            return
        vocab[merged] = len(vocab)
        merges.append((left, right))

    for first, second in (
        ("h", "e"),
        ("l", "l"),
        ("he", "ll"),
        ("hell", "o"),
        (" ", "t"),
        (" ", "a"),
        (" ", "s"),
        ("t", "h"),
        ("th", "e"),
        ("a", "n"),
        ("i", "n"),
        ("o", "n"),
        ("e", "r"),
        ("w", "o"),
        ("r", "l"),
        ("d", "a"),
        ("y", "s"),
        ("s", " "),
        ("a", "r"),
        ("e", "n"),
    ):
        add_text(first, second)
    for extra in extras:
        merged = mapped(extra)
        if merged in vocab:
            continue
        vocab[merged] = len(vocab)
    specials: dict[str, int] = {}
    if end_of_text:
        # 真实的 vocab.json 里这个键**就在词表里**（值是最后一个 id），
        # 同时它又被声明为特殊 token。本包照此办理：两处同一份。
        token = "<|endoftext|>"
        vocab[token] = len(vocab)
        specials[token] = vocab[token]
    return ByteBPETokenizer(vocab=vocab, merges=tuple(merges), name="tiny", specials=specials)


__all__ = [
    "CONTRACTION_SUFFIXES",
    "MERGES_HEADER",
    "PADDING_LONGEST",
    "PADDING_MAX_LENGTH",
    "PADDING_STRATEGIES",
    "ByteBPETokenizer",
    "build_tiny_tokenizer",
    "bytes_to_unicode",
    "get_pairs",
    "merge_ranks",
    "parse_merges",
    "pretokenize",
    "split_on_specials",
]
