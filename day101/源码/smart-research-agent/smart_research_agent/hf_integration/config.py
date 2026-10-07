"""``config``：那份 ``config.json`` 就是模型的形状（day086 / M7-D10）.

`from_pretrained` 的第二步是**解释那份配置**。它不是"元数据"——它是模型的**定义**：

```text
n_embd 与 hidden_size        同一个量、两个键名（两个架构各叫各的）
n_head 与 num_attention_heads 同上
n_layer 与 num_hidden_layers  同上
```

因此本模块要做三件事，而三件都可以被断言：

```text
① 把两个架构的键名收敛成一个 ModelCard（同一个量的两个写法）
② 校验"这份配置描述的模型是否成立"（头数必须整除隐藏维，否则 head_dim 没有定义）
③ 由配置**算出参数量**——不是"读出来的"，而是算出来的（与真实库整数相等）
```

## 第三条为什么值钱

"我理解了这个架构"是一句无法被反驳的话。而一旦把它写成
"给定这几个整数，参数量必须等于某个数"，它就变成了一条能失败的判据：

```text
tiny GPT-2   vocab=100 n_embd=16 n_head=2 n_layer=2 n_positions=32 tie=True   →     8 704
gpt2（124M） vocab=50257 n_embd=768 n_head=12 n_layer=12 n_positions=1024     → 124 439 808
bert-base    vocab=30522 hidden=768 heads=12 layers=12 positions=512 type=2   → 109 482 240
```

三个数都能在真实库里逐位复核（本机的 ``transformers`` 实测值，
见 ``tests/test_hf_real_bridge.py`` 与 ``scripts/hf_integration_demo.py``）。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from smart_research_agent.hf_integration.errors import ConfigError
from smart_research_agent.hf_integration.hub import HubResolver
from smart_research_agent.hf_integration.types import (
    ARCHITECTURE_BERT,
    ARCHITECTURE_GPT2,
    ARCHITECTURES,
    CONFIG_FILE,
    FFN_RATIO,
    REQUIRED_CONFIG_KEYS,
    SIZE_KEYS,
    ModelCard,
    Snapshot,
)

#: 两个架构各自认得的激活函数默认值（``config.json`` 里就写着它）.
DEFAULT_ACTIVATIONS: dict[str, str] = {
    ARCHITECTURE_GPT2: "gelu_new",
    ARCHITECTURE_BERT: "gelu",
}

#: 两个架构各自认得的 eps 默认值（day085 第 2 章那张表的同一个来源）.
DEFAULT_LN_EPS: dict[str, float] = {
    ARCHITECTURE_GPT2: 1e-5,
    ARCHITECTURE_BERT: 1e-12,
}

#: BERT 的句子类型表长度默认值（``type_vocab_size``）.
DEFAULT_TYPE_VOCAB = 2

#: ``tie_word_embeddings`` 的默认值：**True，而且两个架构都是**.
#:
#: 这一条值得单独写下来，因为它推翻了"BERT 不共享词嵌入"这个流传很广的说法：
#: 真实 ``bert-base-uncased`` 的 ``config.json`` 里**根本没有这个键**，
#: 于是全局默认 ``True`` 生效——它的 MLM 头（``cls.predictions.decoder.weight``）
#: 与 ``bert.embeddings.word_embeddings.weight`` 是同一份参数。
#: 证据是一个整数：带共享算出来是 109 482 240，不共享会多出 23 440 896。
DEFAULT_TIE = True

#: 两个架构各自支持哪些激活（不在表里的名字当场拒绝——回退到 relu 会让读数印错模型）.
KNOWN_ACTIVATIONS: dict[str, tuple[str, ...]] = {
    ARCHITECTURE_GPT2: ("gelu_new", "gelu", "relu"),
    ARCHITECTURE_BERT: ("gelu", "relu", "gelu_new"),
}


def architecture_of(raw: dict[str, Any]) -> str:
    """从配置里取出 ``model_type`` 并校验它被本包认得.

    ``model_type`` 缺席时**不**回退到某个默认架构——回退的后果是一份 BERT 的
    读数被印成 GPT-2 的（day085 第 2 章那条"绝不回退到默认画像"的同族）。
    """
    if "model_type" not in raw:
        raise ConfigError(
            f"配置里没有 'model_type' 键（现有的键：{sorted(raw)[:8]}…）："
            "它是决定'用哪条参数量公式'的那一个字段，猜不出来。"
        )
    value = raw["model_type"]
    if not isinstance(value, str):
        raise ConfigError(f"'model_type' 必须是字符串，收到 {value!r}。")
    if value not in ARCHITECTURES:
        raise ConfigError(
            f"本包只给了两个架构（{list(ARCHITECTURES)}）的实现，收到 {value!r}："
            "把不认识的名字硬套上某一条公式，会让读数看起来正确。"
        )
    return value


def _positive_int(raw: dict[str, Any], key: str, *, model_type: str) -> int:
    """取一个必须为正整数的字段（缺键、类型不符、非正分别报清楚）."""
    if key not in raw:
        raise ConfigError(
            f"配置里缺少必需的键 {key!r}（架构 {model_type} 需要它）："
            f"每个架构的必需键见 types.REQUIRED_CONFIG_KEYS[{model_type!r}]。"
        )
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"{key!r} 必须是整数，收到 {value!r}（{type(value).__name__}）："
            "JSON 里的 16.0 与 '16' 都能被读出来，而它们都不是一个合法的维度。"
        )
    if value < 1:
        raise ConfigError(f"{key!r} 必须 >= 1，收到 {value}。")
    return value


def parse_config(raw: dict[str, Any], *, name: str = "") -> ModelCard:
    """把一份 ``config.json`` 解析成 :class:`ModelCard`（**两个架构共用一条路径**）.

    三个值得单独指出的决定：

    ```text
    ① 键名映射来自 types.SIZE_KEYS —— 一个地方写、一个地方读，
       因此"n_embd 与 hidden_size 是同一个量"这件事只有一处实现
    ② 头数必须整除隐藏维 —— 否则 head_dim 不是整数，模型**没有定义**
    ③ uses_token_type 由架构决定 —— GPT-2 没有"句子 A / 句子 B"的概念
    ```

    第四个决定藏在一个默认值里，而它值得单独写一段：``tie_word_embeddings``
    的默认值是 **True，两个架构都是**（见 :data:`DEFAULT_TIE`）。
    真实 ``bert-base-uncased`` 的配置文件里没有这个键，因此它**共享**词嵌入——
    这一点常被说反，而它有一个整数的代价（多出 ``vocab × hidden``）。
    """
    model_type = architecture_of(raw)
    keys = SIZE_KEYS[model_type]
    hidden = _positive_int(raw, keys["hidden"], model_type=model_type)
    heads = _positive_int(raw, keys["heads"], model_type=model_type)
    layers = _positive_int(raw, keys["layers"], model_type=model_type)
    positions = _positive_int(raw, keys["positions"], model_type=model_type)
    vocab = _positive_int(raw, "vocab_size", model_type=model_type)
    if hidden % heads != 0:
        raise ConfigError(
            f"{keys['hidden']}={hidden} 不能被 {keys['heads']}={heads} 整除："
            f"head_dim = {hidden}/{heads} 不是整数，这个模型**没有定义**。"
            "真实的装载会在一次 reshape 上失败，而本包在解析配置的那一刻就拒绝——"
            "报错点离出错点越近，修起来越便宜。"
            "（注意 hf_source.split_heads 对同一个数报 ShapeError：那里要改的是"
            "**这次调用的 heads**，这里要改的是**那份配置文件**。）"
        )
    activation = raw.get(keys["activation"], DEFAULT_ACTIVATIONS[model_type])
    if not isinstance(activation, str):
        raise ConfigError(f"{keys['activation']!r} 必须是字符串，收到 {activation!r}。")
    if activation not in KNOWN_ACTIVATIONS[model_type]:
        raise ConfigError(
            f"架构 {model_type} 不认激活 {activation!r}："
            f"本包认得 {list(KNOWN_ACTIVATIONS[model_type])}。"
            "回退到 relu 的后果是——一份 GPT-2 的读数会被印成 BERT 的"
            "（day085 的 activation_of 里那条'绝不回退'）。"
        )
    ln_eps = raw.get(keys["ln_eps"], DEFAULT_LN_EPS[model_type])
    if isinstance(ln_eps, bool) or not isinstance(ln_eps, (int, float)):
        raise ConfigError(f"{keys['ln_eps']!r} 必须是数，收到 {ln_eps!r}。")
    if not 0.0 < float(ln_eps):
        raise ConfigError(f"{keys['ln_eps']!r} 必须为正，收到 {ln_eps!r}。")
    tie = raw.get("tie_word_embeddings", DEFAULT_TIE)
    if not isinstance(tie, bool):
        raise ConfigError(f"'tie_word_embeddings' 必须是布尔值，收到 {tie!r}。")
    uses_token_type = model_type == ARCHITECTURE_BERT
    type_vocab = DEFAULT_TYPE_VOCAB if uses_token_type else 0
    if uses_token_type and "type_vocab_size" in raw:
        type_vocab = _positive_int(raw, "type_vocab_size", model_type=model_type)
    # 前馈中间维：**配置里有就用配置里的**。它值得单独读一次，因为
    # "前馈是 4·hidden"对 BERT 只是一次碰巧（它的缺省是固定的 3072）。
    intermediate = (
        _positive_int(raw, keys["ffn"], model_type=model_type) if keys["ffn"] in raw else None
    )
    raw_architectures = raw.get("architectures") or ()
    if not isinstance(raw_architectures, (list, tuple)):
        raise ConfigError(f"'architectures' 必须是列表，收到 {raw_architectures!r}。")
    return ModelCard(
        name=name or str(raw.get("_name_or_path", "")) or model_type,
        model_type=model_type,
        hidden=hidden,
        heads=heads,
        layers=layers,
        vocab=vocab,
        positions=positions,
        activation=activation,
        ln_eps=float(ln_eps),
        tie_word_embeddings=tie,
        uses_token_type=uses_token_type,
        type_vocab=type_vocab,
        architectures=tuple(str(item) for item in raw_architectures),
        intermediate=intermediate,
    )


def card_from_snapshot(
    resolver: HubResolver,
    snapshot: Snapshot,
    *,
    path: str = CONFIG_FILE,
    name: str = "",
) -> ModelCard:
    """从一份快照里读出配置并解析（**这是"装载"的最后一步**）."""
    raw = resolver.read_json(snapshot, path)
    return parse_config(raw, name=name or snapshot.repo_id)


def missing_keys(raw: dict[str, Any], *, model_type: str) -> tuple[str, ...]:
    """这份配置缺了哪些必需键（顺序与 :data:`types.REQUIRED_CONFIG_KEYS` 一致）."""
    if model_type not in REQUIRED_CONFIG_KEYS:
        raise ConfigError(f"未知架构 {model_type!r}，没有必需键清单。")
    return tuple(key for key in REQUIRED_CONFIG_KEYS[model_type] if key not in raw)


def parameter_count(card: ModelCard) -> int:
    """按配置算出的参数量（**委托给卡片属性**，保证只有一处实现）."""
    return card.parameter_count


def tie_delta(card: ModelCard) -> int:
    """共享与不共享词嵌入之间的参数量之差（``vocab × hidden``，一个整数）."""
    return card.vocab * card.hidden


def untied_card(card: ModelCard) -> ModelCard:
    """同一份配置、但**不共享**词嵌入的那一版（用来把 tying 的代价量出来）."""
    return replace(card, tie_word_embeddings=False)


def tiny_card(
    model_type: str,
    *,
    vocab: int = 100,
    hidden: int = 16,
    heads: int = 2,
    layers: int = 2,
    positions: int = 32,
    name: str = "",
    **overrides: Any,
) -> ModelCard:
    """造一份**极小但合法**的配置（测试与演示都从它出发）。

    默认值刻意选成"两项都不整齐"的样子（``vocab=100`` 不是 2 的幂、
    ``hidden=16`` 而 ``heads=2`` 是真整除），因为过整齐的数字会让
    "某个数与另一个数恰好相等"这类错误看不出来。
    """
    if model_type == ARCHITECTURE_GPT2:
        raw: dict[str, Any] = {
            "model_type": ARCHITECTURE_GPT2,
            "vocab_size": vocab,
            "n_embd": hidden,
            "n_head": heads,
            "n_layer": layers,
            "n_positions": positions,
            # 刻意**不写** ``n_inner``：GPT-2 的规矩是缺省取 4·hidden，
            # 而"缺省"这条路径必须被走到过（写死一个数就把规矩藏起来了）。
            "activation_function": "gelu_new",
            "layer_norm_epsilon": 1e-5,
            "tie_word_embeddings": True,
        }
    elif model_type == ARCHITECTURE_BERT:
        raw = {
            "model_type": ARCHITECTURE_BERT,
            "vocab_size": vocab,
            "hidden_size": hidden,
            "num_attention_heads": heads,
            "num_hidden_layers": layers,
            "max_position_embeddings": positions,
            # 显式写出来，因为 BERT 的缺省是固定的 3072（与 hidden 无关）——
            # 一个 hidden=16 的玩具卡片若吃缺省，前馈会变成 3072 宽。
            "intermediate_size": FFN_RATIO * hidden,
            "hidden_act": "gelu",
            "layer_norm_eps": 1e-12,
            "type_vocab_size": 2,
            "tie_word_embeddings": True,
        }
    else:
        raise ConfigError(
            f"本包只给了两个架构的配置模板（{list(ARCHITECTURES)}），收到 {model_type!r}。"
        )
    raw.update(overrides)
    return parse_config(raw, name=name or f"{model_type}-tiny")


def reference_cards() -> dict[str, ModelCard]:
    """两个**真实模型**的配置（参数量是可在库里复核的那两个整数）.

    这两个卡片不进快照，只用来说明"参数量公式是对的"——
    它们的读数与真实库逐位相等（``tests/test_hf_real_bridge.py`` 会复核）。
    """
    return {
        "gpt2": parse_config(
            {
                "model_type": ARCHITECTURE_GPT2,
                "vocab_size": 50257,
                "n_embd": 768,
                "n_head": 12,
                "n_layer": 12,
                "n_positions": 1024,
                "activation_function": "gelu_new",
                "layer_norm_epsilon": 1e-5,
                "tie_word_embeddings": True,
            },
            name="gpt2",
        ),
        "bert-base-uncased": parse_config(
            {
                "model_type": ARCHITECTURE_BERT,
                "vocab_size": 30522,
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 12,
                "max_position_embeddings": 512,
                # bert-base 的 3072 是**缺省值**（不是 4×768 推出来的）——两者恰好相等
                "intermediate_size": 3072,
                "hidden_act": "gelu",
                "layer_norm_eps": 1e-12,
                "type_vocab_size": 2,
                "tie_word_embeddings": True,
            },
            name="bert-base-uncased",
        ),
    }


def profile_agreement(card: ModelCard) -> dict[str, object]:
    """把卡片与 day085 的**源码画像**对齐（两条来源必须说同一件事）.

    这是本课的一条跨天对账：day085 的画像来自 ``modeling_*.py`` 里的默认值，
    今天的卡片来自 ``config.json``。两者若不一致，那么"我读的那份源码"
    与"我装的那个模型"就不是同一个东西——而它们的读数会**看起来都对**。
    """
    from smart_research_agent.hf_source.types import profile_of

    profile = profile_of(card.model_type)
    return {
        "model_type": card.model_type,
        "activation": {"config": card.activation, "source": profile.activation},
        "ln_eps": {"config": card.ln_eps, "source": profile.ln_eps},
        "fused_qkv": {"config": card.fused_qkv, "source": profile.fused_qkv},
        "uses_token_type": {
            "config": card.uses_token_type,
            "source": profile.uses_token_type,
        },
        "agreed": (
            card.activation == profile.activation
            and card.ln_eps == profile.ln_eps
            and card.fused_qkv == profile.fused_qkv
            and card.uses_token_type == profile.uses_token_type
        ),
    }


def describe(card: ModelCard) -> str:
    """一行说明（报告里读它）: ``gpt2-tiny | d=16 h=2 L=2 V=100 P=32 | 8 704 参数 | causal``."""
    return (
        f"{card.name} | d={card.hidden} h={card.heads} L={card.layers} "
        f"V={card.vocab} P={card.positions} | {card.parameter_count} 参数 | "
        f"{'causal' if card.causal else 'bidirectional'} | tie={card.tie_word_embeddings}"
    )


def ffn_of(card: ModelCard) -> int:
    """前馈中间维（**配置优先**：BERT 的缺省 3072 与 GPT-2 的缺省 4·hidden 各按各的规矩）."""
    return card.ffn


def check_snapshot_is_a_model(resolver: HubResolver, snapshot: Snapshot) -> ModelCard:
    """自检：这份快照的配置必须能被解析（解析不了就原样抛出那一族错误）.

    本函数存在的理由与 day085 的 ``check_score_magnitude`` 相同：
    它把"这份快照是不是一个模型"这件事**收成一个入口**，
    于是调用方不必自己决定该 ``except`` 哪一族。
    """
    return card_from_snapshot(resolver, snapshot)


__all__ = [
    "DEFAULT_ACTIVATIONS",
    "DEFAULT_LN_EPS",
    "DEFAULT_TIE",
    "DEFAULT_TYPE_VOCAB",
    "KNOWN_ACTIVATIONS",
    "architecture_of",
    "card_from_snapshot",
    "check_snapshot_is_a_model",
    "describe",
    "ffn_of",
    "missing_keys",
    "parameter_count",
    "parse_config",
    "profile_agreement",
    "reference_cards",
    "tie_delta",
    "tiny_card",
    "untied_card",
]
