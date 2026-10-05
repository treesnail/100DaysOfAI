"""与**真实** ``transformers`` / ``torch`` 的对账（day086 / M7-D10）.

本文件与其余 ``test_hf_integration_*`` 的分工是刻意的：

```text
其余文件      纯 Python、零外部依赖 ⇒ 任何环境都能跑（CI 只装 requirements.txt 也能过）
本文件        用**真库**复核本包的几个整数与默认值 ⇒ 没装 transformers/torch 时整份跳过
```

为什么值得单独留一份：本包的核心断言里有三条是"与库一致"类型——

```text
① 参数量公式          config 里那几个整数算出来的总量 == 真模型 ``sum(p.numel())``
② 配置默认值          四个默认值（两个 eps、两个激活）与真 ``*Config()`` 逐个相同
③ 共享词嵌入的默认值   ``tie_word_embeddings`` 的默认值是真库里的那个（两个架构都是 True）
```

没有这一份，"与库一致"就只是一句记录在案的话（它能被反驳的方式只剩"改那个常量"）。
有这一份，它每次跑都在跟真库对一次。**模型都很小**（``vocab=277, hidden=16``），
因此这一份的运行时间是毫秒级。
"""

from __future__ import annotations

import pytest

from smart_research_agent.hf_integration.config import parse_config
from smart_research_agent.hf_integration.types import (
    ARCHITECTURE_BERT,
    ARCHITECTURE_GPT2,
    DEFAULT_REVISION,
    SIZE_KEYS,
)

from tests.hf_integration_samples import config_payload

transformers = pytest.importorskip("transformers", reason="本份对账需要真实 transformers")
torch = pytest.importorskip("torch", reason="本份对账需要真实 torch")


def _real_parameters(model: object) -> int:
    """真模型的参数量（``sum(p.numel())``，与 ``hf_integration`` 的口径同一件事）."""
    return sum(parameter.numel() for parameter in model.parameters())  # type: ignore[attr-defined]


def test_tiny_gpt2_and_bert_parameter_counts_match_the_library() -> None:
    """三张卡片的参数量与真库**整数相等**（GPT-2 共享/不共享、BERT 主干）.

    BERT 那一侧用的是 ``BertModel``：它的参数量是"主干 + pooler"，
    而本包的公式在 ``tie_word_embeddings=True`` 时不加输出投影——
    两者正是同一件事（真实的 bert-base-uncased 就是 109 482 240 那个数）。
    """
    gpt2_payload = config_payload(ARCHITECTURE_GPT2, 277, tie_word_embeddings=True)
    gpt2_untied = config_payload(ARCHITECTURE_GPT2, 277, tie_word_embeddings=False)
    bert_payload = config_payload(ARCHITECTURE_BERT, 277, tie_word_embeddings=True)

    tied = transformers.GPT2LMHeadModel(transformers.GPT2Config(**gpt2_payload))
    untied = transformers.GPT2LMHeadModel(transformers.GPT2Config(**gpt2_untied))
    backbone = transformers.BertModel(transformers.BertConfig(**bert_payload))

    assert parse_config(gpt2_payload).parameter_count == _real_parameters(tied)
    assert parse_config(gpt2_untied).parameter_count == _real_parameters(untied)
    assert parse_config(bert_payload).parameter_count == _real_parameters(backbone)


def test_tying_actually_shares_the_same_tensor_in_the_library() -> None:
    """真库里"共享"就是**同一个张量**（本包的 ``head is word`` 是同一件事）."""
    payload = config_payload(ARCHITECTURE_GPT2, 277, tie_word_embeddings=True)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(**payload))
    assert model.lm_head.weight is model.transformer.wte.weight


@pytest.mark.parametrize(
    ("model_type", "config_class", "activation_key", "eps_key"),
    [
        (ARCHITECTURE_GPT2, "GPT2Config", "activation_function", "layer_norm_epsilon"),
        (ARCHITECTURE_BERT, "BertConfig", "hidden_act", "layer_norm_eps"),
    ],
)
def test_config_defaults_match_the_library(
    model_type: str, config_class: str, activation_key: str, eps_key: str
) -> None:
    """三个默认值逐个与真 ``*Config()`` 相同：激活、eps、以及**共享词嵌入**.

    这条对账复核的是 day085 从源码读到、今天又从 ``config.json`` 读到的那几条默认值——
    而它现在有第三个来源：**真库自己**。
    刻意给一份**最小配置**（只有必需键）：这样"默认值"这件事才真的走到了缺省路径。
    """
    minimal = {
        "model_type": model_type,
        "vocab_size": 64,
        "hidden_size": 16,
        "num_attention_heads": 2,
        "num_hidden_layers": 1,
        "max_position_embeddings": 16,
        "n_embd": 16,
        "n_head": 2,
        "n_layer": 1,
        "n_positions": 16,
        "type_vocab_size": 2,
    }
    real = getattr(transformers, config_class)(**minimal)
    card = parse_config(minimal, name="defaults")
    assert card.activation == getattr(real, activation_key)
    assert card.ln_eps == getattr(real, eps_key)
    assert real.tie_word_embeddings is True
    assert card.tie_word_embeddings is True
    assert card.uses_token_type is (model_type == ARCHITECTURE_BERT)


def test_forward_dimension_defaults_are_different_between_the_two_architectures() -> None:
    """前馈中间维的缺省：GPT-2 是 ``None``（按 4·hidden 推），BERT 是**固定的 3072**.

    这是本课与真库对账时冒出来的一条**看起来一样、其实不一样**。
    它值一个测试，因为"两个架构的前馈都是 4 倍"这句话在 bert-base 上恰好成立。
    """
    from smart_research_agent.hf_integration.types import BERT_DEFAULT_INTERMEDIATE

    gpt2_real = transformers.GPT2Config()
    bert_real = transformers.BertConfig()
    assert gpt2_real.n_inner is None
    assert bert_real.intermediate_size == BERT_DEFAULT_INTERMEDIATE == 3072
    bert_base = transformers.BertConfig(hidden_size=768, num_attention_heads=12)
    assert bert_base.intermediate_size == 3072 == 4 * 768  # 恰好相等，因此容易被误读成规律
    tiny = transformers.BertConfig(hidden_size=16, num_attention_heads=2)
    assert tiny.intermediate_size == 3072 != 4 * 16


def test_size_key_mapping_points_at_real_attributes() -> None:
    """本包那张"同一个量的两个键名"映射，在真 ``*Config`` 上也能对上.

    映射表里每个键都应当是真实配置类的一个属性名——写错一个字母时，
    本包会安静地取到默认值，而"取到默认值"与"这份配置没写"读起来一样。
    """
    for model_type, config_class in (
        (ARCHITECTURE_GPT2, "GPT2Config"),
        (ARCHITECTURE_BERT, "BertConfig"),
    ):
        real = getattr(transformers, config_class)()
        for name, attribute in SIZE_KEYS[model_type].items():
            assert hasattr(real, attribute), (name, attribute)


def test_head_count_divides_hidden_in_the_library_config_too() -> None:
    """真库对"头数不整除隐藏维"的态度：**构造时就报错**（与本包的 ``ConfigError`` 同一取向）."""
    payload = config_payload(ARCHITECTURE_GPT2, 64, n_head=5, n_embd=16)
    with pytest.raises(Exception):
        transformers.GPT2LMHeadModel(transformers.GPT2Config(**payload))
    with pytest.raises(Exception):
        parse_config(payload)


def test_byte_mapping_matches_the_library() -> None:
    """本包的 ``bytes_to_unicode`` 与真库的那个**逐键相同**.

    这是 byte-level BPE 的地基：256 个字节与 256 个可打印字符一一对应。
    映射错一个键，某一段字节就会切错——而它只在**含那个字节的文本**上出错，
    因此"随手试几条都正常"是可能的。
    """
    from smart_research_agent.hf_integration.tokenizer import bytes_to_unicode

    try:
        from transformers.models.gpt2.tokenization_gpt2 import (
            bytes_to_unicode as library_bytes_to_unicode,
        )
    except ImportError:  # pragma: no cover - 真库改名时跳过
        pytest.skip("这个版本的 transformers 里找不到 tokenization_gpt2.bytes_to_unicode")

    assert bytes_to_unicode() == library_bytes_to_unicode()


def test_versions_are_recorded_and_current() -> None:
    """记录在案的版本号与真库报出来的版本一致（"当时接的是哪一版"不是传说）."""
    from smart_research_agent.hf_integration.types import TRANSFORMERS_VERSION

    assert transformers.__version__ == TRANSFORMERS_VERSION
    assert DEFAULT_REVISION == "main"
    assert torch.__version__  # torch 只是被 importorskip 拉进来的，此处只证明它在
