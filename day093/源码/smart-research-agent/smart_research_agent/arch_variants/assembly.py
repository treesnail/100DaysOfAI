"""``assembly.py``：把三个变体**生成**成 PyTorch 脚本，并把那段文本解析回结构（day082）.

与前几课同一条纪律：**包里不 import torch**——

```text
包里的代码     纯 Python（逐位可复核），与 day073~081 一样
生成的脚本     ``variant_script(shape, variant)`` 产出一段**完整可运行**的 PyTorch 代码
结构核对       ``assembly_facts(script)`` 用 ast 把那段文本解析回类名 / nn 模块 / CONFIG
本机实测       在有 torch 的机器上真的跑一遍，把真实参数量写进教程（本章末）
```

## 一、为什么生成的脚本值得存在

前几课的包都是“从零实现”，而这一课要回答的是**“工业实现长什么样”**。
三行对应关系足以说明这件事：

```text
本包                                  PyTorch
encoder_block（pre-LN）               nn.TransformerEncoderLayer(norm_first=True)
decoder_block（九个阶段）              nn.TransformerDecoderLayer（自注意力 + 交叉 + 前馈）
一张显式掩码                           src_mask / tgt_mask / src_key_padding_mask
```

于是“BERT / GPT / T5”这三个名字在工业代码里的样子就是三行：

```python
nn.TransformerEncoder(layer, num_layers)                      # BERT
nn.ModuleList([layer for _ in range(num_layers)]) + 上三角掩码  # GPT
nn.TransformerEncoder(...) + nn.TransformerDecoder(...)        # T5
```

## 二、差异必须被看见，而不是被凑平

PyTorch 的参数量与本包的解析式**不相等**，原因有三条（都是真实的、可枚举的）：

```text
① nn.MultiheadAttention 的四个投影**带偏置**      本包不带（day075 的口径）
② nn.Linear 的前馈带偏置                          本包只有 ffn_b_in / ffn_b_out（一致）
③ nn.Embedding 与位置编码是额外的参数              本包的前向直接吃 (n, d) 的输入
```

本课**不把它们凑平**：生成的脚本会打印自己的参数量，而教程把差值写在旁边
（day080 对 ``4d × N`` 那处差值做的就是这件事）。
"""

from __future__ import annotations

import ast
from typing import Any

from smart_research_agent.arch_variants.errors import ParameterError
from smart_research_agent.arch_variants.types import (
    DEFAULT_LAYERS,
    VARIANT_DECODER_ONLY,
    VARIANT_ENCODER_DECODER,
    VARIANT_ENCODER_ONLY,
    VariantShape,
    validate_variant,
)

#: 生成脚本的运行要求（**写进脚本文本里**：一段没有 torch 就跑不了的代码应当自己说清）.
ASSEMBLY_REQUIREMENT = "需要 PyTorch >= 1.9（nn.TransformerEncoderLayer 的 batch_first 与 norm_first）"

#: 每个变体生成的类名（**这张表要被测试逐键检查**）.
ASSEMBLY_CLASSES: dict[str, str] = {
    VARIANT_ENCODER_ONLY: "EncoderOnlyStack",
    VARIANT_DECODER_ONLY: "DecoderOnlyStack",
    VARIANT_ENCODER_DECODER: "EncoderDecoderStack",
}

#: 每个变体用到的 nn 模块（**顺序即“从输入到输出”的顺序**）.
ASSEMBLY_MODULES: dict[str, tuple[str, ...]] = {
    VARIANT_ENCODER_ONLY: (
        "nn.Embedding",
        "nn.TransformerEncoderLayer",
        "nn.TransformerEncoder",
    ),
    VARIANT_DECODER_ONLY: (
        "nn.Embedding",
        "nn.TransformerEncoderLayer",
        "nn.ModuleList",
    ),
    VARIANT_ENCODER_DECODER: (
        "nn.Embedding",
        "nn.TransformerEncoderLayer",
        "nn.TransformerEncoder",
        "nn.TransformerDecoderLayer",
        "nn.TransformerDecoder",
    ),
}

#: CONFIG 的键（**生成侧与解析侧共用同一张表**：少一个键的脚本会被解析侧认出来）.
ASSEMBLY_CONFIG_KEYS: tuple[str, ...] = (
    "tokens",
    "sources",
    "hidden",
    "ffn",
    "layers",
    "heads",
    "vocab",
    "norm_first",
)

#: 掩码在 PyTorch 里的名字（三个变体各自用到哪些）.
ASSEMBLY_MASKS: dict[str, tuple[str, ...]] = {
    VARIANT_ENCODER_ONLY: ("src_key_padding_mask",),
    VARIANT_DECODER_ONLY: ("src_mask",),
    VARIANT_ENCODER_DECODER: ("src_key_padding_mask", "tgt_mask"),
}


def variant_script(
    shape: VariantShape,
    variant: str,
    *,
    heads: int = 1,
    vocab: int = 32,
    norm_first: bool = True,
) -> str:
    """生成一段**完整可运行**的 PyTorch 脚本（它自己打印参数量）.

    ``norm_first=True`` 对应本包的 ``placement="pre"``——
    这个对应关系不是巧合：PyTorch 的 ``norm_first`` 就是 pre-LN 那个开关。
    """
    if not isinstance(shape, VariantShape):
        raise ParameterError(f"shape 必须是 VariantShape，收到 {type(shape).__name__}。")
    resolved_variant = validate_variant(variant)
    if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
        raise ParameterError(f"heads 必须是 >= 1 的整数，收到 {heads!r}。")
    if shape.hidden % heads != 0:
        raise ParameterError(
            f"隐藏维 {shape.hidden} 不能被 heads={heads} 整除：多头把 d 切成等份，"
            "因此每一份的长度必须相等。"
        )
    if isinstance(vocab, bool) or not isinstance(vocab, int) or vocab < 2:
        raise ParameterError(f"vocab 必须是 >= 2 的整数，收到 {vocab!r}。")
    if not isinstance(norm_first, bool):
        raise ParameterError(f"norm_first 必须是 bool，收到 {norm_first!r}。")
    config_repr = "\n".join(
        [
            f'    "tokens": {shape.tokens},',
            f'    "sources": {shape.source_length},',
            f'    "hidden": {shape.hidden},',
            f'    "ffn": {shape.ffn},',
            f'    "layers": {shape.layers},',
            f'    "heads": {heads},',
            f'    "vocab": {vocab},',
            f'    "norm_first": {norm_first},',
        ]
    )
    header = (
        '"""由 day082 的 ``arch_variants.assembly`` 生成：三个变体的 PyTorch 对照实现.\n\n'
        f"{ASSEMBLY_REQUIREMENT}\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
        "import torch\n"
        "from torch import nn\n\n"
        "CONFIG = {\n"
        f"{config_repr}\n"
        "}\n\n"
    )
    body = _VARIANT_BODIES[resolved_variant]
    return header + body


_ENCODER_ONLY_BODY = '''
def causal_mask(size: int) -> torch.Tensor:
    """上三角掩码：``True`` 表示**不许看**（PyTorch 的约定与 day073 的掩码相反）."""
    return torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)


class EncoderOnlyStack(nn.Module):
    """BERT 一类：只有编码器，自注意力**全开**（每一行都能看到所有位置）."""

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config["vocab"], config["hidden"])
        layer = nn.TransformerEncoderLayer(
            d_model=config["hidden"],
            nhead=config["heads"],
            dim_feedforward=config["ffn"],
            batch_first=True,
            activation="relu",
            norm_first=config["norm_first"],
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config["layers"])

    def forward(self, tokens, padding_mask=None):
        """注意：**不加**因果掩码，只可能加填充掩码."""
        return self.encoder(
            self.embedding(tokens), src_key_padding_mask=padding_mask
        )


def main() -> int:
    """建一个模型、打印参数量（**差值要被看见**，而不是被凑平）."""
    model = EncoderOnlyStack(CONFIG)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"PARAM_COUNT {total}")
    with torch.no_grad():
        model(torch.zeros(2, CONFIG["tokens"], dtype=torch.long))
    return total


if __name__ == "__main__":
    main()
'''

_DECODER_ONLY_BODY = '''
def causal_mask(size: int) -> torch.Tensor:
    """上三角掩码：``True`` 表示**不许看**（PyTorch 的约定与 day073 的掩码相反）."""
    return torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)


class DecoderOnlyStack(nn.Module):
    """GPT 一类：只有解码器，自注意力**因果**（第 i 个位置只看 j <= i）.

    工业实现里它常常就是"一层 ``TransformerEncoderLayer`` + 一张上三角掩码"——
    与本课那句"因果不是一条新代码路径，而是一张掩码"逐字对应。
    """

    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config["vocab"], config["hidden"])
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=config["hidden"],
                    nhead=config["heads"],
                    dim_feedforward=config["ffn"],
                    batch_first=True,
                    activation="relu",
                    norm_first=config["norm_first"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, tokens):
        hidden = self.embedding(tokens)
        mask = causal_mask(hidden.shape[1])
        for block in self.blocks:
            hidden = block(hidden, src_mask=mask)
        return hidden


def main() -> int:
    model = DecoderOnlyStack(CONFIG)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"PARAM_COUNT {total}")
    with torch.no_grad():
        model(torch.zeros(2, CONFIG["tokens"], dtype=torch.long))
    return total


if __name__ == "__main__":
    main()
'''

_ENCODER_DECODER_BODY = '''
def causal_mask(size: int) -> torch.Tensor:
    """上三角掩码：``True`` 表示**不许看**（PyTorch 的约定与 day073 的掩码相反）."""
    return torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)


class EncoderDecoderStack(nn.Module):
    """T5 一类：编码器全开 + 解码器因果 + 交叉注意力（**同一个解码器里两个都占**）.

    ``tgt_mask`` 只加在**自注意力**那一步；交叉注意力那一步**没有**掩码——
    这与 day079 "交叉注意力不能被赋因果掩码"那条拒绝是同一件事。
    """

    def __init__(self, config):
        super().__init__()
        self.source_embedding = nn.Embedding(config["vocab"], config["hidden"])
        self.target_embedding = nn.Embedding(config["vocab"], config["hidden"])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["hidden"],
            nhead=config["heads"],
            dim_feedforward=config["ffn"],
            batch_first=True,
            activation="relu",
            norm_first=config["norm_first"],
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["layers"])
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config["hidden"],
            nhead=config["heads"],
            dim_feedforward=config["ffn"],
            batch_first=True,
            activation="relu",
            norm_first=config["norm_first"],
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config["layers"])

    def forward(self, source, target, source_padding_mask=None):
        memory = self.encoder(
            self.source_embedding(source), src_key_padding_mask=source_padding_mask
        )
        mask = causal_mask(target.shape[1])
        return self.decoder(
            self.target_embedding(target), memory, tgt_mask=mask
        )


def main() -> int:
    model = EncoderDecoderStack(CONFIG)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"PARAM_COUNT {total}")
    with torch.no_grad():
        model(
            torch.zeros(2, CONFIG["sources"], dtype=torch.long),
            torch.zeros(2, CONFIG["tokens"], dtype=torch.long),
        )
    return total


if __name__ == "__main__":
    main()
'''

_VARIANT_BODIES: dict[str, str] = {
    VARIANT_ENCODER_ONLY: _ENCODER_ONLY_BODY,
    VARIANT_DECODER_ONLY: _DECODER_ONLY_BODY,
    VARIANT_ENCODER_DECODER: _ENCODER_DECODER_BODY,
}


def parse_script(script: str) -> ast.Module:
    """把生成的脚本解析成语法树（**语法错当场抛**：一段跑不起来的代码不该被当成产物）."""
    if not isinstance(script, str):
        raise ParameterError(f"script 必须是字符串，收到 {type(script).__name__}。")
    if not script.strip():
        raise ParameterError("script 不能为空。")
    return ast.parse(script)


def class_names(script: str) -> tuple[str, ...]:
    """脚本里定义的类名（按出现顺序）."""
    return tuple(
        node.name
        for node in parse_script(script).body
        if isinstance(node, ast.ClassDef)
    )


def module_names(script: str) -> tuple[str, ...]:
    """脚本里用到的 ``nn.XXX`` 名字（按出现顺序、去重）."""
    found: list[str] = []
    for node in ast.walk(parse_script(script)):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "nn":
                name = f"nn.{node.attr}"
                if name not in found:
                    found.append(name)
    return tuple(found)


def parse_config(script: str) -> dict[str, Any]:
    """把脚本里的 ``CONFIG`` 解析回字典（**用 ast.literal_eval，不执行那段代码**）."""
    for node in parse_script(script).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CONFIG":
                    return dict(ast.literal_eval(node.value))
    raise ParameterError("脚本里没有 CONFIG：解析侧要靠它核对规模，而不是靠读注释。")


def function_names(script: str) -> tuple[str, ...]:
    """脚本里定义的函数名（``main`` 与 ``causal_mask`` 都在里面）."""
    return tuple(
        node.name
        for node in parse_script(script).body
        if isinstance(node, ast.FunctionDef)
    )


def assembly_facts(script: str) -> dict[str, Any]:
    """把一段脚本解析回“它是什么”：类名、nn 模块、CONFIG、是否带 main.

    本函数**不执行**脚本（``ast.literal_eval`` 只认字面量），
    因此它可以在任何环境下跑——这正是“本机没有 torch 也能核对结构”的实现。
    """
    config = parse_config(script)
    missing = [key for key in ASSEMBLY_CONFIG_KEYS if key not in config]
    if missing:
        raise ParameterError(
            f"CONFIG 缺键 {missing}：生成侧与解析侧共用同一张表（ASSEMBLY_CONFIG_KEYS），"
            "因此少一个键会被当场认出来。"
        )
    return {
        "classes": class_names(script),
        "functions": function_names(script),
        "modules": module_names(script),
        "config": config,
        "docstring": (ast.get_docstring(parse_script(script)) or ""),
    }


def script_requires_torch(script: str) -> bool:
    """这段脚本是不是真的 import 了 torch（**不要靠注释判断**）."""
    for node in ast.walk(parse_script(script)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch":
                    return True
        elif isinstance(node, ast.ImportFrom) and node.module == "torch":
            return True
    return False


def config_matches_shape(script: str, shape: VariantShape) -> bool:
    """脚本里的 CONFIG 与形状是否一致（**两侧口径同源**，因此这条断言有意义）."""
    config = parse_config(script)
    return (
        config["tokens"] == shape.tokens
        and config["sources"] == shape.source_length
        and config["hidden"] == shape.hidden
        and config["ffn"] == shape.ffn
        and config["layers"] == shape.layers
    )


def facts_match_variant(script: str, variant: str) -> bool:
    """脚本的类名与 nn 模块是否与变体相符（第 7 章那张对照表就靠它）."""
    resolved = validate_variant(variant)
    facts = assembly_facts(script)
    expected_class = ASSEMBLY_CLASSES[resolved]
    if facts["classes"] != (expected_class,):
        return False
    for module in ASSEMBLY_MODULES[resolved]:
        if module not in facts["modules"]:
            return False
    return True


def all_scripts(shape: VariantShape | None = None, **kwargs: Any) -> dict[str, str]:
    """三个变体各生成一段脚本（实验脚本与测试都用它）."""
    resolved = shape if shape is not None else VariantShape(layers=DEFAULT_LAYERS)
    return {
        variant: variant_script(resolved, variant, **kwargs) for variant in ASSEMBLY_CLASSES
    }


__all__ = [
    "ASSEMBLY_CLASSES",
    "ASSEMBLY_CONFIG_KEYS",
    "ASSEMBLY_MASKS",
    "ASSEMBLY_MODULES",
    "ASSEMBLY_REQUIREMENT",
    "all_scripts",
    "assembly_facts",
    "class_names",
    "config_matches_shape",
    "facts_match_variant",
    "function_names",
    "module_names",
    "parse_config",
    "parse_script",
    "script_requires_torch",
    "variant_script",
]
