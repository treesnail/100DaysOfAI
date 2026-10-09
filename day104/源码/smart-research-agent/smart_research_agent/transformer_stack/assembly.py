"""``transformer_stack`` 的 **PyTorch 组装脚本生成器**（day080 / M7-D5）.

## 为什么“生成一份脚本”而不是“在包里 import torch”

day001 起本项目就有一条纪律：**运行时依赖只有 ``requirements.txt`` 里那几个**。
``transformer_core`` / ``positional_encoding`` / ``encoder_decoder`` 全部是纯 Python 算术，
因此 100 天里的每一课都能在任意环境里导入、跑测试。

而今天的学习目标是“使用 PyTorch 组装 Multi-Head Attention、FFN、LayerNorm”。
两条要求放在一起，本课选了 day052（``sft/hf_script.py``）已经用过的那条路：

```text
包里的代码        纯 Python（逐位可复核），**不 import torch**
生成的脚本        一段可以独立运行的 PyTorch 代码（文本），由测试按结构逐项核对
本机实测          在有 torch 的机器上把那一段真的跑一遍，把真实输出写进教程
```

因此本模块只做两件事：

```text
assembly_script(shape, ...)  把形状编译成一段**完整可运行**的 PyTorch 脚本文本
assembly_facts(script)       把那段文本解析回结构（类名 / 用了哪些 nn 模块 / CONFIG）
```

## 脚本里最要紧的三行

```python
self.norm1 = nn.LayerNorm(CONFIG["hidden"], eps=CONFIG["epsilon"])
self.attention = nn.MultiheadAttention(CONFIG["hidden"], CONFIG["num_heads"], batch_first=True)
self.ffn = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(), nn.Linear(d_ff, d))
```

它们与 day079 的 ``layer_norm`` / ``self_attention`` / ``feed_forward`` **一一对应**，
而 ``eps`` 是同一个 ``1e-05``——这一条不是巧合，Hugging Face 的 ``LayerNorm``
默认值也是它（day085 会看到源码里的那一行）。

## 一个必须说清的差别：参数量对不上

本课手写的注意力**四个投影都没有偏置**（day075 的 ``AttentionParams`` 只有四块矩阵），
而 ``nn.MultiheadAttention`` 默认把 q/k/v 合并成 ``in_proj_weight`` 并**带** ``in_proj_bias``，
``out_proj`` 也带偏置。于是：

```text
本课解析式     4d²                     （四个 (d, d) 投影，无偏置）
nn.MultiheadAttention（bias=True）  4d² + 4d
```

生成的脚本会把两个数都打印出来，而教程里那一节讲的就是这 ``4d`` 与 ``4d`` 从哪来——
**差异必须被看见，而不是被凑平**。
"""

from __future__ import annotations

import ast
from typing import Any

from smart_research_agent.encoder_decoder.types import (
    ACTIVATION_GELU,
    ACTIVATION_RELU,
    DEFAULT_EPSILON,
    NORM_PLACEMENTS,
    NORM_PRE,
    _checked_activation,
    _checked_placement,
)
from smart_research_agent.transformer_stack.errors import ParameterError
from smart_research_agent.transformer_stack.types import StackShape

#: 生成脚本需要的第三方依赖（**只在跑那一段脚本时才需要**）.
ASSEMBLY_REQUIREMENT = "torch>=2.5"

#: 脚本里必须出现的类名（测试逐项核对）.
ASSEMBLY_CLASSES: tuple[str, ...] = ("TransformerBlock", "TransformerStack")

#: 脚本里必须用到的 ``nn`` 模块（少一个就说明那段组装少了一块）.
ASSEMBLY_MODULES: tuple[str, ...] = (
    "LayerNorm",
    "Linear",
    "ModuleList",
    "MultiheadAttention",
    "ReLU",
    "Sequential",
)

#: ``CONFIG`` 里必须有的键（``assembly_facts`` 逐键核对，与 ``StackShape`` 对齐）.
ASSEMBLY_CONFIG_KEYS: tuple[str, ...] = (
    "hidden",
    "ffn",
    "tokens",
    "layers",
    "num_heads",
    "epsilon",
    "placement",
    "activation",
    "batch",
    "analytic_layer_parameters",
    "analytic_total_parameters",
)

_BODY = '''

class TransformerBlock(nn.Module):
    """day079 那六个阶段在 PyTorch 里的写法（pre / post 只差 LN 的位置）."""

    def __init__(self) -> None:
        super().__init__()
        hidden = int(CONFIG["hidden"])
        ffn = int(CONFIG["ffn"])
        activation = str(CONFIG["activation"])
        self.norm1 = nn.LayerNorm(hidden, eps=float(CONFIG["epsilon"]))
        self.attention = nn.MultiheadAttention(
            hidden, int(CONFIG["num_heads"]), batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden, eps=float(CONFIG["epsilon"]))
        # 前馈：Linear → 激活 → Linear（逐位置，因此与序列长度无关）
        non_linearity = nn.ReLU() if activation == "relu" else nn.GELU()
        self.ffn = nn.Sequential(nn.Linear(hidden, ffn), non_linearity, nn.Linear(ffn, hidden))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if CONFIG["placement"] == "pre":
            normed = self.norm1(inputs)
            attended, _ = self.attention(normed, normed, normed, need_weights=False)
            residual = inputs + attended
            return residual + self.ffn(self.norm2(residual))
        attended, _ = self.attention(inputs, inputs, inputs, need_weights=False)
        residual = self.norm1(inputs + attended)
        return self.norm2(residual + self.ffn(residual))


class TransformerStack(nn.Module):
    """把一个块复制 N 份：``nn.ModuleList`` 就是本课的“链”. """

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock() for _ in range(int(CONFIG["layers"]))]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current = inputs
        for block in self.blocks:
            current = block(current)
        return current


def parameter_count(model: nn.Module) -> int:
    """可训练参数个数（与解析式对比时要看清口径：带不带偏置）."""
    return sum(item.numel() for item in model.parameters() if item.requires_grad)


def main() -> None:
    torch.manual_seed(0)
    model = TransformerStack()
    hidden = int(CONFIG["hidden"])
    inputs = torch.randn(int(CONFIG["batch"]), int(CONFIG["tokens"]), hidden)
    outputs = model(inputs)
    if tuple(outputs.shape) != tuple(inputs.shape):
        raise AssertionError("块保形 => 整条链也保形，输出与输入必须同形")
    measured = parameter_count(model)
    analytic = int(CONFIG["analytic_total_parameters"])
    print(f"输入 {tuple(inputs.shape)} -> 输出 {tuple(outputs.shape)}")
    print(f"可训练参数 {measured} 个（nn.MultiheadAttention 的四个投影**带**偏置）")
    print(f"本课解析式 {analytic} 个（四个 (d, d) 投影**不带**偏置）")
    print(f"差值 {measured - analytic} = {int(CONFIG['layers'])} 层 x 4d（每层 4d 个偏置）")


if __name__ == "__main__":
    main()
'''


def assembly_script(
    shape: StackShape,
    *,
    placement: str = NORM_PRE,
    activation: str = ACTIVATION_RELU,
    num_heads: int = 2,
    batch: int = 2,
) -> str:
    """把 ``StackShape`` 编译成一段完整可运行的 PyTorch 脚本（文本）.

    ``num_heads`` 必须整除 ``hidden``——否则 ``nn.MultiheadAttention``
    在**运行到前向时**才会报错，而那时已经离“写错一行”很远了。
    """
    resolved_placement = _checked_placement(placement)
    resolved_activation = _checked_activation(activation)
    if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads < 1:
        raise ParameterError(f"num_heads 必须是 >= 1 的整数，收到 {num_heads!r}。")
    if shape.hidden % num_heads != 0:
        raise ParameterError(
            f"隐藏维 {shape.hidden} 不能被 num_heads={num_heads} 整除："
            "多头要求把隐藏维等分，切不匀会在前向时才炸。"
        )
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ParameterError(f"batch 必须是 >= 1 的整数，收到 {batch!r}。")
    config: dict[str, Any] = {
        "hidden": shape.hidden,
        "ffn": shape.ffn,
        "tokens": shape.tokens,
        "layers": shape.layers,
        "num_heads": num_heads,
        "epsilon": DEFAULT_EPSILON,
        "placement": resolved_placement,
        "activation": resolved_activation,
        "batch": batch,
        "analytic_layer_parameters": shape.layer_parameter_count,
        "analytic_total_parameters": shape.total_parameter_count,
    }
    header = (
        '"""由 day080 的 ``transformer_stack`` 生成的 PyTorch 组装脚本。\n'
        "\n"
        "它把一个编码器块组装出来并复制 N 份，验证前向的形状。原文没有任何手工数字：\n"
        "所有的 d / d_ff / n / N / eps 都来自 ``CONFIG``，而那本字典来自 ``StackShape``。\n"
        "\n"
        f'依赖：pip install "{ASSEMBLY_REQUIREMENT}"（本包本身不 import torch）。\n'
        '"""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "import torch\n"
        "from torch import nn\n"
        "\n"
        f"CONFIG = {config!r}\n"
    )
    return header + _BODY


def assembly_facts(script: str) -> dict[str, Any]:
    """把生成的脚本解析回结构：类名、用到的 ``nn`` 模块、``CONFIG``、还有行数.

    解析失败（语法错）**当场报错**：一段连 ``ast.parse`` 都过不去的脚本
    在被运行之前就已经没有意义了，而“文本里有那三个字母”这种检查
    （``"LayerNorm" in script``）会放过它。
    """
    if not isinstance(script, str) or not script.strip():
        raise ParameterError("script 必须是非空字符串。")
    tree = ast.parse(script)
    classes: list[str] = []
    modules: set[str] = set()
    config: dict[str, Any] | None = None
    has_main = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node.name)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "nn":
                modules.add(node.attr)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CONFIG":
                    config = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            has_main = True
    if config is None:
        raise ParameterError("脚本里找不到 ``CONFIG``：没有它就无法逐键核对形状。")
    return {
        "classes": tuple(classes),
        "modules": tuple(sorted(modules)),
        "config": dict(config),
        "has_main": has_main,
        "line_count": len(script.splitlines()),
    }


def activation_is_supported(activation: str) -> bool:
    """这个激活在生成的脚本里有没有对应实现（``relu`` / ``gelu`` 都有）."""
    return activation in (ACTIVATION_RELU, ACTIVATION_GELU)


def placement_is_supported(placement: str) -> bool:
    """这个摆放位置在生成的脚本里有没有对应分支（pre / post 都有）."""
    return placement in NORM_PLACEMENTS


__all__ = [
    "ASSEMBLY_CLASSES",
    "ASSEMBLY_CONFIG_KEYS",
    "ASSEMBLY_MODULES",
    "ASSEMBLY_REQUIREMENT",
    "activation_is_supported",
    "assembly_facts",
    "assembly_script",
    "placement_is_supported",
]
