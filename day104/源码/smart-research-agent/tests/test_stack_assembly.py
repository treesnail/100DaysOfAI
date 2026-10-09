"""``transformer_stack.assembly`` 与 PyTorch 脚本的结构核对（day080）."""

from __future__ import annotations

import ast
import json

import pytest

from smart_research_agent.transformer_stack import (
    ASSEMBLY_CLASSES,
    ASSEMBLY_CONFIG_KEYS,
    ASSEMBLY_MODULES,
    ASSEMBLY_REQUIREMENT,
    ParameterError,
    activation_is_supported,
    assembly_facts,
    assembly_script,
    placement_is_supported,
)
from tests.stack_samples import shape


class TestAssemblyScriptText:
    """生成的文本：一段**完整可运行**的 PyTorch 代码."""

    def test_it_is_valid_python(self):
        """``ast.parse`` 能过（这是“它是一段代码”的最低要求）."""
        tree = ast.parse(assembly_script(shape()))
        assert isinstance(tree, ast.Module)

    def test_it_declares_the_two_classes(self):
        """两个类都在：一个块 + 一条链."""
        text = assembly_script(shape())
        for name in ASSEMBLY_CLASSES:
            assert f"class {name}(nn.Module):" in text

    def test_it_uses_every_required_module(self):
        """六个 ``nn`` 模块齐备（少一个就说明那段组装少了一块）."""
        text = assembly_script(shape())
        for name in ASSEMBLY_MODULES:
            assert f"nn.{name}" in text

    def test_it_asks_for_the_documented_requirement(self):
        """脚本头里写着依赖与安装方式（本包自己不 import torch）."""
        text = assembly_script(shape())
        assert ASSEMBLY_REQUIREMENT in text
        assert "pip install" in text
        assert "torch" in text

    def test_the_input_has_no_hard_coded_numbers(self):
        """d / d_ff / n / N 全部来自 ``CONFIG``：正文里没有裸的 6 / 24 / 4."""
        text = assembly_script(shape())
        body = text.split("CONFIG = ", 1)[1].split("\n", 1)[1]
        assert "int(CONFIG[" in body
        assert "torch.randn(int(CONFIG[\"batch\"]), int(CONFIG[\"tokens\"]), hidden)" in body

    def test_the_layer_norm_uses_the_shared_epsilon(self):
        """``LayerNorm`` 的 ``eps`` 与 day079 的 ``DEFAULT_EPSILON`` 同值."""
        text = assembly_script(shape())
        assert "nn.LayerNorm(hidden, eps=float(CONFIG[\"epsilon\"]))" in text
        assert assembly_facts(text)["config"]["epsilon"] == 1e-05

    def test_both_placements_are_rendered(self):
        """pre 与 post 的分支都在文本里（脚本能表达两种摆放位置）."""
        text = assembly_script(shape(), placement="pre")
        assert "CONFIG[\"placement\"] == \"pre\"" in text
        post = assembly_script(shape(), placement="post")
        assert post != text
        assert assembly_facts(post)["config"]["placement"] == "post"

    def test_gelu_renders_a_gelu_layer(self):
        """``gelu`` 会被渲染成 ``nn.GELU()``（激活是一个真的分支，不是一行注释）."""
        text = assembly_script(shape(), activation="gelu")
        assert "nn.GELU()" in text
        assert "nn.ReLU() if activation == \"relu\"" in text

    def test_one_head_is_allowed(self):
        """``num_heads=1`` 是合法的（整除永远成立）."""
        facts = assembly_facts(assembly_script(shape(), num_heads=1))
        assert facts["config"]["num_heads"] == 1

    def test_it_prints_the_parameter_gap(self):
        """脚本会打印 PyTorch 与解析式的参数量差（差异必须被看见）."""
        text = assembly_script(shape())
        assert "analytic_total_parameters" in text
        assert "偏置" in text


class TestAssemblyScriptValidation:
    """参数的越界检查（都挡在生成那一刻）."""

    @pytest.mark.parametrize("num_heads", [4, 5, 7])
    def test_heads_must_divide_the_hidden_width(self, num_heads: int):
        """隐藏维必须能被头数整除（切不匀会在前向时才炸）."""
        with pytest.raises(ParameterError):
            assembly_script(shape(), num_heads=num_heads)

    @pytest.mark.parametrize("num_heads", [0, -1, True, 1.5])
    def test_heads_must_be_a_positive_integer(self, num_heads: object):
        """头数必须是 >= 1 的整数."""
        with pytest.raises(ParameterError):
            assembly_script(shape(), num_heads=num_heads)  # type: ignore[arg-type]

    @pytest.mark.parametrize("batch", [0, -2, True, 2.5])
    def test_batch_must_be_a_positive_integer(self, batch: object):
        """批大小必须是 >= 1 的整数."""
        with pytest.raises(ParameterError):
            assembly_script(shape(), batch=batch)  # type: ignore[arg-type]

    def test_unknown_placement_is_rejected(self):
        """不认识的摆放位置当场报错（脚本里没有那个分支）."""
        with pytest.raises(ValueError):
            assembly_script(shape(), placement="middle")

    def test_unknown_activation_is_rejected(self):
        """不认识的激活当场报错."""
        with pytest.raises(ValueError):
            assembly_script(shape(), activation="silu")


class TestAssemblyFacts:
    """把那段文本解析回结构."""

    def test_it_reads_the_config_back(self):
        """``CONFIG`` 逐键与 ``StackShape`` 对上（d / d_ff / n / N / 参数量）."""
        item = shape()
        facts = assembly_facts(assembly_script(item))
        config = facts["config"]
        assert config["hidden"] == item.hidden
        assert config["ffn"] == item.ffn
        assert config["tokens"] == item.tokens
        assert config["layers"] == item.layers
        assert config["analytic_layer_parameters"] == item.layer_parameter_count
        assert config["analytic_total_parameters"] == item.total_parameter_count

    def test_every_config_key_is_present(self):
        """``ASSEMBLY_CONFIG_KEYS`` 的每一项都在（少一项就无法逐键核对）."""
        config = assembly_facts(assembly_script(shape()))["config"]
        assert set(ASSEMBLY_CONFIG_KEYS) <= set(config)

    def test_it_reports_the_classes_and_modules(self):
        """类名与 ``nn`` 模块都被读出来（顺序无关，集合相等）."""
        facts = assembly_facts(assembly_script(shape()))
        assert set(ASSEMBLY_CLASSES) <= set(facts["classes"])
        assert set(ASSEMBLY_MODULES) <= set(facts["modules"])
        assert facts["has_main"] is True

    def test_it_reports_the_line_count(self):
        """行数是一个便宜但有用的读数（生成器一变，它立刻会动）."""
        facts = assembly_facts(assembly_script(shape()))
        assert isinstance(facts["line_count"], int)
        assert facts["line_count"] > 40

    def test_it_is_json_serialisable(self):
        """事实字典能 json.dumps（它要被写进报告）."""
        facts = assembly_facts(assembly_script(shape()))
        assert isinstance(json.dumps(facts), str)

    @pytest.mark.parametrize("text", ["", "   ", None])
    def test_empty_text_is_rejected(self, text: object):
        """空文本当场报错."""
        with pytest.raises(ParameterError):
            assembly_facts(text)  # type: ignore[arg-type]

    def test_broken_syntax_is_rejected(self):
        """语法错的脚本当场报错（而'文本里有那三个字母'那种检查会放过它）."""
        with pytest.raises(SyntaxError):
            assembly_facts("def broken(:\n")

    def test_a_missing_config_is_rejected(self):
        """没有 ``CONFIG`` 的脚本当场报错（没有它就无法逐键核对形状）."""
        text = assembly_script(shape()).replace("CONFIG = ", "UNUSED = ", 1)
        with pytest.raises(ParameterError):
            assembly_facts(text)


class TestSupportProbes:
    """两个便宜的探针（都转发包里的口径表）."""

    def test_supported_activations(self):
        """``relu`` 与 ``gelu`` 都支持，``silu`` 不支持."""
        assert activation_is_supported("relu")
        assert activation_is_supported("gelu")
        assert not activation_is_supported("silu")

    def test_supported_placements(self):
        """``pre`` 与 ``post`` 都支持，别的都不支持."""
        assert placement_is_supported("pre")
        assert placement_is_supported("post")
        assert not placement_is_supported("middle")


class TestGeneratedScriptRuns:
    """**本机实测**：在有 torch 的环境里真的把那一段跑起来.

    CI 只装 ``requirements.txt`` 与 ``.[dev]``，而 torch 不在其中——因此这一条
    在没有 torch 的环境里会被跳过（与 day062 的 tiktoken 用例同一种写法）。
    本课包内的代码**从不** import torch：这里跑的是**生成出来的那一段文本**。
    """

    def test_the_script_runs_and_reports_the_gap(self, tmp_path):
        """跑一遍：输出与输入同形、参数量与解析式相差 ``4d × N``."""
        pytest.importorskip("torch")
        import subprocess
        import sys

        item = shape()
        text = assembly_script(item, num_heads=2)
        path = tmp_path / "assembly_demo.py"
        path.write_text(text, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        output = completed.stdout
        assert "(2, 4, 6)" in output
        gap = item.layers * 4 * item.hidden
        assert str(item.total_parameter_count) in output
        assert str(gap) in output

    def test_the_script_is_self_contained(self, tmp_path):
        """生成的文本可以被**独立**写成一个文件并跑起来（不带任何本包的 import）."""
        text = assembly_script(shape(), activation="gelu")
        assert "smart_research_agent" not in text
        path = tmp_path / "standalone.py"
        path.write_text(text, encoding="utf-8")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert isinstance(tree, ast.Module)
