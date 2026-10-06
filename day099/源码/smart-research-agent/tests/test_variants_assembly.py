"""``arch_variants.assembly`` 的测试：生成 PyTorch 脚本并**用 ast 解析回结构**（day082）."""

from __future__ import annotations

import ast

import pytest

from smart_research_agent.arch_variants import assembly
from smart_research_agent.arch_variants.errors import ParameterError
from smart_research_agent.arch_variants.types import (
    VARIANTS,
    VARIANT_DECODER_ONLY,
    VARIANT_ENCODER_DECODER,
    VARIANT_ENCODER_ONLY,
)
from tests import arch_variants_samples as samples


class TestGeneration:
    """三个变体各生成一段脚本，而它们必须**语法合法**."""

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_script_is_valid_python(self, variant: str) -> None:
        script = assembly.variant_script(samples.sample_shape(), variant)
        assert isinstance(ast.parse(script), ast.Module)

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_script_defines_exactly_one_class(self, variant: str) -> None:
        script = assembly.variant_script(samples.sample_shape(), variant)
        assert assembly.class_names(script) == (assembly.ASSEMBLY_CLASSES[variant],)

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_script_declares_the_torch_requirement(self, variant: str) -> None:
        script = assembly.variant_script(samples.sample_shape(), variant)
        assert assembly.script_requires_torch(script)
        assert assembly.ASSEMBLY_REQUIREMENT in script

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_script_has_a_main(self, variant: str) -> None:
        script = assembly.variant_script(samples.sample_shape(), variant)
        assert "main" in assembly.function_names(script)
        assert "PARAM_COUNT" in script

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_config_matches_the_shape(self, variant: str) -> None:
        shape = samples.sample_shape()
        script = assembly.variant_script(shape, variant)
        assert assembly.config_matches_shape(script, shape)

    @pytest.mark.parametrize("variant", VARIANTS)
    def test_facts_match_the_variant(self, variant: str) -> None:
        script = assembly.variant_script(samples.sample_shape(), variant)
        assert assembly.facts_match_variant(script, variant)

    def test_encoder_only_uses_the_encoder_stack(self) -> None:
        script = assembly.variant_script(samples.sample_shape(), VARIANT_ENCODER_ONLY)
        modules = assembly.module_names(script)
        assert "nn.TransformerEncoder" in modules
        assert "nn.TransformerDecoder" not in modules

    def test_decoder_only_uses_a_module_list(self) -> None:
        """GPT 一类的工业写法：一层 + 一张上三角掩码，而不是另一个类."""
        script = assembly.variant_script(samples.sample_shape(), VARIANT_DECODER_ONLY)
        modules = assembly.module_names(script)
        assert "nn.ModuleList" in modules
        assert "nn.TransformerDecoder" not in modules
        assert "src_mask=mask" in script

    def test_encoder_decoder_uses_both_stacks(self) -> None:
        script = assembly.variant_script(samples.sample_shape(), VARIANT_ENCODER_DECODER)
        modules = assembly.module_names(script)
        assert "nn.TransformerEncoder" in modules
        assert "nn.TransformerDecoder" in modules
        assert "tgt_mask=mask" in script

    def test_scripts_differ_between_variants(self) -> None:
        scripts = assembly.all_scripts(samples.sample_shape())
        assert len(scripts) == 3
        assert len(set(scripts.values())) == 3

    def test_norm_first_follows_the_placement(self) -> None:
        """``norm_first=True`` 就是本包的 ``placement="pre"``——两边口径同源."""
        pre = assembly.variant_script(samples.sample_shape(), VARIANT_ENCODER_ONLY)
        post = assembly.variant_script(
            samples.sample_shape(), VARIANT_ENCODER_ONLY, norm_first=False
        )
        assert assembly.parse_config(pre)["norm_first"] is True
        assert assembly.parse_config(post)["norm_first"] is False

    def test_heads_can_be_more_than_one(self) -> None:
        script = assembly.variant_script(samples.sample_shape(), VARIANT_ENCODER_ONLY, heads=3)
        assert assembly.parse_config(script)["heads"] == 3

    def test_vocab_is_configurable(self) -> None:
        script = assembly.variant_script(samples.sample_shape(), VARIANT_ENCODER_ONLY, vocab=8)
        assert assembly.parse_config(script)["vocab"] == 8


class TestGenerationErrors:
    """生成侧的拒绝（都在“写出一段跑不起来的代码”之前）."""

    def test_shape_type_is_checked(self) -> None:
        with pytest.raises(ParameterError):
            assembly.variant_script("shape", VARIANT_ENCODER_ONLY)  # type: ignore[arg-type]

    def test_variant_is_checked(self) -> None:
        with pytest.raises(ParameterError):
            assembly.variant_script(samples.sample_shape(), "bert")

    @pytest.mark.parametrize("heads", [0, -1, 1.5, True])
    def test_bad_heads_are_rejected(self, heads: object) -> None:
        with pytest.raises(ParameterError):
            assembly.variant_script(
                samples.sample_shape(), VARIANT_ENCODER_ONLY, heads=heads  # type: ignore[arg-type]
            )

    def test_heads_must_divide_the_hidden_dimension(self) -> None:
        with pytest.raises(ParameterError):
            assembly.variant_script(samples.sample_shape(), VARIANT_ENCODER_ONLY, heads=4)

    def test_vocab_must_be_at_least_two(self) -> None:
        with pytest.raises(ParameterError):
            assembly.variant_script(samples.sample_shape(), VARIANT_ENCODER_ONLY, vocab=1)

    def test_norm_first_must_be_bool(self) -> None:
        with pytest.raises(ParameterError):
            assembly.variant_script(
                samples.sample_shape(), VARIANT_ENCODER_ONLY, norm_first=1  # type: ignore[arg-type]
            )


class TestParsing:
    """解析侧：**它不执行脚本**（``ast.literal_eval`` 只认字面量）."""

    def test_parse_script_rejects_non_string(self) -> None:
        with pytest.raises(ParameterError):
            assembly.parse_script(123)  # type: ignore[arg-type]

    def test_parse_script_rejects_empty_text(self) -> None:
        with pytest.raises(ParameterError):
            assembly.parse_script("   ")

    def test_parse_script_rejects_syntax_errors(self) -> None:
        with pytest.raises(SyntaxError):
            assembly.parse_script("def broken(:\n")

    def test_missing_config_is_rejected(self) -> None:
        with pytest.raises(ParameterError):
            assembly.parse_config("import torch\n")

    def test_missing_config_key_is_rejected(self) -> None:
        script = "CONFIG = {'tokens': 4}\n"
        with pytest.raises(ParameterError):
            assembly.assembly_facts(script)

    def test_script_without_torch_reports_false(self) -> None:
        assert not assembly.script_requires_torch("CONFIG = {'tokens': 4}\n")

    def test_import_from_torch_also_counts(self) -> None:
        assert assembly.script_requires_torch("from torch import nn\n")

    def test_assembly_facts_collects_the_docstring(self) -> None:
        facts = assembly.assembly_facts(
            assembly.variant_script(samples.sample_shape(), VARIANT_ENCODER_ONLY)
        )
        assert "day082" in facts["docstring"]

    def test_assembly_facts_of_every_variant_has_the_expected_modules(self) -> None:
        for variant in VARIANTS:
            script = assembly.variant_script(samples.sample_shape(), variant)
            facts = assembly.assembly_facts(script)
            for module in assembly.ASSEMBLY_MODULES[variant]:
                assert module in facts["modules"]

    def test_config_mismatch_is_detected(self) -> None:
        script = assembly.variant_script(
            samples.sample_shape(tokens=4), VARIANT_ENCODER_ONLY
        )
        assert not assembly.config_matches_shape(script, samples.sample_shape(tokens=5))

    def test_facts_match_variant_detects_a_wrong_class(self) -> None:
        script = assembly.variant_script(samples.sample_shape(), VARIANT_ENCODER_ONLY)
        assert not assembly.facts_match_variant(script, VARIANT_DECODER_ONLY)

    def test_facts_match_variant_checks_the_name(self) -> None:
        script = assembly.variant_script(samples.sample_shape(), VARIANT_ENCODER_ONLY)
        with pytest.raises(ParameterError):
            assembly.facts_match_variant(script, "bert")


class TestTables:
    """两张表必须闭合（**少一个键 = 一个变体没有对照实现**）."""

    def test_class_table_covers_every_variant(self) -> None:
        assert set(assembly.ASSEMBLY_CLASSES) == set(VARIANTS)

    def test_module_table_covers_every_variant(self) -> None:
        assert set(assembly.ASSEMBLY_MODULES) == set(VARIANTS)

    def test_mask_table_covers_every_variant(self) -> None:
        assert set(assembly.ASSEMBLY_MASKS) == set(VARIANTS)

    def test_mask_table_matches_the_variant_semantics(self) -> None:
        assert assembly.ASSEMBLY_MASKS[VARIANT_DECODER_ONLY] == ("src_mask",)
        assert "tgt_mask" in assembly.ASSEMBLY_MASKS[VARIANT_ENCODER_DECODER]

    def test_config_keys_are_the_protocol(self) -> None:
        script = assembly.variant_script(samples.sample_shape(), VARIANT_ENCODER_ONLY)
        config = assembly.parse_config(script)
        assert set(config) == set(assembly.ASSEMBLY_CONFIG_KEYS)

    def test_all_scripts_uses_the_default_shape(self) -> None:
        assert set(assembly.all_scripts()) == set(VARIANTS)
