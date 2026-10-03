"""``transformer_stack.verify`` 的六条性质与小工具（day080）."""

from __future__ import annotations

import json

import pytest

from smart_research_agent.transformer_stack import (
    PROPERTY_ASSEMBLY_CARRIES_SHAPE,
    PROPERTY_DETERMINISTIC,
    PROPERTY_IDENTITY_WHEN_BRANCHES_VANISH,
    PROPERTY_MATCHES_LOOP,
    PROPERTY_PARAMETER_COUNT_MATCHES,
    PROPERTY_SHAPE_PRESERVED,
    STACK_PROPERTIES,
    NumericError,
    ParameterError,
    PropertyOutcome,
    PropertyReport,
    StackParameters,
    assembly_script,
    census_of_layer,
    check_assembly_carries_shape,
    check_parameter_count_matches,
    check_properties,
    check_stack_identity_when_branches_vanishes,
    check_stack_is_deterministic,
    check_stack_matches_loop,
    check_stack_preserves_shape,
    checked_vector,
    frobenius_profile,
    make_samples,
    matrix_norm_of,
    parse_script_module_names,
    relative_gradient_change,
    stack_forward,
    stack_stage_lines,
    zero_attention,
    zero_branches,
)
from tests.stack_samples import HIDDEN, LAYERS, TOKENS, inputs, parameters, script, shape


def outcome_of(report: PropertyReport, name: str) -> PropertyOutcome:
    """按名字从报告里取一条（缺项时直接失败，而不是静默返回 None）."""
    for item in report.outcomes:
        if item.name == name:
            return item
    raise AssertionError(f"报告里没有 {name}")


class TestPreservesShape:
    """性质 1：逐层保形 + 链上传递逐位一致."""

    def test_it_passes_on_the_sample(self):
        """样本上通过，证据里带着“几层 / 几处传递”。"""
        forward = stack_forward(parameters(), inputs())
        result = check_stack_preserves_shape(forward)
        assert result.passed
        assert "4 层逐层同形" in result.evidence
        assert "3 处传递逐位一致" in result.evidence
        assert result.description

    def test_it_rejects_a_non_forward_record(self):
        """传进来不是 ``StackForward`` 时当场报错."""
        with pytest.raises(ParameterError):
            check_stack_preserves_shape("not-a-record")  # type: ignore[arg-type]

    def test_it_also_passes_for_a_single_layer(self):
        """一层的链没有“传递”那一段，判据也要成立（``max(depth-1, 0)``）."""
        forward = stack_forward(parameters(layers=1), inputs())
        assert check_stack_preserves_shape(forward).passed


class TestMatchesLoop:
    """性质 2：与手写逐层调用逐位一致（本课最便宜也最值钱的一条）."""

    def test_it_passes_on_the_sample(self):
        """样本上通过：4 层手写循环与 ``stack_forward`` 逐位相等."""
        params = parameters()
        forward = stack_forward(params, inputs())
        result = check_stack_matches_loop(params, inputs(), forward)
        assert result.passed
        assert "逐位相等" in result.evidence

    @pytest.mark.parametrize("placement", ["pre", "post"])
    def test_both_placements_pass(self, placement: str):
        """两种摆放位置都要过（两边必须走同一个 ``block_attention``）."""
        params = parameters()
        forward = stack_forward(params, inputs(), placement=placement)
        result = check_stack_matches_loop(params, inputs(), forward, placement=placement)
        assert result.passed

    def test_it_fails_if_the_parameters_are_shifted_by_one_layer(self):
        """**把参数错位一层**：判据必须亮红（这正是它要抓的那类错误）."""
        params = parameters(layers=3)
        forward = stack_forward(params, inputs())
        from smart_research_agent.transformer_stack import StackParameters

        shifted = StackParameters(blocks=params.blocks[1:], attentions=params.attentions[1:])
        result = check_stack_matches_loop(shifted, inputs(), forward)
        assert not result.passed
        assert "[!!]" in result.summary_line()

    def test_the_evidence_says_what_it_is_worth(self):
        """证据里必须写清“为什么这一条重要”（参数错位不会报形状错误）."""
        params = parameters()
        forward = stack_forward(params, inputs())
        detail = check_stack_matches_loop(params, inputs(), forward).detail
        assert "for" in detail


class TestDeterministic:
    """性质 3：两次调用逐位相同（输出与每一行读数）."""

    def test_it_passes_on_the_sample(self):
        """样本上通过：输出与 4 行读数都逐位相等."""
        params = parameters()
        forward = stack_forward(params, inputs())
        result = check_stack_is_deterministic(params, inputs(), forward)
        assert result.passed
        assert "输出逐位相等 True" in result.evidence
        assert "4 行读数逐位相等 True" in result.evidence

    def test_it_fails_for_a_foreign_record(self):
        """拿一条**别的**前向账来比：判据亮红（读数也对不上）."""
        params = parameters()
        forward = stack_forward(params, inputs())
        other = stack_forward(
            parameters(seed=99), inputs()
        )
        result = check_stack_is_deterministic(params, inputs(), other)
        assert not result.passed

    def test_the_detail_says_why_it_is_deterministic(self):
        """说明里点出“本包没有随机数”——确定性不是运气."""
        params = parameters()
        forward = stack_forward(params, inputs())
        detail = check_stack_is_deterministic(params, inputs(), forward).detail
        assert "rand" in detail or "随机数" in detail


class TestIdentityWhenBranchesVanish:
    """性质 4：每一层两个分支全零 ⇒ 整条链是恒等."""

    def test_it_passes_on_the_sample(self):
        """样本上通过：输出与输入逐位相等."""
        result = check_stack_identity_when_branches_vanishes(parameters(), inputs())
        assert result.passed
        assert "逐位相等" in result.evidence

    def test_it_deflates_the_branches_itself(self):
        """**这个性质自己负责关那两个开关**——传进来的参数是什么状态都不影响结论.

        这一条要说清一个反直觉的地方：性质函数内部会把每一层的两个分支都置零，
        因此“传一份只关了一半的参数进来”**也会通过**。而“只关一个分支 ⇒ 不是恒等”
        这个负例必须另外做（见 ``test_stack_layers.py`` 里的
        ``test_only_the_ffn_out_zero_is_not_enough``）——
        把负例放在这条性质上会得到一个永远为真的断言。
        """
        params = parameters()
        half = StackParameters(
            blocks=tuple(zero_branches(item) for item in params.blocks),
            attentions=params.attentions,
        )
        result = check_stack_identity_when_branches_vanishes(half, inputs())
        assert result.passed

    def test_it_deflates_the_attention_too(self):
        """另一半也一样：只把注意力置零的参数进来，结论仍然是“恒等”."""
        params = parameters()
        half = StackParameters(
            blocks=params.blocks,
            attentions=tuple(zero_attention(item) for item in params.attentions),
        )
        assert check_stack_identity_when_branches_vanishes(half, inputs()).passed

    def test_the_identity_needs_the_residual_to_be_on(self):
        """**残差**那一半不能关：关掉之后这个性质不再成立.

        性质函数只关“两个分支”，而残差那一条路必须留着——否则
        ``y = F(LN(x))`` 在分支全零时退化成零矩阵，而不是 ``x``。
        这一条是**对性质本身的反证**：它证明这条性质测的确实是残差那条 +1 路。
        """
        from smart_research_agent.encoder_decoder.types import NORM_PRE

        params = parameters(layers=1)
        deflated = StackParameters(
            blocks=tuple(zero_branches(item) for item in params.blocks),
            attentions=tuple(zero_attention(item) for item in params.attentions),
        )
        forward = stack_forward(deflated, inputs(), placement=NORM_PRE, use_residual=False)
        assert forward.output != inputs()
        assert all(value == 0.0 for row in forward.output for value in row)

    def test_a_deep_bare_stack_hits_day075_zero_row_guard(self):
        """**一条跨天的边界**：残差关 + 分支全零 + 两层 ⇒ 第二层收到全零行而被拒.

        day075 的 ``self_attention`` 显式拒绝**全零行**（它的 softmax 会给出均匀分布，
        而“均匀分布”与“还没学到”看起来一样）。于是在这个退化的配置里：

        ```text
        一层    输出 = F(LN(x)) = 0        ——还能算完（零矩阵是一个合法的结果）
        两层    第一层把 y 压成全零 ⇒ 第二层的注意力拿到全零行 ⇒ NumericError
        ```

        这不是 bug，而是两天的判据在这里**接上了**：一个把输入压成全零的堆叠
        在 day075 的眼里就是一次“不该交给这一层猜”的调用。
        """
        from smart_research_agent.encoder_decoder.types import NORM_PRE
        from smart_research_agent.transformer_core.errors import (
            NumericError as CoreNumericError,
        )

        params = parameters(layers=2)
        deflated = StackParameters(
            blocks=tuple(zero_branches(item) for item in params.blocks),
            attentions=tuple(zero_attention(item) for item in params.attentions),
        )
        with pytest.raises(CoreNumericError):
            stack_forward(deflated, inputs(), placement=NORM_PRE, use_residual=False)

    def test_the_detail_names_the_three_switches(self):
        """说明里点出“残差开 + 分支全零”，并解释 IEEE 下 ``x + 0`` 是精确的."""
        detail = check_stack_identity_when_branches_vanishes(parameters(), inputs()).detail
        assert "IEEE" in detail


class TestParameterCountMatches:
    """性质 5：解析式 == 逐块数出来的参数量."""

    def test_it_passes_on_the_sample(self):
        """样本上通过：两边都是 1944."""
        result = check_parameter_count_matches(shape(), parameters())
        assert result.passed
        assert "1944" in result.evidence

    def test_it_fails_for_a_shallow_parameter_bundle(self):
        """参数少了三层：判据亮红（层数也对不上）."""
        result = check_parameter_count_matches(shape(), parameters(layers=1))
        assert not result.passed

    def test_the_detail_explains_both_sides(self):
        """说明里点出“解析式与实测各写一遍”的理由."""
        detail = check_parameter_count_matches(shape(), parameters()).detail
        assert "解析式" in detail


class TestAssemblyCarriesShape:
    """性质 6：生成的脚本带着**同一个**形状."""

    def test_it_passes_on_the_sample(self):
        """样本上通过：CONFIG 逐键对上、类与 nn 模块齐备."""
        result = check_assembly_carries_shape(shape(), script())
        assert result.passed
        assert "对不上的 无" in result.evidence

    def test_it_fails_for_a_script_of_another_shape(self):
        """换一份别的形状的脚本：判据亮红（d / d_ff / n / N 都对不上）."""
        result = check_assembly_carries_shape(shape(), script(hidden=8, tokens=6))
        assert not result.passed

    def test_it_fails_when_a_module_is_missing(self):
        """把 ``MultiheadAttention`` 那一行删掉：判据亮红（少了一块组装）."""
        text = script().replace("nn.MultiheadAttention", "nn.Identity")
        result = check_assembly_carries_shape(shape(), text)
        assert not result.passed

    def test_the_detail_says_where_the_numbers_come_from(self):
        """说明里点出“源码里没有手工数字”."""
        detail = check_assembly_carries_shape(shape(), script()).detail
        assert "CONFIG" in detail or "手工数字" in detail


class TestCheckProperties:
    """六条一起跑：名单必须齐、结论必须全绿."""

    def test_the_sample_is_all_green(self):
        """样本上六条全过，且名单与 ``STACK_PROPERTIES`` 逐项对齐."""
        params = parameters()
        report = check_properties(shape(), params, inputs(), script=script())
        assert report.ok
        assert tuple(item.name for item in report.outcomes) == STACK_PROPERTIES
        assert report.failures == ()
        assert "通过 6、失败 0" in report.summary_line()

    def test_it_generates_the_script_when_none_is_given(self):
        """不给脚本时自己按形状生成一份（因此这一条永远不会“因为没传参而跳过”）."""
        report = check_properties(shape(), parameters(), inputs())
        assert report.ok

    def test_the_sample_notes_are_recorded(self):
        """三条边界被带进报告（报告要能自己解释口径）."""
        report = check_properties(shape(), parameters(), inputs())
        assert len(report.notes) == 3
        assert isinstance(json.dumps(report.to_dict()), str)

    def test_an_incomplete_report_is_rejected(self):
        """缺项的报告 ⇒ ``NumericError``（它在'全绿'时看起来一样）."""
        with pytest.raises(NumericError):
            PropertyReport(
                outcomes=(check_stack_preserves_shape(stack_forward(parameters(), inputs())),)
            )

    def test_an_empty_report_is_rejected(self):
        """空报告没有意义."""
        with pytest.raises(ParameterError):
            PropertyReport(outcomes=())


class TestPropertyOutcome:
    """单条读数与两种输出."""

    def test_unknown_names_are_rejected(self):
        """未知的性质名当场报错（名单只有一处实现）."""
        with pytest.raises(ParameterError):
            PropertyOutcome(name="looks_fine", passed=True, evidence="嗯")

    def test_to_dict_carries_the_description(self):
        """读数进字典时带着“这一条在说什么”."""
        params = parameters()
        forward = stack_forward(params, inputs())
        payload = check_stack_preserves_shape(forward).to_dict()
        assert payload["description"]
        assert payload["passed"] is True
        assert set(payload) == {"name", "passed", "evidence", "detail", "description"}

    def test_all_six_names_are_reachable(self):
        """六条性质的名字都能造出一条读数（闭合检查）."""
        for name in STACK_PROPERTIES:
            assert PropertyOutcome(name=name, passed=True, evidence="ok").passed


class TestSmallTools:
    """几个便宜的小工具（演示脚本与测试共用）."""

    def test_stage_lines_cover_the_five_stages(self):
        """五个阶段各一行，且每一行都带着说明与形状."""
        lines = stack_stage_lines()
        assert len(lines) == 5
        assert all("——" in line for line in lines)
        assert "block" in lines[1]

    def test_make_samples_returns_four_pieces(self):
        """样本工厂：形状 / 参数 / 输入 / 目标."""
        sample_shape, params, sample_inputs, sample_target = make_samples()
        assert sample_shape.layers == 4
        assert params.layers == 4
        assert len(sample_inputs) == len(sample_target) == sample_shape.tokens

    def test_make_samples_is_deterministic(self):
        """两次调用给出同一组样本（测试与演示靠这一条对齐）."""
        first = make_samples()
        second = make_samples()
        assert first[1] == second[1]
        assert first[2] == second[2]

    def test_relative_gradient_change(self):
        """相邻比值：``(2, 1, 0.5)`` → ``(0.5, 0.5)``."""
        assert relative_gradient_change((2.0, 1.0, 0.5)) == (0.5, 0.5)

    def test_relative_gradient_change_handles_a_zero(self):
        """零会让比值没有定义——那一位返回 ``0.0``，而不是 nan."""
        assert relative_gradient_change((0.0, 1.0)) == (0.0,)
        assert relative_gradient_change((1.0, 0.0)) == (0.0,)

    def test_relative_gradient_change_of_one_value_is_empty(self):
        """只有一个数时没有“相邻比值”这回事."""
        assert relative_gradient_change((5.0,)) == ()

    def test_relative_gradient_change_rejects_negatives(self):
        """范数序列不能有负数."""
        with pytest.raises(NumericError):
            relative_gradient_change((1.0, -1.0))

    def test_frobenius_profile_matches_the_census(self):
        """逐层出口范数与读数表一致."""
        forward = stack_forward(parameters(), inputs())
        assert frobenius_profile(forward) == tuple(row.output_norm for row in forward.censuses)
        assert len(frobenius_profile(forward)) == LAYERS

    def test_frobenius_profile_rejects_a_foreign_record(self):
        """传进来不是前向账时当场报错."""
        with pytest.raises(ParameterError):
            frobenius_profile(3)  # type: ignore[arg-type]

    def test_census_of_layer_returns_a_dictionary(self):
        """按层号取读数的字典形状."""
        forward = stack_forward(parameters(), inputs())
        payload = census_of_layer(forward, 1)
        assert payload["index"] == 1
        assert set(payload) == {
            "index",
            "input_norm",
            "output_norm",
            "branch1_norm",
            "branch2_norm",
            "max_abs_input",
            "use_residual",
            "gain",
            "carry_share",
        }

    def test_matrix_norm_of_matches_the_package(self):
        """形状为 ``(4, 6)`` 的样本输入的范数可手算."""
        assert matrix_norm_of(inputs()) > 0.0

    def test_checked_vector_accepts_a_sequence(self):
        """一串数会被校验成 ``Vector``."""
        assert checked_vector([1.0, 2.0]) == (1.0, 2.0)

    def test_checked_vector_rejects_a_non_number(self):
        """非数字当场报错（形状口径只有一处实现）."""
        with pytest.raises(ValueError):
            checked_vector([1.0, "二"])  # type: ignore[list-item]

    def test_parse_script_module_names_is_wider_than_nn(self):
        """它给出脚本里**所有**被调用的名字（``nn`` 那一族只是其中一部分）."""
        names = parse_script_module_names(script())
        assert "Manual" not in names
        assert "LayerNorm" in names
        assert "manual_seed" in names

    def test_parse_script_module_names_rejects_broken_text(self):
        """连 ``ast.parse`` 都过不去的文本当场报错."""
        with pytest.raises(ParameterError):
            parse_script_module_names("def 坏(:")

    def test_two_shapes_share_one_sample_helper(self):
        """样本形状与 ``make_shape`` 完全一致（口径只有一处）."""
        assert shape(hidden=HIDDEN, tokens=TOKENS).total_parameter_count == 1944
        assert script(placement="post").count("placement") >= 1

    def test_assembly_script_is_not_empty(self):
        """生成的脚本非空（最后一道便宜的兜底）."""
        assert len(assembly_script(shape()).splitlines()) > 40
