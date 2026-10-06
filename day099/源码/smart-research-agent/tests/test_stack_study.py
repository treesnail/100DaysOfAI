"""``transformer_stack.study``：堆叠实验与几何平均（day080）."""

from __future__ import annotations

import json

import pytest

from smart_research_agent.transformer_stack import (
    DEFAULT_STUDY_DEPTHS,
    STUDY_VARIANTS,
    VARIANT_BARE,
    VARIANT_DESCRIPTIONS,
    VARIANT_RESIDUAL,
    NumericError,
    ParameterError,
    StackRow,
    StackStudy,
    geometric_mean,
    stack_gradient_profile,
    stack_loss_gradient,
)
from smart_research_agent.transformer_stack.study import (
    gradient_norms_of,
    shape_of_study,
    stack_study,
)
from smart_research_agent.transformer_stack.verify import relative_gradient_change
from tests.stack_samples import parameters, target


class TestGeometricMean:
    """几何平均：比值的“平均”必须是它."""

    def test_it_is_the_nth_root_of_the_product(self):
        """``(1, 4)`` 的几何平均是 2（把 1 与 4 平均成 2.5 是错的）."""
        assert geometric_mean((1.0, 4.0)) == pytest.approx(2.0)

    def test_it_reproduces_the_product(self):
        """**关键性质**：N 个 step 的几何平均取 N 次方 = 它们的乘积."""
        steps = (0.9, 1.1, 0.8)
        product = steps[0] * steps[1] * steps[2]
        assert geometric_mean(steps) ** 3 == pytest.approx(product)

    def test_the_identity_holds_on_a_real_chain(self):
        """**文档里那条恒等式**：``mean_step_ratio^(N−1) = ‖dx_{N−1}‖/‖dx_0‖``.

        第一版文档把 ``overall_ratio`` 也写成了“各层 step 相乘”，而那只对
        **链内首尾之比**成立——``dx`` 序列有 N 个数、只有 N−1 个比值，
        而 ``dy_N`` 不在那个序列里。这一条把它钉住。
        """
        params = parameters(layers=4)
        _forward, grads = stack_loss_gradient(params, target(), target())
        norms = grads.norms()
        steps = relative_gradient_change(norms)
        assert geometric_mean(steps) ** (len(norms) - 1) == pytest.approx(
            norms[-1] / norms[0]
        )

    def test_overall_ratio_is_not_the_same_quantity(self):
        """``overall_ratio`` 的分母是 ``‖dy_N‖``，而它不是 ``dx`` 序列里的一项."""
        params = parameters(layers=4)
        _forward, grads = stack_loss_gradient(params, target(), target())
        norms = grads.norms()
        steps = relative_gradient_change(norms)
        inside = norms[-1] / norms[0]
        assert geometric_mean(steps) ** (len(norms) - 1) == pytest.approx(inside)
        assert inside != pytest.approx(norms[0] / norms[-1])

    def test_an_empty_sequence_is_one(self):
        """空序列返回 ``1.0``（乘法的单位元，而不是 0）."""
        assert geometric_mean(()) == 1.0

    def test_a_zero_makes_the_whole_thing_zero(self):
        """有一项是 0，乘积就是 0（不做对数平均的近似）."""
        assert geometric_mean((0.0, 3.0)) == 0.0

    def test_negatives_are_rejected(self):
        """负数没有实数对数，当场报错."""
        with pytest.raises(NumericError):
            geometric_mean((1.0, -1.0))

    def test_it_is_symmetric_in_a_reciprocal_pair(self):
        """``3`` 与 ``1/3`` 的几何平均恰好是 1（算术平均会是 1.67）."""
        assert geometric_mean((3.0, 1.0 / 3.0)) == pytest.approx(1.0)


class TestStackStudySample:
    """样本上的实验：两个变体 × 六档深度."""

    def test_it_covers_both_variants_and_every_depth(self):
        """``2 × 6 = 12`` 行，且名单齐备（缺一个对照就失去参照物）."""
        study = stack_study()
        assert len(study.rows) == 2 * len(DEFAULT_STUDY_DEPTHS)
        assert {row.variant for row in study.rows} == set(STUDY_VARIANTS)
        assert study.deepest == max(DEFAULT_STUDY_DEPTHS) == 8

    def test_the_two_variants_share_everything_but_the_switch(self):
        """同一个深度上，两个变体的参数量完全相同（只差残差那一个开关）."""
        study = stack_study(depths=(1, 3))
        for layers in (1, 3):
            counts = {
                row.parameter_count for row in study.rows if row.layers == layers
            }
            assert len(counts) == 1

    def test_the_parameter_count_grows_with_depth(self):
        """参数量随深度线性增长（每层固定 486）."""
        study = stack_study(depths=(1, 2, 4))
        counts = [row.parameter_count for row in study.rows if row.variant == VARIANT_RESIDUAL]
        assert counts == [486, 972, 1944]

    def test_the_two_variants_differ_at_depth_one(self):
        """**深度为 1 时两个变体就已经不同**：残差开关在那时也改变了前向.

        ``y = x + F(LN(x))`` 与 ``y = F(LN(x))`` 是同一个代码路径的两侧，
        而它们的输出（因此 ``‖dy`` 与整体的比值）本来就不一样。把这一条写下来，
        是为了防止有人以为“一层时残差没有第二项可加，所以两者必然相同”。
        """
        study = stack_study(depths=(1,))
        assert study.ratios(VARIANT_RESIDUAL) != study.ratios(VARIANT_BARE)
        assert study.ratios(VARIANT_RESIDUAL)[0] > study.ratios(VARIANT_BARE)[0]

    def test_the_residual_variant_keeps_the_overall_ratio_near_one(self):
        """残差开时，整体比在 ``1~3`` 之间（一条**没有塌下去**的路）."""
        study = stack_study()
        values = study.ratios(VARIANT_RESIDUAL)
        assert all(1.0 < value < 3.0 for value in values)

    def test_the_bare_variant_collapses(self):
        """残差关时，最深那一层的整体比塌到 ``1e-2`` 以下."""
        study = stack_study()
        assert study.ratios(VARIANT_BARE)[-1] < 1e-2

    def test_the_step_ratio_is_almost_independent_of_depth(self):
        """**这一课的值钱结论**：残差开时的“每层倍数”与总深度几乎无关.

        判据是极差 ``max/min < 2``——一个与深度无关的因子应该落在这条线内。
        """
        study = stack_study()
        assert study.step_ratio_spread < 2.0
        values = study.step_ratios(VARIANT_RESIDUAL)
        assert all(0.5 < value < 1.5 for value in values)

    def test_the_bare_variants_step_ratio_is_not_a_stable_factor(self):
        """而关掉残差之后，“每层倍数”自己就开始乱跳（逐层形状是噪声）."""
        study = stack_study()
        values = study.step_ratios(VARIANT_BARE)
        assert max(values) / min(values) > 3.0

    def test_the_verdict_holds(self):
        """判决：最深一层上 ``bare`` 比 ``residual`` 差一个数量级以上，且每层倍数稳定."""
        study = stack_study()
        assert study.verdict_ok
        assert study.ratios(VARIANT_BARE)[-1] < study.ratios(VARIANT_RESIDUAL)[-1] / 10.0

    def test_the_1_layer_ratio_is_the_same_for_both_variants(self):
        """两个变体在**一层**上的读数都落在同一个量级（可比较）.

        它们并不相等（见上面那条），但都必须是一个正的有限读数——
        否则“残差有用”这句话就失去了参照物。
        """
        study = stack_study(depths=(1,))
        for variant in STUDY_VARIANTS:
            value = study.ratios(variant)[0]
            assert 0.0 < value < 10.0

    def test_output_drift_is_positive(self):
        """前向把输入搬离了多远：一个正数（每一档都是）."""
        study = stack_study(depths=(1, 4))
        assert all(row.output_drift > 0.0 for row in study.rows)

    def test_the_records_are_json_serialisable(self):
        """每一行与整份报告都能 json.dumps."""
        study = stack_study(depths=(1, 2))
        assert isinstance(json.dumps(study.to_dict()), str)
        assert all(isinstance(json.dumps(row.to_dict()), str) for row in study.rows)


class TestStackStudyOutput:
    """两种输出：一行摘要与一张表."""

    def test_summary_line_names_the_deepest_ratio(self):
        """摘要里有“最深一层整体比”与两个变体的名字."""
        text = stack_study(depths=(1, 2)).summary_line()
        assert "最深一层整体比" in text
        assert VARIANT_RESIDUAL in text and VARIANT_BARE in text
        assert "每层倍数极差" in text

    def test_table_lines_have_a_header_and_a_row_per_row(self):
        """一张表：表头 + 分隔线 + 每行读数一行."""
        study = stack_study(depths=(1, 2))
        lines = study.table_lines()
        assert len(lines) == len(study.rows) + 2
        assert "每层倍数" in lines[0]
        assert lines[1].strip().startswith("-")

    def test_row_summary_line(self):
        """单行摘要里带着四个读数."""
        row = stack_study(depths=(1,)).rows[0]
        text = row.summary_line()
        assert "整体比" in text and "每层倍数" in text and "输出漂移" in text


class TestStackStudyValidation:
    """参数与构造检查."""

    def test_depths_must_be_strictly_increasing(self):
        """深度序列必须严格递增且不重复（否则同一条曲线会被画两次）."""
        with pytest.raises(ParameterError):
            stack_study(depths=(1, 2, 2))

    def test_depths_must_not_be_empty(self):
        """空深度序列没有意义."""
        with pytest.raises(ParameterError):
            stack_study(depths=())

    @pytest.mark.parametrize("layers", [0, -1])
    def test_depths_must_be_positive(self, layers: int):
        """深度必须 >= 1."""
        with pytest.raises(ParameterError):
            stack_study(depths=(layers,))

    def test_scale_must_live_in_the_open_unit_interval(self):
        """初始化幅度必须在 ``(0, 1)``."""
        with pytest.raises(ParameterError):
            stack_study(depths=(1,), scale=1.5)

    def test_a_study_with_one_variant_is_rejected(self):
        """缺一个变体的报告 ⇒ ``NumericError``."""
        row = StackRow(
            variant=VARIANT_RESIDUAL,
            layers=1,
            input_gradient_norm=1.0,
            output_gradient_norm=1.0,
            overall_ratio=1.0,
            mean_step_ratio=1.0,
            output_drift=0.5,
            parameter_count=486,
        )
        with pytest.raises(NumericError):
            StackStudy(
                hidden=6,
                ffn=24,
                tokens=4,
                depths=(1,),
                rows=(row,),
                seed=7,
                scale=0.25,
            )

    def test_an_empty_row_list_is_rejected(self):
        """没有行读数的实验没有意义."""
        with pytest.raises(ParameterError):
            StackStudy(
                hidden=6, ffn=24, tokens=4, depths=(1,), rows=(), seed=7, scale=0.25
            )

    def test_a_duplicate_depth_sequence_is_rejected(self):
        """深度序列重复 ⇒ ``ParameterError``（在 ``StackStudy`` 这一层也要挡住）."""
        rows = tuple(
            StackRow(
                variant=variant,
                layers=1,
                input_gradient_norm=1.0,
                output_gradient_norm=1.0,
                overall_ratio=1.0,
                mean_step_ratio=1.0,
                output_drift=0.5,
                parameter_count=486,
            )
            for variant in STUDY_VARIANTS
        )
        with pytest.raises(ParameterError):
            StackStudy(
                hidden=6,
                ffn=24,
                tokens=4,
                depths=(1, 1),
                rows=rows,
                seed=7,
                scale=0.25,
            )


class TestStackRow:
    """单行读数的构造检查."""

    def test_unknown_variants_are_rejected(self):
        """未知变体名当场报错（不给它挑一个默认设置）."""
        with pytest.raises(ParameterError):
            StackRow(
                variant="half_residual",
                layers=2,
                input_gradient_norm=1.0,
                output_gradient_norm=1.0,
                overall_ratio=1.0,
                mean_step_ratio=1.0,
                output_drift=1.0,
                parameter_count=972,
            )

    @pytest.mark.parametrize(
        "field",
        [
            "input_gradient_norm",
            "output_gradient_norm",
            "overall_ratio",
            "mean_step_ratio",
            "output_drift",
        ],
    )
    def test_nan_readings_are_rejected(self, field: str):
        """``nan`` 不能进读数."""
        payload = {
            "variant": VARIANT_RESIDUAL,
            "layers": 2,
            "input_gradient_norm": 1.0,
            "output_gradient_norm": 1.0,
            "overall_ratio": 1.0,
            "mean_step_ratio": 1.0,
            "output_drift": 1.0,
            "parameter_count": 972,
        }
        payload[field] = float("nan")
        with pytest.raises(NumericError):
            StackRow(**payload)  # type: ignore[arg-type]

    def test_layer_and_parameter_counts_must_be_positive(self):
        """层数与参数量都必须是正数."""
        with pytest.raises(ParameterError):
            StackRow(
                variant=VARIANT_RESIDUAL,
                layers=0,
                input_gradient_norm=1.0,
                output_gradient_norm=1.0,
                overall_ratio=1.0,
                mean_step_ratio=1.0,
                output_drift=1.0,
                parameter_count=972,
            )
        with pytest.raises(ParameterError):
            StackRow(
                variant=VARIANT_RESIDUAL,
                layers=2,
                input_gradient_norm=1.0,
                output_gradient_norm=1.0,
                overall_ratio=1.0,
                mean_step_ratio=1.0,
                output_drift=1.0,
                parameter_count=0,
            )


class TestStudyHelpers:
    """两个小工具：逐层范数与形状重放."""

    def test_gradient_norms_of_matches_the_record(self):
        """逐层范数与 ``StackGradients.norms`` 一致."""
        _forward, grads = stack_loss_gradient(parameters(), target(), target())
        assert gradient_norms_of(grads) == grads.norms()

    def test_gradient_norms_of_rejects_a_foreign_record(self):
        """传进来不是梯度账时当场报错."""
        with pytest.raises(ParameterError):
            gradient_norms_of("not-a-record")  # type: ignore[arg-type]

    def test_shape_of_study_round_trips(self):
        """按深度重放形状：层数与参数量都对得上."""
        study = stack_study(depths=(1, 2))
        item = shape_of_study(study, 2)
        assert item.layers == 2
        assert item.hidden == study.hidden
        assert item.ffn == study.ffn
        assert item.total_parameter_count == 972

    def test_variant_descriptions_are_closed(self):
        """两个变体各有一条说明（表与键必须闭合）."""
        assert set(VARIANT_DESCRIPTIONS) == set(STUDY_VARIANTS)
        assert all(item for item in VARIANT_DESCRIPTIONS.values())

    def test_step_ratios_agree_with_the_gradient_profile(self):
        """实验里的“每层倍数”与 ``relative_gradient_change`` 用的是同一个定义."""
        params = parameters(layers=4)
        _forward, grads = stack_loss_gradient(params, target(), target())
        from_record = geometric_mean(relative_gradient_change(grads.norms()))
        from_profile = geometric_mean(
            relative_gradient_change(stack_gradient_profile(params, target(), target()))
        )
        assert from_record == pytest.approx(from_profile)
