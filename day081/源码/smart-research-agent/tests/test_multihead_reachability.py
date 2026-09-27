"""可达集合：单头是一个凸包，多头是 heads 个凸包的闵可夫斯基和（day076 / M7-D2）.

这一份测试守的是这一课**唯一值钱的结论**：

```text
同一批参数（参数量一字不差）下，存在一些目标点：单头**结构上到不了**，双头能到。
```

三个手算的锚点：

```text
单头凸包         线段 (0,1)—(1,0)（第三点是中点，共线）
单头到目标距离   1/√2 = 0.7071067811865475
兑现目标的权重   head 0 → (1,0,0,0)、head 1 → (0,1,0,0)（TV = 1.0）
```

第二个锚点有一条必须写进测试的细节：**公式值与算法值差最后一位**
（``0.7071067811865475`` vs ``0.7071067811865476``）——把两者写成 ``==``
会让某一次无关的重排变成一次假失败。因此这里用 1e-12 的容差，
并把观测到的差写进断言旁边的注释。
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from smart_research_agent.multi_head.errors import (
    MultiHeadError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.multi_head.layers import multi_head_attention
from smart_research_agent.multi_head.reachability import (
    HULL_TOLERANCE,
    MAX_VERTEX_COMBINATIONS,
    WITNESS_ONE_HEAD_DISTANCE,
    WITNESS_TARGET,
    WITNESS_WEIGHTS,
    allowed_positions,
    convex_hull_2d,
    designed_witness,
    distance_to_convex_hull_2d,
    head_contributions,
    minkowski_vertices,
    multi_head_shape_of,
    reachability_report,
    realize_with_weights,
    single_head_contributions,
    witness_parameters_are_shared,
    witness_report,
)
from smart_research_agent.transformer_core.train import (
    default_parameters,
    make_induction_batch,
)
from tests.multihead_samples import (
    WITNESS_DISTANCE,
    WITNESS_DISTRIBUTIONS,
    WITNESS_ONE_PULLS,
    WITNESS_TARGET_2D,
    WITNESS_ZERO_PULLS,
    approx,
    identity_parameters,
)


class TestConvexHull:
    """二维凸包：共线点必须被丢掉（否则"点在多边形内"会面对一个退化多边形）."""

    def test_collinear_points_collapse_to_a_segment(self):
        hull = convex_hull_2d(((1.0, 0.0), (0.0, 1.0), (0.5, 0.5)))
        assert hull == ((0.0, 1.0), (1.0, 0.0))

    def test_square_keeps_four_vertices(self):
        hull = convex_hull_2d(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)))
        assert hull == ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

    def test_single_point(self):
        assert convex_hull_2d(((0.25, 0.75),)) == ((0.25, 0.75),)

    def test_duplicates_are_dropped(self):
        hull = convex_hull_2d(((1.0, 1.0), (1.0, 1.0), (0.0, 0.0)))
        assert hull == ((0.0, 0.0), (1.0, 1.0))

    def test_empty_is_rejected(self):
        with pytest.raises(ShapeError):
            convex_hull_2d(())

    def test_three_dimensions_are_rejected(self):
        with pytest.raises(ParameterError):
            convex_hull_2d(((1.0, 2.0, 3.0),))

    def test_non_finite_is_rejected(self):
        with pytest.raises(ParameterError):
            convex_hull_2d(((float("nan"), 0.0),))


class TestDistanceToHull:
    """"到不了"这件事只有在**距离**这个读数下才是可执行的."""

    def test_hand_computed_perpendicular_distance(self):
        """目标 (1,1) 到线段 (1,0)—(0,1)：垂足恰好是中点，距离 = |1+1−1|/√2."""
        distance = distance_to_convex_hull_2d(
            (1.0, 1.0), [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]
        )
        assert distance == pytest.approx(WITNESS_ONE_HEAD_DISTANCE, abs=1e-12)
        assert approx(distance, math.sqrt(2.0) / 2.0, tolerance=1e-12)

    def test_inside_a_polygon_is_zero(self):
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert distance_to_convex_hull_2d((0.5, 0.5), points) == 0.0
        assert distance_to_convex_hull_2d((1.0, 1.0), points) == 0.0

    def test_single_point_distance(self):
        assert approx(distance_to_convex_hull_2d((0.0, 3.0), [(0.0, 0.0)]), 3.0)

    def test_projection_outside_the_segment_uses_the_endpoint(self):
        """投影落在段外时必须取端点距离——否则会算出一个"垂直的假象"."""
        assert approx(
            distance_to_convex_hull_2d((2.0, 0.0), [(0.0, 0.0), (1.0, 0.0)]), 1.0
        )

    def test_outside_a_polygon_uses_the_nearest_edge(self):
        points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        assert approx(distance_to_convex_hull_2d((2.0, 0.5), points), 1.0)
        assert approx(distance_to_convex_hull_2d((2.0, 2.0), points), math.sqrt(2.0))

    def test_target_must_be_two_dimensional(self):
        with pytest.raises(ParameterError):
            distance_to_convex_hull_2d((1.0, 2.0, 3.0), [(0.0, 0.0)])

    def test_target_must_be_finite(self):
        with pytest.raises(ParameterError):
            distance_to_convex_hull_2d((float("inf"), 0.0), [(0.0, 0.0)])


class TestMinkowskiVertices:
    """网格点集 = heads 个凸包之和（那条定理让"无限多点的和"变成有限件事）."""

    def test_witness_grid_has_nine_unique_points(self):
        witness = designed_witness()
        forward = multi_head_attention(
            witness.params, witness.inputs, heads=witness.heads, causal=True
        )
        sets = head_contributions(forward, witness.row)
        vertices = minkowski_vertices(sets)
        assert len(vertices) == 9
        assert len(set(vertices)) == 9

    def test_single_head_set_is_itself(self):
        grid = ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5))
        assert minkowski_vertices((grid,)) == grid

    def test_no_head_is_rejected(self):
        with pytest.raises(ShapeError):
            minkowski_vertices(())

    def test_empty_group_is_rejected(self):
        with pytest.raises(ShapeError):
            minkowski_vertices((((1.0, 0.0),), ()))

    def test_width_mismatch_is_rejected(self):
        with pytest.raises(ShapeError):
            minkowski_vertices((((1.0, 0.0),), ((0.0,),)))

    def test_zero_width_is_rejected(self):
        with pytest.raises(ShapeError):
            minkowski_vertices((((),),))

    def test_cap_is_enforced(self):
        """``|A|^heads`` 会爆炸——超过上限就拒绝而不是把内存吃光."""
        group = tuple((float(index), 0.0) for index in range(10))
        sets = tuple(group for _ in range(6))  # 10^6 = 1_000_000 > 200_000
        assert 10**6 > MAX_VERTEX_COMBINATIONS
        with pytest.raises(ParameterError, match="网格点"):
            minkowski_vertices(sets)


class TestAllowedPositions:
    """允许的位置：因果掩码下第 i 行看到 0..i（掩码是**唯一**的判据）."""

    def test_causal_rows_grow(self):
        witness = designed_witness()
        forward = multi_head_attention(
            witness.params, witness.inputs, heads=witness.heads, causal=True
        )
        assert allowed_positions(forward, 0) == (0,)
        assert allowed_positions(forward, 3) == (0, 1, 2, 3)

    def test_row_must_be_an_index(self):
        forward = multi_head_attention(
            identity_parameters(4), ((1.0, 0.0, 0.0, 0.0),) * 2, heads=2
        )
        with pytest.raises(ParameterError):
            allowed_positions(forward, 2)
        with pytest.raises(ParameterError):
            allowed_positions(forward, True)

    def test_a_row_without_allowed_positions_is_rejected(self):
        """掩码形状对、但某一行全 False：这一行的分布没有定义（**显式拒绝**）."""
        forward = multi_head_attention(
            identity_parameters(4), ((1.0, 0.0, 0.0, 0.0),) * 2, heads=2
        )
        broken = dataclasses.replace(
            forward, mask=((False, False), (True, True))
        )
        with pytest.raises(ShapeError, match="没有任何允许的位置"):
            allowed_positions(broken, 0)


class TestContributions:
    """贡献向量：第 h 头整行押在位置 j 时交给 W_o 的那一条向量."""

    def test_witness_head_zero_is_along_the_first_axis(self):
        witness = designed_witness()
        forward = multi_head_attention(
            witness.params, witness.inputs, heads=witness.heads, causal=True
        )
        sets = head_contributions(forward, witness.row)
        pulls = WITNESS_ZERO_PULLS
        assert sets[0] == tuple((pull, 0.0) for pull in pulls)

    def test_witness_head_one_is_along_the_second_axis(self):
        witness = designed_witness()
        forward = multi_head_attention(
            witness.params, witness.inputs, heads=witness.heads, causal=True
        )
        sets = head_contributions(forward, witness.row)
        pulls = WITNESS_ONE_PULLS
        assert sets[1] == tuple((0.0, pull) for pull in pulls)

    def test_single_head_contributions_are_the_value_rows(self):
        """``W_o = [[1,0,0,0],[0,0,0,1]]`` ⇒ 单头贡献 = ``(V_j[0], V_j[3])``."""
        witness = designed_witness()
        vertices = single_head_contributions(
            witness.params, witness.inputs, witness.row, causal=True
        )
        assert vertices == tuple(
            (t, s) for t, s in zip(WITNESS_ZERO_PULLS, WITNESS_ONE_PULLS, strict=True)
        )

    def test_same_parameters_for_both_structures(self):
        """结构性对照的前提：两次前向的四个矩阵**逐位相同**."""
        assert witness_parameters_are_shared() is True


class TestWitness:
    """手工设计的见证：单头到不了、双头到得了."""

    def test_witness_fields(self):
        witness = designed_witness()
        assert witness.heads == 2
        assert witness.row == 3
        assert witness.target == WITNESS_TARGET == WITNESS_TARGET_2D
        assert witness.witness_weights == WITNESS_WEIGHTS == WITNESS_DISTRIBUTIONS
        assert witness.expected_one_head_distance == WITNESS_DISTANCE

    def test_witness_shape_is_the_tight_case(self):
        """``head_dim`` **恰好等于**可见位置数：``>=`` 取等号的边界情形."""
        witness = designed_witness()
        shape = multi_head_shape_of(witness.params, witness.heads)
        assert shape.head_dim == 4
        assert shape.head_value == 2
        assert shape.attention.outputs == 2

    def test_report_separates_the_two_structures(self):
        report = witness_report()
        assert report.one_head_reachable is False
        assert report.multi_head_reachable is True
        assert report.separates is True
        assert report.gap > 0.7

    def test_one_head_hull_is_a_segment(self):
        report = witness_report()
        assert report.one_head_hull == ((0.0, 1.0), (1.0, 0.0))
        assert len(report.one_head_vertices) == 4

    def test_multi_head_hull_is_the_unit_square(self):
        report = witness_report()
        assert report.multi_head_hull == (
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        )

    def test_distance_matches_the_hand_computed_formula(self):
        """公式值 ``1/√2`` 与算法值差**最后一位**（1.1e-16）——因此用容差比."""
        report = witness_report()
        assert approx(report.one_head_distance, WITNESS_ONE_HEAD_DISTANCE, tolerance=1e-12)
        assert report.one_head_distance == pytest.approx(
            WITNESS_ONE_HEAD_DISTANCE, abs=1e-15
        )
        assert report.multi_head_distance == 0.0

    def test_witness_weights_realise_the_target(self):
        """构造出来的见证必须能被**代进去核对**，而不是只停在几何断言上."""
        witness = designed_witness()
        forward = multi_head_attention(
            witness.params, witness.inputs, heads=witness.heads, causal=True
        )
        realized = realize_with_weights(forward, witness.row, witness.witness_weights)
        assert realized == WITNESS_TARGET

    def test_witness_weights_are_maximally_different(self):
        """两份分布的 TV = 1.0：这是"多头真的分了两路"的最强形态."""
        first, second = WITNESS_WEIGHTS
        gap = 0.5 * math.fsum(abs(a - b) for a, b in zip(first, second, strict=True))
        assert approx(gap, 1.0)

    def test_report_serialises(self):
        payload = witness_report().to_dict()
        assert payload["separates"] is True
        assert payload["allowed"] == [0, 1, 2, 3]
        assert len(payload["witness"]) == 2

    def test_report_summary_line(self):
        line = witness_report().summary_line()
        assert "单头距 0.7071" in line
        assert "分离" in line

    def test_designed_witness_serialises(self):
        payload = designed_witness().to_dict()
        assert payload["heads"] == 2
        assert payload["target"] == list(WITNESS_TARGET)


class TestReachabilityReportBuild:
    """报告构造的三种显式拒绝：维度、前提、目标点."""

    def test_d_out_must_be_two(self):
        params = default_parameters(6)
        tasks = make_induction_batch(1)
        with pytest.raises(ParameterError, match="d_out = 2"):
            reachability_report(
                params, tasks[0].inputs, 4, heads=1, target=(1.0, 1.0)
            )

    def test_head_dim_must_cover_the_visible_positions(self):
        witness = designed_witness()
        with pytest.raises(ParameterError, match="小于可见位置数"):
            reachability_report(
                witness.params,
                witness.inputs,
                witness.row,
                heads=4,
                target=(1.0, 1.0),
            )

    def test_target_must_be_two_dimensional(self):
        witness = designed_witness()
        with pytest.raises(ParameterError):
            reachability_report(
                witness.params, witness.inputs, witness.row, heads=2, target=(1.0,)
            )

    def test_target_must_be_finite(self):
        witness = designed_witness()
        with pytest.raises(ParameterError):
            reachability_report(
                witness.params,
                witness.inputs,
                witness.row,
                heads=2,
                target=(float("nan"), 1.0),
            )

    def test_default_witness_is_a_placeholder(self):
        """缺省见证只是"每头各取第一个位置"——它是合法分布，但不保证兑现目标."""
        witness = designed_witness()
        report = reachability_report(
            witness.params, witness.inputs, witness.row, heads=2, target=(1.0, 1.0)
        )
        assert report.witness == ((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        assert any("占位" in item for item in report.notes)

    def test_a_target_inside_both_hulls_does_not_separate(self):
        """目标落在单头凸包上时**不分离**——这正是"分离"这个词的判据."""
        witness = designed_witness()
        report = reachability_report(
            witness.params,
            witness.inputs,
            witness.row,
            heads=2,
            target=(0.5, 0.5),
        )
        assert report.one_head_reachable is True
        assert report.multi_head_reachable is True
        assert report.separates is False

    def test_hull_tolerance_is_used_for_reachability(self):
        witness = designed_witness()
        report = reachability_report(
            witness.params, witness.inputs, witness.row, heads=2, target=(1.0, 1.0)
        )
        assert report.one_head_reachable == (
            report.one_head_distance <= HULL_TOLERANCE
        )


class TestRealizeWeights:
    """见证的核对入口：非法分布必须被拒绝（否则它什么都不证明）."""

    def _forward(self):
        witness = designed_witness()
        return multi_head_attention(
            witness.params, witness.inputs, heads=witness.heads, causal=True
        )

    def test_extra_distributions_are_rejected(self):
        forward = self._forward()
        with pytest.raises(ShapeError):
            realize_with_weights(
                forward, 3, ((1.0, 0.0, 0.0, 0.0),) * 3
            )

    def test_wrong_slot_count_is_rejected(self):
        forward = self._forward()
        with pytest.raises(ShapeError):
            realize_with_weights(forward, 3, ((1.0, 0.0), (0.0, 1.0)))

    def test_negative_weight_is_rejected(self):
        forward = self._forward()
        with pytest.raises(MultiHeadError):
            realize_with_weights(
                forward, 3, ((-0.5, 0.5, 0.0, 1.0), (0.0, 1.0, 0.0, 0.0))
            )

    def test_unnormalised_distribution_is_rejected(self):
        forward = self._forward()
        with pytest.raises(MultiHeadError, match="合法的分布"):
            realize_with_weights(
                forward, 3, ((0.1, 0.1, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))
            )

    def test_uniform_weights_give_the_centroid(self):
        """两份均匀分布落在**两个凸包的中点之和**上（一个可手算的点）."""
        forward = self._forward()
        realized = realize_with_weights(
            forward, 3, ((0.25,) * 4, (0.25,) * 4)
        )
        expected = (
            math.fsum(WITNESS_ZERO_PULLS) / 4.0,
            math.fsum(WITNESS_ONE_PULLS) / 4.0,
        )
        assert approx(realized[0], expected[0])
        assert approx(realized[1], expected[1])


class TestReachabilityReportValidation:
    """直接构造报告时的校验（它是 frozen dataclass，没有第二道防线）."""

    def _kwargs(self, **overrides):
        report = witness_report()
        payload = {
            "heads": report.heads,
            "row": report.row,
            "allowed": report.allowed,
            "head_dim": report.head_dim,
            "one_head_vertices": report.one_head_vertices,
            "multi_head_vertices": report.multi_head_vertices,
            "one_head_hull": report.one_head_hull,
            "multi_head_hull": report.multi_head_hull,
            "target": report.target,
            "one_head_distance": report.one_head_distance,
            "multi_head_distance": report.multi_head_distance,
            "witness": report.witness,
        }
        payload.update(overrides)
        return payload

    def test_heads_must_be_positive(self):
        with pytest.raises(ParameterError):
            type(witness_report())(**self._kwargs(heads=0))

    def test_row_must_be_non_negative(self):
        with pytest.raises(ParameterError):
            type(witness_report())(**self._kwargs(row=-1))

    def test_allowed_must_not_be_empty(self):
        with pytest.raises(ShapeError):
            type(witness_report())(**self._kwargs(allowed=()))

    def test_target_must_be_two_dimensional(self):
        with pytest.raises(ParameterError):
            type(witness_report())(**self._kwargs(target=(1.0,)))

    def test_head_dim_must_cover_the_positions(self):
        with pytest.raises(ParameterError):
            type(witness_report())(**self._kwargs(head_dim=1))

    def test_witness_count_must_match_heads(self):
        with pytest.raises(ShapeError):
            type(witness_report())(**self._kwargs(witness=((1.0, 0.0, 0.0, 0.0),)))

    def test_witness_length_must_match_slots(self):
        with pytest.raises(ShapeError):
            type(witness_report())(
                **self._kwargs(witness=((1.0,), (1.0, 0.0, 0.0, 0.0)))
            )

    def test_witness_must_be_a_distribution(self):
        with pytest.raises(MultiHeadError, match="合法的分布"):
            type(witness_report())(
                **self._kwargs(
                    witness=((0.5,) * 4, (0.4,) * 4),
                )
            )

    def test_witness_must_be_non_negative(self):
        with pytest.raises(MultiHeadError, match="负权重"):
            type(witness_report())(
                **self._kwargs(witness=((1.5, -0.5, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)))
            )

    @pytest.mark.parametrize("field", ["one_head_distance", "multi_head_distance"])
    def test_distances_must_be_non_negative(self, field):
        with pytest.raises(ParameterError):
            type(witness_report())(**self._kwargs(**{field: -0.5}))
        with pytest.raises(ParameterError):
            type(witness_report())(**self._kwargs(**{field: float("nan")}))

    def test_notes_are_stringified(self):
        report = type(witness_report())(**self._kwargs(notes=(1, 2)))
        assert report.notes == ("1", "2")


def test_multi_head_shape_of_uses_the_partition():
    witness = designed_witness()
    shape = multi_head_shape_of(witness.params, 1)
    assert shape.head_dim == 8
    assert shape.scale == shape.single_head_scale
