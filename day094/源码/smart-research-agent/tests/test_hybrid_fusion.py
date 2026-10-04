"""day067 ``retrieval.fusion`` 的单元测试：两种融合策略的算术与它们的边界.

融合是这一课最容易"看起来对"的一层：两份名单合成一份，怎么写都像是合理的。
因此这里把两种策略的**算术本身**钉死，而不是只断言"结果不为空"：

```text
rrf      贡献 = 1/(k + rank + 1)——它是**定义**，因此期望值可以逐位写死（1/61 就是 1/61）
weighted 先 min-max 再加权——因此分段线性、端点固定、min==max 时定义为 1.0
```

三条最容易被写错的地方各有专门用例：

```text
1. 跨通道量纲     同一份 "0.9 / 7.31" 在两个策略下走的是两条完全不同的路
2. min == max     单条命中的那一路不允许归一化成 0（否则它的证据被权重乘没）
3. 去重语义       同一条出现在两路 = **合并证据**（channels 两条、贡献相加），
                  而不是"丢掉一条"
```

全部离线、确定性、零网络。
"""

from __future__ import annotations

from typing import Any

import pytest

from smart_research_agent.retrieval.errors import FusionError, QueryError
from smart_research_agent.retrieval.fusion import (
    DEFAULT_ALPHA,
    DEFAULT_RRF_K,
    FUSION_OVERRIDE_KEYS,
    FUSION_RRF,
    FUSION_STRATEGIES,
    FUSION_WEIGHTED,
    FusedHit,
    _assemble,
    contribution_counts,
    fuse,
    reciprocal_rank_fusion,
    weighted_score_fusion,
)
from smart_research_agent.retrieval.lexical import LexicalHit
from smart_research_agent.retrieval.types import (
    CHANNEL_BM25,
    CHANNEL_VECTOR,
    CHANNELS,
    RetrievalHit,
)

# --------------------------------------------------------------------------- #
# 构造助手：两路的命中都用**真实形状**（RetrievalHit / LexicalHit）
# --------------------------------------------------------------------------- #


def vector_hits(*rows: tuple[str, int, float]) -> list[RetrievalHit]:
    """``(id, rank, score)`` → 向量路的命中（``score`` 是余弦，∈ [-1, 1]）."""
    return [
        RetrievalHit(record_id=record_id, score=score, rank=rank)
        for record_id, rank, score in rows
    ]


def bm25_hits(*rows: tuple[str, int, float]) -> list[LexicalHit]:
    """``(id, rank, score)`` → 关键词路的命中（``score`` 是 BM25，∈ [0, ∞)）."""
    return [
        LexicalHit(record_id=record_id, score=score, rank=rank)
        for record_id, rank, score in rows
    ]


def two_channels() -> dict[str, list[Any]]:
    """一份"两路相反"的输入：向量路的第一名与关键词路的第一名不是同一条.

    它同时是"量纲不可比"的实证材料：向量路的分数在 ``[0, 1]`` 里，
    关键词路的分数大了一个数量级——直接相加时后者会单方面决定名次。
    """
    return {
        CHANNEL_VECTOR: vector_hits(("v-1", 0, 0.9), ("v-2", 1, 0.5)),
        CHANNEL_BM25: bm25_hits(("b-1", 0, 7.31), ("v-1", 1, 3.0)),
    }


def scores_of(hits: list[FusedHit]) -> dict[str, float]:
    """``record_id → 融合分数``（断言里最常读的一项）."""
    return {hit.record_id: hit.score for hit in hits}


def by_id(hits: list[FusedHit]) -> dict[str, FusedHit]:
    """``record_id → FusedHit``（要读证据表时用）."""
    return {hit.record_id: hit for hit in hits}


# --------------------------------------------------------------------------- #
# RRF：只看名次
# --------------------------------------------------------------------------- #


class TestReciprocalRankFusion:
    """``score = Σ 1/(k + rank + 1)``：这是定义，因此期望值可以逐位写死. """

    def test_single_channel_uses_the_definition(self) -> None:
        fused = reciprocal_rank_fusion(
            {CHANNEL_VECTOR: vector_hits(("a", 0, 0.9), ("b", 1, 0.8))}
        )
        scores = scores_of(fused)

        assert scores["a"] == pytest.approx(1.0 / 61.0, rel=1e-15)
        assert scores["b"] == pytest.approx(1.0 / 62.0, rel=1e-15)
        assert [hit.record_id for hit in fused] == ["a", "b"]

    @pytest.mark.parametrize("k", [1, 5, 10, 60, 100, 1000])
    def test_first_place_contribution_is_one_over_k_plus_one(self, k: int) -> None:
        fused = reciprocal_rank_fusion({CHANNEL_VECTOR: vector_hits(("a", 0, 0.9))}, k=k)

        assert fused[0].score == pytest.approx(1.0 / (k + 1), rel=1e-15)

    @pytest.mark.parametrize("rank", [0, 1, 2, 3, 5, 20])
    def test_contribution_decreases_with_rank(self, rank: int) -> None:
        """名次越靠后贡献越小（严格递减，且是**同一个公式**）."""
        fused = reciprocal_rank_fusion({CHANNEL_VECTOR: vector_hits(("a", rank, 0.9))})

        assert fused[0].score == pytest.approx(1.0 / (DEFAULT_RRF_K + rank + 1), rel=1e-15)

    def test_default_k_is_the_paper_value(self) -> None:
        assert DEFAULT_RRF_K == 60

    def test_a_shared_record_sums_both_contributions(self) -> None:
        """同一条被两路召回 → 两个贡献相加（这正是"最强证据"的算术表达）. """
        fused = reciprocal_rank_fusion(
            {
                CHANNEL_VECTOR: vector_hits(("shared", 0, 0.9)),
                CHANNEL_BM25: bm25_hits(("shared", 2, 4.0)),
            }
        )

        assert fused[0].record_id == "shared"
        assert fused[0].score == pytest.approx(1.0 / 61.0 + 1.0 / 63.0, rel=1e-15)

    def test_two_channel_ranking_is_hand_computable(self) -> None:
        """两路各一条、分数并列时的完整次序（含第三键 ``record_id``）. """
        fused = reciprocal_rank_fusion(
            {
                CHANNEL_VECTOR: vector_hits(("v-1", 0, 0.9)),
                CHANNEL_BM25: bm25_hits(("b-1", 0, 7.31)),
            }
        )

        # 两条都是 1/61 → 并列 → 由 record_id 决定次序（b-1 < v-1）。
        assert [hit.record_id for hit in fused] == ["b-1", "v-1"]
        assert fused[0].score == fused[1].score == pytest.approx(1.0 / 61.0, rel=1e-15)

    def test_scores_are_descending_and_ranks_contiguous(self) -> None:
        fused = reciprocal_rank_fusion(two_channels())
        scores = [hit.score for hit in fused]

        assert scores == sorted(scores, reverse=True)
        assert [hit.rank for hit in fused] == list(range(len(fused)))

    def test_contributions_sum_to_the_score(self) -> None:
        """``contributions`` 必须能解释 ``score``（否则"为什么排这里"答不出来）. """
        for hit in reciprocal_rank_fusion(two_channels()):
            assert sum(hit.contributions.values()) == pytest.approx(hit.score, rel=1e-15)

    def test_paper_formula_ignores_the_raw_scores_entirely(self) -> None:
        """RRF **不看分数**：把某一路的分数放大 100 倍，结果逐位不变. """
        original = reciprocal_rank_fusion(two_channels())
        scaled = reciprocal_rank_fusion(
            {
                CHANNEL_VECTOR: vector_hits(("v-1", 0, 0.9), ("v-2", 1, 0.5)),
                CHANNEL_BM25: bm25_hits(("b-1", 0, 731.0), ("v-1", 1, 300.0)),
            }
        )

        assert scores_of(original) == pytest.approx(scores_of(scaled), rel=1e-15)
        assert [hit.record_id for hit in original] == [hit.record_id for hit in scaled]

    def test_empty_input_returns_nothing(self) -> None:
        assert reciprocal_rank_fusion({}) == []

    def test_a_channel_with_no_hits_is_allowed(self) -> None:
        """一路一条都没召回是**正常输入**（融合必须能处理它，而不是报错）. """
        fused = reciprocal_rank_fusion(
            {CHANNEL_VECTOR: vector_hits(("a", 0, 0.9)), CHANNEL_BM25: []}
        )

        assert [hit.record_id for hit in fused] == ["a"]
        assert fused[0].channels == (CHANNEL_VECTOR,)

    def test_duplicate_ids_within_one_channel_are_rejected(self) -> None:
        with pytest.raises(FusionError) as excinfo:
            reciprocal_rank_fusion({CHANNEL_VECTOR: vector_hits(("a", 0, 0.9), ("a", 1, 0.8))})

        message = str(excinfo.value)
        assert "重复的 record_id" in message
        assert CHANNEL_VECTOR in message


class TestRRFEvidence:
    """``FusedHit`` 的三张证据表：这条凭什么在这里、每一路给了它什么. """

    def test_channels_are_sorted_by_rank_quality(self) -> None:
        """向量路 rank 3、关键词路 rank 0 → ``channels`` 里关键词在前. """
        fused = reciprocal_rank_fusion(
            {
                CHANNEL_VECTOR: vector_hits(("a", 3, 0.2)),
                CHANNEL_BM25: bm25_hits(("a", 0, 5.0)),
            }
        )

        assert fused[0].channels == (CHANNEL_BM25, CHANNEL_VECTOR)
        assert fused[0].channel_ranks == {CHANNEL_BM25: 0, CHANNEL_VECTOR: 3}

    def test_tied_ranks_put_the_vector_channel_first(self) -> None:
        """名次相同时向量优先——与 ``RetrievalHit.channel`` 的归属规则必须一致. """
        fused = reciprocal_rank_fusion(
            {
                CHANNEL_BM25: bm25_hits(("a", 0, 5.0)),
                CHANNEL_VECTOR: vector_hits(("a", 0, 0.9)),
            }
        )

        assert fused[0].channels == (CHANNEL_VECTOR, CHANNEL_BM25)
        assert fused[0].channels[0] == CHANNEL_VECTOR

    def test_raw_scores_are_kept_side_by_side(self) -> None:
        """``channel_scores`` 保留**原始**分数：两路量纲不同这件事因此可见. """
        fused = reciprocal_rank_fusion(two_channels())
        shared = by_id(fused)["v-1"]

        assert shared.channel_scores[CHANNEL_VECTOR] == pytest.approx(0.9, rel=1e-12)
        assert shared.channel_scores[CHANNEL_BM25] == pytest.approx(3.0, rel=1e-12)
        assert shared.channel_count == 2

    def test_best_rank_is_exposed(self) -> None:
        fused = reciprocal_rank_fusion(two_channels())

        assert by_id(fused)["v-1"].best_rank == 0
        assert by_id(fused)["b-1"].best_rank == 0
        assert by_id(fused)["v-2"].best_rank == 1

    def test_to_dict_round_trips_the_evidence(self) -> None:
        hit = by_id(reciprocal_rank_fusion(two_channels()))["v-1"]
        payload = hit.to_dict()

        assert set(payload) == {
            "rank",
            "record_id",
            "score",
            "channels",
            "channel_ranks",
            "channel_scores",
            "contributions",
        }
        assert payload["channels"] == [CHANNEL_VECTOR, CHANNEL_BM25]
        assert payload["rank"] == 0

    def test_to_dict_can_drop_the_evidence_tables(self) -> None:
        hit = by_id(reciprocal_rank_fusion(two_channels()))["v-1"]

        assert set(hit.to_dict(include_evidence=False)) == {
            "rank",
            "record_id",
            "score",
            "channels",
        }

    def test_summary_line_names_every_channel(self) -> None:
        summary = by_id(reciprocal_rank_fusion(two_channels()))["v-1"].summary_line()

        assert CHANNEL_VECTOR in summary
        assert CHANNEL_BM25 in summary
        assert "2 路" in summary

    def test_empty_evidence_yields_an_empty_contribution_map(self) -> None:
        """一条记录不可能没有通道（它必须来自某一路）——直接构造才触发这条校验. """
        with pytest.raises(FusionError) as excinfo:
            FusedHit(
                record_id="a",
                score=1.0,
                rank=0,
                channels=(),
                channel_ranks={CHANNEL_VECTOR: 0},
                contributions={},
            )

        assert "不自洽" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# weighted：先归一化，再加权
# --------------------------------------------------------------------------- #


class TestWeightedFusion:
    """min-max 归一化 + 加权求和：每一路的量纲在归一化之后才可比. """

    def test_min_max_is_computed_per_channel(self) -> None:
        """两路各自归一化：向量路 ``[1.0, 0.5] → [1.0, 0.0]``、关键词路 ``[7.31, 3.0] → [1, 0]``.

        重点是第二路：它的分数比第一路大一个数量级，归一化之后**两边一样大**——
        否则加权求和会变成"谁的分数量纲大谁说了算"。
        """
        fused = weighted_score_fusion(
            two_channels(),
            {CHANNEL_VECTOR: 0.5, CHANNEL_BM25: 0.5},
        )
        scores = scores_of(fused)

        assert scores["v-1"] == pytest.approx(0.5 * 1.0 + 0.5 * 0.0, rel=1e-12)
        assert scores["v-2"] == pytest.approx(0.5 * 0.0, rel=1e-12)
        assert scores["b-1"] == pytest.approx(0.5 * 1.0, rel=1e-12)

    def test_first_places_of_both_channels_tie_by_id(self) -> None:
        fused = weighted_score_fusion(
            {
                CHANNEL_VECTOR: vector_hits(("v-1", 0, 0.9)),
                CHANNEL_BM25: bm25_hits(("b-1", 0, 7.31)),
            },
            {CHANNEL_VECTOR: 0.5, CHANNEL_BM25: 0.5},
        )

        assert [hit.record_id for hit in fused] == ["b-1", "v-1"]
        assert fused[0].score == fused[1].score == pytest.approx(0.5, rel=1e-12)

    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_alpha_is_the_vector_weight_and_one_minus_alpha_the_bm25_one(
        self, alpha: float
    ) -> None:
        """两路的"唯一第一名"各自的分数恰好是 alpha 与 1-alpha（可直接读出来）. """
        fused = weighted_score_fusion(
            {
                CHANNEL_VECTOR: vector_hits(("v-1", 0, 0.9)),
                CHANNEL_BM25: bm25_hits(("b-1", 0, 7.31)),
            },
            {CHANNEL_VECTOR: alpha, CHANNEL_BM25: 1.0 - alpha},
        )
        scores = scores_of(fused)

        assert scores["v-1"] == pytest.approx(alpha, rel=1e-12)
        assert scores["b-1"] == pytest.approx(1.0 - alpha, rel=1e-12)

    @pytest.mark.parametrize(
        ("alpha", "first"),
        [(0.0, "b-1"), (0.25, "b-1"), (0.5, "b-1"), (0.75, "v-1"), (1.0, "v-1")],
    )
    def test_alpha_sweep_flips_the_order_at_the_midpoint(
        self, alpha: float, first: str
    ) -> None:
        """alpha 扫描：谁排第一随 alpha 越过 0.5 而翻转（这正是权重该有的表现）.

        ``alpha=0.5`` 时两条并列，次序由 ``record_id`` 决定（``b-1`` < ``v-1``），
        因此这一格期望的是 ``b-1``——**并列不是"随机"，它有确定的第三键**。
        """
        fused = fuse(
            {
                CHANNEL_VECTOR: vector_hits(("v-1", 0, 0.9)),
                CHANNEL_BM25: bm25_hits(("b-1", 0, 7.31)),
            },
            strategy=FUSION_WEIGHTED,
            alpha=alpha,
        )

        assert fused[0].record_id == first

    def test_a_shared_record_accumulates_from_both_channels(self) -> None:
        fused = weighted_score_fusion(
            {
                CHANNEL_VECTOR: vector_hits(("shared", 0, 1.0), ("other", 1, 0.5)),
                CHANNEL_BM25: bm25_hits(("shared", 0, 9.0), ("bm", 1, 3.0)),
            },
            {CHANNEL_VECTOR: 0.5, CHANNEL_BM25: 0.5},
        )
        shared = by_id(fused)["shared"]

        assert shared.score == pytest.approx(0.5 + 0.5, rel=1e-12)
        assert shared.channels == (CHANNEL_VECTOR, CHANNEL_BM25)
        assert sum(shared.contributions.values()) == pytest.approx(shared.score, rel=1e-12)

    def test_normalization_is_linear_in_the_middle(self) -> None:
        """分段线性：``[0, 5, 10] → [0.0, 0.5, 1.0]``（权重 1.0 时可直接读出归一化值）. """
        fused = weighted_score_fusion(
            {CHANNEL_VECTOR: vector_hits(("low", 0, 0.0), ("mid", 1, 5.0), ("high", 2, 10.0))},
            {CHANNEL_VECTOR: 1.0},
        )

        assert scores_of(fused)["low"] == pytest.approx(0.0, abs=1e-15)
        assert scores_of(fused)["mid"] == pytest.approx(0.5, rel=1e-12)
        assert scores_of(fused)["high"] == pytest.approx(1.0, rel=1e-12)

    def test_negative_scores_are_normalized_to_the_same_segment(self) -> None:
        """向量路的余弦可以是负数：归一化把 ``[-1, 1] → [0, 1]``（量纲问题在归一化里解决）. """
        fused = weighted_score_fusion(
            {CHANNEL_VECTOR: vector_hits(("neg", 0, -1.0), ("zero", 1, 0.0), ("pos", 2, 1.0))},
            {CHANNEL_VECTOR: 1.0},
        )

        assert scores_of(fused)["neg"] == pytest.approx(0.0, abs=1e-15)
        assert scores_of(fused)["zero"] == pytest.approx(0.5, rel=1e-12)
        assert scores_of(fused)["pos"] == pytest.approx(1.0, rel=1e-12)


class TestMinMaxBoundary:
    """``min == max`` 定义为 1.0（**不是除零，也不是 0**）. """

    @pytest.mark.parametrize("weight", [0.1, 0.25, 0.5, 0.9])
    def test_single_hit_channel_keeps_its_full_weight(self, weight: float) -> None:
        """一路只有一条命中时，它就是那一路上最相关的一条：贡献必须是完整的 weight.

        归一化成 0.0 会让这份**唯一的证据**被权重乘没（0 × weight = 0），
        而它明明是该路的第 1 名。
        """
        fused = weighted_score_fusion(
            {CHANNEL_VECTOR: vector_hits(("only", 0, 0.42))},
            {CHANNEL_VECTOR: weight},
        )

        assert fused[0].score == pytest.approx(weight, rel=1e-12)

    def test_all_equal_scores_are_one(self) -> None:
        """一路内部没有区分度（分数全相同）→ 全体都是 1.0（这一路没区分开它们）. """
        fused = weighted_score_fusion(
            {
                CHANNEL_BM25: bm25_hits(
                    ("a", 0, 3.0), ("b", 1, 3.0), ("c", 2, 3.0)
                )
            },
            {CHANNEL_BM25: 1.0},
        )

        assert scores_of(fused) == pytest.approx({"a": 1.0, "b": 1.0, "c": 1.0}, rel=1e-12)
        # 三条同分 → 次序完全由 record_id 决定（确定性）。
        assert [hit.record_id for hit in fused] == ["a", "b", "c"]

    def test_a_tiny_but_nonzero_span_is_stretched_to_the_full_range(self) -> None:
        """跨度非零但极小时会被拉满到 ``[0, 1]``——**min-max 的固有性质**，写在用例里.

        判据是"精确的零跨度"（``high - low <= 0.0``）而不是"小于某个 epsilon"：
        凭空的 epsilon 会变成第三个要标定的参数，而它的表现（几乎相同的文档
        突然被归一化成 0）比放大噪声更难解释。要避免这种放大请在打分那一侧处理。
        """
        tiny = 1e-15
        fused = weighted_score_fusion(
            {CHANNEL_VECTOR: vector_hits(("a", 0, 1.0), ("b", 1, 1.0 - tiny))},
            {CHANNEL_VECTOR: 1.0},
        )
        scores = scores_of(fused)

        assert scores["a"] == pytest.approx(1.0, rel=1e-12)
        assert scores["b"] == pytest.approx(0.0, abs=1e-12)

    def test_single_hit_channel_plus_another_channel(self) -> None:
        """一路单条、一路多条：单条那条拿满权重，另一路按自身区间展开. """
        fused = weighted_score_fusion(
            {
                CHANNEL_VECTOR: vector_hits(("only", 0, 0.5)),
                CHANNEL_BM25: bm25_hits(("b-1", 0, 8.0), ("b-2", 1, 4.0)),
            },
            {CHANNEL_VECTOR: 0.5, CHANNEL_BM25: 0.5},
        )
        scores = scores_of(fused)

        assert scores["only"] == pytest.approx(0.5, rel=1e-12)
        assert scores["b-1"] == pytest.approx(0.5, rel=1e-12)
        assert scores["b-2"] == pytest.approx(0.0, abs=1e-15)


class TestWeightedWeights:
    """权重表的三项校验 + "缺一路不默认为 0"这条纪律. """

    @pytest.mark.parametrize("missing", [CHANNEL_VECTOR, CHANNEL_BM25])
    def test_a_channel_without_a_weight_is_an_error(self, missing: str) -> None:
        """缺一路的权重直接报错（默认 0 会让这一路的证据**静默消失**）. """
        weights = {CHANNEL_VECTOR: 0.5, CHANNEL_BM25: 0.5}
        weights.pop(missing)

        with pytest.raises(FusionError) as excinfo:
            weighted_score_fusion(two_channels(), weights)

        message = str(excinfo.value)
        assert missing in message
        assert "没有权重" in message

    @pytest.mark.parametrize("weight", [-0.1, -1, float("nan"), float("inf"), "0.5", None, True])
    def test_bad_weight_values_are_rejected(self, weight: Any) -> None:
        with pytest.raises(FusionError):
            weighted_score_fusion(
                {CHANNEL_VECTOR: vector_hits(("a", 0, 0.9))},
                {CHANNEL_VECTOR: weight},
            )

    def test_non_mapping_weights_are_rejected(self) -> None:
        with pytest.raises(FusionError) as excinfo:
            weighted_score_fusion(
                {CHANNEL_VECTOR: vector_hits(("a", 0, 0.9))}, [0.5]
            )

        assert "映射" in str(excinfo.value)

    def test_extra_weight_keys_are_ignored(self) -> None:
        """多给一路的权重不影响结果（它不在这份输入里，因此没有可加的东西）. """
        baseline = weighted_score_fusion(
            {CHANNEL_VECTOR: vector_hits(("a", 0, 0.9))}, {CHANNEL_VECTOR: 0.5}
        )
        with_extra = weighted_score_fusion(
            {CHANNEL_VECTOR: vector_hits(("a", 0, 0.9))},
            {CHANNEL_VECTOR: 0.5, CHANNEL_BM25: 0.5},
        )

        assert scores_of(baseline) == pytest.approx(scores_of(with_extra), rel=1e-15)

    def test_weights_do_not_have_to_sum_to_one(self) -> None:
        """权重不必和为 1（它们只是一个线性组合的系数）——分数会整体缩放. """
        fused = weighted_score_fusion(
            {CHANNEL_VECTOR: vector_hits(("a", 0, 0.9))},
            {CHANNEL_VECTOR: 2.0},
        )

        assert fused[0].score == pytest.approx(2.0, rel=1e-12)

    def test_zero_weight_effectively_drops_a_channel(self) -> None:
        """``0.0`` 是**显式**地说"这一路这次不参与"（与"忘了给它权重"是两件事）. """
        fused = weighted_score_fusion(
            {
                CHANNEL_VECTOR: vector_hits(("v-1", 0, 0.9)),
                CHANNEL_BM25: bm25_hits(("b-1", 0, 7.31)),
            },
            {CHANNEL_VECTOR: 0.0, CHANNEL_BM25: 1.0},
        )

        assert scores_of(fused)["v-1"] == pytest.approx(0.0, abs=1e-15)
        assert fused[0].record_id == "b-1"


# --------------------------------------------------------------------------- #
# 统一入口 fuse
# --------------------------------------------------------------------------- #


class TestFuseDispatch:
    """``fuse`` 是唯一的入口：策略名、alpha、k_rrf、weights 的校验都在这里. """

    def test_defaults_to_rrf(self) -> None:
        """默认策略是 rrf（免标定、"跨通道量纲不可比"这件事由它绕开）. """
        assert FUSION_RRF == "rrf"
        assert FUSION_WEIGHTED == "weighted"
        assert FUSION_STRATEGIES == (FUSION_RRF, FUSION_WEIGHTED)
        assert DEFAULT_ALPHA == 0.5
        assert scores_of(fuse(two_channels())) == pytest.approx(
            scores_of(reciprocal_rank_fusion(two_channels())), rel=1e-15
        )

    @pytest.mark.parametrize("strategy", ["", "RRF", "RRF ", "rank", "bm25", None, 1])
    def test_unknown_strategy_is_rejected(self, strategy: Any) -> None:
        with pytest.raises(FusionError) as excinfo:
            fuse(two_channels(), strategy=strategy)

        assert "未知融合策略" in str(excinfo.value)
        assert "rrf" in str(excinfo.value)

    @pytest.mark.parametrize("alpha", [1.5, -0.1, 2, float("nan"), float("inf"), "0.5", None, True])
    def test_alpha_out_of_range_is_rejected(self, alpha: Any) -> None:
        """alpha 越界在任何策略下都报错（错配置不该因为换了策略而被放过）. """
        for strategy in FUSION_STRATEGIES:
            with pytest.raises(FusionError) as excinfo:
                fuse(two_channels(), strategy=strategy, alpha=alpha)

            assert "alpha" in str(excinfo.value)

    @pytest.mark.parametrize("k_rrf", [0, -1, 1.5, "60", None, True])
    def test_bad_k_rrf_is_rejected(self, k_rrf: Any) -> None:
        with pytest.raises(FusionError) as excinfo:
            fuse(two_channels(), k_rrf=k_rrf)

        assert "k_rrf" in str(excinfo.value)

    def test_rrf_accepts_a_boundary_alpha(self) -> None:
        """alpha 合法时 rrf 照常工作（它不用 alpha，但也不该因为 alpha 而拒绝）. """
        assert fuse(two_channels(), alpha=0.0)
        assert fuse(two_channels(), alpha=1.0)

    def test_rrf_rejects_weights(self) -> None:
        """RRF 的定义里没有权重，给了就报错（而不是"忽略它"）. """
        with pytest.raises(FusionError) as excinfo:
            fuse(two_channels(), weights={CHANNEL_VECTOR: 1.0, CHANNEL_BM25: 0.0})

        message = str(excinfo.value)
        assert "不接受 weights" in message
        assert "weighted" in message

    def test_weighted_uses_alpha_when_weights_are_omitted(self) -> None:
        fused = fuse(
            {
                CHANNEL_VECTOR: vector_hits(("v-1", 0, 0.9)),
                CHANNEL_BM25: bm25_hits(("b-1", 0, 7.31)),
            },
            strategy=FUSION_WEIGHTED,
            alpha=0.25,
        )

        assert scores_of(fused)["v-1"] == pytest.approx(0.25, rel=1e-12)
        assert scores_of(fused)["b-1"] == pytest.approx(0.75, rel=1e-12)

    def test_explicit_weights_win_over_alpha(self) -> None:
        fused = fuse(
            {
                CHANNEL_VECTOR: vector_hits(("v-1", 0, 0.9)),
                CHANNEL_BM25: bm25_hits(("b-1", 0, 7.31)),
            },
            strategy=FUSION_WEIGHTED,
            alpha=0.25,
            weights={CHANNEL_VECTOR: 0.9, CHANNEL_BM25: 0.1},
        )

        assert scores_of(fused)["v-1"] == pytest.approx(0.9, rel=1e-12)
        assert scores_of(fused)["b-1"] == pytest.approx(0.1, rel=1e-12)

    @pytest.mark.parametrize(
        "channel", ["bm25x", "Vector", "bm25 ", " vector", "tag", ""]
    )
    def test_unknown_channel_names_are_rejected(self, channel: str) -> None:
        """通道名是封闭清单：拼错一个名字会凭空多出一路只有一半证据的通道. """
        with pytest.raises(FusionError) as excinfo:
            fuse({channel: vector_hits(("a", 0, 0.9))})

        assert "未知通道名" in str(excinfo.value)
        assert "bm25" in str(excinfo.value)

    def test_the_closed_channel_list_is_the_two_known_channels(self) -> None:
        assert CHANNELS == (CHANNEL_VECTOR, CHANNEL_BM25)

    @pytest.mark.parametrize("strategy", FUSION_STRATEGIES)
    def test_empty_input_returns_an_empty_list(self, strategy: str) -> None:
        assert fuse({}, strategy=strategy) == []

    @pytest.mark.parametrize("strategy", FUSION_STRATEGIES)
    def test_empty_channels_return_an_empty_list(self, strategy: str) -> None:
        """两路都空（"这次一句话都没捞到"）是合法输入，不是错误. """
        assert fuse({CHANNEL_VECTOR: [], CHANNEL_BM25: []}, strategy=strategy) == []

    def test_override_keys_are_the_documented_closed_list(self) -> None:
        """``RetrievalQuery.extra`` 允许的键 —— 封闭清单（见 hybrid）。 """
        assert FUSION_OVERRIDE_KEYS == ("strategy", "alpha", "k_rrf", "weights")


class TestFuseInputValidation:
    """输入形状的错误各自给出出路（这些错误都属于调用方那一族）. """

    @pytest.mark.parametrize("bad", [[], (), "vector", 42, None])
    def test_channel_hits_must_be_a_mapping(self, bad: Any) -> None:
        with pytest.raises(FusionError) as excinfo:
            reciprocal_rank_fusion(bad)

        assert "映射" in str(excinfo.value)

    def test_hits_must_be_a_sequence(self) -> None:
        with pytest.raises(FusionError) as excinfo:
            reciprocal_rank_fusion({CHANNEL_VECTOR: {"record_id": "a"}})

        assert "序列" in str(excinfo.value)

    def test_a_string_is_not_a_sequence_of_hits(self) -> None:
        """``"abc"`` 是 Sequence，但它显然不是一组命中——它必须被拒掉. """
        with pytest.raises(FusionError):
            reciprocal_rank_fusion({CHANNEL_VECTOR: "abc"})

    def test_a_plain_dict_hit_is_rejected_with_a_hint(self) -> None:
        """传 ``SearchHit`` 也会被拒（id 藏在 ``.record.record_id`` 里）——消息要说清. """
        with pytest.raises(FusionError) as excinfo:
            reciprocal_rank_fusion({CHANNEL_VECTOR: [{"record_id": "a", "rank": 0, "score": 1.0}]})

        message = str(excinfo.value)
        assert "没有 record_id" in message
        assert "SearchHit" in message

    @pytest.mark.parametrize(
        ("field", "value"),
        [("record_id", ""), ("record_id", None), ("rank", -1), ("rank", 1.5),
         ("rank", True), ("score", "0.9"), ("score", None), ("score", float("nan"))],
    )
    def test_bad_hit_fields_are_rejected(self, field: str, value: Any) -> None:
        hit = {"record_id": "a", "rank": 0, "score": 0.9}
        hit[field] = value
        with pytest.raises(FusionError):
            reciprocal_rank_fusion({CHANNEL_VECTOR: [Hit(**hit)]})

    def test_empty_channel_name_is_rejected(self) -> None:
        with pytest.raises(FusionError):
            reciprocal_rank_fusion({"": vector_hits(("a", 0, 0.9))})

    def test_fusion_errors_are_query_errors(self) -> None:
        """``FusionError`` 继承 ``QueryError``：``except QueryError`` 能一次收全. """
        with pytest.raises(QueryError):
            fuse(two_channels(), strategy="nope")

        assert issubclass(FusionError, QueryError)

    def test_fused_hit_shape_is_validated(self) -> None:
        with pytest.raises(FusionError):
            FusedHit(record_id="", score=1.0, rank=0)
        with pytest.raises(FusionError):
            FusedHit(record_id="a", score=float("nan"), rank=0)
        with pytest.raises(FusionError):
            FusedHit(record_id="a", score=1.0, rank=-1)
        with pytest.raises(FusionError):
            FusedHit(
                record_id="a",
                score=1.0,
                rank=0,
                channel_ranks={"vector": 0},
                contributions=[],
            )


class Hit:
    """一个最小的"命中"替身（用来传形状不对的字段，见 ``test_bad_hit_fields_are_rejected``）."""

    def __init__(self, record_id: Any, rank: Any, score: Any) -> None:
        self.record_id = record_id
        self.rank = rank
        self.score = score


# --------------------------------------------------------------------------- #
# 确定性与汇总
# --------------------------------------------------------------------------- #


class TestDeterminism:
    """同一份输入任何一次调用都给出同一份输出——包括"通道的书写顺序"不影响结果. """

    def test_repeated_calls_are_identical(self) -> None:
        first = reciprocal_rank_fusion(two_channels())
        second = reciprocal_rank_fusion(two_channels())

        assert first == second
        assert [hit.to_dict() for hit in first] == [hit.to_dict() for hit in second]

    @pytest.mark.parametrize("strategy", FUSION_STRATEGIES)
    def test_channel_dict_order_does_not_matter(self, strategy: str) -> None:
        forward = fuse(two_channels(), strategy=strategy)
        reversed_order = fuse(
            {
                CHANNEL_BM25: two_channels()[CHANNEL_BM25],
                CHANNEL_VECTOR: two_channels()[CHANNEL_VECTOR],
            },
            strategy=strategy,
        )

        assert [hit.record_id for hit in forward] == [hit.record_id for hit in reversed_order]
        assert scores_of(forward) == pytest.approx(scores_of(reversed_order), rel=1e-15)

    def test_second_sort_key_best_rank_breaks_a_constructed_tie(self) -> None:
        """分数并列时先看 ``best_rank``（这里用单元级构造触发，端到端很难造出）.

        RRF 下"分数并列"通常等价于"名次的多重集相同"，因此端到端拿到的是第三键
        （``record_id``）在工作；``best_rank`` 是第二键，它的作用要在**直接构造**
        两张分数表时才能看见——这与 day066 直接调用 ``_diagnose`` 触发
        ``diversity_trimmed`` 是同一条纪律：规则不该只靠端到端路径来验证。
        """
        assembled = _assemble(
            {"x": 1.0, "y": 1.0, "z": 0.5},
            {"x": {CHANNEL_VECTOR: 3}, "y": {CHANNEL_VECTOR: 0}, "z": {CHANNEL_VECTOR: 1}},
            {"x": {CHANNEL_VECTOR: 0.1}, "y": {CHANNEL_VECTOR: 0.2}, "z": {CHANNEL_VECTOR: 0.3}},
            {"x": {CHANNEL_VECTOR: 1.0}, "y": {CHANNEL_VECTOR: 1.0}, "z": {CHANNEL_VECTOR: 0.5}},
        )

        # x 与 y 同分 → best_rank(0) < best_rank(3) → y 在前；z 分数更低 → 最后。
        assert [hit.record_id for hit in assembled] == ["y", "x", "z"]
        assert [hit.rank for hit in assembled] == [0, 1, 2]

    def test_score_is_the_primary_key_over_best_rank(self) -> None:
        """分数**优先于** ``best_rank``：名次好但分数低的那条排在后面. """
        assembled = _assemble(
            {"x": 2.0, "y": 1.0},
            {"x": {CHANNEL_VECTOR: 5}, "y": {CHANNEL_VECTOR: 0}},
            {"x": {CHANNEL_VECTOR: 0.1}, "y": {CHANNEL_VECTOR: 0.2}},
            {"x": {CHANNEL_VECTOR: 2.0}, "y": {CHANNEL_VECTOR: 1.0}},
        )

        assert [hit.record_id for hit in assembled] == ["x", "y"]


class TestContributionCounts:
    """``contribution_counts``：每一路在最终名单里参与过多少条（与"召回几条"不是一回事）. """

    def test_counts_each_channel_once_per_record(self) -> None:
        fused = reciprocal_rank_fusion(two_channels())

        assert contribution_counts(fused) == {CHANNEL_BM25: 2, CHANNEL_VECTOR: 2}

    def test_a_channel_that_contributed_nothing_is_absent(self) -> None:
        """一路一条都没召回时，它不出现在计数里——这正是"召回 0 条"与
        "贡献 0 条"必须分开看的地方：只看召回数时，"召回 3 条但一条都没进名单"
        与"召回 0 条"长得一样。"""
        fused = reciprocal_rank_fusion(
            {CHANNEL_VECTOR: vector_hits(("v-1", 0, 0.9)), CHANNEL_BM25: []}
        )
        counts = contribution_counts(fused)

        assert counts == {CHANNEL_VECTOR: 1}
        assert CHANNEL_BM25 not in counts

    def test_counts_are_sorted_by_channel_name(self) -> None:
        fused = fuse(two_channels(), strategy=FUSION_WEIGHTED)

        assert list(contribution_counts(fused)) == sorted(contribution_counts(fused))

    def test_empty_input_counts_nothing(self) -> None:
        assert contribution_counts([]) == {}

    def test_counts_match_the_evidence_tables(self) -> None:
        fused = fuse(two_channels(), strategy=FUSION_WEIGHTED)
        counts = contribution_counts(fused)

        for channel, expected in counts.items():
            actual = sum(1 for hit in fused if channel in hit.channels)
            assert actual == expected
