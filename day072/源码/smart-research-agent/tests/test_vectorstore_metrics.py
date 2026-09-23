"""day064 ``vectorstore.metrics`` 的单元测试：三度量、两口径、一条换算.

全部离线、确定性，期望值**手算**（样本向量的分量都是 0/0.5/0.6/0.8/10 这类
能手算的数，见 ``vectorstore_samples``）。这一层要钉死两件事：

```text
1. cosine 归一会抹掉模长，ip 不会
2. score（越大越近）与 distance（越小越近）之间只有一个换算式
```

第 2 条与外部库对账直接相关：Chroma 返回 ``distances``（越小越近），
FAISS 返回 ``D``（不同索引符号还不一样），本包内部只用 ``score``。
"""

from __future__ import annotations

from typing import Any

import pytest

from smart_research_agent.vectorstore.errors import FilterError, VectorError
from smart_research_agent.vectorstore.metrics import (
    METRIC_COSINE,
    METRIC_INNER_PRODUCT,
    METRIC_L2,
    METRICS,
    cosine_similarity,
    describe_metrics,
    distance,
    dot,
    norm,
    normalize_metric,
    score,
    similarity_from_distance,
    squared_l2,
)
from tests.vectorstore_samples import (
    RECORD_IDS,
    SAMPLE_QUERIES,
    brute_force_top,
    sample_records,
)

#: 一对"手算得出来"的向量：点积 0.96、平方 L2 距离 0.08、模长都是 1。
Q_TILT = (0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
V_TILT = (0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def records_by_id(metric: str) -> dict[str, Any]:
    """按 id 索引一批样本记录（避免测试里到处写列表推导）."""
    return {record.record_id: record for record in sample_records(metric=metric)}


# --------------------------------------------------------------------------- #
# 三个度量在同一对向量上的值
# --------------------------------------------------------------------------- #


def test_three_metrics_on_the_same_pair() -> None:
    """同一对向量：``ip`` 与 ``cosine`` 数值相同（模长都是 1），``l2`` 是距离."""
    assert dot(Q_TILT, V_TILT) == pytest.approx(0.96)
    assert norm(Q_TILT) == pytest.approx(1.0)

    assert score(METRIC_INNER_PRODUCT, Q_TILT, V_TILT) == pytest.approx(0.96)
    assert score(METRIC_COSINE, Q_TILT, V_TILT) == pytest.approx(0.96)
    assert score(METRIC_L2, Q_TILT, V_TILT) == pytest.approx(-0.08)

    assert squared_l2(Q_TILT, V_TILT) == pytest.approx(0.08)
    assert cosine_similarity(Q_TILT, V_TILT) == pytest.approx(0.96)

    assert distance(METRIC_COSINE, Q_TILT, V_TILT) == pytest.approx(0.04)
    assert distance(METRIC_INNER_PRODUCT, Q_TILT, V_TILT) == pytest.approx(0.04)
    assert distance(METRIC_L2, Q_TILT, V_TILT) == pytest.approx(0.08)


def test_cosine_normalizes_away_magnitude_but_ip_does_not() -> None:
    """关键对照：``r_d`` 的模长是 10 倍，于是 ``ip`` 给 10.0 而 ``cosine`` 给 1.0.

    这条断言是"提供方到底归没归一化"的探针：

    ```text
    ip  + 未归一化 → 10.0（长向量占优）
    cosine        → 1.0 （方向一致就够了，模长被约掉）
    ```
    """
    q_axis = SAMPLE_QUERIES["q_axis"]
    r_d_ip = records_by_id("ip")[RECORD_IDS[3]]
    r_d_cosine = records_by_id("cosine")[RECORD_IDS[3]]

    assert r_d_ip.vector == (10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert score(METRIC_INNER_PRODUCT, q_axis, r_d_ip.vector) == pytest.approx(10.0)

    assert r_d_cosine.vector == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert score(METRIC_COSINE, q_axis, r_d_cosine.vector) == pytest.approx(1.0)


def test_ip_and_cosine_do_not_agree_on_the_top_one() -> None:
    """归一的差别会真的改变榜首：``ip`` 选 ``r_d``，``cosine`` 选 ``r_a``.

    两者在 cosine 下并列（都 1.0），于是平局由 id 升序决定，
    这正好也让"排序规则"与"度量差异"两件事互相印证。
    """
    q_axis = SAMPLE_QUERIES["q_axis"]

    assert brute_force_top(sample_records(metric="ip"), q_axis, "ip", 1)[0][0] == RECORD_IDS[3]
    assert (
        brute_force_top(sample_records(metric="cosine"), q_axis, "cosine", 1)[0][0]
        == RECORD_IDS[0]
    )


# --------------------------------------------------------------------------- #
# score 与 distance 的换算关系
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("metric", [METRIC_COSINE, METRIC_INNER_PRODUCT])
def test_score_and_distance_are_complementary(metric: str) -> None:
    """``cosine`` / ``ip``：``1 − distance == score``（与 Chroma 的定义一致）."""
    query = SAMPLE_QUERIES["q_tilt"]
    for record in sample_records(metric=metric):
        raw = distance(metric, query, record.vector)
        assert 1.0 - raw == pytest.approx(score(metric, query, record.vector))


def test_l2_score_is_negative_distance() -> None:
    """``l2``：``score == −distance``（平方 L2 越小越近，取负即"越大越近"）."""
    query = SAMPLE_QUERIES["q_tilt"]
    for record in sample_records(metric=METRIC_L2):
        raw = distance(METRIC_L2, query, record.vector)
        assert score(METRIC_L2, query, record.vector) == pytest.approx(-raw)


def test_similarity_from_distance_for_each_metric() -> None:
    """三条换算各一条：两个 ``1 − d``、一个 ``−d``."""
    assert similarity_from_distance(METRIC_COSINE, 0.25) == pytest.approx(0.75)
    assert similarity_from_distance(METRIC_INNER_PRODUCT, 0.25) == pytest.approx(0.75)
    assert similarity_from_distance(METRIC_L2, 4.0) == pytest.approx(-4.0)


def test_similarity_from_distance_inverts_distance() -> None:
    """拿 ``distance`` 的输出折回来必须得到 ``score``（两个口径闭环）."""
    query = SAMPLE_QUERIES["q_axis"]
    for metric in METRICS:
        for record in sample_records(metric=metric):
            raw = distance(metric, query, record.vector)
            assert similarity_from_distance(metric, raw) == pytest.approx(
                score(metric, query, record.vector)
            )


# --------------------------------------------------------------------------- #
# 别名与未知值
# --------------------------------------------------------------------------- #


def test_normalize_metric_aliases() -> None:
    """``dot`` / ``euclidean`` / ``cos`` 这些写法都要收敛到规范名."""
    assert normalize_metric("dot") == METRIC_INNER_PRODUCT
    assert normalize_metric("inner-product") == METRIC_INNER_PRODUCT
    assert normalize_metric("euclidean") == METRIC_L2
    assert normalize_metric("cos") == METRIC_COSINE
    assert normalize_metric("  Cosine ") == METRIC_COSINE
    assert normalize_metric("DOT") == METRIC_INNER_PRODUCT


def test_normalize_metric_unknown_lists_options() -> None:
    """未知度量必须报错并列出可选值——静默当成默认值会变成"排序不太对"."""
    with pytest.raises(FilterError) as excinfo:
        normalize_metric("manhattan")

    message = str(excinfo.value)
    assert "manhattan" in message
    for name in METRICS:
        assert name in message


def test_score_accepts_alias() -> None:
    """``score`` 内部先规范化，因此别名在整条链路上都可用."""
    assert score("dot", Q_TILT, V_TILT) == pytest.approx(0.96)
    assert score("euclidean", Q_TILT, V_TILT) == pytest.approx(-0.08)


# --------------------------------------------------------------------------- #
# 维度不一致
# --------------------------------------------------------------------------- #


def test_dimension_mismatch_raises_vector_error() -> None:
    """维度不一致是 ``VectorError``（不是 ``ValueError``）——它指向"编码器换过"."""
    with pytest.raises(VectorError) as excinfo:
        dot((1.0, 2.0), (1.0,))

    assert "维度不一致" in str(excinfo.value)

    for call in (
        lambda: squared_l2((1.0,), (1.0, 2.0)),
        lambda: cosine_similarity((1.0,), (1.0, 2.0)),
        lambda: score(METRIC_COSINE, (1.0,), (1.0, 2.0)),
        lambda: distance(METRIC_INNER_PRODUCT, (1.0,), (1.0, 2.0)),
    ):
        with pytest.raises(VectorError):
            call()


def test_zero_vector_similarity_is_zero_not_an_error() -> None:
    """脏数据（零向量）在**查询侧**不打断整批：返回 0.0（写入侧已由类型拦掉）."""
    assert cosine_similarity((0.0, 0.0), (1.0, 0.0)) == 0.0
    assert cosine_similarity((1.0, 0.0), (0.0, 0.0)) == 0.0


# --------------------------------------------------------------------------- #
# 与独立参照实现对账
# --------------------------------------------------------------------------- #


def test_scores_match_brute_force_reference() -> None:
    """三种度量 × 两个查询，逐条与 ``brute_force_top`` 对账（含排序规则）.

    ``brute_force_top`` 是用 ``math`` 与朴素循环写的一份**独立实现**，
    不复用 ``metrics.py``——如果它也调用 ``metrics``，
    "两者一致"就只证明了"这段代码等于它自己"。
    """
    for metric in METRICS:
        records = sample_records(metric=metric)
        for query_name in ("q_axis", "q_tilt"):
            query = SAMPLE_QUERIES[query_name]
            expected = brute_force_top(records, query, metric, top_k=len(records))
            actual = sorted(
                ((record.record_id, score(metric, query, record.vector)) for record in records),
                key=lambda item: (-item[1], item[0]),
            )

            assert [record_id for record_id, _ in actual] == [
                record_id for record_id, _ in expected
            ]
            for (_, mine), (_, theirs) in zip(actual, expected):
                assert mine == pytest.approx(theirs)


def test_describe_metrics_table() -> None:
    """自述表与 ``METRICS`` 同序，且"要不要归一化"只对 cosine 为真."""
    rows = describe_metrics()

    assert [row["metric"] for row in rows] == list(METRICS)
    assert rows[0]["needs_normalization"] == "true"
    assert rows[1]["needs_normalization"] == "false"
    assert rows[2]["needs_normalization"] == "false"
    assert all(row["better"] == "越大越近" for row in rows)
