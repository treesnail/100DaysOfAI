"""day068 ``retrieval.rerank`` 的形状层与特征层单元测试（不含流水线集成）.

这一层被测的是"任何一个数都能被纸笔验算"这件事的前提：**五副形状**与**四个特征**。
因此断言分六组，每组都盯着"这个字段/这个特征被改坏之后，报告里的哪句话会变成假的"：

```text
常量与顺序   RERANK_MODES / OVERRIDE_KEYS / FEATURE_NAMES 的确切取值与顺序；
             FEATURE_WEIGHTS 之和**逐位**等于 1.0（它保证分数不越界，
             而 min_score 的可标定性建立在那条不变量上）
五副形状     每一条构造期校验的正反例，断言的都是"现象 + 出路"里的**出路**片段
特征层       覆盖率的三个分支、整句包含的词形陷阱、跨度的滑窗最小值、
             长度惩罚的两个分支（含空正文的防除零分支）
替身打分     score_pairs 的确定性、批大小无关、三个计数器在分批下的确切值
主入口       rerank_hits 的七步：参数解析 / 空输入早退 / 切窗口 / 一次调用 /
             先归一化后落刀 / 排序三键 / 装配
输入形状     _read_stage1_hit 的五条校验（字典、缺属性、坏名次、非字符串正文）
```

**刻意不写"软断言"**（``is not None`` / 只判真假）：一条软断言在实现被改坏之后
仍然会通过，它证明不了任何事。所有断言都指向具体数值、具体次序或具体错误消息片段。

期望值来自 ``tests/rerank_samples.py``（十条样本的四个特征与分数都是手算常量）。

全部离线、确定性、零网络。
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from smart_research_agent.retrieval.errors import QueryError, RerankError
from smart_research_agent.retrieval.rerank import (
    DEFAULT_RERANK_BATCH,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RERANK_TOP_N,
    DEFAULT_RERANK_WEIGHT,
    IDEAL_LEN,
    RERANK_FEATURE_NAMES,
    RERANK_MODE_BLEND,
    RERANK_MODE_REPLACE,
    RERANK_MODES,
    RERANK_OVERRIDE_KEYS,
    BaseReranker,
    CrossEncoderReranker,
    LiftProbe,
    LiftReport,
    RerankFeatures,
    RerankHit,
    RerankResult,
    rerank_hits,
)
from smart_research_agent.retrieval.types import RetrievalHit
from tests.rerank_samples import (
    HAND_FEATURES,
    HAND_RERANK_MOVED,
    HAND_RERANK_ORDER,
    HAND_RERANK_SCORES,
    HAND_RERANK_STAGE1_RANKS,
    HAND_SCORES,
    HAND_STAGE1_ORDER,
    HAND_WINDOW3_ORDER,
    LONG_TEXT,
    LONG_TEXT_LENGTH,
    PROXIMITY_ADJACENT,
    RECORD_IDS,
    RECORD_TEXTS,
    RERANK_QUERY,
    FixedReranker,
    NonDictExplainReranker,
    ShortExplainReranker,
    TextExplainReranker,
    TruncatingReranker,
    hand_features,
    stage1_hits,
)

# --------------------------------------------------------------------------- #
# 造样本的小工具（每个都只改一处，让"哪一支坏了"能一眼定位）
# --------------------------------------------------------------------------- #


def make_features(**overrides: Any) -> RerankFeatures:
    """造一副合法特征（缺省是 r-t-08 的那一组：覆盖满、整句命中、相邻）."""
    payload: dict[str, Any] = {
        "term_coverage": 1.0,
        "exact_phrase": 1.0,
        "proximity": PROXIMITY_ADJACENT,
        "length_penalty": 1.0,
    }
    payload.update(overrides)
    return RerankFeatures(**payload)


def make_hit(record_id: str = "r-t-08", **overrides: Any) -> RerankHit:
    """造一条合法重排命中（缺省：重排分 11/12、新名次 0、原来第 7、带两路证据）.

    特征按 ``record_id`` 查 ``HAND_FEATURES``；id 不合法（用例故意要造坏输入）时
    查不到，于是回落空字典——那几种输入在构造期就报错，特征值根本到不了排序。
    """
    hand = HAND_FEATURES.get(record_id, ()) if isinstance(record_id, str) else ()
    payload: dict[str, Any] = {
        "score": 11.0 / 12.0,
        "rank": 0,
        "stage1_score": 0.0,
        "stage1_rank": 7,
        "features": dict(zip(RERANK_FEATURE_NAMES, hand)),
        "channels": ("vector", "bm25"),
    }
    payload.update(overrides)
    return RerankHit(record_id=record_id, **payload)


def make_result(*hits: RerankHit, **overrides: Any) -> RerankResult:
    """造一个合法的重排结果（无命中时自动补一句 ``empty_reason``）."""
    payload: dict[str, Any] = {
        "query_text": RERANK_QUERY,
        "hits": hits,
        "candidates": max(len(hits), 1),
        "scored": max(len(hits), 1),
        "top_n": DEFAULT_RERANK_TOP_N,
        "model": DEFAULT_RERANK_MODEL,
        "mode": RERANK_MODE_REPLACE,
        "empty_reason": "" if hits else "没有候选可比",
    }
    payload.update(overrides)
    return RerankResult(**payload)


def make_probe(**overrides: Any) -> LiftProbe:
    """造一条合法探针."""
    payload: dict[str, Any] = {"query": RERANK_QUERY, "relevant": ("r-t-08",)}
    payload.update(overrides)
    return LiftProbe(**payload)


def make_report(**overrides: Any) -> LiftReport:
    """造一份合法报告（三条指标的键集合必须逐键对齐，因此三张表一起给）."""
    tables = {"recall": 0.4, "reciprocal_rank": 0.5, "ndcg": 0.6}
    payload: dict[str, Any] = {
        "queries": 2,
        "k": 5,
        "before": dict(tables),
        "after": {name: value + 0.1 for name, value in tables.items()},
        "lift": {name: 0.1 for name in tables},
    }
    payload.update(overrides)
    return LiftReport(**payload)


class FakeHit:
    """只带用例关心的那几个属性（用来触发 ``_read_stage1_hit`` 的各条校验）.

    用它而不是 ``RetrievalHit``：字典 / 缺属性 / 坏名次 / 非字符串正文这三种输入
    正常路径造不出来，而它们恰恰是"上游给错了形状"时的真实样子。
    """

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


class MinimalReranker(BaseReranker):
    """最小的合法子类：用来验证基类那两个"可选/缺省"的行为.

    它刻意**不覆盖** ``explain_pairs``——真实重排模型正是这个形状
    （分数来自隐层，没有可以摆出来的特征，返回空字典是**正确**实现）。
    """

    @property
    def name(self) -> str:
        """这个重排器的名字."""
        return "minimal"

    @property
    def dimension(self) -> int:
        """一维打分（真实交叉编码器就是一个相关性 logit）."""
        return 1

    def score_pairs(self, query: str, texts: Any) -> list[float]:
        """一律给 0.5（用例只关心形状与长度）."""
        return [0.5 for _ in texts]

    def describe(self) -> dict[str, Any]:
        """自述."""
        return {"name": "minimal"}


class BadScoreReranker(MinimalReranker):
    """返回**坏分数**的重排器（字符串 / nan）：验证"分数也会被本层校验".

    真实场景是"接进来的重排服务返回了一个字符串"或"某个模型包装里漏了除零"——
    两种都不会让列表长度出错，只会让排序变成字典序或让名次没有答案。
    """

    def __init__(self, value: Any) -> None:
        self._value = value

    def score_pairs(self, query: str, texts: Any) -> list[float]:
        """把那个坏值复制给每一条（长度仍然正确）。"""
        return [self._value for _ in texts]


# --------------------------------------------------------------------------- #
# 常量与顺序
# --------------------------------------------------------------------------- #


class TestConstants:
    """常量不是装饰：每一个都被别处引用，改动会静默改变行为."""

    def test_defaults_are_the_documented_numbers(self) -> None:
        """五个缺省值都是写进文档的数（窗口 20、权重 0.5、批 16、标准长度 240）."""
        assert DEFAULT_RERANK_MODEL == "cross-encoder-teaching-v1"
        assert DEFAULT_RERANK_TOP_N == 20
        assert DEFAULT_RERANK_WEIGHT == 0.5
        assert DEFAULT_RERANK_BATCH == 16
        assert IDEAL_LEN == 240

    def test_model_name_says_teaching(self) -> None:
        """名字里必须带 ``teaching``：报告里出现它就要一眼看出"这不是真实模型"."""
        assert "teaching" in DEFAULT_RERANK_MODEL

    def test_modes_is_a_closed_tuple_in_report_order(self) -> None:
        """两个模式的确切取值与顺序（顺序 = 报告里的顺序，不是字母序）."""
        assert RERANK_MODES == ("replace", "blend")
        assert RERANK_MODE_REPLACE == "replace"
        assert RERANK_MODE_BLEND == "blend"

    def test_override_keys_are_the_five_documented_names(self) -> None:
        """封闭清单：多一个键就报错，因此它的内容（与顺序）是接口的一部分."""
        assert RERANK_OVERRIDE_KEYS == ("enabled", "mode", "model", "top_n", "weight")

    def test_feature_names_match_the_weight_order(self) -> None:
        """四个特征的名字与顺序 = 权重的顺序（两处错位会让分数张冠李戴）."""
        assert RERANK_FEATURE_NAMES == (
            "term_coverage",
            "exact_phrase",
            "proximity",
            "length_penalty",
        )
        assert len(RERANK_FEATURE_NAMES) == len(CrossEncoderReranker.FEATURE_WEIGHTS)

    def test_feature_weights_sum_is_exactly_one(self) -> None:
        """**逐位**等于 1.0（不是"约等于"）：1/2、1/4、1/8、1/8 都可二进制精确表示."""
        weights = CrossEncoderReranker.FEATURE_WEIGHTS

        assert weights == (0.5, 0.25, 0.125, 0.125)
        running = 0.0
        for weight in weights:
            running += weight
        assert running == 1.0
        assert 0.5 + 0.25 == 0.75
        assert 0.75 + 0.125 == 0.875
        assert 0.875 + 0.125 == 1.0

    @pytest.mark.parametrize("index", [1, 2])
    def test_weights_are_monotone_decreasing(self, index: int) -> None:
        """越靠前的特征权重越大（覆盖率是主项，整句包含是强证据）."""
        weights = CrossEncoderReranker.FEATURE_WEIGHTS

        assert weights[index - 1] > weights[index]

    def test_the_two_hedge_features_share_the_smallest_weight(self) -> None:
        """邻近度与长度各占 1/8——它们是对冲项，不是决定性因素（并列而不是递减）."""
        weights = CrossEncoderReranker.FEATURE_WEIGHTS

        assert weights[2] == weights[3] == 0.125
        assert weights[1] / weights[2] == 2.0

    def test_dimension_is_four_not_one(self) -> None:
        """替身的维度是 4（四个可手算特征）——真实模型是 1，这个差异刻意暴露."""
        assert CrossEncoderReranker().dimension == len(RERANK_FEATURE_NAMES) == 4


# --------------------------------------------------------------------------- #
# FEATURE_WEIGHTS 的构造期校验（用子类改类属性，不改源码）
# --------------------------------------------------------------------------- #


class TestFeatureWeightsGuard:
    """权重表的三条校验：等长、非负有限、和逐位等于 1.0."""

    def test_a_short_table_names_both_sides(self) -> None:
        """长度不符时必须同时给出"有几项"与"有几个特征"——否则改哪边是猜的."""

        class ShortWeights(CrossEncoderReranker):
            FEATURE_WEIGHTS = (0.5, 0.5)

        with pytest.raises(RerankError) as excinfo:
            ShortWeights()

        message = str(excinfo.value)
        assert "FEATURE_WEIGHTS 有 2 项" in message
        assert "而特征有 4 个" in message
        assert "两者必须一一对应" in message

    def test_a_long_table_is_rejected_too(self) -> None:
        """多一项同样要报（多出来的那一项会静默改掉分数的量纲）."""

        class LongWeights(CrossEncoderReranker):
            FEATURE_WEIGHTS = (0.5, 0.25, 0.125, 0.125, 0.0)

        with pytest.raises(RerankError) as excinfo:
            LongWeights()

        assert "FEATURE_WEIGHTS 有 5 项" in str(excinfo.value)

    @pytest.mark.parametrize(
        ("weights", "fragment"),
        [
            ((0.5, 0.25, 0.125, "0.125"), "必须是数字"),
            ((0.5, 0.25, 0.125, None), "必须是数字"),
            ((True, 0.25, 0.125, 0.125), "必须是数字"),
            ((0.5, 0.25, 0.125, float("nan")), "必须是非负有限数"),
            ((0.5, 0.25, 0.125, float("inf")), "必须是非负有限数"),
            ((0.5, 0.25, 0.125, -0.125), "必须是非负有限数"),
        ],
    )
    def test_each_bad_weight_is_rejected(self, weights: tuple, fragment: str) -> None:
        """逐项校验：非数字 / 非有限 / 负数各自报出来（负数会让"特征越明显分越低"）."""

        class BadWeights(CrossEncoderReranker):
            FEATURE_WEIGHTS = weights

        with pytest.raises(RerankError) as excinfo:
            BadWeights()

        assert fragment in str(excinfo.value)

    def test_a_sum_that_is_not_one_is_rejected(self) -> None:
        """和必须逐位等于 1.0：它保证分数落在 [0, 1]，也就是 min_score 可标定的前提."""

        class SumTooLow(CrossEncoderReranker):
            FEATURE_WEIGHTS = (0.5, 0.25, 0.125, 0.1)

        with pytest.raises(RerankError) as excinfo:
            SumTooLow()

        message = str(excinfo.value)
        assert "之和必须逐位等于 1.0，当前是 0.975" in message
        assert "本模块刻意选二进制精确可表示的权重" in message

    def test_a_sum_with_a_tiny_float_error_is_rejected(self) -> None:
        """浮点误差级别的偏离也拒绝（不许引入一个 tolerance 去放过它）."""

        class SumOff(CrossEncoderReranker):
            FEATURE_WEIGHTS = (0.1, 0.2, 0.3, 0.40000001)

        with pytest.raises(RerankError) as excinfo:
            SumOff()

        assert "之和必须逐位等于 1.0" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# RerankFeatures
# --------------------------------------------------------------------------- #


class TestRerankFeaturesValidation:
    """四个特征各自的构造期校验：类型 / 有限 / [0, 1]（越界会让 min_score 失去含义）."""

    def test_accepts_the_documented_range(self) -> None:
        """两端的边界值都合法（0.0 与 1.0 都在闭区间里），并被收敛成 float."""
        features = RerankFeatures(
            term_coverage=0.0, exact_phrase=1, proximity=0.5, length_penalty=0.25
        )

        assert features.term_coverage == 0.0
        assert features.exact_phrase == 1.0
        assert isinstance(features.term_coverage, float)

    @pytest.mark.parametrize("name", RERANK_FEATURE_NAMES)
    @pytest.mark.parametrize("value", ["0.5", None, [0.5], {"v": 0.5}, True, False])
    def test_rejects_non_numbers(self, name: str, value: Any) -> None:
        """字符串/布尔/容器都不是特征：字符串会让加权和变成字符串拼接（不报错、只怪）."""
        with pytest.raises(RerankError) as excinfo:
            make_features(**{name: value})

        message = str(excinfo.value)
        assert f"RerankFeatures.{name} 必须是数字" in message
        assert "字符串形式的特征会让加权和变成字符串拼接" in message

    @pytest.mark.parametrize("name", RERANK_FEATURE_NAMES)
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite(self, name: str, value: float) -> None:
        """nan 参与的比较恒为 False → 名次不再由分数决定（除零是最常见的原因）."""
        with pytest.raises(RerankError) as excinfo:
            make_features(**{name: value})

        message = str(excinfo.value)
        assert f"RerankFeatures.{name} 必须是有限数" in message
        assert "请检查算这个特征的那段代码（除零是最常见的原因）" in message

    @pytest.mark.parametrize("name", RERANK_FEATURE_NAMES)
    @pytest.mark.parametrize("value", [-0.000001, -1.0, 1.000001, 2.0, -0.5])
    def test_rejects_out_of_range(self, name: str, value: float) -> None:
        """越界会让加权和超出 [0, 1]，而阈值正是建立在"分数不越界"之上."""
        with pytest.raises(RerankError) as excinfo:
            make_features(**{name: value})

        message = str(excinfo.value)
        assert f"RerankFeatures.{name}={value} 越界" in message
        assert "四个特征都必须落在 [0, 1]" in message

    def test_the_error_is_a_query_error_too(self) -> None:
        """``RerankError`` 继承 ``QueryError``：端点那条"参数问题一律 400"不必加分支."""
        with pytest.raises(RerankError) as excinfo:
            make_features(term_coverage=2.0)

        assert isinstance(excinfo.value, QueryError)


class TestRerankFeaturesProjection:
    """两个投影：``to_dict``（六位小数）与 ``summary_line``（固定顺序的四个四位小数）."""

    def test_to_dict_has_exactly_the_four_keys_in_order(self) -> None:
        """键集合**全等**且顺序 = ``RERANK_FEATURE_NAMES``（报告按它逐行对照）."""
        payload = make_features().to_dict()

        assert set(payload) == set(RERANK_FEATURE_NAMES)
        assert list(payload) == list(RERANK_FEATURE_NAMES)
        assert payload == {
            "term_coverage": 1.0,
            "exact_phrase": 1.0,
            "proximity": 0.333333,
            "length_penalty": 1.0,
        }

    def test_to_dict_rounds_to_six_places(self) -> None:
        """六个小数位：再多会把一次 diff 淹掉，再少会让相邻分数看起来相同."""
        payload = make_features(proximity=1.0 / 7.0, length_penalty=1.0 / 3.0).to_dict()

        assert payload["proximity"] == 0.142857
        assert payload["length_penalty"] == 0.333333

    def test_to_dict_is_json_friendly(self) -> None:
        """每个值都是有限 float（能被 ``json.dumps`` 直接吃掉）."""
        for value in make_features().to_dict().values():

            assert isinstance(value, float)
            assert math.isfinite(value)

    def test_summary_line_lists_the_four_values_in_order(self) -> None:
        """一行摘要必须按固定顺序列出四项——两次运行的摘要要能逐字 diff."""
        line = make_features().summary_line()

        assert line == (
            "特征：term_coverage=1.0000、exact_phrase=1.0000、"
            "proximity=0.3333、length_penalty=1.0000"
        )

    def test_frozen_shape_cannot_be_mutated(self) -> None:
        """frozen：改一个特征只能靠新建（"这份分数不曾被改过"必须成立）."""
        features = make_features()

        with pytest.raises(AttributeError):
            features.term_coverage = 0.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# RerankHit
# --------------------------------------------------------------------------- #


class TestRerankHitValidation:
    """十条构造期校验：id / 两个分数 / blended / 两个名次 / features / channels / scored."""

    def test_accepts_the_documented_shape(self) -> None:
        """一条正常命中：窗口外那条的 ``scored=False`` 与 replace 的 ``blended=None`` 都合法."""
        hit = make_hit()

        assert hit.score == 11.0 / 12.0
        assert hit.rank == 0
        assert hit.stage1_rank == 7
        assert hit.channels == ("vector", "bm25")
        assert hit.scored is True
        assert hit.blended is None

    @pytest.mark.parametrize("record_id", ["", "   ", None, 7, ["r-t-08"]])
    def test_rejects_bad_record_id(self, record_id: Any) -> None:
        """没有 id 就没有"这一条"——重排的第一步就是按 id 把名次写回去."""
        with pytest.raises(RerankError) as excinfo:
            make_hit(record_id=record_id, features={})

        assert "重排的第一步就是按 id 把名次写回去，没有 id 就没有'这一条'" in str(excinfo.value)

    @pytest.mark.parametrize("label", ["score", "stage1_score"])
    @pytest.mark.parametrize("value", ["0.5", None, [0.5], True])
    def test_rejects_non_numeric_scores(self, label: str, value: Any) -> None:
        """字符串形式的分数会让排序变成字典序比较（一个与相关性无关的顺序）."""
        with pytest.raises(RerankError) as excinfo:
            make_hit(**{label: value})

        message = str(excinfo.value)
        assert f"RerankHit.{label} 必须是数字" in message
        assert "字符串会让排序变成字典序比较" in message

    @pytest.mark.parametrize("label", ["score", "stage1_score"])
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_scores(self, label: str, value: float) -> None:
        """nan 会让这条落到任意位置，而重排的全部意义就是名次."""
        with pytest.raises(RerankError) as excinfo:
            make_hit(**{label: value})

        message = str(excinfo.value)
        assert f"RerankHit.{label} 必须是有限数" in message
        assert "nan 会让这条落到任意位置" in message

    def test_blended_zero_is_not_none(self) -> None:
        """``0.0`` 与 ``None`` 是两件事："算出来是 0"和"没算过"."""
        assert make_hit(blended=0.0, score=0.0).blended == 0.0

    @pytest.mark.parametrize("value", ["0.5", [0.5], True])
    def test_rejects_bad_blended(self, value: Any) -> None:
        """非数字的 blended 要让调用方知道"replace 模式请留 None"."""
        with pytest.raises(RerankError) as excinfo:
            make_hit(blended=value)

        message = str(excinfo.value)
        assert "RerankHit.blended 必须是数字或 None" in message
        assert "replace 模式请留 None" in message

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_rejects_non_finite_blended(self, value: float) -> None:
        """blended 是排序第一键：非有限值同样不可接受."""
        with pytest.raises(RerankError) as excinfo:
            make_hit(blended=value)

        assert "RerankHit.blended 必须是有限数" in str(excinfo.value)

    @pytest.mark.parametrize("label", ["rank", "stage1_rank"])
    @pytest.mark.parametrize("value", [-1, 1.5, "0", None, True])
    def test_rejects_bad_ranks(self, label: str, value: Any) -> None:
        """名次从 0 起且必须是整数：错一格会让报告里"第几条"整体错位."""
        with pytest.raises(RerankError) as excinfo:
            make_hit(**{label: value})

        message = str(excinfo.value)
        assert f"RerankHit.{label} 必须是非负整数" in message
        assert "名次从 1 起会让'第几条'这个问题在报告里整体错一格" in message

    @pytest.mark.parametrize("value", [None, "features", ["term_coverage"], 3])
    def test_rejects_non_dict_features(self, value: Any) -> None:
        """特征的形状必须是"按一张固定的表读得出来"的那一种."""
        with pytest.raises(RerankError) as excinfo:
            make_hit(features=value)

        assert "RerankHit.features 必须是字典" in str(excinfo.value)

    def test_rejects_unknown_feature_keys(self) -> None:
        """闭清单：清单外的键会让"这一条为什么得到这个分"无从按表读出."""
        with pytest.raises(RerankError) as excinfo:
            make_hit(features={"term_coverage": 1.0, "magic": 0.5})

        message = str(excinfo.value)
        assert "本模块不认识的键 ['magic']" in message
        assert "自定义重排器请让 explain_pairs 返回空字典" in message

    def test_unknown_keys_are_reported_sorted(self) -> None:
        """多个错键要排序后列出（报告要能被逐行 diff）."""
        with pytest.raises(RerankError) as excinfo:
            make_hit(features={"zeta": 1.0, "alpha": 1.0})

        assert "不认识的键 ['alpha', 'zeta']" in str(excinfo.value)

    def test_feature_values_are_normalized_to_float(self) -> None:
        """整数形式的特征会被收敛成 float（排序比较不再混着两种类型）."""
        hit = make_hit(features={"term_coverage": 1, "proximity": 0})

        assert hit.features == {"term_coverage": 1.0, "proximity": 0.0}
        assert all(isinstance(value, float) for value in hit.features.values())

    @pytest.mark.parametrize("value", ["1.0", None, [1.0], True])
    def test_rejects_non_numeric_feature_values(self, value: Any) -> None:
        """特征值也必须是数字（否则加权和会变成拼接或 TypeError）."""
        with pytest.raises(RerankError) as excinfo:
            make_hit(features={"term_coverage": value})

        assert "RerankHit.features['term_coverage'] 必须是数字" in str(excinfo.value)

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_rejects_non_finite_feature_values(self, value: float) -> None:
        """非有限的"分数来源"要当场报，而不是等到 explain 里打印出一个 nan."""
        with pytest.raises(RerankError) as excinfo:
            make_hit(features={"proximity": value})

        assert "RerankHit.features['proximity'] 必须是有限数" in str(excinfo.value)

    @pytest.mark.parametrize("value", [["vector"], {"vector"}, "vector", None])
    def test_rejects_non_tuple_channels(self, value: Any) -> None:
        """frozen 形状里塞一个可变对象会让"这份结果不曾被改过"不再成立."""
        with pytest.raises(RerankError) as excinfo:
            make_hit(channels=value)

        message = str(excinfo.value)
        assert "RerankHit.channels 必须是 tuple" in message
        assert "写法：channels=('vector', 'bm25')" in message

    @pytest.mark.parametrize("value", [1, 0, "yes", None])
    def test_rejects_non_bool_scored(self, value: Any) -> None:
        """``scored`` 是"到底有没有被打分"的唯一标记，非布尔写法会被当成真值用."""
        with pytest.raises(RerankError) as excinfo:
            make_hit(scored=value)

        assert "非布尔的写法会让它被当成真值用" in str(excinfo.value)


class TestRerankHitViews:
    """``moved`` / ``to_dict`` / ``summary_line``：名次差与两副投影."""

    @pytest.mark.parametrize(
        ("stage1_rank", "rank", "expected"),
        [(7, 0, 7), (0, 5, -5), (3, 3, 0), (4, 3, 1), (9, 8, 1), (8, 9, -1)],
    )
    def test_moved_is_the_rank_difference(self, stage1_rank: int, rank: int, expected: int) -> None:
        """正数 = 上移。注意它把"别人被切掉"也算进来（是名次差，不是"更相关"）."""
        assert make_hit(stage1_rank=stage1_rank, rank=rank).moved == expected

    def test_to_dict_key_set_is_exact(self) -> None:
        """键集合全等（下游按这些键取数，多一个少一个都会让报告错位）."""
        payload = make_hit().to_dict()

        assert set(payload) == {
            "rank",
            "record_id",
            "score",
            "stage1_score",
            "stage1_rank",
            "moved",
            "blended",
            "channels",
            "scored",
            "features",
        }
        assert payload["rank"] == 0
        assert payload["moved"] == 7
        assert payload["score"] == 0.916667
        assert payload["channels"] == ["vector", "bm25"]
        assert payload["scored"] is True

    def test_to_dict_can_omit_features(self) -> None:
        """``include_features=False`` 用于"只看名次怎么动"：20 条 × 4 个特征会淹掉 diff."""
        payload = make_hit().to_dict(include_features=False)

        assert "features" not in payload
        assert set(payload) == {
            "rank",
            "record_id",
            "score",
            "stage1_score",
            "stage1_rank",
            "moved",
            "blended",
            "channels",
            "scored",
        }

    def test_to_dict_keeps_blended_when_present(self) -> None:
        """blend 模式下 ``blended`` 要进投影（replace 下是 None）."""
        payload = make_hit(blended=0.346939, score=0.666667).to_dict()

        assert payload["blended"] == 0.346939

    def test_summary_line_of_a_scored_hit(self) -> None:
        """被打分过的条：一行里要有重排分、新旧名次与移动量（带符号）."""
        assert make_hit().summary_line() == "#0 r-t-08 rerank=0.9167 | stage1 #7 → #0（移动 +7）"

    def test_summary_line_of_a_blended_hit(self) -> None:
        """blend 的那一半要显式出现（否则"这个名次凭什么"少一个输入）."""
        line = make_hit(blended=0.346939, score=0.666667, rank=1, stage1_rank=5).summary_line()

        assert line == "#1 r-t-08 rerank=0.6667 blended=0.3469 | stage1 #5 → #1（移动 +4）"

    def test_summary_line_of_an_unscored_hit(self) -> None:
        """窗口外的那条：必须明说"未打分"（0.0 不会被误读成"最不相关"）."""
        line = make_hit(score=0.0, rank=7, stage1_rank=7, features={}, scored=False).summary_line()

        assert line == "#7 r-t-08 stage1 #7（score=0.0000）| **窗口外，未打分**"


# --------------------------------------------------------------------------- #
# RerankResult
# --------------------------------------------------------------------------- #


class TestRerankResultValidation:
    """十三处构造期校验：账本字段不许互相矛盾."""

    def test_accepts_a_normal_result(self) -> None:
        """一份正常结果：``weight`` 与 ``latency_ms`` 会被收敛成 float."""
        result = make_result(make_hit(), weight=1, latency_ms=3)

        assert result.count == 1
        assert isinstance(result.weight, float)
        assert isinstance(result.latency_ms, float)

    @pytest.mark.parametrize("query_text", ["", "   ", None, 7])
    def test_rejects_bad_query_text(self, query_text: Any) -> None:
        """没有查询这一侧的记录，这份名单就成了一份"不知道在问什么"的名次."""
        with pytest.raises(RerankError) as excinfo:
            make_result(make_hit(), query_text=query_text)

        assert "不知道在问什么" in str(excinfo.value)

    def test_rejects_non_tuple_hits(self) -> None:
        """frozen 形状里的命中列必须是 tuple."""
        with pytest.raises(RerankError) as excinfo:
            make_result(hits=[make_hit()])

        assert "RerankResult.hits 必须是 tuple" in str(excinfo.value)

    @pytest.mark.parametrize("mode", ["Replace", "blended", "", "repalce", None])
    def test_rejects_unknown_mode(self, mode: Any) -> None:
        """模式是封闭清单：拼错的名字会让调用方以为换了一种策略."""
        with pytest.raises(RerankError) as excinfo:
            make_result(make_hit(), mode=mode)

        message = str(excinfo.value)
        assert "可用模式是 replace、blend" in message
        assert "两者的差别正是'信不信第一阶段的分数'" in message

    @pytest.mark.parametrize("name", ["candidates", "scored", "top_n", "dropped_by_min_score"])
    @pytest.mark.parametrize("value", [-1, "2", 1.5, None, True])
    def test_rejects_bad_counts(self, name: str, value: Any) -> None:
        """四个计数各自是一笔账，负数（或非整数）说明有一笔算错了."""
        with pytest.raises(RerankError) as excinfo:
            make_result(make_hit(), **{name: value})

        message = str(excinfo.value)
        assert f"RerankResult.{name} 必须是非负整数" in message
        assert "负数说明有一笔账算错了" in message

    def test_rejects_scored_greater_than_candidates(self) -> None:
        """两个数一起读是"窗口有没有生效"的唯一证据，不一致时它什么都不说明."""
        with pytest.raises(RerankError) as excinfo:
            make_result(make_hit(), candidates=2, scored=3)

        message = str(excinfo.value)
        assert "scored=3 大于 candidates=2" in message
        assert "打分的条数不可能多于输入条数" in message

    def test_scored_equal_to_candidates_is_legal(self) -> None:
        """``scored == candidates`` 是"窗口比候选大"的正常情形（全部打分）."""
        assert make_result(make_hit(), candidates=3, scored=3).scored == 3

    @pytest.mark.parametrize("value", ["0.5", None, [0.5], True])
    def test_rejects_non_numeric_weight(self, value: Any) -> None:
        """权重必须是数字（它决定 blend 里两份分数的比例）."""
        with pytest.raises(RerankError) as excinfo:
            make_result(make_hit(), weight=value)

        assert "RerankResult.weight 必须是数字" in str(excinfo.value)

    @pytest.mark.parametrize("value", [-0.1, 1.1, 2, float("nan"), float("inf")])
    def test_rejects_weight_out_of_range(self, value: Any) -> None:
        """越界会让 blend 的两个权重连同号或超过 1（第一阶段权重是 1−weight）."""
        with pytest.raises(RerankError) as excinfo:
            make_result(make_hit(), weight=value)

        message = str(excinfo.value)
        assert "[0, 1] 里的有限数" in message
        assert "越界会让 blend 的两个权重连同号或超过 1" in message

    @pytest.mark.parametrize("value", [-0.001, float("nan"), float("inf"), float("-inf")])
    def test_rejects_bad_latency(self, value: float) -> None:
        """耗时不是负数——负数会让"这次重排贵不贵"得到一个编出来的答案."""
        with pytest.raises(RerankError) as excinfo:
            make_result(make_hit(), latency_ms=value)

        assert "RerankResult.latency_ms 必须是非负有限数" in str(excinfo.value)

    def test_rejects_hits_with_an_empty_reason(self) -> None:
        """两个字段互相矛盾会让"为什么没有命中"失去意义."""
        with pytest.raises(RerankError) as excinfo:
            make_result(make_hit(), empty_reason="没有候选")

        message = str(excinfo.value)
        assert "有 1 条命中却报告 empty_reason='没有候选'" in message
        assert "两个字段互相矛盾的结果会让'为什么没有命中'这个问题失去意义" in message

    def test_rejects_empty_hits_without_a_reason(self) -> None:
        """空名单的成因有三种，合成一句"没有结果"等于什么都没说."""
        with pytest.raises(RerankError) as excinfo:
            make_result(empty_reason="")

        assert "空名单可能是'上游没给候选'、'被重排阈值切完'、'输入本身是空的'" in str(
            excinfo.value
        )

    @pytest.mark.parametrize("note", ["", "   ", None, 7])
    def test_rejects_empty_notes(self, note: Any) -> None:
        """注记是给人读的，空串只会让报告里多一行空白."""
        with pytest.raises(RerankError) as excinfo:
            make_result(make_hit(), notes=("正常的注记", note))

        assert "注记是给人读的，空串只会让报告里多一行空白" in str(excinfo.value)


class TestRerankResultViews:
    """``count`` / ``is_empty`` / ``window`` / ``moved_ids`` / ``top`` 与三副投影."""

    def test_window_is_the_smaller_of_scored_and_top_n(self) -> None:
        """窗口 = 真正进重排器的那一段长度；``scored`` 不会大于 ``top_n``."""
        assert make_result(make_hit(), scored=3, top_n=5, candidates=10).window == 3
        assert make_result(make_hit(), scored=10, top_n=3, candidates=10).window == 3

    def test_is_empty_and_top(self) -> None:
        """空名单没有第一名（返回 None 而不是抛异常），且 ``is_empty`` 跟着它."""
        result = make_result(empty_reason="没有候选可比", candidates=0, scored=0, top_n=5)

        assert result.is_empty is True
        assert result.count == 0
        assert result.top() is None

    def test_top_returns_the_first_hit(self) -> None:
        """第一名就是 ``hits[0]``（新名次的那一份名单）."""
        first = make_hit(rank=0)
        second = make_hit(record_id="r-t-06", rank=1)

        assert make_result(first, second).top() == first

    def test_moved_ids_lists_only_the_moved_ones_in_new_order(self) -> None:
        """只列名次真的变了的（含"别人被切掉而前移"的那几条），按**新**名次顺序."""
        hits = (
            make_hit("r-t-08", rank=0, stage1_rank=7),
            make_hit("r-t-06", rank=1, stage1_rank=5),
            make_hit("r-t-03", rank=2, stage1_rank=2),
            make_hit("r-t-01", rank=3, stage1_rank=0),
        )

        assert make_result(*hits).moved_ids() == ["r-t-08", "r-t-06", "r-t-01"]

    def test_moved_ids_is_empty_when_nothing_moves(self) -> None:
        """一条都没动时必须是空列表（报告里那句"名次变化：无"靠它判定）."""
        assert make_result(make_hit(rank=7, stage1_rank=7)).moved_ids() == []

    def test_to_dict_key_set_is_exact(self) -> None:
        """顶层键集合全等（``to_summary`` 是**另一副**更小的形状，见下一条用例）."""
        payload = make_result(make_hit()).to_dict()

        assert set(payload) == {
            "query_text",
            "count",
            "candidates",
            "scored",
            "window",
            "top_n",
            "model",
            "mode",
            "weight",
            "dropped_by_min_score",
            "empty_reason",
            "latency_ms",
            "moved",
            "notes",
            "hits",
        }
        assert payload["count"] == 1
        assert payload["moved"] == ["r-t-08"]

    def test_to_dict_hits_are_nested_dicts(self) -> None:
        """嵌套的命中也要能被投影（``include_features=False`` 一路传下去）."""
        payload = make_result(make_hit()).to_dict(include_features=False)

        assert "features" not in payload["hits"][0]

    def test_to_summary_key_set_is_exact(self) -> None:
        """进上游结果的那一小段摘要：**刻意不含正文与特征**."""
        summary = make_result(make_hit(), mode=RERANK_MODE_BLEND, weight=0.5).to_summary()

        assert set(summary) == {
            "model",
            "mode",
            "params",
            "candidates",
            "scored",
            "window",
            "dropped_by_min_score",
            "moved",
            "empty_reason",
        }
        assert summary["params"] == {"top_n": DEFAULT_RERANK_TOP_N, "weight": 0.5}
        assert summary["moved"] == 1
        assert "hits" not in summary

    def test_summary_line_shape(self) -> None:
        """一行摘要：模式、模型、四笔账、命中数与前三条（没有命中时写"（无）"）."""
        line = make_result(
            make_hit("r-t-08", rank=0, stage1_rank=7),
            candidates=10,
            scored=10,
            top_n=10,
        ).summary_line()

        assert line == (
            "重排[replace] cross-encoder-teaching-v1 | 候选 10 | 打分 10（窗口 10）| "
            "阈值切掉 0 | 命中 1 | r-t-08:0.9167"
        )

    def test_summary_line_of_an_empty_result(self) -> None:
        """空名单的前三条写成"（无）"（不许在这里再抛一次异常）."""
        line = make_result(
            empty_reason="没有候选可比", candidates=0, scored=0, top_n=5
        ).summary_line()

        assert line.endswith("| 命中 0 | （无）")


class TestRerankResultExplain:
    """``explain()`` 的四段与四问的对应关系（行数与内容都钉住）."""

    def test_line_count_of_the_full_window(self) -> None:
        """十条全被打分且全都移动 → 2 行结构 + 10 行名次变化 + 1 行窗口注记 = 13 行."""
        hits = tuple(
            make_hit(record_id, rank=index, stage1_rank=HAND_RERANK_STAGE1_RANKS[index])
            for index, record_id in enumerate(HAND_RERANK_ORDER)
        )
        result = make_result(
            *hits,
            candidates=10,
            scored=10,
            top_n=10,
            notes=("只对前 N 条打分（N=top_n=10）：窗口外 0 条一条都没打分",),
        )

        assert len(result.explain()) == 13

    def test_line_count_drops_with_an_empty_window_note(self) -> None:
        """没有注记时少一行（注记是按条渲染的，不是恒定占位）.

        一条都没动时"名次变化：无"那一行**仍在**（它是一句结论，不是明细）。
        """
        result = make_result(make_hit(rank=7, stage1_rank=7), notes=())

        assert len(result.explain()) == 3
        assert result.explain()[2].startswith("名次变化：无")

    def test_first_line_is_the_window_discipline(self) -> None:
        """第一行回答"只对多少条打了分、外面还有多少条"（窗口纪律与它的代价）."""
        result = make_result(make_hit(), candidates=10, scored=3, top_n=3)

        assert result.explain()[0] == (
            "窗口：候选 10 条，只对前 top_n=3 条打分（真正打分 3 条、窗口内 3 条）；"
            "窗口外 7 条一条都没打分，因此它的名次可能本应更靠前"
        )

    @pytest.mark.parametrize(
        ("mode", "weight", "fragment"),
        [
            (
                RERANK_MODE_REPLACE,
                0.5,
                "模式：replace—— 窗口内只按重排分排序，权重不参与（同一次重排内部无需归一化）",
            ),
            (RERANK_MODE_BLEND, 0.5, "模式：blend，重排分权重 0.5（第一阶段分数权重 0.5）"),
            (RERANK_MODE_BLEND, 0.25, "第一阶段分数权重 0.75"),
        ],
    )
    def test_second_line_explains_the_mode_and_dimensions(
        self, mode: str, weight: float, fragment: str
    ) -> None:
        """第二行回答"用的是哪种模式、量纲怎么处理的"（不可比这件事怎么被绕开）."""
        lines = make_result(make_hit(), mode=mode, weight=weight).explain()

        assert fragment in lines[1]
        assert "归一化" in lines[1]

    def test_threshold_line_appears_only_when_something_was_dropped(self) -> None:
        """第三段只在真的切了东西时出现（"没切"不该占一行）."""
        kept = make_result(make_hit(), dropped_by_min_score=0).explain()
        dropped = make_result(
            make_hit(), dropped_by_min_score=3, scored=10, candidates=10
        ).explain()

        assert not any("重排阈值" in line for line in kept)
        assert (
            "重排阈值：切掉 3 条（只作用于被打分过的 10 条）；"
            "窗口外的条既不会被它切、也不会被它救" in dropped
        )

    def test_no_movement_line_is_explicit(self) -> None:
        """一条都没动时要**明说**"这不代表重排没用"——否则会被读成"重排坏了"."""
        joined = "\n".join(make_result(make_hit(rank=7, stage1_rank=7)).explain())

        assert "名次变化：无（窗口内的重排次序与第一阶段一致——" in joined
        assert "这**不代表重排没用**，只代表这次它没有改动任何一对相邻次序" in joined

    def test_empty_reason_line_is_rendered(self) -> None:
        """空结果的唯一原因必须出现在诊断里（它就是第 4 问的答案）."""
        lines = make_result(
            empty_reason="重排阈值 1.0 把窗口内的 10 条全切掉了",
            candidates=10,
            scored=10,
            top_n=10,
        ).explain()

        assert "空结果原因：重排阈值 1.0 把窗口内的 10 条全切掉了" in lines

    def test_notes_are_rendered_one_per_line(self) -> None:
        """每条注记一行（窗口纪律那条是无条件的，因此它必须出现在这里）."""
        lines = make_result(make_hit(), notes=("窗口注记", "阈值注记")).explain()

        assert "注记：窗口注记" in lines
        assert "注记：阈值注记" in lines

    def test_explain_does_not_render_the_notes_header_when_there_are_none(self) -> None:
        """没有注记时一行都不渲染（不是每一种形状都必须带注记）."""
        lines = make_result(make_hit(), notes=()).explain()

        assert not any(line.startswith("注记：") for line in lines)


# --------------------------------------------------------------------------- #
# LiftProbe / LiftReport 的形状
# --------------------------------------------------------------------------- #


class TestLiftProbeValidation:
    """``LiftProbe`` 的两条校验：查询非空 + 金标准是非空的 id 元组."""

    def test_accepts_a_normal_probe(self) -> None:
        """一条正常探针：查询与金标准都原样保留."""
        probe = make_probe(relevant=("r-t-08", "r-t-06"))

        assert probe.query == RERANK_QUERY
        assert probe.relevant == ("r-t-08", "r-t-06")

    @pytest.mark.parametrize("query", ["", "   ", None, 7, ["q"]])
    def test_rejects_bad_query(self, query: Any) -> None:
        """探针要被真的送进检索器，而空查询在检索那一层本来就会被拒."""
        with pytest.raises(RerankError) as excinfo:
            make_probe(query=query)

        assert "空查询在检索那一层本来就会被拒" in str(excinfo.value)

    @pytest.mark.parametrize("relevant", [["r-t-08"], "r-t-08", {"r-t-08"}, None])
    def test_rejects_non_tuple_relevant(self, relevant: Any) -> None:
        """frozen 形状里塞一个 list 会让"这份金标准不曾被改过"不再成立."""
        with pytest.raises(RerankError) as excinfo:
            make_probe(relevant=relevant)

        message = str(excinfo.value)
        assert "LiftProbe.relevant 必须是 tuple" in message
        assert "写法：relevant=('c-a-01', 'c-b-02')" in message

    def test_rejects_an_empty_gold_standard(self) -> None:
        """空标注会让 recall@k 变成 0/0——它算出来的 0.0 看起来像"一条都没召回到"."""
        with pytest.raises(RerankError) as excinfo:
            make_probe(relevant=())

        message = str(excinfo.value)
        assert "空标注会让 recall@k 变成 0/0" in message
        assert "出路：给这条查询标出至少一条相关记录，或把这条探针删掉" in message

    @pytest.mark.parametrize("item", ["", "   ", None, 7])
    def test_rejects_empty_ids_inside_the_gold_standard(self, item: Any) -> None:
        """空 id 永远匹配不上任何命中，而它会让分母变大、指标变小."""
        with pytest.raises(RerankError) as excinfo:
            make_probe(relevant=("r-t-08", item))

        assert "金标准里存的是记录 id，空 id 永远匹配不上任何命中" in str(excinfo.value)

    def test_to_dict_and_summary_line(self) -> None:
        """两个投影：``to_dict`` 的键集合全等；摘要里金标准的条数要写出来."""
        probe = make_probe(relevant=("r-t-08", "r-t-06"))

        assert probe.to_dict() == {"query": RERANK_QUERY, "relevant": ["r-t-08", "r-t-06"]}
        assert probe.summary_line() == (
            f"探针 {RERANK_QUERY!r} | 金标准 2 条：['r-t-08', 'r-t-06']"
        )

    def test_frozen_probe_cannot_be_mutated(self) -> None:
        """frozen：改金标准只能靠新建（它是这份报告里唯一来自外部的东西）."""
        with pytest.raises(AttributeError):
            make_probe().relevant = ("r-t-01",)  # type: ignore[misc]


class TestLiftReportShape:
    """``LiftReport`` 的形状校验与三副投影（三张指标表必须逐键对齐）."""

    def test_accepts_a_normal_report(self) -> None:
        """一份正常报告：三张表的键都是那三个指标名."""
        report = make_report()

        assert report.queries == 2
        assert report.k == 5
        assert set(report.before) == {"recall", "reciprocal_rank", "ndcg"}

    @pytest.mark.parametrize("queries", [-1, 1.5, "2", None, True])
    def test_rejects_bad_query_count(self, queries: Any) -> None:
        """它是"这份报告基于几条探针"，而均值正是在它上面算的."""
        with pytest.raises(RerankError) as excinfo:
            make_report(queries=queries)

        assert "它是'这份报告基于几条探针'，而均值正是在它上面算的" in str(excinfo.value)

    def test_zero_probes_is_legal(self) -> None:
        """空探针列表返回一份"三条指标全 0"的报告而不是报错（但 queries=0 要如实写）."""
        report = LiftReport(
            queries=0,
            k=5,
            before={"recall": 0.0, "reciprocal_rank": 0.0, "ndcg": 0.0},
            after={"recall": 0.0, "reciprocal_rank": 0.0, "ndcg": 0.0},
            lift={"recall": 0.0, "reciprocal_rank": 0.0, "ndcg": 0.0},
        )

        assert report.queries == 0
        assert report.per_query == ()

    @pytest.mark.parametrize("k", [0, -1, 1.5, "5", None, True])
    def test_rejects_bad_k(self, k: Any) -> None:
        """k=0 会让两条指标恒为 0（看起来像"重排毫无效果"）."""
        with pytest.raises(RerankError) as excinfo:
            make_report(k=k)

        message = str(excinfo.value)
        assert "k 必须是 >= 1 的整数" in message
        assert "k=0 会让两个指标恒为 0" in message

    def test_rejects_tables_that_do_not_line_up(self) -> None:
        """三张表必须逐键对齐——少一个键时"这一条提升了多少"就无法回答."""
        with pytest.raises(RerankError) as excinfo:
            make_report(lift={"recall": 0.1, "reciprocal_rank": 0.1})

        message = str(excinfo.value)
        assert "三份指标表键不一致" in message
        assert "少一个键时'这一条提升了多少'就无法回答" in message

    @pytest.mark.parametrize("label", ["before", "after", "lift"])
    @pytest.mark.parametrize("value", ["0.5", None, [0.5], float("nan"), float("inf")])
    def test_rejects_bad_metric_values(self, label: str, value: Any) -> None:
        """nan 的均值会让"这次提升是正是负"永远无法回答."""
        payload = {"recall": 0.4, "reciprocal_rank": 0.5, "ndcg": 0.6}
        payload["recall"] = value
        table = dict(payload)
        overrides = {label: table}
        if label != "before":
            overrides["before"] = dict(payload)
            overrides["before"]["recall"] = 0.4
        with pytest.raises(RerankError) as excinfo:
            make_report(**overrides)

        message = str(excinfo.value)
        assert f"LiftReport.{label}['recall'] 必须是" in message
        assert ("nan 的均值会让'这次提升是正是负'永远无法回答" in message) or (
            "必须是数字" in message
        )

    @pytest.mark.parametrize("per_query", [{"a": 1}, ["a"], "a", None])
    def test_rejects_non_tuple_per_query(self, per_query: Any) -> None:
        """逐条明细必须是 tuple（frozen 形状里的可变对象不算"不曾被改过"）."""
        with pytest.raises(RerankError) as excinfo:
            make_report(per_query=per_query)

        assert "LiftReport.per_query 必须是 tuple" in str(excinfo.value)

    def test_metrics_are_sorted(self) -> None:
        """``metrics`` 是字典序（投影里的顺序以它为准，不依赖插入顺序）."""
        report = make_report(
            before={"ndcg": 0.1, "recall": 0.2, "reciprocal_rank": 0.3},
            after={"ndcg": 0.2, "recall": 0.3, "reciprocal_rank": 0.4},
            lift={"ndcg": 0.1, "recall": 0.1, "reciprocal_rank": 0.1},
        )

        assert report.metrics == ("ndcg", "recall", "reciprocal_rank")

    def test_to_dict_key_set_is_exact(self) -> None:
        """键集合全等，且三张表按 ``metrics`` 的顺序重排."""
        report = make_report(per_query=({"query": "x"},))
        payload = report.to_dict()

        assert set(payload) == {
            "queries",
            "k",
            "metrics",
            "before",
            "after",
            "lift",
            "per_query",
        }
        assert list(payload["before"]) == list(report.metrics)
        assert payload["before"] == {"ndcg": 0.6, "recall": 0.4, "reciprocal_rank": 0.5}
        assert payload["per_query"] == [{"query": "x"}]

    def test_summary_line_shape(self) -> None:
        """一行摘要：探针数 / k / 逐指标的"前 → 后（差）"."""
        line = make_report().summary_line()

        assert line == (
            "重排提升（2 条探针 / k=5）：ndcg 0.6000 → 0.7000（+0.1000）、"
            "recall 0.4000 → 0.5000（+0.1000）、"
            "reciprocal_rank 0.5000 → 0.6000（+0.1000）"
        )

    def test_summary_line_of_an_empty_report(self) -> None:
        """没有指标时写"（无指标）"（而不是抛异常或留一个空尾巴）."""
        report = LiftReport(queries=0, k=5)

        assert report.summary_line() == "重排提升（0 条探针 / k=5）：（无指标）"


# --------------------------------------------------------------------------- #
# BaseReranker 的缺省行为
# --------------------------------------------------------------------------- #


class TestBaseRerankerContract:
    """接口只有三件必须实现的事；``explain_pairs`` 的缺省实现是**正确**的."""

    def test_cannot_instantiate_the_interface(self) -> None:
        """抽象基类不许被直接实例化（缺一件必须实现的事就不该跑起来）."""
        with pytest.raises(TypeError):
            BaseReranker()  # type: ignore[abstract]

    def test_default_explain_pairs_returns_empty_dicts(self) -> None:
        """真实模型没有"可手算的四个特征"，返回空字典是正确实现而不是偷懒."""
        assert MinimalReranker().explain_pairs("q", ["a", "b"]) == [{}, {}]

    def test_default_summary_line_reports_name_and_dimension(self) -> None:
        """基类的一行摘要含名字与维度（"这次是谁打的分"）."""
        assert MinimalReranker().summary_line() == "重排器 'minimal'（1 维打分）"

    def test_the_two_abstract_properties_are_read_only(self) -> None:
        """``name`` / ``dimension`` 是属性：调用方读它们，重排器自己决定怎么给."""
        reranker = MinimalReranker()

        assert reranker.name == "minimal"
        assert reranker.dimension == 1


# --------------------------------------------------------------------------- #
# CrossEncoderReranker：构造与自述
# --------------------------------------------------------------------------- #


class TestCrossEncoderRerankerConstruction:
    """三个构造参数与四副只读视图."""

    def test_defaults(self) -> None:
        """缺省：名字 = 教学替身名、批大小 16、标准长度 240、维度 4."""
        reranker = CrossEncoderReranker()

        assert reranker.name == DEFAULT_RERANK_MODEL
        assert reranker.batch_size == DEFAULT_RERANK_BATCH
        assert reranker.ideal_len == IDEAL_LEN
        assert reranker.dimension == 4

    def test_name_is_stripped(self) -> None:
        """名字首尾空白会被去掉（报告里不该出现" xxx "这种值）."""
        assert CrossEncoderReranker(name="  我的重排器  ").name == "我的重排器"

    @pytest.mark.parametrize("name", ["", "   ", None, 7, ["n"]])
    def test_rejects_bad_name(self, name: Any) -> None:
        """空名字会让重排这一段在报告里失名（"这次是谁打的分"变成空白）."""
        with pytest.raises(RerankError) as excinfo:
            CrossEncoderReranker(name=name)

        message = str(excinfo.value)
        assert "CrossEncoderReranker.name 必须是非空字符串" in message
        assert "空名字会让重排这一段在报告里失名" in message

    @pytest.mark.parametrize("label", ["batch_size", "ideal_len"])
    @pytest.mark.parametrize("value", [0, -1, 1.5, "16", None, True])
    def test_rejects_bad_positive_ints(self, label: str, value: Any) -> None:
        """0 会让"每批 0 条"或"所有文本都打满折"这种错法静默发生."""
        with pytest.raises(RerankError) as excinfo:
            CrossEncoderReranker(**{label: value})

        message = str(excinfo.value)
        assert f"{label} 必须是 >= 1 的整数" in message
        assert "它们各自是一个下限" in message

    def test_accepts_one_as_a_lower_bound(self) -> None:
        """下界 1 本身合法（batch_size=1 就是逐条，ideal_len=1 就是几乎全罚）."""
        reranker = CrossEncoderReranker(batch_size=1, ideal_len=1)

        assert reranker.batch_size == 1
        assert reranker.ideal_len == 1

    def test_describe_is_the_full_self_report(self) -> None:
        """自述的九个键全等：名字 / 类型 / 特征表 / 权重 / 两个参数 + 三个计数器."""
        assert CrossEncoderReranker().describe() == {
            "name": DEFAULT_RERANK_MODEL,
            "kind": "cross-encoder(teaching)",
            "features": ["term_coverage", "exact_phrase", "proximity", "length_penalty"],
            "weights": {
                "term_coverage": 0.5,
                "exact_phrase": 0.25,
                "proximity": 0.125,
                "length_penalty": 0.125,
            },
            "batch_size": 16,
            "ideal_len": 240,
            "calls": 0,
            "scored_pairs": 0,
            "batches": 0,
        }

    def test_describe_kind_says_teaching(self) -> None:
        """``kind`` 里带着 ``(teaching)``：端点上必须看得出"打分的不是真实模型"."""
        assert CrossEncoderReranker().describe()["kind"] == "cross-encoder(teaching)"

    def test_summary_line_is_exact(self) -> None:
        """一行摘要里要有"替身"与"可手算"这两个词（诚实声明不只写在 docstring 里）."""
        assert CrossEncoderReranker().summary_line() == (
            "重排器 'cross-encoder-teaching-v1'（教学级交叉编码器替身，"
            "4 维可手算特征，批大小 16）"
        )

    def test_summary_line_reflects_the_batch_size(self) -> None:
        """批大小换了要跟着变（它是纯开销旋钮，但必须看得出来）."""
        assert "批大小 4" in CrossEncoderReranker(batch_size=4).summary_line()


# --------------------------------------------------------------------------- #
# CrossEncoderReranker：四个特征
# --------------------------------------------------------------------------- #


class TestCrossEncoderRerankerFeatures:
    """四个特征的边界与两个分支：覆盖率 / 整句包含 / 跨度滑窗 / 长度惩罚."""

    @pytest.mark.parametrize("record_id", RECORD_IDS)
    def test_every_sample_record_matches_the_hand_computed_table(self, record_id: str) -> None:
        """十条样本逐条对照 ``HAND_FEATURES``（本文件最要紧的一条断言）."""
        got = CrossEncoderReranker().features_of(RERANK_QUERY, RECORD_TEXTS[record_id])
        expected = HAND_FEATURES[record_id]

        assert got.term_coverage == pytest.approx(expected[0], abs=1e-12)
        assert got.exact_phrase == pytest.approx(expected[1], abs=1e-12)
        assert got.proximity == pytest.approx(expected[2], abs=1e-12)
        assert got.length_penalty == pytest.approx(expected[3], abs=1e-12)

    def test_hand_features_helper_agrees_with_the_implementation(self) -> None:
        """样本模块的 ``hand_features`` 与被测实现给出同一组数（两边独立写出来）."""
        for record_id in RECORD_IDS:
            got = CrossEncoderReranker().features_of(RERANK_QUERY, RECORD_TEXTS[record_id])

            assert got.to_dict() == hand_features(record_id).to_dict()

    def test_coverage_full(self) -> None:
        """两个查询词元都在文档里 → 1.0（r-t-06 的正文两个都有）."""
        features = CrossEncoderReranker().features_of(RERANK_QUERY, "rerank window")

        assert features.term_coverage == 1.0

    def test_coverage_half(self) -> None:
        """只含一个词元 → 0.5（分母是**去重后的查询词元数**）."""
        assert CrossEncoderReranker().features_of(RERANK_QUERY, "window").term_coverage == 0.5

    def test_coverage_zero(self) -> None:
        """一个都不含 → 0.0（0 分不是"相关性低"，而是"这次查询完全没提到它"）."""
        assert CrossEncoderReranker().features_of(RERANK_QUERY, "cache").term_coverage == 0.0

    def test_coverage_ignores_morphology(self) -> None:
        """**词形陷阱**：``reranking`` 不等于 ``rerank``（它认不出词形变化）."""
        assert (
            CrossEncoderReranker().features_of(RERANK_QUERY, "reranking window").term_coverage
            == 0.5
        )

    def test_coverage_deduplicates_repeated_document_terms(self) -> None:
        """文档里把同一个词写三遍不会让覆盖率超过 1.0（分母在查询侧）."""
        assert (
            CrossEncoderReranker().features_of("window", "window window window").term_coverage
            == 1.0
        )

    def test_coverage_is_zero_when_the_query_has_no_terms(self) -> None:
        """查询一个词元都没有（纯标点）时给 0.0：没有证据不等于证据充分."""
        features = CrossEncoderReranker().features_of("？！…", "任何正文")

        assert features.term_coverage == 0.0
        assert features.exact_phrase == 0.0

    def test_coverage_of_a_single_token_query(self) -> None:
        """单个词元的查询：命中给 1.0，不命中给 0.0（分母是 1）."""
        reranker = CrossEncoderReranker()

        assert reranker.features_of("window", "a window here").term_coverage == 1.0
        assert reranker.features_of("window", "a token here").term_coverage == 0.0

    def test_exact_phrase_hit(self) -> None:
        """整句包含（折叠大小写）：查询原文出现在正文里 → 1.0."""
        assert (
            CrossEncoderReranker().features_of(RERANK_QUERY, "like window rerank here").exact_phrase
            == 1.0
        )

    def test_exact_phrase_folds_case(self) -> None:
        """大小写折叠：``WINDOW RERANK`` 也算命中（与 tokenize 的折叠口径一致）."""
        assert (
            CrossEncoderReranker().features_of(RERANK_QUERY, "an WINDOW RERANK here").exact_phrase
            == 1.0
        )

    def test_exact_phrase_miss_when_the_words_are_reordered(self) -> None:
        """词序反了就不算整句命中——但它仍然是**满分覆盖**（两个特征各说各的）."""
        features = CrossEncoderReranker().features_of(RERANK_QUERY, "rerank window")

        assert features.exact_phrase == 0.0
        assert features.term_coverage == 1.0

    def test_exact_phrase_miss_when_a_word_is_missing(self) -> None:
        """少一个词就是没有整句命中（哪怕那个词在文档里出现两次）."""
        assert CrossEncoderReranker().features_of(RERANK_QUERY, "window window").exact_phrase == 0.0

    def test_exact_phrase_is_zero_for_a_blank_query(self) -> None:
        """strip 之后为空串时给 0.0（空串"在"任何文本里，那会让每条都白拿这一项）.

        这一支只能在 ``_features`` 上触发：三个公开入口都会先把空查询拒掉——
        因此这里直接调私有方法，**同时**断言公开入口确实拒了它。
        """
        reranker = CrossEncoderReranker()

        assert reranker._features("   ", "任意正文").exact_phrase == 0.0
        with pytest.raises(RerankError):
            reranker.features_of("   ", "任意正文")

    def test_proximity_adjacent(self) -> None:
        """两个词紧挨着 → 跨度 2 → ``1/(1+2) = 1/3``."""
        features = CrossEncoderReranker().features_of(RERANK_QUERY, "rerank window")

        assert features.proximity == PROXIMITY_ADJACENT
        assert features.proximity == pytest.approx(0.3333333333333333, rel=1e-15)

    def test_proximity_single_token_query(self) -> None:
        """单词查询的跨度是 1 → ``1/(1+1) = 0.5``."""
        assert CrossEncoderReranker().features_of("window", "a window here").proximity == 0.5

    def test_proximity_of_a_wide_window(self) -> None:
        """两个词隔三个词 → 跨度 5 → ``1/(1+5) = 1/6``."""
        assert CrossEncoderReranker().features_of(
            RERANK_QUERY, "rerank a b c window"
        ).proximity == pytest.approx(1.0 / 6.0, rel=1e-15)

    def test_proximity_uses_the_smallest_window_not_the_first(self) -> None:
        """滑窗要取**最小**跨度：``rerank a a a rerank window`` 的最小窗口是最后两个词元.

        若实现只取"第一次覆盖齐全"的那一段，跨度会变成 6（``1/7``）而不是 2（``1/3``）。
        """
        features = CrossEncoderReranker().features_of(RERANK_QUERY, "rerank a a a rerank window")

        assert features.proximity == PROXIMITY_ADJACENT

    def test_proximity_shrinks_to_the_tightest_pair(self) -> None:
        """反向的滑窗：窗口先开到全文（两个词元隔了五个）→ 跨度 7 → ``1/8``."""
        features = CrossEncoderReranker().features_of(RERANK_QUERY, "window a b c d e rerank")

        assert features.proximity == pytest.approx(1.0 / 8.0, rel=1e-15)

    def test_proximity_is_zero_when_coverage_is_incomplete(self) -> None:
        """无法覆盖时是 0.0（不是"很远"）——它和"散落全文"必须区分得开."""
        reranker = CrossEncoderReranker()

        assert reranker.features_of(RERANK_QUERY, "window only here").proximity == 0.0
        assert reranker.features_of("a b", "a " + "x " * 40 + "b").proximity > 0.0

    def test_proximity_is_zero_for_an_empty_document(self) -> None:
        """空正文（0 个词元）时 ``_min_span`` 返回 None → 0.0（不是除零）."""
        features = CrossEncoderReranker().features_of(RERANK_QUERY, "")

        assert features.proximity == 0.0
        assert features.term_coverage == 0.0

    def test_proximity_is_zero_when_the_query_has_no_terms(self) -> None:
        """查询没有词元时段度未定义 → 0.0（0.0 不是"宽度 0"）."""
        assert CrossEncoderReranker().features_of("？！", "？！").proximity == 0.0

    @pytest.mark.parametrize(
        ("text_length", "expected"),
        [(1, 1.0), (16, 1.0), (239, 1.0), (240, 1.0), (480, 0.5), (960, 0.25)],
    )
    def test_length_penalty_branches(self, text_length: int, expected: float) -> None:
        """两个分支：短于/等于标准长度不罚（1.0），长于它按 ``240/len`` 打折."""
        assert CrossEncoderReranker().features_of("q", "a" * text_length).length_penalty == (
            pytest.approx(expected, rel=1e-15)
        )

    def test_length_penalty_of_an_empty_document(self) -> None:
        """空正文走 ``max(len, 1)`` 的防除零分支 → 240/1 = 1.0."""
        assert CrossEncoderReranker().features_of("q", "").length_penalty == 1.0

    def test_length_penalty_uses_the_injected_ideal_len(self) -> None:
        """``ideal_len`` 是唯一参数：改成 10 之后 20 字的正文打五折."""
        reranker = CrossEncoderReranker(ideal_len=10)

        assert reranker.features_of("q", "a" * 20).length_penalty == pytest.approx(0.5, rel=1e-15)
        assert reranker.features_of("q", "a" * 10).length_penalty == 1.0
        assert reranker.features_of("q", "a" * 9).length_penalty == 1.0

    def test_the_long_sample_record_is_the_only_penalised_one(self) -> None:
        """样本里只有 r-t-09 被罚：``240 / 300 = 0.8``（它的长度是写死的常量）."""
        features = CrossEncoderReranker().features_of(RERANK_QUERY, LONG_TEXT)

        assert len(LONG_TEXT) == LONG_TEXT_LENGTH == 300
        assert features.length_penalty == pytest.approx(0.8, rel=1e-15)
        assert features.length_penalty != 1.0

    def test_the_other_nine_records_are_not_penalised(self) -> None:
        """其余九条都短于 240 字 → 惩罚项全是 1.0（"短于标准长度不罚"这一支）."""
        reranker = CrossEncoderReranker()

        assert all(
            reranker.features_of(RERANK_QUERY, RECORD_TEXTS[record_id]).length_penalty == 1.0
            for record_id in RECORD_IDS
            if record_id != "r-t-09"
        )

    def test_every_feature_stays_within_the_unit_interval(self) -> None:
        """四个特征都在 [0, 1]（这是"分数不越界"的**唯一**依据）."""
        reranker = CrossEncoderReranker()
        texts = [RECORD_TEXTS[record_id] for record_id in RECORD_IDS] + ["", "x", LONG_TEXT]
        for text in texts:
            features = reranker.features_of(RERANK_QUERY, text)
            for name in RERANK_FEATURE_NAMES:

                assert 0.0 <= getattr(features, name) <= 1.0

    def test_features_of_rejects_a_non_string_text(self) -> None:
        """一段文本请直接给 str；一个批次请用 explain_pairs."""
        with pytest.raises(RerankError) as excinfo:
            CrossEncoderReranker().features_of(RERANK_QUERY, 42)

        message = str(excinfo.value)
        assert "features_of 的 text 必须是字符串" in message
        assert "一个批次请用 explain_pairs" in message

    @pytest.mark.parametrize("query", [None, 42, ["window"], b"window"])
    def test_features_of_rejects_a_non_string_query(self, query: Any) -> None:
        """查询必须是字符串（要重排一批 id 得先让它们变成命中）."""
        with pytest.raises(RerankError) as excinfo:
            CrossEncoderReranker().features_of(query, "body")

        assert "要重排一批 id 请先把它们变成命中" in str(excinfo.value)

    @pytest.mark.parametrize("query", ["", "   ", "\n", "\t"])
    def test_features_of_rejects_a_blank_query(self, query: str) -> None:
        """空查询会让"整句包含"对每条都无意义，而分数看起来仍然正常."""
        with pytest.raises(RerankError) as excinfo:
            CrossEncoderReranker().features_of(query, "body")

        message = str(excinfo.value)
        assert "空查询在重排里没有定义：请给一段非空文本" in message
        assert "一个不会报错、只会让重排失去依据的输入" in message


# --------------------------------------------------------------------------- #
# CrossEncoderReranker：打分与计数器
# --------------------------------------------------------------------------- #


class TestCrossEncoderRerankerScoring:
    """``score_pairs`` / ``explain_pairs``：确定性、批大小无关、三个计数器."""

    @pytest.mark.parametrize("record_id", RECORD_IDS)
    def test_scores_match_the_hand_computed_table(self, record_id: str) -> None:
        """十条样本的重排分逐条对照 ``HAND_SCORES``（四个特征加权求和）."""
        score = CrossEncoderReranker().score_pairs(RERANK_QUERY, [RECORD_TEXTS[record_id]])[0]

        assert score == pytest.approx(HAND_SCORES[record_id], rel=1e-12)

    def test_full_corpus_scores_are_the_hand_computed_fractions(self) -> None:
        """整批一起打分给出与逐条相同的一串数（权重顺序固定，无累加顺序问题）."""
        texts = [RECORD_TEXTS[record_id] for record_id in RECORD_IDS]
        scores = CrossEncoderReranker().score_pairs(RERANK_QUERY, texts)

        assert scores == pytest.approx(
            [HAND_SCORES[record_id] for record_id in RECORD_IDS], rel=1e-12
        )
        assert scores[0] == 0.125
        assert scores[5] == pytest.approx(2.0 / 3.0, rel=1e-15)
        assert scores[7] == pytest.approx(11.0 / 12.0, rel=1e-15)
        assert scores[8] == 0.1

    def test_empty_texts_is_legal_and_produces_no_batch(self) -> None:
        """"算过 0 批"与"算过 1 批空批"是两件事：空输入不产生批次."""
        reranker = CrossEncoderReranker(batch_size=4)

        assert reranker.score_pairs(RERANK_QUERY, []) == []
        assert reranker.calls == 1
        assert reranker.scored_pairs == 0
        assert reranker.batches == 0

    @pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 9, 10, 11, 16, 1024])
    def test_batch_size_does_not_change_any_score(self, batch_size: int) -> None:
        """批大小是**纯开销旋钮**：换它之后结果变了，一定是因为别的原因."""
        texts = [RECORD_TEXTS[record_id] for record_id in RECORD_IDS]
        scores = CrossEncoderReranker(batch_size=batch_size).score_pairs(RERANK_QUERY, texts)

        assert scores == pytest.approx(
            [HAND_SCORES[record_id] for record_id in RECORD_IDS], rel=1e-15
        )

    @pytest.mark.parametrize(
        ("batch_size", "expected_batches"),
        [(1, 10), (2, 5), (3, 4), (4, 3), (5, 2), (9, 2), (10, 1), (16, 1)],
    )
    def test_counters_under_batching(self, batch_size: int, expected_batches: int) -> None:
        """三个计数器的确切值：一次调用、十条打分、``ceil(10/批大小)`` 批."""
        reranker = CrossEncoderReranker(batch_size=batch_size)
        reranker.score_pairs(RERANK_QUERY, [RECORD_TEXTS[record_id] for record_id in RECORD_IDS])

        assert reranker.calls == 1
        assert reranker.scored_pairs == 10
        assert reranker.batches == expected_batches

    def test_counters_accumulate_across_calls(self) -> None:
        """计数器只增不减（一个实例可被多次查询复用，"这次打了多少"看差量）."""
        reranker = CrossEncoderReranker(batch_size=2)
        reranker.score_pairs(RERANK_QUERY, ["a", "b"])
        reranker.score_pairs(RERANK_QUERY, ["c"])

        assert reranker.calls == 2
        assert reranker.scored_pairs == 3
        assert reranker.batches == 2

    def test_explain_pairs_does_not_count(self) -> None:
        """``explain_pairs`` 不是打分（它只是把打分用的数摆出来），因此不记账."""
        reranker = CrossEncoderReranker()
        reranker.explain_pairs(RERANK_QUERY, ["a", "b", "c"])

        assert reranker.calls == 0
        assert reranker.scored_pairs == 0
        assert reranker.batches == 0

    def test_a_rejected_call_does_not_count(self) -> None:
        """参数不合法时连计数器都不该动（那一次调用根本没发生）."""
        reranker = CrossEncoderReranker()

        with pytest.raises(RerankError):
            reranker.score_pairs(RERANK_QUERY, [1, 2])

        assert reranker.calls == 0
        assert reranker.scored_pairs == 0

    def test_score_pairs_is_deterministic(self) -> None:
        """同一个输入在任何机器上给出逐位相同的列表（四个项、固定顺序）."""
        first = CrossEncoderReranker().score_pairs(RERANK_QUERY, [RECORD_TEXTS["r-t-08"]])
        second = CrossEncoderReranker().score_pairs(RERANK_QUERY, [RECORD_TEXTS["r-t-08"]])

        assert first == second

    def test_describe_counters_follow_the_scoring(self) -> None:
        """自述里的三个计数器与实际打分一致（端点上读到的账不能是假的）."""
        reranker = CrossEncoderReranker(batch_size=4)
        reranker.score_pairs(RERANK_QUERY, [RECORD_TEXTS[record_id] for record_id in RECORD_IDS])
        described = reranker.describe()

        assert (described["calls"], described["scored_pairs"], described["batches"]) == (1, 10, 3)

    def test_explain_pairs_matches_the_features(self) -> None:
        """``explain_pairs`` 的每一行就是 ``features_of().to_dict()``（无状态、可重放）."""
        rows = CrossEncoderReranker().explain_pairs(
            RERANK_QUERY, [RECORD_TEXTS["r-t-08"], RECORD_TEXTS["r-t-09"]]
        )

        assert rows[0] == hand_features("r-t-08").to_dict()
        assert rows[1] == hand_features("r-t-09").to_dict()
        assert rows[0]["length_penalty"] == 1.0
        assert rows[1]["length_penalty"] == 0.8

    def test_explain_pairs_of_an_empty_batch(self) -> None:
        """空批给空列表（与 ``score_pairs`` 同一口径）."""
        assert CrossEncoderReranker().explain_pairs(RERANK_QUERY, []) == []

    @pytest.mark.parametrize("texts", ["window", b"window", None, 42])
    def test_rejects_non_sequence_texts(self, texts: Any) -> None:
        """一段文本会被按字符逐个当成"一篇文章"（一个把逐字打分当批量打分的坑）."""
        with pytest.raises(RerankError) as excinfo:
            CrossEncoderReranker().score_pairs(RERANK_QUERY, texts)

        message = str(excinfo.value)
        assert "必须是序列（列表/元组）" in message
        assert "出路：传 ['第一段', '第二段']，空输入传 []" in message

    @pytest.mark.parametrize("bad", [None, 3, ["window"], {"text": "window"}])
    def test_rejects_non_string_items(self, bad: Any) -> None:
        """批次里的每一项都必须是字符串（命中对象请先取出它的 text）."""
        with pytest.raises(RerankError) as excinfo:
            CrossEncoderReranker().score_pairs(RERANK_QUERY, ["ok", bad])

        message = str(excinfo.value)
        assert "的第 1 项必须是字符串" in message
        assert "命中对象请先取出它的 text；本方法只认文本" in message

    def test_explain_pairs_validates_the_same_way(self) -> None:
        """两个入口用同一套校验（否则"哪一入口会拒"会变成一个需要记住的事）."""
        with pytest.raises(RerankError) as excinfo:
            CrossEncoderReranker().explain_pairs(RERANK_QUERY, ["ok", 3])

        assert "CrossEncoderReranker.explain_pairs 的第 1 项必须是字符串" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# rerank_hits：参数解析与空输入
# --------------------------------------------------------------------------- #


class TestRerankHitsParameters:
    """七步里的第 1 步：``None`` → 缺省值，非法参数当场报（不静默纠正）."""

    def test_none_resolves_to_the_documented_defaults(self) -> None:
        """``top_n`` / ``mode`` / ``model`` 的 ``None`` 各自落到文档里的缺省值."""
        result = rerank_hits([], RERANK_QUERY)

        assert result.top_n == DEFAULT_RERANK_TOP_N
        assert result.mode == RERANK_MODE_REPLACE
        assert result.model == DEFAULT_RERANK_MODEL
        assert result.weight == 0.0

    def test_replace_defaults_the_weight_to_zero(self) -> None:
        """replace 下 weight 缺省记 0.0：它的意思是"没有权重这件事"."""
        assert rerank_hits([], RERANK_QUERY, mode=RERANK_MODE_REPLACE).weight == 0.0

    def test_blend_defaults_the_weight_to_a_half(self) -> None:
        """blend 下缺省 0.5：不确定该信谁，先两边都不占先."""
        assert rerank_hits([], RERANK_QUERY, mode=RERANK_MODE_BLEND).weight == (
            DEFAULT_RERANK_WEIGHT
        )

    def test_model_name_only_goes_into_the_report(self) -> None:
        """``model`` 只进报告（本模块不按名字加载模型）."""
        assert rerank_hits([], RERANK_QUERY, model="ms-marco-MiniLM").model == "ms-marco-MiniLM"

    def test_model_defaults_to_the_reranker_name(self) -> None:
        """缺省取 ``reranker.name``（"这次是谁打的分"必须有一个值）."""
        result = rerank_hits([], RERANK_QUERY, reranker=TruncatingReranker(name="我的模型"))

        assert result.model == "我的模型"

    @pytest.mark.parametrize("top_n", [0, -1, -100, 1.5, "3", True])
    def test_rejects_bad_top_n(self, top_n: Any) -> None:
        """"只对前 0 条打分"不是一个请求；关掉重排要走 ``rerank_enabled=False``."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([], RERANK_QUERY, top_n=top_n)

        message = str(excinfo.value)
        assert "top_n 必须是 >= 1 的整数" in message
        assert "要关掉重排请把 rerank_enabled 设为 False" in message

    def test_top_n_none_means_the_documented_default(self) -> None:
        """``None`` 不是"非法值"而是"没指定"：它落到 ``DEFAULT_RERANK_TOP_N``."""
        assert rerank_hits(stage1_hits(), RERANK_QUERY, top_n=None).top_n == DEFAULT_RERANK_TOP_N

    def test_top_n_one_is_the_extreme_but_legal(self) -> None:
        """``top_n=1`` 合法：窗口里只有第一条（它拿到窗口外候选最好的名次）."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=1)

        assert result.scored == 1
        assert result.window == 1
        assert result.hits[0].record_id == "r-t-01"
        assert result.hits[0].rank == 0

    @pytest.mark.parametrize("mode", ["Replace", "mod", "", "rrf", 3])
    def test_rejects_unknown_mode(self, mode: Any) -> None:
        """拼错的模式名会让调用方以为换了一种策略，而结果看起来"只是没什么变化"."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([], RERANK_QUERY, mode=mode)

        message = str(excinfo.value)
        assert "可用模式是 replace、blend" in message
        assert "拼错的模式名会让调用方以为换了一种策略" in message

    def test_mode_none_means_replace(self) -> None:
        """``None`` 不是"非法值"而是"没指定"：它落到 ``replace``（"先看清重排把谁动了"）."""
        assert rerank_hits(stage1_hits(), RERANK_QUERY, mode=None).mode == RERANK_MODE_REPLACE

    @pytest.mark.parametrize("weight", [-0.1, 1.1, 2])
    def test_rejects_weight_out_of_range(self, weight: float) -> None:
        """越界要给出一条可执行的出路（"只信重排分"该写成什么）."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([], RERANK_QUERY, weight=weight)

        message = str(excinfo.value)
        assert f"weight={float(weight)} 越界" in message
        assert "要表达'只信重排分'请用 mode='replace' 或 weight=1.0" in message

    @pytest.mark.parametrize("weight", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_weight(self, weight: float) -> None:
        """nan 参与的比较恒为 False → blend 的加权和会整列变成 nan."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([], RERANK_QUERY, weight=weight)

        assert "nan 参与的比较恒为 False，于是 blend 的加权和会整列变成 nan" in str(excinfo.value)

    @pytest.mark.parametrize("weight", ["0.5", [0.5], {"w": 1}])
    def test_rejects_non_numeric_weight(self, weight: Any) -> None:
        """字符串形式的权重会被 ``float()`` 悄悄转出来，因此先判类型."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([], RERANK_QUERY, weight=weight)

        assert "weight 必须是数字或 None" in str(excinfo.value)

    @pytest.mark.parametrize("weight", [True, False])
    def test_rejects_bool_weight(self, weight: bool) -> None:
        """布尔是 int 的子类：必须显式拒掉，否则 ``True`` 会被当成 1.0 接受."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([], RERANK_QUERY, weight=weight)

        assert "weight 必须是数字或 None" in str(excinfo.value)

    @pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 1.0, 1, 0])
    def test_accepts_the_documented_weight_range(self, weight: Any) -> None:
        """两端都合法并被收敛成 float（1.0 = 只信重排分，0.0 = 只信第一阶段）."""
        result = rerank_hits([], RERANK_QUERY, mode=RERANK_MODE_BLEND, weight=weight)

        assert result.weight == float(weight)
        assert isinstance(result.weight, float)

    @pytest.mark.parametrize("min_score", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_min_score(self, min_score: float) -> None:
        """nan 与任何分数比较都返回 False → 一条都不会留下，而它在配置里像是个阈值."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([], RERANK_QUERY, min_score=min_score)

        assert "nan 与任何分数比较都返回 False" in str(excinfo.value)

    @pytest.mark.parametrize("min_score", ["0.5", [0.5], {"m": 1}])
    def test_rejects_non_numeric_min_score(self, min_score: Any) -> None:
        """阈值也必须是数字或 None（字符串会被 ``float()`` 悄悄转出来）."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([], RERANK_QUERY, min_score=min_score)

        assert "min_score 必须是数字或 None" in str(excinfo.value)

    @pytest.mark.parametrize("min_score", [-1.0, 0.0, 1.5])
    def test_accepts_any_finite_min_score(self, min_score: float) -> None:
        """阈值本身不设范围（重排分不一定落在 [0, 1]），只要求是有限数."""
        assert rerank_hits([], RERANK_QUERY, min_score=min_score).is_empty is True

    @pytest.mark.parametrize("elapsed", ["1", [1], {"e": 1}])
    def test_rejects_non_numeric_elapsed(self, elapsed: Any) -> None:
        """它是"调用方已经量好的毫秒数"，没有量过请留 ``None``（本函数自己量）."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([], RERANK_QUERY, elapsed=elapsed)

        assert "它是'调用方已经量好的毫秒数'，没有量过请留 None" in str(excinfo.value)

    @pytest.mark.parametrize("elapsed", [float("nan"), -1, -0.001, float("-inf")])
    def test_rejects_bad_elapsed(self, elapsed: Any) -> None:
        """耗时不是负数；负数会让"这次重排贵不贵"得到一个编出来的答案."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([], RERANK_QUERY, elapsed=elapsed)

        assert "耗时不是负数" in str(excinfo.value)

    def test_elapsed_is_used_verbatim_when_given(self) -> None:
        """调用方量好了就用它的（重排是整条链路上最贵的一步，账要能对上）."""
        assert rerank_hits([], RERANK_QUERY, elapsed=12.34567).latency_ms == 12.346

    def test_latency_is_measured_when_not_given(self) -> None:
        """没给就自己量：至少是 0.0（不许出现负数）."""
        assert rerank_hits(stage1_hits(), RERANK_QUERY).latency_ms >= 0.0

    @pytest.mark.parametrize("model", ["", "   ", 3, ["m"]])
    def test_rejects_bad_model_name(self, model: Any) -> None:
        """空白最容易被读成"没有重排"，因此空串要当场拒掉."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([], RERANK_QUERY, model=model)

        message = str(excinfo.value)
        assert "model 必须是非空字符串或 None" in message
        assert "空白最容易被读成'没有重排'" in message

    @pytest.mark.parametrize("query", ["", "   ", None, 42, ["q"]])
    def test_rejects_bad_query(self, query: Any) -> None:
        """查询必须是字符串且非空白（与 ``RetrievalQuery.text`` 同一条纪律）."""
        with pytest.raises(RerankError):
            rerank_hits([], query)

    @pytest.mark.parametrize("reranker", ["cross-encoder", 3, lambda q, t: [1.0]])
    def test_rejects_a_non_base_reranker(self, reranker: Any) -> None:
        """不要传一个函数或一个字符串模型名——本模块**不按名字加载模型**."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([], RERANK_QUERY, reranker=reranker)

        message = str(excinfo.value)
        assert "reranker 必须是 BaseReranker" in message
        assert "不要传一个函数或一个字符串模型名" in message

    @pytest.mark.parametrize("hits", ["hits", b"hits", 3, None, {"a": 1}])
    def test_rejects_non_sequence_hits(self, hits: Any) -> None:
        """要重排一批纯文本请直接用 ``reranker.score_pairs``（不是把它传进来）."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits(hits, RERANK_QUERY)

        message = str(excinfo.value)
        assert "rerank_hits 需要一批命中（序列）" in message
        assert "要重排一批纯文本请直接用 reranker.score_pairs(query, texts)" in message

    def test_a_tuple_of_hits_is_accepted(self) -> None:
        """元组也是序列（``RetrievalResult.hits`` 就是元组）."""
        assert rerank_hits(tuple(stage1_hits()), RERANK_QUERY).candidates == 10


class TestRerankHitsInputShapes:
    """``_read_stage1_hit`` 的五条校验：上游给错形状时要说清缺的是哪一样."""

    def test_a_plain_dict_is_rejected_with_a_way_out(self) -> None:
        """字典能取到那几个键，但"错误的用法活到排序那一步"更糟——要当场响."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([{"record_id": "r-t-01", "score": 1.0, "rank": 0}], RERANK_QUERY)

        message = str(excinfo.value)
        assert "第 0 条命中 没有 record_id 属性（收到 dict）" in message
        assert "出路：传这两者之一，或用 reranker.score_pairs 直接对纯文本打分" in message

    def test_a_hit_missing_score_names_the_attribute(self) -> None:
        """缺 ``score`` 时要点名（"重排只要求命中提供 record_id / score / rank"）."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([FakeHit(record_id="r-t-01", rank=0, text="t")], RERANK_QUERY)

        assert "第 0 条命中 没有 score 属性" in str(excinfo.value)

    def test_a_blank_record_id_is_rejected(self) -> None:
        """空 id 无法把名次写回去（它是重排的第一步）."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits(
                [FakeHit(record_id="  ", score=1.0, rank=0, text="t")], RERANK_QUERY
            )

        assert "第 0 条命中 的 record_id 必须是非空字符串，收到 '  '" in str(excinfo.value)

    @pytest.mark.parametrize("rank", [-1, 1.5, "0", True])
    def test_a_bad_rank_is_rejected(self, rank: Any) -> None:
        """``rank`` 会被原样记进 ``stage1_rank``（"它原来在第几"的答案）."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits(
                [FakeHit(record_id="r-t-01", score=1.0, rank=rank, text="t")], RERANK_QUERY
            )

        message = str(excinfo.value)
        assert "的 rank 必须是非负整数（从 0 起）" in message
        assert "它会被原样记进 stage1_rank（'它原来在第几'的答案）" in message

    def test_a_non_string_text_is_rejected(self) -> None:
        """重排分是对 (查询, 正文) 算的，没有正文就无从打分（空串是合法的）."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([FakeHit(record_id="r-t-01", score=1.0, rank=0, text=7)], RERANK_QUERY)

        message = str(excinfo.value)
        assert "的 text 必须是字符串" in message
        assert "空串是合法的，它会让这条特征全为 0" in message

    def test_a_missing_text_falls_back_to_an_empty_string(self) -> None:
        """没有 ``text`` 属性时回落空串（合法的"这条没有正文"）."""
        result = rerank_hits([FakeHit(record_id="r-t-01", score=1.0, rank=0)], RERANK_QUERY)

        assert result.hits[0].score == 0.125
        assert result.hits[0].features == {"term_coverage": 0.0, "exact_phrase": 0.0,
                                           "proximity": 0.0, "length_penalty": 1.0}

    @pytest.mark.parametrize("channels", [["vector"], "vector", None, 3])
    def test_a_non_tuple_channels_is_rejected(self, channels: Any) -> None:
        """它带着上游的多路证据（未记录时用空元组），重排要原样传下去."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits(
                [FakeHit(record_id="r-t-01", score=1.0, rank=0, text="t", channels=channels)],
                RERANK_QUERY,
            )

        message = str(excinfo.value)
        assert "的 channels 必须是 tuple" in message
        assert "未记录时用空元组 ()" in message

    def test_a_missing_channels_falls_back_to_an_empty_tuple(self) -> None:
        """没有 ``channels`` 属性时回落空元组（单路语义：这个形状不知道通道这件事）."""
        result = rerank_hits([FakeHit(record_id="r-t-01", score=1.0, rank=0, text="t")],
                             RERANK_QUERY)

        assert result.hits[0].channels == ()

    def test_stage1_rank_is_the_position_not_the_upstream_rank(self) -> None:
        """``stage1_rank`` 记的是**输入序列的位置**（上游那个 rank 可能带空洞）."""
        hit = RetrievalHit(record_id="r-t-01", score=1.0, rank=100, text=RECORD_TEXTS["r-t-01"])
        result = rerank_hits([hit], RERANK_QUERY)

        assert result.hits[0].stage1_rank == 0
        assert result.hits[0].stage1_score == 1.0
        assert result.hits[0].moved == 0

    def test_a_custom_reranker_controls_the_order(self) -> None:
        """自定重排器（按 id 给分）决定次序：重排分才是权威次序."""
        reranker = FixedReranker({"r-t-03": 9.0, "r-t-09": 5.0, "r-t-01": 1.0})
        result = rerank_hits(stage1_hits(), RERANK_QUERY, reranker=reranker, top_n=10)

        assert [hit.record_id for hit in result.hits[:3]] == ["r-t-03", "r-t-09", "r-t-01"]
        assert result.hits[0].score == 9.0
        assert result.hits[0].stage1_rank == 2
        assert result.hits[0].stage1_score == 0.6

    def test_score_vector_length_mismatch_is_caught(self) -> None:
        """分数与文本错位**不会报错**（两个都是列表），因此必须由本层拦下."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits(stage1_hits(), RERANK_QUERY, reranker=TruncatingReranker())

        message = str(excinfo.value)
        assert "给了 9 个分数" in message
        assert "而窗口里有 10 条文本：长度不一致" in message
        assert "出路：让 score_pairs 返回与 texts 等长同序的列表" in message

    def test_a_non_sequence_explain_pairs_is_caught(self) -> None:
        """真实模型没有可摆出的特征时返回 ``{}`` 就是正确实现（返回字符串不是）."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits(stage1_hits(), RERANK_QUERY, reranker=TextExplainReranker())

        message = str(excinfo.value)
        assert "的 explain_pairs 返回了 str" in message
        assert "真实模型没有可摆出的特征时返回 {} 就是正确实现" in message

    def test_a_short_explain_pairs_is_caught(self) -> None:
        """特征行数比文本少时，特征会错位到别的条上——必须拦下."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits(stage1_hits(), RERANK_QUERY, reranker=ShortExplainReranker())

        message = str(excinfo.value)
        assert "的 explain_pairs 返回 9 行特征" in message
        assert "出路：让它的长度与 texts 一致" in message

    def test_a_non_dict_explain_row_is_caught(self) -> None:
        """每一项必须是一个 dict（否则"哪一项对应哪个特征"的对应关系就丢了）."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits(stage1_hits(), RERANK_QUERY, reranker=NonDictExplainReranker())

        message = str(excinfo.value)
        assert "的 explain_pairs 返回了 str 项" in message
        assert "而每一项必须是一个 dict（键落在 RERANK_FEATURE_NAMES 里）" in message

    def test_a_minimal_reranker_yields_empty_feature_rows(self) -> None:
        """只实现三件必做之事也能跑：特征那几行是空字典，分数照样可用."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, reranker=MinimalReranker(), top_n=10)

        assert all(hit.features == {} for hit in result.hits)
        assert all(hit.score == 0.5 for hit in result.hits)
        assert [hit.record_id for hit in result.hits] == list(HAND_STAGE1_ORDER)

    @pytest.mark.parametrize("score", ["1.0", None, [1.0]])
    def test_a_non_numeric_stage_one_score_is_rejected(self, score: Any) -> None:
        """上游给的分数是字符串时，排序会变成字典序比较——必须当场拒掉."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([FakeHit(record_id="r-t-01", score=score, rank=0, text="t")], RERANK_QUERY)

        message = str(excinfo.value)
        assert "第 0 条命中 的 score 必须是数字" in message
        assert "字符串形式的分数会被排序当成字典序比较——一个与相关性无关的顺序" in message

    @pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
    def test_a_non_finite_stage_one_score_is_rejected(self, score: float) -> None:
        """nan 会让"这条排第几"变成未定义（它比任何数都既不大于也不小于）."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits([FakeHit(record_id="r-t-01", score=score, rank=0, text="t")], RERANK_QUERY)

        message = str(excinfo.value)
        assert "第 0 条命中 的 score 必须是有限数" in message
        assert "nan 会让'这条排第几'变成未定义" in message

    def test_a_bad_score_from_the_reranker_is_rejected(self) -> None:
        """重排器返回的分数也要过同一套校验（它可能来自一个坏掉的服务包装）."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits(stage1_hits(), RERANK_QUERY, reranker=BadScoreReranker("0.9"))

        message = str(excinfo.value)
        assert "score_pairs 必须是数字" in message
        assert "字符串形式的分数会被排序当成字典序比较" in message

    def test_a_non_finite_score_from_the_reranker_is_rejected(self) -> None:
        """nan 会让名次变成"没有答案"（比"排错了"更糟：它不会报错，只会乱序）."""
        with pytest.raises(RerankError) as excinfo:
            rerank_hits(stage1_hits(), RERANK_QUERY, reranker=BadScoreReranker(float("nan")))

        message = str(excinfo.value)
        assert "score_pairs 必须是有限数" in message
        assert "nan 会让'这条排第几'变成未定义" in message


# --------------------------------------------------------------------------- #
# rerank_hits：窗口纪律
# --------------------------------------------------------------------------- #


class TestRerankHitsWindow:
    """七步里的第 3、4 步：只对前 N 条打分，窗口外一条都不进来."""

    @pytest.mark.parametrize(
        ("top_n", "expected_scored"),
        [(1, 1), (2, 2), (3, 3), (5, 5), (9, 9), (10, 10), (20, 10)],
    )
    def test_the_window_is_the_first_top_n_candidates(
        self, top_n: int, expected_scored: int
    ) -> None:
        """窗口 = 前 ``top_n`` 条；``scored`` 恰好是 ``min(候选, top_n)``."""
        reranker = CrossEncoderReranker()
        result = rerank_hits(stage1_hits(), RERANK_QUERY, reranker=reranker, top_n=top_n)

        assert result.candidates == 10
        assert result.scored == expected_scored
        assert result.window == expected_scored
        assert reranker.scored_pairs == expected_scored
        assert reranker.calls == 1

    def test_top_n_larger_than_candidates_scores_everything(self) -> None:
        """窗口比候选大时全部打分（并且不会因为"取不满"而报错）."""
        reranker = CrossEncoderReranker()
        result = rerank_hits(stage1_hits(), RERANK_QUERY, reranker=reranker, top_n=100)

        assert result.scored == 10
        assert result.window == 10
        assert reranker.scored_pairs == 10
        assert all(hit.scored for hit in result.hits)

    def test_the_tail_is_never_scored(self) -> None:
        """``top_n=3`` 时尾部**一条都没打分**：``scored=False``、``score=0.0``、``features={}``."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=3)
        tail = result.hits[3:]

        assert [hit.record_id for hit in tail] == [
            "r-t-04",
            "r-t-05",
            "r-t-06",
            "r-t-07",
            "r-t-08",
            "r-t-09",
            "r-t-10",
        ]
        for hit in tail:

            assert hit.scored is False
            assert hit.score == 0.0
            assert hit.features == {}
            assert hit.blended is None

    def test_the_tail_keeps_its_input_order_and_rank(self) -> None:
        """窗口外的条按**原顺序**接在重排结果之后，且 ``stage1_rank`` 保持原值."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=3)
        tail = result.hits[3:]

        assert [hit.rank for hit in tail] == [3, 4, 5, 6, 7, 8, 9]
        assert [hit.stage1_rank for hit in tail] == [3, 4, 5, 6, 7, 8, 9]
        assert all(hit.moved == 0 for hit in tail)

    def test_scored_pairs_equals_the_window_not_the_candidates(self) -> None:
        """**窗口纪律的账**：打分的条数由 top_n 决定，与候选数无关."""
        reranker = CrossEncoderReranker(batch_size=2)
        rerank_hits(stage1_hits(), RERANK_QUERY, reranker=reranker, top_n=3)

        assert reranker.scored_pairs == 3
        assert reranker.batches == 2

    def test_the_window_order_is_hand_computable(self) -> None:
        """``top_n=3`` 的完整名单 = 手算常量（窗口内换序 + 窗口外原序）."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=3)

        assert [hit.record_id for hit in result.hits] == list(HAND_WINDOW3_ORDER)

    def test_the_window_moves_records_within_itself(self) -> None:
        """窗口内确实发生了换序（r-t-02 从第 2 名升到第 1 名，r-t-01 降一位）."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=3)

        assert result.hits[0].record_id == "r-t-02"
        assert result.hits[0].stage1_rank == 1
        assert result.hits[0].rank == 0
        assert result.hits[1].record_id == "r-t-01"
        assert result.moved_ids() == ["r-t-02", "r-t-01"]

    def test_window_note_is_unconditional(self) -> None:
        """窗口纪律那条注记**无条件**出现（它说的是"这次重排看不见什么"）."""
        for top_n in (1, 3, 10, 100):
            result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=top_n)

            assert any("一条都没打分" in note for note in result.notes)

    def test_tail_note_appears_only_when_there_is_a_tail(self) -> None:
        """尾部注记只在真有尾部时出现（窗口覆盖全部候选时它不该占一行）."""
        with_tail = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=3)
        without_tail = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10)

        assert any("按**原顺序**接在重排结果之后" in note for note in with_tail.notes)
        assert not any("按**原顺序**接在重排结果之后" in note for note in without_tail.notes)

    def test_tail_note_points_at_top_n_as_the_remedy(self) -> None:
        """"要让它有机会被提前，请调大 top_n"——被限制的能力必须说清怎么放开."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=5)
        note = next(note for note in result.notes if "按**原顺序**" in note)

        assert "窗口外的 5 条" in note
        assert "scored=False、score=0.0" in note
        assert "不是'最不相关'，而是'没有分'" in note

    def test_window_note_counts_the_exact_numbers(self) -> None:
        """注记里的三个数（候选 / 窗口内 / 窗口外）必须与账本一致."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=4)
        note = result.notes[0]

        assert "N=top_n=4" in note
        assert "候选 10 条" in note
        assert "窗口内 4 条全部打过分" in note
        assert "窗口外的 6 条**一条都没打分**" in note


# --------------------------------------------------------------------------- #
# rerank_hits：排序、归一化与阈值
# --------------------------------------------------------------------------- #


class TestRerankHitsOrdering:
    """七步里的第 5~7 步：排序三键、blend 的归一化基准、阈值只切被打分的条."""

    def test_replace_order_matches_the_hand_computed_list(self) -> None:
        """主案例：第一阶段第 7 名被顶到第 0 名（十个分数逐项对照手算常量）."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10)

        assert [hit.record_id for hit in result.hits] == list(HAND_RERANK_ORDER)
        assert [round(hit.score, 6) for hit in result.hits] == list(HAND_RERANK_SCORES)
        assert [hit.stage1_rank for hit in result.hits] == list(HAND_RERANK_STAGE1_RANKS)
        assert [hit.moved for hit in result.hits] == list(HAND_RERANK_MOVED)

    def test_ranks_are_contiguous_from_zero(self) -> None:
        """整份名单一起编号（"第 3 条"必须只有一个含义）."""
        assert [hit.rank for hit in rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10).hits] == (
            list(range(10))
        )

    def test_all_ten_records_move(self) -> None:
        """这次重排把十条全都动了（``moved_ids`` 长度 10）——重排确实生效了."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10)

        assert len(result.moved_ids()) == 10
        assert result.to_summary()["moved"] == 10

    def test_the_exact_phrase_record_wins_by_a_quarter(self) -> None:
        """整句命中（+0.25）把 r-t-06 与 r-t-08 拉开：11/12 对 2/3，差恰好 0.25."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10)
        scores = {hit.record_id: hit.score for hit in result.hits}

        assert scores["r-t-08"] == pytest.approx(11.0 / 12.0, rel=1e-15)
        assert scores["r-t-06"] == pytest.approx(2.0 / 3.0, rel=1e-15)
        assert scores["r-t-08"] - scores["r-t-06"] == pytest.approx(0.25, rel=1e-12)

    def test_ties_are_broken_by_stage1_rank(self) -> None:
        """三个 0.375 **精确并列**：输出次序 = 输入次序（第二键是 stage1_rank）."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10)

        assert result.hits[2].score == result.hits[3].score == result.hits[4].score == 0.375
        assert [hit.record_id for hit in result.hits if hit.score == 0.375] == [
            "r-t-02",
            "r-t-05",
            "r-t-07",
        ]

    def test_tie_order_follows_the_input_not_the_record_id(self) -> None:
        """**第二键的判别用例**：把输入序改成与 id 序相反，并列的次序必须跟着输入走.

        若实现把第二键写成 ``record_id``，这一次的输出会退回 id 升序
        （r-t-02 / r-t-05 / r-t-07），而正确答案是"跟着输入序"（r-t-07 / r-t-05 / r-t-02）。
        """
        result = rerank_hits(list(reversed(stage1_hits())), RERANK_QUERY, top_n=10)

        assert [hit.record_id for hit in result.hits if hit.score == 0.375] == [
            "r-t-07",
            "r-t-05",
            "r-t-02",
        ]

    def test_blend_order_matches_the_hand_computed_list(self) -> None:
        """blend 的名单与加权分逐项对照手算常量（归一化在窗口内做）."""
        result = rerank_hits(
            stage1_hits(), RERANK_QUERY, top_n=10, mode=RERANK_MODE_BLEND, weight=0.5
        )

        assert [hit.record_id for hit in result.hits] != list(HAND_RERANK_ORDER)
        assert [hit.record_id for hit in result.hits] == [
            "r-t-02",
            "r-t-01",
            "r-t-08",
            "r-t-06",
            "r-t-03",
            "r-t-05",
            "r-t-07",
            "r-t-04",
            "r-t-10",
            "r-t-09",
        ]
        assert [round(hit.blended, 6) for hit in result.hits] == [
            0.568367,
            0.515306,
            0.5,
            0.346939,
            0.315306,
            0.168367,
            0.168367,
            0.015306,
            0.015306,
            0.0,
        ]

    def test_blend_records_the_blended_value_and_keeps_the_rerank_score(self) -> None:
        """blend 下两个分数各占一个字段：排序看 ``blended``，报告看 ``score``."""
        top = rerank_hits(
            stage1_hits(), RERANK_QUERY, top_n=10, mode=RERANK_MODE_BLEND, weight=0.5
        ).hits[0]

        assert top.record_id == "r-t-02"
        assert top.score == pytest.approx(0.375, rel=1e-12)
        assert top.blended == pytest.approx(0.568367, rel=1e-6)

    def test_blend_normalises_over_the_whole_scored_window(self) -> None:
        """**归一化基准是窗口内全部被打分的条**，且发生在 ``min_score`` 落刀之前.

        这一条用 ``r-t-06`` 钉住：它的加权分 0.346939 只有在
        "min=0.1（r-t-09）/ max=11/12（r-t-08）"这一组基准下才成立。
        若实现先切掉 < 0.2 的条再归一化，基准会变成 0.375 / 0.916667，
        同一个 r-t-06 会得到 0.269231——一个一眼可辨的差。
        """
        result = rerank_hits(
            stage1_hits(),
            RERANK_QUERY,
            top_n=10,
            mode=RERANK_MODE_BLEND,
            weight=0.5,
            min_score=0.2,
        )
        by_id = {hit.record_id: hit for hit in result.hits}

        assert [hit.record_id for hit in result.hits] == [
            "r-t-02",
            "r-t-08",
            "r-t-06",
            "r-t-05",
            "r-t-07",
        ]
        assert by_id["r-t-06"].blended == pytest.approx(0.3469387755, rel=1e-9)
        assert by_id["r-t-06"].blended != pytest.approx(0.269231, rel=1e-6)
        assert result.scored == 10
        assert result.window == 10
        assert result.dropped_by_min_score == 5

    def test_blend_weight_one_means_only_the_rerank_score(self) -> None:
        """``weight=1.0`` 时 blend 退化成"只看归一化后的重排分" → 次序等于 replace."""
        blend = rerank_hits(
            stage1_hits(), RERANK_QUERY, top_n=10, mode=RERANK_MODE_BLEND, weight=1.0
        )
        replace = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10)

        assert [hit.record_id for hit in blend.hits] == [hit.record_id for hit in replace.hits]
        assert blend.hits[0].blended == pytest.approx(1.0, rel=1e-12)

    def test_blend_weight_zero_means_only_stage_one(self) -> None:
        """``weight=0.0`` 时 blend 完全丢掉重排分 → 次序回到第一阶段（并列按原名次）."""
        result = rerank_hits(
            stage1_hits(), RERANK_QUERY, top_n=10, mode=RERANK_MODE_BLEND, weight=0.0
        )

        assert [hit.record_id for hit in result.hits] == list(HAND_STAGE1_ORDER)
        assert [hit.blended for hit in result.hits] == pytest.approx(
            [1.0, 0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-12
        )

    def test_replace_ignores_the_weight_but_records_it(self) -> None:
        """replace 不看权重，但**记录**它并由 notes 说明——报错会让"试一下 blend"很别扭."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10, weight=0.25)

        assert result.weight == 0.25
        assert all(hit.blended is None for hit in result.hits)
        assert any("mode='replace' 不看权重" in note for note in result.notes)
        assert any("weight=0.25 已记录在结果里但没有参与排序" in note for note in result.notes)

    def test_min_score_only_drops_scored_hits(self) -> None:
        """阈值只作用于被打分过的条：这次切掉 8 条，只剩两个高分."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10, min_score=0.5)

        assert [hit.record_id for hit in result.hits] == ["r-t-08", "r-t-06"]
        assert result.dropped_by_min_score == 8
        assert result.scored == 10
        note = next(note for note in result.notes if note.startswith("min_score=0.5"))

        assert "只作用于被打分过的 10 条" in note
        assert "本次切掉 8 条" in note

    def test_min_score_is_inclusive_of_the_boundary(self) -> None:
        """阈值用的是"小于才切"（``<`` 而不是 ``<=``）：恰好相等的那条留下."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10, min_score=0.375)

        assert [hit.record_id for hit in result.hits] == [
            "r-t-08",
            "r-t-06",
            "r-t-02",
            "r-t-05",
            "r-t-07",
        ]
        assert result.dropped_by_min_score == 5

    def test_min_score_never_touches_the_tail(self) -> None:
        """窗口外的条不受阈值影响（它们连分数都没有），但会跟着一起留/走."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=3, min_score=0.5)

        assert [hit.record_id for hit in result.hits] == [
            "r-t-04",
            "r-t-05",
            "r-t-06",
            "r-t-07",
            "r-t-08",
            "r-t-09",
            "r-t-10",
        ]
        assert result.dropped_by_min_score == 3
        assert all(hit.scored is False for hit in result.hits)
        assert result.empty_reason == ""

    def test_threshold_emptying_the_window_gives_an_explicit_reason(self) -> None:
        """阈值把窗口切空时，``empty_reason`` 必须说清"是阈值太高、不是重排失败"."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10, min_score=1.0)

        assert result.hits == ()
        assert result.is_empty is True
        assert result.dropped_by_min_score == 10
        assert "重排阈值 1.0 把窗口内的 10 条全切掉了" in result.empty_reason
        assert "而窗口外没有别的候选：请降低重排阈值或不设阈值" in result.empty_reason
        assert any("而是阈值定得比全部分数都高" in note for note in result.notes)

    def test_threshold_note_reports_the_exact_count(self) -> None:
        """阈值注记里的两个数字（作用范围 / 切掉几条）都必须是实数."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=4, min_score=0.3)
        note = next(note for note in result.notes if note.startswith("min_score=0.3"))

        assert "只作用于被打分过的 4 条" in note
        assert "本次切掉 3 条" in note
        assert "与关键词路 BM25 的'没有绝对标度'正好相反" in note

    def test_model_name_mismatch_goes_into_the_notes(self) -> None:
        """报告里的模型名与实现的名字不一致时必须说清"打分用的还是实现那个"."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, model="ms-marco-MiniLM")

        assert any(
            "与实现的名字 'cross-encoder-teaching-v1' 不一致" in note for note in result.notes
        )
        assert any("本模块**不按名字加载模型**" in note for note in result.notes)

    def test_no_model_note_when_the_names_agree(self) -> None:
        """名字一致时不加这条注记（它只在"可能被误读"时才有价值）."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, model=DEFAULT_RERANK_MODEL)

        assert not any("不一致" in note for note in result.notes)

    def test_features_are_copied_into_every_scored_hit(self) -> None:
        """窗口内每条的四个特征都要带出来（"这个分是怎么来的"必须能逐项核对）."""
        result = rerank_hits(stage1_hits(), RERANK_QUERY, top_n=10)
        by_id = {hit.record_id: hit for hit in result.hits}

        assert by_id["r-t-08"].features == hand_features("r-t-08").to_dict()
        assert by_id["r-t-09"].features == {
            "term_coverage": 0.0,
            "exact_phrase": 0.0,
            "proximity": 0.0,
            "length_penalty": 0.8,
        }

    def test_channels_survive_the_rerank(self) -> None:
        """通道证据原样带过去（融合的多路证据不能在重排这一步丢掉）."""
        channels = {record_id: ("vector", "bm25") for record_id in RECORD_IDS}
        result = rerank_hits(stage1_hits(channels=channels), RERANK_QUERY, top_n=10)

        assert all(hit.channels == ("vector", "bm25") for hit in result.hits)

    def test_ordering_uses_the_raw_score_not_the_rounded_one(self) -> None:
        """两个分数在第六位上才分得开时，次序仍由**原始**分数决定（不是四舍五入后的）."""
        reranker = FixedReranker({"r-t-02": 0.30000004, "r-t-05": 0.30000001})
        result = rerank_hits(stage1_hits(), RERANK_QUERY, reranker=reranker, top_n=10)

        assert result.hits[0].record_id == "r-t-02"
        assert result.hits[1].record_id == "r-t-05"
        assert result.hits[0].to_dict()["score"] == 0.3
        assert result.hits[1].to_dict()["score"] == 0.3
