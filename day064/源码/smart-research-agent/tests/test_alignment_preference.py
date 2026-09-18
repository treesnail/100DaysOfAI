"""偏好数据测试（M5-D6 A 组之一）：chosen / rejected 的格式、四道体检与分层切分（preference.py）.

本文件与 ``tests/test_alignment_strategy.py`` 成对：那个文件盯"要买多少条"，
这个文件盯"手上这批数据能不能用"。四条承担"证明结论"角色的用例：

1. :meth:`TestSeedPairs.test_fingerprint_matches_the_pinned_value` —— 12 条种子
   偏好样本的内容指纹必须等于 ``5c7cfb2241b63405``。偏好数据比 SFT 数据更
   需要指纹：对齐改变的正是"偏好"，而偏好完全由数据定义，数据一换，两次对齐
   的结论就再也不能逐行比对。
2. :meth:`TestLengthBiasReport.test_seed_totals` /
   :meth:`TestLengthBiasReport.test_seed_by_dimension` —— 总体
   ``chosen_longer_fraction = 0.8333`` 看着像严重偏置，拆到维度上才知道它是
   两个方向相反的合力：``conciseness`` 的 chosen 更短（这一维度的目标就是更短），
   其余四维 chosen 更长。**只看总量会把"维度定义"误判成"标注偏置"。**
3. :meth:`TestNearDuplicates.test_similarity_exactly_at_threshold_is_flagged` ——
   阈值判定是闭区间（``similarity >= threshold``）：两侧 3-gram 相似度恰好
   等于 0.9 的那一对也必须被标出，否则"阈值 0.9"这条线会漏掉刚好压线的样本。
4. :meth:`TestSplitPreferences.test_single_item_per_dimension_leaves_valid_empty` /
   :meth:`TestSplitPreferences.test_two_items_per_dimension_split_one_and_one` ——
   ``n_valid = min(n - 1, max(1, round(n · ratio)))`` 这条算术的两端：每维 1 条时
   验证集为空、必须报错；每维 2 条时验证侧恰好 1 条、训练侧恰好 1 条。
   验证集一旦为空，"何时停止"这个问题就没有数据可答。

所有断言都与 ``preference.py`` 的实现逐字对应；自造样本一律走 :func:`make_pair`
（``PreferencePair.__post_init__`` 会立刻校验，非法样本在构造时就抛错）。

一处与规格书的偏差记录在此（只记录、不改代码）：规格里为"近重复"举的例子是
``chosen="这是答案" / rejected="这是答案。"``，但这一对**无法构造**——
``validate`` 用 ``normalize_text`` 比对两侧，而规范化会抹掉句号，于是这一对
在构造期就被判为"chosen 与 rejected 规范化后相同"。本文件改用
``"这是答案甲乙丙丁戊己庚" / "这是答案甲乙丙丁戊己庚辛"``（相似度恰好 0.9）与
``"…庚辛" / "…庚辛壬"``（0.909091）两条**能构造出来**的近重复样本。
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from smart_research_agent.alignment.preference import (
    ALIGNMENT_DIMENSIONS,
    DEFAULT_VALID_RATIO,
    MAX_LENGTH_RATIO,
    NEAR_DUPLICATE_NGRAM,
    NEAR_DUPLICATE_THRESHOLD,
    SEED_PAIRS,
    PreferencePair,
    _ngram_set,
    dedupe_pairs,
    length_bias_report,
    near_duplicates,
    preference_fingerprint,
    preference_stats,
    read_preferences,
    split_preferences,
    validate_pairs,
    write_preferences,
)
from smart_research_agent.finetune_eval.metrics import FinetuneEvalError, normalize_text
from smart_research_agent.finetune_eval.suites import jaccard

#: 种子偏好样本总数（五个维度各 2~3 条）
SEED_COUNT = 12

#: 种子偏好数据的冻结指纹（``sha256`` 前 16 位）——对齐结果靠它认领"用的是哪一份数据"
PINNED_FINGERPRINT = "5c7cfb2241b63405"

#: 默认切分（``valid_ratio=0.25`` / ``seed=42``）下被划入验证集的样本 id（维度顺序 + id 定序）
PINNED_VALID_IDS = [
    "pref-cite-02",
    "pref-fmt-02",
    "pref-conc-01",
    "pref-refuse-02",
    "pref-honest-02",
]

#: 同一份切分下的训练侧 id
PINNED_TRAIN_IDS = [
    "pref-cite-01",
    "pref-cite-03",
    "pref-fmt-01",
    "pref-conc-02",
    "pref-refuse-01",
    "pref-honest-01",
    "pref-honest-03",
]

#: 种子数据里两个越界样本：一条 rejected 明显更长、一条 chosen 明显更长
PINNED_EXTREME_IDS = ["pref-conc-01", "pref-cite-03"]

#: 近重复样本：11 字 vs 12 字，前者的 3-gram 全被后者包含 → 相似度恰好 9/10 = 0.9
PAIR_AT_THRESHOLD = PreferencePair(
    id="nd-01",
    prompt="近重复体检提问",
    chosen="这是答案甲乙丙丁戊己庚",
    rejected="这是答案甲乙丙丁戊己庚辛",
    dimension="format",
)

#: 近重复样本：12 字 vs 13 字 → 相似度 10/11 = 0.909091
PAIR_ABOVE_THRESHOLD = PreferencePair(
    id="nd-02",
    prompt="近重复体检提问",
    chosen="这是答案甲乙丙丁戊己庚辛",
    rejected="这是答案甲乙丙丁戊己庚辛壬",
    dimension="format",
)

#: 明显不同的一对（相似度 0.0625）：体检仪对"正常样本"必须一声不响
PAIR_DISTINCT = PreferencePair(
    id="ok-01",
    prompt="近重复体检提问",
    chosen="完全不相干的另一种说法",
    rejected="另一套完全不同的表述",
    dimension="honesty",
)


def make_pair(pair_id: str = "custom-01", **overrides) -> PreferencePair:
    """构造一条自造偏好样本：默认最小合法字段，其余用关键字覆盖.

    ``PreferencePair`` 在 ``__post_init__`` 里校验，所以这个 helper 只在
    "故意要合法"的场合使用；要构造非法样本直接显式传非法字段。
    """
    payload: dict = {
        "id": pair_id,
        "prompt": "自造提问：这次对齐要买的是什么？",
        "chosen": "自造的好答案：给出出处，并说明不确定的部分。",
        "rejected": "自造的差答案：含糊其辞，用大概、应该把话说圆。",
        "dimension": "honesty",
    }
    payload.update(overrides)
    return PreferencePair(**payload)


def pair_with_lengths(chosen_len: int, rejected_len: int, pair_id: str = "len-01") -> PreferencePair:
    """用两个不同的字种凑出指定长度的一对（规范化后必然不同，且内容可读）.

    构造"长度比落在某个分档"的样本时，直接写字比数字符可靠：这里只控制
    字符数，``length_ratio`` 完全由它们决定。
    """
    return PreferencePair(
        id=pair_id,
        prompt="长度体检提问",
        chosen="甲" * chosen_len,
        rejected="乙" * rejected_len,
    )


def pairs_with_length_gap(longer: int, shorter: int, *, prefix: str = "gap") -> list[PreferencePair]:
    """构造一批"长度相当"的样本：``longer`` 条 chosen 略长、``shorter`` 条 chosen 略短.

    两侧只差一个字符（9 vs 8 → 长度比 1.125 / 0.889，都落在
    ``MAX_LENGTH_RATIO`` 的容忍区间内），因此 ``extreme_pairs`` 恒为空——
    这样才能单独检验 ``balanced`` 判据里"比例落带"的那一半。
    """
    pairs: list[PreferencePair] = []
    for index in range(longer + shorter):
        if index < longer:
            chosen, rejected = "答案文本甲乙丙丁戊", "答案文本甲乙丙丁"
        else:
            chosen, rejected = "答案文本甲乙丙丁", "答案文本甲乙丙丁戊"
        pairs.append(
            PreferencePair(
                id=f"{prefix}-{index:02d}",
                prompt="长度体检提问",
                chosen=chosen,
                rejected=rejected,
            )
        )
    return pairs


def pairs_with_one_extreme(longer: int = 2, shorter: int = 2) -> list[PreferencePair]:
    """比例落在带内、但含一条越界样本（长度比 4.0）的一组数据.

    ``balanced`` 是"比例落带 **且** 无越界样本"的合取，这一组专门拆开后半个条件：
    比例 0.6 恰好压在带上限，越界样本却让判据必须落到 ``False``。
    """
    pairs = pairs_with_length_gap(longer, shorter, prefix="gap")
    pairs.append(pair_with_lengths(40, 10, pair_id="ext-00"))
    return pairs


@pytest.fixture
def seed_pairs() -> list[PreferencePair]:
    """12 条种子偏好样本的可变副本（每个用例新建一份，避免相互污染）."""
    return list(SEED_PAIRS)


@pytest.fixture
def seed_by_id() -> dict[str, PreferencePair]:
    """种子样本按 id 索引（按 id 取样本比按位置取更耐改）."""
    return {pair.id: pair for pair in SEED_PAIRS}


class TestSeedPairs:
    """种子偏好数据体检：条数 / 维度分布 / 唯一性 / 通过校验 / 定序 / 指纹."""

    def test_seed_pairs_count_is_twelve(self):
        assert len(SEED_PAIRS) == SEED_COUNT

    def test_seed_pairs_is_a_tuple_of_pairs(self):
        assert isinstance(SEED_PAIRS, tuple)
        assert all(isinstance(pair, PreferencePair) for pair in SEED_PAIRS)

    @pytest.mark.parametrize(
        "dimension,expected",
        [("citation", 3), ("format", 2), ("conciseness", 2), ("refusal", 2), ("honesty", 3)],
    )
    def test_declared_dimension_distribution(self, dimension, expected):
        """维度分布必须与声明一致——它是逐维度配额的输入."""
        members = [pair for pair in SEED_PAIRS if pair.dimension == dimension]
        assert len(members) == expected

    def test_seed_dimensions_cover_exactly_the_declared_dimensions(self):
        assert {pair.dimension for pair in SEED_PAIRS} == set(ALIGNMENT_DIMENSIONS)

    def test_seed_ids_are_unique(self):
        ids = [pair.id for pair in SEED_PAIRS]
        assert len(ids) == len(set(ids)) == SEED_COUNT

    @pytest.mark.parametrize("pair", SEED_PAIRS, ids=[pair.id for pair in SEED_PAIRS])
    def test_every_seed_pair_passes_validate(self, pair):
        """每条种子样本都必须自洽（``validate`` 不抛异常，且二次调用幂等）."""
        assert pair.validate() is None
        assert normalize_text(pair.chosen) != normalize_text(pair.rejected)

    def test_seed_pairs_have_no_issues(self):
        """12 条种子数据是"体检全绿"的基准：任何一条坏样本都会让这条红."""
        assert validate_pairs(list(SEED_PAIRS)) == []

    def test_fingerprint_matches_the_pinned_value(self):
        """指纹冻结整份种子：改一个字、加一条样本都会让它红."""
        assert preference_fingerprint(list(SEED_PAIRS)) == PINNED_FINGERPRINT

    def test_alignment_dimensions_order_is_declared(self):
        """五个维度的顺序是报告顺序（也是切分与配额的顺序），必须显式钉住."""
        assert ALIGNMENT_DIMENSIONS == ("citation", "format", "conciseness", "refusal", "honesty")

    def test_module_constants_are_declared(self):
        assert MAX_LENGTH_RATIO == 3.0
        assert NEAR_DUPLICATE_NGRAM == 3
        assert NEAR_DUPLICATE_THRESHOLD == 0.9
        assert DEFAULT_VALID_RATIO == 0.25


class TestPreferencePairValidation:
    """``PreferencePair.validate`` 的失败路径：非法样本必须在构造时就被拦下."""

    def test_minimal_pair_is_accepted(self):
        """正对照：最小合法样本能构造出来，且 metadata 默认为空字典."""
        pair = make_pair()
        assert pair.metadata == {}
        assert pair.dimension == "honesty"

    @pytest.mark.parametrize("blank_id", ["", "   ", "\t", "\n"])
    def test_blank_id_rejected(self, blank_id):
        with pytest.raises(FinetuneEvalError, match="偏好样本的 id 不能为空"):
            make_pair(blank_id)

    @pytest.mark.parametrize("prompt", ["", "   ", "\n", " \t "])
    def test_blank_prompt_rejected(self, prompt):
        with pytest.raises(FinetuneEvalError, match="的 prompt 不能为空"):
            make_pair(prompt=prompt)

    @pytest.mark.parametrize("chosen", ["", "   ", "\n"])
    def test_blank_chosen_rejected(self, chosen):
        """空 chosen 会让这一对在 DPO 里恒偏向 rejected——必须在构造期拦住."""
        with pytest.raises(FinetuneEvalError, match="chosen / rejected 都不能为空"):
            make_pair(chosen=chosen)

    @pytest.mark.parametrize("rejected", ["", "   ", "\t"])
    def test_blank_rejected_rejected(self, rejected):
        with pytest.raises(FinetuneEvalError, match="chosen / rejected 都不能为空"):
            make_pair(rejected=rejected)

    @pytest.mark.parametrize("dimension", ["unknown", "Citation", "citation ", "工具", ""])
    def test_unknown_dimension_rejected(self, dimension):
        """维度名大小写与空格都不容错：拼错会静默改变逐维度的统计口径."""
        with pytest.raises(FinetuneEvalError, match="未知的对齐维度"):
            make_pair(dimension=dimension)

    def test_known_dimensions_are_all_accepted(self):
        for dimension in ALIGNMENT_DIMENSIONS:
            assert make_pair(dimension=dimension).dimension == dimension

    @pytest.mark.parametrize(
        "chosen,rejected",
        [
            ("答案。", "答案"),
            ("答案", " 答案 "),
            ("ABC", "abc"),
            ("答案！", "答案"),
            ("ΔW = B·A", "ΔW=B*A"),
        ],
    )
    def test_normalized_equal_sides_rejected(self, chosen, rejected):
        """规范化后相同的一对在 DPO 里 margin 恒为 0：不产生梯度，却占一条样本.

        这正是"loss 一直等于 ln 2、训练看起来跑过了却什么都没学到"的来源，
        所以它必须在构造期就抛错，而不是留到训练日志里去猜。
        """
        with pytest.raises(FinetuneEvalError, match="规范化后相同"):
            make_pair(chosen=chosen, rejected=rejected)

    def test_normalized_different_sides_are_accepted(self):
        """正对照：只差一个实义字（规范化不会抹掉）的一对是合法的."""
        pair = make_pair(chosen="这是答案甲乙", rejected="这是答案甲乙丙")
        assert normalize_text(pair.chosen) != normalize_text(pair.rejected)

    def test_validate_is_idempotent(self):
        pair = make_pair()
        assert pair.validate() is None
        assert pair.validate() is None

    def test_error_message_carries_the_pair_id(self):
        """错误信息要能定位到具体样本（否则 12 条里改哪条靠猜）."""
        with pytest.raises(FinetuneEvalError, match="custom-07 的 prompt 不能为空"):
            make_pair("custom-07", prompt=" ")

    def test_error_message_for_equal_sides_carries_the_pair_id(self):
        with pytest.raises(FinetuneEvalError, match="custom-08 的 chosen 与 rejected"):
            make_pair("custom-08", chosen="答案。", rejected="答案")

    def test_pair_is_frozen(self):
        """``PreferencePair`` 是不可变的：偏好数据在运行期不该被就地改写."""
        pair = make_pair()
        with pytest.raises(FrozenInstanceError):
            pair.dimension = "format"  # type: ignore[misc]


class TestLengthRatioAndBias:
    """``length_ratio`` 与三档 ``length_bias``：分档边界是闭区间."""

    def test_length_ratio_is_the_character_count_ratio(self):
        pair = pair_with_lengths(30, 10)
        assert len(pair.chosen) == 30
        assert len(pair.rejected) == 10
        assert pair.length_ratio == pytest.approx(3.0)

    @pytest.mark.parametrize(
        "chosen_len,rejected_len,expected",
        [
            (31, 10, "chosen_longer"),
            (40, 10, "chosen_longer"),
            (12, 10, "balanced"),
            (10, 10, "balanced"),
        ],
    )
    def test_chosen_longer_band(self, chosen_len, rejected_len, expected):
        assert pair_with_lengths(chosen_len, rejected_len).length_bias == expected

    @pytest.mark.parametrize(
        "chosen_len,rejected_len,expected",
        [
            (10, 31, "rejected_longer"),
            (10, 40, "rejected_longer"),
            (10, 12, "balanced"),
            (10, 10, "balanced"),
        ],
    )
    def test_rejected_longer_band(self, chosen_len, rejected_len, expected):
        assert pair_with_lengths(chosen_len, rejected_len).length_bias == expected

    @pytest.mark.parametrize("chosen_len,rejected_len", [(30, 10), (10, 30)])
    def test_ratio_exactly_at_the_limit_is_balanced(self, chosen_len, rejected_len):
        """比值恰好等于 ``MAX_LENGTH_RATIO``（或它的倒数）时**不算越界**.

        ``length_bias`` 用的是严格不等式，而 ``extreme_pairs`` 的定义是
        "``length_bias != balanced``"，所以这条边界决定了越界清单一端的口径。
        """
        pair = pair_with_lengths(chosen_len, rejected_len)
        expected_ratio = MAX_LENGTH_RATIO if chosen_len > rejected_len else 1 / MAX_LENGTH_RATIO
        assert pair.length_ratio == pytest.approx(expected_ratio)
        assert pair.length_bias == "balanced"

    def test_just_outside_the_limit_is_flagged(self):
        assert pair_with_lengths(31, 10).length_bias == "chosen_longer"
        assert pair_with_lengths(10, 31).length_bias == "rejected_longer"

    def test_seed_pairs_bias_directions(self, seed_by_id):
        """种子数据里 10 条落在容忍区间内，只有两条越界（方向相反）."""
        biases = {pair.id: pair.length_bias for pair in SEED_PAIRS}
        assert biases["pref-conc-01"] == "rejected_longer"
        assert biases["pref-cite-03"] == "chosen_longer"
        assert sum(1 for value in biases.values() if value == "balanced") == 10


class TestPairSerialization:
    """``to_dict`` / ``from_dict`` / ``summary_line``：数据的进出必须无损."""

    def test_to_dict_matches_the_declared_fields(self):
        pair = make_pair("json-01", metadata={"note": "备注"})
        payload = pair.to_dict()
        assert list(payload) == ["id", "prompt", "chosen", "rejected", "dimension", "metadata"]
        assert payload["id"] == "json-01"
        assert payload["metadata"] == {"note": "备注"}

    def test_to_dict_is_json_serializable(self):
        payload = make_pair().to_dict()
        text = json.dumps(payload, ensure_ascii=False)
        assert json.loads(text) == payload

    def test_from_dict_round_trip(self):
        pair = make_pair("json-02", dimension="refusal", metadata={"source": "手写"})
        assert PreferencePair.from_dict(pair.to_dict()) == pair

    def test_from_dict_round_trip_for_whole_seed_set(self):
        payloads = [pair.to_dict() for pair in SEED_PAIRS]
        assert [PreferencePair.from_dict(item) for item in payloads] == list(SEED_PAIRS)

    def test_from_dict_ignores_unknown_keys(self):
        """未来字段不该让老数据读不进来（``from_dict`` 只取已知键）."""
        payload = make_pair("json-03").to_dict()
        payload["future_field"] = {"x": 1}
        payload["另一把不认识的键"] = 2
        assert PreferencePair.from_dict(payload) == make_pair("json-03")

    def test_from_dict_missing_required_key_rejected(self):
        payload = make_pair().to_dict()
        del payload["chosen"]
        with pytest.raises(TypeError):
            PreferencePair.from_dict(payload)

    def test_summary_line_is_pinned_for_a_seed_pair(self, seed_by_id):
        pair = seed_by_id["pref-cite-01"]
        expected = (
            f"{pair.id} | citation | 长度比 {pair.length_ratio:.2f}（balanced）"
            f" | prompt {pair.prompt[:24]}…"
        )
        assert pair.summary_line() == expected

    def test_summary_line_uses_the_two_decimal_ratio(self, seed_by_id):
        assert "长度比 0.17（rejected_longer）" in seed_by_id["pref-conc-01"].summary_line()

    def test_summary_line_truncates_long_prompts(self):
        pair = make_pair("long-01", prompt="很长的问题" * 20)
        line = pair.summary_line()
        assert pair.prompt[:24] in line
        assert line.endswith("…")
        assert pair.prompt not in line
        assert len(pair.prompt) > 24

    def test_summary_line_is_a_single_line_string(self):
        for pair in SEED_PAIRS:
            line = pair.summary_line()
            assert isinstance(line, str)
            assert "\n" not in line
            assert pair.id in line
            assert pair.dimension in line


class TestValidatePairs:
    """``validate_pairs``：体检结果是"清单"而不是异常，且空数据必须自己报出来."""

    def test_empty_list_reports_the_empty_dataset_issue(self):
        assert validate_pairs([]) == [{"id": "", "problem": "偏好数据集为空"}]

    def test_seed_pairs_have_nothing_to_report(self):
        assert validate_pairs(list(SEED_PAIRS)) == []

    def test_duplicate_id_is_reported(self):
        duplicated = [
            make_pair("dup-01"),
            make_pair("dup-01", chosen="另一个好答案甲乙", rejected="另一个差答案丙丁"),
        ]
        assert validate_pairs(duplicated) == [{"id": "dup-01", "problem": "重复的 id"}]

    def test_every_extra_copy_is_counted(self):
        """重复是"多出来的每一份"各记一条：三条同 id → 两条问题."""
        tripled = [
            make_pair("dup-02"),
            make_pair("dup-02", chosen="另一个好答案甲乙", rejected="另一个差答案丙丁"),
            make_pair("dup-02", chosen="第三个好答案戊己", rejected="第三个差答案庚辛"),
        ]
        assert validate_pairs(tripled) == [
            {"id": "dup-02", "problem": "重复的 id"},
            {"id": "dup-02", "problem": "重复的 id"},
        ]

    def test_issue_entries_have_the_declared_shape(self):
        issue = validate_pairs([])[0]
        assert set(issue) == {"id", "problem"}
        assert isinstance(issue["problem"], str)

    def test_duplicate_ids_across_dimensions_are_still_duplicates(self):
        """id 是主键：换维度不构成"另一条样本"."""
        duplicated = [
            make_pair("dup-03", dimension="citation"),
            make_pair(
                "dup-03",
                dimension="format",
                chosen="另一个好答案甲乙",
                rejected="另一个差答案丙丁",
            ),
        ]
        assert validate_pairs(duplicated) == [{"id": "dup-03", "problem": "重复的 id"}]

    def test_clean_custom_batch_reports_nothing(self):
        clean = [make_pair(f"clean-{index:02d}") for index in range(5)]
        assert validate_pairs(clean) == []


class TestLengthBiasReport:
    """``length_bias_report``：偏好数据里最经典的混淆变量必须被量化."""

    def test_empty_pairs_rejected(self):
        with pytest.raises(FinetuneEvalError, match="不能对空偏好数据集做长度体检"):
            length_bias_report([])

    def test_seed_totals(self, seed_pairs):
        report = length_bias_report(seed_pairs)
        assert report["total"] == SEED_COUNT
        assert report["chosen_longer_fraction"] == 0.8333

    def test_seed_ratio_summary(self, seed_pairs):
        report = length_bias_report(seed_pairs)
        assert report["mean_length_ratio"] == 1.6095
        assert report["min_length_ratio"] == 0.1698
        assert report["max_length_ratio"] == 3.2

    def test_seed_is_not_balanced(self, seed_pairs):
        """0.8333 远超容忍带 [0.4, 0.6]，且存在越界样本——粗筛必须红."""
        assert length_bias_report(seed_pairs)["balanced"] is False

    def test_seed_extreme_pairs(self, seed_pairs):
        """越界样本恰好两条，方向相反：一条 rejected 更长、一条 chosen 更长."""
        extremes = length_bias_report(seed_pairs)["extreme_pairs"]
        assert [item["id"] for item in extremes] == PINNED_EXTREME_IDS
        assert extremes[0]["ratio"] == 0.1698
        assert extremes[0]["bias"] == "rejected_longer"
        assert extremes[1]["ratio"] == 3.2
        assert extremes[1]["bias"] == "chosen_longer"

    def test_every_extreme_pair_is_outside_the_tolerance_band(self, seed_pairs):
        extremes = length_bias_report(seed_pairs)["extreme_pairs"]
        for item in extremes:
            assert not (1 / MAX_LENGTH_RATIO <= item["ratio"] <= MAX_LENGTH_RATIO)

    def test_seed_by_dimension(self, seed_pairs):
        """逐维度分解才是定位问题的依据：conciseness 的 chosen **更短**."""
        by_dimension = length_bias_report(seed_pairs)["by_dimension"]
        assert list(by_dimension) == list(ALIGNMENT_DIMENSIONS)
        assert by_dimension["citation"] == {
            "total": 3,
            "chosen_longer_fraction": 1.0,
            "mean_length_ratio": 2.4337,
        }
        assert by_dimension["format"] == {
            "total": 2,
            "chosen_longer_fraction": 1.0,
            "mean_length_ratio": 1.7473,
        }
        assert by_dimension["conciseness"] == {
            "total": 2,
            "chosen_longer_fraction": 0.0,
            "mean_length_ratio": 0.4369,
        }
        assert by_dimension["refusal"] == {
            "total": 2,
            "chosen_longer_fraction": 1.0,
            "mean_length_ratio": 1.1268,
        }
        assert by_dimension["honesty"] == {
            "total": 3,
            "chosen_longer_fraction": 1.0,
            "mean_length_ratio": 1.797,
        }

    def test_only_conciseness_has_chosen_shorter(self, seed_pairs):
        """总体 0.8333 是两个相反合力叠出来的：拆开看只有 conciseness 反号."""
        by_dimension = length_bias_report(seed_pairs)["by_dimension"]
        assert by_dimension["conciseness"]["chosen_longer_fraction"] == 0.0
        for dimension in ("citation", "format", "refusal", "honesty"):
            assert by_dimension[dimension]["chosen_longer_fraction"] == 1.0

    def test_by_dimension_skips_empty_dimensions(self):
        """只保留实际出现的维度：0 条的分组不该在报告里占一行."""
        citation_only = [pair for pair in SEED_PAIRS if pair.dimension == "citation"]
        by_dimension = length_bias_report(citation_only)["by_dimension"]
        assert list(by_dimension) == ["citation"]
        assert by_dimension["citation"]["total"] == 3

    def test_balanced_case(self):
        """构造 6 条长度相当的样本：比例 0.5、无越界 → ``balanced = True``."""
        report = length_bias_report(pairs_with_length_gap(3, 3))
        assert report["total"] == 6
        assert report["chosen_longer_fraction"] == 0.5
        assert report["mean_length_ratio"] == 1.0069
        assert report["min_length_ratio"] == 0.8889
        assert report["max_length_ratio"] == 1.125
        assert report["extreme_pairs"] == []
        assert report["balanced"] is True

    def test_upper_boundary_of_the_band_is_inclusive(self):
        """比例恰好 0.6 落在带上（闭区间），且无越界样本 → 仍然 balanced."""
        report = length_bias_report(pairs_with_length_gap(6, 4))
        assert report["chosen_longer_fraction"] == 0.6
        assert report["extreme_pairs"] == []
        assert report["balanced"] is True

    def test_fraction_above_the_band_is_not_balanced(self):
        """0.7 越出带上限，即使一条越界样本都没有 → 偏置明显."""
        report = length_bias_report(pairs_with_length_gap(7, 3))
        assert report["chosen_longer_fraction"] == 0.7
        assert report["extreme_pairs"] == []
        assert report["balanced"] is False

    def test_lower_boundary_of_the_band_is_inclusive(self):
        report = length_bias_report(pairs_with_length_gap(4, 6))
        assert report["chosen_longer_fraction"] == 0.4
        assert report["balanced"] is True

    def test_extreme_pair_alone_breaks_the_balance(self):
        """比例压在带上限，但有一条长度比 4.0 的样本 → 判据必须落到 False."""
        report = length_bias_report(pairs_with_one_extreme())
        assert report["chosen_longer_fraction"] == 0.6
        assert report["extreme_pairs"] == [{"id": "ext-00", "ratio": 4.0, "bias": "chosen_longer"}]
        assert report["max_length_ratio"] == 4.0
        assert report["balanced"] is False

    def test_report_keys_are_pinned(self, seed_pairs):
        """报告字段是 API 与文档的契约，键名漂移会让下游读不到."""
        report = length_bias_report(seed_pairs)
        assert set(report) == {
            "total",
            "chosen_longer_fraction",
            "mean_length_ratio",
            "min_length_ratio",
            "max_length_ratio",
            "extreme_pairs",
            "by_dimension",
            "balanced",
        }

    def test_single_pair_report(self):
        report = length_bias_report([pair_with_lengths(20, 10)])
        assert report["total"] == 1
        assert report["chosen_longer_fraction"] == 1.0
        assert report["mean_length_ratio"] == 2.0
        assert report["min_length_ratio"] == report["max_length_ratio"] == 2.0
        assert report["balanced"] is False


class TestNgramSet:
    """``_ngram_set``：近重复体检的底层口径（规范化字符 n-gram）."""

    @pytest.mark.parametrize("order", [0, -1, -3])
    def test_non_positive_order_rejected(self, order):
        with pytest.raises(FinetuneEvalError, match="n-gram 的阶必须为正整数"):
            _ngram_set("答案文本", order)

    def test_default_order_is_the_declared_constant(self):
        assert sorted(_ngram_set("abcd")) == [("a", "b", "c"), ("b", "c", "d")]
        assert NEAR_DUPLICATE_NGRAM == 3

    def test_bigram_set_of_a_short_string(self):
        assert sorted(_ngram_set("abcd", 2)) == [("a", "b"), ("b", "c"), ("c", "d")]

    def test_normalization_is_applied(self):
        """规范化在取 gram 之前发生：大小写与标点都不该产生新 gram."""
        assert _ngram_set("A-B", 2) == _ngram_set("ab", 2) == {("a", "b")}

    def test_text_shorter_than_the_order_gives_an_empty_set(self):
        assert _ngram_set("ab", 3) == set()
        assert _ngram_set("", 3) == set()

    def test_chinese_text_grams(self):
        assert _ngram_set("答案文本", 3) == {("答", "案", "文"), ("案", "文", "本")}

    def test_gram_count_is_length_minus_order_plus_one(self):
        text = "答案文本甲乙丙"
        assert len(normalize_text(text)) == 7
        assert len(_ngram_set(text)) == 5

    def test_identical_twins_have_jaccard_one(self):
        assert jaccard(_ngram_set("答案文本"), _ngram_set("答案文本")) == 1.0


class TestNearDuplicates:
    """``near_duplicates``：chosen 与 rejected 几乎一样的一对没有信息量."""

    @pytest.mark.parametrize("threshold", [-0.01, -1.0, 1.01, 2.0])
    def test_out_of_range_threshold_rejected(self, threshold):
        with pytest.raises(FinetuneEvalError, match="threshold 必须落在"):
            near_duplicates(list(SEED_PAIRS), threshold=threshold)

    def test_seed_pairs_have_no_near_duplicates(self, seed_pairs):
        """12 条手写样本彼此差别都能指出来：默认阈值下一条都不该被标."""
        assert near_duplicates(seed_pairs) == []

    def test_threshold_boundaries_are_accepted(self, seed_pairs):
        """0.0 与 1.0 都是合法阈值（闭区间），不能被判成越界."""
        assert isinstance(near_duplicates(seed_pairs, threshold=0.0), list)
        assert isinstance(near_duplicates(seed_pairs, threshold=1.0), list)

    def test_threshold_one_flags_nothing_for_seeds(self, seed_pairs):
        assert near_duplicates(seed_pairs, threshold=1.0) == []

    def test_threshold_zero_flags_every_pair(self, seed_pairs):
        """阈值下界 0.0 时**所有**样本都算"过于相似"（闭区间的另一端）."""
        flagged = near_duplicates(seed_pairs, threshold=0.0)
        assert len(flagged) == SEED_COUNT
        assert {item["id"] for item in flagged} == {pair.id for pair in SEED_PAIRS}

    def test_empty_list_returns_empty(self):
        assert near_duplicates([]) == []

    def test_similar_pair_is_flagged(self):
        flagged = near_duplicates([PAIR_ABOVE_THRESHOLD])
        assert len(flagged) == 1
        assert flagged[0]["id"] == "nd-02"
        assert flagged[0]["dimension"] == "format"
        assert flagged[0]["similarity"] == pytest.approx(0.909091, abs=1e-6)
        assert flagged[0]["similarity"] >= NEAR_DUPLICATE_THRESHOLD

    def test_similarity_exactly_at_threshold_is_flagged(self):
        """阈值是闭区间：相似度恰好 0.9 也要被标出（9/10 + 少一个字的差别）.

        两条回答只差最后一个字，前者的 9 个 3-gram 全被后者包含，于是
        Jaccard = 9/10。这样的一对在 DPO 里的 margin 接近 0，贡献的梯度也
        接近 0，却照样占一条样本的位置。
        """
        left = _ngram_set(PAIR_AT_THRESHOLD.chosen)
        right = _ngram_set(PAIR_AT_THRESHOLD.rejected)
        assert len(left) == 9 and len(right) == 10
        assert jaccard(left, right) == pytest.approx(0.9)
        flagged = near_duplicates([PAIR_AT_THRESHOLD])
        assert [item["id"] for item in flagged] == ["nd-01"]
        assert flagged[0]["similarity"] == pytest.approx(0.9, abs=1e-6)

    def test_just_below_the_threshold_is_not_flagged(self):
        """把阈值抬到 0.95，同一条样本就不再被标——阈值真的在起作用."""
        assert near_duplicates([PAIR_AT_THRESHOLD], threshold=0.95) == []
        assert near_duplicates([PAIR_ABOVE_THRESHOLD], threshold=0.95) == []

    def test_distinct_pair_is_not_flagged(self):
        assert near_duplicates([PAIR_DISTINCT]) == []

    def test_flag_shape(self):
        flagged = near_duplicates([PAIR_DISTINCT, PAIR_AT_THRESHOLD])
        assert len(flagged) == 1
        item = flagged[0]
        assert set(item) == {"id", "dimension", "similarity", "hint"}
        assert item["id"] == "nd-01"
        assert item["dimension"] == "format"
        assert "两侧过于相似" in item["hint"]

    def test_flags_follow_the_input_order(self):
        flagged = near_duplicates([PAIR_ABOVE_THRESHOLD, PAIR_AT_THRESHOLD])
        assert [item["id"] for item in flagged] == ["nd-02", "nd-01"]

    def test_public_signature_exposes_only_pairs_and_threshold(self):
        """近重复的"阶"由模块常量决定：公共入口只收样本与阈值，口径不可被调用方改."""
        parameters = inspect.signature(near_duplicates).parameters
        assert set(parameters) == {"pairs", "threshold"}
        assert parameters["threshold"].default == NEAR_DUPLICATE_THRESHOLD


class TestDedupePairs:
    """``dedupe_pairs``：只剔"近重复"，**不动长度偏置**."""

    def test_seed_pairs_are_all_kept(self, seed_pairs):
        kept, dropped = dedupe_pairs(seed_pairs)
        assert len(kept) == SEED_COUNT
        assert dropped == []

    def test_flagged_pair_is_dropped(self):
        kept, dropped = dedupe_pairs([PAIR_AT_THRESHOLD, PAIR_DISTINCT])
        assert [pair.id for pair in kept] == ["ok-01"]
        assert [item["id"] for item in dropped] == ["nd-01"]

    def test_kept_plus_dropped_equals_the_total(self):
        """守恒：留下的 + 剔除的必须等于总数，否则"剔了多少条"这个数字不可信."""
        for threshold in (0.0, 0.5, 0.9, 0.95, 1.0):
            kept, dropped = dedupe_pairs(list(SEED_PAIRS), threshold=threshold)
            assert len(kept) + len(dropped) == SEED_COUNT

    def test_everything_dropped_at_threshold_zero(self, seed_pairs):
        """阈值 0.0 时"保留 + 剔除"仍然守恒，且保留侧为空."""
        kept, dropped = dedupe_pairs(seed_pairs, threshold=0.0)
        assert kept == []
        assert len(dropped) == SEED_COUNT

    def test_dropped_entries_carry_the_similarity_and_hint(self):
        _, dropped = dedupe_pairs([PAIR_ABOVE_THRESHOLD])
        assert set(dropped[0]) == {"id", "dimension", "similarity", "hint"}
        assert dropped[0]["similarity"] >= NEAR_DUPLICATE_THRESHOLD

    @pytest.mark.parametrize("threshold", [-0.1, 1.5])
    def test_out_of_range_threshold_rejected(self, threshold):
        with pytest.raises(FinetuneEvalError, match="threshold 必须落在"):
            dedupe_pairs(list(SEED_PAIRS), threshold=threshold)

    def test_empty_list_returns_two_empty_lists(self):
        assert dedupe_pairs([]) == ([], [])

    def test_deduped_batch_is_still_valid(self):
        """剔除之后的批次必须仍然是合法批次（且条数守恒）."""
        kept, dropped = dedupe_pairs([PAIR_AT_THRESHOLD, PAIR_DISTINCT])
        assert validate_pairs(kept) == []
        assert len(kept) + len(dropped) == 2


class TestPreferenceStats:
    """``preference_stats``：偏好数据的统计画像（维度分布 + 长度体检 + 近重复 + 指纹）."""

    def test_empty_pairs_rejected(self):
        with pytest.raises(FinetuneEvalError, match="不能对空偏好数据集做统计"):
            preference_stats([])

    def test_seed_total(self, seed_pairs):
        assert preference_stats(seed_pairs)["total"] == SEED_COUNT

    def test_seed_dimension_distribution(self, seed_pairs):
        assert preference_stats(seed_pairs)["by_dimension"] == {
            "citation": 3,
            "format": 2,
            "conciseness": 2,
            "refusal": 2,
            "honesty": 3,
        }

    def test_seed_mean_character_counts(self, seed_pairs):
        stats = preference_stats(seed_pairs)
        assert stats["mean_chosen_chars"] == 60.25
        assert stats["mean_rejected_chars"] == 46.8333

    def test_by_dimension_has_no_zero_entries(self):
        """0 条的分组不进画像：留一行 ``citation: 0`` 只会让人误以为数据缺失."""
        only_refusal = [pair for pair in SEED_PAIRS if pair.dimension == "refusal"]
        stats = preference_stats(only_refusal)
        assert stats["by_dimension"] == {"refusal": 2}
        assert set(stats["by_dimension"]) <= set(ALIGNMENT_DIMENSIONS)

    def test_length_bias_is_embedded(self, seed_pairs):
        """画像里的长度体检必须是同一份报告（而不是另算一遍的近似值）."""
        stats = preference_stats(seed_pairs)
        assert stats["length_bias"] == length_bias_report(list(SEED_PAIRS))
        assert stats["length_bias"]["chosen_longer_fraction"] == 0.8333

    def test_near_duplicates_are_embedded(self, seed_pairs):
        stats = preference_stats(seed_pairs)
        assert stats["near_duplicates"] == []
        assert stats["near_duplicates"] == near_duplicates(list(SEED_PAIRS))

    def test_fingerprint_is_embedded_and_pinned(self, seed_pairs):
        stats = preference_stats(seed_pairs)
        assert stats["fingerprint"] == PINNED_FINGERPRINT
        assert stats["fingerprint"] == preference_fingerprint(list(SEED_PAIRS))

    def test_stats_keys_are_pinned(self, seed_pairs):
        assert set(preference_stats(seed_pairs)) == {
            "total",
            "by_dimension",
            "mean_chosen_chars",
            "mean_rejected_chars",
            "length_bias",
            "near_duplicates",
            "fingerprint",
        }

    def test_single_pair_stats(self):
        stats = preference_stats([pair_with_lengths(20, 10)])
        assert stats["total"] == 1
        assert stats["mean_chosen_chars"] == 20.0
        assert stats["mean_rejected_chars"] == 10.0
        assert stats["by_dimension"] == {"honesty": 1}


class TestPreferenceFingerprint:
    """``preference_fingerprint``：一次对齐的结果必须能指回"用的是哪一份偏好数据"."""

    def test_empty_pairs_rejected(self):
        with pytest.raises(FinetuneEvalError, match="不能对空偏好数据集取指纹"):
            preference_fingerprint([])

    def test_fingerprint_shape(self):
        fingerprint = preference_fingerprint(list(SEED_PAIRS))
        assert re.fullmatch(r"[0-9a-f]{16}", fingerprint) is not None

    def test_same_content_is_stable(self):
        assert preference_fingerprint(list(SEED_PAIRS)) == preference_fingerprint(list(SEED_PAIRS))

    def test_same_content_built_twice_is_stable(self):
        assert preference_fingerprint([make_pair("fp-01")]) == preference_fingerprint(
            [make_pair("fp-01")]
        )

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda pair: replace(pair, chosen=pair.chosen + "。"), id="chosen"),
            pytest.param(lambda pair: replace(pair, rejected=pair.rejected + "！"), id="rejected"),
            pytest.param(lambda pair: replace(pair, prompt=pair.prompt + "？"), id="prompt"),
            pytest.param(lambda pair: replace(pair, id=pair.id + "-x"), id="id"),
            pytest.param(lambda pair: replace(pair, metadata={"note": "改动"}), id="metadata"),
        ],
    )
    def test_changing_one_field_changes_the_fingerprint(self, seed_pairs, mutate):
        baseline = preference_fingerprint(seed_pairs)
        changed = list(seed_pairs)
        changed[0] = mutate(changed[0])
        assert preference_fingerprint(changed) != baseline

    def test_dropping_a_pair_changes_the_fingerprint(self, seed_pairs):
        assert preference_fingerprint(seed_pairs[:-1]) != preference_fingerprint(seed_pairs)

    def test_reordering_changes_the_fingerprint(self, seed_pairs):
        """指纹包含顺序：同一批样本换个顺序就不是"同一份偏好数据"."""
        assert preference_fingerprint(list(reversed(seed_pairs))) != preference_fingerprint(seed_pairs)

    def test_metadata_key_order_does_not_change_the_fingerprint(self):
        """规范化 JSON（``sort_keys``）是"指纹能回答问题"的前提."""
        left = preference_fingerprint([make_pair("fp-02", metadata={"a": 1, "b": 2})])
        right = preference_fingerprint([make_pair("fp-02", metadata={"b": 2, "a": 1})])
        assert left == right

    def test_fingerprint_matches_the_pinned_seed_value(self, seed_pairs):
        assert preference_fingerprint(seed_pairs) == PINNED_FINGERPRINT


class TestSplitPreferences:
    """``split_preferences``：按维度分层切分，验证集必须非空且逐维度都留了一条."""

    @pytest.mark.parametrize("valid_ratio", [-0.1, -1.0, 1.0, 1.5])
    def test_out_of_range_valid_ratio_rejected(self, seed_pairs, valid_ratio):
        with pytest.raises(FinetuneEvalError, match="valid_ratio 必须落在"):
            split_preferences(seed_pairs, valid_ratio=valid_ratio)

    def test_empty_pairs_rejected(self):
        with pytest.raises(FinetuneEvalError, match="不能对空偏好数据集做切分"):
            split_preferences([])

    def test_default_sizes(self, seed_pairs):
        train, valid = split_preferences(seed_pairs)
        assert len(train) == 7
        assert len(valid) == 5

    def test_default_valid_ids(self, seed_pairs):
        """默认切分（0.25 / seed=42）的结果被冻结：换实现会立刻红."""
        _, valid = split_preferences(seed_pairs)
        assert [pair.id for pair in valid] == PINNED_VALID_IDS

    def test_default_train_ids(self, seed_pairs):
        train, _ = split_preferences(seed_pairs)
        assert [pair.id for pair in train] == PINNED_TRAIN_IDS

    def test_split_is_disjoint_and_complete(self, seed_pairs):
        train, valid = split_preferences(seed_pairs)
        train_ids = {pair.id for pair in train}
        valid_ids = {pair.id for pair in valid}
        assert train_ids & valid_ids == set()
        assert train_ids | valid_ids == {pair.id for pair in seed_pairs}

    def test_split_is_sorted_by_dimension_then_id(self, seed_pairs):
        """顺序恒为"维度顺序 + id"：顺序会漂移的切分没法逐行比对."""
        train, valid = split_preferences(seed_pairs)
        expected = sorted(
            (pair for pair in seed_pairs if pair.dimension in ALIGNMENT_DIMENSIONS),
            key=lambda pair: (ALIGNMENT_DIMENSIONS.index(pair.dimension), pair.id),
        )
        for side in (train, valid):
            positions = [
                (ALIGNMENT_DIMENSIONS.index(pair.dimension), pair.id) for pair in side
            ]
            assert positions == sorted(positions)
        assert {pair.id for pair in expected} == {pair.id for pair in seed_pairs}

    def test_same_seed_is_reproducible(self, seed_pairs):
        first = split_preferences(seed_pairs, seed=7)
        second = split_preferences(seed_pairs, seed=7)
        assert [pair.id for pair in first[0]] == [pair.id for pair in second[0]]
        assert [pair.id for pair in first[1]] == [pair.id for pair in second[1]]

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 123, 2024])
    def test_quota_is_independent_of_the_seed(self, seed_pairs, seed):
        """种子只决定"哪几条"落进验证集，不决定"几条"——配额是纯算术."""
        train, valid = split_preferences(seed_pairs, seed=seed)
        assert len(train) == 7
        assert len(valid) == 5
        assert {pair.id for pair in train} | {pair.id for pair in valid} == {
            pair.id for pair in seed_pairs
        }

    @pytest.mark.parametrize("valid_ratio", [0.0, 0.25, 0.5, 0.99])
    def test_per_dimension_quota_arithmetic(self, seed_pairs, valid_ratio):
        """``n_valid = min(n - 1, max(1, round(n · ratio)))`` 逐浓度核对."""
        train, valid = split_preferences(seed_pairs, valid_ratio=valid_ratio)
        for dimension in ALIGNMENT_DIMENSIONS:
            members = [pair for pair in seed_pairs if pair.dimension == dimension]
            quota = min(len(members) - 1, max(1, round(len(members) * valid_ratio)))
            assert sum(1 for pair in valid if pair.dimension == dimension) == quota
            assert sum(1 for pair in train if pair.dimension == dimension) == len(members) - quota

    def test_zero_ratio_still_reserves_one_validation_item_per_dimension(self, seed_pairs):
        """比例取 0 也**不等于**不切验证集：``max(1, …)`` 的下限仍然生效."""
        train, valid = split_preferences(seed_pairs, valid_ratio=0.0)
        assert len(valid) == 5
        assert {pair.dimension for pair in valid} == set(ALIGNMENT_DIMENSIONS)
        assert [pair.id for pair in valid] == PINNED_VALID_IDS

    def test_high_ratio_keeps_one_item_in_training(self, seed_pairs):
        """比例再高也要给训练侧留 1 条：``min(n - 1, …)`` 的上限是硬护栏."""
        train, valid = split_preferences(seed_pairs, valid_ratio=0.99)
        assert len(train) == len(ALIGNMENT_DIMENSIONS)
        assert len(valid) == 7
        for dimension in ALIGNMENT_DIMENSIONS:
            assert sum(1 for pair in train if pair.dimension == dimension) == 1

    def test_single_item_per_dimension_leaves_valid_empty(self):
        """每个维度只有 1 条 → 验证集为空 → 必须报错（而不是静默继续）.

        空验证集会让训练全程"看起来一直在变好"：过优化曲线只能在留出数据上量。
        """
        one_each = [
            make_pair(
                f"solo-{dimension}",
                dimension=dimension,
                chosen=f"{dimension} 的好答案甲",
                rejected=f"{dimension} 的差答案乙",
            )
            for dimension in ALIGNMENT_DIMENSIONS
        ]
        with pytest.raises(FinetuneEvalError, match="切分后验证集为空"):
            split_preferences(one_each)

    def test_two_items_per_dimension_split_one_and_one(self):
        """每个维度 2 条 → 该维度验证侧恰好 1 条、训练侧恰好 1 条（round(0.5) → 0 → 抬到 1）."""
        two_each = [
            make_pair(
                f"pair-{dimension}-{index}",
                dimension=dimension,
                chosen=f"{dimension} 好答案甲乙{index}",
                rejected=f"{dimension} 差答案丙丁{index}",
            )
            for dimension in ALIGNMENT_DIMENSIONS
            for index in range(2)
        ]
        train, valid = split_preferences(two_each)
        assert len(valid) == len(ALIGNMENT_DIMENSIONS)
        assert len(train) == len(ALIGNMENT_DIMENSIONS)

    @pytest.mark.parametrize("dimension", ALIGNMENT_DIMENSIONS)
    def test_each_dimension_gets_exactly_one_validation_item(self, dimension):
        two_each = [
            make_pair(
                f"pair-{name}-{index}",
                dimension=name,
                chosen=f"{name} 好答案甲乙{index}",
                rejected=f"{name} 差答案丙丁{index}",
            )
            for name in ALIGNMENT_DIMENSIONS
            for index in range(2)
        ]
        train, valid = split_preferences(two_each)
        assert sum(1 for pair in valid if pair.dimension == dimension) == 1
        assert sum(1 for pair in train if pair.dimension == dimension) == 1

    def test_single_item_dimension_stays_in_training(self):
        """单条维度整组留在训练侧（``n <= 1 → n_valid = 0``），其余维度照常切."""
        pairs = [
            make_pair("lonely-cite", dimension="citation"),
        ]
        for dimension in ("format", "conciseness", "refusal", "honesty"):
            for index in range(2):
                pairs.append(
                    make_pair(
                        f"pair-{dimension}-{index}",
                        dimension=dimension,
                        chosen=f"{dimension} 好答案甲乙{index}",
                        rejected=f"{dimension} 差答案丙丁{index}",
                    )
                )
        train, valid = split_preferences(pairs)
        assert "lonely-cite" in {pair.id for pair in train}
        assert "lonely-cite" not in {pair.id for pair in valid}
        assert len(valid) == 4

    def test_two_pairs_in_a_single_dimension_still_split(self):
        """只覆盖一个维度也能切：验证侧 1 条、训练侧 1 条."""
        pairs = [
            make_pair("only-01", dimension="citation"),
            make_pair("only-02", dimension="citation"),
        ]
        train, valid = split_preferences(pairs)
        assert len(train) == 1
        assert len(valid) == 1

    def test_outputs_are_plain_lists(self, seed_pairs):
        train, valid = split_preferences(seed_pairs)
        assert isinstance(train, list)
        assert isinstance(valid, list)
        assert all(isinstance(pair, PreferencePair) for pair in train + valid)


class TestPreferenceIO:
    """``write_preferences`` / ``read_preferences``：JSONL 往返、中文可见、行号进错误信息."""

    def test_empty_pairs_write_rejected(self, tmp_path: Path):
        with pytest.raises(FinetuneEvalError, match="不能把空偏好数据写盘"):
            write_preferences(tmp_path / "empty.jsonl", [])

    def test_write_returns_the_target_path(self, tmp_path: Path):
        target = tmp_path / "prefs.jsonl"
        assert write_preferences(target, list(SEED_PAIRS)) == target
        assert target.exists()

    def test_write_creates_parent_directories(self, tmp_path: Path):
        target = tmp_path / "nested" / "deep" / "prefs.jsonl"
        write_preferences(target, [make_pair("io-01")])
        assert target.is_file()

    def test_write_then_read_round_trip(self, tmp_path: Path):
        target = tmp_path / "prefs.jsonl"
        write_preferences(target, list(SEED_PAIRS))
        assert read_preferences(target) == list(SEED_PAIRS)

    def test_round_trip_keeps_metadata(self, tmp_path: Path):
        target = tmp_path / "meta.jsonl"
        pairs = [make_pair("io-02", metadata={"annotator": "甲", "batch": 3})]
        write_preferences(target, pairs)
        assert read_preferences(target)[0].metadata == {"annotator": "甲", "batch": 3}

    def test_file_is_utf8_and_chinese_is_visible(self, tmp_path: Path):
        """``ensure_ascii=False``：偏好数据是给人看的，中文必须以原文落盘."""
        target = tmp_path / "prefs.jsonl"
        write_preferences(target, list(SEED_PAIRS))
        text = target.read_text(encoding="utf-8")
        assert "拒绝" in text
        assert "\\u" not in text

    def test_every_line_is_one_json_object(self, tmp_path: Path):
        target = tmp_path / "prefs.jsonl"
        write_preferences(target, list(SEED_PAIRS))
        lines = [line for line in target.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == SEED_COUNT
        for line in lines:
            assert isinstance(json.loads(line), dict)

    def test_one_pair_written_as_one_line(self, tmp_path: Path):
        target = tmp_path / "single.jsonl"
        write_preferences(target, [make_pair("io-03")])
        assert target.read_text(encoding="utf-8").strip().count("\n") == 0

    def test_missing_file_rejected(self, tmp_path: Path):
        with pytest.raises(FinetuneEvalError, match="找不到偏好数据文件"):
            read_preferences(tmp_path / "not-there.jsonl")

    def test_invalid_json_reports_the_line_number(self, tmp_path: Path):
        target = tmp_path / "broken.jsonl"
        good = json.dumps(make_pair("io-04").to_dict(), ensure_ascii=False)
        target.write_text(f"{good}\n{{不是 json}}\n", encoding="utf-8")
        with pytest.raises(FinetuneEvalError, match="第 2 行不是合法 JSON"):
            read_preferences(target)

    def test_invalid_field_reports_the_line_number(self, tmp_path: Path):
        """字段非法（未知维度）也要带行号：否则 12 条里改哪条靠猜."""
        payload = make_pair("io-05").to_dict()
        payload["dimension"] = "unknown"
        good = json.dumps(make_pair("io-06").to_dict(), ensure_ascii=False)
        broken = json.dumps(payload, ensure_ascii=False)
        target = tmp_path / "bad-field.jsonl"
        target.write_text(f"{good}\n\n{broken}\n", encoding="utf-8")
        with pytest.raises(FinetuneEvalError, match="第 3 行不合法"):
            read_preferences(target)

    def test_missing_required_field_reports_the_line_number(self, tmp_path: Path):
        payload = make_pair("io-07").to_dict()
        del payload["chosen"]
        target = tmp_path / "missing-field.jsonl"
        target.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        with pytest.raises(FinetuneEvalError, match="第 1 行不合法"):
            read_preferences(target)

    def test_normalized_equal_pair_on_disk_reports_the_line_number(self, tmp_path: Path):
        """磁盘上读回来的样本同样要过"两侧不同"这条判据."""
        payload = make_pair("io-08").to_dict()
        payload["chosen"] = "答案。"
        payload["rejected"] = "答案"
        target = tmp_path / "equal-sides.jsonl"
        target.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        with pytest.raises(FinetuneEvalError, match="第 1 行不合法"):
            read_preferences(target)

    def test_blank_lines_are_skipped(self, tmp_path: Path):
        target = tmp_path / "blank.jsonl"
        line = json.dumps(make_pair("io-09").to_dict(), ensure_ascii=False)
        target.write_text(f"\n{line}\n   \n", encoding="utf-8")
        assert read_preferences(target) == [make_pair("io-09")]

    def test_all_blank_file_rejected(self, tmp_path: Path):
        target = tmp_path / "all-blank.jsonl"
        target.write_text("\n\n   \n", encoding="utf-8")
        with pytest.raises(FinetuneEvalError, match="没有任何样本"):
            read_preferences(target)

    def test_unknown_keys_are_ignored(self, tmp_path: Path):
        """未来字段不该让老数据读不进来（``from_dict`` 只取已知键）."""
        payload = make_pair("io-10").to_dict()
        payload["future_field"] = 1
        target = tmp_path / "future.jsonl"
        target.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
        assert read_preferences(target) == [make_pair("io-10")]

    def test_round_trip_of_a_split_side_is_lossless(self, tmp_path: Path):
        """切分产物落盘再读回必须逐字段一致：否则"训练用的是哪一批"说不清."""
        train, valid = split_preferences(list(SEED_PAIRS))
        train_path = tmp_path / "train.jsonl"
        valid_path = tmp_path / "valid.jsonl"
        write_preferences(train_path, train)
        write_preferences(valid_path, valid)
        assert read_preferences(train_path) == train
        assert read_preferences(valid_path) == valid

    def test_rewriting_the_same_batch_gives_the_same_file(self, tmp_path: Path):
        first = tmp_path / "a.jsonl"
        second = tmp_path / "b.jsonl"
        write_preferences(first, list(SEED_PAIRS))
        write_preferences(second, list(SEED_PAIRS))
        assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
