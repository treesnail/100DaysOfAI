"""领域评估集测试（M5-D5）：把"评什么"钉死，再谈分数（suites.py）.

本文件承担"证明结论"角色的四条用例：

1. :meth:`TestSeedSuite.test_fingerprint_matches_the_pinned_value` —— 18 条种子
   用例的内容指纹必须等于 ``93daad77e78b9721``。指纹是**报告能指回"评的是哪一份
   评估集"的唯一凭据**，它一旦漂移，两次运行的分数就再也不能逐行比对；这条断言
   同时也把"种子的内容"整体冻结住（改一个字都会让它红）。
2. :meth:`TestSplitSuite.test_real_split_covers_all_three_difficulties` —— 默认
   切分出的评估集必须同时含 easy / normal / hard。分层粒度取"桶"而不是
   "桶 × 难度"时，最容易踩的坑是评估集被压成同一个难度档，于是
   ``by_difficulty`` 那张表失去信息量、难例结论无从谈起。
3. :meth:`TestSplitSuite.test_single_item_buckets_leave_the_eval_side_empty` /
   :meth:`TestSplitSuite.test_high_ratio_still_keeps_one_item_in_training` ——
   ``n_eval = min(n - 1, max(min_eval_per_bucket, round(n·ratio)))`` 这条算术的
   两端：一头是"每桶 1 条时评估集为空、必须报错"，另一头是"比例再高也要
   给训练侧留 1 条"。少任何一条，"训练集非空"这个保证就只是注释里的一句话。
4. :meth:`TestDetectLeakage.test_identical_item_is_flagged` —— 注入一条与训练集
   逐字相同的用例后，``passed=False`` / ``max_jaccard == 1.0`` /
   ``leakage_rate == 1.0``。数据泄漏不会报错，只会让分数虚高，所以必须有一条
   用例证明这台体检仪**真的会响**；
   :meth:`TestDetectLeakage.test_leakage_rate_never_exceeds_one` 则把
   ``leakage_rate`` 的口径钉住：分母是**评估条目数**，所以它在任何阈值下都
   不超过 1（配对数仍可能远大于评估条目数，那是另一个字段的事）。

定序也在这里被钉住：:meth:`TestBuildSuite.test_build_suite_has_no_seed_parameter`
与 :meth:`TestBuildSuite.test_repeated_calls_give_the_same_order` 一起说明
"顺序恒为桶序 + id"——顺序会漂移的报告没法逐行比对。

所有断言都与 ``suites.py`` 的实现逐字对应；自造用例一律走 ``EvalItem(...)``
（它的 ``__post_init__`` 会立刻校验，非法用例在构造时就抛错）。
"""

from __future__ import annotations

import inspect
import json
import re
from collections import Counter
from dataclasses import FrozenInstanceError, replace

import pytest

from smart_research_agent.finetune_eval.metrics import FinetuneEvalError
from smart_research_agent.finetune_eval.suites import (
    BUCKET_DESCRIPTIONS,
    BUCKETS,
    DEFAULT_EVAL_RATIO,
    DIFFICULTIES,
    EASY_MAX_LOAD,
    LEAKAGE_NGRAM,
    LEAKAGE_THRESHOLD,
    NORMAL_MAX_LOAD,
    SEED_ITEMS,
    TIGHT_CHAR_BUDGET,
    EvalItem,
    audit_suite,
    build_suite,
    detect_leakage,
    jaccard,
    read_suite,
    split_suite,
    suite_fingerprint,
    suite_stats,
    write_suite,
)

#: 种子用例总数（6 桶 × 3 条）
SEED_COUNT = 18

#: 种子评估集的冻结指纹（``sha256`` 前 16 位）——报告靠它认领"评的是哪一份"
PINNED_FINGERPRINT = "93daad77e78b9721"

#: 默认切分（``eval_ratio=0.25`` / ``seed=42``）下被划入评估集的用例 id
PINNED_EVAL_IDS = frozenset({"cite-02", "tool-01", "fmt-03", "fact-02", "ref-02", "conc-03"})


def make_item(item_id: str = "custom-01", **overrides) -> EvalItem:
    """构造一条自造用例：默认最小合法字段，其余用关键字覆盖.

    ``EvalItem`` 在 ``__post_init__`` 里校验，所以这个 helper 只在
    "故意要合法"的场合使用；要构造非法用例直接显式传非法字段。
    """
    payload: dict = {
        "id": item_id,
        "bucket": "citation",
        "difficulty": "normal",
        "instruction": "自造指令",
        "reference": "自造参考答案",
    }
    payload.update(overrides)
    return EvalItem(**payload)


def declared_order(items) -> list[str]:
    """按实现约定的定序规则（桶顺序 + id）算出期望的 id 序列."""
    return [item.id for item in sorted(items, key=lambda x: (BUCKETS.index(x.bucket), x.id))]


def bucket_counts(items) -> dict:
    """把 Counter 转成普通 dict，避免与 dict 比较时的隐式行为."""
    return dict(Counter(item.bucket for item in items))


def difficulty_counts(items) -> dict:
    """同上，按难度分组."""
    return dict(Counter(item.difficulty for item in items))


@pytest.fixture
def suite() -> list[EvalItem]:
    """默认定序的 18 条种子评估集（每个用例新建一份，避免相互污染）."""
    return build_suite()


@pytest.fixture
def seed_by_id() -> dict[str, EvalItem]:
    """种子用例按 id 索引（按 id 取用例比按位置取更耐改）."""
    return {item.id: item for item in SEED_ITEMS}


class TestSeedSuite:
    """种子用例体检：条数 / 分桶 / 唯一性 / 通过校验 / 定序 / 指纹."""

    def test_seed_items_count_is_eighteen(self):
        assert len(SEED_ITEMS) == SEED_COUNT
        assert len(build_suite()) == SEED_COUNT

    @pytest.mark.parametrize("bucket", BUCKETS)
    def test_each_bucket_has_exactly_three_items(self, bucket):
        """6 个桶各 3 条：桶内条数不等会让"每桶留 1 条"的切分保证失去对称性."""
        members = [item for item in SEED_ITEMS if item.bucket == bucket]
        assert len(members) == 3
        assert len({item.id for item in members}) == 3

    def test_seed_ids_are_unique(self):
        ids = [item.id for item in SEED_ITEMS]
        assert len(ids) == len(set(ids)) == SEED_COUNT

    def test_seed_buckets_cover_exactly_the_declared_buckets(self):
        assert set(bucket_counts(SEED_ITEMS)) == set(BUCKETS)
        assert bucket_counts(SEED_ITEMS) == {bucket: 3 for bucket in BUCKETS}

    @pytest.mark.parametrize("item", SEED_ITEMS, ids=[item.id for item in SEED_ITEMS])
    def test_every_seed_item_passes_validate(self, item):
        """每条种子用例都必须自洽（``validate`` 不抛异常，且二次调用幂等）."""
        assert item.validate() is None
        assert item.format_rules.max_chars is None or item.format_rules.max_chars > 0

    @pytest.mark.parametrize("level,expected", [("easy", 4), ("normal", 9), ("hard", 5)])
    def test_declared_difficulty_distribution(self, level, expected):
        assert difficulty_counts(SEED_ITEMS) == {"easy": 4, "normal": 9, "hard": 5}
        assert difficulty_counts(SEED_ITEMS)[level] == expected

    def test_difficulty_labels_are_all_known(self):
        assert set(difficulty_counts(SEED_ITEMS)) <= set(DIFFICULTIES)

    def test_audit_suite_reports_nothing_for_seeds(self):
        """声明难度与结构负载推导难度必须全部一致——否则"难例上的表现"不可信."""
        assert audit_suite(build_suite()) == []
        assert audit_suite(list(SEED_ITEMS)) == []

    def test_audit_suite_flags_a_mislabelled_item(self):
        """体检是"返回清单"而不是抛异常：标注偏差削弱结论，但不该让流程崩掉."""
        mislabelled = make_item("mislabelled", difficulty="easy", required_facts=("a", "b", "c"))
        findings = audit_suite([mislabelled])
        assert len(findings) == 1
        finding = findings[0]
        assert finding["id"] == "mislabelled"
        assert finding["bucket"] == "citation"
        assert finding["declared"] == "easy"
        assert finding["derived"] == "normal"
        assert finding["structural_load"] == 3
        assert "修正难度标注" in finding["hint"]

    def test_refusal_items_are_exactly_three(self):
        refusal_items = [item for item in SEED_ITEMS if item.expect_refusal]
        assert len(refusal_items) == 3
        assert {item.id for item in refusal_items} == {"ref-01", "ref-02", "ref-03"}

    def test_refusal_flag_lives_only_in_the_refusal_bucket(self):
        """``expect_refusal`` 是安全层开关：它只该出现在 refusal 桶里."""
        for item in SEED_ITEMS:
            assert item.expect_refusal == (item.bucket == "refusal")

    def test_fingerprint_matches_the_pinned_value(self):
        """指纹冻结整份种子：改一个字、加一条用例都会让它红."""
        assert suite_fingerprint(build_suite()) == PINNED_FINGERPRINT

    def test_bucket_descriptions_cover_every_bucket(self):
        """桶名不能只有作者懂：每个桶都要有一句人话解释."""
        assert set(BUCKET_DESCRIPTIONS) == set(BUCKETS)
        assert all(BUCKET_DESCRIPTIONS[bucket].strip() for bucket in BUCKETS)

    def test_items_are_frozen(self):
        """``EvalItem`` 是不可变的：评估集在运行期不该被就地改写."""
        item = build_suite()[0]
        with pytest.raises(FrozenInstanceError):
            item.difficulty = "hard"  # type: ignore[misc]

    def test_seed_items_are_a_tuple_of_eval_items(self):
        assert isinstance(SEED_ITEMS, tuple)
        assert all(isinstance(item, EvalItem) for item in SEED_ITEMS)

    def test_buckets_and_difficulties_order(self):
        """桶顺序与难度顺序都是报告顺序，必须与 ``suites`` 的声明一致."""
        assert BUCKETS == (
            "citation",
            "tool_use",
            "format",
            "factuality",
            "refusal",
            "conciseness",
        )
        assert DIFFICULTIES == ("easy", "normal", "hard")
        assert DEFAULT_EVAL_RATIO == 0.25


class TestEvalItemValidation:
    """``EvalItem.validate`` 的失败路径：非法用例必须在构造时就被拦下."""

    def test_minimal_item_is_accepted(self):
        """正对照：最小合法用例（只有五个必填字段）能构造出来."""
        item = make_item()
        assert item.required_facts == ()
        assert item.format_rules.summary_line() == "无格式约束"

    @pytest.mark.parametrize("blank_id", ["", "   ", "\t"])
    def test_blank_id_rejected(self, blank_id):
        with pytest.raises(FinetuneEvalError, match="用例 id 不能为空"):
            make_item(blank_id)

    def test_unknown_bucket_rejected(self):
        with pytest.raises(FinetuneEvalError, match="未知的能力桶"):
            make_item(bucket="unknown")

    @pytest.mark.parametrize("bucket", ["Citation", "tool-use", "工具"])
    def test_near_miss_bucket_names_rejected(self, bucket):
        """桶名大小写与连字符都不容错：拼错会静默改变分桶统计."""
        with pytest.raises(FinetuneEvalError, match="未知的能力桶"):
            make_item(bucket=bucket)

    def test_unknown_difficulty_rejected(self):
        with pytest.raises(FinetuneEvalError, match="未知的难度"):
            make_item(difficulty="medium")

    @pytest.mark.parametrize("difficulty", ["EASY", "Normal", "trivial"])
    def test_near_miss_difficulty_names_rejected(self, difficulty):
        with pytest.raises(FinetuneEvalError, match="未知的难度"):
            make_item(difficulty=difficulty)

    @pytest.mark.parametrize("instruction", ["", "  ", "\n"])
    def test_blank_instruction_rejected(self, instruction):
        with pytest.raises(FinetuneEvalError, match="指令不能为空"):
            make_item(instruction=instruction)

    @pytest.mark.parametrize("reference", ["", "   "])
    def test_blank_reference_rejected(self, reference):
        with pytest.raises(FinetuneEvalError, match="参考答案不能为空"):
            make_item(reference=reference)

    @pytest.mark.parametrize("fact", ["", " ", "  \t"])
    def test_blank_required_fact_rejected(self, fact):
        """空事实点是任何文本的子串，放进去等于这条用例永远满分."""
        with pytest.raises(FinetuneEvalError, match="事实点不能为空字符串"):
            make_item(required_facts=("合法事实", fact))

    @pytest.mark.parametrize("fact", ["", "   "])
    def test_blank_forbidden_fact_rejected(self, fact):
        with pytest.raises(FinetuneEvalError, match="事实点不能为空字符串"):
            make_item(required_facts=("合法事实",), forbidden_facts=(fact,))

    def test_only_must_not_contain_without_budget_rejected(self):
        """只有"禁止"没有"要求"也没有长度上限 = 注定满分的规则，必须拒绝."""
        with pytest.raises(FinetuneEvalError, match="至少要有 must_contain 或 max_chars"):
            make_item(must_not_contain=("禁止片段",))

    @pytest.mark.parametrize("max_chars", [0, -1])
    def test_non_positive_max_chars_rejected(self, max_chars):
        with pytest.raises(FinetuneEvalError, match="max_chars 必须为正整数"):
            make_item(max_chars=max_chars)

    def test_validate_is_idempotent(self):
        item = make_item(required_facts=("a",), must_contain=("结论：",), max_chars=120)
        assert item.validate() is None
        assert item.validate() is None

    def test_error_message_carries_the_item_id(self):
        """错误信息要能定位到具体用例（否则 18 条里改哪条靠猜）."""
        with pytest.raises(FinetuneEvalError, match="custom-07 的指令不能为空"):
            make_item("custom-07", instruction=" ")


class TestEvalItemDerivedFields:
    """``format_rules`` / ``structural_load`` / ``derived_difficulty`` / ``tags``."""

    @pytest.mark.parametrize(
        "facts,forbidden,must_contain,must_not_contain,max_chars,expected_load,expected_level",
        [
            (0, 0, 0, 0, None, 0, "easy"),
            (1, 0, 0, 0, None, 1, "easy"),
            (2, 0, 0, 0, None, 2, "easy"),
            (3, 0, 0, 0, None, 3, "normal"),
            (4, 0, 0, 0, None, 4, "normal"),
            (5, 0, 0, 0, None, 5, "hard"),
            (6, 0, 0, 0, None, 6, "hard"),
            (0, 1, 0, 0, None, 1, "easy"),
            (0, 2, 0, 0, None, 2, "easy"),
            (0, 0, 1, 0, None, 1, "easy"),
            (0, 0, 2, 0, None, 2, "easy"),
            (0, 0, 0, 1, 200, 1, "easy"),
            (0, 0, 0, 0, TIGHT_CHAR_BUDGET, 1, "easy"),
            (0, 0, 0, 0, TIGHT_CHAR_BUDGET + 1, 0, "easy"),
            (0, 0, 0, 0, 1000, 0, "easy"),
            (4, 0, 0, 0, TIGHT_CHAR_BUDGET, 5, "hard"),
            (2, 1, 1, 1, 120, 6, "hard"),
        ],
    )
    def test_structural_load_and_derived_difficulty(
        self,
        facts,
        forbidden,
        must_contain,
        must_not_contain,
        max_chars,
        expected_load,
        expected_level,
    ):
        """结构负载 = 四类清单条数之和 +（长度预算紧则 +1），再按阈值映射难度."""
        item = make_item(
            required_facts=tuple(f"f{index}" for index in range(facts)),
            forbidden_facts=tuple(f"x{index}" for index in range(forbidden)),
            must_contain=tuple(f"c{index}：" for index in range(must_contain)),
            must_not_contain=tuple(f"n{index}" for index in range(must_not_contain)),
            max_chars=max_chars,
        )
        assert item.structural_load == expected_load
        assert item.derived_difficulty == expected_level

    @pytest.mark.parametrize(
        "base_facts,next_level",
        [(0, "easy"), (1, "easy"), (2, "normal"), (3, "normal"), (4, "hard")],
    )
    def test_adding_one_fact_raises_the_load_and_may_raise_the_level(self, base_facts, next_level):
        """加一个事实点让负载恰好 +1，并可能把难度推到下一档（这就是"更难"的定义）."""
        base = make_item(required_facts=tuple(f"f{index}" for index in range(base_facts)))
        bumped = replace(base, required_facts=(*base.required_facts, "新增事实"))
        assert bumped.structural_load == base.structural_load + 1
        assert bumped.derived_difficulty == next_level

    @pytest.mark.parametrize(
        "max_chars,expected_bonus",
        [
            (None, 0),
            (1, 1),
            (149, 1),
            (TIGHT_CHAR_BUDGET, 1),
            (TIGHT_CHAR_BUDGET + 1, 0),
            (10_000, 0),
        ],
    )
    def test_tight_budget_adds_one_point_of_load(self, max_chars, expected_bonus):
        """长度预算紧（``max_chars <= TIGHT_CHAR_BUDGET``）时负载 +1：同样是"多做一件事"."""
        item = make_item(max_chars=max_chars)
        assert item.structural_load == expected_bonus

    def test_tight_budget_threshold_is_inclusive(self):
        assert (
            make_item(max_chars=TIGHT_CHAR_BUDGET).structural_load
            == make_item(max_chars=TIGHT_CHAR_BUDGET + 1).structural_load + 1
        )

    @pytest.mark.parametrize(
        "facts,expected_level",
        [
            (EASY_MAX_LOAD, "easy"),
            (EASY_MAX_LOAD + 1, "normal"),
            (NORMAL_MAX_LOAD, "normal"),
            (NORMAL_MAX_LOAD + 1, "hard"),
        ],
    )
    def test_difficulty_thresholds_boundary(self, facts, expected_level):
        """难度阈值本身是闭区间上界：``load == EASY_MAX_LOAD`` 仍是 easy."""
        item = make_item(required_facts=tuple(f"f{index}" for index in range(facts)))
        assert item.structural_load == facts
        assert item.derived_difficulty == expected_level

    def test_format_rules_property_mirrors_the_three_fields(self):
        item = make_item(
            must_contain=("结论：", "要点："), must_not_contain=("无关",), max_chars=120
        )
        rules = item.format_rules
        assert rules.must_contain == ("结论：", "要点：")
        assert rules.must_not_contain == ("无关",)
        assert rules.max_chars == 120

    def test_format_rules_default_is_empty(self):
        rules = make_item().format_rules
        assert rules.must_contain == ()
        assert rules.must_not_contain == ()
        assert rules.max_chars is None

    @pytest.mark.parametrize("item", SEED_ITEMS, ids=[item.id for item in SEED_ITEMS])
    def test_tags_equal_bucket_and_difficulty(self, item):
        assert item.tags == (item.bucket, item.difficulty)
        assert len(item.tags) == 2

    def test_load_counts_every_list_kind(self):
        """四类清单都计入负载——漏掉任何一类，"结构负载"就不叫结构负载了."""
        item = make_item(
            required_facts=("a",),
            forbidden_facts=("b",),
            must_contain=("结论：",),
            must_not_contain=("c",),
            max_chars=None,
        )
        assert item.structural_load == 4

    def test_declared_and_derived_levels_agree_for_seeds(self):
        for item in SEED_ITEMS:
            assert item.derived_difficulty == item.difficulty


class TestEvalItemSerialization:
    """``to_dict`` / ``from_dict`` / ``summary_line``：落盘与回读必须无损."""

    @pytest.mark.parametrize("item", SEED_ITEMS, ids=[item.id for item in SEED_ITEMS])
    def test_to_dict_from_dict_round_trip(self, item):
        assert EvalItem.from_dict(item.to_dict()) == item

    def test_to_dict_keeps_tuples(self):
        """``to_dict`` 不落盘派生属性（``format_rules`` 是 @property）."""
        payload = make_item(required_facts=("a", "b")).to_dict()
        assert payload["required_facts"] == ("a", "b")
        assert "format_rules" not in payload
        assert set(payload) == {
            "id",
            "bucket",
            "difficulty",
            "instruction",
            "reference",
            "required_facts",
            "forbidden_facts",
            "must_contain",
            "must_not_contain",
            "max_chars",
            "expect_refusal",
            "metadata",
        }

    def test_from_dict_restores_tuples_from_lists(self):
        """JSON 里没有元组：回读时四个清单字段必须重新变成元组（哈希/比较才稳定）."""
        payload = make_item(
            required_facts=("a",),
            forbidden_facts=("b",),
            must_contain=("结论：",),
            must_not_contain=("c",),
        ).to_dict()
        restored = EvalItem.from_dict(json.loads(json.dumps(payload)))
        assert restored.required_facts == ("a",)
        assert restored.forbidden_facts == ("b",)
        assert restored.must_contain == ("结论：",)
        assert restored.must_not_contain == ("c",)
        assert all(
            isinstance(field, tuple)
            for field in (
                restored.required_facts,
                restored.forbidden_facts,
                restored.must_contain,
                restored.must_not_contain,
            )
        )

    def test_from_dict_ignores_unknown_keys(self):
        payload = make_item().to_dict()
        payload["future_field"] = 1
        payload["format_rules"] = {"max_chars": 1}
        restored = EvalItem.from_dict(payload)
        assert restored == make_item()
        assert "future_field" not in restored.to_dict()

    def test_from_dict_missing_required_key_raises_type_error(self):
        with pytest.raises(TypeError):
            EvalItem.from_dict({"id": "only-id"})

    def test_from_dict_still_validates(self):
        payload = make_item().to_dict()
        payload["bucket"] = "unknown"
        with pytest.raises(FinetuneEvalError, match="未知的能力桶"):
            EvalItem.from_dict(payload)

    def test_to_dict_is_json_serializable(self):
        """经过一次真正的 ``json.dumps`` / ``loads`` 往返后仍能还原成同一条用例
        （JSON 里没有元组，四个清单字段回读时由 ``from_dict`` 转回元组）."""
        payload = make_item(metadata={"day": 53}).to_dict()
        round_tripped = json.loads(json.dumps(payload, ensure_ascii=False))
        assert EvalItem.from_dict(round_tripped) == make_item(metadata={"day": 53})

    @pytest.mark.parametrize("item", SEED_ITEMS, ids=[item.id for item in SEED_ITEMS])
    def test_summary_line_is_printable(self, item):
        line = item.summary_line()
        assert line.startswith(f"{item.id} | {item.bucket}/{item.difficulty}")
        assert f"负载 {item.structural_load}" in line
        assert f"事实点 {len(item.required_facts)}" in line
        assert "格式：" in line

    def test_summary_line_mentions_format_rules(self):
        line = make_item(must_contain=("结论：",), max_chars=120).summary_line()
        assert "必须含 结论：" in line
        assert "不超 120 字" in line


class TestBuildSuite:
    """``build_suite``：筛选 + 定序（顺序恒为"桶序 + id"，且**没有** ``seed`` 参数）."""

    def test_returns_everything_by_default(self, suite):
        assert len(suite) == SEED_COUNT
        assert {item.id for item in suite} == {item.id for item in SEED_ITEMS}

    @pytest.mark.parametrize("bucket", BUCKETS)
    def test_bucket_filter(self, bucket):
        selected = build_suite(buckets=(bucket,))
        assert len(selected) == 3
        assert all(item.bucket == bucket for item in selected)

    @pytest.mark.parametrize("level,expected", [("easy", 4), ("normal", 9), ("hard", 5)])
    def test_difficulty_filter(self, level, expected):
        selected = build_suite(difficulties=(level,))
        assert len(selected) == expected
        assert all(item.difficulty == level for item in selected)

    def test_combined_filter(self):
        selected = build_suite(buckets=("citation",), difficulties=("hard",))
        assert [item.id for item in selected] == ["cite-03"]

    def test_multiple_buckets_keep_the_declared_order(self):
        selected = build_suite(buckets=("refusal", "citation"))
        assert [item.bucket for item in selected] == ["citation"] * 3 + ["refusal"] * 3

    def test_no_match_raises(self):
        with pytest.raises(FinetuneEvalError, match="筛选条件没有命中任何用例"):
            build_suite(buckets=("citation",), difficulties=("easy",))

    def test_unknown_bucket_name_yields_no_match(self):
        with pytest.raises(FinetuneEvalError, match="筛选条件没有命中任何用例"):
            build_suite(buckets=("unknown",))

    def test_repeated_calls_give_the_same_order(self):
        """重复调用给出同一份顺序：定序里没有随机性，报告才能逐行比对.

        三条断言一起钉住"顺序是纯函数"：同一份输入调两次一样、显式传
        ``SEED_ITEMS`` 与使用缺省值一样（缺省值就是它）、并且等于实现
        声明过的"桶序 + id"。
        """
        first = [item.id for item in build_suite()]
        second = [item.id for item in build_suite()]
        assert first == second
        assert build_suite() == build_suite(list(SEED_ITEMS))
        assert first == declared_order(SEED_ITEMS)

    def test_build_suite_has_no_seed_parameter(self):
        """``build_suite`` **没有** ``seed`` 参数：定序不靠随机数.

        随机性只出现在真正需要它的地方（``split_suite`` 里"同一难度档取哪一条"、
        自助法重采样），它们各自带种子；把一个用不上的 ``seed`` 留在签名里，
        会让读者以为顺序会随种子变化，而"顺序变了"正是报告无法逐行比对的原因。
        """
        parameters = inspect.signature(build_suite).parameters
        assert "seed" not in parameters
        assert tuple(parameters) == ("items", "buckets", "difficulties")

    def test_order_is_bucket_order_then_id(self, suite):
        assert [item.id for item in suite] == declared_order(suite)
        indexes = [BUCKETS.index(item.bucket) for item in suite]
        assert indexes == sorted(indexes)

    def test_custom_items_are_filtered_and_sorted(self):
        items = (
            make_item("b-01", bucket="tool_use"),
            make_item("a-02", bucket="citation"),
            make_item("a-01", bucket="citation"),
        )
        selected = build_suite(items)
        assert [item.id for item in selected] == ["a-01", "a-02", "b-01"]

    def test_filter_keeps_only_requested_difficulties(self):
        selected = build_suite(difficulties=("easy", "hard"))
        assert difficulty_counts(selected) == {"easy": 4, "hard": 5}

    def test_returns_a_new_list(self, suite):
        """每次调用都要给出新对象：就地改写返回的列表不该污染种子."""
        first = build_suite()
        first.clear()
        assert len(build_suite()) == SEED_COUNT
        assert len(SEED_ITEMS) == SEED_COUNT


class TestSuiteStats:
    """``suite_stats``：桶 / 难度分布、负载、字数预算、拒答数、指纹."""

    def test_empty_suite_rejected(self):
        with pytest.raises(FinetuneEvalError, match="不能对空评估集做统计"):
            suite_stats([])

    def test_totals_and_distributions(self, suite):
        stats = suite_stats(suite)
        assert stats["total"] == SEED_COUNT
        assert stats["by_bucket"] == {bucket: 3 for bucket in BUCKETS}
        assert stats["by_difficulty"] == {"easy": 4, "normal": 9, "hard": 5}
        assert stats["refusal_items"] == 3

    def test_by_bucket_only_keeps_non_zero_buckets(self):
        """零计数桶不进报告：一张全是 0 的表只会稀释真正需要看的那几行."""
        stats = suite_stats(build_suite(buckets=("citation",)))
        assert stats["by_bucket"] == {"citation": 3}
        assert stats["by_difficulty"] == {"normal": 2, "hard": 1}

    def test_fact_totals(self, suite):
        stats = suite_stats(suite)
        assert stats["required_facts_total"] == 39
        assert stats["forbidden_facts_total"] == 3
        assert stats["required_facts_total"] == sum(len(item.required_facts) for item in suite)

    def test_load_statistics(self, suite):
        stats = suite_stats(suite)
        assert stats["mean_structural_load"] == pytest.approx(3.6111, abs=1e-4)
        assert stats["max_structural_load"] == 6
        assert stats["max_structural_load"] == max(item.structural_load for item in suite)

    def test_mean_char_budget(self, suite):
        stats = suite_stats(suite)
        assert stats["mean_char_budget"] == pytest.approx(198.8889, abs=1e-4)
        assert stats["mean_char_budget"] == pytest.approx(
            sum(item.max_chars for item in suite) / len(suite), abs=1e-4
        )

    def test_mean_char_budget_is_none_without_budgets(self):
        """一条预算都没有时返回 ``None``，而不是拿 0 当"没有限制"."""
        items = [make_item("no-budget-01"), make_item("no-budget-02", required_facts=("a",))]
        stats = suite_stats(items)
        assert stats["mean_char_budget"] is None
        assert stats["mean_structural_load"] == pytest.approx(0.5)
        assert stats["max_structural_load"] == 1

    def test_budget_with_the_tight_flag_is_counted_once(self):
        stats = suite_stats([make_item(max_chars=TIGHT_CHAR_BUDGET)])
        assert stats["mean_char_budget"] == TIGHT_CHAR_BUDGET
        assert stats["mean_structural_load"] == 1

    def test_fingerprint_field_matches_the_helper(self, suite):
        assert suite_stats(suite)["fingerprint"] == suite_fingerprint(suite)

    @pytest.mark.parametrize("item_id,bucket", [("cite-01", "citation"), ("ref-02", "refusal")])
    def test_single_item_statistics(self, item_id, bucket, seed_by_id):
        stats = suite_stats([seed_by_id[item_id]])
        assert stats["total"] == 1
        assert stats["by_bucket"] == {bucket: 1}
        assert stats["max_structural_load"] >= stats["mean_structural_load"]


class TestSuiteFingerprint:
    """``suite_fingerprint``：报告要能指回"评的是哪一份评估集"."""

    def test_empty_suite_rejected(self):
        with pytest.raises(FinetuneEvalError, match="不能对空评估集取指纹"):
            suite_fingerprint([])

    def test_same_content_is_stable(self, suite):
        assert suite_fingerprint(list(suite)) == suite_fingerprint(list(suite))

    def test_pinned_value(self, suite):
        assert suite_fingerprint(suite) == PINNED_FINGERPRINT

    def test_fingerprint_is_sixteen_hex_characters(self, suite):
        assert re.fullmatch(r"[0-9a-f]{16}", suite_fingerprint(suite))

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(
                lambda item: replace(item, reference=item.reference + "。"), id="reference"
            ),
            pytest.param(
                lambda item: replace(item, instruction=item.instruction + "？"), id="instruction"
            ),
            pytest.param(
                lambda item: replace(item, required_facts=(*item.required_facts, "另一个事实")),
                id="facts",
            ),
            pytest.param(
                lambda item: replace(item, max_chars=(item.max_chars or 100) + 1),
                id="max-chars",
            ),
            pytest.param(lambda item: replace(item, metadata={"note": "改动"}), id="metadata"),
        ],
    )
    def test_changing_one_item_changes_the_fingerprint(self, suite, mutate):
        """改一个字、加一个事实点、调一次字数上限——指纹都必须变."""
        baseline = suite_fingerprint(suite)
        changed = list(suite)
        changed[0] = mutate(changed[0])
        assert suite_fingerprint(changed) != baseline

    def test_dropping_an_item_changes_the_fingerprint(self, suite):
        assert suite_fingerprint(suite[:-1]) != suite_fingerprint(suite)

    def test_reordering_changes_the_fingerprint(self, suite):
        """指纹包含顺序：同一个集合换个顺序就不是"同一份评估过程"."""
        reversed_suite = list(reversed(suite))
        assert suite_fingerprint(reversed_suite) != suite_fingerprint(suite)

    def test_key_order_in_metadata_does_not_change_the_fingerprint(self):
        """规范化 JSON（``sort_keys``）是"指纹能回答问题"的前提."""
        left = suite_fingerprint([make_item(metadata={"a": 1, "b": 2})])
        right = suite_fingerprint([make_item(metadata={"b": 2, "a": 1})])
        assert left == right


class TestJaccard:
    """``jaccard``：泄漏体检的距离函数，空集合的语义必须显式定义."""

    def test_both_sides_empty_is_zero(self):
        """两边都空 = 无信息，不算泄漏（而不是"完全一致"）."""
        assert jaccard(set(), set()) == 0.0

    def test_identical_sets_is_one(self):
        grams = {("a", "b"), ("b", "c")}
        assert jaccard(grams, grams) == 1.0

    def test_one_side_empty_is_zero(self):
        assert jaccard({("a",)}, set()) == 0.0
        assert jaccard(set(), {("a",)}) == 0.0

    def test_disjoint_sets_is_zero(self):
        assert jaccard({("a",)}, {("b",)}) == 0.0

    def test_partial_overlap(self):
        left = {("a",), ("b",)}
        right = {("b",), ("c",)}
        assert jaccard(left, right) == pytest.approx(1 / 3)

    def test_symmetry(self):
        left = {("a",), ("b",), ("c",)}
        right = {("b",), ("d",)}
        assert jaccard(left, right) == jaccard(right, left)

    def test_is_bounded_by_one(self):
        left = {("a",), ("b",)}
        right = {("a",), ("b",), ("c",)}
        assert 0.0 <= jaccard(left, right) <= 1.0


class TestDetectLeakage:
    """``detect_leakage``：训练集污染评估集会静默抬高分数，所以必须先体检."""

    @pytest.mark.parametrize("ngram", [0, -1, -3])
    def test_non_positive_ngram_rejected(self, suite, ngram):
        with pytest.raises(FinetuneEvalError, match="ngram 必须为正整数"):
            detect_leakage(suite, suite, ngram=ngram)

    @pytest.mark.parametrize("threshold", [-0.01, -1.0, 1.01, 2.0])
    def test_out_of_range_threshold_rejected(self, suite, threshold):
        with pytest.raises(FinetuneEvalError, match="threshold 必须落在"):
            detect_leakage(suite, suite, threshold=threshold)

    def test_identical_item_is_flagged(self, suite):
        """注入一条与训练集逐字相同的用例：体检仪必须立刻响."""
        train, _ = split_suite(suite)
        report = detect_leakage(train, [train[0]])
        assert report["passed"] is False
        assert report["max_jaccard"] == 1.0
        assert report["leakage_rate"] == 1.0
        assert report["suspicious_pairs"] == [
            {"train_id": train[0].id, "eval_id": train[0].id, "jaccard": 1.0}
        ]

    def test_identical_item_at_threshold_one_is_still_flagged(self, suite):
        """阈值判定是闭区间：``similarity >= threshold``，等于 1.0 时也拦."""
        train, _ = split_suite(suite)
        report = detect_leakage(train, [train[0]], threshold=1.0)
        assert report["passed"] is False
        assert len(report["suspicious_pairs"]) == 1

    def test_real_split_passes(self, suite):
        """18 条彼此独立的用例：最大相似度远低于 0.5."""
        train, evaluation = split_suite(suite)
        report = detect_leakage(train, evaluation)
        assert report["passed"] is True
        assert report["max_jaccard"] < 0.1
        assert report["suspicious_pairs"] == []

    def test_real_split_numbers(self, suite):
        train, evaluation = split_suite(suite)
        report = detect_leakage(train, evaluation)
        assert report["ngram"] == LEAKAGE_NGRAM == 3
        assert report["threshold"] == LEAKAGE_THRESHOLD == 0.5
        assert report["train_items"] == 12
        assert report["eval_items"] == 6
        assert report["max_jaccard"] == pytest.approx(0.054348, abs=1e-6)
        assert report["leakage_rate"] == 0.0

    def test_custom_ngram_is_recorded_and_raises_the_overlap(self, suite):
        """降阶会抬高重叠：unigram 下相似度上去了，但仍在阈值以下."""
        train, evaluation = split_suite(suite)
        report = detect_leakage(train, evaluation, ngram=1)
        assert report["ngram"] == 1
        assert report["max_jaccard"] > 0.3
        assert report["passed"] is True

    def test_empty_evaluation_side(self, suite):
        """没有评估条目时"无泄漏"是可判定的（分母为空 → 泄漏率 0）."""
        train, _ = split_suite(suite)
        report = detect_leakage(train, [])
        assert report["eval_items"] == 0
        assert report["max_jaccard"] == 0.0
        assert report["leakage_rate"] == 0.0
        assert report["passed"] is True

    def test_empty_train_side(self, suite):
        """训练侧为空时没有任何可配对的 gram，绝不误报."""
        report = detect_leakage([], list(suite))
        assert report["train_items"] == 0
        assert report["max_jaccard"] == 0.0
        assert report["passed"] is True

    def test_threshold_zero_flags_every_pair(self, suite):
        """阈值下界 0.0 是允许的，但此时**所有配对**都算可疑（含 0 相似度）.

        ``leakage_rate`` 的分母是**评估条目数**（不是配对数）：阈值取 0 时
        每个评估条目都至少与训练集里的某一条配上了，于是它是 1.0——而
        ``suspicious_pairs`` 是"训练 × 评估"的配对数，可以远大于 1。
        两个数字各回答一个问题："有多少评估条目被污染"与"有多少对可疑"。
        """
        train, evaluation = split_suite(suite)
        report = detect_leakage(train, evaluation, threshold=0.0)
        assert len(report["suspicious_pairs"]) >= len(evaluation)
        assert len(report["suspicious_pairs"]) == len(train) * len(evaluation)
        assert report["leakage_rate"] == 1.0
        assert report["leakage_rate"] <= 1.0
        assert len(report["leaked_eval_items"]) == len(evaluation)
        assert report["passed"] is False

    @pytest.mark.parametrize("threshold", [0.0, 0.25, 0.5, 1.0])
    def test_leakage_rate_never_exceeds_one(self, suite, threshold):
        """``leakage_rate`` 是"被可疑命中的评估条目占比"，任何阈值下都在 [0, 1].

        它是"率"，所以必须不超过 1：一个名叫"率"却大于 1 的数字没法解释。
        这里顺带把它与分母的关系也钉住——``leakage_rate`` 恰好等于
        ``leaked_eval_items`` 的条数除以评估条目数。
        """
        train, evaluation = split_suite(suite)
        report = detect_leakage(train, evaluation, threshold=threshold)
        assert 0.0 <= report["leakage_rate"] <= 1.0
        assert report["leakage_rate"] == pytest.approx(
            len(report["leaked_eval_items"]) / len(evaluation)
        )

    def test_report_keeps_the_requested_threshold(self, suite):
        train, evaluation = split_suite(suite)
        report = detect_leakage(train, evaluation, threshold=0.9)
        assert report["threshold"] == 0.9
        assert report["passed"] is True


class TestSplitSuite:
    """``split_suite``：按桶分层切分，两边都非空，且评估集必须含难例."""

    @pytest.mark.parametrize("eval_ratio", [-0.1, -1.0, 1.0, 1.5])
    def test_out_of_range_ratio_rejected(self, suite, eval_ratio):
        with pytest.raises(FinetuneEvalError, match="eval_ratio 必须落在"):
            split_suite(suite, eval_ratio=eval_ratio)

    @pytest.mark.parametrize("min_eval", [-1, -5])
    def test_negative_min_eval_per_bucket_rejected(self, suite, min_eval):
        with pytest.raises(FinetuneEvalError, match="min_eval_per_bucket 不能为负数"):
            split_suite(suite, min_eval_per_bucket=min_eval)

    def test_empty_suite_rejected(self):
        with pytest.raises(FinetuneEvalError, match="不能对空评估集做切分"):
            split_suite([])

    def test_default_split_sizes(self, suite):
        train, evaluation = split_suite(suite)
        assert len(train) == 12
        assert len(evaluation) == 6

    def test_default_split_is_disjoint_and_complete(self, suite):
        train, evaluation = split_suite(suite)
        train_ids = {item.id for item in train}
        eval_ids = {item.id for item in evaluation}
        assert train_ids & eval_ids == set()
        assert train_ids | eval_ids == {item.id for item in suite}
        assert len(train) + len(evaluation) == len(suite)

    def test_real_split_covers_all_three_difficulties(self, suite):
        """轮转取法的意义所在：评估集不能退化成"同一个难度档的六条"."""
        _, evaluation = split_suite(suite)
        assert {item.difficulty for item in evaluation} == set(DIFFICULTIES)

    def test_default_split_difficulty_distribution(self, suite):
        _, evaluation = split_suite(suite)
        assert difficulty_counts(evaluation) == {"easy": 1, "normal": 3, "hard": 2}
        assert difficulty_counts(evaluation)["hard"] >= 1

    def test_default_split_ids_are_pinned(self, suite):
        """``seed=42`` 下"每个难度档里取哪一条"是确定的（报告要能逐行比对）."""
        _, evaluation = split_suite(suite)
        assert {item.id for item in evaluation} == PINNED_EVAL_IDS

    def test_both_sides_keep_every_bucket(self, suite):
        train, evaluation = split_suite(suite)
        assert bucket_counts(train) == {bucket: 2 for bucket in BUCKETS}
        assert bucket_counts(evaluation) == {bucket: 1 for bucket in BUCKETS}

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_same_seed_is_deterministic(self, suite, seed):
        first = split_suite(suite, seed=seed)
        second = split_suite(suite, seed=seed)
        assert [item.id for item in first[0]] == [item.id for item in second[0]]
        assert [item.id for item in first[1]] == [item.id for item in second[1]]

    def test_seed_decides_which_item_within_a_level(self, suite):
        """种子决定"同一个难度档里取哪一条"，但不改变两侧的条数."""
        default_eval = {item.id for item in split_suite(suite, seed=42)[1]}
        other_eval = {item.id for item in split_suite(suite, seed=0)[1]}
        assert default_eval != other_eval
        assert len(default_eval) == len(other_eval) == 6

    def test_single_item_buckets_leave_the_eval_side_empty(self, suite):
        """每桶 1 条时整桶留在训练侧 → 评估集为空，必须报错而不是"静默空集"."""
        singles = [next(item for item in suite if item.bucket == bucket) for bucket in BUCKETS]
        assert len(singles) == 6
        with pytest.raises(FinetuneEvalError, match="切分后评估集为空"):
            split_suite(singles)

    def test_high_ratio_still_keeps_one_item_in_training(self, suite):
        """``min(n - 1, …)`` 是"训练集非空"的硬保证：比例再高也要留 1 条."""
        train, evaluation = split_suite(suite, eval_ratio=0.9)
        assert len(evaluation) == 12
        assert len(train) == 6
        assert bucket_counts(train) == {bucket: 1 for bucket in BUCKETS}
        assert bucket_counts(evaluation) == {bucket: 2 for bucket in BUCKETS}

    @pytest.mark.parametrize(
        "eval_ratio,expected_eval", [(0.0, 6), (0.25, 6), (0.5, 12), (0.75, 12), (0.99, 12)]
    )
    def test_quota_follows_the_arithmetic(self, suite, eval_ratio, expected_eval):
        """``n_eval = min(n-1, max(min_eval, round(n·ratio)))``，``n = 3``.

        ``0.25 → round(0.75) = 1``、``0.5 → round(1.5) = 2``（银行家舍入）、
        ``0.75 → round(2.25) = 2``、``0.99 → 3`` 被 ``n - 1 = 2`` 封顶。
        """
        train, evaluation = split_suite(suite, eval_ratio=eval_ratio)
        assert len(evaluation) == expected_eval
        assert len(train) == SEED_COUNT - expected_eval

    def test_zero_ratio_with_zero_min_eval_leaves_eval_empty(self):
        """比例与下限同时为 0 → 配额 0 → 评估集为空（必须报错，不能静默通过）."""
        with pytest.raises(FinetuneEvalError, match="切分后评估集为空"):
            split_suite(build_suite(), eval_ratio=0.0, min_eval_per_bucket=0)

    def test_min_eval_per_bucket_raises_the_quota(self, suite):
        """``min_eval_per_bucket=2`` 时每桶取 2 条（仍在 ``n - 1`` 以内）."""
        train, evaluation = split_suite(suite, min_eval_per_bucket=2)
        assert len(evaluation) == 12
        assert len(train) == 6
        assert bucket_counts(evaluation) == {bucket: 2 for bucket in BUCKETS}

    def test_min_eval_per_bucket_zero_matches_the_default_quota(self, suite):
        default_eval = {item.id for item in split_suite(suite)[1]}
        zero_min_eval = {item.id for item in split_suite(suite, min_eval_per_bucket=0)[1]}
        assert default_eval == zero_min_eval

    def test_quota_is_capped_by_n_minus_one(self, suite):
        """哪怕下限给到 99，每桶也只拿走 ``n - 1`` 条（训练侧永不空）."""
        train, evaluation = split_suite(suite, min_eval_per_bucket=99)
        assert len(evaluation) == 12
        assert len(train) == 6

    def test_custom_bucket_sizes_are_handled(self, suite):
        """只有 2 条的桶也能切：n=2 → 配额 min(1, …) = 1."""
        two_item_bucket = [item for item in suite if item.bucket == "citation"][:2]
        single_group = [item for item in suite if item.bucket == "refusal"][:1]
        train, evaluation = split_suite([*two_item_bucket, *single_group])
        assert len(evaluation) == 1
        assert len(train) == 2

    def test_training_side_empty_is_impossible_for_these_seeds(self, suite):
        """反证：只要每桶 ≥ 2 条，训练侧就不可能被掏空."""
        train, _ = split_suite(suite, eval_ratio=0.999)
        assert train
        assert len(train) == len(BUCKETS)


class TestSuitePersistence:
    """``write_suite`` / ``read_suite``：JSONL 往返 + 行号进错误信息."""

    def test_empty_suite_write_rejected(self, tmp_path):
        with pytest.raises(FinetuneEvalError, match="不能把空评估集写盘"):
            write_suite(tmp_path / "suite.jsonl", [])

    def test_write_creates_parent_directories(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "suite.jsonl"
        assert not path.parent.exists()
        written = write_suite(path, build_suite())
        assert written == path
        assert path.is_file()

    def test_write_returns_the_target_path(self, tmp_path):
        path = tmp_path / "suite.jsonl"
        assert write_suite(str(path), build_suite()) == path

    def test_round_trip_preserves_every_item(self, tmp_path, suite):
        path = write_suite(tmp_path / "suite.jsonl", suite)
        restored = read_suite(path)
        assert restored == list(suite)
        assert [item.id for item in restored] == [item.id for item in suite]

    def test_round_trip_preserves_field_types(self, tmp_path, seed_by_id):
        item = seed_by_id["cite-03"]
        path = write_suite(tmp_path / "suite.jsonl", [item])
        restored = read_suite(path)[0]
        assert restored.required_facts == item.required_facts
        assert restored.forbidden_facts == item.forbidden_facts
        assert restored.must_contain == item.must_contain
        assert restored.max_chars == item.max_chars
        assert restored.expect_refusal is False
        assert restored.metadata == {}

    def test_file_keeps_chinese_text_unescaped(self, tmp_path, suite):
        """评估集是要被人 review 的：留中文原文，不写 ``\\uXXXX``."""
        path = write_suite(tmp_path / "suite.jsonl", suite)
        text = path.read_text(encoding="utf-8")
        assert "来源：" in text
        assert "\\u" not in text
        assert "peft/layers.py" in text

    def test_file_is_one_json_object_per_line(self, tmp_path, suite):
        path = write_suite(tmp_path / "suite.jsonl", suite)
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        lines = text.splitlines()
        assert len(lines) == SEED_COUNT
        for lineno, line in enumerate(lines, start=1):
            payload = json.loads(line)
            assert payload["id"] == suite[lineno - 1].id
            assert sorted(payload) == sorted(line_keys())

    def test_metadata_survives_the_round_trip(self, tmp_path):
        item = make_item("meta-01", metadata={"day": 53, "nested": {"a": 1}})
        restored = read_suite(write_suite(tmp_path / "suite.jsonl", [item]))[0]
        assert restored.metadata == {"day": 53, "nested": {"a": 1}}
        assert restored == item

    def test_blank_lines_are_skipped(self, tmp_path, seed_by_id):
        path = tmp_path / "suite.jsonl"
        first = json.dumps(seed_by_id["cite-01"].to_dict(), ensure_ascii=False, sort_keys=True)
        path.write_text(f"\n{first}\n\n", encoding="utf-8")
        assert [item.id for item in read_suite(path)] == ["cite-01"]

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(FinetuneEvalError, match="找不到评估集文件"):
            read_suite(tmp_path / "no-such-suite.jsonl")

    def test_invalid_json_reports_the_line_number(self, tmp_path, seed_by_id):
        """行号进错误信息：18 行里坏一行，不报行号等于让人逐行试."""
        path = tmp_path / "suite.jsonl"
        first = json.dumps(seed_by_id["cite-01"].to_dict(), ensure_ascii=False, sort_keys=True)
        path.write_text(f"{first}\n{{不是合法 JSON\n", encoding="utf-8")
        with pytest.raises(FinetuneEvalError, match="第 2 行不是合法 JSON"):
            read_suite(path)

    def test_invalid_json_on_the_first_line(self, tmp_path):
        path = tmp_path / "suite.jsonl"
        path.write_text("{broken\n", encoding="utf-8")
        with pytest.raises(FinetuneEvalError, match="第 1 行不是合法 JSON"):
            read_suite(path)

    def test_invalid_field_reports_the_line_number(self, tmp_path, seed_by_id):
        path = tmp_path / "suite.jsonl"
        first = json.dumps(seed_by_id["cite-01"].to_dict(), ensure_ascii=False, sort_keys=True)
        bad = {**seed_by_id["tool-01"].to_dict(), "bucket": "unknown"}
        path.write_text(f"{first}\n{json.dumps(bad, ensure_ascii=False)}\n", encoding="utf-8")
        with pytest.raises(FinetuneEvalError, match="第 2 行不合法"):
            read_suite(path)

    def test_missing_required_field_reports_the_line_number(self, tmp_path):
        path = tmp_path / "suite.jsonl"
        path.write_text('{"id": "only-id"}\n', encoding="utf-8")
        with pytest.raises(FinetuneEvalError, match="第 1 行不合法"):
            read_suite(path)

    def test_blank_only_file_rejected(self, tmp_path):
        path = tmp_path / "suite.jsonl"
        path.write_text("\n\n   \n", encoding="utf-8")
        with pytest.raises(FinetuneEvalError, match="没有任何用例"):
            read_suite(path)

    def test_empty_file_rejected(self, tmp_path):
        path = tmp_path / "suite.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(FinetuneEvalError, match="没有任何用例"):
            read_suite(path)

    def test_read_returns_a_list_of_eval_items(self, tmp_path, suite):
        restored = read_suite(write_suite(tmp_path / "suite.jsonl", suite))
        assert isinstance(restored, list)
        assert all(isinstance(item, EvalItem) for item in restored)


def line_keys() -> set[str]:
    """``to_dict`` 的字段表（落盘行的键集合）."""
    return set(make_item().to_dict())
