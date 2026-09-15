"""领域文本指标测试（M5-D5）：把"哪个答案更好"拆成可复现的数字.

``finetune_eval/metrics.py`` 是本课唯一一个**完全纯函数**的模块：同一对输入
必须给出同一个输出，不依赖模型、不依赖随机数、不依赖环境。这个性质决定了
本文件的断言方式——**期望值全部手算**，不拿被测函数自己算出来的中间量
当基准。三条承担"证明结论"角色的用例：

1. :meth:`TestNormalizeText.test_delta_w_notation_forms_match` —— ``ΔW = B·A``
   与 ``ΔW=B*A`` 规范化后必须相等（乘号与空格都是"表达差异"，不是"事实差异"），
   这是事实点命中能容忍排版的**唯一依据**；
2. :meth:`TestChrf.test_beta_one_equals_harmonic_mean_of_macro_pr` —— ``β = 1``
   时 chrF 必须退化成宏平均精确率与召回率的调和平均，且宏平均由测试**自己
   重算一遍**（``_macro_precision_recall``）。这条不变式能同时钉住分子、
   分母与"排序取用 sacrebleu 形式"（``chrR`` 与 ``chrP`` 的位置）；
3. :meth:`TestComputeComponents.test_format_decays_by_violation_count` ——
   ``format`` 分量按 ``1/(1+违规数)`` 衰减（1 条违规 0.5、2 条违规 1/3），
   不是 0/1 二值。它决定了"少写一个必须片段"不会与"长度超一倍"得同一个分。

``chrf`` 的阶数口径单独记录一条（它决定"短文本能不能拿满分"）：

- **某一阶 n-gram 在两侧都不存在时跳过该阶**。跳过之后，``n`` 超过文本
  长度不再把分数压低：4 个字的相同文本在默认 ``n = 6`` 下也是 1.0
  （见 :meth:`TestChrf.test_identical_short_text_scores_one`）。本文件自用的
  ``_macro_precision_recall`` 按同一口径重算宏平均，否则它算出来的期望值
  会比实现少一截。注意"只有一侧没有这一阶"仍记 ``0.0``——那是真的量不到
  重叠，不是"没有东西可量"。

另外一条刻意记录的**实现口径**：

- ``normalize_text`` 只做折叠与剔除、**不做排序**，所以 ``ABA`` 与 ``AAB``
  规范化后仍是两串不同的文本（``aba`` / ``aab``），能区分的（见
  :meth:`TestNormalizeText.test_does_not_reorder`）。
"""

from __future__ import annotations

from collections import Counter

import pytest

from smart_research_agent.finetune_eval.metrics import (
    COMPONENT_NAMES,
    DEFAULT_WEIGHTS,
    REFUSAL_MARKERS,
    WEIGHT_TOLERANCE,
    FactRecall,
    FinetuneEvalError,
    FormatRules,
    MetricBreakdown,
    char_ngrams,
    chrf,
    compute_components,
    fact_recall,
    forbidden_hits,
    forbidden_ratio,
    format_violations,
    is_refusal,
    lcs_length,
    mean,
    normalize_text,
    rouge_l,
    token_f1,
    tokenize,
    validate_weights,
    weighted_total,
)

# ---------------------------------------------------------------------------
# 测试自用的独立重算与样例构造
# ---------------------------------------------------------------------------


def _macro_precision_recall(prediction: str, reference: str, n: int) -> tuple[float, float]:
    """**独立**重算 chrF 的宏平均精确率 / 召回率（不调用被测的 ``char_ngrams``）.

    与实现的口径一致的两点：按 ``normalize_text`` + 字符级切分；**某一阶在
    两侧都取不到任何 gram 时跳过该阶**、只有一侧取不到时记 ``0.0``。
    跳过那一支是"短文本也能拿满分"的依据：文本比 ``n`` 短时，高阶上
    没有任何东西可量，把它算成"零重叠"会凭空压低分数。
    """
    predicted = normalize_text(prediction)
    expected = normalize_text(reference)
    precisions: list[float] = []
    recalls: list[float] = []
    for order in range(1, n + 1):
        predicted_grams = Counter(
            tuple(predicted[index : index + order]) for index in range(len(predicted) - order + 1)
        )
        expected_grams = Counter(
            tuple(expected[index : index + order]) for index in range(len(expected) - order + 1)
        )
        if not predicted_grams and not expected_grams:
            continue
        overlap = sum((predicted_grams & expected_grams).values())
        precisions.append(overlap / sum(predicted_grams.values()) if predicted_grams else 0.0)
        recalls.append(overlap / sum(expected_grams.values()) if expected_grams else 0.0)
    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)


def mixed_components() -> dict[str, float]:
    """一份分量只有 0 / 1、便于手算加权和的样例（手算 = 0.40 + 0.15 + 0.05）."""
    return {
        "fact_recall": 1.0,
        "token_f1": 0.0,
        "rouge_l": 1.0,
        "chrf": 0.0,
        "format": 1.0,
        "refusal": 0.0,
    }


def half_components() -> dict[str, float]:
    """六个分量全为 0.5 的样例（任何合法权重下的加权和都是 0.5）."""
    return {name: 0.5 for name in COMPONENT_NAMES}


# ---------------------------------------------------------------------------
# 规范化与分词
# ---------------------------------------------------------------------------


class TestNormalizeText:
    """``NFKC → casefold → 剔除 P/Z/C/S`` 三步，各自的职责必须可分辨."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("１２３", "123"),
            ("１", "1"),
            ("ＡＢ", "ab"),
            ("（中文）", "中文"),
            ("Ⅻ", "xii"),
        ],
    )
    def test_nfkc_folds_compatibility_characters(self, raw, expected):
        """NFKC 把全角数字 / 全角字母 / 兼容字符折成 ASCII 形态."""
        assert normalize_text(raw) == expected

    def test_casefold_lowercases_latin(self):
        """大小写折叠：``B`` 与 ``b`` 对事实点命中是同一个字符."""
        assert normalize_text("ABC") == "abc"

    def test_casefold_folds_greek_delta(self):
        """``Δ`` 折叠为 ``δ``——这一条直接决定 ``ΔW`` 这类事实点能不能命中."""
        assert normalize_text("ΔW") == "δw"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("a b", "ab"),
            ("a\tb", "ab"),
            ("a\nb", "ab"),
            ("a\u00a0b", "ab"),
            ("a,b", "ab"),
            ("a：b", "ab"),
            ("a。b", "ab"),
            ("a-b", "ab"),
            ("a+b", "ab"),
            ("a=b", "ab"),
            ("a·b", "ab"),
            ("a*b", "ab"),
        ],
    )
    def test_strips_punctuation_space_and_symbols(self, raw, expected):
        """标点（P）、分隔符（Z）、控制符（C）、符号（S）全部剔除."""
        assert normalize_text(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("中文测试", "中文测试"),
            ("ひらがな", "ひらがな"),
            ("カタカナ", "カタカナ"),
            ("한글", "한글"),
        ],
    )
    def test_keeps_cjk_characters(self, raw, expected):
        """中日韩字符属于字母类（Lo），必须原样保留——否则中文指标全是 0."""
        assert normalize_text(raw) == expected

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("v2.0", "v20"),
            ("75.31", "7531"),
            ("2026年", "2026年"),
        ],
    )
    def test_keeps_digits(self, raw, expected):
        """数字只在剔除标点后保留（``75.31`` 里的小数点会掉，但数字本身不丢）."""
        assert normalize_text(raw) == expected

    def test_delta_w_notation_forms_match(self):
        """``ΔW = B·A`` 与 ``ΔW=B*A`` 规范化后必须相等（空格 / 乘号都是排版差异）."""
        assert normalize_text("ΔW = B·A") == normalize_text("ΔW=B*A")

    def test_delta_w_notation_exact_value(self):
        """两个写法规范化后都等于 ``δwba``——空格、``=``、``·``、``*`` 全部被剔除."""
        assert normalize_text("ΔW = B·A") == "δwba"
        assert normalize_text("ΔW=B*A") == "δwba"

    @pytest.mark.parametrize(("raw", "expected"), [("✓", ""), ("€", ""), ("★", "")])
    def test_strips_symbols(self, raw, expected):
        """符号类（S）整类剔除：勾选标记、货币符号都不参与事实点匹配."""
        assert normalize_text(raw) == expected

    def test_combining_mark_is_composed_not_dropped(self):
        """组合记号（M）不被剔除：``e`` + U+0301 经 NFKC 组合为 ``é`` 而非丢掉重音."""
        assert normalize_text("e\u0301") == "é"

    def test_whitespace_only_is_empty(self):
        """纯空白规范化后是空串（于是"两边都空"的边界可以由此触发）."""
        assert normalize_text("   \n\t ") == ""

    def test_punctuation_only_is_empty(self):
        """纯标点同样规范化成空串."""
        assert normalize_text("!!!") == ""

    def test_does_not_reorder(self):
        """**实现口径**：规范化只做折叠与剔除、不做排序，所以 ``ABA`` ≠ ``AAB``.

        模块文档里"``ABA`` 与 ``AAB`` 也无法区分"的说法与实现对不上（两者
        分别是 ``aba`` 与 ``aab``）。真正"无法区分顺序"的是事实点命中所用的
        **子串包含**判定，而不是规范化本身。
        """
        assert normalize_text("ABA") == "aba"
        assert normalize_text("AAB") == "aab"
        assert normalize_text("ABA") != normalize_text("AAB")

    @pytest.mark.parametrize("raw", ["ΔW = B·A", "ABC", "中文，测试！", "１２３", ""])
    def test_idempotent(self, raw):
        """规范化是幂等的：再折一次不该再变（否则"规范化后的子串命中"不可复现）."""
        assert normalize_text(normalize_text(raw)) == normalize_text(raw)


class TestTokenize:
    """字符级分词：规范化后的每个字符算一个 token."""

    def test_empty_string_gives_empty_list(self):
        """空串得空列表（``[]``，不是 ``[""]``）."""
        assert tokenize("") == []

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("中文 a", ["中", "文", "a"]),
            ("１２３", ["1", "2", "3"]),
            ("ABC", ["a", "b", "c"]),
            ("中，文", ["中", "文"]),
        ],
    )
    def test_character_level(self, raw, expected):
        """一个字符一个 token，标点与空白不产生 token."""
        assert tokenize(raw) == expected

    def test_punctuation_only_gives_empty_list(self):
        """纯标点在规范化阶段就被清空，因此 token 列表也是空的."""
        assert tokenize("!!!") == []

    @pytest.mark.parametrize("raw", ["中文答案", "ΔW = B·A", "", "a b c"])
    def test_matches_normalize_text(self, raw):
        """``tokenize`` 就是 ``list(normalize_text(...))``——两条路径不能有分歧."""
        assert tokenize(raw) == list(normalize_text(raw))


# ---------------------------------------------------------------------------
# 词面指标
# ---------------------------------------------------------------------------


class TestTokenF1:
    """字符多重集 F1：三种边界必须显式定义，重叠按**多重集**计数."""

    def test_both_empty_is_one(self):
        """空答案与空答案是"完全一致"，得 1.0（而不是 0.0 或 nan）."""
        assert token_f1("", "") == 1.0

    def test_both_empty_after_normalization_is_one(self):
        """规范化后两边都空（纯标点）同样按"都空"处理."""
        assert token_f1("!!!", "　　") == 1.0

    @pytest.mark.parametrize(
        ("prediction", "reference"), [("中文", ""), ("", "中文"), ("中文", "!!!")]
    )
    def test_one_side_empty_is_zero(self, prediction, reference):
        """只有一边为空 → 0.0：一方什么都没说，谈不上相似."""
        assert token_f1(prediction, reference) == 0.0

    def test_no_overlap_is_zero(self):
        """重叠为 0 → 0.0（不做"都含标点所以相似"这类加分）."""
        assert token_f1("abcd", "wxyz") == 0.0

    def test_identical_is_one(self):
        """完全相同的文本得 1.0."""
        assert token_f1("中文答案", "中文答案") == 1.0

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("中文答案", "中文"),
            ("aab", "ab"),
            ("", "中文"),
            ("abc", "xyz"),
            ("结论: OK", "结论：ok"),
        ],
    )
    def test_symmetric(self, left, right):
        """F1 对调换两侧保持不变（P 与 R 对称地进入调和平均）."""
        assert token_f1(left, right) == pytest.approx(token_f1(right, left))

    @pytest.mark.parametrize(
        ("prediction", "reference", "expected"),
        [
            ("aab", "ab", 0.8),
            ("aaa", "a", 0.5),
            ("aa", "aa", 1.0),
            ("abc", "cba", 1.0),
        ],
    )
    def test_multiset_counts_duplicates(self, prediction, reference, expected):
        """重复字符按**个数**计：``aab`` vs ``ab`` 的 P=2/3、R=2/2 → F1=0.8."""
        assert token_f1(prediction, reference) == pytest.approx(expected)

    def test_order_insensitive(self):
        """多重集语义对顺序不敏感：``abc`` 与 ``cba`` 的 F1 是 1.0（而 LCS 只有 1）."""
        assert token_f1("abc", "cba") == 1.0
        assert lcs_length(list("abc"), list("cba")) == 1

    @pytest.mark.parametrize(
        ("prediction", "reference"),
        [("中文答案", "中文"), ("aab", "ab"), ("abcd", "wxyz"), ("", "")],
    )
    def test_stays_within_unit_interval(self, prediction, reference):
        """F 度量必须落在 [0, 1]（越界的分数会让"通过阈值"这个约定失效）."""
        assert 0.0 <= token_f1(prediction, reference) <= 1.0

    def test_tolerates_case_and_punctuation(self):
        """规范化带来的容忍：大小写与冒号全半角差异不影响 F1."""
        assert token_f1("结论: OK", "结论：ok") == 1.0


class TestLcsLength:
    """最长公共子序列：对**顺序**敏感，是词面 F1 之外的另一把尺子."""

    @pytest.mark.parametrize(("left", "right"), [([], []), ([], list("ab")), (list("ab"), [])])
    def test_empty_sequence_is_zero(self, left, right):
        """任一序列为空 → 0（没有公共子序列可言）."""
        assert lcs_length(left, right) == 0

    @pytest.mark.parametrize("sequence", ["", "a", "abc", "中文答案"])
    def test_identical_returns_length(self, sequence):
        """两个相同序列的 LCS 就是它自己的长度."""
        items = list(sequence)
        assert lcs_length(items, items) == len(items)

    def test_order_sensitive_abc_vs_cba(self):
        """``A B C`` 与 ``C B A`` 的顺序完全不同：LCS 只有 1（而多重集 F1 是 1.0）."""
        assert lcs_length(list("ABC"), list("CBA")) == 1

    def test_prefix_is_shorter(self):
        """前缀关系的 LCS 等于短的那条的长度."""
        assert lcs_length(list("ab"), list("abcd")) == 2

    def test_subsequence(self):
        """子序列（不连续）也算：``ace`` 是 ``abcde`` 的子序列，LCS = 3."""
        assert lcs_length(list("ace"), list("abcde")) == 3

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            ("abcdef", "acf", 3),
            ("abc", "abc", 3),
            ("abc", "xyz", 0),
            ("ab", "ba", 1),
            ("aaa", "aa", 2),
            ("中文测试", "中文", 2),
        ],
    )
    def test_table(self, left, right, expected):
        """手算表：覆盖相等、无重叠、逆序、重复字符与中文."""
        assert lcs_length(list(left), list(right)) == expected

    def test_accepts_arbitrary_sequence_types(self):
        """签名收的是 ``Sequence[str]``，元组与列表都要能用."""
        assert lcs_length(("a", "b", "c"), ["a", "c"]) == 2

    def test_single_element(self):
        """单元素序列的边界（滚动数组的下标从 1 开始，容易差一位）."""
        assert lcs_length(["a"], ["a"]) == 1
        assert lcs_length(["a"], ["b"]) == 0


class TestRougeL:
    """ROUGE-L 的 F 度量：``β`` 是有争议的常数，必须显式."""

    @pytest.mark.parametrize("beta", [0.0, -1.0, -0.5])
    def test_non_positive_beta_rejected(self, beta):
        """``β <= 0`` 会让分母失去意义，直接报错而不是给出 nan."""
        with pytest.raises(FinetuneEvalError, match="beta 必须为正数"):
            rouge_l("中文", "中文", beta=beta)

    def test_both_empty_is_one(self):
        """两边都空 → 1.0."""
        assert rouge_l("", "") == 1.0

    @pytest.mark.parametrize(("prediction", "reference"), [("中文", ""), ("", "中文")])
    def test_one_side_empty_is_zero(self, prediction, reference):
        """只有一边空 → 0.0（注意 ``""`` vs ``"!!!"`` 是"两边都空"，见下一条）."""
        assert rouge_l(prediction, reference) == 0.0

    def test_punctuation_only_counts_as_empty(self):
        """规范化之后两边都空 → 1.0（"都是空答案"仍然算完全一致）."""
        assert rouge_l("", "!!!") == 1.0

    def test_zero_lcs_is_zero(self):
        """LCS 为 0（没有任何公共子序列）→ 0.0."""
        assert rouge_l("abc", "xyz") == 0.0

    def test_identical_is_one(self):
        """相同文本 → 1.0."""
        assert rouge_l("中文答案", "中文答案") == 1.0

    def test_hand_computed_f1(self):
        """手算：prediction 2 字、reference 4 字、LCS=2 → P=1.0、R=0.5 → F1=2/3."""
        assert rouge_l("ab", "abcd", beta=1.0) == pytest.approx(2 * 1.0 * 0.5 / (0.5 + 1.0))

    def test_beta_one_and_half_differ_when_p_not_r(self):
        """**β 是有效参数**：``P ≠ R`` 时 ``β = 1``（0.6667）与 ``β = 0.5``（0.8333）不同值."""
        beta_one = rouge_l("ab", "abcd", beta=1.0)
        beta_half = rouge_l("ab", "abcd", beta=0.5)
        assert beta_one == pytest.approx(2 / 3)
        assert beta_half == pytest.approx(5 / 6)
        assert beta_one != pytest.approx(beta_half)

    def test_beta_half_hand_computed(self):
        """``β = 0.5`` 时按 ``(1+0.25)PR / (R + 0.25P)`` 手算一遍（分母里换的是 P）."""
        precision, recall = 1.0, 0.5
        expected = (1 + 0.25) * precision * recall / (recall + 0.25 * precision)
        assert rouge_l("ab", "abcd", beta=0.5) == pytest.approx(expected)

    def test_beta_one_is_symmetric(self):
        """``β = 1`` 时 F1 对称：``f(a, b) == f(b, a)``."""
        assert rouge_l("ab", "abcd", beta=1.0) == pytest.approx(rouge_l("abcd", "ab", beta=1.0))

    def test_beta_larger_than_one_approaches_recall(self):
        """``β > 1`` 加重召回：分数向召回值移动（这里召回 0.5 低于 F1 0.6667，故变小）."""
        assert rouge_l("ab", "abcd", beta=2.0) == pytest.approx(5 / 9)
        assert rouge_l("ab", "abcd", beta=2.0) < rouge_l("ab", "abcd", beta=1.0)
        assert rouge_l("ab", "abcd", beta=1e6) == pytest.approx(0.5)

    def test_beta_smaller_than_one_approaches_precision(self):
        """``β → 0`` 时分数向精确率移动（这里是 P = 1.0）."""
        assert rouge_l("ab", "abcd", beta=1e-6) == pytest.approx(1.0)

    def test_equal_precision_and_recall_is_their_common_value(self):
        """P == R 时任何 ``β`` 都退化成同一个值（这里是 1/3）."""
        assert rouge_l("abc", "cba", beta=1.0) == pytest.approx(1 / 3)
        assert rouge_l("abc", "cba", beta=2.0) == pytest.approx(1 / 3)

    @pytest.mark.parametrize(
        ("prediction", "reference", "beta", "expected"),
        [
            ("ab", "abcd", 1.0, 2 / 3),
            ("a", "a", 1.0, 1.0),
            ("abc", "cba", 1.0, 1 / 3),
            ("ab", "abcd", 0.5, 5 / 6),
            ("ab", "abcd", 2.0, 5 / 9),
        ],
    )
    def test_table(self, prediction, reference, beta, expected):
        """手算表：把 ``β`` 的影响钉成一组确定数字."""
        assert rouge_l(prediction, reference, beta=beta) == pytest.approx(expected)


class TestCharNgrams:
    """字符 n-gram 多重集：阶越界的两种处理（``<=0`` 报错、``>len`` 返回空）."""

    @pytest.mark.parametrize("order", [0, -1, -7])
    def test_non_positive_order_rejected(self, order):
        """``order <= 0`` 没有定义，直接报错（避免"空 Counter = 全不匹配"的假信号）."""
        with pytest.raises(FinetuneEvalError, match="n-gram 的阶必须为正整数"):
            char_ngrams(list("abc"), order)

    @pytest.mark.parametrize(
        ("characters", "order"),
        [("ab", 3), ("a", 2), ("", 1), ("中文", 5)],
    )
    def test_order_greater_than_length_is_empty(self, characters, order):
        """``order > len`` 时返回**空** Counter（而不是报错）."""
        assert char_ngrams(list(characters), order) == Counter()

    @pytest.mark.parametrize(
        ("characters", "order", "expected"),
        [
            ("aab", 1, Counter({("a",): 2, ("b",): 1})),
            ("aaa", 2, Counter({("a", "a"): 2})),
            ("abc", 2, Counter({("a", "b"): 1, ("b", "c"): 1})),
            ("ab", 2, Counter({("a", "b"): 1})),
        ],
    )
    def test_multiset_counts(self, characters, order, expected):
        """多重集语义：同一个 gram 出现两次就计两个."""
        assert char_ngrams(list(characters), order) == expected

    def test_order_equals_length_is_single_gram(self):
        """``order == len`` 时只有一个 gram（窗口数 = len - order + 1）."""
        assert char_ngrams(list("abc"), 3) == Counter({("a", "b", "c"): 1})

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_total_count_is_window_count(self, order):
        """总计数必须等于窗口数 ``len - order + 1``（差一位是最常见的实现错误）."""
        characters = list("abcdef")
        assert sum(char_ngrams(characters, order).values()) == len(characters) - order + 1

    def test_empty_sequence_gives_empty_counter(self):
        """空序列取任何合法阶都得到空 Counter."""
        assert char_ngrams([], 1) == Counter()


class TestChrf:
    """chrF：字符 1..n-gram 的宏平均，再按 ``β`` 合成 F 度量."""

    @pytest.mark.parametrize("n", [0, -1])
    def test_invalid_n_rejected(self, n):
        """``n < 1`` 报错（没有任何阶可平均）."""
        with pytest.raises(FinetuneEvalError, match="chrf 的 n 至少为 1"):
            chrf("中文", "中文", n=n)

    @pytest.mark.parametrize("beta", [0.0, -2.0])
    def test_non_positive_beta_rejected(self, beta):
        """``β <= 0`` 报错，与 ``rouge_l`` 同一约定."""
        with pytest.raises(FinetuneEvalError, match="beta 必须为正数"):
            chrf("中文", "中文", beta=beta)

    def test_both_empty_is_one(self):
        """两边都空 → 1.0."""
        assert chrf("", "", n=1) == 1.0

    @pytest.mark.parametrize(
        ("prediction", "reference"),
        [("中文", ""), ("", "中文")],
    )
    def test_one_side_empty_is_zero(self, prediction, reference):
        """只有一边空 → 0.0（``""`` vs ``"!!!"`` 是"两边都空"，见下一条）."""
        assert chrf(prediction, reference, n=1) == 0.0

    def test_punctuation_only_counts_as_empty(self):
        """规范化后两边都空 → 1.0（chrF 与 ``token_f1`` / ``rouge_l`` 同一约定）."""
        assert chrf("!!!", "", n=1) == 1.0

    @pytest.mark.parametrize("n", [1, 2, 3, 6])
    def test_no_overlap_is_zero(self, n):
        """宏平均任一项为 0（重叠为 0）→ 0.0，不返回 nan."""
        assert chrf("abcd", "wxyz", n=n) == 0.0

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_identical_text_is_one_when_n_covers_length(self, n):
        """``n`` 不超过文本长度时，相同文本 → 1.0（每个阶的 P 与 R 都是 1）."""
        assert chrf("中文测试", "中文测试", n=n) == 1.0

    def test_identical_long_text_is_one_with_default_n(self):
        """文本长于 ``n``（这里是 8 字 ≥ 默认 6）时，相同文本 → 1.0."""
        assert chrf("中文测试用例文本", "中文测试用例文本") == 1.0

    def test_identical_short_text_scores_one(self):
        """相同文本在**任何** ``n`` 下都是 1.0——包括文本比 ``n`` 短的情形.

        ``中文测试`` 只有 4 个字：默认 ``n = 6`` 时 5 / 6 两阶在两侧都取不到
        gram，按口径**跳过**；剩下的 1~4 阶都是完美重叠，于是宏平均
        P = R = 1.0，任何 ``β`` 下都是 1.0。``n = 4`` 恰好齐平、``n = 5``
        起开始有阶被跳过，三个取值一起把"跳过"这件事钉住。
        """
        assert chrf("中文测试", "中文测试") == pytest.approx(1.0)
        assert chrf("中文测试", "中文测试", n=4) == pytest.approx(1.0)
        assert chrf("中文测试", "中文测试", n=5) == pytest.approx(1.0)
        assert chrf("中文测试", "中文测试", n=6) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("prediction", "reference", "n"),
        [
            ("ab", "abc", 1),
            ("ab", "abc", 2),
            ("abcd", "abc", 3),
            ("低秩矩阵", "低秩矩阵与乘积", 4),
            ("中文测试", "中文测试", 6),
        ],
    )
    def test_beta_one_equals_harmonic_mean_of_macro_pr(self, prediction, reference, n):
        """**β = 1 的不变式**：结果必须等于宏平均 P 与 R 的调和平均（自己重算一遍）.

        宏平均由本文件独立重算（``_macro_precision_recall``），口径与被测实现
        一致：两侧都没有这一阶时跳过。最后那一组 ``("中文测试", "中文测试", 6)``
        正是这条口径的读数——4 个字的文本有 2 阶被跳过，期望值是 **1.0**
        （而不是把跳过的阶算成零重叠得到的 4/6）。
        """
        precision, recall = _macro_precision_recall(prediction, reference, n)
        harmonic = 2 * precision * recall / (precision + recall)
        assert chrf(prediction, reference, n=n, beta=1.0) == pytest.approx(harmonic)
        if prediction == reference:
            # 相同文本的期望值恒为 1.0：n 超过文本长度只让高阶被跳过，
            # 不会让分数下降（见 test_identical_short_text_scores_one）。
            assert harmonic == pytest.approx(1.0)

    def test_beta_two_is_recall_weighted_hand_value(self):
        """``β = 2``（缺省）按 sacrebleu 形式手算：``P = 1.0``、``R = 2/3`` → ``10/14``."""
        macro_precision, macro_recall = _macro_precision_recall("ab", "abc", 1)
        assert (macro_precision, macro_recall) == (1.0, pytest.approx(2 / 3))
        expected = (1 + 4) * macro_precision * macro_recall / (4 * macro_precision + macro_recall)
        assert chrf("ab", "abc", n=1, beta=2.0) == pytest.approx(expected)
        assert chrf("ab", "abc", n=1, beta=2.0) == pytest.approx(10 / 14)

    def test_beta_larger_than_one_approaches_recall(self):
        """``β > 1`` 加重召回：分数向召回值（2/3）移动，故低于 ``β = 1`` 的值（0.8）."""
        assert chrf("ab", "abc", n=1, beta=2.0) < chrf("ab", "abc", n=1, beta=1.0)
        assert chrf("ab", "abc", n=1, beta=1e6) == pytest.approx(2 / 3)

    def test_beta_smaller_than_one_approaches_precision(self):
        """``β → 0`` 时分数向精确率（1.0）移动."""
        assert chrf("ab", "abc", n=1, beta=1e-6) == pytest.approx(1.0)

    def test_n_changes_the_value(self):
        """``n`` 是有效参数：阶数变了，宏平均的组成就变了，数值必须跟着变."""
        assert chrf("低秩矩阵", "低秩矩阵与乘积", n=1) != pytest.approx(
            chrf("低秩矩阵", "低秩矩阵与乘积", n=6)
        )

    @pytest.mark.parametrize(
        ("prediction", "reference"), [("ab", "abc"), ("abcd", "wxyz"), ("中文测试", "中文测试")]
    )
    def test_stays_within_unit_interval(self, prediction, reference):
        """F 度量必须落在 [0, 1]."""
        assert 0.0 <= chrf(prediction, reference) <= 1.0


# ---------------------------------------------------------------------------
# 事实点与禁项
# ---------------------------------------------------------------------------


class TestFactRecall:
    """事实点召回：规范化后子串命中，并把**分母交出来**."""

    def test_empty_checklist_scores_one(self):
        """没有要求等于要求都满足：``score = 1.0`` 且 ``total = 0``."""
        result = fact_recall("任意输出", ())
        assert (result.score, result.total, result.hit, result.missing) == (1.0, 0, 0, ())

    def test_empty_fact_rejected(self):
        """``""`` 是任何文本的子串，放进清单会让用例永远满分——必须报错."""
        with pytest.raises(FinetuneEvalError, match="事实点不能为空字符串"):
            fact_recall("任意输出", ("低秩", ""))

    def test_whitespace_fact_rejected(self):
        """纯空白事实点同样被拒绝（否则它也是恒真断言）."""
        with pytest.raises(FinetuneEvalError, match="事实点不能为空字符串"):
            fact_recall("任意输出", ("   ",))

    def test_all_hit(self):
        """全部命中时 ``missing`` 为空元组、``score`` 为 1.0."""
        result = fact_recall("低秩矩阵的乘积", ("低秩", "矩阵"))
        assert result == FactRecall(hit=2, total=2, score=1.0, missing=())

    def test_missing_reported_in_input_order(self):
        """缺失清单只含没命中的项，且顺序与输入一致."""
        result = fact_recall("只说了低秩", ("低秩", "ΔW", "梯度"))
        assert result.missing == ("ΔW", "梯度")
        assert result.hit == 1

    def test_score_is_ratio(self):
        """``score = 命中数 / 总数``（这里是 1/3）."""
        assert fact_recall("低秩", ("低秩", "ΔW", "梯度")).score == pytest.approx(1 / 3)

    def test_normalization_tolerance_on_punctuation(self):
        """事实点 ``ΔW = B·A`` 在输出写成 ``ΔW=B*A`` 时仍然命中."""
        result = fact_recall("增量是 ΔW=B*A 的乘积", ("ΔW = B·A",))
        assert result.hit == 1

    def test_normalization_tolerance_on_case(self):
        """大小写差异不影响命中（``LoRA`` vs ``lora``）."""
        assert fact_recall("lora 冻结基座", ("LoRA",)).hit == 1

    def test_normalization_tolerance_on_fullwidth_colon(self):
        """全角 / 半角冒号与空格都不影响事实点命中（格式规则才走原文匹配）."""
        assert fact_recall("来源: 源码", ("来源：源码",)).hit == 1

    def test_miss_on_different_fact(self):
        """事实点必须在输出里**真的出现**，近义改写不算命中（子串判定的已知代价）."""
        assert fact_recall("矩阵是低阶的", ("低秩",)).hit == 0

    def test_empty_prediction_misses_all(self):
        """输出为空 → 全部缺失（而不是"没有可比内容"这种模糊状态）."""
        result = fact_recall("", ("低秩", "矩阵"))
        assert (result.hit, result.missing) == (0, ("低秩", "矩阵"))

    def test_summary_line_when_all_hit(self):
        """``summary_line`` 的全部命中分支."""
        assert fact_recall("低秩矩阵", ("低秩", "矩阵")).summary_line() == "事实点 2/2（全部命中）"

    def test_summary_line_when_missing(self):
        """``summary_line`` 的缺失分支（单条）."""
        assert fact_recall("低秩", ("低秩", "ΔW")).summary_line() == "事实点 1/2（缺失 ΔW）"

    def test_summary_line_when_multiple_missing(self):
        """``summary_line`` 的缺失分支（多条用顿号连接）."""
        line = fact_recall("低秩", ("低秩", "ΔW", "梯度")).summary_line()
        assert line == "事实点 1/3（缺失 ΔW、梯度）"

    def test_result_is_immutable(self):
        """``FactRecall`` 是 frozen 数据类：结果一旦给出就不该被就地改写."""
        result = fact_recall("低秩", ("低秩",))
        with pytest.raises(Exception):
            result.hit = 99  # type: ignore[misc]


class TestForbiddenHitsAndRatio:
    """禁项：与事实点用同一套规范化，但返回**原文片段**."""

    def test_empty_checklist_returns_empty_tuple(self):
        """没有禁项 → 空元组（而不是 None）."""
        assert forbidden_hits("任意输出", ()) == ()

    def test_no_hit_returns_empty_tuple(self):
        """一个都没踩到 → 空元组."""
        assert forbidden_hits("正常输出", ("编造", "放得下")) == ()

    def test_hit_returns_original_fragment(self):
        """返回的是**原文片段**（报告是给人看的，不是给人看规范化结果的）."""
        assert forbidden_hits("增量是 ΔW=B*A", ("ΔW = B·A",)) == ("ΔW = B·A",)

    def test_normalization_tolerance(self):
        """大小写与标点差异同样被容忍（``能省 4 倍`` vs ``能省4倍``）."""
        assert forbidden_hits("zero2 能省4倍显存", ("能省 4 倍",)) == ("能省 4 倍",)

    def test_multiple_hits_keep_input_order(self):
        """多条命中按输入顺序返回."""
        hits = forbidden_hits("既是编造又放得下", ("放得下", "编造"))
        assert hits == ("放得下", "编造")

    def test_hits_are_tuple(self):
        """返回类型是元组（可哈希、可比较，便于断言）."""
        assert isinstance(forbidden_hits("编造", ("编造",)), tuple)

    def test_ratio_without_checklist_is_zero(self):
        """没有禁项 → "没有需要避免的东西"，瑕疵为 0.0（与事实点的"空即 1.0"相反）."""
        assert forbidden_ratio("任意输出", ()) == 0.0

    @pytest.mark.parametrize(
        ("prediction", "checklist", "expected"),
        [
            ("正常输出", ("编造", "放得下"), 0.0),
            ("编造了数字", ("编造", "放得下"), 0.5),
            ("既编造又放得下", ("编造", "放得下"), 1.0),
            ("编造", ("编造", "放得下", "能省 4 倍"), 1 / 3),
        ],
    )
    def test_ratio_is_hits_over_total(self, prediction, checklist, expected):
        """比例 = 命中数 / 禁项总数."""
        assert forbidden_ratio(prediction, checklist) == pytest.approx(expected)

    def test_ratio_matches_hits_length(self):
        """比例必须与 ``forbidden_hits`` 的命中条数一致（不能各算各的）."""
        prediction = "编造了 75.31 这个数字"
        checklist = ("编造", "75.31", "放得下")
        assert forbidden_ratio(prediction, checklist) == pytest.approx(
            len(forbidden_hits(prediction, checklist)) / len(checklist)
        )


# ---------------------------------------------------------------------------
# 格式契约
# ---------------------------------------------------------------------------


class TestFormatRules:
    """格式规则的自洽性校验与摘要文案."""

    @pytest.mark.parametrize("max_chars", [0, -1, -100])
    def test_non_positive_max_chars_rejected(self, max_chars):
        """``max_chars <= 0`` 报错（0 字上限不是"无限制"）."""
        with pytest.raises(FinetuneEvalError, match="max_chars 必须为正整数"):
            FormatRules(max_chars=max_chars).validate()

    def test_only_must_not_contain_rejected(self):
        """只有"禁止"、既没有"必须"也没有长度上限的规则几乎必然通过，必须拒绝."""
        rules = FormatRules(must_not_contain=("不好",))
        with pytest.raises(FinetuneEvalError, match="至少要有 must_contain 或 max_chars 之一"):
            rules.validate()

    def test_must_not_contain_with_max_chars_is_valid(self):
        """补上长度上限之后，同一条"只有禁止"的规则就成了自洽规则."""
        assert FormatRules(must_not_contain=("不好",), max_chars=100).validate() is None

    def test_must_contain_only_is_valid(self):
        """只有"必须"是自洽的（它本身就可以被判假）."""
        assert FormatRules(must_contain=("结论：",)).validate() is None

    def test_must_contain_with_max_chars_is_valid(self):
        """必须片段 + 长度上限（最常见的组合）合法."""
        assert FormatRules(must_contain=("结论：",), max_chars=120).validate() is None

    def test_empty_rules_are_valid(self):
        """默认空规则合法：它表示"这一条用例没有格式约束"."""
        assert FormatRules().validate() is None

    def test_defaults_are_empty(self):
        """三个字段的缺省值都是"无约束"."""
        rules = FormatRules()
        assert (rules.must_contain, rules.must_not_contain, rules.max_chars) == ((), (), None)

    @pytest.mark.parametrize(
        ("rules", "expected"),
        [
            (FormatRules(), "无格式约束"),
            (FormatRules(must_contain=("结论：",)), "必须含 结论："),
            (FormatRules(max_chars=100), "不超 100 字"),
            (FormatRules(must_not_contain=("不好",), max_chars=10), "禁止含 不好；不超 10 字"),
            (
                FormatRules(
                    must_contain=("结论：", "依据："), must_not_contain=("风险",), max_chars=100
                ),
                "必须含 结论：、依据：；禁止含 风险；不超 100 字",
            ),
        ],
    )
    def test_summary_line_combinations(self, rules, expected):
        """摘要文案覆盖三种单项组合、全组合与空规则."""
        assert rules.summary_line() == expected


class TestFormatViolations:
    """逐条检查格式契约：**原文**匹配，返回空元组才算合规."""

    def test_compliant_returns_empty_tuple(self):
        """合规 → 空元组（这是"没有违规"的唯一表示）."""
        rules = FormatRules(must_contain=("结论：",), must_not_contain=("风险",), max_chars=100)
        assert format_violations("结论：可以上线。", rules) == ()

    def test_missing_required_reports_one(self):
        """缺一个必须片段 → 恰好一条违规，文案可复核."""
        assert format_violations("没有冒号的结论", FormatRules(must_contain=("结论：",))) == (
            "缺少必需片段：'结论：'",
        )

    def test_forbidden_fragment_reports_one(self):
        """出现一个禁止片段 → 恰好一条违规."""
        assert format_violations(
            "这里有风险提示", FormatRules(must_contain=("风险",), must_not_contain=("风险",))
        ) == ("出现禁止片段：'风险'",)

    def test_too_long_reports_one(self):
        """超长 → 一条违规，且把实际长度与上限都写出来（可复核）."""
        assert format_violations("x" * 11, FormatRules(max_chars=10)) == ("长度 11 超出上限 10",)

    def test_three_violations_in_order(self):
        """三类违规按"必须 → 禁止 → 长度"的顺序各报一条."""
        rules = FormatRules(must_contain=("结论：",), must_not_contain=("风险",), max_chars=5)
        assert format_violations("风险很大啊哈", rules) == (
            "缺少必需片段：'结论：'",
            "出现禁止片段：'风险'",
            "长度 6 超出上限 5",
        )

    def test_length_exactly_at_limit_is_compliant(self):
        """边界：``len(text) == max_chars`` 不算超长."""
        assert format_violations("x" * 10, FormatRules(max_chars=10)) == ()

    def test_fullwidth_colon_must_appear_verbatim(self):
        """**原文匹配**：``结论：`` 里的全角冒号必须原样出现，半角冒号不算命中."""
        rules = FormatRules(must_contain=("结论：",))
        assert format_violations("结论: 半角冒号", rules) == ("缺少必需片段：'结论：'",)
        assert format_violations("结论：全角冒号", rules) == ()

    def test_forbidden_fragment_is_literal_not_normalized(self):
        """禁止片段同样**不**做规范化：``不 好`` 不会命中 ``不好``.

        若这里改用 ``normalize_text``，标点与空白会被抹掉，"禁止"就再也拦不住
        任何东西——规则形同虚设。
        """
        rules = FormatRules(must_contain=("不",), must_not_contain=("不 好",))
        assert format_violations("不好", rules) == ()
        assert format_violations("不 好", rules) == ("出现禁止片段：'不 好'",)

    def test_invalid_rules_are_rejected_before_checking(self):
        """``format_violations`` 内部先 ``validate()``：非法规则直接报错."""
        with pytest.raises(FinetuneEvalError, match="max_chars 必须为正整数"):
            format_violations("任意文本", FormatRules(max_chars=0))

    def test_empty_text_with_required_fragment(self):
        """空文本对"必须含"必然是违规，对"不超长"则合规."""
        assert format_violations("", FormatRules(must_contain=("结论：",))) == (
            "缺少必需片段：'结论：'",
        )
        assert format_violations("", FormatRules(max_chars=10)) == ()

    def test_empty_rules_on_any_text_is_compliant(self):
        """空规则对任何文本都合规（这正是"只为禁止、不给长度"的规则必须被拒的原因）."""
        assert format_violations("任意文本，随便写。", FormatRules()) == ()


class TestIsRefusal:
    """拒答检测：只回答"拒了吗"，不回答"拒得好不好"."""

    @pytest.mark.parametrize("marker", REFUSAL_MARKERS)
    def test_every_marker_is_detected(self, marker):
        """清单里的每个标记词都必须能被检出（清单本身要被测试守着）."""
        assert is_refusal(f"这件事我{marker}配合") is True

    @pytest.mark.parametrize(
        "text",
        ["我可以帮你做这个", "这是一段正常答案", "", "建议你先看文档"],
    )
    def test_non_marker_text_is_false(self, text):
        """没有标记词 → False（不返回"安全 / 不安全"这种二义结论）."""
        assert is_refusal(text) is False

    def test_marker_at_the_end(self):
        """标记词出现在句中任何位置都能检出."""
        assert is_refusal("抱歉，我无法协助。") is True

    def test_no_normalization_so_spaced_marker_misses(self):
        """**实现口径**：原文子串匹配，不做规范化——``无 法`` 不算拒答标记."""
        assert is_refusal("我无 法回答") is False
        assert is_refusal("我无法回答") is True


# ---------------------------------------------------------------------------
# 权重与加权和
# ---------------------------------------------------------------------------


class TestValidateWeights:
    """权重字典的两项校验：键集合一致 + 和等于 1.0."""

    def test_default_weights_are_valid(self):
        """缺省权重必须自洽（否则所有报告都算不出来）."""
        assert validate_weights(DEFAULT_WEIGHTS) is None

    def test_default_weights_sum_to_one(self):
        """缺省权重之和为 1.0（精度内）."""
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_component_names_follow_default_weights(self):
        """``COMPONENT_NAMES`` 就是缺省权重的键顺序（报告顺序依赖它）."""
        assert COMPONENT_NAMES == tuple(DEFAULT_WEIGHTS)

    def test_unknown_key_rejected(self):
        """多一个键 → 报错（否则它会被静默忽略，权重和也不再是 1.0）."""
        with pytest.raises(FinetuneEvalError, match="未知的分量"):
            validate_weights({**DEFAULT_WEIGHTS, "bleu": 0.0})

    def test_missing_key_rejected(self):
        """少一个键 → 报错（否则某个分量静默不参与打分）."""
        incomplete = {name: value for name, value in DEFAULT_WEIGHTS.items() if name != "chrf"}
        with pytest.raises(FinetuneEvalError, match="缺少分量权重"):
            validate_weights(incomplete)

    def test_negative_value_rejected(self):
        """负数权重 → 报错（总分可能越界）."""
        with pytest.raises(FinetuneEvalError, match="必须是非负有限数"):
            validate_weights({**DEFAULT_WEIGHTS, "fact_recall": -0.4})

    def test_infinite_value_rejected(self):
        """``inf`` → 报错（有限性检查先于求和检查）."""
        with pytest.raises(FinetuneEvalError, match="必须是非负有限数"):
            validate_weights({**DEFAULT_WEIGHTS, "fact_recall": float("inf")})

    def test_nan_value_rejected(self):
        """``nan`` → 报错（nan 会让一切比较恒为 ``False``）."""
        with pytest.raises(FinetuneEvalError, match="必须是非负有限数"):
            validate_weights({**DEFAULT_WEIGHTS, "fact_recall": float("nan")})

    def test_sum_not_one_rejected(self):
        """全 0 → 和不为 1.0，报错（总分恒为 0 的报告没有信息量）."""
        with pytest.raises(FinetuneEvalError, match="权重之和必须为 1.0"):
            validate_weights({name: 0.0 for name in COMPONENT_NAMES})

    def test_within_tolerance_passes(self):
        """和与 1.0 的差在 ``WEIGHT_TOLERANCE`` 以内 → 通过（浮点加法不满足结合律）."""
        weights = {**DEFAULT_WEIGHTS, "fact_recall": 0.40 + WEIGHT_TOLERANCE / 2}
        assert validate_weights(weights) is None

    def test_beyond_tolerance_rejected(self):
        """超出容差 → 报错（容差不是"随便差一点都行"）."""
        weights = {**DEFAULT_WEIGHTS, "fact_recall": 0.40 + WEIGHT_TOLERANCE * 2}
        with pytest.raises(FinetuneEvalError, match="权重之和必须为 1.0"):
            validate_weights(weights)


class TestWeightedTotal:
    """分量加权和：每个分量都必须落在 [0, 1]，越界即报错."""

    def test_uses_default_weights(self):
        """不传权重时用 ``DEFAULT_WEIGHTS``：全 0.5 的分量得到 0.5."""
        assert weighted_total(half_components()) == pytest.approx(0.5)

    def test_hand_computed_mixed_sum(self):
        """手算：``0.40·1 + 0.20·0 + 0.15·1 + 0.15·0 + 0.05·1 + 0.05·0 = 0.60``."""
        assert weighted_total(mixed_components()) == pytest.approx(0.40 + 0.15 + 0.05)

    def test_all_zero_and_all_one(self):
        """两个端点：全 0 → 0.0；全 1 → 1.0（总分必须仍落在 [0, 1]）."""
        assert weighted_total({name: 0.0 for name in COMPONENT_NAMES}) == 0.0
        assert weighted_total({name: 1.0 for name in COMPONENT_NAMES}) == pytest.approx(1.0)

    def test_boundary_values_accepted(self):
        """0.0 与 1.0 是合法分量值（闭区间）."""
        components = {name: 0.0 for name in COMPONENT_NAMES}
        components["chrf"] = 1.0
        assert weighted_total(components) == pytest.approx(DEFAULT_WEIGHTS["chrf"])

    def test_missing_component_rejected(self):
        """缺分量 → 报错（缺的那一项会被当成 0，总分被压低）."""
        with pytest.raises(FinetuneEvalError, match="缺少分量"):
            weighted_total({name: 0.5 for name in COMPONENT_NAMES if name != "chrf"})

    def test_unknown_component_rejected(self):
        """未知分量 → 报错（它没有任何权重可乘）."""
        with pytest.raises(FinetuneEvalError, match="未知的分量"):
            weighted_total({**half_components(), "bleu": 0.5})

    @pytest.mark.parametrize("value", [-0.01, -1.0])
    def test_below_zero_rejected(self, value):
        """分量小于 0 → 报错（真正的问题是指标本身，不该让它静默算出总分）."""
        with pytest.raises(FinetuneEvalError, match="必须落在"):
            weighted_total({**half_components(), "chrf": value})

    @pytest.mark.parametrize("value", [1.01, 2.0])
    def test_above_one_rejected(self, value):
        """分量大于 1 → 报错."""
        with pytest.raises(FinetuneEvalError, match="必须落在"):
            weighted_total({**half_components(), "chrf": value})

    def test_nan_component_rejected(self):
        """``nan`` 分量 → 报错（有限性检查与区间检查在同一条分支里）."""
        with pytest.raises(FinetuneEvalError, match="必须落在"):
            weighted_total({**half_components(), "chrf": float("nan")})

    def test_custom_weights_are_used(self):
        """自定义权重生效：把 1.0 全押在 ``chrf`` 上，总分就等于 ``chrf`` 的值."""
        custom = {name: (1.0 if name == "chrf" else 0.0) for name in COMPONENT_NAMES}
        components = {name: 0.0 for name in COMPONENT_NAMES}
        components["chrf"] = 0.25
        assert weighted_total(components, custom) == pytest.approx(0.25)

    def test_invalid_custom_weights_rejected(self):
        """自定义权重自身先被校验（和不为 1.0 直接报错）."""
        with pytest.raises(FinetuneEvalError, match="权重之和必须为 1.0"):
            weighted_total(half_components(), {name: 0.0 for name in COMPONENT_NAMES})


class TestMean:
    """安全求均值：空序列定义为 0.0（不返回 nan）."""

    def test_empty_list_is_zero(self):
        """空序列 → 0.0."""
        assert mean([]) == 0.0

    def test_empty_generator_is_zero(self):
        """生成器同样被接受（先把可迭代对象收成列表）."""
        assert mean(value for value in []) == 0.0

    def test_average(self):
        """普通均值."""
        assert mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_single_value(self):
        """单元素序列返回它自己."""
        assert mean([7.0]) == 7.0

    def test_accepts_tuple(self):
        """元组也能用（签名收的是 ``Iterable[float]``）."""
        assert mean((0.0, 1.0)) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 六个分量与加权和
# ---------------------------------------------------------------------------


class TestComputeComponents:
    """``compute_components``：只算分量，不做合格判定."""

    def test_component_keys_are_complete(self):
        """六个分量键齐全，且顺序就是 ``COMPONENT_NAMES``."""
        breakdown = compute_components("中文答案", "中文答案")
        assert tuple(breakdown.components) == COMPONENT_NAMES

    def test_no_violation_gives_format_one(self):
        """零违规 → ``format = 1/(1+0) = 1.0``."""
        rules = FormatRules(must_contain=("结论：",), max_chars=100)
        assert (
            compute_components("结论：答案", "结论：答案", rules=rules).components["format"] == 1.0
        )

    def test_one_violation_gives_format_half(self):
        """1 条违规 → ``1/2``（缺失片段扣一半，而不是归零）."""
        rules = FormatRules(must_contain=("结论：",))
        breakdown = compute_components("没有标记的文本", "参考", rules=rules)
        assert breakdown.components["format"] == pytest.approx(1 / 2)

    def test_two_violations_give_format_third(self):
        """2 条违规 → ``1/3``（衰减：违规越多分越低，但没有跳变）."""
        rules = FormatRules(must_contain=("结论：", "依据："))
        breakdown = compute_components("结论 内容", "参考", rules=rules)
        assert breakdown.components["format"] == pytest.approx(1 / 3)

    def test_format_decays_by_violation_count(self):
        """把三条曲线的值一起钉住：``1/(1+k)``（k = 0, 1, 2）."""
        compliant = compute_components(
            "结论：依据：", "参考", rules=FormatRules(must_contain=("结论：", "依据："))
        )
        one = compute_components(
            "依据：", "参考", rules=FormatRules(must_contain=("结论：", "依据："))
        )
        two = compute_components(
            "无关文本", "参考", rules=FormatRules(must_contain=("结论：", "依据："))
        )
        assert compliant.components["format"] == 1.0
        assert one.components["format"] == pytest.approx(1 / 2)
        assert two.components["format"] == pytest.approx(1 / 3)

    @pytest.mark.parametrize(
        ("text", "expect_refusal", "expected"),
        [
            ("这是正常答案", False, 1.0),
            ("我无法提供", False, 0.0),
            ("我无法提供", True, 1.0),
            ("这是正常答案", True, 0.0),
        ],
    )
    def test_refusal_is_binary(self, text, expect_refusal, expected):
        """``refusal`` 只有 1.0 / 0.0 两种取值（"拒了一半"不是一个状态）."""
        breakdown = compute_components(text, "参考", expect_refusal=expect_refusal)
        assert breakdown.components["refusal"] == expected

    def test_fact_recall_component_matches_function(self):
        """``fact_recall`` 分量必须与单独调用 ``fact_recall`` 的值一致（同一口径）."""
        breakdown = compute_components("只说了低秩", "参考", required_facts=("低秩", "ΔW"))
        assert breakdown.components["fact_recall"] == pytest.approx(
            fact_recall("只说了低秩", ("低秩", "ΔW")).score
        )

    def test_lexical_components_are_one_for_identical_text(self):
        """预测与参考相同时，三个词面分量都必须是 1.0（除 format 之外的满配）."""
        breakdown = compute_components("中文测试答案", "中文测试答案")
        assert breakdown.components["token_f1"] == 1.0
        assert breakdown.components["rouge_l"] == 1.0
        assert breakdown.components["chrf"] == 1.0

    def test_default_rules_when_none(self):
        """``rules=None`` 等价于空规则：没有任何格式约束，``format`` 为 1.0."""
        assert compute_components("任意文本", "参考").components["format"] == 1.0

    def test_default_required_facts_is_empty(self):
        """不传事实点清单 → "没有要求"→ ``fact_recall`` 为 1.0."""
        assert compute_components("任意文本", "参考").components["fact_recall"] == 1.0

    def test_total_is_weighted_sum_of_six_components(self):
        """``total`` 必须等于六项按 ``DEFAULT_WEIGHTS`` 的加权和（写在断言里，不复用函数）."""
        breakdown = compute_components(
            "结论：低秩矩阵 A 与 B 的乘积。",
            "结论：低秩矩阵 A 与 B 的乘积。",
            required_facts=("低秩",),
            rules=FormatRules(must_contain=("结论：",), max_chars=200),
        )
        components = breakdown.components
        expected = (
            components["fact_recall"] * 0.40
            + components["token_f1"] * 0.20
            + components["rouge_l"] * 0.15
            + components["chrf"] * 0.15
            + components["format"] * 0.05
            + components["refusal"] * 0.05
        )
        assert breakdown.total == pytest.approx(expected)

    def test_total_matches_weighted_total(self):
        """``total`` 与独立调用 ``weighted_total`` 的结果一致."""
        breakdown = compute_components("部分正确的答案", "参考答案文本")
        assert breakdown.total == pytest.approx(weighted_total(breakdown.components))

    def test_total_stays_within_unit_interval(self):
        """总分落在 [0, 1]（权重和为 1、分量都在 [0, 1] 的直接结论）."""
        breakdown = compute_components("完全不同的输出", "参考答案文本")
        assert 0.0 <= breakdown.total <= 1.0

    def test_to_dict_follows_component_order(self):
        """``to_dict`` 的分量顺序必须是 ``COMPONENT_NAMES``（顺序变了报告不可比对）."""
        breakdown = compute_components("中文答案", "中文答案")
        assert tuple(breakdown.to_dict()["components"]) == COMPONENT_NAMES

    def test_to_dict_values_match_components(self):
        """``to_dict`` 是投影：值与 ``components`` / ``total`` 逐一相同."""
        breakdown = compute_components("中文答案", "其他答案")
        payload = breakdown.to_dict()
        assert payload["components"] == dict(breakdown.components)
        assert payload["total"] == breakdown.total

    def test_to_dict_is_json_serializable(self):
        """投影结果必须能 ``json.dumps``（报告要落盘）."""
        import json

        payload = compute_components("中文答案", "中文答案").to_dict()
        assert json.loads(json.dumps(payload, ensure_ascii=False))["total"] == pytest.approx(
            compute_components("中文答案", "中文答案").total
        )

    def test_summary_line_lists_every_component(self):
        """``summary_line`` 可打印，且六个分量名一个不少."""
        line = compute_components("中文答案", "中文答案").summary_line()
        assert line.startswith("总分 ")
        for name in COMPONENT_NAMES:
            assert f"{name}=" in line

    def test_summary_line_hand_formatted(self):
        """手工构造的 ``MetricBreakdown`` 也要能打印（分量与总分一起留档）."""
        breakdown = MetricBreakdown(components=half_components(), total=0.5)
        line = breakdown.summary_line()
        assert line.startswith("总分 0.5000 | ")
        assert line.endswith("refusal=0.500")

    def test_identical_inputs_give_identical_outputs(self):
        """纯函数性质：同一对输入必须给出同一份分量与总分."""
        first = compute_components("结论：答案", "结论：答案", required_facts=("答案",))
        second = compute_components("结论：答案", "结论：答案", required_facts=("答案",))
        assert first == second
