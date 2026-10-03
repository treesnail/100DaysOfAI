"""day057 近重复指纹测试：shingle / Jaccard / MinHash 签名（全部离线、确定性）.

本文件守的是**近重复判据的数学底座**，不是"函数能跑通"：

1. ``shingles`` 的边界（短文本退化、空文本、``k<1``）决定了短样本会不会
   被"空集合之间 Jaccard = 1.0"这条约定误判成彼此重复；
2. ``hash_shingle`` 必须**跨进程稳定**。一旦它换成内建 ``hash()``，
   同一份数据每次运行都会得到不同的签名，"这次重复率 12%"就变成一句
   无法复现的话；
3. ``jaccard_exact`` 是**定义**、``estimate_jaccard`` 是**估计**。
   测试必须同时钉住两者及两者之间的差距——阈值留不留裕度全靠这个差距；
4. 下面若干条断言里的数字是**实测标定值**（本课程语料上跑出来的），
   不是推导出来的。它们变了就意味着指纹口径变了，必须当场解释。
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from smart_research_agent.domain_data.errors import DomainDataError
from smart_research_agent.domain_data.fingerprint import (
    DEFAULT_NUM_PERM,
    DEFAULT_SHINGLE_K,
    MINHASH_EMPTY,
    MINHASH_PRIME,
    MINHASH_SEED,
    estimate_jaccard,
    exact_key,
    fingerprint_example,
    fingerprint_text,
    hash_shingle,
    jaccard_exact,
    minhash_signature,
    normalize_for_fingerprint,
    permutation_params,
    shingles,
)
from smart_research_agent.finetune.cleaner import DEDUPE_KEY_LENGTH, dedupe_key
from smart_research_agent.finetune.schema import TrainingExample

#: 实测标定用的三条中文提问。前两条是"同一份语料被加了开场白"，
#: 第三条与第一条**意思几乎一样但字面不同**——本模块明确不抓语义重复。
QUESTION = "如何评估 RAG 的检索质量？"
QUESTION_PREFIXED = "请帮我看看：如何评估 RAG 的检索质量？"
QUESTION_PARAPHRASE = "怎么评估 RAG 的检索效果？"

#: 与上面三条毫无字面重叠的对照样本（用于"确实不像"的断言）
FAR_QUESTION = "如何配置 MCP 服务器并接入本地文件系统？请给出可复现的步骤。"

#: 归一化后的第一条提问（NFKC 把全角 ``？`` 变成半角 ``?``）
QUESTION_NORMALIZED = "如何评估 rag 的检索质量?"


def make_example(prompt: str, output: str = "可以从命中率与召回率评估。") -> TrainingExample:
    """构造一条最小可用的训练样本（不读磁盘、不依赖真实数据文件）."""
    return TrainingExample(instruction=prompt, output=output)


class TestShingles:
    """字符 n-gram 集合：切分口径 + 三个边界."""

    def test_long_text_yields_all_three_grams(self):
        """15 个字符 → 13 个 3-gram（``len - k + 1``），且内容逐条可核对."""
        grams = shingles(QUESTION)
        assert len(grams) == 13
        assert "如何评" in grams
        assert "质量?" in grams  # 全角问号已被 NFKC 归一成半角
        assert all(len(gram) == DEFAULT_SHINGLE_K for gram in grams)

    def test_normalization_matches_cleaner(self):
        """指纹的"看起来一样"必须与 day048 精确去重同一个定义，否则两份报告会打架."""
        assert normalize_for_fingerprint(QUESTION) == QUESTION_NORMALIZED
        assert normalize_for_fingerprint(f"  {QUESTION}  ") == QUESTION_NORMALIZED
        assert shingles(f"  {QUESTION}  ") == shingles(QUESTION)

    def test_full_width_and_case_are_folded(self):
        """NFKC 折叠全角、小写化折叠大小写：``ＲＡＧ`` 与 ``RAG`` 不可是两个 shingle."""
        assert shingles("ＲＡＧ") == shingles("RAG") == frozenset({"rag"})

    def test_text_shorter_than_k_degrades_to_whole_string(self):
        """短于 k 时退化成"整串一个 shingle"。

        若不这样处理，短文本会得到空集合，而空集合之间的 Jaccard 是 1.0
        ——"好"与"差"会被判成完全重复。
        """
        assert shingles("如何", k=3) == frozenset({"如何"})
        assert shingles("好", k=5) == frozenset({"好"})

    def test_length_equal_to_k_is_exactly_one_shingle(self):
        assert shingles("abc", k=3) == frozenset({"abc"})
        assert shingles("abcd", k=3) == frozenset({"abc", "bcd"})

    @pytest.mark.parametrize("text", ["", "   ", "\n\n"])
    def test_blank_text_yields_empty_set(self, text):
        """空白文本归一化后为空 → 空集合（不是"一个空 shingle"）."""
        assert shingles(text) == frozenset()

    @pytest.mark.parametrize("k", [0, -1, -7])
    def test_window_below_one_rejected(self, k):
        """``k=0`` 会让每条文本得到无意义结果，必须在入口拒绝而不是静默返回."""
        with pytest.raises(DomainDataError, match="shingle 窗口 k 必须 >= 1"):
            shingles(QUESTION, k=k)


class TestJaccardExact:
    """精确 Jaccard：MinHash 要逼近的定义，测试拿它当基准."""

    def test_both_empty_is_one(self):
        """两条空文本"确实是同一件事"——这条约定让精确值与估计值在全空输入上一致."""
        assert jaccard_exact(frozenset(), frozenset()) == 1.0

    @pytest.mark.parametrize(
        "left, right",
        [(frozenset(), frozenset({"abc"})), (frozenset({"abc"}), frozenset())],
    )
    def test_one_side_empty_is_zero(self, left, right):
        assert jaccard_exact(left, right) == 0.0

    def test_identical_sets_is_one(self):
        assert jaccard_exact(shingles(QUESTION), shingles(QUESTION)) == 1.0

    def test_disjoint_sets_is_zero(self):
        assert jaccard_exact(shingles(QUESTION), shingles(FAR_QUESTION)) == 0.0

    def test_calibrated_prefix_copy_pair(self):
        """实测：加 6 字前缀的同一条语料，精确 Jaccard 只有 0.6842.

        （并集 19、交集 13 → 13/19；短中文提问上字符 3-gram 的粒度很粗。）
        """
        assert jaccard_exact(shingles(QUESTION), shingles(QUESTION_PREFIXED)) == pytest.approx(
            0.6842, abs=1e-4
        )

    def test_calibrated_paraphrase_pair(self):
        """实测：语义相近但字面不同的改写对，精确 Jaccard 为 0.4444（4/9）。

        注意：``fingerprint.py`` 模块 docstring 里写的 0.3000 与当前实现不符
        ——这里按**实测**钉住 0.4444。两处口径若长期不一致，评审时会说不清
        "到底哪个数字是近重复判据的输入"。语义重复本就不该由本模块抓，
        因此无论 0.3000 还是 0.4444，它都必须落在阈值之下。
        """
        assert jaccard_exact(shingles(QUESTION), shingles(QUESTION_PARAPHRASE)) == pytest.approx(
            0.4444, abs=1e-4
        )


class TestHashShingle:
    """shingle → 64 位整数：必须可复现，且不是内建 ``hash``."""

    def test_matches_blake2b_definition(self):
        """与 ``blake2b(digest_size=8)`` 的大端整数完全一致（定义即断言）."""
        expected = int.from_bytes(hashlib.blake2b(b"RAG", digest_size=8).digest(), "big")
        assert hash_shingle("RAG") == expected

    def test_stable_across_processes_while_builtin_hash_is_not(self):
        """跨进程稳定性：换 ``PYTHONHASHSEED`` 后本函数不变、内建 ``hash`` 会变.

        这是本模块选择 ``blake2b`` 的**唯一理由**，也必须是一条可执行的断言：
        若哪天有人把它换回 ``hash()``，签名每次运行都不同，去重结果不可复现。
        """
        script = (
            "from smart_research_agent.domain_data.fingerprint import hash_shingle;"
            "print(hash_shingle('RAG'), hash('RAG'))"
        )
        root = Path(__file__).resolve().parents[1]
        env = {
            **os.environ,
            "PYTHONPATH": f"{root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        }
        outputs = {}
        for seed in ("0", "1"):
            completed = subprocess.run(
                [sys.executable or "python", "-c", script],
                capture_output=True,
                text=True,
                env={**env, "PYTHONHASHSEED": seed},
                check=True,
            )
            fingerprint_value, builtin_value = completed.stdout.split()
            outputs[seed] = (fingerprint_value, builtin_value)

        assert outputs["0"][0] == outputs["1"][0] == str(hash_shingle("RAG"))
        assert outputs["0"][1] != outputs["1"][1], "内建 hash 加了盐，正是不能用它的原因"


class TestPermutationParams:
    """``h(x) = (a*x + b) mod p`` 的置换参数：可复现 + 取值域正确."""

    def test_same_seed_is_reproducible(self):
        assert permutation_params(8, seed=MINHASH_SEED) == permutation_params(8, seed=MINHASH_SEED)

    def test_different_seed_gives_different_params(self):
        """种子是签名的一部分：换种子必须换参数，否则"seed"是个摆设."""
        assert permutation_params(8, seed=42) != permutation_params(8, seed=43)

    @pytest.mark.parametrize("num_perm", [1, 8, 64])
    def test_length_matches_num_perm(self, num_perm):
        assert len(permutation_params(num_perm)) == num_perm

    def test_a_is_never_zero_and_params_are_in_range(self):
        """``a=0`` 会让所有输入映射到 ``b``（签名退化），因此 ``a`` 恒 >= 1."""
        params = permutation_params(64)
        assert all(1 <= a < MINHASH_PRIME for a, _ in params)
        assert all(0 <= b < MINHASH_PRIME for _, b in params)

    @pytest.mark.parametrize("num_perm", [0, -3])
    def test_num_perm_below_one_rejected(self, num_perm):
        with pytest.raises(DomainDataError, match="置换个数 num_perm 必须 >= 1"):
            permutation_params(num_perm)


class TestMinHashSignature:
    """签名长度恒定 + 空集合占位 + 与 shingle 顺序无关."""

    @pytest.mark.parametrize("num_perm", [1, 8, 64])
    def test_length_is_always_num_perm(self, num_perm):
        """长度恒定，才可能逐位比较；长度不同即参数不匹配（见下一个测试类）."""
        signature = minhash_signature(shingles(QUESTION), num_perm=num_perm)
        assert len(signature) == num_perm

    def test_default_length_is_module_constant(self):
        assert len(minhash_signature(shingles(QUESTION))) == DEFAULT_NUM_PERM

    def test_empty_shingles_give_all_placeholder_values(self):
        """空集合 → 全 ``MINHASH_EMPTY``：保证"空 vs 空"估计 1.0、"空 vs 非空"估计 0."""
        signature = minhash_signature(frozenset(), num_perm=16)
        assert set(signature) == {MINHASH_EMPTY}
        assert MINHASH_EMPTY == 0

    def test_deterministic_and_order_insensitive(self):
        """MinHash 取的是最小值，与遍历顺序无关——顺序敏感会让同一份文本给出两条签名."""
        assert minhash_signature(shingles(QUESTION)) == minhash_signature(shingles(QUESTION))
        assert minhash_signature(["abc", "bcd"]) == minhash_signature(["bcd", "abc"])

    def test_seed_changes_signature(self):
        """签名里含 seed：换 seed 后同一条文本的签名必须不同（否则 seed 无效）."""
        assert minhash_signature(shingles(QUESTION), seed=43) != minhash_signature(
            shingles(QUESTION), seed=MINHASH_SEED
        )

    def test_estimate_matches_calibrated_ratio(self):
        """实测：前缀复制对的估计值 0.5469 = 35/64（签名位数固定，故比值可数）."""
        left = minhash_signature(shingles(QUESTION))
        right = minhash_signature(shingles(QUESTION_PREFIXED))
        assert estimate_jaccard(left, right) == pytest.approx(0.5469, abs=1e-4)


class TestEstimateJaccard:
    """估计量与它的边界约定."""

    def test_identical_signatures_is_one(self):
        signature = minhash_signature(shingles(QUESTION))
        assert estimate_jaccard(signature, signature) == 1.0

    def test_all_placeholder_signatures_is_one(self):
        """``((0,)*3, (0,)*3)`` → 1.0：两条空文本的估计值必须与精确值一致."""
        assert estimate_jaccard((0,) * 3, (0,) * 3) == 1.0

    def test_empty_signatures_is_one(self):
        """长度 0 的两条空签名：没有位次可以"不匹配"，故为 1.0（与代码分支一致）."""
        assert estimate_jaccard((), ()) == 1.0

    def test_unrelated_texts_yield_zero(self):
        left = minhash_signature(shingles(QUESTION))
        right = minhash_signature(shingles(FAR_QUESTION))
        assert estimate_jaccard(left, right) == 0.0

    def test_length_mismatch_rejected(self):
        """长度不同意味着 ``num_perm``/``seed`` 不同，给分数就是给一个没有意义的数字."""
        with pytest.raises(DomainDataError, match="签名长度不一致"):
            estimate_jaccard((1, 2, 3), (1, 2))

    def test_estimate_is_a_proportion_of_positions(self):
        """逐位相等比例：手算 2/4 的签名必须给 0.5，钉住"无偏估计"的定义."""
        assert estimate_jaccard((1, 2, 3, 4), (1, 2, 9, 9)) == 0.5


class TestCalibration:
    """阈值能不能定准，取决于这些实测数字本身是否可信."""

    def test_prefix_copy_is_measurably_near(self):
        """前缀复制对：精确 0.6842、估计 0.5469。

        两个数字都**低于**常见经验阈值 0.8——这就是本课把阈值标定到 0.7
        的理由：取 0.8 会让近重复链路静默空转。
        """
        exact = jaccard_exact(shingles(QUESTION), shingles(QUESTION_PREFIXED))
        estimate = estimate_jaccard(
            minhash_signature(shingles(QUESTION)),
            minhash_signature(shingles(QUESTION_PREFIXED)),
        )
        assert exact == pytest.approx(0.6842, abs=1e-4)
        assert estimate == pytest.approx(0.5469, abs=1e-4)
        assert exact < 0.8 and estimate < 0.8

    def test_paraphrase_is_below_any_sane_threshold(self):
        """语义改写对的精确值 0.4444 明显低于 0.7：字面重复与语义重复是两条链路."""
        exact = jaccard_exact(shingles(QUESTION), shingles(QUESTION_PARAPHRASE))
        assert exact == pytest.approx(0.4444, abs=1e-4)
        assert exact < 0.7


class TestFingerprintObject:
    """``Fingerprint`` 数据类与 ``fingerprint_text`` / ``fingerprint_example``."""

    def test_text_fingerprint_fields(self):
        fingerprint = fingerprint_text(QUESTION)
        assert fingerprint.key == exact_key(QUESTION)
        assert len(fingerprint.key) == DEDUPE_KEY_LENGTH
        assert fingerprint.shingle_count == 13
        assert len(fingerprint.signature) == DEFAULT_NUM_PERM

    def test_similarity_uses_signature(self):
        left = fingerprint_text(QUESTION)
        right = fingerprint_text(QUESTION_PREFIXED)
        assert left.similarity(right) == pytest.approx(0.5469, abs=1e-4)

    def test_same_key_is_near_even_at_threshold_one(self):
        """键相同必然 True，**不依赖估计**：100% 确定的重复不能交给概率裁决.

        阈值取 1.0（估计值不可能达到的上界）时仍然必须是 True，才说明
        "精确键优先"这条分支真的生效。
        """
        left = fingerprint_text(QUESTION)
        right = fingerprint_text(f"  {QUESTION}  ")
        assert left.key == right.key
        assert left.is_near(right, threshold=1.0) is True

    def test_unrelated_texts_are_not_near(self):
        assert (
            fingerprint_text(QUESTION).is_near(fingerprint_text(FAR_QUESTION), threshold=0.7)
            is False
        )

    def test_example_fingerprint_ignores_output(self):
        """两条"同问不同答"的样本必须给出**同一个 key**。

        口径与 day048 一致：只用 prompt。否则"同问不同答"会从"冲突"
        变成"不同"，重复数突然变小，同一份数据在两套口径下得到两个结论。
        """
        first = make_example(QUESTION, output="答案甲：看命中率。")
        second = make_example(QUESTION, output="完全不同的答案乙，长度也不一样，还带标点。")
        left = fingerprint_example(first)
        right = fingerprint_example(second)
        assert left.key == right.key
        assert left.signature == right.signature
        assert left.is_near(right, threshold=1.0) is True

    def test_example_fingerprint_covers_input_field(self):
        """比较对象是 ``prompt_text``（instruction + input），不是裸 instruction."""
        with_input = TrainingExample(instruction=QUESTION, input="材料 A", output="答案。")
        other_input = TrainingExample(instruction=QUESTION, input="材料 B", output="答案。")
        assert fingerprint_example(with_input).key != fingerprint_example(other_input).key

    def test_default_parameters_are_module_constants(self):
        assert DEFAULT_SHINGLE_K == 3
        assert DEFAULT_NUM_PERM == 64
        assert MINHASH_SEED == 42


class TestExactKey:
    """精确键：与 day048 ``dedupe_key`` 同定义，是两级去重的第一级."""

    def test_matches_cleaner_dedupe_key_on_same_prompt(self):
        """同一个 prompt 上两者必须逐字符相同，否则"精确"与"近"两级会互相打架."""
        example = make_example(QUESTION)
        assert exact_key(example.prompt_text) == dedupe_key(example)

    def test_length_and_normalization(self):
        assert exact_key(QUESTION) == exact_key(f"  {QUESTION}  ")
        assert exact_key("ＲＡＧ") == exact_key("RAG")
        assert len(exact_key(QUESTION)) == 16

    def test_different_prompts_differ(self):
        assert exact_key(QUESTION) != exact_key(QUESTION_PREFIXED)
