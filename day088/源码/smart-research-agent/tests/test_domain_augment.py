"""day057 增强算子测试（M5-D8）：不变式、守门、轮转与"只在增强样本之间去重".

本文件把 ``augment.py`` 模块开头的三条纪律逐条钉成断言，而不是靠注释保证：

1. **只改 prompt 侧**：任何算子产出的 ``output`` 与入参**逐字相同**，
   ``source``/``license``/``system``/``input`` 原样继承，``tags`` 追加
   ``aug:<算子名>``——增强样本永久可查，且出处不会被改写；
2. **能改写答案正确性的增强必须自证**：``constraint`` 只在"原答案本来就
   满足这条约束"时才追加，不满足就跳过并记 ``guard_rejected``；
3. **默认不启用会降低质量的算子**：``noise`` 在 ``AUGMENT_OPS`` 里但不在
   ``DEFAULT_OPS`` 里，且 ``OPS["noise"].risky is True``。

外加一条容易被漏掉、但必须写明的取舍：**增强阶段的去重只在增强样本之间
进行**（索引里不放原样本），否则 ``prefix`` 类增强会被自己的亲本判成重复。

全部断言只用内存样本，不联网、不用随机数、不依赖当前工作目录。
"""

from __future__ import annotations

import json

import pytest

from smart_research_agent.domain_data.augment import (
    AUG_APPLIED,
    AUG_GUARD_REJECTED,
    AUG_NO_CHANGE,
    AUG_TAG_PREFIX,
    AUGMENT_OPS,
    CHINESE_RATIO_FLOOR,
    CONSTRAINTS,
    DEFAULT_MAX_PER_EXAMPLE,
    DEFAULT_OPS,
    LENGTH_CONSTRAINT_CHARS,
    NOISE_VARIANTS,
    OPS,
    PARAPHRASE_PAIRS,
    PREFIX_VARIANTS,
    augment_dataset,
    augment_example,
    augment_ops_table,
    augmented_ops,
    cjk_ratio,
    does_not_restate,
    is_augmented,
    is_mostly_chinese,
    render_augment_table,
    resolve_ops,
    transform_constraint,
    transform_noise,
    transform_paraphrase,
    transform_prefix,
    within_chars,
)
from smart_research_agent.domain_data.errors import DomainDataError
from smart_research_agent.finetune.schema import TrainingExample


def full_example(
    instruction: str = "什么是 RAG？",
    output: str = "RAG 是检索增强生成，它先把相关片段检索出来，再交给模型生成答案。",
) -> TrainingExample:
    """带齐**全部治理字段**的样本：用来验证"增强不改出处"这条不变式.

    ``input``/``system``/``source``/``license`` 四个字段必须都有非空值，
    否则"原样继承"的断言会退化成"两边都是空字符串"，测不出问题。
    """
    return TrainingExample(
        instruction=instruction,
        output=output,
        input="待处理的材料正文",
        system="你是一个严谨的领域问答助手",
        source="seed",
        tags=("rag",),
        license="CC-BY-4.0",
    )


def chinese_example(instruction: str = "如何评估检索质量？") -> TrainingExample:
    """一条三条约束**全部满足**的样本（中文为主、不超长、不复述问题）."""
    return TrainingExample(
        instruction=instruction,
        output="用召回率与精确率一起评估，两者必须结合看，单看一个都会被误导。",
        source="seed",
        license="CC-BY-4.0",
    )


def long_chinese_example(chars: int = 300) -> TrainingExample:
    """一条"中文为主但超过 200 字"的样本：用来单点触发长度约束的守门."""
    instruction = "请说明 RAG 的检索阶段。"
    body = "它先用向量相似度召回候选片段。"
    output = (body * (chars // len(body) + 1))[:chars]
    return TrainingExample(instruction=instruction, output=output, source="seed")


def restating_example() -> TrainingExample:
    """一条"答案开头复述了问题"的样本：用来单点触发复述约束的守门."""
    instruction = "什么是 RAG？请介绍它的检索阶段。"
    return TrainingExample(
        instruction=instruction,
        output=f"{instruction}RAG 的检索阶段负责把候选片段找出来。",
        source="seed",
    )


class TestOpRegistry:
    """算子登记表：``AUGMENT_OPS`` 与 ``OPS`` 必须同源，且默认表由 risky 推出."""

    def test_augment_ops_matches_op_table_keys_and_order(self):
        """``AUGMENT_OPS`` 与 ``OPS`` 的键集合与**顺序**都必须一致.

        两个名字指同一份清单：文档与报表面用 ``AUGMENT_OPS``（元组、有序），
        查表用 ``OPS``（字典）。一旦它们漂移，``by_op`` 报告里就会出现
        一个"文档里没写过"的算子名。
        """
        assert AUGMENT_OPS == ("prefix", "constraint", "paraphrase", "noise")
        assert tuple(OPS) == AUGMENT_OPS

    def test_default_ops_is_derived_from_risky_flags(self):
        """默认启用的算子 = 非 risky 的那些；``noise`` 必须在列但默认关闭.

        这样断言（而不是抄一份字面量）是为了让"给流水线加一个算子之前，
        先问它出错时谁会报警"这条纪律在代码层面可执行：任何人把新算子
        标成 ``risky=True``，默认表自动把它排除。
        """
        assert DEFAULT_OPS == ("prefix", "constraint", "paraphrase")
        assert "noise" not in DEFAULT_OPS
        assert OPS["noise"].risky is True
        assert all(OPS[name].risky is False for name in DEFAULT_OPS)
        assert DEFAULT_OPS == tuple(name for name in AUGMENT_OPS if not OPS[name].risky)

    def test_every_op_carries_metadata_and_a_callable(self):
        """每个算子都要有改善维度、做法说明与可调用的纯函数实现."""
        for name, op in OPS.items():
            assert op.name == name
            assert op.dimension and op.description
            assert callable(op.transform)

    def test_resolve_ops_preserves_order(self):
        """``resolve_ops`` 按入参顺序解析，且返回的正是登记表里的对象."""
        resolved = resolve_ops(DEFAULT_OPS)
        assert resolved == tuple(OPS[name] for name in DEFAULT_OPS)

    def test_resolve_ops_rejects_unknown_names_early(self):
        """未知算子名必须在**构造期**抛 ``DomainDataError``.

        算子名打错最糟的表现是静默少生成一批样本（``by_op`` 少一个键但
        没有任何报错），所以这里显式断言"立刻报错"。
        """
        with pytest.raises(DomainDataError, match="未知的增强算子"):
            resolve_ops(("prefix", "nope"))

    def test_ops_table_enabled_flag_is_computed_from_default_ops(self):
        """对照表的 ``enabled_by_default`` 由 ``DEFAULT_OPS`` 现场算出，不是抄写."""
        table = augment_ops_table()
        assert [row["op"] for row in table] == list(AUGMENT_OPS)
        for row in table:
            assert row["enabled_by_default"] is (row["op"] in DEFAULT_OPS)
            assert row["risky"] is OPS[row["op"]].risky
            assert row["dimension"] == OPS[row["op"]].dimension

    def test_render_table_has_exactly_one_row_per_op(self):
        """渲染出的 markdown 行数 = 表头 2 行 + 每算子 1 行（文档由代码生成）."""
        rendered = render_augment_table()
        lines = [line for line in rendered.splitlines() if line.strip()]
        assert len(lines) == 2 + len(AUGMENT_OPS)
        for name in AUGMENT_OPS:
            assert f"| `{name}` |" in rendered


class TestPrefixRotation:
    """``prefix``：四个固定变体按 ``offset`` 轮转，不引入任何随机数."""

    def test_four_offsets_yield_four_distinct_variants(self):
        """``offset=0..3`` 得到四个**不同**的前缀，且都是纯前置语（原文是后缀）."""
        example = chinese_example()
        texts = []
        for offset, variant in enumerate(PREFIX_VARIANTS):
            result = transform_prefix(example, offset)
            assert result.status == AUG_APPLIED
            assert result.text == f"{variant}{example.instruction}"
            assert result.text.endswith(example.instruction)
            texts.append(result.text)
        assert len(set(texts)) == len(PREFIX_VARIANTS)

    def test_offset_wraps_around_after_the_last_variant(self):
        """``offset=4`` 回到第一个变体：轮转是取模，不是"用完了就没有"."""
        example = chinese_example()
        assert transform_prefix(example, 4).text == transform_prefix(example, 0).text
        assert transform_prefix(example, 4).text == f"{PREFIX_VARIANTS[0]}{example.instruction}"

    def test_augment_example_inherits_output_after_prefix(self):
        """``prefix`` 只改 instruction：答案逐字不动，且打上 ``aug:prefix`` 标签."""
        example = full_example()
        outcome = augment_example(example, OPS["prefix"], offset=1)
        assert outcome.status == AUG_APPLIED
        assert outcome.example is not None
        assert outcome.example.instruction == f"{PREFIX_VARIANTS[1]}{example.instruction}"
        assert outcome.example.output == example.output


class TestChineseRatioFloor:
    """``is_mostly_chinese`` 的阈值与边界（守门的第一道判据）."""

    def test_floor_is_exactly_0_3(self):
        """阈值 0.3 是模块常量；测试直接引用它，避免"代码改了断言没改"."""
        assert CHINESE_RATIO_FLOOR == 0.3

    def test_ratio_is_cjk_over_total_characters(self):
        """口径：中日韩表意文字数 / 总字符数（空文本按 0.0 计，不抛错）."""
        assert cjk_ratio("") == 0.0
        assert cjk_ratio("abc") == 0.0
        assert cjk_ratio("中文") == 1.0
        assert cjk_ratio("a中") == 0.5

    def test_floor_is_inclusive_on_both_sides(self):
        """边界 ``== 0.3`` 算"像中文"（``>=``），``0.2`` 不算.

        10 个字符里 3 个汉字恰好是 0.3：这条边界决定了"请用中文回答。"
        这一条约束会不会被追加，必须钉住。
        """
        assert is_mostly_chinese("中文中abcdefg") is True
        assert is_mostly_chinese("中文abcdefgh") is False
        assert is_mostly_chinese("") is False


class TestConstraintGuard:
    """``constraint`` 的三条守门：中文占比、长度上限、不复述问题."""

    def test_chinese_constraint_is_blocked_by_an_english_answer(self):
        """英文答案必须把"请用中文回答。"这条约束**拦下来**.

        否则会产出一条自相矛盾的增强样本（参考答案是英文，指令却要求中文）。
        断言分两层：判定器本身为 False，且轮转跳过了它、改用下一条约束——
        这正是"跳过是正确行为"的具体形态。
        """
        english = TrainingExample(
            instruction="What is RAG?",
            output=(
                "RAG combines a retriever with a generator so the model "
                "can answer with private knowledge."
            ),
            source="external",
            license="CC-BY-4.0",
        )
        assert CONSTRAINTS[0].text == "请用中文回答。"
        assert CONSTRAINTS[0].satisfied_by(english) is False
        assert is_mostly_chinese(english.output) is False

        result = transform_constraint(english, 0)
        assert result.status == AUG_APPLIED
        assert "请用中文回答" not in result.text
        assert result.text == f"{english.instruction}\n{CONSTRAINTS[1].text}"

    def test_length_constraint_is_blocked_by_an_overlong_answer(self):
        """``within_chars(200)`` 对超长答案为 False；守门跳过它并改追加下一条.

        给一条 300 字的答案追加"控制 200 字以内"会让参考答案越界，
        所以必须由守门拦下——同时它不影响"请用中文回答"这条（答案确实像中文）。
        """
        long_example = long_chinese_example()
        assert len(long_example.output) == 300
        assert within_chars(LENGTH_CONSTRAINT_CHARS)(long_example) is False
        assert CONSTRAINTS[1].satisfied_by(long_example) is False
        assert is_mostly_chinese(long_example.output) is True

        assert CONSTRAINTS[1].text == f"回答请控制在 {LENGTH_CONSTRAINT_CHARS} 字以内。"
        # offset=0 命中第一条约束（中文），长度约束不参与这一轮
        assert transform_constraint(long_example, 0).text == (
            f"{long_example.instruction}\n{CONSTRAINTS[0].text}"
        )
        # offset=1 先撞上长度约束 → 被拦 → 轮转到"不要复述问题"（该答案满足）
        skipped = transform_constraint(long_example, 1)
        assert skipped.status == AUG_APPLIED
        assert skipped.text == f"{long_example.instruction}\n{CONSTRAINTS[2].text}"

    def test_restatement_constraint_is_blocked_by_a_restating_answer(self):
        """``does_not_restate`` 对"答案开头是问题原文"的样本为 False."""
        restating = restating_example()
        assert does_not_restate(restating) is False
        assert CONSTRAINTS[2].satisfied_by(restating) is False
        assert does_not_restate(chinese_example()) is True

        # offset=2 命中复述约束 → 被拦 → 轮转回第一条（中文、长度都满足）
        result = transform_constraint(restating, 2)
        assert result.status == AUG_APPLIED
        assert "不要复述问题" not in result.text
        assert result.text == f"{restating.instruction}\n{CONSTRAINTS[0].text}"

    def test_missing_text_is_not_treated_as_restatement(self):
        """指令或答案为空时不算"复述问题"（空文本无从复述，按通过处理）.

        这条边界的分工要说清：空样本该由**清洗阶段**的 ``empty_instruction``
        规则拒掉，守门只负责"追加约束前先自证"。守门不能因为字段为空就把
        样本判死，否则两个阶段的责任就混了。
        """
        assert does_not_restate(TrainingExample(instruction="", output="答案足够长。")) is True
        assert does_not_restate(TrainingExample(instruction="问题是什么？", output="")) is True

    def test_all_three_blocked_returns_guard_rejected(self):
        """三条约束全被拦下时返回 ``AUG_GUARD_REJECTED``，且不产出样本.

        这条样本刻意同时满足三个"该被拦"的条件：答案以英文为主、超过 200 字、
        开头复述了问题。三个 offset 都必须得到同一个结论——守门与起点无关。
        """
        blocked_example = TrainingExample(
            instruction="什么是 RAG？请介绍它的检索阶段。",
            output="什么是 RAG？请介绍它的检索阶段。" + "x" * 300,
            source="seed",
        )
        for offset in range(len(CONSTRAINTS)):
            result = transform_constraint(blocked_example, offset)
            assert result.status == AUG_GUARD_REJECTED
            assert result.text == ""
            assert f"{len(CONSTRAINTS)} 条约束均被守门拦下" in result.detail

        outcome = augment_example(blocked_example, OPS["constraint"])
        assert outcome.status == AUG_GUARD_REJECTED
        assert outcome.example is None  # 不产出样本，而不是产出一条自相矛盾的样本
        assert json.loads(json.dumps(outcome.to_dict(), ensure_ascii=False))["status"] == (
            AUG_GUARD_REJECTED
        )


class TestParaphrase:
    """``paraphrase``：只替换第一处、只用受控词表，绝不自由改写."""

    def test_only_the_first_occurrence_is_replaced(self):
        """换一处就够：换得越多语义漂移风险越大."""
        instruction = "如何做 A 以及如何做 B"
        result = transform_paraphrase(TrainingExample(instruction=instruction, output="答案"), 0)
        assert result.status == AUG_APPLIED
        assert result.text == "怎么做 A 以及如何做 B"
        assert result.text.count("如何") == 1  # 第二处保持原样

    def test_every_offset_replaces_exactly_one_controlled_pair(self):
        """对 8 个 offset 逐个断言：结果必须等于"某条受控词对替换第一处"."""
        for offset, (source, target) in enumerate(PARAPHRASE_PAIRS):
            instruction = f"{source}做这件事"
            result = transform_paraphrase(
                TrainingExample(instruction=instruction, output="答案"), offset
            )
            assert result.status == AUG_APPLIED
            assert result.text == f"{target}做这件事"
            assert any(
                result.text == instruction.replace(left, right, 1)
                for left, right in PARAPHRASE_PAIRS
            )

    def test_instruction_without_a_controlled_word_yields_no_change(self):
        """指令里没有可替换词 → ``AUG_NO_CHANGE``（不是 applied，也不是 guard）."""
        example = TrainingExample(instruction="RAG 的检索阶段负责什么", output="答案")
        result = transform_paraphrase(example, 0)
        assert result.status == AUG_NO_CHANGE
        assert result.text == ""

        outcome = augment_example(example, OPS["paraphrase"])
        assert outcome.status == AUG_NO_CHANGE
        assert outcome.example is None


class TestNoiseOp:
    """``noise``：存在、可用、但**默认关闭**（质量打分察觉不到它）."""

    def test_noise_transform_still_works_when_explicitly_enabled(self):
        """显式启用时算子照常工作：首尾成对插入填充词."""
        example = chinese_example()
        head, tail = NOISE_VARIANTS[0]
        result = transform_noise(example, 0)
        assert result.status == AUG_APPLIED
        assert result.text == f"{head}{example.instruction}{tail}"

    def test_default_pipeline_ops_never_produce_noise_samples(self):
        """默认算子表跑出来的样本里**没有** ``aug:noise`` 标签，显式启用才有."""
        batch = [chinese_example("如何评估检索质量？"), chinese_example("为什么需要重排？")]

        default_result, default_report = augment_dataset(batch, max_per_example=1)
        tags = [tag for example in default_result for tag in augmented_ops(example)]
        assert "noise" not in tags
        assert set(default_report.by_op) <= set(DEFAULT_OPS)

        noise_result, noise_report = augment_dataset(batch, ops=("noise",), max_per_example=1)
        assert noise_report.by_op == {"noise": 2}
        assert all("aug:noise" in example.tags for example in noise_result[len(batch) :])


class TestAugmentExampleInvariants:
    """``augment_example`` 的三条不变式，对**每一个算子**逐条断言."""

    def test_output_is_verbatim_and_governance_fields_are_inherited(self):
        """最重要的一条：``output`` 逐字相同，出处字段原样继承，tags 追加 ``aug:<op>``.

        为什么逐条断言而不是抽查一个算子：增强一旦改了 output，就从"换一种问法"
        变成了"生成新答案"，而生成就可能出错——一条错了的增强样本会在训练里
        被当成金标准反复强化。
        """
        example = full_example()
        for name, op in OPS.items():
            outcome = augment_example(example, op, offset=0)
            assert outcome.status == AUG_APPLIED, name
            assert outcome.op == name
            produced = outcome.example
            assert produced is not None
            assert produced.output == example.output, name
            assert produced.input == example.input, name
            assert produced.system == example.system, name
            assert produced.source == example.source, name
            assert produced.license == example.license, name
            assert produced.tags == example.tags + (f"{AUG_TAG_PREFIX}{name}",), name
            assert augmented_ops(produced) == (name,)
            assert is_augmented(produced) is True
            assert produced.instruction != example.instruction  # prompt 侧确实变了
        # 入参本身没有被就地修改（增强不回头污染原样本）
        assert example.tags == ("rag",)
        assert example.instruction == "什么是 RAG？"

    def test_augmented_outcome_projection_is_json_serialisable(self):
        """``to_dict`` 必须能直接 ``json.dumps``（报告要落盘）."""
        outcome = augment_example(full_example(), OPS["prefix"], offset=0)
        payload = json.loads(json.dumps(outcome.to_dict(), ensure_ascii=False))
        assert payload["status"] == AUG_APPLIED
        assert payload["op"] == "prefix"
        assert payload["instruction"].endswith(full_example().instruction)

    def test_original_tags_are_preserved_in_order(self):
        """原标签在前、``aug:*`` 在后：标签是追加，不是替换."""
        example = full_example()
        example.tags = ("rag", "embedding")
        produced = augment_example(example, OPS["constraint"]).example
        assert produced is not None
        assert produced.tags == ("rag", "embedding", "aug:constraint")


class TestAugmentDataset:
    """``augment_dataset``：原样本优先、确定性轮转、只在增强样本之间去重."""

    def test_originals_come_first_and_are_the_same_objects(self):
        """返回列表的前 ``len(examples)`` 条**就是原样本本身**且顺序不变.

        用 ``is`` 而不是 ``==`` 断言：增强只做追加，从不重排或替换，
        因此"增强前后逐条对照"才能是一行代码的事。
        """
        batch = [chinese_example("如何评估检索质量？"), chinese_example("为什么需要重排？")]
        result, _ = augment_dataset(batch, max_per_example=1)
        assert len(result) == len(batch) + 2
        for original, produced in zip(batch, result[: len(batch)]):
            assert produced is original

    def test_default_max_per_example_is_one(self):
        """每条原样本缺省只产出 1 个增强样本（增强样本天然比原样本同质）."""
        assert DEFAULT_MAX_PER_EXAMPLE == 1
        batch = [chinese_example("如何评估检索质量？")]
        result, report = augment_dataset(batch)
        assert report.applied == 1
        assert len(result) == len(batch) + 1

    def test_zero_or_empty_ops_generate_nothing(self):
        """``max_per_example=0`` 与 ``ops=()`` 都不产出，但原样本照样返回."""
        batch = [chinese_example(), chinese_example("为什么需要重排？")]
        for kwargs in ({"max_per_example": 0}, {"ops": ()}):
            result, report = augment_dataset(batch, **kwargs)
            assert len(result) == len(batch)
            assert report.inputs == len(batch)
            assert report.applied == 0
            assert report.generated == 0
            assert report.by_op == {}
            assert report.expansion_ratio == 0.0

    def test_negative_max_per_example_raises(self):
        """负数是参数错误，必须响亮地拒绝（静默当成 0 会让调用方以为生效了）."""
        with pytest.raises(DomainDataError, match="max_per_example 不能为负"):
            augment_dataset([chinese_example()], max_per_example=-1)

    def test_accounting_identities_hold(self):
        """两本账必须对得上：``inputs + generated == len(结果)``，
        ``generated == applied - dropped_duplicates``，``sum(by_op) == generated``."""
        batch = [
            chinese_example("如何评估检索质量？"),
            chinese_example("什么是向量数据库？"),
            chinese_example("为什么检索之后还要再做一次重排？"),
        ]
        result, report = augment_dataset(batch, max_per_example=3)
        assert report.inputs == len(batch)
        assert report.generated == report.applied - report.dropped_duplicates
        assert len(result) == report.inputs + report.generated
        assert sum(report.by_op.values()) == report.generated
        assert report.expansion_ratio == pytest.approx(report.generated / report.inputs)

    def test_rotation_exercises_every_default_op_in_one_batch(self):
        """确定性轮转保证每个算子在同一批里都被行使（覆盖度优先于单条丰富度）.

        三条互不相似的指令 × 3 个尝试位，``offset = 样本序号 + 第几次尝试``
        让 ``prefix``/``constraint``/``paraphrase`` 各被行使 3 次。
        """
        batch = [
            chinese_example("如何评估检索质量？"),
            chinese_example("什么是向量数据库？"),
            chinese_example("为什么检索之后还要再做一次重排？"),
        ]
        result, report = augment_dataset(batch, ops=DEFAULT_OPS, max_per_example=3)
        assert report.by_op == {"prefix": 3, "constraint": 3, "paraphrase": 3}
        assert report.guard_rejected == {}
        assert report.no_change == {}
        assert report.dropped_duplicates == 0
        assert len(result) == len(batch) + 9

    def test_dedupe_scope_excludes_the_parents(self):
        """增强样本不会被自己的**亲本**判重：``prefix`` 天生与亲本高度重叠.

        实测：``"请帮我看看：X"`` 与 ``"X"`` 的 Jaccard 约 0.87，远高于阈值
        0.7。若把原样本也放进索引，几乎每条 prefix 增强都会被自己的亲本判成
        重复，整条增强链路大面积空转。因此这里断言"亲本不参与比较"：
        两条**互不相同**的原样本各产出 1 条增强，丢弃数必须是 0。
        """
        batch = [chinese_example("如何评估检索质量？"), chinese_example("为什么需要重排？")]
        result, report = augment_dataset(batch, ops=("prefix",), max_per_example=1)
        assert report.applied == 2
        assert report.dropped_duplicates == 0
        assert len(result) == 4
        assert all(augmented_ops(example) == ("prefix",) for example in result[2:])

    def test_identical_parents_collide_and_the_second_is_dropped(self):
        """两条**本来就相同**的原样本被同一算子改成同一个问法 → 第二条被丢弃.

        这是增强阶段去重要防的那一件事（"两条相近原样本被改成几乎一样的问法"）。
        注意它需要 ``max_per_example == len(PREFIX_VARIANTS)`` 才会发生：
        ``offset = 样本序号 + 第几次尝试``，两条相同样本只有落到同一个
        ``offset`` 上才会得到同一个变体。下面同时断言这个边界——
        ``max_per_example=1`` 时两条样本分别拿到变体 0 与变体 1，
        指令并不相同，因此一条都不该被丢。
        """
        identical = [
            TrainingExample(
                instruction="如何评估 RAG 的检索质量？", output="答案一。", source="seed"
            ),
            TrainingExample(
                instruction="如何评估 RAG 的检索质量？", output="答案二。", source="seed"
            ),
        ]

        single, single_report = augment_dataset(identical, ops=("prefix",), max_per_example=1)
        assert single_report.dropped_duplicates == 0
        assert single[2].instruction != single[3].instruction  # 变体不同 → 不算重复

        full, full_report = augment_dataset(
            identical, ops=("prefix",), max_per_example=len(PREFIX_VARIANTS)
        )
        assert full_report.applied == 2 * len(PREFIX_VARIANTS)
        assert full_report.dropped_duplicates >= 1  # 撞车的增强样本被丢弃
        assert len(full) == 2 + full_report.generated
        assert full_report.by_op == {"prefix": full_report.generated}

    def test_report_properties_and_json_projection(self):
        """``expansion_ratio``/``blocked``/``unchanged`` 与两张明细表逐项一致.

        批次里刻意放一条"三条约束全被拦"的样本与一条"没有可替换同义词"的样本，
        这样 ``guard_rejected`` 与 ``no_change`` 都非空——空集合上的断言
        （``0 == 0``）测不出任何东西。
        """
        blocked = TrainingExample(
            instruction="什么是 RAG？请介绍它的检索阶段。",
            output="什么是 RAG？请介绍它的检索阶段。" + "x" * 300,
            source="seed",
        )
        batch = [blocked, chinese_example("RAG 的检索阶段")]
        _, report = augment_dataset(batch, max_per_example=3)
        assert report.guard_rejected == {"constraint": 1}
        assert report.no_change == {"paraphrase": 1}
        assert report.blocked == sum(report.guard_rejected.values()) == 1
        assert report.unchanged == sum(report.no_change.values()) == 1
        assert report.generated == sum(report.by_op.values())
        assert report.expansion_ratio == pytest.approx(report.generated / report.inputs)
        payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
        assert payload["inputs"] == report.inputs
        assert payload["applied"] == report.applied
        assert payload["by_op"] == report.by_op
        assert payload["guard_rejected"] == report.guard_rejected
        assert payload["no_change"] == report.no_change
        assert "守门拦下" in report.summary_line()
        assert "撞车丢弃" in report.summary_line()

    def test_empty_input_reports_zero_ratio(self):
        """空输入不抛错，``expansion_ratio`` 按 0.0 计（不是除零崩溃）."""
        result, report = augment_dataset([])
        assert result == []
        assert report.inputs == 0
        assert report.expansion_ratio == 0.0
