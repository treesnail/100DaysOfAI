"""数据清洗与质量过滤测试（day048）：清洗规则、七条默认规则、去重与报告计数."""

from __future__ import annotations

from smart_research_agent.finetune.cleaner import (
    DECISION_DUPLICATE,
    DECISION_KEPT,
    PLACEHOLDER_MARKERS,
    DatasetCleaner,
    QualityRule,
    clean_text,
    dedupe_key,
    default_rules,
    normalize_whitespace,
    strip_control_chars,
)
from smart_research_agent.finetune.schema import TrainingExample


def make_example(
    instruction: str = "什么是 RAG？",
    output: str = "检索增强生成（RAG）是先检索再生成的技术。",
    tags: tuple[str, ...] = (),
) -> TrainingExample:
    """构造一条干净样本（输出默认超过 8 字符，能通过 output_too_short）."""
    return TrainingExample(instruction=instruction, output=output, tags=tags)


class TestCleanText:
    """清洗总入口：NFKC → 控制字符 → 空白折叠."""

    def test_nfkc_normalizes_fullwidth_letters(self):
        assert clean_text("ＲＡＧ 与 微调") == "RAG 与 微调"

    def test_nfkc_normalizes_fullwidth_punctuation(self):
        """全角问号在 NFKC 下归一为半角——这是去重能识别变体的前提."""
        assert clean_text("什么是 RAG？") == "什么是 RAG?"

    def test_nfkc_normalizes_nbsp(self):
        assert clean_text("a\u00a0b") == "a b"

    def test_removes_zero_width_space(self):
        assert clean_text("注\u200b入") == "注入"

    def test_removes_nul_and_control_chars(self):
        assert clean_text("a\x00b\x07c") == "abc"

    def test_collapses_repeated_spaces(self):
        assert clean_text("a    b") == "a b"

    def test_converts_tabs_to_space(self):
        assert normalize_whitespace("a\t\tb") == "a b"

    def test_keeps_single_newline(self):
        """ReAct 轨迹依赖换行，清洗不能把结构压平."""
        assert clean_text("Thought: 想一想\nAction: calculator") == (
            "Thought: 想一想\nAction: calculator"
        )

    def test_collapses_blank_lines(self):
        assert clean_text("a\n\n\nb") == "a\nb"

    def test_strips_surrounding_whitespace(self):
        assert clean_text("   \n 答案 \n  ") == "答案"

    def test_empty_inputs(self):
        assert clean_text("") == ""
        assert normalize_whitespace("") == ""
        assert strip_control_chars("") == ""

    def test_strip_control_chars_keeps_newline_and_tab(self):
        assert strip_control_chars("a\nb\tc") == "a\nb\tc"


class TestDedupeKey:
    """去重指纹：清洗 + 小写之后相同即视为重复."""

    def test_case_and_whitespace_insensitive(self):
        first = dedupe_key(make_example(instruction="RAG   是什么？"))
        second = dedupe_key(make_example(instruction="rag 是什么？"))
        assert first == second

    def test_fullwidth_variant_is_duplicate(self):
        assert dedupe_key(make_example(instruction="ＲＡＧ 是什么？")) == dedupe_key(
            make_example(instruction="RAG 是什么？")
        )

    def test_different_prompts_differ(self):
        assert dedupe_key(make_example(instruction="问题一")) != dedupe_key(
            make_example(instruction="问题二")
        )

    def test_key_length(self):
        assert len(dedupe_key(make_example())) == 16


class TestDefaultRules:
    """七条默认规则：每条各给一个反例与一个正例."""

    def setup_method(self):
        self.rules = {
            rule.name: rule
            for rule in default_rules(
                min_output_chars=8,
                max_output_chars=20,
                max_instruction_chars=10,
                banned_patterns=("违法",),
            )
        }

    def test_rule_names_and_order(self):
        assert list(self.rules) == [
            "output_too_short",
            "output_too_long",
            "instruction_too_long",
            "empty_instruction",
            "placeholder_output",
            "banned_pattern",
            "unverified_source",
        ]

    def test_output_too_short(self):
        rule = self.rules["output_too_short"]
        assert rule.check(make_example(output="太短")) is False
        assert rule.check(make_example(output="八个字符以上的答案")) is True

    def test_output_too_long(self):
        rule = self.rules["output_too_long"]
        assert rule.check(make_example(output="长" * 21)) is False
        assert rule.check(make_example(output="长" * 20)) is True

    def test_instruction_too_long(self):
        rule = self.rules["instruction_too_long"]
        assert rule.check(make_example(instruction="问" * 11)) is False
        assert rule.check(make_example(instruction="问" * 10)) is True

    def test_empty_instruction(self):
        rule = self.rules["empty_instruction"]
        assert rule.check(make_example(instruction="   ")) is False
        assert rule.check(make_example(instruction="问题")) is True

    def test_placeholder_output(self):
        rule = self.rules["placeholder_output"]
        assert rule.check(make_example(output="TODO：待补充完整说明")) is False
        assert rule.check(make_example(output="这是一句完整的答案")) is True
        assert "lorem" in PLACEHOLDER_MARKERS

    def test_banned_pattern(self):
        rule = self.rules["banned_pattern"]
        assert rule.check(make_example(output="这可能涉及违法内容")) is False
        assert rule.check(make_example(output="这是合法内容")) is True

    def test_unverified_source(self):
        rule = self.rules["unverified_source"]
        assert rule.check(make_example(tags=("from:eval", "unverified"))) is False
        assert rule.check(make_example(tags=("from:eval", "react-trace"))) is True

    def test_custom_rule_table_can_be_injected(self):
        rule = QualityRule(
            name="no_answer_marker",
            reason="不允许出现固定串",
            check=lambda example: "?" not in example.output,
        )
        cleaner = DatasetCleaner(rules=[rule])
        kept, report = cleaner.run([make_example(output="答案?")])
        assert kept == []
        assert report.drop_reasons == {"no_answer_marker": 1}


class TestDatasetCleanerRun:
    """run()：清洗 → 规则 → 去重，并逐条记账."""

    def test_returns_cleaned_examples(self):
        cleaner = DatasetCleaner()
        kept, _ = cleaner.run(
            [make_example(instruction="  什么是  RAG ? ", output=" 这是一个足够长的答案 ")]
        )
        assert kept[0].instruction == "什么是 RAG ?"
        assert kept[0].output == "这是一个足够长的答案"

    def test_does_not_mutate_input(self):
        cleaner = DatasetCleaner()
        original = make_example(instruction="  原始  ")
        cleaner.run([original])
        assert original.instruction == "  原始  "

    def test_dedupe_keeps_first_occurrence(self):
        cleaner = DatasetCleaner()
        examples = [
            make_example(instruction="RAG 是什么？", output="这是第一个足够长的答案"),
            make_example(instruction="rag   是什么？", output="这是第二个足够长的答案"),
        ]
        kept, report = cleaner.run(examples)
        assert len(kept) == 1
        assert kept[0].output == "这是第一个足够长的答案"
        assert report.duplicates == 1
        assert report.drop_reasons[DECISION_DUPLICATE] == 1

    def test_dedupe_disabled(self):
        cleaner = DatasetCleaner(dedupe=False)
        kept, report = cleaner.run([make_example(), make_example()])
        assert len(kept) == 2
        assert report.duplicates == 0
        assert DECISION_DUPLICATE not in report.drop_reasons

    def test_rule_rejection_counted_by_name(self):
        cleaner = DatasetCleaner(min_output_chars=8)
        kept, report = cleaner.run([make_example(output="短"), make_example(instruction="另一问")])
        assert len(kept) == 1
        assert report.drop_reasons == {"output_too_short": 1}

    def test_decisions_record_kept_and_rejected(self):
        cleaner = DatasetCleaner(min_output_chars=8)
        _, report = cleaner.run([make_example(), make_example(output="短")])
        assert [d.rule for d in report.decisions] == [DECISION_KEPT, "output_too_short"]

    def test_report_counts_and_rate(self):
        cleaner = DatasetCleaner(min_output_chars=8)
        _, report = cleaner.run(
            [
                make_example(instruction="q1"),
                make_example(instruction="q2"),
                make_example(instruction="q3", output="短"),
                make_example(instruction="q4", output="短"),
            ]
        )
        assert report.total == 4
        assert report.kept == 2
        assert report.rejected == 2
        assert report.keep_rate == 0.5
        payload = report.to_dict()
        assert payload["total"] == 4
        assert payload["kept"] == 2
        assert payload["rejected"] == 2
        assert payload["drop_reasons"] == {"output_too_short": 2}
        assert len(payload["decisions"]) == 4
        assert payload["decisions"][0]["rule"] == DECISION_KEPT
        assert payload["decisions"][0]["source"] == ""

    def test_empty_input_report(self):
        cleaner = DatasetCleaner()
        kept, report = cleaner.run([])
        assert kept == []
        assert report.total == 0
        assert report.keep_rate == 0.0
        assert report.to_dict()["drop_reasons"] == {}
