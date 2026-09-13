"""数据采集测试（day048）：评估轨迹的 5/2/1 分布、红队安全样本、容错与归因.

真实数据断言直接读仓库里的 ``data/eval/*.jsonl``——采集层的语义（哪些轨迹
能用、哪些必须被标记）与真实语料强绑定，用假数据测不出问题。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from smart_research_agent.finetune.cleaner import DatasetCleaner
from smart_research_agent.finetune.collector import (
    EVAL_TASK_SOURCE,
    REDTEAM_SOURCE,
    SAFETY_REFUSAL_TEMPLATE,
    DataCollector,
    DataSource,
    EvalTaskSource,
    JSONLSource,
    RedTeamSource,
    SourceSpec,
    default_collector,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_TASKS = DATA_DIR / "eval" / "agent_tasks.jsonl"
REDTEAM_CASES = DATA_DIR / "eval" / "redteam_cases.jsonl"

COLLECTOR_LOGGER = "smart_research_agent.finetune.collector"


def write_jsonl(path: Path, records: list[dict]) -> Path:
    """把若干字典写成 JSONL（父目录自动创建）."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


class TestEvalTaskSource:
    """评估轨迹源的 5（保留）/ 2（unverified）/ 1（跳过）分布."""

    def test_distribution_on_real_data(self):
        source = EvalTaskSource(EVAL_TASKS)
        examples = source.load()
        assert source.kept == 5
        assert source.unverified == 2
        assert source.skipped == 1
        assert len(examples) == 7  # 保留 + unverified，跳过的不产出样本

    def test_kept_sample_fields(self):
        examples = EvalTaskSource(EVAL_TASKS).load()
        first = next(e for e in examples if e.instruction == "计算 123 乘以 456 的结果")
        assert first.source == EVAL_TASK_SOURCE
        assert first.tags == ("from:eval", "react-trace")
        assert "Final Answer: 结果是 56088" in first.output
        assert first.output.count("\n") >= 1  # ReAct 轨迹保留换行结构

    def test_wrong_answer_is_tagged_unverified(self):
        examples = EvalTaskSource(EVAL_TASKS).load()
        wrong = next(e for e in examples if e.instruction == "计算 7 乘以 8 等于多少")
        assert "unverified" in wrong.tags
        assert "99" in wrong.output  # 期望 56，实际 99
        assert wrong.source == EVAL_TASK_SOURCE

    def test_sample_without_answer_contains_is_skipped(self):
        examples = EvalTaskSource(EVAL_TASKS).load()
        assert all("3 加 5" not in example.instruction for example in examples)

    def test_missing_file_warns_and_returns_empty(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING, logger=COLLECTOR_LOGGER):
            examples = EvalTaskSource(tmp_path / "absent.jsonl").load()
        assert examples == []
        assert any("文件不存在" in record.message for record in caplog.records)

    def test_empty_task_is_skipped(self, tmp_path):
        path = write_jsonl(
            tmp_path / "tasks.jsonl",
            [
                {"id": "x", "task": "", "answer_contains": "1", "mock_responses": ["a"]},
                {
                    "id": "y",
                    "task": "问题",
                    "answer_contains": "1",
                    "mock_responses": ["Final Answer: 1"],
                },
            ],
        )
        source = EvalTaskSource(path)
        assert len(source.load()) == 1
        assert source.kept == 1


class TestRedTeamSource:
    """红队源：攻击 payload 转成安全拒答正样本."""

    def test_count_and_tags(self):
        examples = RedTeamSource(REDTEAM_CASES).load()
        assert len(examples) == 16
        assert all(example.source == REDTEAM_SOURCE for example in examples)
        assert all(example.is_safety_example() for example in examples)
        assert all(example.tags[:2] == ("from:redteam", "safety") for example in examples)

    def test_output_is_fixed_refusal_template(self):
        examples = RedTeamSource(REDTEAM_CASES).load()
        assert {example.output for example in examples} == {SAFETY_REFUSAL_TEMPLATE}

    def test_categories_are_kept_in_tags(self):
        categories = {example.tags[2] for example in RedTeamSource(REDTEAM_CASES).load()}
        assert categories == {"prompt_injection", "jailbreak", "pii_leak", "tool_abuse"}

    def test_missing_payload_is_skipped(self, tmp_path, caplog):
        path = write_jsonl(
            tmp_path / "redteam.jsonl",
            [
                {"id": "a", "category": "jailbreak", "description": "无 payload"},
                {"id": "b", "category": "jailbreak", "payload": "攻击串"},
            ],
        )
        with caplog.at_level(logging.WARNING, logger=COLLECTOR_LOGGER):
            examples = RedTeamSource(path).load()
        assert len(examples) == 1
        assert any("缺少 payload" in record.message for record in caplog.records)


class TestJSONLSource:
    """通用 JSONL 源：坏行跳过、来源补齐."""

    def test_parses_and_fills_source_and_license(self, tmp_path):
        path = write_jsonl(
            tmp_path / "seed.jsonl",
            [{"instruction": "问题", "output": "答案"}],
        )
        spec = SourceSpec(name="seed", path=path, license="CC-BY-4.0")
        examples = JSONLSource(spec).load()
        assert len(examples) == 1
        assert examples[0].source == "seed"
        assert examples[0].license == "CC-BY-4.0"

    def test_keeps_explicit_source(self, tmp_path):
        path = write_jsonl(
            tmp_path / "seed.jsonl",
            [{"instruction": "问题", "output": "答案", "source": "external"}],
        )
        spec = SourceSpec(name="seed", path=path, license="CC-BY-4.0")
        assert JSONLSource(spec).load()[0].source == "external"

    def test_bad_json_line_is_skipped_with_warning(self, tmp_path, caplog):
        path = tmp_path / "broken.jsonl"
        path.write_text(
            '{"instruction": "好样本", "output": "答案"}\n'
            "{这一行不是 JSON}\n"
            '{"instruction": "另一条", "output": "答案"}\n',
            encoding="utf-8",
        )
        spec = SourceSpec(name="broken", path=path)
        with caplog.at_level(logging.WARNING, logger=COLLECTOR_LOGGER):
            examples = JSONLSource(spec).load()
        assert len(examples) == 2
        assert any("JSON 解析失败" in record.message for record in caplog.records)

    def test_invalid_sample_is_skipped_with_warning(self, tmp_path, caplog):
        path = write_jsonl(
            tmp_path / "bad_sample.jsonl",
            [{"instruction": "缺少答案"}, {"instruction": "问题", "output": "答案"}],
        )
        spec = SourceSpec(name="bad_sample", path=path)
        with caplog.at_level(logging.WARNING, logger=COLLECTOR_LOGGER):
            examples = JSONLSource(spec).load()
        assert len(examples) == 1
        assert any("样本解析失败" in record.message for record in caplog.records)

    def test_non_object_line_is_skipped(self, tmp_path, caplog):
        path = tmp_path / "list.jsonl"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger=COLLECTOR_LOGGER):
            examples = JSONLSource(SourceSpec(name="list", path=path)).load()
        assert examples == []
        assert any("不是 JSON 对象" in record.message for record in caplog.records)

    def test_missing_file_returns_empty(self, tmp_path, caplog):
        spec = SourceSpec(name="absent", path=tmp_path / "absent.jsonl")
        with caplog.at_level(logging.WARNING, logger=COLLECTOR_LOGGER):
            assert JSONLSource(spec).load() == []
        assert caplog.records

    def test_blank_lines_are_skipped(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        path.write_text(
            "\n".join(
                [
                    '{"instruction": "问题一", "output": "答案一"}',
                    "",
                    "   ",
                    '{"instruction": "问题二", "output": "答案二"}',
                ]
            ),
            encoding="utf-8",
        )
        examples = JSONLSource(SourceSpec(name="blanks", path=path)).load()
        assert len(examples) == 2

    def test_chat_format_spec(self, tmp_path):
        path = write_jsonl(
            tmp_path / "chat.jsonl",
            [
                {
                    "messages": [
                        {"role": "user", "content": "问题"},
                        {"role": "assistant", "content": "答案"},
                    ]
                }
            ],
        )
        spec = SourceSpec(name="chat", path=path, fmt="chat")
        examples = JSONLSource(spec).load()
        assert examples[0].instruction == "问题"
        assert examples[0].tags == ()


class TestDataCollector:
    """采集器：合并多源、全局去重与按源归因."""

    def make_sources(self, tmp_path) -> list[DataSource]:
        seed_path = write_jsonl(
            tmp_path / "seed.jsonl",
            [{"instruction": f"种子问题 {i}", "output": "种子答案内容足够长"} for i in range(3)],
        )
        return [JSONLSource(SourceSpec(name="seed", path=seed_path))]

    def test_source_kept_counts(self, tmp_path):
        sources = self.make_sources(tmp_path)
        dirty = write_jsonl(
            tmp_path / "dirty.jsonl",
            [
                {"instruction": "太短的答案", "output": "短"},
                {"instruction": "合法问题", "output": "这是一条足够长的合法答案"},
            ],
        )
        sources.append(JSONLSource(SourceSpec(name="dirty", path=dirty)))
        bundle = DataCollector(sources, cleaner=DatasetCleaner(min_output_chars=8)).collect()
        assert bundle.source_kept == {"seed": 3, "dirty": 1}
        assert len(bundle.examples) == 4
        assert bundle.report.drop_reasons == {"output_too_short": 1}

    def test_dedupe_is_global_across_sources(self, tmp_path):
        first = write_jsonl(
            tmp_path / "a.jsonl",
            [{"instruction": "重复问题", "output": "来自第一个源的足够长的答案"}],
        )
        second = write_jsonl(
            tmp_path / "b.jsonl",
            [{"instruction": "  重复问题  ", "output": "来自第二个源的足够长的答案"}],
        )
        bundle = DataCollector(
            [
                JSONLSource(SourceSpec(name="a", path=first)),
                JSONLSource(SourceSpec(name="b", path=second)),
            ]
        ).collect()
        assert len(bundle.examples) == 1
        assert bundle.examples[0].output == "来自第一个源的足够长的答案"
        assert bundle.source_kept == {"a": 1, "b": 0}
        assert bundle.report.duplicates == 1

    def test_source_stats_matches_bundle(self, tmp_path):
        collector = DataCollector(self.make_sources(tmp_path))
        assert collector.source_stats() == collector.collect().source_kept

    def test_len_reflects_examples(self, tmp_path):
        bundle = DataCollector(self.make_sources(tmp_path)).collect()
        assert len(bundle) == len(bundle.examples) == 3

    def test_missing_source_contributes_zero(self, tmp_path):
        bundle = DataCollector(
            [JSONLSource(SourceSpec(name="absent", path=tmp_path / "absent.jsonl"))]
        ).collect()
        assert bundle.source_kept == {"absent": 0}
        assert bundle.examples == []

    def test_datasource_is_abstract(self):
        with pytest.raises(TypeError):
            DataSource()  # type: ignore[abstract]


class TestDefaultCollector:
    """default_collector：按 data_dir 装配三个源（用 tmp_path 造一套数据）."""

    def build_tree(self, tmp_path) -> Path:
        finetune_dir = tmp_path / "data" / "finetune"
        eval_dir = tmp_path / "data" / "eval"
        write_jsonl(
            finetune_dir / "seed_examples.jsonl",
            [{"instruction": f"种子 {i}", "output": "这是一条足够长的种子答案"} for i in range(2)],
        )
        write_jsonl(
            eval_dir / "agent_tasks.jsonl",
            [
                {
                    "id": "t-1",
                    "task": "计算 1 加 1",
                    "expected_tools": ["calculator"],
                    "answer_contains": "2",
                    "mock_responses": ["Thought: 用计算器\nFinal Answer: 结果是 2"],
                }
            ],
        )
        write_jsonl(
            eval_dir / "redteam_cases.jsonl",
            [{"id": "r-1", "category": "jailbreak", "payload": "角色扮演越狱", "description": "x"}],
        )
        return finetune_dir

    def test_collects_three_sources(self, tmp_path):
        bundle = default_collector(self.build_tree(tmp_path)).collect()
        assert bundle.source_kept == {"seed": 2, EVAL_TASK_SOURCE: 1, REDTEAM_SOURCE: 1}
        assert len(bundle.examples) == 4
        assert bundle.report.rejected == 0

    def test_eval_trace_enters_dataset(self, tmp_path):
        bundle = default_collector(self.build_tree(tmp_path)).collect()
        trace = next(e for e in bundle.examples if e.source == EVAL_TASK_SOURCE)
        assert "react-trace" in trace.tags
        assert trace.output.startswith("Thought:")
