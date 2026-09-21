"""数据集画像、切分与落盘测试（day048）：统计口径、可复现切分与 JSONL 落盘."""

from __future__ import annotations

from pathlib import Path

import pytest

from smart_research_agent.finetune.cleaner import FilterReport
from smart_research_agent.finetune.collector import DatasetBundle
from smart_research_agent.finetune.dataset import (
    DatasetStats,
    build_dataset,
    compute_stats,
    dump_bundle,
    split_dataset,
)
from smart_research_agent.finetune.schema import CHAT, TrainingExample, load_jsonl
from smart_research_agent.llm.tokenizer import TokenCounter

#: 仓库里种子样本清洗后的保留数下限（随种子文件增长只会更多，不会更少）
DEFAULT_SEED_MIN = 14


class CountingStub:
    """固定返回 10 的计数器：用来证明 compute_stats 真的用了注入的 counter."""

    def count_tokens(self, text: str, model: str | None = None) -> int:
        return 10


def make_examples(count: int = 5) -> list[TrainingExample]:
    return [
        TrainingExample(
            instruction=f"问题 {index}",
            output=f"答案 {index} 的正文内容",
            source="seed" if index % 2 == 0 else "eval/agent_tasks",
            tags=("rag",) if index % 2 == 0 else ("from:eval", "react-trace"),
        )
        for index in range(count)
    ]


class TestComputeStats:
    """数据集画像：空集不抛错、长度与分布口径确定."""

    def test_empty_list_returns_zeros(self):
        stats = compute_stats([])
        assert isinstance(stats, DatasetStats)
        assert stats.count == 0
        assert stats.avg_instruction_chars == 0.0
        assert stats.avg_output_chars == 0.0
        assert stats.min_output_chars == 0
        assert stats.max_output_chars == 0
        assert stats.avg_estimated_tokens == 0.0
        assert stats.source_distribution == {}
        assert stats.tag_distribution == {}

    def test_counts_and_average_lengths(self):
        examples = [
            TrainingExample(instruction="一二三", output="答案"),
            TrainingExample(instruction="一二", output="答案答案"),
        ]
        stats = compute_stats(examples)
        assert stats.count == 2
        assert stats.avg_instruction_chars == 2.5
        assert stats.avg_output_chars == 3.0
        assert stats.min_output_chars == 2
        assert stats.max_output_chars == 4

    def test_uses_default_offline_counter(self):
        examples = make_examples(2)
        counter = TokenCounter(prefer_tiktoken=False)
        expected = sum(
            counter.count_tokens(e.prompt_text) + counter.count_tokens(e.output) for e in examples
        ) / 2
        assert compute_stats(examples).avg_estimated_tokens == expected

    def test_injected_counter_is_used(self):
        stats = compute_stats(make_examples(3), counter=CountingStub())
        assert stats.avg_estimated_tokens == 20.0  # (prompt + output) 各 10

    def test_distributions(self):
        stats = compute_stats(make_examples(4))
        assert stats.source_distribution == {"seed": 2, "eval/agent_tasks": 2}
        assert stats.tag_distribution["rag"] == 2
        assert stats.tag_distribution["from:eval"] == 2

    def test_to_dict_rounds_averages(self):
        payload = compute_stats(make_examples(3)).to_dict()
        assert set(payload) == {
            "count",
            "avg_instruction_chars",
            "avg_output_chars",
            "min_output_chars",
            "max_output_chars",
            "avg_estimated_tokens",
            "source_distribution",
            "tag_distribution",
        }
        assert payload["count"] == 3
        assert isinstance(payload["avg_instruction_chars"], float)

    def test_summary_line(self):
        line = compute_stats(make_examples(3)).summary_line()
        assert "样本 3 条" in line
        assert "token/条" in line


class TestSplitDataset:
    """切分：可复现、两侧非空、比例越界报错."""

    def test_same_seed_is_reproducible(self):
        examples = make_examples(20)
        first_train, first_eval = split_dataset(examples, eval_ratio=0.2, seed=7)
        second_train, second_eval = split_dataset(examples, eval_ratio=0.2, seed=7)
        assert first_train == second_train
        assert first_eval == second_eval

    def test_different_seed_changes_partition(self):
        examples = make_examples(40)
        _, eval_a = split_dataset(examples, eval_ratio=0.25, seed=1)
        _, eval_b = split_dataset(examples, eval_ratio=0.25, seed=2)
        assert [e.instruction for e in eval_a] != [e.instruction for e in eval_b]

    def test_eval_count_from_ratio(self):
        examples = make_examples(37)
        train, eval_set = split_dataset(examples, eval_ratio=0.2, seed=42)
        assert len(eval_set) == 7
        assert len(train) == 30
        assert len(train) + len(eval_set) == 37

    def test_ratio_zero_still_yields_non_empty_eval(self):
        train, eval_set = split_dataset(make_examples(10), eval_ratio=0.0, seed=42)
        assert len(eval_set) == 1
        assert len(train) == 9

    def test_no_overlap(self):
        train, eval_set = split_dataset(make_examples(10), eval_ratio=0.3, seed=42)
        assert not {e.instruction for e in train} & {e.instruction for e in eval_set}

    def test_two_examples_split_both_non_empty(self):
        train, eval_set = split_dataset(make_examples(2), eval_ratio=0.9, seed=42)
        assert len(train) == 1
        assert len(eval_set) == 1

    def test_single_example_keeps_train(self):
        train, eval_set = split_dataset(make_examples(1), eval_ratio=0.2, seed=42)
        assert len(train) == 1
        assert eval_set == []

    def test_empty_dataset(self):
        assert split_dataset([], eval_ratio=0.2, seed=42) == ([], [])

    @pytest.mark.parametrize("ratio", [-0.1, 1.0, 1.5])
    def test_invalid_ratio_raises(self, ratio):
        with pytest.raises(ValueError, match="eval_ratio"):
            split_dataset(make_examples(10), eval_ratio=ratio)


class TestDumpBundle:
    """落盘：写出的文件能被 load_jsonl 读回，条数与切分一致."""

    def make_bundle(self, count: int = 10) -> DatasetBundle:
        return DatasetBundle(
            examples=make_examples(count),
            report=FilterReport(total=count, kept=count, rejected=0, duplicates=0),
            source_kept={"seed": count},
        )

    def test_writes_train_and_eval(self, tmp_path):
        paths = dump_bundle(self.make_bundle(10), tmp_path / "out")
        assert set(paths) == {"train", "eval"}
        assert paths["train"].name == "train.jsonl"
        assert paths["eval"].parent.is_dir()
        train = load_jsonl(paths["train"])
        eval_set = load_jsonl(paths["eval"])
        assert len(train) == 8
        assert len(eval_set) == 2
        assert train[0].keys() == {"instruction", "input", "output"}
        assert {record["instruction"] for record in train} | {
            record["instruction"] for record in eval_set
        } == {f"问题 {index}" for index in range(10)}

    def test_returns_paths_for_existing_dir(self, tmp_path):
        out_dir = tmp_path / "deep" / "nested"
        paths = dump_bundle(self.make_bundle(5), out_dir)
        assert paths["train"] == Path(out_dir) / "train.jsonl"
        assert paths["train"].exists()
        assert paths["eval"].exists()

    def test_chat_format_round_trip(self, tmp_path):
        paths = dump_bundle(self.make_bundle(4), tmp_path / "chat", fmt=CHAT)
        records = load_jsonl(paths["train"])
        assert all("messages" in record for record in records)
        assert {m["role"] for m in records[0]["messages"]} == {"user", "assistant"}

    def test_seed_controls_split(self, tmp_path):
        first = dump_bundle(self.make_bundle(10), tmp_path / "a", seed=1)
        second = dump_bundle(self.make_bundle(10), tmp_path / "b", seed=1)
        assert load_jsonl(first["eval"]) == load_jsonl(second["eval"])

    def test_default_format_is_alpaca(self, tmp_path):
        paths = dump_bundle(self.make_bundle(3), tmp_path / "default")
        assert "instruction" in load_jsonl(paths["train"])[0]


class TestBuildDataset:
    """端到端：真实语料 → 数据集包."""

    def test_build_from_repository_data(self):
        bundle = build_dataset()
        assert bundle.examples
        assert bundle.report.kept == len(bundle.examples)
        assert bundle.source_kept["seed"] >= DEFAULT_SEED_MIN
        assert bundle.source_kept["eval/agent_tasks"] == 5
        assert bundle.source_kept["eval/redteam_cases"] == 16
