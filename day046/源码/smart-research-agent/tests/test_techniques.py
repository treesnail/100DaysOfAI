"""四种 Prompt 技巧构造器的单元测试（day036）."""

from __future__ import annotations

import pytest

from smart_research_agent.agent.techniques import (
    COT_INSTRUCTION,
    chain_of_thought,
    few_shot,
    tree_of_thoughts,
    zero_shot,
)
from smart_research_agent.llm.mock import MockLLM

TASK = "一个水池有进水管和出水管，单开进水管 4 小时注满，单开出水管 6 小时放空。两管同开，几小时注满？"
EXAMPLES = [
    ("1 + 1 = ?", "2"),
    ("3 * 4 = ?", "12"),
]


class TestZeroShot:
    def test_returns_single_user_message(self):
        messages = zero_shot(TASK)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert TASK in messages[0].content

    def test_contains_no_examples(self):
        content = zero_shot(TASK)[0].content
        assert "示例" not in content


class TestFewShot:
    def test_examples_and_task_all_present(self):
        content = few_shot(TASK, EXAMPLES)[0].content
        for inp, out in EXAMPLES:
            assert inp in content
            assert out in content
        assert TASK in content

    def test_example_ordering_preserved(self):
        content = few_shot(TASK, EXAMPLES)[0].content
        assert content.index("1 + 1") < content.index("3 * 4") < content.index(TASK)

    def test_empty_examples_rejected(self):
        with pytest.raises(ValueError, match="至少需要一个示例"):
            few_shot(TASK, [])

    def test_single_user_message(self):
        assert len(few_shot(TASK, EXAMPLES)) == 1


class TestChainOfThought:
    def test_appends_cot_instruction(self):
        content = chain_of_thought(TASK)[0].content
        assert TASK in content
        assert COT_INSTRUCTION in content
        assert content.index(TASK) < content.index(COT_INSTRUCTION)

    def test_differs_from_zero_shot_only_by_instruction(self):
        """CoT = zero-shot 任务 + 思维链引导语，其余结构一致."""
        zero = zero_shot(TASK)[0].content
        cot = chain_of_thought(TASK)[0].content
        assert cot.startswith(zero)


class TestTreeOfThoughts:
    def test_generates_n_distinct_branch_prompts(self):
        plan = tree_of_thoughts(TASK, n_branches=3)
        assert plan.n_branches == 3
        contents = [msgs[0].content for msgs in plan.branch_messages]
        assert len(set(contents)) == 3  # 三条分支提示互不相同
        for content in contents:
            assert TASK in content

    def test_invalid_branch_count_rejected(self):
        with pytest.raises(ValueError):
            tree_of_thoughts(TASK, n_branches=0)

    def test_synthesis_messages_collect_branch_outputs(self):
        plan = tree_of_thoughts(TASK, n_branches=2)
        messages = plan.synthesis_messages(["思路A：12 小时", "思路B：12 小时"])
        assert len(messages) == 1
        content = messages[0].content
        assert TASK in content
        assert "思路A" in content and "思路B" in content

    def test_synthesis_requires_exact_branch_count(self):
        plan = tree_of_thoughts(TASK, n_branches=3)
        with pytest.raises(ValueError, match="3 条分支输出"):
            plan.synthesis_messages(["只有一条"])


class TestTechniquesWithMockLLM:
    """同一任务喂给四种技巧，MockLLM 演示输出差异."""

    def test_all_techniques_produce_valid_messages(self):
        llm = MockLLM(responses=["12 小时"] * 10)
        assert llm.chat(zero_shot(TASK)) == "12 小时"
        assert llm.chat(few_shot(TASK, EXAMPLES)) == "12 小时"
        assert llm.chat(chain_of_thought(TASK)) == "12 小时"

        plan = tree_of_thoughts(TASK, n_branches=2)
        outputs = [llm.chat(msgs) for msgs in plan.branch_messages]
        final = llm.chat(plan.synthesis_messages(outputs))
        assert final == "12 小时"
        # 四种技巧发出的 prompt 互不相同
        prompts = [call[0].content for call in llm.calls]
        assert len(set(prompts)) == len(prompts)

    def test_tot_consumes_2n_plus_calls(self):
        """ToT 的 LLM 调用数 = 分支数 + 1 次汇总，比单链技巧贵 n+1 倍."""
        llm = MockLLM(responses=["x"] * 10)
        plan = tree_of_thoughts(TASK, n_branches=3)
        outputs = [llm.chat(msgs) for msgs in plan.branch_messages]
        llm.chat(plan.synthesis_messages(outputs))
        assert len(llm.calls) == 4
