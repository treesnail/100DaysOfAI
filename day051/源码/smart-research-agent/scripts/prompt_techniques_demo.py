"""day036 编程题参考：四技巧输出对比 + PromptLibrary 版本管理演示.

运行方式（项目根目录）：
    python scripts/prompt_techniques_demo.py
全程离线：LLM 由 MockLLM 脚本化扮演，持久化写入系统临时目录。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smart_research_agent.agent.prompt_library import PromptLibrary, PromptTemplate
from smart_research_agent.agent.techniques import (
    chain_of_thought,
    few_shot,
    tree_of_thoughts,
    zero_shot,
)
from smart_research_agent.llm.mock import MockLLM

TASK = "把『人工智能正在改变科研工作流』翻译成英文"


def demo_techniques() -> None:
    """同一任务用四种技巧各跑一次（ToT 用 2 分支），断言提示词互不相同."""
    llm = MockLLM(
        responses=[
            "AI is transforming research workflows.",  # zero-shot
            "AI is transforming research workflows.",  # few-shot
            "逐词分析……AI is transforming research workflows.",  # CoT
            "分支1：直译……",  # ToT 分支 1
            "分支2：意译……",  # ToT 分支 2
            "综合两分支，最终答案：AI is transforming research workflows.",  # ToT 汇总
        ]
    )
    examples = [("把『你好』翻译成英文", "Hello"), ("把『谢谢』翻译成英文", "Thank you")]

    prompts = {
        "zero_shot": zero_shot(TASK),
        "few_shot": few_shot(TASK, examples),
        "cot": chain_of_thought(TASK),
    }
    for name, messages in prompts.items():
        print(f"[{name}] 回复: {llm.chat(messages)}")

    plan = tree_of_thoughts(TASK, n_branches=2)
    branch_outputs = [llm.chat(msgs) for msgs in plan.branch_messages]
    final = llm.chat(plan.synthesis_messages(branch_outputs))
    print(f"[tot] 分支输出: {branch_outputs}")
    print(f"[tot] 汇总回复: {final}")

    # 四技巧的首次提示词必须互不相同；ToT 调用数 = 分支数 + 1 次汇总
    first_prompts = [msgs[0].content for msgs in prompts.values()]
    assert len(set(first_prompts)) == 3
    assert len(llm.calls) == 3 + 2 + 1
    print(f"[断言] 四种技巧提示词互不相同，LLM 总调用 {len(llm.calls)} 次（ToT 占 3 次）")


def demo_library() -> None:
    """注册 -> 渲染 -> 历史查询 -> JSONL 持久化往返."""
    lib = PromptLibrary()
    lib.register(
        PromptTemplate(
            name="translate",
            template="把『{text}』翻译成{lang}",
            version="v1",
            changelog="初始版本：直接指令式",
        )
    )
    lib.register(
        PromptTemplate(
            name="translate",
            template="你是专业译者。请把『{text}』翻译成{lang}，只输出译文，禁止解释。",
            version="v2",
            changelog="补充角色定义与输出约束（角色/约束两要素）",
        )
    )

    rendered = lib.render("translate", text="你好", lang="英文")
    assert rendered == "你是专业译者。请把『你好』翻译成英文，只输出译文，禁止解释。"
    assert [t.version for t in lib.history("translate")] == ["v1", "v2"]
    assert lib.get("translate").version == "v2"  # 默认取最新
    print(f"[library] 渲染结果: {rendered}")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prompts.jsonl"
        lib.save_jsonl(path)
        loaded = PromptLibrary()
        assert loaded.load_jsonl(path) == 2
        assert loaded.get("translate", "v1").template == "把『{text}』翻译成{lang}"
        assert loaded.render("translate", text="你好", lang="英文") == rendered
    print("[library] JSONL 持久化往返一致，版本历史完整保留")


if __name__ == "__main__":
    demo_techniques()
    demo_library()
    print("全部断言通过")
