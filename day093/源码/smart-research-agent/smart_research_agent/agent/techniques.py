"""四种经典 Prompt 技巧的可复用构造器（day036）.

每个构造器返回渲染好的 messages 列表，可直接喂给 BaseLLM.chat：

- zero_shot:        直接给任务，不给示例——依赖模型的预训练泛化能力；
- few_shot:         任务前附输入/输出示例——用示例"校准"格式与风格；
- chain_of_thought: 追加思维链引导语——把推理过程摊到生成过程里；
- tree_of_thoughts: 多分支探索 + 汇总裁决——把单链推理升级为搜索。

ToT 返回 ToTPlan：branch_messages 是 n 条相互独立的分支提示（同一任务、
不同探索视角），分支各自调用 LLM 后，用 synthesis_messages() 把各分支输出
拼成汇总提示，让模型裁决出最终答案。
"""

from __future__ import annotations

from dataclasses import dataclass

from smart_research_agent.llm.base import Message

#: CoT 引导语：显式要求模型把推理步骤写出来再给结论
COT_INSTRUCTION = "让我们一步一步思考，先写出推理过程，再给出最终答案。"

#: ToT 各分支的探索视角，分支数超出时循环使用
_BRANCH_ANGLES = [
    "从最常规、最稳妥的思路出发",
    "换一个完全不同的角度，尝试非常规的思路",
    "先找出问题中最容易出错的地方，围绕它展开",
    "先给出粗略估计，再逐步细化",
]


def zero_shot(task: str) -> list[Message]:
    """Zero-shot：只给任务本身，不提供任何示例."""
    return [Message(role="user", content=f"请完成以下任务：\n\n{task}")]


def few_shot(task: str, examples: list[tuple[str, str]]) -> list[Message]:
    """Few-shot：任务前附若干 (输入, 输出) 示例.

    示例的作用不是"教知识"，而是校准输出的格式、粒度与风格——
    模型会模仿示例的呈现方式，即使示例内容本身很简单。
    """
    if not examples:
        raise ValueError("few_shot 至少需要一个示例")
    parts = ["请参照以下示例完成最后的任务。", ""]
    for i, (inp, out) in enumerate(examples, 1):
        parts.append(f"示例 {i}：")
        parts.append(f"输入：{inp}")
        parts.append(f"输出：{out}")
        parts.append("")
    parts.append(f"任务：{task}")
    parts.append("输出：")
    return [Message(role="user", content="\n".join(parts))]


def chain_of_thought(task: str) -> list[Message]:
    """CoT：在 zero-shot 任务后追加思维链引导语.

    原理：自回归模型生成每个 token 都以前面所有 token 为条件。
    让模型先写推理步骤，等于把中间计算结果"落到纸面"，后续 token
    可以直接利用这些中间结果，而不是在单次前向传播里心算答案。
    """
    return [
        Message(role="user", content=f"请完成以下任务：\n\n{task}\n\n{COT_INSTRUCTION}")
    ]


@dataclass
class ToTPlan:
    """Tree-of-Thoughts 的提示计划：n 条分支提示 + 汇总提示生成器."""

    task: str
    branch_messages: list[list[Message]]

    @property
    def n_branches(self) -> int:
        return len(self.branch_messages)

    def synthesis_messages(self, branch_outputs: list[str]) -> list[Message]:
        """把各分支输出拼成汇总提示，让模型评估并给出最终答案."""
        if len(branch_outputs) != self.n_branches:
            raise ValueError(
                f"需要 {self.n_branches} 条分支输出，实际收到 {len(branch_outputs)} 条"
            )
        parts = [
            f"原始任务：{self.task}",
            "",
            "以下是多位解题者从不同思路给出的候选解答：",
        ]
        for i, output in enumerate(branch_outputs, 1):
            parts.append(f"\n--- 候选 {i} ---")
            parts.append(output)
        parts.extend(
            [
                "",
                "请完成：",
                "1. 评估每个候选解答的推理是否严密、结论是否正确",
                "2. 综合所有候选（可取最优者，也可融合多个候选的优点），给出最终答案",
            ]
        )
        return [Message(role="user", content="\n".join(parts))]


def tree_of_thoughts(task: str, n_branches: int = 3) -> ToTPlan:
    """ToT：把一个推理任务拆成 n_branches 条独立分支 + 一次汇总.

    思想来源：单链 CoT 一旦前面想错，错误会沿链条放大；ToT 把推理
    当作搜索问题——并行展开多条思路（分支），再由汇总步评估、比较、
    择优，用更多计算换取对单点失败的鲁棒性。
    """
    if n_branches < 1:
        raise ValueError("n_branches 必须 >= 1")
    branches = []
    for i in range(n_branches):
        angle = _BRANCH_ANGLES[i % len(_BRANCH_ANGLES)]
        branches.append(
            [
                Message(
                    role="user",
                    content=(
                        f"请完成以下任务：\n\n{task}\n\n"
                        f"要求：{angle}。先写出推理过程，再给出最终答案。"
                    ),
                )
            ]
        )
    return ToTPlan(task=task, branch_messages=branches)
