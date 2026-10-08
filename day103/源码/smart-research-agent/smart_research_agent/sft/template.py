"""SFT 监督文本渲染：把 ``TrainingExample`` 变成"带监督边界的一段文本"（M5-D2）.

SFT（监督微调）与预训练在**数据形态**上的唯一区别是：训练序列里有一段
是"提问"、有一段是"标准答案"，而 loss 只应该算在答案上。所以数据准备
的第一件事不是编码，而是**把边界标出来**：

    ┌──────────────── 前缀（system + user，不参与 loss）────────────────┐
    <|im_start|>system
    你是智研 AI 助手……<|im_end|>
    <|im_start|>user
    RAG 与微调的区别是什么？<|im_end|>
    <|im_start|>assistant
    检索增强生成（RAG）通过外部知识库……<|im_end|>
    └──────────────────── 监督区间（参与 loss）────────────────────┘

``prompt_chars`` / ``supervised_chars`` 这两个字符偏移就是本模块的全部产出。
为什么用**字符偏移**而不是"前缀字符串"：

1. 下游（``encoding.encode_supervised``）必须把前缀与监督区间**分别编码
   再拼接**，而不是编码整段文本再切片——因为 BPE 的合并可能跨越边界，
   先编码再切片会让"prompt 的最后一个 token"和"answer 的第一个 token"
   被错误地合并成一个属于两者的 token。字符偏移是唯一能表达"从哪里切开"
   且不依赖具体分词器的表示。
2. 字符数可以直接对照 day048 的 ``DatasetStats``（平均指令 26.8 字 /
   平均输出 88.1 字），**渲染是否截断了答案，一眼可查**。

三种模板都实现了同一件事，区别只是特殊标记的写法：
``CHATML``（Qwen 系）、``LLAMA3``（Llama 3 系）、``PLAIN``（无特殊标记的
兜底格式，用于把监督文本喂给不认 chat 模板的模型）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from smart_research_agent.finetune.schema import TrainingExample

#: ChatML 模板（Qwen / InternLM 等）：``<|im_start|>role\\ncontent<|im_end|>``
CHATML = "chatml"
#: Llama 3 模板：``<|start_header_id|>role<|end_header_id|>\\n\\ncontent<|eot_id|>``
LLAMA3 = "llama3"
#: 无特殊标记的兜底的指令格式（alpaca 风格），用于不认 chat 模板的模型
PLAIN = "plain"

#: 支持的模板全集；新增模板时同步修改 ``render_supervised`` 的分派
SUPPORTED_TEMPLATES: tuple[str, ...] = (CHATML, LLAMA3, PLAIN)

#: 缺省模板：ChatML 是当前开源指令模型里最通用的一种
DEFAULT_TEMPLATE = CHATML

#: 默认系统提示词：与 SmartResearch Agent 的定位一致（Agent 场景里可用
#: ``system`` 字段覆盖）。刻意写成常量而不是"让模型自由发挥"，因为
#: **SFT 的目标就是把这个人的设定内化进权重**，它必须稳定可复现。
DEFAULT_SYSTEM_PROMPT = (
    "你是智研 AI 助手，一个善于检索与推理的研究助理。"
    "回答要准确、具体，涉及步骤时按条列出。"
)


class SFTTemplateError(ValueError):
    """模板名不合法（继承 ``ValueError``，便于调用方按类型捕获）.

    与 day048 的 ``DatasetFormatError`` 同构：调用方关心的是"这份数据
    不能用于训练"，而不是底层拼接细节。
    """


@dataclass(frozen=True)
class RenderedSample:
    """一条渲染好的监督样本：文本 + 两个字符偏移.

    ``frozen=True`` 是刻意的：渲染结果是一份**不可变的数据快照**，
    编码阶段只读它、不改它；要换模板请重新渲染。

    不变式（``__post_init__`` 之外也应当被调用方信任）::

        text[prompt_chars : prompt_chars + supervised_chars] == 监督区间
        prompt_chars + supervised_chars == len(text)
    """

    text: str
    #: 前缀（system + user）的字符数；等于监督区间的起始下标
    prompt_chars: int
    #: 监督区间（answer + 轮次终止符）的字符数
    supervised_chars: int
    #: 使用的模板名（供日志与 API 自述）
    template: str

    def __post_init__(self) -> None:
        if self.prompt_chars < 0 or self.supervised_chars <= 0:
            raise SFTTemplateError(
                f"渲染结果的偏移不合法：prompt_chars={self.prompt_chars}, "
                f"supervised_chars={self.supervised_chars}"
            )
        if self.prompt_chars + self.supervised_chars != len(self.text):
            raise SFTTemplateError(
                "渲染结果的偏移之和与文本长度不一致："
                f"{self.prompt_chars} + {self.supervised_chars} != {len(self.text)}"
            )

    @property
    def prompt_text(self) -> str:
        """前缀文本（system + user，不参与 loss）."""
        return self.text[: self.prompt_chars]

    @property
    def supervised_text(self) -> str:
        """监督区间文本（answer + 轮次终止符，参与 loss）."""
        return self.text[self.prompt_chars : self.prompt_chars + self.supervised_chars]

    @property
    def total_chars(self) -> int:
        """整段文本的字符数（= 前缀 + 监督区间）."""
        return len(self.text)


def _chatml_turns(system: str | None, user: str, answer: str) -> tuple[str, str, str]:
    """ChatML 的三段拼接（前缀 / 答案 / 终止符）."""
    prefix = ""
    if system:
        prefix += f"<|im_start|>system\n{system}<|im_end|>\n"
    prefix += f"<|im_start|>user\n{user}<|im_end|>\n"
    prefix += "<|im_start|>assistant\n"
    return prefix, answer, "<|im_end|>\n"


def _llama3_turns(system: str | None, user: str, answer: str) -> tuple[str, str, str]:
    """Llama 3 的三段拼接（前缀 / 答案 / 终止符）.

    Llama 3 的 header 之后是**两个换行**（``\\n\\n``），这是该模板的
    固定约定；少一个换行会让模型看到与 SFT 时不同的前缀，属于典型的
    "训练/推理模板不一致"缺陷。
    """
    prefix = "<|begin_of_text|>"
    if system:
        prefix += (
            "<|start_header_id|>system<|end_header_id|>\n\n" f"{system}<|eot_id|>"
        )
    prefix += (
        "<|start_header_id|>user<|end_header_id|>\n\n" f"{user}<|eot_id|>"
    )
    prefix += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prefix, answer, "<|eot_id|>"


def _plain_turns(system: str | None, user: str, answer: str) -> tuple[str, str, str]:
    """无特殊标记的兜底格式（alpaca 风格 + 一个换行终止符）."""
    prefix = ""
    if system:
        prefix += f"{system}\n\n"
    prefix += f"### 指令\n{user}\n\n### 回答\n"
    return prefix, answer, "\n"


def render_supervised(
    example: TrainingExample,
    *,
    template: str = DEFAULT_TEMPLATE,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    include_turn_terminator: bool = True,
) -> RenderedSample:
    """把一条样本渲染成监督文本，并给出前缀 / 监督区间的字符偏移.

    ``system_prompt`` 的三态语义（与 day041 的 ``default_embedding()`` 同一种
    "显式 None vs 缺省值"纪律）：

    - 缺省（不传）：用 ``DEFAULT_SYSTEM_PROMPT``；
    - 传 ``None``：**不要 system 轮**（用于把"无 system"也作为一种训练条件）；
    - 传字符串：用该字符串。

    样本自带的 ``example.system`` 优先级最高——**样本级设定不该被全局
    缺省值盖掉**，否则"这条样本要带角色设定"的意图会丢失。
    """
    if template not in SUPPORTED_TEMPLATES:
        raise SFTTemplateError(
            f"不支持的模板 {template!r}，可选：{', '.join(SUPPORTED_TEMPLATES)}"
        )

    user_text = example.prompt_text
    if not user_text.strip():
        raise SFTTemplateError("样本的 prompt_text 为空，无法渲染监督文本")
    answer_text = example.output
    if not answer_text.strip():
        raise SFTTemplateError("样本的 output 为空，无法渲染监督文本（答案区间必须非空）")

    effective_system = example.system if example.system else system_prompt

    if template == CHATML:
        prefix, answer, terminator = _chatml_turns(effective_system, user_text, answer_text)
    elif template == LLAMA3:
        prefix, answer, terminator = _llama3_turns(effective_system, user_text, answer_text)
    else:
        prefix, answer, terminator = _plain_turns(effective_system, user_text, answer_text)

    # 轮次终止符是否计入监督区间：**应该计入**。模型需要学会"在哪里停下"，
    # 而 EOS/终止符正是这个信号；不计入会让模型学会"永远继续写下去"。
    supervised = answer + terminator if include_turn_terminator else answer
    return RenderedSample(
        text=prefix + supervised,
        prompt_chars=len(prefix),
        supervised_chars=len(supervised),
        template=template,
    )


def render_supervised_list(
    examples: list[TrainingExample],
    *,
    template: str = DEFAULT_TEMPLATE,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    include_turn_terminator: bool = True,
) -> list[RenderedSample]:
    """批量渲染（保持输入顺序）——训练集与评估集必须用同一套渲染参数."""
    return [
        render_supervised(
            example,
            template=template,
            system_prompt=system_prompt,
            include_turn_terminator=include_turn_terminator,
        )
        for example in examples
    ]


def load_training_examples(path: str | Path) -> list[TrainingExample]:
    """读取 day048 落盘的 ``train.jsonl`` / ``eval.jsonl`` 为样本列表.

    这与 ``finetune.schema.load_jsonl`` 的分工：本函数**多走一步** ``parse_example``，
    因此它校验的是"这些记录能否用于训练"，而不只是"这个文件能不能读"。
    坏记录直接抛错（离线固化数据集要严格）——这与 day048 采集器的宽松策略
    形成互补：**线上采集容错，离线固化严格**。
    """
    from smart_research_agent.finetune.schema import ALPACA, load_jsonl, parse_example

    return [parse_example(record, ALPACA) for record in load_jsonl(path)]
