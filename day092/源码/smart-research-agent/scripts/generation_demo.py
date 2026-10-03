#!/usr/bin/env python
"""day069 演示脚本：RAG 生成器与 Prompt（M6-D8）——把答案折成一份可核对的账.

九节，全部**离线、确定性、零网络、零 API Key**，只依赖本包与标准库：

```text
1. v1 与 v2 两份模板并排对照     同一份上下文渲染出的提示词：字符数、小节标题、编号条目
2. 正常生成                     答案 + 引用表 + grounding 六个数字 + 提示词片段（片段进了提示词）
3. 引用核对三态                 全部合法 / 含越界 [9]（幻觉引用）/ 一条引用都没有
4. coverage 与 grounded 扫描表   编号集合从 0 条到全部被引用，逐行看两个数字与回退
5. 三种拒答句式 + 一条反例        命中 DECLINE_MARKERS 与"含'资料'但不命中句式"
6. 空回复与 llm_error            两条回退路径的 fallback_reason / answer / llm_called
7. require_citation 四格矩阵      开 / 关 × 有 / 无合法引用
8. 空上下文早退                  llm.calls == 0（一次都没调）与 prompt_text == ""
9. pipeline.RagPipeline 集成对照   answer() 的 check / fallback_reason 两个新字段与旧字段并排
```

## 这一课要证明的那句话

```text
调用成功、答案通顺、引用格式像模像样   ← 这三件事都不能说明答案被核对过
引用编号全部落在那次提示词里           ← 只有这一条能说明
```

``RAGGenerator`` 因此不做"事实核查"，它做的是**编号层面的对账**：把答案里的
``[n]`` 与那次提示词的编号表放在一起，折成三个数字（有效引用 / 幻觉引用 /
未被引用的片段）与一个判定（``grounded = valid 非空 且 invalid 为空``）。
"这段话是否忠实于片段"属于语义判断，不在这一层（见 ``retrieval/types.py`` 的边界
声明）；"这个答案好不好"属于 day071 的 RAG 评估。这两句话决定了本脚本不去碰
任何真实模型，也决定了它的每一行输出都只是**编号的算术**。

## 为什么用假 LLM 而不是真模型

要证明的两条断言都**与模型无关**：一是空上下文时一次都不调（护栏的证据是
``llm.calls`` 这个计数器），二是引用核对的结论只由"答案里的编号"与"提示词里的
编号"两串字决定。因此本脚本注入 ``MockLLM``（按脚本回复、并记下每次调用）与一个
"一旦被调用就抛异常"的假 ``BaseLLM``——**它们没被调用过，就是护栏生效的证据**
（与 ``docs/retrieval.md`` 第 7 节给出的测试写法同源）。

## 样本与输出

``tests/`` 里也有一份小库，但**脚本不 import tests**：脚本要能被复制走单独运行，
一旦依赖测试目录，复制到别处就跑不起来。这里自带 6 条记录（向量分量只用
0 / 0.6 / 0.8 / 1.0，都已是单位向量，因此余弦分数**可以手算**），编码器是按文本
查表的确定性替身——"哪条最近"由人给定，不是猜出来的。

结果同时打印到 stdout 并写入 ``outputs/generation_demo.txt``；运行（cwd 为仓库根）::

    cd day069/源码/smart-research-agent
    PYTHONPATH=. python scripts/generation_demo.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

# 本脚本要能直接 ``python scripts/generation_demo.py`` 跑起来（cwd 是仓库根）。
# 以脚本方式启动时 ``sys.path[0]`` 是 ``scripts/``，仓库根不在其中，
# 而本快照的包**没有 pip 安装**——因此这里显式把仓库根塞进 sys.path。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_research_agent.config import settings  # noqa: E402
from smart_research_agent.llm.base import BaseLLM, Message  # noqa: E402
from smart_research_agent.llm.embedding import EmbeddingProvider  # noqa: E402
from smart_research_agent.llm.mock import MockLLM  # noqa: E402
from smart_research_agent.retrieval import (  # noqa: E402
    CURRENT_PROMPT_VERSION,
    CUSTOM_PROMPT_VERSION,
    DECLINE_MARKERS,
    DEFAULT_PROMPT_VERSION,
    FALLBACK_NO_CONTEXT,
    FALLBACK_REASON_DESCRIPTIONS,
    FALLBACK_REASONS,
    GENERATION_OVERRIDE_KEYS,
    PROMPT_CHANGELOG,
    PROMPT_VERSIONS,
    RAG_PROMPT_V1,
    RAG_PROMPT_V2,
    Citation,
    PackedContext,
    RagAnswer,
    RAGGenerator,
    RagPipeline,
    RetrievalQuery,
    Retriever,
    build_retriever,
    pack_context,
    prompt_template_of,
)
from smart_research_agent.retrieval.generation import REQUIRED_PROMPT_FIELDS  # noqa: E402
from smart_research_agent.retrieval.pipeline import OVERRIDE_KEYS  # noqa: E402
from smart_research_agent.vectorstore import FlatVectorStore  # noqa: E402
from smart_research_agent.vectorstore.types import make_record  # noqa: E402

#: 本层的边界（原文供教程引用，见 ``retrieval/generation.py`` 的模块 docstring）.
LAYER_BOUNDARY = (
    "它做的事：答案里的 [n] 与那次提示词里的编号一一对账（编号层面）；"
    "它不做的事：判断这句话是否忠实于片段（语义层面）。"
)

#: 输出目录与文件名（``outputs/`` 在仓库 ``.gitignore`` 里，不污染仓库）.
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_NAME = "generation_demo.txt"

#: 样本向量维度。取 8、分量只用 0 / 0.6 / 0.8 / 1.0，于是余弦分数可以手算：
#: ``(0.8, 0.6) · (1, 0) = 0.8``、``(0.8, 0.6) · (0.8, 0.6) = 1.0`` 这类。
DEMO_DIMENSION = 8

#: 两份文档（day062 的 ``parent_doc_id``；``retrieval_max_per_doc`` 缺省 0 = 不限）.
DOC_MANUAL = "doc-manual"
DOC_GEN = "doc-gen"

#: 六个样本记录：``(id, 向量, 正文, 文档, 标题路径)``.
#:
#: 三条 Generation 相关的记录（k-02 / k-03 / k-04）刻意排在名次前列：这样第 2 节的
#: 提示词片段里读到的正是"生成器在讲什么"，而不是随便一段向量库介绍。
#: 第四条轴上的 k-05 与 k-06 给出两个较小的、彼此不同的分数——第 4 节的扫描表因此
#: 有一段"部分引用"的素材。
DEMO_SPECS: tuple[tuple[Any, ...], ...] = (
    ("k-01", (1.0, 0.0, 0, 0, 0, 0, 0, 0),
     "向量库回答'给一个向量谁最近'；检索器回答'给一句话取回哪几条'。", DOC_MANUAL,
     "检索手册 > 分工"),
    ("k-02", (0.8, 0.6, 0, 0, 0, 0, 0, 0),
     "生成器把答案里的 [n] 与那次提示词的编号对账，折成有效 / 幻觉 / 未引用三个数字。",
     DOC_GEN, "生成器 > 引用溯源"),
    ("k-03", (0.6, 0.8, 0, 0, 0, 0, 0, 0),
     "检索为空时一次 LLM 都不调，答案取兜底答复并给出三条出路。", DOC_MANUAL,
     "生成器 > 护栏"),
    ("k-04", (0.0, 1.0, 0, 0, 0, 0, 0, 0),
     "提示词 v2 按资料片段 / 问题 / 约束 / 输出格式四段式重写，并新增固定拒答句式。",
     DOC_GEN, "生成器 > 提示词"),
    ("k-05", (0.0, 0.6, 0.8, 0, 0, 0, 0, 0),
     "覆盖率 = 有效引用 / 给出去的片段数，保留四位小数。", DOC_GEN, "生成器 > 接地"),
    ("k-06", (0.0, 0.0, 1.0, 0, 0, 0, 0, 0),
     "幻觉引用指答案里的编号不在那次提示词里，它的条数要进报告。", DOC_GEN,
     "生成器 > 接地"),
)

#: 六个记录 id（顺序 = 插入顺序）：用它做"整库都返回"的 top_k。
RECORD_IDS: tuple[str, ...] = tuple(str(spec[0]) for spec in DEMO_SPECS)

#: id → 正文（核对"片段真的进了提示词"时要用它）.
TEXT_BY_ID: dict[str, str] = {str(spec[0]): str(spec[2]) for spec in DEMO_SPECS}

#: 演示问题（它同时是检索查询与送进提示词的那个问题）.
DEMO_QUESTION = "生成器怎么核对答案里的引用"

#: 查询文本 → 查询向量。这张表是**断言的一部分**：题目与"谁最近"由人给定。
QUERY_VECTORS: dict[str, tuple[float, ...]] = {
    DEMO_QUESTION: (0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}

#: 兜底查询向量（查询文本不在表里时用它）.
DEFAULT_QUERY_VECTOR: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

#: 正常答案：先结论后依据，且引用了两条真在提示词里的编号。
ANSWER_OK = (
    "生成器做的是编号层面的对账：把答案里的 [n] 与那次提示词的编号一一对上 [1][2]。\n"
    "依据 [1]：它把答案折成有效 / 幻觉 / 未引用三个数字。\n"
    "依据 [2]：检索为空时一次 LLM 都不调，答案取兜底答复。"
)

#: 含幻觉引用的答案：``[9]`` 越界（那次只给了 6 条片段）。
ANSWER_HALLUCINATED = "结论：生成器会核对引用 [1]，并且总会追溯到第 9 条片段 [9]。"

#: 一条引用都没有的答案（"没有引用"与"引用了不存在的编号"是两件事）。
ANSWER_NO_CITATION = "结论：生成器会对答案做编号层面的对账，但它这次忘了标注出处。"


def title(text: str) -> None:
    """打印一节标题."""
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def short(text: str, width: int = 42) -> str:
    """一行摘要（超长截断加省略号）——正文只用来核对，不占满屏幕."""
    return text if len(text) <= width else text[: width - 1] + "…"


def numbered_items(template: str) -> list[tuple[str, int]]:
    """模板里每个 ``## 小节`` 下的编号条目数（v1 没有小节标题，整段算一节）.

    v1 的四条与 v2 的六条约束都是形如 ``1. …`` 的行，因此数它们不需要理解语义；
    把它写成函数而不是硬编码"4 条 / 6 条"，是为了让第 1 节打印出来的数字
    **永远等于模板里真实的行数**。
    """
    sections: dict[str, int] = {}
    current = "（无小节标题）"
    sections[current] = 0
    for line in template.splitlines():
        if line.startswith("## "):
            current = line.strip()[3:]
            sections.setdefault(current, 0)
            continue
        if re.match(r"^\d+\.\s", line):
            sections[current] = sections.get(current, 0) + 1
    return [(name, count) for name, count in sections.items() if count]


# --------------------------------------------------------------------------- #
# 样本构造（离线小库 / 确定性编码器 / 检索器）
# --------------------------------------------------------------------------- #


class TableEmbedding(EmbeddingProvider):
    """按文本查表的确定性编码器（本脚本的向量表，理由见模块 docstring）."""

    def __init__(
        self,
        *,
        table: dict[str, tuple[float, ...]],
        dimension: int,
        default: tuple[float, ...],
    ) -> None:
        self._table = dict(table)
        self._dimension = dimension
        self._default = tuple(default)
        self.calls: list[str] = []

    @property
    def dimension(self) -> int:
        """向量维度（与库的维度必须一致，否则检索器在编码那一步就拒）."""
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """查表；表里没有的文本用兜底向量（并记一次调用）."""
        self.calls.append(text)
        return list(self._table.get(text, self._default))


def demo_records() -> list[Any]:
    """六条 ``VectorRecord``（元数据按 day062 的口径：``parent_doc_id`` / ``source`` / …）."""
    return [
        make_record(
            record_id=str(spec[0]),
            vector=tuple(float(value) for value in spec[1]),
            text=str(spec[2]),
            metadata={
                "parent_doc_id": str(spec[3]),
                "source": f"docs/{spec[3]}.md",
                "heading_path": str(spec[4]),
            },
            metric="cosine",
        )
        for spec in DEMO_SPECS
    ]


def demo_store() -> FlatVectorStore:
    """装好六条样本记录的 flat 库（维度与编码器必须一致）."""
    store = FlatVectorStore(metric="cosine", dimension=DEMO_DIMENSION)
    store.upsert(demo_records())
    return store


def empty_store() -> FlatVectorStore:
    """一个**空**库（维度固定为 8，与 ``demo_store`` 同规格）；空库是合法状态."""
    return FlatVectorStore(metric="cosine", dimension=DEMO_DIMENSION)


def demo_embedding() -> TableEmbedding:
    """按 ``QUERY_VECTORS`` 查表的确定性编码器（8 维，与样本库同规格）."""
    return TableEmbedding(
        table=QUERY_VECTORS,
        dimension=DEMO_DIMENSION,
        default=DEFAULT_QUERY_VECTOR,
    )


def demo_retriever(store: FlatVectorStore | None = None, **overrides: Any) -> Retriever:
    """装配一个检索器（``build_retriever``；``**overrides`` 逐个覆盖）."""
    return build_retriever(store or demo_store(), demo_embedding(), **overrides)


def demo_hits() -> list[Any]:
    """检索器给出的完整命中序列（分数降序 + 同分按 id 升序）."""
    result = demo_retriever().retrieve(
        RetrievalQuery(text=DEMO_QUESTION, top_k=len(RECORD_IDS))
    )
    return list(result.hits)


def demo_context(*, take: int | None = None, **overrides: Any) -> PackedContext:
    """把命中打包成带编号、有预算的上下文（默认整库六条都进包）.

    ``take`` 用来取前 N 条命中——第 4 节的扫描表要一份"片段数固定"的上下文，
    它因此不依赖检索顺序之外的东西。
    """
    hits = demo_hits()
    if take is not None:
        hits = hits[:take]
    return pack_context(hits, **overrides)


def empty_context() -> PackedContext:
    """空上下文：没有命中可打包时 ``pack_context([])`` 的产物（不是异常）."""
    return pack_context([])


def generator_with(answer: str, **overrides: Any) -> tuple[RAGGenerator, MockLLM]:
    """一个"按脚本回一句话"的生成器：返回 ``(生成器, 那个假 LLM)``.

    返回 LLM 是为了断言 ``llm.calls``——"这次到底调了几次"与"模型被问到了什么"
    都必须能被读出来，而不是只能相信代码。
    """
    llm = MockLLM(responses=[answer])
    return RAGGenerator(llm, model="mock-llm", **overrides), llm


class BoomLLM(BaseLLM):
    """一旦被调用就抛异常的假 LLM（模拟超时 / 鉴权 / 网络 / 额度）.

    它是 ``llm_error`` 那条路的唯一构造方式：真实提供方的这几类失败都是
    **可预期**的，因此必须变成一份带原因的结果，而不是把链路打断。
    """

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """记下这次调用，然后抛异常（证明"调用抛异常"是一条合法路径）."""
        self.calls.append(messages)
        raise RuntimeError("模拟超时：提供方在 30s 内没有返回")


def grounding_row(check: Any) -> str:
    """把一份核对报告压成一行（六个数字 + 接地判定）——第 4 节的扫描表用它."""
    return (
        f"给了 {check.given} | cited {list(check.cited)} | valid {list(check.valid)} | "
        f"invalid {list(check.invalid)} | unused {list(check.unused)} | "
        f"覆盖 {check.coverage:.4f} | 接地 {check.grounded}"
    )


# --------------------------------------------------------------------------- #
# 第 1 节：v1 与 v2 两份模板并排对照
# --------------------------------------------------------------------------- #


def section_1_prompt_versions() -> None:
    """同一份上下文渲染出的两份提示词：字符数、小节标题、编号条目、变更日志."""
    title("第 1 节：v1 与 v2 两份模板并排对照（同一份上下文渲染）")
    print(f"边界（本层的原文）：{LAYER_BOUNDARY}")
    print(f"受管版本（**封闭清单**）PROMPT_VERSIONS = {PROMPT_VERSIONS}")
    print(f"当前生效 CURRENT_PROMPT_VERSION = {CURRENT_PROMPT_VERSION!r}、"
          f"缺省 DEFAULT_PROMPT_VERSION = {DEFAULT_PROMPT_VERSION!r}、"
          f"注入模板的版本号 CUSTOM_PROMPT_VERSION = {CUSTOM_PROMPT_VERSION!r}")
    print(f"占位符清单 REQUIRED_PROMPT_FIELDS = {REQUIRED_PROMPT_FIELDS}"
          "（构造期用 string.Formatter().parse 校验，缺任何一个报 ContextError）")
    print(f"settings.retrieval_prompt_version = {settings.retrieval_prompt_version!r}"
          "（缺省版本就写在这里，与 DEFAULT_PROMPT_VERSION 对齐）")
    print(f"变更日志 PROMPT_CHANGELOG 覆盖 {len(PROMPT_CHANGELOG)} 个版本：")
    for version in PROMPT_VERSIONS:
        print(f"    {version}：{PROMPT_CHANGELOG[version]}")
    print("版本 → 模板（prompt_template_of 取的就是这两份，清单外的版本号当场报错）：")
    print(f"    {PROMPT_VERSIONS[0]} → RAG_PROMPT_V1（{len(RAG_PROMPT_V1)} 字）")
    print(f"    {PROMPT_VERSIONS[1]} → RAG_PROMPT_V2（{len(RAG_PROMPT_V2)} 字）")

    context = demo_context()
    print(f"\n同一份上下文：{context.count} 条片段、{context.char_count} 字；"
          f"问题 {DEMO_QUESTION!r}")

    rendered: dict[str, str] = {}
    for version in PROMPT_VERSIONS:
        template = prompt_template_of(version)
        text = template.format(context=context.text, question=DEMO_QUESTION)
        rendered[version] = text
        headers = [line.strip() for line in text.splitlines() if line.startswith("## ")]
        items = numbered_items(template)
        print(f"\n  --- {version}：模板 {len(template)} 字 → 渲染后 {len(text)} 字；"
              f"小节标题 {len(headers)} 个：{headers}")
        print(f"      编号条目（按小节）：{items}；"
              f"合计 {sum(count for _name, count in items)} 条")
        print("      前 7 行：")
        for line in text.splitlines()[:7]:
            print(f"        | {line}")

    v1_len, v2_len = len(rendered[PROMPT_VERSIONS[0]]), len(rendered[PROMPT_VERSIONS[1]])
    print(f"\n  字符数差异：v1 = {v1_len}、v2 = {v2_len}、"
          f"差 = {v2_len - v1_len}（v2 / v1 = {v2_len / v1_len:.4f} 倍）")
    print(f"  两份模板逐字相同：{rendered[PROMPT_VERSIONS[0]] == rendered[PROMPT_VERSIONS[1]]}"
          "（v2 是一次重写，不是 v1 的复制）")
    print(f"  DECLINE_MARKERS（拒答句式的**封闭清单**，{len(DECLINE_MARKERS)} 条）："
          f"{list(DECLINE_MARKERS)}")
    print("  读法：v1 把'资料 / 问题 / 要求'混在一段里，模型只能靠位置猜；v2 把四件事"
          "\n  各自给了小节标题，并把约束从 4 条扩到 6 条。多出来的两条不是修辞："
          "\n  '固定拒答句式'让'模型自己说没有'变成可检测（DECLINE_MARKERS），"
          "\n  '每条结论后至少一个引用'让覆盖率这两笔账才有意义。")


# --------------------------------------------------------------------------- #
# 第 2 节：正常生成
# --------------------------------------------------------------------------- #


def section_2_normal() -> None:
    """答案 + 引用表 + grounding 六个数字 + 提示词片段（证明片段真的进了提示词）."""
    title("第 2 节：正常生成（答案 + 引用表 + grounding 六个数字 + 提示词片段）")
    context = demo_context()
    generator, llm = generator_with(ANSWER_OK)
    print(f"生成器：prompt_version={generator.prompt_version!r}、"
          f"temperature={generator.temperature}、max_tokens={generator.max_tokens}、"
          f"require_citation={generator.require_citation}")
    print(f"describe() = {generator.describe()}")
    print(f"上下文：{context.summary_line()}")

    result = generator.generate(DEMO_QUESTION, context)
    print(f"\n  summary_line()：{result.summary_line()}")
    print(f"  fallback_reason={result.fallback_reason!r}、llm_called={result.llm_called}、"
          f"is_fallback={result.is_fallback}、grounded={result.grounded}")
    print(f"  假 LLM 被调用了 {len(llm.calls)} 次；"
          f"它的入参是 {len(llm.calls[0])} 条消息、第一条 role={llm.calls[0][0].role!r}")
    print("\n  答案：")
    for line in result.answer.splitlines():
        print(f"    {line}")

    print("\n  引用表（那次提示词里的编号 → 命中）：")
    for citation in result.citations:
        print(f"    {citation.summary_line()}")
    first_citation: Citation = result.citations[0]
    print(f"  第一条引用的编号与记录：marker={first_citation.marker}、"
          f"record_id={first_citation.record_id!r}、source={first_citation.source!r}、"
          f"heading_path={first_citation.heading_path!r}")

    check = result.check
    print("\n  grounding 六个数字：")
    print(f"    given（给出去的片段数） = {check.given}")
    print(f"    cited（答案里出现的编号）= {list(check.cited)}")
    print(f"    valid（落在提示词里）    = {list(check.valid)}")
    print(f"    invalid（落不到提示词里）= {list(check.invalid)}  ← 幻觉引用")
    print(f"    unused（给了没引用）     = {list(check.unused)}")
    print(f"    coverage（valid / given）= {check.coverage:.4f}；"
          f"grounded = {check.grounded}")
    print(f"  逐编号检查行（{len(check.checks)} 条）：")
    for row in check.checks:
        print(f"    {row.summary_line()}")

    prompt_text = result.prompt_text
    print(f"\n  提示词：{len(prompt_text)} 字（to_dict()['prompt_chars'] = "
          f"{result.to_dict()['prompt_chars']}）")
    first_block = next(line for line in prompt_text.splitlines() if line.startswith("[1]"))
    fragment = context.used_hits[0].text
    print(f"    第一块的表头：{first_block}")
    print(f"    该片段的正文前 24 字：{short(fragment, 24)}")
    print(f"    {'该片段整段都在提示词里' if fragment in prompt_text else '片段不在提示词里'}"
          f"：{fragment in prompt_text}")
    print("    '## 资料片段'一节的内容（前 5 行）：")
    lines = prompt_text.splitlines()
    start = lines.index("## 资料片段") + 1
    for line in lines[start:start + 5]:
        print(f"      | {short(line, 66)}")

    print("\n  explain()（四段：怎么问 / 能不能核对 / 为什么回退 / 注记）：")
    for line in result.explain():
        print(f"    {line}")

    print(f"\n  to_dict() 的键（{len(result.to_dict())} 个，含 prompt_text）：")
    print(f"    {sorted(result.to_dict())}")
    without_prompt = sorted(result.to_dict(include_prompt=False))
    print("    include_prompt=False 时少一个键："
          f"{without_prompt == sorted(set(result.to_dict()) - {'prompt_text'})}")
    print(f"  to_summary() 的键（{len(result.to_summary())} 个，**刻意不含 prompt_text**）：")
    print(f"    {sorted(result.to_summary())}")


# --------------------------------------------------------------------------- #
# 第 3 节：引用核对三态
# --------------------------------------------------------------------------- #


def section_3_three_states() -> None:
    """全部合法 / 含越界 [9]（幻觉引用）/ 一条引用都没有——三组六个数字与 checks."""
    title("第 3 节：引用核对三态（有效引用 / 幻觉引用 / 没有引用）")
    context = demo_context()
    print(f"上下文固定：{context.count} 条片段，编号 [1]~[{context.count}]"
          f"（record_id：{[c.record_id for c in context.citations]}）")
    cases: tuple[tuple[str, str], ...] = (
        ("① 全部合法", ANSWER_OK),
        ("② 含越界 [9]", ANSWER_HALLUCINATED),
        ("③ 一条引用都没有", ANSWER_NO_CITATION),
    )
    for label, answer in cases:
        generator, _llm = generator_with(answer)
        result = generator.generate(DEMO_QUESTION, context)
        check = result.check
        print(f"\n  {label}：fallback_reason={result.fallback_reason!r}、"
              f"grounded={result.grounded}")
        print(f"    答案：{short(result.answer, 60)}")
        print(f"    {grounding_row(check)}")
        print(f"    checks（{len(check.checks)} 条）：")
        for row in check.checks:
            print(f"      {row.summary_line()}")
        print("    explain() 里点名的两行：")
        for line in check.explain():
            if "幻觉" in line or "未引用" in line or "覆盖率" in line or "没有出现" in line:
                print(f"      {line}")

    print(
        "\n  三组要一起看：①②的区别只在 invalid（幻觉引用），②③的区别只在 cited 是否为空。"
        "\n  第一组的 coverage 是部分覆盖（引了 2 条、给了 6 条），后两组分别是'引了不存在的'"
        "\n  与'什么都没引'——处置动作因此各不相同（换模型 / 改提示词）。"
    )


# --------------------------------------------------------------------------- #
# 第 4 节：coverage 与 grounded 的扫描表
# --------------------------------------------------------------------------- #


def section_4_coverage_scan() -> None:
    """编号集合从 0 条到全部被引用：两个数字（coverage / grounded）怎么动."""
    title("第 4 节：coverage 与 grounded 的扫描表（编号集合从 0 条到全部被引用）")
    context = demo_context(take=4)
    total = context.count
    print(f"上下文固定为 {total} 条片段（take=4），编号 [1]~[{total}]；"
          "答案里的编号集合逐个变大：")
    variants: tuple[tuple[str, str], ...] = (
        ("（无）", "这是一个没有标注任何出处的结论。"),
        ("[1]", "结论 [1]。"),
        ("[1][2]", "结论 [1][2]。"),
        ("[1][2][3]", "结论 [1][2][3]。"),
        ("[1][2][3][4]", "结论 [1][2][3][4]。"),
        ("[1][9]", "结论 [1][9]（9 越界）。"),
        ("[0]", "结论 [0]（0 不是编号）。"),
    )
    print(f"\n  {'引用集合':<12} | {'cited':<12} | {'valid':<10} | {'invalid':<8} | "
          f"{'unused':<14} | {'coverage':>8} | grounded")
    for label, answer in variants:
        generator, _llm = generator_with(answer)
        check = generator.generate(DEMO_QUESTION, context).check
        print(f"  {label:<12} | {str(list(check.cited)):<12} | {str(list(check.valid)):<10} | "
              f"{str(list(check.invalid)):<8} | {str(list(check.unused)):<14} | "
              f"{check.coverage:>8.4f} | {check.grounded}")
    print(
        "\n  读法：coverage = len(valid) / len(known)（**4 位小数**，"
        "口径与 PackedContext.fill_ratio 相同），"
        "\n  分母是给出去的片段数（这里恒为 4）。grounded = 'valid 非空 且 invalid 为空'。"
        "\n  两处要看清："
        f"\n    1) '（无）'与'[0]'两行的 coverage 都是 {0.0:.4f}、都不接地——"
        "\n       [0] 不是编号（编号从 1 起），它只进报告自己的注记，不进 cited/valid/invalid；"
        "\n    2) '[1][9]' 的 valid 非空（[1]）但 invalid 也非空（[9]）→ 接地为 False。"
        "\n       它是本表里唯一一行'看起来引了东西、但不接地'的样本。"
    )


# --------------------------------------------------------------------------- #
# 第 5 节：三种拒答句式 + 一条反例
# --------------------------------------------------------------------------- #


def section_5_declines() -> None:
    """三种固定拒答句式各跑一遍，再加一条"含'资料'但不命中句式"的反例."""
    title("第 5 节：三种拒答句式 + 一条反例（拒答可检测的前提）")
    context = demo_context()
    print(f"v2 要求模型在资料不足时以固定句式开头，清单 DECLINE_MARKERS 与"
          f"\n  RAG_PROMPT_V2 的第 2 条约束是**同一份**（{len(DECLINE_MARKERS)} 条）：")
    for marker in DECLINE_MARKERS:
        print(f"    {marker!r}")
    print("  检测只做**归一化后的包含判定**（去空白与标点），不做正则、不做模糊匹配——"
          "\n  理由是可枚举、可测试、可解释（写句式的清单与检测的清单是同一份）。")

    cases: tuple[str, ...] = (
        f"{DECLINE_MARKERS[0]}，缺少关于覆盖率分母的说明。",
        f"{DECLINE_MARKERS[1]}，片段里没有讲回退族的判定顺序。",
        f"{DECLINE_MARKERS[2]}：现有片段与这道题无关。",
        "这段资料里出现了'向量'两个字，但我要谈的是另一件事。",
    )
    for index, answer in enumerate(cases, start=1):
        generator, _llm = generator_with(answer)
        result = generator.generate(DEMO_QUESTION, context)
        hit = next(
            (note for note in result.notes if "命中固定句式" in note),
            "（未命中任何句式）",
        )
        print(f"\n  {index}. 答案：{answer}")
        print(f"     fallback_reason={result.fallback_reason!r}、"
              f"is_fallback={result.is_fallback}、grounded={result.grounded}")
        print(f"     {hit}")
    print(
        "\n  第 4 条是刻意的反例：它含'资料'两个字，但**不命中任何固定句式**，"
        "\n  因此 fallback_reason 不是 model_declined。这正是'固定句式'要买的东西——"
        "\n  若检测靠'答案里有没有 资料 两个字'，这条反例会被误判成一次拒答，"
        "\n  于是把一次正常回答丢掉。"
    )


# --------------------------------------------------------------------------- #
# 第 6 节：空回复与 llm_error
# --------------------------------------------------------------------------- #


def section_6_fallbacks() -> None:
    """两条回退路径（empty_reply / llm_error）：fallback_reason / answer / llm_called."""
    title("第 6 节：空回复与 llm_error（两条“调过了”的回退路径）")
    context = demo_context()
    print("这两条路与 no_context 的关键差别：**它们都真的调过模型**"
          "（llm_called=True）——"
          "\n  护栏管的是'该不该把片段交给模型'，管不了'给了片段之后调用失败'。")

    empty_llm = MockLLM(responses=["   \n  "])
    empty_gen = RAGGenerator(empty_llm, model="mock-llm")
    empty_result = empty_gen.generate(DEMO_QUESTION, context)
    print(f"\n  ① 空回复（模型返回 {empty_result.answer!r}）：")
    print(f"     fallback_reason={empty_result.fallback_reason!r}、"
          f"llm_called={empty_result.llm_called}、"
          f"answer.strip() == ''：{empty_result.answer.strip() == ''}")
    print(f"     answer 原样保留（{len(empty_result.answer)} 个字符）："
          f"{empty_result.answer!r}——它不是'知识库里没有相关内容'（那是 no_context）")
    print(f"     假 LLM 被调用了 {len(empty_llm.calls)} 次；"
          f"引用的片段仍有 {len(empty_result.citations)} 条（上下文都在，只是答案是空的）")
    for note in empty_result.notes:
        if "empty_reply" in note or "max_tokens" in note:
            print(f"     注记：{note}")

    boom = BoomLLM()
    boom_gen = RAGGenerator(boom, model="mock-llm")
    error_result = boom_gen.generate(DEMO_QUESTION, context)
    print("\n  ② 调用抛异常：")
    print(f"     fallback_reason={error_result.fallback_reason!r}、"
          f"llm_called={error_result.llm_called}、"
          f"answer={short(error_result.answer, 40)!r}")
    print(f"     假 LLM 被调用了 {len(boom.calls)} 次；"
          f"answer == FALLBACK_NO_CONTEXT：{error_result.answer == FALLBACK_NO_CONTEXT}")
    for note in error_result.notes:
        if "llm_error" in note or "确实" in note:
            print(f"     注记：{short(note, 72)}")
    print(
        "\n  两条都算'合法状态'而不是异常：调用方拿到的是一份带 fallback_reason 的结果，"
        "\n  而不是一个栈。empty_reply 的处置是查提供方（超时 / 内容过滤 / max_tokens 太小），"
        "\n  llm_error 的处置是查提供方配置与网络——与 no_context（去改检索）是三件不同的事。"
    )


# --------------------------------------------------------------------------- #
# 第 7 节：require_citation 四格矩阵
# --------------------------------------------------------------------------- #


def section_7_require_citation() -> None:
    """开关（开 / 关）× 有无合法引用 = 四格；只有一格会丢掉答案."""
    title("第 7 节：require_citation 开关的四格矩阵")
    context = demo_context()
    print(f"settings.retrieval_require_citation = {settings.retrieval_require_citation}"
          "（缺省 False = 新能力默认不生效）")
    print("它是**一道闸门**而不是一个阈值：打开之后，'一条合法引用都没有'的答案会被整条"
          "\n  丢掉（答案取兜底答复），因此缺省关着——先让默认可核对率跑一段时间。")
    print(f"\n  {'require_citation':<17} | {'有合法引用':<10} | {'fallback_reason':<20} | "
          f"{'is_fallback':<11} | answer 前 16 字")
    for require in (False, True):
        for answer in (ANSWER_OK, ANSWER_NO_CITATION):
            generator, _llm = generator_with(answer, require_citation=require)
            result = generator.generate(DEMO_QUESTION, context)
            has_valid = bool(result.check.valid)
            print(f"  {str(require):<17} | {str(has_valid):<10} | "
                  f"{result.fallback_reason!r:<20} | {str(result.is_fallback):<11} | "
                  f"{short(result.answer, 16)}")
    print(
        "\n  四格里只有一格（require_citation=True 且没有合法引用）走了回退："
        "\n  fallback_reason='unusable_citations'，answer 换成兜底答复。"
        "\n  另外三格都落在正常路径上——包括'关着开关且没有引用'那一格："
        "\n  它作为报告里的一项（coverage=0）被看见，而不是被拦下。"
    )


# --------------------------------------------------------------------------- #
# 第 8 节：空上下文早退
# --------------------------------------------------------------------------- #


def section_8_no_context() -> None:
    """空上下文：llm.calls == 0（一次都没调）与 prompt_text == ""."""
    title("第 8 节：空上下文早退（一次 LLM 都不调）")
    context = empty_context()
    print(f"pack_context([]) 的产物：is_empty={context.is_empty}、count={context.count}、"
          f"char_count={context.char_count}、citations={len(context.citations)}")
    print("它**不是异常**：没有命中可打包时上下文为空是合法状态"
          "（与检索层的 no_data 同一句纪律）。")

    llm = MockLLM(responses=["这段答案永远不会被用上，因为护栏更靠上游"])
    generator = RAGGenerator(llm, model="mock-llm")
    result = generator.generate(DEMO_QUESTION, context)
    print(f"\n  generate()：fallback_reason={result.fallback_reason!r}、"
          f"llm_called={result.llm_called}")
    print(f"    llm.calls == 0（一次都没调）：{llm.calls == []}（实际 {len(llm.calls)} 次）")
    print(f"    prompt_text == ''：{result.prompt_text == ''}"
          f"（实际 {len(result.prompt_text)} 字）")
    print(f"    citations == ()：{result.citations == ()}")
    print(f"    check.given={result.check.given}、cited={list(result.check.cited)}、"
          f"grounded={result.grounded}")
    print(f"    answer == FALLBACK_NO_CONTEXT：{result.answer == FALLBACK_NO_CONTEXT}")
    print(f"    latency_ms={result.latency_ms}（什么都没发生，因此接近 0）")
    print("    注记：")
    for note in result.notes:
        print(f"      {note}")
    print(
        "\n  为什么护栏写在**渲染之前**：没有片段时模型仍会写出一段通顺的答案，"
        "\n  而读的人看不出区别——那正是 RAG 最贵的一种失败。因此这一次连提示词都没有"
        "\n  渲染（prompt_text 为空），Generation 的 __post_init__ 会把这三条钉住。"
    )


# --------------------------------------------------------------------------- #
# 第 9 节：pipeline.RagPipeline 集成对照
# --------------------------------------------------------------------------- #


def section_9_pipeline() -> None:
    """同一份检索结果：answer() 的 check / fallback_reason 与旧字段并排打印."""
    title("第 9 节：pipeline.RagPipeline 集成对照（新字段与旧字段并排）")
    print(f"RagPipeline.answer 的覆盖键 OVERRIDE_KEYS = {OVERRIDE_KEYS}"
          "（前两个由本层消费、temperature 转发给生成器）")
    print(f"生成器的覆盖键 GENERATION_OVERRIDE_KEYS = {GENERATION_OVERRIDE_KEYS}"
          "（两份清单各自封闭、各自报错）")

    llm = MockLLM(responses=[ANSWER_OK])
    pipeline = RagPipeline(demo_retriever(), llm)
    print(f"\n  pipeline.describe() 里的生成组：{pipeline.generator.describe()}")
    answer: RagAnswer = pipeline.answer(DEMO_QUESTION)
    print("\n  ① 命中路径：summary_line()")
    print(f"     {answer.summary_line()}")
    print(f"     旧字段：question={answer.question!r}、citations={len(answer.citations)} 条、"
          f"llm_called={answer.llm_called}、"
          f"context={answer.context.summary_line() if answer.context else None}")
    print(f"     新字段 check（接地核对报告）：{grounding_row(answer.check)}")
    print(f"     新字段 fallback_reason：{answer.fallback_reason!r}、"
          f"grounded={answer.grounded}")
    print(f"     检索的账：empty_reason={answer.retrieval.empty_reason!r}、"
          f"hits={len(answer.retrieval.hits)} 条")
    print(f"     to_dict() 的键（{len(answer.to_dict())} 个）：{sorted(answer.to_dict())}")
    print(f"     注记 {len(answer.notes)} 条（前两条）：")
    for note in answer.notes[:2]:
        print(f"       {short(note, 76)}")

    empty_llm = MockLLM(responses=["这一段永远不会被用上"])
    empty_pipeline = RagPipeline(demo_retriever(empty_store()), empty_llm)
    empty_answer: RagAnswer = empty_pipeline.answer(DEMO_QUESTION)
    print("\n  ② 检索为空（护栏）：summary_line()")
    print(f"     {empty_answer.summary_line()}")
    print(f"     llm_called={empty_answer.llm_called}、"
          f"check={empty_answer.check}（None = 这次没走到生成那一步）、"
          f"fallback_reason={empty_answer.fallback_reason!r}")
    print(f"     假 LLM 被调用了 {len(empty_llm.calls)} 次"
          f"（护栏证据：llm.calls == 0 → {empty_llm.calls == []}）")
    print(f"     empty_reason={empty_answer.retrieval.empty_reason!r}、"
          f"citations={empty_answer.citations}、"
          f"answer == FALLBACK_NO_CONTEXT：{empty_answer.answer == FALLBACK_NO_CONTEXT}")
    print(f"     grounded 属性（check 为 None 时返回 False，不抛异常）："
          f"{empty_answer.grounded}")
    print("     FALLBACK_REASON_DESCRIPTIONS 的键"
          f"（封闭清单，{len(FALLBACK_REASON_DESCRIPTIONS)} 个）：")
    print(f"       {sorted(FALLBACK_REASON_DESCRIPTIONS)}")
    print(f"     FALLBACK_REASONS（判定优先级，不含空串哨兵）：{list(FALLBACK_REASONS)}")
    print(
        "\n  两处的护栏守的是同一件事：本层管'检索为空'（empty_reason 四种），"
        "\n  生成器管'打包为空'（PackedContext.is_empty）——两处都不许把'没有片段'"
        "\n  变成'让模型自由发挥'。而 fallback_reason 用**同一套词汇表**"
        "\n  （generation.FALLBACK_REASON_DESCRIPTIONS），因此两层的空结果"
        "\n  报出来的是同一个名字，统计口径不会分家。"
    )


# --------------------------------------------------------------------------- #
# 输出：同时写终端与文件
# --------------------------------------------------------------------------- #


class _Tee:
    """把写往 stdout 的内容**同时**送到终端与文件（模块 docstring 的硬要求）."""

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def main() -> int:
    """按节运行演示，并把完整输出同时写入 ``outputs/generation_demo.txt``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_NAME
    original = sys.stdout
    with output_path.open("w", encoding="utf-8") as handle:
        sys.stdout = _Tee(original, handle)
        try:
            for section in (
                section_1_prompt_versions,
                section_2_normal,
                section_3_three_states,
                section_4_coverage_scan,
                section_5_declines,
                section_6_fallbacks,
                section_7_require_citation,
                section_8_no_context,
                section_9_pipeline,
            ):
                section()
            print(f"\n演示完成（工作目录 {Path.cwd()}）；输出已同时写入 {output_path}")
            print("九节全部离线：零网络、零 API Key、零新增依赖；"
                  "grounding 的六个数字、覆盖率与回退判定都是真算出来的。")
        finally:
            sys.stdout = original
    return 0


if __name__ == "__main__":
    sys.exit(main())
