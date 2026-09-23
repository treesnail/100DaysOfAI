"""RAG 生成链路：检索 → 打包 → 生成 → 引用（M6-D5；day069 起生成交给 generation）.

本模块是 day066 的后半段：它把 ``retriever`` 与 ``context`` 串成一条**能回答
问题**的链路，并在链路上钉死一条硬护栏：

```text
检索为空 → 一次 LLM 都不调，直接返回 fallback_answer（llm_called=False）
```

## 为什么这条护栏是本课最重要的东西

```text
检索为空 + 调用 LLM   模型手里没有片段，却仍然会写出一段通顺的答案
                      → 它们全来自预训练记忆，而读的人看不出区别
检索为空 + 不调用     答案明确写着"知识库中没有检索到相关内容"，并给出三条出路
```

第一行的产物**看起来最像成功**：语法正确、语气确定、引用格式也像模像样。
它是 RAG 系统最贵的一种失败——因为没有异常、没有告警，只有"用户问了一个
知识库里没有的问题，却拿到了一个编出来的答案"。因此在 ``answer`` 里，
空结果的判断在**渲染提示词之前**：护栏写在调用的上游，而不是调用的下游。

## day069 之后：本模块从"生成者"变成"装配线"

day066 的第四步只有"渲染 + 调一次模型"，day069 把这一步拆成了
``generation.RAGGenerator``：多版本提示词、六种回退（含超时与空回复）、
引用接地校验、逐次覆盖。本模块因此**只留三件事**：

```text
1) 空检索护栏   在打包与渲染**之前**判空，一次 LLM 都不调（比生成器更上游）
2) 打包预算     max_context_chars / per_hit_chars 归它（打包本来就是它的事）
3) 装配         注入生成器，或不注入时按参数自建一个（生成组只有一份定义）
```

护栏于是有了两个把手，且各自守着自己那一段：

```text
本模块      检索为空（empty_reason 四种）→ 不打包、不渲染、不生成
生成器      打包为空（PackedContext.is_empty）→ 不渲染、不调用
```

两处都不许把"没有片段"变成"让模型自由发挥"，理由与上面那一节完全相同。

## 历史常量继续从本模块导出（转发，不是搬家）

``RAG_ANSWER_PROMPT_V1`` / ``RAG_ANSWER_PROMPT`` / ``RAG_ANSWER_PROMPT_VERSION`` /
``FALLBACK_NO_CONTEXT`` / ``REQUIRED_PROMPT_FIELDS`` 这五个名字**永不删**：
调用方（端点、演示脚本、测试）零改动。它们现在是 ``generation`` 的转发——
模板住哪，清单就住哪，本模块只负责"这些名字还在原地"。

``RAG_ANSWER_PROMPT`` 从 day069 起指向 **v2**（四段式 + 固定拒答句式）。
这是一次**显式的版本升级**，不是顺手改文案：它会让没传 ``prompt_version`` 的
调用方一起换模板，因此那一天必须被留痕（见 ``generation.PROMPT_CHANGELOG``，
以及把默认版本号钉在 ``settings.retrieval_prompt_version`` 上的理由）。

## 降级为什么要进 ``notes``

三条降级都不阻断链路，因此它们唯一的痕迹就是 ``notes``：

```text
单条被截断      这次回答可能"少半句"（truncated_hits > 0）
整包丢尾        这次回答可能"少几条依据"（dropped_hits 非空）
索引有漂移      这次的依据来自一份可能过期的清单（可能少召回）
```

前两条的处置是调预算，第三条的处置是重建清单——三种不同的动作，
所以在 ``notes`` 里是三条各自可读的话，而不是一个布尔标志。
day069 起 ``notes`` 后面还会接上生成器的那一段（回退原因与出路、幻觉引用、
未引用的条数、被忽略的参数），顺序 = 链路的顺序：**先打包，再生成**。

## 与既有包的接缝

- **上游**：``Retriever``（day066~day068）交出 ``RetrievalResult``——它已经
  回答了"为什么是这几条"，本模块只管"怎么问"；
- **脚下**：``context.pack_context``（预算与编号）与 ``generation.RAGGenerator``
  （模板、回退、接地）；本模块不重写其中任何一件；
- **下游**：day071 的 RAG 评估读 ``RagAnswer.check``（三组编号与覆盖率）与
  ``fallback_reason`` 做实验分组；
- **端点**：``api.routes`` 的 ``/retrieval/answer``（它用 ``pipeline.prompt_version``
  与 ``answer.to_dict()``，两者在 day069 只增不改）；
- **手册与脚本**：``docs/retrieval.md`` 与 ``scripts/retrieval_demo.py``。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from smart_research_agent.config import settings
from smart_research_agent.llm.base import BaseLLM
from smart_research_agent.retrieval.context import Citation, PackedContext, pack_context
from smart_research_agent.retrieval.errors import ContextError, GenerationError, QueryError
from smart_research_agent.retrieval.generation import (
    CURRENT_PROMPT,
    CURRENT_PROMPT_VERSION,
    FALLBACK_NO_CONTEXT,
    FALLBACK_REASON_DESCRIPTIONS,
    FALLBACK_REASON_NO_CONTEXT,
    RAG_PROMPT_V1,
    REQUIRED_PROMPT_FIELDS,
    Generation,
    GroundingReport,
    RAGGenerator,
)
from smart_research_agent.retrieval.retriever import Retriever
from smart_research_agent.retrieval.types import (
    EMPTY_REASON_DESCRIPTIONS,
    RetrievalQuery,
    RetrievalResult,
)

# 下面这一个常量与上面那个导入都是 day069 起的**转发**：
# ``REQUIRED_PROMPT_FIELDS`` 由 ``generation`` 定义（模板住哪，清单就住在哪），
# 本模块只负责"这些名字还在原地"——调用方零改动，改模板请去 generation。

#: 允许在 ``answer(**overrides)`` 里覆盖的三个参数。
#: **day069 保持不变**（向后兼容）：前两个由本层消费，``temperature`` 转发给
#: 生成器。它**不是** ``generation.GENERATION_OVERRIDE_KEYS`` 的超集——
#: 两份清单服务两层的调用方，各自封闭、各自报错。
OVERRIDE_KEYS: tuple[str, ...] = ("max_context_chars", "per_hit_chars", "temperature")

#: 本层**转发**给生成器的覆盖键（只有它一个）。
#: 为什么单独列出来：本层消费的是两个预算，而温度的校验与生效都在 generation
#: 里——写在这里是为了让"哪些键只是路过"一眼可见（不做这件事的话，读代码的人
#: 会以为温度也在本层被校验过一遍）。
_PASSTHROUGH_KEYS: tuple[str, ...] = ("temperature",)

#: 历史常量转发（day069）：这四个名字**永不删**，调用方零改动。
#: 表在 ``generation`` 里（模板与清单同住一处），这里只是别名。
RAG_ANSWER_PROMPT_V1 = RAG_PROMPT_V1
RAG_ANSWER_PROMPT = CURRENT_PROMPT
RAG_ANSWER_PROMPT_VERSION = CURRENT_PROMPT_VERSION


@dataclass(frozen=True)
class RagAnswer:
    """一次 RAG 问答的完整交代：问题 + 答案 + 依据 + 这次到底调没调模型（M6-D5）.

    ```text
    question        规范化之后的问题（与 RetrievalQuery.text 同一份）
    answer          模型给的答案，或 fallback_answer
    citations       答案里那些 [n] 的对照表（只有进了上下文的命中）
    context         打进提示词的那段文本及其账（未调 LLM 时为 None）
    retrieval       这次检索的完整交代（含 empty_reason 与索引状态）
    llm_called      这一次到底调没调模型——护栏是否生效的唯一证据
    notes           降级注记：截断 / 丢尾 / 索引漂移 / 生成器那一组
    check           day069 新增：接地核对报告（None = 这次没走到生成那一步）
    fallback_reason day069 新增：这次答案为什么不是"模型给的可核对答案"
    ```

    ``llm_called`` 单独成为一个字段而不是"看 answer 等不等于 fallback"：
    **判定不能依赖字符串比较**——兜底答复将来改一个字，那种判定就静默失效。
    它是护栏的**证据**：``llm_called=False`` 且 ``citations=()`` 表示
    "模型这次没有机会自由发挥"。

    day069 加的两个字段是**只增不改**的（照 day068 给 ``RetrievalHit`` 加
    ``rerank_score`` 的先例）：``check`` 回答"答案里的编号能不能被片段核对"，
    ``fallback_reason`` 用 ``generation.FALLBACK_REASON_DESCRIPTIONS`` 那套
    封闭清单说明"这一次为什么不算数"。空检索那条路上生成器一次都没被调用，
    因此 ``check=None``（没有核对可言），``fallback_reason`` 记 ``no_context``
    ——它是本层自己的护栏，报出来的名字必须与生成器那套一致（一个词汇表，
    两处使用），否则"这一批里有几次没让模型发挥"就要分两个键来统计。
    """

    question: str
    answer: str
    citations: tuple[Citation, ...] = ()
    context: PackedContext | None = None
    retrieval: RetrievalResult | None = None
    llm_called: bool = False
    notes: tuple[str, ...] = ()
    check: GroundingReport | None = None
    fallback_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise QueryError(
                f"RagAnswer.question 必须是非空字符串，收到 {self.question!r}："
                "一份不知在回答什么问题的答案无法被复核。"
            )
        if not isinstance(self.answer, str):
            raise QueryError(
                f"RagAnswer.answer 必须是字符串，收到 {type(self.answer).__name__}："
                "模型回复的形状在 RagPipeline 里已经被收敛成文本。"
            )
        if self.check is not None and not isinstance(self.check, GroundingReport):
            raise QueryError(
                f"RagAnswer.check 必须是 GroundingReport 或 None，"
                f"收到 {type(self.check).__name__}："
                "None 的含义是'这次没走到生成那一步'（空检索），"
                "它不是'核对没通过'——要表达后者请让它带着三组编号进来。"
            )
        if self.fallback_reason not in FALLBACK_REASON_DESCRIPTIONS:
            raise QueryError(
                f"RagAnswer.fallback_reason 只能是 "
                f"{list(FALLBACK_REASON_DESCRIPTIONS)} 之一，"
                f"收到 {self.fallback_reason!r}："
                "它与 generation.FALLBACK_REASONS 是**同一份封闭清单**"
                "（空串表示'没有回退'），自由文本会让'这一批里有几次不可交付'"
                "无法统计。出路：用 generation 里那几个常量。"
            )

    @property
    def grounded(self) -> bool:
        """这份答案有没有可核对的依据（``check`` 为 ``None`` 时为 ``False``）.

        ``check`` 为 ``None``（空检索那条路）返回 ``False`` 而不是抛异常：
        "没有核对可言"在报告里的正确读法是"没有依据"，不是一次崩溃。
        """
        return self.check is not None and self.check.grounded

    def to_dict(self, *, include_context: bool = True) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典.

        ``include_context=True`` 是默认值（与 ``RetrievalResult.to_dict`` 相反）：
        这里的消费者主要是**排查一次回答**（"模型到底看到了什么"），
        而那正是上下文本身。只要排序与统计时请显式传 ``False``。

        day069 新增的两个键（``check`` / ``fallback_reason``）加在**末尾**：
        键集合只增，老消费者读到的键一个都不少也不改。

        ``prompt_version`` 这个既有的键报的是**当前默认版本号**
        （``RAG_ANSWER_PROMPT_VERSION``，day069 起是 v2）：``RagAnswer`` 上
        没有为此新增字段，而"这一次真正用的是哪一版"由生成器的结果回答——
        逐次覆盖 ``prompt_version`` 时，``notes`` 里也有一句"本次覆盖了提示词版本"。
        要按版本严格分组（day071 的实验）请读 ``Generation.prompt_version``。
        """
        return {
            "question": self.question,
            "answer": self.answer,
            "llm_called": self.llm_called,
            "prompt_version": RAG_ANSWER_PROMPT_VERSION,
            "citations": [citation.to_dict() for citation in self.citations],
            "context": (
                self.context.to_dict()
                if include_context and self.context is not None
                else None
            ),
            "retrieval": self.retrieval.to_dict() if self.retrieval is not None else None,
            "notes": list(self.notes),
            "check": self.check.to_dict() if self.check is not None else None,
            "fallback_reason": self.fallback_reason,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（day069 起把核对与回退也压在这一行里）."""
        question = self.question.replace("\n", " ")[:24]
        answer = self.answer.replace("\n", " ")[:36]
        called = "已调用" if self.llm_called else "未调用（检索为空）"
        parts = [
            f"问：{question}",
            f"答：{answer}",
            f"引用 {len(self.citations)} 条",
            f"LLM {called}",
        ]
        if self.check is not None:
            state = "通过" if self.check.grounded else "未通过"
            parts.append(f"接地{state}（覆盖 {self.check.coverage:.1%}）")
        if self.fallback_reason:
            parts.append(f"回退 {self.fallback_reason}")
        return " | ".join(parts)


class RagPipeline:
    """检索 → 打包 → 生成 → 引用（构造期定预算与装配，运行期只做四步）.

    构造参数分两组，**每组各自决定一件可以被单独讨论的事**：

    ```text
    预算类    max_context_chars / per_hit_chars   "给模型多少空间"（传给 pack_context）
    生成类    generator，或 prompt / prompt_version / fallback_answer /
              temperature / max_tokens / require_citation
                                                  "怎么问、问不到时说什么"
    ```

    ``None`` 在这里有明确含义：**"没指定，去读 settings"**
    （两个预算读 ``retrieval_max_context_chars`` / ``retrieval_per_hit_chars``，
    生成组的四个读 ``retrieval_*``，见 ``generation.RAGGenerator``）。
    于是"项目默认"只有一处定义，而端点与演示脚本不必各抄一遍。

    **生成组有两种装配方式，且只能选一种**：

    ```text
    注入 generator   整组生成参数由它说了算（本类不再接受生成组的逐项参数）
    不注入           本类用注入的 llm 自建一个 RAGGenerator（逐项参数生效）
    ```

    两种都给时本类当场报 ``GenerationError``：静默让其中一边生效，会让另一边
    **看起来也生效了**——而"这次到底是哪一组参数在跑"必须在报告里唯一。

    检索器与 LLM 都是**注入**的，且**生成器用的必须是同一个 LLM 对象**（本类会
    核对）：端点拦住的是本层持有的那个 LLM（"未配置 LLM"的检查读它），
    要是生成器拿的是另一个，"拦住的"与"真正发出去的"就成了两件事。
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        *,
        prompt: str | None = None,
        max_context_chars: int | None = None,
        per_hit_chars: int | None = None,
        fallback_answer: str = FALLBACK_NO_CONTEXT,
        temperature: float | None = None,
        generator: RAGGenerator | None = None,
        prompt_version: str | None = None,
        max_tokens: int | None = None,
        require_citation: bool = False,
    ) -> None:
        if not isinstance(retriever, Retriever):
            raise QueryError(
                f"retriever 必须是 Retriever，收到 {type(retriever).__name__}。"
                "RAG 链路的前半段是检索器：它负责'取哪几条'，本类负责'怎么问'。"
            )
        if not isinstance(llm, BaseLLM):
            raise QueryError(
                f"llm 必须是 BaseLLM，收到 {type(llm).__name__}。"
                "任何实现 BaseLLM.chat 的对象都可以注入（测试用假实现是刻意的用法）。"
            )
        self._retriever = retriever
        self._llm = llm
        self._max_context_chars = _resolve_budget(
            "max_context_chars", max_context_chars, settings.retrieval_max_context_chars
        )
        self._per_hit_chars = _resolve_budget(
            "per_hit_chars", per_hit_chars, settings.retrieval_per_hit_chars
        )
        self._generator = _resolve_generator(
            generator,
            llm,
            prompt=prompt,
            prompt_version=prompt_version,
            fallback_answer=fallback_answer,
            temperature=temperature,
            max_tokens=max_tokens,
            require_citation=require_citation,
        )

    # ------------------------------------------------------------------ 只读视图

    @property
    def prompt_version(self) -> str:
        """当前提示词版本（报告与评估按它分组）——day069 起来自生成器."""
        return self._generator.prompt_version

    @property
    def llm(self) -> BaseLLM:
        """注入的模型（端点在"未配置 LLM"时要在构造前就拦住）."""
        return self._llm

    @property
    def generator(self) -> RAGGenerator:
        """本层用的生成器（只读；端点与演示脚本用它读生成参数与兜底答复）."""
        return self._generator

    def describe(self) -> dict[str, Any]:
        """这一层的配置 + 检索器的现状 + **生成器那一组**（端点直接返回它）.

        ``temperature`` 取自生成器而不是本层：构造期那个 ``None`` 的含义是
        "去读 settings"，因此报告里要给的必须是**已解析**的那一个数。
        """
        return {
            "prompt_version": self.prompt_version,
            "max_context_chars": self._max_context_chars,
            "per_hit_chars": self._per_hit_chars,
            "temperature": self._generator.temperature,
            "retriever": self._retriever.describe(),
            "generator": self._generator.describe(),
        }

    # ------------------------------------------------------------------ 问答入口

    def answer(self, question: str | RetrievalQuery, **overrides: Any) -> RagAnswer:
        """回答一个问题（四步：检索 → 判空 → 打包 → 生成与核对）.

        第 2 步（判空）**在打包、渲染与生成之前**，这是护栏在代码里的位置：
        检索为空时 ``pack_context`` 与生成器都不会被调用——"一次 LLM 都不调"
        这条纪律由此有了两个把手（本模块管"检索为空"，生成器管"打包为空"）。

        ``overrides`` 允许逐次覆盖 ``max_context_chars`` / ``per_hit_chars`` /
        ``temperature``（例如端点按请求体里的预算回答）。前两个由本层消费，
        ``temperature`` **转发**给生成器（校验只有一份实现，在 generation 里）。
        不认识的键会报 ``QueryError`` 并列出合法取值——**静默忽略一个覆盖参数**
        会让调用方以为它的预算生效了。
        """
        max_chars, per_hit, generation_overrides = self._resolve_overrides(overrides)
        retrieval = self._retriever.retrieve(question)
        # 用**已规范化**的问题文本：一次检索里的 question 与送进提示词的问题
        # 必须是同一份（与 RetrievalResult.query 存已解析默认值是同一个理由）。
        question_text = retrieval.query.text

        if retrieval.is_empty:
            return RagAnswer(
                question=question_text,
                answer=self._generator.fallback_answer,
                citations=(),
                context=None,
                retrieval=retrieval,
                llm_called=False,
                notes=tuple(_empty_notes(retrieval)),
                check=None,
                fallback_reason=FALLBACK_REASON_NO_CONTEXT,
            )

        context = pack_context(
            retrieval.hits,
            max_chars=max_chars,
            per_hit_chars=per_hit,
        )
        generation: Generation = self._generator.generate(
            question_text, context, **generation_overrides
        )
        # 注记的顺序 = 链路的顺序：先"这次打包削掉了什么"，再"这次回答与引用
        # 对得上吗"。两段各自可读，因此不合并成一句。
        notes = _degrade_notes(context, retrieval, max_chars=max_chars, per_hit=per_hit)
        notes.extend(generation.notes)
        return RagAnswer(
            question=question_text,
            answer=generation.answer,
            citations=generation.citations,
            context=context,
            retrieval=retrieval,
            llm_called=generation.llm_called,
            notes=tuple(notes),
            check=generation.check,
            fallback_reason=generation.fallback_reason,
        )

    def answer_many(self, questions: Sequence[str | RetrievalQuery]) -> list[RagAnswer]:
        """批量问答（逐条调用 ``answer``，顺序与入参一致）.

        与 ``Retriever.retrieve_many`` 同样是**刻意的逐条**：每次问答的
        ``llm_called`` / ``notes`` / 上下文占用都是独立的证据，
        而"这一批总共花了多少"与"其中一次为什么没调模型"是两个问题。
        """
        return [self.answer(item) for item in questions]

    # ------------------------------------------------------------------ 内部

    def _resolve_overrides(self, overrides: dict[str, Any]) -> tuple[int, int, dict[str, Any]]:
        """解析逐次覆盖，并**明确拒绝**不认识的键（见 ``answer``）.

        返回值第三项是**转发给生成器**的那一份（只有 ``temperature``）：
        本层消费的是两个预算，温度的校验与生效都在 generation 里——
        同一套校验只该有一份实现，否则"温度越界"会在两个地方各有一套文案，
        而两份文案迟早会对同一个值给出不同的结论。
        """
        unknown = sorted(key for key in overrides if key not in OVERRIDE_KEYS)
        if unknown:
            raise QueryError(
                f"不认识的覆盖参数：{unknown}。可用的键只有 {list(OVERRIDE_KEYS)}。"
                "静默忽略一个覆盖参数会让调用方以为它生效了——"
                "例如把 max_context_chars 拼成 max_context_char 之后，"
                "预算仍然是默认值，而答案看起来只是'变短了'。"
            )
        max_chars = _resolve_budget(
            "max_context_chars",
            overrides.get("max_context_chars", self._max_context_chars),
            self._max_context_chars,
        )
        per_hit = _resolve_budget(
            "per_hit_chars",
            overrides.get("per_hit_chars", self._per_hit_chars),
            self._per_hit_chars,
        )
        forwarded = {key: overrides[key] for key in _PASSTHROUGH_KEYS if key in overrides}
        return max_chars, per_hit, forwarded


# --------------------------------------------------------------------------- #
# 装配：两种生成方式只能选一种
# --------------------------------------------------------------------------- #


def _resolve_generator(
    generator: RAGGenerator | None,
    llm: BaseLLM,
    *,
    prompt: str | None,
    prompt_version: str | None,
    fallback_answer: str,
    temperature: float | None,
    max_tokens: int | None,
    require_citation: bool,
) -> RAGGenerator:
    """装配本层要用的生成器（注入的那个，或按参数自建一个）.

    自建那条路上，**生成组的参数逐个透传**（含 ``None``：它的含义是"去读
    settings"，因此本层不预先解析它们——解析只有生成器那一份实现）。

    注入那条路上做三项核对，每一项都对应一种会让人读错报告的情形：

    ```text
    形状不是 RAGGenerator   本层要读它的 prompt_version / temperature
    用的不是同一个 llm      端点拦住的那个与真正发出去的那个会分家
    同时还逐项给了生成参数   两组参数同时"看起来生效"
    ```
    """
    if generator is None:
        return RAGGenerator(
            llm,
            prompt_version=prompt_version,
            prompt=prompt,
            fallback_answer=fallback_answer,
            temperature=temperature,
            max_tokens=max_tokens,
            require_citation=require_citation,
        )
    if not isinstance(generator, RAGGenerator):
        raise GenerationError(
            f"generator 必须是 RAGGenerator（或留 None 让本类自建），"
            f"收到 {type(generator).__name__}。"
            "本层要读它的 prompt_version / temperature / fallback_answer，"
            "形状不对会让 describe() 与空检索护栏在运行期才炸。"
            "出路：注入 generation.RAGGenerator，或把这一项留 None。"
        )
    if generator.llm is not llm:
        raise GenerationError(
            "注入的 generator 用的模型与本层收到的 llm 不是同一个对象。"
            "端点拦住的是本层持有的那个（'未配置 LLM'的检查读它），"
            "而真正发出去的提示词由生成器交给另一个——两份报告会互相矛盾。"
            "出路：把同一个对象同时传给两处（或只传 llm，让本类自建生成器）。"
        )
    # 逐项参数的"给没给"判定：``fallback_answer`` 用**与缺省值是否相同**来判
    # （它有一个非空缺省），因此显式传了同一个字符串时看不出区别——那种情况
    # 本来就无害（两组参数取值一致），所以不为它另设哨兵。
    conflicting = [
        name
        for name, given in (
            ("prompt", prompt is not None),
            ("prompt_version", prompt_version is not None),
            ("fallback_answer", fallback_answer != FALLBACK_NO_CONTEXT),
            ("temperature", temperature is not None),
            ("max_tokens", max_tokens is not None),
            ("require_citation", require_citation),
        )
        if given
    ]
    if conflicting:
        raise GenerationError(
            f"注入 generator 时不能再逐项给生成参数：{conflicting}。"
            "两种装配方式回答的是同一个问题（'这一组参数由谁定'），"
            "同时给会让其中一边**看起来生效**而实际上被忽略。"
            "出路：要么只注入 generator，要么把这几项交给本类自建生成器。"
        )
    return generator


# --------------------------------------------------------------------------- #
# 注记
# --------------------------------------------------------------------------- #


def _empty_notes(retrieval: RetrievalResult) -> list[str]:
    """空结果时记的三条注记（**为什么没调模型**必须被写出来）.

    第一句给原因，第二句给护栏本身，第三句给出路——与
    ``FALLBACK_NO_CONTEXT`` 的三条出路是同一条纪律：
    "没有找到"必须跟一个下一步动作，否则它只是一次失败宣告。
    """
    reason = EMPTY_REASON_DESCRIPTIONS.get(retrieval.empty_reason, "未知原因")
    return [
        f"检索为空（empty_reason={retrieval.empty_reason}）：{reason}",
        "按硬护栏没有调用 LLM：检索为空时不许让模型自由发挥——"
        "answer 直接取 fallback_answer，llm_called=False、citations=()",
        "建议：换一种说法、放宽过滤条件（时间范围 / 元数据），"
        "或确认索引版本是否包含这批资料",
    ]


def _degrade_notes(
    context: PackedContext,
    retrieval: RetrievalResult,
    *,
    max_chars: int,
    per_hit: int,
) -> list[str]:
    """三类降级各记一条（截断 / 丢尾 / 索引漂移，见模块 docstring）."""
    notes: list[str] = []
    if context.truncated_hits:
        notes.append(
            f"有 {context.truncated_hits} 条命中被单块上限 {per_hit} 字截断："
            "这些片段是半截的，答案可能缺一句结论——要完整片段请调大 "
            "settings.retrieval_per_hit_chars。"
        )
    if context.dropped_hits:
        notes.append(
            f"预算 {max_chars} 字放不下，从尾部丢掉了 {len(context.dropped_hits)} 条命中"
            f"（{', '.join(hit.record_id for hit in context.dropped_hits[:3])}）："
            "被丢的是名次最靠后的那几条，答案可能少一部分依据——"
            "要更多依据请调大 settings.retrieval_max_context_chars。"
        )
    if retrieval.index_state.has_drift:
        notes.append(
            f"索引漂移 {len(retrieval.index_state.drift)} 项："
            "本次检索基于可能过期的清单，结果可能少召回"
            f"（{retrieval.index_state.drift[0]}）——"
            "请重建清单，或确认这次少召回是否可接受。"
        )
    return notes


# --------------------------------------------------------------------------- #
# 参数解析（构造期与逐次覆盖共用同一套校验）
# --------------------------------------------------------------------------- #


def _resolve_budget(label: str, value: int | None, default: int) -> int:
    """解析一个预算参数（``None`` → 默认值；必须是 >= 1 的整数）."""
    resolved = default if value is None else value
    if not isinstance(resolved, int) or isinstance(resolved, bool):
        raise ContextError(
            f"{label} 必须是整数，收到 {type(resolved).__name__}（值为 {resolved!r}）"
        )
    if resolved < 1:
        raise ContextError(
            f"{label}={resolved}：预算为 0 意味着没有任何片段能进提示词，"
            "而那会让一次正常的检索退化成一个'没有资料的提问'。"
            "要'不限'请给一个足够大的正数，不要给 0。"
        )
    return resolved


__all__ = [
    "FALLBACK_NO_CONTEXT",
    "OVERRIDE_KEYS",
    "RAG_ANSWER_PROMPT",
    "RAG_ANSWER_PROMPT_V1",
    "RAG_ANSWER_PROMPT_VERSION",
    "REQUIRED_PROMPT_FIELDS",
    "RagAnswer",
    "RagPipeline",
]
