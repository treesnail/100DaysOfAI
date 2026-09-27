"""套件：把"评测集 + 语料 + 一组变体"装成一次可复现的实验（day071）.

前三天的模块各自回答一件事：``types`` 说清形状，``runner`` 跑一条用例，
``diagnose`` 与 ``baseline`` 判好坏。本模块回答最后一个问题：
**一次实验由什么组成、怎么复现它**。

```text
评测集（cases）      JSONL：query + relevant + reference（data/eval/rag_eval.jsonl）
语料（corpus）       JSONL：id + text（data/eval/rag_corpus.jsonl）
变体（variant）      "这次与 baseline 相比**只改了哪一项**"
装配（assemble）     store + embedding + llm + variant → (RagPipeline, Retriever)
```

## 为什么要有"变体"这个形状

day070 的第七章写下过三条整理纪律，第一条是**一次只改一个旋钮**。变体就是
这条纪律的数据形状：``Variant`` 只允许声明几个具名旋钮（提示词版本、是否
强制引用、top_k、上下文预算），而 ``note`` 那一栏要求写下"这个变体改了哪一个"。

```text
baseline      当前默认（提示词 v2、不强制引用、不重排）
prompt-v1     只换提示词版本 v2 → v1        （day069 留下的 A/B 伏笔）
cite-gate     只开 require_citation 闸门
tight-budget  只收紧上下文预算               （用来量 packed_away 那一门失败）
```

一个变体**不允许**一次改两件不相关的事（比如既换提示词又改 top_k）：
那样跑出来的差值无法归因，而"无法归因的实验"与"没有实验"在信息量上是一样的
（与 ``measure_lift`` 坚持用同一批命中是同一条纪律）。

## 装配路径与 settings

``build_suite`` 是``build_retriever`` 的同构物：``None`` 的含义是"去读 settings"
（``rag_eval_*`` 一组），于是"项目默认"只有一处定义，端点与演示脚本不必
把同一批默认值各抄一遍。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smart_research_agent.evaluation.rag_eval import load_jsonl
from smart_research_agent.llm.base import BaseLLM
from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.rag_debug.diagnose import diagnose_all
from smart_research_agent.rag_debug.errors import CaseDataError, DiagnosisError
from smart_research_agent.rag_debug.report import RagEvalReport, build_report
from smart_research_agent.rag_debug.runner import DEFAULT_EVAL_K, RagEvalRunner
from smart_research_agent.rag_debug.types import DEFAULT_MIN_COVERAGE, RagEvalCase
from smart_research_agent.retrieval.generation import (
    PROMPT_VERSION_V1,
    PROMPT_VERSIONS,
)
from smart_research_agent.retrieval.pipeline import RagPipeline
from smart_research_agent.retrieval.retriever import Retriever, build_retriever
from smart_research_agent.utils.logger import get_logger
from smart_research_agent.vectorstore.base import VectorBackend
from smart_research_agent.vectorstore.types import VectorRecord, WriteReport, make_record

logger = get_logger(__name__)

#: 缺省标签（一份基线的名字；``baseline:baseline`` 这样的两段式由报告拼）。
DEFAULT_SUITE_LABEL = "baseline"

#: 收紧预算那个变体的上下文预算（用来量 packed_away：预算小到必然丢尾）。
TIGHT_BUDGET_CHARS = 200


@dataclass(frozen=True)
class Variant:
    """一个变体：**名字 + 只改的那一项**（加上一句"它改了什么"的说明）.

    ``None`` 在每个旋钮上的含义是"不指定，去读 settings"——与 ``Retriever``
    的构造参数、``RAGGenerator`` 的生成参数逐字相同的一条纪律：
    **项目默认只有一处定义**，而"这个变体到底改了什么"必须能被读出来
    （``describe()`` 与端点的 ``variants`` 字段都直接展示 ``note``）。
    """

    name: str
    prompt_version: str | None = None
    require_citation: bool = False
    top_k: int | None = None
    max_context_chars: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DiagnosisError(
                f"Variant.name 必须是非空字符串，收到 {self.name!r}："
                "报告按它分组（'这次是哪个变体'），空名字会让两次实验在报告里合并。"
            )
        if self.prompt_version is not None and self.prompt_version not in PROMPT_VERSIONS:
            raise DiagnosisError(
                f"Variant.prompt_version 只能是 None 或 {list(PROMPT_VERSIONS)} 之一，"
                f"收到 {self.prompt_version!r}："
                "受管模板只有这两版（day069 的 PROMPT_CHANGELOG）；"
                "自由文本会让'我换了哪一版'这件事无法复现——"
                "要换自定义模板请注入 RAGGenerator（那种覆盖不进提示词版本的分组）。"
            )
        if not isinstance(self.require_citation, bool):
            raise DiagnosisError(
                f"Variant.require_citation 必须是布尔值，收到 "
                f"{type(self.require_citation).__name__}"
            )
        for name in ("top_k", "max_context_chars"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise DiagnosisError(
                    f"Variant.{name} 必须是 >= 1 的整数或 None，收到 {value!r}："
                    "None 的含义是'去读 settings'，而 0 条命中 / 0 字预算"
                    "不是一次可讨论的实验。"
                )

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return {
            "name": self.name,
            "prompt_version": self.prompt_version,
            "require_citation": self.require_citation,
            "top_k": self.top_k,
            "max_context_chars": self.max_context_chars,
            "note": self.note,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return f"{self.name}（{self.note or '未说明改了什么'}）"


#: 四个内置变体（顺序 = 它还是默认集合时的运行顺序）.
DEFAULT_VARIANTS: tuple[Variant, ...] = (
    Variant(name="baseline", note="当前默认配置（提示词 v2、不强制引用）"),
    Variant(
        name="prompt-v1",
        prompt_version=PROMPT_VERSION_V1,
        note="只换提示词版本：v2 → v1（day069 留下的 A/B 伏笔）",
    ),
    Variant(
        name="cite-gate",
        require_citation=True,
        note="只开 require_citation 闸门（不可核对的答案会被整条丢掉）",
    ),
    Variant(
        name="tight-budget",
        max_context_chars=TIGHT_BUDGET_CHARS,
        note="只收紧上下文预算（用来量 packed_away：预算小到必然丢尾）",
    ),
)


def load_cases(path: str | Path) -> tuple[RagEvalCase, ...]:
    """从 JSONL 读评测集（复用 ``evaluation.rag_eval.load_jsonl``）.

    字段名与 day029 的 ``data/eval/rag_eval.jsonl`` **完全一致**（``query`` /
    ``relevant_ids`` / ``relevance_grades`` / ``reference_answer``）——
    同一份评测集要被两个时代的两套评估读到，改字段名等于让旧评估失效。
    """
    records = load_jsonl(path)
    cases: list[RagEvalCase] = []
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise CaseDataError(
                f"评测集第 {position} 行不是 JSON 对象，收到 {type(record).__name__}"
            )
        if "query" not in record or "relevant_ids" not in record:
            raise CaseDataError(
                f"评测集第 {position} 行缺少 query 或 relevant_ids："
                f"收到的键是 {sorted(record)}。"
                "这两个字段是四个检索指标的全部输入，缺一个就没有分母。"
            )
        relevant = record["relevant_ids"]
        if isinstance(relevant, str) or not isinstance(relevant, (list, tuple)):
            raise CaseDataError(
                f"评测集第 {position} 行的 relevant_ids 必须是列表，"
                f"收到 {type(relevant).__name__}"
            )
        grades = record.get("relevance_grades") or {}
        cases.append(
            RagEvalCase(
                query=str(record["query"]),
                relevant=tuple(str(item) for item in relevant),
                reference=str(record.get("reference_answer") or ""),
                grades={str(key): float(value) for key, value in dict(grades).items()},
            )
        )
    return tuple(cases)


def load_corpus(path: str | Path) -> tuple[dict[str, Any], ...]:
    """从 JSONL 读语料（每行形如 ``{"id": ..., "text": ...}``）."""
    records = load_jsonl(path)
    for position, record in enumerate(records, start=1):
        if not isinstance(record, dict) or "id" not in record or "text" not in record:
            shape = (
                sorted(record) if isinstance(record, dict) else type(record).__name__
            )
            raise CaseDataError(
                f"语料第 {position} 行必须是含 id 与 text 的 JSON 对象，"
                f"收到 {shape!r}"
            )
    return tuple(records)


def index_corpus(
    store: VectorBackend, embedding: EmbeddingProvider, corpus: Sequence[dict[str, Any]]
) -> WriteReport:
    """把语料编码并写进库，返回 ``WriteReport``（新增/覆盖/未变/跳过四个数）.

    用 ``vectorstore.types.make_record`` 造记录而不是手拼 ``VectorRecord``：
    维度校验、元数据校验、id 长度校验都在那一处（day064 的成果），
    在这里再写一遍的结果是两套判据在同一个库上给出不同结论。

    ``unchanged`` 这个数在评测里尤其有用：**"这次评估到底编码了几条"**
    决定了它花了多少钱、以及"结果变了"是不是因为向量变了
    （增量评估时，向量未变是"结果可比"的前提）。
    """
    records: list[VectorRecord] = [
        make_record(
            record_id=str(item["id"]),
            vector=embedding.embed(str(item["text"])),
            text=str(item["text"]),
        )
        for item in corpus
    ]
    report = store.upsert(records)
    logger.info(
        "语料入库：新增 %d / 覆盖 %d / 未变 %d / 跳过 %d",
        report.added,
        report.updated,
        report.unchanged,
        report.skipped,
    )
    return report


class RagEvalSuite:
    """一次实验的完整装配：评测集 + 语料 + 变体表 + 判据（**全部注入**）.

    构造参数分四组：

```text
数据类   cases / corpus               "拿什么评"
口径类   label / k / min_coverage /   "怎么算、怎么判"
         require_grounded
变体类   variants                     "要跑哪几种配置"
```

    ``k`` / ``min_coverage`` / ``require_grounded`` 三个判据进套件而不是进
    每个函数：一次实验的全部用例必须用同一套判据，否则"这一批的坏例率"
    就是一个混合了多种口径的数字（而它正是要进基线的那个）。
    """

    def __init__(
        self,
        cases: Sequence[RagEvalCase],
        corpus: Sequence[dict[str, Any]] = (),
        *,
        label: str = DEFAULT_SUITE_LABEL,
        k: int = DEFAULT_EVAL_K,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
        require_grounded: bool = True,
        variants: Sequence[Variant] | None = None,
    ) -> None:
        if not isinstance(label, str) or not label.strip():
            raise DiagnosisError(
                f"label 必须是非空字符串，收到 {label!r}（报告按它分组）"
            )
        if not cases:
            raise CaseDataError(
                "评测集不能为空：一份 0 条用例的评估会产出一张全 0 的指标表，"
                "而它在字节上与'全部失败'的报告很像——"
                "出路：先给评测集补上带金标准的用例（见 data/eval/rag_eval.jsonl）。"
            )
        for case in cases:
            if not isinstance(case, RagEvalCase):
                raise CaseDataError(
                    f"cases 里出现了 {type(case).__name__}：只收 RagEvalCase"
                )
        chosen = tuple(DEFAULT_VARIANTS if variants is None else variants)
        if not chosen:
            raise DiagnosisError(
                "variants 不能为空：至少要有一个变体，否则这份套件没有可跑的实验"
            )
        names = [item.name for item in chosen]
        if len(set(names)) != len(names):
            raise DiagnosisError(
                f"variants 里有重名：{names}。"
                "报告按名字分组，重名会让两次实验在报告里合并成一次。"
            )
        # 标注形态必须一致：`measure_ids` 按"有没有分级标注"在两个 NDCG 口径里
        # 选一个（分级版来自 day029、二值 @k 版来自 day068）。一批里两种混用
        # 会让 mean_ndcg 变成两个口径的平均，而报告里看不出这件事——
        # 那正是"数字都对、结论错的"最典型的一类。
        with_grades = [case.query for case in cases if case.grades]
        if with_grades and len(with_grades) != len(cases):
            raise CaseDataError(
                f"同一批用例的标注形态不一致：{len(with_grades)} 条有分级标注、"
                f"其余 {len(cases) - len(with_grades)} 条没有。"
                "有分级标注时 NDCG 走分级口径（增益 = 2^rel - 1），"
                "没有时走二值口径（命中记 1）——两者不能混在一个均值里。"
                "出路：要么给全部用例补上 relevance_grades，要么全部留空。"
            )
        self._cases = tuple(cases)
        self._corpus = tuple(corpus)
        self.label = label
        self.k = k
        self.min_coverage = min_coverage
        self.require_grounded = require_grounded
        self._variants = chosen
        # 判据的合法性交给 runner 与 diagnose 那两处（**校验只有一份实现**）：
        # 这里先构造一次再丢掉，于是"k=0"这类配置错误在装配那一刻就响，
        # 而不是等到第 7 条用例上才炸。
        RagEvalRunner(
            _validator_pipeline(),
            k=k,
            min_coverage=min_coverage,
            require_grounded=require_grounded,
        )

    # ------------------------------------------------------------------ 只读视图

    @property
    def cases(self) -> tuple[RagEvalCase, ...]:
        """评测集（逐条读出，调用方可以自己再过滤）."""
        return self._cases

    @property
    def corpus(self) -> tuple[dict[str, Any], ...]:
        """语料（原样读出，便于报告里回显"评的是哪一批资料"）."""
        return self._corpus

    @property
    def variants(self) -> tuple[Variant, ...]:
        """变体表（第一个是 A/B 的参照物）."""
        return self._variants

    def variant(self, name: str) -> Variant:
        """按名字取一个变体（不认识的名字会报错并列出可用名）."""
        for item in self._variants:
            if item.name == name:
                return item
        raise DiagnosisError(
            f"不认识的变体 {name!r}：可用的是 {[item.name for item in self._variants]}。"
            "静默退回第一个变体会让一次'我以为跑的是 prompt-v1'的实验"
            "悄悄跑成 baseline——而报告里的数字完全正常。"
        )

    def describe(self) -> dict[str, Any]:
        """这份套件的现状（端点与演示脚本直接展示它）."""
        return {
            "label": self.label,
            "cases": len(self._cases),
            "corpus": len(self._corpus),
            "k": self.k,
            "min_coverage": self.min_coverage,
            "require_grounded": self.require_grounded,
            "variants": [item.to_dict() for item in self._variants],
        }

    # ------------------------------------------------------------------ 装配

    @classmethod
    def from_paths(
        cls,
        cases_path: str | Path,
        corpus_path: str | Path,
        **overrides: Any,
    ) -> RagEvalSuite:
        """从两个 JSONL 文件装载（``**overrides`` 原样透传给构造函数）."""
        return cls(
            load_cases(cases_path),
            load_corpus(corpus_path),
            **overrides,
        )

    def index_into(
        self, store: VectorBackend, embedding: EmbeddingProvider
    ) -> WriteReport:
        """把语料灌进库（**评估的前置条件**：库为空时每条用例都会归因成 no_data）."""
        return index_corpus(store, embedding, self._corpus)

    def assemble(
        self,
        store: VectorBackend,
        embedding: EmbeddingProvider,
        llm: BaseLLM,
        variant: Variant | str | None = None,
    ) -> tuple[RagPipeline, Retriever]:
        """装配 ``(pipeline, retriever)``（**同一次装配的两个视图**）.

        为什么要同时交出检索器：报告里的 ``index_version`` 来自
        ``Retriever.index_state``（"这批数字查的是哪一版索引"），
        而 ``RagPipeline`` 刻意不暴露它持有的检索器（那是它的内部结构）。
        在这里拿一次是最省事、也最不容易出错的做法——
        曲线做法（从 pipeline 的私有属性取）会在某次重构后静默拿到 ``None``。
        """
        spec = self.variant(variant) if isinstance(variant, str) else (variant or self._variants[0])
        if not isinstance(spec, Variant):
            raise DiagnosisError(
                f"variant 必须是 Variant、变体名或 None，收到 {type(spec).__name__}"
            )
        retriever = build_retriever(
            store,
            embedding,
            default_top_k=spec.top_k,
        )
        pipeline = RagPipeline(
            retriever,
            llm,
            prompt_version=spec.prompt_version,
            max_context_chars=spec.max_context_chars,
            require_citation=spec.require_citation,
        )
        # 逐次覆盖会替换构造期默认值，因此报告里必须报**已解析**的那一个
        # （与 /retrieval/status 回显已解析预算同一条纪律）。
        return pipeline, retriever

    def build_pipeline(
        self,
        store: VectorBackend,
        embedding: EmbeddingProvider,
        llm: BaseLLM,
        variant: Variant | str | None = None,
    ) -> RagPipeline:
        """只要 pipeline 的便捷入口（``assemble`` 的第一个返回值）."""
        return self.assemble(store, embedding, llm, variant)[0]

    # ------------------------------------------------------------------ 运行

    def run_variant(
        self,
        store: VectorBackend,
        embedding: EmbeddingProvider,
        llm: BaseLLM,
        variant: Variant | str | None = None,
        *,
        judge: BaseLLM | None = None,
    ) -> RagEvalReport:
        """跑一个变体，返回一份报告（**评测集逐条、判据统一**）.

        报告里的 ``prompt_version`` 取自**生成器**（``pipeline.prompt_version``）
        而不是变体声明：变体里那个 ``None`` 的含义是"去读 settings"，
        而报告要报的必须是**真的生效**的那一版——两者在"settings 被改过"
        时就会分家，而分家的表现是"报告说用了 v2，实际用的是 v1"。
        """
        spec = self.variant(variant) if isinstance(variant, str) else (variant or self._variants[0])
        pipeline, retriever = self.assemble(store, embedding, llm, spec)
        runner = RagEvalRunner(
            pipeline,
            k=self.k,
            min_coverage=self.min_coverage,
            require_grounded=self.require_grounded,
            judge=judge,
        )
        outcomes = runner.run(self._cases)
        bad_cases = diagnose_all(
            outcomes,
            min_coverage=self.min_coverage,
            require_grounded=self.require_grounded,
        )
        report = build_report(
            outcomes,
            bad_cases,
            label=f"{self.label}:{spec.name}",
            k=self.k,
            prompt_version=pipeline.prompt_version,
            index_version=retriever.index_state.version_id,
        )
        logger.info("变体 %s 评估完成：%s", spec.name, report.summary_line())
        return report

    def run_all(
        self,
        store: VectorBackend,
        embedding: EmbeddingProvider,
        llm: BaseLLM,
        *,
        judge: BaseLLM | None = None,
    ) -> tuple[RagEvalReport, ...]:
        """跑完全部变体（顺序与 ``variants`` 一致；第一个是 A/B 的参照物）.

        逐条同步、不并发：一次评估里的模型调用是**要付费**的，而并发会让
        "这一批花了多少"变得难以归因；更重要的是，``judge`` 是一个有状态的
        ``BaseLLM``（``MockLLM`` 会按顺序弹脚本），并发会让它的脚本错配。
        """
        return tuple(
            self.run_variant(store, embedding, llm, spec, judge=judge)
            for spec in self._variants
        )


def build_suite(
    *,
    cases_path: str | None = None,
    corpus_path: str | None = None,
    label: str | None = None,
    k: int | None = None,
    min_coverage: float | None = None,
    require_grounded: bool | None = None,
    variants: Sequence[Variant] | None = None,
) -> RagEvalSuite:
    """按 ``settings.rag_eval_*`` 装配一份套件（端点与演示脚本的统一入口）.

    ``None`` 的语义与 ``build_retriever`` 逐字相同：**"没指定，去读 settings"**。
    路径类参数尤其如此——``settings.rag_eval_cases_path`` 指的是**仓库内**
    那份评测集，于是"评估跑的是哪一份数据"不需要在调用点写第三遍。
    """
    from smart_research_agent.config import settings

    return RagEvalSuite.from_paths(
        cases_path or settings.rag_eval_cases_path,
        corpus_path or settings.rag_eval_corpus_path,
        label=label or DEFAULT_SUITE_LABEL,
        k=settings.rag_eval_top_k if k is None else k,
        min_coverage=(
            settings.rag_eval_min_coverage if min_coverage is None else min_coverage
        ),
        require_grounded=(
            settings.rag_eval_require_grounded
            if require_grounded is None
            else require_grounded
        ),
        variants=variants,
    )


def _validator_pipeline() -> RagPipeline:
    """造一条**只用于参数校验**的 pipeline（不跑任何用例，也不读盘）.

    为什么需要它：``RagEvalRunner`` 的构造参数校验（k / min_coverage /
    require_grounded）是这一层判据的唯一定义处，而套件要在装配那一刻就把
    配置错误喊出来。这里用一个空库 + 查表编码器 + 假模型拼出一条最小链路，
    它一次都不会被调用——``RagPipeline`` 的构造是纯参数解析，没有 I/O。
    """
    from smart_research_agent.llm.embedding import MockEmbedding
    from smart_research_agent.llm.mock import MockLLM
    from smart_research_agent.vectorstore.flat import FlatVectorStore

    return RagPipeline(
        Retriever(FlatVectorStore(), MockEmbedding()),
        MockLLM(),
    )


__all__ = [
    "DEFAULT_SUITE_LABEL",
    "DEFAULT_VARIANTS",
    "TIGHT_BUDGET_CHARS",
    "RagEvalSuite",
    "Variant",
    "build_suite",
    "index_corpus",
    "load_cases",
    "load_corpus",
]
