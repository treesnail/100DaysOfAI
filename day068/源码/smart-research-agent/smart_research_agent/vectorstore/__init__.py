"""向量库包（M6-D3）：把 ``knowledge_records()`` 变成"可以按相似度查的东西".

day061 把八种格式收敛成 ``Document``，day062 把它切成 ``Chunk``，
并交出一份"四个键"的记录（``doc_id`` / ``source`` / ``text`` / ``metadata``）。
今天这份记录终于要落到一个真正的存储里——而这一层最容易被低估，
因为"存向量、查最近"听起来只是一个 ``if``。

先把这个包最重要的一句话放在最前面：

```text
一个向量库其实是两个库拼起来的
├─ 向量索引   (int64 id, float32 向量) → "谁离得最近"
└─ 记录表     id → (原文, 元数据)        → "命中的那条是什么"
```

Chroma 把两者做在一起；**FAISS 只做前者**——它的 ``IndexFlatIP``
连 id 都不存，必须用 ``IndexIDMap2`` 包一层。真实项目里配 FAISS 的标准做法
就是"FAISS 管向量 + SQLite/JSON 管文本"，本包的 ``faiss_backend``
把这张旁挂表写成了 ``<snapshot>.records.json``——**这就是那句话最直白的样子**。

## 十个模块，每个回答一个问题

```text
errors.py         这一层会怎么失败（环境 / 数据 / 调用 / 查询 四族分开）
metrics.py        谁更近？三个库三种符号约定，本包只保留两套口径
types.py          记录、命中、结果、写入报告长什么样（元数据为什么必须更严）
filters.py        where 子句：把元数据筛选写成可移植的声明
base.py           六个抽象原语 + 全部公共行为（校验/过滤/阈值/稳定排序）
flat.py           纯 Python 暴力检索的**参照实现**（零依赖、逐位可复现）
faiss_backend.py  FAISS：id 映射、-1 填充、两文件快照、过滤为什么必须预筛
chroma_backend.py Chroma：集合命名、distance 语义、默认 EF 会偷偷下载模型
registry.py       后端选型 + "缺什么、怎么装、还能用什么"的三段式报错
evaluate.py       跨后端对账（parity）与索引体检（谁是唯一事实）
pipeline.py       knowledge_records → 编码 → 写库 → 检索（本课**逐条**编码）
```

## 三条贯穿全包的纪律

1. **统一口径只有一处定义**。排序、阈值、报告一律用 ``score``（越大越近）；
   ``distance``（越小越近）只在"与外部库对账"时出现。Chroma 的 ``distances``
   与 FAISS 的 ``D`` 符号约定相反，换算只写在 ``metrics.similarity_from_distance``
   一处——**写两遍，第二次就会抄错一个符号，而结果只是"换个后端变笨了"**；
2. **过滤必须在排序之前**。FAISS 没有元数据过滤能力，"先取 top-k 再筛"
   会少返回（尽管库里还有符合条件的记录排在后面），而且不报错。
   因此 ``_find_ids`` 是抽象的一部分（见 ``base.py``）；
3. **"没做成"要分家**。``WriteReport`` 把"新增/覆盖/未变/跳过"分成四个数，
   ``IngestReport`` 再把"数据问题（skipped）"与"环境问题（failed）"分开，
   剩下不认识的异常一律上抛。**只有把它们分开，"这次为什么什么都没变"
   才能被回答**——而 day065 的增量索引正要靠这个答案省钱。

## 与既有包的接缝

- **上游**：``chunking``（day062）的 ``knowledge_records()`` 是唯一的输入形状；
  ``llm.embedding``（day041）提供编码能力；
- **下游**：day065 的 ``indexing`` 包在本层之上加批量编码、向量缓存与增量对账；
  day066 的检索器在本层之上加 Top-K 之外的过滤与路由；
  day071 的 RAG 评估会直接复用 ``evaluate.py`` 的对账与体检结构；
- **端点**：``api.routes`` 的 ``/vectorstore/*`` 与演示脚本
  ``scripts/vectorstore_demo.py``；手册在 ``docs/vector_store.md``。

## 包的导入成本（两个可选后端**不在导入时加载**）

```text
import smart_research_agent.vectorstore        → 零可选依赖
    ├─ flat / types / filters / metrics / base / registry / evaluate / pipeline  立即加载
    ├─ FaissVectorStore                         取用时才加载（该模块顶层 import numpy）
    └─ ChromaVectorStore                        取用时才加载（该模块依赖 chromadb）
```

理由：``faiss_backend`` 在模块顶层 ``import numpy``（faiss 的 Python 绑定本身
就依赖 numpy），而 ``chroma_backend`` 要拿 ``chromadb``。若在这里急切导入它们，
**"只想用 flat"的人也会因为缺 numpy/faiss/chromadb 而在 import 阶段崩掉**——
而 flat 是零依赖的参照实现，它的可用性不该被另外两个后端拖累。

因此这两个后端用模块级 ``__getattr__``（PEP 562）按需加载：
``from smart_research_agent.vectorstore import FaissVectorStore`` 与
``vectorstore.ChromaVectorStore`` 都照常可用，只是**第一次取用时才真正 import**。
日常选型走 ``registry.resolve_backend("faiss")``，它本来就在函数体内延迟导入，
会先做一次依赖探测，再给出"缺什么、怎么装、还能用什么"的三段式提示。
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from smart_research_agent.vectorstore.base import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    UNKNOWN_DIMENSION,
    VECTORSTORE_LIMITATIONS,
    VECTORSTORE_OUT_OF_SCOPE,
    VectorBackend,
)
from smart_research_agent.vectorstore.errors import (
    BackendUnavailable,
    FilterError,
    RecordError,
    VectorError,
    VectorStoreError,
)
from smart_research_agent.vectorstore.evaluate import (
    DEFAULT_PARITY_TOP_K,
    SCORE_TOLERANCE,
    ParityRow,
    compare_backends,
    compare_metrics,
    index_health,
    rank_agreement,
    recall_at_k,
    verify_parity,
)
from smart_research_agent.vectorstore.filters import (
    ARRAY_FIELD_OPERATORS,
    LIST_VALUE_OPERATORS,
    LOGICAL_OPERATORS,
    SUPPORTED_OPERATORS,
    compile_filter,
    describe_filter,
    filter_fields,
    match_metadata,
)
from smart_research_agent.vectorstore.flat import (
    FlatVectorStore,
    build_flat_store,
)
from smart_research_agent.vectorstore.metrics import (
    METRIC_ALIASES,
    METRIC_COSINE,
    METRIC_INNER_PRODUCT,
    METRIC_L2,
    METRIC_NOTES,
    METRICS,
    cosine_similarity,
    describe_metrics,
    distance,
    dot,
    norm,
    normalize_metric,
    score,
    similarity_from_distance,
    squared_l2,
)
from smart_research_agent.vectorstore.pipeline import (
    EMBEDDING_TEXT_FIELD,
    FALLBACK_TEXT_FIELD,
    RECORD_ID_FIELD,
    IngestReport,
    VectorIngestPipeline,
    embedding_input_field,
    embedding_text,
    record_from_knowledge,
    record_id_of,
)
from smart_research_agent.vectorstore.registry import (
    BACKEND_CHROMA,
    BACKEND_FACTORIES,
    BACKEND_FAISS,
    BACKEND_FLAT,
    BACKEND_PROFILES,
    BackendProfile,
    backend_names,
    backend_profile,
    create_backend,
    describe_backends,
    is_available,
    missing_requirements,
    resolve_backend,
)
from smart_research_agent.vectorstore.types import (
    MAX_RECORD_ID_LENGTH,
    METADATA_SCALAR_TYPES,
    SearchHit,
    SearchResult,
    StoreInfo,
    VectorRecord,
    WriteReport,
    assert_metadata,
    is_valid_metadata_value,
    make_record,
    metadata_type_name,
    score_hit,
    sort_hits,
    validate_vector,
)

if TYPE_CHECKING:  # pragma: no cover - 只为类型检查器与 IDE 提供跳转
    from smart_research_agent.vectorstore.chroma_backend import (
        COLLECTION_NAME_MAX_LENGTH,
        COLLECTION_NAME_MIN_LENGTH,
        DEFAULT_COLLECTION_NAME,
        SPACE_BY_METRIC,
        VECTOR_EQUAL_TOLERANCE,
        ChromaVectorStore,
        validate_collection_name,
    )
    from smart_research_agent.vectorstore.faiss_backend import (
        FAISS_ID_NOT_FOUND,
        INT64_ID_MIN,
        RECORDS_SUFFIX,
        FaissVectorStore,
        from_int64_id,
        resolve_faiss_module,
        to_int64_id,
    )

#: 两个可选后端：**符号名 → 它所在模块**。见模块 docstring 的"包的导入成本"。
LAZY_EXPORTS: dict[str, str] = {
    "COLLECTION_NAME_MAX_LENGTH": "chroma_backend",
    "COLLECTION_NAME_MIN_LENGTH": "chroma_backend",
    "DEFAULT_COLLECTION_NAME": "chroma_backend",
    "FAISS_ID_NOT_FOUND": "faiss_backend",
    "INT64_ID_MIN": "faiss_backend",
    "RECORDS_SUFFIX": "faiss_backend",
    "SPACE_BY_METRIC": "chroma_backend",
    "VECTOR_EQUAL_TOLERANCE": "chroma_backend",
    "ChromaVectorStore": "chroma_backend",
    "FaissVectorStore": "faiss_backend",
    "from_int64_id": "faiss_backend",
    "resolve_faiss_module": "faiss_backend",
    "to_int64_id": "faiss_backend",
    "validate_collection_name": "chroma_backend",
}


def __getattr__(name: str) -> Any:
    """按需加载两个可选后端里的符号（PEP 562）.

    取用之后会把结果**写回模块命名空间**（``globals()[name] = value``），
    因此这次导入只发生一次——第二次取用走的是普通属性查找。
    未登记的符号抛 ``AttributeError``（不是 ``BackendUnavailable``）：
    **"这个名字不在包里"与"它需要的库没装"是两件事**，
    只有前者应该让 ``hasattr`` 返回 False。
    """
    module_name = LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """把延迟导出的符号也列进 ``dir()``（否则 IDE 补全看不到它们）."""
    return sorted({*globals(), *LAZY_EXPORTS})

__all__ = [
    "ARRAY_FIELD_OPERATORS",
    "BACKEND_CHROMA",
    "BACKEND_FACTORIES",
    "BACKEND_FAISS",
    "BACKEND_FLAT",
    "BACKEND_PROFILES",
    "COLLECTION_NAME_MAX_LENGTH",
    "COLLECTION_NAME_MIN_LENGTH",
    "DEFAULT_COLLECTION_NAME",
    "DEFAULT_PARITY_TOP_K",
    "DEFAULT_TOP_K",
    "EMBEDDING_TEXT_FIELD",
    "FAISS_ID_NOT_FOUND",
    "FALLBACK_TEXT_FIELD",
    "INT64_ID_MIN",
    "LIST_VALUE_OPERATORS",
    "LOGICAL_OPERATORS",
    "MAX_RECORD_ID_LENGTH",
    "MAX_TOP_K",
    "METADATA_SCALAR_TYPES",
    "METRICS",
    "METRIC_ALIASES",
    "METRIC_COSINE",
    "METRIC_INNER_PRODUCT",
    "METRIC_L2",
    "METRIC_NOTES",
    "RECORDS_SUFFIX",
    "RECORD_ID_FIELD",
    "SCORE_TOLERANCE",
    "SPACE_BY_METRIC",
    "SUPPORTED_OPERATORS",
    "UNKNOWN_DIMENSION",
    "VECTORSTORE_LIMITATIONS",
    "VECTORSTORE_OUT_OF_SCOPE",
    "VECTOR_EQUAL_TOLERANCE",
    "BackendProfile",
    "BackendUnavailable",
    "ChromaVectorStore",
    "FaissVectorStore",
    "FilterError",
    "FlatVectorStore",
    "IngestReport",
    "ParityRow",
    "RecordError",
    "SearchHit",
    "SearchResult",
    "StoreInfo",
    "VectorBackend",
    "VectorError",
    "VectorIngestPipeline",
    "VectorRecord",
    "VectorStoreError",
    "WriteReport",
    "assert_metadata",
    "backend_names",
    "backend_profile",
    "build_flat_store",
    "compare_backends",
    "compare_metrics",
    "compile_filter",
    "cosine_similarity",
    "create_backend",
    "describe_backends",
    "describe_filter",
    "describe_metrics",
    "distance",
    "dot",
    "embedding_input_field",
    "embedding_text",
    "filter_fields",
    "from_int64_id",
    "index_health",
    "is_available",
    "is_valid_metadata_value",
    "make_record",
    "match_metadata",
    "metadata_type_name",
    "missing_requirements",
    "norm",
    "normalize_metric",
    "rank_agreement",
    "recall_at_k",
    "record_from_knowledge",
    "record_id_of",
    "resolve_backend",
    "resolve_faiss_module",
    "score",
    "score_hit",
    "similarity_from_distance",
    "sort_hits",
    "squared_l2",
    "to_int64_id",
    "validate_collection_name",
    "validate_vector",
    "verify_parity",
]
