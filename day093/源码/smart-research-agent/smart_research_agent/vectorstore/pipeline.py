"""知识库记录 → 向量库：把 day062 切出来的片段真正落进索引里（M6-D3）.

day062 交出来的东西长这样（``ChunkSet.knowledge_records()``）：

```text
{"doc_id": <chunk_id>, "source": 来源, "text": 原文片段, "metadata": {… retrieval_text …}}
```

而向量库要的是另一副形状：``VectorRecord``（``record_id`` + 向量 + 原文 + 扁平元数据）。
中间那一步——**"哪段文本送去编码、编码不出来算谁的、写库失败又算谁的"**——
此前没有任何模块负责：分块器不该知道有向量库，向量库也不该知道块是怎么切的
（``base.py`` 的 docstring 把这条边界写成了明文：本层"只接收算好的向量"）。

本模块就是这一段的负责人，而且只负责这一段。它**不依赖 registry**
（后端由调用方注入实例）——这样"装配哪个后端"与"怎么摄取"是两个可以分开演进的
决定，而不是一个互相牵制的决定。

## 本课刻意的选择：逐条编码

```text
day064（本模块）  逐条 embedding → 直接 upsert。embedding_calls == 被编码的条数
day065            批量编码 + 向量缓存 + 增量对账；embedding_calls 会显著小于条数
```

**逐条编码是一个被刻意写下来的决定，不是"还没优化"**：

- 它让 ``IngestReport.embedding_calls`` 有一个**精确**的含义——"这次实际编码了几段文本"。
  有了它，day065 的任何优化才产生一个**可比较的数字**（"省下了多少次编码"），
  而不是一种感觉；如果这里就用 ``len(records)`` 顶替，那个数字会永远等于条数，
  于是"缓存有没有生效"这个问题在报告里根本无法被回答。
- 它让失败边界变得简单：第 N 条编码失败时，前 N-1 条的去向已经确定，
  报告不需要回答"一批里第几条炸了"这种批量 API 普遍答不好的问题。

代价同样是明写在明面上的：**同样的 K 段文本，这里会发起 K 次调用**。
神经模型上这是一次前向 vs K 次前向的差别，云端 API 上是 1 次 HTTP vs K 次
（也是真实的账单差别）。这笔账留给 day065 用"批量 + 缓存"来还——
本课先把**基线**测出来，并把那个数字的含义钉死。

## "存的"与"编码的"不是同一份文本

```text
text             原文片段          → 落库（命中之后要还给用户看的就是它）
retrieval_text   面包屑 + 正文     → 送进 embedding（它才是"该被编码的那份文本"）
```

day062 把标题面包屑放进了 ``metadata["retrieval_text"]``，并在
``knowledge_records()`` 的 docstring 里写明"向量化时用它"——本模块兑现那一条。
一句"阈值设为 0.85"单独看没有话题信息，而它上面的标题（"语义缓存"）
才是"缓存相似度怎么配"能命中它的原因；把面包屑扔掉，等于主动放弃这部分召回。

``embedding_input_field(record)`` 把这件事变成**可核对的事实**：
报告与调试时能直接问出"这一条用的是哪个字段"，而不必靠读代码相信它。
缺 ``retrieval_text`` 的记录（例如不是 day062 产出的、人为拼的记录）
回落到 ``text``，同样是显式的、有函数可查的行为。

## 三种"没做成"必须分家

与 ``WriteReport`` 的三态、day061 的三类状态是同一条纪律：**混在一起，
报错之后的第一个动作就只能靠猜。**

```text
skipped   数据问题：缺 doc_id / text 与 retrieval_text 都是空白   → 调用方修数据
failed    环境与匹配问题：VectorStoreError（维度不一致、后端整批拒绝）→ 修环境或重建索引
异常上抛   不在"我们认识的失败"里的异常一律向上抛，**不许记进报告**
```

第三条最容易被违反：一个宽泛的 ``except Exception`` 会把"后端实现里有个
``TypeError``"这种真 bug 记成一行 ``failed=1``，于是它永远不会被修——
**报告只能收留我们认识的失败，不认识的失败必须让调用栈把它喊出来。**

## 放弃了什么（代价写在明面上）

| 放弃的东西 | 代价 | 谁在什么时候还这笔账 |
|-----------|------|---------------------|
| 批量编码 | K 段文本 → K 次 ``embed`` 调用 | day065 的批量 + 缓存 |
| 向量缓存 | 重放同一批会**重新编码**（但库不变，报告是 ``unchanged``） | day065 的增量对账 |
| 增量索引 | 每次 ingest 都要遍历整批输入 | day065 的索引清单 |
| 元数据瘦身 | ``retrieval_text`` 原样落库，存储约翻一倍 | 刻意保留"存的和编码的不一样"这个证据 |
| 多线程写入 | 与后端一致，单进程使用 | 不在课程范围 |

## 谁依赖它

```text
day065 的 indexing 包    在 ingest 之上加批量编码 + 缓存 + 增量对账，
                         并与本模块的 IngestReport 逐字段对账（embedding_calls 是基准列）
day066 的检索器           search() 是同一条编码路径（同一份 embedding）
scripts / 端点            stats() 直接回显摄取配置，便于"这次是在什么条件下入库的"
tests/                    全部离线：TableEmbedding 把向量写死，期望值手算得出来
```
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from smart_research_agent.config import settings
from smart_research_agent.llm.embedding import EmbeddingProvider
from smart_research_agent.vectorstore.base import VectorBackend
from smart_research_agent.vectorstore.errors import VectorError, VectorStoreError
from smart_research_agent.vectorstore.metrics import METRIC_COSINE
from smart_research_agent.vectorstore.types import SearchResult, VectorRecord, make_record

#: 优先编码的字段名：day062 放在 ``metadata`` 里的检索视图（面包屑 + 正文）。
EMBEDDING_TEXT_FIELD = "retrieval_text"

#: 回落字段名：原文片段。它同时也是**落库**的那份文本。
FALLBACK_TEXT_FIELD = "text"

#: 知识库记录里承载 id 的键（day062 的约定：这个位置放的是 ``chunk_id``）。
RECORD_ID_FIELD = "doc_id"


# --------------------------------------------------------------------------- #
# 形状转换：四个纯函数（不碰后端、不碰 embedding，因此可以单独核对）
# --------------------------------------------------------------------------- #


def record_id_of(record: dict[str, Any]) -> str:
    """取知识库记录的 id（``doc_id`` 位置上的 ``chunk_id``）.

    缺失、``None``、纯空白一律返回空串——**"没有 id"与"id 是空串"在
    本模块看来是同一件事**：它们都会在向量库的 ``VectorRecord`` 那里被拒
    （id 是更新与删除的唯一依据），所以要在编码之前就把它判成 ``skipped``，
    而不是白白花掉一次 embedding。

    这里刻意不替调用方生成一个 id：一条自动拼出来的 id（例如按序号）
    在下一批数据里就对不上了，于是同一条内容会在库里出现两次——
    **那比丢掉这条记录更难查**。
    """
    value = record.get(RECORD_ID_FIELD)
    if value is None:
        return ""
    return str(value).strip()


def embedding_text(record: dict[str, Any]) -> str:
    """返回**该被编码**的那份文本：``retrieval_text`` 优先，缺则 ``text``.

    "缺"的判据是**空白**而不是"键不存在"：一个只有换行与空格的面包屑
    与没有面包屑是同一件事（它编码出来不会带来任何话题信息），
    因此判定统一用 ``str.strip()``。回落是静默的——这正是
    ``embedding_input_field`` 存在的理由：静默的回落必须有一个可查的出口。
    """
    preferred = _field_text(record, EMBEDDING_TEXT_FIELD)
    if preferred.strip():
        return preferred
    return _field_text(record, FALLBACK_TEXT_FIELD)


def embedding_input_field(record: dict[str, Any]) -> str:
    """返回这条记录**实际**会用的字段名（``retrieval_text`` 或 ``text``）.

    存在的理由是把"优先 retrieval_text"从一句承诺变成一次可断言的事实：
    报告、调试脚本与测试都能直接问出这一条走的是哪个字段。
    两个字段都是空白时它也返回 ``text``（"回落到它、然后在下一步被跳过"），
    因为这条记录的归宿是 ``skipped``，字段名已经不改变任何结果。
    """
    if _field_text(record, EMBEDDING_TEXT_FIELD).strip():
        return EMBEDDING_TEXT_FIELD
    return FALLBACK_TEXT_FIELD


def record_from_knowledge(
    record: dict[str, Any],
    vector: list[float] | tuple[float, ...],
    *,
    metric: str = METRIC_COSINE,
) -> VectorRecord:
    """把一条知识库记录 + 一个算好的向量，变成一条 ``VectorRecord``.

    三个必须说清的地方：

    ```text
    record_id  取 doc_id（chunk_id），不做任何改写：id 是跨模块的契约
    text       取**原文片段**（不是 retrieval_text）：命中之后要给用户看的是它
    metadata   day062 交来的那份原样保留，另外补一个 source
    ```

    ``metadata`` 原样保留（含 ``retrieval_text``）是刻意的：本模块是翻译层，
    不是编辑器——悄悄删掉一个键，会让"库里那条记录与产出它的块对不上"
    变成一件查不出来的事。代价是存储翻倍，写在模块 docstring 的取舍表里。

    补 ``source`` 的理由：``VectorRecord`` 没有独立的来源字段，
    不把它折进 metadata 就等于**丢掉**了这条信息（无法按来源过滤、
    也无法在命中之后告诉用户出处）。冲突时以原始 metadata 为准。
    """
    metadata = dict(_metadata_of(record))
    source = record.get("source")
    if source is not None and "source" not in metadata:
        metadata["source"] = str(source)
    return make_record(
        record_id_of(record),
        vector,
        _field_text(record, FALLBACK_TEXT_FIELD),
        metadata,
        metric=metric,
    )


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IngestReport:
    """一次摄取的结果：**五种去向各自一个数字**，外加逐条的原因台账.

    ```text
    seen             这批输入一共有多少条
    written          真正写进库的条数（added + updated）
    unchanged        同一 id、内容逐位相同 → 没有改动任何东西（day065 增量对账的地基）
    skipped          数据问题：缺 id / 文本全空白 → 调用方修数据
    failed           环境与匹配问题：维度不一致、后端整批拒绝 → 修环境或重建索引
    embedding_calls  本次**实际**调用 ``embedding.embed`` 的次数（逐条编码，见模块 docstring）
    ```

    恒等式 ``seen == written + unchanged + skipped + failed`` 是这份报告的
    自检点：对不上就说明有一条记录的归宿没有说法。

    ``failures`` 是 ``(record_id, 原因)`` 的逐条台账（跳过与失败都在里面），
    原因字符串带 ``"跳过："`` / ``"失败："`` 前缀，便于直接 grep 出该找谁修。
    ``doc_id`` 缺失的记录，id 位置是空串——那正是它被跳过的原因。
    """

    seen: int
    written: int
    unchanged: int
    skipped: int
    failed: int
    embedding_calls: int
    backend: str
    metric: str
    dimension: int
    failures: tuple[tuple[str, str], ...] = ()

    @property
    def ok(self) -> bool:
        """这批数据是否**干净地**进了库（有跳过或有失败都算不干净）.

        ``skipped`` 也让它为 ``False``：跳过意味着调用方给的数据有洞，
        而"报告是绿的"应该等价于"这批数据没有任何一条需要人再看一眼"。
        """
        return self.skipped == 0 and self.failed == 0

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（端点与报告共用一份）."""
        return {
            "seen": self.seen,
            "written": self.written,
            "unchanged": self.unchanged,
            "skipped": self.skipped,
            "failed": self.failed,
            "embedding_calls": self.embedding_calls,
            "backend": self.backend,
            "metric": self.metric,
            "dimension": self.dimension,
            "ok": self.ok,
            "failures": [
                {"record_id": record_id, "reason": reason}
                for record_id, reason in self.failures
            ],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（演示脚本与日志用它）."""
        return (
            f"{self.backend}/{self.metric}/{self.dimension}d | 输入 {self.seen} → "
            f"写入 {self.written}（未变 {self.unchanged}）| 跳过 {self.skipped} | "
            f"失败 {self.failed} | 编码调用 {self.embedding_calls} 次"
        )


# --------------------------------------------------------------------------- #
# 编排
# --------------------------------------------------------------------------- #


class VectorIngestPipeline:
    """摄取编排器：一个后端 + 一个编码器 + 两个可覆盖的检索默认值.

    依赖全部由构造注入（``backend`` / ``embedding``），因此本类
    **不依赖 ``registry``**，也就能与"装配哪个后端"分开测试：
    注入 ``FlatVectorStore`` 即可完全离线、确定性、零网络。

    ``top_k`` / ``min_score`` 在这里是**第二级**默认值（第三级是 ``settings``，
    第一级是每次调用的显式参数）——与 day062 的 ``chunking_policy_from``
    是同一套"项目级基线 + 单次覆盖"的三级优先级。
    """

    def __init__(
        self,
        backend: VectorBackend,
        embedding: EmbeddingProvider,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> None:
        self.backend = backend
        self.embedding = embedding
        self._top_k = top_k
        self._min_score = min_score

    # ---------------------------------------------------------------- 摄取

    def ingest(self, records: Iterable[dict[str, Any]]) -> IngestReport:
        """把一批知识库记录编成向量并写进库，返回逐去向的报告.

        流程刻意排成"**先把能判的都判掉，再编码，再一次性写库**"：

        ```text
        1. 缺 id            → skipped（连编码都不发起：一次调用都不该浪费）
        2. 文本全空白        → skipped
        3. 逐条 embed(text) → embedding_calls 在这里 +1（本课的刻意选择，见模块 docstring）
        4. 维度与库不一致     → failed（**单条失败不阻断整批**）
        5. 记录构造被拒       → failed（RecordError 也是 VectorStoreError，只吞这一族）
        6. 剩下的**一次**交给 backend.upsert（批量语义在后端，见 base._write）
        ```

        第 4 步是本方法唯一需要解释的"提前判定"：后端的维度校验发生在
        ``_prepare`` 里，它是**整批**语义——一条 16 维的记录混进 8 维的库，
        基础实现会在构造到那一条时直接抛错，于是这一批其余的干净记录
        一条都进不去。**一条脏数据不该毁掉整批**，所以维度在这里逐条判：
        判出的失败记进 ``failed``，其余的照常入库。

        编码之后抛出的非 ``VectorStoreError`` 一律向上抛（含编码器自身抛出的
        任何异常）：本模块只收留"我们认识的失败"。
        """
        seen = 0
        skipped = 0
        failed = 0
        embedding_calls = 0
        failures: list[tuple[str, str]] = []
        prepared: list[VectorRecord] = []
        #: 期望维度：库已定维就用库的；库还空着时用**本批第一条**定下来，
        #: 这样"同一批里维度不一致"在空库上也能被逐条判出来。
        expected_dimension = self.backend.dimension if self.backend.has_dimension else 0

        for raw in records:
            seen += 1

            record_id = record_id_of(raw)
            if not record_id:
                skipped += 1
                failures.append(
                    (
                        record_id,
                        f"跳过：缺少 {RECORD_ID_FIELD}（知识库记录的唯一键就是 chunk_id）。"
                        "本模块不会替它编一个 id——自动拼出来的 id 在下一批就对不上，"
                        "同一条内容会在库里出现两次。",
                    )
                )
                continue

            text = embedding_text(raw)
            if not text.strip():
                skipped += 1
                failures.append(
                    (
                        record_id,
                        f"跳过：{FALLBACK_TEXT_FIELD} 与 {EMBEDDING_TEXT_FIELD} 都是空白。"
                        "编码一段空文本没有意义（零向量会被向量库直接拒收），"
                        "请在上游修数据。",
                    )
                )
                continue

            # 逐条编码：这里的 +1 就是 embedding_calls 的全部来源。
            vector = self.embedding.embed(text)
            embedding_calls += 1

            if expected_dimension and len(vector) != expected_dimension:
                failed += 1
                failures.append(
                    (record_id, _dimension_failure(expected_dimension, len(vector), record_id))
                )
                continue

            try:
                record = record_from_knowledge(raw, vector, metric=self.backend.metric)
            except VectorStoreError as exc:
                # RecordError 也属于这一族（id 超长、元数据类型越界、零向量…）：
                # 它们的共同点是"构造这条记录时就已经知道它进不去"，
                # 与"环境不认识它"是两回事，但归宿相同——记 failed，不打断整批。
                failed += 1
                failures.append((record_id, f"失败：记录构造被拒：{exc}"))
                continue

            if not expected_dimension:
                expected_dimension = record.dimension
            prepared.append(record)

        written = 0
        unchanged = 0
        if prepared:
            try:
                # 一次交给后端：**批量语义属于后端**（added/updated/unchanged
                # 的逐条判定写在 base._write 与各后端 _upsert 里，不在这里重写）。
                write_report = self.backend.upsert(prepared)
            except VectorStoreError as exc:
                # 走到了这里说明前端能判的都判过了，是后端对**整批**的拒绝
                # （例如换库之后维度对不上、外部服务不可用）。逐条记账而不是
                # 记一条汇总，是为了让 failures 与 failed 计数保持一一对应。
                failed += len(prepared)
                failures.extend(
                    (record.record_id, f"失败：整批写入被后端拒绝：{exc}")
                    for record in prepared
                )
            else:
                written = write_report.written
                unchanged = write_report.unchanged
                # 后端自己跳过的条数（护栏）也记入 skipped：宁可让它出现在
                # 报告里被追问，也不要在计数上悄悄丢掉几条。
                skipped += write_report.skipped

        return IngestReport(
            seen=seen,
            written=written,
            unchanged=unchanged,
            skipped=skipped,
            failed=failed,
            embedding_calls=embedding_calls,
            backend=self.backend.name,
            metric=self.backend.metric,
            dimension=self.backend.dimension,
            failures=tuple(failures),
        )

    # ---------------------------------------------------------------- 检索

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        where: dict[str, Any] | None = None,
        min_score: float | None = None,
    ) -> SearchResult:
        """检索：查询文本走**与入库完全相同**的编码路径.

        同一个 ``embedding`` 实例、同一个 ``embed`` 方法——这一点不是细节：
        查询侧与写入侧用了不同的编码器（或不同的归一化口径），
        检索结果会以"排序不太对"的形式表现出来，而不会报任何错。

        维度在进入后端之前先查一次，是为了让错误发生在**最靠近病因**的地方
        （查询文本 → 向量这一步），而不是等后端在排序时才发现；
        报错信息与 ``base.query`` 的那一条口径一致。
        """
        vector = self.embedding.embed(query)
        if self.backend.has_dimension and len(vector) != self.backend.dimension:
            raise VectorError(
                f"查询向量是 {len(vector)} 维，本库是 {self.backend.dimension} 维："
                "查询用的编码器与建库用的不是同一个。"
                "出路是用建库时的编码器重新编码查询，或重建索引——"
                "截断或补零都会让排序静默失去意义。"
            )
        return self.backend.query(
            vector,
            self._resolve_top_k(top_k),
            where=where,
            min_score=self._resolve_min_score(min_score),
        )

    # ---------------------------------------------------------------- 元信息

    def stats(self) -> dict[str, Any]:
        """当前配置与状态（端点 ``/vectorstore/stats`` 直接回显它）.

        ``embedding_input_field`` 报的是**优先字段**（缺它才回落到 ``text``），
        因此它是一个常量式的承诺，而不是某一次摄取的统计量——
        逐条的实际情况由 ``embedding_input_field(record)`` 现问现答。
        """
        return {
            "backend": self.backend.name,
            "metric": self.backend.metric,
            "dimension": self.backend.dimension,
            "count": self.backend.count(),
            "top_k": self._resolve_top_k(None),
            "min_score": self._resolve_min_score(None),
            "embedding": type(self.embedding).__name__,
            "embedding_dimension": self.embedding.dimension,
            "embedding_input_field": EMBEDDING_TEXT_FIELD,
        }

    # ---------------------------------------------------------------- 默认值

    def _resolve_top_k(self, override: int | None) -> int:
        """``top_k`` 的三级优先级：调用参数 > 构造参数 > ``settings``."""
        if override is not None:
            return int(override)
        if self._top_k is not None:
            return int(self._top_k)
        return int(settings.vector_default_top_k)

    def _resolve_min_score(self, override: float | None) -> float | None:
        """``min_score`` 的三级优先级：调用参数 > 构造参数 > ``settings``.

        ``None`` 是"不设阈值"而不是"阈值 0"：在 ``l2`` 度量下分数是负数，
        阈值 0 会把**全部命中**都切掉——这也是 ``settings.vector_min_score``
        的默认值必须是 ``None`` 的原因。
        """
        if override is not None:
            return float(override)
        if self._min_score is not None:
            return float(self._min_score)
        default = settings.vector_min_score
        return None if default is None else float(default)


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #


def _metadata_of(record: dict[str, Any]) -> dict[str, Any]:
    """取记录的 ``metadata``；不是字典时当成空字典（并让后续校验去响）."""
    metadata = record.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _field_text(record: dict[str, Any], field: str) -> str:
    """按字段名取文本：先看 ``metadata``，再看记录顶层.

    两个位置都查，是因为"语义上属于元数据"的 ``retrieval_text`` 由 day062
    放在 ``metadata`` 里，而 ``text`` 在顶层——**同一个取值函数要能同时
    服务这两种位置**，否则调用方得记住"哪个字段在哪一层"，
    而这种记忆会在第一次换产出方时失效。非字符串值一律 ``str()``。
    """
    metadata = _metadata_of(record)
    if field in metadata:
        value: Any = metadata[field]
    elif field in record:
        value = record[field]
    else:
        return ""
    return "" if value is None else str(value)


def _dimension_failure(expected: int, actual: int, record_id: str) -> str:
    """维度不一致的失败说明（必须点名"维度"，且给出出路）."""
    return (
        f"失败：维度不一致：本库是 {expected} 维，本次编码得到 {actual} 维"
        f"（记录 {record_id!r}）。维度不同说明库与当前编码器不是同一套——"
        "换过 embedding 提供方就要重新编码并重建索引，"
        "不要截断或补零（那会让排序结果静默地失去意义）。"
    )


__all__ = [
    "EMBEDDING_TEXT_FIELD",
    "FALLBACK_TEXT_FIELD",
    "RECORD_ID_FIELD",
    "IngestReport",
    "VectorIngestPipeline",
    "embedding_input_field",
    "embedding_text",
    "record_from_knowledge",
    "record_id_of",
]
