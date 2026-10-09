"""向量库实现之间的对账：同一批记录、同一个查询，几个实现必须给出同一份排名（M6-D3）.

day062 的 ``chunking/evaluate.py`` 评估"哪种分块策略检索得更准"；
本模块回答一个更前置的问题：**几个后端是不是在回答同一个问题**。

```text
flat      ``metrics.score`` 的直译            ← 参照实现（谁都得与它一致）
faiss     归一化向量的内积（借来的等式）        ← 见 faiss_backend 的模块 docstring
chroma    原生 ``distances`` 折回 score        ← 它的"距离"与 FAISS 的方向相反
```

三者给出同一份排名**不是自然事实**，而是三处独立约定恰好对齐的结果：

```text
度量的公式   cos / dot / −Σ(a−b)²               → metrics.py 一处定义
距离的方向   Chroma 越小越近、FAISS 越大越近    → similarity_from_distance 一处换算
平局的断法   分数降序、id 升序                  → types.sort_hits 一处实现
```

任何一处没对齐，症状都**不是报错**，而是"换了个后端之后检索效果变差了"。
所以对账的产物必须能回答"第几名开始不同、期望是谁、实际是谁"——
只说一句"结果不一致"等于什么都没说（见 ``first_divergence`` 的措辞）。

## 参照实现必须是独立的（本模块最重要的取舍）

真正的参照实现是 ``tests/vectorstore_samples.brute_force_top``：
一段只用 ``math`` 重写三种度量公式、**不复用** ``metrics.py`` 的朴素代码。
判定的依据必须与被判定的实现相互独立，否则它只是在复述实现。

但 ``evaluate.py`` 是生产代码，**不能** ``import tests``（测试件不该进生产依赖）。
因此它接受一个 ``reference`` 参数（``[(id, score), ...]``），由调用方注入：

```text
传了 reference   → 拿它当期望值（测试注入 brute_force_top 或手工排出的排名）
没传 reference   → 用 ``metrics.score`` 现算一遍（下称"降级参照"）
```

**降级参照为什么弱**（这条必须记住）：它算出来的就是被测后端同一套公式、
同一份输入、同一个排序函数的结果。于是"一致"只证明**它等于它自己**——

```text
flat 的 _ranked_ids 调用的正是 metrics.score   → agreed 恒等于 top_k
"实现里少排了一条"  → 降级参照会与后端一起少排（它读的也是同一批记录）
"平局的断法反了"    → 降级参照与后端共享同一个排序键，会一起反
```

所以降级路径产出的"全绿"没有任何信息量，它只适合"给人临时看一眼"，
**不能作为 CI 的判据**：真正的对账必须显式传入独立实现算出的期望值
（``tests/test_vectorstore_parity.py`` 里每一条对账都这么写）。

## 两个指标各回答一半问题

```text
rank_agreement   逐位相同的前缀长度   → "顺序对不对"
recall_at_k      前 k 条的集合交叠    → "选出来的这批是不是同一批"
```

只用其中一个都会漏：只看 ``agreed`` 会把"集合相同、顺序不同"当成严重不一致
（它可能只是一个同分平局的断法差异）；只看 ``overlap`` 则完全看不见顺序错乱
（``overlap == 1.0`` 而 ``agreed == 0`` 是一种真实且危险的状态）。
两个一起看才能分清"选错了"和"排错了"。

## compare_backends 不依赖 registry

``registry`` 由 D 号任务同刻开发（它负责"按名字拿到一个后端"与
"缺依赖时给出三段式安装指引"）。本模块只负责**对账**，因此它**不 import
registry**，而是直接接收：

```text
VectorBackend 实例            → 用它本身（度量在构造时定死，只对它那个度量出一行）
可调用工厂 factory(*, metric, dimension, path) → 每个度量建一个新实例
(名称, 实例或工厂) 二元组      → 显式指定报告里的名字
```

"可选依赖缺失"的判据也随之从 ``registry.is_available``（只能看 import 得进不进）
换成**"能不能真的构造出一个实例来"**——后者才是真正要回答的问题：
装了 chromadb 但没有 PersistentClient 的安装，``is_available`` 会说"可用"，
而构造会直接失败。

## 这一层刻意不做的事

```text
不量化近似索引的召回损失   → 那是"过采样 + recall 实验"，属于索引调优
不测延迟/内存/构建耗时     → evaluation.perf_baseline 已有专门的基线表
不替调用方发现后端         → 见上：只接受注入的实例或工厂
不修改记录本身             → 对账是读路径；写库只发生在"这批记录还没进去"的后端上
```

## 谁依赖它

```text
tests/test_vectorstore_parity.py        本模块的全部断言都在那里
/vectorstore/parity 端点（装配阶段）      直接 json.dumps(verify_parity(rows)) 就能返回
```

``compare_metrics`` 有一条值得先记住的事实（它解释了样本为什么要造两遍）：
``types.make_record`` **只在 ``cosine`` 下做 L2 归一化**，因此
用 ``sample_records(metric="cosine")`` 构造的数据里所有向量都已经是单位长度——
此时 ``ip`` 与 ``cosine`` 的排名**完全相同**，看不出两个度量的差别。
要看见差别，必须用 ``sample_records(metric="ip")``（不归一化）构造：
那时 ``r_d`` 的 10 倍模长才会把 ``ip`` 的 top-1 顶到另一条记录上。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from smart_research_agent.vectorstore.base import VectorBackend
from smart_research_agent.vectorstore.errors import BackendUnavailable, VectorStoreError
from smart_research_agent.vectorstore.flat import FlatVectorStore
from smart_research_agent.vectorstore.metrics import normalize_metric
from smart_research_agent.vectorstore.metrics import score as metric_score

#: 缺省的对账深度。与 ``base.DEFAULT_TOP_K`` 取同一个值不是巧合：
#: 对账的深度与线上检索的深度不同，"逐位一致"这句话就不再指向同一件事。
DEFAULT_PARITY_TOP_K = 5

#: 分数比较的绝对容差（**名次仍然必须逐位相同**）.
#: flat 用 ``math.fsum``、FAISS 的 C++ 内核按 float32 累加、Chroma 存 float32
#: 再读回来，于是同一个查询可能给出 ``0.9999999999999998`` 与 ``1.0``——
#: 这两个数的名次完全一致，用 ``==`` 比较只会换来一堆假失败。
#: 反过来不成立：**任何名次差异都不能用容差解释**，因为本包的排序规则
#: （分数降序、id 升序）是确定的（见 ``types.sort_hits``）。
SCORE_TOLERANCE = 1e-6

#: 一份参照排名：``[(记录 id, 分数), ...]``，**已按分数降序、id 升序**.
ReferenceRanking = Sequence[tuple[str, float]]
#: 参照的两种给法：单度量给一份排名；多度量给 ``{度量: 排名}``.
ReferenceLike = ReferenceRanking | Mapping[str, ReferenceRanking]
#: 能构造出一个后端的可调用对象（签名必须是 ``(*, metric, dimension, path)``）.
BackendFactory = Callable[..., VectorBackend]
#: 后端规格的三种写法（见模块 docstring 的"compare_backends 不依赖 registry"）.
BackendSpec = VectorBackend | BackendFactory | tuple[str, VectorBackend | BackendFactory]


@dataclass(frozen=True)
class ParityRow:
    """一个 ``(后端, 度量)`` 组合与参照实现的对账结论（一行就是一句可读的话）.

    字段分成三组，服务三种读者：

    ```text
    身份   backend / metric / available       → 这一行说的是谁
    规模   count / top_k                      → 在什么条件下比
    结论   agreed / overlap / first_divergence / note → 比出了什么
    ```

    ``available=False`` 的行是一等公民而不是异常：**装了 faiss 的机器与
    没装的机器，对账报告必须长得一样**（只差一个布尔），否则并行的两个人
    会拿到两份无法比较的报告（见 ``compare_backends``）。
    """

    backend: str
    metric: str
    available: bool
    count: int
    top_k: int
    #: 与参照实现**逐位相同的前缀长度**（``rank_agreement`` 的返回值）.
    agreed: int
    #: recall@k：参照的前 k 条里有多少出现在候选的前 k 条里（集合口径）.
    overlap: float
    #: 空串表示逐位一致；否则形如 ``"第 2 名：期望 '1c9f…'，实际 '7b3e…'"``.
    first_divergence: str
    #: 一句话补充：不可用时的原因（含安装指引），可用时是本次的 top-1 是谁.
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（端点直接用）."""
        return {
            "backend": self.backend,
            "metric": self.metric,
            "available": self.available,
            "count": self.count,
            "top_k": self.top_k,
            "agreed": self.agreed,
            "overlap": round(self.overlap, 6),
            "first_divergence": self.first_divergence,
            "note": self.note,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要（演示脚本与报告逐行打印）."""
        if not self.available:
            return f"{self.backend:<8} | {self.metric:<6} | 不可用 | {self.note or '缺少可选依赖'}"
        expected = min(self.top_k, self.count)
        state = "逐位一致" if self.agreed >= expected else f"首个分歧在第 {self.agreed + 1} 名"
        return (
            f"{self.backend:<8} | {self.metric:<6} | {self.count} 条 | top-{self.top_k} | "
            f"一致 {self.agreed}/{expected} | 交叠 {self.overlap:.1%} | {state} | {self.note}"
        )


def scores_within_tolerance(left: float, right: float) -> bool:
    """两个分数是否在 ``SCORE_TOLERANCE`` 之内相同（**只用于分数，不用于名次**）.

    它的唯一用途是把"数值精度差异"与"真的排错了"分开——见
    ``_first_divergence``：当两个 id 不同而分数在容差内时，那句分歧说明里
    会多一句"差异来自平局断法"，因为那才是下一步该去看的地方。
    """
    return abs(float(left) - float(right)) <= SCORE_TOLERANCE


def recall_at_k(
    reference_ids: Sequence[str], candidate_ids: Sequence[str], k: int
) -> float:
    """recall@k：**参照的前 k 条里有多少出现在候选的前 k 条里**.

    三个边界都返回 ``0.0`` 而不是抛异常，因为它们都是"照实说没有可比的东西"：

    ```text
    k <= 0            没有要看的深度
    参照为空          没有"应该出现的东西"
    一条都不交叠      真实结论就是 0
    ```

    分母取 ``len(reference[:k])`` 而不是 ``k``：参照不足 k 条时（库里记录少、
    参照实现被截断），用 ``k`` 当分母会把"只看得到 2 条、2 条都在"报成 0.4
    ——那个数字与被测实现无关，是参照自己短了。
    """
    if k <= 0:
        return 0.0
    reference = list(reference_ids)[:k]
    if not reference:
        return 0.0
    candidate = set(list(candidate_ids)[:k])
    matched = sum(1 for record_id in reference if record_id in candidate)
    return matched / len(reference)


def rank_agreement(reference_ids: Sequence[str], candidate_ids: Sequence[str]) -> int:
    """逐位相同的前缀长度（第几名开始不同）.

    为什么是**前缀**而不是"相同位置的个数"：一旦第 3 名不同，
    第 6 名的"相同"已经不再意味着同一件事——两个排名的尾部是从不同的
    分支长出来的，把它们逐位比下去只会得到一个既不是 0 也不是全长的数，
    而这个数没法指导任何人去看哪一行。前缀长度则直接回答了
    "前面几个名次可以信"。

    候选比参照短时按短的那个截断：``[a, b, c]`` 与 ``[a]`` 的一致长度是 1，
    缺的那两条由 ``recall_at_k`` 去说（它会给出 1/3），两个指标各管一半。
    """
    agreed = 0
    for expected, actual in zip(reference_ids, candidate_ids):
        if expected != actual:
            break
        agreed += 1
    return agreed


def compare_backends(
    records: Sequence[Any],
    query: Sequence[float],
    *,
    metrics: Sequence[str] = ("cosine",),
    top_k: int = DEFAULT_PARITY_TOP_K,
    backends: Sequence[BackendSpec] | Mapping[str, BackendSpec] | None = None,
    reference: ReferenceLike | None = None,
) -> list[ParityRow]:
    """同一批记录 × 同一个查询 → 每个 ``(后端, 度量)`` 一行对账结论.

    ``backends`` 的三种写法、以及"为什么不用 registry"，都写在模块 docstring 里。
    缺省（``None``）只对 ``flat`` 自己：**这是刻意的**——缺省配置下不存在
    "两个实现互相对账"这件事，它只给你一份"参照实现的自述"。
    真正的对账必须显式注入第二个后端。

    行序是**度量优先**：同一个度量下各后端的行连在一起。
    对账问的是"同一个度量下几个后端一致吗"，按度量分组才好读。

    错误处理只有一条规则（实现在 ``_compare_one``）：

    ```text
    BackendUnavailable / ImportError  → 折成一行 available=False（报告在任何机器上都齐）
    其它任何异常                        → 原样上抛（把真 bug 记成一行会让对账假绿）
    ```

    ``records`` 会被写进每一个新构造出来的后端（``upsert``，幂等）。
    传入的后端**实例**同样会被写一次：这样"对账前先保证比的是同一批数据"
    这件事由本函数负责，而不是靠调用方记得自己先 load 一遍。
    """
    prepared = list(records)
    if not prepared:
        raise ValueError(
            "对账至少需要一条记录：没有记录就没有任何排名可以对照，"
            "此时返回空列表会让人误以为'后端之间一致'。"
        )
    if top_k < 1:
        raise ValueError(f"top_k 必须 >= 1，收到 {top_k}（要 0 条不是一个有意义的检索请求）")
    resolved_metrics = [normalize_metric(name) for name in metrics]
    if not resolved_metrics:
        raise ValueError("至少要给出一个度量，否则一行报告都产不出来")

    dimension = prepared[0].dimension
    specs = _specs_as_list(backends)
    rows: list[ParityRow] = []
    for metric in resolved_metrics:
        for spec in specs:
            hint, target = _resolve_spec(spec)
            if isinstance(target, VectorBackend) and target.metric != metric:
                # 实例的度量在构造时就定死了（见 base.metric 的说明：不允许运行期切换）。
                # 与其拿一个 cosine 的库冒充 ip 的结果，不如不出这一行。
                continue
            rows.append(
                _compare_one(
                    target,
                    hint,
                    prepared,
                    query,
                    metric=metric,
                    top_k=top_k,
                    dimension=dimension,
                    reference=reference,
                )
            )
    return rows


def compare_metrics(
    records: Sequence[Any],
    query: Sequence[float],
    *,
    top_k: int = DEFAULT_PARITY_TOP_K,
    metrics: Sequence[str] = ("cosine", "ip", "l2"),
    backend: BackendSpec | None = None,
    reference: ReferenceLike | None = None,
) -> list[ParityRow]:
    """同一个后端 × 三种度量 → 每个度量一行，回答"换个度量名次变了几个".

    参照的口径：**以 ``metrics`` 里第一个被对账的度量为基线**。

    ```text
    传了 reference        → 每个度量各用自己那份期望值（Mapping 按度量给）
    没传 reference        → 基线的期望值由 metrics.score 现算（降级参照，弱，见模块 docstring），
                          其余度量则与"基线在本后端上的排名"对照
    ```

    为什么"其余度量与基线对照"是对的：这一层的产品化的那句话是
    "**换度量会换掉答案**"，而这句话只有在"同一份数据、同一个实现、
    只换度量"的条件下才说得通。所以 ``ip`` 那一行的 ``first_divergence``
    会写成 ``第 1 名：期望 <cosine 的 top-1>，实际 <ip 的 top-1>``——
    它直接就是一条可以贴进选型记录的结论。

    两个必须记住的事实：

    ```text
    用 sample_records(metric="cosine") 构造   → 向量已归一化，ip 与 cosine 名次相同
    用 sample_records(metric="ip") 构造       → 未归一化，ip 的 top-1 会被模长顶走
    ```

    ``backend`` 与 ``compare_backends`` 接受同样的三种写法；缺省是
    ``FlatVectorStore``（零可选依赖的参照实现，任何机器上都能跑）。
    """
    prepared = list(records)
    if not prepared:
        raise ValueError("对账至少需要一条记录：没有记录就没有任何排名可以对照")
    if top_k < 1:
        raise ValueError(f"top_k 必须 >= 1，收到 {top_k}")
    resolved_metrics = [normalize_metric(name) for name in metrics]
    if not resolved_metrics:
        raise ValueError("至少要给出一个度量，否则一行报告都产不出来")

    hint, target = _resolve_spec(backend if backend is not None else FlatVectorStore)
    dimension = prepared[0].dimension
    rows: list[ParityRow] = []
    #: 基线排名（第一个真正被对账的度量的期望值），后续度量与它对照.
    anchor: list[tuple[str, float]] | None = None
    for metric in resolved_metrics:
        if isinstance(target, VectorBackend) and target.metric != metric:
            continue
        try:
            store, backend_name = _build_store(target, metric, dimension)
            store.upsert(prepared)
        except (BackendUnavailable, ImportError) as exc:
            rows.append(_unavailable_row(hint, metric, top_k, str(exc)))
            continue
        result = store.query(query, top_k)
        candidate = [(hit.record.record_id, float(hit.score)) for hit in result.hits]
        expected = _resolve_reference(reference, metric)
        if expected is None and anchor is not None:
            expected = anchor
        if expected is None:
            # 基线：由包内朴素打分现算（降级参照，见模块 docstring 的"为什么弱"）.
            expected = _naive_reference(prepared, query, metric, top_k)
        expected = list(expected)[:top_k]
        if anchor is None:
            anchor = list(expected)
        rows.append(
            _build_row(
                backend=backend_name,
                metric=store.metric,
                count=store.count(),
                top_k=top_k,
                expected=expected,
                candidate=candidate,
            )
        )
    return rows


def verify_parity(rows: Sequence[ParityRow]) -> dict[str, Any]:
    """把若干行对账结论收敛成一个"能不能过"的判断（端点与 CI 的入口）.

    返回 ``{"ok", "rows", "failures", "checked", "available"}``：

    ```text
    ok        所有行都可用、且逐位一致、且集合交叠为 1.0
    rows      逐行 to_dict()（可直接 json.dumps，端点就是这么返回的）
    failures  每一句都写清是哪个后端/度量、错在哪一名（可照着修）
    ```

    ``failures`` 用**句子的形式**而不是错误码：对账报告是给人看的，
    而"faiss/cosine 名次与参照不一致（一致 3/5）：第 4 名 期望 X 实际 Y"
    这一句里已经包含了定位与现象。

    额外两个计数（``checked`` / ``available``）不是装饰：一个"因为全都不可用
    所以没有失败项"的报告必须是可见的，否则 ``ok`` 会在什么都没比的情况下为真。
    """
    prepared = list(rows)
    failures: list[str] = []
    for row in prepared:
        if not row.available:
            failures.append(
                f"{row.backend}/{row.metric} 不可用：{row.note or '缺少可选依赖'}"
            )
            continue
        usable = min(row.top_k, row.count)
        if row.agreed < usable:
            failures.append(
                f"{row.backend}/{row.metric} 名次与参照不一致"
                f"（一致 {row.agreed}/{usable}）：{row.first_divergence}"
            )
        elif row.overlap < 1.0 - SCORE_TOLERANCE:
            failures.append(
                f"{row.backend}/{row.metric} 前 {row.top_k} 条的集合对不上"
                f"（交叠 {row.overlap:.1%}）：{row.first_divergence}"
            )
    return {
        "ok": not failures,
        "rows": [row.to_dict() for row in prepared],
        "failures": failures,
        "checked": len(prepared),
        "available": sum(1 for row in prepared if row.available),
    }


def index_health(backend: VectorBackend) -> dict[str, Any]:
    """给一个后端做四项体检，返回可读结论（``/vectorstore/health`` 的取材）.

    ``base.py`` 的模块 docstring 说"一个向量库其实是两个库：向量索引 + 记录表"，
    而那个设计最典型的事故形态是**索引里有、记录表里没有**：检索会命中一个
    没有原文的 id，``base.query`` 遇到它会跳过并继续——于是症状只是
    "结果少了一条"，没有任何异常指向"删除没做干净"或"两个文件不是一次写出的"。

    因此这四项各自盯一种不同步（都是查得出来、但平时看不见的）：

    ```text
    count        count() 与 ids() 的长度是否一致（两端各说一个数时，信谁都不对）
    unique_ids   ids() 里有没有重复（重复会让"条数"与"去重后的条数"分家）
    dimension    每条记录的维度是否与 backend.dimension 一致（换编码器未重建库）
    retrievable  ids() 里的每一个都能被 get_many 取回来（**上面那条事故形态**）
    ```

    ``healthy`` 是四项的与；``checks`` 里每一项都带一句人类可读的 detail，
    因为"哪一项没过"比"过没过"重要得多。
    """
    ids = list(backend.ids())
    count = backend.count()
    duplicated = sorted(
        record_id for record_id, seen in Counter(ids).items() if seen > 1
    )
    records = backend.get_many(ids)
    found = {record.record_id for record in records}
    missing = [record_id for record_id in ids if record_id not in found]
    dimension = backend.dimension
    mismatched = (
        [
            f"{record.record_id}={record.dimension}d"
            for record in records
            if record.dimension != dimension
        ]
        if dimension
        else []
    )

    checks = [
        {
            "check": "count",
            "ok": count == len(ids),
            "detail": f"count()={count}，ids()={len(ids)}",
        },
        {
            "check": "unique_ids",
            "ok": not duplicated,
            "detail": (
                "无重复 id"
                if not duplicated
                else f"重复 id：{', '.join(duplicated[:3])}"
            ),
        },
        {
            "check": "dimension",
            "ok": not mismatched,
            "detail": (
                "维度未定（空库）"
                if not dimension
                else (
                    f"全部 {dimension} 维"
                    if not mismatched
                    else f"维度不一致：{', '.join(mismatched[:3])}"
                )
            ),
        },
        {
            "check": "retrievable",
            "ok": not missing,
            "detail": (
                f"{len(ids)} 个 id 全部取回"
                if not missing
                else f"记录表里取不回：{', '.join(missing[:3])}"
                "（向量索引里有、记录表里没有 → 检索会静默少一条）"
            ),
        },
    ]
    healthy = all(bool(item["ok"]) for item in checks)
    failed = [str(item["check"]) for item in checks if not item["ok"]]
    return {
        "backend": backend.name,
        "metric": backend.metric,
        "dimension": dimension,
        "count": count,
        "unique_ids": not duplicated,
        "duplicate_ids": duplicated,
        "missing_ids": missing,
        "dimension_mismatches": mismatched,
        "checks": checks,
        "healthy": healthy,
        "summary": (
            "四项体检全部通过"
            if healthy
            else f"未通过：{', '.join(failed)}"
        ),
    }


# ---------------------------------------------------------------------------- #
# 内部工具
# ---------------------------------------------------------------------------- #


def _specs_as_list(
    backends: Sequence[BackendSpec] | Mapping[str, BackendSpec] | None,
) -> list[BackendSpec]:
    """把 ``backends`` 的两种容器写法收敛成列表（``{名称: 工厂}`` 也允许）."""
    if backends is None:
        return [FlatVectorStore]
    if isinstance(backends, Mapping):
        return [(name, target) for name, target in backends.items()]
    return list(backends)


def _resolve_spec(spec: BackendSpec) -> tuple[str, VectorBackend | BackendFactory]:
    """把一份后端规格拆成 ``(名称提示, 实例或工厂)``；不认识的写法直接报错.

    名称提示只在"构造失败、拿不到实例的 ``name``"时才会被写进报告，
    因此它允许是个粗略的名字（裸工厂就用其函数名）——但**必须有**，
    否则一份不可用的后端在报告里会变成一行没有归属的空白。
    """
    if isinstance(spec, tuple):
        if len(spec) != 2:
            raise VectorStoreError(
                f"后端规格写成 (名称, 实例或工厂) 的二元组，收到长度 {len(spec)} 的元组"
            )
        name, target = spec
        if not isinstance(target, VectorBackend) and not callable(target):
            raise VectorStoreError(
                f"后端规格 {name!r} 的第二项必须是 VectorBackend 实例或可调用工厂，"
                f"收到 {type(target).__name__}"
            )
        return str(name), target
    if isinstance(spec, VectorBackend):
        return spec.name, spec
    if callable(spec):
        return str(getattr(spec, "__name__", "backend")), spec
    raise VectorStoreError(
        f"无法识别的后端规格 {spec!r}：请传 VectorBackend 实例、"
        "可调用工厂（关键字 metric=/dimension=/path=），或 (名称, 实例或工厂) 二元组。"
    )


def _build_store(
    target: VectorBackend | BackendFactory,
    metric: str,
    dimension: int,
) -> tuple[VectorBackend, str]:
    """按规格造出一个后端实例；返回 ``(实例, 报告里的名字)``.

    ``path=""`` 一律不落盘：对账是**内存里的一次计算**，让它在磁盘上留下
    三个后端的快照，只会给下一次"这个目录怎么多了几个文件"增加一个谜。

    名字取 ``store.name``（后端自己声明的，flat/faiss/chroma）而不是规格里的
    名称提示：类对象形式的工厂（``FlatVectorStore``）其 ``__name__`` 是
    ``"FlatVectorStore"``，而报告里应该写的是 ``"flat"``。
    """
    if isinstance(target, VectorBackend):
        return target, target.name
    store = target(metric=metric, dimension=dimension, path="")
    return store, store.name


def _resolve_reference(
    reference: ReferenceLike | None, metric: str
) -> list[tuple[str, float]] | None:
    """取出某个度量的参照排名；没给就返回 ``None``（交给降级路径）.

    ``Mapping`` 里缺这个度量时**直接报错**而不是回退到降级参照：
    回退会让一次"照着 cosine 的期望值去核对 ip"的对账看起来全绿，
    而那正是 E.3.4 想抓住的那类静默错误。
    """
    if reference is None:
        return None
    if isinstance(reference, Mapping):
        value = reference.get(metric)
        if value is None:
            raise VectorStoreError(
                f"参照实现没有给出度量 {metric!r} 的期望排名："
                f"参照里只有 {sorted(reference)}。请为每个参与对账的度量各给一份，"
                "不要拿另一套度量的期望值充数。"
            )
        return list(value)
    return list(reference)


def _naive_reference(
    records: Sequence[Any],
    query: Sequence[float],
    metric: str,
    top_k: int,
) -> list[tuple[str, float]]:
    """降级参照：用 ``metrics.score`` 现算一遍朴素排名（**弱，见模块 docstring**）.

    它与 ``flat._ranked_ids`` 是同一套公式、同一个排序键，因此
    ``flat`` 与它的 ``agreed`` 必然等于 ``top_k``——**这不是"通过了对账"，
    只是"读了同一段代码两遍"**。
    """
    scored = [
        (record.record_id, metric_score(metric, query, record.vector))
        for record in records
    ]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:top_k]


def _first_divergence(
    expected: Sequence[tuple[str, float]],
    candidate: Sequence[tuple[str, float]],
) -> str:
    """第一个不同的名次，写成"第几名：期望谁、实际谁".

    措辞里必须同时有"期望"与"实际"两个词（测试直接断言这两个片段）：
    对账的价值在于指名道姓，一句"结果不一致"对读报告的人没有任何帮助。

    两边长度不等时用 ``（缺失）`` 占位：候选比参照短，本身就是一个
    必须被看见的事实（通常是库里记录不足或被阈值切掉了）。
    """
    for position in range(max(len(expected), len(candidate))):
        left = expected[position] if position < len(expected) else None
        right = candidate[position] if position < len(candidate) else None
        if left is not None and right is not None and left[0] == right[0]:
            continue
        message = (
            f"第 {position + 1} 名：期望 {left[0]!r}，实际 {right[0]!r}"
            if left is not None and right is not None
            else (
                f"第 {position + 1} 名：期望 {left[0]!r}，实际 （缺失）"
                if left is not None
                else f"第 {position + 1} 名：期望 （缺失），实际 {right[0]!r}"
            )
        )
        if left is not None and right is not None and scores_within_tolerance(
            left[1], right[1]
        ):
            message += (
                f"（两者分数差在 {SCORE_TOLERANCE:g} 的容差内："
                "名次差异来自平局断法，不是数值精度）"
            )
        return message
    return ""


def _compare_one(
    target: VectorBackend | BackendFactory,
    hint: str,
    records: Sequence[Any],
    query: Sequence[float],
    *,
    metric: str,
    top_k: int,
    dimension: int,
    reference: ReferenceLike | None,
) -> ParityRow:
    """一个 ``(后端, 度量)`` 组合的完整对账（``compare_backends`` 的单步）."""
    try:
        store, backend_name = _build_store(target, metric, dimension)
        store.upsert(records)
    except (BackendUnavailable, ImportError) as exc:
        # 只有"缺依赖 / 服务不可达"这一族被折成一行；其它异常一律上抛——
        # 把真 bug 记成一行报告，会让对账在没有比过任何东西的情况下变绿。
        return _unavailable_row(hint, metric, top_k, str(exc))

    result = store.query(query, top_k)
    candidate = [(hit.record.record_id, float(hit.score)) for hit in result.hits]
    expected = _resolve_reference(reference, metric)
    if expected is None:
        expected = _naive_reference(records, query, metric, top_k)
    return _build_row(
        backend=backend_name,
        metric=store.metric,
        count=store.count(),
        top_k=top_k,
        expected=list(expected)[:top_k],
        candidate=candidate,
    )


def _build_row(
    *,
    backend: str,
    metric: str,
    count: int,
    top_k: int,
    expected: Sequence[tuple[str, float]],
    candidate: Sequence[tuple[str, float]],
) -> ParityRow:
    """把两份排名折成一行报告（两个指标 + 第一处分歧 + top-1）."""
    expected_ids = [record_id for record_id, _ in expected]
    candidate_ids = [record_id for record_id, _ in candidate]
    return ParityRow(
        backend=backend,
        metric=metric,
        available=True,
        count=count,
        top_k=top_k,
        agreed=rank_agreement(expected_ids, candidate_ids),
        overlap=recall_at_k(expected_ids, candidate_ids, top_k),
        first_divergence=_first_divergence(expected, candidate),
        note=f"top-1={candidate_ids[0]}" if candidate_ids else "top-1=（空）",
    )


def _unavailable_row(
    backend: str, metric: str, top_k: int, reason: str
) -> ParityRow:
    """不可用的那一行：``available=False``、``agreed=0``、``first_divergence="后端不可用"``.

    三个取值都是固定的（不是随手填的）：报告的形状必须与机器无关，
    这样"这台机器没装 faiss"才只表现为一个布尔与一句原因，
    而不是一份少了几行、没法与别人对照的报告。
    """
    return ParityRow(
        backend=backend,
        metric=metric,
        available=False,
        count=0,
        top_k=top_k,
        agreed=0,
        overlap=0.0,
        first_divergence="后端不可用",
        note=reason,
    )


__all__ = [
    "DEFAULT_PARITY_TOP_K",
    "SCORE_TOLERANCE",
    "BackendFactory",
    "BackendSpec",
    "ParityRow",
    "ReferenceLike",
    "ReferenceRanking",
    "compare_backends",
    "compare_metrics",
    "index_health",
    "rank_agreement",
    "recall_at_k",
    "scores_within_tolerance",
    "verify_parity",
]
