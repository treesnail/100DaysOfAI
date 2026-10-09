"""``faiss`` 后端：**一个只认 (int64, float32) 的向量索引 + 一张本类持有的记录表**（M6-D3）.

``base.py`` 的模块 docstring 说"一个向量库其实是两个库"。FAISS 把这句话
摊得最开——它连 id 都不替我们存：

```text
faiss.IndexFlatIP / IndexFlatL2     只存 float32 向量；检索返回的是"行号"
faiss.IndexIDMap2(flat)             补一层 id 映射：add_with_ids / remove_ids / id_map
本类的 dict[str, VectorRecord]       存 text 与 metadata（FAISS 没有它们的位置）
磁盘上的两个文件                     索引文件 + <path>.records.json
```

因此本类的工作可以一句话说完：**把字符串 id 映射成 int64、把向量交给
FAISS、把原文与元数据留在自己手里**。所有"多出来的代码"都来自那张映射表。

## 谁是主：记录表

与 ``base.py`` 的约定一致，**记录表是主、向量索引是从**：

```text
删除   先按记录表确定"确实存在的那些 id"，再让索引跟随，最后拿索引的
       返回值与请求数对账（不等就报错——那是"两个库不同步"的证据）
查询   索引只回答"谁最近"，命中之后一律回记录表取原文与元数据
写入   先算出整批的计划，再动索引，最后提交记录表（不让半批数据落库）
```

反过来（以索引为主）会留下一种很难查的状态：索引里还有那条向量、
记录表里已经没有它的原文，于是检索命中一个"没有内容的 id"。

## 它放弃了什么（每一条都是可见的代价）

| 放弃的东西 | 具体代价 |
|-----------|---------|
| 原生元数据过滤 | 带 ``where`` 的检索要全量 ``search`` 一次再筛 |
| 字符串 id | 要做一次 sha256 映射，理论上可能撞号 |
| 维度可后定 | 构造时就必须给 ``dimension``，且建立后不可变 |
| 近似索引下的过滤 | 换成 IVF/HNSW 后"全量再筛"就不成立了 |
| 一份文件落盘 | 索引与记录表是两个文件，必须成对存在 |

上表的第一条不是工程折扣，而是**正确性问题**——下面单独说它。

## 过滤：唯一正确的做法是"先筛候选，再在候选里排序"

FAISS **没有任何元数据过滤能力**。于是"带 ``where`` 的检索"只有一条正确路径：
先按条件筛出候选 id，再在候选集里排序取前 K。反过来（先取全库 top-K、
再丢掉不符合条件的）会让**返回条数少于 K，尽管库里还有符合条件的记录
排在更后面**——这个 bug 不报错，只表现为"加了过滤之后结果变少了"。

``IndexFlat`` 是精确索引，因此这里可以走"全量 ``search`` + 过滤"这条笨办法：

```text
IndexFlat（本课）     search(x, ntotal) → 全量排序 → 过滤 → 取前 limit   ✅ 精确
IVF / HNSW（未来）    search(x, ntotal) 会退化成暴力，等于放弃索引的意义
                       → 那时只能"过采样再承认近似"，这是换索引类型的真实代价
```

**这是"过滤"在向量库里最容易被忽略的一笔账**：过滤的正确性依赖索引
能给出全量排序，而所有为了速度而做的索引（IVF/HNSW/PQ）恰恰不保证它。

## id 映射：为什么是哈希，不是自增号

```text
自增号       需要一张全局计数器：进程重启后从 1 开始 → 同一个字符串 id 会拿到新号
哈希         纯函数：同一个 id 在任何进程、任何顺序下都得到同一个 int64
```

代价有两个，都必须写在明面上：**哈希不可逆**（反查要靠本类的
``_reverse`` 表，所以那张表不是缓存而是状态），以及**理论上会撞号**
（撞号必须报错而不是静默覆盖——否则一条记录会凭空消失，而库里没有任何异常）。

值域取 ``[1, 2**63-2]`` 而不是整个 int64：避开 ``-1``（FAISS 的
"此处没有结果"保留值，官方明确建议 id 不要用它）与 ``0``（"未分配"的常见哨兵）。

## 向量口径：``cosine`` 复用 ``IndexFlatIP``

FAISS 没有"余弦索引"这种东西。本包的 ``cosine`` 复用 ``IndexFlatIP``：
``base._prepare`` 在写入前已经按余弦口径把向量 L2 归一化、``base.query``
也把查询向量归一化了，于是"归一化向量的内积"就是余弦相似度。
**这个等式是借来的**——一旦有人绕过 ``base`` 直接往索引里塞未归一化的向量，
``cosine`` 就会静默退化成 ``ip``。

顺带一个必须记住的差别（换算错误的来源）：

```text
IndexFlatL2   D 是平方 L2 距离       越小越近   → score = −D
IndexFlatIP   D 是内积（相似度）      越大越近   → score =  D
Chroma        distances 一律越小越近   → score = 1 − distance（cosine/ip）
```

本类用 ``metrics.similarity_from_distance`` 做统一换算：**先按 FAISS 的口径
把 D 折成"Chroma 口径的距离"，再交给那个函数**。内积那一支是
``1 − (1 − D) = D``，看着像白转一圈，但它把"两个库的距离定义相反"
这件事写在了同一个地方（见 ``_to_score``）。

## 谁依赖它

```text
registry（D 号）        ``vector_backend="faiss"`` 时按 requires 决定可不可用
evaluate.verify_parity 与 flat 逐条对账（本模块的测试已经先用 brute_force_top 对过一遍）
tests/faiss_fakes.py   用假 faiss 模块覆盖全部逻辑，不需要真库、不需要联网
```

不被依赖的一点：**本模块不 import faiss**。模块级只有 ``numpy``
（它本来就是 faiss-cpu 的依赖）；``faiss`` 通过 ``faiss_module=``
注入或惰性导入，缺失时给的是"该装哪个包"，而不是一个裸的 ImportError。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from smart_research_agent.vectorstore.base import VectorBackend
from smart_research_agent.vectorstore.errors import (
    BackendUnavailable,
    VectorError,
    VectorStoreError,
)
from smart_research_agent.vectorstore.metrics import (
    METRIC_COSINE,
    METRIC_INNER_PRODUCT,
    METRIC_L2,
    similarity_from_distance,
)
from smart_research_agent.vectorstore.types import VectorRecord, WriteReport

#: 快照格式版本（与 flat 的快照各自独立：它们是两份不同的文件格式）.
#: 加字段不用升版，改字段含义/换编码必须升版——理由见 ``flat.SNAPSHOT_VERSION``。
SNAPSHOT_VERSION = 1

#: 记录表的旁挂文件名后缀。**索引文件本身不含任何字符串**，
#: 所以这个后缀不是"习惯"，而是"两个库"里第二个库的文件名。
RECORDS_SUFFIX = ".records.json"

#: int64 id 的上界（同时也是哈希取模的模数）：``2**63 - 2``.
#: 不取 ``2**63 - 1``（int64 最大值）是因为要留出 ``-1`` 这个保留值所在的
#: 语义空间；``to_int64_id`` 返回 ``[1, INT64_ID_MODULUS]``，永远不碰 ``-1``/``0``。
INT64_ID_MODULUS = 2**63 - 2

#: 合法 id 的下界（不用 0，理由见模块 docstring）.
INT64_ID_MIN = 1

#: FAISS 的"此处没有结果"保留值：``search`` 返回的 ``I`` 里用它填充空位。
FAISS_ID_NOT_FOUND = -1

#: 索引类型名（同时也是 ``describe_extra`` 里的取值）.
INDEX_TYPE_IP = "IndexFlatIP"
INDEX_TYPE_L2 = "IndexFlatL2"

#: 真库 ``MetricType`` 的取值（``faiss.METRIC_INNER_PRODUCT = 0`` / ``METRIC_L2 = 1``）.
#: 只在"接管一个外部索引"时用来核对度量：一个带错度量的索引会让**整个排序反向**，
#: 而症状只是"结果都不太像"，看不出是索引类型错了。
FAISS_METRIC_INNER_PRODUCT = 0
FAISS_METRIC_L2 = 1

#: 度量 → 索引类型。``cosine`` 与 ``ip`` 共用内积索引：
#: 余弦的归一化已经由 ``base`` 在写入与查询两侧做完（见模块 docstring）。
METRIC_TO_INDEX_TYPE: dict[str, str] = {
    METRIC_COSINE: INDEX_TYPE_IP,
    METRIC_INNER_PRODUCT: INDEX_TYPE_IP,
    METRIC_L2: INDEX_TYPE_L2,
}

#: 本模块要求 faiss 提供的符号。少一个就说明"装的不是我们写代码时对着的那个库"，
#: 因此在构造时就报出来——而不是等到第一次写入时抛一个 AttributeError。
REQUIRED_FAISS_API: tuple[str, ...] = (
    "IndexFlatIP",
    "IndexFlatL2",
    "IndexIDMap2",
    "IDSelectorBatch",
    "write_index",
    "read_index",
)


def _id_digest(record_id: str) -> bytes:
    """字符串 id 的哈希摘要（sha256）.

    单独抽成一个函数只为一件事：**让"撞号"这条分支可被测试**。
    ``sha256`` 的真实输出撞号概率约等于零，因此测试必须能替换掉它
    （monkeypatch 一个常量摘要），否则那一段"冲突必须报错"的代码
    永远只在纸面上存在。
    """
    return hashlib.sha256(record_id.encode("utf-8")).digest()


def to_int64_id(record_id: str) -> int:
    """字符串 id → 稳定的 int64（FAISS 只认整数 id）.

    ```text
    int64 = 1 + (sha256(id)[:8] 的大端无符号整数  %  (2**63 − 2))
    ```

    三个刻意的选择：

    - **纯函数而不是自增计数**：同一个 id 在任何进程、任何插入顺序下都得到
      同一个 int64。自增号会让"从快照恢复"与"重新导入"给出两套号，
      而索引文件里的号一旦与映射表对不上，检索会静默地返回空结果；
    - **取前 8 字节就够**：int64 只有 63 位可用，多取的部分不会增加区分度；
    - **值域 ``[1, 2**63-2]``**：避开 ``-1``（FAISS 的保留值，见
      ``FAISS_ID_NOT_FOUND``）与 ``0``（"未分配"的常见哨兵）。

    代价：哈希不可逆，反查必须靠调用方自己维护的 ``int64 → id`` 表；
    以及理论上会撞号——撞号由 ``FaissVectorStore`` 在写入前检出并报错。
    """
    digest = _id_digest(str(record_id))
    return INT64_ID_MIN + int.from_bytes(digest[:8], "big") % INT64_ID_MODULUS


def from_int64_id(value: int) -> int:
    """把 FAISS 返回的整数收敛成 Python ``int``，并挡掉两个非法取值.

    它**不是** ``to_int64_id`` 的逆（sha256 是单向的，逆要靠 ``_reverse`` 表）。
    它的用途是对账与测试：``search`` 回来的 ``I`` 是一个 ``int64`` 矩阵，
    里面混着两类东西——真实的记录 id，以及 ``-1`` 这个"此处没有结果"的填充。
    在把任何一格当成 id 用之前，必须先过这一关：

    ```text
    -1            → VectorError（它是保留值，不是记录：丢掉它，不是翻译它）
    值域之外       → VectorError（索引里的 id 不是本类写进去的，两边不是一套）
    ```

    返回值是 ``int`` 而不是 ``np.int64``：拿 numpy 标量当字典键虽然能用，
    但 ``np.int64`` 与 ``int`` 的哈希一致只是实现细节，**在库里混用两种
    整数类型是一个迟早会出问题的习惯**，在边界上收敛掉最省事。
    """
    resolved = int(value)
    if resolved == FAISS_ID_NOT_FOUND:
        raise VectorError(
            f"FAISS 的 {FAISS_ID_NOT_FOUND} 是「此处没有结果」的保留值，不是记录 id："
            "结果不足时 I 的尾部会用 -1 填充，必须先按 -1 丢弃，再拿剩下的去查记录表。"
        )
    if resolved < INT64_ID_MIN or resolved > INT64_ID_MODULUS:
        raise VectorError(
            f"int64 id {resolved} 落在本类的 id 值域 [{INT64_ID_MIN}, {INT64_ID_MODULUS}] "
            "之外：本类写进索引的 id 都经过 to_int64_id，出现值域外的 id 说明"
            "索引文件与本类的映射表不是一套（例如换了实现或换了后端）。"
        )
    return resolved


def _to_score(metric: str, value: float, *, similarity: bool) -> float:
    """把 FAISS 返回的数值折成本包的 ``score``（**越大越近**）.

    ``metrics.similarity_from_distance`` 的入参是"外部库的距离"
    （越小越近，与 Chroma 同名同义），而 FAISS 两种索引返回的口径不同：

    ```text
    IndexFlatL2   D = Σ(a−b)²        已经是距离            → 直接交进去
    IndexFlatIP   D = Σ(a·b)         是相似度（越大越近）  → 先折成 1 − D 再交进去
    ```

    ``1 − D`` 正是 ``metrics`` 里 Chroma 口径的 ``ip`` 距离，因此这一支
    写成"先统一口径，再统一换算"：**换算规则只有一处（metrics），
    两个库的距离方向差异只有一处（这里）**。若把它写成"IP 直接返回 D"，
    两条规则就散开在两个 if 里，而将来任何一次改动都只会改到其中一处。
    """
    distance = 1.0 - float(value) if similarity else float(value)
    return similarity_from_distance(metric, distance)


def resolve_faiss_module(module: Any | None = None) -> Any:
    """取到一个"能用的" faiss 模块：注入优先，其次惰性导入，最后核对 API 齐不齐.

    三级顺序对应三种真实场景：

    ```text
    module 给定    测试注入假模块；或调用方自己管依赖（例如多进程里预加载过）
    未给定         惰性 import：模块级 import 会让"没装 faiss"变成一个
                   连 flat 后端都用不了的 ImportError——一个后端不该拖垮两个
    都拿到之后     核对 REQUIRED_FAISS_API：版本不匹配要在构造点报出来，
                   而不是等到某次删除时才 AttributeError
    ```
    """
    if module is not None:
        resolved = module
    else:
        try:
            import faiss as resolved
        except ImportError as exc:
            raise BackendUnavailable(
                "未安装 faiss，无法使用 faiss 后端。请先执行：pip install faiss-cpu"
                "（Windows 有官方 wheel，不需要编译；装完 import faiss 验证一下）。"
                "不想装也可以换 vector_backend=flat（缺省值）或 chroma。"
            ) from exc
    missing = [name for name in REQUIRED_FAISS_API if not hasattr(resolved, name)]
    if missing:
        raise BackendUnavailable(
            f"faiss 模块缺少 {', '.join(missing)}：本包按 faiss-cpu 1.15 的 API 编写，"
            "缺少这些符号通常意味着本机装的是更旧的版本，或装成了另一个同名包。"
            "请执行 pip install -U faiss-cpu 后重试。"
        )
    return resolved


class FaissVectorStore(VectorBackend):
    """FAISS 索引 + 本类的记录表（见模块 docstring 的"两个库"）.

    六个原语全部在本类里实现，公共行为（校验、归一化、过滤、阈值、排序、
    写入报告）仍然由 ``VectorBackend`` 提供——**这与 flat 是同一套骨架**，
    因此两个后端的差异只可能来自"一次向量检索"和"一次持久化"。

    构造期的维度校验也由基类负责：``base.VectorBackend._check_dimension``
    是**唯一**的实现（见那里的说明），本类不再重复覆写它。
    """

    name = "faiss"

    #: 两个可选依赖：``numpy`` 写在这里不是为了"提示去装"，
    #: 而是因为 FAISS 的入参就是 numpy 数组——没有 numpy 就没有这个后端。
    requires: tuple[str, ...] = ("faiss", "numpy")

    supports_filter = True
    supports_delete = True
    supports_persistence = True

    #: **FAISS 只管向量**：``text`` 与 ``metadata`` 全在本类的记录表里，
    #: 因此 ``_find_ids`` 只能扫记录表（不覆写，用 ``base`` 的默认实现）。
    native_metadata = False

    def __init__(
        self,
        *,
        metric: str = METRIC_COSINE,
        dimension: int | None = None,
        faiss_module: Any | None = None,
        path: str = "",
        index: Any | None = None,
    ) -> None:
        """构造一个 faiss 库：**维度在这里就必须确定**，索引也在这里建立.

        ``dimension`` 为什么必填（与 flat 最大的不同）：索引一旦建立，
        ``d`` 就写进了它的内存布局，之后不能改——所以"先写一批数据、
        让库自己学出维度"这条 flat 走得很顺的路，在 FAISS 上没有等价物。
        未给维度且没有现成的 ``index`` 时直接报错，并在消息里给出两条出路。

        ``index`` 用于"接管一个已经建立好的索引"（``load`` 内部走的就是它）：
        传进来的若是裸 ``IndexFlat*``（没有 ``id_map``），本类会自己包一层
        ``IndexIDMap2``——但**空索引才能接管**：一个已经有向量的裸索引
        没法还原"行号 → 外部 id"，硬接管只会让检索返回一堆无效 id。

        ``path`` 只是"默认的落盘位置"（``persist()`` 不带参数时用它）。
        与 flat 不同，本类**不在构造时自动 load**：FAISS 的快照是两个文件，
        自动载入意味着构造可能因为"另一个文件不存在"而失败，而调用方
        未必是来读数据的（例如只是想建一个新库再写进去）。
        """
        super().__init__(metric=metric, dimension=dimension, location=path)
        self._module = resolve_faiss_module(faiss_module)
        self._records: dict[str, VectorRecord] = {}
        #: int64 → record_id 的反查表。**它是状态而不是缓存**：
        #: 哈希不可逆，没有它就取不回记录。
        self._reverse: dict[int, str] = {}
        self._index_type = METRIC_TO_INDEX_TYPE[self.metric]
        if index is not None:
            self._index = self._adopt_index(index)
            if not self.has_dimension:
                self._dimension = self._check_dimension(int(self._index.d))
        else:
            if not self.has_dimension:
                raise VectorStoreError(
                    "FAISS 在建索引之前就要知道维度，请传入 dimension，或先写入一批数据"
                    "（后者指：用 flat 这类无维度要求的后端先把向量算好并落盘，"
                    "再把它的维度传给本类）。本实现不接受「先建库、维度晚点再说」——"
                    "索引的 d 一旦建立就不可变，一个 d 未定的索引无法被创建。"
                )
            self._index = self._create_index(self.dimension)
        index_dimension = int(self._index.d)
        if index_dimension != self.dimension:
            raise VectorError(
                f"索引的维度是 {index_dimension}，本实例的 dimension={self.dimension}："
                "维度不一致说明这个索引不是按当前编码器建的（换过 embedding 提供方？）。"
                "请清库重建——截断或补零会让排序静默失去意义。"
            )

    def _create_index(self, dimension: int) -> Any:
        """按度量建一个**精确**索引，并用 ``IndexIDMap2`` 包一层.

        两件事都在这一行里：

        ```text
        索引类型   cosine/ip → IndexFlatIP ；l2 → IndexFlatL2
        包一层     只有 IndexIDMap2 才有 add_with_ids / id_map
        ```
        """
        factory = (
            self._module.IndexFlatL2
            if self._index_type == INDEX_TYPE_L2
            else self._module.IndexFlatIP
        )
        return self._module.IndexIDMap2(factory(dimension))

    def _adopt_index(self, index: Any) -> Any:
        """接管一个外部索引：核对度量类型，有 ``id_map`` 直接用，裸索引包一层（且必须为空）.

        为什么用 ``id_map`` 判断而不是 ``hasattr(index, "add_with_ids")``：
        真库的 ``IndexFlat`` 也有 ``add_with_ids``（只是会抛 not implemented），
        而 ``id_map`` 只有 ``IndexIDMap*`` 才有——它才是"能不能表达外部 id"
        的判据。

        度量类型必须核对：``IndexFlatL2`` 的 D 越小越近、``IndexFlatIP`` 越大越近，
        拿错一个的后果是**排序整体反向**（返回的正好是最不像的几条），
        而它不报错，只表现为"检索效果突然很差"。
        """
        expected = (
            FAISS_METRIC_L2 if self._index_type == INDEX_TYPE_L2 else FAISS_METRIC_INNER_PRODUCT
        )
        actual = getattr(index, "metric_type", None)
        if actual is not None and int(actual) != expected:
            raise VectorError(
                f"索引的度量类型是 {int(actual)}，而本实例的 metric={self.metric!r} "
                f"需要 {expected}（{self._index_type}）：度量不同会让排序整体反向。"
                "请用与索引匹配的 metric 新建实例，或按当前度量重建索引。"
            )
        if hasattr(index, "id_map"):
            return index
        if int(getattr(index, "ntotal", 0)) > 0:
            raise VectorError(
                "不能接管一个已经有向量的裸索引：它只有行号、没有外部 id，"
                "行号到记录 id 的对应关系无法还原（硬接管会让检索返回一堆无效 id）。"
                "请从空索引开始，或用 load() 读取本类写出的索引文件。"
            )
        return self._module.IndexIDMap2(index)

    # ------------------------------------------------------------------ 原语

    def count(self) -> int:
        """库里的记录条数（以**记录表**为准，见 ``base`` 的"记录表是主"）."""
        return len(self._records)

    def ids(self) -> list[str]:
        """全部 id，**升序**（理由见 ``base.ids``：插入序会随分批方式变化）."""
        return sorted(self._records)

    def get(self, record_id: str) -> VectorRecord | None:
        """取一条记录；FAISS 那边没有原文与元数据，所以只能查记录表."""
        return self._records.get(record_id)

    def _upsert(self, records: Sequence[VectorRecord]) -> WriteReport:
        """判定 added / updated / unchanged，并**同步**维护索引与记录表.

        顺序是刻意的：**先算完整个计划、再动索引、最后提交记录表**。
        半个批次写进索引而记录表没跟上，会留下"索引里有一条没有原文的向量"
        ——它不报错，只表现为检索偶尔返回一条被跳过的命中（``base.query``
        遇到记录表里没有的 id 会跳过并继续），而没有人会怀疑"写入写了一半"。

        ``unchanged`` 的记录**完全不碰索引**（``continue``）：重放同一批数据
        时，索引不该有任何写操作。若改成"无条件先删后加"，观察到的计数
        仍然是 ``unchanged``，但索引文件被重写了一遍——**"这次重建省掉了多少
        工作"这个数字就失真了**（day065 的增量索引正是靠它）。
        """
        prepared = list(records)
        keys = self._validate_ids(prepared)
        added = updated = unchanged = 0
        plan: list[tuple[int, VectorRecord]] = []
        replaced: list[int] = []
        for record, key in zip(prepared, keys):
            existing = self._records.get(record.record_id)
            if existing is None:
                added += 1
            elif self._same_record(existing, record):
                unchanged += 1
                continue
            else:
                updated += 1
                replaced.append(key)
            plan.append((key, record))
        if replaced:
            selector = self._module.IDSelectorBatch(np.asarray(replaced, dtype=np.int64))
            self._index.remove_ids(selector)
        if plan:
            matrix = np.ascontiguousarray(
                [list(record.vector) for _, record in plan], dtype="float32"
            )
            vector_ids = np.asarray([key for key, _ in plan], dtype=np.int64)
            self._index.add_with_ids(matrix, vector_ids)
        for key, record in plan:
            self._records[record.record_id] = record
            self._reverse[key] = record.record_id
        return WriteReport(added=added, updated=updated, unchanged=unchanged)

    def _validate_ids(self, records: Sequence[VectorRecord]) -> list[int]:
        """预检整批记录的 int64 映射，**在动索引之前**把撞号报出来.

        为什么要有一趟预检：撞号是"两条不同的记录抢同一个 id"，
        若在逐条写入的过程中才发现，前半批已经进了索引——
        报错之后库处于一个既不完整、又看不出哪里不完整的状态。
        预检之后撞号是"整批拒绝"，库里一个字节都没变。

        同一批里的两个不同 id 撞号也在这一趟里被发现（``seen`` 随着遍历增长），
        不需要为它单独写一条分支。
        """
        keys: list[int] = []
        seen: dict[int, str] = dict(self._reverse)
        for record in records:
            key = to_int64_id(record.record_id)
            owner = seen.get(key)
            if owner is not None and owner != record.record_id:
                raise VectorStoreError(
                    f"id 哈希冲突：{record.record_id!r} 与 {owner!r} 都映射到 int64 {key}。"
                    f"本类把字符串 id 哈希到 [1, {INT64_ID_MODULUS}] 区间，"
                    "这是哈希冲突，概率极低但必须报出来——静默覆盖会让一条记录凭空消失。"
                    "请改其中一个 id，或换一个直接存字符串 id 的后端（flat / chroma）。"
                )
            seen[key] = record.record_id
            keys.append(key)
        return keys

    @staticmethod
    def _same_record(existing: VectorRecord, incoming: VectorRecord) -> bool:
        """两条记录是否**逐项相同**（``unchanged`` 的唯一依据，与 flat 同一口径）.

        三个字段分开比较而不是比较整个 ``VectorRecord``：一旦将来给记录
        加一个"仅用于展示"的字段（比如入库时间），整体比较会静默地永远不相等，
        于是每次重放都报 ``updated``。
        """
        return (
            existing.vector == incoming.vector
            and existing.text == incoming.text
            and existing.metadata == incoming.metadata
        )

    def _remove(self, ids: Sequence[str]) -> int:
        """删除给定 id：索引与记录表**各删一次**，并用返回值对账.

        对账这一步不是形式主义：``remove_ids`` 返回的是**索引真正删掉的条数**，
        而请求条数来自记录表。两者不等意味着索引与记录表已经不同步
        （例如有人绕过本类直接动了索引），这种状态必须喊出来——
        否则它只会在某次检索中表现为"少了一条结果"。
        """
        targets = list(dict.fromkeys(record_id for record_id in ids if record_id in self._records))
        if not targets:
            return 0
        keys = [to_int64_id(record_id) for record_id in targets]
        selector = self._module.IDSelectorBatch(np.asarray(keys, dtype=np.int64))
        removed = int(self._index.remove_ids(selector))
        for record_id in targets:
            self._records.pop(record_id, None)
            self._reverse.pop(to_int64_id(record_id), None)
        if removed != len(targets):
            raise VectorStoreError(
                f"删除对账失败：请求删除 {len(targets)} 条，索引只报告删掉 {removed} 条。"
                "向量索引与记录表不同步——通常是有人绕过了本类直接改索引，"
                "或快照的两个文件不是同一次写出的。请用 persist()/load() 重建这一对文件。"
            )
        return removed

    def _ranked_ids(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        restrict: frozenset[str] | None = None,
    ) -> list[tuple[str, float]]:
        """向量排序：返回最多 ``limit`` 个 ``(id, score)``，**分数降序**.

        三条与真库有关的细节，每条都对应一种"结果悄悄的错"：

        ```text
        入参形状    np.ascontiguousarray(vector, "float32").reshape(1, -1)
                    —— FAISS 只接受 (n, d) 的 float32，行向量也不行
        检索深度    无 restrict：min(limit, ntotal)（多要的 k 只会拿到 -1 填充）
                    有 restrict：ntotal（**全量**，见下）
        -1 与换算   I 里的 -1 必须丢弃（它是"此处没有结果"，不是记录 id）；
                    D 用 _to_score 折成本包口径（IP 是相似度、L2 是距离）
        ```

        ``restrict`` 这一支是本文件最贵的一段：它把整个库排序一遍再筛。
        为什么值得：``IndexFlat`` 是**精确**索引，全量排序后过滤得到的就是
        "候选集内真正最像的 limit 条"——与 flat 逐条一致。若换成 IVF/HNSW，
        这个做法就不再成立（全量 search 等于放弃索引的意义），那时只剩
        "过采样 + 接受近似"一条路，**"过滤"与"近似"天生互相消耗**。
        """
        total = int(self._index.ntotal)
        if total == 0:
            return []
        query = np.ascontiguousarray(vector, dtype="float32").reshape(1, -1)
        depth = total if restrict is not None else min(limit, total)
        distances, raw_ids = self._index.search(query, depth)
        similarity = self._index_type == INDEX_TYPE_IP
        ranked: list[tuple[str, float]] = []
        for distance, raw_id in zip(distances[0], raw_ids[0]):
            neighbor = int(raw_id)
            if neighbor == FAISS_ID_NOT_FOUND:
                # 结果不足时尾部全是 -1：丢掉，不是翻译。
                continue
            record_id = self._reverse.get(neighbor)
            if record_id is None:
                # 索引与记录表不同步（见 _remove 的对账）：读路径跳过而不是抛错。
                continue
            if restrict is not None and record_id not in restrict:
                continue
            score = _to_score(self.metric, float(distance), similarity=similarity)
            ranked.append((record_id, score))
            if len(ranked) >= limit:
                break
        return ranked

    # ------------------------------------------------------------------ 元信息

    def describe_extra(self) -> dict[str, str]:
        """``/vectorstore/stats`` 的后端专属字段.

        ``index_ntotal`` 与 ``records`` 刻意分成两个数字：它们的差就是
        "两个库是否同步"的体检指标（正常情况下永远相等）。
        """
        return {
            "index_type": self._index_type,
            "index_ntotal": str(int(self._index.ntotal)),
            "records": str(self.count()),
            "id_range": f"{INT64_ID_MIN}..{INT64_ID_MODULUS}",
            "records_sidecar": RECORDS_SUFFIX,
            "snapshot_version": str(SNAPSHOT_VERSION),
        }

    # ------------------------------------------------------------------ 持久化

    def persist(self, path: str | None = None) -> str:
        """把"两个库"分别落盘：索引文件 + ``<path>.records.json``，返回主路径.

        为什么是两份而不是一份：``faiss.write_index`` 只能写它自己的二进制
        格式，而 metadata/原文**根本不在索引里**。把它们塞进索引文件是不可能的
        （那才是最该避免的 hack：一份格式混装的文件会让两边都无法独立升级）。
        代价是"两个文件必须成对存在"——``load`` 会强制这一点。

        全量重写、目录自动创建、记录按 id 升序写出（与 flat 同样三条理由：
        快照要么完整要么不存在、调用方不必先 mkdir、两份快照逐字节可比）。
        """
        target = str(path or self.location or "")
        if not target:
            raise BackendUnavailable(
                "faiss 后端没有可写的路径：构造时 path 为空，本次也没有传入 path。"
                "faiss 的索引不能只活在内存里（write_index 需要一个文件），"
                "请用 FaissVectorStore(path=...) 或 persist(path=...) 指定落盘位置。"
            )
        index_path = Path(target)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self._module.write_index(self._index, str(index_path))
        sidecar = Path(str(index_path) + RECORDS_SUFFIX)
        snapshot = {
            "version": SNAPSHOT_VERSION,
            "backend": self.name,
            "metric": self.metric,
            "dimension": self.dimension,
            "count": self.count(),
            "index_ntotal": int(self._index.ntotal),
            "records": [
                {
                    "record_id": record_id,
                    "vector": list(self._records[record_id].vector),
                    "text": self._records[record_id].text,
                    "metadata": dict(self._records[record_id].metadata),
                }
                for record_id in sorted(self._records)
            ],
        }
        sidecar.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._location = str(index_path)
        return self._location

    def load(self, path: str | None = None) -> None:
        """读回索引与记录表，并**在替换内存状态之前**核对它们是不是一套.

        四道校验，各挡一种"读进来之后错得看不出来"：

        ```text
        version          格式换了            → VectorStoreError（点明"快照版本"）
        metric           度量不同            → VectorError（近邻不是同一批）
        dimension        维度不同            → VectorError（库与编码器不是一套）
        count            声明的条数对不上    → VectorStoreError（文件被截断/手改）
        index_ntotal     **索引与记录表不等** → VectorError（"两个库"不同步）
        id_map 集合      索引里的 id 不是这一批 → VectorError（两个文件不是一次写出的）
        ```

        ``id_map`` 那一条是 FAISS 特有的：即使两边条数相同，也可能"内容对不上"
        （例如把另一个库的索引文件拷了过来）。真库的 ``IndexIDMap2`` 暴露
        ``id_map``，本类在装载时顺手比一次集合——这是把"两个库"这个隐患
        关在装载点上的最后一道门。

        维度只核对、不"补写"：与 flat 不同，本后端的维度在构造时就确定了
        （索引建不起来就没有这个对象），因此不存在"载入时才学到维度"这条路。

        载入是"替换"而不是"合并"（与 flat 同一理由：合并会让库里多出来的
        那几条永远删不掉，而调用方以为它们是旧的）。
        """
        target = str(path or self.location or "")
        if not target:
            raise BackendUnavailable(
                "faiss 后端没有可读的路径：构造时 path 为空，本次也没有传入 path。"
            )
        index_path = Path(target)
        sidecar = Path(str(index_path) + RECORDS_SUFFIX)
        if not index_path.exists():
            raise BackendUnavailable(f"faiss 索引文件不存在：{target}")
        if not sidecar.exists():
            raise BackendUnavailable(
                f"faiss 的记录旁挂文件不存在：{sidecar}。"
                "索引与记录表必须成对出现——只有索引时，命中的 id 没有原文与元数据，"
                "半个库比没有库更危险。"
            )
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        version = payload.get("version")
        if version != SNAPSHOT_VERSION:
            raise VectorStoreError(
                f"快照版本与当前代码不匹配：文件是 version={version!r}，"
                f"当前代码是 version={SNAPSHOT_VERSION}。"
                "请用与快照同版本的代码读取，或用当前代码重建快照。"
            )
        snapshot_metric = str(payload.get("metric"))
        if snapshot_metric != self.metric:
            raise VectorError(
                f"快照的 metric={snapshot_metric!r} 与本实例的 metric={self.metric!r} "
                "不一致：同一批向量在两种度量下的最近邻不是同一批。"
                "出路是**用 snapshot 的 metric 新建实例**"
                f"（例如 FaissVectorStore(metric={snapshot_metric!r}, dimension=...)），"
                "而不是让一个已有实例改度量。"
            )
        snapshot_dimension = int(payload.get("dimension", 0))
        if self.has_dimension and snapshot_dimension != self.dimension:
            raise VectorError(
                f"快照的 dimension={snapshot_dimension} 与本实例的 dimension="
                f"{self.dimension} 不一致：库与当前编码器不是同一套。"
                "出路是**用 snapshot 的 dimension 新建实例**，不要截断或补零。"
            )
        records = payload.get("records", [])
        declared = int(payload.get("count", len(records)))
        if declared != len(records):
            raise VectorStoreError(
                f"快照自相矛盾：count={declared}，但 records 有 {len(records)} 条。"
                "这是一份被截断或手改过的文件，不能作为建库依据。"
            )
        loaded: dict[str, VectorRecord] = {}
        reverse: dict[int, str] = {}
        for item in records:
            record = VectorRecord(
                record_id=item["record_id"],
                vector=tuple(float(value) for value in item["vector"]),
                text=str(item.get("text", "")),
                metadata=dict(item.get("metadata", {})),
            )
            key = to_int64_id(record.record_id)
            owner = reverse.get(key)
            if owner is not None and owner != record.record_id:
                raise VectorStoreError(
                    f"快照里的 id 发生哈希冲突：{record.record_id!r} 与 {owner!r} "
                    f"都映射到 int64 {key}，这份快照不能载入。"
                )
            loaded[record.record_id] = record
            reverse[key] = record.record_id
        try:
            index = self._module.read_index(str(index_path))
        except Exception as exc:
            raise BackendUnavailable(
                f"读取 faiss 索引失败：{target}（{type(exc).__name__}: {exc}）。"
                "文件可能不是本类写出的索引，或已被截断。"
            ) from exc
        if int(index.ntotal) != len(loaded):
            raise VectorError(
                f"索引与记录表不同步：索引里有 {int(index.ntotal)} 条向量，"
                f"记录表里有 {len(loaded)} 条记录。"
                "这两个文件必须来自同一次 persist()——请重新生成快照，"
                "不要手工拼装（那会让检索命中一些没有原文的 id）。"
            )
        id_map = getattr(index, "id_map", None)
        if id_map is not None and set(int(value) for value in id_map.tolist()) != set(reverse):
            raise VectorError(
                "索引里的 id 集合与记录表不一致：两个文件不是同一次写出的"
                "（例如把另一个库的索引文件拷了过来）。请重新生成快照。"
            )
        self._index = self._adopt_index(index)
        self._records = loaded
        self._reverse = reverse
        self._location = str(index_path)


__all__ = [
    "FAISS_ID_NOT_FOUND",
    "FAISS_METRIC_INNER_PRODUCT",
    "FAISS_METRIC_L2",
    "INDEX_TYPE_IP",
    "INDEX_TYPE_L2",
    "INT64_ID_MIN",
    "INT64_ID_MODULUS",
    "METRIC_TO_INDEX_TYPE",
    "RECORDS_SUFFIX",
    "REQUIRED_FAISS_API",
    "SNAPSHOT_VERSION",
    "FaissVectorStore",
    "from_int64_id",
    "resolve_faiss_module",
    "to_int64_id",
]
