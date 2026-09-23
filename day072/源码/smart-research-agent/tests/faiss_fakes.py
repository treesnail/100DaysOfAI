"""测试用的最小假 FAISS：**用 numpy 做精确检索**，并还原真库的几条硬行为（M6-D3）.

本机没有安装 ``faiss``（也不该为了跑测试去装一个几百 MB 的二进制包），
因此 ``faiss_backend`` 的测试只能靠在构造时注入一个"长得像 faiss"的对象。
这个文件就是那个对象。

## 为什么不能只写一个 `MagicMock`

``MagicMock`` 对任何属性都返回一个可调用对象，于是适配器写成什么样它都"通过"：

```text
适配器忘了把 ids 转成 int64        → Mock 照样接受（真库 TypeError）
适配器用了 IndexIDMap2.add         → Mock 照样接受（真库断言失败）
适配器自己发明了一个构造参数       → Mock 照样接受（真库 TypeError）
适配器没处理 search 的 -1 填充     → Mock 返回不了 -1，这条分支永远测不到
```

**一份不会变红的假实现，等于没有测试。** 因此下面这几条真库行为必须被"写死"：

| 真库行为 | 为什么必须还原 |
|---------|--------------|
| ``x`` 必须是 float32 的二维数组 | 少转一次就是运行时崩溃 |
| ``ids`` 必须是 int64 | int32 在 Windows 上"看起来是对的" |
| 构造函数只接受 ``d`` | 防止适配器发明参数 |
| ``IndexIDMap2.add`` 会失败 | 真库要求用 ``add_with_ids`` |
| 结果不足时 ``I`` 用 ``-1`` 填充 | ``-1`` 是保留值，必须丢弃 |
| L2 的 ``D`` 用 ``+Inf`` 填充 | 哨兵值，不能当距离用 |
| 内积索引的 ``D`` **越大越近** | 与 Chroma 的 distance 口径相反 |
| ``IDSelectorBatch`` 只收 int64 数组 | 适配器的删除路径靠它 |

上表各条落在：``_require_float32_matrix`` / ``_require_int64_vector`` /
``_FakeFlatIndex.__init__`` / ``FakeIndexIDMap2.add`` / ``_FakeFlatIndex.search`` /
``FakeIndexFlatIP`` / ``FakeIDSelectorBatch``。

## 假实现与真库的两处"已知不同"

1. ``FakeIndexFlatIP`` 直接实现了 ``add_with_ids``（真 ``IndexFlat`` 的
   ``add_with_ids`` 会抛 "not implemented"）。规格要求这个假模块提供它，
   以便索引对象也能被单独测试；**适配器走的仍然是
   ``IndexFlatIP → IndexIDMap2 → add_with_ids`` 这条真路径**，
   所以这条差异不会掩盖任何真实故障。
2. 真 faiss 的 Python wrapper 会对 ``x`` 做
   ``np.ascontiguousarray(x, dtype="float32")``（即**替调用方转换**），
   本文件把 dtype 当成硬要求（更严）。更严的方向是安全的：
   适配器一旦少写一次显式转换就会立刻失败，而不是依赖 wrapper 兜底。

## 谁用它

只有 ``tests/test_vectorstore_faiss.py``。文件名不以 ``test_`` 开头，
pytest 不会收集它（它本身不是测试，是测试的**道具**）。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

#: 与真实 ``faiss.MetricType`` 取值一致（``METRIC_INNER_PRODUCT = 0`` / ``METRIC_L2 = 1``）.
METRIC_INNER_PRODUCT = 0
METRIC_L2 = 1

#: 结果不足时 ``I`` 的填充值（真库的保留值，官方建议 id 不要用 -1）.
FAISS_ID_NOT_FOUND = -1


def _require_float32_matrix(x: Any, *, who: str) -> np.ndarray:
    """校验入参是 ``float32`` 的二维数组；不是就 ``TypeError``.

    这一条对应真库 wrapper 的类型要求，也是"适配器必须自己做转换"的证据：
    假模块在这里严格，适配器少做一次 ``np.ascontiguousarray(..., "float32")``
    就会立刻变红，而不是等到装上真库那天才崩。
    """
    array = np.asarray(x)
    if array.dtype != np.float32:
        raise TypeError(
            f"{who} 需要 float32 的 numpy 数组，收到 dtype={array.dtype}。"
            "真库的 wrapper 不接受 float64/float16 的向量——"
            "适配器必须显式转换，不能指望 numpy 自动升/降级。"
        )
    if array.ndim != 2:
        raise TypeError(f"{who} 需要形状 (n, d) 的二维数组，收到 {array.shape}")
    return array


def _require_int64_vector(ids: Any, *, who: str) -> np.ndarray:
    """校验入参是 ``int64`` 的一维数组；不是就 ``TypeError``.

    为什么要专门盯 int64：Windows 上 ``np.asarray([1, 2])`` 的默认 dtype 是
    **int32**，而它在算术上与 int64 看不出区别——于是"少转一次 dtype"
    这类问题只在真库（或本假模块）上才会暴露。
    """
    array = np.asarray(ids)
    if array.dtype != np.int64:
        raise TypeError(
            f"{who} 需要 int64 的 id 数组，收到 dtype={array.dtype}。"
            "真库的 add_with_ids / IDSelectorBatch 都要求 int64，"
            "而其它整数类型在数值上看不出区别——所以要显式转换。"
        )
    if array.ndim != 1:
        raise TypeError(f"{who} 需要一维 id 数组，收到 {array.shape}")
    return array


class _FakeFlatIndex:
    """``IndexFlatIP`` / ``IndexFlatL2`` 的公共实现：**精确检索，不存外部 id**（见模块 docstring）.

    真 ``IndexFlat`` 的"id"就是**行号**：``add`` 之后行号 0..n-1 就是它的 id，
    删除之后行号会被重新压实。这里用同一套语义（``self._ids`` 与行号对齐），
    这样 ``FakeIndexIDMap2`` 才能像真库那样"按行号翻译成外部 id"。
    """

    #: 子类覆盖（见 ``FakeIndexFlatIP`` / ``FakeIndexFlatL2``）.
    _metric_type = METRIC_INNER_PRODUCT

    def __init__(self, d: int, **kwargs: Any) -> None:
        """只接受维度 ``d``；任何关键字参数都 ``TypeError``（照真库的签名）.

        这一条是给适配器准备的"防发明"护栏：真库的 ``IndexFlatIP(d)``
        没有 ``metric_type=`` 之类的参数，适配器一旦自己发明一个，
        在假模块上必须先炸，而不是等到装上真库才炸。
        """
        if kwargs:
            raise TypeError(
                f"{type(self).__name__}() 不接受关键字参数 {sorted(kwargs)}："
                "真实 faiss 的 IndexFlat* 只接受位置参数 d。"
            )
        self._d = int(d)
        self._vectors: np.ndarray | None = None
        self._ids = np.zeros(0, dtype=np.int64)
        #: 每次 ``search`` 的入参快照（测试支撑：用来断言适配器传的是 float32）。
        self.searches: list[np.ndarray] = []
        #: 每次 ``remove_ids`` 删掉的行号（测试支撑：用来核对删除路径）。
        self.removals: list[list[int]] = []

    @property
    def d(self) -> int:
        """向量维度."""
        return self._d

    @property
    def ntotal(self) -> int:
        """索引里的向量总数."""
        return 0 if self._vectors is None else int(self._vectors.shape[0])

    @property
    def metric_type(self) -> int:
        """度量类型（与真库的 ``MetricType`` 取值一致）."""
        return self._metric_type

    def add(self, x: Any) -> None:
        """按 ``add`` 写入（行号即 id），对应真库的隐式顺序 id."""
        matrix = _require_float32_matrix(x, who=f"{type(self).__name__}.add")
        self._check_dimension(matrix)
        base = self.ntotal
        self._append(matrix)
        self._ids = np.concatenate(
            [self._ids, np.arange(base, base + matrix.shape[0], dtype=np.int64)]
        )

    def add_with_ids(self, x: Any, ids: Any) -> None:
        """带外部 id 写入（真 ``IndexFlat`` 不支持，见模块 docstring 的已知差异一）."""
        matrix = _require_float32_matrix(x, who=f"{type(self).__name__}.add_with_ids")
        vector_ids = _require_int64_vector(ids, who=f"{type(self).__name__}.add_with_ids")
        self._check_dimension(matrix)
        if vector_ids.shape[0] != matrix.shape[0]:
            raise RuntimeError(
                f"x 有 {matrix.shape[0]} 行、ids 有 {vector_ids.shape[0]} 个："
                "真库在这里会断言失败，所以这里也拒绝。"
            )
        self._append(matrix)
        self._ids = np.concatenate([self._ids, vector_ids])

    def search(self, x: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
        """精确检索；结果不足时 ``I`` 用 ``-1``、``D`` 用哨兵填充.

        两个哨兵与真库一致：**L2 用 ``+Inf``、内积用 ``-Inf``**。
        它们不是"差不多大"的近似值，而是"这里没有结果"的显式标记——
        适配器必须靠 ``I == -1`` 判断，不能靠 ``D`` 猜。
        """
        matrix = _require_float32_matrix(x, who=f"{type(self).__name__}.search")
        self.searches.append(np.array(matrix, copy=True))
        self._check_dimension(matrix)
        rows = matrix.shape[0]
        depth = int(k)
        if depth <= 0:
            return (
                np.zeros((rows, 0), dtype=np.float32),
                np.zeros((rows, 0), dtype=np.int64),
            )
        pad = float("inf") if self._metric_type == METRIC_L2 else float("-inf")
        distances = np.full((rows, depth), pad, dtype=np.float32)
        found_ids = np.full((rows, depth), FAISS_ID_NOT_FOUND, dtype=np.int64)
        found = min(depth, self.ntotal)
        if found == 0:
            return distances, found_ids
        scores = self._scores(matrix)
        if self._metric_type == METRIC_L2:
            # 平方 L2 越小越近：升序
            order = np.argsort(scores, axis=1, kind="stable")
        else:
            # 内积越大越近：降序（stable 让并列时按行号，与真库的堆序一致）
            order = np.argsort(-scores, axis=1, kind="stable")
        top = order[:, :found]
        distances[:, :found] = np.take_along_axis(scores, top, axis=1)
        found_ids[:, :found] = self._ids[top]
        return distances, found_ids

    def remove_ids(self, selector: Any) -> int:
        """按选择器删除并**重新压实行号**（真 ``IndexFlat`` 的行为），返回删掉的条数."""
        if self.ntotal == 0:
            return 0
        removed_mask = np.array(
            [selector.is_member(int(value)) for value in self._ids], dtype=bool
        )
        removed = int(removed_mask.sum())
        self.removals.append([int(value) for value in self._ids[removed_mask]])
        keep = ~removed_mask
        assert self._vectors is not None
        self._vectors = self._vectors[keep]
        # 行号重新压实：0..n-1（真 IndexFlat 的 id 就是行号）
        self._ids = np.arange(int(keep.sum()), dtype=np.int64)
        return removed

    def _scores(self, matrix: np.ndarray) -> np.ndarray:
        assert self._vectors is not None
        if self._metric_type == METRIC_L2:
            diff = matrix[:, None, :] - self._vectors[None, :, :]
            return np.sum(diff * diff, axis=2)
        return matrix @ self._vectors.T

    def _check_dimension(self, matrix: np.ndarray) -> None:
        if matrix.shape[1] != self._d:
            raise RuntimeError(
                f"维度不匹配：索引是 {self._d} 维，收到 {matrix.shape[1]} 维"
                "（真库在 C++ 层断言，映射成 RuntimeError）。"
            )

    def _append(self, matrix: np.ndarray) -> None:
        current = self._vectors
        if current is None or current.shape[0] == 0:
            self._vectors = np.array(matrix, dtype=np.float32, copy=True)
            return
        self._vectors = np.concatenate([current, matrix]).astype(np.float32, copy=False)


class FakeIndexFlatIP(_FakeFlatIndex):
    """``faiss.IndexFlatIP``：内积（**D 越大越近**）."""

    _metric_type = METRIC_INNER_PRODUCT


class FakeIndexFlatL2(_FakeFlatIndex):
    """``faiss.IndexFlatL2``：平方 L2 距离（**D 越小越近**）."""

    _metric_type = METRIC_L2


class FakeIDSelectorBatch:
    """``faiss.IDSelectorBatch``：一个 int64 id 的集合.

    真库的 ``remove_ids`` 只接受选择器对象而不是 id 列表，因此适配器必须
    自己构造它——这里保持同样的形状，顺带把"必须是 int64"这条要求也带上。
    """

    def __init__(self, ids: Any) -> None:
        array = _require_int64_vector(ids, who="IDSelectorBatch")
        self.ids = [int(value) for value in array]
        self._members = set(self.ids)

    def is_member(self, value: int) -> bool:
        """该 id 是否在集合里（真库 ``IDSelector::is_member``）."""
        return int(value) in self._members

    def __len__(self) -> int:
        return len(self.ids)


class FakeIndexIDMap2:
    """``faiss.IndexIDMap2``：把"内部行号"与"外部 int64 id"分开记.

    真库的这一层是必需的：``IndexFlat`` 只有行号，``IndexIDMap2`` 才让它
    能表达外部 id。因此本假实现同样维护一张 ``id_map``（行号 → 外部 id），
    ``search`` 之后把行号翻译回外部 id，删除时再翻译回去。
    """

    def __init__(self, index: _FakeFlatIndex) -> None:
        self.index = index
        self._ids = np.zeros(0, dtype=np.int64)
        #: 每次 ``search`` 的入参快照（测试支撑：断言适配器传的是 float32）。
        self.searches: list[np.ndarray] = []
        #: 每次 ``add_with_ids`` 的 (x.dtype, ids.dtype)（测试支撑）。
        self.add_calls: list[tuple[np.dtype, np.dtype]] = []

    @property
    def d(self) -> int:
        """向量维度."""
        return self.index.d

    @property
    def ntotal(self) -> int:
        """索引里的向量总数（外部 id 的个数）."""
        return int(self.index.ntotal)

    @property
    def metric_type(self) -> int:
        """度量类型（真库转发给内部索引）."""
        return int(self.index.metric_type)

    @property
    def id_map(self) -> np.ndarray:
        """行号 → 外部 id 的映射（真库对外暴露 ``id_map``；本类用它做对账）."""
        return np.array(self._ids, copy=True)

    def add_with_ids(self, x: Any, ids: Any) -> None:
        """带 id 写入（**本类唯一合法的写入入口**）."""
        matrix = _require_float32_matrix(x, who="IndexIDMap2.add_with_ids")
        vector_ids = _require_int64_vector(ids, who="IndexIDMap2.add_with_ids")
        if vector_ids.shape[0] != matrix.shape[0]:
            raise RuntimeError(
                f"x 有 {matrix.shape[0]} 行、ids 有 {vector_ids.shape[0]} 个："
                "真库在这里会断言失败，所以这里也拒绝。"
            )
        self.add_calls.append((matrix.dtype, vector_ids.dtype))
        self.index.add(matrix)
        self._ids = np.concatenate([self._ids, vector_ids])

    def add(self, x: Any) -> None:
        """**必须失败**：真 ``IndexIDMap`` 的 ``add`` 无法表达外部 id.

        ``add`` 只能给出行号，而 ``IndexIDMap2`` 的价值就在于"外部 id"。
        真库在这里会抛异常（C++ 侧断言），所以这里也必须抛——
        否则适配器写成 ``index.add(...)`` 也能"跑通"，直到装上真库才炸。
        """
        raise RuntimeError(
            "add 不能用于 IndexIDMap2（真库会在这里断言失败）："
            f"必须用 add_with_ids(x, ids)，收到 x 形状 {np.asarray(x).shape}。"
        )

    def search(self, x: Any, k: int) -> tuple[np.ndarray, np.ndarray]:
        """转发给内部索引，并把行号翻译成外部 id（``-1`` 原样保留）."""
        matrix = _require_float32_matrix(x, who="IndexIDMap2.search")
        self.searches.append(np.array(matrix, copy=True))
        distances, rows = self.index.search(matrix, k)
        if rows.size == 0:
            return distances, rows
        mapped = np.full(rows.shape, FAISS_ID_NOT_FOUND, dtype=np.int64)
        mask = rows >= 0
        if bool(mask.any()):
            mapped[mask] = self._ids[rows[mask]]
        return distances, mapped

    def remove_ids(self, selector: FakeIDSelectorBatch) -> int:
        """把"外部 id 选择器"翻译成"行号选择器"，再交给内部索引（与真库同构）."""
        if self.ntotal == 0:
            return 0
        positions = np.array(
            [
                position
                for position, external in enumerate(self._ids)
                if selector.is_member(int(external))
            ],
            dtype=np.int64,
        )
        removed = int(self.index.remove_ids(FakeIDSelectorBatch(positions)))
        if positions.size:
            keep = np.ones(self._ids.shape[0], dtype=bool)
            keep[positions] = False
            self._ids = self._ids[keep]
        return removed


def write_index(index: Any, path: Any) -> None:
    """把索引落盘（真库写二进制索引文件；这里用 pickle，够用且可读性不重要）."""
    Path(str(path)).write_bytes(pickle.dumps(index))


def read_index(path: Any) -> Any:
    """读回索引（与 ``write_index`` 成对；文件损坏时抛异常，让调用方去兜住）."""
    return pickle.loads(Path(str(path)).read_bytes())


class FakeFaissModule:
    """把上面几件东西组装成一个 ``faiss`` 模块的样子.

    用实例属性而不是类属性：``self.write_index = write_index`` 拿到的是
    **函数本身**（不会像类属性那样被绑成方法），调用形状与真模块一致。
    """

    def __init__(self) -> None:
        self.__name__ = "faiss_fakes"
        self.IndexFlatIP = FakeIndexFlatIP
        self.IndexFlatL2 = FakeIndexFlatL2
        self.IndexIDMap2 = FakeIndexIDMap2
        self.IDSelectorBatch = FakeIDSelectorBatch
        self.write_index = write_index
        self.read_index = read_index
        self.METRIC_INNER_PRODUCT = METRIC_INNER_PRODUCT
        self.METRIC_L2 = METRIC_L2


class FakeFaissModuleMissingIndexFlat(FakeFaissModule):
    """缺少 ``IndexFlatIP`` 的变体：模拟"版本不匹配 / 装成了另一个同名包".

    适配器必须在构造时就把它认出来，并给出"去升级 faiss-cpu"的指引——
    而不是等到第一次写入时抛 ``AttributeError: module has no attribute``。
    """

    def __init__(self) -> None:
        super().__init__()
        del self.IndexFlatIP


class FakeFaissModuleMissingIDSelectorBatch(FakeFaissModule):
    """缺少 ``IDSelectorBatch`` 的变体（删除路径会用到它，同样要在构造时发现）."""

    def __init__(self) -> None:
        super().__init__()
        del self.IDSelectorBatch


def make_fake_faiss_module() -> FakeFaissModule:
    """构造一个全新的假 faiss 模块（**每个测试各拿一个**，避免状态串味）."""
    return FakeFaissModule()


__all__ = [
    "FAISS_ID_NOT_FOUND",
    "METRIC_INNER_PRODUCT",
    "METRIC_L2",
    "FakeFaissModule",
    "FakeFaissModuleMissingIDSelectorBatch",
    "FakeFaissModuleMissingIndexFlat",
    "FakeIDSelectorBatch",
    "FakeIndexFlatIP",
    "FakeIndexFlatL2",
    "FakeIndexIDMap2",
    "make_fake_faiss_module",
    "read_index",
    "write_index",
]
