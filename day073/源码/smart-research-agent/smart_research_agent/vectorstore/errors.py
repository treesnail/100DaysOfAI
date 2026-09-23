"""向量库会怎么失败（M6-D3）.

day061 的失败族是"文件打不开 / 格式不认识"，day062 的是"参数自相矛盾"。
向量库的失败族与它们都不同：**入口是一次批量写入，出口是一次近似度量的排序**，
两条路都会以"结果看起来还行"的方式坏掉。因此这里把异常分成四族，
每一族对应**不同的修复人**：

```text
BackendUnavailable   环境问题：没装 faiss / chromadb / 客户端连不上
                     → 修依赖，或换一个后端（不是改数据）
VectorError          数据问题：维度不一致 / 分量不是有限数 / 混入零向量
                     → 重新编码，或清库重建（不是改调用方）
RecordError          调用问题：id 为空、add 撞上已存在的 id、元数据类型越界
                     → 改调用方（不是改环境）
FilterError          查询问题：where 子句写法不合法
                     → 改查询（与本条记录无关，是查询本身错了）
```

把四族混成一个大 `VectorStoreError` 是很常见的做法，代价是**报错之后的
第一个动作永远要靠猜**：看到"向量库错误"你无法判断该 `pip install`、
该重跑 embedding，还是该改一行调用代码。四族分开，报错信息自己就带上了
该做什么（见每个异常类的 docstring）。

一个具体的例子说明为什么 `VectorError` 与 `RecordError` 必须分开：

```text
维度不一致    384 维的库被写进 768 维的向量   → 是"库与编码器不匹配"，要重建库
id 重复       add() 撞上已存在的 id          → 是"调用方该用 upsert"，数据没问题
```

前者换一个调用方式也修不好，后者改一个方法名就好了。
"""

from __future__ import annotations


class VectorStoreError(Exception):
    """向量库相关错误的基类.

    只用来做"一类都能接住"的兜底，**不携带任何具体语义**——
    需要判断该做什么的时候，请分别捕获下面四个子类。
    """


class BackendUnavailable(VectorStoreError):
    """后端不可用：缺少可选依赖，或外部服务不可达.

    典型触发点：本机没有安装 ``faiss-cpu`` / ``chromadb``，
    或者 Chroma 的服务端模式连不上。

    **这是一种环境事实，不是数据问题**：同一份数据换个后端就能写进去，
    所以这条异常的出现绝不意味着需要重建索引。构造它的代码应当把
    安装指引一并写进消息（见 ``registry.resolve_backend``）——
    报错的价值在于"下一步做什么"，而"缺少模块 faiss"这句话本身
    并没有告诉任何人该装哪个包。
    """


class VectorError(VectorStoreError):
    """向量本身不合法：维度不一致、分量非有限、零向量.

    这一族错误指向**数据与库的匹配关系**，而不是调用方式：

    - 维度不一致：库已经按 384 维建好，写进来一个 768 维的向量
      （常见原因：换过 embedding 提供方但没重建库）；
    - 分量非有限：``nan`` / ``inf``。它们不会让写入失败，
      但会让**之后每一次排序都是错的**，且错得不报错——
      ``nan`` 参与的任何比较都返回 False，于是这条记录会落到任意位置；
    - 零向量：方向没有定义，余弦相似度对它是 0/0。
    """


class RecordError(VectorStoreError):
    """记录不合法：id 缺失/超长、``add`` 撞上已存在的 id、元数据类型越界.

    最后一条最容易被忽略：向量库对元数据的**类型**是有硬约束的
    （Chroma 只接受字符串、整数、浮点数、布尔值，以及**同类型**标量数组）。
    本包把这条约束提升为公共契约（见 ``types.VectorRecord``），
    因为一个"在纯内存后端跑通、换 Chroma 就 500"的数据形状，
    不该等到上线那天才被发现。
    """


class FilterError(VectorStoreError):
    """``where`` 子句不合法：未知运算符、值类型不对、``$and`` 不是列表.

    与本条记录无关——**是查询本身写错了**。因此它的消息里必须列出
    支持的运算符（见 ``filters.SUPPORTED_OPERATORS``），
    让写错的查询能自己改对，而不是去搜文档。
    """


__all__ = [
    "BackendUnavailable",
    "FilterError",
    "RecordError",
    "VectorError",
    "VectorStoreError",
]
