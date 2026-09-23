"""``rag_ops`` 的失败族：**这一层会怎么失败，以及每一种该谁去修**（day072）.

分族的依据与 ``rag_debug.errors`` / ``retrieval.errors`` 逐字相同——
**按"谁的错、该谁去修"分**，而不是按"哪一行抛的"。运维层的报告里最有用的
那句话同样不是"报错了"，而是"这个错该改什么"：

```text
SyncError      账本或快照不可信（水位脏了、快照身份不符、路径非法）  → 修数据 / 新开账本
ScheduleError  调度参数或时钟用错了（间隔为负、抖动比间隔还大）      → 改调用
HealthError    健康判据用错了（不认识的检查项、状态表缺键）          → 改调用
MetricsError   监控采样用错了（不认识的指标名、单位不符、时间倒序）  → 改调用
DeployError    编排规格不合规（缺探针、镜像没打 tag、卷没声明）      → 改规格
RagOpsError    本族基类：其余"这一层自己的拒绝"
```

五族都继承 ``ValueError``（与 ``RetrievalError`` / ``RagDebugError`` 同一句话）：
参数写错是**调用方**的问题，因此端点的"参数一律 400"不必为这一族加分支
（见 ``api.routes`` 里 ``except RagOpsError`` 的那一处）。

## 为什么 SyncError 与 DeployError 必须分开

它们是这一课最容易混为一谈的两类失败，而处置动作完全不同：

```text
SyncError    同步的**状态**不对   → 下一页要读的账已经不可信，先修账再跑
DeployError  编排的**规格**不对   → 根本没打算跑起来，先改 compose 再发布
```

把两者合成一个 ``RagOpsError`` 之后，报告里只会剩下一句"运维失败"——
而"账脏了要重建"与"compose 写错了要改文件"是两个人、两个时间点要做的事。
"""

from __future__ import annotations


class RagOpsError(ValueError):
    """``rag_ops`` 这一族错误的基类（判据与 ``RagDebugError`` 同源）."""


class SyncError(RagOpsError):
    """同步的账本或快照不可信（水位指向不存在的快照、身份长度不对、来源路径非法）.

    出路永远是**先把账修好**，而不是"忽略这一条继续跑"：
    水位回退一次，下一次增量就会把**已经处理过的来源当成新的**重跑一遍
    （重复写库不报错），或者把**真正变了的来源当成没变**（漏更新也不报错）。
    两种都没有异常指向它，因此本族一律当场拒绝。
    """


class ScheduleError(RagOpsError):
    """调度参数或时钟用错了（间隔 <= 0、抖动 >= 间隔、退避上限小于基数）.

    出路是**改调用**：调度参数不是"运行期数据"，它们是被写进配置的一次决定。
    """


class HealthError(RagOpsError):
    """健康判据用错了（不认识的检查项、没见过的状态、缺键的检查表）.

    出路是**改调用**：检查项是一张封闭清单（``HEALTH_CHECKS``），
    自由文本会让"这个实例到底查过哪几项"无法被统计。
    """


class MetricsError(RagOpsError):
    """监控采样用错了（不认识的指标名、单位与指标不符、同一序列时间倒序）.

    出路是**改调用**：指标名是一张封闭清单（``OPS_METRICS``），
    而"顺手记一个自定义名字"的后果是它永远进不了汇总表——
    与"这个指标一直是正常值"在报告里长得一样。
    """


class DeployError(RagOpsError):
    """编排规格不合规（缺探针、镜像没打 tag、卷没在顶层声明、端口撞车）.

    出路是**改规格**：这些规则全部可以在发布之前静态查出来，
    而放到运行期去发现的代价是"容器起来了，但库是空的、镜像不可追溯"。
    """


__all__ = [
    "DeployError",
    "HealthError",
    "MetricsError",
    "RagOpsError",
    "ScheduleError",
    "SyncError",
]
