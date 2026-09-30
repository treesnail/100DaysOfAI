"""检索错误族：分开，因为**修这几族的是几拨不同的人**（M6-D5 / M6-D6 / M6-D7 / M6-D8）.

day064 的 ``vectorstore.errors`` 也分了族（环境/数据/调用/查询），
今天这一层沿用同一条纪律，但**分的依据换了一个**：
向量库那一层按"失败发生在哪"分，检索这一层按"**谁该去修**"分。

```text
族                 现象                                  谁去修
QueryError         空查询 / top_k 越界 / 过滤条件冲突 /     调用方（写错了参数）
                   时间范围倒置 / 路由名不存在 /            → 改调用点，不改库
                   k1 与 b 取值非法                        → 改调用点或改配置
IndexStateError    库为空却要求过滤 / 维度不符 /           索引运维（库与配置对不上）
                   编码器返回坏向量 / 严格模式下清单漂移     → 重建或对齐索引
ContextError       预算放不下一条命中 / 单块上限为 0 /      打包配置（阈值定得太狠）
                   提示词模板缺占位符                       → 调预算与模板
LexicalError       关键词索引与语料对不上（库里报 N 条、    索引运维（索引没跟上语料）
                   只取回 M 条；索引里的文档全无词元）      → 重建关键词索引 / 修入库口径
FusionError        融合策略名不存在 / alpha 越界 /          调用方（写错了参数）
                   k_rrf < 1 / 某一路没有权重               → 改这一次的调用参数
RerankError        重排参数写错：top_n < 1 / weight 越界 /   调用方（写错了参数）
                   mode 不在封闭清单里 / 传进来的"重排器"    → 改这一次的调用参数或装配
                   不是 BaseReranker
GenerationError    生成参数写错：版本名不在封闭清单里 /     调用方（写错了参数）
                   temperature 越界 / max_tokens < 1 /       → 改这一次的调用参数或装配
                   注入的不是 BaseLLM / overrides 键不认识
```

**为什么不合成一个 ``RetrievalError``**：合起来之后，"检索返回 500" 这个现象
就同时对应三种完全不同的动作——查调用点、重建索引、调预算。而这三件事由不同的人
在同一天的不同时间做；混成一族之后，接报的人必须先读代码才能判断该找谁。

**day067 新增的两族沿用同一条判据**（不是按"发生在新代码里"分）：

```text
LexicalError 是 RetrievalError 的直系子类   改调用点是走不通的——关键词索引少了文档，
                                          只能重建它或修入库口径，所以它不进 QueryError
FusionError  继承 QueryError                改这一次的调用参数就能走通
                                          （strategy 拼错、alpha 给了 1.5、k_rrf=0）
```

``FusionError`` 继承 ``QueryError`` 而不是并列：它描述的就是"这次请求的参数不对"，
继承之后 ``except QueryError`` 的老代码**自动**把它收进去（端点层那条
"参数问题一律 400"的通道不需要再加一个分支），而需要区分时又能单独
``except FusionError``——两边的代价都是零，这正是"子类"该有的用法。

**day068 新增的 ``RerankError`` 照抄的就是这一先例**（不是照抄"新代码就新开一族"）：

```text
它的四种触发都是参数问题   top_n < 1 / weight 越界 / mode 不在封闭清单里 /
                          传进来的重排器不是 BaseReranker
处置动作                  改这一次的调用点或改装配，**不需要**动索引、不会重建任何东西
因此                      继承 QueryError（而不是 RetrievalError 直系）
```

与 ``LexicalError`` 并成对照：那一族是"索引没跟上语料"，改调用点走不通，
所以它挂在 ``RetrievalError`` 下；**族的分法看动作，不看是哪一天写的代码**。

**day069 的 ``GenerationError`` 是这条判据的第三次应用**（不是"新模块配新族"）：

```text
它的五种触发都是参数问题   版本名不在封闭清单里 / temperature 越界 /
                          max_tokens < 1 / 注入的不是 BaseLLM /
                          overrides 的键不认识
处置动作                   改这一次的调用点或改装配，**不需要**动索引、不重建任何东西
因此                       继承 QueryError（与 FusionError / RerankError 逐字同理）
```

它**不**收留两类别的失败，那两类各有自己的去处：

```text
模板缺占位符 / 模板不是合法格式串   那是"提示词装配"的问题 → ContextError
调用抛异常 / 模型返回空串 / 没有合法引用
                                  那是**合法状态**，返回带 fallback_reason 的结果
```

第二行是这一层最容易被写错的一处：模型超时与"资料里没有相关内容"在报告里长得
像，但它们不是异常——把可预期的失败抛出去，调用方就只能拿到一个栈，
而拿不到"这次为什么不算数"。

每一族的报错模板（都是"**现象 + 出路**"两句，而不是一个错误码）：

```text
QueryError        "top_k=0 没有意义 → 要探测'库里有没有数据'请用 backend.count()"
IndexStateError   "查询向量 384 维、库 768 维 → 编码器与建库的不是同一套，请重建索引"
ContextError      "一条命中至少需要 32 字，预算只有 20 → 调大 retrieval_max_context_chars"
LexicalError      "库里报 8 条、只取回 7 条 → 记录表与索引不同步，请重建关键词索引"
FusionError       "alpha=1.5 越界 → alpha 是**向量通道的权重**，必须落在 [0, 1]"
RerankError       "top_n=0 不是窗口 → 要关掉重排请把 rerank_enabled 设为 False"
GenerationError   "未知的提示词版本 'v3' → 受管版本只有 ['v1', 'v2']，请用受管版本号"
```

这几族都**不是**"程序崩了"：它们全是**可预期的拒绝**，因此消息里必须带上出路。
真正不在预期内的异常（例如某个后端实现里有个 ``TypeError``）**不上抛到本族**——
那种错误必须让调用栈把它喊出来（与 ``vectorstore.pipeline`` 的同一条纪律）。
"""

from __future__ import annotations


class RetrievalError(Exception):
    """本包所有错误的基类（``except RetrievalError`` 能一次收全）.

    捕获它的人通常是端点层与演示脚本：它们要把"检索没做成"渲染成一条
    人类可读的结论。**但端点层绝不该用它兜住内部异常**——
    这条纪律写在 ``pipeline`` 的那一层，本类只是让"刻意抛出的失败"
    有一个共同的祖先。
    """


class QueryError(RetrievalError):
    """调用方的问题：空查询 / top_k 越界 / 过滤条件冲突 / 时间范围倒置 / 路由名不存在.

    判据只有一条：**这条路改调用点就能走通**，不需要动索引、不需要重建库。
    例如 ``top_k=0``、``{"$and": [{"a": 1}]}`` 里把 ``a`` 同时写成时间字段、
    ``route="手册库"`` 而这个名从未注册过——这些都是"再问一次就行了"。

    模板：``QueryError("空查询在向量检索里没有定义：请给一段非空文本")``
    ——现象（空查询）后面**紧跟**出路（给一段非空文本）。
    """


class IndexStateError(RetrievalError):
    """库侧的问题：库为空却要求过滤、维度不符、编码器返回坏向量、严格模式下清单漂移.

    判据与 ``QueryError`` 互补：**这条路改调用点是走不通的**，
    必须去动索引（重建、对齐编码器、修清单）。因此把这族单独拿出来，
    是为了让"检索器说库不对"这件事有一个不会被误读成"我参数写错了"的载体。

    模板：``IndexStateError("查询向量 768 维、本库 384 维：编码器与建库的
    不是同一套，请用同一个提供方重建索引")``。
    """


class ContextError(RetrievalError):
    """上下文打包与提示词装配的问题：预算放不下一条命中、单块上限为 0、模板缺占位符.

    这族看着最琐碎，却是唯一一族**只与配置有关**的错误：
    它跟库、跟查询、跟编码器都无关，只跟"你打算给模型留多少空间"有关
    （见 ``config.retrieval_max_context_chars`` / ``retrieval_per_hit_chars``）。

    模板：``ContextError("一条命中至少需要 32 字，本预算只给了 20 字：
    请调大 retrieval_max_context_chars 或降低 min_hit_chars")``。
    """


class LexicalError(RetrievalError):
    """关键词索引侧的问题：索引里的文档与库里对不上、索引里的文档全无词元.

    day067 新增，判据与 ``IndexStateError`` 完全一致——**改调用点走不通**，
    只能去动索引。它与 ``IndexStateError`` 并列而不是合并，理由只有一个：
    出问题时"该重建的是哪一个索引"是接报人的第一个问题，而
    ``向量库对不上`` 与 ``关键词索引对不上`` 的处置动作不同
    （重建向量库要走 day065 的 indexing，重建关键词索引只是
    ``LexicalIndex.from_backend(backend)`` 一次调用）。

    模板：``LexicalError("库里报 8 条、只取回 7 条：记录表与索引不同步，
    请重建关键词索引（LexicalIndex.from_backend）")``。
    """


class FusionError(QueryError):
    """融合参数的问题：策略名不存在 / ``alpha`` 越界 / ``k_rrf < 1`` / 某一路没有权重.

    继承 ``QueryError``（见模块 docstring 的说明）：它就是"这次请求的参数不对"，
    因此（1）``except QueryError`` 能一次收全，（2）端点那条"参数问题一律 400"
    的通道不必再加一个分支，（3）需要区分时仍可单独 ``except FusionError``。

    模板：``FusionError("alpha=1.5 越界：alpha 是**向量通道的权重**，必须落在 [0, 1]；
    关键词通道的权重是 1 - alpha，因此给它 1.5 等于给关键词路 -0.5")``。
    """


class RerankError(QueryError):
    """重排参数的问题：``top_n < 1`` / ``weight`` 越界 / ``mode`` 不在封闭清单里 /
    传进来的重排器不是 ``BaseReranker``.

    **继承 ``QueryError``，理由与 ``FusionError`` 逐字相同**：这四件事都属于
    "这次调用/这次装配写错了参数"，改一处就能走通——不需要动索引、不需要重建
    任何东西。因此（1）``except QueryError`` 能一次收全，（2）端点那条
    "参数问题一律 400"的通道不必再加分支，（3）需要区分时仍可单独
    ``except RerankError``（例如想把"重排参数写错"与"融合参数写错"分别记日志）。

    它**不**收留两类别的失败，那两类各有自己的族：

    ```text
    重排器是坏的（编码器/模型返回坏数值）  那是"实现或装配对不上"，不是参数问题
    上游给不出候选（阈值/融合之后为空）    那是合法状态，返回空结果而不是报错
    ```

    模板：``RerankError("top_n=0 不是窗口：它必须 >= 1；要关掉重排请把
    rerank_enabled 设为 False（'只对前 0 条打分'不是一个请求）")``。
    """


class GenerationError(QueryError):
    """生成参数的问题：版本名不在封闭清单里 / ``temperature`` 越界 /
    ``max_tokens < 1`` / 注入的不是 ``BaseLLM`` / ``overrides`` 的键不认识.

    **继承 ``QueryError``，理由与 ``FusionError`` / ``RerankError`` 逐字相同**：
    这五件事都属于"这次调用或这次装配写错了参数"，改一处就能走通——不需要动
    索引、不需要重建任何东西。因此（1）``except QueryError`` 能一次收全，
    （2）端点那条"参数问题一律 400"的通道不必再加分支，（3）需要区分时仍可
    单独 ``except GenerationError``（例如把"生成参数写错"与"重排参数写错"
    分别记日志）。

    它**不**收留两类别的失败，那两类各有自己的族（见模块 docstring）：

    ```text
    模板缺占位符 / 不是合法格式串     那是提示词装配 → ContextError
    调用抛异常 / 空回复 / 没有合法引用  那是合法状态 → 带 fallback_reason 的结果
    ```

    模板：``GenerationError("未知的提示词版本 'v3'：受管版本只有 ['v1', 'v2']；
    版本号是报告与评估的分组键，清单外的取值会让'这一版好不好'无法与任何一版
    模板对上。出路：用上面那几个版本号，或在装配期注入 prompt=...")``。
    """


__all__ = [
    "ContextError",
    "FusionError",
    "GenerationError",
    "IndexStateError",
    "LexicalError",
    "QueryError",
    "RerankError",
    "RetrievalError",
]
