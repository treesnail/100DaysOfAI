"""``hf_integration`` 的失败族：**把生态接进来时，每一种失败该谁去修**（day086 / M7-D10）.

分族的依据与前十天逐字相同——**按“谁的错、该谁去修”分**。今天这件事第一次从
"模型的算术"挪到了"模型的**搬运与装配**"上，因此族的名字也换了一批：

```text
ShapeError       形状不符     padding 长度与 mask 不齐、池化维度对不上、
                              注意力掩码与输入行数不一致                          → 改调用
ParameterError   参数越界     max_length 非正、padding 策略不认识、
                              top_k 与词表冲突、把生成管线当特征管线用              → 改调用
ConfigError      配置不成立   config.json 缺键、类型不符、头数不能整除隐藏维、
                              架构名不认识                                        → 改模型或改配置
HubError         缓存/解析失败 仓库不存在、版本不存在、离线模式下缓存里没有这个文件  → 改环境
TokenError       词表不成立   未知 token id、merges 里有非法行、词表与 tokenizer 不同源 → 改数据
AssemblyError    组装不成立   两个各自合法的部件拼不成一个合法整体（名单对不上、
                              双向模型上整批跑、词表与 id 不同源）                     → 改组装
NumericError     数值不可用   mask 全 0 的均值池化、非有限数、一行之和不是 1        → 改数据或改实现
IntegrationError 本族基类
```

## 一、本课**仍然缺席**的那一族：``GradientError``

day085 缺席它的理由是"源码精读读的是推理路径"。**今天这个理由更强了**：
本课一行反向都不写——`from_pretrained` 之后，模型的每一层都已经是别人写好的；
本课做的是**解析、缓存、分词、池化、编排**，它们全部落在**前向之外**。

```text
day075~081   有手写反向            ⇒ 有 GradientError 那一族
day082~086   没有手写反向          ⇒ 这一族连续缺席五天
```

一条纪律在这里第三次兑现：**能"改推导"的地方必须是有人真写了推导的地方。**

## 二、本课**新增**的三族：``ConfigError`` / ``HubError`` / ``TokenError``（以及 ``AssemblyError``）

前三天的族全部围绕"我算出来的数不对"。今天第一次出现三类**与算术无关**的失败：

```text
ConfigError   "这份配置描述的模型不成立"   n_embd=16 而 n_head=5
                                        （16 不能被 5 整除 ⇒ 没有 head_dim 这个整数）
HubError      "东西根本不在本地"           local_files_only=True 而缓存里没有这个 revision
                                        （**它不是参数错**：参数全对，只是环境里没有）
TokenError    "词表对不上"                vocab.json 里有 100 个 id，而 merges.txt
                                        里的合并对引用了第 101 个
```

三者的出路各不相同，而**这正是它们必须分开的理由**：

```text
ConfigError  → 改配置或换模型      （数得改，或者换个尺寸）
HubError     → 改环境             （联网、换 revision、或把 local_files_only 关掉）
TokenError   → 改数据             （换成与该 tokenizer 同源的那份文本/词表）
```

``AssemblyError`` 与本包一天前遇到的那一族同名同类：它说的是"两边各自都对，
而拼起来不成立"——因此修法是**同时看两边**，不是改一个数。

day070 起的那条纪律在这里第四次兑现：
**"没写该怎么办"与"这一族不需要处理"读起来是一样的**，因此三族的出路必须分别写出来。

## 三、跨包的继承关系（**这张图也要写下来**）

```text
ValueError
├── MathError（day073）
├── TransformerError（day075）
├── EncoderDecoderError（day079）
├── VariantError（day082）
├── ExplainError（day083）
├── SourceError（day085）
└── IntegrationError（day086，本模块）
    ├── ShapeError / NumericError / ParameterError   （多继承：既是 day075 族、也是本族）
    ├── HubError                                      （本层独有：它**不是**参数错）
    └── ConfigError → ParameterError                  （特例：配置也是一种调用参数）
        TokenError  → ParameterError                  （同上：词表也是一种入参）
        AssemblyError → ParameterError                （同上：组装不成立也是一种参数失败）
```

与前两层一样，本层**继承 day075 的三族而不是彼此**：``hf_source`` 与本包是
**兄弟**。因此 ``except transformer_core.errors.ShapeError`` 能同时兜住六层，
而 ``except hf_source.errors.ShapeError`` **兜不住本层**——
这一条有它自己的用处：day085 的读数与 day086 的读数是两件事，
它们在报告里必须能分开统计。
"""

from __future__ import annotations

from smart_research_agent.transformer_core.errors import (
    NumericError as CoreNumericError,
)
from smart_research_agent.transformer_core.errors import (
    ParameterError as CoreParameterError,
)
from smart_research_agent.transformer_core.errors import (
    ShapeError as CoreShapeError,
)


class IntegrationError(ValueError):
    """``hf_integration`` 这一族错误的基类（判据与 ``SourceError`` 同源）."""


class ShapeError(CoreShapeError, IntegrationError):
    """形状不符：padding 长度与 mask 不齐、池化的行数对不上、掩码行数与输入不一致.

    出路是**改调用**。同时继承 ``transformer_core.errors.ShapeError``：
    前天写的 ``except ShapeError`` 代码不需要改动就能兜住本层的失败。
    """


class ParameterError(CoreParameterError, IntegrationError):
    """参数越界：``max_length`` 非正、``padding`` 策略不认识、``top_k`` 非正.

    出路是**改调用**。本包不接受"参数不合法就取个默认值"这种兜底——
    一个被兜住的 ``max_length=0`` 会退化成"原样返回"，而它与"这一批本来就很短"
    在前几步的输出上**看起来一样**（day081 第 10.2 节的'漏传'在这里第四次现形）。
    """


class NumericError(CoreNumericError, IntegrationError):
    """数值不可用：非有限数、一行之和不是 1、掩码全 0 的均值池化.

    第三条值得单独说：均值的分母是 ``mask`` 的和，而一行全 0 的掩码会让它变成
    ``0/0``。Hugging Face 的实现里这会产生一个 ``nan`` 并**安静地传下去**，
    本包把它当场拒绝——一旦 ``nan`` 进入池化结果，它会在相似度、聚类、
    检索排序里表现出"这条和谁都不像"，而那看起来像一条**数据结论**。
    """


class ConfigError(ParameterError):
    """配置不成立：``config.json`` 缺键、类型不符、头数不能整除隐藏维、架构名不认识.

    它与 :class:`ParameterError` 的差别不在"合法性"，而在**来源**：
    ``ParameterError`` 说的是"你这次调用的一个数写错了"，
    而 ``ConfigError`` 说的是"这份配置文件描述的模型不成立"——
    后者的修法通常是**换一个模型**或**补上缺的那几个键**，而不是改一个调用参数。

    值得单独指出的第一条：``n_embd=16`` 而 ``n_head=5`` 不是"参数越界"，
    而是**这个模型没有定义**（``head_dim = 16/5`` 不是整数）。
    真实装载时它会以一个 ``reshape`` 失败的形式出现，
    而本包在**解析配置的那一刻**就拒绝——报错点离出错点越近，修起来越便宜。
    """


class HubError(IntegrationError):
    """缓存与解析失败：仓库不存在、版本不存在、离线模式下缓存里没有这个文件.

    这是本课程**第一次**出现"不是参数错"的一族：调用方把每一个参数都写对了，
    而环境里就是没有那份东西。因此它**不继承** :class:`ParameterError`——
    把它归到"改调用"里会指向一个错误的修法（把 ``local_files_only`` 改成
    ``False`` 只是把失败推迟到网络上，而网络那一步的失败长得**完全一样**）。

    三句真实世界里最常见的原文都归这一族：

    ```text
    RepositoryNotFoundError      仓库名写错 / 私有库没有权限
    RevisionNotFoundError        分支或 commit 不存在
    LocalEntryNotFoundError      离线（或断网）而缓存里没有这个文件
    ```
    """


class TokenError(ParameterError):
    """词表不成立：未知 token id、``merges.txt`` 里有非法行、词表与合并表不同源.

    它与 :class:`ParameterError` 的差别在**数据侧**：
    "``max_length=0``"是调用方写错了一个数，
    而"``merges.txt`` 第 42 行引用了词表里没有的片段"是**那份文件坏了**。

    本族里最值钱的一条是**同源性**：``vocab.json`` 与 ``merges.txt``
    必须来自同一个 tokenizer。两个不同来源的文件**能拼出一个可用的分词器**，
    而它切出来的 token 与训练时不一致——那时模型不会报错，
    只会给出**看起来通顺但语义漂移**的结果。
    """


class AssemblyError(ParameterError):
    """组装不成立：两个各自合法的部件拼不成一个合法的整体.

    三条典型情形（都是"形状对、语义错"那一族）：

    ```text
    性质名单对不上      跑出来的结论集合与 types 里那份名单不是同一个集合
                        ⇒ 报告里会安静地少一行或多一行
    填充策略与模型不配  双向模型上仍然整批跑（填充会改写真实位置那一行）
    词表与 id 不同源    一个能跑通的组合，切出来的 token 却是别人的
    ```

    它与 day085 的同名族**恰好同名同类**（都继承 :class:`ParameterError`），
    而理由也一样：它要求调用方**同时**看两个部件，修法与"改一个数"不同。
    """


#: 七个族各自“该谁去修”（**这张表要被测试逐键检查**）.
FAMILY_OUTCOMES: dict[str, str] = {
    "ShapeError": "改调用：padding 与 mask、池化与行数、掩码与输入必须先对齐",
    "ParameterError": "改调用：max_length / padding / top_k 都是调用点的一次决定",
    "ConfigError": "改模型或改配置：缺键、类型、头数与隐藏维属于'这份配置不成立'",
    "HubError": "改环境：联网、换 revision、或先补缓存——它**不是**参数错",
    "TokenError": "改数据：词表与合并表必须同源，id 必须都在表里",
    "AssemblyError": "改组装：两份各自合法的东西拼不成立时，两边要**同时**看",
    "NumericError": "改数据或改实现：非有限数、和不为 1、全 0 掩码的均值都属于'数值不可用'",
}

#: 本模块真正导出的七个异常类名（与 ``FAMILY_OUTCOMES`` 逐键对齐的闭合检查）.
_FAMILY_CLASSES: dict[str, type[BaseException]] = {
    "ShapeError": ShapeError,
    "ParameterError": ParameterError,
    "ConfigError": ConfigError,
    "HubError": HubError,
    "TokenError": TokenError,
    "AssemblyError": AssemblyError,
    "NumericError": NumericError,
}

if set(FAMILY_OUTCOMES) != set(_FAMILY_CLASSES):  # pragma: no cover - 只在有人改表时触发
    raise IntegrationError(
        "失败族的两张表不一致：FAMILY_OUTCOMES 的键必须与 _FAMILY_CLASSES 逐一对应——"
        "少一个键的那一族在报告里只剩类名、没有'该怎么办'，"
        "而'没写该怎么办'与'这一族不需要处理'读起来是一样的。"
    )

#: 本课**连续缺席**的那一族（写进常量，让"缺席"是一个可断言的事实而不是一段散文）.
ABSENT_FAMILY = "GradientError"

#: 缺席的理由（与 ``hf_source`` 的那一条**不同**，这一条要写清楚）.
ABSENT_FAMILY_REASON = (
    "本课一行反向都不写：from_pretrained 之后每一层都是别人写好的，"
    "本课做的是解析、缓存、分词、池化与编排，全部落在前向之外。"
    "能'改推导'的地方必须是有人真写了推导的地方。"
)

__all__ = [
    "ABSENT_FAMILY",
    "ABSENT_FAMILY_REASON",
    "FAMILY_OUTCOMES",
    "AssemblyError",
    "ConfigError",
    "HubError",
    "IntegrationError",
    "NumericError",
    "ParameterError",
    "ShapeError",
    "TokenError",
]
