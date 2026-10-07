"""后端注册表：把"选哪个向量库"变成一次**可解释的决定**（M6-D3）.

到这一步为止，``flat`` / ``faiss`` / ``chroma`` 各自回答了"怎么做"。
但没有人回答那个更早的问题：**该用哪一个，以及为什么现在用不了。**
两种常见的答法都不行：

```text
try:
    import faiss                     # 一个 import 决定整条链路能不能启动
    backend = "faiss"
except ImportError:
    backend = "flat"                 # 静默降级：使用者以为自己在用 faiss
```

第一种把"用哪个后端"变成了一个模块级副作用（没装 faiss 的人连 flat 都用不了）；
第二种更糟——它**不报错**，只是检索质量与预期不同，而报告里没有任何痕迹。

本模块给出第三种答法：把后端当成一份**声明**（``BackendProfile``）。

```text
BackendProfile  它需要什么依赖 / 能不能落盘 / 元数据在不在同一个库
                / 缺依赖时敲哪条命令 / 它适合什么、代价是什么
describe_backends()   把上面这些 + "这台机器上现在能不能用" 一次说清（可直接 json.dumps）
```

于是"选后端"这件事**在不 import 任何可选库的前提下**就能被回答，
端点 ``GET /vectorstore/backends`` 可以把答案原样交给使用者。

## 三段式错误信息：本模块存在的核心价值

"缺少模块 faiss"这句话没有告诉任何人下一步该做什么。因此本模块报的
每一条不可用错误都必须是三句：

```text
后端 faiss 不可用：缺少 faiss（numpy 已满足）。
安装：pip install faiss-cpu
或者改用：flat（零可选依赖，结果逐位可复现）
```

```text
缺什么          让使用者知道是自己的环境还是别人的代码
怎么装          一条**可复制粘贴**的命令（不是"请安装 faiss"）
还能用什么      现场可用的替代品，并且带上它适合什么
```

第三句最容易被省掉，也最有用：进度不该被一个可选依赖卡死。
这也是 ``flat`` 在 profile 表里排第一的原因——它 ``requires=()``，
**永远可用**，因此"或者改用"这一句在任何机器上都给得出一个真答案。

## 延迟导入：为什么 registry 自己不能 import 那两个后端

``faiss_backend`` 在模块级依赖 ``numpy``；``chroma_backend`` 会连带导入
本包 llm 层。若 ``registry`` 在模块顶层 import 它们，那么"只想用 flat 的人"
会在 **import 阶段**崩掉——而 flat 明明零依赖。因此：

```text
注册表本身        只依赖 stdlib + 本包的核心五模块 + config（都不含可选依赖）
BACKEND_FACTORIES 每个值是一个**函数**，真正的 import 发生在函数体里（调用时）
resolve_backend   ImportError → BackendUnavailable（带上面那三句），而不是让它冒泡
```

顺带一个必须显式说的实现细节：``missing_requirements`` 调用
``importlib.util.find_spec`` 时用的是**属性访问**（``importlib.util.find_spec(...)``）
而不是 ``from importlib.util import find_spec``。区别在于前者可以被
monkeypatch 替换——"缺依赖"与"依赖齐全"两条分支因此都能在测试里
被真实走到，而不是靠"本机装没装 faiss"来决定覆盖了哪一半。

## 三层优先级：settings 是基线，override 是第二级

```text
第一级  config.Settings 的 vector_* 字段         项目级基线（"这个库怎么跑"）
第二级  create_backend(**overrides)             单次调用覆盖（脚本 / 端点参数）
第三级  调用点显式传参（resolve_backend 的具名参数）  最具体的一次
```

与 day062 的"默认参数 + 单次覆盖"是同一条纪律：**基线集中在一处，
覆盖只能收紧、不能发散**。``vector_persist_path`` 为空串时传 ``path=""``
——即"明确不落盘"，而不是"替我挑一个默认目录"（理由见 ``config.py``）。

## 谁依赖它

```text
vectorstore/__init__.py            装配阶段导出 backend_names / describe_backends
evaluate.compare_backends（E 号）  用 is_available 把"不可用"变成一条报告行而不是异常
pipeline（F 号）                    create_backend 拿到的实例就是它的后端
scripts/vectorstore_demo.py         演示脚本按 settings 选后端
GET /vectorstore/backends           直接把 describe_backends() 当响应体
```

## 这一层刻意不做的事

```text
不自己实现后端       只做"选择 + 构造"，六个原语属于各后端（见 base.py）
不静默降级           缺依赖就报错并给指引，不偷偷换成 flat
不给后端发明参数     每个后端只收到它自己接受的键（见 _BACKEND_PARAMS 表）
不做运行期切换       后端与度量在构造时定下（见 base.VectorBackend.metric 的说明）
不缓存可用性         find_spec 的结果会随"刚装完一个包"变化，缓存只会带来陈旧答案
```
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from smart_research_agent.config import settings as app_settings
from smart_research_agent.vectorstore.base import VectorBackend
from smart_research_agent.vectorstore.errors import BackendUnavailable, VectorStoreError
from smart_research_agent.vectorstore.metrics import METRIC_COSINE, normalize_metric

#: 三个后端的名字。写成常量而不是各处字面量，是因为它们同时出现在
#: profile 表、工厂表、参数表、错误信息与端点响应里——**五处必须逐字一致**。
BACKEND_FLAT = "flat"
BACKEND_FAISS = "faiss"
BACKEND_CHROMA = "chroma"


@dataclass(frozen=True)
class BackendProfile:
    """一个后端的**声明**：它需要什么、适合什么、缺依赖时怎么办.

    为什么用数据而不是"从类属性上反射"（``FaissVectorStore.requires`` 里
    已经有依赖列表了）：反射要求**先把模块 import 进来**——而那正是本模块
    要避免的事（见模块 docstring 的延迟导入）。代价是这张表与三个类
    之间可能出现不一致，因此由测试逐字段对账
    （``test_vectorstore_registry.py`` 里把两边钉在一起）。

    字段分成三组，服务三类读者：

    ```text
    决策用   description / notes              → 人：该选哪一个
    环境用   requires / install_hint          → 运维：缺什么、怎么装
    运行用   supports_persistence / native_metadata / default_metric → 代码：怎么用
    ```
    """

    #: 后端名（``resolve_backend`` 的键，端点会回显）.
    name: str
    #: 一句话：它适合什么（"或者改用：flat（**这一句**）"直接用它）.
    description: str
    #: 需要的第三方包（``flat`` 为空元组 = 永远可用）.
    requires: tuple[str, ...]
    #: 能否把状态落到磁盘.
    supports_persistence: bool
    #: 向量与元数据是否在同一个存储里（决定 ``_find_ids`` 是扫表还是问原生库）.
    native_metadata: bool
    #: 默认度量。三个后端都是 ``cosine``——但**仍然写出来**：它是"选后端时
    #: 度量不该跟着变"这句话的凭据，而不是三个碰巧相同的字面量。
    default_metric: str
    #: 缺依赖时给出的**可执行**安装命令；``flat`` 为空串（它没有可选依赖）.
    install_hint: str
    #: 一句话代价/限制（"它放弃了什么"，与各后端模块 docstring 的取舍表对应）.
    notes: str


#: 后端清单。**顺序就是报告与端点的顺序**：按"依赖递增 / 可控性递减"阅读
#: ——先看零依赖可复现的 flat，再看只管向量的 faiss，最后看自带存储的 chroma。
BACKEND_PROFILES: tuple[BackendProfile, ...] = (
    BackendProfile(
        name=BACKEND_FLAT,
        description="零可选依赖，结果逐位可复现",
        requires=(),
        supports_persistence=True,
        native_metadata=True,
        default_metric=METRIC_COSINE,
        install_hint="",
        notes="每次查询全量扫描（O(N·d)），整库必须放进内存",
    ),
    BackendProfile(
        name=BACKEND_FAISS,
        description="只管向量的精确索引，元数据要自己管",
        requires=("faiss", "numpy"),
        supports_persistence=True,
        native_metadata=False,
        default_metric=METRIC_COSINE,
        install_hint="pip install faiss-cpu",
        notes="维度必填；带 where 的检索要全量 search 之后再筛",
    ),
    BackendProfile(
        name=BACKEND_CHROMA,
        description="向量与元数据在同一个集合里，原生支持 where",
        requires=("chromadb",),
        supports_persistence=True,
        native_metadata=True,
        default_metric=METRIC_COSINE,
        install_hint="pip install chromadb",
        notes="依赖一个外部客户端/目录；自带 HNSW，默认参数会影响结果",
    ),
)

#: name → profile（内部查表用；对外一律走 ``backend_profile`` 以获得校验与报错）.
_PROFILES_BY_NAME: dict[str, BackendProfile] = {
    profile.name: profile for profile in BACKEND_PROFILES
}

#: 每个后端构造时**只**接受这些参数。这张表是本模块对 D.3 第 4 条的落实：
#: ``chroma`` 需要 ``collection``，``flat``/``faiss`` 没有这个概念——
#: 给它们传 ``collection`` 会得到一个 ``TypeError``，而"多传一个参数"
#: 这种错误在调用方看起来完全不像配置问题。**在分派处就按表裁剪**，
#: 让"不是所有后端都接受同样关键字"这件事只在这里出现一次。
_BACKEND_PARAMS: dict[str, tuple[str, ...]] = {
    BACKEND_FLAT: ("metric", "dimension", "path"),
    BACKEND_FAISS: ("metric", "dimension", "path"),
    BACKEND_CHROMA: ("metric", "dimension", "path", "collection"),
}

#: ``create_backend`` 允许用短名覆盖的配置项（其余键按"后端专有参数"透传）.
CREATE_OVERRIDE_KEYS: tuple[str, ...] = (
    "backend",
    "metric",
    "dimension",
    "path",
    "collection",
)

#: 查询期参数：不属于"怎么建库"，而是"怎么搜库"。它们在 ``pipeline`` 里
#: 有一条自己的三级优先级链（调用参数 > 构造参数 > settings），
#: 因此 ``create_backend`` 明确拒收而不是默默忽略（理由见 ``create_backend``）。
QUERY_TIME_KEYS: tuple[str, ...] = ("top_k", "min_score")


# --------------------------------------------------------------------------- #
# 工厂：延迟导入的唯一落点
# --------------------------------------------------------------------------- #


def _build_flat(**params: Any) -> VectorBackend:
    """构造 flat 后端（函数内 import，见模块 docstring）.

    flat 零依赖，本可以在模块顶层 import；三个工厂写成同一种形状是为了
    **让"新增第四个后端"只有一种改法**——照着抄一个函数、在两张表里各加一行。
    混着写（有的顶层导、有的函数内导）时，下一个人很容易照着顶层那种抄，
    而那正好会把"缺库就崩在 import 阶段"重新引进来。
    """
    from smart_research_agent.vectorstore.flat import build_flat_store

    return build_flat_store(**params)


def _build_faiss(**params: Any) -> VectorBackend:
    """构造 faiss 后端（函数内 import：``faiss_backend`` 模块级依赖 numpy）."""
    from smart_research_agent.vectorstore.faiss_backend import FaissVectorStore

    return FaissVectorStore(**params)


def _build_chroma(**params: Any) -> VectorBackend:
    """构造 chroma 后端（函数内 import：本包 llm 层的可用性是调用期事实）."""
    from smart_research_agent.vectorstore.chroma_backend import ChromaVectorStore

    return ChromaVectorStore(**params)


#: name → 构造函数。**值本身不含 import**：真正的 import 在被调用时发生，
#: 所以把这张表放在模块顶层是安全的（这正是"延迟导入"要的形状）。
BACKEND_FACTORIES: dict[str, Callable[..., VectorBackend]] = {
    BACKEND_FLAT: _build_flat,
    BACKEND_FAISS: _build_faiss,
    BACKEND_CHROMA: _build_chroma,
}


# --------------------------------------------------------------------------- #
# 声明查询：不 import 任何可选库
# --------------------------------------------------------------------------- #


def backend_names() -> tuple[str, ...]:
    """全部后端名（顺序 = 阅读顺序 = 端点回显顺序）."""
    return tuple(profile.name for profile in BACKEND_PROFILES)


def backend_profile(name: str) -> BackendProfile:
    """取一个后端的声明；未知名字 → ``VectorStoreError`` 并列出可选值.

    与 ``metrics.normalize_metric`` 同一条纪律：**宁可报错，不要挑一个
    "看起来差不多"的后端**。把 ``"semantic"`` 静默当成 flat 的代价是
    使用者以为自己在用某个语义后端，而实际拿到的是暴力扫描——
    这个差别不会报错，只会体现在检索质量与耗时上。
    """
    key = str(name).strip().lower()
    if key in _PROFILES_BY_NAME:
        return _PROFILES_BY_NAME[key]
    raise VectorStoreError(
        f"未知后端 {name!r}，可选：{', '.join(backend_names())}。"
        "三者按「依赖递增 / 可控性递减」排列：flat 零依赖且结果逐位可复现，"
        "faiss 只做向量索引，chroma 把向量与元数据放在同一个集合里。"
    )


def missing_requirements(name: str) -> list[str]:
    """这个后端还缺哪些包（按 ``requires`` 的顺序；空列表 = 现在就能用）.

    用 ``importlib.util.find_spec`` 而不是真的 import：``import faiss``
    会把整个库加载进内存（几百 MB 与一次磁盘 IO），而"它装没装"这个问题
    只需要读一遍 ``sys.path``。**探测的代价与使用的代价是两件事。**

    对 ``faiss`` 而言 ``requires = ("faiss", "numpy")`` 不是凑数：faiss-cpu
    的 Python 绑定在导入期就要 numpy，所以"numpy 在、faiss 不在"与
    "两个都不在"给出的安装指引必须是同一句话里的两个事实
    （见 ``_missing_phrase`` 的"（numpy 已满足）"）。
    """
    profile = backend_profile(name)
    return [package for package in profile.requires if not _module_present(package)]


def is_available(name: str) -> bool:
    """这个后端现在能不能用（= ``missing_requirements`` 为空）.

    调用方（``evaluate.compare_backends``、端点）据此把"本机没装 faiss"
    变成一条**报告行**而不是一次异常：同一份报告在任何机器上都长得一样，
    只是某一行的 ``available`` 是 ``False``。
    """
    return not missing_requirements(name)


def describe_backends() -> list[dict[str, Any]]:
    """三个后端的完整说明（端点 ``/vectorstore/backends`` 的响应体）.

    每一项都能被 ``json.dumps`` 直接序列化：``requires`` 与 ``missing``
    用 list 而不是 tuple，布尔值是真的布尔值而不是 ``"true"``。
    **端点形状不该在这一层之外被拼装**——拼装意味着两处对"有哪些字段"
    各有一份说法，而它们会在某次改动后分叉。

    ``available`` 与 ``missing`` 一起给：只给一个布尔值时，使用者看到
    ``false`` 还得自己去猜缺的是哪一个包。
    """
    return [_profile_row(profile) for profile in BACKEND_PROFILES]


def _profile_row(profile: BackendProfile) -> dict[str, Any]:
    """把一份声明渲染成可 JSON 序列化的一行（含当场探测到的可用性）."""
    missing = missing_requirements(profile.name)
    return {
        "name": profile.name,
        "description": profile.description,
        "requires": list(profile.requires),
        "available": not missing,
        "missing": list(missing),
        "install_hint": profile.install_hint,
        "supports_persistence": profile.supports_persistence,
        "native_metadata": profile.native_metadata,
        "default_metric": profile.default_metric,
        "notes": profile.notes,
    }


# --------------------------------------------------------------------------- #
# 构造：resolve_backend / create_backend
# --------------------------------------------------------------------------- #


def resolve_backend(
    name: str | None = None,
    *,
    metric: str | None = None,
    dimension: int | None = None,
    path: str | None = None,
    collection: str | None = None,
    **kwargs: Any,
) -> VectorBackend:
    """按名字建一个后端实例（本模块唯一的"真的动手"的入口）.

    五个具名参数都允许缺省，缺省值取自 ``settings``——**这就是"项目级基线 +
    单次覆盖"里的那两级**：``name=None`` 用 ``vector_backend``，
    ``metric=None`` 用 ``vector_metric``，``path=None`` 用 ``vector_persist_path``
    （空串 → 不落盘），``collection=None`` 用 ``vector_collection``。
    ``dimension`` 没有对应的配置项：维度是**编码器的事实**，不是偏好，
    因此它只能来自调用点（或由后端在第一次写入时自己定下）。

    校验顺序是刻意的（先不变量、后环境）：

    ```text
    1. 后端名    未知名字无条件报错        → 与装没装东西无关
    2. 度量      未知度量无条件报错        → 同上（normalize_metric）
    3. 依赖      requires 有缺 → BackendUnavailable（三段式：缺什么/怎么装/用什么）
    4. 导入      函数内 import 失败 → 同上（少见的"文件在但导不进来"）
    5. 构造      后端自己的错误一概**不吞**（例如 faiss 的"维度必填"）
    ```

    第 1、2 步放在最前的理由：一个拼错的名字或度量，**装上依赖之后依然错**。
    先报环境问题会让使用者装完一轮才发现参数是错的。

    第 5 步是本方法唯一"什么都不做"的地方，而且必须如此：``dimension=None``
    对 faiss 是**必然失败**的（索引建立前就要知道 d），那时该出现的是
    ``faiss_backend`` 那条把两条出路都写清楚的消息，而不是本层再包一遍。

    ``**kwargs`` 按原样透传给后端构造（例如测试注入 ``faiss_module`` /
    ``chromadb_module`` / ``client``）；拼错的键会得到后端自己的
    ``TypeError``——**比在这一层维护一张"合法键"白名单更不容易过期**。
    """
    profile = backend_profile(name if name is not None else app_settings.vector_backend)
    resolved_metric = normalize_metric(metric if metric is not None else app_settings.vector_metric)
    missing = missing_requirements(profile.name)
    if missing:
        raise BackendUnavailable(_unavailable_message(profile, missing))

    params: dict[str, Any] = {
        "metric": resolved_metric,
        "dimension": dimension,
        "path": _resolve_path(path),
    }
    if "collection" in _BACKEND_PARAMS[profile.name]:
        params["collection"] = (
            collection if collection is not None else app_settings.vector_collection
        )
    params.update(kwargs)

    try:
        return BACKEND_FACTORIES[profile.name](**params)
    except ImportError as exc:
        # 走到这里说明"find_spec 说在，import 却失败"——典型原因是模块残缺、
        # 平台不匹配的 wheel，或 sys.modules 里被塞了一个 None（测试会用到）。
        # 这一支必须存在：它是"探测与使用不是一回事"的兜底。
        raise BackendUnavailable(_import_failure_message(profile, exc)) from exc


def create_backend(settings_obj: Any = None, **overrides: Any) -> VectorBackend:
    """从配置建一个后端：``settings`` 是基线，``overrides`` 优先级最高.

    覆盖项的短名与配置字段的对应关系（**写成表，因为它是一份翻译契约**）：

    ```text
    backend      vector_backend         flat / faiss / chroma
    metric       vector_metric          走 normalize_metric 的别名表
    dimension    （无配置项）            维度是编码器的事实，只能由调用点给
    path         vector_persist_path    空串 = 明确不落盘
    collection   vector_collection      只有 chroma 用得到（见 _BACKEND_PARAMS）
    ```

    其余的关键字原样透传给 ``resolve_backend``（供高级用法注入客户端/模块）。

    **``top_k`` 与 ``min_score`` 被明确拒收**，而不是忽略：它们是查询期参数
    （``vector_default_top_k`` / ``vector_min_score``），而本函数返回的是
    **一个后端实例**——后端没有"默认取几条"这个概念。静默忽略会让
    ``create_backend(top_k=3)`` 看起来生效了，而实际取的还是 5；
    正确的位置是 ``vectorstore.pipeline.VectorIngestPipeline``，
    那里才有那条三级优先级链（调用参数 > 构造参数 > settings）。
    """
    source = settings_obj if settings_obj is not None else app_settings

    rejected = [key for key in overrides if key in QUERY_TIME_KEYS]
    if rejected:
        raise VectorStoreError(
            f"create_backend 不接受查询期参数 {sorted(rejected)}："
            "top_k / min_score 是**检索**参数，而本函数返回的是一个后端实例，"
            "后端没有「默认取几条」这个概念。"
            "请改用 vectorstore.pipeline.VectorIngestPipeline"
            "（它的 top_k / min_score 优先级链是：调用参数 > 构造参数 > settings）。"
        )

    accepted = {key: overrides[key] for key in CREATE_OVERRIDE_KEYS if key in overrides}
    passthrough = {
        key: value for key, value in overrides.items() if key not in CREATE_OVERRIDE_KEYS
    }

    return resolve_backend(
        accepted.get("backend", _setting(source, "vector_backend")),
        metric=accepted.get("metric", _setting(source, "vector_metric")),
        dimension=accepted.get("dimension"),
        path=accepted.get("path", _setting(source, "vector_persist_path")),
        collection=accepted.get("collection", _setting(source, "vector_collection")),
        **passthrough,
    )


def _setting(source: Any, field: str) -> Any:
    """从 settings 对象读一个字段；缺字段时给出**可修**的报错.

    不直接 ``getattr`` 是为了让"传了一个自定义配置对象但字段名写错"这件事
    在调用点就报出来，而不是变成 ``AttributeError: 'SimpleNamespace' object
    has no attribute 'vector_backend'`` 这种看不到期望形状的消息。
    """
    if not hasattr(source, field):
        raise VectorStoreError(
            f"settings 对象缺少 {field!r}（收到 {type(source).__name__}）："
            "create_backend 从 settings 里读向量库配置，"
            f"请传一个 config.Settings 实例（它有 {field}），"
            "或任何提供同样字段名的对象。"
        )
    return getattr(source, field)


def _resolve_path(path: str | None) -> str:
    """把 ``path`` 收敛成字符串；``None`` 表示"用 settings 的基线".

    ``""`` 与 ``None`` 在这里是**两个不同的意思**（这是本函数唯一的难点）：
    ``""`` 是调用方明确说"这次不要落盘"，``None`` 是"按项目基线来"。
    两者在配置里的表示恰好都是空串，但语义不同——把它们都当成
    "没给" 会让"我想覆盖掉配置里那个路径、这次只在内存里跑"变得无法表达。
    """
    if path is None:
        return str(app_settings.vector_persist_path or "")
    return str(path)


# --------------------------------------------------------------------------- #
# 三段式错误信息
# --------------------------------------------------------------------------- #


def _missing_phrase(profile: BackendProfile, missing: list[str]) -> str:
    """第一句的宾语：缺什么（并把"已满足"的依赖一并说出来）.

    为什么要把"已满足"写进去：``faiss`` 缺的常常只是 faiss 本身而 numpy 在，
    这时"缺少 faiss、numpy"会让使用者去装一个已经有了的包；
    而只说"缺少 faiss"又丢掉"numpy 这个前提已经成立"这个事实。
    两件事都写下来，使用者才能判断该装几个包。
    """
    satisfied = [package for package in profile.requires if package not in missing]
    if satisfied:
        return f"{'、'.join(missing)}（{'、'.join(satisfied)} 已满足）"
    return "、".join(missing)


def _unavailable_message(profile: BackendProfile, missing: list[str]) -> str:
    """三段式：缺什么 → 怎么装 → 还能用什么（见模块 docstring）."""
    return (
        f"后端 {profile.name} 不可用：缺少 {_missing_phrase(profile, missing)}。\n"
        f"安装：{profile.install_hint}\n"
        f"或者改用：{_available_alternatives(profile.name)}"
    )


def _import_failure_message(profile: BackendProfile, exc: ImportError) -> str:
    """同一套三段式，但第一句说的是"探测说在、导入失败"这种少见情形."""
    return (
        f"后端 {profile.name} 不可用：缺少 {'、'.join(profile.requires) or '（无）'}"
        f"（导入失败：{exc}）。\n"
        f"安装：{profile.install_hint}\n"
        f"或者改用：{_available_alternatives(profile.name)}"
    )


def _available_alternatives(excluded: str) -> str:
    """「还能用什么」那一句的内容：现场可用、且不是它自己的后端.

    过滤掉不可用的后端（否则会给出"改用 chroma"而 chroma 同样没装——
    这种建议比没有建议更浪费时间）。``flat`` 的 ``requires`` 是空元组，
    因此这一句在任何机器上都不会是空的。
    """
    rows = [
        profile
        for profile in BACKEND_PROFILES
        if profile.name != excluded and is_available(profile.name)
    ]
    # 这一支只有"连 flat 都不可用"时才会走到，而 flat 零依赖——
    # 因此它是**防御性**的，不属于任何可达路径。仍然保留一句话而不是
    # 返回空串：三段式的第三句宁可说"没有别的可用后端"，
    # 也不能变成"或者改用："这种把使用者晾在半路的样子。
    if not rows:  # pragma: no cover - 见上面三行的说明：可达性依赖于 flat 的 requires 为空
        return "（本机没有别的可用后端；flat 零依赖，理论上总会可用）"
    return "、".join(f"{profile.name}（{profile.description}）" for profile in rows)


def _module_present(package: str) -> bool:
    """``importlib.util.find_spec`` 的防御性包装（探测失败一律当作"不在"）.

    为什么用属性访问而不是 ``from importlib.util import find_spec``：
    绑定到本地名字之后，monkeypatch 替换 ``importlib.util.find_spec``
    就影响不到本模块了——而"缺依赖 / 依赖齐全"两条分支的测试**必须**
    能替换它，否则覆盖了哪一半完全取决于本机装了什么。
    """
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        # find_spec 对"父包不存在"的 dotted name 会抛 ModuleNotFoundError，
        # 对空串抛 ValueError。对调用方来说这两种都等价于"不可用"。
        return False


__all__ = [
    "BACKEND_CHROMA",
    "BACKEND_FACTORIES",
    "BACKEND_FAISS",
    "BACKEND_FLAT",
    "BACKEND_PROFILES",
    "CREATE_OVERRIDE_KEYS",
    "QUERY_TIME_KEYS",
    "BackendProfile",
    "backend_names",
    "backend_profile",
    "create_backend",
    "describe_backends",
    "is_available",
    "missing_requirements",
    "resolve_backend",
]
