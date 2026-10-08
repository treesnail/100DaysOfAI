"""编排规格：把"怎么部署"写成一份可以被**静态校验**的数据（M6-D10 / day072）.

day022 已经交付过一份 ``docker-compose.yml``（MCP Server 那一套）。本模块
要解决的**不是**"再写一份 yml"，而是这一课真正的难题：

```text
yaml 是数据，不是代码 —— 写错的 compose 不会在提交时失败，只会在部署时表现成
"容器起来了，但库是空的""镜像不可追溯""探针永远绿""卷没声明，重建即丢"
```

因此这一层的形态是 ``spec（数据）→ validate（判据）→ render（渲染）``，
而不是"手写一份 yml 再写脚本检查它"：

```text
ComposeSpec / ServiceSpec    规格：服务、镜像、端口、卷、环境、探针、依赖
validate_spec                11 条静态判据（每一条都对应一类部署事故）
render_compose               确定性的 YAML 文本（不含时间、键按名排序）
check_committed_compose      让"仓库里那份文件"与"代码渲染出来的那份"必须一致
```

## 三条纪律

**1. 会出错的东西必须能被静态查出来。**
"探针没配""镜像用了 ``latest``""卷没在顶层声明"这三类问题都能在提交时
被查出来，因此它们就不该留到部署时去发现。11 条判据都在
:func:`validate_spec` 里，且每条消息都写清"为什么"与"怎么改"。

**2. 渲染必须是确定性的。**
同一份规格渲染两次必须逐字节相同（键按名排序、没有时间戳、没有随机名），
否则 ``check_committed_compose`` 会永远报"文件不一致"，
而那个告警会在三天内被人静音——之后真正的漂移也一起被静音了。

**3. 那份 yml 要进 git，并且由测试守着它。**
规格是代码（能被 review、能 diff），渲染产物是它的**快照**。
``tests/test_rag_ops_deploy.py`` 会逐字节比对二者：手改 yml 会当场变红，
而不是等到下次部署时带着一个没人知道的偏配置上线。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smart_research_agent.config import settings
from smart_research_agent.rag_ops.errors import DeployError
from smart_research_agent.rag_ops.types import utc_now

# --------------------------------------------------------------------------- #
# 三个服务角色
# --------------------------------------------------------------------------- #

#: 对外提供 HTTP 服务的容器（FastAPI 应用）。
SERVICE_API = "rag-api"

#: 向量库服务（默认用官方 Chroma 镜像，见 ``CHROMA_IMAGE``）。
SERVICE_STORE = "vector-store"

#: 定时同步的批处理容器（跑 ``python -m smart_research_agent.rag_ops.worker``）。
SERVICE_WORKER = "rag-ops-sync"

#: 三个角色（顺序 = 渲染顺序 = 启动依赖顺序）。
SERVICE_ROLES: tuple[str, ...] = (SERVICE_STORE, SERVICE_API, SERVICE_WORKER)

#: 每个角色的一句话说明（进报告与端点，少一个键 = 某个容器是干什么的没人知道）。
SERVICE_ROLE_DESCRIPTIONS: dict[str, str] = {
    SERVICE_STORE: "向量库服务：保存向量与元数据的**状态**，必须挂持久卷",
    SERVICE_API: "应用服务：提供 /rag/* 与 /rag/ops/* 接口，探针指向 /rag/ops/health",
    SERVICE_WORKER: "定时同步容器：按调度策略扫语料、增量重建索引、落盘账本",
}

#: 向量库镜像（**官方镜像 + 固定 tag**）。
#: 固定 tag 是这一课从 Chroma 官方文档搬来的一条纪律：
#: "Avoid relying on latest for production or repeatable environments."
#: （见 docs/rag_ops.md 的"镜像与探针"一节）
CHROMA_IMAGE = "chromadb/chroma"

#: 向量库镜像的 tag。**2026-09 的稳定版**，也是官方 compose 示例里用的那一个。
#: 升级它必须与 ``docs/rag_ops.md`` 里那条"先验证再升"的说明一起改。
CHROMA_TAG = "1.5.3"

#: 向量库的心跳路径（官方 API：``GET /api/v2/heartbeat``，返回一个纳秒时间戳）。
CHROMA_HEARTBEAT_PATH = "/api/v2/heartbeat"

#: 应用镜像的仓库名（tag 由配置给，缺省 ``v0.1.0``）。
IMAGE_REPOSITORY = "smart-research-rag-api"

#: 应用镜像的 Dockerfile 文件名。**不叫 ``Dockerfile``**：本仓库根目录
#: 已经有一份 ``Dockerfile``（day022 的 MCP Server 镜像），两者共用一个名字
#: 的后果是"``docker build .`` 构建出哪一套"取决于谁最后改的文件。
API_DOCKERFILE = "Dockerfile.api"

#: 应用容器内监听端口。**取 8080 而不是 8000**：本仓库既有的
#: ``docker-compose.yml``（MCP Server）已经占了 8000，两者同时跑时
#: 冲突的表现是"后启动的那个起不来"，而根因在另一个文件里。
API_CONTAINER_PORT = 8080

#: 应用容器里的四个路径（**容器内的绝对路径**，与宿主机的相对路径不是一回事）.
#: 把它们写成常量而不是散在各处，是因为"卷挂载点"与"环境变量里的路径"
#: 必须逐字一致——不一致的表现是"应用往容器自己的可写层写账本"，
#: 而容器一重建账本就不见了，且没有任何报错。
CONTAINER_APP_DIR = "/app"
CONTAINER_CORPUS_DIR = "/app/data/knowledge"
CONTAINER_INDEX_DIR = "/app/data/index"
CONTAINER_OPS_DIR = "/app/data/ops"

#: 应用容器的健康探针路径（``/rag/ops/health``：它答的是"这个实例还能不能
#: 服务知识库问题"，而不是 ``/health`` 的"进程还活着"）。
API_HEALTH_PATH = "/rag/ops/health"

#: 应用容器的启动命令。**用 uvicorn 的 factory 模式，而不是
#: ``python -m smart_research_agent.api.app``**：那个入口的 ``main()``
#: 绑的是 ``127.0.0.1:8000``（本地开发的正确默认值），
#: 而容器里必须绑 ``0.0.0.0``，否则宿主机端口转发一个包都进不来——
#: 表现是"容器 healthy、端口开着、请求超时"，而日志里什么都没有。
API_COMMAND: tuple[str, ...] = (
    "python",
    "-m",
    "uvicorn",
    "smart_research_agent.api.app:create_app",
    "--factory",
    "--host",
    "0.0.0.0",
    "--port",
    str(API_CONTAINER_PORT),
)

#: 探针的三种写法（Docker Compose 的 ``test`` 首元素）。
PROBE_TEST_FORMS: tuple[str, ...] = ("CMD", "CMD-SHELL", "NONE")

#: 环境变量名的合法形状（十二要素：配置存环境，名字一律大写）。
ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")

#: 镜像 tag 里不允许出现的值（不可追溯：今天拉的 ``latest`` 与明天拉的
#: ``latest`` 是两个镜像，而它们看起来是同一行配置）。
FORBIDDEN_IMAGE_TAGS: tuple[str, ...] = ("latest", "")

#: 容器里**不许**被卷覆盖的路径前缀：它们是镜像的一部分（系统目录与代码目录）。
#: 这条清单刻意不写成"路径深度必须 >= 2"——那种规则会把官方 Chroma 镜像
#: 推荐的 ``/data`` 一起拦掉（见 ``docs/rag_ops.md`` 的"卷与挂载点"一节）。
RESERVED_MOUNT_PREFIXES: tuple[str, ...] = (
    "/bin",
    "/boot",
    "/dev",
    "/etc",
    "/lib",
    "/lib64",
    "/proc",
    "/sbin",
    "/sys",
    "/usr",
    "/var/run",
    "/app/smart_research_agent",
)


# --------------------------------------------------------------------------- #
# 规格的形状
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PortMapping:
    """一条端口映射：``"8080:8080"`` 的宿主端口与容器端口."""

    host: int
    container: int

    def __post_init__(self) -> None:
        for label, value in (("host", self.host), ("container", self.container)):
            if not 1 <= int(value) <= 65535:
                raise DeployError(
                    f"端口映射的 {label} 端口必须在 1~65535 之间，收到 {value}："
                    "宿主机端口填 0 表示'随机分配'，那会让外部再也找不到这个服务。"
                )

    def to_dict(self) -> dict[str, int]:
        """可 json.dumps 的形状."""
        return {"host": self.host, "container": self.container}

    def render(self) -> str:
        """渲染成 compose 的那一行字符串."""
        return f"{self.host}:{self.container}"


@dataclass(frozen=True)
class VolumeMount:
    """一条卷挂载：**顶层声明的卷名** → 容器内绝对路径.

    ``source`` 刻意只接受"卷名"而不是"宿主路径"：绑定挂载（``./data:/app/data``）
    会把"这个容器写在哪"变成一台机器上的一个目录，而命名卷由编排层管理，
    重建容器不会丢数据——这一课要的正是后者。
    """

    source: str
    target: str
    read_only: bool = False

    def __post_init__(self) -> None:
        if not self.source:
            raise DeployError("卷名不能为空。")
        if "/" in self.source or "\\" in self.source:
            raise DeployError(
                f"卷挂载的来源必须是**顶层声明的卷名**（收到 {self.source!r}）："
                "绑定挂载会把'数据在哪'绑到一台机器的一个目录上，"
                "而命名卷才能在容器重建后保住索引。"
            )
        if self.target == "/":
            raise DeployError(
                "容器内路径不能是根目录 '/'：整个文件系统会被一个空卷覆盖，"
                "镜像里的代码与依赖全部消失。"
            )
        if not self.target.startswith("/"):
            raise DeployError(
                f"容器内路径必须是绝对路径（收到 {self.target!r}）："
                "相对路径会落在镜像的 WORKDIR 上，而那个位置由基础镜像决定。"
            )
        if self.target.endswith("/"):
            raise DeployError(f"容器内路径不能以 '/' 结尾（收到 {self.target!r}）。")

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {"source": self.source, "target": self.target, "read_only": self.read_only}

    def render(self) -> str:
        """渲染成 compose 的那一行字符串（只读时带 ``:ro``）."""
        suffix = ":ro" if self.read_only else ""
        return f"{self.source}:{self.target}{suffix}"


@dataclass(frozen=True)
class HealthProbe:
    """一个健康探针：命令 + 四个时间参数（单位一律秒，渲染时补 ``s``）.

    ``start_period_seconds`` 是这里最容易被忽略、又最影响可用性的一个：
    应用启动要先把索引加载进内存（本课程是几十毫秒，真实语料是几十秒），
    没有启动宽限期的探针会在启动期间连续失败，把容器判成不健康。
    """

    test: tuple[str, ...]
    interval_seconds: int = 30
    timeout_seconds: int = 5
    retries: int = 3
    start_period_seconds: int = 15

    def __post_init__(self) -> None:
        if not self.test:
            raise DeployError("探针命令不能为空：没有命令的探针永远返回 0（也就是永远健康）。")
        if self.test[0] not in PROBE_TEST_FORMS:
            raise DeployError(
                f"探针命令的第一段必须是 {list(PROBE_TEST_FORMS)} 之一，"
                f"收到 {self.test[0]!r}：Compose 用它区分'直接执行'与'交给 shell'。"
            )
        if self.test[0] == "NONE":
            raise DeployError(
                "探针写成 NONE 等于声明'我没有探针'：那种情况请用 ServiceSpec.probe_exempt "
                "写明理由，而不是让一个永远成功的探针冒充健康检查。"
            )
        for label, value in (
            ("interval_seconds", self.interval_seconds),
            ("timeout_seconds", self.timeout_seconds),
            ("start_period_seconds", self.start_period_seconds),
        ):
            if int(value) <= 0:
                raise DeployError(f"探针的 {label} 必须为正数，收到 {value}。")
        if int(self.retries) < 1:
            raise DeployError(f"探针的 retries 必须 >= 1，收到 {self.retries}。")
        if int(self.timeout_seconds) >= int(self.interval_seconds):
            raise DeployError(
                f"探针超时 {self.timeout_seconds}s 不小于间隔 {self.interval_seconds}s："
                "超时比间隔还长时，探针永远在等上一次的结果，实际上退化成随机判定。"
            )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（含那条可被复制粘贴的探针命令）."""
        return {
            "test": list(self.test),
            "test_line": " ".join(self.test),
            "interval_seconds": self.interval_seconds,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
            "start_period_seconds": self.start_period_seconds,
        }

    def render(self) -> list[str]:
        """渲染成 compose 里 ``healthcheck`` 那几行（缩进由调用方补）."""
        return [
            f"test: {_render_list(self.test)}",
            f"interval: {self.interval_seconds}s",
            f"timeout: {self.timeout_seconds}s",
            f"retries: {self.retries}",
            f"start_period: {self.start_period_seconds}s",
        ]


@dataclass(frozen=True)
class ServiceSpec:
    """一个服务的规格（见模块 docstring 的三条纪律）.

    ``probe_exempt`` 是这份规格里最"这一课"的字段：

    ```text
    对外服务的容器必须有探针        → 没有探针的容器，编排层看不见它坏
    批处理容器没有端口可探          → 它的存活信号是账本与监控指标，不是 HTTP
    因此必须"要么有探针、要么写明豁免理由" —— 缺失必须是有意的，并且写下来
    ```
    """

    name: str
    image: str
    command: tuple[str, ...] = ()
    ports: tuple[PortMapping, ...] = ()
    volumes: tuple[VolumeMount, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()
    health_probe: HealthProbe | None = None
    probe_exempt: str = ""
    depends_on: tuple[str, ...] = ()
    restart: str = "unless-stopped"
    build_dockerfile: str = ""
    build_context: str = "."

    def __post_init__(self) -> None:
        if self.name not in SERVICE_ROLE_DESCRIPTIONS:
            raise DeployError(
                f"不认识的服务名 {self.name!r}：可用角色 {list(SERVICE_ROLES)}。"
                "自由命名会让'哪些容器属于这一套编排'变成一个需要读完整份 yml 的问题。"
            )
        if not self.image:
            raise DeployError(f"服务 {self.name!r} 必须指定镜像。")
        if self.build_dockerfile and not self.build_context.strip():
            raise DeployError(
                f"服务 {self.name!r} 声明了 build_dockerfile 但 build_context 是空的："
                "缺上下文时 compose 会去猜，而猜出来的目录里可能有一份同名的 Dockerfile。"
            )
        object.__setattr__(self, "command", tuple(str(item) for item in self.command))
        object.__setattr__(
            self, "depends_on", tuple(sorted({str(item) for item in self.depends_on}))
        )
        env: dict[str, str] = {}
        for key, value in self.environment:
            if key in env:
                raise DeployError(
                    f"服务 {self.name!r} 的环境变量 {key!r} 出现了两次："
                    "后一个会静默覆盖前一个，而读的人只会看到其中一行。"
                )
            env[str(key)] = str(value)
        object.__setattr__(self, "environment", tuple(sorted(env.items())))

    # ------------------------------------------------------------------ 查询

    def env(self) -> dict[str, str]:
        """环境变量查表（判定"这个服务用的是哪个后端"时用它）."""
        return dict(self.environment)

    def volume_names(self) -> tuple[str, ...]:
        """这个服务挂载的卷名（升序）."""
        return tuple(sorted({mount.source for mount in self.volumes}))

    def port_pairs(self) -> tuple[tuple[int, int], ...]:
        """端口映射（宿主，容器）的元组."""
        return tuple((item.host, item.container) for item in self.ports)

    def image_tag(self) -> str:
        """镜像的 tag（没有 tag 时返回空串）.

        只取"最后一个 ``:`` 之后且不含 ``/``"的那一段：带端口的仓库名
        （``localhost:5000/app``）里那个 ``5000`` 是**端口不是 tag**，
        把它当成 tag 会让"没打 tag"这件事逃过 R4。
        """
        _head, sep, tail = self.image.rpartition(":")
        if not sep or "/" in tail:
            return ""
        return tail

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（含角色说明与探针）."""
        return {
            "name": self.name,
            "description": SERVICE_ROLE_DESCRIPTIONS[self.name],
            "image": self.image,
            "image_tag": self.image_tag(),
            "command": list(self.command),
            "ports": [item.to_dict() for item in self.ports],
            "volumes": [item.to_dict() for item in self.volumes],
            "environment": dict(self.environment),
            "health_probe": self.health_probe.to_dict() if self.health_probe else None,
            "probe_exempt": self.probe_exempt,
            "depends_on": list(self.depends_on),
            "restart": self.restart,
            "build_dockerfile": self.build_dockerfile,
            "build_context": self.build_context if self.build_dockerfile else "",
        }


@dataclass(frozen=True)
class ComposeSpec:
    """一整套编排规格：项目名 + 三个服务 + 顶层卷 + 网络.

    构造期只做"内部一致性"的检查（名字唯一），跨服务的判据全部在
    :func:`validate_spec` 里——因为那些判据要给出**可读的编号清单**，
    而不是在构造某一个服务时抛一句局部错误。
    """

    project_name: str
    services: tuple[ServiceSpec, ...] = ()
    volumes: tuple[str, ...] = ()
    network: str = "rag-net"
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.project_name:
            raise DeployError("项目名不能为空：它决定容器名前缀与网络名。")
        services = tuple(sorted(self.services, key=lambda item: item.name))
        seen: list[str] = []
        for service in services:
            if service.name in seen:
                raise DeployError(f"服务 {service.name!r} 在规格里出现了两次。")
            seen.append(service.name)
        volumes = tuple(sorted({str(item) for item in self.volumes}))
        if any(not item for item in volumes):
            raise DeployError("顶层卷名不能为空。")
        object.__setattr__(self, "services", services)
        object.__setattr__(self, "volumes", volumes)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def service(self, name: str) -> ServiceSpec | None:
        """按名字取一个服务；没有时返回 ``None``."""
        for item in self.services:
            if item.name == name:
                return item
        return None

    def service_names(self) -> tuple[str, ...]:
        """全部服务名（升序）."""
        return tuple(item.name for item in self.services)

    def to_dict(self, *, include_services: bool = True) -> dict[str, Any]:
        """可 json.dumps 的形状（``include_services=False`` 时只报概览）."""
        payload: dict[str, Any] = {
            "project_name": self.project_name,
            "network": self.network,
            "volumes": list(self.volumes),
            "services": list(self.service_names()),
            "metadata": dict(self.metadata),
        }
        if include_services:
            payload["detail"] = [item.to_dict() for item in self.services]
        return payload


# --------------------------------------------------------------------------- #
# 渲染
# --------------------------------------------------------------------------- #


def _render_list(items: tuple[str, ...]) -> str:
    """把一串值渲染成 compose 认的行内列表：``["CMD", "curl", "-f", "..."]``."""
    body = ", ".join(f'"{item}"' for item in items)
    return f"[{body}]"


def _render_scalar(value: str) -> str:
    """渲染一个 YAML 标量：**只有必要时才加引号，且只加双引号**.

    "必要时"的判据是"这个值不加引号会被 YAML 读成别的东西"：

    ```text
    含 ':' 或 '#'        裸写会被读成分隔符或注释（镜像名一定命中这一条）
    首尾有空格           裸写会被静默去掉（那正是最坏的一种"看起来对"）
    以 YAML 指示符开头    - ? : , [ ] { } & * ! | > % @ ` 与引号本身
    ```

    两类值**无法**被本渲染器表达，因此当场拒绝（而不是想办法转义）：
    含双引号的值、含换行或制表符的值。它们要出现在配置里，
    应该改的是那份配置，而不是让渲染器长出一套转义规则
    （转义规则是"两边都要对"的东西，而它只会被其中一边测试）。
    """
    if not value:
        raise DeployError("空串不是一个可渲染的配置值（空值在多数镜像里等价于'没设置'）。")
    if any(ch in value for ch in ('"', "\n", "\t", "\r")):
        raise DeployError(
            f"配置值 {value!r} 含无法表达的字符（双引号或换行）："
            "本渲染器只输出双引号字符串，不做转义——请改配置，而不是改渲染器。"
        )
    if value != value.strip():
        raise DeployError(
            f"配置值 {value!r} 有首尾空格：YAML 裸标量会静默去掉它们，"
            "于是配置里写的与实际生效的不是同一个值。"
        )
    indicator = "-?:,[]{}&*!|>%@`'"
    if any(token in value for token in (":", "#")) or value[0] in indicator:
        return f'"{value}"'
    return value


def render_compose(spec: ComposeSpec) -> str:
    """把规格渲染成一份确定性的 compose 文件（**同一份规格永远同一份文本**）.

    渲染顺序固定：文件头注释 → ``name`` → ``services``（按服务名字典序）
    → 每个服务的固定字段顺序 → ``volumes``（按名排序）→ ``networks``。
    没有时间戳、没有随机名、没有"按插入顺序"，因此它可以直接进 git 被 diff。
    """
    header = [
        "# 智研 AI 助手 —— RAG 生产化编排（day072 / M6-D10）",
        "#",
        "# 本文件由 smart_research_agent.rag_ops.deploy.render_compose() 渲染：",
        "#   请勿手改。tests/test_rag_ops_deploy.py 会逐字节比对它与默认规格，",
        "#   手改会当场变红（而不是等下一次部署带着一份没人知道的偏配置上线）。",
        "# 重新渲染：python scripts/rag_ops_demo.py --write-compose",
        "#",
        "# 三个容器：vector-store（向量库）/ rag-api（应用）/ rag-ops-sync（定时同步）。",
        f"# 变更记录：{spec.metadata.get('note', '（无）')}",
    ]
    lines: list[str] = list(header)
    lines.append(f"name: {_render_scalar(spec.project_name)}")
    lines.append("")
    lines.append("services:")
    for service in spec.services:
        lines.append(f"  {service.name}:")
        lines.append(f"    image: {_render_scalar(service.image)}")
        if service.build_dockerfile:
            lines.append("    build:")
            lines.append(f"      context: {_render_scalar(service.build_context)}")
            lines.append(f"      dockerfile: {_render_scalar(service.build_dockerfile)}")
        if service.command:
            lines.append(f"    command: {_render_list(service.command)}")
        lines.append(f"    restart: {_render_scalar(service.restart)}")
        if service.ports:
            lines.append("    ports:")
            for port in service.ports:
                lines.append(f'      - "{port.render()}"')
        if service.environment:
            lines.append("    environment:")
            for key, value in service.environment:
                lines.append(f"      {key}: {_render_scalar(value)}")
        if service.volumes:
            lines.append("    volumes:")
            for mount in service.volumes:
                lines.append(f'      - "{mount.render()}"')
        if service.health_probe is not None:
            lines.append("    healthcheck:")
            for row in service.health_probe.render():
                lines.append(f"      {row}")
        if service.depends_on:
            lines.append("    depends_on:")
            for name in service.depends_on:
                lines.append(f"      - {name}")
    lines.append("")
    lines.append("volumes:")
    for name in spec.volumes:
        lines.append(f"  {_render_scalar(name)}:")
    lines.append("")
    lines.append("networks:")
    lines.append(f"  {_render_scalar(spec.network)}:")
    lines.append("    driver: bridge")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 静态判据
# --------------------------------------------------------------------------- #


def _covers_reserved_path(target: str) -> bool:
    """挂载点是否落在"镜像自带"的路径上（等于该路径，或它的子路径）."""
    for prefix in RESERVED_MOUNT_PREFIXES:
        if target == prefix or target.startswith(f"{prefix}/"):
            return True
    return False


def validate_spec(spec: ComposeSpec) -> tuple[str, ...]:
    """11 条静态判据，返回**问题清单**（空元组表示通过）.

    每条判据都写成"现象 + 后果 + 怎么改"，因为这份清单会同时出现在
    测试失败信息、CI 日志与 ``/rag/ops/status`` 的响应里。
    """
    problems: list[str] = []
    names = spec.service_names()

    # R1 服务角色齐全（缺了哪一个，整套编排都跑不起来）
    for role in SERVICE_ROLES:
        if role not in names:
            problems.append(
                f"R1 缺少服务 {role!r}（{SERVICE_ROLE_DESCRIPTIONS[role]}）："
                "三个角色缺一不可，缺了它这一套编排没有意义。"
            )

    # R2 探针：要么有、要么写明豁免（不能都没有，也不能同时给）
    for service in spec.services:
        has_probe = service.health_probe is not None
        exempted = bool(service.probe_exempt.strip())
        if not has_probe and not exempted:
            problems.append(
                f"R2 服务 {service.name!r} 既没有健康探针，也没有写明豁免理由："
                "没有探针的容器在编排层看来永远'活着'，它坏了没人知道。"
            )
        if has_probe and exempted:
            problems.append(
                f"R2 服务 {service.name!r} 同时给了探针与豁免理由："
                "两者只能有一个——同时存在时，读的人不知道该信哪个。"
            )

    # R3 至少有一个持久卷，且挂载的卷必须在顶层声明
    mounted = {name for service in spec.services for name in service.volume_names()}
    if not mounted:
        problems.append(
            "R3 没有任何服务挂载持久卷：向量库与索引是**状态**，"
            "容器重建会把它们一起丢掉（而重新灌库要花一次完整编码的时间）。"
        )
    for name in sorted(mounted - set(spec.volumes)):
        problems.append(
            f"R3 卷 {name!r} 被挂载但没有在顶层 volumes 里声明："
            "Compose 会为它创建一个匿名卷，容器重建后那份数据再也找不回来。"
        )
    for name in sorted(set(spec.volumes) - mounted):
        problems.append(
            f"R3 顶层声明的卷 {name!r} 没有任何服务挂载它："
            "声明了不用的卷通常是改配置时漏删的，它会一直占着磁盘而没人知道里面是什么。"
        )

    # R4 镜像必须带显式 tag，且不得是 latest
    for service in spec.services:
        tag = service.image_tag()
        if tag in FORBIDDEN_IMAGE_TAGS:
            problems.append(
                f"R4 服务 {service.name!r} 的镜像 {service.image!r} 没有可追溯的 tag："
                "'latest'（或没有 tag）意味着今天与明天拉到的是两个镜像，"
                "而配置看起来一模一样。请固定到具体版本。"
            )

    # R5 宿主端口不得重复、不得为 0
    seen_ports: dict[int, str] = {}
    for service in spec.services:
        for host, _container in service.port_pairs():
            if host in seen_ports:
                problems.append(
                    f"R5 宿主端口 {host} 被 {seen_ports[host]!r} 与 {service.name!r} 同时占用："
                    "后启动的那个容器会起不来，而报错出现在另一个服务的日志里。"
                )
            seen_ports[host] = service.name

    # R6 环境变量：名字大写（十二要素），值可被安全渲染
    for service in spec.services:
        for key, value in service.environment:
            if not ENVIRONMENT_NAME_PATTERN.match(key):
                problems.append(
                    f"R6 服务 {service.name!r} 的环境变量名 {key!r} 不是全大写："
                    "配置进环境时一律大写，小写名在 shell 与 CI 里会被当成另一件事。"
                )
            if not value:
                problems.append(
                    f"R6 服务 {service.name!r} 的环境变量 {key!r} 是空值："
                    "空值在多数镜像里等价于'没设置'，那与'显式设为空'是两回事。"
                )
            try:
                _render_scalar(value)
            except DeployError as exc:
                problems.append(f"R6 服务 {service.name!r} 的环境变量 {key!r} 无法渲染：{exc}")

    # R7 挂载点不能覆盖镜像自带的目录
    # （挂到根目录 "/" 在 VolumeMount 的**构造期**就被拒了，因此这里不重复判它：
    #  同一个规则有两处实现时，两处迟早会对同一份输入给出不同的结论。）
    for service in spec.services:
        for mount in service.volumes:
            if _covers_reserved_path(mount.target):
                problems.append(
                    f"R7 服务 {service.name!r} 把卷挂到了 {mount.target!r}："
                    "它落在镜像自带的路径上（系统目录或应用代码目录），"
                    "挂载会遮住镜像里的内容。数据目录请用 /data 或 /app/data/... 这类路径。"
                )

    # R8 用 chroma 后端时必须等向量库服务起来
    api = spec.service(SERVICE_API)
    if api is not None:
        backend = api.env().get("VECTOR_BACKEND", "")
        if backend == "chroma" and SERVICE_STORE not in api.depends_on:
            problems.append(
                f"R8 服务 {SERVICE_API!r} 声明了 VECTOR_BACKEND=chroma，"
                f"但没有 depends_on: [{SERVICE_STORE!r}]："
                "应用会先于向量库起来，然后连着失败一分钟（直到向量库就绪）——"
                "而那段时间的报错看起来像应用自身的问题。"
            )

    # R9 depends_on 必须指向真实存在的服务，且不能指向自己
    for service in spec.services:
        for target in service.depends_on:
            if target == service.name:
                problems.append(f"R9 服务 {service.name!r} 依赖自己。")
            elif target not in names:
                problems.append(
                    f"R9 服务 {service.name!r} 依赖了不存在的服务 {target!r}："
                    "Compose 会拒绝这份文件，而错误信息里只有 target 的名字。"
                )

    # R10 有启动命令的服务，命令不能是空串
    for service in spec.services:
        if service.command and not any(item.strip() for item in service.command):
            problems.append(f"R10 服务 {service.name!r} 的命令是空串：请给出可执行的命令。")

    # R11 顶层卷名不得重复（去重发生在构造期，这里查"声明了但没写清用途"的卷）
    for name in spec.volumes:
        if name != name.strip():
            problems.append(f"R11 顶层卷名 {name!r} 有首尾空格。")

    return tuple(problems)


def require_valid(spec: ComposeSpec) -> None:
    """校验不过就抛 :class:`DeployError`，消息里带**全部**问题（不是第一条）.

    只报第一条的校验器会让人改一轮跑一轮；而这份规格的判据之间是独立的
    （探针与镜像没关系），因此一次性给出全部问题既可行也更快。
    """
    problems = validate_spec(spec)
    if problems:
        body = "\n".join(f"  {index}. {item}" for index, item in enumerate(problems, start=1))
        raise DeployError(
            f"编排规格有 {len(problems)} 个问题，拒绝渲染与部署：\n{body}"
        )


# --------------------------------------------------------------------------- #
# 默认规格与落盘
# --------------------------------------------------------------------------- #


def default_spec(
    *,
    image_tag: str | None = None,
    api_port: int | None = None,
    store_port: int | None = None,
    corpus_dir: str | None = None,
    backend: str | None = None,
    note: str = "day072 首次交付：三容器编排（向量库 + 应用 + 定时同步）",
) -> ComposeSpec:
    """按 ``settings`` 装配默认规格（端点、演示脚本与测试三方共用的那一份）.

    装配清单：

    ```text
    vector-store   chromadb/chroma:1.5.3    卷 chroma-data:/data       探针 /api/v2/heartbeat
    rag-api        本项目镜像:<tag>          卷 rag-index:/app/data/index（+ 语料只读卷 + 账本卷）
                                            探针 /rag/ops/health（应用自己的探针）
    rag-ops-sync   本项目镜像:<tag>          同一个索引卷与账本卷（它就是要写它们）
                                            无 HTTP 端口 → 写明探针豁免理由
    ```

    ``corpus_dir`` 指的是**容器内**的语料路径（缺省 ``/app/data/knowledge``），
    不是宿主机上那个 ``data/knowledge``：compose 文件描述的是容器里的样子，
    而"宿主机的哪份文件进了哪个卷"是编排层的事（靠卷声明表达，不靠路径表达）。

    账本刻意挂在**独立的一个卷**（``rag-ops-state``）上，而不是顺手写进索引卷：
    两者都能"重建容器后保住"，但它们的生命周期不同——索引可以被整库重建，
    而账本一旦丢了，下一次同步就会退化成一次全量（幂等，但昂贵）。
    """
    tag = settings.rag_ops_image_tag if image_tag is None else str(image_tag)
    host_api = int(settings.rag_ops_api_port if api_port is None else api_port)
    host_store = int(settings.rag_ops_store_port if store_port is None else store_port)
    corpus = str(CONTAINER_CORPUS_DIR if corpus_dir is None else corpus_dir)
    resolved_backend = str(backend if backend is not None else settings.vector_backend)
    ledger_in_container = f"{CONTAINER_OPS_DIR}/sync_ledger.json"

    store = ServiceSpec(
        name=SERVICE_STORE,
        image=f"{CHROMA_IMAGE}:{CHROMA_TAG}",
        ports=(PortMapping(host=host_store, container=8000),),
        volumes=(VolumeMount(source="chroma-data", target="/data"),),
        environment=(
            ("ANONYMIZED_TELEMETRY", "FALSE"),
            ("IS_PERSISTENT", "TRUE"),
        ),
        health_probe=HealthProbe(
            test=("CMD", "curl", "-f", f"http://localhost:8000{CHROMA_HEARTBEAT_PATH}"),
            interval_seconds=30,
            timeout_seconds=5,
            retries=3,
            start_period_seconds=10,
        ),
        restart="unless-stopped",
    )
    api = ServiceSpec(
        name=SERVICE_API,
        image=f"{IMAGE_REPOSITORY}:{tag}",
        command=API_COMMAND,
        ports=(PortMapping(host=host_api, container=API_CONTAINER_PORT),),
        volumes=(
            VolumeMount(source="rag-index", target=CONTAINER_INDEX_DIR),
            VolumeMount(source="rag-ops-state", target=CONTAINER_OPS_DIR),
            VolumeMount(source="rag-corpus", target=corpus, read_only=True),
        ),
        environment=(
            ("LOG_LEVEL", "INFO"),
            ("RAG_OPS_LEDGER_PATH", ledger_in_container),
            ("RAG_OPS_SOURCE_DIR", corpus),
            ("VECTOR_BACKEND", resolved_backend),
            ("VECTOR_PERSIST_PATH", f"{CONTAINER_INDEX_DIR}/vectors.json"),
        ),
        health_probe=HealthProbe(
            test=("CMD", "curl", "-f", f"http://localhost:{API_CONTAINER_PORT}{API_HEALTH_PATH}"),
            interval_seconds=30,
            timeout_seconds=5,
            retries=3,
            start_period_seconds=20,
        ),
        depends_on=(SERVICE_STORE,) if resolved_backend == "chroma" else (),
        restart="unless-stopped",
        build_dockerfile=API_DOCKERFILE,
    )
    worker = ServiceSpec(
        name=SERVICE_WORKER,
        image=f"{IMAGE_REPOSITORY}:{tag}",
        command=(
            "python",
            "-m",
            "smart_research_agent.rag_ops.worker",
            "--loop",
            "--interval",
            str(settings.rag_ops_interval_minutes),
        ),
        volumes=(
            VolumeMount(source="rag-index", target=CONTAINER_INDEX_DIR),
            VolumeMount(source="rag-ops-state", target=CONTAINER_OPS_DIR),
            VolumeMount(source="rag-corpus", target=corpus),
        ),
        environment=(
            ("RAG_OPS_BACKOFF_BASE_SECONDS", str(settings.rag_ops_backoff_base_seconds)),
            ("RAG_OPS_LEDGER_PATH", ledger_in_container),
            ("RAG_OPS_SOURCE_DIR", corpus),
            ("VECTOR_BACKEND", resolved_backend),
            ("VECTOR_PERSIST_PATH", f"{CONTAINER_INDEX_DIR}/vectors.json"),
        ),
        probe_exempt=(
            "批处理容器：它不监听端口，因此没有可探活的 HTTP 面。"
            "存活信号由账本（/app/data/ops/sync_ledger.json）与监控指标 "
            "run_failures / sync_age_minutes 给出——见 docs/rag_ops.md 的'探针与豁免'一节"
        ),
        depends_on=(SERVICE_STORE,) if resolved_backend == "chroma" else (),
        restart="unless-stopped",
        build_dockerfile=API_DOCKERFILE,
    )
    return ComposeSpec(
        project_name="smart-research-rag",
        services=(store, api, worker),
        volumes=("chroma-data", "rag-corpus", "rag-index", "rag-ops-state"),
        network="rag-net",
        metadata={"note": note, "image_tag": tag, "rendered_by": "rag_ops.deploy"},
    )


def write_compose(spec: ComposeSpec, path: str | Path) -> Path:
    """校验后落盘（**先校验再写**：不合法就不该在磁盘上留下半成品）."""
    require_valid(spec)
    target = Path(path)
    if not str(target):
        raise DeployError("compose 落盘路径不能是空串。")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_compose(spec), encoding="utf-8")
    return target


def check_committed_compose(spec: ComposeSpec, path: str | Path) -> str:
    """比对"仓库里那份文件"与"这份规格渲染出来的文本"，一致时返回空串.

    不一致时返回**第一处不同的行号与两行内容**：只报"不一致"的消息
    会让人去逐行 diff 一份上百行的 yml，而那条消息本身就该指出现场。
    """
    target = Path(path)
    if not target.exists():
        return (
            f"compose 文件 {str(target)!r} 不存在："
            "先跑 python scripts/rag_ops_demo.py --write-compose"
        )
    expected = render_compose(spec)
    actual = target.read_text(encoding="utf-8")
    if actual == expected:
        return ""
    expected_lines = expected.splitlines()
    actual_lines = actual.splitlines()
    for index in range(max(len(expected_lines), len(actual_lines))):
        left = expected_lines[index] if index < len(expected_lines) else "（文件到此结束）"
        right = actual_lines[index] if index < len(actual_lines) else "（文件到此结束）"
        if left != right:
            return (
                f"{str(target)!r} 第 {index + 1} 行与规格渲染结果不一致：\n"
                f"  仓库里：{right}\n"
                f"  规格算：{left}\n"
                "（手改 compose 不是错误，但必须同时改 specs：见 docs/rag_ops.md）"
            )
    return f"{str(target)!r} 与规格渲染结果不一致（内容相同但行尾或编码不同）。"


def describe(spec: ComposeSpec) -> dict[str, Any]:
    """给 ``/rag/ops/status`` 用的一段摘要（规格 + 判据结论 + 与仓库文件的一致性）."""
    problems = validate_spec(spec)
    path = Path(settings.rag_ops_compose_path)
    return {
        "spec": spec.to_dict(),
        "summary_lines": compose_summary_lines(spec),
        "valid": not problems,
        "problems": list(problems),
        "rules": len(VALIDATION_RULES),
        "compose_path": str(path),
        "compose_committed": path.exists(),
        "compose_drift": check_committed_compose(spec, path) if path.exists() else "",
        "checked_at": utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def compose_summary_lines(spec: ComposeSpec) -> list[str]:
    """三个服务各一行摘要（演示脚本与端点共用，避免两处各拼一遍）."""
    lines: list[str] = []
    for service in spec.services:
        ports = ",".join(port.render() for port in service.ports) or "无端口"
        volumes = ",".join(f"{mount.source}→{mount.target}" for mount in service.volumes) or "无卷"
        probe = (
            "探针 " + " ".join(service.health_probe.test)
            if service.health_probe is not None
            else "无探针（已写明豁免）"
        )
        lines.append(
            f"{service.name}: {service.image} | {ports} | {volumes} | {probe} | {service.restart}"
        )
    return lines


#: 判据清单（``describe`` 报它的长度：**判据本身也要能被数出来**）。
VALIDATION_RULES: tuple[str, ...] = (
    "R1 三个服务角色齐全",
    "R2 探针与豁免理由二选一",
    "R3 持久卷被挂载且在顶层声明",
    "R4 镜像 tag 可追溯（不得 latest）",
    "R5 宿主端口不重复",
    "R6 环境变量名全大写、值可安全渲染",
    "R7 卷挂载点不覆盖镜像自带的路径",
    "R8 chroma 后端必须 depends_on 向量库",
    "R9 depends_on 指向真实服务",
    "R10 命令不得为空串",
    "R11 顶层卷名合法",
)


__all__ = [
    "API_COMMAND",
    "API_CONTAINER_PORT",
    "API_DOCKERFILE",
    "API_HEALTH_PATH",
    "CHROMA_HEARTBEAT_PATH",
    "CHROMA_IMAGE",
    "CHROMA_TAG",
    "CONTAINER_APP_DIR",
    "CONTAINER_CORPUS_DIR",
    "CONTAINER_INDEX_DIR",
    "CONTAINER_OPS_DIR",
    "ENVIRONMENT_NAME_PATTERN",
    "FORBIDDEN_IMAGE_TAGS",
    "IMAGE_REPOSITORY",
    "PROBE_TEST_FORMS",
    "RESERVED_MOUNT_PREFIXES",
    "SERVICE_API",
    "SERVICE_ROLES",
    "SERVICE_ROLE_DESCRIPTIONS",
    "SERVICE_STORE",
    "SERVICE_WORKER",
    "VALIDATION_RULES",
    "ComposeSpec",
    "HealthProbe",
    "PortMapping",
    "ServiceSpec",
    "VolumeMount",
    "check_committed_compose",
    "compose_summary_lines",
    "default_spec",
    "describe",
    "render_compose",
    "require_valid",
    "validate_spec",
    "write_compose",
]
