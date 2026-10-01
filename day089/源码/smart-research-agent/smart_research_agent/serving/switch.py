"""模型切换：让 LLM 调用层同时持有云端与专属模型，并按流量策略分配（M5-D11）.

day033 的 ``ModelRouter`` 与今天的 ``ModelSwitcher`` 都在"选一个模型"，
但它们选的是**两个不同的东西**，把它们混为一谈会让两条策略互相打架：

| | `ModelRouter`（day033） | `ModelSwitcher`（今天） |
|---|---|---|
| 选什么 | **候选**：同一个任务交给哪个模型 | **流量**：这一批请求里哪些走新模型 |
| 判据 | 任务复杂度 vs 模型能力 | 部署阶段（灰度 5% → 50% → 100%） |
| 目标是 | 省钱（别用大炮打蚊子） | **换模型**：让新模型逐步接管生产 |
| 失败时 | 沿降级链换更强的模型 | 回落到**已验证的旧路径**（云端） |
| 生命周期 | 长期存在 | **过渡态**：切完 100% 就该被删掉 |

最后一行是本类唯一"短命"的设计目标：``ModelSwitcher`` 是**灰度期间的基础设施**，
它的终点是 ``dedicated_ratio=1.0`` 之后再被摘掉。因此它必须能打印出
"现在是什么状态、还差多少才切完"，而不是只当一个透明的代理。

## 分桶为什么用哈希而不是随机数

``bucket = sha256(prompt)[:8] / 2**32``，落在 ``[0, 1)``，与 ``dedicated_ratio``
比较。三条理由，一条代价：

1. **同一个 prompt 永远去同一个模型**。用户重复问同一个问题时，两次得到的
   不会一会儿来自云端、一会儿来自专属——**线上体验的抖动本身就是一种事故**；
2. **可复现**。测试与 demo 里没有随机数，"这次分流到了哪些 prompt"可以逐条复现，
   也可以人工指定"就这一个 prompt 走专属"来复现一次线上问题；
3. 与 day058 的 ``run_id``、day059 的 ``run_id_for`` 同一条纪律：
   **确定性让"这个数字是哪来的"可被回答**。

代价必须说清楚：**哈希分桶不是随机分流**。``dedicated_ratio=0.05`` 的含义是
"哈希落在前 5% 的**那些 prompt**"，而不是"随机 5% 的请求"。同一句话被问一万次，
它要么全在专属侧、要么全在云端侧。要做统计意义上的 A/B，必须引入随机数——
本课程刻意不引（它会让每天的实验结果不可复现），因此这条限制被写进
``route_table()`` 的"代价"一列，而不是藏起来。

## 降级只向一个方向

``dedicated`` 失败时回落到 ``cloud``（``fail_open=True``，缺省）；
**``cloud`` 失败时绝不回落到 ``dedicated``**。这个不对称是刻意的：

> 云端是已验证路径，专属模型是新增路径。反向降级等于"在云端出故障时，
> 把流量交给一个还没通过验证的模型"——两个故障叠加，而且第二个故障的
> 根因会被第一个掩盖。

## 影子流量：先看再切

``shadow_ratio`` 决定的请求会**同时调用专属模型，但只返回云端的结果**。
它是灰度之前的一步：在真实请求上测专属模型的表现，而用户完全不受影响。
代价是这部分请求要付两次推理成本——所以它有一个独立的开关与一个计数字段
（``shadow_calls``），而不是藏在 ``dedicated_ratio`` 里。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.serving.errors import ServingError
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 两条例内路由的取值（它们不是模型名，而是**路径名**）.
ROUTE_DEDICATED = "dedicated"
ROUTE_CLOUD = "cloud"
ROUTES: tuple[str, ...] = (ROUTE_DEDICATED, ROUTE_CLOUD)

#: 分桶用的哈希盐：影子分流用**另一个盐**，否则它与主分流完全同序，
#: "抽 5% 出来做影子"会退化成"抽前 5%"——而那 5% 恰好全是同一批 prompt。
BUCKET_SALT_PRIMARY = "primary"
BUCKET_SALT_SHADOW = "shadow"

#: 从哈希里取多少字节做分桶：4 字节 = 2**32 个格子，
#: 因此比率的精度约为 2.3e-10，远细于任何有意义的流量比例。
BUCKET_BYTES = 4
BUCKET_SPACE = float(1 << (8 * BUCKET_BYTES))


@dataclass(frozen=True)
class TrafficPolicy:
    """流量策略：两个比率 + 一个开关.

    ``dedicated_ratio`` 与 ``shadow_ratio`` 是两个**独立的比率**而不是
    "影子是灰度的一部分"：影子请求的服务结果来自云端，灰度请求的服务结果
    来自专属——它们的**用户可见行为**完全不同，混成一个数字就再也说不清
    "现在有多少用户已经真正在用专属模型了"。
    """

    dedicated_ratio: float = 0.0
    shadow_ratio: float = 0.0
    fail_open: bool = True

    def __post_init__(self) -> None:
        for name in ("dedicated_ratio", "shadow_ratio"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ServingError(f"{name} 必须落在 [0, 1]，收到 {value}")

    @property
    def is_fully_cut_over(self) -> bool:
        """是否已经完全切到专属模型（切换的终点，之后本类即可被摘掉）."""
        return self.dedicated_ratio >= 1.0

    @property
    def is_bypass(self) -> bool:
        """是否等同"完全绕过切换器"（0% 专属、0% 影子）."""
        return self.dedicated_ratio <= 0.0 and self.shadow_ratio <= 0.0

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典（含派生量）."""
        payload = asdict(self)
        payload["is_fully_cut_over"] = self.is_fully_cut_over
        payload["is_bypass"] = self.is_bypass
        return payload


@dataclass
class RouteRecord:
    """一次调用的路由记录（"为什么这次走了它"）.

    可变对象（与 day058/059 里那些 frozen 的判定结果不同）：记录是在调用
    **过程中**被补齐的——先写下决策，调用成功后再补 ``served_by``，
    失败时补 ``errors``。这三件事发生在三个时刻，用一个 frozen 对象表达
    只能靠"产生三个对象再合并"，反而更容易漏字段。
    """

    index: int
    bucket: float
    route: str
    reason: str
    shadow: bool = False
    fallback_used: bool = False
    served_by: str = ""
    errors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise ServingError(f"未知路由 {self.route!r}，可选 {', '.join(ROUTES)}")

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 ``json.dumps`` 的字典."""
        return asdict(self)

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        flags = []
        if self.shadow:
            flags.append("影子")
        if self.fallback_used:
            flags.append("已降级")
        suffix = f" [{'/'.join(flags)}]" if flags else ""
        return (
            f"#{self.index} bucket={self.bucket:.6f} → {self.route}"
            f"（实际 {self.served_by or '未完成'}）{suffix} | {self.reason}"
        )


def bucket_of(text: str, *, salt: str = BUCKET_SALT_PRIMARY) -> float:
    """把任意文本映射到 ``[0, 1)`` 上的一个稳定分桶值.

    ``sha256`` 的**前 4 字节**转成无符号整数，再除以 ``2**32``。
    用前 4 字节而不是全部 32 字节：``ratio`` 的分辨率只需到万分之一，
    而短哈希更容易在日志里一眼比对。
    """
    digest = hashlib.sha256(f"{salt}\u0000{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:BUCKET_BYTES], "big") / BUCKET_SPACE


class ModelSwitcher(BaseLLM):
    """云端与专属模型之间的流量切换器（实现 ``BaseLLM``，对 Agent 透明）.

    ``cloud`` 与 ``dedicated`` 都是 ``BaseLLM``：前者是已经验证过的路径
    （云端 API 或 day045 的 ``LocalModel``），后者是本次要接管的专属模型。
    两个名字只是**日志里的标签**，不参与任何判定——**判定只看比率**。
    """

    def __init__(
        self,
        cloud: BaseLLM,
        dedicated: BaseLLM,
        *,
        policy: TrafficPolicy | None = None,
        cloud_name: str = ROUTE_CLOUD,
        dedicated_name: str = ROUTE_DEDICATED,
    ):
        if cloud is None or dedicated is None:
            raise ServingError("云端与专属模型都不能为 None：切换器不负责凭空造模型")
        self.cloud = cloud
        self.dedicated = dedicated
        self.policy = policy or TrafficPolicy()
        self.cloud_name = cloud_name
        self.dedicated_name = dedicated_name
        self.switch_log: list[RouteRecord] = []
        #: 最后一次**实际提供服务的叶子模型**（day046 的用量与计费挂在它上面：
        #: 切换器是包装器，它不产生 token，把包装器交给计费器会在
        #: "包装器没有 usage_log" 处断链——与 day033 的 ``ModelRouter`` 同一条理由）。
        self._last_used: BaseLLM | None = None

    # ------------------------------------------------------------------ 计数器
    @property
    def last_used_llm(self) -> BaseLLM | None:
        """最后一次实际生效的叶子模型（从未调用过时为 ``None``）."""
        return self._last_used

    @property
    def dedicated_served(self) -> int:
        """**用户看到的答案来自专属模型**的请求数.

        影子请求永远不会落在这里——影子的定义就是"只返回云端结果"。
        """
        return sum(
            1 for item in self.switch_log if item.served_by == self.dedicated_name and not item.shadow
        )

    @property
    def cloud_served(self) -> int:
        """**用户看到的答案来自云端**的请求数（**含影子请求**）.

        影子请求的用户可见答案就是云端给的，因此它必须计入这里：
        这条计数与 ``dedicated_served`` 一起，应当恰好等于**已完成的请求数**。
        本课第一版把影子请求从两侧都排除掉了，于是"影子请求"在计数上
        不属于任何一侧——而那正是最需要被数清楚的一类请求。
        """
        return sum(1 for item in self.switch_log if item.served_by == self.cloud_name)

    @property
    def fallback_count(self) -> int:
        """发生了"专属失败 → 云端兜住"的请求数.

        这个计数是 ``fail_open=True`` 的**代价表**：它大于 0 说明专属模型
        正在被云端悄悄兜着，而"一切正常"的假象正是这样产生的。
        """
        return sum(1 for item in self.switch_log if item.fallback_used)

    @property
    def shadow_calls(self) -> int:
        """影子调用总数（这些请求付了两次推理成本）."""
        return sum(1 for item in self.switch_log if item.shadow)

    @property
    def shadow_failures(self) -> int:
        """影子调用里失败的次数（**不影响用户**，是灰度的前置信号）."""
        return sum(
            1
            for item in self.switch_log
            if item.shadow and item.errors
        )

    def describe(self) -> dict[str, Any]:
        """输出切换器的运行档案（日志、``/health``、``/models`` 都用它）."""
        return {
            "cloud_name": self.cloud_name,
            "dedicated_name": self.dedicated_name,
            "policy": self.policy.to_dict(),
            "requests": len(self.switch_log),
            "dedicated_served": self.dedicated_served,
            "cloud_served": self.cloud_served,
            "fallback_count": self.fallback_count,
            "shadow_calls": self.shadow_calls,
            "shadow_failures": self.shadow_failures,
            "last_used": None if self._last_used is None else type(self._last_used).__name__,
            "supports_vision": self.supports_vision,
        }

    # ------------------------------------------------------------------ 路由
    @staticmethod
    def _last_user_text(messages: list[Message]) -> str:
        """取最近一条 user 消息作为分桶键（与 day033 的复杂度估计同一取材口径）."""
        return next((m.content for m in reversed(messages) if m.role == "user"), "")

    def _decide(self, messages: list[Message]) -> tuple[str, float, bool, str]:
        """算出这次的首选路径、分桶值、是否要打影子、以及人类可读的理由.

        **一句话被问一万次也只会走同一侧**，这正是哈希分桶与随机分流的分野。
        """
        key = self._last_user_text(messages)
        primary = bucket_of(key, salt=BUCKET_SALT_PRIMARY)
        if primary < self.policy.dedicated_ratio:
            return (
                ROUTE_DEDICATED,
                primary,
                False,
                f"分桶 {primary:.6f} < 专属比例 {self.policy.dedicated_ratio}",
            )
        shadow_bucket = bucket_of(key, salt=BUCKET_SALT_SHADOW)
        shadow = shadow_bucket < self.policy.shadow_ratio
        reason = f"分桶 {primary:.6f} >= 专属比例 {self.policy.dedicated_ratio}"
        if shadow:
            reason += (
                f"；影子分桶 {shadow_bucket:.6f} < 影子比例 {self.policy.shadow_ratio}"
                "（额外调用专属模型，但只返回云端结果）"
            )
        return ROUTE_CLOUD, primary, shadow, reason

    def _open_record(self, route: str, bucket: float, shadow: bool, reason: str) -> RouteRecord:
        """开一条记录（决策先落档，成败随后补）."""
        record = RouteRecord(
            index=len(self.switch_log) + 1,
            bucket=round(bucket, 6),
            route=route,
            reason=reason,
            shadow=shadow,
        )
        self.switch_log.append(record)
        return record

    def _run_shadow(self, messages: list[Message], record: RouteRecord, call: Any) -> None:
        """执行一次影子调用：**只记录、只计数，绝不向上抛**.

        影子路径的唯一职责是"提前发现专属模型的问题"，因此它的异常
        必须被吞在这里——一次影子失败导致用户请求 500，是把观测手段
        变成了故障源。失败次数由 ``shadow_failures`` 单独暴露。
        """
        try:
            call(self.dedicated)
        except Exception as exc:  # noqa: BLE001 - 影子路径吞掉一切异常
            record.errors.append(f"{self.dedicated_name}(shadow): {exc}")
            logger.warning("影子调用失败（不影响用户请求）：%s", exc)

    def chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """按流量策略选择路径；专属失败且 ``fail_open`` 时回落到云端."""
        route, bucket, shadow, reason = self._decide(messages)
        record = self._open_record(route, bucket, shadow, reason)

        def call(leaf: BaseLLM) -> str:
            return leaf.chat(messages, temperature=temperature, max_tokens=max_tokens)

        if route == ROUTE_DEDICATED:
            try:
                reply = call(self.dedicated)
            except Exception as exc:  # noqa: BLE001
                if not self.policy.fail_open:
                    record.errors.append(f"{self.dedicated_name}: {exc}")
                    record.served_by = ""
                    raise ServingError(
                        f"专属模型调用失败且策略为 fail_open=False（严格模式）：{exc}"
                    ) from exc
                record.fallback_used = True
                record.errors.append(f"{self.dedicated_name}: {exc}")
                reply = call(self.cloud)
                record.served_by = self.cloud_name
                self._last_used = self.cloud
            else:
                record.served_by = self.dedicated_name
                self._last_used = self.dedicated
            # 这里**刻意没有** "if record.shadow: record.shadow = False" 那一步：
            # ``_decide`` 里走到专属分支时返回的 shadow 恒为 False（"打影子"只
            # 在云端分支上判定），因此那个条件是一个**恒为假**的分支——
            # 而 day059 已经记过这笔账：一条永远为真的检查不是保护，而是噪声。
            # 这个不变式由 ``test_shadow_is_never_set_on_a_dedicated_route`` 钉住，
            # 将来谁改了 ``_decide`` 的返回约定，那条测试会立刻变红。
            return reply

        if shadow:
            self._run_shadow(messages, record, lambda leaf: call(leaf))
        reply = call(self.cloud)
        record.served_by = self.cloud_name
        self._last_used = self.cloud
        return reply

    def stream(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """流式版本：**已产出片段之后不再降级**.

        这是与 day033 ``ModelRouter.stream`` 的一处刻意差异。那里在异常时直接
        换下一个模型重新流式，代价是调用方可能已经收到前 20 个字——
        于是最终输出是"两个模型的两段话拼在一起"，而**它不会报错**。
        切换器的语义更严格：一旦有片段落地，就把它当成"这次答案已经开始写了"，
        宁可报一次明确的错误，也不产生一段看不出问题的拼接文本。
        """
        route, bucket, shadow, reason = self._decide(messages)
        record = self._open_record(route, bucket, shadow, reason)

        if route == ROUTE_DEDICATED:
            produced = False
            try:
                for chunk in self.dedicated.stream(
                    messages, temperature=temperature, max_tokens=max_tokens
                ):
                    produced = True
                    record.served_by = self.dedicated_name
                    self._last_used = self.dedicated
                    yield chunk
            except Exception as exc:  # noqa: BLE001
                record.errors.append(f"{self.dedicated_name}: {exc}")
                if produced or not self.policy.fail_open:
                    record.served_by = "" if not produced else record.served_by
                    raise ServingError(
                        "专属模型流式失败"
                        + ("（已产出片段，不再降级：拼接两段回答不会报错，只会答错）" if produced else "")
                        + f"：{exc}"
                    ) from exc
                record.fallback_used = True
                logger.warning("专属模型流式首片前失败，回落到云端：%s", exc)
            else:
                return

        if shadow:
            self._run_shadow(
                messages,
                record,
                lambda leaf: "".join(
                    leaf.stream(messages, temperature=temperature, max_tokens=max_tokens)
                ),
            )
        for chunk in self.cloud.stream(messages, temperature=temperature, max_tokens=max_tokens):
            record.served_by = self.cloud_name
            self._last_used = self.cloud
            yield chunk

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """工具调用版本：与 ``chat`` 相同的路由与降级语义（day037/039 的协议不变）."""
        route, bucket, shadow, reason = self._decide(messages)
        record = self._open_record(route, bucket, shadow, reason)

        def call(leaf: BaseLLM) -> dict[str, Any]:
            return leaf.chat_with_tools(
                messages, tools=tools, temperature=temperature, max_tokens=max_tokens
            )

        if route == ROUTE_DEDICATED:
            try:
                reply = call(self.dedicated)
            except Exception as exc:  # noqa: BLE001
                if not self.policy.fail_open:
                    record.errors.append(f"{self.dedicated_name}: {exc}")
                    raise ServingError(
                        f"专属模型 function calling 失败且策略为严格模式：{exc}"
                    ) from exc
                record.fallback_used = True
                record.errors.append(f"{self.dedicated_name}: {exc}")
                reply = call(self.cloud)
                record.served_by = self.cloud_name
                self._last_used = self.cloud
            else:
                record.served_by = self.dedicated_name
                self._last_used = self.dedicated
            return reply

        if shadow:
            self._run_shadow(messages, record, lambda leaf: call(leaf))
        reply = call(self.cloud)
        record.served_by = self.cloud_name
        self._last_used = self.cloud
        return reply

    @property
    def supports_vision(self) -> bool:
        """两侧**都**要支持视觉，切换器才可以声明支持视觉.

        用 ``and`` 而不是 ``or``：声明 ``or`` 会让"专属模型不支持视觉"
        在切流量时才暴露成一次运行期失败；用 ``and`` 则让能力声明
        与"最坏情况"一致——这与 day040 把纯文本模型直接跳过是同一种保守。
        """
        return bool(
            getattr(self.cloud, "supports_vision", False)
            and getattr(self.dedicated, "supports_vision", False)
        )

    def chat_vision(
        self,
        messages: list[Message],
        image_data_url: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """视觉请求：只走**两条路径里支持视觉的那一条**（day040 的路由规则）.

        与 ``chat`` 的差别只有一处：能力不满足时**直接换另一侧**，
        而不是先失败再回落——因为"这个模型没有视觉能力"是一个可以事先
        问清楚的配置事实（day045 的 ``LocalModelSpec.supports_vision``），
        不是一次运行期意外。
        """
        route, bucket, shadow, reason = self._decide(messages)
        record = self._open_record(route, bucket, shadow, reason)
        dedicated_ok = bool(getattr(self.dedicated, "supports_vision", False))
        cloud_ok = bool(getattr(self.cloud, "supports_vision", False))
        if not dedicated_ok and not cloud_ok:
            raise ServingError("云端与专属模型都不支持视觉，无法处理图像请求")

        def call(leaf: BaseLLM) -> str:
            return leaf.chat_vision(
                messages, image_data_url, temperature=temperature, max_tokens=max_tokens
            )

        if route == ROUTE_DEDICATED and dedicated_ok:
            try:
                reply = call(self.dedicated)
            except Exception as exc:  # noqa: BLE001
                record.errors.append(f"{self.dedicated_name}: {exc}")
                if not self.policy.fail_open or not cloud_ok:
                    record.fallback_used = False
                    raise ServingError(f"专属视觉模型调用失败：{exc}") from exc
                record.fallback_used = True
                reply = call(self.cloud)
                record.served_by = self.cloud_name
                self._last_used = self.cloud
            else:
                record.served_by = self.dedicated_name
                self._last_used = self.dedicated
            return reply
        if route == ROUTE_DEDICATED and not dedicated_ok:
            record.reason = f"{record.reason}；专属模型不支持视觉，本次改走云端"
        if shadow and dedicated_ok:
            self._run_shadow(messages, record, lambda leaf: call(leaf))
        elif shadow:
            # 专属模型没有视觉能力 → 影子调用**根本没有发生**，
            # 因此它不能被计入 ``shadow_calls``（那个计数回答的是
            # "有多少请求付了两次推理成本"）。本课第一版漏了这一步，
            # 于是报告里出现了"打了影子"而实际上一次都没调用过。
            record.shadow = False
            record.reason = f"{record.reason}；专属模型不支持视觉，本次没有打影子"
        reply = call(self.cloud)
        record.served_by = self.cloud_name
        self._last_used = self.cloud
        return reply


def split_summary(prompts: list[str], policy: TrafficPolicy) -> dict[str, Any]:
    """给定一批 prompt，算出血缘分流的结果（**不调用任何模型**）.

    它是"先看清楚会怎么分，再决定要不要切流量"的工具：``dedicated_ratio=0.2``
    在一批真实问题上到底分走多少条，可以在这里先算一遍。同时它也是
    "哈希分桶不是随机分流"这条限制的**可执行证据**——同一批 prompt 跑两次
    得到的是同一份名单。
    """
    dedicated = 0
    shadow = 0
    for prompt in prompts:
        primary = bucket_of(prompt, salt=BUCKET_SALT_PRIMARY)
        if primary < policy.dedicated_ratio:
            dedicated += 1
            continue
        if bucket_of(prompt, salt=BUCKET_SALT_SHADOW) < policy.shadow_ratio:
            shadow += 1
    total = len(prompts)
    return {
        "requests": total,
        "dedicated": dedicated,
        "cloud": total - dedicated,
        "shadow": shadow,
        "dedicated_ratio_requested": policy.dedicated_ratio,
        "dedicated_ratio_actual": round(dedicated / total, 6) if total else 0.0,
    }


def route_table(policy: TrafficPolicy | None = None) -> list[dict[str, Any]]:
    """切换策略的对照表（API 的自我描述端点与文档同源）.

    表里带 ``cost`` 一列：**每一条策略都有代价**，把它们写出来是让"我们为什么
    选它"变成一个可以被复核的决定，而不是一次口味选择。
    """
    resolved = policy or TrafficPolicy()
    return [
        {
            "field": "dedicated_ratio",
            "value": resolved.dedicated_ratio,
            "meaning": "分桶值小于它的请求由专属模型直接服务（用户看到专属的答案）",
            "cost": "哈希分桶不是随机分流：同一句话被问一万次也只会走同一侧",
        },
        {
            "field": "shadow_ratio",
            "value": resolved.shadow_ratio,
            "meaning": "落在云端侧且影子分桶命中时，额外调用专属模型但**只返回云端结果**",
            "cost": "这部分请求付两次推理成本；影子失败只记数、不影响用户",
        },
        {
            "field": "fail_open",
            "value": resolved.fail_open,
            "meaning": "专属模型失败时回落到云端（严格模式置 False 用于测真实成功率）",
            "cost": (
                "fail_open=True 时「一切正常」可能是云端兜出来的："
                "必须同时看 fallback_count"
            ),
        },
        {
            "field": "is_fully_cut_over",
            "value": resolved.is_fully_cut_over,
            "meaning": "是否已切到 100%（切换器的终点：此后它应当被摘掉）",
            "cost": "永远留在 100% 的切换器是一层没有收益的间接层",
        },
    ]


__all__ = [
    "BUCKET_BYTES",
    "BUCKET_SALT_PRIMARY",
    "BUCKET_SALT_SHADOW",
    "ROUTES",
    "ROUTE_CLOUD",
    "ROUTE_DEDICATED",
    "ModelSwitcher",
    "RouteRecord",
    "TrafficPolicy",
    "bucket_of",
    "route_table",
    "split_summary",
]
