"""day060 ``serving.switch`` 的单元测试：流量切换、影子流量与单向降级."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import pytest

from smart_research_agent.llm.base import BaseLLM, Message
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.serving.errors import ServingError
from smart_research_agent.serving.switch import (
    BUCKET_BYTES,
    BUCKET_SALT_PRIMARY,
    BUCKET_SALT_SHADOW,
    ROUTE_CLOUD,
    ROUTE_DEDICATED,
    ROUTES,
    ModelSwitcher,
    RouteRecord,
    TrafficPolicy,
    bucket_of,
    route_table,
    split_summary,
)


class _StubLLM(BaseLLM):
    """测试用叶子模型：按开关成功或抛异常，并记录被调用的次数."""

    def __init__(
        self,
        *,
        reply: str = "stub-reply",
        chunks: tuple[str, ...] = ("st", "ub"),
        vision: bool = False,
        fail_chat: str = "",
        fail_stream: str = "",
        chunks_before_stream_failure: int | None = None,
        fail_tool: str = "",
        fail_vision: str = "",
        tool_reply: dict | None = None,
    ) -> None:
        self.reply = reply
        self.chunks = chunks
        self._vision = vision
        self.fail_chat = fail_chat
        self.fail_stream = fail_stream
        self.chunks_before_stream_failure = chunks_before_stream_failure
        self.fail_tool = fail_tool
        self.fail_vision = fail_vision
        self.tool_reply = tool_reply or {"content": reply}
        self.chat_calls = 0
        self.stream_calls = 0
        self.tool_calls = 0
        self.vision_calls = 0

    @property
    def supports_vision(self) -> bool:
        """视觉能力由构造参数声明（模拟 day045 的部署侧事实）."""
        return self._vision

    def chat(
        self, messages: list[Message], temperature: float = 0.7, max_tokens: int = 1024
    ) -> str:
        self.chat_calls += 1
        if self.fail_chat:
            raise RuntimeError(self.fail_chat)
        return self.reply

    def stream(
        self, messages: list[Message], temperature: float = 0.7, max_tokens: int = 1024
    ) -> Iterator[str]:
        self.stream_calls += 1
        index = 0
        for chunk in self.chunks:
            limit = (
                self.chunks_before_stream_failure
                if self.chunks_before_stream_failure is not None
                else len(self.chunks)
            )
            if self.fail_stream and index >= limit:
                raise RuntimeError(self.fail_stream)
            index += 1
            yield chunk
        if self.fail_stream and self.chunks_before_stream_failure is None:
            raise RuntimeError(self.fail_stream)

    def chat_with_tools(
        self,
        messages: list[Message],
        tools: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict:
        self.tool_calls += 1
        if self.fail_tool:
            raise RuntimeError(self.fail_tool)
        return dict(self.tool_reply)

    def chat_vision(
        self,
        messages: list[Message],
        image_data_url: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        self.vision_calls += 1
        if self.fail_vision:
            raise RuntimeError(self.fail_vision)
        return self.reply


def _messages(text: str = "什么是 LoRA？") -> list[Message]:
    return [Message(role="user", content=text)]


# --------------------------------------------------------------------------- #
# 分桶：确定性
# --------------------------------------------------------------------------- #


def test_bucket_is_deterministic_and_in_range() -> None:
    """同一个 prompt 永远落在同一个桶里——**这是灰度期间体验不抖动的根据**."""
    value = bucket_of("什么是 DPO？")
    assert value == bucket_of("什么是 DPO？")
    assert 0.0 <= value < 1.0
    assert BUCKET_BYTES == 4


def test_bucket_matches_the_documented_formula() -> None:
    """``sha256(salt + \\0 + text)`` 的前 4 字节对 ``2**32`` 取比值（可逐位复算）."""
    text = "什么是 LoRA？"
    digest = hashlib.sha256(f"{BUCKET_SALT_PRIMARY}\u0000{text}".encode("utf-8")).digest()
    expected = int.from_bytes(digest[:4], "big") / float(1 << 32)
    assert bucket_of(text) == expected


def test_primary_and_shadow_salts_give_different_buckets() -> None:
    """影子用另一个盐：否则"抽 5% 做影子"会退化成"抽前 5%"（都是同一批 prompt）."""
    text = "什么是 QLoRA？"
    assert bucket_of(text, salt=BUCKET_SALT_PRIMARY) != bucket_of(text, salt=BUCKET_SALT_SHADOW)


def test_route_names_are_exactly_two() -> None:
    assert ROUTES == (ROUTE_DEDICATED, ROUTE_CLOUD)


# --------------------------------------------------------------------------- #
# 策略
# --------------------------------------------------------------------------- #


def test_traffic_policy_validates_ratios() -> None:
    with pytest.raises(ServingError, match="dedicated_ratio 必须落在"):
        TrafficPolicy(dedicated_ratio=1.5)
    with pytest.raises(ServingError, match="shadow_ratio 必须落在"):
        TrafficPolicy(shadow_ratio=-0.1)


def test_traffic_policy_flags_bypass_and_full_cutover() -> None:
    default = TrafficPolicy()
    assert default.is_bypass is True
    assert default.is_fully_cut_over is False
    assert TrafficPolicy(dedicated_ratio=1.0).is_fully_cut_over is True
    assert TrafficPolicy(dedicated_ratio=1.0).is_bypass is False
    assert TrafficPolicy(shadow_ratio=1.0).is_bypass is False
    payload = TrafficPolicy(dedicated_ratio=0.5).to_dict()
    assert payload == {
        "dedicated_ratio": 0.5,
        "shadow_ratio": 0.0,
        "fail_open": True,
        "is_fully_cut_over": False,
        "is_bypass": False,
    }


def test_route_record_rejects_unknown_routes() -> None:
    with pytest.raises(ServingError, match="未知路由"):
        RouteRecord(index=1, bucket=0.1, route="edge", reason="r")


def test_route_record_summary_marks_flags() -> None:
    record = RouteRecord(
        index=3, bucket=0.25, route=ROUTE_DEDICATED, reason="r", shadow=True, fallback_used=True
    )
    line = record.summary_line()
    assert "#3" in line
    assert "影子" in line and "已降级" in line
    assert record.to_dict()["index"] == 3


def test_route_record_summary_without_flags_has_no_suffix() -> None:
    """平铺记录里不该出现空的方括号——**没发生的事不要留下痕迹**."""
    line = RouteRecord(index=1, bucket=0.0, route=ROUTE_CLOUD, reason="r").summary_line()
    assert "[" not in line
    assert "未完成" in line  # served_by 还没补上时也看得出来


def test_shadow_is_never_set_on_a_dedicated_route() -> None:
    """**不变式**：走专属的请求永远不会被标记成影子.

    ``_decide`` 只在云端分支上判定影子，因此 ``chat`` / ``stream`` /
    ``chat_with_tools`` 里那句"如果走了专属就把 shadow 置回 False"是恒为假的
    分支——本课把它删掉了（day059 的账：恒为真的检查不是保护，而是噪声）。
    这条测试就是删它的依据：只要 ``_decide`` 的约定被改坏，它立刻变红。
    """
    cloud, dedicated = _StubLLM(), _StubLLM(chunks=("a",), tool_reply={"content": "t"})
    switcher = ModelSwitcher(
        cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0, shadow_ratio=1.0)
    )
    switcher.chat(_messages("问题一"))
    list(switcher.stream(_messages("问题二")))
    switcher.chat_with_tools(_messages("问题三"), tools=[])
    assert len(switcher.switch_log) == 3
    assert all(item.route == ROUTE_DEDICATED for item in switcher.switch_log)
    assert switcher.shadow_calls == 0
    assert not any(item.shadow for item in switcher.switch_log)


def test_switcher_rejects_missing_arms() -> None:
    with pytest.raises(ServingError, match="都不能为 None"):
        ModelSwitcher(None, MockLLM())  # type: ignore[arg-type]
    with pytest.raises(ServingError, match="都不能为 None"):
        ModelSwitcher(MockLLM(), None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# chat：路由与降级
# --------------------------------------------------------------------------- #


def test_dedicated_ratio_one_routes_everything_to_the_dedicated_model() -> None:
    cloud, dedicated = _StubLLM(reply="cloud"), _StubLLM(reply="dedicated")
    switcher = ModelSwitcher(
        cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0)
    )
    assert switcher.chat(_messages()) == "dedicated"
    assert dedicated.chat_calls == 1
    assert cloud.chat_calls == 0
    assert switcher.dedicated_served == 1
    assert switcher.cloud_served == 0
    assert switcher.last_used_llm is dedicated
    assert switcher.switch_log[0].served_by == ROUTE_DEDICATED


def test_zero_ratio_routes_everything_to_the_cloud() -> None:
    cloud, dedicated = _StubLLM(reply="cloud"), _StubLLM(reply="dedicated")
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy())
    assert switcher.chat(_messages()) == "cloud"
    assert switcher.cloud_served == 1
    assert switcher.dedicated_served == 0
    assert switcher.last_used_llm is cloud
    assert switcher.switch_log[0].shadow is False


def test_same_prompt_always_lands_on_the_same_side() -> None:
    """哈希分桶的核心性质：**同一句话被问一万次也只会走同一侧**."""
    prompt = "请对比 LoRA 与全参微调的显存占用"
    policy = TrafficPolicy(dedicated_ratio=bucket_of(prompt) + 0.01)
    cloud, dedicated = _StubLLM(), _StubLLM()
    switcher = ModelSwitcher(cloud, dedicated, policy=policy)
    for _ in range(5):
        switcher.chat(_messages(prompt))
    assert switcher.dedicated_served == 5
    assert switcher.cloud_served == 0


def test_dedicated_failure_falls_back_to_the_cloud() -> None:
    """``fail_open=True``（缺省）时专属失败由云端兜住，但**这件事被计数**."""
    cloud = _StubLLM(reply="cloud")
    dedicated = _StubLLM(fail_chat="CUDA out of memory")
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0))
    assert switcher.chat(_messages()) == "cloud"
    assert switcher.fallback_count == 1
    assert switcher.cloud_served == 1
    record = switcher.switch_log[0]
    assert record.fallback_used is True
    assert record.route == ROUTE_DEDICATED
    assert "CUDA out of memory" in record.errors[0]
    assert switcher.last_used_llm is cloud


def test_strict_mode_propagates_the_dedicated_failure() -> None:
    """``fail_open=False`` 用于测专属模型的**真实成功率**（不许云端兜）."""
    cloud = _StubLLM()
    dedicated = _StubLLM(fail_chat="connection refused")
    switcher = ModelSwitcher(
        cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0, fail_open=False)
    )
    with pytest.raises(ServingError, match="严格模式"):
        switcher.chat(_messages())
    assert cloud.chat_calls == 0
    assert switcher.fallback_count == 0
    assert switcher.switch_log[0].served_by == ""


def test_both_arms_failing_surfaces_the_last_error() -> None:
    cloud = _StubLLM(fail_chat="cloud down")
    dedicated = _StubLLM(fail_chat="gpu gone")
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0))
    with pytest.raises(RuntimeError, match="cloud down"):
        switcher.chat(_messages())
    assert switcher.fallback_count == 1


# --------------------------------------------------------------------------- #
# 影子流量
# --------------------------------------------------------------------------- #


def test_shadow_traffic_returns_cloud_result_but_calls_dedicated() -> None:
    """影子请求**付两次推理成本**，但用户看到的是云端答案."""
    cloud = _StubLLM(reply="cloud")
    dedicated = _StubLLM(reply="dedicated")
    switcher = ModelSwitcher(
        cloud, dedicated, policy=TrafficPolicy(shadow_ratio=1.0)
    )
    assert switcher.chat(_messages()) == "cloud"
    assert switcher.shadow_calls == 1
    assert dedicated.chat_calls == 1
    assert switcher.switch_log[0].shadow is True
    assert switcher.cloud_served == 1


def test_served_counters_add_up_to_the_finished_requests() -> None:
    """两条"服务计数"必须恰好等于已完成的请求数——**影子请求也要算进去**.

    本课第一版把影子请求从两侧都排除掉了，于是"影子请求"在计数上不属于
    任何一侧，而两侧之和也对不上请求总数。这类缺陷不会报错，
    只会让"现在有多少请求真的在用专属模型"这个数字悄悄变小。
    """
    cloud, dedicated = _StubLLM(reply="cloud"), _StubLLM(reply="dedicated")
    switcher = ModelSwitcher(
        cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0, shadow_ratio=1.0)
    )
    switcher.chat(_messages("问题一"))
    switcher.chat(_messages("问题一"))  # 同一句话：稳定分桶仍然走专属
    assert switcher.dedicated_served + switcher.cloud_served == len(switcher.switch_log) == 2
    assert switcher.dedicated_served == 2

    shadow_switcher = ModelSwitcher(
        cloud, dedicated, policy=TrafficPolicy(shadow_ratio=1.0)
    )
    shadow_switcher.chat(_messages("问题二"))
    assert shadow_switcher.shadow_calls == 1
    assert shadow_switcher.cloud_served == 1
    assert shadow_switcher.dedicated_served == 0
    assert (
        shadow_switcher.cloud_served + shadow_switcher.dedicated_served
        == len(shadow_switcher.switch_log)
    )


def test_shadow_failure_never_breaks_the_user_request() -> None:
    """影子路径的唯一职责是提前发现问题——**它自己不能变成故障源**."""
    cloud = _StubLLM(reply="cloud")
    dedicated = _StubLLM(fail_chat="shadow boom")
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(shadow_ratio=1.0))
    assert switcher.chat(_messages()) == "cloud"
    assert switcher.shadow_failures == 1
    assert switcher.fallback_count == 0


def test_shadow_is_not_recorded_when_the_request_already_uses_dedicated() -> None:
    """已经真的走专属的请求不再另打影子（那会变成两次调用同一个模型）."""
    cloud, dedicated = _StubLLM(), _StubLLM()
    switcher = ModelSwitcher(
        cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0, shadow_ratio=1.0)
    )
    switcher.chat(_messages())
    assert switcher.shadow_calls == 0
    assert dedicated.chat_calls == 1


def test_shadow_is_skipped_after_the_cutover_is_complete() -> None:
    """切到 100% 之后影子没有意义（专属已经在服务了）."""
    cloud, dedicated = _StubLLM(), _StubLLM()
    switcher = ModelSwitcher(
        cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0, shadow_ratio=1.0)
    )
    assert switcher.policy.is_fully_cut_over is True
    switcher.chat(_messages())
    assert switcher.shadow_calls == 0


# --------------------------------------------------------------------------- #
# 流式：已产出片段之后不再降级
# --------------------------------------------------------------------------- #


def test_stream_from_dedicated_yields_all_chunks() -> None:
    cloud, dedicated = _StubLLM(chunks=("a", "b", "c")), _StubLLM(chunks=("x", "y"))
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0))
    assert "".join(switcher.stream(_messages())) == "xy"
    assert switcher.switch_log[0].served_by == ROUTE_DEDICATED


def test_stream_falls_back_when_the_first_chunk_never_arrived() -> None:
    """首片未产出即失败 → 沿单向降级换云端（用户什么都没看到，切换是干净的）."""
    cloud = _StubLLM(chunks=("c1", "c2"))
    dedicated = _StubLLM(fail_stream="model loading", chunks_before_stream_failure=0)
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0))
    assert "".join(switcher.stream(_messages())) == "c1c2"
    assert switcher.fallback_count == 1
    assert switcher.switch_log[0].served_by == ROUTE_CLOUD


def test_stream_refuses_to_splice_two_models_answers() -> None:
    """**本课与 day033 的刻意差异**：已产出片段后报错，而不是拼接两段回答.

    拼接出来的文本不会报错，只会答错——因此宁可报一次明确的错误。
    """
    cloud = _StubLLM(chunks=("c1",))
    dedicated = _StubLLM(fail_stream="decode crashed", chunks_before_stream_failure=1)
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0))
    stream = switcher.stream(_messages())
    assert next(stream) == "st"
    with pytest.raises(ServingError, match="拼接两段回答不会报错"):
        next(stream)
    assert cloud.stream_calls == 0


def test_stream_in_strict_mode_raises_before_the_first_chunk() -> None:
    cloud = _StubLLM(chunks=("c1",))
    dedicated = _StubLLM(fail_stream="nope", chunks_before_stream_failure=0)
    switcher = ModelSwitcher(
        cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0, fail_open=False)
    )
    with pytest.raises(ServingError, match="严格模式|专属模型流式失败"):
        list(switcher.stream(_messages()))
    assert cloud.stream_calls == 0


def test_stream_shadow_calls_dedicated_without_affecting_output() -> None:
    cloud = _StubLLM(chunks=("c",))
    dedicated = _StubLLM(chunks=("d",))
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(shadow_ratio=1.0))
    assert "".join(switcher.stream(_messages())) == "c"
    assert switcher.shadow_calls == 1
    assert dedicated.stream_calls == 1


def test_stream_shadow_failure_is_swallowed() -> None:
    cloud = _StubLLM(chunks=("c",))
    dedicated = _StubLLM(fail_stream="shadow down", chunks_before_stream_failure=0)
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(shadow_ratio=1.0))
    assert "".join(switcher.stream(_messages())) == "c"
    assert switcher.shadow_failures == 1


# --------------------------------------------------------------------------- #
# function calling
# --------------------------------------------------------------------------- #


def test_tool_calls_are_routed_like_chat() -> None:
    cloud = _StubLLM(tool_reply={"content": "cloud-tool"})
    dedicated = _StubLLM(tool_reply={"content": "dedicated-tool"})
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0))
    reply = switcher.chat_with_tools(_messages(), tools=[{"type": "function"}])
    assert reply == {"content": "dedicated-tool"}
    assert switcher.switch_log[0].served_by == ROUTE_DEDICATED


def test_tool_call_failure_falls_back_to_the_cloud() -> None:
    cloud = _StubLLM(tool_reply={"content": "cloud-tool"})
    dedicated = _StubLLM(fail_tool="tools unsupported")
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0))
    assert switcher.chat_with_tools(_messages(), tools=[]) == {"content": "cloud-tool"}
    assert switcher.fallback_count == 1


def test_tool_call_strict_mode_raises() -> None:
    cloud = _StubLLM()
    dedicated = _StubLLM(fail_tool="tools unsupported")
    switcher = ModelSwitcher(
        cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0, fail_open=False)
    )
    with pytest.raises(ServingError, match="strict|严格模式"):
        switcher.chat_with_tools(_messages(), tools=[])
    assert cloud.tool_calls == 0


def test_tool_calls_can_run_in_shadow_mode() -> None:
    cloud = _StubLLM(tool_reply={"content": "cloud-tool"})
    dedicated = _StubLLM(tool_reply={"content": "dedicated-tool"})
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(shadow_ratio=1.0))
    assert switcher.chat_with_tools(_messages(), tools=[]) == {"content": "cloud-tool"}
    assert switcher.shadow_calls == 1


def test_tool_calls_skip_the_shadow_path_when_it_is_off() -> None:
    """影子比例 0（缺省）时**一次额外调用都不能发生**——那是白花的钱."""
    cloud = _StubLLM(tool_reply={"content": "cloud-tool"})
    dedicated = _StubLLM(tool_reply={"content": "dedicated-tool"})
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy())
    assert switcher.chat_with_tools(_messages(), tools=[]) == {"content": "cloud-tool"}
    assert switcher.shadow_calls == 0
    assert dedicated.tool_calls == 0


# --------------------------------------------------------------------------- #
# 视觉能力
# --------------------------------------------------------------------------- #


def test_supports_vision_requires_both_sides() -> None:
    """用 ``and`` 而不是 ``or``：能力声明必须与**最坏情况**一致."""
    assert (
        ModelSwitcher(_StubLLM(vision=True), _StubLLM(vision=True)).supports_vision is True
    )
    assert (
        ModelSwitcher(_StubLLM(vision=True), _StubLLM(vision=False)).supports_vision is False
    )
    assert (
        ModelSwitcher(_StubLLM(vision=False), _StubLLM(vision=True)).supports_vision is False
    )


def test_vision_request_uses_the_dedicated_model_when_it_can_see() -> None:
    cloud = _StubLLM(reply="cloud", vision=True)
    dedicated = _StubLLM(reply="dedicated", vision=True)
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0))
    assert switcher.chat_vision(_messages("这张图里是什么"), "data:image/png;base64,AA") == (
        "dedicated"
    )
    assert dedicated.vision_calls == 1


def test_vision_request_goes_to_the_cloud_when_dedicated_cannot_see() -> None:
    """能力不满足时**直接换另一侧**，而不是先失败再回落.

    "这个模型没有视觉能力"是可以事先问清楚的配置事实（day045 的
    ``LocalModelSpec.supports_vision``），不是一次运行期意外。
    """
    cloud = _StubLLM(reply="cloud", vision=True)
    dedicated = _StubLLM(reply="dedicated", vision=False)
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0))
    assert switcher.chat_vision(_messages("看图"), "data:image/png;base64,AA") == "cloud"
    assert dedicated.vision_calls == 0
    assert "专属模型不支持视觉" in switcher.switch_log[0].reason


def test_vision_request_fails_when_neither_side_can_see() -> None:
    switcher = ModelSwitcher(
        _StubLLM(vision=False), _StubLLM(vision=False), policy=TrafficPolicy(dedicated_ratio=1.0)
    )
    with pytest.raises(ServingError, match="都不支持视觉"):
        switcher.chat_vision(_messages("看图"), "data:image/png;base64,AA")


def test_vision_failure_falls_back_to_the_cloud() -> None:
    cloud = _StubLLM(reply="cloud", vision=True)
    dedicated = _StubLLM(vision=True, fail_vision="vision kernel crashed")
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0))
    assert switcher.chat_vision(_messages("看图"), "data:image/png;base64,AA") == "cloud"
    assert switcher.fallback_count == 1


def test_vision_failure_without_cloud_vision_raises() -> None:
    cloud = _StubLLM(vision=False)
    dedicated = _StubLLM(vision=True, fail_vision="vision kernel crashed")
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0))
    with pytest.raises(ServingError, match="专属视觉模型调用失败"):
        switcher.chat_vision(_messages("看图"), "data:image/png;base64,AA")


def test_vision_shadow_is_recorded_but_output_comes_from_the_cloud() -> None:
    cloud = _StubLLM(reply="cloud", vision=True)
    dedicated = _StubLLM(reply="dedicated", vision=True)
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(shadow_ratio=1.0))
    assert switcher.chat_vision(_messages("看图"), "data:image/png;base64,AA") == "cloud"
    assert switcher.shadow_calls == 1
    assert dedicated.vision_calls == 1
    assert switcher.switch_log[0].shadow is True


def test_vision_shadow_is_not_counted_when_dedicated_cannot_see() -> None:
    """``shadow_calls`` 回答的是"有多少请求付了两次推理成本"——没发生的调用不能计入.

    本课第一版在视觉路径上漏了这一步：记录被标成"影子"，而专属模型
    一次都没有被调用，于是报告里的那个数字凭空多了一份。
    """
    cloud = _StubLLM(reply="cloud", vision=True)
    dedicated = _StubLLM(reply="dedicated", vision=False)
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(shadow_ratio=1.0))
    assert switcher.chat_vision(_messages("看图"), "data:image/png;base64,AA") == "cloud"
    assert switcher.shadow_calls == 0
    assert dedicated.vision_calls == 0
    assert "本次没有打影子" in switcher.switch_log[0].reason


# --------------------------------------------------------------------------- #
# 可观测性
# --------------------------------------------------------------------------- #


def test_describe_exposes_counters_and_policy() -> None:
    cloud = _StubLLM(reply="cloud")
    dedicated = _StubLLM(fail_chat="down")
    switcher = ModelSwitcher(cloud, dedicated, policy=TrafficPolicy(dedicated_ratio=1.0))
    switcher.chat(_messages())
    payload = switcher.describe()
    assert payload["requests"] == 1
    assert payload["fallback_count"] == 1
    assert payload["cloud_served"] == 1
    assert payload["dedicated_name"] == ROUTE_DEDICATED
    assert payload["policy"]["dedicated_ratio"] == 1.0
    assert payload["last_used"] == "_StubLLM"
    assert payload["supports_vision"] is False


def test_describe_before_any_call_has_no_last_used() -> None:
    switcher = ModelSwitcher(_StubLLM(), _StubLLM())
    assert switcher.describe()["last_used"] is None
    assert switcher.last_used_llm is None
    assert switcher.switch_log == []


def test_last_used_llm_supports_cost_accounting() -> None:
    """day046 的计费读的是**叶子**的 ``usage_log``：包装器不产生 token."""
    leaf = MockLLM(responses=["来自云端"])
    switcher = ModelSwitcher(leaf, _StubLLM(), policy=TrafficPolicy())
    switcher.chat(_messages())
    used = switcher.last_used_llm
    assert used is leaf
    assert used is not None and used.usage_log


def test_split_summary_counts_without_calling_any_model() -> None:
    prompts = [f"问题 {i}" for i in range(20)]
    payload = split_summary(prompts, TrafficPolicy(dedicated_ratio=0.25, shadow_ratio=0.5))
    assert payload["requests"] == 20
    assert payload["dedicated"] + payload["cloud"] == 20
    assert payload["dedicated_ratio_requested"] == 0.25
    assert 0.0 <= payload["dedicated_ratio_actual"] <= 1.0
    # 纯计算：没有任何模型被调用
    assert payload["shadow"] <= payload["cloud"]


def test_split_summary_is_reproducible() -> None:
    prompts = [f"问题 {i}" for i in range(10)]
    policy = TrafficPolicy(dedicated_ratio=0.3, shadow_ratio=0.4)
    assert split_summary(prompts, policy) == split_summary(prompts, policy)


def test_split_summary_handles_an_empty_batch() -> None:
    payload = split_summary([], TrafficPolicy(dedicated_ratio=0.5))
    assert payload["requests"] == 0
    assert payload["dedicated_ratio_actual"] == 0.0


def test_split_summary_on_a_full_cutover_sends_everything_to_dedicated() -> None:
    payload = split_summary([f"q{i}" for i in range(5)], TrafficPolicy(dedicated_ratio=1.0))
    assert payload["dedicated"] == 5
    assert payload["cloud"] == 0


def test_route_table_lists_each_field_with_its_cost() -> None:
    rows = route_table(TrafficPolicy(dedicated_ratio=0.05))
    fields = [row["field"] for row in rows]
    assert fields == ["dedicated_ratio", "shadow_ratio", "fail_open", "is_fully_cut_over"]
    ratio_row = rows[0]
    assert ratio_row["value"] == 0.05
    assert "哈希分桶不是随机分流" in ratio_row["cost"]
    assert "fallback_count" in rows[2]["cost"]
    assert rows[3]["value"] is False


def test_route_table_defaults_are_the_dataclass_defaults() -> None:
    """缺省表与 ``TrafficPolicy()`` 同源（文档与代码只有一份权威）."""
    rows = route_table()
    assert rows[0]["value"] == TrafficPolicy().dedicated_ratio
    assert rows[2]["value"] is True
