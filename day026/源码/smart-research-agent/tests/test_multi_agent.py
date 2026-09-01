"""多 Agent 协作测试（MockLLM 驱动，完全离线）."""

from __future__ import annotations

from smart_research_agent.agent.message import Message
from smart_research_agent.agent.orchestrator import MultiAgentOrchestrator
from smart_research_agent.agent.roles import AnalystAgent, ResearcherAgent, WriterAgent
from smart_research_agent.llm.mock import MockLLM


def _make_orchestrator(responses: list[str]) -> tuple[MultiAgentOrchestrator, MockLLM]:
    llm = MockLLM(responses=responses)
    orchestrator = MultiAgentOrchestrator(
        researcher=ResearcherAgent(llm=llm),
        analyst=AnalystAgent(llm=llm),
        writer=WriterAgent(llm=llm),
    )
    return orchestrator, llm


_SCRIPT = ["资料：RAG 是检索增强生成", "要点：1. 检索 2. 生成", "文章：RAG 技术漫谈……"]


class TestMessage:
    def test_to_dict_round_trip(self):
        msg = Message(sender="a", receiver="b", content="hi", metadata={"stage": "research"})
        data = msg.to_dict()
        assert data == {
            "sender": "a",
            "receiver": "b",
            "content": "hi",
            "metadata": {"stage": "research"},
        }

    def test_metadata_defaults_to_empty_dict(self):
        msg = Message(sender="a", receiver="b", content="hi")
        assert msg.metadata == {}
        assert msg.to_dict()["metadata"] == {}

    def test_default_metadata_not_shared(self):
        m1 = Message(sender="a", receiver="b", content="x")
        m2 = Message(sender="a", receiver="b", content="y")
        m1.metadata["k"] = "v"
        assert "k" not in m2.metadata


class TestRoles:
    def test_roles_have_distinct_system_prompts(self):
        llm = MockLLM(responses=["ok"] * 3)
        researcher = ResearcherAgent(llm=llm)
        analyst = AnalystAgent(llm=llm)
        writer = WriterAgent(llm=llm)
        prompts = {researcher.system_prompt, analyst.system_prompt, writer.system_prompt}
        assert len(prompts) == 3

    def test_role_calls_llm_once_with_system_and_user(self):
        llm = MockLLM(responses=["资料清单"])
        researcher = ResearcherAgent(llm=llm)
        result = researcher.work(task="收集资料")
        assert result == "资料清单"
        assert len(llm.calls) == 1
        system_msg, user_msg = llm.calls[0]
        assert system_msg.role == "system"
        assert "研究员" in system_msg.content
        assert "收集资料" in user_msg.content

    def test_context_is_passed_into_prompt(self):
        llm = MockLLM(responses=["要点"])
        analyst = AnalystAgent(llm=llm)
        analyst.work(task="提炼要点", context="上游的研究资料")
        user_msg = llm.calls[0][1]
        assert "上游的研究资料" in user_msg.content

    def test_custom_name_overrides_default(self):
        agent = ResearcherAgent(llm=MockLLM(), name="研究员甲")
        assert agent.name == "研究员甲"


class TestOrchestrator:
    def test_pipeline_produces_article(self):
        orchestrator, _ = _make_orchestrator(_SCRIPT)
        result = orchestrator.run("RAG 技术")
        assert result["topic"] == "RAG 技术"
        assert result["research"] == _SCRIPT[0]
        assert result["analysis"] == _SCRIPT[1]
        assert result["article"] == _SCRIPT[2]

    def test_three_roles_called_in_order(self):
        orchestrator, llm = _make_orchestrator(_SCRIPT)
        orchestrator.run("RAG 技术")
        assert len(llm.calls) == 3
        # 三次调用的 system prompt 依次属于研究员、分析师、撰稿人
        assert "研究员" in llm.calls[0][0].content
        assert "分析师" in llm.calls[1][0].content
        assert "撰稿人" in llm.calls[2][0].content

    def test_upstream_output_flows_downstream(self):
        orchestrator, llm = _make_orchestrator(_SCRIPT)
        orchestrator.run("RAG 技术")
        # 分析师收到的 user 消息里包含研究员的产出
        assert _SCRIPT[0] in llm.calls[1][1].content
        # 撰稿人收到的 user 消息里包含分析师的产出
        assert _SCRIPT[1] in llm.calls[2][1].content

    def test_message_log_records_full_trace(self):
        orchestrator, _ = _make_orchestrator(_SCRIPT)
        result = orchestrator.run("RAG 技术")
        log = result["message_log"]
        assert len(log) == 4
        pairs = [(m["sender"], m["receiver"]) for m in log]
        assert pairs == [
            ("orchestrator", "researcher"),
            ("researcher", "analyst"),
            ("analyst", "writer"),
            ("writer", "orchestrator"),
        ]
        stages = [m["metadata"]["stage"] for m in log]
        assert stages == ["dispatch_research", "research", "analysis", "article"]
        # 日志中的交接内容与各阶段产出一一对应
        assert log[1]["content"] == result["research"]
        assert log[2]["content"] == result["analysis"]
        assert log[3]["content"] == result["article"]

    def test_message_log_reset_between_runs(self):
        orchestrator, _ = _make_orchestrator(_SCRIPT * 2)
        orchestrator.run("话题一")
        orchestrator.run("话题二")
        assert len(orchestrator.message_log) == 4
