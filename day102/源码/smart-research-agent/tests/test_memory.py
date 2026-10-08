"""短期记忆与会话管理测试."""

from __future__ import annotations

import pytest

from smart_research_agent.llm.base import Message
from smart_research_agent.memory.session import SessionManager
from smart_research_agent.memory.short_term import ShortTermMemory


def _msg(role: str, n: int) -> Message:
    return Message(role=role, content=f"msg-{role}-{n}")


class TestShortTermMemory:
    def test_add_and_get(self):
        mem = ShortTermMemory(max_messages=5)
        mem.add(_msg("user", 1))
        assert len(mem.get_all()) == 1

    def test_truncation_keeps_system_and_drops_oldest(self):
        mem = ShortTermMemory(max_messages=4)
        mem.add(Message(role="system", content="sys"))
        for i in range(5):
            mem.add(_msg("user", i))
        all_msgs = mem.get_all()
        assert len(all_msgs) == 4
        assert all_msgs[0].role == "system"
        # 最老的 user 消息 msg-user-0、msg-user-1 应被丢弃
        contents = [m.content for m in all_msgs]
        assert "msg-user-0" not in contents
        assert "msg-user-4" in contents

    def test_invalid_max_messages(self):
        with pytest.raises(ValueError):
            ShortTermMemory(max_messages=1)

    def test_clear(self):
        mem = ShortTermMemory()
        mem.add(_msg("user", 1))
        mem.clear()
        assert mem.get_all() == []


class TestSessionManager:
    def test_sessions_are_isolated(self):
        mgr = SessionManager()
        s1 = mgr.create_session()
        s2 = mgr.create_session()
        mgr.get_memory(s1).add(_msg("user", 1))
        assert len(mgr.get_memory(s1).get_all()) == 1
        assert len(mgr.get_memory(s2).get_all()) == 0

    def test_unknown_session_raises(self):
        mgr = SessionManager()
        with pytest.raises(KeyError):
            mgr.get_memory("not-exist")

    def test_close_session(self):
        mgr = SessionManager()
        sid = mgr.create_session()
        mgr.close_session(sid)
        assert sid not in mgr.active_sessions()
