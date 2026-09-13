"""会话管理：多会话的记忆隔离."""

from __future__ import annotations

import uuid

from smart_research_agent.memory.short_term import ShortTermMemory


class SessionManager:
    """按 session_id 隔离各会话的短期记忆."""

    def __init__(self, max_messages: int = 20):
        self._max_messages = max_messages
        self._sessions: dict[str, ShortTermMemory] = {}

    def create_session(self) -> str:
        session_id = uuid.uuid4().hex[:12]
        self._sessions[session_id] = ShortTermMemory(max_messages=self._max_messages)
        return session_id

    def get_memory(self, session_id: str) -> ShortTermMemory:
        if session_id not in self._sessions:
            raise KeyError(f"会话不存在: {session_id}")
        return self._sessions[session_id]

    def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def active_sessions(self) -> list[str]:
        return list(self._sessions.keys())
