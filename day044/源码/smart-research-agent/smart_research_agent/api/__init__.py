"""FastAPI 服务层（day038）：把 SmartResearch Agent 暴露为 HTTP API."""

from smart_research_agent.api.app import create_app

__all__ = ["create_app"]
