"""FastAPI 应用工厂（day038）.

工厂模式而非模块级 ``app = FastAPI()`` 的原因：测试与生产可以各自创建
app 实例并注入不同的 LLM/Agent——测试注入 MockLLM 实现完全离线，
生产由默认工厂按 settings 构建真实客户端。``app.state`` 是注入的落脚点：
路由通过 ``request.app.state`` 读取被注入的对象（见 routes.py）。

端点统一用同步 ``def`` 声明：Agent 循环（多轮 LLM 调用 + 工具执行）是
阻塞式同步代码，声明为 ``async def`` 会阻塞事件循环；声明为 ``def``
FastAPI 会自动把它丢进线程池执行，异步服务器与同步业务代码由此兼容。
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from smart_research_agent.agent.fc_agent import FunctionCallingAgent
from smart_research_agent.api.schemas import ErrorResponse
from smart_research_agent.config import settings
from smart_research_agent.llm.base import BaseLLM
from smart_research_agent.llm.mock import MockLLM
from smart_research_agent.tools.calculator import CalculatorTool
from smart_research_agent.tools.registry import ToolRegistry
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 服务版本，随 /health 与 OpenAPI 信息返回
API_VERSION = "0.1.0"


def default_registry() -> ToolRegistry:
    """生产默认工具集：注册全部内置工具."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return registry


def default_llm() -> BaseLLM:
    """生产默认 LLM：有 API Key 用真实客户端，否则退回 MockLLM 占位.

    退回 Mock 而非直接报错，是为了让服务在"尚未配置密钥"的开发机上也能
    启动并自检（/health 可用）；真正调用对话接口时会得到 Mock 的兜底回复，
    日志中的告警会提示配置缺失。
    """
    if settings.openai_api_key:
        from smart_research_agent.llm.openai_compatible import OpenAICompatibleLLM

        return OpenAICompatibleLLM()
    logger.warning("未配置 OPENAI_API_KEY，LLM 退回 MockLLM 占位实现")
    return MockLLM(default="（未配置 LLM，这是 Mock 占位回复）")


def create_app(
    llm: BaseLLM | None = None,
    agent_factory: Callable[[], FunctionCallingAgent] | None = None,
) -> FastAPI:
    """创建 FastAPI 应用实例.

    参数（依赖注入入口，测试用）：
      - llm: /chat 使用的 LLM 客户端；None 时按 settings 构建默认实现；
      - agent_factory: /agent/run 的 Agent 工厂，**每个请求调用一次**——
        Agent 持有单次任务的执行轨迹（call_history），跨请求复用会串数据。
        None 时基于 llm（或默认 LLM）与默认工具注册表构建。
    """
    app = FastAPI(
        title="SmartResearch Agent API",
        version=API_VERSION,
        description="智研 AI 助手后端服务：对话与 Agent 任务接口",
    )

    resolved_llm = llm or default_llm()
    app.state.llm = resolved_llm
    if agent_factory is None:
        registry = default_registry()
        agent_factory = lambda: FunctionCallingAgent(resolved_llm, registry)  # noqa: E731
    app.state.agent_factory = agent_factory

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """把未捕获异常统一转为 ErrorResponse（HTTP 500）.

        不让框架默认的纯文本 500 泄漏到客户端：响应结构稳定（ErrorResponse），
        客户端可以按契约解析错误，而不是猜文本格式。
        """
        logger.exception("请求处理异常: %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                detail=str(exc) or "内部错误", error_type=type(exc).__name__
            ).model_dump(),
        )

    from smart_research_agent.api.routes import router

    app.include_router(router)
    return app


def main() -> None:
    """以 uvicorn 启动服务（python -m smart_research_agent.api.app）."""
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
