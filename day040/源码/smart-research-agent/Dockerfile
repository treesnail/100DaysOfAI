# 智研 AI 助手 - MCP Server 镜像
# 基于官方 Python 3.13 slim 镜像：体积小、包含运行所需的最小系统依赖
FROM python:3.13-slim

# 设置工作目录，容器内的所有后续操作都在此目录下进行
WORKDIR /app

# slim 镜像不带 curl，而 HEALTHCHECK 需要它来探测 /health
# --no-install-recommends 避免安装建议依赖；清理 apt 缓存减小层体积
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 镜像分层缓存：
# 只要 requirements.txt 不变，依赖层就不需要重新构建
COPY requirements.txt ./

# 安装运行时依赖
# --no-cache-dir 减小镜像体积；--upgrade pip 保证 pip 版本足够新
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 再复制项目源码（源码变化频繁，放在依赖层之后，避免源码修改导致依赖重装）
COPY smart_research_agent/ ./smart_research_agent/
COPY pyproject.toml ./

# 以 editable 模式安装本项目
RUN pip install --no-cache-dir -e .

# 环境变量注入（十二要素：配置存环境，而非写死在代码里）
# 容器必须监听 0.0.0.0，否则宿主机端口转发进不来
# 这些值可在 docker run -e 或 compose 的 environment 中覆盖
ENV MCP_TRANSPORT=sse \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

# 声明 SSE 服务端口（文档性声明，实际映射由 -p / compose ports 决定）
EXPOSE 8000

# 非 root 用户运行，降低安全风险
# logs/ 由应用在运行时创建，/app 需要对运行用户可写
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# 健康检查：每 30s 请求一次 /health，连续 3 次失败判定 unhealthy
# --start-period 给服务启动留出宽限期，期间失败不计入
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 默认以 SSE 模式启动 MCP Server，监听 8000 端口
CMD ["python", "-m", "smart_research_agent.mcp_server.server", "--transport", "sse"]
