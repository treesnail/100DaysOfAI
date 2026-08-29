# 智研 AI 助手 - 基础镜像
# 基于官方 Python 3.11 slim 镜像，体积小且包含生产运行所需的最小依赖
FROM python:3.11-slim

# 设置工作目录，容器内的所有后续操作都在此目录下进行
WORKDIR /app

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

# 以 editable 模式安装本项目，使 smart-research-agent 命令可用
RUN pip install --no-cache-dir -e .

# 非 root 用户运行，降低安全风险
RUN useradd -m appuser
USER appuser

# 默认启动命令：运行项目入口
CMD ["python", "-m", "smart_research_agent.main"]
