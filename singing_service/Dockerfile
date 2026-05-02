# singing_service/Dockerfile
# 歌声微服务：纯 CPU Mock 模式，无需 GPU
FROM python:3.11-slim

WORKDIR /app

# 配置 pip 清华镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install Python dependencies (Mock 模式不需要 torch/transformers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY . .

# 环境变量
ENV HOST=0.0.0.0
ENV PORT=8002

# Mock 模式启动快
HEALTHCHECK --interval=15s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')" || exit 1

CMD ["python", "main.py"]