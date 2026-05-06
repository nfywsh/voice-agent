# tts_service/Dockerfile
# TTS 微服务：纯 CPU，通过 DashScope API 调用 Qwen3-TTS
FROM python:3.11-slim

WORKDIR /app

# 配置 pip 清华镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install Python dependencies (不再需要 torch/transformers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY . .

# 环境变量 (运行时可覆盖)
ENV HOST=0.0.0.0
ENV PORT=8001
ENV DASHSCOPE_TTS_MODEL=qwen3-tts-flash-realtime

# Health check (API 模式启动快，不需要 120s start_period)
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD ["python", "main.py"]