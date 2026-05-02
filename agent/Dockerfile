# agent/Dockerfile
# Agent 服务：纯 CPU，通过 DashScope API 调用 LLM/ASR/TTS
FROM python:3.11-slim

WORKDIR /app

# 配置 pip 清华镜像源
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY . .

# Default environment variables (can be overridden at runtime)
ENV LIVEKIT_URL=ws://livekit:7880
ENV TTS_SERVICE_URL=http://tts-service:8001
ENV SINGING_SERVICE_URL=http://singing-service:8002
ENV DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

CMD ["python", "agent.py", "start"]