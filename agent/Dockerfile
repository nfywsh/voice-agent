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

# Patch audio_recognition.py to add debug logging
# VAD event debug
RUN sed -i 's/async def _on_vad_event(self, ev: vad.VADEvent) -> None:/async def _on_vad_event(self, ev: vad.VADEvent) -> None:\n        import sys\n        print(f"ARDebug vad_event: {ev.type}", flush=True, file=sys.stderr)/' /usr/local/lib/python3.11/site-packages/livekit/agents/voice/audio_recognition.py

# push_audio debug - print every 100 frames going to STT pipeline
RUN sed -i 's/if not skip_stt and self._stt_pipeline is not None:/if not skip_stt and self._stt_pipeline is not None:\n            import sys\n            if not hasattr(self, '"'"'_push_audio_count'"'"'):\n                self._push_audio_count = 0\n            self._push_audio_count += 1\n            if self._push_audio_count % 100 == 0:\n                print(f"ARDebug push_audio: stt_pipeline_id={id(self._stt_pipeline)}, frames={self._push_audio_count}", flush=True, file=sys.stderr)/' /usr/local/lib/python3.11/site-packages/livekit/agents/voice/audio_recognition.py

# Copy agent code
COPY . .

# Default environment variables (can be overridden at runtime)
ENV LIVEKIT_URL=ws://livekit:7880
ENV TTS_SERVICE_URL=http://tts-service:8001
ENV SINGING_SERVICE_URL=http://singing-service:8002
ENV DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

CMD ["python", "agent.py", "start"]