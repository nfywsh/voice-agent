"""Configuration module for voice agent."""

import json
import os

# LLM template parameters (thinking mode, etc.), thinking disabled by default
LLM_CHAT_TEMPLATE_KWARGS = json.loads(
    os.environ.get("LLM_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}')
)

# Chat history retention turns
CHAT_HISTORY_MAX_TURNS = int(os.environ.get("CHAT_HISTORY_MAX_TURNS", "10"))

# VAD configuration
VAD_THRESHOLD = float(os.environ.get("VAD_THRESHOLD", "0.5"))
VAD_MIN_SPEECH = float(os.environ.get("VAD_MIN_SPEECH", "0.2"))
VAD_MIN_SILENCE = float(os.environ.get("VAD_MIN_SILENCE", "0.3"))

# TTS chunk configuration
TTS_FIRST_CHUNK_MIN = int(os.environ.get("TTS_FIRST_CHUNK_MIN", "30"))
TTS_MAX_CHUNK = int(os.environ.get("TTS_MAX_CHUNK", "300"))
TTS_CHUNK_WAIT_SEC = float(os.environ.get("TTS_CHUNK_WAIT_SEC", "5.0"))
TTS_MAX_CONCURRENT = int(os.environ.get("TTS_MAX_CONCURRENT", "3"))
TTS_TIMEOUT = float(os.environ.get("TTS_TIMEOUT", "30"))

# Singing configuration
SING_AGENT_URL = os.environ.get("SING_AGENT_URL", "http://localhost:8080")
SINGING_TIMEOUT = float(os.environ.get("SINGING_TIMEOUT", "30"))

# LLM configuration
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "5"))
QWEN3_TTS_TIMEOUT = float(os.environ.get("QWEN3_TTS_TIMEOUT", "120"))