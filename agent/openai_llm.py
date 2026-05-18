# agent/openai_llm.py
"""OpenAI 格式 LLM 适配器

支持 OpenAI Chat Completion API 格式:
  POST /v1/chat/completions

通过环境变量配置:
  OPENAI_LLM_BASE_URL: API 基地址 (默认 https://api.openai.com/v1)
  OPENAI_LLM_API_KEY: API 密钥
  OPENAI_LLM_MODEL: 模型名 (默认 gpt-4o)

Docker 网络内直接通过服务名访问:
  - VLLM 服务: http://vllm:8000/v1/chat/completions
"""

import logging
import os

from livekit.plugins import openai as lk_openai

logger = logging.getLogger(__name__)


def create_llm() -> lk_openai.LLM:
    """工厂函数：从环境变量创建 OpenAI 格式 LLM 实例。

    支持的模型:
    - OpenAI GPT 系列 (gpt-4o, gpt-4-turbo, etc.)
    - VLLM 部署的模型 (Qwen3.6-35B, etc.)
    - 其他 OpenAI 兼容 API

    配置优先级:
    1. OPENAI_LLM_* 环境变量 (新版配置)
    2. LLM_* 环境变量 (旧版配置，兼容)
    """
    api_key = os.environ.get("OPENAI_LLM_API_KEY", "")
    if not api_key:
        api_key = os.environ.get("LLM_API_KEY", "")

    base_url = os.environ.get("OPENAI_LLM_BASE_URL", "")
    if not base_url:
        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")

    model = os.environ.get("OPENAI_LLM_MODEL", "")
    if not model:
        model = os.environ.get("LLM_MODEL", "gpt-4o")

    timeout = float(os.environ.get("OPENAI_LLM_TIMEOUT", "120"))

    logger.info(f"Creating OpenAI LLM: base_url={base_url}, model={model}")

    return lk_openai.LLM(
        api_key=api_key,
        base_url=base_url,
        model=model,
        http_client=None,  # 使用默认 Client
    )