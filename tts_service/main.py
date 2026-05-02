# tts_service/main.py
"""Qwen3-TTS 流式语音合成微服务 [已完成 - DashScope API 改造]

通过阿里云 DashScope API（模型 qwen3-tts-vd-2026-01-26）进行语音合成，
不再需要本地 GPU 和模型文件。

API 调用方式：
  POST https://dashscope.aliyuncs.com/compatible-mode/v1/audio/speech
  使用 OpenAI 兼容接口格式，流式返回 PCM/WAV 音频数据。

接口规格：
  POST /tts/stream — 流式语音合成，返回 chunked 音频
  GET  /health     — 健康检查
  POST /tts/reload — 无操作（保留接口兼容性）
"""

import asyncio
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# DashScope TTS API 配置
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
DASHSCOPE_TTS_MODEL = os.environ.get("DASHSCOPE_TTS_MODEL", "qwen3-tts-vd-2026-01-26")


class DashScopeTTSClient:
    """DashScope TTS API 客户端。

    使用 OpenAI 兼容接口调用 DashScope TTS。
    接口：POST {base_url}/audio/speech
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def synthesize_stream(
        self,
        text: str,
        voice: str = "Chelsie",
        speed: float = 1.0,
        response_format: str = "pcm",
    ) -> aiohttp.ClientResponse:
        """调用 DashScope TTS API，返回流式响应。

        Args:
            text: 合成文本
            voice: 音色（DashScope TTS 支持的音色名）
            speed: 语速倍率
            response_format: 音频格式 pcm / wav / mp3

        Returns:
            aiohttp.ClientResponse: 流式音频响应
        """
        session = await self._get_session()

        url = f"{self._base_url}/audio/speech"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": response_format,
        }

        resp = await session.post(
            url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30.0),
        )

        if resp.status != 200:
            error_text = await resp.text()
            logger.error(f"DashScope TTS API error {resp.status}: {error_text}")
            raise Exception(f"DashScope TTS API error {resp.status}: {error_text}")

        return resp

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


# 全局 TTS 客户端
tts_client: Optional[DashScopeTTSClient] = None


# ============================================================
# FastAPI 应用
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化 TTS 客户端"""
    global tts_client

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        logger.error("DASHSCOPE_API_KEY not set! TTS service will not work.")
    else:
        tts_client = DashScopeTTSClient(
            api_key=api_key,
            base_url=DASHSCOPE_BASE_URL,
            model=DASHSCOPE_TTS_MODEL,
        )
        logger.info(f"TTS service ready (model={DASHSCOPE_TTS_MODEL})")

    yield

    if tts_client:
        await tts_client.close()


app = FastAPI(
    title="Qwen3-TTS Service (DashScope API)",
    description="流式语音合成微服务 — 通过 DashScope API 调用 Qwen3-TTS",
    version="3.0.0",
    lifespan=lifespan,
)


# ============================================================
# 请求/响应模型
# ============================================================

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="合成文本")
    voice: str = Field(default="Chelsie", description="音色标识（DashScope TTS 支持的音色）")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="语速倍率")
    sample_rate: Optional[int] = Field(default=None, description="输出采样率（兼容参数，API 原生 24kHz）")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    sample_rate: int
    api_mode: str


# ============================================================
# API 端点
# ============================================================

@app.post("/tts/stream")
async def stream_tts(request: TTSRequest, req: Request):
    """流式 TTS 端点。调用 DashScope API，返回流式音频数据。"""
    if not tts_client:
        raise HTTPException(status_code=503, detail="TTS service not initialized (missing DASHSCOPE_API_KEY)")

    # 检查客户端断连
    if await req.is_disconnected():
        return

    # 截断过长文本
    text = request.text[:500]
    was_truncated = len(request.text) > 500

    async def audio_generator():
        """异步音频生成器：流式从 DashScope 获取音频数据。"""
        try:
            resp = await tts_client.synthesize_stream(
                text=text,
                voice=request.voice,
                speed=request.speed,
                response_format="pcm",
            )

            # 流式读取音频数据
            async for chunk in resp.content.iter_chunked(4096):
                if await req.is_disconnected():
                    break
                yield chunk

        except asyncio.CancelledError:
            logger.info("[tts] Stream cancelled")
        except Exception as e:
            logger.error(f"[tts] Stream error: {e}")
            # 返回静音作为降级
            silence = np.zeros(24000 // 10, dtype=np.int16)  # 100ms @ 24kHz
            yield silence.tobytes()

    headers = {
        "X-Sample-Rate": "24000",
        "X-Voice": request.voice,
        "X-Model": DASHSCOPE_TTS_MODEL,
        "Cache-Control": "no-cache",
    }
    if was_truncated:
        headers["X-Truncated"] = "true"

    return StreamingResponse(
        audio_generator(),
        media_type="audio/x-raw",
        headers=headers,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查端点。"""
    return HealthResponse(
        status="ok" if tts_client else "degraded",
        model_loaded=tts_client is not None,
        device="dashscope-api",
        sample_rate=24000,
        api_mode="dashscope",
    )


@app.post("/tts/reload")
async def reload_model():
    """重新初始化 TTS 客户端（保留接口兼容性）。"""
    global tts_client
    if tts_client:
        await tts_client.close()

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if api_key:
        tts_client = DashScopeTTSClient(
            api_key=api_key,
            base_url=DASHSCOPE_BASE_URL,
            model=DASHSCOPE_TTS_MODEL,
        )
        return {"status": "reloaded", "model": DASHSCOPE_TTS_MODEL}
    else:
        raise HTTPException(status_code=500, detail="DASHSCOPE_API_KEY not set")


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )
