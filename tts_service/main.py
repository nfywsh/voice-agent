# tts_service/main.py
"""Qwen3-TTS 流式语音合成微服务 — 使用 DashScope QwenTtsRealtime SDK

SDK 事件协议：
  session.created         → 连接建立
  session.updated         → session.update 确认
  input_text_buffer.committed → 文本已提交
  response.created        → 开始生成响应
  response.audio.delta    → 音频数据块（base64 编码）
  response.done           → 本次响应完成
  session.finished        → 会话结束

接口规格：
  POST /tts/stream — 流式语音合成，返回 chunked PCM 24kHz
  GET  /health     — 健康检查
  POST /tts/reload — 重新初始化
"""

import asyncio
import base64
import logging
import os
import queue
import threading
from typing import AsyncGenerator, Optional

import dashscope
from dashscope.audio.qwen_tts_realtime import (
    AudioFormat,
    QwenTtsRealtime,
    QwenTtsRealtimeCallback,
)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")
TTS_MODEL = os.environ.get("DASHSCOPE_TTS_MODEL", "qwen3-tts-flash-realtime")
DEFAULT_VOICE = os.environ.get("DASHSCOPE_TTS_VOICE", "Cherry")


class TTSCallback(QwenTtsRealtimeCallback):
    """TTS 回调：将音频块通过 Queue 传递给调用方。"""

    def __init__(self, audio_queue: queue.Queue, finish_event: threading.Event):
        super().__init__()
        self.audio_queue = audio_queue
        self.finish_event = finish_event
        self.error: Optional[Exception] = None

    def on_open(self):
        logger.info("[TTSCallback] connection opened")

    def on_close(self, close_status_code, close_msg):
        logger.info(f"[TTSCallback] connection closed: {close_status_code} {close_msg}")
        self.finish_event.set()

    def on_event(self, response):
        try:
            r = response if isinstance(response, dict) else {}
            t = r.get("type", "")
            if t == "response.audio.delta":
                pcm = base64.b64decode(r["delta"])
                self.audio_queue.put(pcm)
            elif t in ("response.done", "session.finished"):
                self.audio_queue.put(b"")  # 结束标记
                self.finish_event.set()
            elif t == "error":
                err_msg = r.get("error", "unknown")
                self.error = Exception(f"TTS error: {err_msg}")
                self.audio_queue.put(b"")
                self.finish_event.set()
        except Exception as e:
            logger.error(f"[TTSCallback] parse error: {e}")
            self.error = e
            self.audio_queue.put(b"")


def synthesize_sync(text: str, voice: str, speed: float) -> tuple[queue.Queue, threading.Event, QwenTtsRealtime]:
    """同步合成：在子线程中调用 SDK，音频块通过 Queue 返回。

    Returns:
        (audio_queue, finish_event, tts_instance) — 调用方用 finish_event 等待完成
    """
    audio_queue: queue.Queue = queue.Queue()
    finish_event = threading.Event()

    callback = TTSCallback(audio_queue, finish_event)
    tts = QwenTtsRealtime(
        model=TTS_MODEL,
        callback=callback,
        url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
    )

    try:
        tts.connect()
        tts.update_session(
            voice=voice,
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            speed_ratio=speed,
            mode="server_commit",
        )
        # 分句追加文本以获得更好的流式效果
        for segment in text.split("。"):
            if segment.strip():
                tts.append_text(segment + "。")
        tts.commit()
    except Exception as e:
        logger.error(f"[synthesize_sync] TTS error: {e}")
        callback.error = e
        audio_queue.put(b"")
        finish_event.set()

    return audio_queue, finish_event, tts


async def synthesize_async(
    text: str,
    voice: str = DEFAULT_VOICE,
    speed: float = 1.0,
) -> AsyncGenerator[bytes, None]:
    """异步包装器：在线程池运行同步 SDK 调用。"""
    loop = asyncio.get_event_loop()
    audio_queue, finish_event, tts = await loop.run_in_executor(
        None, synthesize_sync, text, voice, speed
    )

    try:
        while True:
            # 等待音频块可用（超时检测finish_event，防止无限阻塞）
            try:
                chunk = await loop.run_in_executor(None, audio_queue.get, True, 0.5)
            except queue.Empty:
                if finish_event.is_set():
                    break
                continue

            if not chunk:  # 结束标记
                break
            yield chunk

            if finish_event.is_set() and audio_queue.empty():
                break
    except Exception as e:
        logger.error(f"[synthesize_async] error: {e}")
    finally:
        try:
            await loop.run_in_executor(None, tts.finish)
        except Exception:
            pass


# ============================================================
# FastAPI 应用
# ============================================================

app = FastAPI(
    title="Qwen3-TTS Service",
    description="流式语音合成微服务 — DashScope QwenTtsRealtime SDK",
    version="4.0.0",
)


@app.on_event("startup")
async def startup():
    if not dashscope.api_key:
        logger.error("DASHSCOPE_API_KEY not set!")
    else:
        logger.info(f"TTS service ready (model={TTS_MODEL}, voice={DEFAULT_VOICE})")


# ============================================================
# 请求/响应模型
# ============================================================

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="合成文本")
    voice: str = Field(default=DEFAULT_VOICE, description="音色标识")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="语速倍率")


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
    """流式 TTS 端点。使用 QwenTtsRealtime SDK 流式输出 PCM。"""
    if not dashscope.api_key:
        raise HTTPException(
            status_code=503,
            detail="TTS service not initialized (missing DASHSCOPE_API_KEY)",
        )

    if await req.is_disconnected():
        return

    text = request.text[:500]

    async def audio_generator():
        try:
            async for chunk in synthesize_async(
                text=text,
                voice=request.voice,
                speed=request.speed,
            ):
                if await req.is_disconnected():
                    break
                yield chunk
        except asyncio.CancelledError:
            logger.info("[tts] Stream cancelled")
        except Exception as e:
            logger.error(f"[tts] Stream error: {e}")
            # 降级：返回静音 100ms @ 24kHz
            import numpy as np
            silence = np.zeros(2400, dtype=np.int16)
            yield silence.tobytes()

    return StreamingResponse(
        audio_generator(),
        media_type="audio/x-raw",
        headers={
            "X-Sample-Rate": "24000",
            "X-Voice": request.voice,
            "X-Model": TTS_MODEL,
            "Cache-Control": "no-cache",
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if dashscope.api_key else "degraded",
        model_loaded=bool(dashscope.api_key),
        device="dashscope-sdk",
        sample_rate=24000,
        api_mode="websocket",
    )


@app.post("/tts/reload")
async def reload_model():
    if not dashscope.api_key:
        raise HTTPException(status_code=500, detail="DASHSCOPE_API_KEY not set")
    return {"status": "ok", "model": TTS_MODEL}


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