# singing_service/main.py
"""歌声合成微服务 — 纯 Mock 模式 [已完成 - 简化改造]

VibeVoice 模型暂无线上 API 可用，当前仅保留 Mock 模式返回测试音频。
后续接入线上歌声合成 API 时，只需替换 _generate_singing_audio() 方法。

接口规格：
  POST /sing    — 歌声合成（当前返回 Mock 正弦波音频）
  GET  /health  — 健康检查
  DELETE /cache — 清除缓存
"""

import asyncio
import hashlib
import logging
import os
import struct
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000


# ============================================================
# 歌曲缓存（LRU，减少重复生成）
# ============================================================

class LRUCache:
    """简单的 LRU 缓存。"""

    def __init__(self, max_size: int = 20, max_audio_seconds: int = 180):
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._max_size = max_size
        self._max_audio_seconds = max_audio_seconds

    def _make_key(self, lyrics: str, speaker_id: str) -> str:
        content = f"{speaker_id}:{lyrics}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, lyrics: str, speaker_id: str) -> Optional[bytes]:
        key = self._make_key(lyrics, speaker_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, lyrics: str, speaker_id: str, audio: bytes) -> None:
        max_bytes = self._max_audio_seconds * SAMPLE_RATE * 2
        if len(audio) > max_bytes:
            return
        key = self._make_key(lyrics, speaker_id)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
        self._cache[key] = audio


singing_cache = LRUCache(max_size=20)


# ============================================================
# Mock 歌声生成
# ============================================================

def _generate_mock_singing(sr: int = SAMPLE_RATE, duration: float = 5.0) -> bytes:
    """生成简单的正弦波测试音频（C大调音阶）。"""
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    # C大调音阶：C4 → D4 → E4 → F4 → G4 → A4 → B4 → C5
    freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    samples_per_note = len(t) // len(freqs)
    audio = np.zeros(len(t), dtype=np.float32)
    for i, freq in enumerate(freqs):
        start = i * samples_per_note
        end = min(start + samples_per_note, len(t))
        note_t = t[start:end] - t[start]
        audio[start:end] = 0.3 * np.sin(2 * np.pi * freq * note_t)

    # 淡入淡出
    fade = int(0.05 * sr)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)

    return (audio * 32767).astype(np.int16).tobytes()


# ============================================================
# FastAPI 应用
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：Mock 模式无需加载模型。"""
    logger.info("Singing service ready (MOCK mode — returning sine wave test audio)")
    yield


app = FastAPI(
    title="Singing Service (Mock)",
    description="歌声合成微服务 — 当前为 Mock 模式",
    version="3.0.0",
    lifespan=lifespan,
)


# ============================================================
# 请求模型
# ============================================================

class SingingRequest(BaseModel):
    lyrics: str = Field(..., min_length=1, description="歌词文本，格式为 'Speaker 1: 歌词'")
    title: str = Field(default="即兴歌曲", description="歌曲标题")
    style: str = Field(default="流行", description="歌曲风格")
    speaker_id: str = Field(default="Speaker 1", description="说话人标识")
    sample_rate: Optional[int] = Field(default=None, description="输出采样率，默认 24000")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    sample_rate: int
    mock_mode: bool


# ============================================================
# API 端点
# ============================================================

@app.post("/sing")
async def sing_song(request: SingingRequest, req: Request):
    """歌声合成端点。当前为 Mock 模式，返回正弦波测试音频。"""
    output_sr = request.sample_rate or SAMPLE_RATE

    async def audio_generator():
        """异步音频生成器。"""
        try:
            # 检查缓存
            cached = singing_cache.get(request.lyrics, request.speaker_id)
            if cached:
                logger.info(f"[sing] Cache hit: {request.title}")
                chunk_size = 8192
                for i in range(0, len(cached), chunk_size):
                    if await req.is_disconnected():
                        return
                    yield cached[i:i + chunk_size]
                    await asyncio.sleep(0.01)
                return

            # Mock 模式：生成测试音频
            logger.info(f"[sing] Generating mock audio: {request.title} (style={request.style})")
            audio_bytes = _generate_mock_singing(sr=output_sr, duration=5.0)

            # 尝试缓存
            try:
                singing_cache.put(request.lyrics, request.speaker_id, audio_bytes)
            except Exception:
                pass

            # 分块流式返回
            chunk_size = 8192
            for i in range(0, len(audio_bytes), chunk_size):
                if await req.is_disconnected():
                    return
                yield audio_bytes[i:i + chunk_size]
                await asyncio.sleep(0.01)

            logger.info(f"[sing] Finished: {request.title}")

        except asyncio.CancelledError:
            logger.info(f"[sing] Cancelled: {request.title}")
        except Exception as e:
            logger.error(f"[sing] Error: {e}")
            # 返回静音作为降级
            silence = np.zeros(output_sr // 10, dtype=np.int16)
            yield silence.tobytes()

    headers = {
        "X-Sample-Rate": str(output_sr),
        "X-Title": request.title,
        "X-Style": request.style,
        "X-Mock": "true",
        "Cache-Control": "no-cache",
    }

    return StreamingResponse(
        audio_generator(),
        media_type="audio/x-raw",
        headers=headers,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查端点。"""
    return HealthResponse(
        status="ok",
        model_loaded=True,  # Mock 模式视为已加载
        device="mock",
        sample_rate=SAMPLE_RATE,
        mock_mode=True,
    )


@app.delete("/cache")
async def clear_cache():
    """清除歌曲缓存。"""
    global singing_cache
    singing_cache = LRUCache(max_size=20)
    return {"status": "cache cleared"}


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8002"))

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )
