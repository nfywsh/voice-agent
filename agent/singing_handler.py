# agent/singing_handler.py
"""歌声合成处理器 — 对接 Singing Service 微服务，支持流式音频输出与 24kHz→48kHz 重采样"""

import asyncio
import logging
from typing import AsyncGenerator, Optional

import aiohttp
import numpy as np
from scipy.signal import resample_poly
from math import gcd

logger = logging.getLogger(__name__)

OUTPUT_SAMPLE_RATE = 48000
SINGING_SAMPLE_RATE = 24000


def _resample_24k_to_48k(pcm_bytes: bytes, source_rate: int = SINGING_SAMPLE_RATE,
                          target_rate: int = OUTPUT_SAMPLE_RATE) -> bytes:
    """将 16-bit PCM 音频从 source_rate 重采样到 target_rate。"""
    if source_rate == target_rate:
        return pcm_bytes
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    audio_resampled = resample_poly(audio, target_rate // gcd(target_rate, source_rate),
                                     source_rate // gcd(target_rate, source_rate))
    audio_int16 = np.clip(audio_resampled * 32767, -32768, 32767).astype(np.int16)
    return audio_int16.tobytes()


class SingingHandler:
    """歌声合成处理器，对接 Singing Service 微服务。

    职责：
    1. 调用 POST /sing 接口，流式获取歌声音频
    2. 将 24kHz PCM 重采样到 48kHz
    3. 返回可在 LiveKit 推流的音频帧
    """

    def __init__(
        self,
        service_url: str = "http://localhost:8002",
        timeout: float = 30.0,
        mock_mode: bool = False,
    ):
        self._service_url = service_url.rstrip("/")
        self._timeout = timeout
        self._mock_mode = mock_mode

    async def sing_stream(
        self,
        lyrics: str,
        title: str = "即兴歌曲",
        style: str = "流行",
        speaker_id: str = "Speaker 1",
    ) -> AsyncGenerator[bytes, None]:
        """流式获取歌声音频（已重采样到 48kHz 16bit mono PCM）。

        Args:
            lyrics: 歌词文本，每行格式应为 "Speaker 1: 歌词内容"
            title: 歌曲标题
            style: 歌曲风格
            speaker_id: 说话人标识

        Yields:
            bytes: 48kHz 16bit mono PCM 音频块
        """
        if self._mock_mode:
            async for chunk in self._mock_sing(title):
                yield chunk
            return

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self._service_url}/sing",
                    json={
                        "lyrics": lyrics,
                        "title": title,
                        "style": style,
                        "speaker_id": speaker_id,
                    },
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"Singing service error {resp.status}: {error_text}")
                        return

                    buffer = b""
                    async for chunk in resp.content.iter_chunked(4096):
                        buffer += chunk
                        # 累积到足够一帧再输出（约 20ms @ 48kHz = 3840 bytes）
                        if len(buffer) >= 3840:
                            resampled = _resample_24k_to_48k(buffer)
                            yield resampled
                            buffer = b""

                    # 输出剩余数据
                    if buffer:
                        resampled = _resample_24k_to_48k(buffer)
                        yield resampled

            except asyncio.TimeoutError:
                logger.error(f"Singing request timed out after {self._timeout}s")
            except Exception as e:
                logger.error(f"Singing stream error: {e}")

    async def _mock_sing(self, title: str) -> AsyncGenerator[bytes, None]:
        """Mock 模式：生成 1 秒静音 + 简单正弦波测试音。用于开发调试。"""
        import struct

        # 生成 1 秒的 440Hz 正弦波 @ 48kHz 16bit mono
        duration = 1.0
        freq = 440.0
        sample_rate = OUTPUT_SAMPLE_RATE
        num_samples = int(duration * sample_rate)

        samples = []
        for i in range(num_samples):
            t = i / sample_rate
            value = int(8000 * (0.5 + 0.5 * np.sin(2 * np.pi * freq * t)))  # 简单旋律感
            samples.append(struct.pack("<h", max(-32768, min(32767, value))))

        chunk_size = 3840  # 20ms
        data = b"".join(samples)
        for i in range(0, len(data), chunk_size):
            yield data[i:i + chunk_size]
            await asyncio.sleep(0.02)  # 模拟流式延迟

    async def check_health(self) -> bool:
        """检查歌声服务是否健康。"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self._service_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5.0),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False