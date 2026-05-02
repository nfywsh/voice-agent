# agent/tts_adapter.py
"""Qwen3-TTS 适配器 — 对接 TTS 微服务（内部调用 DashScope API），支持流式合成与 24kHz→48kHz 重采样

TTS 微服务现在通过阿里云 DashScope API（模型 qwen3-tts-vd-2026-01-26）进行语音合成，
不再需要本地 GPU 模型。微服务对外接口保持不变，Agent 无感知。"""

import asyncio
import logging
import io
from typing import Optional

import aiohttp
import numpy as np
from scipy.signal import resample_poly

from livekit.agents import tts

logger = logging.getLogger(__name__)

# LiveKit 内部使用 48kHz，Qwen3-TTS 输出 24kHz
OUTPUT_SAMPLE_RATE = 48000
TTS_SAMPLE_RATE = 24000


def _resample_24k_to_48k(pcm_bytes: bytes, source_rate: int = TTS_SAMPLE_RATE,
                          target_rate: int = OUTPUT_SAMPLE_RATE) -> bytes:
    """将 16-bit PCM 音频从 source_rate 重采样到 target_rate。

    Qwen3-TTS 输出 24kHz/16bit/mono，推流到 LiveKit 需要 48kHz。
    """
    if source_rate == target_rate:
        return pcm_bytes

    # bytes → int16 numpy array
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # scipy 重采样：24k → 48k = 原始 * 2 / 1
    up = target_rate
    down = source_rate
    from math import gcd
    g = gcd(up, down)
    audio_resampled = resample_poly(audio, up // g, down // g)

    # float32 → int16 → bytes
    audio_int16 = np.clip(audio_resampled * 32767, -32768, 32767).astype(np.int16)
    return audio_int16.tobytes()


class QwenTTSAdapter(tts.TTS):
    """Qwen3-TTS 适配器，对接本地 TTS 微服务。

    继承 livekit.agents.tts.TTS，实现 synthesize 方法。
    流式输出通过 QwenTTSStream 实现。
    """

    def __init__(
        self,
        service_url: str = "http://localhost:8001",
        voice: str = "default",
        speed: float = 1.0,
        timeout: float = 10.0,
    ):
        super().__init__(
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=1,
        )
        self._service_url = service_url.rstrip("/")
        self._voice = voice
        self._speed = speed
        self._timeout = timeout

    async def synthesize(self, text: str) -> tts.AudioFrame:
        """非流式合成：请求 TTS 服务并返回完整音频帧。"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self._service_url}/tts/stream",
                    json={
                        "text": text,
                        "voice": self._voice,
                        "speed": self._speed,
                    },
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"TTS service error {resp.status}: {error_text}")
                        # 返回静音帧作为降级
                        return tts.AudioFrame(
                            data=b"\x00\x00" * 4800,  # 100ms 静音 @ 48kHz 16bit
                            sample_rate=OUTPUT_SAMPLE_RATE,
                            num_channels=1,
                        )

                    raw_audio = await resp.read()
                    resampled = _resample_24k_to_48k(raw_audio)
                    return tts.AudioFrame(
                        data=resampled,
                        sample_rate=OUTPUT_SAMPLE_RATE,
                        num_channels=1,
                    )
            except asyncio.TimeoutError:
                logger.error(f"TTS request timed out after {self._timeout}s")
                return tts.AudioFrame(
                    data=b"\x00\x00" * 4800,
                    sample_rate=OUTPUT_SAMPLE_RATE,
                    num_channels=1,
                )
            except Exception as e:
                logger.error(f"TTS synthesis error: {e}")
                return tts.AudioFrame(
                    data=b"\x00\x00" * 4800,
                    sample_rate=OUTPUT_SAMPLE_RATE,
                    num_channels=1,
                )

    def stream(self) -> "QwenTTSStream":
        """返回流式合成器。"""
        return QwenTTSStream(self)


class QwenTTSStream(tts.SynthesizeStream):
    """Qwen3-TTS 流式合成器。

    工作流程：
    1. Agent 通过 push_text() 推送文本
    2. 对 TTS 服务发起流式请求
    3. 逐块读取音频数据，重采样到 48kHz 后输出
    4. Agent 通过 on_output 回调接收音频帧
    """

    def __init__(self, adapter: QwenTTSAdapter):
        super().__init__()
        self._adapter = adapter
        self._session: Optional[aiohttp.ClientSession] = None
        self._response: Optional[aiohttp.ClientResponse] = None
        self._text_queue: asyncio.Queue = asyncio.Queue()

    async def push_text(self, text: str) -> None:
        """推送文本到队列。"""
        await self._text_queue.put(text)

    async def _run(self) -> None:
        """主循环：从队列取文本，请求 TTS，输出音频帧。"""
        self._session = aiohttp.ClientSession()
        try:
            while True:
                text = await self._text_queue.get()
                if text is None:  # 哨兵值，结束流
                    break

                if not text.strip():
                    continue

                try:
                    async with self._session.post(
                        f"{self._adapter._service_url}/tts/stream",
                        json={
                            "text": text,
                            "voice": self._adapter._voice,
                            "speed": self._adapter._speed,
                        },
                        timeout=aiohttp.ClientTimeout(total=self._adapter._timeout),
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            logger.error(f"TTS stream error {resp.status}: {error_text}")
                            continue

                        # 逐块读取音频
                        buffer = b""
                        async for chunk in resp.content.iter_chunked(4096):
                            buffer += chunk
                            # 累积到足够一帧再输出（约 20ms @ 48kHz = 1920 samples = 3840 bytes）
                            if len(buffer) >= 3840:
                                resampled = _resample_24k_to_48k(buffer)
                                frame = tts.AudioFrame(
                                    data=resampled,
                                    sample_rate=OUTPUT_SAMPLE_RATE,
                                    num_channels=1,
                                )
                                self._output_ch.send_nowait(frame)
                                buffer = b""

                        # 输出剩余数据
                        if buffer:
                            resampled = _resample_24k_to_48k(buffer)
                            frame = tts.AudioFrame(
                                data=resampled,
                                sample_rate=OUTPUT_SAMPLE_RATE,
                                num_channels=1,
                            )
                            self._output_ch.send_nowait(frame)

                except asyncio.TimeoutError:
                    logger.error(f"TTS stream timed out after {self._adapter._timeout}s")
                except Exception as e:
                    logger.error(f"TTS stream error: {e}")

        finally:
            if self._session:
                await self._session.close()
            self._session = None

    async def aclose(self) -> None:
        """关闭流。"""
        await self._text_queue.put(None)  # 发送哨兵值结束主循环