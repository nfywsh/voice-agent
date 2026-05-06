# agent/tts_adapter.py
"""Qwen3-TTS 适配器 — 对接 TTS 微服务（内部调用 DashScope API），支持流式合成与 24kHz→48kHz 重采样

TTS 微服务现在通过阿里云 DashScope API（模型 qwen3-tts-vd-2026-01-26）进行语音合成，
不再需要本地 GPU 模型。微服务对外接口保持不变，Agent 无感知。

SDK v1.5.7 适配：
- synthesize() 返回 ChunkedStream（内部类 QwenTTSChunkedStream）
- stream() 返回 SynthesizeStream（QwenTTSStream）
- _run(output_emitter: AudioEmitter) 接收 AudioEmitter 参数
"""

import asyncio
import logging
import uuid
from typing import Optional

import aiohttp
import numpy as np
from scipy.signal import resample_poly

from livekit.agents import tts
from livekit.agents.tts.tts import AudioEmitter
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

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

    继承 livekit.agents.tts.TTS，实现 synthesize 和 stream 方法。
    """

    def __init__(
        self,
        service_url: str = "http://localhost:8001",
        voice: str = "default",
        speed: float = 1.0,
        timeout: float = 10.0,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=1,
        )
        self._service_url = service_url.rstrip("/")
        self._voice = voice
        self._speed = speed
        self._timeout = timeout

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "QwenTTSChunkedStream":
        """非流式合成：请求 TTS 服务并返回 ChunkedStream。"""
        return QwenTTSChunkedStream(self, text, conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "QwenTTSStream":
        """返回流式合成器。"""
        return QwenTTSStream(self, conn_options=conn_options)


class QwenTTSChunkedStream(tts.ChunkedStream):
    """非流式合成的 ChunkedStream 实现。

    请求 TTS 服务获取完整音频，通过 AudioEmitter 输出。
    """

    def __init__(
        self,
        adapter: QwenTTSAdapter,
        text: str,
        conn_options: APIConnectOptions,
    ):
        super().__init__(tts=adapter, input_text=text, conn_options=conn_options)
        self._adapter = adapter

    async def _run(self, output_emitter: AudioEmitter) -> None:
        """请求 TTS 服务并通过 AudioEmitter 输出音频帧。"""
        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._adapter._service_url}/tts/stream",
                    json={
                        "text": self._input_text,
                        "voice": self._adapter._voice,
                        "speed": self._adapter._speed,
                    },
                    timeout=aiohttp.ClientTimeout(total=self._adapter._timeout),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"TTS service error {resp.status}: {error_text}")
                        return

                    # 逐块读取音频，重采样后输出
                    buffer = b""
                    async for chunk in resp.content.iter_chunked(4096):
                        buffer += chunk
                        # 累积到足够一帧再输出（约 20ms @ 48kHz = 1920 samples = 3840 bytes）
                        if len(buffer) >= 3840:
                            resampled = _resample_24k_to_48k(buffer)
                            output_emitter.push(resampled)
                            buffer = b""

                    # 输出剩余数据
                    if buffer:
                        resampled = _resample_24k_to_48k(buffer)
                        output_emitter.push(resampled)

            output_emitter.flush()

        except asyncio.TimeoutError:
            logger.error(f"TTS request timed out after {self._adapter._timeout}s")
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")


class QwenTTSStream(tts.SynthesizeStream):
    """Qwen3-TTS 流式合成器。

    工作流程：
    1. Agent 通过 push_text() 推送文本（由 base class 管理 _input_ch）
    2. _run() 从 _input_ch 读取文本，累积后一次性请求 TTS 服务
    3. 流式读取音频数据，重采样到 48kHz 后通过 AudioEmitter 输出
    4. 使用单一 segment，避免 segment 计数不匹配
    """

    def __init__(
        self,
        adapter: QwenTTSAdapter,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        super().__init__(tts=adapter, conn_options=conn_options)
        self._adapter = adapter

    async def _run(self, output_emitter: AudioEmitter) -> None:
        """主循环：从 _input_ch 取文本，累积后一次性 TTS，通过 AudioEmitter 输出。"""
        logger.info(f"[QwenTTSStream._run] started, waiting for text input...")
        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",  # LiveKit 会自动处理重采样
            stream=True,
        )

        # 累积所有文本，一次性合成（避免每个 token 建一个 segment）
        full_text = ""
        async for item in self._input_ch:
            if isinstance(item, str) and item.strip():
                full_text += item

        logger.info(f"[QwenTTSStream._run] accumulated text length: {len(full_text)}, text: {full_text[:100]}")

        if not full_text.strip():
            logger.info("[QwenTTSStream._run] empty text, skipping TTS")
            return

        segment_id = str(uuid.uuid4())
        output_emitter.start_segment(segment_id=segment_id)
        logger.info(f"[QwenTTSStream._run] calling TTS service: {self._adapter._service_url}")

        self._pcm_frames_received = 0
        self._pcm_bytes_received = 0

        session = aiohttp.ClientSession()
        try:
            async with session.post(
                f"{self._adapter._service_url}/tts/stream",
                json={
                    "text": full_text,
                    "voice": self._adapter._voice,
                    "speed": self._adapter._speed,
                },
                timeout=aiohttp.ClientTimeout(total=self._adapter._timeout),
            ) as resp:
                logger.info(f"[QwenTTSStream._run] TTS response status: {resp.status}")
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"[QwenTTSStream._run] TTS stream error {resp.status}: {error_text}")
                    return

                buffer = b""
                chunk_count = 0
                async for chunk in resp.content.iter_chunked(4096):
                    buffer += chunk
                    if len(buffer) >= 3840:
                        resampled = _resample_24k_to_48k(buffer)
                        logger.info(f"[QwenTTSStream._run] pushing PCM frame: {len(resampled)} bytes, frame samples: {len(resampled)//2}")
                        output_emitter.push(resampled)
                        self._pcm_frames_received += 1
                        self._pcm_bytes_received += len(resampled)
                        buffer = b""
                        chunk_count += 1
                        logger.debug(f"[QwenTTSStream._run] pushed chunk {chunk_count}")

                if buffer:
                    resampled = _resample_24k_to_48k(buffer)
                    logger.info(f"[QwenTTSStream._run] pushing final PCM frame: {len(resampled)} bytes, frame samples: {len(resampled)//2}")
                    output_emitter.push(resampled)
                    self._pcm_frames_received += 1
                    self._pcm_bytes_received += len(resampled)
                    chunk_count += 1

                logger.info(f"[QwenTTSStream._run] TTS completed, total chunks: {chunk_count}, total PCM bytes: {self._pcm_bytes_received}, total PCM frames: {self._pcm_frames_received}")

        except asyncio.TimeoutError:
            logger.error(f"[QwenTTSStream._run] TTS stream timed out after {self._adapter._timeout}s")
        except Exception as e:
            logger.error(f"[QwenTTSStream._run] TTS stream error: {e}")
        finally:
            if session:
                await session.close()

        output_emitter.end_segment()
        logger.info(f"[QwenTTSStream._run] finished, segment ended, total PCM bytes pushed: {self._pcm_bytes_received}")
