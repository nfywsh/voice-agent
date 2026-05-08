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
import re
import time
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
        timeout: float = 30.0,
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
        import time
        t0 = time.monotonic()
        logger.info(f"[QwenTTSStream._run] started, waiting for text input...")
        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",  # LiveKit 会自动处理重采样
            stream=True,
        )
        output_emitter.start_segment(segment_id=str(uuid.uuid4()))

        # 流式处理：边接收 LLM 文本边分片发送给 TTS，保持流式输出
        # _input_ch 是异步生成器，yield 出 LLM 的每个文本片段
        full_text = ""
        pending_text = ""  # 未发送给 TTS 的累积文本
        first_sent = False  # 是否已发送第一片（首片不等待，直接发）
        last_send_time = t0  # 上次发送 TTS 的时间
        pcm_bytes_received = 0  # 累计接收的 PCM 字节数（本地变量，非实例属性）
        session = aiohttp.ClientSession()

        # 发送策略：
        # - 首片：30 字符 或 遇到句末符 立即发（低延迟）
        # - 后续：等待上一片 TTS 完成后再发下一片（避免重叠/乱序）
        #   OR 等待超 5 秒强制发（防断档，LLM 太慢时最多等 5s）
        MAX_WAIT_SEC = 5.0   # 最长等待秒数，超时强制发送
        MAX_TTS_CHUNK = 300  # TTS 单次最大字符数（API 限制 500，留余量）

        async def send_tts_chunk(text: str) -> None:
            """发送单个文本分片给 TTS 服务并流式输出音频"""
            nonlocal first_sent, last_send_time, pcm_bytes_received
            if not text.strip():
                return
            tts_start = time.monotonic()
            async with session.post(
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
                    logger.error(f"[QwenTTSStream._run] TTS chunk error {resp.status}: {error_text}")
                    return
                buffer = b""
                async for chunk_data in resp.content.iter_chunked(4096):
                    buffer += chunk_data
                    if len(buffer) >= 3840:
                        resampled = _resample_24k_to_48k(buffer)
                        output_emitter.push(resampled)
                        pcm_bytes_received += len(resampled)
                        buffer = b""
                if buffer:
                    resampled = _resample_24k_to_48k(buffer)
                    output_emitter.push(resampled)
                    pcm_bytes_received += len(resampled)
            logger.info(f"[QwenTTSStream._run] TTS chunk ({len(text)} chars) done in {time.monotonic() - tts_start:.3f}s, bytes={pcm_bytes_received}")
            first_sent = True
            last_send_time = time.monotonic()

        try:
            async for item in self._input_ch:
                if isinstance(item, str) and item.strip():
                    full_text += item
                    pending_text += item

                    # 检查是否应该发送
                    time_since_last = time.monotonic() - last_send_time
                    timeout_trigger = first_sent and time_since_last >= MAX_WAIT_SEC

                    if not first_sent:
                        # 首片：30 字符 或 句末符 立即发
                        can_send = len(pending_text) >= 30 or re.search(r'[。！？；\n]', pending_text)
                    elif timeout_trigger:
                        # 后续超时保底：强制发送
                        can_send = True
                    else:
                        # 正常情况：等上一片完成再检查
                        can_send = False

                    while can_send and pending_text:
                        if not first_sent and len(pending_text) < 30:
                            break  # 首片不足 30 字，等

                        # 找到最后一个句子结束符作为切割点
                        m = re.search(r'[。！？；\n](.{0,30})$', pending_text)
                        if m:
                            cut_pos = pending_text.rfind(m.group(0)) + 1
                        else:
                            # 没有句末符，硬切
                            cut_pos = min(len(pending_text), MAX_TTS_CHUNK)

                        send_text = pending_text[:cut_pos]
                        pending_text = pending_text[cut_pos:]

                        if send_text:
                            reason = "first" if not first_sent else f"timeout({time_since_last:.1f}s)"
                            logger.info(f"[QwenTTSStream._run] sending chunk: {len(send_text)} chars, reason={reason}, pending={len(pending_text)}")
                            await send_tts_chunk(send_text)

                        # 重新计算 timeout（因为可能已经过了几秒）
                        time_since_last = time.monotonic() - last_send_time
                        timeout_trigger = first_sent and time_since_last >= MAX_WAIT_SEC

                        # 首片发出后，后续检查：超时 OR 有足够文本 OR 句末符 OR 收到新文本
                        if pending_text:
                            time_since_last_now = time.monotonic() - last_send_time
                            timeout_now = first_sent and time_since_last_now >= MAX_WAIT_SEC
                            can_send = timeout_now or len(pending_text) >= 30 or bool(re.search(r'[。！？；\n]', pending_text))
                        else:
                            can_send = False  # 等待下一个 item 触发 or 超时

            # 处理剩余文本（等最后一篇 TTS 完成）
            while pending_text.strip():
                m = re.search(r'[。！？；\n](.{0,30})$', pending_text)
                if m:
                    cut_pos = pending_text.rfind(m.group(0)) + 1
                else:
                    cut_pos = min(len(pending_text), MAX_TTS_CHUNK)
                send_text = pending_text[:cut_pos]
                pending_text = pending_text[cut_pos:]
                if send_text:
                    await send_tts_chunk(send_text)

            total_tts_time = time.monotonic() - t0
            logger.info(f"[QwenTTSStream._run] TTS completed, total time: {total_tts_time:.3f}s, bytes: {pcm_bytes_received}")

        except asyncio.TimeoutError:
            logger.error(f"[QwenTTSStream._run] TTS stream timed out after {self._adapter._timeout}s")
        except Exception as e:
            logger.error(f"[QwenTTSStream._run] TTS stream error: {e}")
        finally:
            if session:
                await session.close()

        output_emitter.end_segment()
        logger.info(f"[QwenTTSStream._run] finished, total PCM bytes: {pcm_bytes_received}")
