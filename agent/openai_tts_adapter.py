# agent/openai_tts_adapter.py
"""OpenAI 格式 TTS 适配器

支持 OpenAI Audio Speech API 格式:
  POST /v1/audio/speech

通过环境变量配置:
  OPENAI_TTS_BASE_URL: API 基地址 (默认 https://api.openai.com/v1)
  OPENAI_TTS_API_KEY: API 密钥
  OPENAI_TTS_MODEL: 模型名 (默认 tts-1)
  OPENAI_TTS_VOICE: 音色 (默认 alloy)

Docker 网络内直接通过服务名访问:
  - VLLM TTS 服务: http://tts-service:8001/v1/audio/speech
"""

import asyncio
import logging
import os
import uuid
from typing import TYPE_CHECKING, Optional

import aiohttp
import numpy as np
from scipy.signal import resample_poly
from livekit.agents import tts
from livekit.agents.tts.tts import AudioEmitter
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

if TYPE_CHECKING:
    from monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)

OUTPUT_SAMPLE_RATE = 48000
TTS_SAMPLE_RATE = 24000


def _resample(pcm_bytes: bytes, source_rate: int, target_rate: int) -> bytes:
    """将 16-bit PCM 音频从 source_rate 重采样到 target_rate."""
    if source_rate == target_rate:
        return pcm_bytes

    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    from math import gcd
    g = gcd(up := target_rate, down := source_rate)
    audio_resampled = resample_poly(audio, up // g, down // g)
    audio_int16 = np.clip(audio_resampled * 32767, -32768, 32767).astype(np.int16)
    return audio_int16.tobytes()


class OpenAITTSAdapter(tts.TTS):
    """OpenAI 格式 TTS 适配器

    继承 livekit.agents.tts.TTS，实现 synthesize 和 stream 方法。
    支持 OpenAI Audio Speech API 格式。

    配置参数（通过环境变量或构造参数设置）：
    - max_tts_chunk: 后续分片最大字符数（默认 300）
    - first_chunk_min: 首片最小字符数（默认 30）
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "tts-1",
        voice: str = "alloy",
        speed: float = 1.0,
        timeout: float = 30.0,
        max_tts_chunk: int = 300,
        first_chunk_min: int = 30,
        metrics: "MetricsCollector" = None,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=1,
        )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._voice = voice
        self._speed = speed
        self._timeout = timeout
        self._max_tts_chunk = max_tts_chunk
        self._first_chunk_min = first_chunk_min
        self._metrics = metrics

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "OpenAITTSChunkedStream":
        """非流式合成：请求 TTS 服务并返回 ChunkedStream。"""
        return OpenAITTSChunkedStream(self, text, conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "OpenAITTSStream":
        """返回流式合成器。"""
        return OpenAITTSStream(self, conn_options=conn_options)


class OpenAITTSChunkedStream(tts.ChunkedStream):
    """非流式合成的 ChunkedStream 实现。"""

    def __init__(
        self,
        adapter: OpenAITTSAdapter,
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
                    f"{self._adapter._base_url}/audio/speech",
                    json={
                        "model": self._adapter._model,
                        "input": self._input_text,
                        "voice": self._adapter._voice,
                        "response_format": "pcm",
                        "speed": self._adapter._speed,
                    },
                    headers={"Authorization": f"Bearer {self._adapter._api_key}"},
                    timeout=aiohttp.ClientTimeout(total=self._adapter._timeout),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"OpenAI TTS error {resp.status}: {error_text}")
                        return

                    buffer = b""
                    async for chunk in resp.content.iter_chunked(4096):
                        buffer += chunk
                        if len(buffer) >= 3840:
                            resampled = _resample(buffer, TTS_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
                            output_emitter.push(resampled)
                            buffer = b""

                    if buffer:
                        resampled = _resample(buffer, TTS_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
                        output_emitter.push(resampled)

            output_emitter.flush()

        except asyncio.TimeoutError:
            logger.error(f"OpenAI TTS request timed out after {self._adapter._timeout}s")
        except Exception as e:
            logger.error(f"OpenAI TTS synthesis error: {e}")


class OpenAITTSStream(tts.SynthesizeStream):
    """OpenAI 格式流式合成器

    工作流程：
    1. Agent 通过 push_text() 推送文本
    2. _run() 从 _input_ch 读取文本，按规则切分后发送到 TTS 服务
    3. 流式接收音频并输出到 AudioEmitter
    """

    def __init__(
        self,
        adapter: OpenAITTSAdapter,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        super().__init__(tts=adapter, conn_options=conn_options)
        self._adapter = adapter

    async def _run(self, output_emitter: AudioEmitter) -> None:
        """流式 TTS 主循环"""
        import re
        import time

        t0 = time.monotonic()
        logger.info(f"[OpenAITTSStream._run] started")

        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=str(uuid.uuid4()))

        MAX_TTS_CHUNK = self._adapter._max_tts_chunk
        FIRST_CHUNK_MIN = self._adapter._first_chunk_min
        MAX_WAIT_SEC = float(os.environ.get("TTS_CHUNK_WAIT_SEC", "5.0"))

        session = aiohttp.ClientSession()

        def _maybe_metrics():
            return getattr(self._adapter, '_metrics', None)

        pending_text = ""
        first_sent = False
        last_send_time = t0
        pcm_bytes_sent = 0

        try:
            async for item in self._input_ch:
                if isinstance(item, str) and item.strip():
                    pending_text += item

                    time_since_last = time.monotonic() - last_send_time
                    timeout_trigger = first_sent and time_since_last >= MAX_WAIT_SEC

                    if not first_sent:
                        can_send = len(pending_text) >= FIRST_CHUNK_MIN or re.search(r'[。！？；\n]', pending_text)
                    elif timeout_trigger:
                        can_send = True
                    else:
                        can_send = False

                    while can_send and pending_text:
                        if not first_sent and len(pending_text) < FIRST_CHUNK_MIN:
                            break

                        cut_pos = 0
                        for m in re.finditer(r'[。！？；\n]', pending_text):
                            if m.end() <= MAX_TTS_CHUNK:
                                cut_pos = m.end()
                            if m.end() == MAX_TTS_CHUNK:
                                break
                        if cut_pos == 0:
                            cut_pos = min(len(pending_text), MAX_TTS_CHUNK)

                        send_text = pending_text[:cut_pos]
                        pending_text = pending_text[cut_pos:]

                        if send_text:
                            m = _maybe_metrics()
                            if m:
                                m.tts_chunk_sent(len(send_text))

                            async with session.post(
                                f"{self._adapter._base_url}/audio/speech",
                                json={
                                    "model": self._adapter._model,
                                    "input": send_text,
                                    "voice": self._adapter._voice,
                                    "response_format": "pcm",
                                    "speed": self._adapter._speed,
                                },
                                headers={"Authorization": f"Bearer {self._adapter._api_key}"},
                                timeout=aiohttp.ClientTimeout(total=self._adapter._timeout),
                            ) as resp:
                                if resp.status != 200:
                                    error_text = await resp.text()
                                    logger.error(f"OpenAI TTS chunk error {resp.status}: {error_text}")
                                    continue

                                if not first_sent:
                                    if m:
                                        m.tts_start()
                                    first_sent = True

                                buffer = b""
                                async for chunk_data in resp.content.iter_chunked(4096):
                                    buffer += chunk_data
                                    if len(buffer) >= 3840:
                                        resampled = _resample(buffer, TTS_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
                                        output_emitter.push(resampled)
                                        output_emitter.flush()
                                        pcm_bytes_sent += len(resampled)

                                        if m and pcm_bytes_sent == len(resampled):
                                            m.tts_first_audio()

                                        buffer = b""

                                if buffer:
                                    resampled = _resample(buffer, TTS_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
                                    output_emitter.push(resampled)
                                    output_emitter.flush()
                                    pcm_bytes_sent += len(resampled)

                            last_send_time = time.monotonic()

                        time_since_last = time.monotonic() - last_send_time
                        timeout_trigger = first_sent and time_since_last >= MAX_WAIT_SEC
                        if pending_text:
                            timeout_now = first_sent and time_since_last >= MAX_WAIT_SEC
                            can_send = timeout_now or len(pending_text) >= 30 or bool(re.search(r'[。！？；\n]', pending_text))
                        else:
                            can_send = False

            _m = _maybe_metrics()
            if _m:
                _m.tts_end()

        except asyncio.TimeoutError:
            logger.error(f"OpenAI TTS stream timed out after {self._adapter._timeout}s")
        except Exception as e:
            logger.error(f"OpenAI TTS stream error: {e}")
        finally:
            if session:
                await session.close()

        output_emitter.end_segment()
        logger.info(f"[OpenAITTSStream._run] finished, total PCM bytes: {pcm_bytes_sent}")


def create_tts() -> tts.TTS:
    """工厂函数：从环境变量创建 OpenAI 格式 TTS 实例。"""
    api_key = os.environ.get("OPENAI_TTS_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_TTS_API_KEY environment variable is required")

    base_url = os.environ.get(
        "OPENAI_TTS_BASE_URL",
        "https://api.openai.com/v1"
    )
    model = os.environ.get("OPENAI_TTS_MODEL", "tts-1")
    voice = os.environ.get("OPENAI_TTS_VOICE", "alloy")
    speed = float(os.environ.get("OPENAI_TTS_SPEED", "1.0"))
    timeout = float(os.environ.get("OPENAI_TTS_TIMEOUT", "30"))
    max_tts_chunk = int(os.environ.get("OPENAI_TTS_MAX_CHUNK", "300"))
    first_chunk_min = int(os.environ.get("OPENAI_TTS_FIRST_CHUNK_MIN", "30"))

    logger.info(f"Creating OpenAI TTS: base_url={base_url}, model={model}, voice={voice}")
    return OpenAITTSAdapter(
        api_key=api_key,
        base_url=base_url,
        model=model,
        voice=voice,
        speed=speed,
        timeout=timeout,
        max_tts_chunk=max_tts_chunk,
        first_chunk_min=first_chunk_min,
    )