# agent/qwen3_tts_adapter.py
"""Qwen3-TTS-Base 适配器

VLLM 部署的 Qwen3-TTS-Base，本地路径 172.17.1.53:8021

必须参数:
  - task_type: "Base"
  - ref_audio: 参考音频 (base64 data URL)
  - ref_text: 参考音频转写文本

通过环境变量配置:
  QWEN3_TTS_BASE_URL: API 基地址 (默认 http://172.17.1.53:8021)
  QWEN3_TTS_REF_AUDIO: 参考音频路径 (默认 /data/voice-temp/voices/voice_8fe34d12.wav)
  QWEN3_TTS_REF_TEXT: 参考音频转写 (默认 "这是一段参考音色的示例文本")
  QWEN3_TTS_TIMEOUT: 请求超时秒数 (默认 30)
  OPENAI_TTS_MAX_CHUNK: 最大分片字符数 (默认 300)
  OPENAI_TTS_FIRST_CHUNK_MIN: 首片最小字符数 (默认 30)
"""

import asyncio
import base64
import logging
import os
import uuid
from math import gcd
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
DEFAULT_REF_AUDIO_PATH = "/data/voice-temp/voices/voice_8fe34d12.wav"
DEFAULT_REF_TEXT = "这是一段参考音色的示例文本"


def _resample(pcm_bytes: bytes, source_rate: int, target_rate: int) -> bytes:
    if source_rate == target_rate:
        return pcm_bytes
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    g = gcd(up := target_rate, down := source_rate)
    audio_resampled = resample_poly(audio, up // g, down // g)
    audio_int16 = np.clip(audio_resampled * 32767, -32768, 32767).astype(np.int16)
    return audio_int16.tobytes()


def _load_ref_audio_base64(path: str) -> Optional[str]:
    """读取参考音频文件并转为 base64 data URL"""
    try:
        with open(path, 'rb') as f:
            audio_data = f.read()
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        return f"data:audio/wav;base64,{audio_b64}"
    except Exception as e:
        logger.warning(f"[Qwen3TTSAdapter] Failed to load ref_audio {path}: {e}")
        return None


class Qwen3TTSAdapter(tts.TTS):
    def __init__(
        self,
        base_url: str = "http://172.17.1.53:8021",
        ref_audio_path: str = DEFAULT_REF_AUDIO_PATH,
        ref_text: str = DEFAULT_REF_TEXT,
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
        self._base_url = base_url.rstrip("/")
        self._ref_audio_path = ref_audio_path
        self._ref_text = ref_text
        self._timeout = timeout
        self._max_tts_chunk = max_tts_chunk
        self._first_chunk_min = first_chunk_min
        self._metrics = metrics
        self._ref_audio_b64: Optional[str] = None

    def set_ref_audio(self, path: str, ref_text: Optional[str] = None) -> None:
        """运行时设置参考音色（从 session_params 注入）"""
        self._ref_audio_path = path
        if ref_text is not None:
            self._ref_text = ref_text
        self._ref_audio_b64 = None

    def _get_ref_audio(self) -> Optional[str]:
        if not self._ref_audio_b64:
            self._ref_audio_b64 = _load_ref_audio_base64(self._ref_audio_path)
        return self._ref_audio_b64

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "Qwen3TTSChunkedStream":
        return Qwen3TTSChunkedStream(self, text, conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "Qwen3TTSStream":
        return Qwen3TTSStream(self, conn_options=conn_options)


class Qwen3TTSChunkedStream(tts.ChunkedStream):
    def __init__(self, adapter: Qwen3TTSAdapter, text: str, conn_options: APIConnectOptions):
        super().__init__(tts=adapter, input_text=text, conn_options=conn_options)
        self._adapter = adapter

    async def _run(self, output_emitter: AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
        )
        ref_audio = self._adapter._get_ref_audio()
        if not ref_audio:
            logger.error("[Qwen3TTSAdapter] No ref_audio available")
            return
        connector = aiohttp.TCPConnector(ssl=False)
        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.post(
                    f"{self._adapter._base_url}/v1/audio/speech",
                    json={
                        "model": "Qwen3-TTS-Base",
                        "input": self._input_text,
                        "response_format": "pcm",
                        "task_type": "Base",
                        "ref_audio": ref_audio,
                        "ref_text": self._adapter._ref_text,
                    },
                    headers={"Authorization": "Bearer placeholder"},
                    timeout=aiohttp.ClientTimeout(total=self._adapter._timeout),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"Qwen3TTS error {resp.status}: {error_text}")
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
            logger.error(f"Qwen3TTS request timed out after {self._adapter._timeout}s")
        except Exception as e:
            logger.error(f"Qwen3TTS synthesis error: {e}")


class Qwen3TTSStream(tts.SynthesizeStream):
    def __init__(self, adapter: Qwen3TTSAdapter, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        super().__init__(tts=adapter, conn_options=conn_options)
        self._adapter = adapter

    async def _run(self, output_emitter: AudioEmitter) -> None:
        import time

        t0 = time.monotonic()
        logger.info("[Qwen3TTSStream._run] started")

        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=str(uuid.uuid4()))

        ref_audio = self._adapter._get_ref_audio()
        if not ref_audio:
            logger.error("[Qwen3TTSStream] No ref_audio available, cannot synthesize")
            return

        logger.info(f"[Qwen3TTSStream] ref_audio loaded: {len(ref_audio)} chars")
        connector = aiohttp.TCPConnector(ssl=False)
        session = aiohttp.ClientSession(connector=connector)

        def _maybe_metrics():
            return getattr(self._adapter, '_metrics', None)

        pending_text = ""
        pcm_bytes_sent = 0
        item_count = 0

        logger.info(f"[Qwen3TTSStream] _input_ch type: {type(self._input_ch)}, waiting for text...")

        try:
            async for item in self._input_ch:
                item_count += 1
                if item_count <= 5:
                    logger.info(f"[Qwen3TTSStream] item {item_count}: type={type(item).__name__}, value={repr(item)[:80]}")

                if isinstance(item, str):
                    pending_text += item
                elif item is None:
                    continue
                else:
                    # FlushSentinel or other - try to flush what we have
                    if pending_text:
                        logger.info(f"[Qwen3TTSStream] Sentinel received after {item_count} items, pending_text len={len(pending_text)}")
                        break

            logger.info(f"[Qwen3TTSStream] _input_ch exhausted: items={item_count}, pending_text='{pending_text[:100]}...'")

            if not pending_text.strip():
                logger.warning("[Qwen3TTSStream] No text accumulated, returning")
                return

            # Send the full pending_text to TTS in one request
            logger.info(f"[Qwen3TTSStream] Sending single TTS request for {len(pending_text)} chars")
            m = _maybe_metrics()
            if m:
                m.tts_start()

            try:
                async with session.post(
                    f"{self._adapter._base_url}/v1/audio/speech",
                    json={
                        "model": "Qwen3-TTS-Base",
                        "input": pending_text,
                        "response_format": "pcm",
                        "task_type": "Base",
                        "ref_audio": ref_audio,
                        "ref_text": self._adapter._ref_text,
                    },
                    headers={"Authorization": "Bearer placeholder"},
                    timeout=aiohttp.ClientTimeout(total=self._adapter._timeout),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"Qwen3TTS chunk error {resp.status}: {error_text}")
                        return

                    buffer = b""
                    first_audio_sent = False
                    async for chunk_data in resp.content.iter_chunked(4096):
                        buffer += chunk_data
                        if len(buffer) >= 3840:
                            resampled = _resample(buffer, TTS_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
                            output_emitter.push(resampled)
                            output_emitter.flush()
                            pcm_bytes_sent += len(resampled)
                            if m and not first_audio_sent:
                                m.tts_first_audio()
                                first_audio_sent = True
                            buffer = b""
                    if buffer:
                        resampled = _resample(buffer, TTS_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
                        output_emitter.push(resampled)
                        output_emitter.flush()
                        pcm_bytes_sent += len(resampled)
                        if m and not first_audio_sent:
                            m.tts_first_audio()
                            first_audio_sent = True

                    if m:
                        m.tts_end()

            except asyncio.TimeoutError:
                logger.error(f"Qwen3TTS request timed out after {self._adapter._timeout}s")
            except Exception as e:
                logger.error(f"Qwen3TTS synthesis error: {e}")

        except asyncio.TimeoutError:
            logger.error(f"Qwen3TTS stream timed out after {self._adapter._timeout}s")
        except Exception as e:
            logger.error(f"Qwen3TTS stream error: {e}")
        finally:
            if session:
                await session.close()

        output_emitter.end_segment()
        logger.info(f"[Qwen3TTSStream._run] finished, total PCM bytes: {pcm_bytes_sent}")


def create_tts() -> tts.TTS:
    base_url = os.environ.get("QWEN3_TTS_BASE_URL", "http://172.17.1.53:8021")
    ref_audio_path = os.environ.get("QWEN3_TTS_REF_AUDIO", DEFAULT_REF_AUDIO_PATH)
    ref_text = os.environ.get("QWEN3_TTS_REF_TEXT", DEFAULT_REF_TEXT)
    timeout = float(os.environ.get("QWEN3_TTS_TIMEOUT", "30"))
    max_tts_chunk = int(os.environ.get("OPENAI_TTS_MAX_CHUNK", "300"))
    first_chunk_min = int(os.environ.get("OPENAI_TTS_FIRST_CHUNK_MIN", "30"))

    logger.info(f"Creating Qwen3TTS adapter: base_url={base_url}, ref_audio={ref_audio_path}")
    return Qwen3TTSAdapter(
        base_url=base_url,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        timeout=timeout,
        max_tts_chunk=max_tts_chunk,
        first_chunk_min=first_chunk_min,
    )