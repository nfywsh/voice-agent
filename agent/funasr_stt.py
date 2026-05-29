# agent/funasr_stt.py
"""FunASR HTTP 语音识别适配器

对接本地 funasr-old 服务 (HTTP 接口):
  POST /  - 发送 JSON {"audio": "/path/to/file", "language": "zh"}

通过环境变量配置:
  OPENAI_ASR_BASE_URL: API 基地址 (默认 http://nginx_gateway:80/asr)
  OPENAI_ASR_API_KEY: API 密钥 (占位符即可)
  OPENAI_ASR_MODEL: 模型名 (默认 fun-asr-2512)
"""

import asyncio
import logging
import os
import tempfile
import uuid
import wave
from typing import Optional

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import stt, utils
from livekit.agents.language import LanguageCode
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr
from livekit.agents.utils import AudioBuffer

logger = logging.getLogger(__name__)

_ASR_SAMPLE_RATE = 16000


def _resample_to_16k(pcm_bytes: bytes, source_rate: int) -> bytes:
    """将音频重采样到 16kHz"""
    if source_rate == _ASR_SAMPLE_RATE:
        return pcm_bytes
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    ratio = _ASR_SAMPLE_RATE / source_rate
    indices = np.arange(0, len(audio), ratio)
    resampled = np.interp(indices, np.arange(len(audio)), audio).astype(np.int16)
    return resampled.tobytes()


class FunASRSTT(stt.STT):
    """FunASR HTTP 语音识别适配器

    支持流式识别（streaming）和单次识别（_recognize_impl）。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "http://nginx_gateway:80/asr",
        model: str = "fun-asr-2512",
        language: str = "zh",
        vad: "Optional[object]" = None,  # silero.VAD passed from agent.py
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=False,
            )
        )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._language = language
        self._vad = vad  # For VAD event handling when used without StreamAdapterWrapper

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """非流式识别：发送完整音频，返回识别结果。"""
        lang = language if isinstance(language, str) else self._language

        logger.info(f"[FunASRSTT] _recognize_impl called, buffer type={type(buffer)}, lang={lang}")

        if isinstance(buffer, list):
            all_samples = []
            sample_rate = None
            for frame in buffer:
                if sample_rate is None:
                    sample_rate = frame.sample_rate
                all_samples.append(np.frombuffer(bytes(frame.data), dtype=np.int16))
            if not all_samples:
                logger.info("[FunASRSTT] Empty buffer, returning empty transcript")
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language=LanguageCode(lang), text="", confidence=0.0)],
                )
            audio_np = np.concatenate(all_samples)
            sample_rate = sample_rate or _ASR_SAMPLE_RATE
        else:
            sample_rate = buffer.sample_rate
            audio_np = np.frombuffer(bytes(buffer.data), dtype=np.int16)

        max_amp = float(np.max(np.abs(audio_np)))
        logger.info(f"[FunASRSTT] Audio buffer: samples={len(audio_np)}, sample_rate={sample_rate}, duration={len(audio_np)/sample_rate:.2f}s, max_amplitude={max_amp}")

        pcm_16k = _resample_to_16k(audio_np.tobytes(), sample_rate)

        asr_temp_dir = "/data/asr-temp"
        os.makedirs(asr_temp_dir, exist_ok=True)
        tmp_path = os.path.join(asr_temp_dir, f"asr_{uuid.uuid4().hex}.wav")
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(_ASR_SAMPLE_RATE)
            wf.writeframes(pcm_16k)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._base_url,
                    json={
                        "audio": tmp_path,
                        "model": self._model,
                        "language": lang,
                    },
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=aiohttp.ClientTimeout(total=30.0),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"FunASR error {resp.status}: {error_text}")
                        return stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[stt.SpeechData(language=LanguageCode(lang), text="", confidence=0.0)],
                        )

                    result = await resp.json()
                    text = result.get("text", "")
                    logger.info(f"[FunASRSTT] ASR result: '{text}'")
                    return stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[stt.SpeechData(language=LanguageCode(lang), text=text, confidence=1.0)],
                    )
        except Exception as e:
            logger.error(f"[FunASRSTT] ASR request failed: {e}")
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[stt.SpeechData(language=LanguageCode(lang), text="", confidence=0.0)],
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = APIConnectOptions(),
    ) -> "FunASRSTTStream":
        """返回流式识别器。

        如果配置了 VAD，使用 StreamAdapterWrapper 进行 VAD 驱动的流式识别。
        否则返回原生 FunASRSTTStream（等待 FlushSentinel）。
        """
        if self._vad:
            return stt.StreamAdapter(
                stt=self,
                vad=self._vad,
            ).stream(language=language, conn_options=conn_options)
        return FunASRSTTStream(self, language=language, conn_options=conn_options)


class FunASRSTTStream(stt.RecognizeStream):
    """FunASR 流式识别器

    从 audio_ch 接收音频帧，收集完成后调用 HTTP API 识别。
    """

    def __init__(
        self,
        stt: FunASRSTT,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = APIConnectOptions(),
    ):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=_ASR_SAMPLE_RATE)
        self._language = language if isinstance(language, str) else stt._language
        logger.info(f"[FunASRSTT] Stream created")

    async def _run(self) -> None:
        """收集音频帧，识别后发送结果。"""
        logger.info(f"[FunASRSTT] _run started")
        try:
            audio_buffer = bytearray()
            frame_count = 0
            flush_received = False
            actual_sample_rate = _ASR_SAMPLE_RATE  # Will be overridden by first frame's rate

            async for item in self._input_ch:
                if isinstance(item, stt.RecognizeStream._FlushSentinel):
                    # FlushSentinel received - stop collecting, proceed to ASR
                    logger.info(f"[FunASRSTT] FlushSentinel received, frames={frame_count}, buffer={len(audio_buffer)}")
                    flush_received = True
                    break
                elif isinstance(item, rtc.AudioFrame):
                    pcm_data = item.data
                    if isinstance(pcm_data, memoryview):
                        pcm_data = bytes(pcm_data)
                    audio_buffer.extend(pcm_data)
                    frame_count += 1
                    if actual_sample_rate == _ASR_SAMPLE_RATE:
                        actual_sample_rate = item.sample_rate
                    logger.info(f"[FunASRSTT] collected frame {frame_count}, total size: {len(audio_buffer)}, sample_rate={item.sample_rate}")

            logger.info(f"[FunASRSTT] _run finished collecting: frames={frame_count}, buffer_size={len(audio_buffer)}, flush={flush_received}, actual_sr={actual_sample_rate}")

            if not audio_buffer:
                self._event_ch.send_nowait(stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language=LanguageCode(self._language), text="", confidence=0.0)],
                ))
                return

            # Use actual_sample_rate (from first frame) for correct resampling
            # RecognizeStream may already resample to _ASR_SAMPLE_RATE, but we defensively check
            pcm_16k = _resample_to_16k(bytes(audio_buffer), actual_sample_rate)

            asr_temp_dir = "/data/asr-temp"
            os.makedirs(asr_temp_dir, exist_ok=True)
            tmp_path = os.path.join(asr_temp_dir, f"asr_{uuid.uuid4().hex}.wav")
            with wave.open(tmp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(_ASR_SAMPLE_RATE)
                wf.writeframes(pcm_16k)

            logger.info(f"[FunASRSTT] Calling ASR API with {len(pcm_16k)} bytes")

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self._stt._base_url,
                        json={
                            "audio": tmp_path,
                            "model": self._stt._model,
                            "language": self._language,
                        },
                        headers={"Authorization": f"Bearer {self._stt._api_key}"},
                        timeout=aiohttp.ClientTimeout(total=30.0),
                    ) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            text = result.get("text", "")
                            logger.info(f"[FunASRSTT] ASR result: {text}")
                            self._event_ch.send_nowait(stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                alternatives=[stt.SpeechData(language=LanguageCode(self._language), text=text, confidence=1.0)],
                            ))
                            # Send END_OF_SPEECH to signal end of user speech turn
                            self._event_ch.send_nowait(stt.SpeechEvent(
                                type=stt.SpeechEventType.END_OF_SPEECH,
                                alternatives=[],
                            ))
                        else:
                            error_text = await resp.text()
                            logger.error(f"FunASR error {resp.status}: {error_text}")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"[FunASRSTT] _run error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info("[FunASRSTT] Stream finished")


def create_stt(vad: "Optional[object]" = None) -> stt.STT:
    """工厂函数：从环境变量创建 FunASR STT 实例。

    Args:
        vad: VAD 实例，用于启用 VAD 驱动的流式识别
    """
    api_key = os.environ.get("OPENAI_ASR_API_KEY", "placeholder")
    base_url = os.environ.get("OPENAI_ASR_BASE_URL", "http://nginx_gateway:80/asr")
    model = os.environ.get("OPENAI_ASR_MODEL", "fun-asr-2512")
    language = os.environ.get("OPENAI_ASR_LANGUAGE", "zh")

    logger.info(f"Creating FunASR STT: base_url={base_url}, model={model}, language={language}, vad={'Yes' if vad else 'No'}")
    return FunASRSTT(
        api_key=api_key,
        base_url=base_url,
        model=model,
        language=language,
        vad=vad,
    )