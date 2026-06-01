# agent/openai_stt.py
"""OpenAI 格式 ASR 适配器

支持 OpenAI Audio Transcription API 格式:
  POST /v1/audio/transcriptions

通过环境变量配置:
  OPENAI_ASR_BASE_URL: API 基地址 (默认 https://api.openai.com/v1)
  OPENAI_ASR_API_KEY: API 密钥
  OPENAI_ASR_MODEL: 模型名 (默认 whisper-1)

Docker 网络内直接通过服务名访问:
  - VLLM/其他 ASR 服务: http://asr-service:8030/v1/audio/transcriptions
"""

import asyncio
import base64
import logging
import os
import tempfile
import uuid
from typing import Optional

import aiohttp
import numpy as np
from livekit import rtc
from livekit.agents import stt
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


class OpenAISTT(stt.STT):
    """OpenAI 格式语音识别适配器

    支持流式识别（streaming）和单次识别（_recognize_impl）。
    使用 OpenAI Audio Transcription API 格式。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "whisper-1",
        language: str = "zh",
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

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """非流式识别：发送完整音频，返回识别结果。"""
        lang = language if isinstance(language, str) else self._language

        if isinstance(buffer, list):
            all_samples = []
            sample_rate = None
            for frame in buffer:
                if sample_rate is None:
                    sample_rate = frame.sample_rate
                all_samples.append(np.frombuffer(bytes(frame.data), dtype=np.int16))
            if not all_samples:
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(language=LanguageCode(lang), text="", confidence=0.0)],
                )
            audio_np = np.concatenate(all_samples)
            sample_rate = sample_rate or _ASR_SAMPLE_RATE
        else:
            sample_rate = buffer.sample_rate
            audio_np = np.frombuffer(bytes(buffer.data), dtype=np.int16)

        pcm_16k = _resample_to_16k(audio_np.tobytes(), sample_rate)

        # 保存到 /data/voice-temp/asr 目录（容器内挂载路径，对应 ASR 服务的要求）
        asr_temp_dir = "/data/voice-temp/asr"
        os.makedirs(asr_temp_dir, exist_ok=True)
        import wave
        tmp_path = os.path.join(asr_temp_dir, f"asr_{uuid.uuid4().hex}.wav")
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(_ASR_SAMPLE_RATE)
            wf.writeframes(pcm_16k)

        try:
            async with aiohttp.ClientSession() as session:
                # ASR 服务接受文件路径字符串而非 multipart 文件上传
                async with session.post(
                    f"{self._base_url}/audio/transcriptions",
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
                        logger.error(f"OpenAI ASR error {resp.status}: {error_text}")
                        return stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[stt.SpeechData(language=LanguageCode(lang), text="", confidence=0.0)],
                        )

                    result = await resp.json()
                    text = result.get("text", "")
                    return stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[stt.SpeechData(language=LanguageCode(lang), text=text, confidence=1.0)],
                    )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = APIConnectOptions(),
    ) -> "OpenAISTTStream":
        """返回流式识别器。"""
        return OpenAISTTStream(self, language=language, conn_options=conn_options)


class OpenAISTTStream(stt.RecognizeStream):
    """OpenAI 格式流式语音识别

    对于流式识别，OpenAI API 使用 WebSocket 格式。
    如果上游服务不支持 WebSocket，则回退到非流式。
    """

    def __init__(
        self,
        stt: OpenAISTT,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = APIConnectOptions(),
    ):
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=_ASR_SAMPLE_RATE)
        self._language = language if isinstance(language, str) else stt._language
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._task_id = str(uuid.uuid4())
        self._audio_buffer = bytearray()
        self._recv_task: Optional[asyncio.Task] = None
        self._end_of_stream_sent = False
        logger.info(f"[OpenAISTT] Stream created: stream_id={id(self)}, task_id={self._task_id}")

    async def _run(self) -> None:
        """主循环：从 _input_ch 读取音频帧，发送到 ASR 服务。"""
        try:
            await self._connect()
            self._recv_task = asyncio.create_task(self._recv_loop())
            frame_count = 0

            async for item in self._input_ch:
                if isinstance(item, rtc.AudioFrame):
                    pcm_data = item.data
                    if isinstance(pcm_data, memoryview):
                        pcm_data = bytes(pcm_data)

                    self._audio_buffer.extend(pcm_data)
                    frame_count += 1

                    while len(self._audio_buffer) >= 3200:
                        chunk = bytes(self._audio_buffer[:3200])
                        self._audio_buffer = self._audio_buffer[3200:]
                        if self._ws and not self._ws.closed:
                            await self._ws.send_bytes(chunk)

            if self._ws and not self._ws.closed:
                if self._audio_buffer:
                    await self._ws.send_bytes(bytes(self._audio_buffer))
                    self._audio_buffer.clear()

                finish_msg = {"type": "input_audio_buffer.speech_stopped"}
                try:
                    await self._ws.send_str(json.dumps(finish_msg))
                except Exception as e:
                    logger.warning(f"[OpenAISTT] Error sending finish: {e}")

            if self._recv_task and not self._recv_task.done():
                try:
                    await asyncio.wait_for(self._recv_task, timeout=5.0)
                except asyncio.TimeoutError:
                    self._recv_task.cancel()

        except Exception as e:
            logger.error(f"[OpenAISTT] _run error: {e}")
        finally:
            if self._ws and not self._ws.closed:
                await self._ws.close()
            if self._session:
                await self._session.close()
                self._session = None
            self._ws = None

    async def _connect(self) -> None:
        """建立 WebSocket 连接"""
        self._session = aiohttp.ClientSession()

        ws_url = f"{self._stt._base_url.replace('http', 'ws')}/audio/transcriptions/stream"
        try:
            self._ws = await self._session.ws_connect(
                ws_url,
                headers={"Authorization": f"bearer {self._stt._api_key}"},
                timeout=aiohttp.ClientWSTimeout(ws_close=10.0),
            )

            init_msg = {
                "type": "session开始",
                "audio_model": self._stt._model,
                "language": self._language,
            }
            await self._ws.send_str(json.dumps(init_msg))
            logger.info(f"[OpenAISTT] Connected: {self._task_id}")

        except Exception as e:
            logger.error(f"[OpenAISTT] WebSocket connect failed: {e}")
            if self._session:
                await self._session.close()
                self._session = None
            raise

    async def _recv_loop(self) -> None:
        """持续接收识别结果"""
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    import json
                    data = json.loads(msg.data)
                    event_type = data.get("type", "")

                    if event_type == "transcription":
                        text = data.get("text", "")
                        if text:
                            speech_event = stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                alternatives=[stt.SpeechData(
                                    language=LanguageCode(self._language),
                                    text=text,
                                    confidence=1.0,
                                )],
                            )
                            self._event_ch.send_nowait(speech_event)

                    elif event_type == "session_stop":
                        break

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"[OpenAISTT] WebSocket error: {self._ws.exception()}")
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"[OpenAISTT] Recv loop error: {e}")
        finally:
            if not self._end_of_stream_sent:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.END_OF_STREAM,
                        alternatives=[],
                    )
                )


import json


def create_stt() -> stt.STT:
    """工厂函数：从环境变量创建 OpenAI 格式 STT 实例。"""
    api_key = os.environ.get("OPENAI_ASR_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_ASR_API_KEY environment variable is required")

    base_url = os.environ.get(
        "OPENAI_ASR_BASE_URL",
        "https://api.openai.com/v1"
    )
    model = os.environ.get("OPENAI_ASR_MODEL", "whisper-1")
    language = os.environ.get("OPENAI_ASR_LANGUAGE", "zh")

    logger.info(f"Creating OpenAI STT: base_url={base_url}, model={model}, language={language}")
    return OpenAISTT(
        api_key=api_key,
        base_url=base_url,
        model=model,
        language=language,
    )