# agent/dashscope_stt.py
"""DashScope Fun-ASR 流式语音识别适配器

通过 WebSocket 对接阿里云 DashScope 实时语音识别 API（模型 fun-asr-realtime）。
继承 livekit.agents.stt.STT，实现 _recognize_impl 和 stream 两个方法。

协议参考：https://help.aliyun.com/model-studio/developer-reference/error-code

音频流：
  LiveKit 管道输入 48kHz PCM → 下采样到 16kHz → WebSocket 发送到 DashScope → 接收识别结果
"""

import asyncio
import json
import logging
import os
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

# DashScope WebSocket 实时 ASR 端点
_DASHSCOPE_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"

# 下采样参数：48kHz → 16kHz
_ASR_SAMPLE_RATE = 16000


def _downsample_48k_to_16k(pcm_bytes: bytes) -> bytes:
    """将 48kHz 16bit mono PCM 下采样到 16kHz。简单每 3 个样本取 1 个。"""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    downsampled = audio[::3]
    return downsampled.tobytes()


class DashScopeSTT(stt.STT):
    """DashScope Fun-ASR 语音识别适配器。

    支持流式识别（streaming）和单次识别（_recognize_impl）。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "fun-asr-realtime",
        language: str = "zh",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
            )
        )
        self._api_key = api_key
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

        # Handle AudioBuffer which can be a single frame or a list of frames
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
            if sample_rate is None:
                sample_rate = _ASR_SAMPLE_RATE
        else:
            sample_rate = buffer.sample_rate
            audio_np = np.frombuffer(bytes(buffer.data), dtype=np.int16)

        if sample_rate != _ASR_SAMPLE_RATE:
            audio_np = audio_np[::sample_rate // _ASR_SAMPLE_RATE]

        pcm_16k = audio_np.tobytes()

        import tempfile
        import wave

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            with wave.open(f, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(_ASR_SAMPLE_RATE)
                wf.writeframes(pcm_16k)

        try:
            base_url = os.environ.get(
                "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            async with aiohttp.ClientSession() as session:
                with open(tmp_path, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('file', f, filename='audio.wav', content_type='audio/wav')
                    data.add_field('model', self._model)
                    data.add_field('language', lang)
                    data.add_field('response_format', 'json')

                    async with session.post(
                        f"{base_url}/audio/transcriptions",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        data=data,
                        timeout=aiohttp.ClientTimeout(total=30.0),
                    ) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            logger.error(f"DashScope ASR error {resp.status}: {error_text}")
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
            os.unlink(tmp_path)

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = APIConnectOptions(),
    ) -> "DashScopeSTTStream":
        """返回流式识别器。"""
        return DashScopeSTTStream(self, language=language, conn_options=conn_options)


class DashScopeSTTStream(stt.RecognizeStream):
    """DashScope 流式语音识别。

    通过 WebSocket 持续发送音频帧，实时接收识别结果。

    工作流程：
    1. _run() 被 base class 自动调用
    2. 建立 WebSocket 连接并发送 run-task
    3. 从 _input_ch 读取 AudioFrame → 发送到 WS
    4. 接收识别结果 → 转换为 SpeechEvent → 通过 _event_ch 输出
    5. _input_ch 关闭时发送 finish-task 结束
    """

    def __init__(
        self,
        stt: DashScopeSTT,
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
        import sys
        print(f"[DashScopeSTT] DEBUG Stream created: stream_id={id(self)}, task_id={self._task_id}", flush=True)
        logger.info(f"[DashScopeSTT] Stream created: stream_id={id(self)}, task_id={self._task_id}")

    async def _run(self) -> None:
        """主循环：从 _input_ch 读取音频帧，发送到 DashScope WebSocket。"""
        import threading
        print(f"[DashScopeSTT] DEBUG _run executing: stream_id={id(self)}, task_id={self._task_id}", flush=True)
        logger.info(f"[DashScopeSTT] _run executing: stream_id={id(self)}, task_id={self._task_id}, thread={threading.current_thread().name}")
        try:
            await self._connect()
            print(f"[DashScopeSTT] DEBUG WebSocket connected: stream_id={id(self)}, task_id={self._task_id}, ws_id={id(self._ws)}", flush=True)
            logger.info(f"[DashScopeSTT] WebSocket connected: stream_id={id(self)}, task_id={self._task_id}, ws_id={id(self._ws)}")
            self._recv_task = asyncio.create_task(self._recv_loop())
            frame_count = 0

            async for item in self._input_ch:
                if isinstance(item, rtc.AudioFrame):
                    # Base class already resampled to 16kHz (sample_rate=16000)
                    pcm_data = item.data
                    if isinstance(pcm_data, memoryview):
                        pcm_data = bytes(pcm_data)

                    self._audio_buffer.extend(pcm_data)
                    frame_count += 1
                    if frame_count % 50 == 0:
                        logger.info(f"[DashScopeSTT] frames received: {frame_count}, buffer: {len(self._audio_buffer)}")
                    # Send every 3200 bytes (100ms @ 16kHz 16bit mono)
                    while len(self._audio_buffer) >= 3200:
                        chunk = bytes(self._audio_buffer[:3200])
                        self._audio_buffer = self._audio_buffer[3200:]
                        if self._ws and not self._ws.closed:
                            await self._ws.send_bytes(chunk)
                else:
                    # FlushSentinel: flush remaining audio buffer
                    if self._audio_buffer and self._ws and not self._ws.closed:
                        await self._ws.send_bytes(bytes(self._audio_buffer))
                        self._audio_buffer.clear()

            # _input_ch closed - send finish-task
            logger.info(f"[DashScopeSTT] _input_ch closed, total frames: {frame_count}, sending finish-task")
            if self._ws and not self._ws.closed:
                if self._audio_buffer:
                    await self._ws.send_bytes(bytes(self._audio_buffer))
                    self._audio_buffer.clear()

                finish_msg = {
                    "header": {
                        "action": "finish-task",
                        "task_id": self._task_id,
                    }
                }
                try:
                    await self._ws.send_str(json.dumps(finish_msg))
                except Exception as e:
                    logger.warning(f"[DashScopeSTT] Error sending finish: {e}")

            # Wait for recv task to complete
            if self._recv_task and not self._recv_task.done():
                try:
                    await asyncio.wait_for(self._recv_task, timeout=5.0)
                except asyncio.TimeoutError:
                    self._recv_task.cancel()

        except Exception as e:
            logger.error(f"[DashScopeSTT] _run error: {e}")
        finally:
            if self._ws and not self._ws.closed:
                await self._ws.close()
            if self._session:
                await self._session.close()
                self._session = None
            self._ws = None
            logger.info("[DashScopeSTT] Stream finished")

    async def _connect(self) -> None:
        """建立 WebSocket 连接并发送 run-task。"""
        self._session = aiohttp.ClientSession()

        try:
            self._ws = await self._session.ws_connect(
                _DASHSCOPE_WS_URL,
                headers={"Authorization": f"bearer {self._stt._api_key}"},
                timeout=aiohttp.ClientWSTimeout(ws_close=10.0),
            )

            run_task_msg = {
                "header": {
                    "action": "run-task",
                    "task_id": self._task_id,
                    "streaming": "duplex",
                },
                "payload": {
                    "task_group": "audio",
                    "task": "asr",
                    "function": "recognition",
                    "model": self._stt._model,
                    "parameters": {
                        "sample_rate": _ASR_SAMPLE_RATE,
                        "format": "pcm",
                        "enable_intermediate_results": True,
                        "enable_punctuation_prediction": True,
                        "enable_inverse_text_normalization": True,
                    },
                    "input": {},
                },
            }

            await self._ws.send_str(json.dumps(run_task_msg))
            logger.info(f"[DashScopeSTT] Connected, task_id={self._task_id}")

        except Exception as e:
            logger.error(f"[DashScopeSTT] WebSocket connect failed: {e}")
            if self._session:
                await self._session.close()
                self._session = None
            raise

    async def _recv_loop(self) -> None:
        """持续接收 WebSocket 识别结果，转换为 SpeechEvent。"""
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    header = data.get("header", {})
                    event = header.get("event", "")

                    logger.info(f"[DashScopeSTT] recv: event={event}, data={data}")

                    if event == "result-generated":
                        payload = data.get("payload", {})
                        output = payload.get("output", {})
                        sentence = output.get("sentence", {})
                        text = sentence.get("text", "")
                        is_sentence_end = sentence.get("sentence_end", False)
                        logger.info(f"[DashScopeSTT] sentence_end check: is_sentence_end={is_sentence_end}, text={repr(text)}")

                        if text:
                            event_type = (
                                stt.SpeechEventType.FINAL_TRANSCRIPT
                                if is_sentence_end
                                else stt.SpeechEventType.INTERIM_TRANSCRIPT
                            )
                            speech_event = stt.SpeechEvent(
                                type=event_type,
                                alternatives=[stt.SpeechData(
                                    language=LanguageCode(self._language),
                                    text=text,
                                    confidence=1.0,
                                )],
                            )
                            logger.info(f"[DashScopeSTT] Sending speech_event: type={event_type}, text={text}")
                            self._event_ch.send_nowait(speech_event)
                            # sentence_end=True means end of user speech → signal end of stream AND end of speech
                            # END_OF_SPEECH triggers VAD reset in audio_recognition.py (update_vad call at line 940)
                            # END_OF_STREAM signals the STT stream is done
                            if is_sentence_end:
                                logger.info(f"[DashScopeSTT] Sending END_OF_SPEECH after final transcript (stream stays open for next turn)")
                                self._event_ch.send_nowait(
                                    stt.SpeechEvent(
                                        type=stt.SpeechEventType.END_OF_SPEECH,
                                        alternatives=[],
                                    )
                                )
                                # NOTE: Do NOT send END_OF_STREAM here!
                                # END_OF_STREAM closes RecognizeStream._input_ch, which causes
                                # _run() to exit and the stream to never accept more frames.
                                # For multi-turn conversations, we need the stream to stay open.
                                # VAD handles turn detection via END_OF_SPEECH from VAD, not STT.
                                self._end_of_stream_sent = True
                        elif is_sentence_end and not text:
                            # Empty sentence_end signals end of turn (silence period)
                            # Don't send END_OF_STREAM here, let VAD handle it
                            pass

                    elif event == "task-failed":
                        error_code = header.get("error_code", "unknown")
                        error_msg = data.get("payload", {}).get("message", "unknown error")
                        logger.error(f"[DashScopeSTT] Task failed: code={error_code}, msg={error_msg}, data={data}")
                        break

                    elif event == "task-finished":
                        logger.info("[DashScopeSTT] Task finished")
                        break

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"[DashScopeSTT] WebSocket error: {self._ws.exception()}")
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"[DashScopeSTT] Recv loop error (expected on stream end): {e}")
        finally:
            if not self._end_of_stream_sent:
                logger.info("[DashScopeSTT] Sending END_OF_STREAM in finally (fallback)")
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.END_OF_STREAM,
                        alternatives=[],
                    )
                )
            if self._ws and not self._ws.closed:
                await self._ws.close()
            if self._session:
                await self._session.close()
                self._session = None
            self._ws = None
            logger.info("[DashScopeSTT] Stream finished")


def create_stt() -> DashScopeSTT:
    """工厂函数：从环境变量创建 DashScope STT 实例。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY environment variable is required")

    model = os.environ.get("DASHSCOPE_ASR_MODEL", "fun-asr-realtime")
    language = os.environ.get("DASHSCOPE_ASR_LANGUAGE", "zh")

    return DashScopeSTT(
        api_key=api_key,
        model=model,
        language=language,
    )
