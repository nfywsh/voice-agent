# agent/dashscope_stt.py
"""DashScope Fun-ASR 流式语音识别适配器

通过 WebSocket 对接阿里云 DashScope 实时语音识别 API（模型 fun-asr-2025-11-07）。
继承 livekit.agents.stt.STT，实现 recognize 和 stream 两个方法。

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

logger = logging.getLogger(__name__)

# DashScope WebSocket 实时 ASR 端点
_DASHSCOPE_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/inference"

# 下采样参数：48kHz → 16kHz
_INPUT_SAMPLE_RATE = 48000
_ASR_SAMPLE_RATE = 16000


def _downsample_48k_to_16k(pcm_bytes: bytes) -> bytes:
    """将 48kHz 16bit mono PCM 下采样到 16kHz。简单每 3 个样本取 1 个。"""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    # 48k → 16k = 取每第3个样本
    downsampled = audio[::3]
    return downsampled.tobytes()


class DashScopeSTT(stt.STT):
    """DashScope Fun-ASR 语音识别适配器。

    支持流式识别（streaming）和单次识别（recognize）。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "fun-asr-2025-11-07",
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

    async def recognize(
        self,
        buffer: rtc.AudioBuffer,
        *,
        language: str | None = None,
    ) -> stt.SpeechEvent:
        """非流式识别：发送完整音频，返回识别结果。

        使用 DashScope 的 HTTP 接口一次性识别。
        """
        lang = language or self._language

        # 获取 16kHz PCM 数据
        frames = buffer.data
        if hasattr(buffer, 'sample_rate') and buffer.sample_rate != _ASR_SAMPLE_RATE:
            # 下采样
            audio_np = np.array(frames, dtype=np.float32)
            if buffer.sample_rate == _INPUT_SAMPLE_RATE:
                audio_np = audio_np[::_INPUT_SAMPLE_RATE // _ASR_SAMPLE_RATE]
            pcm_16k = (audio_np * 32767).astype(np.int16).tobytes()
        else:
            pcm_16k = np.array(frames, dtype=np.int16).tobytes()

        # 使用 DashScope OpenAI 兼容接口（非流式）
        # DashScope 的 /v1/audio/transcriptions 端点
        import tempfile
        import wave

        # 写入临时 WAV 文件
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
                                alternatives=[stt.SpeechData(text="", confidence=0.0)],
                            )

                        result = await resp.json()
                        text = result.get("text", "")
                        return stt.SpeechEvent(
                            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                            alternatives=[stt.SpeechData(text=text, confidence=1.0)],
                        )
        finally:
            os.unlink(tmp_path)

    def stream(self) -> "DashScopeSTTStream":
        """返回流式识别器。"""
        return DashScopeSTTStream(self)


class DashScopeSTTStream(stt.SpeechStream):
    """DashScope 流式语音识别。

    通过 WebSocket 持续发送音频帧，实时接收识别结果。

    工作流程：
    1. 建立 WebSocket 连接
    2. 发送 run-task 消息启动识别任务
    3. push_frame() 推入音频帧 → 下采样到 16kHz → 通过 WS 发送
    4. 接收识别结果 → 转换为 SpeechEvent → 通过 _event_queue 输出
    5. finish() 发送 finish-task 结束
    """

    def __init__(self, stt: DashScopeSTT):
        super().__init__(stt)
        self._stt = stt
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._task_id = str(uuid.uuid4())
        self._audio_buffer = bytearray()
        self._recv_task: Optional[asyncio.Task] = None

    async def _connect(self) -> None:
        """建立 WebSocket 连接并发送 run-task。"""
        self._session = aiohttp.ClientSession()

        try:
            self._ws = await self._session.ws_connect(
                _DASHSCOPE_WS_URL,
                headers={"Authorization": f"Bearer {self._stt._api_key}"},
                timeout=aiohttp.ClientWSTimeout(ws_close=10.0),
            )

            # 发送 run-task 启动识别
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
                },
            }

            await self._ws.send_str(json.dumps(run_task_msg))
            logger.info(f"[DashScopeSTT] Connected, task_id={self._task_id}")

            # 启动接收协程
            self._recv_task = asyncio.create_task(self._recv_loop())

        except Exception as e:
            logger.error(f"[DashScopeSTT] WebSocket connect failed: {e}")
            if self._session:
                await self._session.close()
                self._session = None
            raise

    async def _recv_loop(self) -> None:
        """持续接收 WebSocket 识别结果，转换为 SpeechEvent。”"""
        if not self._ws:
            return

        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    header = data.get("header", {})
                    event = header.get("event", "")

                    if event == "result-generated":
                        payload = data.get("payload", {})
                        output = payload.get("output", {})
                        sentence = output.get("sentence", {})
                        text = sentence.get("text", "")
                        is_final = payload.get("is_final", False)

                        if text:
                            event_type = (
                                stt.SpeechEventType.FINAL_TRANSCRIPT
                                if is_final
                                else stt.SpeechEventType.INTERIM_TRANSCRIPT
                            )
                            speech_event = stt.SpeechEvent(
                                type=event_type,
                                alternatives=[stt.SpeechData(text=text, confidence=1.0)],
                            )
                            self._event_queue.put_nowait(speech_event)

                    elif event == "task-failed":
                        error_msg = data.get("payload", {}).get("message", "unknown error")
                        logger.error(f"[DashScopeSTT] Task failed: {error_msg}")
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
            logger.error(f"[DashScopeSTT] Recv loop error: {e}")
        finally:
            # 通知流结束
            self._event_queue.put_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_STREAM,
                    alternatives=[],
                )
            )

    async def push_frame(self, frame: rtc.AudioFrame) -> None:
        """推入音频帧（48kHz），下采样到 16kHz 后发送到 DashScope。"""
        # 延迟建立连接（首次 push_frame 时才连接）
        if self._ws is None:
            await self._connect()

        # 下采样 48kHz → 16kHz
        pcm_48k = frame.data
        if isinstance(pcm_48k, memoryview):
            pcm_48k = bytes(pcm_48k)

        pcm_16k = _downsample_48k_to_16k(pcm_48k)

        # 累积到一定量再发送（减少 WebSocket 消息频率）
        self._audio_buffer.extend(pcm_16k)

        # 每 3200 字节发送一次（100ms @ 16kHz 16bit mono）
        if len(self._audio_buffer) >= 3200:
            chunk = bytes(self._audio_buffer[:3200])
            self._audio_buffer = self._audio_buffer[3200:]

            if self._ws and not self._ws.closed:
                await self._ws.send_bytes(chunk)

    async def finish(self) -> None:
        """结束流式识别。"""
        # 发送剩余缓冲区
        if self._audio_buffer and self._ws and not self._ws.closed:
            await self._ws.send_bytes(bytes(self._audio_buffer))
            self._audio_buffer.clear()

        # 发送 finish-task
        if self._ws and not self._ws.closed:
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

        # 等待接收协程结束
        if self._recv_task and not self._recv_task.done():
            try:
                await asyncio.wait_for(self._recv_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._recv_task.cancel()

        # 关闭连接
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

    model = os.environ.get("DASHSCOPE_ASR_MODEL", "fun-asr-2025-11-07")
    language = os.environ.get("DASHSCOPE_ASR_LANGUAGE", "zh")

    return DashScopeSTT(
        api_key=api_key,
        model=model,
        language=language,
    )
