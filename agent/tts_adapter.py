# agent/tts_adapter.py
"""Qwen3-TTS 适配器 — 对接 TTS 微服务（内部调用 DashScope API），支持流式合成与 24kHz→48kHz 重采样

TTS 微服务现在通过阿里云 DashScope API（模型 qwen3-tts-instruct-flash-realtime-2026-01-22）进行语音合成，
不再需要本地 GPU 模型。微服务对外接口保持不变，Agent 无感知。

SDK v1.5.7 适配：
- synthesize() 返回 ChunkedStream（内部类 QwenTTSChunkedStream）
- stream() 返回 SynthesizeStream（QwenTTSStream）
- _run(output_emitter: AudioEmitter) 接收 AudioEmitter 参数
"""

import asyncio
import logging
import os
import re
import time
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

# LiveKit 内部使用 48kHz，Qwen3-TTS 输出 24kHz
OUTPUT_SAMPLE_RATE = 48000
TTS_SAMPLE_RATE = 24000


def _resample_24k_to_48k(pcm_bytes: bytes, source_rate: int = TTS_SAMPLE_RATE,
                          target_rate: int = OUTPUT_SAMPLE_RATE) -> bytes:
    """将 16-bit PCM 音频从 source_rate 重采样到 target_rate.

    Qwen3-TTS 输出 24kHz/16bit/mono，推流到 LiveKit 需要 48kHz。
    """
    if source_rate == target_rate:
        return pcm_bytes

    # bytes → int16 numpy array
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # scipy 重采样：24k → 48k = 原始 * 2 / 1
    from math import gcd
    g = gcd(up := target_rate, down := source_rate)
    audio_resampled = resample_poly(audio, up // g, down // g)

    # float32 → int16 → bytes
    audio_int16 = np.clip(audio_resampled * 32767, -32768, 32767).astype(np.int16)
    return audio_int16.tobytes()


class QwenTTSAdapter(tts.TTS):
    """Qwen3-TTS 适配器，对接本地 TTS 微服务。

    继承 livekit.agents.tts.TTS，实现 synthesize 和 stream 方法。

    配置参数（通过环境变量或构造参数设置）：
    - max_tts_chunk: 后续分片最大字符数（默认 300，保护 API 限流）
    - first_chunk_min: 首片最小字符数（默认 30，低延迟优先）
    - chunk_wait_sec: 后续分片保底等待秒数（默认 5.0）
    - max_concurrent: 最大并发 TTS 请求数（默认 3）
    """

    def __init__(
        self,
        service_url: str = "http://localhost:8001",
        voice: str = "default",
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
        self._service_url = service_url.rstrip("/")
        self._voice = voice
        self._speed = speed
        self._timeout = timeout
        self._max_tts_chunk = max_tts_chunk
        self._first_chunk_min = first_chunk_min
        self._metrics = metrics

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
    """Qwen3-TTS 流式合成器（并行流水线模式）。

    工作流程：
    1. Agent 通过 push_text() 推送文本（由 base class 管理 _input_ch）
    2. _run() 从 _input_ch 读取文本，按规则切分后并行发送给 TTS 服务
    3. 各分片音频先放入顺序队列，输出线程按分片顺序串行输出到 AudioEmitter
    4. 保证音频播放不乱序：先收到的分片必须先输出，后续分片等前分片完成才输出

    并行策略：
    - 最多 MAX_CONCURRENT_TTS 个 TTS 请求同时进行（保护 API 限流）
    - 每个分片的音频先缓存，输出前检查自己是否在队列最前端
    - 只有在队列前端的分片才能输出，确保播放顺序正确

    配置参数（环境变量）：
    - TTS_MAX_CHUNK: 后续分片最大字符数（默认 300）
    - TTS_FIRST_CHUNK_MIN: 首片最小字符数（默认 30）
    - TTS_CHUNK_WAIT_SEC: 后续分片保底等待秒数（默认 5.0）
    - TTS_MAX_CONCURRENT: 最大并发 TTS 请求数（默认 3）
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
        """并行流水线主循环：边收文本边分片，并行发送，顺序输出。"""
        t0 = time.monotonic()
        logger.info(f"[QwenTTSStream._run] started (parallel pipeline mode)")
        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=OUTPUT_SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )
        output_emitter.start_segment(segment_id=str(uuid.uuid4()))

        # ========== 配置参数 ==========
        MAX_TTS_CHUNK = self._adapter._max_tts_chunk
        FIRST_CHUNK_MIN = self._adapter._first_chunk_min
        MAX_WAIT_SEC = float(os.environ.get("TTS_CHUNK_WAIT_SEC", "5.0"))
        MAX_CONCURRENT_TTS = int(os.environ.get("TTS_MAX_CONCURRENT", "3"))

        session = aiohttp.ClientSession()

        def _maybe_metrics():
            return getattr(self._adapter, '_metrics', None)

        def _record_chunk_done(seq: int, chars: int, send_time: float,
                               success: bool, error_msg: str, bytes_sent: int) -> None:
            """记录 TTS 分片完成时序到 metrics"""
            m = _maybe_metrics()
            if m:
                done_time = time.monotonic()
                m.tts_chunk_done(seq, chars, send_time, done_time, success, error_msg, bytes_sent)

        # ========== 并行流水线核心数据结构 ==========
        chunk_seq = 0                   # 分片序号（单调递增）
        in_flight_tasks: list[asyncio.Task] = []  # 正在进行的 TTS 请求任务
        chunk_buffers: dict[int, tuple[bytes, bool]] = {}  # seq -> (pcm_bytes, has_audio)
        delivered_thru = -1             # 已完成输出的最大序号
        sem = asyncio.Semaphore(MAX_CONCURRENT_TTS)  # 控制最大并发数
        chunk_done_event = asyncio.Event()  # 新分片完成时通知
        chunk_errors: list[Exception] = []
        pcm_bytes_sent = 0              # 累计发送的 PCM 字节数（输出到 emitter 的）

        # ========== 分片发送协程（并行，音频缓存不直接输出） ==========
        async def send_tts_chunk_parallel(text: str, seq: int) -> None:
            """并行发送单个文本分片，音频缓存到 chunk_buffers，不直接输出。"""
            nonlocal pcm_bytes_sent
            task_create_time = time.monotonic()
            send_time = task_create_time  # 在 sem 获取前记录，接近真实发送时间
            metrics = _maybe_metrics()
            chars = len(text)
            bytes_sent = 0
            if metrics:
                metrics.tts_chunk_sent(chars)
            logger.info(f"[QwenTTSStream] [task_id={seq}] task created, "
                        f"chars={chars}, wait_for_sem={task_create_time - t0:.3f}s since _run start")
            try:
                async with sem:  # 控制并发数
                    sem_acquire_time = time.monotonic()
                    logger.info(f"[QwenTTSStream] [task_id={seq}] sem acquired, "
                                f"queue_delay={sem_acquire_time - task_create_time:.3f}s since task_create")
                    async with session.post(
                        f"{self._adapter._service_url}/tts/stream",
                        json={
                            "text": text,
                            "voice": self._adapter._voice,
                            "speed": self._adapter._speed,
                        },
                        timeout=aiohttp.ClientTimeout(total=self._adapter._timeout),
                    ) as resp:
                        req_start_time = time.monotonic()
                        logger.info(f"[QwenTTSStream] [task_id={seq}] HTTP request started, "
                                    f"http_delay={req_start_time - sem_acquire_time:.3f}s")
                        if resp.status != 200:
                            error_text = await resp.text()
                            logger.error(f"[QwenTTSStream] TTS chunk {seq} error {resp.status}: {error_text}")
                            chunk_buffers[seq] = (b"", False)
                            _record_chunk_done(seq, chars, send_time, False, f"HTTP {resp.status}: {error_text[:50]}", 0)
                            chunk_done_event.set()
                            return
                        buffer = b""
                        has_audio = False
                        async for chunk_data in resp.content.iter_chunked(4096):
                            buffer += chunk_data
                            if len(buffer) >= 3840:
                                resampled = _resample_24k_to_48k(buffer)
                                existing = chunk_buffers.get(seq, (b"", False))
                                chunk_buffers[seq] = (existing[0] + resampled, True)
                                bytes_sent += len(resampled)
                                pcm_bytes_sent += len(resampled)
                                buffer = b""
                                has_audio = True
                        if buffer:
                            resampled = _resample_24k_to_48k(buffer)
                            existing = chunk_buffers.get(seq, (b"", False))
                            chunk_buffers[seq] = (existing[0] + resampled, existing[1] or True)
                            bytes_sent += len(resampled)
                            pcm_bytes_sent += len(resampled)
                            has_audio = True
                        if seq not in chunk_buffers:
                            chunk_buffers[seq] = (b"", False)
                    done_time = time.monotonic()
                    is_first = seq == 0
                    if metrics:
                        if is_first:
                            metrics.tts_start()
                    logger.info(f"[QwenTTSStream] TTS chunk {seq} ({chars} chars) done in {done_time - send_time:.3f}s, has_audio={has_audio}, bytes={bytes_sent}")
                    _record_chunk_done(seq, chars, send_time, True, "", bytes_sent)
                    chunk_done_event.set()
            except asyncio.TimeoutError:
                done_time = time.monotonic()
                logger.error(f"[QwenTTSStream] TTS chunk {seq} timed out after {self._adapter._timeout}s")
                chunk_buffers[seq] = (b"", False)
                _record_chunk_done(seq, chars, send_time, False, f"timeout after {self._adapter._timeout}s", 0)
                chunk_done_event.set()
            except Exception as e:
                done_time = time.monotonic()
                logger.error(f"[QwenTTSStream] TTS chunk {seq} error: {e}")
                chunk_errors.append(e)
                chunk_buffers[seq] = (b"", False)
                _record_chunk_done(seq, chars, send_time, False, str(e)[:50], 0)
                chunk_done_event.set()

        # ========== 顺序输出协程（等前沿分片就绪后才输出） ==========
        async def drain_ordered_output() -> None:
            """按分片序号顺序输出：只有序号=delivered_thru+1 的分片到达后才能输出。"""
            nonlocal delivered_thru
            while True:
                next_seq = delivered_thru + 1
                # 如果下一个分片还没到达，等待
                if next_seq not in chunk_buffers:
                    # 没有待输出分片，等待事件触发
                    try:
                        await asyncio.wait_for(chunk_done_event.wait(), timeout=2.0)
                        chunk_done_event.clear()
                    except asyncio.TimeoutError:
                        # 超时：检查是否所有任务都完成了
                        if not in_flight_tasks:
                            # 重新检查：chunk 可能刚好在超时前被添加
                            if next_seq not in chunk_buffers:
                                break  # 确实没有待处理分片，退出
                            # 否则继续处理刚到达的分片
                            continue
                        continue
                    if not in_flight_tasks and next_seq not in chunk_buffers:
                        break
                    continue

                buf, has_audio = chunk_buffers[next_seq]
                if buf:
                    output_emitter.push(buf)
                    output_emitter.flush()
                    # tts_first_audio 在首次实际输出音频时记录
                    m = _maybe_metrics()
                    if m and delivered_thru == -1:
                        m.tts_first_audio()
                    logger.info(f"[QwenTTSStream] delivered chunk {next_seq}, {len(buf)} bytes")
                delivered_thru = next_seq
                # 循环继续处理下一个

        try:
            # ========== 文本分片主循环 ==========
            pending_text = ""
            first_sent = False
            last_send_time = t0

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

                        # 切割文本：优先在句末符处切，最多切 MAX_TTS_CHUNK 字符
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
                            reason = "first" if not first_sent else f"timeout({time_since_last:.1f}s)"
                            seq = chunk_seq
                            chunk_seq += 1
                            task = asyncio.create_task(send_tts_chunk_parallel(send_text, seq))
                            in_flight_tasks.append(task)
                            first_sent = True
                            last_send_time = time.monotonic()
                            logger.info(f"[QwenTTSStream] [task_id={seq}] task created at {last_send_time - t0:.3f}s since _run start, "
                                        f"chars={len(send_text)}, reason={reason}, pending={len(pending_text)}, total_in_flight={len(in_flight_tasks)}")

                        time_since_last = time.monotonic() - last_send_time
                        timeout_trigger = first_sent and time_since_last >= MAX_WAIT_SEC
                        if pending_text:
                            timeout_now = first_sent and time_since_last >= MAX_WAIT_SEC
                            can_send = timeout_now or len(pending_text) >= 30 or bool(re.search(r'[。！？；\n]', pending_text))
                        else:
                            can_send = False

            # 处理剩余文本
            while pending_text.strip():
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
                    seq = chunk_seq
                    chunk_seq += 1
                    task = asyncio.create_task(send_tts_chunk_parallel(send_text, seq))
                    in_flight_tasks.append(task)
                    logger.info(f"[QwenTTSStream] [task_id={seq}] REMAINING task created at "
                                f"{time.monotonic() - t0:.3f}s since _run start, chars={len(send_text)}, total_in_flight={len(in_flight_tasks)}")

            # ========== 并行发送中，启动顺序输出协程 ==========
            output_task = asyncio.create_task(drain_ordered_output())

            if in_flight_tasks:
                await asyncio.gather(*in_flight_tasks, return_exceptions=True)
            in_flight_tasks.clear()

            await output_task

            # 校验：所有分片都已输出
            final_seq = chunk_seq - 1
            if delivered_thru < final_seq:
                logger.warning(f"[QwenTTSStream] Audio chunks lost: delivered {delivered_thru} of {final_seq}")

            total_tts_time = time.monotonic() - t0
            logger.info(f"[QwenTTSStream._run] TTS completed, total time: {total_tts_time:.3f}s, bytes: {pcm_bytes_sent}")
            _m = _maybe_metrics()
            if _m:
                _m.tts_end()

        except asyncio.TimeoutError:
            logger.error(f"[QwenTTSStream._run] TTS stream timed out after {self._adapter._timeout}s")
        except Exception as e:
            logger.error(f"[QwenTTSStream._run] TTS stream error: {e}")
        finally:
            if session:
                await session.close()

        output_emitter.end_segment()
        logger.info(f"[QwenTTSStream._run] finished, total PCM bytes: {pcm_bytes_sent}")