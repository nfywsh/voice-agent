# agent/monitoring/metrics.py
"""Prometheus 指标暴露模块 + per-room per-request 时序追踪

两套接口分工：
- /metrics → Prometheus 集群聚合指标（P50/P95/QPS）
- /stats/rooms/{room_id} → per-room 调试（各环节耗时、卡住检测）

使用方式：
  # 在 agent.py 中初始化
  from monitoring.metrics import MetricsCollector, create_app

  metrics = MetricsCollector()
  metrics_app, _ = create_app(port=8082, collector=metrics)

  # 在业务代码中调用
  metrics.request_start(room_id, user_id)
  metrics.asr_start()
  metrics.asr_final(text)
  metrics.llm_start()
  metrics.llm_first_token()
  metrics.llm_end()
  metrics.tts_start()
  metrics.tts_first_audio()
  metrics.tts_end()
  metrics.request_end(request_id)
"""

import contextvars
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    REGISTRY,
)

# ============================================================
# 指标定义（Prometheus 全局聚合，不分房间）
# ============================================================

LLM_DURATION = Histogram(
    "voice_llm_duration_seconds",
    "LLM 处理时长 (秒)",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
)
LLM_FIRST_TOKEN = Histogram(
    "voice_llm_first_token_seconds",
    "LLM 首 token 延迟 (秒)",
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0),
)
TTS_FIRST_AUDIO = Histogram(
    "voice_tts_first_audio_seconds",
    "TTS 首音频延迟 (秒)",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0),
)
TTS_TOTAL_DURATION = Histogram(
    "voice_tts_total_duration_seconds",
    "TTS 总处理时长 (秒)",
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0),
)
STT_DURATION = Histogram(
    "voice_stt_duration_seconds",
    "STT 识别时长 (秒)",
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
)
VAD_LATENCY = Histogram(
    "voice_vad_latency_seconds",
    "VAD 检测到说话到识别完成延迟",
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
)
SESSION_ACTIVE = Gauge("voice_session_active", "当前活跃会话数")
SESSION_TOTAL = Counter("voice_session_total", "累计会话数")
VAD_TRIGGERED = Counter("voice_vad_triggered_total", "VAD 触发次数", ["is_speech"])
ASR_INTERIM = Counter("voice_asr_interim_total", "ASR 中间结果次数")
ASR_FINAL = Counter("voice_asr_final_total", "ASR 最终识别次数")
ASR_ERROR = Counter("voice_asr_error_total", "ASR 错误次数")
ERROR_TOTAL = Counter("voice_error_total", "错误统计", ["stage"])
TRANSCRIPT_LENGTH = Histogram(
    "voice_transcript_length_chars",
    "识别文本长度 (字符数)",
    buckets=(5, 10, 20, 30, 50, 75, 100, 150, 200),
)


# ============================================================
# Per-request 时序数据结构
# ============================================================

@dataclass
class TTSChunkTrace:
    """单个 TTS 分片的时序"""
    seq: int
    chars: int
    send_time: float          # 发送到 TTS 服务的绝对时间（monotonic）
    done_time: Optional[float] = None   # TTS 完成时间
    success: bool = True
    error_msg: str = ""
    bytes_sent: int = 0       # 发送的 PCM 字节数

    @property
    def duration(self) -> Optional[float]:
        if self.send_time and self.done_time:
            return self.done_time - self.send_time
        return None

    @property
    def timed_out(self) -> bool:
        return not self.success and "timeout" in self.error_msg.lower()

    def to_dict(self, session_start: Optional[float] = None) -> dict:
        """将 monotonic 时间转为相对时间显示

        Args:
            session_start: 会话开始的 monotonic 时间，用于计算相对偏移
        """
        def fmt_absmono(m: float) -> str:
            """monotonic 时间转相对偏移 HH:MM:SS 格式"""
            if session_start is not None:
                return f"{round(m - session_start, 3):+.3f}s"
            return datetime.fromtimestamp(m).strftime("%H:%M:%S")

        send_str = fmt_absmono(self.send_time) if self.send_time else "—"
        done_str = fmt_absmono(self.done_time) if self.done_time else "—"
        dur = f"{round(self.duration, 3)}s" if self.duration else "—"
        if not self.success:
            result = f"failed {self.error_msg[:20]}"
        elif self.bytes_sent == 0:
            result = "failed no_audio"
        else:
            result = "success"
        return {
            "seq": self.seq,
            "chars": self.chars,
            "send_time": send_str,
            "done_time": done_str,
            "duration": dur,
            "result": result,
            "bytes": self.bytes_sent,
        }


@dataclass
class RequestTrace:
    """单次请求的完整时序（VAD 检测到 → TTS 播完）"""
    request_id: str
    room_id: str
    user_id: str
    status: str = "in_progress"  # in_progress | done | error
    created_at: float = field(default_factory=time.monotonic)
    created_at_wall: float = field(default_factory=time.time)  # wall-clock for cross-process display

    asr_start: Optional[float] = None
    asr_end: Optional[float] = None
    asr_text: str = ""

    llm_start: Optional[float] = None
    llm_first_token: Optional[float] = None
    llm_end: Optional[float] = None
    llm_input: str = ""
    llm_output: str = ""

    tts_start: Optional[float] = None
    tts_first_audio: Optional[float] = None
    tts_end: Optional[float] = None
    tts_chunk_count: int = 0
    tts_chars_sent: int = 0
    tts_chunks: list[TTSChunkTrace] = field(default_factory=list)  # 每分片详细时序

    @property
    def asr_duration(self) -> Optional[float]:
        if self.asr_start and self.asr_end:
            return self.asr_end - self.asr_start
        return None

    @property
    def llm_duration(self) -> Optional[float]:
        if self.llm_start and self.llm_end:
            return self.llm_end - self.llm_start
        return None

    @property
    def llm_ttft(self) -> Optional[float]:
        if self.llm_start and self.llm_first_token:
            return self.llm_first_token - self.llm_start
        return None

    @property
    def tts_duration(self) -> Optional[float]:
        if self.tts_start and self.tts_end:
            return self.tts_end - self.tts_start
        return None

    @property
    def tts_ttfb(self) -> Optional[float]:
        if self.tts_start and self.tts_first_audio:
            return self.tts_first_audio - self.tts_start
        return None

    @property
    def e2e_duration(self) -> Optional[float]:
        if self.asr_start and self.tts_end:
            return self.tts_end - self.asr_start
        return None

    def tts_chunk_record(self, seq: int, chars: int, send_time: float,
                          done_time: Optional[float], success: bool,
                          error_msg: str, bytes_sent: int) -> None:
        """记录单个 TTS 分片完成"""
        self.tts_chunks.append(TTSChunkTrace(
            seq=seq, chars=chars, send_time=send_time,
            done_time=done_time, success=success,
            error_msg=error_msg, bytes_sent=bytes_sent,
        ))

    @property
    def e2e_duration(self) -> Optional[float]:
        if self.asr_start and self.tts_end:
            return self.tts_end - self.asr_start
        return None


@dataclass
class RoomStats:
    """房间级别的统计（滚动窗口）"""
    room_id: str
    active_request: Optional[RequestTrace] = None
    recent_traces: list[RequestTrace] = field(default_factory=list)
    total_requests: int = 0
    total_errors: int = 0


# ============================================================
# 卡住检测阈值配置
# ============================================================

STUCK_THRESHOLDS = {
    "asr": 5.0,       # ASR 超过 5s 无结果 → 卡住
    "llm_ttft": 10.0, # LLM 超过 10s 无首 token → 卡住
    "tts": 5.0,       # TTS 超过 5s 无音频 → 卡住
}


def _detect_current_phase(trace: RequestTrace) -> Optional[str]:
    """根据 trace 各环节 timestamp 判断当前处于哪个阶段"""
    if trace.asr_end is None:
        return "asr"
    if trace.llm_end is None:
        if trace.llm_start is not None:
            return "llm"
        return "llm"  # ASR 结束但 LLM 还没开始
    if trace.tts_end is None:
        return "tts"
    return None  # 全部完成


def _detect_stuck(trace: RequestTrace, now: float) -> tuple[Optional[str], float, bool]:
    """检测请求是否卡在某个阶段

    Returns:
        (phase, elapsed_seconds, is_stuck)
    """
    phase = _detect_current_phase(trace)
    if phase is None:
        return None, 0.0, False

    if phase == "asr":
        if trace.asr_start is None:
            return phase, 0.0, False
        elapsed = now - trace.asr_start
        return phase, elapsed, elapsed > STUCK_THRESHOLDS["asr"]
    elif phase == "llm":
        if trace.llm_start is None:
            elapsed = now - (trace.asr_end or trace.created_at)
            return phase, elapsed, elapsed > STUCK_THRESHOLDS["llm_ttft"]
        if trace.llm_first_token is None:
            elapsed = now - trace.llm_start
            return phase, elapsed, elapsed > STUCK_THRESHOLDS["llm_ttft"]
        return phase, now - trace.llm_start, False
    elif phase == "tts":
        if trace.tts_start is None:
            elapsed = now - (trace.llm_end or trace.created_at)
            return phase, elapsed, elapsed > STUCK_THRESHOLDS["tts"]
        if trace.tts_first_audio is None:
            elapsed = now - trace.tts_start
            return phase, elapsed, elapsed > STUCK_THRESHOLDS["tts"]
        return phase, now - trace.tts_start, False
    return None, 0.0, False


def _trace_to_dict(trace: Optional[RequestTrace]) -> Optional[dict]:
    if trace is None:
        return None

    def fmt(monotonic: Optional[float]) -> Optional[str]:
        if monotonic is None:
            return None
        offset = monotonic - (trace.created_at or monotonic)
        return f"{round(offset, 3):+.3f}s"

    def fmt_hhmmss(wall_ts: Optional[float]) -> Optional[str]:
        """wall-clock 时间转 HH:MM:SS 格式"""
        if wall_ts is None:
            return None
        return datetime.fromtimestamp(wall_ts).strftime("%H:%M:%S")

    # 构建 TTS 分片明细表（按 seq 升序排列）
    tts_chunks_table = []
    if trace.tts_chunks:
        for chunk in sorted(trace.tts_chunks, key=lambda c: c.seq):
            tts_chunks_table.append(chunk.to_dict(trace.created_at))
    elif trace.tts_chars_sent > 0:
        tts_chunks_table.append({
            "seq": 0,
            "chars": trace.tts_chars_sent,
            "send_time": "—",
            "done_time": "—",
            "duration": "—",
            "result": "（无分片明细）",
            "bytes": 0,
        })

    return {
        "request_id": trace.request_id,
        "status": trace.status,
        "created_at": fmt_hhmmss(trace.created_at_wall),
        "e2e_duration": f"{round(trace.e2e_duration, 2)}s" if trace.e2e_duration else "—",
        "asr": {
            "start": fmt(trace.asr_start),
            "end": fmt(trace.asr_end),
            "duration": f"{round(trace.asr_duration, 3)}s" if trace.asr_duration else "—",
            "text": trace.asr_text[:80] + "..." if len(trace.asr_text) > 80 else trace.asr_text,
        },
        "llm": {
            "start": fmt(trace.llm_start),
            "first_token": fmt(trace.llm_first_token),
            "end": fmt(trace.llm_end),
            "duration": f"{round(trace.llm_duration, 3)}s" if trace.llm_duration else "—",
            "ttft": f"{round(trace.llm_ttft, 3):+.3f}s" if trace.llm_ttft else "—",
            "input": trace.llm_input[:200] + "..." if len(trace.llm_input) > 200 else trace.llm_input,
            "output": trace.llm_output[:300] + "..." if len(trace.llm_output) > 300 else trace.llm_output,
        },
        "tts": {
            "start": fmt(trace.tts_start),
            "first_audio": fmt(trace.tts_first_audio),
            "end": fmt(trace.tts_end),
            "duration": f"{round(trace.tts_duration, 3)}s" if trace.tts_duration else "—",
            "ttfb": f"{round(trace.tts_ttfb, 3):+.3f}s" if trace.tts_ttfb else "—",
            "chunk_count": trace.tts_chunk_count,
            "chars_sent": trace.tts_chars_sent,
            "chunks_table": tts_chunks_table,
        },
    }


def _fmt_ts(monotonic: Optional[float], session_start: Optional[float] = None) -> Optional[str]:
    """将 monotonic 时间转为相对于 session 开始的秒数字符串"""
    if monotonic is None:
        return None
    # 用 session_start 做基准，没有则直接显示原值
    offset = monotonic - (session_start or monotonic)
    return f"{round(offset, 3):+.3f}s"


# ============================================================
# MetricsCollector - agent 业务代码调用
# ============================================================

class MetricsCollector:
    """指标收集器，同时支持 Prometheus 全局聚合和 per-request 时序追踪"""

    __slots__ = (
        "_session_start",
        "_session_start_wall",  # wall-clock time.time()，用于 REST API 显示时分秒
        "_request_traces",
        "_room_stats",
        "_current_request_id",
    )

    def __init__(self):
        self._session_start: Optional[float] = None
        self._session_start_wall: Optional[float] = None  # time.time() 对应的绝对时间
        # Per-request 时序存储
        self._request_traces: dict[str, RequestTrace] = {}
        self._room_stats: dict[str, RoomStats] = {}
        # contextvars 用于在 async 调用链中传递当前 request_id
        self._current_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
            "current_request_id", default=""
        )

    # ---- Prometheus 全局指标（兼容旧接口）----
    def session_start(self) -> None:
        # 避免重复调用：THREAD 模式下多个 job 共享同一 process，
        # 后续 job 不应重置 session_start，否则旧 job trace 的相对时间会变成负数
        if self._session_start is not None:
            return
        self._session_start = time.monotonic()
        self._session_start_wall = time.time()
        SESSION_ACTIVE.inc()
        SESSION_TOTAL.inc()

    def session_end(self) -> None:
        SESSION_ACTIVE.dec()
        self._session_start = None
        self._session_start_wall = None

    def vad_triggered(self, is_speech: bool) -> None:
        """记录 VAD 触发（Prometheus 全局）并更新当前请求的 asr_start"""
        VAD_TRIGGERED.labels(is_speech=str(is_speech).lower()).inc()
        if is_speech:
            # 尝试更新当前请求的 ASR 开始时间
            request_id = self._current_request_id.get()
            if request_id:
                trace = self._request_traces.get(request_id)
                if trace and trace.asr_start is None:
                    trace.asr_start = time.monotonic()

    def stt_start(self) -> None:
        """兼容旧接口，已被 request_start + asr_start 替代"""

    def stt_interim(self, text: str) -> None:
        ASR_INTERIM.inc()
        TRANSCRIPT_LENGTH.observe(len(text))

    def stt_final(self, text: str) -> None:
        ASR_FINAL.inc()
        TRANSCRIPT_LENGTH.observe(len(text))
        request_id = self._current_request_id.get()
        if request_id:
            trace = self._request_traces.get(request_id)
            if trace:
                now = time.monotonic()
                if trace.asr_end is None:
                    trace.asr_end = now
                # 如果 asr_start 还没设置（VAD 未触发），则用 ass_end 倒推一个合理值
                if trace.asr_start is None:
                    trace.asr_start = now - 1.0  # 假设 ASR 耗时约 1s
                # VAD 会把用户语音切分成多段，每段都会触发 stt_final，
                # 用拼接而不是覆盖来保留完整内容
                if trace.asr_text:
                    trace.asr_text += " " + text
                else:
                    trace.asr_text = text

    def stt_error(self) -> None:
        ASR_ERROR.inc()
        ERROR_TOTAL.labels(stage="stt").inc()

    def llm_start(self) -> None:
        request_id = self._current_request_id.get()
        if request_id:
            trace = self._request_traces.get(request_id)
            if trace and trace.llm_start is None:
                trace.llm_start = time.monotonic()

    def llm_first_token(self) -> None:
        request_id = self._current_request_id.get()
        if request_id:
            trace = self._request_traces.get(request_id)
            if trace and trace.llm_first_token is None:
                trace.llm_first_token = time.monotonic()
                # 同时记录 Prometheus 全局指标
                if trace.llm_start:
                    LLM_FIRST_TOKEN.observe(time.monotonic() - trace.llm_start)

    def llm_end(self) -> None:
        request_id = self._current_request_id.get()
        if request_id:
            trace = self._request_traces.get(request_id)
            if trace:
                trace.llm_end = time.monotonic()
                # 记录 Prometheus 全局 LLM 时长
                if trace.llm_start:
                    LLM_DURATION.observe(time.monotonic() - trace.llm_start)
                if trace.asr_start:
                    VAD_LATENCY.observe(time.monotonic() - trace.asr_start)

    def llm_error(self) -> None:
        ERROR_TOTAL.labels(stage="llm").inc()

    def llm_input(self, text: str) -> None:
        request_id = self._current_request_id.get()
        if request_id:
            trace = self._request_traces.get(request_id)
            if trace:
                trace.llm_input = text

    def llm_output(self, text: str) -> None:
        request_id = self._current_request_id.get()
        if request_id:
            trace = self._request_traces.get(request_id)
            if trace:
                trace.llm_output += text

    def tts_start(self) -> None:
        request_id = self._current_request_id.get()
        if request_id:
            trace = self._request_traces.get(request_id)
            if trace and trace.tts_start is None:
                trace.tts_start = time.monotonic()

    def tts_first_audio(self) -> None:
        request_id = self._current_request_id.get()
        if request_id:
            trace = self._request_traces.get(request_id)
            if trace and trace.tts_first_audio is None:
                trace.tts_first_audio = time.monotonic()
                if trace.tts_start:
                    TTS_FIRST_AUDIO.observe(time.monotonic() - trace.tts_start)

    def tts_end(self) -> None:
        request_id = self._current_request_id.get()
        if request_id:
            trace = self._request_traces.get(request_id)
            if trace:
                trace.tts_end = time.monotonic()
                if trace.tts_start:
                    TTS_TOTAL_DURATION.observe(time.monotonic() - trace.tts_start)

    def tts_error(self) -> None:
        ERROR_TOTAL.labels(stage="tts").inc()

    def error(self, stage: str) -> None:
        ERROR_TOTAL.labels(stage=stage).inc()

    # ---- Per-request 时序追踪（新增）----
    def request_start(self, room_id: str, user_id: str) -> str:
        """开始一个请求，返回 request_id"""
        # 清理同一房间的旧 trace（job restart 时，上一个 job 的 active_request 需要结束）
        # 旧 job 的 LLM/TTS 回调可能还在异步执行，这些回调会继续写到旧 trace
        # （找到就写，找不到就静默失败，不影响业务）
        rs = self._room_stats.get(room_id)
        if rs and rs.active_request:
            stale = rs.active_request
            # 只清理不在 _request_traces 中的（已在别的 job 的 finally 中处理过）
            if stale.request_id in self._request_traces:
                stale.status = "interrupted"
                rs.active_request = None
                rs.total_requests += 1
                rs.recent_traces.append(stale)
                if len(rs.recent_traces) > 10:
                    rs.recent_traces = rs.recent_traces[-10:]
                self._request_traces.pop(stale.request_id, None)
                logger.info(f"[metrics] Cleaned up stale trace {stale.request_id} for room={room_id}")

        request_id = str(uuid.uuid4())[:8]
        trace = RequestTrace(
            request_id=request_id,
            room_id=room_id,
            user_id=user_id,
        )
        self._request_traces[request_id] = trace
        self._room_stats.setdefault(room_id, RoomStats(room_id=room_id))
        self._room_stats[room_id].active_request = trace
        self._current_request_id.set(request_id)
        return request_id

    def request_end(self, request_id: str, status: str = "done") -> None:
        """请求结束，移出活跃状态"""
        trace = self._request_traces.get(request_id)
        if not trace:
            # trace 可能已通过其他 request_id 提前移到 recent_traces（如并发 job 互相覆盖）
            # 检查 room_stats 中是否已有该 request_id 的 trace
            for rs in self._room_stats.values():
                for t in rs.recent_traces:
                    if t.request_id == request_id:
                        return  # 已记录过，跳过
            return
        trace.status = status
        if trace.room_id in self._room_stats:
            rs = self._room_stats[trace.room_id]
            rs.active_request = None
            rs.total_requests += 1
            if status == "error":
                rs.total_errors += 1
            rs.recent_traces.append(trace)
            # 保留最近 10 条
            if len(rs.recent_traces) > 10:
                rs.recent_traces = rs.recent_traces[-10:]
        # 清除 contextvars
        current = self._current_request_id.get()
        if current == request_id:
            self._current_request_id.set("")
        # 记录 Prometheus 全局错误
        if status == "error":
            ERROR_TOTAL.labels(stage="request").inc()

    def asr_start(self) -> None:
        """ASR 开始识别（由 user_state_changed speaking 触发）"""
        request_id = self._current_request_id.get()
        if request_id:
            trace = self._request_traces.get(request_id)
            if trace and trace.asr_start is None:
                trace.asr_start = time.monotonic()

    def asr_final(self, text: str) -> None:
        """ASR 最终识别完成"""
        request_id = self._current_request_id.get()
        if request_id:
            trace = self._request_traces.get(request_id)
            if trace:
                trace.asr_end = time.monotonic()
                # 每轮对话独立 trace，ASR 结果直接覆盖（不再拼接）
                trace.asr_text = text

    def asr_interim(self, text: str) -> None:
        """ASR 中间结果（覆盖而非拼接，用于调试）"""
        request_id = self._current_request_id.get()
        if request_id:
            trace = self._request_traces.get(request_id)
            if trace:
                trace.asr_text = text  # 覆盖

    def tts_chunk_sent(self, chars: int) -> None:
        """每次 TTS 分片发送时调用"""
        request_id = self._current_request_id.get()
        if request_id:
            trace = self._request_traces.get(request_id)
            if trace:
                trace.tts_chunk_count += 1
                trace.tts_chars_sent += chars

    def tts_chunk_done(self, seq: int, chars: int, send_time: float,
                        done_time: float, success: bool,
                        error_msg: str, bytes_sent: int) -> None:
        """TTS 分片完成时调用，记录详细时序"""
        request_id = self._current_request_id.get()
        if request_id:
            trace = self._request_traces.get(request_id)
            if trace:
                trace.tts_chunk_record(seq, chars, send_time, done_time,
                                        success, error_msg, bytes_sent)

    # ---- Per-room 查询接口（供 REST API 调用）----
    def get_room_stats(self, room_id: str) -> dict:
        """获取房间当前状态和最近请求"""
        rs = self._room_stats.get(room_id)
        if not rs:
            return {
                "room_id": room_id,
                "active_requests": [],
                "recent_traces": [],
                "total_requests": 0,
                "total_errors": 0,
            }
        # active_requests: 只展示当前 active_request（entrypoint 级别的当前请求）
        # 不展示 _request_traces 中其他的（那些是旧 job 的 trace，回调可能还在写）
        all_active = []
        if rs.active_request and rs.active_request.status == "in_progress":
            if _is_meaningful_trace(rs.active_request):
                all_active.append(_trace_to_dict(rs.active_request))
        # recent_traces: 只展示已结束的 trace（status != in_progress）
        recent = [t for t in rs.recent_traces if t.status != "in_progress"]
        recent = recent[-5:]
        recent = [t for t in recent if _is_meaningful_trace(t)]
        return {
            "room_id": room_id,
            "active_requests": all_active,
            "recent_traces": [_trace_to_dict(t) for t in recent],
            "total_requests": rs.total_requests,
            "total_errors": rs.total_errors,
        }

    def get_active_request(self, room_id: str) -> Optional[RequestTrace]:
        """获取房间当前正在处理的请求"""
        rs = self._room_stats.get(room_id)
        if not rs:
            return None
        return rs.active_request

    def get_current_request_id(self) -> str:
        """获取当前上下文的 request_id（供 tts_adapter 等内部调用）"""
        return self._current_request_id.get()


def _is_meaningful_trace(t: RequestTrace) -> bool:
    """判断 trace 是否有实际业务数据（过滤 THREAD 池幽灵条目）"""
    return bool(
        t.asr_text
        or t.asr_start is not None
        or t.llm_start is not None
        or t.tts_start is not None
        or t.status == "error"
    )


# ============================================================
# FastAPI app - 暴露 /metrics 和 /stats 端点
# ============================================================

def create_app(port: int = 8082, collector: MetricsCollector = None):
    """创建 FastAPI app，暴露监控端点

    Args:
        port: 监听端口，默认 8082
        collector: MetricsCollector 实例（会被 FastAPI app 持有用于 REST 查询）
    """
    from fastapi import FastAPI, Response

    app = FastAPI(title="Voice Agent Metrics")
    _collector = collector if collector is not None else MetricsCollector()

    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/stats")
    async def stats():
        """全局统计（Prometheus 指标的直接映射）"""
        def get_histogram_stats(histogram):
            try:
                count = int(histogram._buckets[-1].get())
                if count == 0:
                    return {"count": 0, "p50": 0, "p95": 0, "p99": 0}
                buckets = getattr(histogram, '_buckets', [])
                upper_bounds = getattr(histogram, '_upper_bounds', [])
                if not buckets or not upper_bounds:
                    return {"count": count, "p50": 0, "p95": 0, "p99": 0}
                bound_counts = {}
                for bound, bucket in zip(upper_bounds, buckets):
                    bound_counts[bound] = bucket.get()
                cumsum = 0
                p50 = p95 = p99 = 0
                for bound, bucket_count in sorted(bound_counts.items()):
                    cumsum += bucket_count
                    if p50 == 0 and cumsum >= count * 0.50:
                        p50 = bound
                    if p95 == 0 and cumsum >= count * 0.95:
                        p95 = bound
                    if p99 == 0 and cumsum >= count * 0.99:
                        p99 = bound
                        break
                return {"count": count, "p50": round(p50, 3), "p95": round(p95, 3), "p99": round(p99, 3)}
            except Exception:
                return {"count": 0, "p50": 0, "p95": 0, "p99": 0}

        def get_counter_value(counter):
            try:
                val = getattr(counter, '_value', None)
                if val is None:
                    return 0
                return float(val.get()) if hasattr(val, 'get') else float(val)
            except Exception:
                return 0

        def get_gauge_value(gauge):
            try:
                val = getattr(gauge, '_value', None)
                if val is None:
                    return 0
                return float(val.get()) if hasattr(val, 'get') else float(val)
            except Exception:
                return 0

        # 列出最近有活动的房间
        recent_rooms = list(_collector._room_stats.keys())[-10:]

        return {
            "timestamp": datetime.now().isoformat(),
            "recent_rooms": recent_rooms,
            "session": {
                "active": get_gauge_value(SESSION_ACTIVE),
                "total": get_counter_value(SESSION_TOTAL),
            },
            "llm": {
                "duration": get_histogram_stats(LLM_DURATION),
                "first_token": get_histogram_stats(LLM_FIRST_TOKEN),
            },
            "tts": {
                "first_audio": get_histogram_stats(TTS_FIRST_AUDIO),
                "total_duration": get_histogram_stats(TTS_TOTAL_DURATION),
            },
            "stt": {
                "duration": get_histogram_stats(STT_DURATION),
                "interim_count": get_counter_value(ASR_INTERIM),
                "final_count": get_counter_value(ASR_FINAL),
                "error_count": get_counter_value(ASR_ERROR),
            },
            "vad": {
                "latency": get_histogram_stats(VAD_LATENCY),
                "triggered_speech": get_counter_value(VAD_TRIGGERED.labels(is_speech="true")),
                "triggered_silence": get_counter_value(VAD_TRIGGERED.labels(is_speech="false")),
            },
            "errors": {
                "llm": get_counter_value(ERROR_TOTAL.labels(stage="llm")),
                "tts": get_counter_value(ERROR_TOTAL.labels(stage="tts")),
                "stt": get_counter_value(ERROR_TOTAL.labels(stage="stt")),
                "vad": get_counter_value(ERROR_TOTAL.labels(stage="vad")),
                "connection": get_counter_value(ERROR_TOTAL.labels(stage="connection")),
            },
        }

    @app.get("/stats/rooms/{room_id}")
    async def room_stats(room_id: str):
        """该房间当前状态和最近请求的各环节耗时"""
        return _collector.get_room_stats(room_id)

    @app.get("/stats/rooms/{room_id}/active")
    async def room_active(room_id: str):
        """该房间正在处理中的请求，各环节是否卡住"""
        trace = _collector.get_active_request(room_id)
        if not trace:
            return {"room_id": room_id, "active_request": None}
        now = time.monotonic()
        phase, elapsed, is_stuck = _detect_stuck(trace, now)
        threshold = STUCK_THRESHOLDS.get(phase, 999) if phase else 0
        return {
            "room_id": room_id,
            "active_request": {
                "request_id": trace.request_id,
                "status": trace.status,
                "phase": phase,
                "elapsed": {
                    "asr": round(now - trace.asr_start, 3) if trace.asr_start else None,
                    "llm": round(now - trace.llm_start, 3) if trace.llm_start else None,
                    "tts": round(now - trace.tts_start, 3) if trace.tts_start else None,
                },
                "stuck_detection": {
                    "phase": phase,
                    "waited_seconds": round(elapsed, 3),
                    "threshold_seconds": threshold,
                    "is_stuck": is_stuck,
                },
            },
        }

    return app, _collector