# agent/monitoring/metrics.py
"""Prometheus 指标暴露模块

使用方式：
  # 在 agent.py 中初始化
  from monitoring.metrics import MetricsCollector, create_app

  metrics_app, metrics_collector = create_app(port=8081)
  # 在各阶段调用 metrics_collector.record_xxx()

  # FastAPI app 会自动暴露 /metrics 端点
  # Prometheus 从 :8081/metrics 拉取

指标列表:
- voice_llm_duration_seconds: LLM 处理时长
- voice_llm_first_token_seconds: LLM 首 token 延迟
- voice_tts_first_audio_seconds: TTS 首音频延迟
- voice_tts_total_duration_seconds: TTS 总时长
- voice_stt_duration_seconds: STT 识别时长
- voice_vad_latency_seconds: VAD检测到说话→LLM完成
- voice_vad_triggered_total: VAD 触发次数 (is_speech 标签)
- voice_session_active: 当前活跃会话数 (Gauge)
- voice_session_total: 累计会话数 (Counter)
- voice_asr_interim_total: ASR 中间结果次数
- voice_asr_final_total: ASR 最终识别次数
- voice_asr_error_total: ASR 错误次数
- voice_error_total: 各阶段错误 (stage=llm|tts|stt|vad|connection)
- voice_transcript_length_chars: 识别文本长度分布
"""

import time
from typing import Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY,
)

# ============================================================
# 指标定义
# ============================================================

LLM_DURATION = Histogram(
    "voice_llm_duration_seconds",
    "LLM 处理时长 (秒)",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
)
LLM_FIRST_TOKEN = Histogram(
    "voice_llm_first_token_seconds",
    "LLM 首 token 延迟 (秒)",
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.5, 1.0),
)
TTS_FIRST_AUDIO = Histogram(
    "voice_tts_first_audio_seconds",
    "TTS 首音频延迟 (秒)",
    buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0),
)
TTS_TOTAL_DURATION = Histogram(
    "voice_tts_total_duration_seconds",
    "TTS 总处理时长 (秒)",
    buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
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
# MetricsCollector - agent 业务代码调用
# ============================================================

class MetricsCollector:
    """指标收集器，在 agent 业务代码中各阶段调用"""

    __slots__ = (
        "_session_start", "_llm_start", "_tts_start", "_stt_start",
        "_vad_speech_time", "_asr_text",
    )

    def __init__(self):
        self._session_start: Optional[float] = None
        self._llm_start: Optional[float] = None
        self._tts_start: Optional[float] = None
        self._stt_start: Optional[float] = None
        self._vad_speech_time: Optional[float] = None
        self._asr_text: str = ""

    # ---- Session ----
    def session_start(self) -> None:
        self._session_start = time.monotonic()
        SESSION_ACTIVE.inc()
        SESSION_TOTAL.inc()

    def session_end(self) -> None:
        SESSION_ACTIVE.dec()
        for attr in self.__slots__:
            setattr(self, attr, None)

    # ---- VAD ----
    def vad_triggered(self, is_speech: bool) -> None:
        VAD_TRIGGERED.labels(is_speech=str(is_speech).lower()).inc()
        if is_speech:
            self._vad_speech_time = time.monotonic()

    # ---- STT ----
    def stt_start(self) -> None:
        self._stt_start = time.monotonic()
        self._asr_text = ""

    def stt_interim(self, text: str) -> None:
        ASR_INTERIM.inc()
        self._asr_text = text
        TRANSCRIPT_LENGTH.observe(len(text))

    def stt_final(self, text: str) -> None:
        ASR_FINAL.inc()
        self._asr_text = text
        if self._stt_start:
            STT_DURATION.observe(time.monotonic() - self._stt_start)
        TRANSCRIPT_LENGTH.observe(len(text))

    def stt_error(self) -> None:
        ASR_ERROR.inc()
        ERROR_TOTAL.labels(stage="stt").inc()

    # ---- LLM ----
    def llm_start(self) -> None:
        self._llm_start = time.monotonic()

    def llm_first_token(self) -> None:
        if self._llm_start:
            LLM_FIRST_TOKEN.observe(time.monotonic() - self._llm_start)

    def llm_end(self) -> None:
        if self._llm_start:
            LLM_DURATION.observe(time.monotonic() - self._llm_start)
            if self._vad_speech_time:
                VAD_LATENCY.observe(time.monotonic() - self._vad_speech_time)
            self._llm_start = None

    def llm_error(self) -> None:
        ERROR_TOTAL.labels(stage="llm").inc()

    # ---- TTS ----
    def tts_start(self) -> None:
        self._tts_start = time.monotonic()

    def tts_first_audio(self) -> None:
        if self._tts_start:
            TTS_FIRST_AUDIO.observe(time.monotonic() - self._tts_start)

    def tts_end(self) -> None:
        if self._tts_start:
            TTS_TOTAL_DURATION.observe(time.monotonic() - self._tts_start)
            self._tts_start = None

    def tts_error(self) -> None:
        ERROR_TOTAL.labels(stage="tts").inc()

    # ---- 通用 ----
    def error(self, stage: str) -> None:
        ERROR_TOTAL.labels(stage=stage).inc()


# ============================================================
# FastAPI app - 暴露 /metrics 端点
# ============================================================

def create_app(port: int = 8082, collector: MetricsCollector = None):
    """创建 FastAPI app，暴露 /metrics 和 /health 端点

    Args:
        port: 监听端口，默认 8082 (区别于 LiveKit 内部 8081)
        collector: MetricsCollector 实例，若不传则使用全局注册的指标（推荐）
    """
    from fastapi import FastAPI, Response

    app = FastAPI(title="Voice Agent Metrics")
    # 若传入 collector 则使用传入的实例；否则创建新的（统计数据会为 0，
    # 因为实际的 MetricsCollector 在 entrypoint 中是另一个实例，
    # 但 Prometheus 指标是模块级单例，所以 /metrics 端点不受影响）
    _collector = collector if collector is not None else MetricsCollector()

    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/stats")
    async def stats():
        """格式化统计信息，便于调试"""
        import json

        def get_histogram_stats(histogram):
            """从 Histogram 获取 P50/P95/P99"""
            try:
                # prometheus_client Histogram: _buckets 是 list[MutexValue], _upper_bounds 是 list[float]
                # _buckets[-1].get() = total count, _sum.get() = total sum
                # buckets[i] 的值表示 <= _upper_bounds[i] 的观测值数量（累积）
                try:
                    count = int(histogram._buckets[-1].get())
                except Exception:
                    count = 0
                if count == 0:
                    return {"count": 0, "p50": 0, "p95": 0, "p99": 0}
                buckets = getattr(histogram, '_buckets', [])
                upper_bounds = getattr(histogram, '_upper_bounds', [])
                if not buckets or not upper_bounds:
                    return {"count": count, "p50": 0, "p95": 0, "p99": 0}
                # 构建 bound -> cumulative_count 字典
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
            except Exception as e:
                return {"count": 0, "p50": 0, "p95": 0, "p99": 0, "error": str(e)}

        def get_counter_value(counter):
            """从 Counter 获取总数"""
            try:
                val = getattr(counter, '_value', None)
                if val is None:
                    return 0
                return float(val.get()) if hasattr(val, 'get') else float(val)
            except Exception:
                return 0

        def get_gauge_value(gauge):
            """从 Gauge 获取当前值"""
            try:
                val = getattr(gauge, '_value', None)
                if val is None:
                    return 0
                return float(val.get()) if hasattr(val, 'get') else float(val)
            except Exception:
                return 0

        return {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
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
                "false_trigger_rate": 0,
            },
            "errors": {
                "llm": get_counter_value(ERROR_TOTAL.labels(stage="llm")),
                "tts": get_counter_value(ERROR_TOTAL.labels(stage="tts")),
                "stt": get_counter_value(ERROR_TOTAL.labels(stage="stt")),
                "vad": get_counter_value(ERROR_TOTAL.labels(stage="vad")),
                "connection": get_counter_value(ERROR_TOTAL.labels(stage="connection")),
            },
        }

    return app, collector


# ============================================================
# agent.py 集成示例
# ============================================================
"""
在 agent.py 中使用：

from monitoring.metrics import create_app
import uvicorn

# 创建 metrics app
metrics_app, metrics_collector = create_app(port=8081)

# 在各阶段调用
async def entrypoint(ctx: JobContext):
    metrics_collector.session_start()

    # VAD 检测
    metrics_collector.vad_triggered(is_speech=True)

    # STT 识别
    metrics_collector.stt_final(text="识别的文本")
    metrics_collector.stt_start()

    # LLM 处理
    metrics_collector.llm_start()
    # ... LLM 调用 ...
    metrics_collector.llm_first_token()
    metrics_collector.llm_end()

    # TTS 合成
    metrics_collector.tts_start()
    # ... TTS 调用 ...
    metrics_collector.tts_first_audio()
    metrics_collector.tts_end()

# 启动 metrics HTTP 服务 (可选，并行于主服务)
def run_metrics_server():
    uvicorn.run(metrics_app, host="0.0.0.0", port=8081)

import threading
threading.Thread(target=run_metrics_server, daemon=True).start()

# 注意：Docker 部署时需暴露 8081 端口
"""