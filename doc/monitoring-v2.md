# 语音助手监控改进方案

## 1. 背景与目标

### 1.1 现有监控的问题

| 问题 | 现状 | 影响 |
|------|------|------|
| TTS 指标从未被调用 | `tts_start()` / `tts_first_audio()` / `tts_end()` 定义了但 `agent.py` 从未调用 | TTS 延迟、吞吐量数据全为 0 |
| VAD 指标从未调用 | `vad_triggered()` 定义了但从未调用 | VAD 触发次数、误触发率无法统计 |
| 所有指标是全局聚合 | Prometheus Histogram 只能看集群维度的 P50/P95/P99 | 无法定位"房间 room-xxx 为什么响应慢" |
| TTS 时序起点错误 | `tts_start()` 在 LLM 结束后调用，而非第一个 TTS chunk 发出时 | TTS 首字延迟不准确 |
| 请求级时序丢失 | 每个 request 没有独立的 trace_id / 各环节 timestamp | 无法串联一个请求的全链路 |

### 1.2 改进目标

1. **修复 TTS/VAD 指标调用链路** — 让 Prometheus 集群监控有数据
2. **添加 per-room per-request 时序记录** — 支持按房间号查询每个请求各环节的耗时
3. **支持实时调试** — 输入房间号，能看到当前请求哪个环节卡住了、用时多久
4. **两套接口分工明确** — Prometheus 用于集群告警和大盘，REST API 用于单房间调试

---

## 2. 架构设计

### 2.1 接口分工

| 接口 | 用途 | 使用方 |
|------|------|--------|
| `GET :8082/metrics` | Prometheus 集群聚合指标（P50/P95/P99、QPS、错误率） | Prometheus + Grafana 告警大盘 |
| `GET :8082/stats/rooms/{room_id}` | 该房间当前+最近请求的各环节耗时 | 运维调试 |
| `GET :8082/stats/rooms/{room_id}/active` | 该房间正在处理中的请求，各环节是否卡住 | 实时诊断 |
| `GET :8082/health` | 健康检查 | 存活探针 |

### 2.2 数据流

```
                    Prometheus 集群
                         │
                    :8082/metrics
                         │
┌─────────────────────────────────────────────────────┐
│  agent 进程 (单 Pod)                                 │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ MetricsCollector (全局单例)                  │   │
│  │  - 全局 Histogram (Prometheus聚合)           │   │
│  │  - 全局 Counter/Gauge                        │   │
│  │  - _request_traces: dict[request_id, Trace]  │   │
│  └─────────────────────────────────────────────┘   │
│                         ▲                           │
│   ┌─────────────────────┼─────────────────────┐    │
│   │                     │                     │    │
│ agent.py              TTS Adapter           VAD   │
│ llm_node()         QwenTTSStream          回调    │
│   │                     │                     │    │
│   │  tts_start()        │ tts_start()        │vad_  │
│   │  tts_first_audio()  │ tts_first_audio()  │triggered()
│   │  tts_end()          │ tts_end()          │      │
│   └─────────────────────┴─────────────────────┘    │
└─────────────────────────────────────────────────────┘
                         │
                    :8082/stats/rooms/{room_id}
                         │
                    人工/运维工具
```

---

## 3. 数据模型

### 3.1 Trace — 单次请求的完整时序

```python
@dataclass
class RequestTrace:
    request_id: str          # UUID，一次语音交互（VAD 检测到 → TTS 播完）
    room_id: str
    user_id: str
    status: str              # "in_progress" | "done" | "error"
    created_at: float        # time.monotonic()，绝对值用于超时判断

    # ASR 环节（用户说话 → 文字识别完成）
    asr_start: Optional[float] = None
    asr_end: Optional[float] = None
    asr_text: str = ""

    # LLM 环节（文字 → 首个 token → 生成完毕）
    llm_start: Optional[float] = None
    llm_first_token: Optional[float] = None
    llm_end: Optional[float] = None

    # TTS 环节（流式，第一个 chunk 发出 → 最后一个 chunk 播完）
    tts_start: Optional[float] = None              # 首次调用 TTS 的时间
    tts_first_audio: Optional[float] = None        # 首个音频帧返回的时间
    tts_end: Optional[float] = None                # 最后一个 chunk 完成的时间
    tts_chunk_count: int = 0                       # TTS 分片数量
    tts_chars_sent: int = 0                        # 发送给 TTS 的总字符数

    # 各环节计算的耗时（秒），方便调试
    @property
    def asr_duration(self) -> Optional[float]: ...
    @property
    def llm_duration(self) -> Optional[float]: ...
    @property
    def llm_ttft(self) -> Optional[float]: ...     # time to first token
    @property
    def tts_duration(self) -> Optional[float]: ...
    @property
    def tts_ttfb(self) -> Optional[float]: ...     # time to first byte/audio
    @property
    def e2e_duration(self) -> Optional[float]: ... # asr_start → tts_end
```

### 3.2 RoomStats — 房间级别的统计（滚动窗口）

```python
@dataclass
class RoomStats:
    room_id: str
    active_request: Optional[RequestTrace] = None   # 当前正在处理的请求
    recent_traces: list[RequestTrace] = []          # 最近完成的 N 条（默认保留 10 条）
    total_requests: int = 0
    total_errors: int = 0

    # 滚动窗口统计（最近 5 分钟）
    asr_count_5m: int = 0
    llm_count_5m: int = 0
    tts_count_5m: int = 0
    error_count_5m: int = 0
```

### 3.3 MetricsCollector 改造

```python
class MetricsCollector:
    # 全局 Prometheus 指标（不变，继续聚合跨 Pod 跨请求的数据）
    _llm_duration: Histogram
    _llm_first_token: Histogram
    _tts_first_audio: Histogram
    _tts_total_duration: Histogram
    _stt_duration: Histogram
    ...

    # Per-request 时序（内存字典，key = request_id）
    _request_traces: dict[str, RequestTrace]

    # Per-room 统计（内存字典，key = room_id）
    _room_stats: dict[str, RoomStats]

    # 请求级别上报（新增）
    def request_start(self, room_id: str, user_id: str) -> str:
        """开始一个请求，返回 request_id"""

    def asr_start(self, request_id: str) -> None:
        """ASR 开始识别"""

    def asr_final(self, request_id: str, text: str) -> None:
        """ASR 最终识别完成"""

    def llm_start(self, request_id: str) -> None:
        """LLM 开始推理"""

    def llm_first_token(self, request_id: str) -> None:
        """LLM 首个 token"""

    def llm_end(self, request_id: str) -> None:
        """LLM 生成完毕"""

    def tts_start(self, request_id: str) -> None:
        """TTS 首次调用（发第一个文本 chunk）"""

    def tts_first_audio(self, request_id: str) -> None:
        """TTS 首个音频帧返回"""

    def tts_end(self, request_id: str) -> None:
        """TTS 最后一个 chunk 完成"""

    def request_end(self, request_id: str, status: str = "done") -> None:
        """请求结束"""

    # 查询接口（新增）
    def get_room_stats(self, room_id: str) -> dict:
        """获取房间当前状态和最近请求"""

    def get_active_request(self, room_id: str) -> Optional[RequestTrace]:
        """获取房间当前正在处理的请求"""
```

---

## 4. 接口设计

### 4.1 `GET /stats/rooms/{room_id}`

返回该房间当前状态和最近请求的各环节耗时。

**响应示例：**
```json
{
  "room_id": "room-123",
  "active_request": null,
  "recent_traces": [
    {
      "request_id": "a1b2c3d4",
      "status": "done",
      "asr_duration": 0.83,
      "llm_duration": 1.24,
      "llm_ttft": 0.31,
      "tts_duration": 3.21,
      "tts_ttfb": 0.45,
      "e2e_duration": 5.28,
      "tts_chunk_count": 3,
      "tts_chars_sent": 156,
      "created_at": "2026-05-08T10:23:45.123Z"
    }
  ],
  "stats_5m": {
    "asr_count": 12,
    "llm_count": 12,
    "tts_count": 12,
    "error_count": 0
  }
}
```

### 4.2 `GET /stats/rooms/{room_id}/active`

返回该房间正在处理中的请求，各环节是否卡住。

**响应示例：**
```json
{
  "room_id": "room-123",
  "active_request": {
    "request_id": "e5f6g7h8",
    "status": "in_progress",
    "phase": "tts",
    "elapsed": {
      "asr": 0.83,
      "llm": 1.24,
      "tts": 2.10
    },
    "stuck_detection": {
      "phase": "tts",
      "waited_seconds": 2.10,
      "threshold_seconds": 5.0,
      "is_stuck": false
    }
  }
}
```

**卡住检测逻辑：**
- `asr` 阶段：超过 5s 无结果 → 卡住
- `llm` 阶段：超过 30s 无首 token → 卡住
- `tts` 阶段：超过 5s 无音频返回 → 卡住

### 4.3 `GET /stats`（增强）

在现有 `/stats` JSON 基础上，新增 `recent_rooms` 字段，列出最近有活动的房间。

---

## 5. TTS 指标调用链路改造

### 5.1 问题

当前 `agent.py` 里没有 TTS 回调，`tts_adapter.py` 的 `QwenTTSStream._run()` 也没有上报 metrics。

### 5.2 解决方案

TTS 适配器持有对 `MetricsCollector` 的引用，在各阶段调用上报。

**方案 A：构造函数传入 collector（推荐）**
```python
class QwenTTSAdapter(tts.TTS):
    def __init__(self, ..., metrics: MetricsCollector = None):
        self._metrics = metrics
```

**方案 B：通过 `agent._metrics` 间接传递**
- `QwenTTSStream` 需要能访问到 `agent._metrics`
- 但 `QwenTTSStream` 是通过 `AgentSession` 构造的，不在 agent 的直接控制下
- 方案 A 更直接简洁

**TTS 时序起点修正：**
- `tts_start` = 第一个文本 chunk 发给 TTS 服务的时刻（在 `_run()` 的 `send_tts_chunk()` 里调用）
- `tts_first_audio` = TTS 首个音频帧返回的时刻（在 `buffer >= 3840` 首次满足时调用）
- `tts_end` = 最后一个 chunk 完成的时刻

---

## 6. VAD 指标调用链路改造

### 6.1 问题

`vad_triggered()` 从未被调用。

### 6.2 解决方案

在 `AgentSession` 的 VAD 事件回调中调用。查看 SDK v1.5.7，VAD 事件通过 `session.on("vad_speech")` 等事件暴露。

但 `agent.py` 目前没有注册 VAD 事件的回调。需要添加：
```python
@session.on("vad_detected")
def on_vad_detected(is_speech: bool):
    metrics.vad_triggered(is_speech)
```

**注意：** 需要确认 LiveKit Agents SDK v1.5.7 的 VAD 事件名称，常见的是 `vad_speech_start` / `vad_speech_end`。

---

## 7. 请求 ID 生成与关联

### 7.1 request_id 生成时机

在 `on_user_turn_completed` 或 VAD 检测到 speech 开始时生成，作为 `RequestTrace` 的唯一标识。

### 7.2 各环节如何拿到 request_id

当前设计下，`entrypoint()` 创建 `MetricsCollector`，注入 `agent._metrics`。

问题：`tts_adapter.py` 没有 `agent` 引用，只有 `QwenTTSAdapter` 实例。

解决方案：
1. `QwenTTSAdapter` 持有 `metrics` 引用
2. `tts_start/tts_first_audio/tts_end` 携带 `request_id`，但流式处理中 `request_id` 需要从外部传入
3. 更简洁的方案：`tts_adapter` 的流式对象不直接持有 `request_id`，而是在 `agent.py` 的 `session.on("llm_end")` / `session.on("tts_*")` 事件中处理

**最终方案：**
- 通过 `request_id` 参数传递给 `synthesize()` / `stream()`
- 但 SDK 的 `SynthesizeStream` 不支持传额外参数
- 改用 contextvars 或 thread-local storage：在 `llm_node()` 调用前设置当前 `request_id`，TTS 内部通过 `get_current_request_id()` 获取

**最简方案：**
- 在 `MetricsCollector` 里维护一个 `current_request_id` 的 thread-local 变量
- `llm_node` 开始时 set，结束后 clear
- TTS 回调通过 `get_current_request_id()` 获取

---

## 8. 文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `agent/monitoring/metrics.py` | 新增 `RequestTrace` / `RoomStats` 数据结构，改造 `MetricsCollector`，新增 per-request 时序记录和 room 查询接口 |
| `agent/tts_adapter.py` | 在 `QwenTTSStream._run()` 中调用 metrics 回调（tts_start / tts_first_audio / tts_end） |
| `agent/agent.py` | 注册 VAD 事件回调，在 `llm_node` 中设置/清除 current_request_id，注入 metrics 到相关组件 |
| `doc/monitoring-v2.md` | 本文档 |

---

## 9. 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `METRICS_PORT` | `8082` | metrics HTTP 服务端口 |
| `MAX_RECENT_TRACES` | `10` | 每个房间保留最近 N 条 trace |
| `STUCK_THRESHOLD_ASR` | `5.0` | ASR 卡住阈值（秒） |
| `STUCK_THRESHOLD_LLM_TTFT` | `10.0` | LLM 首 token 卡住阈值（秒） |
| `STUCK_THRESHOLD_TTS` | `5.0` | TTS 无音频卡住阈值（秒） |

---

## 10. 不纳入本方案的范围

1. **持久化** — trace 数据只存内存，Pod 重启后丢失。生产环境如需持久化，可后续对接 Redis/ClickHouse。
2. **Grafana 改造** — 当前 Prometheus 指标体系不变，只修复 TTS/VAD 指标调用链路使其有数据。
3. **分布式追踪** — 不引入 Jaeger/Zipkin，保持轻量。