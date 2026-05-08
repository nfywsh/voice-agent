# 语音助手监控系统

## 1. 概述

Prometheus + Grafana 监控体系，agent 自身暴露 `/metrics` 端点，Prometheus 定期拉取。

**核心原则**：
- agent 内嵌 `/metrics` HTTP 端点（方案 A/B 通用）
- 调试：直接 `curl localhost:8081/metrics`
- 集群：Prometheus 从每个 agent Pod 的 :8081/metrics 拉取

## 2. 指标清单

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `voice_llm_duration_seconds` | Histogram | LLM 处理总时长 |
| `voice_llm_first_token_seconds` | Histogram | LLM 首 token 延迟 |
| `voice_tts_first_audio_seconds` | Histogram | TTS 首音频延迟 |
| `voice_tts_total_duration_seconds` | Histogram | TTS 总合成时长 |
| `voice_stt_duration_seconds` | Histogram | STT 识别时长 |
| `voice_vad_latency_seconds` | Histogram | VAD检测→LLM完成 |
| `voice_vad_triggered_total` | Counter | VAD触发次数 (is_speech 标签) |
| `voice_asr_interim_total` | Counter | ASR中间结果次数 |
| `voice_asr_final_total` | Counter | ASR最终识别次数 |
| `voice_asr_error_total` | Counter | ASR错误次数 |
| `voice_session_active` | Gauge | 当前活跃会话数 |
| `voice_session_total` | Counter | 累计会话数 |
| `voice_error_total` | Counter | 各阶段错误 (stage 标签) |
| `voice_transcript_length_chars` | Histogram | 识别文本长度 |

## 3. 架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Compose / K8s                     │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ agent 容器                                              │ │
│  │ - agent.py 业务逻辑                                     │ │
│  │ - :8081/metrics (prometheus_client)                    │ │
│  └────────────────────────────────────────────────────────┘ │
│                         │                                    │
│          ┌──────────────┴──────────────┐                    │
│          │  Prometheus 拉取            │                    │
│          │  targets: [agent:8081]      │                    │
│          └──────────────┬──────────────┘                    │
│                         ▼                                    │
│                   ┌──────────┐                              │
│                   │ Grafana  │ ← 可视化 + 告警               │
│                   └──────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

## 4. 快速开始

### 4.1 本地调试

```bash
# 1. 安装依赖
pip install prometheus_client fastapi uvicorn

# 2. 验证 metrics 模块
python -c "from monitoring.metrics import create_app; print('OK')"

# 3. agent.py 中集成 (见 5.1)

# 4. 启动后调试
curl http://localhost:8082/metrics | grep "^voice_" | head -20
```

### 4.2 Docker 部署

**docker-compose.yml**:
```yaml
agent:
  build:
    context: ./agent
    dockerfile: Dockerfile
  ports:
    - "8081:8081"   # metrics 端点
  environment:
    ENABLE_METRICS: "true"
```

**prometheus.yml**:
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'voice-agent'
    static_configs:
      - targets: ['agent:8081']
```

## 5. agent.py 集成

### 5.1 基础集成

```python
from monitoring.metrics import create_app
import uvicorn
import threading

# 创建 metrics app
metrics_app, metrics_collector = create_app(port=8081)

# 启动 metrics HTTP 服务 (daemon thread)
def run_metrics():
    uvicorn.run(metrics_app, host="0.0.0.0", port=8081, log_level="warning")

threading.Thread(target=run_metrics, daemon=True).start()

# 在业务代码中调用
async def entrypoint(ctx: JobContext):
    metrics_collector.session_start()
    # ... VAD/STT/LLM/TTS 流程中调用对应方法 ...
    metrics_collector.session_end()
```

### 5.2 各阶段调用时机

```python
# VAD 检测到语音开始
metrics_collector.vad_triggered(is_speech=True)

# 用户说完，STT 开始识别
metrics_collector.stt_start()

# 收到 ASR 中间结果 (可选)
metrics_collector.stt_interim(text="用户说...")

# 收到 ASR 最终结果
metrics_collector.stt_final(text="用户说的完整内容")

# LLM 开始处理
metrics_collector.llm_start()

# LLM 输出第一个 token
metrics_collector.llm_first_token()

# LLM 处理完成
metrics_collector.llm_end()

# TTS 开始合成
metrics_collector.tts_start()

# TTS 输出第一个音频块
metrics_collector.tts_first_audio()

# TTS 合成完成
metrics_collector.tts_end()

# 错误记录
metrics_collector.error(stage="llm")  # stage: llm|tts|stt|vad|connection
```

## 6. Grafana 看板

### 6.1 核心 SQL

**P99 LLM 延迟**:
```sql
histogram_quantile(0.99, rate(voice_llm_duration_seconds_bucket[5m]))
```

**P50 TTS 首字延迟**:
```sql
histogram_quantile(0.50, rate(voice_tts_first_audio_seconds_bucket[5m]))
```

**QPS**:
```sql
rate(voice_llm_duration_seconds_count[1m])  # LLM QPS
rate(voice_asr_final_total[1m])              # ASR QPS
```

**VAD 误触发率**:
```sql
sum(rate(voice_vad_triggered_total{is_speech="false"}[5m]))
/ sum(rate(voice_vad_triggered_total[5m]))
```

**错误率**:
```sql
sum(rate(voice_error_total[5m])) / sum(rate(voice_session_total[5m]))
```

### 6.2 告警规则

```yaml
groups:
  - name: voice-agent-alerts
    rules:
      - alert: HighLLMLatency
        expr: histogram_quantile(0.95, rate(voice_llm_duration_seconds_bucket[5m])) > 2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "LLM P95 延迟超过 2 秒"

      - alert: HighTTSLatency
        expr: histogram_quantile(0.95, rate(voice_tts_first_audio_seconds_bucket[5m])) > 1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "TTS P95 首字延迟超过 1 秒"

      - alert: HighVADFalseTrigger
        expr: |
          sum(rate(voice_vad_triggered_total{is_speech="false"}[5m]))
          / sum(rate(voice_vad_triggered_total[5m])) > 0.3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "VAD 误触发率超过 30%"

      - alert: HighErrorRate
        expr: |
          sum(rate(voice_error_total[5m]))
          / sum(rate(voice_session_total[5m])) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "错误率超过 5%"
```

## 7. 高并发扩展

| 并发会话数 | Agent 副本 | 内存建议 | 说明 |
|-----------|-----------|---------|------|
| ~10 | 1 | 2Gi | 调试/小规模 |
| ~50 | 3 | 2Gi x 3 | 中等规模 |
| ~200 | 10 | 2Gi x 10 | 生产高并发 |

Prometheus 拉取所有 agent 副本，Grafana 自动聚合。

## 8. 依赖

```txt
# requirements.txt
prometheus_client>=0.19.0
fastapi>=0.110.0
uvicorn>=0.27.0
```

## 9. 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_METRICS` | `false` | 是否启用 /metrics 端点 |
| `METRICS_PORT` | `8081` | metrics HTTP 端口 |