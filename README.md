# 全双工语音聊天机器人 [已完成]

基于 LiveKit + DashScope API 的全双工实时语音对话系统，支持自然打断、Function Calling 和歌声合成。所有 AI 能力通过云端 API 调用，无需本地 GPU。

## 架构图

```
用户麦克风 → LiveKit → Agent
                         ↓
                   独立 LLM API (Qwen3.5-122B-W8A8)
                         ↓
             ┌───────────┴───────────┐
             ↓                       ↓
       DashScope Fun-ASR      DashScope Qwen3-TTS
             ↓                       ↓
       重采样 16k→48kHz       重采样 24k→48kHz → LiveKit → 用户扬声器
```

**数据流说明**：
- **ASR 路径**：用户语音 → LiveKit → Agent → DashScope Fun-ASR（16kHz）→ 重采样 48kHz → LLM
- **TTS 路径**：LLM 输出 → DashScope Qwen3-TTS（24kHz）→ 重采样 48kHz → LiveKit → 用户扬声器
- **LLM 路径**：直接调用独立 API 端点（非 DashScope）

| 服务 | 端口 | 说明 |
|------|------|------|
| Nginx | 80 | 反向代理 + 静态文件 |
| Next.js Backend | 3000 | Token 生成 API |
| LiveKit Server | 7880/7881 | WebRTC SFU 信令 + 媒体转发 |
| TTS Service | 8001 | DashScope Qwen3-TTS 代理（24kHz→48kHz 重采样） |
| Singing Service | 8002 | 歌声合成（Mock 正弦波模式） |
| Agent | - | LiveKit Worker，核心对话逻辑 |

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入必要的 API Key
```

必需的环境变量：

| 变量 | 说明 |
|------|------|
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API Key（ASR/TTS 用） |
| `LLM_API_KEY` | 独立 LLM API Key |
| `LLM_BASE_URL` | 独立 LLM API 端点地址 |
| `LIVEKIT_API_KEY` | LiveKit API Key |
| `LIVEKIT_API_SECRET` | LiveKit API Secret |

获取 DashScope API Key：https://dashscope.console.aliyun.com/

### 2. 启动服务

```bash
docker-compose up -d --build
```

### 3. 访问

打开浏览器访问 http://localhost

## 环境变量完整说明

| 变量 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `DASHSCOPE_API_KEY` | 是 | - | 阿里云 DashScope API Key（ASR/TTS 共用） |
| `DASHSCOPE_BASE_URL` | 否 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | DashScope API 基地址 |
| `DASHSCOPE_ASR_MODEL` | 否 | `fun-asr-2025-11-07` | Fun-ASR 实时语音识别模型 |
| `DASHSCOPE_TTS_MODEL` | 否 | `qwen3-tts-vd-2026-01-26` | Qwen3-TTS 语音合成模型 |
| `LLM_API_KEY` | 是 | - | 独立 LLM API Key |
| `LLM_BASE_URL` | 否 | `https://jiajiatemp.duckdns.org:30002/` | 独立 LLM API 端点 |
| `LLM_MODEL` | 否 | `Qwen3.5-122B-W8A8` | LLM 模型名称 |
| `LLM_TIMEOUT` | 否 | `5` | LLM 请求超时（秒） |
| `LIVEKIT_API_KEY` | 是 | - | LiveKit API Key |
| `LIVEKIT_API_SECRET` | 是 | - | LiveKit API Secret |
| `LIVEKIT_URL` | 否 | `ws://livekit:7880` | LiveKit WebSocket 地址（Agent 用） |
| `SYSTEM_PROMPT` | 否 | 内置默认 | System Prompt（简单模式） |
| `PROMPT_SERVICE_URL` | 否 | - | System Prompt HTTP 拉取地址（生产模式） |
| `TTS_SERVICE_URL` | 否 | `http://tts-service:8001` | TTS 微服务地址 |
| `SINGING_SERVICE_URL` | 否 | `http://singing-service:8002` | 歌声服务地址 |
| `TTS_TIMEOUT` | 否 | `10` | TTS 请求超时（秒） |
| `SINGING_TIMEOUT` | 否 | `30` | 歌声生成超时（秒） |
| `VAD_THRESHOLD` | 否 | `0.5` | VAD 语音检测阈值 |
| `VAD_MIN_SPEECH` | 否 | `0.2` | VAD 最小语音时长（秒） |
| `VAD_MIN_SILENCE` | 否 | `0.3` | VAD 最小静音时长（秒） |
| `RTC_PORT_START` | 否 | `50000` | WebRTC UDP 端口起始 |
| `RTC_PORT_END` | 否 | `50050` | WebRTC UDP 端口结束 |

## API 端点说明

### LLM 端点（独立 API）

```
https://jiajiatemp.duckdns.org:30002/
```

- **用途**：大语言模型推理（Qwen3.5-122B-W8A8）
- **接口**：OpenAI 兼容 API（`/v1/chat/completions`）
- **认证**：通过 `LLM_API_KEY` 环境变量配置

### DashScope API

```
https://dashscope.aliyuncs.com/compatible-mode/v1
```

- **用途**：ASR（Fun-ASR）和 TTS（Qwen3-TTS）
- **认证**：通过 `DASHSCOPE_API_KEY` 环境变量配置
- **文档**：https://help.aliyun.com/zh/dashscope/

| 模型 | 端点 | 采样率 | 说明 |
|------|------|--------|------|
| Fun-ASR | `/v1/audio/transcriptions` | 16kHz | 实时语音识别 |
| Qwen3-TTS | `/v1/audio/speech` | 24kHz | 语音合成 |

## 常见问题（FAQ）

**Q: 启动后提示 "DASHSCOPE_API_KEY is required"？**

A: 确保在 `.env` 文件中正确配置了 `DASHSCOPE_API_KEY`。获取地址：https://dashscope.console.aliyun.com/

**Q: 启动后提示 "LLM_API_KEY is required"？**

A: 确保在 `.env` 文件中正确配置了 `LLM_API_KEY`。这是独立 LLM API 的密钥，不同于 DashScope API Key。

**Q: 无法连接 LiveKit 服务器？**

A: 检查 `LIVEKIT_API_KEY` 和 `LIVEKIT_API_SECRET` 是否正确，以及 `LIVEKIT_URL` 是否可访问（docker-compose 内部为 `ws://livekit:7880`）。

**Q: TTS 语音质量差或延迟高？**

A: 确保网络可以访问 DashScope API。TTS 输出为 24kHz，内部会重采样到 48kHz。

**Q: 如何修改 System Prompt？**

A: 方式 A（简单模式）：设置 `SYSTEM_PROMPT` 环境变量。方式 B（生产模式）：设置 `PROMPT_SERVICE_URL`，Agent 会定时从该地址拉取 Prompt。

**Q: 歌声服务返回错误？**

A: 当前歌声服务为 Mock 模式，返回正弦波音频。检查 `SINGING_SERVICE_URL` 是否可访问。

**Q: 如何开启噪声消除？**

A: Agent 配置中已默认开启 `noise_cancellation=True`，可减少回声误触发。如需关闭，修改 `agent/agent.py` 中的 `RoomInputOptions`。

## 开发调试说明

各服务可独立启动，无需启动全部 docker-compose 服务。

### Agent（核心对话逻辑）

```bash
cd agent
pip install -r requirements.txt
# 需要配置 DASHSCOPE_API_KEY, LLM_API_KEY, LLM_BASE_URL, LIVEKIT_URL 等环境变量
python agent.py
```

### TTS Service

```bash
cd tts_service
pip install -r requirements.txt
# 需要配置 DASHSCOPE_API_KEY
python main.py
```

### Singing Service（Mock 模式）

```bash
cd singing_service
pip install -r requirements.txt
python main.py
```

### Frontend（Next.js）

```bash
cd frontend
npm install
npm run dev
```

### Backend（Token Service）

```bash
cd backend
npm install
npm run dev
```

### 单独测试 docker-compose 中的某个服务

```bash
# 只启动指定服务（不启动 agent）
docker-compose up -d livekit nginx backend tts-service

# 查看日志
docker-compose logs -f agent

# 进入容器调试
docker-compose exec agent bash
```

## 常用脚本

```bash
bash scripts/start-all.sh       # 启动所有服务
bash scripts/stop-all.sh        # 停止所有服务
bash scripts/logs.sh            # 查看所有服务日志
bash scripts/logs.sh agent      # 查看 Agent 日志
bash scripts/logs.sh tts        # 查看 TTS 服务日志
bash scripts/integration-tests.sh  # 运行集成测试
```

## 与旧版的主要区别

| 项目 | 旧版（本地模型） | 新版（DashScope API） |
|------|-----------------|---------------------|
| ASR | Deepgram / Whisper 本地 | DashScope Fun-ASR |
| LLM | 自建推理服务 | 独立 API 端点 |
| TTS | 本地 Qwen3-TTS 模型 | DashScope Qwen3-TTS API |
| Singing | 本地 VibeVoice 模型 | Mock 模式（正弦波） |
| GPU | 必需 (3 个模型) | 不需要 |
| 模型文件 | ~10GB+ | 无需下载 |
| TTS Docker 镜像 | ~10GB (含 torch) | ~200MB (纯 API 代理) |
| 启动时间 | 2-5 分钟 | <10 秒 |

## 许可证

MIT