# 全双工语音聊天机器人

基于 LiveKit + DashScope API 的全双工实时语音对话系统，支持自然打断、Function Calling 和歌声合成。

## 架构概览

```
浏览器 → Nginx → LiveKit Server → Agent (Python)
                                    ↕
                           DashScope Fun-ASR (语音识别)
                           DashScope Qwen3.5-122B (大语言模型)
                           DashScope Qwen3-TTS (语音合成)
                           Mock 歌声服务 (开发调试)
```

**核心变化**：所有 AI 能力（ASR/LLM/TTS）通过阿里云 DashScope API 调用，**不再需要本地 GPU 和模型文件**。

| 服务 | 端口 | 说明 |
|------|------|------|
| Nginx | 80/443 | 反向代理 + 静态文件 |
| Next.js | 3000 | Token 生成 API |
| LiveKit | 7880/7881 | WebRTC 信令 + 媒体转发 |
| TTS Service | 8001 | DashScope Qwen3-TTS 代理 |
| Singing Service | 8002 | 歌声合成（Mock 模式） |
| Agent | - | LiveKit Worker，核心对话逻辑 |

## DashScope API 配置

| 模型 | 模型 ID | 用途 |
|------|---------|------|
| ASR | `fun-asr-2025-11-07` | 实时语音识别（Fun-ASR） |
| LLM | `Qwen3.5-122B-W8A8` | 大语言模型推理 |
| TTS | `qwen3-tts-vd-2026-01-26` | 语音合成 |

所有模型共用一个 `DASHSCOPE_API_KEY`，通过 OpenAI 兼容接口 (`/compatible-mode/v1`) 调用。

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填入:
# - DASHSCOPE_API_KEY  （阿里云 DashScope API Key，必需）
# - LIVEKIT_API_KEY / LIVEKIT_API_SECRET
```

获取 DashScope API Key: https://dashscope.console.aliyun.com/

### 2. 生成 SSL 证书（测试用）

```bash
mkdir -p nginx/ssl
cd nginx/ssl
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout privkey.pem -out fullchain.pem -days 365
```

### 3. 构建并启动

```bash
# 构建前端
cd frontend && npm install && npm run build && cd ..

# 启动所有服务
docker-compose up -d --build

# 或使用启动脚本
bash scripts/start-all.sh
```

### 4. 访问

打开浏览器访问 http://localhost

## 环境变量

完整变量列表见 `.env.example`，关键配置：

| 变量 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `DASHSCOPE_API_KEY` | 是 | - | 阿里云 DashScope API Key |
| `DASHSCOPE_BASE_URL` | 否 | `https://dashscope.aliyuncs.com/compatible-mode/v1` | API 基地址 |
| `DASHSCOPE_ASR_MODEL` | 否 | `fun-asr-2025-11-07` | 语音识别模型 |
| `DASHSCOPE_TTS_MODEL` | 否 | `qwen3-tts-vd-2026-01-26` | 语音合成模型 |
| `LLM_MODEL` | 否 | `Qwen3.5-122B-W8A8` | 大语言模型 |
| `LIVEKIT_API_KEY` | 是 | - | LiveKit API Key |
| `LIVEKIT_API_SECRET` | 是 | - | LiveKit API Secret |
| `SYSTEM_PROMPT` | 否 | 内置默认 | 系统提示词 |
| `VAD_THRESHOLD` | 否 | 0.5 | VAD 语音检测阈值 |

## 与旧版的主要区别

| 项目 | 旧版（本地模型） | 新版（DashScope API） |
|------|-----------------|---------------------|
| ASR | Deepgram / Whisper 本地 | DashScope Fun-ASR |
| LLM | 自建推理服务 | DashScope Qwen3.5 |
| TTS | 本地 Qwen3-TTS 模型 | DashScope Qwen3-TTS API |
| Singing | 本地 VibeVoice 模型 | Mock 模式（正弦波） |
| GPU | 必需 (3 个模型) | 不需要 |
| 模型文件 | ~10GB+ | 无需下载 |
| TTS Docker 镜像 | ~10GB (含 torch) | ~200MB (纯 API 代理) |
| 启动时间 | 2-5 分钟 | <10 秒 |

## 开发

各服务可独立开发和调试：

```bash
# Agent (需要 DASHSCOPE_API_KEY)
cd agent && pip install -r requirements.txt && python agent.py

# TTS Service (需要 DASHSCOPE_API_KEY)
cd tts_service && pip install -r requirements.txt && python main.py

# Singing Service (Mock 模式，无需 API Key)
cd singing_service && pip install -r requirements.txt && python main.py

# Frontend
cd frontend && npm install && npm run dev

# Backend (Token Service)
cd backend && npm install && npm run dev
```

## 运行测试

```bash
# 集成测试（需先 docker-compose up -d）
bash scripts/integration-tests.sh

# Agent 单元测试
cd agent && pip install pytest pytest-asyncio && pytest tests/ -v

# TTS 单元测试
cd tts_service && pip install pytest pytest-asyncio httpx && pytest tests/ -v

# Singing 单元测试
cd singing_service && pip install pytest pytest-asyncio httpx && pytest tests/ -v
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

## 许可证

MIT