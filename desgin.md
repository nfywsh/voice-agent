# 全双工语音聊天机器人 - 项目介绍与设计文档

## 1. 项目概述

### 1.1 项目背景

本项目旨在构建一个具备**全双工实时对话**、**自然打断**、**工具调用**及**唱歌能力**的语音聊天机器人。系统采用模块化级联架构，整合业界领先的开源模型（Qwen3.6-35B-A3B、Qwen3-TTS、VibeVoice-1.5B），并基于 LiveKit 实时通信框架实现低延迟的语音交互体验。

### 1.2 核心特性

| 特性               | 描述                                                                 |
| ------------------ | -------------------------------------------------------------------- |
| **全双工语音对话** | 支持边说边听，用户可随时打断 AI 发言，交互自然流畅                   |
| **多模态模型集成** | 自研级联流水线：ASR → LLM → TTS，支持流式处理降低首包延迟            |
| **Function Calling** | LLM 可调用外部工具（天气查询、网页搜索、唱歌等），扩展性强          |
| **歌声合成**       | 独立歌声模型 VibeVoice-1.5B，支持根据歌词生成旋律并演唱               |
| **实时通信**       | 基于 WebRTC 的 LiveKit 框架，保障低延迟音频传输与房间管理             |
| **生产级部署**     | Docker Compose 一键编排，Nginx 反向代理支持 HTTPS/WSS，GPU 资源隔离   |

### 1.3 技术栈

| 层级       | 技术选型                                                        |
| ---------- | --------------------------------------------------------------- |
| **前端**   | React + Vite + LiveKit Client SDK + WebRTC                       |
| **后端**   | Python (FastAPI) + LiveKit Agents SDK + Next.js (Token Service) |
| **AI 模型** | Qwen3.6-35B-A3B (LLM)、Qwen3-TTS (流式 TTS)、VibeVoice-1.5B (歌声) |
| **ASR**    | FunASR (本地实时识别)                                             |
| **实时通信** | LiveKit Server (开源 RTC 引擎)                                   |
| **部署**   | Docker Compose + Nginx + NVIDIA Docker Toolkit (GPU 支持)        |

---

## 2. 系统架构设计

### 2.1 总体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                              User Browser                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │   React UI   │◄──►│ LiveKit SDK  │◄──►│ WebRTC (Audio Track) │   │
│  └──────────────┘    └──────────────┘    └──────────────────────┘   │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ HTTPS / WSS
                              ┌────▼────┐
                              │  Nginx  │ (Reverse Proxy + SSL)
                              └────┬────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
┌────────▼────────┐      ┌─────────▼─────────┐      ┌───────▼───────┐
│   Next.js       │      │   LiveKit Server  │      │   Agent       │
│  (Token API)    │      │   (RTC Engine)    │      │  (Python)     │
└─────────────────┘      └─────────┬─────────┘      └───────┬───────┘
                                   │                         │
                                   │               ┌─────────▼─────────┐
                                   │               │  LLM (Qwen3.6)    │
                                   │               └─────────┬─────────┘
                                   │                         │
                          ┌────────┴────────┐        ┌───────┴───────┐
                          │                 │        │               │
                   ┌──────▼──────┐   ┌──────▼──────┐ │  ┌────────────▼────────┐
                   │  VLLM TTS   │   │  Sing Agent │ │  │  Tool Execution      │
                   │ (Qwen3-TTS) │   │  (8080)     │ │  │ (Weather, Search等)  │
                   └─────────────┘   └─────────────┘ │  └─────────────────────┘
                                                       │
                                          ┌────────────▼────────────┐
                                          │   FunASR ASR (Local)     │
                                          └──────────────────────────┘
```

### 2.2 模块职责说明

| 模块               | 职责                                                                   | 对外接口         |
| ------------------ | ---------------------------------------------------------------------- | ---------------- |
| **前端 (React)**   | 提供用户界面，采集麦克风音频，播放 AI 语音，显示对话记录                | -                |
| **Nginx**          | 反向代理，HTTPS 终结，WebSocket 升级，静态文件服务                      | 80/443 端口      |
| **Next.js 后端**   | 生成 LiveKit 访问 Token，保障房间安全                                   | `/api/token`     |
| **LiveKit Server** | WebRTC 信令服务，房间管理，音视频流转发                                 | 7880/7881 端口   |
| **Agent**          | 对话逻辑编排，调用 ASR/LLM/TTS，管理打断与工具调用                      | LiveKit Worker   |
| **LLM**            | 大语言模型推理，生成对话文本，支持 Function Calling                      | vLLM 兼容 API    |
| **VLLM TTS**       | 文本转语音，流式生成对话音频，通过 qwen3_tts_adapter 直连 VLLM TTS      | VLLM TTS :8021   |
| **Sing Agent**     | 歌词到歌声合成，流式返回歌声音频                                        | Sing Agent :8080 |
| **ASR**            | 实时语音识别，将用户语音转为文本                                        | FunASR HTTP API  |

### 2.3 数据流（一次典型对话）

1.  用户通过浏览器加入 LiveKit 房间。
2.  用户说话，音频通过 WebRTC 发送到 LiveKit Server。
3.  Agent 从 LiveKit 获取音频流，通过 FunASR ASR 转为文本。
4.  Agent 将文本发送给 LLM（Qwen3.6），LLM 可能返回普通文本或工具调用。
5.  若为工具调用（如唱歌），Agent 调用 Sing Agent（:8080）获取音频，直接推流；若为普通文本，则调用 VLLM TTS 流式合成语音。
6.  Agent 将合成的音频流通过 LiveKit 推送给用户。
7.  对话过程中，VAD 持续检测用户是否打断，若检测到新语音则中止当前 TTS 播放，切换至聆听状态。

---

## 3. 模块详细设计

### 3.1 前端模块（React）

**技术要点**：
- 使用 `@livekit/components-react` 提供的 `LiveKitRoom`、`VoiceAssistantControlBar`、`BarVisualizer` 等组件快速搭建 UI。
- 通过 `useVoiceAssistant` Hook 监听 Agent 状态（聆听、思考、说话）。
- 实时显示对话转写记录（需配合 Agent 端推送的转写事件）。

**关键代码片段（VoiceRoom.jsx）**：
```jsx
<LiveKitRoom
  serverUrl={livekitUrl}
  token={token}
  audio={true}
  connect={true}
>
  <VoiceAssistantControlBar onStateChange={handleAgentChange} />
  <BarVisualizer barCount={20} />
</LiveKitRoom>
```

### 3.2 Token 服务（Next.js）

**设计理由**：LiveKit 要求客户端在连接时提供有效的访问令牌（JWT）。将 Token 生成逻辑放在后端可避免在前端暴露 API Secret。

**接口定义**：
```
GET /api/token?room={roomName}&username={userName}
Response: { "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." }
```

**安全注意事项**：
- `LIVEKIT_API_KEY` 和 `LIVEKIT_API_SECRET` 必须通过环境变量注入，严禁硬编码。
- 生产环境可增加用户认证中间件，限制 Token 发放权限。

### 3.3 LiveKit Agent（Python）

Agent 是整个系统的核心协调者，基于 `livekit-agents` SDK 构建。

#### 3.3.1 核心类与职责

| 类/组件            | 职责                                                                 |
| ------------------ | -------------------------------------------------------------------- |
| `VoiceAssistant`   | 继承 `Agent`，定义系统指令和工具函数 (`@function_tool` 装饰器)       |
| `VLLM LLM Adapter` | 实现 `LLM` 抽象类，对接本地 Qwen3.6-35B-A3B 推理服务（OpenAI 兼容 API）|
| `Qwen3TTSAdapter`  | 实现 `TTS` 抽象类，对接本地 VLLM TTS 服务，支持流式合成                    |
| `SingingHandler`   | 独立处理唱歌请求，调用 Sing Agent 并返回音频流                  |

#### 3.3.2 System Prompt 注入设计

System Prompt 由外部系统注入，Agent 不硬编码任何业务人设，仅提供入口和结构约束。

```python
class VoiceAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=SYSTEM_PROMPT,  # 由环境变量或外部服务注入
            # ...
        )

# ============================================================
# System Prompt 注入入口
# ============================================================
# System Prompt 应由外部系统注入，本模块不硬编码任何业务人设。
# 注入方式：
#   1. 环境变量 SYSTEM_PROMPT：适合简单 demo / 测试场景
#   2. HTTP 接口动态拉取：适合生产环境，支持热更新
#
# Prompt 内容应包含：
#   - 角色设定（你是一个语音助手…）
#   - 工具调用规范（简洁调用，避免冗余参数）
#   - 打断行为的说明（用户可在任意时刻打断）
#   - 对话格式约束（短句优先，避免超长回复导致 TTS 延迟高）
# ============================================================

# --- 方式 A: 环境变量注入（简单 demo / 测试） ---
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    # 以下为默认 fallback，仅在未设置环境变量时使用
    "你是一个友好的语音助手。你能够通过工具调用来为用户唱歌。"
    "回复应简洁自然，适合语音播报。当用户要求唱歌时，调用 sing_a_song 工具。",
)

# --- 方式 B: HTTP 接口动态拉取（高并发 / 生产环境） ---
# 适合需要 Prompt 热更新、A/B 测试或按用户分组的场景。
# 实现思路：
#   1. 新建 PromptService 类，启动时和定期轮询远端 API 获取最新 Prompt
#   2. Agent 初始化时从 PromptService 获取 Prompt
#   3. 支持按 room / userId 维度下发不同 Prompt
#
# 示例骨架：
#   class PromptService:
#       def __init__(self, endpoint: str, refresh_interval: int = 300):
#           self._endpoint = endpoint
#           self._cache: dict[str, str] = {}
#           self._refresh_interval = refresh_interval
#
#       async def get_prompt(self, room_id: str = "default") -> str:
#           if room_id not in self._cache or self._is_stale(room_id):
#               async with httpx.AsyncClient() as client:
#                   resp = await client.get(f"{self._endpoint}?room={room_id}")
#                   self._cache[room_id] = resp.json()["prompt"]
#           return self._cache[room_id]
#
# 两种方式的对比：
# ┌──────────────┬─────────────────────┬──────────────────────────────┐
# │              │ 环境变量注入（A）    │ HTTP 动态拉取（B）            │
# ├──────────────┼─────────────────────┼──────────────────────────────┤
# │ 适用场景     │ Demo / 测试 / 单租户 │ 生产 / 多租户 / 需热更新      │
# │ 更新方式     │ 重启服务             │ 运行时轮询自动更新            │
# │ 实现复杂度   │ 低                   │ 中                            │
# │ 多 Prompt    │ 不支持               │ 支持按 room/user 分组         │
# │ 依赖         │ 无额外依赖           │ 需要 Prompt 管理 API 服务     │
# └──────────────┴─────────────────────┴──────────────────────────────┘
```

#### 3.3.3 打断处理机制

打断是全双工语音对话的核心交互行为，本系统采用"取消旧推理 + 合并新会话"的策略：

```
打断时序图：

用户说话 ───► ASR 转写 ───► VAD 检测到新语音
                                    │
                    ┌───────────────┤
                    │               │
                    ▼               ▼
            Agent.interrupt()   保留用户当前语音转写文本
            ├─ 停止当前 LLM 推理（cancel 正在进行的 streaming 请求）
            ├─ 清空 TTS 输出队列（停止当前音频播放）
            └─ 切换到聆听状态
                                    │
                                    ▼
                    用户停止说话，VAD 检测到静默
                                    │
                                    ▼
                    将用户新的语音转写文本与之前未完成对话的上下文合并
                                    │
                                    ▼
                    发起新的 LLM 会话请求（包含完整上下文）
```

**关键实现要点**：
- VAD 使用 Silero VAD 模型，持续监听用户音频流。
- 当检测到用户在 AI 说话期间开始讲话时，Agent 调用 `interrupt()` 方法。
- `interrupt()` 会立即取消当前 LLM 的 streaming 请求（通过 `asyncio.Task.cancel()`），清空 TTS 播放队列。
- 用户的完整语音输入由 ASR 转写完成后，与之前的对话历史合并，发起新一轮 LLM 请求。
- 合并逻辑：保留原始 system prompt 和历史消息，将打断时 LLM 已输出的部分文本作为"已说内容"标记，用户新输入追加到消息列表末尾。

```python
class VoiceAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=SYSTEM_PROMPT,
            vad=silero.VAD.load(
                min_speech_duration=0.2,    # 最短语音持续时间（秒），低于此阈值的片段视为噪声
                min_silence_duration=0.3,   # 最短静默持续时间（秒），超过此阈值认为说话结束
                activation_threshold=0.5,   # 语音激活概率阈值，高于此值判定为有语音
                sample_rate=16000,          # VAD 输入采样率，必须与 ASR 输入一致
            ),
            stt=deepgram.STT(),
            llm=QwenLLMAdapter(),
            tts=QwenTTSAdapter(),
        )
        self._current_llm_task: asyncio.Task | None = None

    async def on_user_turn_completed(self, message: str):
        """用户一轮发言结束后触发。如果当前有正在进行的 LLM 推理，先取消再发起新请求。"""
        if self._current_llm_task and not self._current_llm_task.done():
            self._current_llm_task.cancel()
            # 注意：取消后 LLM 之前已流式输出的部分文本会保留在对话历史中
            # 作为 assistant 消息的一部分，供新请求参考上下文

        self._current_llm_task = asyncio.create_task(
            self._process_llm_response(message)
        )

    async def _process_llm_response(self, user_message: str):
        """处理 LLM 响应，包含流式输出和工具调用。"""
        try:
            # 构建包含完整历史的消息列表
            messages = self._build_messages(user_message)
            async for chunk in self.llm.astream(messages):
                if self._current_llm_task.cancelled():
                    break
                await self._handle_chunk(chunk)
        except asyncio.CancelledError:
            # 打断时正常取消，不需要报错
            pass
```

#### 3.3.4 工具调用流程

```python
@function_tool
async def sing_a_song(self, title: str, lyrics: str, style: str = "流行") -> str:
    """当用户要求唱歌时调用。

    实际音频生成由 SingingHandler 完成，这里返回提示文本。
    SingingHandler 会异步将歌声音频流推送到 LiveKit 房间。

    Args:
        title: 歌曲名称
        lyrics: 歌词全文
        style: 歌曲风格，如"流行"、"民谣"、"摇滚"等
    """
    # 异步启动歌声生成，不阻塞 LLM 的 tool_call 返回
    asyncio.create_task(
        self._singing_handler.sing(title=title, lyrics=lyrics, style=style)
    )
    return f"正在演唱《{title}》，风格：{style}"
```

当 LLM 返回 `tool_calls` 时，LiveKit Agents 框架会自动调用对应的函数，函数返回值会作为新的消息发送回 LLM。

**工具调用约束（在 System Prompt 中声明）**：
- 仅在用户明确表达意图时调用工具，不要主动猜测。
- 唱歌工具调用时，需 LLM 自行生成合适歌词（如果用户未提供完整歌词）。
- 工具调用返回后，LLM 应给出简短的过渡语（如"好的，我来为你唱一首"），不要重复工具返回的文本。

### 3.4 TTS Service（Qwen3-TTS）

#### 3.4.1 服务封装

- 基于 FastAPI 构建独立微服务。
- 启动时加载 `Qwen3-TTS-12Hz-1.7B-CustomVoice` 模型至 GPU。
- 提供 `/tts/stream` 端点，返回流式音频数据。

**接口定义**：
```
POST /tts/stream
Content-Type: application/json

Request Body:
{
    "text": "你好，很高兴认识你",
    "voice": "default",          // 可选，音色标识（如使用 CustomVoice 特性）
    "speed": 1.0,                // 可选，语速倍率，默认 1.0
    "sample_rate": 24000         // 可选，输出采样率，默认 24000
}

Response:
Content-Type: audio/x-wav; streaming=true
Transfer-Encoding: chunked

[流式 WAV 音频块，每个块约 20-50ms 音频数据]
```

#### 3.4.2 音频格式规范

| 参数              | 值                  | 说明                                         |
| ----------------- | ------------------- | -------------------------------------------- |
| 采样率 (output)   | 24000 Hz            | Qwen3-TTS 默认输出采样率                     |
| 位深度            | 16-bit PCM          | 标准 WAV 格式                                |
| 声道数            | 1 (mono)            | 语音场景无需立体声                           |
| 流式分块大小      | ~20-50ms / 块       | 首包延迟约 200ms，后续持续输出               |

> **注意**：LiveKit 内部使用 48kHz/Opus 编码传输。Agent 在推流到 LiveKit 前需将 TTS 输出的 24kHz PCM 重采样到 48kHz，然后由 LiveKit SDK 自动编码为 Opus。重采样逻辑应在 `QwenTTSAdapter` 中处理。

#### 3.4.3 性能优化

- 使用 `bfloat16` 精度加载模型以节省显存。
- 音频分块输出，模拟流式效果（Qwen3-TTS 本身支持流式输出，可通过迭代器逐步返回）。
- 首 Token 延迟约 200ms，端到端首包延迟取决于 LLM 输出速度 + TTS 首包延迟。

#### 3.4.4 错误处理

| 错误场景             | 处理策略                                                 |
| -------------------- | -------------------------------------------------------- |
| 模型加载失败         | 服务启动时校验模型文件完整性，失败则拒绝启动并告警       |
| GPU OOM              | 返回 HTTP 503，Agent 侧收到后回退到简化提示重试一次      |
| 请求超时 (>10s)      | 返回 HTTP 504，Agent 侧 TTS 超时后播放预设提示音         |
| 输入文本过长 (>500字)| 截断并返回警告 header `X-Truncated: true`                |
| 无效 voice 参数      | 回退到默认音色，返回 `X-Fallback-Voice: default` header  |

### 3.5 Singing Service（VibeVoice-1.5B）

#### 3.5.1 模型能力说明

VibeVoice-1.5B 是微软开源的语音/歌声合成模型，具有以下特性：
- **输入**：文本（歌词 + 对话标记，格式为每行 `Speaker 1: 歌词内容`），不需要提供旋律 MIDI 或音高信息。模型能够根据文本内容和说话人标记，自动推断出合适的韵律和旋律走向。
- **输出**：原始音频波形（PCM），采样率通常为 24kHz 或 16kHz。
- **音色控制**：通过对话标记中的 Speaker ID 区分不同说话人/歌手，模型内置支持多种音色。
- **流式能力**：VibeVoice 原生支持逐 chunk 流式输出音频，推理过程中可边生成边推送。

> **关键设计决策**：LLM 负责生成歌词内容（基于用户请求的歌曲主题/风格），VibeVoice 则根据歌词文本自动生成旋律和歌声。无需额外的"旋律生成"模块——VibeVoice 内部已融合了从文本到旋律再到声波的能力。

#### 3.5.2 服务封装

不再使用 `subprocess` 调用推理脚本，改为构建独立的 FastAPI 微服务，与 TTS Service 架构一致。

**接口定义**：
```
POST /sing
Content-Type: application/json

Request Body:
{
    "lyrics": "Speaker 1: 星光洒落在肩上\nSpeaker 1: 夜风轻抚过脸庞",
    "style": "流行",              // 可选，风格提示（目前主要作为日志标记，
                                  // VibeVoice 暂不直接支持风格控制参数，
                                  // 风格差异由 LLM 生成不同歌词内容来间接体现）
    "speaker_id": "Speaker 1",   // 可选，默认 "Speaker 1"
    "sample_rate": 24000         // 可选，输出采样率，默认 24000
}

Response:
Content-Type: audio/x-wav; streaming=true
Transfer-Encoding: chunked

[流式 WAV 音频块]
```

**服务实现要点**：
- 启动时预加载 VibeVoice 模型到 GPU，避免首次请求冷启动延迟。
- 使用 `bfloat16` 精度加载以节省显存（约 3GB）。
- 模型推理通过 Python API 直接调用，不使用 subprocess。
- 流式返回：模型推理过程中，每生成一段音频 chunk 即通过 `StreamingResponse` 推送给 Agent。
- 支持并发请求队列，单 GPU 情况下串行推理，排队等待。

```python
# singing_service/app.py（核心骨架）
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import numpy as np

app = FastAPI()
model = None  # 全局模型实例，启动时加载

@app.on_event("startup")
async def load_model():
    global model
    # 从 HuggingFace 加载 VibeVoice 模型
    model = load_vibevoice("microsoft/VibeVoice-1.5B", dtype=torch.bfloat16)

@app.post("/sing")
async def sing(request: SingRequest):
    """流式生成歌声音频。"""
    async def audio_stream():
        # 模型流式推理，逐 chunk 输出
        for chunk in model.stream_generate(
            text=request.lyrics,
            speaker=request.speaker_id,
            sample_rate=request.sample_rate,
        ):
            # 将 numpy 数组转为 16-bit PCM 字节
            pcm_bytes = (chunk * 32767).astype(np.int16).tobytes()
            yield pcm_bytes

    return StreamingResponse(
        audio_stream(),
        media_type="audio/x-wav",
        headers={"X-Sample-Rate": str(request.sample_rate)},
    )
```

#### 3.5.3 歌唱场景完整流程

```
用户说"唱一首关于星空的歌"
        │
        ▼
    ASR 转写文本
        │
        ▼
    LLM 识别意图 → 调用 sing_a_song 工具
        │
        ├─ LLM 生成歌词内容：
        │  "Speaker 1: 星光洒落在肩上
        │   Speaker 1: 夜风轻抚过脸庞
        │   Speaker 1: 月牙弯弯挂天上
        │   Speaker 1: 梦在心间静静流淌"
        │
        ▼
    Agent 调用 Sing Agent（端口 8080）
        │
        ▼
    Sing Agent 流式返回歌声音频
        │
        ▼
    Agent 将音频流重采样至 48kHz → 推送到 LiveKit
        │
        ▼
    用户浏览器播放歌声
```

#### 3.5.4 延迟与用户体验优化

| 优化项               | 实现方式                                                       |
| -------------------- | -------------------------------------------------------------- |
| 模型预加载           | 服务启动时即加载模型到 GPU，避免首次请求 10s+ 冷启动           |
| 流式输出             | 模型边生成边推送，用户在 1-2s 内开始听到歌声                    |
| 等待提示             | Agent 在调用 Sing Agent 前先通过 TTS 说"好的，我来唱一首…"，填充等待时间 |
| 歌曲缓存             | 对相同歌词请求做结果缓存（LRU, max=50），避免重复推理           |
| 超时兜底             | 设置 30s 推理超时，超时后返回预设提示音并告知用户"歌声生成超时"  |

### 3.6 LLM 推理服务（Qwen3.6-35B-A3B）

#### 3.6.1 部署方案

Qwen3.6-35B-A3B 是 MoE 架构，使用 **VLLM** 部署。暴露与 OpenAI 兼容的 `/v1/chat/completions` 端点。

**高并发方案**（详见第 7 节）：
- VLLM 支持 prefix caching，对多用户共享 system prompt 的场景有显著优化。
- 启用 `--enable-prefix-caching` 参数，使相同 system prompt 前缀的请求共享 KV Cache。

#### 3.6.2 Function Calling 定义

```json
{
  "type": "function",
  "function": {
    "name": "sing_a_song",
    "description": "当用户要求唱歌时调用此工具。工具会根据歌词生成旋律并演唱。你需要根据用户的要求生成合适的歌词。",
    "parameters": {
      "type": "object",
      "properties": {
        "title": {
          "type": "string",
          "description": "歌曲名称"
        },
        "lyrics": {
          "type": "string",
          "description": "歌词全文，使用换行符分隔每句歌词，每句格式为 'Speaker 1: 歌词内容'"
        },
        "style": {
          "type": "string",
          "description": "歌曲风格，如流行、民谣、摇滚等",
          "default": "流行"
        }
      },
      "required": ["title", "lyrics"]
    }
  }
}
```

#### 3.6.3 LLM 推理超时与降级

| 场景                   | 处理策略                                                       |
| ---------------------- | -------------------------------------------------------------- |
| LLM 首包超时 (>5s)     | Agent 通过 TTS 播放"让我想想…"提示音                           |
| LLM 推理中断（打断）   | cancel 当前 streaming task，保留已输出文本为 assistant 消息     |
| LLM 服务不可用 (>10s)  | Agent 播放预设错误提示音，并记录错误日志                       |
| 响应过长 (>500 tokens)  | 在 system prompt 中约束 LLM 控制回复长度；超长时截断并合成已输出部分 |

### 3.7 ASR 服务（FunASR）

#### 3.7.1 选型与集成

- FunASR 提供低延迟的实时流式语音识别，支持中文。
- 在 Agent 中通过 `FunASRSTT` 或 `OpenAISTT` 插件直接集成，通过 HTTP API 调用本地 FunASR 服务。

**备选方案**：可替换为 OpenAI Whisper API 以降低本地部署依赖，但需额外处理流式输入。

#### 3.7.2 音频格式参数

| 参数              | 值              | 说明                                         |
| ----------------- | --------------- | -------------------------------------------- |
| 采样率 (input)    | 16000 Hz        | FunASR 推荐的语音识别采样率                  |
| 编码格式          | linear16 (PCM)  | LiveKit 从 WebRTC Opus 解码后输出 PCM        |
| 语言              | zh              | 中文语音识别，可设置 `auto` 开启自动语言检测 |
| 标点              | 开启            | FunASR 开启智能标点，提升转写可读性          |

#### 3.7.3 备选 ASR 方案：OpenAI Whisper API

如果需要纯离线部署（不依赖云端），可使用 OpenAI Whisper API：

```python
# 备选 ASR 配置（OpenAI Whisper）
from agent.openai_stt import create_stt

stt = create_stt()  # 通过 OPENAI_ASR_* 环境变量配置
# 注意：Whisper 本身不支持真正的流式输入，需配合 VAD 做段切分后批量识别
# 延迟比 FunASR 高约 200-500ms，适合对延迟不敏感的场景
```

### 3.8 音频管线参数汇总

整个系统中音频需经过多次采样率转换，以下是全链路参数：

```
用户麦克风 (48kHz, Opus)
    │
    ▼  LiveKit 解码
Agent 接收 (48kHz, PCM, mono)
    │
    ├─► VAD 检测 (重采样到 16kHz, PCM) ──► Silero VAD
    │
    ├─► ASR 识别 (重采样到 16kHz, PCM) ──► FunASR
    │
    ▼  TTS/Singing 输出
TTS 生成 (24kHz, PCM, mono)
    │
    ▼  重采样
Agent 输出 (48kHz, PCM, mono)
    │
    ▼  LiveKit 编码
用户扬声器 (48kHz, Opus)
```

| 环节             | 采样率     | 格式          | 声道  | 说明                               |
| ---------------- | ---------- | ------------- | ----- | ----------------------------------- |
| WebRTC 传输      | 48kHz      | Opus          | mono  | WebRTC/Opus 标准格式               |
| VAD 输入         | 16kHz      | PCM (S16LE)   | mono  | Silero VAD 训练采样率              |
| ASR 输入         | 16kHz      | PCM (S16LE)   | mono  | FunASR 推荐输入格式                |
| TTS 输出         | 24kHz      | PCM (S16LE)   | mono  | Qwen3-TTS 默认输出格式             |
| Singing 输出     | 24kHz      | PCM (S16LE)   | mono  | VibeVoice 默认输出格式             |
| Agent → LiveKit  | 48kHz      | PCM (S16LE)   | mono  | 推流前需从 24kHz 重采样到 48kHz    |

**重采样实现**：在 `QwenTTSAdapter` 和 `SingingHandler` 中使用 `librosa.resample` 或 `scipy.signal.resample_poly` 将 24kHz → 48kHz。

### 3.9 VAD 配置参数

Silero VAD 的关键参数需根据实际场景调优：

| 参数                    | 默认值  | 推荐范围      | 说明                                                     |
| ----------------------- | ------- | ------------- | -------------------------------------------------------- |
| `min_speech_duration`   | 0.2s    | 0.15 - 0.3s   | 低于此阈值的语音片段视为噪声，过短会导致吞音           |
| `min_silence_duration`  | 0.3s    | 0.2 - 0.5s    | 静默超过此阈值判定为说话结束，过短会频繁误触发         |
| `activation_threshold`  | 0.5     | 0.4 - 0.6     | 语音概率阈值，越高误触发越少但可能漏检                   |
| `sample_rate`           | 16000   | 16000 (固定)  | Silero VAD 仅支持 8kHz 和 16kHz，语音场景用 16kHz       |
| `prefix_padding_ms`     | 300ms   | 200 - 500ms   | 每个 VAD 窗口前的音频填充，帮助模型识别语音起始        |

> **调优建议**：在全双工场景下，`activation_threshold` 建议设为 0.5-0.6 以避免 AI 自身的语音被麦克风捕获后误触发打断。同时可在 Agent 侧增加"回声消除"逻辑：当 TTS 正在播放时，适当提高 VAD 阈值或延迟打断判定。

---

## 4. 部署与运维

### 4.1 Docker Compose 编排

所有服务均通过 Docker 容器化，`docker-compose.yml` 定义了完整的依赖关系。

**启动命令**：
```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 LiveKit 密钥、FunASR 服务地址等

# 2. 生成 SSL 证书（测试用）
mkdir -p nginx/ssl
cd nginx/ssl
openssl req -x509 -newkey rsa:4096 -nodes -keyout privkey.pem -out fullchain.pem -days 365

# 3. 构建并启动所有服务
docker-compose up -d --build
```

### 4.2 硬件要求

| 组件           | 最低显存要求 | 推荐配置          |
| -------------- | ------------ | ----------------- |
| Qwen3.5-35B    | 40 GB        | 2x A100 40GB      |
| Qwen3-TTS      | 4 GB         | RTX 4090 或以上   |
| VibeVoice-1.5B | 4 GB         | RTX 4090 或以上   |
| 其他服务       | 2 GB         | 普通 CPU 即可     |

> **注意**：若显存不足，可将 TTS 和 Singing 服务部署在同一 GPU，但需注意推理时的显存峰值。建议使用 `CUDA_VISIBLE_DEVICES` 环境变量隔离 GPU 分配。

### 4.3 监控与日志

- **LiveKit**：通过 `livekit-server --config` 可开启 Prometheus 指标导出。
- **Agent/TTS/Singing**：标准输出日志可通过 `docker-compose logs -f [service]` 查看。
- **建议**：集成 ELK 或 Loki 进行日志聚合。

### 4.4 扩展性考虑

- **横向扩展**：LiveKit Server 支持集群模式；Agent 可启动多个 Worker 并通过 Redis 共享状态。
- **模型服务化**：LLM、TTS 等可作为独立微服务，通过负载均衡提升并发能力。

---

## 5. 并发架构设计

### 5.1 简单 Demo / 测试方案

适用于单机、少量并发（1-5 用户同时在线）的场景。

```
┌─────────────────────────────────────────────────┐
│              单机 Docker Compose                 │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │LiveKit   │  │Agent     │  │Next.js Token  │  │
│  │Server    │  │(1 Worker)│  │Service        │  │
│  │(1 实例)  │  │          │  │               │  │
│  └──────────┘  └────┬─────┘  └───────────────┘  │
│                     │                            │
│         ┌───────────┼───────────┐                │
│         │           │           │                │
│  ┌──────▼───┐ ┌─────▼────┐ ┌───▼──────────┐    │
│  │LLM       │ │VLLM TTS  │ │Sing Agent    │    │
│  │(VLLM)    │ │(8021)    │ │(8080)        │    │
│  │(1 GPU)   │ │(1 GPU)   │ │(1 GPU)       │    │
│  └──────────┘ └──────────┘ └──────────────┘    │
│                                                  │
│  GPU 分配: GPU0 -> LLM, GPU1 -> TTS + Singing   │
└─────────────────────────────────────────────────┘
```

**关键配置**：
- Agent 启动 1 个 Worker 进程，单进程处理所有房间。
- TTS 和 Singing 共享同一 GPU（通过 `CUDA_VISIBLE_DEVICES=1` 控制），可以交替使用但不会同时推理，显存峰值约 7GB。
- LLM 独占 GPU0，VLLM 启用 prefix caching 优化 system prompt 共享。
- Nginx 在同一台机器上做反向代理。

**启动命令**：
```bash
# .env 配置
SYSTEM_PROMPT="你是一个友好的语音助手..."
LLM_GPU=0
TTS_SINGING_GPU=1

docker-compose --profile demo up -d
```

### 5.2 高并发生产方案

适用于 10+ 用户同时在线、需要高可用和低延迟的场景。

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Nginx (负载均衡 + SSL)                       │
└────────────┬──────────────────────────────────┬────────────────────┘
             │                                  │
    ┌────────▼────────┐                ┌────────▼────────┐
    │  LiveKit Server  │                │  LiveKit Server  │
    │  (Node 1)        │◄──Redis──────►│  (Node 2)        │
    └────────┬─────────┘                └────────┬─────────┘
             │                                  │
    ┌────────▼──────────────────────────────────▼────────┐
    │              Agent Worker Pool                      │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
    │  │ Worker 1 │  │ Worker 2 │  │ Worker N │  ...    │
    │  └──────────┘  └──────────┘  └──────────┘         │
    │     (每个 Worker 处理 1 个房间，水平扩展)           │
    └─────────────────────────┬──────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
     ┌────────▼──────┐ ┌─────▼──────┐ ┌──────▼───────┐
     │ LLM Cluster   │ │ VLLM TTS   │ │ Sing Agent   │
     │ (VLLM x2,     │ │ (2 实例,   │ │ (2 实例,    │
     │  prefix cache) │ │  负载均衡) │ │  队列调度)   │
     └───────────────┘ └────────────┘ └──────────────┘
```

**关键设计**：

1. **Agent Worker 池**：每个 Worker 处理一个房间的对话逻辑。使用 LiveKit 的 `WorkerType.ROOM` 模式，当一个房间被创建时自动派发空闲 Worker。Worker 数量可水平扩展。

2. **LLM 集群**：部署 2+ 个 VLLM 实例，通过 Nginx upstream 做负载均衡。所有实例共享相同 system prompt 前缀，VLLM 的 prefix caching 可使重复前缀的请求复用 KV Cache，显著降低首包延迟和显存占用。

3. **TTS / Sing Agent 服务池**：各部署 2 个实例，通过请求队列实现并发。Sing Agent 因其推理耗时较长（3-5s/首歌），建议增加请求队列深度和超时控制。

4. **Prompt 动态拉取**（见 3.3.2 节方式 B）：生产环境需要按租户/房间维度下发不同 system prompt，需部署 Prompt 管理 API 服务。

5. **GPU 资源隔离**：
   - LLM 实例独占 GPU（2x A100-40GB）
   - TTS 实例共享 GPU（每个实例约 3-4GB 显存）
   - Singing 实例共享 GPU（每个实例约 3-4GB 显存）

**Docker Compose 扩展**：
```bash
# 启动 3 个 Agent Worker
docker-compose up -d --scale agent=3

# 启动多个 LLM 实例（VLLM 支持多实例负载均衡）
docker-compose up -d --scale llm=2
```

### 5.3 并发能力估算

| 部署方案       | Agent Worker | LLM 并发 | TTS 并发 | Singing 并发 | 预估并发用户 |
| -------------- | ------------ | -------- | -------- | ------------ | ------------ |
| 简单 Demo      | 1            | 1        | 1        | 1 (串行)     | 1-3          |
| 中等配置       | 4            | 2        | 2        | 2            | 5-10         |
| 高并发生产     | 10+          | 4+       | 4+       | 4+ (队列)    | 20-50+       |

> **瓶颈分析**：LLM 推理通常是最先触达的瓶颈。VLLM 的 continuous batching 和 prefix caching 可显著提升吞吐。当 LLM 延迟上升时，优先扩展 LLM 实例数量。

---

## 6. 错误处理与降级策略

### 6.1 全链路错误处理

```
用户说话 ──► ASR ──► LLM ──► TTS ──► 用户听到
              │         │         │
              ▼         ▼         ▼
          降级:切换   降级:重试  降级:提示音
          OpenAI ASR  或播提示  或文本展示
```

| 阶段           | 错误场景                       | 处理策略                                                           | 用户感知                                   |
| -------------- | ------------------------------ | ------------------------------------------------------------------ | ------------------------------------------ |
| **ASR**        | FunASR 服务超时 (>5s)          | 自动切换到 OpenAI ASR（需配置），降级提示延迟增加 200ms            | 几乎无感知，识别略有延迟                   |
| **ASR**        | FunASR 服务不可用             | 切换到 OpenAI ASR，记录告警日志                                     | 延迟增加，但功能正常                       |
| **ASR**        | 两个 ASR 都不可用              | Agent 播放语音提示"抱歉，语音识别暂时不可用"                        | 功能不可用                                 |
| **LLM**        | 首包超时 (>5s)                 | Agent 播放"让我想想…"提示音                                        | 用户感知到等待，但有反馈                   |
| **LLM**        | 推理完全超时 (>15s)            | Agent 播放"抱歉，我一时没有想好"                                   | 用户感知到失败                             |
| **LLM**        | 服务不可用                     | Agent 播放预设错误提示音，记录错误日志                              | 功能不可用                                 |
| **LLM**        | Function Call 格式异常         | 忽略异常 tool_call，将其作为普通文本回复处理                        | 工具调用失败，但对话继续                   |
| **TTS**        | 服务返回 503 (OOM)             | 重试一次，若仍失败则播放预设提示音                                  | 可能听到"嘟"声而非完整语音                |
| **TTS**        | 请求超时 (>10s)                | 超时后播放预设提示音，将 LLM 回复文本推送到前端展示                 | 语音不可用，但可看文字                     |
| **TTS**        | 文本过长 (>500字)              | 截断并合成前 500 字，前端展示完整文本                               | 语音不完整，但可看文字                     |
| **Singing**    | Sing Agent 返回 503            | Agent 回退到 TTS"念歌词"模式（用普通 TTS 朗读歌词）                 | 无歌声但有语音                             |
| **Singing**    | 推理超时 (>30s)                | Agent 播放"歌声生成超时，请稍后再试"                                | 明确的超时反馈                             |
| **Singing**    | 排队等待过长 (>60s)            | 返回 429 Too Many Requests，Agent 告知用户"当前唱歌请求较多"        | 明确的繁忙反馈                             |
| **LiveKit**    | WebRTC 连接断开                | 前端自动重连 (LiveKit SDK 内置)，Agent 保留房间状态 30s            | 短暂中断后恢复                             |
| **LiveKit**    | 房间创建失败                   | 前端提示重试，后端检查 LiveKit Server 状态                          | 用户需手动重试                             |

### 6.2 前端降级展示

当 TTS 不可用时，Agent 通过 LiveKit DataChannel 将 LLM 回复的文本推送到前端，前端以文字气泡形式展示：

```typescript
// 前端监听 Agent 的 data message
room.on('data-received', (payload) => {
  const message = JSON.parse(new TextDecoder().decode(payload));
  if (message.type === 'transcript') {
    // 在对话区域显示文字消息
    appendTranscript(message.role, message.text);
  }
});
```

### 6.3 健康检查

所有微服务提供 `/health` 端点，Docker Compose 配置健康检查。Agent 服务通过 LiveKit 自带的健康检查机制确保可用性。

---

## 7. 功能测试场景

| 场景               | 预期结果                                                         |
| ------------------ | ---------------------------------------------------------------- |
| 用户说"你好"       | AI 以语音回复问候语，延迟 < 300ms                                |
| 用户在 AI 说话时打断 | AI 立即停止说话，开始聆听新输入                                   |
| 用户说"今天天气怎么样" | AI 调用 `get_weather` 工具，返回指定城市天气信息                  |
| 用户说"唱首歌给我听" | AI 调用 `sing_a_song`，生成歌词并调用歌声服务，播放歌曲           |
| 用户说"搜索最新新闻" | AI 调用 `search_web` 工具，返回模拟搜索结果                       |
| TTS 服务不可用     | AI 回复以文字形式展示在前端                                       |
| 用户在歌声播放时打断 | 歌声立即停止，AI 切换到聆听模式                                   |
| 多用户同时使用      | 各用户独立房间，互不干扰                                          |

---

## 8. 附录

### 8.1 环境变量清单

| 变量名                  | 描述                               | 示例值                         | 必需 |
| ----------------------- | ---------------------------------- | ------------------------------ | ---- |
| `LIVEKIT_API_KEY`       | LiveKit API Key                    | `devkey`                       | 是   |
| `LIVEKIT_API_SECRET`    | LiveKit API Secret                 | `secret`                       | 是   |
| `LIVEKIT_URL`           | LiveKit Server 地址（内部）        | `ws://livekit:7880`            | 是   |
| `LLM_BASE_URL`          | LLM 服务地址（OpenAI 兼容）        | `http://llm-server:8000/v1`    | 是   |
| `LLM_MODEL`             | 模型名称                           | `Qwen3.6-35B-A3B`              | 是   |
| `OPENAI_ASR_BASE_URL`   | FunASR 服务地址                   | `http://funasr:8000/v1`        | 是   |
| `OPENAI_ASR_API_KEY`    | FunASR API Key                     | `placeholder`                  | 是   |
| `OPENAI_ASR_MODEL`      | FunASR 模型名称                    | `fun-asr-2512`                 | 是   |
| `QWEN3_TTS_BASE_URL`    | VLLM TTS 服务地址                  | `http://host.docker.internal:8021` | 是   |
| `SING_AGENT_URL`        | Sing Agent 服务地址                | `http://sing_agent:8080`       | 是   |
| `SYSTEM_PROMPT`         | 系统提示词（简单 demo 方式）       | `你是一个友好的语音助手…`      | 否   |
| `PROMPT_SERVICE_URL`    | Prompt 动态拉取地址（生产方式）    | `http://prompt-service:8010`   | 否   |
| `LLM_TIMEOUT`           | LLM 首包超时阈值（秒）             | `5`                            | 否   |
| `TTS_TIMEOUT`           | TTS 请求超时阈值（秒）             | `10`                           | 否   |
| `SINGING_TIMEOUT`       | 歌声推理超时阈值（秒）             | `30`                           | 否   |
| `VAD_THRESHOLD`         | VAD 语音激活阈值                   | `0.5`                          | 否   |
| `VAD_MIN_SPEECH`        | VAD 最短语音时长（秒）             | `0.2`                          | 否   |
| `VAD_MIN_SILENCE`       | VAD 最短静默时长（秒）             | `0.3`                          | 否   |

### 8.2 常见问题排查

| 问题现象                 | 可能原因                         | 解决方法                               |
| ------------------------ | -------------------------------- | -------------------------------------- |
| 前端无法获取 Token       | Next.js 服务未启动或环境变量错误 | 检查 `backend` 容器日志，确认密钥正确 |
| WebRTC 连接失败          | Nginx 未正确代理 WebSocket       | 检查 `nginx.conf` 中 `/livekit` 配置  |
| TTS 返回 500             | 模型未下载或显存不足             | 检查服务日志，确认模型路径正确         |
| 唱歌功能长时间无响应     | VibeVoice 推理较慢               | 正常现象，可增加等待提示优化体验       |
| LLM 工具调用未触发       | 系统提示词未强调 Function Calling | 优化 `instructions`，明确告知模型调用工具 |
| 用户打断时 AI 继续说话   | VAD 阈值过高或回声未消除         | 降低 `VAD_THRESHOLD`，检查音频回环    |
| 歌声只有念白没有旋律     | VibeVoice 输入格式错误           | 确认歌词格式为 `Speaker 1: 歌词`      |
| 多用户时 Agent 响应变慢  | LLM 吞吐瓶颈                    | 增加 LLM 实例数，启用 prefix caching   |

### 8.3 Nginx 配置参考

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate     /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;

    # 前端静态文件
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }

    # Next.js Token API
    location /api/ {
        proxy_pass http://token-service:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # LiveKit WebSocket + HTTP
    location /livekit/ {
        proxy_pass http://livekit:7880/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
}
```

---

*文档版本：v3.0*
*最后更新：2026-06-01*
*变更记录：更新 LLM 为 Qwen3.6-35B-A3B，ASR 替换为 FunASR，TTS/Singing Service 替换为 VLLM TTS 和 Sing Agent，删除 Deepgram/Whisper/SGLang 引用*
