# LLM 思考模式与聊天历史管理设计

## 1. 概述

为语音助手 Agent 增加两个功能：
1. **LLM 思考模式开关**：控制模型是否启用思考/推理模式
2. **聊天历史管理**：限制历史长度，支持外部注入，避免历史无限累积

同时实现：
3. **LLM Reasoning 内容过滤**：思考模式下，reasoning 内容不传给 TTS
4. **TTS 流式分片策略**：首字低延迟 + 顺序无重叠 + 超时保底

## 2. 思考模式开关

### 2.1 设计目标

- 默认关闭（降低延迟，节省 token）
- 按用户+房间独立配置
- 不持久化存储，仅在 session 生命周期内生效
- 支持调用时动态修改（通过工具函数）

### 2.2 实现方案

**LLM 请求参数抽象层**

为支持多模型兼容，引入 `LLM_CHAT_TEMPLATE_KWARGS` 环境变量配置：

```bash
# .env 配置
LLM_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'
```

**模型参数映射**

| 模型 | 参数路径 | 说明 |
|------|----------|------|
| DashScope Qwen | `extra_body.chat_template_kwargs.enable_thinking` | bool |
| OpenAI GPT | `extra_kwargs.reasoning_effort` | low/medium/high |
| Gemini | `extra_kwargs.thinking_budget` | int, 0 = 关闭 |

**存储设计**

```python
# 内存存储，key = f"{room_id}:{user_id}"
_thinking_mode_store: dict[str, bool] = {}

def get_thinking_mode(room_id: str, user_id: str) -> bool:
    return _thinking_mode_store.get(f"{room_id}:{user_id}", False)

def set_thinking_mode(room_id: str, user_id: str, enabled: bool) -> None:
    _thinking_mode_store[f"{room_id}:{user_id}"] = enabled
```

### 2.3 工具函数接口

Agent 提供两个工具函数供语音调用：

```python
@function_tool
async def set_thinking_mode(self, enabled: bool) -> str:
    """开启或关闭思考模式"""
    # 设置后当前 session 生效，session 结束自动清除

@function_tool
async def get_thinking_mode_status(self) -> str:
    """查询当前思考模式状态"""
```

### 2.4 HTTP API 接口（供外部系统调用）

```
# 设置思考模式
POST /api/agent/thinking-mode
{
    "room": "room-123",
    "user": "user-456",
    "enabled": true
}

# 获取思考模式状态
GET /api/agent/thinking-mode?room=room-123&user=user-456
```

### 2.5 动态修改实现

通过重写 `VoiceAssistant.llm_node()` 实现真正的动态思考模式切换：

```python
async def llm_node(self, chat_ctx, tools, model_settings=None) -> AsyncIterable[llm.ChatChunk]:
    # 每次调用时读取当前思考模式状态
    room_id = getattr(self, '_room_id', None) or ...
    user_id = getattr(self, '_user_id', None) or ...
    is_thinking = get_thinking_mode(room_id, user_id)

    extra_kwargs = {
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": is_thinking}
        }
    }

    async with self.session.llm.chat(chat_ctx=chat_ctx, tools=tools, extra_kwargs=extra_kwargs) as stream:
        async for chunk in stream:
            # reasoning 过滤（见第 6 节）
            ...
            yield chunk
```

**关键机制**：`room_id` 和 `user_id` 在 `entrypoint` 中通过 `agent._room_id` / `agent._user_id` 注入，确保 `llm_node` 能正确查找每用户的思考模式状态。

## 3. 聊天历史管理

### 3.1 设计目标

- 默认保留最近 N 轮对话（N 通过 `CHAT_HISTORY_MAX_TURNS` 配置，默认 10）
- 支持外部系统注入聊天历史
- 自动截断避免历史无限累积
- 区分内部（语音）消息和外部（文字）消息来源（预留）

### 3.2 配置项

```bash
# .env 配置
CHAT_HISTORY_MAX_TURNS=10  # 保留最近 10 轮对话
```

### 3.3 实现方案

**ChatHistoryManager 类**

```python
class ChatHistoryManager:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns

    def inject_messages(self, chat_ctx: ChatContext, messages: list[dict]) -> None:
        """注入外部消息到聊天历史"""
        for msg in messages:
            chat_ctx.add_message(role=msg["role"], content=msg["content"])
        self.truncate(chat_ctx)

    def truncate(self, chat_ctx: ChatContext) -> None:
        """截断聊天历史到指定轮数"""
        chat_ctx.truncate(max_items=self.max_turns)
```

**截断时机**

在 `VoiceAssistant.on_user_turn_completed()` 中自动截断：

```python
async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
    """用户说完一句话后，自动截断聊天历史避免无限累积"""
    self.chat_history_manager.truncate(turn_ctx)
```

### 3.4 外部注入接口

```bash
# 注入聊天历史
POST /api/agent/inject-history
{
    "room": "room-123",
    "user": "user-456",
    "messages": [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好，有什么可以帮你？"}
    ]
}
```

注入后的消息会自动截断到 `CHAT_HISTORY_MAX_TURNS` 轮。

## 4. 文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `agent/agent.py` | 添加思考模式存储、ChatHistoryManager、工具函数、`llm_node` reasoning 过滤 |
| `agent/tts_adapter.py` | 流式分片策略：首片低延迟 + await 顺序发送 + 超时保底 |
| `doc/design-llm-thinking-and-history.md` | 本文档 |

## 5. 配置项汇总

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LLM_CHAT_TEMPLATE_KWARGS` | `{"enable_thinking": false}` | LLM 模板参数 |
| `CHAT_HISTORY_MAX_TURNS` | `10` | 聊天历史保留轮数 |
| `TTS_TIMEOUT` | `10` | TTS 单次请求超时（秒） |

## 6. LLM Reasoning 内容过滤

### 6.1 问题背景

启用思考模式时，LLM 输出中 `reasoning` 字段包含大量内部推理内容（可能数千字），不应发送给 TTS 合成语音。

API 返回格式（思考模式）：

```json
{
  "choices": [{
    "index": 0,
    "delta": {
      "reasoning": "\n模型推理内容...",
      "content": "\n\n实际回复内容..."
    }
  }]
}
```

### 6.2 实现方案

在 `agent.py` 的 `llm_node` 中过滤：

```python
async def llm_node(self, chat_ctx, tools, model_settings=None) -> AsyncIterable[llm.ChatChunk]:
    async with self.session.llm.chat(chat_ctx=chat_ctx, tools=tools, extra_kwargs=extra_kwargs) as stream:
        async for chunk in stream:
            if chunk.delta:
                reasoning = chunk.delta.extra.get("reasoning") if chunk.delta.extra else None
                content = chunk.delta.content

                # reasoning chunk 没有 content，跳过（不传给 TTS）
                if reasoning is not None and not content:
                    logger.debug(f"[llm_node] Skip reasoning-only chunk")
                    continue

                yield chunk  # 有 content 的正常传下游
```

**关键点**：
- `reasoning` 在 `delta.extra` 中（provider-specific 字段）
- 如果 chunk 同时有 `reasoning` 和 `content`，正常传给 TTS
- 如果 chunk 只有 `reasoning`，跳过，不发任何音频

### 6.3 TTS 端不再需要正则过滤

之前在 TTS 端用正则去掉 `<think>...</think>` 标签的方案被放弃，原因：
- 标签可能不闭合导致误删正文
- reasoning 在 API 层已经被过滤，TTS 端收到的都是纯正文

## 7. TTS 流式分片策略

### 7.1 问题背景

TTS 服务（DashScope Qwen3-TTS）限制单次请求最多 500 字符。高并发时 LLM 吞吐可能降至 20 token/s，需要防止语音播放断档。

### 7.2 设计目标

1. **首字低延迟**：用户说完后尽快听到第一个字
2. **顺序无重叠**：避免多片音频在播放端叠加/乱序
3. **超时保底**：LLM 慢时最多等 5 秒就发下一片

### 7.3 分片发送策略

| 阶段 | 触发条件 | 说明 |
|------|----------|------|
| 首片 | ≥30 字符 **或** 遇到句末符（。！？；\n） | 立即发，最小化首字延迟 |
| 后续片 | 上一片 TTS 完成 **且** 新文本已累积 ≥1 个字符 | 顺序发送，前一片完成才发下一片 |
| 超时保底 | 上一片发出后已等待 ≥5 秒 | 强制发送，防止 LLM 过慢时断档 |

**API 上限**：MAX_TTS_CHUNK = 300 字符（API 限制 500，留余量）
**分片断点**：优先在句末符（。！？；\n）处切割，保持句子完整

### 7.4 实现方案

```python
async def _run(self, output_emitter: AudioEmitter) -> None:
    first_sent = False          # 首片标记
    last_send_time = t0         # 上次发送时间（用于超时检测）
    MAX_WAIT_SEC = 5.0          # 超时阈值

    async def send_tts_chunk(text: str) -> None:
        nonlocal first_sent, last_send_time
        # 发起 TTS 请求，流式读取音频并通过 output_emitter.push() 输出
        await session.post(...)
        first_sent = True
        last_send_time = time.monotonic()

    async for item in self._input_ch:
        pending_text += item

        time_since_last = time.monotonic() - last_send_time
        timeout_trigger = first_sent and time_since_last >= MAX_WAIT_SEC

        if not first_sent:
            can_send = len(pending_text) >= 50 or re.search(r'[。！？；\n]', pending_text)
        elif timeout_trigger:
            can_send = True
        else:
            can_send = False

        while can_send and pending_text:
            # 在句末符处切割
            m = re.search(r'[。！？；\n](.{0,50})$', pending_text)
            cut_pos = pending_text.rfind(m.group(0)) + 1 if m else len(pending_text)
            if cut_pos < len(pending_text) * 0.3:  # 找不到句末符，硬切
                cut_pos = min(len(pending_text), MAX_TTS_CHUNK - 20)

            send_text = pending_text[:cut_pos]
            pending_text = pending_text[cut_pos:]

            await send_tts_chunk(send_text)

            if not first_sent:
                can_send = len(pending_text) >= 50 or re.search(r'[。！？；\n]', pending_text)
            else:
                can_send = False  # 等待下一个 item 或超时触发
```

### 7.5 为什么不会重叠播放

1. **单 segment**：整个回答只有一个 `start_segment` / `end_segment`，所有音频在同一流中
2. **`await send_tts_chunk()`**：每片 TTS 完成后才发下一片，无并发
3. **`output_emitter.push()`**：SDK 底层按顺序排队播放，不会叠加

### 7.6 配置项

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `TTS_TIMEOUT` | `30` | TTS 单次请求超时（秒） |

（`MAX_WAIT_SEC = 5.0` 和 `MAX_TTS_CHUNK = 300` 为代码常量，暂不需要环境变量化）

## 8. 文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `agent/agent.py` | 添加思考模式存储、ChatHistoryManager、工具函数、`llm_node` reasoning 过滤 |
| `agent/tts_adapter.py` | 流式分片策略：首片低延迟 + await 顺序发送 + 超时保底 |
| `doc/design-llm-thinking-and-history.md` | 本文档 |

## 9. 配置项汇总

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LLM_CHAT_TEMPLATE_KWARGS` | `{"enable_thinking": false}` | LLM 模板参数 |
| `CHAT_HISTORY_MAX_TURNS` | `10` | 聊天历史保留轮数 |
| `TTS_TIMEOUT` | `10` | TTS 单次请求超时（秒） |

## 10. 注意事项

1. **思考模式参数兼容性**：`enable_thinking` 是 DashScope Qwen 的参数格式，换模型时需调整 `MODEL_THINKING_PARAM_MAP` 映射表
2. **动态思考模式**：通过 `llm_node` 每次调用时读取 `_thinking_mode_store` 状态，实现真正的动态切换
3. **截断时机**：截断在 `on_user_turn_completed` 中执行，确保用户说话后及时清理
4. **session 生命周期**：思考模式存储在内存中，session 结束后自动清除
5. **工具函数命名**：避免与 OpenAI 工具规范冲突，使用 `set_thinking_mode` 和 `get_thinking_mode_status`
6. **TTS 分片断点**：优先在句末符切割，如找不到则硬切（最大 480 字符）
7. **reasoning 过滤位置**：在 `llm_node` 层过滤，不传给下游（AgentSession → TTS）