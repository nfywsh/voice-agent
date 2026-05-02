# 音频采样率转换设计文档

## 1. 问题描述

在全双工语音聊天机器人系统中，音频需要经过多个处理节点，每个节点对采样率的要求不同。如果采样率不匹配，会导致以下问题：

- **TTS/歌声输出失真**：24kHz 音频被 LiveKit 按 48kHz 解读，声音变快变尖（花栗鼠效应）
- **ASR 识别率下降**：48kHz 音频直接送入期望 16kHz 输入的 Deepgram，识别错误率飙升
- **VAD 误触发**：Silero VAD 模型训练在 16kHz 数据上，输入 48kHz 会导致语音检测阈值失效

## 2. 全链路音频格式

```
用户麦克风 (48kHz, Opus)
    │
    ▼  LiveKit 解码
Agent 接收 (48kHz, PCM, mono)
    │
    ├─► VAD 检测 (重采样到 16kHz, PCM) ──► Silero VAD
    │
    ├─► ASR 识别 (重采样到 16kHz, PCM) ──► Deepgram
    │
    ▼  TTS/Singing 输出
TTS 生成 (24kHz, PCM, mono)
    │
    ▼  重采样 (24kHz → 48kHz)
Singing 生成 (24kHz, PCM, mono)
    │
    ▼  重采样 (24kHz → 48kHz)
Agent 输出 (48kHz, PCM, mono)
    │
    ▼  LiveKit 编码
用户扬声器 (48kHz, Opus)
```

## 3. 各环节采样率转换详情

### 3.1 输入侧：麦克风 48kHz → ASR/VAD 16kHz

| 参数 | 值 | 说明 |
|------|-----|------|
| WebRTC 传输采样率 | 48kHz | Opus 编解码器标准格式 |
| VAD 输入采样率 | 16kHz | Silero VAD 训练采样率，仅支持 8kHz 和 16kHz |
| ASR 输入采样率 | 16kHz | Deepgram 推荐输入格式 |
| 编码格式 | linear16 (PCM) | LiveKit 从 WebRTC Opus 解码后输出 |

**处理方式**：LiveKit Agents SDK 内置自动重采样。

具体来说，`deepgram.STT()` 和 `silero.VAD.load()` 在接收到 48kHz PCM 流时，SDK 内部会自动完成 48kHz→16kHz 的下采样。开发者无需手动处理。

**代码位置**：`agent/agent.py`

```python
# VAD — SDK 内部自动将 48kHz 输入下采样到 16kHz
vad = silero.VAD.load(
    activation_threshold=0.5,
    min_speech_duration=0.2,
    min_silence_duration=0.3,
)

# STT — SDK 内部自动将 48kHz 输入下采样到 16kHz
stt = deepgram.STT(
    api_key=deepgram_key,
    language="zh-CN",
)
```

### 3.2 输出侧（TTS）：24kHz → 48kHz

| 参数 | 值 | 说明 |
|------|-----|------|
| TTS 输出采样率 | 24kHz | Qwen3-TTS 默认输出采样率 |
| LiveKit 推流采样率 | 48kHz | WebRTC/Opus 标准格式 |
| 位深度 | 16-bit PCM | 全链路统一 |
| 声道数 | 1 (mono) | 语音场景无需立体声 |

**处理方式**：在 `QwenTTSAdapter` 中手动做 24kHz→48kHz 上采样。

**代码位置**：`agent/tts_adapter.py`

```python
from scipy.signal import resample_poly
from math import gcd

OUTPUT_SAMPLE_RATE = 48000  # LiveKit 要求
TTS_SAMPLE_RATE = 24000     # Qwen3-TTS 输出

def _resample_24k_to_48k(pcm_bytes: bytes,
                          source_rate: int = TTS_SAMPLE_RATE,
                          target_rate: int = OUTPUT_SAMPLE_RATE) -> bytes:
    """将 16-bit PCM 音频从 source_rate 重采样到 target_rate。"""
    if source_rate == target_rate:
        return pcm_bytes

    # bytes → int16 numpy array → float32 归一化
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # scipy 多相重采样（高质量，抗混叠）
    g = gcd(target_rate, source_rate)  # 24k→48k: gcd=24000, up=2, down=1
    audio_resampled = resample_poly(audio, target_rate // g, source_rate // g)

    # float32 → int16 → bytes
    audio_int16 = np.clip(audio_resampled * 32767, -32768, 32767).astype(np.int16)
    return audio_int16.tobytes()
```

**重采样算法选择**：

| 算法 | 质量 | 速度 | 选择原因 |
|------|------|------|----------|
| `scipy.signal.resample_poly` | 高 | 中 | ✅ 使用多相滤波，抗混叠，适合语音 |
| `librosa.resample` | 高 | 慢 | 依赖 librosa（体积大），不必要 |
| 线性插值 | 低 | 快 | 语音频段有明显失真，不采用 |
| `scipy.signal.resample`（FFT） | 中 | 快 | 周期信号效果好，但对语音不如多相 |

### 3.3 输出侧（歌声）：24kHz → 48kHz

与 TTS 相同，VibeVoice-1.5B 输出 24kHz PCM，需要上采样到 48kHz。

**代码位置**：`agent/singing_handler.py`

```python
# 与 tts_adapter.py 中相同的重采样逻辑
def _resample_24k_to_48k(pcm_bytes, source_rate=24000, target_rate=48000):
    # ... 同上 ...
```

**注意**：歌声重采样函数在 `singing_handler.py` 中独立实现（而非从 tts_adapter 导入），是为了保持模块独立性，避免两个微服务之间的代码耦合。

## 4. 流式输出中的重采样策略

TTS 和歌声服务都是流式返回音频（每次返回一小块），而非等待全部生成完毕再返回。这带来一个关键问题：**块边界处的重采样如何处理？**

### 方案：累积缓冲 + 帧对齐

```
TTS/Singing 流式输出 (24kHz, 块大小 4096 bytes)
    │
    ▼
累积到足够一帧 (≥ 1920 bytes @ 24kHz = 40ms)
    │
    ▼
对累积数据做完整重采样 (24kHz → 48kHz)
    │
    ▼
输出重采样后的帧 (48kHz)
    │
    ▼
推送到 LiveKit (48kHz, Opus 编码)
```

**关键参数**：

| 参数 | 值 | 说明 |
|------|-----|------|
| TTS 流式块大小 | 4096 bytes | 从 TTS 服务每次读取的字节数 |
| 累积阈值 | 3840 bytes @ 48kHz (≈ 1920 bytes @ 24kHz) | 约等于 20ms @ 48kHz 的音频量 |
| 歌声流式块大小 | 4096 bytes | 从 Singing 服务每次读取的字节数 |

**代码位置**：`agent/tts_adapter.py` 中 `QwenTTSStream._run()` 方法

```python
# 逐块读取音频
buffer = b""
async for chunk in resp.content.iter_chunked(4096):
    buffer += chunk
    # 累积到足够一帧再输出（约 20ms @ 48kHz = 1920 samples = 3840 bytes）
    if len(buffer) >= 1920:  # 24kHz 下的阈值（重采样后变为 48kHz 3840 bytes）
        resampled = _resample_24k_to_48k(buffer)
        frame = tts.AudioFrame(
            data=resampled,
            sample_rate=48000,
            num_channels=1,
        )
        self._output_ch.send_nowait(frame)
        buffer = b""

# 输出剩余数据
if buffer:
    resampled = _resample_24k_to_48k(buffer)
    frame = tts.AudioFrame(...)
    self._output_ch.send_nowait(frame)
```

## 5. 采样率配置参数汇总

所有采样率相关参数集中在代码中定义，方便统一调整：

```python
# agent/tts_adapter.py
OUTPUT_SAMPLE_RATE = 48000   # LiveKit 推流采样率（固定，不可修改）
TTS_SAMPLE_RATE = 24000      # Qwen3-TTS 输出采样率

# agent/singing_handler.py
OUTPUT_SAMPLE_RATE = 48000   # LiveKit 推流采样率（固定，不可修改）
SINGING_SAMPLE_RATE = 24000  # VibeVoice 输出采样率

# agent/agent.py 中 VAD 配置
# VAD 输入采样率由 SDK 自动处理，固定为 16kHz
vad = silero.VAD.load(
    activation_threshold=0.5,  # 可调：语音检测灵敏度
    min_speech_duration=0.2,   # 可调：最短语音持续时间（秒）
    min_silence_duration=0.3,  # 可调：最短静默持续时间（秒）
)
```

## 6. 常见问题排查

| 现象 | 可能原因 | 排查方法 |
|------|---------|---------|
| AI 声音变快变尖（花栗鼠） | TTS 24kHz 未重采样直接推 48kHz | 检查 `tts_adapter.py` 的 `OUTPUT_SAMPLE_RATE` 是否为 48000 |
| AI 声音变慢变沉 | 48kHz 音频被误标为 24kHz | 检查 `AudioFrame` 的 `sample_rate` 参数 |
| ASR 识别乱码/错误率高 | Deepgram 收到非 16kHz 音频 | 检查 SDK 版本是否支持自动重采样 |
| VAD 频繁误触发/漏检 | VAD 收到非 16kHz 音频 | 检查 `silero.VAD.load()` 参数 |
| 歌声失真 | singing_handler 未做重采样 | 检查 `singing_handler.py` 的 `_resample_24k_to_48k` |
| 流式音频有"卡顿"感 | 重采样块缓冲区太小 | 增大累积阈值至 40ms 对应字节数 |
| 流式音频有"重叠"杂音 | 重采样块缓冲区边界不对齐 | 确保每次重采样是对累积的连续数据操作，而非对单个块独立重采样 |

## 7. 测试验证

### 7.1 采样率正确性验证

```python
# 测试脚本：验证重采样后音频长度正确
import numpy as np
from scipy.signal import resample_poly
from math import gcd

def test_resample():
    # 生成 1 秒 24kHz 正弦波
    t = np.linspace(0, 1, 24000, dtype=np.float32)
    audio_24k = np.sin(2 * np.pi * 440 * t)
    pcm_24k = (audio_24k * 32767).astype(np.int16).tobytes()

    # 重采样
    audio = np.frombuffer(pcm_24k, dtype=np.int16).astype(np.float32) / 32768.0
    g = gcd(48000, 24000)
    audio_48k = resample_poly(audio, 48000 // g, 24000 // g)
    pcm_48k = np.clip(audio_48k * 32767, -32768, 32767).astype(np.int16).tobytes()

    # 验证
    assert len(audio_48k) == 48000, f"Expected 48000 samples, got {len(audio_48k)}"
    print(f"24kHz: {len(audio_24k)} samples → 48kHz: {len(audio_48k)} samples ✅")

test_resample()
```

### 7.2 端到端验证流程

1. 启动所有服务
2. 打开前端页面，加入房间
3. 说"你好"，听 AI 回复是否正常语速、音调
4. 说"唱一首关于星空的歌"，听歌声是否正常
5. 在 AI 说话时打断，检查打断是否立即生效

---

*文档版本：v1.0*
*创建日期：2026-05-01*
*关联设计文档：desgin.md 第 3.8 节*