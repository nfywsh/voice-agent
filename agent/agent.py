# agent/agent.py
"""LiveKit 全双工语音 Agent 主程序

架构说明：
- VoiceAssistant 继承 Agent，定义系统指令和工具函数
- FunASRSTT / OpenAISTT 通过 HTTP 对接本地 Fun-ASR 实时语音识别
- LLM 通过 OpenAI 兼容 API 对接本地 VLLM (Qwen3.6-35B-A3B)
- Qwen3TTSAdapter 对接本地 VLLM TTS (Qwen3-TTS-Base)，24kHz→48kHz 重采样
- SingingHandler 对接 sing_agent 歌声服务，流式推歌声音频到 LiveKit

API 配置（全部通过环境变量注入）：
- OPENAI_ASR_BASE_URL: Fun-ASR HTTP 地址
- OPENAI_LLM_BASE_URL: VLLM LLM 地址
- QWEN3_TTS_BASE_URL: VLLM TTS 地址
- SING_AGENT_URL: 歌声服务地址
- LLM_CHAT_TEMPLATE_KWARGS: LLM 模板参数 JSON，默认 {"enable_thinking": false}
- CHAT_HISTORY_MAX_TURNS: 聊天历史保留轮数，默认 10

打断机制：
- VAD 检测到用户说话 → Agent 自动 interrupt() → 取消当前 LLM 推理
- 等用户新输入完成 → 合并上下文 → 发起新 LLM 请求

System Prompt 注入：
- 环境变量 SYSTEM_PROMPT（简单 demo）
- HTTP 接口 PROMPT_SERVICE_URL（生产环境，支持热更新）

思考模式：
- 默认关闭，按 room+user 独立存储在内存中
- 通过 _thinking_mode_store 管理，session 生命周期内有效
- 调用 LLM 时通过 extra_kwargs.chat_template_kwargs 传递

聊天历史：
- 默认保留最近 N 轮（CHAT_HISTORY_MAX_TURNS 配置）
- 通过 ChatHistoryManager 管理注入和截断
- 在 on_user_turn_completed 中自动截断
"""

import asyncio
import json
import logging
import os
from typing import AsyncIterable, Optional

import aiohttp
import numpy as np
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    ChatMessage,
    JobContext,
    RoomInputOptions,
    cli,
    function_tool,
    llm,
)
from livekit.plugins import silero

from dashscope_stt import DashScopeSTT
from singing_handler import SingingHandler, _resample_24k_to_48k
from tts_adapter import QwenTTSAdapter
from monitoring.metrics import MetricsCollector
from livekit.plugins import openai as lk_openai

load_dotenv()

# ============================================================
# 配置项
# ============================================================

# LLM 模板参数（思考模式等），默认关闭思考
LLM_CHAT_TEMPLATE_KWARGS = json.loads(
    os.environ.get("LLM_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}')
)

# 聊天历史保留轮数
CHAT_HISTORY_MAX_TURNS = int(os.environ.get("CHAT_HISTORY_MAX_TURNS", "10"))


# ============================================================
# 思考模式存储（内存）
# ============================================================
# key 格式: "{room_id}:{user_id}"
# 仅在 session 生命周期内有效，session 结束后自动清除
_thinking_mode_store: dict[str, bool] = {}


def get_thinking_mode(room_id: str, user_id: str) -> bool:
    """获取当前 session 的思考模式状态"""
    return _thinking_mode_store.get(f"{room_id}:{user_id}", False)


def set_thinking_mode(room_id: str, user_id: str, enabled: bool) -> None:
    """设置当前 session 的思考模式状态"""
    _thinking_mode_store[f"{room_id}:{user_id}"] = enabled
    logger.info(f"[thinking_mode] room={room_id}, user={user_id}, enabled={enabled}")


def clear_thinking_mode(room_id: str, user_id: str) -> None:
    """清除思考模式状态（session 结束时调用）"""
    key = f"{room_id}:{user_id}"
    if key in _thinking_mode_store:
        del _thinking_mode_store[key]


# ============================================================
# 聊天历史管理器
# ============================================================


class ChatHistoryManager:
    """聊天历史管理器，提供外部注入和截断能力"""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns

    def inject_messages(self, chat_ctx: ChatContext, messages: list[dict]) -> None:
        """注入外部消息到聊天历史

        Args:
            chat_ctx: Agent 的 ChatContext 实例
            messages: 消息列表，格式为 [{"role": "user"|"assistant", "content": "..."}]
        """
        for msg in messages:
            chat_ctx.add_message(role=msg["role"], content=msg["content"])
        logger.info(f"[chat_history] Injected {len(messages)} messages, truncating to {self.max_turns} turns")
        self.truncate(chat_ctx)

    def truncate(self, chat_ctx: ChatContext) -> None:
        """截断聊天历史到指定轮数"""
        chat_ctx.truncate(max_items=self.max_turns)


# ============================================================
# 日志配置
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("dashscope_stt").setLevel(logging.DEBUG)
logging.getLogger("livekit.agents.voice.agent_activity").setLevel(logging.DEBUG)
logging.getLogger("livekit.agents.voice.audio_recognition").setLevel(logging.DEBUG)
logging.getLogger("livekit.agents.voice.room_io").setLevel(logging.DEBUG)
logging.getLogger("livekit.agents.stt").setLevel(logging.DEBUG)
logging.getLogger("livekit.agents.llm").setLevel(logging.DEBUG)
logging.getLogger("livekit.plugins.silero").setLevel(logging.DEBUG)

# ============================================================
# System Prompt 注入入口
# ============================================================
# System Prompt 应由外部系统注入，本模块不硬编码任何业务人设。
# 注入方式：
#   1. 环境变量 SYSTEM_PROMPT：适合简单 demo / 测试场景
#   2. HTTP 接口 PROMPT_SERVICE_URL：适合生产环境，支持热更新
#
# Prompt 内容应包含：
#   - 角色设定（你是一个语音助手…）
#   - 工具调用规范（简洁调用，避免冗余参数）
#   - 打断行为的说明（用户可在任意时刻打断）
#   - 对话格式约束（短句优先，避免超长回复导致 TTS 延迟高）
# ============================================================

_DEFAULT_SYSTEM_PROMPT = (
    "你是一个友好、热情的语音助手，具有唱歌能力。\n"
    "## 工具调用规范\n"
    "- 仅在用户明确表达意图时调用工具\n"
    "- 唱歌时，你需要根据用户要求创作歌词，格式为 LRC 带时间戳：\n"
    "  [start]\n"
    "  [intro]\n"
    "  [00:00.00][verse]\n"
    "  歌词内容第一句\n"
    "  [00:00.05][verse]\n"
    "  歌词内容第二句\n"
    "  （注：[start] 和 [intro] 标记结构，[00:xx.xx] 为时间戳，[verse] 为行标签，可选 rap/melodic/chorus 等；前奏 [intro] 不超过 5 秒）\n"
    "- 工具调用返回后，给出简短的过渡语，不要重复工具返回的文本，更不要生成歌词内容\n"
    "- 重要：唱歌工具（sing_a_song）调用后，**禁止**再生成或朗读歌词，歌曲音频会直接播放\n"
    "## 对话风格\n"
    "- 回复简洁自然，适合语音播报\n"
    "- 避免超长回复（控制在 100 字以内），用户可以随时打断\n"
    "- 用中文回答所有问题\n"
)


class PromptService:
    """System Prompt 动态拉取服务（生产环境方案）。

    适合需要 Prompt 热更新、A/B 测试或按用户分组的场景。
    使用方式：设置环境变量 PROMPT_SERVICE_URL，Agent 启动时自动拉取。
    """

    def __init__(self, endpoint: str, refresh_interval: int = 300):
        self._endpoint = endpoint
        self._refresh_interval = refresh_interval
        self._cache: dict[str, str] = {}

    async def get_prompt(self, room_id: str = "default") -> str:
        """拉取指定房间的 System Prompt。"""
        if room_id not in self._cache:
            try:
                async with aiohttp.ClientSession() as session:
                    resp = await session.get(
                        f"{self._endpoint}?room={room_id}",
                        timeout=aiohttp.ClientTimeout(total=5.0),
                    )
                    if resp.status == 200:
                        data = await resp.json()
                        self._cache[room_id] = data.get("prompt", _DEFAULT_SYSTEM_PROMPT)
                    else:
                        logger.warning(f"Prompt service returned {resp.status}, using default")
                        self._cache[room_id] = _DEFAULT_SYSTEM_PROMPT
            except Exception as e:
                logger.warning(f"Failed to fetch prompt: {e}, using default")
                self._cache[room_id] = _DEFAULT_SYSTEM_PROMPT
        return self._cache[room_id]


def _get_system_prompt() -> str:
    """获取 System Prompt（环境变量方式，简单 demo）。"""
    return os.environ.get("SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT)


async def _fetch_session_params(room_id: str):
    """从 backend 获取会话参数 (system_prompt, chat_history)"""
    backend_url = os.environ.get("BACKEND_URL", "http://backend:3000")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{backend_url}/api/session/params?room={room_id}",
                timeout=aiohttp.ClientTimeout(total=3.0),
            ) as resp:
                if resp.ok:
                    data = await resp.json()
                    logger.info(f"[_fetch_session_params] room={room_id}: "
                                f"system_prompt={'set' if data.get('system_prompt') else 'None'}, "
                                f"chat_history={len(data.get('chat_history', []))} messages")
                    return data
    except Exception as e:
        logger.warning(f"[_fetch_session_params] failed for room={room_id}: {e}")
    return None


# ============================================================
# 主 Agent 类
# ============================================================

class VoiceAssistant(Agent):
    """全双工语音助手"""

    def __init__(self, *, singing_handler: SingingHandler, instructions: str = None) -> None:
        if instructions is None:
            instructions = _get_system_prompt()
        super().__init__(instructions=instructions)
        self.singing_handler = singing_handler
        self.chat_history_manager = ChatHistoryManager(max_turns=CHAT_HISTORY_MAX_TURNS)
        self._chat_history_injected = False
        self._pending_chat_history = None
        # 歌曲播放状态
        self._tts_idle_event = asyncio.Event()
        self._tts_idle_event.set()  # 初始为空闲
        self._pending_song_path: Optional[str] = None
        self._song_playback_task: Optional[asyncio.Task] = None

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """用户说完一句话后，自动截断聊天历史避免无限累积"""
        self.chat_history_manager.truncate(turn_ctx)
        # 通知 metrics 当前 turn 结束（触发 request_end，但保留 trace 以供查询）
        metrics = getattr(self, '_metrics', None)
        if metrics:
            request_id = metrics._current_request_id.get()
            if request_id:
                metrics.request_end(request_id)

    # ============================================================
    # 思考模式控制
    # ============================================================

    def set_thinking(self, enabled: bool, room_id: str = "", user_id: str = "") -> None:
        """设置当前 Agent 的思考模式

        Args:
            enabled: 是否启用思考模式
            room_id: 房间 ID（用于日志）
            user_id: 用户 ID（用于日志）
        """
        # 从 session 中获取 room 和 user 信息
        if not room_id and self.session:
            room_id = self.session.room.name if self.session.room else "unknown"
        if not user_id and self.session:
            user_id = self.session.user_id or "unknown"

        set_thinking_mode(room_id, user_id, enabled)

    def is_thinking(self, room_id: str = "", user_id: str = "") -> bool:
        """获取当前 Agent 的思考模式状态"""
        if not room_id and self.session:
            room_id = self.session.room.name if self.session.room else "unknown"
        if not user_id and self.session:
            user_id = self.session.user_id or "unknown"

        return get_thinking_mode(room_id, user_id)

    # ============================================================
    # LLM 节点重写（支持动态思考模式）
    # ============================================================

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings=None,
    ) -> AsyncIterable[llm.ChatChunk | str | None]:
        """重写 LLM 节点，在调用前根据思考模式动态注入 extra_kwargs

        通过 self.session.llm.chat() 直接调用，传入 extra_kwargs 控制思考模式。
        同时过滤掉 reasoning 字段（思考内容），只将正文 content 传给下游（LLM/Agent 会自动发给 TTS）。
        """
        # 从 session 获取 room 信息
        # 优先使用 agent 实例自身存储的 room_id（entrypoint 传入的 ctx.room.name）
        session_room = getattr(self.session, 'room', None)
        room_id = getattr(self, '_room_id', None) or (session_room.name if session_room else "default_room")
        # user_id 同样使用 entrypoint 传入的值
        user_id = getattr(self, '_user_id', None) or f"agent_{room_id}"
        # 构建 chat_template_kwargs 顶层参数（VLLM 只识别顶层，不识别 extra_body 包装）
        # DashScope 格式也放在顶层，与 vllm_llm.py 保持一致
        is_thinking = get_thinking_mode(room_id, user_id)
        # VLLM recognizes chat_template_kwargs at TOP LEVEL (NOT extra_body wrapped)
        extra_kwargs = {
            "chat_template_kwargs": {"enable_thinking": is_thinking}
        }

        import time
        t0 = time.monotonic()
        first_chunk = True

        logger.info(f"[llm_node] Thinking mode: {is_thinking} for room={room_id}, tools count={len(tools)}")

        # 获取 metrics 实例（从 self._metrics 或全局）
        metrics = getattr(self, '_metrics', None)

        if metrics:
            metrics.llm_start()
            # 提取对话内容用于监控显示（精简格式：只保留 role + content）
            items = chat_ctx.to_dict().get("items", [])
            simplified = [
                {"role": it.get("role"), "content": it.get("content")}
                for it in items
                if it.get("role") in ("system", "user", "assistant") and it.get("content")
            ]
            import json
            metrics.llm_input(json.dumps(simplified, ensure_ascii=False))

        # 通过 activity 访问 llm（与 SDK 默认实现一致的方式）
        activity = self._get_activity_or_raise()
        activity_llm = activity.llm

        async with activity_llm.chat(
            chat_ctx=chat_ctx,
            tools=tools,
            extra_kwargs=extra_kwargs,
        ) as stream:
            async for chunk in stream:
                if first_chunk:
                    logger.info(f"[llm_node] First token after {time.monotonic() - t0:.3f}s")
                    if metrics:
                        metrics.llm_first_token()
                    first_chunk = False

                if metrics and hasattr(chunk, 'delta') and hasattr(chunk.delta, 'content') and chunk.delta.content:
                    metrics.llm_output(chunk.delta.content)

                yield chunk

        if metrics:
            metrics.llm_end()
        logger.info(f"[llm_node] Done, total time: {time.monotonic() - t0:.3f}s")

    # ============================================================
    # 工具函数
    # ============================================================

    @function_tool
    async def sing_a_song(self, title: str = "", lyrics: str = "", style: str = "流行") -> str:
        """当用户要求唱歌时调用此工具。

        歌词格式（LRC 带时间戳）：
        [start]
        [intro]
        [00:00.00][verse]
        歌词内容第一句
        [00:00.05][verse]
        歌词内容第二句

        决策逻辑：
        - lyrics 为空（简单请求如"唱首歌"）→ 随机歌曲 + 音色转换（快速）
        - lyrics 有内容（复杂请求如"唱一首关于...的歌"）→ /sing/generate 创作（较慢）

        Args:
            title: 歌曲名称（可选，为空则使用即兴歌曲）
            lyrics: LRC 格式歌词（可选，为空则使用随机歌曲 + 音色转换）
            style: 歌曲风格，如流行、民谣、摇滚等
        """
        room_id = getattr(self, '_room_id', None) or "default"
        logger.info(f"[sing_a_song] title={title}, style={style}, lyrics_len={len(lyrics)}, room={room_id}")

        # 返回简短回复，让 LLM 继续生成自然的后续内容（TTS 会播放）
        # 歌曲在后台生成，完成后等 TTS 空闲再播放
        # 重要：清除 idle 状态，确保歌曲等待本次 TTS 完成
        self._tts_idle_event.clear()
        asyncio.create_task(self._generate_and_play_song(title, lyrics, style, room_id))
        return "好的，正在为你创作这首歌曲，请稍候..."

    async def _generate_and_play_song(self, title: str, lyrics: str, style: str, room_id: str):
        """后台生成歌曲，完成后等待 TTS 空闲再播放"""
        song_path = None
        SING_AGENT_URL = os.environ.get("SING_AGENT_URL", "http://localhost:8080")

        try:
            async with aiohttp.ClientSession() as session:
                if not lyrics:
                    # 快速路径：随机歌曲 + 音色转换
                    async with session.get(
                        f"{SING_AGENT_URL}/sing/random-song",
                        timeout=aiohttp.ClientTimeout(total=10.0),
                    ) as resp:
                        if not resp.ok:
                            logger.error(f"[_generate_and_play_song] random-song failed: {resp.status}")
                            return
                        data = await resp.json()
                        song_path = data.get("song_path")

                    voice_path = await self._get_room_voice_path(room_id)
                    if voice_path:
                        async with session.post(
                            f"{SING_AGENT_URL}/sing/convert-direct",
                            json={"song_path": song_path, "voice_path": voice_path},
                            timeout=aiohttp.ClientTimeout(total=60.0),
                        ) as resp:
                            if resp.ok:
                                result = await resp.json()
                                song_path = result.get("result_path")
                else:
                    # 生成路径
                    voice_path = await self._get_room_voice_path(room_id)
                    payload = {"lyrics": lyrics, "style": style, "model_type": "acestep", "duration": 90}
                    if voice_path:
                        payload["voice_path"] = voice_path
                    async with session.post(
                        f"{SING_AGENT_URL}/sing/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=120.0),
                    ) as resp:
                        if resp.ok:
                            result = await resp.json()
                            song_path = result.get("song_path") or result.get("vocal_path")

            if not song_path:
                logger.error(f"[_generate_and_play_song] No song_path generated")
                return

            logger.info(f"[_generate_and_play_song] Song generated: {song_path}, waiting for TTS idle")

            # 等待 TTS 播放完成（此时 LLM 的后续内容正在 TTS 播放）
            await self._tts_idle_event.wait()

            # TTS 播完了，开始播放歌曲
            logger.info(f"[_generate_and_play_song] TTS idle, playing song now")
            self._song_playback_task = asyncio.create_task(self._play_song_file(song_path))

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[_generate_and_play_song] Error: {e}")

    async def _get_room_voice_path(self, room_id: str) -> Optional[str]:
        """获取room对应的音色文件路径，优先用 agent 入房时缓存的，无则查 backend"""
        # 优先用 entrypoint 入房时缓存的 ref_audio_path
        cached = getattr(self, '_ref_audio_path', None)
        if cached:
            logger.info(f"[_get_room_voice_path] using cached ref_audio_path: {cached}")
            return cached
        # Fallback: 查 backend session
        backend_url = os.environ.get("BACKEND_URL", "http://backend:3000")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{backend_url}/api/session/params?room={room_id}",
                    timeout=aiohttp.ClientTimeout(total=3.0),
                ) as resp:
                    if resp.ok:
                        data = await resp.json()
                        voice_path = data.get("ref_audio_path")
                        logger.info(f"[_get_room_voice_path] session_data={data}")
                        if voice_path:
                            return voice_path
                    else:
                        logger.warning(f"[_get_room_voice_path] HTTP {resp.status}")
        except Exception as e:
            logger.warning(f"[_get_room_voice_path] {e}")
        # 最后 fallback: 环境变量默认音色
        default_voice = os.environ.get("QWEN3_TTS_REF_AUDIO", "")
        if default_voice:
            logger.info(f"[_get_room_voice_path] using env default: {default_voice}")
            return default_voice
        return None

    async def _play_song_file(self, file_path: str) -> None:
        """播放本地歌曲文件（支持 MP3/WAV 等格式，重采样到 48kHz 后推流到 LiveKit）"""
        try:
            import av
            audio_source = rtc.AudioSource(sample_rate=48000, num_channels=1)
            track = rtc.LocalAudioTrack.create_audio_track("song-audio", audio_source)

            # 获取 room 并发布轨道
            # AgentSession → room_io.room (AgentSession 没有直接 room 属性)
            logger.info(f"[_play_song_file] self={id(self)}, session={id(getattr(self, 'session', None))}")
            session = getattr(self, 'session', None)
            room_io = getattr(session, 'room_io', None) if session else None
            room = getattr(room_io, 'room', None) if room_io else None
            logger.info(f"[_play_song_file] room_io={id(room_io)}, room={id(room)}, local_participant={id(getattr(room, 'local_participant', None)) if room else None}")
            if room:
                publication = await room.local_participant.publish_track(track)
                track_sid = publication.sid
                logger.info(f"[_play_song_file] Published track for: {file_path}, sid={track_sid}")
            else:
                logger.warning(f"[_play_song_file] No room available, audio won't be heard")
                return

            # 使用 PyAV 解码任意格式音频
            input_container = av.open(file_path)
            input_stream = input_container.streams.audio[0]
            input_stream.thread_type = 'AUTO'

            source_rate = input_stream.rate
            total_frames = 0
            logger.info(f"[_play_song_file] Playing: {file_path} @ {source_rate}Hz, format={input_stream.format.name}")

            # 24kHz->48kHz: 2048 samples = 2048*2/24000 = 85.3ms per frame -> resample后 = 4096 samples
            # 分段发送，每段 20ms @ 48kHz = 960 samples = 1920 bytes
            chunk_samples = 960  # 20ms @ 48kHz mono
            resampled_48k = _resample_24k_to_48k
            pending = b""
            for packet in input_container.demux(input_stream):
                for frame in packet.decode():
                    # 重采样到 48kHz mono 16bit
                    audio_np = frame.to_ndarray()
                    if audio_np.ndim == 2:
                        if audio_np.shape[0] == 1:
                            audio_np = audio_np[0]
                        else:
                            audio_np = audio_np.mean(axis=0).astype(np.int16)
                    pcm_bytes = audio_np.tobytes()
                    pcm_48k = resampled_48k(pcm_bytes, source_rate, 48000)
                    pending += pcm_48k
                    # 累积到 20ms 一段再发送
                    while len(pending) >= chunk_samples * 2:
                        segment = pending[:chunk_samples * 2]
                        pending = pending[chunk_samples * 2:]
                        frame_out = rtc.AudioFrame(
                            data=segment,
                            sample_rate=48000,
                            num_channels=1,
                            samples_per_channel=chunk_samples,
                        )
                        await audio_source.capture_frame(frame_out)
                        total_frames += 1

            input_container.close()
            logger.info(f"[_play_song_file] Total frames sent: {total_frames}, duration: {total_frames * 0.02:.1f}s")
            await room.local_participant.unpublish_track(track_sid)
            await audio_source.aclose()
            logger.info(f"[_play_song_file] Finished: {file_path}")

        except asyncio.CancelledError:
            logger.info(f"[_play_song_file] Cancelled: {file_path}")
        except Exception as e:
            logger.error(f"[_play_song_file] Error: {e}")

    @function_tool
    async def get_weather(self, city: str) -> str:
        """获取指定城市的天气信息。

        Args:
            city: 城市名称，如"北京"、"上海"
        """
        # TODO: 接入真实天气 API
        logger.info(f"[get_weather] city={city}")
        return f"{city}今天晴，气温 15-25°C，空气质量良好。"

    @function_tool
    async def search_web(self, query: str) -> str:
        """搜索网络获取实时信息。

        Args:
            query: 搜索关键词
        """
        # TODO: 接入真实搜索 API
        logger.info(f"[search_web] query={query}")
        return (
            f"根据搜索结果，关于「{query}」的信息如下："
            "根据公开资料，该问题涉及的内容目前没有明确的官方结论。"
            "建议您关注相关官方渠道以获取最新信息。"
        )

    @function_tool
    async def set_thinking_mode(self, enabled: bool) -> str:
        """开启或关闭思考模式。思考模式会让模型进行更深入的推理，但响应会更慢。

        Args:
            enabled: true 开启思考模式，false 关闭思考模式
        """
        room_id = getattr(self, '_room_id', None) or "unknown"
        user_id = getattr(self, '_user_id', None) or "unknown"
        set_thinking_mode(room_id, user_id, enabled)
        status = "开启" if enabled else "关闭"
        logger.info(f"[set_thinking_mode] {status}思考模式 for room={room_id}, user={user_id}")
        return f"思考模式已{status}。"

    @function_tool
    async def get_thinking_mode_status(self) -> str:
        """查询当前思考模式的状态"""
        room_id = getattr(self, '_room_id', None) or "unknown"
        user_id = getattr(self, '_user_id', None) or "unknown"
        is_thinking = get_thinking_mode(room_id, user_id)
        status = "开启" if is_thinking else "关闭"
        return f"当前思考模式状态：{status}"

    # ============================================================
    # 歌声音频推流
    # ============================================================

    async def _push_singing_audio(
        self, title: str, lyrics: str, style: str
    ) -> None:
        """从 Singing Service 获取歌声音频并推流到 LiveKit 房间。

        这是最关键的音频推流逻辑：
        1. 调用 SingingHandler 获取 48kHz PCM 音频流
        2. 将音频帧写入 AgentSession 的输出源
        3. LiveKit SDK 自动将音频推送到用户浏览器
        """
        try:
            # 先通过 TTS 播放过渡提示
            logger.info(f"[singing] Starting to sing: {title}")

            audio_source = rtc.AudioSource(
                sample_rate=48000,
                num_channels=1,
            )

            # 流式获取并推送歌声音频
            async for pcm_chunk in self.singing_handler.sing_stream(
                lyrics=lyrics,
                title=title,
                style=style,
            ):
                # 将 PCM bytes 封装为 AudioFrame 并推送
                frame = rtc.AudioFrame(
                    data=pcm_chunk,
                    sample_rate=48000,
                    num_channels=1,
                    samples_per_channel=len(pcm_chunk) // 2,  # 16bit = 2 bytes/sample
                )
                await audio_source.capture_frame(frame)

            logger.info(f"[singing] Finished singing: {title}")

        except asyncio.CancelledError:
            logger.info(f"[singing] Cancelled: {title}")
        except Exception as e:
            logger.error(f"[singing] Error: {e}")


# ============================================================
# ASR 配置（支持多种后端）
# ============================================================

def _create_stt(vad=None):
    """创建 STT 实例。

    根据环境变量选择后端：
    - OPENAI_ASR_BASE_URL 包含 "nginx_gateway" 或 "localhost"/"127.0.0.1": 使用 FunASR HTTP (funasr_stt.py)
    - OPENAI_ASR_API_KEY: 使用 OpenAI 格式 ASR (openai_stt.py)
    - DASHSCOPE_API_KEY: 使用 DashScope Fun-ASR (dashscope_stt.py)

    优先级：本地 FunASR > OpenAI > DashScope
    """
    base_url = os.environ.get("OPENAI_ASR_BASE_URL", "")

    # 本地 FunASR HTTP (funasr-old)
    if "funasr-old" in base_url or "nginx_gateway" in base_url or ("localhost" in base_url) or ("127.0.0.1" in base_url):
        from funasr_stt import create_stt as create_funasr_stt
        logger.info("Using FunASR HTTP (local)")
        return create_funasr_stt(vad=vad)

    # OpenAI 格式 ASR
    openai_asr_key = os.environ.get("OPENAI_ASR_API_KEY", "")
    if openai_asr_key:
        from openai_stt import create_stt as create_openai_stt
        logger.info("Using OpenAI format ASR")
        return create_openai_stt()

    # 回退到 DashScope Fun-ASR
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Either OPENAI_ASR_BASE_URL or DASHSCOPE_API_KEY is required. "
            "Set it in .env or docker-compose.yml"
        )
    model = os.environ.get("DASHSCOPE_ASR_MODEL", "fun-asr-realtime")
    language = os.environ.get("DASHSCOPE_ASR_LANGUAGE", "zh")
    logger.info(f"Using DashScope ASR: model={model}, language={language}")
    return DashScopeSTT(api_key=api_key, model=model, language=language)


# ============================================================
# LLM 思考模式参数注入
# ============================================================
# DashScope Qwen 使用 chat_template_kwargs.enable_thinking 控制思考模式
# 其他模型可能有不同的参数名，需要在此映射

# 模型参数映射表（后续换模型时在这里添加映射）
MODEL_THINKING_PARAM_MAP = {
    # DashScope Qwen 系列
    "qwen": {"param": "chat_template_kwargs", "key": "enable_thinking"},
    # OpenAI GPT 系列（使用 reasoning_effort）
    "gpt": {"param": "extra_kwargs", "key": "reasoning_effort"},
    # Gemini 系列（使用 thinking_budget）
    "gemini": {"param": "extra_kwargs", "key": "thinking_budget"},
}


def build_llm_kwargs(room_id: str, user_id: str) -> dict:
    """构建 LLM 调用参数，包含当前思考模式状态

    Args:
        room_id: 房间 ID
        user_id: 用户 ID

    Returns:
        包含思考模式参数的 kwargs dict，可传入 LLM 构造器或 chat() 调用
    """
    is_thinking = get_thinking_mode(room_id, user_id)
    model = os.environ.get("LLM_MODEL", "qwen3.5-122b-a10b").lower()

    # 根据模型类型选择参数映射
    kwargs = {}
    for model_prefix, param_map in MODEL_THINKING_PARAM_MAP.items():
        if model_prefix in model:
            if param_map["param"] == "chat_template_kwargs":
                # DashScope 格式
                template_kwargs = json.loads(
                    os.environ.get("LLM_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}')
                )
                template_kwargs["enable_thinking"] = is_thinking
                kwargs["extra_kwargs"] = {"chat_template_kwargs": template_kwargs}
            else:
                # 其他格式（extra_kwargs 直接）
                extra_kwargs = kwargs.get("extra_kwargs", {})
                extra_kwargs[param_map["key"]] = is_thinking
                kwargs["extra_kwargs"] = extra_kwargs
            break
    else:
        # 默认：尝试 DashScope 格式
        template_kwargs = json.loads(
            os.environ.get("LLM_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}')
        )
        template_kwargs["enable_thinking"] = is_thinking
        kwargs["extra_kwargs"] = {"chat_template_kwargs": template_kwargs}

    logger.debug(f"[llm_kwargs] room={room_id}, thinking={is_thinking}, kwargs={kwargs}")
    return kwargs


# ============================================================
# LLM 封装类（支持思考模式动态注入）
# ============================================================


class ThinkingModeLLM:
    """LLM 封装类，支持在每次调用时动态注入思考模式参数

    使用方式：
        1. 用 ThinkingModeLLM 包装原始 LLM 实例
        2. 每次对话前通过 set_thinking_mode() 设置思考模式
        3. 调用 wrapped_llm.chat() 时自动注入思考参数
    """

    def __init__(self, llm, room_id: str = "", user_id: str = ""):
        self._llm = llm
        self._room_id = room_id
        self._user_id = user_id

    async def chat(self, chat_ctx: ChatContext, **kwargs) -> None:
        """带思考模式注入的 chat 调用"""
        extra = build_llm_kwargs(self._room_id, self._user_id)

        # 合并 extra_kwargs
        if "extra_kwargs" in kwargs:
            kwargs["extra_kwargs"] = {**kwargs["extra_kwargs"], **extra.get("extra_kwargs", {})}
        else:
            kwargs["extra_kwargs"] = extra.get("extra_kwargs", {})

        await self._llm.chat(chat_ctx, **kwargs)

    def __getattr__(self, name):
        """委托其他属性到原始 LLM"""
        return getattr(self._llm, name)


# ============================================================
# 全局 metrics 单例（entrypoint 共享同一实例，确保 HTTP server 和 room 数据指向同一个 collector）
# ============================================================
_global_metrics: Optional[MetricsCollector] = None


def _get_metrics() -> MetricsCollector:
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


# ============================================================
# 入口函数
# ============================================================

async def entrypoint(ctx: JobContext):
    """Agent 会话入口"""
    logger.info(f"[entrypoint] Agent starting, room: {ctx.room.name}, dispatch_id={getattr(ctx, 'dispatch_id', 'N/A')}")
    logger.info(f"[entrypoint] num_idle_processes env: {os.environ.get('LIVEKIT_NUM_IDLE_PROCESSES', 'NOT SET')}")

    metrics = _get_metrics()

    # 初始化歌声处理器
    singing_url = os.environ.get("SINGING_SERVICE_URL", "http://localhost:8002")
    singing_mock = os.environ.get("SINGING_MOCK_MODE", "false").lower() == "true"
    singing_handler = SingingHandler(
        service_url=singing_url,
        timeout=float(os.environ.get("SINGING_TIMEOUT", "30")),
        mock_mode=singing_mock,
    )

    # 创建 VAD
    vad = silero.VAD.load(
        activation_threshold=float(os.environ.get("VAD_THRESHOLD", "0.5")),
        min_speech_duration=float(os.environ.get("VAD_MIN_SPEECH", "0.2")),
        min_silence_duration=float(os.environ.get("VAD_MIN_SILENCE", "0.3")),
    )

    # 创建 STT（传入 VAD 以启用 VAD 驱动的流式识别）
    stt = _create_stt(vad=vad)

    # 创建 LLM 适配器（使用独立 API 端点）
    # 注意：openai plugin 必须在主线程注册，所以放在模块顶部 import（THREAD 模式下 entrypoint 运行在主线程）

    # 优先使用 OpenAI 格式 LLM 配置
    openai_llm_key = os.environ.get("OPENAI_LLM_API_KEY", "")
    openai_llm_url = os.environ.get("OPENAI_LLM_BASE_URL", "")
    openai_llm_model = os.environ.get("OPENAI_LLM_MODEL", "")

    # 优先使用 VLLM 专用 LLM（处理 chat_template_kwargs 顶层注入）
    # VLLM 的 chat_template_kwargs 必须是请求体顶层字段，不能在 extra_body 中
    if os.environ.get("USE_VLLM_LLM", "") == "1":
        from vllm_llm import create_llm as create_vllm_llm
        llm = create_vllm_llm()
        logger.info(f"[entrypoint] Using VLLM LLM (custom adapter for chat_template_kwargs)")
    elif openai_llm_key and openai_llm_url:
        # 使用 OpenAI 格式 LLM（不支持 VLLM 的 chat_template_kwargs 顶层注入）
        from openai_llm import create_llm as create_openai_llm
        llm = create_openai_llm()
        logger.info(f"[entrypoint] Using OpenAI format LLM: base_url={openai_llm_url}, model={openai_llm_model}")
    else:
        # 使用原始 LLM 配置
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get(
            "LLM_BASE_URL", "https://jiajiatemp.duckdns.org:30002/v1/"
        )
        llm_model = os.environ.get("LLM_MODEL", "qwen3.5-122b-a10b")

        if not llm_api_key:
            raise ValueError(
                "Either OPENAI_LLM_API_KEY or LLM_API_KEY is required for LLM. "
                "Set it in .env or docker-compose.yml"
            )

        logger.info(f"[entrypoint] LLM config: base_url={llm_base_url}, model={llm_model}")
        llm = lk_openai.LLM(
            api_key=llm_api_key,
            base_url=llm_base_url,
            model=llm_model,
        )

    # 获取 room 信息
    room_id = ctx.room.name
    user_id = f"agent_{room_id}"
    logger.info(f"[entrypoint] Room: {room_id}, User: {user_id}")

    # 使用全局 metrics 单例（确保 HTTP server 和 room 数据指向同一个 collector）
    metrics = _get_metrics()
    metrics.session_start()

    # 创建 TTS 适配器（支持 Qwen3-TTS-Base、OpenAI 格式和 DashScope 格式）
    qwen3_tts_url = os.environ.get("QWEN3_TTS_BASE_URL", "")
    openai_tts_key = os.environ.get("OPENAI_TTS_API_KEY", "")
    openai_tts_url = os.environ.get("OPENAI_TTS_BASE_URL", "")

    if qwen3_tts_url:
        from qwen3_tts_adapter import create_tts as create_qwen3_tts
        tts_adapter = create_qwen3_tts()
        tts_adapter._metrics = metrics
        logger.info(f"[entrypoint] Using Qwen3-TTS-Base: base_url={qwen3_tts_url}")
    elif openai_tts_key and openai_tts_url:
        from openai_tts_adapter import create_tts as create_openai_tts
        tts_adapter = create_openai_tts()
        tts_adapter._metrics = metrics
        logger.info(f"[entrypoint] Using OpenAI format TTS: base_url={openai_tts_url}")
    else:
        tts_url = os.environ.get("TTS_SERVICE_URL", "http://localhost:8001")
        tts_adapter = QwenTTSAdapter(
            service_url=tts_url,
            voice=os.environ.get("DASHSCOPE_TTS_VOICE", "Cherry"),
            timeout=float(os.environ.get("TTS_TIMEOUT", "30")),
            max_tts_chunk=int(os.environ.get("TTS_MAX_CHUNK", "300")),
            first_chunk_min=int(os.environ.get("TTS_FIRST_CHUNK_MIN", "30")),
            metrics=metrics,  # 注入 metrics 用于 TTS 内部上报
        )

    # 测试 LLM API 连通性
    try:
        import openai
        test_api_key = openai_llm_key if openai_llm_key else llm_api_key
        test_base_url = openai_llm_url if openai_llm_url else llm_base_url
        test_model = openai_llm_model if (openai_llm_model and openai_llm_url) else llm_model
        oai_client = openai.AsyncOpenAI(api_key=test_api_key, base_url=test_base_url)
        logger.info(f"[entrypoint] Testing LLM API connectivity...")
        resp = await oai_client.chat.completions.create(
            model=test_model,
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=10,
        )
        logger.info(f"[entrypoint] LLM API OK: {resp.choices[0].message.content}")
        await oai_client.close()
    except Exception as e:
        logger.error(f"[entrypoint] LLM API test FAILED: {type(e).__name__}: {e}")

    # 从 backend 获取会话参数
    session_params = None
    try:
        session_params = await _fetch_session_params(room_id)
    except Exception as e:
        logger.warning(f"[entrypoint] Session params fetch error: {e}")

    # 优先级: session params > env var > default
    if session_params and session_params.get("system_prompt"):
        effective_system_prompt = session_params["system_prompt"]
    else:
        effective_system_prompt = os.environ.get("SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT)

    # 存储 chat_history 在首次 on_user_turn_completed 时注入
    _pending_chat_history = session_params.get("chat_history") if session_params else None

    # 创建 Agent 实例
    agent = VoiceAssistant(singing_handler=singing_handler, instructions=effective_system_prompt)
    agent._metrics = metrics  # 注入 metrics 收集器
    agent._room_id = room_id  # 注入 room_id（用于 llm_node 中查找思考模式）
    agent._user_id = user_id  # 注入 user_id
    agent._pending_chat_history = _pending_chat_history

    # 存储音色路径（从 session params 或环境变量获取，供 sing_a_song 直接使用）
    ref_audio_path = None
    if session_params:
        ref_audio_path = session_params.get("ref_audio_path")
    if not ref_audio_path:
        ref_audio_path = os.environ.get("QWEN3_TTS_REF_AUDIO", "")
    agent._ref_audio_path = ref_audio_path
    logger.info(f"[entrypoint] ref_audio_path for room={room_id}: {ref_audio_path!r}")

    # 开始 per-request 时序追踪（必须在 session.start() 之前，确保回调能访问到 active request）
    request_id = metrics.request_start(room_id, user_id)
    agent._request_id = request_id
    _this_request_id = request_id
    logger.info(f"[entrypoint] Request {request_id} started for room={room_id}")

    # 创建 AgentSession
    session = AgentSession(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts_adapter,
    )

    # 从 session_params 注入参考音色（Qwen3-TTS-Base 直接克隆，流式输出）
    if session_params:
        ref_audio_path = session_params.get("ref_audio_path")
        ref_text = session_params.get("ref_text")
        if ref_audio_path and hasattr(tts_adapter, 'set_ref_audio'):
            tts_adapter.set_ref_audio(ref_audio_path, ref_text)
            logger.info(f"[entrypoint] TTS ref_audio injected: path={ref_audio_path}, ref_text={ref_text}")

    # 注册 session 事件回调（用于监控指标）
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev):
        """用户语音被识别为文字后记录指标（ev 是 UserInputTranscribedEvent）

        per-turn split 检测：如果当前 trace 已进展到 LLM 或更后阶段，
        则说明用户在上一轮回复结束前又说了新的话（打断场景），此时创建新 trace。
        """
        if ev.is_final:
            request_id = metrics._current_request_id.get()
            trace = metrics._request_traces.get(request_id) if request_id else None
            # 如果当前 trace 的 LLM 已开始，说明这是新一轮语音输入
            if trace and trace.llm_start is not None:
                # 前一轮尚未结束，但用户又说了新的话——创建新 trace
                new_request_id = metrics.request_start(room_id, user_id)
                metrics._current_request_id.set(new_request_id)
                agent._request_id = new_request_id
                logger.info(f"[user_input_transcribed] Per-turn split: llm_start={trace.llm_start}, creating new trace {new_request_id}")
            metrics.stt_final(ev.transcript)
        else:
            metrics.stt_interim(ev.transcript)

    # 注册 user_state_changed 事件（VAD 触发检测）
    @session.on("user_state_changed")
    def on_user_state_changed(ev):
        """用户状态变化（VAD 检测到说话/停说话）"""
        if str(ev.new_state) == "UserState.speaking":
            metrics.vad_triggered(is_speech=True)

    # 注册 usage_updated 事件，获取 SDK 内置延迟指标
    @session.on("session_usage_updated")
    def on_session_usage_updated(ev):
        """接收 SDK 内置的延迟指标（ev 是 SessionUsageUpdatedEvent）"""
        logger.debug(f"[session_usage_updated] {ev}")

    # 注册数据通道处理（处理前端发送的思考模式切换命令）
    def handle_data_packet(packet: rtc.DataPacket) -> None:
        """处理来自前端的数据通道消息"""
        try:
            payload = packet.data.decode("utf-8")
            data = json.loads(payload)
            if data.get("type") == "set_thinking_mode":
                enabled = data.get("enabled", False)
                set_thinking_mode(room_id, user_id, enabled)
                logger.info(f"[data_channel] set_thinking_mode: {enabled} for room={room_id}, user={user_id}")
        except Exception as e:
            logger.debug(f"[data_channel] Failed to parse data packet: {e}")

    ctx.room.on("data_received", handle_data_packet)

    # 注册 Agent 状态变化事件（用于歌曲排队播放）
    # agent_state_changed: "initializing" | "idle" | "listening" | "thinking" | "speaking"
    @session.on("agent_state_changed")
    def on_agent_state_changed(ev):
        """Agent 状态变为 speaking 时标记 TTS 忙碌，变为 idle/listening 时标记空闲"""
        logger.info(f"[tts_state] agent_state_changed: {ev.old_state} -> {ev.new_state}")
        if ev.new_state == "speaking":
            agent._tts_idle_event.clear()
        elif ev.new_state in ("idle", "listening"):
            agent._tts_idle_event.set()

    # 注册 session 关闭回调
    @session.on("close")
    def on_session_close():
        """session 关闭时清理思考模式状态"""
        clear_thinking_mode(room_id, user_id)
        logger.info(f"[entrypoint] Session closed, cleaned up thinking mode for room={room_id}, user={user_id}")

    await session.start(
        room=ctx.room,
        agent=agent,
    )

    logger.info(f"[entrypoint] Agent ready in room: {ctx.room.name}")

    # 保持 entrypoint 存活，持续监听用户语音
    # AgentSession 由 VAD 驱动，自动处理后续用户语音
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        logger.info(f"[entrypoint] Job cancelled, room: {ctx.room.name}")
    finally:
        metrics.request_end(_this_request_id)
        logger.info(f"[entrypoint] Request {_this_request_id} ended")


if __name__ == "__main__":
    # 启动 metrics HTTP 服务（在主进程中，提前启动确保一定可用）
    try:
        import threading
        import uvicorn
        from monitoring.metrics import create_app

        metrics_app, _ = create_app(port=8082, collector=_get_metrics())

        def run_metrics():
            uvicorn.run(metrics_app, host="0.0.0.0", port=8082, log_level="warning")

        metrics_thread = threading.Thread(target=run_metrics, daemon=True, name="metrics-server")
        metrics_thread.start()
        print(f"[main] Metrics server started on :8082, alive={metrics_thread.is_alive()}", flush=True)
    except Exception as e:
        print(f"[main] Failed to start metrics server: {e}", flush=True)

    # 启动 Agent Worker
    # 使用 THREAD 模式而非 PROCESS 模式，确保所有 job 共享同一个 Prometheus REGISTRY
    # （PROCESS 模式下每个子进程有独立 REGISTRY，主进程的 /metrics 看不到子进程指标）
    # 注意：THREAD 模式下 GIL 可能影响 CPU 密集型任务，但对 I/O 密集的语音 agent 影响可接受
    cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="voice-agent",
            num_idle_processes=int(os.environ.get("LIVEKIT_NUM_IDLE_PROCESSES", "1")),
            job_executor_type=agents.JobExecutorType.THREAD,
        )
    )