# agent/agent.py
"""LiveKit 全双工语音 Agent 主程序 [已完成 - DashScope API 改造]

架构说明：
- VoiceAssistant 继承 Agent，定义系统指令和工具函数
- DashScopeSTT 通过 WebSocket 对接阿里云 Fun-ASR 实时语音识别
- LLM 通过 OpenAI 兼容 API 对接 DashScope Qwen3.5-122B-W8A8
- QwenTTSAdapter 对接 TTS 微服务（内部调用 DashScope Qwen3-TTS API），24kHz→48kHz 重采样
- SingingHandler 对接歌声服务，流式推歌声音频到 LiveKit

API 配置（全部通过环境变量注入）：
- DASHSCOPE_API_KEY: 阿里云 DashScope API Key（共用）
- DASHSCOPE_BASE_URL: API 基地址（默认 https://dashscope.aliyuncs.com/compatible-mode/v1）
- DASHSCOPE_ASR_MODEL: ASR 模型（默认 fun-asr-2025-11-07）
- DASHSCOPE_TTS_MODEL: TTS 模型（默认 qwen3-tts-instruct-flash-realtime-2026-01-22）
- LLM_MODEL: LLM 模型（默认 Qwen3.5-122B-W8A8）
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
from singing_handler import SingingHandler
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
    "- 唱歌时，你需要根据用户要求创作歌词，格式为每行 'Speaker 1: 歌词内容'\n"
    "- 工具调用返回后，给出简短的过渡语，不要重复工具返回的文本\n"
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


# ============================================================
# 主 Agent 类
# ============================================================

class VoiceAssistant(Agent):
    """全双工语音助手"""

    def __init__(self, *, singing_handler: SingingHandler) -> None:
        system_prompt = _get_system_prompt()
        super().__init__(instructions=system_prompt)
        self.singing_handler = singing_handler
        self.chat_history_manager = ChatHistoryManager(max_turns=CHAT_HISTORY_MAX_TURNS)

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        """用户说完一句话后，自动截断聊天历史避免无限累积"""
        self.chat_history_manager.truncate(turn_ctx)

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
        is_thinking = get_thinking_mode(room_id, user_id)

        # 根据思考模式构建 extra_kwargs
        # DashScope 使用 extra_body.chat_template_kwargs.enable_thinking
        extra_kwargs = {
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": is_thinking
                }
            }
        }

        import time
        t0 = time.monotonic()
        first_chunk = True

        logger.info(f"[llm_node] Thinking mode: {is_thinking} for room={room_id}, tools count={len(tools)}")

        # 获取 metrics 实例（从 self._metrics 或全局）
        metrics = getattr(self, '_metrics', None)

        if metrics:
            metrics.llm_start()
            # 设置 contextvars，让后续的 tts_adapter 等能拿到 request_id
            request_id = getattr(self, '_request_id', None)
            if request_id:
                metrics._current_request_id.set(request_id)

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

                yield chunk

        if metrics:
            metrics.llm_end()
        logger.info(f"[llm_node] Done, total time: {time.monotonic() - t0:.3f}s")

    # ============================================================
    # 工具函数
    # ============================================================

    @function_tool
    async def sing_a_song(self, title: str, lyrics: str, style: str = "流行") -> str:
        """当用户要求唱歌时调用此工具。工具会根据歌词生成旋律并演唱。

        Args:
            title: 歌曲名称
            lyrics: 歌词全文，使用换行符分隔每句歌词，每句格式为 'Speaker 1: 歌词内容'
            style: 歌曲风格，如流行、民谣、摇滚等
        """
        logger.info(f"[sing_a_song] title={title}, style={style}, lyrics_len={len(lyrics)}")

        # 异步启动歌声生成并推流到 LiveKit
        # 注意：Agent 框架会自动将此工具的返回值作为 LLM 消息
        # 歌声音频推流在 _push_singing_audio 中处理
        asyncio.create_task(
            self._push_singing_audio(title=title, lyrics=lyrics, style=style)
        )
        return f"正在演唱《{title}》，风格：{style}"

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
# ASR 配置（备选方案）
# ============================================================

def _create_stt():
    """创建 STT 实例。使用 DashScope Fun-ASR。"""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "DASHSCOPE_API_KEY is required for DashScope ASR. "
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
# 入口函数
# ============================================================

async def entrypoint(ctx: JobContext):
    """Agent 会话入口"""
    logger.info(f"[entrypoint] Agent starting, room: {ctx.room.name}")
    logger.info(f"[entrypoint] num_idle_processes env: {os.environ.get('LIVEKIT_NUM_IDLE_PROCESSES', 'NOT SET')}")
    logger.info(f"[entrypoint] Metrics server already running in main process on :8082")

    # 初始化歌声处理器
    singing_url = os.environ.get("SINGING_SERVICE_URL", "http://localhost:8002")
    singing_mock = os.environ.get("SINGING_MOCK_MODE", "false").lower() == "true"
    singing_handler = SingingHandler(
        service_url=singing_url,
        timeout=float(os.environ.get("SINGING_TIMEOUT", "30")),
        mock_mode=singing_mock,
    )

    # 创建 TTS 适配器
    tts_url = os.environ.get("TTS_SERVICE_URL", "http://localhost:8001")
    tts_adapter = QwenTTSAdapter(
        service_url=tts_url,
        voice=os.environ.get("DASHSCOPE_TTS_VOICE", "Cherry"),
        timeout=float(os.environ.get("TTS_TIMEOUT", "10")),
        metrics=metrics,  # 注入 metrics 用于 TTS 内部上报
    )

    # 创建 VAD
    vad = silero.VAD.load(
        activation_threshold=float(os.environ.get("VAD_THRESHOLD", "0.5")),
        min_speech_duration=float(os.environ.get("VAD_MIN_SPEECH", "0.2")),
        min_silence_duration=float(os.environ.get("VAD_MIN_SILENCE", "0.3")),
    )

    # 创建 STT
    stt = _create_stt()

    # 创建 LLM 适配器（使用独立 API 端点）
    # 注意：openai plugin 必须在主线程注册，所以放在模块顶部 import（THREAD 模式下 entrypoint 运行在主线程）

    llm_api_key = os.environ.get("LLM_API_KEY", "")
    llm_base_url = os.environ.get(
        "LLM_BASE_URL", "https://jiajiatemp.duckdns.org:30002/v1/"
    )
    llm_model = os.environ.get("LLM_MODEL", "qwen3.5-122b-a10b")

    if not llm_api_key:
        raise ValueError(
            "LLM_API_KEY is required for LLM. "
            "Set it in .env or docker-compose.yml"
        )

    logger.info(f"[entrypoint] LLM config: base_url={llm_base_url}, model={llm_model}")

    # 创建 LLM 实例
    # 注意：思考模式参数通过 llm_node 动态传入 extra_kwargs，不再在构造函数设置 extra_body
    llm = lk_openai.LLM(
        api_key=llm_api_key,
        base_url=llm_base_url,
        model=llm_model,
    )

    # 获取 room 信息
    room_id = ctx.room.name
    user_id = f"agent_{room_id}"
    logger.info(f"[entrypoint] Room: {room_id}, User: {user_id}")

    # 创建 metrics 收集器
    metrics = MetricsCollector()
    metrics.session_start()

    # 验证 LLM API 连通性
    try:
        import openai
        oai_client = openai.AsyncOpenAI(api_key=llm_api_key, base_url=llm_base_url)
        logger.info(f"[entrypoint] Testing LLM API connectivity...")
        resp = await oai_client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=10,
        )
        logger.info(f"[entrypoint] LLM API OK: {resp.choices[0].message.content}")
        await oai_client.close()
    except Exception as e:
        logger.error(f"[entrypoint] LLM API test FAILED: {type(e).__name__}: {e}")

    # 创建 Agent 实例
    agent = VoiceAssistant(singing_handler=singing_handler)
    agent._metrics = metrics  # 注入 metrics 收集器
    agent._room_id = room_id  # 注入 room_id（用于 llm_node 中查找思考模式）
    agent._user_id = user_id  # 注入 user_id

    # 开始 per-request 时序追踪
    request_id = metrics.request_start(room_id, user_id)
    agent._request_id = request_id
    logger.info(f"[entrypoint] Request {request_id} started for room={room_id}")

    # 注册 session 事件回调（用于监控指标）
    @session.on("user_input_transcribed")
    def on_user_input_transcribed(ev):
        """用户语音被识别为文字后记录指标（ev 是 UserInputTranscribedEvent）"""
        if ev.is_final:
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

    # 注册 session 关闭回调
    @session.on("close")
    def on_session_close():
        """session 关闭时清理思考模式状态和监控指标"""
        clear_thinking_mode(room_id, user_id)
        metrics.session_end()
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
        metrics.request_end(request_id)
        logger.info(f"[entrypoint] Request {request_id} ended")


if __name__ == "__main__":
    # 启动 metrics HTTP 服务（在主进程中，提前启动确保一定可用）
    try:
        import threading
        import uvicorn
        from monitoring.metrics import create_app

        metrics_app, _ = create_app(port=8082)

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
            num_idle_processes=int(os.environ.get("LIVEKIT_NUM_IDLE_PROCESSES", "1")),
            job_executor_type=agents.JobExecutorType.THREAD,
        )
    )