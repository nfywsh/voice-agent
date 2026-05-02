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
- DASHSCOPE_TTS_MODEL: TTS 模型（默认 qwen3-tts-vd-2026-01-26）
- LLM_MODEL: LLM 模型（默认 Qwen3.5-122B-W8A8）

打断机制：
- VAD 检测到用户说话 → Agent 自动 interrupt() → 取消当前 LLM 推理
- 等用户新输入完成 → 合并上下文 → 发起新 LLM 请求

System Prompt 注入：
- 环境变量 SYSTEM_PROMPT（简单 demo）
- HTTP 接口 PROMPT_SERVICE_URL（生产环境，支持热更新）
"""

import asyncio
import logging
import os

import aiohttp
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    cli,
    function_tool,
)
from livekit.plugins import silero

from dashscope_stt import DashScopeSTT
from singing_handler import SingingHandler
from tts_adapter import QwenTTSAdapter

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

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
        return f"关于「{query}」的搜索结果：暂无相关信息。"

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
    model = os.environ.get("DASHSCOPE_ASR_MODEL", "fun-asr-2025-11-07")
    language = os.environ.get("DASHSCOPE_ASR_LANGUAGE", "zh")
    logger.info(f"Using DashScope ASR: model={model}, language={language}")
    return DashScopeSTT(api_key=api_key, model=model, language=language)


# ============================================================
# 入口函数
# ============================================================

async def entrypoint(ctx: JobContext):
    """Agent 会话入口"""
    logger.info(f"Agent starting, room: {ctx.room.name}")

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
        timeout=float(os.environ.get("TTS_TIMEOUT", "10")),
    )

    # 创建 VAD
    vad = silero.VAD.load(
        activation_threshold=float(os.environ.get("VAD_THRESHOLD", "0.5")),
        min_speech_duration=float(os.environ.get("VAD_MIN_SPEECH", "0.2")),
        min_silence_duration=float(os.environ.get("VAD_MIN_SILENCE", "0.3")),
    )

    # 创建 STT
    stt = _create_stt()

    # 创建 LLM 适配器（使用 DashScope OpenAI 兼容 API）
    from livekit.plugins import openai as lk_openai

    dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    dashscope_base_url = os.environ.get(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    llm_model = os.environ.get("LLM_MODEL", "Qwen3.5-122B-W8A8")

    if not dashscope_api_key:
        raise ValueError(
            "DASHSCOPE_API_KEY is required for LLM. "
            "Set it in .env or docker-compose.yml"
        )

    llm = lk_openai.LLM(
        api_key=dashscope_api_key,
        base_url=dashscope_base_url,
        model=llm_model,
    )

    # 创建 Agent 实例
    agent = VoiceAssistant(singing_handler=singing_handler)

    # 创建会话
    # 注意：AgentSession 是 livekit-agents SDK 的核心类，
    # 它串联 STT → LLM → TTS 的流水线，并处理 VAD 打断
    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts_adapter,
        vad=vad,
        # turn_detector 由 VAD 驱动，无需额外配置
    )

    # 启动会话
    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=True,  # 开启噪声消除，减少回声误触发
        ),
    )

    # 生成开场白
    await session.generate_reply(
        instructions="向用户打招呼，简短介绍自己是一个会唱歌的语音助手。"
    )

    logger.info(f"Agent joined room: {ctx.room.name}")


if __name__ == "__main__":
    # 启动 Agent Worker
    cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="voice-assistant",
        )
    )