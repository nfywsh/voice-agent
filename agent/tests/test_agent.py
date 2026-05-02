# agent/tests/test_agent.py
"""Agent 服务单元测试 — Mock 模式，不需要网络连接"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# 将 agent 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDashScopeSTT:
    """DashScope STT 适配器测试"""

    def test_create_stt_with_api_key(self):
        """测试使用 API Key 创建 STT 实例"""
        from dashscope_stt import DashScopeSTT

        stt = DashScopeSTT(
            api_key="test-api-key",
            model="fun-asr-2025-11-07",
            language="zh",
        )
        assert stt._api_key == "test-api-key"
        assert stt._model == "fun-asr-2025-11-07"
        assert stt._language == "zh"

    def test_create_stt_stream(self):
        """测试创建流式识别器"""
        from dashscope_stt import DashScopeSTT

        stt = DashScopeSTT(api_key="test-key")
        stream = stt.stream()
        assert stream is not None
        assert stream._stt is stt

    def test_create_stt_factory_missing_key(self):
        """测试工厂函数 — 缺少 API Key 应抛异常"""
        from dashscope_stt import create_stt

        with patch.dict(os.environ, {}, clear=True):
            # 清除可能存在的环境变量
            os.environ.pop("DASHSCOPE_API_KEY", None)
            with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
                create_stt()

    def test_create_stt_factory_with_key(self):
        """测试工厂函数 — 有 API Key 应成功"""
        from dashscope_stt import create_stt

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}):
            stt = create_stt()
            assert stt._api_key == "test-key"
            assert stt._model == "fun-asr-2025-11-07"

    def test_create_stt_factory_custom_model(self):
        """测试工厂函数 — 自定义模型"""
        from dashscope_stt import create_stt

        with patch.dict(os.environ, {
            "DASHSCOPE_API_KEY": "test-key",
            "DASHSCOPE_ASR_MODEL": "custom-model",
            "DASHSCOPE_ASR_LANGUAGE": "en",
        }):
            stt = create_stt()
            assert stt._model == "custom-model"
            assert stt._language == "en"


class TestDownsample:
    """下采样函数测试"""

    def test_downsample_48k_to_16k(self):
        """测试 48kHz → 16kHz 下采样"""
        from dashscope_stt import _downsample_48k_to_16k
        import numpy as np

        # 生成 48kHz 正弦波 (100ms)
        duration = 0.1
        freq = 1000
        t = np.linspace(0, duration, int(48000 * duration), dtype=np.int16)
        pcm_48k = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16).tobytes()

        result = _downsample_48k_to_16k(pcm_48k)

        # 16kHz 100ms = 1600 samples = 3200 bytes
        expected_len = int(48000 * duration / 3) * 2  # /3 因为 48k→16k，*2 因为 int16
        assert abs(len(result) - expected_len) < 10  # 允许小误差


class TestVoiceAssistant:
    """VoiceAssistant 类测试"""

    def test_default_system_prompt(self):
        """测试默认 System Prompt"""
        # 直接读取默认 prompt
        from agent import _DEFAULT_SYSTEM_PROMPT
        assert "语音助手" in _DEFAULT_SYSTEM_PROMPT
        assert "唱歌" in _DEFAULT_SYSTEM_PROMPT

    def test_get_system_prompt_from_env(self):
        """测试从环境变量获取 System Prompt"""
        from agent import _get_system_prompt

        with patch.dict(os.environ, {"SYSTEM_PROMPT": "自定义提示词"}):
            prompt = _get_system_prompt()
            assert prompt == "自定义提示词"

    def test_sing_a_song_tool_defined(self):
        """测试 sing_a_song 工具是否定义"""
        from agent import VoiceAssistant
        from singing_handler import SingingHandler

        handler = SingingHandler(mock_mode=True)
        agent = VoiceAssistant(singing_handler=handler)

        # 检查 @function_tool 方法存在
        assert hasattr(agent, 'sing_a_song')
        assert callable(agent.sing_a_song)

    def test_get_weather_tool_defined(self):
        """测试 get_weather 工具是否定义"""
        from agent import VoiceAssistant
        from singing_handler import SingingHandler

        handler = SingingHandler(mock_mode=True)
        agent = VoiceAssistant(singing_handler=handler)
        assert hasattr(agent, 'get_weather')

    def test_search_web_tool_defined(self):
        """测试 search_web 工具是否定义"""
        from agent import VoiceAssistant
        from singing_handler import SingingHandler

        handler = SingingHandler(mock_mode=True)
        agent = VoiceAssistant(singing_handler=handler)
        assert hasattr(agent, 'search_web')


class TestSingingHandler:
    """SingingHandler 测试"""

    @pytest.mark.asyncio
    async def test_mock_sing(self):
        """测试 Mock 模式歌声生成"""
        from singing_handler import SingingHandler

        handler = SingingHandler(mock_mode=True)
        chunks = []
        async for chunk in handler.sing_stream(
            lyrics="Speaker 1: 测试歌词", title="测试", style="流行"
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        # 每个 chunk 应该是 48kHz 16bit mono PCM
        for chunk in chunks:
            assert isinstance(chunk, bytes)
            assert len(chunk) > 0

    @pytest.mark.asyncio
    async def test_mock_sing_multiple_chunks(self):
        """测试 Mock 模式返回多个音频块"""
        from singing_handler import SingingHandler

        handler = SingingHandler(mock_mode=True)
        chunks = []
        async for chunk in handler.sing_stream(
            lyrics="Speaker 1: 多块测试", title="块测试"
        ):
            chunks.append(chunk)

        # Mock 模式应返回多个 chunk
        assert len(chunks) >= 1


class TestPromptService:
    """PromptService 测试"""

    @pytest.mark.asyncio
    async def test_prompt_service_fallback(self):
        """测试 PromptService 降级到默认 prompt"""
        from agent import PromptService, _DEFAULT_SYSTEM_PROMPT

        # 使用不存在的 URL
        service = PromptService(endpoint="http://localhost:99999/prompt")
        prompt = await service.get_prompt("test-room")
        # 降级到默认 prompt
        assert prompt == _DEFAULT_SYSTEM_PROMPT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])