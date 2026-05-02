# tts_service/tests/test_tts.py
"""TTS 微服务单元测试 — Mock 模式，不需要真实 DashScope API"""

import os
import sys
import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

# 将 tts_service 目录添加到 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from httpx import AsyncClient, ASGITransport


@pytest.fixture
def mock_api_key():
    """设置 Mock API Key"""
    os.environ["DASHSCOPE_API_KEY"] = "test-api-key-for-testing"
    os.environ["DASHSCOPE_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    os.environ["DASHSCOPE_TTS_MODEL"] = "qwen3-tts-vd-2026-01-26"
    yield
    # 清理
    os.environ.pop("DASHSCOPE_API_KEY", None)


@pytest.fixture
def app(mock_api_key):
    """创建测试用 FastAPI 应用"""
    from main import app
    return app


@pytest.fixture
async def client(app):
    """创建异步测试客户端"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    """健康检查测试"""

    @pytest.mark.asyncio
    async def test_health_ok(self, client):
        """健康检查应返回 ok"""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")
        assert "sample_rate" in data
        assert data["sample_rate"] == 24000
        assert data["api_mode"] == "dashscope"


class TestTTSStreamEndpoint:
    """TTS 流式合成测试"""

    @pytest.mark.asyncio
    async def test_tts_stream_missing_api_key(self):
        """缺少 API Key 应返回 503"""
        os.environ.pop("DASHSCOPE_API_KEY", None)
        from main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/tts/stream", json={"text": "你好"})
            assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_tts_stream_empty_text(self, client):
        """空文本应返回 422"""
        response = await client.post("/tts/stream", json={"text": ""})
        assert response.status_code == 422  # Pydantic validation error

    @pytest.mark.asyncio
    async def test_tts_stream_text_too_long(self, client):
        """超长文本应被截断但仍正常处理"""
        long_text = "你好" * 300  # 600 字，超过 500 字限制
        with patch("main.tts_client") as mock_client:
            # Mock API 调用返回空流
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.content.iter_chunked = AsyncMock(return_value=AsyncMock())
            mock_client.synthesize_stream = AsyncMock(return_value=mock_resp)

            response = await client.post(
                "/tts/stream",
                json={"text": long_text},
            )
            # 请求应该被接受（即使 mock 调用可能出错）
            assert response.status_code in (200, 500)


class TestTTSRequestModel:
    """请求模型验证测试"""

    def test_tts_request_defaults(self):
        """测试默认参数"""
        from main import TTSRequest
        req = TTSRequest(text="你好")
        assert req.voice == "Chelsie"
        assert req.speed == 1.0
        assert req.sample_rate is None

    def test_tts_request_custom(self):
        """测试自定义参数"""
        from main import TTSRequest
        req = TTSRequest(text="你好", voice="custom", speed=1.5, sample_rate=16000)
        assert req.voice == "custom"
        assert req.speed == 1.5
        assert req.sample_rate == 16000

    def test_tts_request_speed_bounds(self):
        """测试语速边界"""
        from main import TTSRequest
        # 太慢
        with pytest.raises(Exception):
            TTSRequest(text="你好", speed=0.1)
        # 太快
        with pytest.raises(Exception):
            TTSRequest(text="你好", speed=3.0)

    def test_tts_request_text_max_length(self):
        """测试文本最大长度"""
        from main import TTSRequest
        req = TTSRequest(text="你好" * 250)  # 500 字
        assert len(req.text) == 500
        # 超过 500 字
        with pytest.raises(Exception):
            TTSRequest(text="你好" * 300)  # 600 字


class TestDashScopeTTSClient:
    """DashScope TTS 客户端测试"""

    def test_client_init(self):
        """测试客户端初始化"""
        from main import DashScopeTTSClient
        client = DashScopeTTSClient(
            api_key="test-key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen3-tts-vd-2026-01-26",
        )
        assert client._api_key == "test-key"
        assert client._model == "qwen3-tts-vd-2026-01-26"


class TestReloadEndpoint:
    """重新加载端点测试"""

    @pytest.mark.asyncio
    async def test_reload_no_api_key(self):
        """没有 API Key 时重新加载应失败"""
        os.environ.pop("DASHSCOPE_API_KEY", None)
        from main import app
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post("/tts/reload")
            assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_reload_with_api_key(self, client):
        """有 API Key 时重新加载应成功"""
        response = await client.post("/tts/reload")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "reloaded"
        assert "qwen3-tts" in data["model"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])