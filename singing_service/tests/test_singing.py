# singing_service/tests/test_singing.py
"""Singing 微服务单元测试 — Mock 模式"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from httpx import AsyncClient, ASGITransport


@pytest.fixture
def app():
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
        """健康检查应返回 ok，mock_mode=true"""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["device"] == "mock"
        assert data["sample_rate"] == 24000
        assert data["mock_mode"] is True


class TestSingEndpoint:
    """歌声合成测试"""

    @pytest.mark.asyncio
    async def test_sing_basic(self, client):
        """基本歌声请求应返回音频数据"""
        response = await client.post(
            "/sing",
            json={
                "lyrics": "Speaker 1: 测试歌词",
                "title": "测试歌曲",
                "style": "流行",
            },
        )
        assert response.status_code == 200
        assert response.headers["x-mock"] == "true"
        assert response.headers["x-sample-rate"] == "24000"
        # 应返回音频数据
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_sing_custom_sample_rate(self, client):
        """自定义采样率"""
        response = await client.post(
            "/sing",
            json={
                "lyrics": "Speaker 1: 测试",
                "title": "测试",
                "sample_rate": 16000,
            },
        )
        assert response.status_code == 200
        assert response.headers["x-sample-rate"] == "16000"

    @pytest.mark.asyncio
    async def test_sing_empty_lyrics(self, client):
        """空歌词应返回 422"""
        response = await client.post(
            "/sing",
            json={"lyrics": "", "title": "测试"},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_sing_cache_hit(self, client):
        """第二次请求相同歌词应命中缓存"""
        lyrics = "Speaker 1: 缓存测试歌词"
        # 第一次请求
        r1 = await client.post(
            "/sing",
            json={"lyrics": lyrics, "title": "缓存测试1"},
        )
        assert r1.status_code == 200

        # 第二次请求（应命中缓存）
        r2 = await client.post(
            "/sing",
            json={"lyrics": lyrics, "title": "缓存测试2"},
        )
        assert r2.status_code == 200
        # 两次结果长度应相同
        assert len(r1.content) == len(r2.content)


class TestCacheEndpoint:
    """缓存管理测试"""

    @pytest.mark.asyncio
    async def test_clear_cache(self, client):
        """清除缓存应成功"""
        response = await client.delete("/cache")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cache cleared"


class TestLRUCache:
    """LRU 缓存单元测试"""

    def test_cache_put_and_get(self):
        """测试基本存取"""
        from main import LRUCache
        cache = LRUCache(max_size=3)
        cache.put("歌词1", "Speaker 1", b"audio1")
        assert cache.get("歌词1", "Speaker 1") == b"audio1"

    def test_cache_miss(self):
        """测试缓存未命中"""
        from main import LRUCache
        cache = LRUCache()
        assert cache.get("不存在的歌词", "Speaker 1") is None

    def test_cache_eviction(self):
        """测试 LRU 淘汰"""
        from main import LRUCache
        cache = LRUCache(max_size=2)
        cache.put("歌词1", "Speaker 1", b"audio1")
        cache.put("歌词2", "Speaker 1", b"audio2")
        cache.put("歌词3", "Speaker 1", b"audio3")  # 应淘汰歌词1
        assert cache.get("歌词1", "Speaker 1") is None
        assert cache.get("歌词2", "Speaker 1") == b"audio2"
        assert cache.get("歌词3", "Speaker 1") == b"audio3"

    def test_cache_oversized_audio(self):
        """超大音频不缓存"""
        from main import LRUCache
        cache = LRUCache(max_size=5, max_audio_seconds=1)
        # >1s @ 24kHz 16bit = 48000 bytes
        big_audio = b"\x00" * 50000
        cache.put("大音频", "Speaker 1", big_audio)
        assert cache.get("大音频", "Speaker 1") is None


class TestMockSingingGeneration:
    """Mock 歌声生成测试"""

    def test_generate_mock_singing(self):
        """测试 Mock 歌声生成输出"""
        from main import _generate_mock_singing
        audio = _generate_mock_singing(sr=24000, duration=5.0)
        assert isinstance(audio, bytes)
        assert len(audio) > 0
        # 5s @ 24kHz 16bit = 240000 bytes
        expected_len = 24000 * 5 * 2
        assert len(audio) == expected_len

    def test_generate_mock_singing_custom_duration(self):
        """测试自定义时长"""
        from main import _generate_mock_singing
        audio = _generate_mock_singing(sr=24000, duration=2.0)
        expected_len = 24000 * 2 * 2
        assert len(audio) == expected_len


if __name__ == "__main__":
    pytest.main([__file__, "-v"])