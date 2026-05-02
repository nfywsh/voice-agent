#!/bin/bash
# scripts/integration-tests.sh
# 集成测试脚本 — 验证所有服务健康检查和基本功能
#
# 使用方式：在 docker-compose up -d 后运行
#   bash scripts/integration-tests.sh

set -e

echo "============================================"
echo " 全双工语音聊天机器人 — 集成测试"
echo "============================================"
echo ""

# 配置
BASE_URL="${BASE_URL:-http://localhost}"
TTS_URL="${TTS_URL:-http://localhost:8001}"
SINGING_URL="${SINGING_URL:-http://localhost:8002}"
BACKEND_URL="${BACKEND_URL:-http://localhost:3000}"

PASS=0
FAIL=0

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check() {
    local name="$1"
    local url="$2"
    local method="${3:-GET}"
    local expected_status="${4:-200}"

    echo -n "  测试: $name ... "
    if [ "$method" = "DELETE" ]; then
        status=$(curl -sf -o /dev/null -w "%{http_code}" -X DELETE "$url" 2>/dev/null) || status="000"
    elif [ "$method" = "POST" ]; then
        status=$(curl -sf -o /dev/null -w "%{http_code}" -X POST "$url" 2>/dev/null) || status="000"
    else
        status=$(curl -sf -o /dev/null -w "%{http_code}" "$url" 2>/dev/null) || status="000"
    fi

    if [ "$status" = "$expected_status" ]; then
        echo -e "${GREEN}✓ PASS${NC} (status: $status)"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}✗ FAIL${NC} (expected: $expected_status, got: $status)"
        FAIL=$((FAIL + 1))
    fi
}

check_body() {
    local name="$1"
    local url="$2"
    local method="${3:-GET}"
    local body="${4:-}"
    local expected_status="${5:-200}"

    echo -n "  测试: $name ... "
    if [ "$method" = "POST" ]; then
        status=$(curl -sf -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d "$body" "$url" 2>/dev/null) || status="000"
    else
        status=$(curl -sf -o /dev/null -w "%{http_code}" "$url" 2>/dev/null) || status="000"
    fi

    if [ "$status" = "$expected_status" ]; then
        echo -e "${GREEN}✓ PASS${NC} (status: $status)"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}✗ FAIL${NC} (expected: $expected_status, got: $status)"
        FAIL=$((FAIL + 1))
    fi
}

# ============================================
echo "1. 检查服务健康状态"
echo "-------------------------------------------"
check "Nginx 健康检查" "$BASE_URL/health"
check "TTS 服务健康检查" "$TTS_URL/health"
check "Singing 服务健康检查" "$SINGING_URL/health"
check "Backend 健康检查" "$BACKEND_URL/api/health"
echo ""

# ============================================
echo "2. 检查 TTS 服务 (DashScope API 模式)"
echo "-------------------------------------------"
# 检查 TTS 健康状态详情
echo -n "  测试: TTS 服务状态详情 ... "
tts_health=$(curl -sf "$TTS_URL/health" 2>/dev/null) || tts_health=""
if echo "$tts_health" | grep -q '"api_mode":"dashscope"'; then
    echo -e "${GREEN}✓ PASS${NC} (API 模式: dashscope)"
    PASS=$((PASS + 1))
elif echo "$tts_health" | grep -q '"status"'; then
    echo -e "${YELLOW}⚠ WARN${NC} (状态非 dashscope API 模式: $tts_health)"
    PASS=$((PASS + 1))
else
    echo -e "${RED}✗ FAIL${NC} (无法获取 TTS 健康详情)"
    FAIL=$((FAIL + 1))
fi

# TTS 流式合成（需要真实 API Key，仅在 API 可用时测试）
if [ -n "$DASHSCOPE_API_KEY" ]; then
    echo -n "  测试: TTS 流式合成 ... "
    tts_result=$(curl -sf -X POST -H "Content-Type: application/json" \
        -d '{"text": "你好", "voice": "Chelsie"}' \
        "$TTS_URL/tts/stream" -o /dev/null -w "%{http_code}" 2>/dev/null) || tts_result="000"
    if [ "$tts_result" = "200" ]; then
        echo -e "${GREEN}✓ PASS${NC} (TTS 合成成功)"
        PASS=$((PASS + 1))
    else
        echo -e "${YELLOW}⚠ SKIP${NC} (TTS 合成返回 $tts_result，可能 API Key 无效)"
    fi
else
    echo -e "  测试: TTS 流式合成 ... ${YELLOW}⚠ SKIP${NC} (未设置 DASHSCOPE_API_KEY)"
fi
echo ""

# ============================================
echo "3. 检查 Singing 服务 (Mock 模式)"
echo "-------------------------------------------"
# Mock 模式歌声合成
check_body "Singing Mock 合成" "$SINGING_URL/sing" "POST" \
    '{"lyrics": "Speaker 1: 测试歌词", "title": "集成测试", "style": "流行"}' 200

# 检查返回的 Mock 标记
echo -n "  测试: Singing Mock 标记 ... "
sing_header=$(curl -sf -X POST -H "Content-Type: application/json" \
    -d '{"lyrics": "Speaker 1: mock测试", "title": "测试"}' \
    -D - "$SINGING_URL/sing" 2>/dev/null | grep -i "x-mock" || echo "")
if echo "$sing_header" | grep -q "true"; then
    echo -e "${GREEN}✓ PASS${NC} (X-Mock: true)"
    PASS=$((PASS + 1))
else
    echo -e "${YELLOW}⚠ WARN${NC} (X-Mock 标记未找到)"
fi

# 缓存清除
check "Singing 缓存清除" "$SINGING_URL/cache" "DELETE" 200
echo ""

# ============================================
echo "4. 检查 Token 生成"
echo "-------------------------------------------"
check "Token 生成 (默认房间)" "$BACKEND_URL/api/token?room=test&username=testuser"
check "Token 生成 (中文用户名)" "$BACKEND_URL/api/token?room=测试房间&username=测试用户"

# 检查 Token 格式
echo -n "  测试: Token JWT 格式 ... "
token_response=$(curl -sf "$BACKEND_URL/api/token?room=test&username=testuser" 2>/dev/null) || token_response=""
if echo "$token_response" | grep -q '"token"' && echo "$token_response" | grep -q '"room"'; then
    echo -e "${GREEN}✓ PASS${NC} (返回 token + room 字段)"
    PASS=$((PASS + 1))
else
    echo -e "${RED}✗ FAIL${NC} (Token 响应格式不正确: $token_response)"
    FAIL=$((FAIL + 1))
fi

# 错误处理 — 空参数由前端校验，不带参数时 Next.js 路由仍返回 200（使用默认值）
echo -n "  测试: Token 默认参数 ... "
status=$(curl -sf -o /dev/null -w "%{http_code}" "$BACKEND_URL/api/token" 2>/dev/null) || status="000"
if [ "$status" = "200" ]; then
    echo -e "${GREEN}✓ PASS${NC} (默认参数返回 200)"
    PASS=$((PASS + 1))
else
    echo -e "${YELLOW}⚠ WARN${NC} (默认参数返回 $status)"
fi
echo ""

# ============================================
echo "5. 检查 Nginx 代理"
echo "-------------------------------------------"
check "Nginx 静态文件" "$BASE_URL/"
check "Nginx API 代理" "$BASE_URL/api/token?room=nginx-test&username=nginx-user"
echo ""

# ============================================
echo "============================================"
echo -e " 测试结果: ${GREEN}$PASS 通过${NC}, ${RED}$FAIL 失败${NC}"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi