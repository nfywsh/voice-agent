#!/bin/bash
# scripts/start-all.sh
# 启动所有服务 — 本地开发模式
# [已完成]
#
# 使用方式：bash scripts/start-all.sh

set -e

echo "============================================"
echo " 全双工语音聊天机器人 — 启动所有服务"
echo "============================================"
echo ""

# 检查 .env 文件
if [ ! -f .env ]; then
    echo "⚠️  未找到 .env 文件，从 .env.example 复制..."
    cp .env.example .env
    echo "✓ 已创建 .env，请编辑配置后重新运行"
    echo "  特别是 DASHSCOPE_API_KEY 必须设置！"
    exit 1
fi

# 检查 DASHSCOPE_API_KEY
if grep -q "DASHSCOPE_API_KEY=your_" .env 2>/dev/null || grep -q "DASHSCOPE_API_KEY=$" .env 2>/dev/null; then
    echo "❌ DASHSCOPE_API_KEY 未配置！请编辑 .env 设置你的 API Key"
    echo "   获取 API Key: https://dashscope.console.aliyun.com/"
    exit 1
fi

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

if ! docker compose version &> /dev/null && ! docker-compose version &> /dev/null; then
    echo "❌ Docker Compose 未安装"
    exit 1
fi

# 选择 docker compose 命令
if docker compose version &> /dev/null; then
    COMPOSE="docker compose"
else
    COMPOSE="docker-compose"
fi

# 检查前端构建产物
if [ ! -d frontend/dist ]; then
    echo "📦 前端未构建，正在构建..."
    (cd frontend && npm config set registry https://registry.npmmirror.com && npm install && npm run build)
    if [ $? -ne 0 ]; then
        echo "❌ 前端构建失败，请手动构建："
        echo "   cd frontend && npm install && npm run build"
        exit 1
    fi
    echo "✓ 前端构建完成"
else
    echo "✓ 前端构建产物已存在 (frontend/dist/)"
fi

echo ""
echo "🔧 构建所有 Docker 镜像..."
$COMPOSE build

echo ""
echo "🚀 启动所有服务..."
$COMPOSE up -d

echo ""
echo "⏳ 等待服务就绪..."

# 等待服务健康的函数
wait_for_service() {
    local service=$1
    local url=$2
    local max_attempts=${3:-30}
    local attempt=1

    echo -n "  等待 $service..."
    while [ $attempt -le $max_attempts ]; do
        if wget -qO- "$url" > /dev/null 2>&1; then
            echo " ✓"
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
    done
    echo " ✗ (超时)"
    return 1
}

# 按依赖顺序检查服务就绪
wait_for_service "LiveKit" "http://localhost:7880/" 20
wait_for_service "Backend" "http://localhost:3000/api/health" 20
wait_for_service "TTS Service" "http://localhost:8001/health" 15
wait_for_service "Singing Service" "http://localhost:8002/health" 15
wait_for_service "Nginx" "http://localhost:80/" 10

echo ""
echo "📊 服务状态:"
$COMPOSE ps

echo ""
echo "============================================"
echo " 🎉 服务启动完成！"
echo ""
echo " 前端页面:  http://localhost"
echo " LiveKit:   ws://localhost:7880"
echo " TTS API:   http://localhost:8001/health"
echo " 歌声 API:  http://localhost:8002/health"
echo " Token API: http://localhost:3000/api/health"
echo ""
echo " 查看日志:  bash scripts/logs.sh"
echo " 查看日志:  bash scripts/logs.sh agent"
echo " 停止服务:  bash scripts/stop-all.sh"
echo " 运行测试:  bash scripts/integration-tests.sh"
echo "============================================"