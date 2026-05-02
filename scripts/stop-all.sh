#!/bin/bash
# scripts/stop-all.sh
# 停止所有服务
# [已完成]
#
# 使用方式：bash scripts/stop-all.sh

set -e

# 选择 docker compose 命令
if docker compose version &> /dev/null 2>&1; then
    COMPOSE="docker compose"
else
    COMPOSE="docker-compose"
fi

echo "🛑 停止所有服务..."
$COMPOSE down

echo ""
echo "✓ 所有服务已停止"
echo "  重新启动: bash scripts/start-all.sh"