#!/bin/bash
# scripts/logs.sh
# 查看服务日志
# [已完成]
#
# 使用方式：
#   bash scripts/logs.sh          # 查看所有服务日志
#   bash scripts/logs.sh agent    # 查看 agent 服务日志
#   bash scripts/logs.sh tts      # 查看 TTS 服务日志
#   bash scripts/logs.sh singing  # 查看 Singing 服务日志

SERVICE="${1:-}"

# 选择 docker compose 命令
if docker compose version &> /dev/null 2>&1; then
    COMPOSE="docker compose"
else
    COMPOSE="docker-compose"
fi

if [ -z "$SERVICE" ]; then
    $COMPOSE logs -f --tail=100
else
    # 映射简短名称到完整服务名
    case "$SERVICE" in
        agent)      SERVICE_NAME="agent" ;;
        tts)        SERVICE_NAME="tts-service" ;;
        singing)    SERVICE_NAME="singing-service" ;;
        backend)    SERVICE_NAME="backend" ;;
        nginx)      SERVICE_NAME="nginx" ;;
        livekit)    SERVICE_NAME="livekit" ;;
        *)          SERVICE_NAME="$SERVICE" ;;
    esac
    $COMPOSE logs -f --tail=100 "$SERVICE_NAME"
fi