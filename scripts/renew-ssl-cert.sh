#!/bin/bash
#===============================================================================
# SSL 证书自动申请脚本 (使用 DynV6 DNS API)
#
# 使用方式:
#   ./scripts/renew-ssl-cert.sh <域名> <DynV6_Token>
#
# 示例:
#   ./scripts/renew-ssl-cert.sh futurechat.dns.army 3uZTSwHwJRTWTxmaYfMxgSrN4DP4JB
#
# 说明:
#   - 证书存放在项目 nginx/ssl/ 目录
#   - 使用 acme.sh + DynV6 DNS 验证
#   - 证书会自动安装到 nginx/ssl/ 目录
#   - 需要重启 nginx 容器使配置生效: docker compose restart nginx
#===============================================================================

set -e

DOMAIN="${1:-}"
TOKEN="${2:-}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查参数
if [ -z "$DOMAIN" ] || [ -z "$TOKEN" ]; then
    log_error "用法: $0 <域名> <DynV6_Token>"
    echo ""
    echo "示例:"
    echo "  $0 futurechat.dns.army 3uZTSwHwJRTWTxmaYfMxgSrN4DP4JB"
    exit 1
fi

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SSL_DIR="$PROJECT_DIR/nginx/ssl"

log_info "开始申请 SSL 证书..."
log_info "域名: $DOMAIN"
log_info "证书目录: $SSL_DIR"

# 创建证书目录
mkdir -p "$SSL_DIR"

# 设置环境变量
export DYNV6_TOKEN="$TOKEN"

# 申请证书 (使用 zerossl CA)
log_info "正在通过 DynV6 DNS 验证申请证书..."
~/.acme.sh/acme.sh --issue \
    --dns dns_dynv6 \
    -d "$DOMAIN" \
    --dnssleep 120 \
    --zerossl \
    --always-force-domain

if [ $? -eq 0 ]; then
    log_info "证书申请成功!"
else
    log_error "证书申请失败!"
    exit 1
fi

# 安装证书到项目目录
log_info "正在安装证书到 $SSL_DIR ..."

~/.acme.sh/acme.sh --install-cert -d "$DOMAIN" \
    --key-file "$SSL_DIR/${DOMAIN}.key" \
    --fullchain-file "$SSL_DIR/${DOMAIN}.pem" \
    --reloadcmd "echo 'Certificate renewed at $(date)'"

if [ $? -eq 0 ]; then
    log_info "证书安装成功!"
else
    log_error "证书安装失败!"
    exit 1
fi

# 验证证书文件
log_info "验证证书文件..."
if [ -f "$SSL_DIR/${DOMAIN}.key" ] && [ -f "$SSL_DIR/${DOMAIN}.pem" ]; then
    log_info "证书文件已生成:"
    ls -la "$SSL_DIR/${DOMAIN}".*
else
    log_error "证书文件缺失!"
    exit 1
fi

# 清理旧容器 (如果有)
log_info "清理旧的 nginx 容器..."
cd "$PROJECT_DIR"
docker compose rm -f nginx 2>/dev/null || true

# 重启 nginx 服务
log_info "重启 nginx 服务..."
docker compose up -d nginx

# 等待 nginx 启动
sleep 3

# 验证 HTTPS
log_info "验证 HTTPS 连接..."
HTTP_CODE=$(curl -k -s -o /dev/null -w "%{http_code}" https://localhost:40082/health 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
    log_info "HTTPS 服务正常! (HTTP $HTTP_CODE)"
else
    log_warn "HTTPS 服务可能异常 (HTTP $HTTP_CODE), 请手动检查"
fi

echo ""
log_info "=================================="
log_info "SSL 证书更新完成!"
log_info "域名: $DOMAIN"
log_info "证书: $SSL_DIR/${DOMAIN}.pem"
log_info "密钥: $SSL_DIR/${DOMAIN}.key"
log_info "HTTPS 端口: 40082"
log_info "=================================="