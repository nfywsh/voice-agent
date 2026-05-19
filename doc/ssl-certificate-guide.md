# SSL 证书配置指南

本文档说明如何为 voice-agent nginx 配置 HTTPS 证书。

## 目录结构

```
voice-agent/
├── nginx/
│   ├── nginx.conf          # Nginx 配置 (包含 HTTP 80 和 HTTPS 40082)
│   └── ssl/                # SSL 证书目录
│       ├── futurechat.dns.army.key
│       └── futurechat.dns.army.pem
├── scripts/
│   └── renew-ssl-cert.sh   # 证书自动申请脚本
└── docker-compose.yml
```

## 证书申请脚本

### 使用方式

```bash
# 完整命令
./scripts/renew-ssl-cert.sh <域名> <DynV6_Token>

# 示例：申请 futurechat.dns.army 证书
./scripts/renew-ssl-cert.sh futurechat.dns.army 3uZTSwHwJRTWTxmaYfMxgSrN4DP4JB
```

### 脚本功能

1. 通过 DynV6 DNS API 自动验证域名所有权
2. 向 Let's Encrypt / ZeroSSL 申请 SSL 证书
3. 自动安装证书到 `nginx/ssl/` 目录
4. 重启 nginx 容器使配置生效
5. 验证 HTTPS 连接

### 先决条件

- acme.sh 已安装 (`~/.acme.sh/acme.sh`)
- 域名已托管在 DynV6 并配置正确的 DNS Token
- DynV6 Token 需有该域名的管理权限

## 手动操作步骤

### 1. 申请新证书

```bash
# 设置 DynV6 Token
export DYNV6_TOKEN="your-dynv6-token"

# 申请证书 (使用 Zerossl CA)
~/.acme.sh/acme.sh --issue \
    --dns dns_dynv6 \
    -d your-domain.dns.army \
    --dnssleep 120 \
    --zerossl
```

### 2. 安装证书

```bash
DOMAIN="your-domain.dns.army"
SSL_DIR="/data/script/voice-agent/nginx/ssl"

mkdir -p "$SSL_DIR"

~/.acme.sh/acme.sh --install-cert -d "$DOMAIN" \
    --key-file "$SSL_DIR/${DOMAIN}.key" \
    --fullchain-file "$SSL_DIR/${DOMAIN}.pem" \
    --reloadcmd "echo 'Certificate renewed'"
```

### 3. 修改 nginx 配置

编辑 `nginx/nginx.conf`，更新证书路径：

```nginx
ssl_certificate /etc/nginx/ssl/your-domain.dns.army.pem;
ssl_certificate_key /etc/nginx/ssl/your-domain.dns.army.key;
```

### 4. 重启 nginx

```bash
cd /data/script/voice-agent
docker compose restart nginx
```

### 5. 验证

```bash
# 检查端口
docker port voice-agent-nginx-1

# 测试 HTTPS
curl -k https://localhost:40082/health
```

## 更换域名步骤

如果需要使用新域名：

1. **在新域名管理面板添加 DynV6 DNS**
   - 创建 TXT 记录 `_acme-challenge.your-new-domain.dns.army`
   - 指向 DynV6 提供的验证值

2. **申请新证书**
   ```bash
   ./scripts/renew-ssl-cert.sh your-new-domain.dns.army your-dynv6-token
   ```

3. **更新 nginx 配置**
   ```bash
   # 修改 nginx/nginx.conf 中的证书路径
   ssl_certificate /etc/nginx/ssl/your-new-domain.dns.army.pem;
   ssl_certificate_key /etc/nginx/ssl/your-new-domain.dns.army.key;
   ```

4. **重启 nginx**
   ```bash
   docker compose restart nginx
   ```

## 证书自动续期

acme.sh 会自动处理证书续期（通常在到期前 30 天）。

续期后的 reloadcmd 配置为 `echo 'Certificate renewed'`，如需更完整的处理（如重启 nginx），可修改为：

```bash
--reloadcmd "docker exec voice-agent-nginx-1 nginx -s reload"
```

## 常见问题

### Q: 证书申请失败 "The TXT record has been added but DNS propagation check failed"

**原因**: DNS 记录尚未传播到所有 DNS 服务器。

**解决**:
1. 增加等待时间: `--dnssleep 180`
2. 或禁用 DNS 检查: `--dnsleep 0` 然后手动验证 DNS

### Q: curl 返回 400 Bad Request

**原因**: 证书路径配置错误或证书文件权限问题。

**解决**:
```bash
# 检查证书文件权限
docker exec voice-agent-nginx-1 ls -la /etc/nginx/ssl/

# 测试 nginx 配置
docker exec voice-agent-nginx-1 nginx -t
```

### Q: 想使用其他 DNS 服务商

acme.sh 支持多种 DNS API，修改 `--dns dns_xxx` 参数即可：

| DNS 服务商 | 参数 | 环境变量 |
|-----------|------|---------|
| DynV6 | `--dns dns_dynv6` | `DYNV6_TOKEN` |
| CloudFlare | `--dns dns_cf` | `CF_Token` |
|阿里云 | `--dns dns_ali` | `Ali_Key` / `Ali_Secret` |

详见: https://github.com/acmesh-official/acme.sh/wiki/dnsapi