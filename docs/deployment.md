# 全双工语音聊天机器人 - 生产环境部署手册

> 本文档详述如何在生产环境（Ubuntu 20.04/22.04）部署全双工语音聊天机器人系统。
>
> 架构概览：LiveKit SFU + 独立 LLM API（DashScope 兼容）+ DashScope ASR/TTS，所有 AI 能力通过 API 调用，无需本地 GPU。

---

## 1. 生产环境准备

### 1.1 服务器要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 4 核 | 8 核+ |
| 内存 | 8 GB | 16 GB+ |
| 硬盘 | 50 GB SSD | 100 GB+ SSD |
| GPU | **无需**（所有 AI 能力通过 API 调用） | — |
| 操作系统 | Ubuntu 20.04 / 22.04 LTS | Ubuntu 22.04 LTS |
| 网络带宽 | 10 Mbps | 50 Mbps+ |
| 公网 IP | 需要（弹性 IP） | 需要（固定公网 IP） |

**说明**：
- 本系统不依赖本地 GPU，AI 能力通过 API 调用实现
- LLM：独立端点 `https://jiajiatemp.duckdns.org:30002/`（用户自建或第三方）
- ASR/TTS：阿里云 DashScope API（Fun-ASR + Qwen3-TTS）
- 占用资源主要是 LiveKit SFU 的媒体转发，需根据并发用户数评估带宽

### 1.2 域名与 SSL 证书

**域名要求**：
- 需要一个已备案的域名（如 `voice.example.com`）
- 支持二级域名

**SSL 证书获取（选其一）**：

**方案 A：Let's Encrypt 免费证书（推荐）**
```bash
# 安装 certbot
sudo apt update
sudo apt install -y certbot python3-certbot-nginx

# 停止 nginx（如果正在运行）
sudo systemctl stop nginx

# 获取证书（需域名已解析到本机）
sudo certbot certonly --standalone -d voice.example.com --agree-tos --email admin@example.com --non-interactive

# 证书存放位置
# /etc/letsencrypt/live/voice.example.com/fullchain.pem
# /etc/letsencrypt/live/voice.example.com/privkey.pem
```

**方案 B：阿里云/腾讯云免费证书**
- 在云服务商控制台申请免费 SSL 证书
- 下载 Nginx 版本的证书文件
- 上传到服务器的 `/etc/ssl/certs/` 和 `/etc/ssl/private/` 目录

### 1.3 DNS 解析配置

在域名服务商控制台添加以下记录：

| 记录类型 | 主机记录 | 记录值 | TTL |
|---------|---------|-------|-----|
| A | voice | `<服务器公网IP>` | 600 |
| A | `*` | `<服务器公网IP>` | 600 |

**验证 DNS 生效**：
```bash
# 本地验证
nslookup voice.example.com

# 在服务器上验证（确保防火墙开放 80/443 端口）
curl -I https://voice.example.com
```

### 1.4 服务器安全组/防火墙配置

**阿里云/腾讯云安全组入站规则**：

| 协议 | 端口范围 | 用途 |
|-----|---------|------|
| TCP | 80 | HTTP（用于 Let's Encrypt 验证）|
| TCP | 443 | HTTPS |
| TCP | 7880 | LiveKit HTTP |
| TCP | 7881 | LiveKit TCP 转发 |
| UDP | 50000-50050 | LiveKit 媒体传输（RTC）|

**服务器防火墙**：
```bash
# 开放端口
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 7880/tcp
sudo ufw allow 7881/tcp
sudo ufw allow 50000:50050/udp

# 启用防火墙
sudo ufw enable
sudo ufw status
```

---

## 2. 部署步骤

### 2.1 Ubuntu 环境准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y curl wget git vim htop net-tools nmon

# 检查系统版本
cat /etc/os-release
# 确保是 Ubuntu 20.04 或 22.04
```

### 2.2 Docker 安装

```bash
# 安装 Docker（使用阿里云镜像加速）
curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun

# 启动 Docker
sudo systemctl start docker
sudo systemctl enable docker

# 添加当前用户到 docker 组（免 sudo）
sudo usermod -aG docker $USER
newgrp docker

# 验证安装
docker --version
docker-compose --version
```

### 2.3 NVIDIA Docker Toolkit 安装（可选，仅用于未来本地 GPU 支持）

> **注意**：当前版本不需要本地 GPU，本步骤仅记录备用。

```bash
# 添加 NVIDIA 仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 安装 NVIDIA Docker Toolkit
sudo apt update
sudo apt install -y nvidia-docker2

# 重启 Docker
sudo systemctl restart docker

# 验证
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### 2.4 项目文件准备

```bash
# 创建项目目录
sudo mkdir -p /opt/voice-agent
cd /opt/voice-agent

# 克隆项目（如果有私有仓库）
# git clone https://github.com/your-org/voice-agent.git .

# 如果是手动上传，将项目文件复制到该目录
# scp -r ./voice-agent/* root@your-server:/opt/voice-agent/

# 设置权限
sudo chown -R $USER:$USER /opt/voice-agent
cd /opt/voice-agent
```

### 2.5 环境变量配置（生产环境）

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑生产环境配置
vim .env
```

**生产环境 `.env` 关键配置**：

```bash
# ============================================
# 生产环境配置
# ============================================

# ============ DashScope API 配置 ============
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_ASR_MODEL=fun-asr-2025-11-07
DASHSCOPE_TTS_MODEL=qwen3-tts-vd-2026-01-26

# ============ LiveKit 配置（生产环境请修改！）============
LIVEKIT_API_KEY=voice_prod_key
LIVEKIT_API_SECRET=<随机生成的强密码，至少32字符>
# 推荐生成随机密钥：
# openssl rand -hex 32

# ============ LLM 配置（独立 API）============
LLM_BASE_URL=https://jiajiatemp.duckdns.org:30002/
LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
LLM_MODEL=Qwen3.5-122B-W8A8

# ============ VAD 配置 ============
VAD_THRESHOLD=0.5
VAD_MIN_SPEECH=0.2
VAD_MIN_SILENCE=0.3

# ============ 超时配置 ============
LLM_TIMEOUT=5
TTS_TIMEOUT=10
SINGING_TIMEOUT=30

# ============ RTC 端口范围 ============
RTC_PORT_START=50000
RTC_PORT_END=50050
```

**安全建议**：
- 立即修改 `LIVEKIT_API_KEY` 和 `LIVEKIT_API_SECRET`
- 使用 `openssl rand -hex 32` 生成随机密钥
- 不要在代码仓库中存储真实的 API Key

### 2.6 生产环境 docker-compose.prod.yml

在项目根目录创建 `docker-compose.prod.yml`：

```yaml
# docker-compose.prod.yml
# 生产环境配置 - 包含资源限制和日志轮转
version: "3.8"

services:
  # ============ 1. Nginx 反向代理 + HTTPS ============
  nginx:
    image: nginx:alpine
    container_name: voice-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./frontend/dist:/usr/share/nginx/html:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      backend:
        condition: service_healthy
      livekit:
        condition: service_healthy
    networks:
      - voice-network
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:80/health"]
      interval: 15s
      timeout: 3s
      retries: 3
      start_period: 5s
    restart: always
    # 生产环境资源限制
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
    # 日志配置 - 日志轮转
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
        compress: "true"

  # ============ 2. Next.js Token 服务 ============
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: voice-backend
    environment:
      LIVEKIT_API_KEY: ${LIVEKIT_API_KEY}
      LIVEKIT_API_SECRET: ${LIVEKIT_API_SECRET}
    networks:
      - voice-network
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:3000/api/health"]
      interval: 15s
      timeout: 3s
      retries: 3
      start_period: 30s
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 1G
        reservations:
          cpus: '1.0'
          memory: 512M
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"
        compress: "true"

  # ============ 3. LiveKit Server ============
  livekit:
    image: livekit/livekit-server:v1.7
    container_name: voice-livekit
    ports:
      - "7880:7880"
      - "7881:7881"
      - "${RTC_PORT_START:-50000}-${RTC_PORT_END:-50050}:${RTC_PORT_START:-50000}-${RTC_PORT_END:-50050}/udp"
    environment:
      LIVEKIT_KEYS: "${LIVEKIT_API_KEY}: ${LIVEKIT_API_SECRET}"
    command: --config /etc/livekit.yaml
    volumes:
      - ./livekit.yaml:/etc/livekit.yaml:ro
      - ./data/livekit:/data
    networks:
      - voice-network
    healthcheck:
      test: ["CMD", "wget", "-qO-", "http://localhost:7880/"]
      interval: 10s
      timeout: 3s
      retries: 5
      start_period: 10s
    restart: always
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 2G
        reservations:
          cpus: '2.0'
          memory: 1G
    logging:
      driver: "json-file"
      options:
        max-size: "200m"
        max-file: "5"
        compress: "true"

  # ============ 4. TTS 代理服务 ============
  tts-service:
    build:
      context: ./tts_service
      dockerfile: Dockerfile
    container_name: voice-tts
    environment:
      DASHSCOPE_API_KEY: ${DASHSCOPE_API_KEY}
      DASHSCOPE_BASE_URL: ${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}
      DASHSCOPE_TTS_MODEL: ${DASHSCOPE_TTS_MODEL:-qwen3-tts-vd-2026-01-26}
      HOST: "0.0.0.0"
      PORT: "8001"
    networks:
      - voice-network
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 10s
    restart: always
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"

  # ============ 5. 歌声服务 ============
  singing-service:
    build:
      context: ./singing_service
      dockerfile: Dockerfile
    container_name: voice-singing
    environment:
      HOST: "0.0.0.0"
      PORT: "8002"
    networks:
      - voice-network
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 5s
    restart: always
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"

  # ============ 6. LiveKit Agent ============
  agent:
    build:
      context: ./agent
      dockerfile: Dockerfile
    container_name: voice-agent
    environment:
      LIVEKIT_URL: ws://livekit:7880
      LIVEKIT_API_KEY: ${LIVEKIT_API_KEY}
      LIVEKIT_API_SECRET: ${LIVEKIT_API_SECRET}
      DASHSCOPE_API_KEY: ${DASHSCOPE_API_KEY}
      DASHSCOPE_BASE_URL: ${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}
      DASHSCOPE_ASR_MODEL: ${DASHSCOPE_ASR_MODEL:-fun-asr-2025-11-07}
      LLM_BASE_URL: ${LLM_BASE_URL:-https://jiajiatemp.duckdns.org:30002/}
      LLM_API_KEY: ${LLM_API_KEY}
      LLM_MODEL: ${LLM_MODEL:-Qwen3.5-122B-W8A8}
      TTS_SERVICE_URL: http://tts-service:8001
      SINGING_SERVICE_URL: http://singing-service:8002
      SYSTEM_PROMPT: ${SYSTEM_PROMPT:-}
      PROMPT_SERVICE_URL: ${PROMPT_SERVICE_URL:-}
      VAD_THRESHOLD: ${VAD_THRESHOLD:-0.5}
      VAD_MIN_SPEECH: ${VAD_MIN_SPEECH:-0.2}
      VAD_MIN_SILENCE: ${VAD_MIN_SILENCE:-0.3}
      LLM_TIMEOUT: ${LLM_TIMEOUT:-5}
      TTS_TIMEOUT: ${TTS_TIMEOUT:-10}
      SINGING_TIMEOUT: ${SINGING_TIMEOUT:-30}
    depends_on:
      livekit:
        condition: service_healthy
      tts-service:
        condition: service_healthy
      singing-service:
        condition: service_healthy
    networks:
      - voice-network
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
    logging:
      driver: "json-file"
      options:
        max-size: "200m"
        max-file: "5"
        compress: "true"

networks:
  voice-network:
    driver: bridge
```

### 2.7 Nginx HTTPS 配置

创建 `nginx/ssl` 目录并放置证书：

```bash
mkdir -p /opt/voice-agent/nginx/ssl
```

**方案 A：Let's Encrypt 证书**（证书已保存在 `/etc/letsencrypt/live/voice.example.com/`）
```bash
# 复制证书到项目目录
sudo cp /etc/letsencrypt/live/voice.example.com/fullchain.pem /opt/voice-agent/nginx/ssl/fullchain.pem
sudo cp /etc/letsencrypt/live/voice.example.com/privkey.pem /opt/voice-agent/nginx/ssl/privkey.pem
sudo chown -R $USER:$USER /opt/voice-agent/nginx/ssl
```

**方案 B：商业证书**
```bash
# 将下载的证书文件上传到 ssl 目录
# 假设证书文件名为：
#   certificate.crt -> fullchain.pem
#   private.key -> privkey.pem
mv certificate.crt /opt/voice-agent/nginx/ssl/fullchain.pem
mv private.key /opt/voice-agent/nginx/ssl/privkey.pem
```

**创建生产环境 Nginx 配置** `nginx/nginx.conf`（覆盖原有配置）：

```nginx
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    # 日志格式
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    sendfile        on;
    keepalive_timeout 65;

    # Gzip 压缩
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/wasm;
    gzip_min_length 1024;

    # 上游服务定义
    upstream backend {
        server backend:3000;
    }

    upstream livekit {
        server livekit:7880;
    }

    # HTTP -> HTTPS 重定向
    server {
        listen 80;
        server_name voice.example.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name voice.example.com;

        # SSL 证书配置
        ssl_certificate /etc/nginx/ssl/fullchain.pem;
        ssl_certificate_key /etc/nginx/ssl/privkey.pem;

        # SSL 安全配置
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 1d;

        # 安全头
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

        # OCSP Stapling
        ssl_stapling on;
        ssl_stapling_verify on;

        # 客户端最大上传大小（用于 token 请求）
        client_max_body_size 10m;

        # 健康检查
        location /health {
            return 200 'ok';
            add_header Content-Type text/plain;
        }

        # 前端静态文件
        location / {
            root   /usr/share/nginx/html;
            index  index.html index.htm;
            try_files $uri $uri/ /index.html;

            location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff2?)$ {
                expires 7d;
                add_header Cache-Control "public, immutable";
            }
        }

        # Token API 代理
        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 5s;
            proxy_read_timeout 30s;
        }

        # LiveKit WebSocket + HTTP 代理
        location /livekit/ {
            proxy_pass http://livekit/;
            proxy_http_version 1.1;

            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";

            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            proxy_read_timeout 86400s;
            proxy_send_timeout 86400s;

            proxy_buffering off;
            proxy_cache off;
        }
    }
}
```

**注意**：将 `voice.example.com` 替换为你的实际域名。

### 2.8 SSL 证书自动续期

Let's Encrypt 证书有效期 90 天，需设置自动续期：

```bash
# 编辑 crontab
sudo crontab -e

# 添加以下行（每天检查证书，过期前30天自动续期）
0 0 * * * certbot renew --quiet --deploy-hook "docker exec voice-nginx nginx -s reload"
```

### 2.9 启动服务

```bash
cd /opt/voice-agent

# 构建并启动所有服务
docker-compose -f docker-compose.prod.yml up -d --build

# 查看服务状态
docker-compose -f docker-compose.prod.yml ps

# 查看实时日志
docker-compose -f docker-compose.prod.yml logs -f
```

**验证服务启动**：
```bash
# 检查所有容器健康状态
docker ps

# 健康检查
curl -I https://voice.example.com/health
curl -I https://voice.example.com/api/health

# 检查 LiveKit WebSocket 连接
wscat -c wss://voice.example.com/livekit/
```

---

## 3. API 端点说明

### 3.1 LLM API（独立端点）

| 配置项 | 值 |
|-------|-----|
| Base URL | `https://jiajiatemp.duckdns.org:30002/` |
| API Key | `sk-1r59zPAgUXYNMFcAXpjR6rnU0YNqNtjyP5CjRKzhqSD9PqRn` |
| 模型 | `Qwen3.5-122B-W8A8` |
| 用途 | 对话生成、意图理解、工具调用 |
| 认证方式 | Bearer Token |
| 协议 | OpenAI 兼容（DashScope 兼容模式）|

**示例请求**：
```bash
curl -X POST https://jiajiatemp.duckdns.org:30002/chat/completions \
  -H "Authorization: Bearer sk-1r59zPAgUXYNMFcAXpjR6rnU0YNqNtjyP5CjRKzhqSD9PqRn" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-122B-W8A8",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 500
  }'
```

### 3.2 ASR/TTS API（阿里云 DashScope）

> **注意**：需要单独申请 DashScope API Key

| 服务 | 端点 | 模型 | 用途 |
|-----|------|-----|------|
| ASR | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `fun-asr-2025-11-07` | 实时语音识别 |
| TTS | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen3-tts-vd-2026-01-26` | 语音合成 |

**DashScope 申请流程**：
1. 访问阿里云 DashScope 控制台：https://dashscope.console.aliyun.com/
2. 开通服务（如未开通）
3. 创建 API Key 并妥善保存
4. 将 API Key 填入 `.env` 的 `DASHSCOPE_API_KEY`

**DashScope 计费说明**：
- Fun-ASR：按识别时长计费
- Qwen3-TTS：按生成字符数计费
- 具体价格以阿里云官网为准

### 3.3 其他端点

| 服务 | 内部地址 | 外部端口 |
|-----|---------|---------|
| Nginx | `http://nginx:80` | 443 (HTTPS) |
| Backend | `http://backend:3000` | 通过 Nginx 代理 |
| LiveKit | `ws://livekit:7880` | 通过 Nginx 代理 |
| TTS Service | `http://tts-service:8001` | 内部 |
| Singing Service | `http://singing-service:8002` | 内部 |

---

## 4. 运维

### 4.1 日志查看

**实时日志**：
```bash
# 查看所有服务日志
docker-compose -f docker-compose.prod.yml logs -f

# 查看特定服务日志
docker-compose -f docker-compose.prod.yml logs -f nginx
docker-compose -f docker-compose.prod.yml logs -f agent
docker-compose -f docker-compose.prod.yml logs -f livekit

# 查看最近 100 行日志
docker-compose -f docker-compose.prod.yml logs --tail 100

# 搜索错误日志
docker-compose -f docker-compose.prod.yml logs | grep -i error
```

**持久化日志位置**（已配置日志轮转）：
```
/opt/voice-agent/logs/nginx/     # Nginx 访问日志
/var/lib/docker/                 # Docker 容器日志（json-file 格式）
```

**日志轮转配置**：
- 单个日志文件最大 100MB（LiveKit 200MB）
- 保留最多 5 个轮转文件
- 超过后自动压缩

### 4.2 服务重启

**重启单个服务**：
```bash
docker-compose -f docker-compose.prod.yml restart agent
```

**重启所有服务**：
```bash
docker-compose -f docker-compose.prod.yml restart
```

**停止并重建**：
```bash
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d --build
```

**查看重启历史**：
```bash
docker inspect voice-agent | grep -A 5 "RestartCount"
docker stats --no-stream
```

### 4.3 Prometheus + Grafana 监控接入

#### 4.3.1 启用 LiveKit Prometheus 指标

编辑 `livekit.yaml`，取消注释 prometheus 配置：

```yaml
# Prometheus 指标导出
prometheus:
  port: 9090
  exporter:
    enabled: true
```

重启 LiveKit 服务：
```bash
docker-compose -f docker-compose.prod.yml restart livekit
```

#### 4.3.2 Prometheus 配置

创建 `prometheus.yml`：

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  # LiveKit 指标
  - job_name: 'livekit'
    static_configs:
      - targets: ['livekit:9090']
    metrics_path: /metrics

  # Nginx 指标（需启用 nginx-exporter）
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:9113']

  # 后端服务指标（如果后端有 /metrics 端点）
  - job_name: 'backend'
    static_configs:
      - targets: ['backend:3000']
```

#### 4.3.3 Grafana 看板导入

1. 登录 Grafana
2. 导入看板：
   - LiveKit 官方看板 ID: `15822`（LiveKit Analytics）
   - 或使用 `15821`（LiveKit Cloud）

3. 自定义看板查询示例：
```promql
# 在线房间数
livekitRooms

# 当前参与者数
livekitParticipants

# 服务健康状态
up{job="livekit"}
```

### 4.4 数据备份

#### 4.4.1 需要备份的内容

| 内容 | 路径 | 备份频率 |
|-----|------|---------|
| 配置文件 | `/opt/voice-agent/.env` | 每次修改时 |
| SSL 证书 | `/opt/voice-agent/nginx/ssl/` | 证书更新时 |
| LiveKit 配置 | `/opt/voice-agent/livekit.yaml` | 每次修改时 |
| 日志文件 | `/opt/voice-agent/logs/` | 每日 |

#### 4.4.2 备份脚本

创建 `/opt/voice-agent/scripts/backup.sh`：

```bash
#!/bin/bash
# backup.sh - 生产环境数据备份脚本

BACKUP_DIR="/opt/backups/voice-agent"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p ${BACKUP_DIR}

# 备份配置文件
tar -czf ${BACKUP_DIR}/config_${DATE}.tar.gz \
  /opt/voice-agent/.env \
  /opt/voice-agent/nginx/ssl/ \
  /opt/voice-agent/livekit.yaml \
  /opt/voice-agent/nginx/nginx.conf

# 备份 docker-compose 配置
cp /opt/voice-agent/docker-compose.prod.yml ${BACKUP_DIR}/docker-compose.prod.yml_${DATE}

# 备份 SSL 证书续期脚本
cp /etc/cron.d/certbot-renewal ${BACKUP_DIR}/certbot-renewal_${DATE} 2>/dev/null || true

# 清理 7 天前的备份
find ${BACKUP_DIR} -name "*.tar.gz" -mtime +7 -delete
find ${BACKUP_DIR} -name "*.yml_*" -mtime +7 -delete

echo "[$(date)] Backup completed: config_${DATE}.tar.gz"
```

添加执行权限并配置定时任务：
```bash
chmod +x /opt/voice-agent/scripts/backup.sh

# 每天凌晨 3 点执行备份
sudo crontab -e
# 添加：0 3 * * * /opt/voice-agent/scripts/backup.sh >> /var/log/backup.log 2>&1
```

---

## 5. 故障排查

### 问题 1：服务启动后立即退出

**症状**：容器启动后几秒内退出，`docker ps` 看不到容器。

**排查步骤**：
```bash
# 查看容器退出原因
docker-compose -f docker-compose.prod.yml ps -a

# 查看详细日志
docker-compose -f docker-compose.prod.yml logs <service_name>

# 检查环境变量是否正确加载
docker exec voice-agent env | grep -E "LLM_|DASHSCOPE_|LIVEKIT_"
```

**常见原因**：
- `.env` 文件缺少必需的环境变量
- 镜像构建失败（检查 Dockerfile）
- 端口被占用（`netstat -tlnp | grep 7880`）

**解决措施**：
```bash
# 重新构建镜像
docker-compose -f docker-compose.prod.yml build --no-cache agent

# 检查端口占用
sudo lsof -i :7880
sudo lsof -i :3000

# 使用正确的环境变量文件
docker-compose -f docker-compose.prod.yml --env-file /opt/voice-agent/.env up -d
```

---

### 问题 2：WebSocket 连接失败

**症状**：客户端无法连接 LiveKit，日志显示 WebSocket 升级失败。

**排查步骤**：
```bash
# 检查 LiveKit 服务状态
curl -I http://localhost:7880/

# 检查 Nginx WebSocket 代理配置
docker exec voice-nginx nginx -t

# 查看 LiveKit 日志
docker-compose -f docker-compose.prod.yml logs livekit | tail -50
```

**常见原因**：
- Nginx 配置缺少 `proxy_set_header Upgrade` 和 `Connection` 头
- 防火墙未开放 7880 端口
- LiveKit 配置中 `use_external_ip` 未正确设置

**解决措施**：
```bash
# 确保 Nginx 配置包含 WebSocket 支持（见本文档 2.7 节）

# 检查防火墙
sudo ufw status
sudo iptables -L -n | grep 7880

# 如果是云服务器，确保安全组已开放 7880 端口

# 编辑 livekit.yaml，确保云服务器环境设置正确
# use_external_ip: true
# node_ip: "<服务器公网IP>"
```

---

### 问题 3：LLM API 调用超时

**症状**：对话无响应，日志显示 `LLM_TIMEOUT` 或 `Connection timeout`。

**排查步骤**：
```bash
# 测试 LLM API 连通性
curl -v -X POST https://jiajiatemp.duckdns.org:30002/chat/completions \
  -H "Authorization: Bearer <LLM_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3.5-122B-W8A8","messages":[{"role":"user","content":"test"}],"max_tokens":10}'

# 查看 agent 日志中的超时错误
docker-compose -f docker-compose.prod.yml logs agent | grep -i timeout
```

**常见原因**：
- 独立 LLM API 端点不可达（网络问题或服务宕机）
- API Key 错误或已过期
- 网络延迟过高（增加 `LLM_TIMEOUT` 值）

**解决措施**：
```bash
# 编辑 .env，增加超时时间
LLM_TIMEOUT=15

# 重启 agent 服务
docker-compose -f docker-compose.prod.yml restart agent

# 检查 API Key 是否正确
docker exec voice-agent env | grep LLM_API_KEY
```

---

### 问题 4：DashScope API 错误

**症状**：ASR 或 TTS 调用失败，日志显示 `DashScope API error`。

**排查步骤**：
```bash
# 测试 DashScope API 连通性
curl -v -X POST https://dashscope.aliyuncs.com/compatible-mode/v1/audio/transcriptions \
  -H "Authorization: Bearer <DASHSCOPE_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model":"fun-asr-2025-11-07","file":"test"}'

# 查看 tts-service 日志
docker-compose -f docker-compose.prod.yml logs tts-service | tail -50
```

**常见原因**：
- DashScope API Key 未配置或错误
- API 账户余额不足或配额用尽
- 阿里云 DashScope 服务宕机

**解决措施**：
```bash
# 检查 API Key
docker exec voice-tts env | grep DASHSCOPE_API_KEY

# 登录阿里云控制台检查账户余额和服务状态
# https://dashscope.console.aliyun.com/

# 确认 API Key 有权限访问对应服务
```

---

### 问题 5：TTS 无声音输出

**症状**：对话正常但没有语音回复，agent 日志显示 TTS 调用成功。

**排查步骤**：
```bash
# 测试 TTS 服务
curl -X POST http://localhost:8001/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"你好","language":"zh"}'

# 检查音频采样率是否正确（项目要求 16kHz）
# 查看 agent 日志中的音频编码信息
docker-compose -f docker-compose.prod.yml logs agent | grep -i audio
```

**常见原因**：
- 浏览器自动播放策略阻止音频播放
- 音频采样率不匹配（需要 16kHz）
- 前端音量设置为 0

**解决措施**：
```bash
# 确保前端使用正确的音频配置
# 参考：docs/audio-resampling.md

# 在前端添加用户交互以触发音频播放
# （浏览器要求用户首次交互后才能自动播放音频）
```

---

### 问题 6：Nginx 502 Bad Gateway

**症状**：访问网站返回 502 错误。

**排查步骤**：
```bash
# 检查 upstream 服务是否健康
curl -I http://localhost:3000/api/health  # backend
curl -I http://localhost:7880/             # livekit

# 查看 Nginx 错误日志
docker exec voice-nginx cat /var/log/nginx/error.log

# 检查容器状态
docker-compose -f docker-compose.prod.yml ps
```

**常见原因**：
- backend 或 livekit 服务未启动
- 服务健康检查失败
- Nginx 与后端容器网络不通

**解决措施**：
```bash
# 重启所有服务
docker-compose -f docker-compose.prod.yml restart

# 等待服务就绪后重试
sleep 10
curl -I https://voice.example.com/api/health
```

---

### 问题 7：证书续期失败

**症状**：Let's Encrypt 证书过期，HTTPS 连接失败。

**排查步骤**：
```bash
# 检查证书过期时间
sudo certbot certificates

# 手动测试续期
sudo certbot renew --dry-run
```

**常见原因**：
- 域名未正确解析到服务器
- 80 端口被其他服务占用
- certbot 无法访问 `.well-known/acme-challenge/` 路径

**解决措施**：
```bash
# 确保 80 端口未被占用
sudo lsof -i :80

# 手动触发续期
sudo certbot renew --force-renewal

# 如果仍失败，手动重新获取证书
sudo certbot certonly --standalone -d voice.example.com
```

---

### 问题 8：内存不足导致容器被 OOM Kill

**症状**：`dmesg | grep -i oom` 显示容器被 kill，服务反复重启。

**排查步骤**：
```bash
# 检查内存使用
docker stats --no-stream

# 查看系统内存
free -h

# 检查 dmesg 中的 OOM 记录
dmesg | grep -i "out of memory"
dmesg | grep -i "killed process"
```

**常见原因**：
- 服务器内存不足（低于 8GB）
- 单个容器内存限制过高
- 同时运行了其他占用内存的服务

**解决措施**：
```bash
# 降低容器内存限制（编辑 docker-compose.prod.yml）
# 例如将 agent 从 2G 降到 1G

# 增加服务器内存或关闭其他服务

# 调整 Java/Node.js 堆内存（如适用）
```

---

### 问题 9：端口被占用导致启动失败

**症状**：`docker-compose up` 报错 `Bind for port 7880 failed: address already in use`。

**排查步骤**：
```bash
# 查看端口占用
sudo lsof -i :7880
sudo lsof -i :7881
sudo netstat -tlnp | grep -E '7880|7881|50000'
```

**常见原因**：
- 之前运行的 LiveKit 或其他服务未完全停止
- 多个 docker-compose 实例同时运行

**解决措施**：
```bash
# 停止占用端口的进程
sudo kill <PID>

# 或停止所有 docker 容器
docker-compose -f docker-compose.prod.yml down
docker stop $(docker ps -aq)

# 重新启动
docker-compose -f docker-compose.prod.yml up -d
```

---

### 问题 10：Docker 镜像拉取失败

**症状**：`docker-compose up` 报错 `repository not found` 或 `connection timeout`。

**排查步骤**：
```bash
# 检查 Docker 配置是否使用国内镜像
docker info | grep -i mirror

# 测试 Docker Hub 连通性
curl -I https://registry.hub.docker.com
```

**常见原因**：
- Docker Hub 被墙（国内服务器常见）
- 未配置镜像加速器

**解决措施**：
```bash
# 配置阿里云镜像加速（免费）
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<EOF
{
  "registry-mirrors": [
    "https://mirror.ccs.tencentyun.com",
    "https://docker.mirrors.ustc.edu.cn"
  ]
}
EOF

# 重启 Docker
sudo systemctl restart docker

# 重新拉取镜像
docker-compose -f docker-compose.prod.yml pull
```

---

## 附录：快速检查清单

### 部署前检查
- [ ] 服务器系统：Ubuntu 20.04 / 22.04 LTS
- [ ] 内存 >= 8GB
- [ ] 域名已备案并解析到服务器
- [ ] SSL 证书已获取
- [ ] 安全组已开放 80/443/7880/7881/50000-50050 端口
- [ ] `.env` 文件已正确配置所有 API Key

### 部署后验证
- [ ] `curl -I https://voice.example.com/health` 返回 200
- [ ] `docker ps` 显示所有 6 个容器运行中
- [ ] WebSocket 连接测试通过
- [ ] LLM API 调用成功
- [ ] TTS 服务响应正常

### 运维检查（每日）
- [ ] 日志无 Error 日志
- [ ] 内存使用率 < 80%
- [ ] 磁盘使用率 < 80%
- [ ] 容器健康状态正常

---

*文档版本：1.0*
*最后更新：2026-05-02*
*[已完成]*