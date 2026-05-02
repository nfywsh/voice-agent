# 踩坑经验记录

## T10: Docker Compose 全链路启动

### 问题 1: backend 服务缺少 healthcheck
**问题描述**：nginx depends_on backend，配置了 `condition: service_healthy`，但 backend 服务没有定义 healthcheck，导致 nginx 永远无法启动。

**影响**：即使 backend 启动成功，nginx 也会因为无法满足 depends_on 条件而处于 waiting 状态。

**解决方案**：
```yaml
# docker-compose.yml 中为 backend 添加 healthcheck
backend:
  # ... 其他配置 ...
  healthcheck:
    test: ["CMD", "wget", "-qO-", "http://localhost:3000/api/health"]
    interval: 15s
    timeout: 3s
    retries: 3
    start_period: 10s
```

**验证**：
- backend 的 Dockerfile 已包含 HEALTHCHECK 指令（指向 /api/health）
- 但 docker-compose.yml 中没有声明式 healthcheck，导致 `docker compose up` 时 depends_on 条件无法满足
- 必须同时在 Dockerfile 和 docker-compose.yml 中配置

### 问题 2: start-all.sh 使用固定 sleep
**问题描述**：原脚本使用 `sleep 8` 等待服务就绪，不够健壮，可能因机器性能导致服务未就绪就继续。

**解决方案**：
改为循环检查各服务健康端点，确保服务真正就绪后再继续：
```bash
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
```

### 问题 3: 服务端口参考
- TTS Service: 8001
- Singing Service: 8002
- Backend (Next.js): 3000
- LiveKit: 7880
- Nginx: 80

### 验证清单
1. 所有脚本已设置可执行权限 (chmod +x)
2. docker-compose.yml 中所有依赖服务的 healthcheck 都已配置
3. start-all.sh 按依赖顺序等待服务就绪（先 LiveKit/Backend，再 TTS/Singing，最后 Nginx）
4. logs.sh 支持服务名称映射（tts -> tts-service, singing -> singing-service 等）

### 问题 4: agent.py 使用错误的 LLM 环境变量
**问题描述**：用户配置了独立的 LLM API 端点（`https://jiajiatemp.duckdns.org:30002/`），docker-compose.yml 也正确传递了 `LLM_BASE_URL` 和 `LLM_API_KEY`，但 `agent/agent.py` 实际读取的是 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_BASE_URL`，导致实际调用了 DashScope 而非独立端点。

**影响**：独立 LLM API 配置完全未生效，Agent 仍使用 DashScope API。

**问题代码**（agent.py 第 280-296 行）：
```python
dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "")
dashscope_base_url = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# ... 后续使用 dashscope_api_key 和 dashscope_base_url 创建 LLM
```

**修复方案**：将 `DASHSCOPE_API_KEY`/`DASHSCOPE_BASE_URL` 替换为 `LLM_API_KEY`/`LLM_BASE_URL`：
```python
llm_api_key = os.environ.get("LLM_API_KEY", "")
llm_base_url = os.environ.get(
    "LLM_BASE_URL", "https://jiajiatemp.duckdns.org:30002/"
)
```

**验证**：
1. 确认 docker-compose.yml 中 agent 服务有 `LLM_BASE_URL` 和 `LLM_API_KEY` 环境变量（已有）
2. 确认 agent/.env 也添加了这两个变量（缺失，需补充）
3. 修复 agent.py 后，独立 LLM API 才能真正生效

### 已知限制
- agent 服务没有 healthcheck（因为是 Long-running 进程，无 HTTP 端点）
- agent 依赖 tts-service 和 singing-service，但这两个服务有 healthcheck，所以整体启动顺序可控

## T14: 前端环境准备

### 配置检查结果

#### 1. Vite 配置 (`frontend/vite.config.js`)
- **proxy 设置**: `/api` → `http://localhost:3000` - **正确**
- **build.outDir**: `dist` - **正确**

#### 2. Nginx 配置 (`nginx/nginx.conf`)
- **静态文件路径**: `root /usr/share/nginx/html` - 需要确认 docker-compose 中映射到 `frontend/dist`
- **SPA fallback**: `try_files $uri $uri/ /index.html` - **已配置**

#### 3. Frontend package.json
- **build 命令**: `vite build` - **正确**
- **LiveKit 依赖**: `@livekit/components-react`, `@livekit/components-styles`, `livekit-client`, `livekit-server-sdk` - **已安装**

#### 4. 环境变量问题
- **问题**: 缺少 `.env.production` 文件
- **当前状态**: 只有 `.env` 文件，`VITE_LIVEKIT_URL=ws://localhost:7880`（开发环境直连）
- **生产环境需求**: 需要 `.env.production` 配置正确的 LiveKit URL：
  ```
  VITE_LIVEKIT_URL=wss://your-domain.com/livekit
  ```
- **影响**: 生产环境前端无法正确连接到 LiveKit 服务

#### 5. 构建产物检查
- **问题**: `frontend/dist/` 目录不存在
- **原因**: 尚未执行构建
- **修复**: 部署前需执行 `cd frontend && npm install && npm run build`

### 修复建议

1. **创建 `frontend/.env.production`**:
```bash
VITE_LIVEKIT_URL=wss://your-domain.com/livekit
```

2. **验证 docker-compose 静态文件映射**: 确认 nginx 服务的 volume 配置类似：
```yaml
nginx:
  volumes:
    - ./frontend/dist:/usr/share/nginx/html:ro
```

3. **部署前构建**: 执行 `cd frontend && npm install && npm run build`

### 验证清单
- [ ] `.env.production` 已创建并配置正确的 LiveKit URL
- [ ] `frontend/dist/` 目录存在且包含构建产物
- [ ] docker-compose.yml 中 nginx volume 正确映射到 `frontend/dist`
- [ ] 生产环境 `VITE_LIVEKIT_URL` 使用 wss 协议

## T0: Docker 镜像预拉取

### 问题: 镜像拉取失败
**环境**: WSL2 中无法直接 pull 镜像

**尝试的镜像源**:
1. `docker.io/library/python:3.11-slim` - TLS handshake timeout
2. `mirror.gcr.io/library/python:3.11-slim` - TLS handshake timeout
3. `registry.cn-hangzhou.aliyuncs.com/library/python:3.11-slim` - pull access denied
4. `docker.chenxue.cc/library/python:3.11-slim` - EOF

**分析**: WSL2 环境网络问题，Docker daemon 虽已安装但访问外网受限

**解决方案**: 需要在 Windows 主机上配置 Docker 镜像加速器

**需要在 Windows 配置的镜像加速器**:
```json
{
  "registry-mirrors": [
    "https://docker.chenxue.cc",
    "https://dockerhub.azk8s.cn",
    "https://reg-mirror.qiniu.com"
  ]
}
```

**手动拉取命令**（在 Windows Docker Desktop 或 WSL2 中）:
```bash
docker pull python:3.11-slim
docker pull node:20-alpine
docker pull nginx:alpine
docker pull livekit/livekit-server:v1.7
```