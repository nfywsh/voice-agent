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