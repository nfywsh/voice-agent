# 全双工语音聊天机器人 — 并行开发任务清单

> **⚠️ 开始任何任务前必读：**
> 网络操作（pip/npm/docker/模型下载）必须使用国内代理，详见 `CLAUDE.md` 第 0 节。
> 建议第一个 subagent 先执行基础设施准备（镜像预拉取、pip 镜像配置），再分发其他任务。
> 参见下方「执行顺序与并行策略」中的推荐启动顺序。

本文档将整个项目拆分为多个独立任务，供多 subagent 并行执行。
每个任务描述清晰、依赖明确、交付物具体，可同时分配给不同 agent 独立完成。

---

## 任务图谱

```
阶段 0：基础设施（所有任务的依赖，需最先完成）
    │
    ├── [T0] 完善 docker-compose.yml + .env 配置
    ├── [T1] Nginx + LiveKit 配置
    └── [T2] 模型文件准备说明文档

阶段 1：微服务独立开发（4 个服务可并行）
    │
    ├── [T3] Agent 服务开发 + Docker
    ├── [T4] TTS Service 开发 + Docker
    ├── [T5] Singing Service 开发 + Docker
    └── [T6] Token Service (Next.js) 开发 + Docker

阶段 2：前端开发
    │
    ├── [T7] 前端页面开发 + 样式
    └── [T8] 前端构建 + Nginx 集成

阶段 3：联调与测试
    │
    ├── [T9] 单元测试 + 集成测试
    ├── [T10] Docker Compose 全链路启动
    └── [T11] 音频采样率专项测试

阶段 4：文档与交付
    │
    ├── [T12] README 完善
    └── [T13] 部署手册编写
```

---

## 阶段 0：基础设施（所有任务的前置依赖）

### [T0] 完善 docker-compose.yml + 环境配置

**负责人**：任意一个 agent（建议第一个启动）

**依赖**：无

**任务内容**：
1. 确认 `docker-compose.yml` 中所有服务端口、环境变量、GPU 配置完整
2. 确认 `agent/.env`、`.env.example` 包含所有必需变量
3. 添加 `docker-compose.override.yml` 用于本地开发覆盖（如 Mock 模式）
4. 验证 GPU 隔离配置：`CUDA_VISIBLE_DEVICES` 在 tts-service 和 singing-service 间的分配

**交付物**：
- `docker-compose.yml`（已验证）
- `.env.example`（完整，所有变量有注释）
- `docker-compose.override.yml`（本地开发用，可选）

---

### [T1] Nginx + LiveKit 配置

**负责人**：独立 agent

**依赖**：无

**任务内容**：
1. 完善 `nginx/nginx.conf`：
   - WebSocket 超时设为 86400s
   - 添加 `/livekit/` 路径代理
   - 添加 gzip 压缩配置
   - 添加安全响应头（X-Frame-Options 等）
2. 完善 `livekit.yaml`：
   - 配置 `rtc.use_external_ip: true`（适配云服务器）
   - 添加 `room.auto_create: true`
   - 添加 Prometheus metrics 导出配置（注释说明）
3. 生成测试用 SSL 证书脚本 `scripts/generate-ssl.sh`

**交付物**：
- `nginx/nginx.conf`（已完善）
- `livekit.yaml`（已完善）
- `scripts/generate-ssl.sh`（可执行脚本）

---

### [T2] 模型文件准备说明

**负责人**：独立 agent

**依赖**：无

**任务内容**：
1. 编写 `docs/models-download.md`，说明三个模型的下载方式：
   - Qwen3-TTS：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`（HuggingFace）
   - VibeVoice：`microsoft/VibeVoice-1.5B`（HuggingFace）
   - Whisper（备选 ASR）：`openai/whisper-large-v3`
2. 编写 `scripts/download-models.sh` 自动下载脚本（使用 `huggingface-cli` 或 `snapshot_download`）
3. 确认 `models/` 目录结构符合 docker-compose volume 挂载要求

**交付物**：
- `docs/models-download.md`
- `scripts/download-models.sh`

---

## 阶段 1：微服务独立开发（4 个服务可 100% 并行）

### [T3] Agent 服务开发 + Docker

**负责人**：独立 agent

**依赖**：[T0]

**任务内容**：
1. **核心代码完善**（基于已有 `agent/agent.py`）：
   - 确认 `entrypoint()` 中 VAD 参数从环境变量正确读取
   - 确认 `SINGING_MOCK_MODE` 环境变量控制歌声是否走 Mock
   - 确认 LLM 通过 `livekit.plugins.openai.LLM` 接入（而非自建适配器）
   - 确认 STT 降级逻辑（Deepgram → Whisper）正确
2. **测试 Mock 模式**：
   - 写一个独立测试脚本 `agent/test_mock.py`，在无 TTS/Singing 模型时验证基本对话流程
   - 测试 LLM 的 Function Calling 是否正确触发 `sing_a_song` 工具
3. **Docker 化**：
   - 完善 `agent/Dockerfile`，确保 `livekit-agents` 及所有依赖正确安装
   - 确认 `PYTHONUNBUFFERED=1` 设置（日志实时输出）
4. **编写测试**：
   - `agent/tests/test_agent.py`：测试 `@function_tool` 装饰器、`sing_a_song` 参数传递

**交付物**：
- `agent/agent.py`（最终版）
- `agent/Dockerfile`（已验证可构建）
- `agent/test_mock.py`（可独立运行）
- `agent/tests/test_agent.py`

---

### [T4] TTS Service 开发 + Docker

**负责人**：独立 agent

**依赖**：[T0]

**任务内容**：
1. **核心代码完善**（基于已有 `tts_service/main.py`）：
   - 确认模型加载使用正确的 HuggingFace 模型 ID
   - 确认 `generate_stream()` 中的推理 API 与 Qwen3-TTS 实际 API 一致
   - 如果 Qwen3-TTS 官方 API 与代码不符，修正 `generate_stream()` 实现
   - 添加文本长度截断逻辑（>500 字截断并返回 `X-Truncated: true` header）
2. **健康检查完善**：
   - `/health` 端点返回 `model_loaded`、`device`、`sample_rate`
   - `/tts/reload` 端点支持运行时重新加载模型
3. **Mock 模式**（用于开发无 GPU 环境）：
   - 添加 `ENABLE_MOCK=true` 环境变量时，返回静音或正弦波测试音频
   - Mock 下 TTS 请求响应时间 <100ms，方便前端调试
4. **Docker 化**：
   - 完善 `tts_service/Dockerfile`
   - 确认 `torch` 使用 `cuda` 版本的 wheel
   - 确认 `HEALTHCHECK` 设置 `start_period: 120s`（模型加载慢）
5. **单元测试**：
   - `tts_service/tests/test_tts.py`：测试 `/tts/stream` 返回音频格式、`/health` 返回正确状态

**交付物**：
- `tts_service/main.py`（最终版，含 Mock 模式）
- `tts_service/Dockerfile`（已验证可构建）
- `tts_service/tests/test_tts.py`
- `tts_service/requirements.txt`（如需补充依赖）

---

### [T5] Singing Service 开发 + Docker

**负责人**：独立 agent

**依赖**：[T0]

**任务内容**：
1. **核心代码完善**（基于已有 `singing_service/main.py`）：
   - 确认 VibeVoice 模型加载使用正确的 HuggingFace 模型 ID
   - 确认 `generate_singing()` 中的推理 API 与 VibeVoice 实际 API 一致
   - 如果 VibeVoice API 与代码不符，修正实现（可参考微软官方 VibeVoice 仓库文档）
   - 确认歌词格式处理：`Speaker 1: 歌词内容` 每行格式正确
2. **缓存机制**：
   - 确认 `LRUCache` 正常工作（相同 lyrics 不重复推理）
   - 添加 `/cache` 清除接口
3. **Mock 模式**（默认开启，便于开发调试）：
   - `ENABLE_MOCK=true` 时返回简单正弦波歌声音频（440Hz+523Hz 交替）
   - Mock 模式响应时间 <500ms
4. **Docker 化**：
   - 完善 `singing_service/Dockerfile`
   - 确认 `HEALTHCHECK` 设置 `start_period: 180s`
   - 确认 GPU 隔离（通过 `CUDA_VISIBLE_DEVICES`）
5. **单元测试**：
   - `singing_service/tests/test_singing.py`：测试 `/sing` 返回音频、`/health`、缓存命中

**交付物**：
- `singing_service/main.py`（最终版，含 Mock + 缓存）
- `singing_service/Dockerfile`（已验证可构建）
- `singing_service/tests/test_singing.py`

---

### [T6] Token Service (Next.js) 开发 + Docker

**负责人**：独立 agent

**依赖**：[T0]

**任务内容**：
1. **核心代码完善**（基于已有 `backend/app/api/token/route.ts`）：
   - 确认 JWT 有效期设为 1 小时
   - 确认 `roomJoin: true`、`canPublish: true`、`canSubscribe: true`、`canPublishData: true` 权限
   - 确认参数校验（room/user 名称长度限制、特殊字符过滤）
2. **错误处理**：
   - API Key 未配置时返回 500 + 明确错误信息（而非空指针）
   - 请求格式错误时返回 400 + 错误详情
3. **健康检查路由**：
   - 添加 `GET /api/health` 返回 `{status: "ok"}`
4. **Docker 化**：
   - 完善 `backend/Dockerfile`
   - 确认 `standalone` output 模式正确配置
   - 确认 `HEALTHCHECK` 使用 `/api/health` 端点
5. **Next.js 配置**：
   - 确认 `next.config.js` 的 `output: 'standalone'` 正确
   - 确认 `tsconfig.json` 存在且正确

**交付物**：
- `backend/app/api/token/route.ts`（最终版）
- `backend/Dockerfile`（已验证可构建）
- `backend/tsconfig.json`

---

## 阶段 2：前端开发（2 个任务可并行）

### [T7] 前端页面开发 + 样式

**负责人**：独立 agent

**依赖**：[T0]（需要 `.env` 中的 `VITE_LIVEKIT_URL`）

**任务内容**：
1. **核心组件**（基于已有代码）：
   - 确认 `VoiceRoom.jsx` 使用正确的 LiveKit SDK API
   - `useVoiceAssistant()` 返回的 state（listening/thinking/speaking/connecting）正确显示
   - `BarVisualizer` 正确渲染 AI 说话时的音频波形
   - `useDataChannel` 监听 Agent 推送的文本消息（用于 TTS 降级展示）
2. **错误处理**：
   - Token 获取失败时显示明确错误 + 重试按钮
   - WebRTC 连接断开时自动重连（LiveKit SDK 内置） + 状态提示
   - Agent 异常断开时页面提示"连接已断开"并提供返回入口
3. **样式完善**：
   - 确认 `App.css` 已有样式覆盖了所有组件状态
   - 深色主题适配（背景 #0f0f1a，文字 #e0e0e0）
   - 移动端响应式布局（768px 断点）
4. **性能优化**：
   - 确认 `messages` 数组超过 50 条时只保留最新 50 条（防止内存泄漏）
   - 使用 `React.memo` 避免不必要的重渲染

**交付物**：
- `frontend/src/components/VoiceRoom.jsx`（最终版）
- `frontend/src/App.css`（最终版）
- `frontend/src/App.jsx`（最终版）
- `frontend/src/components/Visualizer.jsx`（简化版）

---

### [T8] 前端构建 + Nginx 集成

**负责人**：独立 agent

**依赖**：[T7]（需前端代码完成）

**任务内容**：
1. **Vite 配置**：
   - 确认 `vite.config.js` 的 proxy 设置正确（`/api` → Next.js 3000 端口）
   - 确认 `build.outDir: 'dist'`
2. **生产构建**：
   - 执行 `npm run build`，验证构建成功无警告
   - 确认 `dist/` 目录包含所有静态资源
3. **Nginx 集成**：
   - 确认 `nginx.conf` 中静态文件路径指向 `frontend/dist`
   - 确认 SPA fallback 配置（`try_files $uri $uri/ /index.html`）
4. **环境变量**：
   - 生产环境 `.env.production` 配置 `VITE_LIVEKIT_URL` 为正式域名
5. **最终检查**：
   - `index.html` 存在且引入正确的入口 JS
   - 构建产物中无 `__VITE_PROXY__` 等开发期标记残留

**交付物**：
- `frontend/dist/`（构建产物，可直接部署）
- `frontend/vite.config.js`（已验证）
- `nginx.conf`（静态文件路径正确）

---

## 阶段 3：联调与测试（3 个任务可部分并行）

### [T9] 单元测试 + 集成测试

**负责人**：独立 agent

**依赖**：[T3]、[T4]、[T5]、[T6]

**任务内容**：
1. **各服务单元测试**（4 个服务并行执行各自测试）：
   - `agent/tests/test_agent.py`（T3 交付物）
   - `tts_service/tests/test_tts.py`（T4 交付物）
   - `singing_service/tests/test_singing.py`（T5 交付物）
2. **集成测试脚本** `scripts/integration-tests.sh`：
   ```bash
   #!/bin/bash
   set -e

   echo "=== 1. 测试所有服务健康检查 ==="
   curl -f http://localhost:8001/health || { echo "TTS service down"; exit 1; }
   curl -f http://localhost:8002/health || { echo "Singing service down"; exit 1; }
   curl -f http://localhost:3000/api/health || { echo "Backend down"; exit 1; }

   echo "=== 2. 测试 TTS 流式输出 ==="
   curl -X POST http://localhost:8001/tts/stream \
     -H "Content-Type: application/json" \
     -d '{"text": "你好"}' --no-buffer | head -c 1024 && echo "TTS OK"

   echo "=== 3. 测试 Singing Mock 模式 ==="
   curl -X POST http://localhost:8002/sing \
     -H "Content-Type: application/json" \
     -d '{"lyrics": "Speaker 1: 测试歌词", "title": "测试"}' --no-buffer | head -c 1024 && echo "Singing OK"

   echo "=== 4. 测试 Token 生成 ==="
   TOKEN=$(curl -s "http://localhost:3000/api/token?room=test&username=user" | jq -r .token)
   [ -n "$TOKEN" ] && [ "$TOKEN" != "null" ] && echo "Token API OK" || { echo "Token API FAIL"; exit 1; }

   echo "=== 全部测试通过 ==="
   ```
3. **测试覆盖报告**：统计各服务的测试覆盖率（可选，使用 `pytest --cov`）

**交付物**：
- 各服务 `tests/` 目录
- `scripts/integration-tests.sh`（可执行）
- 测试覆盖率报告（可选）

---

### [T10] Docker Compose 全链路启动

**负责人**：独立 agent

**依赖**：[T3]、[T4]、[T5]、[T6]、[T7]（前端构建完成）

**任务内容**：
1. **启动脚本** `scripts/start-all.sh`：
   ```bash
   #!/bin/bash
   set -e
   echo "=== 构建所有镜像 ==="
   docker-compose build

   echo "=== 启动所有服务 ==="
   docker-compose up -d

   echo "=== 等待服务就绪 ==="
   sleep 10
   for svc in livekit backend nginx; do
     until docker-compose exec $svc wget -qO- http://localhost:80/ 2>/dev/null || \
           docker-compose exec $svc curl -sf http://localhost:7880/ 2>/dev/null; do
       echo "Waiting for $svc..."
       sleep 3
     done
   done

   echo "=== 服务状态 ==="
   docker-compose ps

   echo "=== 访问地址 ==="
   echo "前端: http://localhost"
   echo "LiveKit: ws://localhost:7880"
   ```
2. **停止脚本** `scripts/stop-all.sh`
3. **日志查看脚本** `scripts/logs.sh [service_name]`
4. **验证全链路**：
   - 浏览器打开 `http://localhost`
   - 输入房间名、用户名，加入房间
   - 确认 AI 语音回复正常
   - 确认打断功能生效

**交付物**：
- `scripts/start-all.sh`（可执行）
- `scripts/stop-all.sh`（可执行）
- `scripts/logs.sh`（可执行）

---

### [T11] 音频采样率专项测试

**负责人**：独立 agent

**依赖**：[T10]（全链路服务运行）

**任务内容**：
1. **重采样正确性测试**：
   - 用 `sox` 或 `ffmpeg` 生成标准 24kHz 1kHz 正弦波测试音频
   - 送入 TTS 服务，验证输出是 48kHz 且音调正确（无变调）
   - 用 `sox` 验证输出文件的实际采样率：`sox input.wav -n stat`
2. **ASR 识别质量测试**：
   - 用 `sox` 生成标准 16kHz 测试音频（录音或 TTS 输出重采样得到）
   - 送入 Deepgram，验证识别准确率
   - 对比：48kHz 直接送入 vs 正确 16kHz 送入，识别率应有显著差异
3. **VAD 打断测试**：
   - 在 AI 说话时（播放 TTS 音频时），对着麦克风说话
   - 验证 AI 立即停止当前说话并开始聆听
   - 多次测试，统计打断成功率
4. **歌声采样率测试**：
   - Singing Mock 模式输出正弦波（已知频率）
   - 录音回放，验证音高无变化（无花栗鼠效应）
5. **测试报告**：
   - 输出 `docs/audio-test-report.md`，包含测试用例、结果、数据截图

**交付物**：
- `docs/audio-test-report.md`
- `scripts/audio-test.sh`（音频专项测试可执行脚本）
- 测试用音频文件 `test-audio/`

---

## 阶段 4：文档与交付

### [T12] README 完善

**负责人**：独立 agent

**依赖**：所有功能开发完成

**任务内容**：
1. 重写 `README.md`，包含：
   - 项目简介（3 句话内）
   - 架构图（纯文本版）
   - 快速开始（3 步：配环境变量 → 下载模型 → docker-compose up）
   - 环境变量完整说明表
   - GPU 配置说明（多卡分配）
   - 常见问题（FAQ）
   - 开发调试说明（各服务独立启动方式）
2. 确保 README 内容与实际文件路径、命令一致

**交付物**：`README.md`（可直接给新成员阅读）

---

### [T13] 部署手册编写

**负责人**：独立 agent

**依赖**：[T12]

**任务内容**：
1. 编写 `docs/deployment.md`，包含：
   - **生产环境准备**：
     - 服务器要求（CPU、内存、GPU 型号、网络）
     - 域名 + SSL 证书配置
     - DNS 解析配置
   - **部署步骤**：
     - Ubuntu 20.04 / 22.04 环境准备
     - Docker + NVIDIA Docker Toolkit 安装
     - 模型文件下载（使用 `scripts/download-models.sh`）
     - 环境变量配置（生产环境 `.env`）
     - `docker-compose.prod.yml` 编写（含资源限制、日志轮转）
     - Nginx HTTPS 配置（真实证书）
   - **运维**：
     - 日志查看（`docker-compose logs -f`）
     - 服务重启
     - 监控（Prometheus + Grafana 接入说明）
     - 数据备份（模型文件、配置）
   - **故障排查**：
     - 常见 10 种生产环境问题及解决步骤

**交付物**：`docs/deployment.md`

---

## 执行顺序与并行策略

### 推荐启动顺序

```
第一波（3 个 agent 并行）：
  Agent A → [T0] + [T1] + [T2]
  Agent B → [T3]（依赖 T0）
  Agent C → [T6]（依赖 T0）

第二波（4 个 agent 并行，等 T0 完成）：
  Agent A → [T4]（依赖 T0）
  Agent B → [T5]（依赖 T0）
  Agent C → [T7]（依赖 T0）
  Agent D → [T8 前半：vite 配置]（依赖 T0）

第三波（等 T3/T4/T5/T6 完成）：
  Agent A → [T9]（依赖 T3/T4/T5/T6）
  Agent B → [T10]（依赖 T3/T4/T5/T6/T8）
  Agent C → [T11]（依赖 T10）

第四波（等 T9/T10/T11 完成）：
  Agent A → [T12]
  Agent B → [T13]
```

### 并行度说明

| 阶段 | 并行 agent 数 | 关键依赖 |
|------|-------------|---------|
| 阶段 0 | 3 | 全部无依赖 |
| 阶段 1 | 4 | 均依赖 T0 |
| 阶段 2 | 2 | 均依赖 T0 |
| 阶段 3 | 3 | 依赖阶段 1 完成 |
| 阶段 4 | 2 | 依赖阶段 3 完成 |

### 总任务数：13 个
### 理论最短完成时间：4 轮（4 串并行阶段）

---

*文档版本：v1.0*
*创建日期：2026-05-01*
*维护者：项目团队*