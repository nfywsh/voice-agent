# 本地部署清单

以下操作需要在你的本地电脑完成。

## 1. 前端构建

```bash
cd D:\FAE\voice-agent\frontend
npm config set registry https://registry.npmmirror.com
npm install
npm run build
```

> 构建产物会输出到 `frontend/dist/`，Nginx 会自动挂载。

## 2. Docker 构建与启动

```bash
cd D:\FAE\voice-agent

# 确认 .env 配置正确
cat .env
# 重点检查: DASHSCOPE_API_KEY 是否为你的真实 Key

# 构建并启动
docker compose up -d --build
# 或者使用启动脚本
bash scripts/start-all.sh
```

## 3. 验证各服务

```bash
# 检查所有容器状态
docker compose ps

# 查看 Agent 日志（重点关注 DashScope 连接是否正常）
docker compose logs -f agent

# 运行集成测试
bash scripts/integration-tests.sh
```

手动验证检查点：

| 检查 | 命令/操作 | 期望结果 |
|------|----------|---------|
| 健康检查 | `curl http://localhost/health` | `ok` |
| TTS 健康检查 | `curl http://localhost:8001/health` | `{"status":"ok","api_mode":"dashscope",...}` |
| Singing 健康检查 | `curl http://localhost:8002/health` | `{"status":"ok","mock_mode":true,...}` |
| Token 生成 | `curl "http://localhost:3000/api/token?room=test&username=user"` | 返回 JWT token |
| 前端页面 | 浏览器打开 `http://localhost` | 显示语音助手界面 |
| 语音对话 | 输入房间名加入 → 说话 | AI 用 DashScope 语音回复 |

## 4. 常见问题排查

### TTS 服务 503
```
检查 DASHSCOPE_API_KEY 是否正确
检查模型名 qwen3-tts-vd-2026-01-26 是否可用
docker compose logs tts-service
```

### Agent ASR 连接失败
```
检查 DashScope Fun-ASR WebSocket 连接
docker compose logs agent | grep DashScopeSTT
```

### Agent LLM 调用失败
```
检查 DASHSCOPE_API_KEY 和模型名 Qwen3.5-122B-W8A8
docker compose logs agent | grep LLM
```

### 前端无法连接 LiveKit
```
确认浏览器访问的是 ws://localhost:7880（不是 wss://）
检查前端 .env 中 VITE_LIVEKIT_URL 配置
docker compose logs nginx
```

### 容器启动失败
```
查看具体容器日志：
docker compose logs <service-name>
常见原因：端口冲突、.env 缺少必需变量
```

## 5. 本地开发模式（不用 Docker）

如果不想用 Docker，也可以分别启动各服务：

```bash
# 终端 1: Token 服务
cd backend && npm install && npm run dev

# 终端 2: TTS 服务
cd tts_service && pip install -r requirements.txt && python main.py

# 终端 3: Singing 服务
cd singing_service && pip install -r requirements.txt && python main.py

# 终端 4: LiveKit Server
# 下载 livekit-server: https://github.com/livekit/livekit/releases
livekit-server --config livekit.yaml

# 终端 5: Agent
cd agent && pip install -r requirements.txt && python agent.py

# 终端 6: 前端
cd frontend && npm run dev
```

## 6. 后续可选操作

- [ ] 将 Singing 服务接入真实歌声合成 API（当 VibeVoice 有线上版本时）
- [ ] 配置 SSL 证书（生产环境）
- [ ] 添加 .gitignore 排除 .env 和 node_modules