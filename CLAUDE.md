# CLAUDE.md

## 0. 网络与资源准备准则（首要）

**所有网络操作（镜像拉取、包安装、模型下载）必须使用国内代理/镜像，否则必然失败。**

### 镜像源配置

**Docker 镜像** — 优先使用国内加速器：
```bash
# 方法 1: Docker Hub 官方镜像（实测可用）
docker pull python:3.11-slim
# 方法 2: 中科大镜像（备用）
docker pull mirror.gcr.io/library/python:3.11-slim
# 方法 3: 阿里云容器镜像 ACR（推荐生产环境）
docker pull registry.cn-hangzhou.aliyuncs.com/xxx/python:3.11-slim
```

**pip 安装** — 必须配置国内源：
```bash
# 永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 或阿里云
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
```
国内常用源：
- 清华：`https://pypi.tuna.tsinghua.edu.cn/simple`
- 阿里云：`https://mirrors.aliyun.com/pypi/simple`
- 腾讯云：`https://mirrors.cloud.tencent.com/pypi/simple`

**npm 安装** — 配置国内镜像：
```bash
npm config set registry https://registry.npmmirror.com
```

**apt 安装** — 配置国内镜像（如需）：
```bash
sed -i 's|http://deb.debian.org|http://mirrors.aliyun.com/debian|g' /etc/apt/sources.list
```

**Conda 环境** — 配置国内镜像：
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
```

**HuggingFace 模型下载** — 使用镜像或 `HF_ENDPOINT`：
```bash
# 方法 1: 设置 HF 镜像（推荐）
export HF_ENDPOINT=https://hf-mirror.com

# 方法 2: 使用 huggingface-cli
huggingface-cli download microsoft/VibeVoice-1.5B --local-dir ./models/VibeVoice-1.5B
```

**Web 搜索** — 国内环境优先：
- 使用 `WebSearch` 时，关键词用中文搜索国内技术博客和文档
- 优先参考知乎、CSDN、简书、官方中文文档

### 资源准备优先原则

**新任务启动前，先开一个 agent 专门准备基础设施（镜像、环境），其他 agent 等待或并行做无网络依赖的工作。**

具体判断：
- 需要 `pip install`、`npm install` → 先配 pip/npm 镜像
- 需要 `docker pull`、`docker build` → 先拉取/构建基础镜像
- 需要 `conda create`、大型 pip 包（如 torch、transformers）→ 先建 conda 环境
- 需要下载 HuggingFace 模型（>1GB）→ 先用镜像下载

**推荐流程**：
1. 新任务开始前，检查所需资源是否就绪
2. 如有缺失，立即创建 `prepare-xxx` 任务优先执行
3. 其他 agent 可并行做"纯代码编写"（无网络依赖）直到资源就绪

---

## 1. 编码前先思考

**不要假设。不要掩饰困惑。明确呈现权衡。**

在实现之前：
- 明确写出你的假设。如果不确定，就提问。
- 如果存在多种解释，先把它们列出来，不要默默自行选择。
- 如果有更简单的方法，就直接指出来。在有必要时提出异议。
- 如果有不清楚的地方，就停下来。说清楚困惑点，并提问。

## 2. 简单优先

**只写解决问题所需的最少代码。不做任何预设性扩展。**

- 不要加入超出需求范围的功能。
- 不要为一次性代码做抽象。
- 不要加入未被要求的"灵活性"或"可配置性"。
- 不要为不可能发生的场景写错误处理。
- 如果你写了 200 行，但 50 行就够，就重写。

问问自己："一个资深工程师会认为这太复杂了吗？" 如果答案是会，那就继续简化。

## 3. 外科手术式修改

**只改必须改的内容。只清理你自己造成的问题。**

编辑现有代码时：
- 不要"顺手优化"相邻代码、注释或格式。
- 不要重构没有坏掉的部分。
- 保持现有风格，即使你个人会写成别的样子。
- 如果发现无关的死代码，可以指出，但不要删除。

当你的改动产生遗留项时：
- 删除那些因你的修改而变成未使用的 import、变量或函数。
- 不要删除原本就存在的死代码，除非被明确要求。

检验标准：每一行改动都应当能直接追溯到用户请求。

## 4. 目标驱动执行

**先定义成功标准，再循环推进，直到验证通过。**

把任务转换成可验证的目标：
- "添加校验" → "先为非法输入写测试，再让测试通过"
- "修复这个 bug" → "先写能复现它的测试，再让测试通过"
- "重构 X" → "确保改动前后测试都通过"

对于多步骤任务，先给出简短计划：
```
1. [步骤] → 验证：[检查项]
2. [步骤] → 验证：[检查项]
3. [步骤] → 验证：[检查项]
```

强有力的成功标准能让你独立闭环推进。弱成功标准（"把它弄好"）则会不断需要额外澄清。

---

## 5. 多 Agent 协作准则（subagent 任务清单）

### 任务清单执行原则

当有子任务需要分配给多个 subagent 并行执行时：

**第一步：基础设施准备（任何代码任务之前）**

所有需要网络的子任务之前，必须先开一个 `prepare` agent 专门处理：
- Docker 基础镜像预拉取（`docker pull python:3.11-slim`、`docker pull nvidia/cuda:12.4-cudnn8-runtime-ubuntu22.04` 等）
- pip/npm/apt 镜像源配置脚本
- Conda 环境创建（如需 GPU Python 环境）
- HuggingFace 模型预下载到 `models/` 目录

**第二步：任务分发**

按 `doc/task-plan-for-subagents.md` 中的任务图谱分发：
- 阶段 0（T0/T1/T2）：无依赖，3 个 agent 并行启动
- 阶段 1（T3/T4/T5/T6）：依赖 T0 完成后，启动 4 个 agent 并行
- 阶段 2（T7/T8）：依赖 T0，启动 2 个 agent 并行
- 后续阶段按依赖顺序执行

**第三步：模型选择策略**

根据任务复杂度选择合适的模型，避免浪费算力：

| 任务类型 | 模型选择 | 说明 |
|---------|---------|------|
| 简单快速任务（代码补全、简单问答、格式检查） | `Qwen3.5-122B-W8A8` | W8A8 量化，支持 256k 上下文，延迟低 |
| 复杂全面任务（架构设计、多文件重构、全面测试） | `GLM5.1` | 性能强，但接近 150k 上下文时需压缩 |
| 模型调用异常 | `MiniMax-2.7` | 作为兜底模型，响应快，稳定性高 |

**上下文压缩原则**：
- GLM5.1 接近 150k token 时，主动压缩历史对话（保留最近 N 轮或关键上下文）
- 压缩方式：删除中间轮次的详细描写，保留关键结论和当前任务目标
- 判断方法：单轮请求 token 数 > 50k 时考虑压缩

**第四步：交接确认**

每个 agent 完成自己负责的任务后，必须：
1. 在文件中标注"[已完成]"注释
2. 将具体发现（如 API 不匹配、依赖版本冲突）明确写出
3. 供后续 agent 参考，避免重复踩坑