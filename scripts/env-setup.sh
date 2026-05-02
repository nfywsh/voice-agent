#!/bin/bash
# D:\FAE\env\ 路径配置脚本
# 在 WSL 环境下将 Docker、Conda、pip、npm、HuggingFace 缓存统一配置到 D:\FAE\env\ 下，避免 C 盘膨胀

set -e

mkdir -p D:\FAE\env\conda\envs
mkdir -p D:\FAE\env\conda\pkgs
mkdir -p D:\FAE\env\pip-packages
mkdir -p D:\FAE\env\npm-global
mkdir -p D:\FAE\env\hf-cache
mkdir -p D:\FAE\env\docker-data

echo "=== D:\FAE\env\ 路径配置 ==="

# 1. Conda 配置
if command -v conda &> /dev/null; then
    conda config --prepend envs_dirs D:\FAE\env\conda\envs
    conda config --prepend pkgs_dirs D:\FAE\env\conda\pkgs
    echo "[OK] Conda 环境目录已配置到 D:\FAE\env\conda"
else
    echo "[SKIP] Conda 未安装，跳过"
fi

# 2. pip 配置
pip config set global.target D:\FAE\env\pip-packages
pip config set global.cache-dir D:\FAE\env\pip-packages\cache
echo "[OK] pip 包目录已配置到 D:\FAE\env\pip-packages"

# 3. npm 配置
npm config set prefix D:\FAE\env\npm-global
echo "[OK] npm 全局包已配置到 D:\FAE\env\npm-global"

# 4. HuggingFace 缓存（使用镜像 + 本地缓存）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=D:\FAE\env\hf-cache
echo "[OK] HuggingFace 缓存已配置到 D:\FAE\env\hf-cache (镜像: hf-mirror.com)"

# 5. Docker Desktop 数据迁移（如需）
# 将以下内容添加到 Docker Desktop → Settings → General → Docker Engine：
# {
#   "data-root": "D:\\\\FAE\\\\env\\\\docker-data"
# }
echo "[INFO] Docker 数据目录建议配置到 D:\FAE\env\docker-data"
echo "[INFO] 方法：Docker Desktop → Settings → Resources → Advanced → Disk image location"

# 6. 生成持久化环境变量（写入 ~/.bashrc）
cat >> ~/.bashrc << 'EOF'

# === D:\FAE\env\ 路径配置 ===
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=D:\FAE\env\hf-cache
export PIP_TARGET=D:\FAE\env\pip-packages
EOF

echo ""
echo "=== 配置完成 ==="
echo "目录结构："
ls -la D:\FAE\env\