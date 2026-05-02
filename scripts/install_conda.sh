# 1. 下载 Miniconda 安装脚本到 D:\FAE\env\conda
mkdir -p D:\FAE\env\conda
cd D:\FAE\env\conda

# 2. 下载 Miniconda（使用清华镜像）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh


# 3. 安装到 D:\FAE\env\conda（不创建 base 环境）
bash miniconda.sh -b -p /mnt/d/FAE\env\conda

# 4. 配置路径
export PATH=/mnt/d/FAE/env/conda/bin:$PATH

# 5. 配置国内镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --set show_channel_urls yes

# 6. 持久化到 ~/.bashrc
echo 'export PATH=/mnt/d/FAE/env/conda/bin:$PATH' >> ~/.bashrc
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
echo 'export HF_HOME=/mnt/d/FAE/env/hf-cache' >> ~/.bashrc

# 7. 验证
conda --version