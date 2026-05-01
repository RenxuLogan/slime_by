#!/bin/bash

set -ex

# 开启全局代理，利用本机运行的 Mihomo 代理加速网络请求
export http_proxy=http://127.0.0.1:8890
export https_proxy=http://127.0.0.1:8890
export all_proxy=http://127.0.0.1:8890
export no_proxy="localhost,127.0.0.1,tsinghua.edu.cn,aliyun.com"

# 强制为 Git 屏蔽任何全局的 URL 替换配置
export GIT_CONFIG_GLOBAL=/dev/null

# create conda
# 如果本机已经有 micromamba，则跳过下载安装过程
if [ ! -f "$HOME/.local/bin/micromamba" ]; then
  curl -L micro.mamba.pm/install.sh -o install.sh
  chmod +x install.sh
  yes '' | ./install.sh
  rm install.sh
else
  echo "micromamba already exists, skipping installation."
fi
export PS1=tmp
mkdir -p ~/.cargo/
touch ~/.cargo/env

# 在非交互式 shell 中，~/.bashrc 开头的 case $- in *i*) 保护会导致后续配置被跳过
# 我们需要手动设置 micromamba 路径，并手动初始化 mamba 环境

export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"

# 配置 micromamba 使用 conda-forge 官方源，防止大文件下载（如 nccl）时由于镜像不稳定而中断
cat << 'EOF' > ~/.mambarc
channels:
  - conda-forge
show_channel_urls: true
default_channels:
  - defaults
EOF

micromamba create -n slime python=3.12 pip -c conda-forge -y --override-channels
micromamba activate slime

# 配置 pip 国内源并安装 uv

pip install uv
# 后续使用 uv pip 进行加速安装
alias pip="uv pip"

# 使用 micromamba 在环境内部安装 CUDA 12.9 及相关依赖
# install cuda 12.9 as it's the default cuda version for torch 
micromamba install -n slime cuda cuda-nvtx cuda-nvtx-dev nccl -c nvidia/label/cuda-12.9.1 -y 
micromamba install -n slime -c conda-forge cudnn -y

# 导出环境变量让后续编译（如 flash-attn、apex）能找到 micromamba 内部的 CUDA 工具链
export CUDA_HOME=$MAMBA_ROOT_PREFIX/envs/slime
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# 设置特定的版本号
export SGLANG_COMMIT="bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2"
export MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"

# 将 BASE_DIR 修改为 /home/brx/data
export BASE_DIR=${BASE_DIR:-"$HOME/data"}
cd $BASE_DIR


# 安装CUDA
pip install cuda-python==12.9
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# install sglang
if [ ! -d "sglang" ]; then
  # 使用 HTTPS 方式进行克隆
  git clone https://github.com/sgl-project/sglang.git
fi
cd sglang
git checkout ${SGLANG_COMMIT}
# Install the python packages
pip install -e "python[all]"
cd ..

# 将 编译ninja
pip install cmake ninja


# flash attn
# the newest version megatron supports is v2.7.4.post1
MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
pip install flash-linear-attention==0.4.1

# 降低 apex 编译时的 --parallel 数，防止 OOM 
NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

pip install git+https://gh-proxy.com/https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall
pip install git+https://gh-proxy.com/https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

# 安装 sglang-router
if [ ! -f "sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl" ]; then
  wget https://gh-proxy.com/https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl
fi
pip install https://gh-proxy.com/https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/sglang_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall

# 安装 megatron
cd $BASE_DIR
if [ ! -d "Megatron-LM" ]; then
  # 使用 HTTPS 方式进行克隆
  git clone https://gh-proxy.com/https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
cd Megatron-LM/
git checkout ${MEGATRON_COMMIT}
pip install -e .
cd ..

# install slime and apply patches

# if slime does not exist locally, clone it
if [ ! -d "$BASE_DIR/slime" ]; then
  # 使用 HTTPS 方式进行克隆
  git clone https://gh-proxy.com/https://github.com/THUDM/slime.git
fi
cd slime/
export SLIME_DIR=$BASE_DIR/slime
pip install -e .

# https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29 #（前面的编译需要用到固定的版本）重新安装回来
pip install "numpy<2"

# 添加补丁
cd $BASE_DIR/sglang
git apply $SLIME_DIR/docker/patch/v0.5.9/sglang.patch
cd $BASE_DIR/Megatron-LM
git apply $SLIME_DIR/docker/patch/v0.5.9/megatron.patch