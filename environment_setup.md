# Slime 项目环境搭建记录

本文档记录了使用 `micromamba` 为 Slime 项目配置开发环境的详细流程。本次搭建通过 `micromamba` 在虚拟环境内部独立安装 CUDA 12.9 工具链，实现环境隔离，不依赖物理机的 CUDA 版本。

## 1. 做什么

我们需要搭建 Slime 项目的 Python 开发环境，包括：
1. 使用 `micromamba`（官方 conda-forge 源）创建 Python 3.12 虚拟环境。
2. 使用 `micromamba` 在虚拟环境内部独立安装完整的 CUDA 12.9 工具链，实现环境隔离，避免依赖物理机的 CUDA 版本。
3. 全局安装 `uv` 替换 `pip` 以实现超高速的依赖安装，全程通过本地 Mihomo (Clash Meta) 提供代理。
4. 安装 PyTorch 等依赖，使用 `cu129`（CUDA 12.9）版本的预编译包。
5. 手动下载和安装 `sglang` 以及构建工具 `ninja`、`cmake`，所有 GitHub 拉取均通过本地 Mihomo 代理加速，并使用 SSH 克隆方式（`git@github.com:...`）避开污染规则。
6. 编译安装 Flash Attention 及 Megatron-LM 等深度学习依赖库，适当降低了并行编译数防止 OOM。

## 2. 怎么做

### 附加说明1：后台挂起运行
由于脚本执行时间较长，强烈建议使用 `nohup` 将其挂起在后台运行，以防 SSH 连接断开导致安装失败。您可以运行以下命令：

```bash
nohup ./build_conda.sh > build_conda.log 2>&1 &
```

执行后，您可以通过以下命令实时查看安装进度和日志：
```bash
tail -f build_conda.log
```

在安装并激活 Python 3.12 后，由于我们已经配置了本地代理，可以直接使用官方源，并安装极速包管理器 `uv`，用 `alias pip="uv pip"` 替代原生 pip，从而极大缩短依赖安装时间。
```bash
micromamba create -n slime python=3.12 pip -c conda-forge -y --override-channels
micromamba activate slime

# 安装 uv 并开启 pip 加速
pip install uv
alias pip="uv pip"
```

为了彻底解决 `nvcc` 找不到的问题，我们在环境内部直接使用 micromamba 安装与 PyTorch 匹配的 CUDA 12.9 工具链，并导出相应的环境变量。
```bash
# install cuda 12.9 as it's the default cuda version for torch
micromamba install -n slime cuda cuda-nvtx cuda-nvtx-dev nccl -c nvidia/label/cuda-12.9.1 -y
micromamba install -n slime -c conda-forge cudnn -y

# 导出环境变量让后续编译（如 flash-attn、apex）能找到 micromamba 内部的 CUDA 工具链
export CUDA_HOME=$MAMBA_ROOT_PREFIX/envs/slime
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# 开启全局代理，利用本机运行的 Mihomo 代理加速 GitHub 拉取
export http_proxy=http://127.0.0.1:8890
export https_proxy=http://127.0.0.1:8890
export all_proxy=http://127.0.0.1:8890
```

### 附加说明2：后台挂起运行
如需加速 GitHub 资源下载，可将相关地址替换为镜像站前缀 `https://gh-proxy.com/`，例如：  
`https://gh-proxy.com/https://github.com/xxx/yyy/archive/refs/tags/v1.0.0.tar.gz`

### 第二步：安装 PyTorch
```bash
pip install cuda-python==12.9
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129
```

### 第三步：手动克隆构建工具与大型仓库
由于项目中涉及到大量的源码编译，我们需要构建工具 `ninja`。
同时为了避免脚本中的 `git clone` 受全局 URL 替换等错误配置的影响，对于 `sglang`、`Megatron-LM` 以及 `slime`，脚本已经更改为使用基于 SSH 的代理拉取方式（`git clone git@github.com:...`），您需要提前配置好 SSH 的 `ProxyCommand` 和公钥：
```bash
pip install cmake ninja
```

### 第四步：安装其他依赖与补丁应用
随后脚本将依次源码安装 `flash-attn`、`apex`、`Megatron-LM`，并拉取 `slime` 源码应用相应补丁：
```bash
pip install nvidia-cudnn-cu12==9.16.0.29
```

### 第五步：处理依赖版本兼容问题
在所有的基础依赖安装完成后，由于目前上游依赖存在一些严格的版本限制（比如较新的 `huggingface-hub` 和旧版本 `transformers` 的冲突），我们需要将一些可能自动被装成过高版本的库降级：
```bash
pip install "huggingface-hub<1.0"
```

### 第六步：日常激活环境与运行任务
在未来的每次登录中，您需要手动初始化 micromamba 并激活 `slime` 虚拟环境。只有在环境激活后，才能运行训练脚本（比如 `ray` 命令）：
```bash
export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate slime
```
*提示：您可以将这四行命令追加到您的 `~/.bashrc` 文件末尾，这样以后每次登录就只需要执行 `micromamba activate slime` 即可。*

## 3. 遇到的问题和解决方案

### 问题 1: 手动下载 sglang 或 GitHub 仓库时网络超时或提示找不到仓库
- **表现**：执行 `git clone` 或使用 `pip install git+https://github.com/...` 时，如果服务器访问 GitHub 受限，或者环境内被配置了错误的 `url.insteadOf` 全局替换，会导致仓库拉取失败或脚本挂起。
- **解决方案**：我们在本地启动了 Mihomo 代理（端口 8890/8891），并在脚本顶端导出了 `http_proxy`、`https_proxy` 环境变量让 pip 等工具走代理。针对顽固的 Git URL 全局污染，我们直接将大的代码仓库拉取改为 SSH 方式（`git clone git@github.com:...`），并通过修改 `~/.ssh/config` 设置 `ProxyCommand` 让 SSH 流量通过代理的 SOCKS5 端口，彻底解决网络和污染问题。

### 问题 2: 手动下载 ninja 或编译库时发生 `ninja: build stopped` 等错误
- **表现**：在安装 `flash-attn` 或 `apex` 时，通过 `pip` 会调用 `ninja` 进行 C++ 扩展的并行编译，由于服务器 CPU 或内存资源有限，并发量过大可能导致编译进程被 OOM 杀死。
- **解决方案**：原脚本中已针对 `flash-attn` 设置了 `MAX_JOBS=64`，如果仍然失败，请将 `MAX_JOBS` 调低（例如 `MAX_JOBS=8` 或 `4`）；同理 `apex` 中的 `--parallel 8` 也可酌情降低以减小内存占用。如果 `pip install ninja` 失败，也可以选择从系统包管理器安装 `sudo apt install ninja-build`。

### 问题 3: flash-attn 编译报错 `FileNotFoundError: No such file or directory: '/usr/local/cuda-xx/bin/nvcc'`
- **表现**：PyTorch 及依赖包均安装成功，但在执行 `pip install flash-attn` 时，报错中断提示找不到 `nvcc`。
- **原因**：物理机上可能不存在预期版本的 CUDA（如 `cuda-13.0`），或者 `CUDA_HOME` 环境变量指向了错误路径。
- **解决方案**：放弃依赖系统级的 CUDA 工具，转而使用 `micromamba` 直接在虚拟环境中安装对应版本的 CUDA 工具链（如 CUDA 12.9），并让 `CUDA_HOME` 指向虚拟环境前缀：
  ```bash
  micromamba install -n slime cuda cuda-nvtx cuda-nvtx-dev nccl -c nvidia/label/cuda-12.9.1 -y
  micromamba install -n slime -c conda-forge cudnn -y

  export CUDA_HOME=$MAMBA_ROOT_PREFIX/envs/slime
  export PATH=$CUDA_HOME/bin:$PATH
  export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
  ```
  这样可以做到环境完全隔离，避免任何本机路径问题。修改后，您可以重新运行脚本恢复编译。

### 问题 4: `micromamba install` 下载 nccl 等大文件时频发 `Transferred a partial file` 错误
- **表现**：执行环境内部 CUDA 或 cudnn 安装时，由于 `nccl`（约 300MB）等包体积极大，经过外网代理下载容易出现 `Transferred a partial file` 并反复 retry。
- **解决方案**：将脚本中的 `conda-forge` 频道替换为清华镜像源，同时在环境变量中设置 `export no_proxy="localhost,127.0.0.1,tsinghua.edu.cn"` 使国内镜像流量直连，避免被代理掐断：
  ```bash
  cat << 'EOF' > ~/.mambarc
  channels:
    - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  show_channel_urls: true
  default_channels:
    - defaults
  EOF
  ```
  同时在脚本顶部代理设置中加入 `export no_proxy="localhost,127.0.0.1,tsinghua.edu.cn"`。

### 问题 5: PyTorch 安装成了错误的 CUDA 版本 (如 cu128 而非 cu129)
- **表现**：执行环境检查发现 `torch.version.cuda` 输出为 12.8，而非预期的 12.9。这会导致后续编译依赖 CUDA 的扩展（如 flash-attn、apex）时出现版本不匹配错误。
- **原因**：脚本中指定 `--index-url https://download.pytorch.org/whl/cu129` 安装 PyTorch 时，因为网络请求官方源超时而失败。由于脚本未设置遇到错误即停止，继续往下执行，在后续安装 `sglang` 等包时触发了自动依赖解析，pip 自动从默认的国内镜像源（如阿里云）拉取了基于 CUDA 12.8 编译的默认版本。
- **解决方案**：使用环境对应的 pip 绝对路径（或激活环境后）卸载错误的包并重新安装：
  ```bash
  /home/brx/micromamba/envs/slime/bin/pip uninstall torch torchvision torchaudio -y
  /home/brx/micromamba/envs/slime/bin/pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129
  ```

### 问题 6: 脚本执行一半静默退出（如安装完 sglang 后停止）
- **表现**：查看日志发现，脚本在输出 `Successfully installed sglang...` 后就结束了，没有任何明显报错，但后续的 `flash-attn`、`apex` 编译以及 `Megatron-LM` 的克隆等均未执行。
- **原因**：在执行类似 `pip install -e .` 等耗时操作时，可能由于内存波动（OOM Killer）或者外部终端断开导致 bash 脚本意外终止，且未抛出直观错误。
- **解决方案**：将脚本中未执行的部分（从 `pip install cmake ninja` 开始到最后）提取保存为新脚本（例如 `build_conda_part2.sh`），然后继续挂后台执行：
  ```bash
  nohup ./build_conda_part2.sh > build_conda_part2.log 2>&1 &
  ```

### 问题 7: Git Clone (SSH) 报错 `OpenSSL version mismatch` 或其它鉴权问题
- **表现**：脚本执行 `git clone ssh://git@github.com/...` 或者通过 pip 安装 git 仓库时，由于本地环境可能缺少 SSH 密钥或者因环境变量导致 OpenSSL 冲突（如 `OpenSSL version mismatch`），导致代码拉取失败。
- **解决方案**：针对这种环境依赖强的情况，最简单稳妥的做法是**直接弃用 SSH 方式，全面改回 HTTPS 方式拉取代码**，并确保我们在脚本开头已经配置好了 HTTP/HTTPS 代理。
  - 将脚本里所有的 `git@github.com:...` 或 `git+ssh://...` 替换为 `https://github.com/...` 和 `git+https://...` 即可。

### 问题 8: FlashInfer/SGLang 运行时 JIT 编译报错 `ld: cannot find -lcuda`
- **表现**：运行训练代码时，SGLang 初始化 CUDA Graph 触发底层 FlashInfer 的实时动态编译（JIT），在最后的链接阶段编译器（如 `gcc` 的 `ld`）报错中断，提示找不到 `libcuda.so` 通信库：`/usr/bin/ld: cannot find -lcuda: No such file or directory`。
- **原因**：`libcuda.so` 是显卡驱动提供的核心通信库，存在于宿主物理机系统（如 `/usr/lib/x86_64-linux-gnu/libcuda.so`）。由于我们为了环境隔离，在 `micromamba` 中独立安装了全套的 CUDA 工具链和 C++ 编译器（GCC 等），这导致 Conda/Micromamba 内部的链接器被设定为“只搜索虚拟环境内的 `lib` 目录”，不再去系统全局目录寻找依赖，从而找不到物理机的驱动库。

**补充说明：这个问题是使用 micromamba 一定会出现的吗？**
是的，只要在 micromamba/conda 中使用了它们自带的 C++ 编译器链，几乎一定会遇到这个“隔离机制”带来的副作用。背后的原理如下：

1. **Conda/Micromamba 的极致隔离**：为了保证环境的可复现性，micromamba 在安装 `gcc` 等编译工具时，故意修改了编译器的行为。它限制了内部链接器（`ld`）只能在 `$CONDA_PREFIX/lib`（即你的环境目录）下去寻找动态库，强行阻断了它去读取物理机 `/usr/lib` 目录的行为。
2. **`libcuda.so` 的特殊性**：一般的依赖库都可以被打包进 micromamba 环境里，唯独 `libcuda.so` 不行。因为这个文件不属于 CUDA Toolkit，它属于**英伟达显卡底层驱动**，必须和当前物理机内核中的 NVIDIA 驱动版本绝对一致。因此，它只能存在于宿主机的系统目录下。

*注：如果不使用 micromamba 在环境内重装 CUDA 和编译器，而是完全依赖物理机上自带的 GCC 编译器和系统级 CUDA 工具链，系统的链接器默认会扫描全局系统目录，就不会出现此问题。*

#### 解决方案与实践场景
处理这个问题时，我们需要根据宿主机的 CUDA 环境现状，采取不同的策略：

**情况 1：宿主机驱动支持但是已安装版本较低，需要在虚拟环境中独立安装 CUDA 12.9+**
- **场景描述**：这是最推荐的做法。通过 `nvidia-smi` 确认宿主机的显卡驱动上限（CUDA Version） $\ge$ 13.0，允许我们无视宿主机上已安装的旧版 CUDA Toolkit（例如宿主机的 `nvcc -V` 只有 12.8）。
- **脚本配置**：在 `build_conda.sh` 脚本中，直接让 micromamba 在虚拟环境中安装与 PyTorch 绑定的 12.9 版本，完全隔离物理机的 12.8：
  ```bash
  micromamba install -n slime cuda cuda-nvtx cuda-nvtx-dev nccl -c nvidia/label/cuda-12.9.1 -y
  ```
- **补齐桩文件**：在虚拟环境内部署好 12.9 后，只需将物理机驱动的 `libcuda.so` 软链接到环境内即可解决 JIT 编译报错：
  ```bash
  ln -s /usr/lib/x86_64-linux-gnu/libcuda.so /home/brx/micromamba/envs/slime/lib/libcuda.so
  ```

**情况 2：宿主机中已经安装好了 CUDA 12.9+**
- **处理方式**：宿主机已经老老实实装好了目标版本的 CUDA，所以在 `build_conda.sh` 脚本中可以放弃使用 micromamba 独立安装 12.9：
  1. 不要在环境内执行 `micromamba install cuda ...`。
  2. 在脚本中直接将环境变量指向物理机已经装好的 12.9 目录：
     ```bash
     export CUDA_HOME=/usr/local/cuda-12.9
     export PATH=$CUDA_HOME/bin:$PATH
     export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
     ```
  3. **无需在环境内手动补齐桩文件**：因为你现在直接使用了物理机的原生环境路径，系统自带的编译器和链接器默认就能在 `/usr/lib/x86_64-linux-gnu/` 下找到 `libcuda.so`。如果因特殊原因依然找不到，你应该将宿主机的驱动桩文件链接到物理机的 CUDA 系统库中：
     ```bash
     sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so /usr/local/cuda-12.9/lib64/libcuda.so
     ```