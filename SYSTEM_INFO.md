# 🖥️ 系统硬件与 Python 环境信息 / System Hardware & Environment

> 本文件供 AI 助手快速了解本机的硬件配置、Python 开发环境和 Git 配置。
> 最后更新: 2026-06-25

---

## 🔧 硬件配置 / Hardware

| 组件 | 型号/规格 |
|------|-----------|
| **操作系统** | Windows 11 Home China (64-bit, Build 2009) |
| **CPU** | 12th Gen Intel Core i7-12650H (10核16线程) |
| **内存** | 16 GB DDR4 3200MHz (2×8GB, Micron) |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop GPU |
| **显存** | 8 GB GDDR6 (8188 MiB) |
| **GPU 驱动** | 596.49 |
| **CUDA Compute Capability** | 8.9 |
| **磁盘 1 (系统)** | YMTC PC300 512GB NVMe SSD |
| **磁盘 2 (数据)** | KIOXIA EXCERIA PLUS G3 2TB NVMe SSD |

### GPU 兼容性摘要

- **CUDA 版本支持**: 驱动支持最高 CUDA 12.8
- **Compute Capability**: 8.9（支持 FP16/BF16 Tensor Core、RT Core）
- **适用场景**: 中小型模型训练/推理（LLaMA-7B QLoRA、SDXL LoRA、YOLO 系列等均可胜任）
- **显存限制**: 8GB VRAM，大模型需使用量化（4-bit/8-bit）或 LoRA 微调策略

---

## 🐍 Conda 环境 / Python Environments

### 环境总览

| 环境名 | Python 版本 | 路径 | 主要用途 |
|--------|------------|------|----------|
| `base` | 3.13.9 | `D:\anaconda3` | Anaconda 基础环境 |
| `ai_env` | 3.10.20 | `C:\Users\15703\.conda\envs\ai_env` | AI 通用环境 |
| `GPUpytorch-env` ⭐ | 3.11.14 | `D:\anaconda3\envs\GPUpytorch-env` | GPU PyTorch 主环境 |

> ⭐ = 推荐 AI 开发首选环境

---

### GPUpytorch-env (GPU 开发环境)

```
路径:     D:\anaconda3\envs\GPUpytorch-env
Python:   3.11.14
PyTorch:  2.11.0+cu128
CUDA:     12.8
GPU:      ✅ 可用 (RTX 4060 Laptop, 8GB)
```

**激活命令 (bash/终端):**
```bash
conda activate GPUpytorch-env
```

**运行脚本:**
```bash
D:/anaconda3/envs/GPUpytorch-env/python.exe <脚本名>.py
```

**安装包:**
```bash
D:/anaconda3/envs/GPUpytorch-env/python.exe -m pip install <包名>
```

---

### ai_env (AI 通用环境)

```
路径:     C:\Users\15703\.conda\envs\ai_env
Python:   3.10.20
PyTorch:  未安装
```

---

### base (Anaconda 基础)

```
路径:     D:\anaconda3
Python:   3.13.9
```

---

## 🔗 Git / GitHub 配置

| 项目 | 值 |
|------|-----|
| **Git 用户名** | `konodioda39` |
| **Git 邮箱** | `141110572+konodioda39@users.noreply.github.com` |
| **GitHub 账户** | <https://github.com/konodioda3939> |
| **AIGC 项目仓库** | <https://github.com/konodioda3939/aigc-game-asset-pipeline> |
| **仓库 SSH 地址** | `git@github.com:konodioda3939/aigc-game-asset-pipeline.git` |
| **认证方式** | SSH 密钥（`~/.ssh/id_rsa`，已添加到 GitHub） |
| **SSH 连通性** | ✅ `ssh -T git@github.com` 认证成功 |
| **本地分支** | `master` |
| **远程分支** | `main` |

### 推送代码到 GitHub

```bash
cd d:/aigc-project
git push origin master:main
```

> ⚠️ **用 SSH 不用 HTTPS**：国内网络 HTTPS 直连 GitHub 会断连（`Connection was reset`），必须用 SSH。

### 提交并推送（完整流程）

```bash
cd d:/aigc-project
git add .
git commit -m "描述你的改动"
git push origin master:main
```

---

## 🗂️ 项目路径速查

| 项目 | 路径 |
|------|------|
| **AIGC 主项目** | `d:\aigc-project` |
| **Python 推理服务** | `d:\aigc-project\inference_server\` |
| **LoRA 权重** | `d:\aigc-project\lora_output\adapter_model.safetensors` |
| **Unity 插件源码** | `d:\aigc-project\unity_plugin\Assets\Editor\AIGCAssetGenerator\` |
| **Unity 项目** | `D:\Unity\My_AIGC_Testproject` |
| **Unity 内插件位置** | `D:\Unity\My_AIGC_Testproject\Assets\Editor\AIGCAssetGenerator\` |
| **模型缓存（不上传 Git）** | `d:\aigc-project\cache\hub\`（~8GB） |

---

## 🌐 网络注意事项

| 场景 | 问题 | 解决方案 |
|------|------|----------|
| HuggingFace 下载 | 国内直连超时 | 设 `HF_ENDPOINT=https://hf-mirror.com`（必须在 `import transformers` 之前） |
| GitHub HTTPS | 直连断连 | **用 SSH**（`git@github.com:...`），已配置完毕 |
| pip 安装慢 | 国外源慢 | 可用 `pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <包名>` |

---

## 📋 AI 助手速查卡 / Quick Reference for AI

```
OS:              Windows 11 Home China 64-bit
CPU:             i7-12650H (12th Gen, 10C/16T)
RAM:             16 GB DDR4
GPU:             RTX 4060 Laptop 8GB (CC 8.9)
GPU Driver:      596.49
Max CUDA:        CUDA 12.8
Primary Env:     GPUpytorch-env (Python 3.11, PyTorch 2.11+cu128)
Primary Env Path: D:\anaconda3\envs\GPUpytorch-env
Workspace:       d:\Code\ComputerVision_ChenFei
AIGC Project:    d:\aigc-project
GitHub:          konodioda3939 (SSH auth)
Git Push:        cd d:\aigc-project && git push origin master:main
```

---

## ⚠️ 注意事项

1. **Conda 命令**: 在 bash 终端中需使用完整路径 `D:\anaconda3\Scripts\conda.exe`，或先执行 `conda init bash`
2. **显存限制**: 8GB VRAM，训练全精度大模型需使用 QLoRA、DeepSpeed ZeRO 等显存优化策略
3. **Compute Capability 8.9**: 支持 BF16、FlashAttention、torch.compile 等现代特性
4. **Python 环境优先级**: 所有 AI/深度学习项目优先使用 `GPUpytorch-env`
5. **Git 推送用 SSH**: 不要切回 HTTPS，否则又会断连
6. **HuggingFace 镜像必须**: 任何涉及 `transformers`/`diffusers` 的脚本必须在 import 之前设置 `HF_ENDPOINT`
