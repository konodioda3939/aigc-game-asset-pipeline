# 🎮 AIGC Game Asset Pipeline

> **输入文字描述 → AI 自动生成游戏素材 → 一键导入 Unity 直接用**

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Unity](https://img.shields.io/badge/Unity-2022.3%20LTS-black?logo=unity)](https://unity.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.1.0-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Diffusers-Counterfeit--V2.5-yellow)](https://huggingface.co/gsdf/Counterfeit-V2.5)
[![GPU](https://img.shields.io/badge/GPU-RTX%204060%20Laptop-76B900?logo=nvidia)](https://www.nvidia.com/)

---

## 📖 这是什么？

一个**从文字到游戏资产**的 AI 管线，分三个阶段：

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  阶段 1          │ ──→ │  阶段 2           │ ──→ │  阶段 3          │
│  LoRA 风格微调   │     │  Python 推理 API  │     │  Unity 编辑器插件 │
│                 │     │                  │     │                 │
│  48张原神图      │     │  FastAPI 本地服务  │     │  窗口输入文字     │
│  → 训练 LoRA    │     │  接收 prompt      │     │  → AI 生成图片   │
│  → 学会原神风格  │     │  → 返回 PNG 图片  │     │  → 导入为资产    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

**一句话总结**：在 Unity 里打开窗口，输入「一把金色长剑」，点按钮，10 秒后图片出现在项目资源里，拖到场景就能用。

---

## 🎬 演示视频

📹 **[观看完整演示](./2026-06-25%2019-27-02.mp4)**（2 分钟录屏）

> 视频展示了从启动推理服务到资产出现在 Unity 场景中的完整流程。

---

## 🏗️ 项目结构

```
aigc-project/
├── README.md                       ← 📖 本文件
├── CLAUDE.md                       ← AI 协作指南
│
├── data/
│   ├── style_images/               ← 48张原始参考图（原神角色 AI 绘图）
│   └── processed/                  ← 512×512 裁切 + 人工修正标注
│
├── lora_output/
│   ├── adapter_model.safetensors   ← 🔑 LoRA 权重（12.2 MB）
│   ├── adapter_config.json         ← LoRA 配置（rank=16）
│   ├── checkpoint-200~1200/        ← 训练中间保存点
│   └── comparison/                 ← 有无 LoRA 对比图
│
├── inference_server/
│   ├── main.py                     ← FastAPI 入口（/generate + /health）
│   ├── model_loader.py             ← SD + LoRA 加载器（全局单例）
│   ├── start.bat                   ← 🚀 双击启动脚本
│   ├── requirements.txt            ← Python 依赖
│   └── outputs/                    ← 生成图片存档
│
├── unity_plugin/
│   └── Assets/Editor/AIGCAssetGenerator/
│       ├── AIGCWindow.cs           ← Editor 窗口 UI
│       ├── AIGCClient.cs           ← HTTP 客户端
│       ├── AssetImporter.cs        ← 资产导入
│       └── AIGCSettings.cs         ← Preferences 配置
│
├── train_lora.py                   ← LoRA 训练脚本
├── preprocess.py                   ← 图片裁剪预处理
├── caption.py                      ← WD14 ONNX 自动打标
├── inference_compare.py            ← LoRA 加载前后对比推理
│
└── 2026-06-25 19-27-02.mp4         ← 🎥 演示录屏
```

---

## 🚀 快速开始

### 环境要求

| 组件 | 要求 |
|------|------|
| Python | 3.10+ |
| CUDA | 12.x（8 GB 显存） |
| Unity | 2022.3 LTS |
| 操作系统 | Windows 10/11 |

### 三步跑通

**第 1 步：启动推理服务**

```bash
# 双击这个文件
inference_server/start.bat
```

看到 `服务已就绪 → http://127.0.0.1:8000` 表示启动成功。

**第 2 步：安装 Unity 插件**

将 `unity_plugin/Assets/Editor/AIGCAssetGenerator/` 文件夹复制到你的 Unity 项目的 `Assets/Editor/` 目录下。

**第 3 步：开始生成**

Unity 菜单栏 → `Tools` → `AI Asset Generator` → 输入描述 → 点生成 → 点导入。

---

## 🔬 阶段详情

### 阶段 1：LoRA 风格微调

**目标**：让 Stable Diffusion 学会画原神动漫风格。

**数据**：48 张原神角色 AI 绘图，覆盖雷电将军、芙宁娜、胡桃、纳西妲等角色。

**流程**：

```
Pinterest/Danbooru 收集
       ↓
48张原神风格图
       ↓ preprocess.py → 中心裁切 512×512
data/processed/
       ↓ caption.py → WD14 ONNX 自动打标 + 人工修正
data/processed/*.txt
       ↓ train_lora.py
Counterfeit-V2.5 + LoRA (rank=16, 1200步)
       ↓
adapter_model.safetensors (12.2 MB)
```

**训练配置**：

| 参数 | 值 |
|------|-----|
| 基座模型 | Counterfeit-V2.5（动漫专用 SD 1.5） |
| LoRA Rank | 16 |
| 学习率 | 1e-4 |
| 等效 Batch | 4（1 × 梯度累积） |
| 训练步数 | 1200 |
| 可训参数 | 3.19M / 862.71M（0.37%） |
| 目标模块 | to_q, to_k, to_v, to_out.0 |

**关键技术问题与解决**：

| 问题 | 解决 |
|------|------|
| PEFT + fp16 冲突 | `model.to(device)` 不带 dtype，保留 LoRA 为 fp32 |
| 梯度爆炸 | GradScaler + clip_grad_norm 双保险 |
| ONNX 打标全返回灰度 | 检查模型内嵌预处理，不做外部归一化 |

---

### 阶段 2：Python 推理 API

**目标**：搭建本地 HTTP 服务，接收 prompt 返回 AI 生成的图片。

**接口一览**：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/generate` | POST | 传入 prompt，返回 PNG 图片 |
| `/health` | GET | 服务状态检测 |
| `/docs` | GET | Swagger 可视化文档（可手动测试） |

**请求参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prompt` | 必填 | 英文描述，最长 1000 字符 |
| `steps` | 25 | 推理步数（10~100） |
| `guidance_scale` | 7.5 | 引导强度（1~20） |
| `seed` | 随机 | 固定种子可复现 |

**测试性能**（RTX 4060 Laptop, 512×512, 25 steps）：

| prompt | 耗时 | 大小 |
|--------|------|------|
| 雷电将军角色 | 4.1s | 460 KB |
| 游戏剑图标 | 3.9s | 482 KB |
| 芙宁娜角色 | 4.2s | 377 KB |

**设计亮点**：

- 模型全局单例，只加载一次，所有请求复用
- LoRA 通过 `merge_and_unload()` 融合进 UNet，推理速度与不加 LoRA 相同
- `start.bat` 双击启动，对非技术人员友好
- 所有 `print()` 加 `flush=True`，启动进度即时可见

---

### 阶段 3：Unity Editor 插件

**目标**：在 Unity 编辑器里一键生成并导入游戏资产。

**窗口界面**：

```
┌─────────────────────────────────────┐
│  AI 资产生成器                       │
│                                     │
│  ● 服务在线               [重新检测] │
│                                     │
│  画面描述 (Prompt)                   │
│  ┌───────────────────────────────┐  │
│  │ a golden sword icon, RPG style│  │
│  └───────────────────────────────┘  │
│                                     │
│  资产类型: [图标] [贴图] [UI元素]    │
│  ▶ 高级选项                          │
│                                     │
│  [          生成 (Generate)        ] │
│                                     │
│  生成结果预览                         │
│  ┌───────────────────────────────┐  │
│  │      (Texture2D 预览)          │  │
│  └───────────────────────────────┘  │
│                                     │
│  [导入到项目]    [打开输出文件夹]      │
└─────────────────────────────────────┘
```

**四个模块**：

| 文件 | 职责 |
|------|------|
| `AIGCWindow.cs` | Editor 窗口 UI（菜单入口、输入框、预览、按钮） |
| `AIGCClient.cs` | HTTP 通信（调用 Python 服务的 `/generate` 接口） |
| `AssetImporter.cs` | 图片 → PNG → Unity 资产（自动命名、纹理导入配置） |
| `AIGCSettings.cs` | 全局配置（Preferences 面板，API 地址/默认参数） |

**设计亮点**：

- 打开窗口自动检测服务状态（绿色在线 / 红色离线）
- 三种资产类型自动追加风格关键词（图标 → `clean design, centered`，贴图 → `seamless, texture`）
- 生成异步不卡 Editor UI
- 导入自动配置 sRGB、FilterMode、WrapMode
- 重名自动编号（chest → chest_001 → chest_002）

---

## 🛠️ 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 深度学习框架 | PyTorch 2.x + CUDA | 模型训练与推理 |
| 模型生态 | HuggingFace Diffusers + PEFT | SD 管线 + LoRA |
| 基座模型 | Counterfeit-V2.5 | 动漫专用 SD 1.5 微调 |
| 打标模型 | WD SwinV2 Tagger v3 (ONNX) | 图像自动标签 |
| 推理服务 | FastAPI + Uvicorn | HTTP API |
| 游戏引擎 | Unity 2022.3 LTS | Editor 插件 + 资产导入 |
| GPU | NVIDIA RTX 4060 Laptop (8GB) | 本地推理 |

---

## 🎯 下一步

- [ ] 批量生成（多种子变体，挑最好的）
- [ ] 训练更多风格 LoRA（科幻、像素、卡通）
- [ ] 图片后期处理（背景去除、亮度调整）
- [ ] ComfyUI 工作流集成
- [ ] Web 前端（非 Unity 用户也能用）

---

## 📄 License

MIT

---

> 🤖 本项目在 [Claude Code](https://claude.ai/code) 辅助下完成，全流程对话式开发。
