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

一个**从文字/草图到游戏资产**的 AI 管线，分五个阶段：

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ 阶段 1    │─→│ 阶段 2    │─→│ 阶段 3    │─→│ 阶段 4    │─→│ 阶段 5    │
│ LoRA 微调 │  │ Python   │  │ Unity    │  │ ControlNet│  │ TripoSR   │
│          │  │ API      │  │ 插件      │  │ 可控生成   │  │ 图片→3D   │
│ 48张原神图│  │ FastAPI  │  │ 输入文字  │  │ 草图/线稿  │  │ 上传图片  │
│ →训练LoRA│  │ →PNG图片 │  │ →生成图片 │  │ →AI精修   │  │ →3D模型  │
│ →原神风格│  │          │  │ →导入资产 │  │ →保持结构  │  │ →Unity    │
└──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

---

## 🎬 演示视频

| 阶段 | 视频 | 内容 |
|------|------|------|
| 阶段 1 | 📹 [1. LoRA 文生图](./1.Lora生图.mp4) | LoRA 风格微调 → FastAPI 文本生图 |
| 阶段 2+4 | 📹 [2. ControlNet 修图](./2.ControlNet修图.mp4) | Unity 插件 + 草图/线稿 → AI 精修 |
| 阶段 5 | 📹 [3. TripoSR 转 3D](./3.TripoSR转3d.mp4) | 图片去背景 → AI 生成 3D 模型 → Unity |

---

## 🏗️ 项目结构

```
aigc-project/
├── README.md                       ← 📖 本文件
├── CLAUDE.md                       ← AI 协作指南
├── ControlNet阶段总结.md            ← ControlNet 阶段总结
├── ControlNet 可控生成 — 实施计划.md ← ControlNet 原始计划
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
│   ├── main.py                     ← FastAPI 入口（/generate + /generate-controlled + /health）
│   ├── model_loader.py             ← SD + LoRA + ControlNet 加载器（全局单例）
│   ├── start.bat                   ← 🚀 双击启动脚本
│   ├── requirements.txt            ← Python 依赖
│   ├── install_controlnet.bat      ← ControlNet 依赖安装
│   └── outputs/                    ← 生成图片存档（含参考图/预处理图/生成图）
│
├── unity_plugin/
│   └── Assets/Editor/AIGCAssetGenerator/
│       ├── AIGCWindow.cs           ← Editor 窗口 UI（文生图 + 草图精修）
│       ├── AIGCClient.cs           ← HTTP 客户端（/generate + /generate-controlled）
│       ├── AssetImporter.cs        ← 资产导入
│       └── AIGCSettings.cs         ← Preferences 配置
│
├── train_lora.py                   ← LoRA 训练脚本
├── preprocess.py                   ← 图片裁剪预处理
├── caption.py                      ← WD14 ONNX 自动打标
├── inference_compare.py            ← LoRA 加载前后对比推理
│
├── 1.Lora生图.mp4                   ← 🎥 阶段1 演示：LoRA 文生图
├── 2.ControlNet修图.mp4             ← 🎥 阶段2+4 演示：ControlNet 草图精修
└── 3.TripoSR转3d.mp4                ← 🎥 阶段5 演示：图片转 3D 模型
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
| `/generate` | POST | 传入 prompt，返回 PNG 图片（纯文本生图） |
| `/generate-controlled` | POST | 上传参考图 + prompt，AI 保持结构精修（详见阶段 4） |
| `/generate-3d` | POST | 上传图片 → AI 生成 3D 模型（详见阶段 5） |
| `/health` | GET | 服务状态检测 |
| `/docs` | GET | Swagger 可视化文档（可手动测试） |

**`/generate` 请求参数**：

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
| `AIGCWindow.cs` | Editor 窗口 UI（文生图 + 草图精修双模式、输入框、预览、按钮） |
| `AIGCClient.cs` | HTTP 通信（`/generate` + `/generate-controlled` 接口） |
| `AssetImporter.cs` | 图片 → PNG → Unity 资产（自动命名、纹理导入配置） |
| `AIGCSettings.cs` | 全局配置（Preferences 面板，API 地址/默认参数） |

**设计亮点**：

- 打开窗口自动检测服务状态（绿色在线 / 红色离线）
- 三种资产类型自动追加风格关键词（图标 → `clean design, centered`，贴图 → `seamless, texture`）
- 生成异步不卡 Editor UI
- 导入自动配置 sRGB、FilterMode、WrapMode
- 重名自动编号（chest → chest_001 → chest_002）

---

### 阶段 4：ControlNet 可控生成

**目标**：在文本生图基础上，增加「草图/线稿 → AI 精修」的可控生成能力。

**三种控制方式**：

| 模式 | 用途 | 举例 |
|------|------|------|
| **Canny 线稿精修** | 清晰轮廓/线稿 → 保持结构上色 | 画出武器轮廓 → AI 生成精修版游戏图标 |
| **Scribble 草图生成** | 随手涂鸦 → 概念图 | 涂鸦建筑轮廓 → AI 生成完整场景图 |
| **Depth 深度保持** | 照片/3D图 → 保持前后空间关系 | 3D 白模截图 → AI 生成带材质的场景 |

**流程**：

```text
用户上传参考图 (Unity 拖入或 API 上传)
       ↓
服务端自动预处理（Canny 提取线稿 / Scribble / Depth 提取深度）
       ↓
ControlNet + 融合了 LoRA 的 UNet → 生成精修图
       ↓
返回 PNG + 自动存档（参考图、预处理图、生成图各一份）
```

**API 参数**（`POST /generate-controlled`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `image` | 必填 | 参考图文件（PNG/JPG） |
| `prompt` | 必填 | 英文描述 |
| `control_mode` | `"canny"` | 控制方式：`canny` / `scribble` / `depth` |
| `steps` | 25 | 推理步数 |
| `guidance_scale` | 7.5 | 引导强度 |
| `control_strength` | 0.8 | 控制力度（0.1~2.0） |
| `max_size` | 768 | 输入图最大边长（SD 1.5 原生 512，过大自动缩放） |
| `seed` | 随机 | 固定种子可复现 |

**设计亮点**：

- 与 txt2img 管线共享 UNet/VAE/TextEncoder，不重复占显存
- 三模式合计最大约 4.2GB（8GB 显存轻松跑）
- 智能按需下载（每个模式 ~725MB fp16），一次下载永久缓存，重启秒加载
- 输入图自动缩放（长边 > 768 等比缩小），避免 1024×1024 撑爆显存
- Unity 插件支持模式切换 + 参考图拖入 + 实时预览

> 📖 详细文档见 [ControlNet阶段总结.md](./ControlNet阶段总结.md)

---

### 阶段 5：图片转 3D 模型 

**目标**：上传一张物体/角色图片，AI 自动生成带贴图的 3D 模型（.glb），可直接导入 Unity 作为 Prefab。

**技术选型**：TripoSR（Stability AI 开源）—— 单阶段前馈模型，~725MB fp16 权重，8GB 显存可跑。

**流程**：

```text
用户上传参考图 (Unity 拖入或 API 上传)
       ↓
rembg 去背景（自动抠图）
       ↓
resize_foreground（裁剪至主体区域）
       ↓
TripoSR 推理 → 提取 3D mesh（Marching Cubes 分块处理）
       ↓
导出 .glb（贴图内嵌）→ 存档
       ↓
Unity：写入 Assets/ → ModelImporter → 自动创建 Prefab → Ping 到 Project 窗口
```

**API 接口**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `image` | 必填 | 物体/角色参考图 |
| `output_format` | `"glb"` | `glb`（Unity 原生支持）或 `obj` |
| `resolution` | 256 | Mesh 精度：128=快速预览, 256=标准, 512=高精度 |

**测试数据**（RTX 4060 Laptop, resolution=256）：

| 输入图 | 结果 |
|--------|------|
| 橘子照片 | ✅ 球形 + 大致颜色正确（有白斑和方块感） |
| 白色咖啡杯（白底） | ❌ rembg 抠图失败（白物+白底分不清） |
| 二次元角色 | ❌ 一坨（TripoSR 没见过风格化角色） |

**设计亮点**：

- 下载与缓存策略同阶段 4（HF 镜像优先 → 智能按需下载），模型缓存复用
- HF 权重键名自动转换（适配新版 ViT 架构）
- torchmcubes 用 CPU 兼容层替代（skimage），无需 Visual Studio 编译
- 渲染器分块处理（chunk_size=4096），8GB 显存安全
- rembg 去背景 + resize_foreground 裁剪，全自动预处理
- 中间结果自动存档（去背景图 + 预处理图 + 最终模型），方便排查问题

**当前局限**：

| 局限 | 原因 | 改善方向 |
|------|------|----------|
| 简单物体效果好，复杂角色崩 | TripoSR 训练数据为真实 3D 扫描（Objaverse） | 换用角色专用模型 |
| 背面/遮挡区域颜色不准 | 单视角输入，背面靠模型猜 | 多视角输入（TRELLIS） |
| 曲面有方块感 | Marching Cubes 网格分辨率限制 | 提高 resolution 或换 FlexiCubes |

> **核心价值**：整条「图片 → 3D → Unity Prefab」管线已打通。后续只需升级底模就可以改善质量，不用改架构。

---

## 🛠️ 技术栈

> 顶部 badges 为快速概览，下表补充各技术在管线中的具体角色：

| 层级 | 技术 | 在管线中的角色 |
|------|------|----------------|
| 训练与推理 | PyTorch 2.x + CUDA 12.8 | 全流程计算后端 |
| 基座模型 | Counterfeit-V2.5 + LoRA（PEFT, rank=16） | 动漫风格图像生成 |
| 可控生成 | ControlNet（Canny / Scribble / Depth） | 草图/线稿 → AI 精修，保持结构 |
| 3D 重建 | TripoSR（Stability AI） | 单图 → 带贴图 3D mesh（.glb） |
| 自动标注 | WD SwinV2 Tagger v3（ONNX Runtime） | 训练数据自动打标 |
| 推理服务 | FastAPI + Uvicorn | 本地 HTTP API（`/generate` 等接口） |
| 游戏引擎 | Unity 2022.3 LTS | Editor 插件 + 资产一键导入 |
| 硬件 | NVIDIA RTX 4060 Laptop（8 GB） | 本地实时推理 |

---

## 🎯 下一步

- [x] ~~图片转 3D 模型~~ ✅ 管线已打通（TripoSR），需升级底模改善质量
- [ ] 升级 3D 底模（TRELLIS / Unique3D，改善角色和背面质量）
- [ ] PBR 材质批量生成（游戏生产中实际表现）
- [ ] 训练自己的 ControlNet（用游戏素材风格）
- [ ] 批量生成 + 多种子变体（一次生成多张，挑最好的）
- [ ] 训练更多风格 LoRA（科幻、像素、卡通）
- [ ] 图片后期处理（背景去除、亮度调整）
- [ ] Web 前端（非 Unity 用户也能用）

---

## 📄 License

MIT

---

> 🤖 本项目在 [Claude Code](https://claude.ai/code) 辅助下完成，全流程对话式开发。
