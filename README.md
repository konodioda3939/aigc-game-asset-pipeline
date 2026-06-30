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

一个**从文字/草图到游戏资产**的 AI 管线，分八个阶段：

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ 阶段 1    │─→│ 阶段 2    │─→│ 阶段 3    │─→│ 阶段 4    │─→│ 阶段 5    │─→│ 阶段 6    │─→│ 阶段 7    │─→│ 阶段 8    │
│ LoRA 微调 │  │ Python   │  │ Unity    │  │ ControlNet│  │ TripoSR   │  │ PBR 材质  │  │ 工作流引擎 │  │ ComfyUI  │
│          │  │ API      │  │ 插件      │  │ 可控生成   │  │ 图片→3D   │  │ 自动生成   │  │ 4条产线   │  │ 节点图   │
│ 48张原神图│  │ FastAPI  │  │ 输入文字  │  │ 草图/线稿  │  │ 上传图片  │  │ 输入描述   │  │ 角色/图标  │  │ 可视化   │
│ →训练LoRA│  │ →PNG图片 │  │ →生成图片 │  │ →AI精修   │  │ →3D模型  │  │ →完整材质  │  │ 场景/UI   │  │ API调用  │
│ →原神风格│  │          │  │ →导入资产 │  │ →保持结构  │  │ →Unity    │  │ →Unity.mat│  │ 一键切换   │  │ 标准工具 │
└──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

---

## 🎬 演示视频

| 阶段 | 视频 | 内容 |
|------|------|------|
| 阶段 1 | 📹 [1. LoRA 文生图](./1.Lora生图.mp4) | LoRA 风格微调 → FastAPI 文本生图 |
| 阶段 2+4 | 📹 [2. ControlNet 修图](./2.ControlNet修图.mp4) | Unity 插件 + 草图/线稿 → AI 精修 |
| 阶段 5 | 📹 [3. TripoSR 转 3D](./3.TripoSR转3d.mp4) | 图片去背景 → AI 生成 3D 模型 → Unity |
| 阶段 6 | 📹 [4. PBR 材质生成](./4.生成PBR材质.mp4) | prompt → StableMaterials 生成贴图 → Unity Material |
| 阶段 7 | 🌐 [Web UI](http://127.0.0.1:8000/workflow-ui/) | 4 条工作流 → 选产线 → 填描述 → 点生成 |
| 阶段 8 | 📹 [5. ComfyUI 工作流](./5.简易ComfyUI工作流.mp4) | 4 套 .json 节点图 → 可视化节点编排 → API 调用 |

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
├── comfyui_workflows/               ← ComfyUI 工作流文件
│   ├── workflows/                    ← 4 套 .json 节点图
│   ├── custom_nodes/                 ← 自定义节点（StableMaterials）
│   ├── start_comfyui.bat             ← ComfyUI 启动脚本
│   └── extra_model_paths.yaml        ← 模型路径配置
│
├── scripts/                          ← 辅助脚本
│   ├── lora_convert.py               ← PEFT→ComfyUI LoRA 格式转换
│   └── create_workflows.py           ← 工作流 JSON 生成器
│
├── 1.Lora生图.mp4                   ← 🎥 阶段1 演示
├── 2.ControlNet修图.mp4             ← 🎥 阶段2+4 演示
├── 3.TripoSR转3d.mp4                ← 🎥 阶段5 演示
├── 4.生成PBR材质.mp4                 ← 🎥 阶段6 演示
└── 5.简易ComfyUI工作流.mp4           ← 🎥 阶段8 演示
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
| `/generate-pbr` | POST | 输入材质描述 → AI 生成 4 张 PBR 贴图 ZIP（详见阶段 6） |
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
| 边界伪影（方壳） | TripoSR 三平面表示的立方体边界噪声 | 尚无有效解决方案，后续换底模 |

> **核心价值**：整条「图片 → 3D → Unity Prefab」管线已打通。后续只需升级底模就可以改善质量，不用改架构。

---

### 阶段 6：PBR 材质自动生成

**目标**：输入文字描述（如 "rough stone wall"），AI 自动生成完整的 PBR 材质贴图集，一键创建 Unity Standard Shader Material。

**技术选型**：StableMaterials（`gvecchio/StableMaterials`）—— 专用 PBR 材质生成管线，基于 MatFuse 架构。

**流程**：

```text
用户输入材质描述 (prompt)
       ↓
SD 管线卸载到 CPU（腾显存）
       ↓
StableMaterials LCM 推理（4 步，~5-10 秒）
       ↓
生成 5 张 PBR 贴图（BaseColor / Normal / Height / Roughness / Metallic）
       ↓
服务端打包：Metallic(R) + Smoothness(1-Roughness, A) → Unity _MetallicGlossMap
       ↓
所有贴图 ZIP → 返回
       ↓
Unity：解压 → 导入贴图 → 配置 TextureImporter → 创建 Triplanar .mat
```

**API 参数**（`POST /generate-pbr`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prompt` | 必填 | 材质描述（英文），如 `"rough stone wall"` |
| `tileable` | `true` | 无缝平铺（Feature Rolling） |
| `steps` | 25 | 推理步数（5~50） |
| `guidance_scale` | 10.0 | 引导强度（1~20） |
| `seed` | 随机 | 固定种子可复现 |

**返回**：`application/zip`，含 `basecolor.png`、`normal.png`、`metallic_smoothness.png`（R=金属度 A=光滑度）、`height.png`。

**Triplanar Shader**（`AIGC/TriplanarPBR`）：

PBR 材质默认使用 Triplanar（三平面）纹理投射，**无视模型的 UV 坐标**，从 X/Y/Z 三个世界方向投射纹理并根据法线自动混合。这意味着 **TripoSR 等 AI 生成的模型（UV 差/无 UV）直接丢上去就能正常显示材质**，不会因 UV 丢失而变成纯色。

可在 Material Inspector 的 Shader 下拉菜单中随时切换回 `Standard`。

**设计亮点**：

- StableMaterials 是独立架构（不依赖 SD 1.5），懒加载
- 标准 scheduler 25 步推理，512×512 输出（LCM scheduler 需配合专用 unet_lcm 权重，标准 UNet 直接用会输出纯色）
- Feature Rolling 生成无缝纹理（游戏地面/墙壁必备）
- 服务端用 PIL 将 Metallic + Smoothness 打包为 Unity Standard Shader 兼容的 RGBA 格式
- SD 管线与 PBR 管线自动卸载/恢复，8GB 显存安全共存
- 下载策略同 ControlNet/TripoSR：HF 镜像优先 → 直连回退
- Triplanar Shader 解决 AI 模型 UV 缺失问题，Cube 和复杂模型均正常显示

**当前局限**：

| 局限 | 说明 |
|------|------|
| 模型为独立架构 | 不与现有 SD/LoRA 共享组件，首次下载 ~2-3GB |
| 只擅长写实材质 | 训练数据为真实 PBR 材质（MatSynth），非风格化 |
| 复杂空间关系可能不准 | 对复杂图案细节的精度有限 |
| 单次生成一张材质 | 不支持批量（需多次调用） |

---

### 阶段 7：简易游戏美术工作流引擎

**四条产线**：

| 工作流 | 输入 | 输出 | 底层引擎 |
|--------|------|------|----------|
| 🎭 角色概念图 | 文字描述 | 1024×576 角色转身图（正面/侧面/背面/3/4） | SD+LoRA txt2img |
| 🎯 游戏素材生成 | 文字 / 参考图+ControlNet | 图标/场景/UI（三种风格一键切换） | SD+LoRA + ControlNet（canny/scribble/depth） |
| 📦 3D 模型生成 | 物体/道具图片 | .obj/.glb 3D 模型 | TripoSR（去背景 → 推理 → Marching Cubes） |
| 🧱 PBR 材质 | 材质描述 | ZIP 含 7 张贴图（颜色/法线/粗糙度/金属度） | StableMaterials |

**三种使用方式**：

| 方式 | 入口 | 说明 |
|------|------|------|
| 🌐 Web 界面 | `http://127.0.0.1:8000/workflow-ui/` | 双击 start_workflows.bat → 浏览器打开 → 4 张卡片 → 填表单 → 生成 |
| 🔌 Unity 插件 | `Tools → AI Asset Generator → 工作流` | 生成完一键导入项目 |
| 📡 API 调用 | `POST /workflows/run` | 程序化调用，可集成到任何工具 |

**各工作流的关键设计**：

| 工作流 | 关键策略 |
|--------|----------|
| 角色概念 | 「character turnaround sheet」prompt 模板，单次生成多角度，角色天然一致 |
| 游戏素材 | 三种风格后缀（icon/scene/ui）+ 可选 ControlNet 参考图精修；图标风格自动 rembg 去背景居中 |
| 3D 模型 | 输入图自动缩放到 ≤1024px；默认 OBJ 格式 Unity 原生导入；rembg 自动去背景 |
| PBR 材质 | 自动卸载 SD 腾显存 → StableMaterials 推理 → 打包 MetallicSmoothness → 恢复 SD |

**设计亮点**：

- **零额外显存**：复用已有管线全局单例，所有工作流用 `torch.no_grad()` + `torch.cuda.empty_cache()`
- **零新模型下载**：全部使用阶段 1-6 已缓存的模型
- **Prompt 模板分离**：JSON 文件可独立编辑，美术能自己调风格
- **三端统一**：Web UI + Unity + API 用同一套后端
- **输入图自动缩放**：大图自动缩到合理尺寸，避免显存爆炸和极慢推理
- **懒加载**：ControlNet/TripoSR/StableMaterials 按需加载，不浪费启动时间

---

## 🛠️ 技术栈

> 顶部 badges 为快速概览，下表补充各技术在管线中的具体角色：

| 层级 | 技术 | 在管线中的角色 |
|------|------|----------------|
| 训练与推理 | PyTorch 2.x + CUDA 12.8 | 全流程计算后端 |
| 基座模型 | Counterfeit-V2.5 + LoRA（PEFT, rank=16） | 动漫风格图像生成 |
| 可控生成 | ControlNet（Canny / Scribble / Depth） | 草图/线稿 → AI 精修，保持结构 |
| 3D 重建 | TripoSR（Stability AI） | 单图 → 带贴图 3D mesh（.glb） |
| PBR 材质 | StableMaterials（MatFuse + LCM） | prompt → 完整 PBR 贴图集（.mat） |
| 工作流引擎 | Python 工作流编排 + Prompt 模板引擎 | 4 条标准化游戏美术产线 |
| ComfyUI 工作流 | ComfyUI 0.26 + 自定义节点 + 4 套 .json 节点图 | 可视化节点图 + API 调用 |
| 自动标注 | WD SwinV2 Tagger v3（ONNX Runtime） | 训练数据自动打标 |
| 推理服务 | FastAPI + Uvicorn | 本地 HTTP API（`/generate` 等接口 + `/workflows/run`） |
| 游戏引擎 | Unity 2022.3 LTS | Editor 插件 + 工作流模式 + 资产一键导入 |
| 硬件 | NVIDIA RTX 4060 Laptop（8 GB） | 本地实时推理 |

---

### 阶段 8：ComfyUI 游戏美术工作流

**目标**：把 4 条产线做成 ComfyUI 标准节点图（.json），通过 API 调用。

**与阶段 7 的关系**：
- 阶段 7 是轻量级 Python 引擎（复用全局单例，零新依赖）
- 阶段 8 是标准 ComfyUI 工作流（行业标准工具，可视化节点图）
- 两套系统**互不干扰**：ComfyUI 端口 8188，FastAPI 端口 8000

**四条 ComfyUI 工作流**：

| 工作流 | 文件 | 输入 | 输出 | 核心节点 |
|--------|------|------|------|----------|
| 🎭 角色概念图 | `character_concept.json` | 文字描述 | 1024×576 角色转身图 | DiffusersLoader → KSampler(30步) → VAEDecode |
| 🎯 游戏素材 | `asset_icon_text.json` | 文字描述 | 512×512 游戏图标 | DiffusersLoader → KSampler(25步) → SaveImage |
| 📦 3D 模型 | `model_3d.json` | 物体照片 | .glb 3D 模型 + 网页预览 | TripoSRLoader → TripoSRSampler → MeshSave → Preview3D |
| 🧱 PBR 材质 | `pbr_material.json` | 材质描述 | 5 张 PBR 贴图 | StableMaterials(自定义节点) → SaveImage×5 |

**自定义节点**：

| 节点 | 来源 | 用途 |
|------|------|------|
| `ComfyUI-StableMaterials` | 结合ai工具编写 | StableMaterials PBR 材质生成 |
| `ComfyUI-Flowty-TripoSR` | flowtyone（已修复键名兼容） | TripoSR 图片转 3D mesh |
| `comfyui_controlnet_aux` | Fannovel16 | ControlNet 预处理器(Canny/HED/Depth) |
| `comfyui_remove_background` | d3cker | rembg 去背景 |

**使用方式**：
```bash
# 启动
双击 ComfyUI/start_comfyui.bat
浏览器打开 http://127.0.0.1:8188

# 使用工作流
把 ComfyUI/workflows/*.json 拖入浏览器窗口
改 prompt → 点 Queue Prompt

# API 调用
POST http://127.0.0.1:8188/prompt
Body: {"prompt": <workflow_json>}
```

---

## 🎯 下一步

- [x] ~~图片转 3D 模型~~ ✅ 管线已打通（TripoSR），需升级底模改善质量
- [x] ~~PBR 材质批量生成~~ ✅ 管线已打通（StableMaterials），prompt → 完整材质球
- [x] ~~游戏美术工作流引擎~~ ✅ 4 条标准化产线，零新模型依赖
- [x] ~~ComfyUI 游戏美术工作流~~ ✅ 4 套 .json 节点图 + API，
- [ ] 升级 3D 底模（TRELLIS / Unique3D，改善角色和背面质量）
- [ ] 训练自己的 ControlNet（用游戏素材风格）
- [ ] 批量生成 + 多种子变体（一次生成多张，挑最好的）
- [ ] 训练更多风格 LoRA（科幻、像素、卡通）
- [ ] 图片后期处理（背景去除、亮度调整）
- [ ] Web 前端（非 Unity 用户也能用）

---

## 📄 License

MIT

---

