# CLAUDE.md — AIGC 游戏资产管线

## 项目概述

从**文字/草图**到**游戏资产**的 AI 管线，五个阶段全部打通：

```
文字描述 → LoRA 生图 → Unity 一键导入 → ControlNet 草图精修 → TripoSR 图片转 3D
```

**当前进度：**

| 里程碑 | 内容 | 状态 |
|--------|------|------|
| 里程碑1 | LoRA 风格微调（48张原神参考图 → 训练 → 12.2MB LoRA 权重） | ✅ |
| 里程碑2 | Python 推理 API（FastAPI 本地 HTTP 服务 + /generate 接口） | ✅ |
| 里程碑3 | Unity Editor 插件（文生图 + 草图精修双模式、状态检测、一键导入） | ✅ |
| 里程碑4 | ControlNet 可控生成（Canny 线稿 / Scribble 草图 / Depth 深度） | ✅ |
| 里程碑5 | TripoSR 图片转 3D 模型（去背景 → 推理 → .glb → Unity Prefab） | ✅ |
| 里程碑6 | StableMaterials PBR 材质生成（prompt → BaseColor/Normal/Roughness/Metallic → Unity Material） | ✅ |

**已知局限：**

| 问题 | 说明 |
|------|------|
| TripoSR 边界伪影（方壳） | 三平面表示的立方体边界噪声，尚无有效代码解决方案，后续换底模 |
| TripoSR 只擅长真实物体 | 训练数据为 Objaverse（真实 3D 扫描），二次元角色会崩 |
| ControlNet 大图会慢 | SD 1.5 原生 512×512，超过 768 会自动缩放（8GB 显存限制） |
| PBR 只擅长写实材质 | 训练数据为 MatSynth（真实 PBR），非风格化/动漫材质效果差 |
| PBR 与 SD 不能同时加载 | 8GB 显存限制，自动卸载/恢复策略透明处理 |

---

## 用户约束（极其重要）

### 用户画像
- **用户是技术小白**，不懂编程、不懂 AI 术语、不懂命令行
- 用户的母语是中文，所有沟通使用中文
- 用户的需求描述是「日常用语」，不是「技术规格」

### 交互原则
1. **以产品经理思维理解需求**：用户说「我想让图更好看」→ 你要自己翻译成技术语言（可能是调整 CFG scale、换基座模型、增加训练步数、调整分辨率等），然后给出 2-3 个具体方案让用户选，不要追问技术细节
2. **永远不要假设用户懂技术**：不要问「你想用哪个 optimizer？AdamW 还是 SGD？」—— 这类问题对用户没有意义。你应该问：「你更看重训练速度快，还是最终效果更精细？」
3. **需求模糊时主动补全**：用户说「帮我改一下那个脚本」，你要根据上下文推断是哪个脚本、可能要改什么，然后确认：「你是想改训练脚本里的学习率吗？还是想调整打标脚本的阈值？」
4. **每次操作前用大白话解释**：
   - ❌ 「我会调整 `train_lora.py` 里的 `learning_rate` 从 1e-4 改到 5e-5」
   - ✅ 「我会把训练参数里的『学习速度』调慢一半，这样模型学得更稳，但需要多花一点时间」
5. **给出选项而非开放式问题**：当需要用户做选择时，给出 2-3 个具体选项，每个选项说明优缺点，用大白话描述
6. **不自动推送 GitHub**：修改代码后只 commit 不 push，等用户审查确认功能正常后再推送

### 输出规范
- 所有回复使用中文
- 技术概念必须附带通俗解释
- 涉及文件操作时，说明「改了哪个文件」「改了什么」「为什么这样改」「对最终效果有什么影响」

---

## 项目技术栈

| 层级 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.11 + CUDA 12.8 |
| 基座模型 | Counterfeit-V2.5（动漫专用 SD 1.5） + LoRA（PEFT, rank=16） |
| 可控生成 | ControlNet v1.0（Canny / Scribble / Depth），权重 ~1.4GB/模式 |
| 3D 重建 | TripoSR（Stability AI），单图 → 带贴图 3D mesh（.glb），权重 ~1.68GB |
| 打标模型 | WD SwinV2 Tagger v3（ONNX Runtime，CPU 推理） |
| 推理服务 | FastAPI + Uvicorn（本地 HTTP，全局单例加载） |
| 游戏引擎 | Unity 2022.3 LTS（Editor 插件 + 资产一键导入） |
| Python 环境 | `D:/anaconda3/envs/GPUpytorch-env/python.exe`（Python 3.11） |
| GPU | NVIDIA GeForce RTX 4060 Laptop（8 GB 显存） |

---

## 项目文件结构

```
d:\aigc-project\
├── README.md                       ← 📖 项目文档（给人类看）
├── CLAUDE.md                       ← AI 协作指南（给 AI 看）
├── SYSTEM_INFO.md                  ← 硬件/环境/Git 配置
│
├── data/
│   ├── style_images/               ← 48张原始参考图（原神角色 AI 绘图）
│   └── processed/                  ← 512×512 裁切 + 人工修正标注
│
├── lora_output/
│   ├── adapter_model.safetensors   ← 🔑 LoRA 权重（12.2 MB）
│   ├── adapter_config.json         ← LoRA 配置（rank=16）
│   └── checkpoint-200~1200/        ← 训练中间保存点
│
├── inference_server/               ← 🔑 推理服务（核心）
│   ├── main.py                     ← FastAPI 入口（3 个接口 + 健康检查）
│   ├── model_loader.py             ← 模型加载器（SD+LoRA+ControlNet+TripoSR，全局单例）
│   ├── start.bat                   ← 🚀 双击启动脚本
│   ├── requirements.txt            ← Python 依赖
│   ├── install_controlnet.bat      ← ControlNet 额外依赖安装
│   ├── install_triposr.bat         ← TripoSR 额外依赖安装
│   ├── torchmcubes.py              ← torchmcubes CPU 兼容层
│   ├── torchmcubes_compat.py       ← torchmcubes 兼容层（旧版）
│   └── outputs/                    ← 生成图片/3D 模型存档
│
├── unity_plugin/                   ← Unity Editor 插件
│   └── Assets/Editor/AIGCAssetGenerator/
│       ├── AIGCWindow.cs           ← Editor 窗口 UI（双模式 + 预览 + 导入）
│       ├── AIGCClient.cs           ← HTTP 客户端（/generate + /generate-controlled + /generate-3d）
│       ├── AssetImporter.cs        ← 图片/3D 资产自动导入
│       └── AIGCSettings.cs         ← Preferences 全局配置
│
├── TripoSR/                        ← TripoSR 源码（克隆自 Stability AI）
│   ├── tsr/system.py               ← TSR 模型类 + extract_mesh
│   ├── tsr/utils.py                ← remove_background + resize_foreground
│   └── tsr/models/                 ← NeRF 渲染器 + Transformer + IsoSurface
│
├── torchmcubes/                    ← torchmcubes 源码（克隆，CPU 兼容）
│
├── train_lora.py                   ← LoRA 训练脚本
├── preprocess.py                   ← 图片裁剪预处理（中心裁切 512×512）
├── caption.py                      ← WD14 ONNX 自动打标
├── inference_compare.py            ← LoRA 加载前后对比推理
├── check_env.py                    ← 环境检测（GPU、PyTorch、CUDA）
├── check_images.py                 ← 图片质量检查
├── review_tags.py                  ← 标注审查（打印所有标签）
├── fix_tags.py                     ← 批量修正标注（全局增删标签）
│
├── 1.Lora生图.mp4                  ← 🎥 阶段1 演示
├── 2.ControlNet修图.mp4            ← 🎥 阶段2+4 演示
├── 3.TripoSR转3d.mp4               ← 🎥 阶段5 演示
│
└── cache/hub/                      ← 模型缓存（~8GB，不上传 Git）
    ├── models--gsdf--Counterfeit-V2.5/   ← 基座模型（~2GB）
    └── controlnet/                       ← ControlNet 权重（每模式 ~1.4GB）
```

---

## API 接口一览

| 接口 | 方法 | 用途 |
|------|------|------|
| `/generate` | POST | 纯文本生图（prompt → PNG），向后兼容 |
| `/generate-controlled` | POST | ControlNet 可控生成（参考图 + prompt → 精修图） |
| `/generate-3d` | POST | TripoSR 图片转 3D 模型（图片 → .glb） |
| `/generate-pbr` | POST | StableMaterials PBR 材质生成（prompt → ZIP 含 4 张贴图） |
| `/health` | GET | 服务状态 + 已加载模型信息 |
| `/docs` | GET | Swagger 可视化文档（可手动测试） |

详见 README.md 中各阶段的参数说明。

---

## 关键技术细节（供 AI 参考）

### 打标系统
- 模型：`SmilingWolf/wd-swinv2-tagger-v3`，ONNX Runtime，CPU 推理
- 输入格式：NHWC，原始 [0, 255] 像素值（**不要归一化**，ONNX 模型已内嵌预处理）
- 阈值：置信度 0.35
- 国内网络需设 `HF_ENDPOINT = 'https://hf-mirror.com'`（必须在 import transformers 之前）

### LoRA 训练
- 基座模型：`gsdf/Counterfeit-V2.5`
- 配置：rank=16, lr=1e-4, batch=1×梯度累积4=等效4, 1200步
- 目标模块：仅 Attention 投影层（to_q, to_k, to_v, to_out.0）
- **PEFT + fp16 注意**：`model.to(device)` 不带 dtype，否则 LoRA 层被意外转 fp16
- **必须同时使用** GradScaler + clip_grad_norm
- **LoRA 基座必须一致**：在哪个模型训练就在哪个模型推理

### 推理加载（txt2img）
- PEFT 保存的 LoRA 不能直接用 `pipe.load_lora_weights()` 加载
- 正确方式：`PeftModel.from_pretrained(pipe.unet, path)` → `merge_and_unload()`
- 融合后推理速度和不加 LoRA 一样快（LoRA 已被吸收进 UNet 原始权重）
- scheduler 使用 DPMSolverMultistepScheduler（比默认 DDPM 快 2-3 倍）

### ControlNet 可控生成

**三种模式：**

| 模式 | 预处理 | 用途 |
|------|--------|------|
| `canny` | OpenCV Canny 边缘检测 | 清晰轮廓/线稿 → 保持结构上色 |
| `scribble` | HED 检测器（controlnet_aux） | 随手涂鸦 → 概念图 |
| `depth` | MiDaS 深度估计（controlnet_aux） | 照片/3D图 → 保持空间关系 |

**模型下载策略**（`model_loader.py`）：
- **优先级**：HuggingFace 镜像（hf-mirror.com）→ HuggingFace 直连 → ModelScope 兜底
- **智能按需下载**：只下载 config.json + 一个权重文件（优先 fp16 ~725MB，回退 fp32 ~1.45GB），不全仓库下载
- **404 不重试**：`_download_single_file` 对 404 立即抛出，上层秒级回退到 fp32
- **缓存复用**：下载到 `cache/hub/controlnet/<repo-id>/`，重启跳过下载
- **管线共享**：ControlNet 与 txt2img 共享 UNet/VAE/TextEncoder，不重复占显存

**输入图自动缩放**：
- SD 1.5 原生 512×512，超过 768 后速度暴跌（8GB 显存）
- `_resize_for_controlnet()` 自动将长边 > 768 的图等比缩小，尺寸对齐到 8 的倍数
- API 参数 `max_size` 可覆盖（默认 768）

**scribble 预处理注意**：
- HED 检测器可能改变输出尺寸，代码有对齐逻辑（`main.py` 第 2.5 步）
- controlnet_aux 首次使用会下载 HED/MiDaS 模型，有 mediapipe/timm 警告（可忽略）

### TripoSR 图片转 3D

**预处理流程**（`main.py` `/generate-3d`）：
```
上传图片 → remove_background (rembg 抠图)
         → resize_foreground (ratio=0.85，裁剪主体并居中)
         → 转 RGB（灰色背景合成）
         → 存入 TripoSR 模型
```

**推理与提取**：
```
TSR 前向推理 → scene_codes (triplane 特征)
             → extract_mesh(scene_codes, resolution=256, threshold=25.0)
             → Marching Cubes 提取等值面
             → trimesh 导出 .glb（贴图内嵌）
```

**显存优化**：
- 渲染器分块：`model.renderer.set_chunk_size(4096)`，避免 256³ 的 16M 点同时查询
- OOM 自动降级：推理/提取失败时 resolution 减半重试
- 提取后 `del scene_codes` + `torch.cuda.empty_cache()`

**torchmcubes 兼容**：
- `torchmcubes/` 是克隆的 C++ 扩展源码，需要 Visual Studio 编译
- `inference_server/torchmcubes.py` 是纯 Python CPU 兼容层（基于 skimage），无需编译
- `inference_server/torchmcubes_compat.py` 是旧版兼容层，保留以防回退
- `model_loader.py` 将 `inference_server/` 加入 `sys.path` 优先于 `torchmcubes/`

**已知局限**：
- **边界伪影（方壳）**：三平面立方体边界噪声，密度阈值/连通分量/壳面检测均无法根治，后续换底模解决
- 只擅长真实物体（训练数据 Objaverse），二次元角色会崩
- rembg 抠图在白色物体+白色背景时可能失败

---

## 推理服务（inference_server/）

### 架构
- **全局单例**：SD + LoRA 只加载一次，所有请求复用 UNet/VAE/TextEncoder
- **懒加载**：ControlNet 和 TripoSR 模型按需加载，首次使用自动下载
- **设备**：CUDA（fp16）优先，CPU（fp32）回退
- **内存优化**：`attention_slicing` 已启用

### 启动
```bash
# 方式1：双击（最简单）
inference_server/start.bat

# 方式2：命令行
cd d:\aigc-project\inference_server
D:/anaconda3/envs/GPUpytorch-env/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

启动后访问 `http://127.0.0.1:8000/docs` 可视化测试。

### 重要约定
- **Windows 启动**：用 `python -m uvicorn` 而非裸 `uvicorn`（后者不在 PATH）
- **print 必须加 flush=True**：否则启动日志被缓冲，用户看不到进度会以为卡了
- **代码修改需重启服务才能生效**：关掉窗口 → 重新双击 start.bat

---

### PBR 材质生成（StableMaterials）

**模型信息**：
- 模型 ID：`gvecchio/StableMaterials`，基于 MatFuse 架构（改编自 LDM），独立于 SD 1.5
- 输出 5 张 512×512 PBR 贴图：BaseColor、Normal、Height、Roughness、Metallic
- 推理：标准 scheduler 25 步（~15-25 秒），支持 Feature Rolling 无缝纹理
- **注意**：LCM Scheduler 必须配合 `unet_lcm` 权重使用，直接套用标准 UNet 会崩溃（纯色输出）
- 权重：~2-3GB（fp16），通过 `trust_remote_code=True` 加载自定义 diffusers pipeline
- 加载策略：HF 镜像优先 → HF 直连回退，缓存到 `cache/hub/pbr/`

**VRAM 管理**：
- 8GB 无法同时容纳 SD 1.5 + StableMaterials
- 策略：PBR 调用时自动将 SD+ControlNet 卸载到 CPU（`_offload_sd_pipeline()`）
- 推理完成后恢复 SD 到 GPU（`_restore_sd_pipeline()`）
- 使用 `threading.Lock` 防止并发访问冲突
- **注意**：fp16 pipeline 移到 CPU 会有警告（可忽略，只用于释放显存，不在 CPU 上推理）

**纹理打包**（服务端 PIL/numpy）：
- `metallic_smoothness.png`：R 通道 = Metallic，A 通道 = Smoothness（1 - Roughness）
- 此为 Unity Standard Shader `_MetallicGlossMap` 的预期格式
- ZIP 中额外附带原始 `roughness_raw.png` 和 `metallic_raw.png` 供调试

**API 参数**（`POST /generate-pbr`）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prompt` | 必填 | 材质描述（英文） |
| `tileable` | `true` | 无缝平铺（Feature Rolling） |
| `steps` | 25 | 推理步数（5-50） |
| `guidance_scale` | 10.0 | 引导强度 |

**Triplanar Shader（`AIGC/TriplanarPBR`）**：
- 世界坐标三平面纹理投射，**完全无视模型 UV**
- AI 生成的 3D 模型（TripoSR 等）UV 坐标差/无 UV 直接可用
- 从 X/Y/Z 三个轴投射纹理，根据表面法线加权混合
- 支持 BaseColor / Normal / MetallicSmoothness（与 StableMaterials 输出对应）
- 关键实现细节：
  - 用 `INTERNAL_DATA` + `WorldNormalVector()` 获取 TBN 矩阵（不能用自定义 vert，会破坏 TBN 传递）
  - Triplanar 法线必须从世界空间转回**切线空间**再赋值 `o.Normal`（否则光照全黑）
  - 法线混合时用顶点法线的符号修正投影方向

**Unity 集成**：
- `AssetImporter.SaveAsPBRMaterial()` 为每种贴图类型配置正确的 TextureImporter
- BaseColor → sRGB Default；Normal → sRGB Off NormalMap；MetallicSmoothness → sRGB Off
- **默认使用 `AIGC/TriplanarPBR` shader**（而非 Standard），确保所有模型都正常显示
- 回退：Triplanar shader 未找到时自动回退到 Standard

**已知局限**：
- 只擅长写实材质（训练数据 MatSynth ~6,198 种 PBR），非风格化
- 复杂空间关系/图案精度有限
- 独立架构不与现有 SD/LoRA 共享组件
- Triplanar 法线混合在极端锐角处可能有轻微不连续

---

## 网络注意事项

| 场景 | 配置 |
|------|------|
| HuggingFace 下载 | `HF_ENDPOINT = 'https://hf-mirror.com'`（必须在 import diffusers 之前） |
| HuggingFace 超时 | `HF_HUB_DOWNLOAD_TIMEOUT = '600'`（10 分钟） |
| GitHub 推送 | **用 SSH**（`git@github.com:konodioda3939/aigc-game-asset-pipeline.git`），不用 HTTPS |
| pip 安装 | 可用清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |

---

## 常用操作

### Python 环境
```bash
# 所有脚本都用这个 Python
D:/anaconda3/envs/GPUpytorch-env/python.exe <脚本名>.py

# 安装包
D:/anaconda3/envs/GPUpytorch-env/python.exe -m pip install <包名>
```

### Git
```bash
cd d:/aigc-project

# 查看状态
git status

# 提交（不推送）
git add <文件>
git commit -m "描述"

# 推送（用户确认后）
git push origin master:main
```

### 检查环境
```bash
D:/anaconda3/envs/GPUpytorch-env/python.exe check_env.py
```

---

> **核心理念**：用户不需要懂技术，用户只需要告诉你「想要什么效果」。把技术细节留给自己，把简单选择留给用户。
