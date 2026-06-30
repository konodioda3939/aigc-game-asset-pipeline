# 技术备忘 / TECHNICAL_NOTES

> 本文件是 [CLAUDE.md](../CLAUDE.md) 外迁的「踩坑手册」。**平时不进上下文，改到对应模块时再读对应章节。**
> 项目全貌 / API 接口 / 文件结构见 [README.md](../README.md)；硬件 / Python 环境 / Git 推送见 [../SYSTEM_INFO.md](../SYSTEM_INFO.md)。

**目录**

- [打标系统](#打标系统)
- [LoRA 训练与推理加载](#lora-训练与推理加载)
- [推理服务架构（inference_server）](#推理服务架构inference_server)
- [ControlNet 可控生成](#controlnet-可控生成)
- [TripoSR 图片转 3D](#triposr-图片转-3d)
- [PBR 材质 / StableMaterials](#pbr-材质--stablematerials)
- [工作流引擎（里程碑 7）](#工作流引擎里程碑-7)
- [ComfyUI 工作流（里程碑 8）](#comfyui-工作流里程碑-8)
- [网络注意事项](#网络注意事项)

---

## 打标系统

- 模型：`SmilingWolf/wd-swinv2-tagger-v3`，ONNX Runtime，CPU 推理
- 输入格式：NHWC，原始 [0, 255] 像素值（**不要归一化**，ONNX 模型已内嵌预处理）
- 阈值：置信度 0.35
- 国内网络需设 `HF_ENDPOINT = 'https://hf-mirror.com'`（必须在 import transformers 之前）

---

## LoRA 训练与推理加载

**训练配置**：
- 基座模型：`gsdf/Counterfeit-V2.5`
- 参数：rank=16, lr=1e-4, batch=1×梯度累积4=等效4, 1200步
- 目标模块：仅 Attention 投影层（to_q, to_k, to_v, to_out.0）
- 可训参数：3.19M / 862.71M（0.37%）

**训练坑**：
- **PEFT + fp16 注意**：`model.to(device)` 不带 dtype，否则 LoRA 层被意外转 fp16
- **必须同时使用** GradScaler + clip_grad_norm（防梯度爆炸）
- **LoRA 基座必须一致**：在哪个模型训练就在哪个模型推理

**推理加载（txt2img）**：
- PEFT 保存的 LoRA **不能**直接用 `pipe.load_lora_weights()` 加载
- 正确方式：`PeftModel.from_pretrained(pipe.unet, path)` → `merge_and_unload()`
- 融合后推理速度和不加 LoRA 一样快（LoRA 已被吸收进 UNet 原始权重）
- scheduler 使用 DPMSolverMultistepScheduler（比默认 DDPM 快 2-3 倍）

---

## 推理服务架构（inference_server）

- **全局单例**：SD + LoRA 只加载一次，所有请求复用 UNet/VAE/TextEncoder
- **懒加载**：ControlNet 和 TripoSR 模型按需加载，首次使用自动下载
- **设备**：CUDA（fp16）优先，CPU（fp32）回退
- **内存优化**：`attention_slicing` 已启用

**启动**：
```bash
# 方式1：双击（最简单）
inference_server/start.bat

# 方式2：命令行
cd d:\aigc-project\inference_server
D:/anaconda3/envs/GPUpytorch-env/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```
启动后访问 `http://127.0.0.1:8000/docs` 可视化测试。

**重要约定**：
- **Windows 启动**：用 `python -m uvicorn` 而非裸 `uvicorn`（后者不在 PATH）
- **print 必须加 flush=True**：否则启动日志被缓冲，用户看不到进度会以为卡了
- **代码修改需重启服务才能生效**：关掉窗口 → 重新双击 start.bat

---

## ControlNet 可控生成

**三种模式**：

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

---

## TripoSR 图片转 3D

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

## PBR 材质 / StableMaterials

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
|--------|--------|------|
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

## 工作流引擎（里程碑 7）

**核心理念**：将 AI 能力封装为标准化「产线」，美术人员不需要懂技术——选择工作流 → 输入描述 → 点生成。

> **与计划书的差异**：原定搭建 ComfyUI 工作流，但 ComfyUI（~10GB+）在 8GB 显存下无法与现有管线共存。实际落地为**轻量级工作流引擎**，复用 SD 1.5 + LoRA + ControlNet + TripoSR + StableMaterials 全局单例。零额外显存、零新模型下载。核心理念一致：标准化产线。

**4 条工作流**：

| 工作流 | 输入 | 输出 | 管线 |
|--------|------|------|------|
| `character_concept` 角色概念图 | 文字描述 | 1024×576 角色转身图（多角度单图） | txt2img + turnaround sheet prompt |
| `asset_generator` 游戏素材生成 | 文字 + 可选参考图 | 图标/场景/UI（三种风格切换） | txt2img 或 ControlNet（canny/scribble/depth） |
| `model_3d` 3D 模型生成 | 物体图片 | .obj/.glb 3D 模型 | TripoSR（rembg→推理→Marching Cubes） |
| `pbr_material` PBR 材质 | 材质描述 | ZIP 含 7 张 PBR 贴图 | StableMaterials |

**架构**：
- 零额外显存（复用 SD 1.5 + LoRA + ControlNet 全局单例）
- 零新模型下载（全部使用已有缓存）
- Prompt 模板从代码分离到 `prompts/*.json`（美术可自行调风格）
- Web UI（`/workflow-ui/`）+ Unity 插件 + API 三端统一

**API 接口**：

| 接口 | 方法 | 用途 |
|------|------|------|
| `/workflows` | GET | 列出所有工作流 + 输入参数 schema |
| `/workflows/run` | POST | 执行工作流（multipart/form-data） |
| `/workflow-ui/` | GET | Web 演示界面（静态页面） |

**文件结构**：
```
inference_server/
├── workflows/           ← 工作流编排（BaseWorkflow + 4 个子类）
│   ├── workflow_base.py        ← 基类（txt2img/img2img/ControlNet 封装 + 后处理）
│   ├── character_concept.py    ← 角色转身图（turnaround sheet prompt）
│   ├── asset_generator.py      ← 游戏素材生成（三种风格 × 两种模式）
│   ├── model_3d.py             ← 3D 模型（TripoSR 封装，含输入图自动缩放）
│   └── pbr_material.py         ← PBR 材质（StableMaterials 封装）
├── prompts/             ← Prompt 模板引擎 + 4 个 JSON 模板
├── web_ui/              ← Web 演示界面（index.html，单页应用）
├── main.py              ← + /workflows、/workflows/run、422 错误日志、StaticFiles mount
└── start_workflows.bat  ← 一键启动（含 Web UI 提示）
```

**Unity 集成**：
- `AIGCWindow.cs`：GenerationModeNames 新增「工作流」，动态表单（风格/ControlNet 模式选择器）
- `AIGCClient.cs`：新增 `RunWorkflow()` 方法（处理 PNG/ZIP/GLB/OBJ 四种响应）
- `AssetImporter.cs`：新增 `SaveWorkflowZip()` → PNG 自动导入
- `WorkflowPresets.cs`：4 个工作流预设定义（character_concept / asset_generator / model_3d / pbr_material）

**设计亮点**：
- 角色概念图用「character turnaround sheet」单次生成多角度，天然角色一致（不再用 img2img 链）
- 游戏素材三种风格（图标/场景/UI）统一入口，公共 ControlNet 管线
- 输入图自动缩放到 ≤768px（ControlNet）或 ≤1024px（TripoSR），防止显存爆炸
- API 422 错误自动记录表单字段详情，快速排查前端参数遗漏
- Web UI 生成结果留在弹窗内展示，不再自动关闭，用户看完手动关

---

## ComfyUI 工作流（里程碑 8）

**目标**：把已有的 4 条 Python 工作流迁移到 ComfyUI 节点图，通过标准 API 调用。面试可演示。

**与里程碑 7 的关系**：
- 里程碑 7 是轻量级 Python 引擎（复用全局单例，零新依赖）
- 里程碑 8 是标准 ComfyUI 工作流（行业标准工具，可视化节点图）
- 两套系统**互不干扰**：ComfyUI 端口 8188，FastAPI 端口 8000

**ComfyUI 安装信息**：
- 位置：`d:\aigc-project\ComfyUI\`
- 启动：双击 `ComfyUI/start_comfyui.bat` → 浏览器打开 `http://127.0.0.1:8188`
- 端口：8188（不与 FastAPI 8000 冲突）
- 启动参数：`--fp16-unet --fp16-vae --use-pytorch-cross-attention --reserve-vram 1.0`
- 环境变量：`TQDM_DISABLE=1`（避免后台运行进度条报错）

**模型路径**（全部复用已有缓存，不重复下载）：

| 模型 | ComfyUI 路径 | 来源 |
|------|-------------|------|
| Counterfeit-V2.5 | `models/diffusers/Counterfeit-V2.5/` | HF 缓存复制 |
| LoRA (原神风格) | `models/loras/CounterfeitGenshin-LoRA.safetensors` | PEFT 格式转换 |
| ControlNet ×3 | `models/controlnet/control_v11p_sd15_*.safetensors` | HF 缓存复制 |
| TripoSR | `models/checkpoints/TripoSR_model.ckpt` | HF 缓存复制 |

**4 套工作流 JSON**（位于 `ComfyUI/workflows/`）：

| 文件 | 工作流 | 核心节点 | 状态 |
|------|--------|----------|------|
| `character_concept.json` | 角色转身图 | DiffusersLoader → CLIPTextEncode → KSampler(1024×576, 30步) → VAEDecode | ✅ |
| `asset_icon_text.json` | 游戏图标(文字) | DiffusersLoader → KSampler(512×512, 25步) → SaveImage | ✅ |
| `model_3d.json` | 图片转3D | LoadImage → TripoSRModelLoader → TripoSRSampler → TripoSRViewer | ⚠️ 未测试 |
| `pbr_material.json` | PBR材质 | StableMaterials(自定义节点) → SaveImage×5 | ✅ |

**自定义节点**（位于 `ComfyUI/custom_nodes/`）：

| 节点 | 来源 | 用途 |
|------|------|------|
| `ComfyUI-Flowty-TripoSR` | flowtyone (GitHub SSH) | TripoSR 图片→3D mesh |
| `comfyui_remove_background` | d3cker (GitHub SSH) | rembg 去背景 |
| `ComfyUI-StableMaterials` | **自己编写** | StableMaterials PBR 材质生成 |
| `comfyui_controlnet_aux` | Fannovel16 (GitHub SSH) | ControlNet 预处理器(Canny/HED/Depth) |

**API 调用方式**：
```python
# 提交工作流
POST http://127.0.0.1:8188/prompt
Body: {"prompt": <workflow_json>, "client_id": "..."}
# 返回: {"prompt_id": "..."}

# 查询结果
GET http://127.0.0.1:8188/history/{prompt_id}

# 下载图片
GET http://127.0.0.1:8188/view?filename=...&type=output
```

**LoRA 转换**（`scripts/lora_convert.py`）：
- PEFT 格式键名 `base_model.model.xxx.lora_A.weight` → ComfyUI 格式 `lora_unet_xxx.lora_down.weight`
- 256 个权重键全部成功转换，输出 12.2MB
- 在 ComfyUI 中使用 `LoraLoader` 节点，strength 推荐 0.7-1.0

**已知问题**：
- `RemoveBackground` 节点输出 MASK 而非 IMAGE，素材图标工作流暂不含去背景步骤
- Flowty TripoSR 使用旧版 ViT 键名，HF 新版 checkpoint 可能需要键名转换（同 model_loader.py 的 `_remap_triposr_keys`）
- StableMaterials 首次加载需 30-60 秒（2-3GB 权重），后续调用秒级

---

## 网络注意事项

| 场景 | 配置 |
|------|------|
| HuggingFace 下载 | `HF_ENDPOINT = 'https://hf-mirror.com'`（必须在 import diffusers 之前） |
| HuggingFace 超时 | `HF_HUB_DOWNLOAD_TIMEOUT = '600'`（10 分钟） |
| GitHub 推送 | **用 SSH**（`git@github.com:konodioda3939/aigc-game-asset-pipeline.git`），不用 HTTPS |
| pip 安装 | 可用清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
