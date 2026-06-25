# CLAUDE.md — AIGC 项目（LoRA 风格微调）

## 项目概述

这是一个 AIGC（AI Generated Content）项目，核心目标是通过 LoRA 微调技术，让 Stable Diffusion 学会特定风格的图像生成。

**当前进度：**

| 里程碑 | 内容 | 状态 |
|--------|------|------|
| 里程碑1 | LoRA 风格微调（48张原神参考图 → 训练 → 12.2MB LoRA 权重） | ✅ |
| 里程碑2 | Python 推理 API（FastAPI 本地 HTTP 服务 + /generate 接口） | ✅ |

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

### 输出规范
- 所有回复使用中文
- 技术概念必须附带通俗解释
- 涉及文件操作时，说明「改了哪个文件」「改了什么」「为什么这样改」「对最终效果有什么影响」

---

## 项目技术栈

| 层级 | 技术 |
|------|------|
| 深度学习框架 | PyTorch + CUDA |
| 模型生态 | HuggingFace Diffusers + PEFT |
| 基座模型 | Counterfeit-V2.5（动漫专用 SD 1.5 微调） |
| 打标模型 | WD SwinV2 Tagger v3（ONNX Runtime） |
| Web 框架 | FastAPI + Uvicorn（推理服务） |
| Python 环境 | `D:/anaconda3/envs/GPUpytorch-env/python.exe` |
| GPU | NVIDIA GeForce RTX 4060 Laptop（8 GB 显存） |

## 项目文件结构

```
d:\aigc-project\
├── data/
│   ├── style_images/          ← 原始参考图（48张 jpg）
│   └── processed/             ← 512×512 裁切 png + 人工修正后的 .txt 标注
├── lora_output/               ← 训练产出（adapter_model.safetensors 12.2MB）
│   ├── adapter_model.safetensors  ← 🔑 核心产出：LoRA 权重文件
│   ├── adapter_config.json        ← LoRA 配置
│   ├── checkpoint-200~1200/       ← 训练中间保存点
│   └── comparison/                ← 里程碑1 加载前后对比图
├── inference_server/          ← 里程碑2：推理 API 服务
│   ├── main.py                ← FastAPI 入口（/generate + /health）
│   ├── model_loader.py        ← 加载 SD + LoRA（全局单例，只加载一次）
│   ├── start.bat              ← 🔑 双击启动脚本（小白专用）
│   ├── requirements.txt       ← 服务依赖清单
│   └── outputs/               ← 生成图片自动存档
├── preprocess.py              ← 图片裁剪预处理（中心裁切 512×512）
├── caption.py                 ← WD14 ONNX 自动打标
├── check_env.py               ← 环境检测（GPU、PyTorch、CUDA）
├── check_images.py            ← 图片质量检查
├── review_tags.py             ← 标注审查工具（打印所有标签）
├── fix_tags.py                ← 批量修正标注（全局增删标签）
├── train_lora.py              ← LoRA 训练脚本（核心）
├── inference_compare.py       ← 推理对比脚本（有无 LoRA 对比）
└── cache/                     ← VAE/CLIP 编码缓存 + HuggingFace 模型缓存
    └── hub/                   ← 下载的基座模型（~2GB）
```

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

### 推理加载
- PEFT 保存的 LoRA 不能直接用 `pipe.load_lora_weights()` 加载
- 正确方式：`PeftModel.from_pretrained(pipe.unet, path)` → `merge_and_unload()`
- 融合后推理速度和不加 LoRA 一样快（LoRA 已被吸收进 UNet 原始权重）
- scheduler 使用 DPMSolverMultistepScheduler（比默认 DDPM 快 2-3 倍）

### 推理服务
- 框架：FastAPI + Uvicorn
- 全局单例：管线只加载一次，所有请求复用
- 设备：CUDA（fp16）优先，CPU（fp32）回退
- 内存优化：`attention_slicing` 已启用（8GB 卡够用）
- **Windows 启动**：用 `python -m uvicorn` 而非裸 `uvicorn`（后者不在 PATH）
- **print 必须加 flush=True**：否则启动日志被缓冲，用户看不到进度会以为卡了
- FastAPI 自带 `/docs` 可视化接口页面，适合技术小白手动测试

---

## 常用操作

### 启动推理服务

**方式1：双击 start.bat（最简单）**
```
双击 d:\aigc-project\inference_server\start.bat
```

**方式2：命令行**
```bash
cd d:\aigc-project\inference_server
D:/anaconda3/envs/GPUpytorch-env/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

启动后访问：
- API 文档（可视化测试）：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

### 停止推理服务
- 关掉命令行窗口
- 或按 `Ctrl+C`

### 命令行测试生成
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "1girl, raiden shogun, purple hair, masterpiece"}' \
  -o output.png
```

### 运行 Python 脚本
```bash
cd d:\aigc-project
D:/anaconda3/envs/GPUpytorch-env/python.exe <脚本名>.py
```

### 检查环境
```bash
D:/anaconda3/envs/GPUpytorch-env/python.exe check_env.py
```

### 安装推理服务依赖（首次使用需执行一次）
```bash
cd d:\aigc-project\inference_server
D:/anaconda3/envs/GPUpytorch-env/python.exe -m pip install -r requirements.txt
```

---

> **核心理念**：用户不需要懂技术，用户只需要告诉你「想要什么效果」。把技术细节留给自己，把简单选择留给用户。
