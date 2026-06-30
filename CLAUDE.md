# CLAUDE.md — AIGC 游戏资产管线（AI 协作索引卡）

> **本文件是轻量索引**：只放高频命令、不可违反的铁律决策、用户协作约束。
> 庞大的技术细节**不常驻上下文**，按需查阅下方「文档地图」里的引用文件。

---

## 📚 文档地图（需要细节时再读，不要全读）

| 想了解 | 去看 |
|--------|------|
| 项目全貌 / 8 阶段详情 / API 参数 / 文件结构（人类视角） | [README.md](README.md) |
| 硬件 / Python 环境 / Git 推送流程 | [SYSTEM_INFO.md](SYSTEM_INFO.md) |
| **技术踩坑备忘**（改到某模块时，读对应章节） | [docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md) |

`docs/TECHNICAL_NOTES.md` 章节：打标系统 / LoRA 训练与加载 / 推理服务架构 / ControlNet / TripoSR / PBR·StableMaterials / 工作流引擎(里程碑7) / ComfyUI(里程碑8) / 网络注意。

---

## 项目一句话

从**文字/草图**到**游戏资产**的 AI 管线：`文字 → LoRA 生图 → Unity 导入 → ControlNet 草图精修 → TripoSR 图片转 3D → StableMaterials PBR 材质 → 工作流引擎 → ComfyUI 节点图`。

**8 个里程碑全部完成**（LoRA 微调 / 推理 API / Unity 插件 / ControlNet / TripoSR / PBR / 工作流引擎 / ComfyUI）。阶段详情与演示视频见 [README.md](README.md)。

---

## 🚨 用户协作约束（最高优先级，每次对话必读）

### 用户画像
- **用户是技术小白**，不懂编程、不懂 AI 术语、不懂命令行
- 用户的母语是中文，所有沟通使用中文
- 用户的需求描述是「日常用语」，不是「技术规格」

### 交互原则

1. **以产品经理思维理解需求**：用户说「我想让图更好看」→ 你自己翻译成技术语言，给出 2-3 个具体方案让用户选，不要追问技术细节
2. **永远不要假设用户懂技术**：不要问「用 AdamW 还是 SGD？」——应问「你更看重训练速度快，还是最终效果更精细？」
3. **需求模糊时主动补全**：根据上下文推断，然后确认
4. **每次操作前用大白话解释**：说明「改了哪个文件 / 改了什么 / 为什么 / 对最终效果的影响」
5. **给出选项而非开放式问题**：每个选项用大白话说清优缺点
6. **不自动推送 GitHub**：修改代码后只 commit 不 push，等用户审查确认功能正常后再推送

### 输出规范

- 所有回复使用中文；技术概念必须附带通俗解释

---

## ⚡ 常用命令

### 启动推理服务（FastAPI，端口 8000）

```bash
# 方式1：双击（最简单）
inference_server/start.bat

# 方式2：命令行
cd d:\aigc-project\inference_server
D:/anaconda3/envs/GPUpytorch-env/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```
启动后 `http://127.0.0.1:8000/docs` 可视化测试。**代码改了必须重启服务才生效**；`print()` 必须加 `flush=True`，否则日志被缓冲用户会以为卡住。

### Python 环境（所有 AI 脚本只用这一个）

```bash
D:/anaconda3/envs/GPUpytorch-env/python.exe <脚本>.py        # Python 3.11 + PyTorch 2.11 + CUDA 12.8
D:/anaconda3/envs/GPUpytorch-env/python.exe -m pip install <包名>   # 装包（可用清华源）
```

### 训练 / 预处理脚本

`train_lora.py`（LoRA 训练）、`preprocess.py`（图片裁切）、`caption.py`（WD14 打标）、`check_env.py`（环境检测）——都用上面的 Python 跑。

### Git 推送（必须 SSH，本地 master → 远程 main）

```bash
cd d:/aigc-project && git push origin master:main
```
> ⚠️ 用 SSH 不用 HTTPS（国内 HTTPS 直连 GitHub 会断连）。完整 Git/SSH 信息见 [SYSTEM_INFO.md](SYSTEM_INFO.md)。

---

## 🔒 铁律架构决策（不可违反，违反必踩坑）

1. **8GB 显存是天花板**：SD 与 StableMaterials/PBR 不能同时驻留显存，靠卸载/恢复（`_offload_sd_pipeline` / `_restore_sd_pipeline`）共存；大图自动缩放（ControlNet 输入 ≤768，TripoSR ≤1024）。
2. **LoRA 加载必须 `merge_and_unload()`**：PEFT 保存的 LoRA 不能用 `pipe.load_lora_weights()`，要用 `PeftModel.from_pretrained(pipe.unet, path)` → `merge_and_unload()`。
3. **`HF_ENDPOINT='https://hf-mirror.com'` 必须在 `import transformers/diffusers` 之前设置**（国内网络，否则下载超时）。
4. **模型全局单例 + 懒加载**：SD+LoRA 只加载一次，所有请求复用；ControlNet / TripoSR / StableMaterials 按需首次加载。
5. **torchmcubes 用 CPU 兼容层**（`inference_server/torchmcubes.py`，基于 skimage），不编译 C++ 扩展；`model_loader.py` 把 `inference_server/` 优先加入 `sys.path`。
6. **AI 生成的 3D 模型默认用 `AIGC/TriplanarPBR` shader**（三平面投射，无视 UV），不是 Standard——否则 UV 缺失会变纯色。
7. **里程碑 7 用轻量 Python 引擎而非 ComfyUI**：ComfyUI(~10GB) 在 8GB 显存下无法与主管线共存，故工作流引擎复用全局单例（零新显存、零新模型）。
8. **GitHub 推送用 SSH 不用 HTTPS**。

> 各决策的实现细节、参数、回退策略见 [docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md) 对应章节。

---

## 📊 当前进度与已知短板

**进度**：8/8 里程碑全部完成（详见 [README.md](README.md)）。

**已知短板**（模型本身局限，非 bug；改对应模块时务必记得）：
- **TripoSR**：只擅长真实物体，二次元角色会崩；有边界「方壳」伪影（三平面表示噪声，暂无根治，待换底模）；白物+白底抠图可能失败。
- **PBR / StableMaterials**：只擅长写实材质（训练数据 MatSynth），非风格化；独立架构不与 SD 共享组件。
- **ControlNet**：SD 1.5 原生 512×512，超过 768 自动缩放（8GB 限制）。
- **ComfyUI**：StableMaterials 首次加载慢（30-60s，下载 2-3GB 权重）；Flowty TripoSR 用旧版 ViT 键名，HF 新版 checkpoint 可能需键名转换。
- **显存互斥**：SD 与 PBR 不能同时加载（8GB 限制，按需卸载/恢复）。

---

> **核心理念**：用户不需要懂技术，用户只需要告诉你「想要什么效果」。把技术细节留给自己，把简单选择留给用户。
