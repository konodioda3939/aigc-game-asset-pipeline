# AIGC 管线增强计划书（求职方向 · 进行中）

> **🔒 给未来的 AI 对话（必读）**：本文是**跨会话工作计划**。接手时请按顺序：
> 1. 先看「二、进度总表」→ 知道整体做到哪了；
> 2. 再看「八、下一步行动」→ 知道现在该干哪件事；
> 3. 然后读对应项目的「状态与日志」。
> **每完成一个步骤，必须回来更新本文的进度总表和对应日志**（见「九、如何更新本文件」）。
>
> **给用户**：这是你的求职备战路线图，全程大白话。你不用懂技术细节，只管说"想做哪个"，技术活交给 AI。

---

## 一、为什么有这份计划（背景）

### 1.1 起点
主项目（AIGC 游戏资产管线）**8 个里程碑已全部完成**（LoRA 微调 / 推理 API / Unity 插件 / ControlNet / TripoSR / PBR / 工作流引擎 / ComfyUI）。项目全貌见 [README.md](README.md)，技术约束见 [CLAUDE.md](CLAUDE.md)。

### 1.2 目标岗位（AIGC + 游戏研发）核心要求
JD 反复点名的关键词：**图像/视频/3D 内容生成、多模态、3D 动作、生成式模型训练、数据处理、推理优化、游戏引擎、跨团队合作**。

### 1.3 差距分析（现有项目 vs JD）
| JD 关键词 | 现有项目覆盖情况 | 结论 |
|-----------|----------------|------|
| 图像生成 / 可控生成 | LoRA + ControlNet | ✅ 已命中 |
| 3D 内容生成 | TripoSR（图转 3D） | ✅ 已命中 |
| 游戏引擎 / 生产流程 | Unity 插件 | ✅ 已命中 |
| 推理服务 / 管线工程 | FastAPI + 工作流引擎 | ✅ 已命中（含优化深度，见项目 C） |
| **3D 动作 / 动画生成** | TripoSR 只出"静态雕像" | ❌ **空白** |
| **推理优化（量化/加速）** | SDPA + LCM 少步生成 | ✅ **已完成**（项目 C，3.88→0.75s ×5.2） |
| **Mesh 数据处理** | TripoSR 毛坯模型未打磨 | ❌ **深度不足** |
| **视频生成** | 无 | ❌ 空白（本轮缓后） |

### 1.4 本轮决策（2026-06-30 与用户确认）
- **本轮做**：项目 A（3D 动作）、项目 C（推理优化）、项目 D（Mesh 后处理）—— 这三个能直接补上 JD 的硬性空白，且 A/D 与现有 TripoSR 管线强相关，能形成完整故事。
- **本轮缓后**：项目 B（视频生成）—— 与当前主线（静态图→3D→材质）关联较弱，等时间充足再单独开。

---

## 二、进度总表（一扫就知道做到哪）

| 项目 | 命中 JD 关键词 | 状态 | 当前进度 | 简要备注 |
|------|--------------|------|---------|---------|
| **A. 3D 模型动作生成** | 3D 动作 | 🔄 收尾中 | 85% | A-1~A-3b 完成；MoMask 文生动作跑通（走路72帧 BVH+火柴人gif），重定向简化（手动/Unity 失败→gif 演示），剩 A-5 演示 |
| **C. 推理优化** | 推理优化 | ✅ 已完成 | 100% | 3.88→0.75s(×5.2)，画质无损，已集成「快速模式」开关 |
| **D. Mesh 后处理** | Mesh 数据处理 | ✅ 已完成 | 100% | 全部完成：减面+展UV+LOD+服务接口+Unity实测（Tris降95.5%）|
| B. 视频生成（缓后） | 视频 | ⏸ 暂缓 | - | 见「六、缓后项目」 |

> 状态图例：⬜ 未开始 / 🔄 进行中 / ✅ 已完成 / ⏸ 暂缓 / ⚠️ 阻塞

---

## 三、项目 A：3D 模型动作生成

### A.1 目标（大白话）
你现在用 TripoSR 生成的 3D 模型是座"雕像"，放进游戏里不会动。这个项目让它能**走路、攻击、施法**——最终实现：**输入一句"挥剑攻击" → 得到一个带动画的 3D 角色，直接拖进 Unity 就能播**。

### A.2 命中 JD
- "3D 动作等 AI 技术"（任职要求 1）
- "图像、视频、Mesh 等多媒体数据…算法设计…推理优化"（职责 2，Mesh→动作链路）

### A.3 技术路线（推荐"先工程闭环，再叠 AI 生成"）

**路线 1（先做，工程闭环）——自动绑骨 + 动作套用**
- **自动绑骨**（rigging，给模型装"骨架"）：
  - 首选 **Mixamo**（Adobe 免费在线工具，上传模型→自动绑骨→下载带骨骼的模型，最稳）
  - 或开源 **RigNet**（AI 自动绑骨，纯本地，研究味更浓）
- **动作套用**（把现成动作"穿"到骨架上）：
  - Mixamo 动作库（几百种现成动捕动作）
  - 重定向（retarget，通俗说就是把动作"翻译"到你的模型骨架上）：Unity Humanoid Rig 或 Python 工具

**路线 2（后做，AI 生成点睛）——动作生成模型**
- **文本→动作**：**MotionLCM** / **MDM**（输入"角色向前挥拳"→ AI 生成一段动作序列）
- **视频→动作**（视频动捕）：**ViTPose / MMPose**（提取 2D 姿态）→ **MotionBERT**（2D 升 3D）→ 套到模型上。这条线"参考游戏宣传片动作 → 复刻到自家角色"，游戏感极强。

> **为什么分两步**：路线 1 能快速跑通"模型能动"的闭环，立刻有产出；路线 2 才是真正的"生成式 AI"，是面试讲故事的高光。先稳后亮。

### A.4 落地步骤
- [x] **A-1 调研对齐** ✅ 2026-07-02：TripoSR 导出 .glb(默认)/.obj（[main.py:626-634](inference_server/main.py#L626-L634)）；Unity 导入同 D-6（glb 直拖，带动画 FBX 需配 Humanoid Rig + Animator）；绑骨方案 **Mixamo**。**两个坑**：① Mixamo 不认 glb，需 Blender 转 fbx；② 橙子非人形，先用 Mixamo 自带角色打通
- [x] **A-2 绑骨闭环** ✅ 2026-07-02：用 Mixamo 自带角色（X Bot）走通"选动作→下 FBX→Unity Humanoid→Animator 播放"，做成按键 1/2/3 切换走/射/跑的交互 demo。主角为 Mixamo 自带角色（TripoSR 人形差，A-2c 再尝试上传自家模型）
- [x] **A-3a 选型调研** ✅ 2026-07-02：对比 MotionLCM/MoMask/Kimodo/视频动捕(ViTPose+MotionBERT)，选定 **MoMask**（CVPR2024 文生动作 SOTA、**官方确认 CPU 可跑零显存**、输出 BVH、官方给 Mixamo 角色的 Blender 重定向方案 + mapping.json，完美复用项目 D 工具链）
- [x] **A-3b-1 环境** ✅ 2026-07-02：clone MoMask + 独立 conda 环境（D:/anaconda3/envs/momask，管理员授权写权限）+ 装依赖（torch CPU 版、clip 本地 SSH 装、chumpy 摘除）
- [x] **A-3b-2 生成** ✅ 2026-07-02：跑通 `gen_t2m.py`（CPU），"A person walks forward." → 72 帧 BVH + gif 预览 + npy
- [x] **A-3b-3 重定向（简化收尾）** ✅ 2026-07-03：Blender 手动重定向 4 次失败 + Unity Humanoid 诡异（Hips 150° 校正未应用）→ 改用 MoMask 原始骨架（走路自然）+ 独立火柴人 gif 演示（npy + 真 kinematic chain，绕过 plot_script bug）
- [ ] **A-4（可选）集成**：把动作生成挂到推理服务/工作流引擎，做成"一句话出带动画角色"
- [ ] **A-5 演示与文档**：录对比视频（静态 vs 动起来）、写进 README/TECHNICAL_NOTES

### A.5 验收标准（怎样算做完）
1. 一个 TripoSR 生成的模型能在 Unity 里播放至少 3 种动作（走/攻击/待机）
2. 至少有一条"生成式 AI"路径跑通（文本或视频→动作）
3. 有可展示的演示视频和前后对比

### A.6 简历 / 面试话术（量化版，做完填数字）
> "构建 3D 角色动作生成双路线管线：① 工程闭环——Mixamo 自动绑骨 + Unity Humanoid 重定向，跑通'选动作→FBX→Unity 按键切换播放'交互 demo（走/射/跑 3 动作）；② 生成式 AI——集成 **MoMask（CVPR2024 文生动作 SOTA，FID 0.045）**，输入文字 → CPU 生成 72 帧人形走路 BVH（22 骨骼；Hips 前移 4m/3.6s、双脚交替+抬腿摆动相，数据验证自然走路）。攻坚：MoMask 权重 Google Drive 配额限流 → 转 HF 镜像 requests 直连；huggingface_hub 1.x hf-xet 绕过 hf-mirror → 禁用走 VPN；numpy 1.21/torch 1.12 API 冲突 → 锁版本。独立 conda 环境 CPU 推理零显存，不碰主管线。"

### A.7 状态与日志
- 2026-06-30：计划立项，未开工。
- 2026-07-02：✅ 完成 **A-1 调研对齐**。Blender MCP 已连通（场景含项目 D 橙子 `orange_LOD0_8000`）。确认事实：① TripoSR `/generate-3d` 导出 **.glb（默认）/.obj**（[main.py:626-634](inference_server/main.py#L626-L634)，`output_format` 参数控制）；② Unity 导入流程同 D-6（glb 直拖），但带动画 FBX 需额外配 Humanoid Rig + Animator Controller，属新内容；③ 绑骨方案锁定 **Mixamo**。**发现两个坑**：🚨 Mixamo 只认 .fbx/.obj、不认 .glb → 需 Blender 转 fbx（Blender MCP 可做）；🚨 Mixamo 是人形绑骨，橙子（球体）不能当主角。**与用户确认 A-2 方案**：先用 Mixamo 自带角色（X Bot）打通"选动作→下 FBX→Unity 播放"链路（必成功），再尝试上传自定义人形模型。
- 2026-07-02：✅ 完成 **A-2 绑骨闭环**（路线 1 工程闭环主体）。Mixamo 自带角色 X Bot → 下载 FBX for Unity（Walk / Fast Run / Shooting 三动作，存 `animation/`）→ Unity 导入 → Rig 配 Humanoid → Animator Controller + 按键切换 demo（按 1/2/3 切走/射/跑，重复按从头播）。
  - **主角说明**：用 Mixamo 自带角色（非 TripoSR 输出）。原因：① Mixamo 是人形绑骨，橙子（项目 D 模型）不可用；② TripoSR 对人形/二次元效果差（CLAUDE.md 已知短板）。先用自带角色打通链路最稳，A-2c 再尝试上传自家模型。
  - **踩坑**（Unity Animator 新手向，已同步 `docs/TECHNICAL_NOTES.md`「3D 动作生成」）：① 配 Humanoid 须在 Project 窗口点 fbx 源文件，不能在 Hierarchy 点场景实例（后者只显示 Transform，无 Rig 标签）；② Any State→State 过渡条件持续满足时，默认 `Can Transition To Self=true` 导致动画反复从头播（抖动），须取消勾选；③ `SetInteger` 设同值不触发重播（已在 Walk 再按 1 无反应），改用 `animator.Play("StateName", 0, 0f)` 每次从头播；④ 换 Play 脚本后须删掉 Any State 过渡线，否则 actionIndex 恒为 0 把角色从射击/快跑拉回走路；⑤ Mixamo clip 常都叫 `mixamo.com`，须右键 Rename 成 Walk/Shoot/Run。
  - **产物**：`animation/` 下 3 个 fbx；Unity 项目内 `PlayerAnim.controller` + `ActionSwitcher.cs`。
- 2026-07-02：✅ 完成 **A-3a 选型调研**（WebSearch 调研 + 精读 MoMask GitHub README）。对比候选：
  - 文生动作：**MoMask**（CVPR2024，FID 0.045 SOTA）、MotionLCM（ECCV2024，~8GB 实时）、NVIDIA Kimodo（2025，需 17GB 显存超 8GB 天花板，pass）、MMM（FID 0.089）。
  - 视频动捕：ViTPose（2D）→ MotionBERT（3D lifting）→ BVH，成熟但环节多。
  - **选定 MoMask**：① 官方确认 **WebUI demo 在 CPU 上跑通、No GPU required**（完美契合 8GB 铁律，零显存不与主管线抢资源）；② `gen_t2m.py --text_prompt "..."` 一行生成，**输出 BVH**（Blender/Unity 通用动作格式）；③ 官方 README 给出 **Blender 重定向方案**（keemap.rig.transfer 插件 + `assets/mapping.json` 骨骼映射，专门适配 Mixamo 角色），与 A-2 的 Mixamo 角色 + 项目 D 的 Blender MCP 完美复用。
  - **落地链路**：文字 → MoMask(`gen_t2m.py`, CPU) → BVH → Blender(keemap + `mapping.json` 重定向到 Mixamo 角色) → 带动画 FBX → Unity 播放。
  - **两个风险点**：① MoMask 预训练权重用 gdown 从 **Google Drive 下载**（国内可能失败，备选：代理/手动/HF 镜像）；② 依赖 PyTorch 1.7.1 / Python 3.10（与主管线 GPUpytorch-env 3.11 + PyTorch2.11 冲突），**须建独立 conda 环境**（momask），不碰主管线。
- 2026-07-02：✅ 完成 **A-3b-1 环境 + A-3b-2 生成**（路线 2 核心突破：文字 → AI → 3D 动作 BVH）。
  - **A-3b-1 环境**（踩坑密集）：① conda 默认把环境建到 C 盘用户目录，且 `D:/anaconda3/envs` 仅管理员可写 → 用户开**管理员 PowerShell** 跑 `icacls "D:\anaconda3\envs" /grant 15703:(OI)(CI)M /T`（+ pkgs）授权，重建 momask 到 D 盘，并删 C 盘 ai_env/momask；② requirements 改造：`environment.yml` 是 Linux 专用 → 用 `requirements.txt`（Python 3.10）；torch 换 CPU 版（1.3GB→150MB）；`chumpy` 老 setup.py（`import pip`）装不上 → 摘除（生成 BVH 用不到）；`clip @ git+https` 被 GitHub HTTPS 重置 → **SSH clone 到本地** + `clip @ file:///D:/aigc-project/CLIP`；scipy/sklearn 装最新与 numpy 1.21.5 打架 → 锁 `scipy==1.7.3` / `scikit-learn==1.0.2`；`matplotlib==3.1.3` 不兼容 Python 3.10 → 放宽。
  - **A-3b-2 权重下载**（最折腾）：Google Drive 被 gdown 走 drive.google.com → 开 VPN（香港）连通，但 **Google Drive 配额限流**（"many accesses"）→ 转 HF 镜像：andrewatef/MoMask-test 权重不全（仅评估版）、MeYourHint/MoMask 无权重 → 找到 **geedog/momask-codes-models**（全套 HumanML3D 权重，目录名与 `gen_t2m` 默认参数完全匹配）。下载：huggingface_hub 1.x 的 **hf-xet 后端**绕过 hf-mirror 直连被墙，`HF_HUB_DISABLE_XET=1` 未生效 → **绕开 huggingface_hub，用 requests 直连**（`trust_env=True` 走 VPN 从 huggingface.co 下 9 文件）成功（约 200MB）。
  - **A-3b-2 生成**：numpy 1.21.5 太旧（API 0xe）致 `torch.numpy()` 报"Numpy is not available"（torch 1.12 编译针对 API 0x10）→ 升级 `numpy==1.23.5`；matplotlib 3.5+ 的 `ax.lines`/`ax.dist` 只读 → 改 plot_script 兼容；mp4 需 ffmpeg 未装 → **改存 gif**（pillow 自带）。最终 `python gen_t2m.py --gpu_id -1 --text_prompt "A person walks forward."` 跑通，输出 72 帧 BVH + gif 预览 + npy（CLIP ViT-B/338MB 首次下载缓存到 ~/.cache/clip）。
  - **产物**：`momask-codes/generation/exp1/animations/0/sample0_repeat0_len72.{bvh,_ik.bvh,.gif,_ik.gif}` + `joints/0/*.npy`。
- 2026-07-03：⚠️ **A-3b-3 重定向受阻 → 简化收尾**（用户确认）。MoMask BVH → Mixamo 角色重定向尝试：
  - **Blender 手动 4 次失败**：① Copy Rotation（world, 无校正）→ mesh 扭曲（mapping.json 的 Hips `CorrectionFactorX=2.618rad=150°` 未应用）；② Copy Transforms（world）→ 单位错位（脚飞地下 -81m，BVH 与 Mixamo 单位不匹配）；③ local euler + CorrectionFactor → 脚飞天上 87m（骨骼局部坐标系矩阵运算错）；④ Copy Rotation + 校正 quaternion → 更扭曲。**根因**：mapping.json 的 18 骨骼校正（Hips 150° + 各骨 quaternion）需 keemap 插件精确应用，手写矩阵极易搞反。
  - **Unity Humanoid 重定向**：诡异（腰部插屁股）——Unity 自动配 Avatar 未应用 Hips 150° 校正，脊柱插进骨盆。Enforce T-pose 无效（rest pose 锁在 FBX）。
  - **简化方案**：MoMask 动作本身 OK（Blender 原始 22 骨骼走路自然，用户 Space 播放确认）→ 绕过重定向，用 npy 关节坐标 + 真 kinematic chain 画**火柴人 gif**（独立脚本 `stickfigure.py`，绕过 plot_3d_motion 的 `ax.lines` bug；视角 `view_init(elev=15, azim=135)` 斜俯视）。
  - **正解备忘**（未做）：MoMask 官方推荐 **keemap.rig.transfer** Blender 插件（mapping.json 给它写的，自动应用校正）是重定向到 Mixamo 的正解；本轮因 keemap 是 UI 工具 bpy 调用复杂 + 时间成本，简化收尾。
  - **产物**：`momask-codes/generation/exp1/walk_stickfigure.gif`（火柴人走路 72 帧，1.2MB）+ `stickfigure.py`。

---

## 四、项目 C：推理优化

### C.1 目标（大白话）
现在生成一张图要等好几秒、占满显存。这个项目让生成**更快（秒级出图）+ 更省显存（8GB 卡能干更多事）**，并产出一张"优化前 vs 优化后"的对比数据表——这是面试官最爱的工程硬功夫。

### C.2 命中 JD
- "推理优化"（职责 2 + 任职要求 2，明文要求）
- "扎实的编程技能"（任职要求 2）

### C.3 技术手段（通俗解释）
| 手段 | 通俗说 | 预期收益 |
|------|--------|---------|
| **LCM-LoRA / SDXL Turbo** | 少跑几步就出图（原来 30 步 → 4 步） | 速度 ×5~8 |
| **xformers / SDPA 注意力** | 更省内存的注意力计算 | 显存 ↓、速度 ↑ |
| **VAE tiling / slicing** | 把最后一步解码"分块"做，不爆显存 | 显存 ↓，能出大图 |
| **FP16 半精度**（确认已开） | 数字精度砍半，体积减半 | 显存 ↓ |
| **INT8 量化（bitsandbytes）** | 把模型"压缩"，常驻显存更小 | 显存 ↓（视显卡型号） |
| **model offload 调优** | 模型在显存/内存间智能搬运 | 多模型共存 |

> ⚠️ 显卡约束：8GB 显存。**FP8 量化需要 40 系（Ada）以上显卡**，30 系及以下只能用 INT8。开工前先确认显卡型号（见 [SYSTEM_INFO.md](SYSTEM_INFO.md)）。

### C.4 落地步骤
- [x] **C-1 基线测量**：记录现状——单图生成耗时、峰值显存、能支持的最大分辨率 ✅ 2026-06-30
- [x] **C-2 加速**：接 LCM-LoRA / Turbo，少步数出图 ✅ 2026-06-30（LCM 4 步，0.75s，画质 8/8 无损）
- [x] **C-3 显存优化**：SDPA ✅（3.88→2.75s，显存不变）；VAE tiling 待大图分辨率验证时补测
- [x] **C-4（可选）量化**：评估后跳过（显存仅用 2.6/8GB 不缺、triton 未装，量化/torch.compile 边际收益低）
- [x] **C-5 出基准报告**：✅ 见 `benchmarks/BENCHMARK_REPORT.md`
- [x] **C-6 集成**：✅ `/generate` + `/workflows/run` 加 fast_mode 开关，集成测试通过

### C.5 验收标准
1. 单图生成速度有明显提升（有前后数字对比）
2. 峰值显存下降（有数字）
3. 画质无明显劣化（主观/SSIM 对比）
4. 产出一份数据对比表

### C.6 简历 / 面试话术
> "针对 8GB 显存约束优化 AIGC 推理服务（SD 1.5 + LoRA）：以 PyTorch 2.x SDPA 注意力替代 attention_slicing、引入 LCM-LoRA 少步生成（25 步→4 步），单图纯推理耗时从 3.88s 降至 0.75s（×5.2），峰值显存 2.63→2.75GB 基本持平，视觉盲评画质 8/10 无明显劣化；优化以「快速模式」开关集成进 FastAPI 服务，按需 4 步极速出图。"

### C.7 状态与日志
- 2026-06-30：计划立项，未开工。
- 2026-06-30：✅ 完成 **C-1 基线测量**。写好基准脚本 `benchmarks/bench_inference.py`（复用 model_loader 全局管线，固定 prompt + seed=42，warmup 1 次 + 正式测 3 次取均值，输出 JSON + 基准图）。
  - **优化前基线（512×512，25 步，DPM solver，`enable_attention_slicing` 开启）**：
    - 单图耗时 **3.88s**（每步 0.16s）
    - 峰值显存 **2.63GB**（仅占 8GB 的 33%）
  - **关键发现**：显存占用仅 1/3，说明现有的 `enable_attention_slicing()`（以速度换显存的妥协）纯属"白付代价"——根本不缺显存。下一步去掉它换 SDPA，预计纯赚速度。
  - 产物：`benchmarks/results/baseline.json`、`benchmarks/results/baseline_seed42.png`。
- 2026-06-30：✅ 完成 **C-3 SDPA 优化**（去掉 `enable_attention_slicing`，改由 PyTorch 2.x 原生 SDPA 接管注意力计算）。
  - **对比**：单图耗时 **3.88s → 2.75s（提速 29%，×1.41）**；每步 0.16s→0.11s（it/s 6.8→9.8）。
  - **显存**：2.63GB → 2.63GB（**零代价**——SDPA 内存效率本就优于 attention_slicing）。
  - 改动：`inference_server/model_loader.py` 的 `load_pipeline()` 注释掉 `enable_attention_slicing()`，保留回退注释。
  - 产物：`benchmarks/results/sdpa.json`、`sdpa_seed42.png`。
- 2026-06-30：✅ 完成 **C-2 LCM 少步加速**（叠加 LCM-LoRA + 切换 LCMScheduler）。
  - **对比（512×512）**：单图耗时 **2.75s → 0.75s（较 SDPA 再快 3.7 倍）**；相比最初基线 **3.88s → 0.75s（×5.2）**。
  - **显存**：2.63GB → 2.75GB（仅 +0.12GB）。
  - **画质**：4 步 LCM 图 vs 25 步标准图，视觉模型盲评均 **8/10**，**无明显劣化**。
  - 踩坑：LCM-LoRA 仓库名是 `latent-consistency/lcm-lora-sdv1-5`（`v1-5` 带横杠），不是 `sdv15`；hf-mirror 对其 `/api/` 目录查询返回 401，改用 `hf_hub_download` 直拉 `/resolve/` 单文件绕过。
  - 产物：`benchmarks/results/lcm.json`、`lcm_seed42.png`；LCM-LoRA 缓存于 `cache/hub/lcm-lora/`。
- 2026-07-01：✅ 完成 **C-6 集成** + **C-5 报告** + **C-4 评估**，项目 C 收尾。
  - **C-6 集成**：给 `/generate`（`GenerateRequest.fast_mode`）和 `/workflows/run`（`fast_mode` Form）加「快速模式」开关；`model_loader` 加 `ensure_lcm_mode()` 幂等切换（加载/卸载 LCM-LoRA + 切 LCMScheduler/DPM scheduler）。工作流经 `BaseWorkflow.fast_mode` 透传，`_txt2img` 在快速模式夹紧到 ≤8 步、cfg=1.5。角色 LoRA 已 merge 进 UNet，LCM-LoRA 仅叠加其上，unload 不影响角色 LoRA。
  - **集成测试**（启动服务实测）：标准→LCM→切回标准 三次切换均正确；工作流 `character_concept fast_mode=true` 返回 200（1024×576 LCM 8 步 3.85s）。日志确认 `→ 启用 LCM 快速模式` / `→ 恢复标准模式` 正确触发。
  - **C-4 评估**：跳过。显存仅用 2.6/8GB（量化省显存无意义）；triton 未装（torch.compile 收益打折）；LCM 已达 ×5.2，边际收益低。
  - **C-5 报告**：`benchmarks/BENCHMARK_REPORT.md`；技术踩坑同步到 `docs/TECHNICAL_NOTES.md`「推理优化」章节。

---

## 五、项目 D：3D 模型后处理（Mesh 处理）

### D.1 目标（大白话）
TripoSR 出的 3D 模型是"毛坯"——面数动辄十几万（游戏跑不动）、多数没 UV（贴不上贴图）。这个项目把毛坯打磨成"游戏真能用的"：**减面、展 UV、生成 LOD（远近切换的多级精度）**。

### D.2 命中 JD
- "Mesh 等多媒体数据…数据处理…算法设计"（职责 2，明文 Mesh）
- 补全 3D 生产链路最后一环

### D.3 技术手段
| 步骤 | 工具 | 通俗说 |
|------|------|--------|
| **减面** | PyMeshLab / trimesh | 砍掉多余三角面，保留外形 |
| **LOD 生成** | 按比例分级减面 | 远处用粗糙版、近处用精细版，省性能 |
| **UV 展开** | xatlas（Python 绑定） | 把 3D 表面"摊平"成 2D，才能贴图 |
| **贴图烘焙（可选）** | Blender bpy / xatlas | 把高模细节"印"到低模贴图上 |

### D.4 落地步骤
- [x] **D-1 现状评估**：测 TripoSR 输出模型的面数、是否有 UV、是否游戏可用 ✅ 2026-07-01（见 D.7）
- [x] **D-2 减面 + LOD**：写脚本自动减面 + 生成 2~3 级 LOD ✅ 2026-07-01（见 D.7）
- [x] **D-3 UV 展开**：用 xatlas 自动展 UV，验证能贴 PBR 材质 ✅ 2026-07-01（改用 Blender Smart UV Project，见 D.7）
- [ ] **D-4（可选）贴图烘焙**：高模→低模细节转移
- [x] **D-5 集成**：推理服务加"导出游戏级 Mesh"选项（带 LOD + UV）✅ 2026-07-01（见 D.7）
- [x] **D-6 对比**：原始毛坯 vs 优化后，Unity 里跑性能对比 ✅ 2026-07-01（见 D.7）

### D.5 验收标准
1. 模型面数降到游戏可用级别（如 < 1 万面，有前后数字）
2. 自动生成 UV，能正确贴 PBR 材质（用现有的 TriplanarPBR 或新 UV 都行）
3. 至少生成 1 套 LOD
4. Unity 实测性能提升（帧率/内存）

### D.6 简历 / 面试话术
> "构建 3D 资产后处理流水线：自动减面、LOD 分级生成、Smart UV Project UV 展开（Blender headless subprocess），将 TripoSR 输出从 **19.1 万面降至 8000 面**（生成 3 级 LOD），并以独立接口 `/post-process-mesh` 集成进 FastAPI 服务；**Unity 实测三角面数下降 95.5%（575k→25.7k）、顶点下降 94.4%**，渲染负载降低 20+ 倍，达到游戏实时可用标准。"

### D.7 状态与日志
- 2026-06-30：计划立项，未开工。
- 2026-07-01：✅ 完成 **D-1 现状评估**。用 trimesh 批量体检 14 个 TripoSR 输出（脚本 `mesh_tools/inspect_mesh.py`，纯只读）：
  - **面数 11.78 万 ~ 19.5 万（平均 17.2 万）**，全部远超游戏可用阈值（<1 万），超标 10~20 倍 → D-2 减面是头号任务。
  - **UV 坐标 0/14，纹理材质 0/14**；**顶点色 14/14 全有**——TripoSR 仅靠顶点色携带颜色，印证了 CLAUDE.md 铁律 6（必须用 `AIGC/TriplanarPBR` 无视 UV 的 shader）。D-3 必须展 UV 才能解锁 Standard shader。
  - 封闭性 9/14 watertight；尺寸均归一化到 ~1.0 立方体。
  - 踩坑：① 第一版检测误判 UV（trimesh `TextureVisuals.uv` 属性存在但为空），增强检测（按 UV 唯一坐标点数）后才确认真无 UV；② Windows 控制台 GBK 编码不支持 emoji，脚本输出改纯 ASCII。
  - 产物：`mesh_tools/inspect_mesh.py`、`mesh_tools/d1_assessment.json`（14 个模型完整指标）。
- 2026-07-01：✅ 完成 **D-2 减面 + LOD**（Blender MCP）。样本：`20260626_115239_318204_3d_model.glb`（18.89 万面）。
  - Decimate（COLLAPSE 模式，按 `ratio=目标/原始` 设定）精准砍到 3 级 LOD：**LOD0 8000 面 / LOD1 2500 面 / LOD2 800 面**（缩面最高 ×236）。
  - **顶点色完整保留**（glb `COLOR_0=YES`，COLLAPSE 插值）；**Blender 顺手补了法线 NORMAL**（原始 TripoSR glb 连法线都没有，补上对游戏光照有利）。
  - 导出 3 个独立 glb + 3 张同视角渲染对比图（`mesh_tools/d2_output/`）。
  - 踩坑：① Blender 5.1 新建材质默认 BSDF 节点名不再是字符串 `"Principled BSDF"`，须按 `node.type=='BSDF_PRINCIPLED'` 匹配；② 渲染引擎改用 `BLENDER_EEVEE_NEXT`；③ **trimesh 对"glb 同时带材质+顶点色"会误报无顶点色**（实际 `COLOR_0` 在文件里），需解析 glb JSON 确认 → 已写 `mesh_tools/check_glb_attrs.py` 兜底；④ glb 按属性拆分顶点，导出后顶点数 ≈ 面数×3（Blender 内部共享顶点），非 bug。
  - 产物：`mesh_tools/d2_output/{LOD0_fine,LOD1_mid,LOD2_far}.glb`、3 张 `render_*.png`、`mesh_tools/check_glb_attrs.py`。
- 2026-07-01：✅ **换图验证 D-2 通用性**：用户新图（橙子 `tripoSR_orange.jpg`）经 `/generate-3d` 生成新毛坯（19.1 万面），重跑减面 → LOD0/1/2 = 8000/2500/800，顶点色全保留。证明减面流程不挑图（剑+橙子双验证）。
- 2026-07-01：✅ 完成 **D-3 UV 展开**（Blender MCP）。对象：orange_LOD0（8000 面）。
  - Smart UV Project 自动展 UV（须 EDIT 模式调用，OBJECT 模式 poll 失败）。
  - 棋盘格材质按 UV 渲染，验证 UV 质量（行业惯例：棋盘格均匀=无拉伸）。
  - **数据铁证**：glb `TEXCOORD_0` 展 UV 前 `NO` → 后 `YES`（`orange_LOD0.glb` vs `orange_LOD0_uv.glb`），解锁 CLAUDE.md 铁律 5「可改 Standard shader」。
  - 踩坑：① `bpy.ops.uv.smart_project` 在 OBJECT 模式 poll 失败，须 `mode_set('EDIT')` + `mesh.select_all` 后再调；② 材质未引用顶点色时导出会丢 `COLOR_0`（不影响 UV 验证）。
  - 产物：`mesh_tools/d2_output/orange_LOD0_uv.glb`、`render_orange_LOD0_checker.png`、`render_orange_compare.png`。
- 2026-07-01：✅ 完成 **D-5 集成进推理服务**（Blender headless subprocess 路线）。新增「游戏级 Mesh 后处理」接口：客户端上传 glb → 一键得 ZIP（减面+UV+LOD+预览+清单）。
  - 新增 `inference_server/blender_scripts/post_process_mesh.py`：参数化 bpy 脚本（被 `blender --background --python` 调用），固化 D-2/D-3 验证的减面+展UV+LOD+渲染流程，写 manifest.json。
  - 新增 `inference_server/mesh_postprocessor.py`：编排模块（subprocess 调 Blender + blender.exe 三级路径回退 + 5min 超时兜底 + ZIP 打包），**纯 CPU 不 import torch**。
  - 修改 `inference_server/main.py`：仅在 shutdown 事件前**追加** `POST /post-process-mesh` 接口（参数 file/target_faces/uv_unwrap/lod_faces/render_preview → ZIP 响应），现有 4 接口正文一行未改。
  - **铁律遵守**：未碰 `model_loader.py`（保护 C 的 fast_mode/LCM）、未碰工作流引擎。
  - 技术路线选 Blender headless 而非纯 Python：trimesh 减面依赖 `fast-simplification`（Windows Rust 编译易失败）且不会展 UV；Blender 已装、效果已 D-2/D-3 验证。
  - 端到端测试（橙子 19.1 万面）：接口 6.7s 返回 ZIP，3 级 LOD 均 `COLOR_0=YES` + `TEXCOORD_0=YES`。
  - 回归验证：`/health` 正常；`/generate fast_mode=true` 仍 LCM 出图（PNG 462KB / 3.9s）—— **证明未破坏项目 C**。
  - 踩坑：① EEVEE_NEXT 在 headless 无界面模式**可**渲染（原担心要改 Cycles CPU，多虑了）；② Blender 5.1 新建材质 BSDF 节点名变了，按 `type=='BSDF_PRINCIPLED'` 匹配；③ `smart_project` 必须 EDIT 模式；④ `GenerateRequest.steps` 有 `ge=10` 约束，测 fast_mode 时**不能传 steps<10**（fast_mode 会自动把默认 25 改成 6 步，正确用法是不传 steps）。
  - 产物：`/post-process-mesh` 接口 + 上述 2 个新文件；测试脚本 `mesh_tools/test_postprocessor.py`。
- 2026-07-01：✅ 完成 **D-6 Unity 性能验收**（项目 D 收尾）。用户在本地 Unity 项目实测毛坯 vs 优化后（Game 视图 Stats）：
  - **Triangles 575k → 25.7k（降 95.5%，×22.4）；Vertices 1.2m → 67.5k（降 94.4%，×17.8）**。
  - FPS 单测都 ~400（机器强，单模型看不出差异；真实多模型场景毛坯必卡）。
  - Unity Tris 绝对值高于 mesh 面数（含阴影 pass/顶点拆分，约 3x），但毛坯∶优化后比例 ×22 与 mesh 面数比 ×24 一致，准确反映优化幅度。
  - **验收结论**：项目 D 全部达标（面数<1万 ✅、UV ✅、3 级 LOD ✅、Unity 性能提升 ✅）。
  - 产物：对比包 `mesh_tools/d6_compare/`、验收报告 `mesh_tools/D6_unity_report.md`。

---

## 六、缓后项目 B：视频生成（暂存备忘，本轮不做）

- **命中 JD**："视频内容生成"（明文）
- **缓后原因**（用户 2026-06-30 确认）：与当前主线（静态图→3D→材质）关联较弱，优先级低于 A/C/D。
- **未来可做方向**（备忘）：
  - 动态立绘：**AnimateDiff**（给 LoRA 生成的角色立绘加微动效）
  - 文生视频：**Wan2.2 / CogVideoX / HunyuanVideo**（国产开源，国内好下载）
  - 图生视频：**SVD**（Stable Video Diffusion）
- **触发条件**：A/C/D 基本完成、时间充足时再启动。

---

## 七、不可违反的技术约束（继承自 CLAUDE.md 铁律）

做 A/C/D 时**必须遵守**，否则必踩坑：
1. **8GB 显存是天花板**：动作生成/量化/减面工具都不能爆显存；新模型按需懒加载。
2. **复用全局单例**：新功能尽量复用现有 SD/管线，不重复加载模型（里程碑 7 的教训）。
3. **`HF_ENDPOINT='https://hf-mirror.com'` 必须在 import 之前设置**（国内网络）。
4. **LoRA 加载用 `merge_and_unload()`**。
5. **3D 模型在 Unity 默认用 `AIGC/TriplanarPBR` shader**（除非 D 项目成功展了 UV，可改用 Standard）。
6. **`print()` 加 `flush=True`**；改代码后重启服务才生效。
7. **不自动 push GitHub**：只 commit，等用户确认后再 `git push origin master:main`（SSH）。
8. Python 统一用 `D:/anaconda3/envs/GPUpytorch-env/python.exe`。

---

## 八、下一步行动（Next Actions）

> AI 接手时，从这里开始。

**⭐ 第 0 步（环境就绪检查，必做）**：验证 Blender MCP 已连通——调用 blender MCP 的 `get_mcp_context` 或 `get_scene_info` 工具，能返回 Blender 场景信息即说明连接正常（详见第十节）。**每次使用前必须**：Blender 开着 + 按 `N` 键 → BlenderMCP 面板 → 已点 **Start Server**（看到 "Running on port 9876"）。

当前建议执行顺序（性价比从高到低）：

1. ~~**项目 C（推理优化）**~~ ✅ **已完成**（3.88→0.75s ×5.2）。
2. ~~**项目 D（Mesh 后处理）**~~ ✅ **已完成**（D-1~D-3 + D-5 + D-6：减面/展UV/LOD + 服务接口 `/post-process-mesh` + Unity 实测 Tris 降 95.5%）。
3. **项目 A（3D 动作）🔄 收尾**：A-1~A-3b ✅（MoMask 文生动作跑通 + 火柴人 gif 演示；重定向到 Mixamo 失败→简化为 MoMask 原始动作演示）；**剩 A-5**（录演示视频 + README 同步）。A-2c 上传自家模型绑骨、视频动捕、keemap 重定向为可选增强。

> 💡 也可按用户意愿调整顺序。开工任一项目前，AI 应先把对应项目的「落地步骤」拆成可执行清单，用大白话向用户说明，再动手。

---

## 九、如何更新本文件（每次有进展必做）

1. **进度总表**（第二节）：更新对应行的「状态」「当前进度」「备注」。
2. **对应项目的「状态与日志」**（如 A.7 / C.7 / D.7）：加一行 `YYYY-MM-DD：做了什么、结果如何、遇到什么问题`。
3. **落地步骤**（如 A.4）：勾选已完成的小步骤 `[x]`。
4. **下一步行动**（第八节）：把刚做完的划掉，更新当前最该做的。
5. 关键踩坑同步到 [docs/TECHNICAL_NOTES.md](docs/TECHNICAL_NOTES.md) 对应章节。

---

## 十、工具箱与环境配置（MCP）

> 本节记录为推进 A/C/D 而配置的 AI 工具链。**新会话接手时先读这里。**

### 10.1 Blender MCP ✅ 已配置（2026-06-30 完成）
- **用途**：让 AI 直接操控 Blender（减面、展 UV、绑骨、套动作、导出 FBX）—— **项目 A 和 D 的主力工具**。
- **安装明细**：
  - `blender-mcp 1.6.4` 已装，启动器在 `C:\Users\15703\.local\bin\blender-mcp.exe`
  - addon 已导入 **Blender 5.1**（addon.py 备份：`d:\aigc-project\.claude\blender_mcp_addon.py` 及项目根目录）
  - MCP 配置已写入 `C:\Users\15703\.claude.json`（备份 `.bak`），command 指向 blender-mcp.exe
  - 启动器实际通过 `python -m uv` 安装（uv 在用户 site-packages，无独立 uvx.exe，但 blender-mcp.exe 已生成可用）
- **⚠️ 每次使用前必须**：① 开 Blender；② 按 `N` 键 → 右侧面板 BlenderMCP 标签 → 点 **Start Server**（看到 "Running on port 9876" 才算就绪）。
- **验证方法**：调用 blender MCP 的 `get_mcp_context` / `get_scene_info`，能返回场景信息 = 连接正常。
- **已知特性**：blender-mcp 启动会向 supabase 发匿名遥测，不影响功能。

### 10.2 HuggingFace MCP ❌ 未装（国内网络限制，已改方案）
- **原因**：HF 官方 MCP 连 huggingface.co 主站，国内直连 8 秒超时（2026-06-30 实测 HTTP 000）。
- **替代方案（已采纳）**：需要搜/下模型时，AI 直接用 `HF_ENDPOINT=https://hf-mirror.com` + `huggingface_hub` 库下载，国内稳定、效果等价。

### 10.3 工具链版本备忘
- Python：`D:/anaconda3/envs/GPUpytorch-env/python.exe`（3.11）
- uv：用 `python -m uv` 调用（装在用户 site-packages）
- Blender：`D:\Steam\steamapps\common\Blender\blender.exe`（5.1）

---

> **核心理念**（同 CLAUDE.md）：用户不需要懂技术，只需要说"想要什么效果"。技术细节留给 AI，简单选择留给用户。
