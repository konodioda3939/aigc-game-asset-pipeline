# 推理优化基准报告（项目 C）

> SD 1.5 + LoRA 文生图，在 **8GB 显存（RTX 4060 Laptop）** 约束下的推理性能优化。
> 本报告是「推理优化」方向的硬数据素材，所有数字均可复现（见文末「复现方法」）。
> 生成日期：2026-07-01。

---

## 一、测试环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 4060 Laptop（8GB GDDR6，Ada 架构，Compute Capability 8.9） |
| PyTorch | 2.11.0 + CUDA 12.8 |
| 基座模型 | SD 1.5（Counterfeit-V2.5）+ 自训练角色 LoRA（已 `merge_and_unload` 进 UNet） |
| 注意力 | PyTorch 2.x 原生 SDPA（`scaled_dot_product_attention`） |
| 测试设定 | 512×512，固定 prompt + seed=42，预热 1 次 + 正式测 3 次取均值 |

---

## 二、优化手段

| # | 手段 | 通俗说 | 风险 |
|---|------|--------|------|
| 1 | 去掉 `enable_attention_slicing`，改由 **SDPA** 接管 | 原来用「切片注意力」省显存但慢；实测显存根本不缺，换成 PyTorch 2.x 自带的更快算法 | 极低 |
| 2 | 叠加 **LCM-LoRA** + LCMScheduler | 把出图步数从 25 步压到 **4 步**（少步蒸馏） | 中（画质需验证） |

> **量化（INT8/FP8）/ torch.compile 经评估跳过**：显存仅用 2.6/8GB（不缺显存，量化省显存无意义）；triton 未安装（torch.compile 收益打折）；LCM 已达 ×5.2，边际收益低。

---

## 三、性能对比（纯推理，512×512）

| 方案 | 步数 | 单图耗时 | 峰值显存 | 较基线加速 | 画质（盲评） |
|------|:---:|:---:|:---:|:---:|:---:|
| 基线（attention_slicing） | 25 | 3.88s | 2.63GB | — | 8/10 |
| + SDPA 注意力 | 25 | 2.75s | 2.63GB | ×1.41 | 8/10 |
| **+ LCM 4 步出图** | **4** | **0.75s** | 2.75GB | **×5.17** | **8/10** |

**核心结论**：单图纯推理 **3.88s → 0.75s（×5.17）**，峰值显存基本持平（2.63→2.75GB，+0.12GB），画质盲评 8/10 无明显劣化。

### 每步推理速度
- 基线：6.8 步/秒（0.16s/步）
- SDPA：9.8 步/秒（0.11s/步，**每步快 44%**）
- LCM：单步 0.19s（LCM 单步略重，但只需 4 步即可出图）

---

## 四、画质评估

用视觉模型对「同 prompt + 同 seed」的 **25 步标准图** 与 **4 步 LCM 图** 盲评：

| 方案 | 评分 | 评价 |
|------|:---:|------|
| 25 步标准 | 8/10 | 清晰度高，发丝/服饰细节丰富，无模糊/噪点/五官畸变 |
| 4 步 LCM | 8/10 | 清晰度高，细节丰富，色彩过渡自然，无结构崩坏 |

**结论**：4 步 LCM 画质与 25 步标准持平，完全可用于快速预览 / 批量出图。

基准图：[`results/baseline_seed42.png`](results/baseline_seed42.png) / [`sdpa_seed42.png`](results/sdpa_seed42.png) / [`lcm_seed42.png`](results/lcm_seed42.png)。

---

## 五、集成（已落地到推理服务）

优化以 **「快速模式」开关** 集成进 FastAPI 推理服务，**默认关闭，现有功能零影响**：

- `POST /generate` 加 `fast_mode` 字段（JSON body）
- `POST /workflows/run` 加 `fast_mode` 字段（form-data）
- `model_loader.ensure_lcm_mode(active)` 幂等切换：
  - `active=True` → 加载 LCM-LoRA + 切 LCMScheduler
  - `active=False` → 卸载 LCM-LoRA + 恢复 DPMSolverMultistepScheduler
- 角色 LoRA 已 `merge_and_unload` 进 UNet 基础权重，LCM-LoRA 仅作为 peft adapter 叠加其上；开关关闭时 `unload_lora_weights()` 只移除 adapter，**不影响角色 LoRA**，标准/LCM 可安全互切。

### 集成测试（HTTP 端到端，启动服务实测）
标准 → LCM → 切回标准 三次切换均正确（日志确认 `→ 启用 LCM 快速模式` / `→ 恢复标准模式`）；工作流 `character_concept fast_mode=true` 返回 200。
> 注：HTTP 端到端耗时含网络 + 图片编码 + 存盘开销，高于纯推理（如 LCM 端到端 ~1.9s vs 纯推理 0.75s），这层开销对所有方案固定。

---

## 六、简历 / 面试话术

> "针对 8GB 显存约束优化 AIGC 推理服务（SD 1.5 + LoRA）：以 PyTorch 2.x SDPA 注意力替代 attention_slicing、引入 LCM-LoRA 少步生成（25 步→4 步），单图纯推理耗时从 **3.88s 降至 0.75s（×5.2）**，峰值显存 **2.63→2.75GB 基本持平**，视觉盲评画质 **8/10 无明显劣化**；优化以「快速模式」开关集成进 FastAPI 服务，按需 4 步极速出图。"

---

## 七、复现方法

```bash
# Python 环境：D:/anaconda3/envs/GPUpytorch-env/python.exe

# 1. SDPA 优化后（model_loader 已默认 SDPA）
python benchmarks/bench_inference.py --label sdpa

# 2. LCM 4 步快速模式
python benchmarks/bench_inference.py --label lcm --lcm

# 结果产出：benchmarks/results/<label>.json + <label>_seed42.png
```

> 基线（attention_slicing）数据为历史记录：需在 `model_loader.load_pipeline()` 临时恢复 `pipe.enable_attention_slicing()` 后用 `--label baseline` 重跑。

---

## 八、踩坑备忘

- **LCM-LoRA 仓库名** 是 `latent-consistency/lcm-lora-sdv1-5`（`v1-5` 带横杠），不是 `sdv15`——后者 404。
- **hf-mirror 对该仓库的 `/api/` 目录查询返回 401**；改用 `hf_hub_download` 直拉 `/resolve/` 单文件绕过（公开文件不触发认证）。
- **`enable_attention_slicing` 会覆盖 SDPA**：必须关掉它，PyTorch 2.x 的 SDPA（`AttnProcessor2_0`）才会生效。
- LCM 模式必须配低 CFG（1.0–2.0）和 LCMScheduler，否则出图崩；标准模式切回时务必同时卸载 LoRA 和换回 DPM scheduler。
