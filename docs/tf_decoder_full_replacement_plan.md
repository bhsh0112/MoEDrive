### `_tf_decoder` 完整替换为 MoE 的方案说明（Full Replacement Plan）

本文档面向“把原有 `_tf_decoder` **彻底替换**掉”的需求，给出可执行的实现方案、训练注意事项（DDP unused 参数）、推荐超参和验证步骤。  
适用仓库：`MOEDrive`（NavSim 训练/评测链路）。

---

### 1. 这里的“完整替换”到底指什么

在 Transfuser / DiffusionDrive agent 中，`_tf_decoder` 原本是：

- `torch.nn.TransformerDecoder(...)`

本项目的“完整替换”指：

- 在模型代码中 **不再实例化** `torch.nn.TransformerDecoder`；
- 统一改为我们自定义的 MoE decoder 模块，并在 forward 时 **产出 aux 信息**（`moe_aux_loss / usage` 等）供训练 loss 与日志使用；
- 保持输出张量契约不变：`query_out` 仍为 `(B, Q, D)`，下游 trajectory/agent head 无需改动。

> 注意：即使我们说“完整替换 `_tf_decoder`”，也有两种不同“力度”的 MoE 设计（见第 3 节）。两者都满足“不再用 torch 的 TransformerDecoder”。

---

### 2. 影响范围（确切文件）

#### 2.1 需要替换的模型文件（已完成）

- `MOEDrive/navsim/agents/transfuser/transfuser_model.py`
- `MOEDrive/navsim/agents/diffusiondrive/transfuser_model_v2.py`

两处的 `_tf_decoder` 已切换为 MoE 实现。

#### 2.2 MoE 实现承载文件

- `MOEDrive/navsim/agents/moe_transformer_decoder.py`

其中包含：

- FFN‑MoE（token 级）：`MoETransformerDecoderLayer` + `MoETransformerDecoder`
- 整层 MoE（样本级）：`MoEFullDecoderLayer` + `MoELayerwiseTransformerDecoder`

#### 2.3 训练 loss 与日志（必须打通）

为确保 MoE 真实训练并可观测，我们要求 loss 侧消费 `moe_aux_loss` 并 log 各项标量：

- `MOEDrive/navsim/agents/transfuser/transfuser_loss.py`
- `MOEDrive/navsim/agents/diffusiondrive/transfuser_loss.py`

---

### 3. 两种“完整替换”的实现路线（推荐你明确选择）

#### 3.1 路线 A：FFN‑MoE（“只 MoE 化 FFN”，更稳）

**核心思路**：保留 decoder layer 的 self‑attn 与 cross‑attn，仅把 FFN 子层替换为 MoE（多个 FFN 专家 + token 级 top‑k 路由）。

- **优点**：
  - 注意力结构不动，行为漂移小；
  - 常见、稳定、调参成本低；
  - token 路由次数多，usage 统计更平滑。
- **缺点**：
  - “专家化”主要发生在 FFN 容量上；attention 不分专家。

对应模块：
- `MoETransformerDecoderLayer`
- `MoETransformerDecoder`

#### 3.2 路线 B：整层 MoE（“更激进”，专家=完整 decoder layer）

**核心思路**：把“一个 decoder layer（self‑attn + cross‑attn + FFN）”作为专家；每个 layer 位置有 E 个专家层；router 在**样本级**选 top‑k 专家并对整层输出加权融合。

- **优点**：
  - 专家不仅包含 FFN，还包含 attention 参数，容量更大、专门化更彻底；
  - 更符合“全层专家”的强 MoE 设想。
- **缺点（重要）**：
  - **DDP 会出现 unused parameters**（因为某些专家在某 step 没被选到）；
  - 统计粒度更粗（样本级），小 batch 下 usage 波动更大；
  - 调参风险更高。

对应模块：
- `MoEFullDecoderLayer`
- `MoELayerwiseTransformerDecoder`

> 目前仓库默认是路线 B（整层 MoE），因为你明确要求“更激进的修改”。

---

### 4. 训练必改点：DDP unused parameters（整层 MoE 必须）

你遇到的报错：

- `RuntimeError: ... parameters that were not used ...`

原因：
- 整层 MoE 每 step 只激活 top‑k 个专家层；
- 未激活专家的参数不会参与 loss 反传；
- Lightning 的 DDP 默认会报错。

**解决**：把 strategy 改成：

- `ddp_find_unused_parameters_true`

你可以二选一：

1) **命令行覆盖（最直接）**  
在训练命令加：`trainer.params.strategy=ddp_find_unused_parameters_true`

2) **使用 MoE 专用训练配置（更省心）**  
已新增：`MOEDrive/navsim/planning/script/config/training/default_training_moe.yaml`  
其中默认 `trainer.params.strategy: ddp_find_unused_parameters_true`

---

### 5. 推荐超参与起步建议

MoE 相关关键参数（两条路线通用）：

- `moe_num_experts`：建议 `4` 起步
- `moe_top_k`：建议 `2`
- `moe_router_temperature`：`1.0`
- `moe_load_balance_coef`：建议 `1e-2` 起步（防塌缩）
- `moe_router_z_loss_coef`：建议 `1e-3`（防 router logits 过大）
- `moe_aux_loss_weight`：建议 `1.0`

经验法则：

- usage 长期塌缩到单一专家：提高 `moe_load_balance_coef`
- 训练不稳定/NaN：提高 `moe_router_z_loss_coef` 或降低学习率（更进一步的训练配置调参）

---

### 6. 验证清单（你应该按这个顺序测）

#### 6.1 单元级 sanity check（不依赖数据集）

脚本：`MOEDrive/scripts/sanity_check_moe_decoder.py`

- `--variant ffn`：验证 FFN‑MoE
- `--variant layer`：验证整层 MoE
- `--variant both`：两者都跑

检查项：
- 输出 shape 必须是 `(B, Q, D)`
- `moe_aux_loss` 非零且无 NaN/Inf
- `moe_usage_fraction` 合理（小 batch 波动大属正常）

#### 6.2 训练 smoke run（几百 step）

目标：确认 MoE 指标在 Lightning log 中持续产出，且不会出现专家长期塌缩。

重点观察：
- `train/moe_aux_loss`
- `train/moe_usage_fraction_e0..e{E-1}`

#### 6.3 全训练 + PDM 评测

评测指标在：
- `exp/<experiment_name>/<timestamp>/log.txt`
- `exp/<experiment_name>/<timestamp>/<timestamp>.csv` 的 `token=average` 行

---

### 7. 训练命令模板（整层 MoE，推荐）

#### 7.1 用命令行覆盖 strategy（最直接）

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  agent=diffusiondrive_agent \
  experiment_name=training_diffusiondrive_agent_layerwise_moe \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=100 \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  +agent.config.moe_num_experts=4 \
  +agent.config.moe_top_k=2 \
  +agent.config.moe_router_temperature=1.0 \
  +agent.config.moe_load_balance_coef=1e-2 \
  +agent.config.moe_router_z_loss_coef=1e-3 \
  +agent.config.moe_aux_loss_weight=1.0
```

#### 7.2 使用 MoE 专用 training config（更省心）

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  --config-name default_training_moe \
  agent=diffusiondrive_agent \
  experiment_name=training_diffusiondrive_agent_layerwise_moe \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=100 \
  cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
  use_cache_without_dataset=True \
  force_cache_computation=False
```

---

### 8. FAQ

#### Q1：为什么 FFN‑MoE 不一定需要 `find_unused_parameters`？

FFN‑MoE 是 token 级路由，通常每个 step 不同 token 会覆盖更多专家；但在极端情况下仍可能出现某些专家未被选中。若你也遇到 DDP 报错，同样可以启用 `ddp_find_unused_parameters_true`。

#### Q2：为什么整层 MoE 采用“样本级路由”而不是 token 级？

整层专家包含 attention，attention 依赖 token 间关系。token 级路由会破坏 attention 的语义并显著增加实现复杂度与不稳定性。样本级路由是工程上更稳的折中。


