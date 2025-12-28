### `_tf_decoder` → MoE 的整体说明文档（MOEDrive）

本文档说明在 **MOEDrive** 中如何将 Transfuser 系列模型的 `_tf_decoder`（原 `torch.nn.TransformerDecoder`）**完全替换**为 **MoE（Mixture-of-Experts）Transformer Decoder**，并打通训练期 MoE 正则与指标记录，保证训练、评测可验证、可复现。

---

### 1. 改动目标与范围

#### 1.1 目标

- 将以下两处模型中的 `_tf_decoder` 从 **vanilla TransformerDecoder** 替换为 **MoETransformerDecoder**：
  - `MOEDrive/navsim/agents/transfuser/transfuser_model.py`
  - `MOEDrive/navsim/agents/diffusiondrive/transfuser_model_v2.py`
- 替换后保持下游 head 的输入输出契约不变：
  - `query_out` 仍然是 `(B, Q, D)`，能继续 `split(self._query_splits, dim=1)`，并被 trajectory/agent head 使用。
- 训练时让 MoE 正则真实生效：
  - `moe_aux_loss` 加入总 loss，用于防止专家塌缩并稳定 router。
- 训练时自动记录 MoE 指标到 Lightning logger：
  - `train/moe_aux_loss`
  - `train/moe_load_balance_loss`
  - `train/moe_router_z_loss`
  - `train/moe_usage_fraction_e0...e{E-1}`（每个专家一个标量）

#### 1.2 范围说明（我们做了什么、没做什么）

- **做了**：只在 Transformer decoder layer 内部把 FFN 替换为 MoE FFN（保留 self-attn/cross-attn）。
- **没做**：没有把整个 planning diffusion decoder 替换为 MoE planner（那是更大范围的改动链路）。

---

### 2. 设计原则：为什么只替换 FFN 为 MoE

MoE 常见且最稳定的落点是 Transformer block 的 **FFN 子层**：

- self-attn/cross-attn 负责对齐与检索信息，是 Transfuser 的结构核心，贸然 MoE 化注意力更容易引入不稳定和行为漂移；
- FFN 是容量瓶颈与梯度冲突聚集点，MoE 在这里可实现“专门化”，与多任务 MoE（例如 ScaleZero）缓解冲突的动机一致；
- Transfuser 的 query token 数很小（`1 + num_bounding_boxes`），MoE 计算开销低、可控。

---

### 3. 关键文件与模块说明

#### 3.1 MoE decoder 的实现文件

**文件**：`MOEDrive/navsim/agents/moe_transformer_decoder.py`

包含模块：

- `MoEConfig`
  - MoE 结构与正则项的配置（专家数、top-k、温度、loss 系数等）。
- `MoEFeedForward`
  - token-level top‑k 路由 MoE FFN；
  - 返回 `(y, aux)`，其中 `aux` 包含：
    - `moe_aux_loss`
    - `moe_load_balance_loss`
    - `moe_router_z_loss`
    - `moe_usage_counts / moe_usage_fraction`
- `MoETransformerDecoderLayer`
  - 结构：self-attn → cross-attn → MoE FFN（残差+LayerNorm）。
- `MoETransformerDecoder`
  - 堆叠多层 layer，并汇总所有层的 aux loss 与 usage 统计；
  - 返回 `(output, aux)`。

#### 3.2 替换 `_tf_decoder` 的文件

- `MOEDrive/navsim/agents/transfuser/transfuser_model.py`
- `MOEDrive/navsim/agents/diffusiondrive/transfuser_model_v2.py`

替换点：

- `self._tf_decoder = nn.TransformerDecoder(...)`
  → `self._tf_decoder = MoETransformerDecoder(...)`

并把 forward 中的调用从：

- `query_out = self._tf_decoder(query, keyval)`

改为：

- `query_out, moe_aux = self._tf_decoder(query, keyval)`

然后把 `moe_aux` 的关键信息写入 `output` dict：

- `moe_aux_loss`
- `moe_load_balance_loss`
- `moe_router_z_loss`
- `moe_usage_fraction`
- `moe_usage_counts`

#### 3.3 训练 loss 与 logging 的关键链路

Lightning wrapper：`MOEDrive/navsim/planning/training/agent_lightning_module.py`

该 wrapper 在每个 step 会：

- `prediction = agent.forward(...)`
- `loss_dict = agent.compute_loss(...)`
- 对 `loss_dict` 中的每个 item 调用 `self.log(f"{prefix}/{k}", v, ...)`

因此要让 MoE 指标“自动出现在训练日志里”，需要满足：

- `transfuser_loss` 返回 `loss_dict`（而非单标量）；
- `loss_dict` 里是 **标量 tensor**（向量会导致无法直接 log）。

我们做了两处改动：

- `MOEDrive/navsim/agents/transfuser/transfuser_loss.py`
  - 从返回标量改为返回 `loss_dict`；
  - 将 `moe_usage_fraction` 向量展开成 `moe_usage_fraction_e{i}`。
- `MOEDrive/navsim/agents/diffusiondrive/transfuser_loss.py`
  - 原本就返回 `loss_dict`；
  - 增加记录 `moe_load_balance_loss / moe_router_z_loss`；
  - 增加 `moe_usage_fraction_e{i}` 标量项。

---

### 4. 接口契约（张量形状与兼容性）

MoE decoder 与 PyTorch decoder 的关键契约一致（`batch_first=True`）：

- `tgt/query`：`(B, Q, D)`
- `memory/keyval`：`(B, S, D)`
- 输出 `query_out`：`(B, Q, D)`

其中：

- `Q = 1 + num_bounding_boxes`（trajectory token + agent tokens）
- `S = 8*8 + 1`（BEV 网格 token + status token）
- `D = tf_d_model`

这保证下游逻辑：

- `trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)`

不需要改。

---

### 5. MoE 路由与正则项说明

#### 5.1 路由（Top‑k routing）

对每个 token `(b, t)`：

- `router_logits = Linear(D -> E)(x)` → 选择 top‑k experts；
- 对 top‑k logits 做 softmax 得到权重；
- 只对被选中的专家执行前向并加权聚合。

#### 5.2 负载均衡损失（Load balance）

目的：防止专家塌缩（所有 token 都走同一个 expert）。

实现：Switch 风格 importance loss（对 router softmax 概率在 batch/token 上求均值并做平方惩罚）。

#### 5.3 Router z-loss

目的：防止 router logits 过大导致训练不稳定。

实现：对 `logsumexp(router_logits)` 的平方均值做惩罚。

#### 5.4 总 aux loss

`moe_aux_loss = moe_load_balance_loss + moe_router_z_loss`

并被加入训练总 loss：

`total_loss += moe_aux_loss_weight * moe_aux_loss`

---

### 6. 推荐默认超参（当前仓库默认值）

在两份 `TransfuserConfig` 中，我们写入了以下默认值（建议作为起点）：

- `moe_num_experts = 4`
- `moe_top_k = 2`
- `moe_router_temperature = 1.0`
- `moe_load_balance_coef = 1e-2`
- `moe_router_z_loss_coef = 1e-3`
- `moe_aux_loss_weight = 1.0`

调参建议：

- 如果专家塌缩：提高 `moe_load_balance_coef`（例如 `1e-2 → 5e-2`）
- 如果训练不稳定/NaN：提高 `moe_router_z_loss_coef` 或降低 router 学习率（更进一步的改动）

---

### 7. 训练命令（Hydra struct 模式的注意点）

Hydra/OmegaConf 开启了 struct 模式时，如果配置里没有预声明字段，命令行覆盖必须使用 `+`：

- 错：`agent.config.moe_num_experts=4`
- 对：`+agent.config.moe_num_experts=4`

如果你已经在 `TransfuserConfig` 里写入了字段（本仓库当前已写入），则一般不需要 `+`；但在某些配置组合/struct 严格模式下仍建议保留 `+` 以保证可覆盖。

---

### 8. 如何验证 MoE 是否生效

#### 8.1 训练期验证（强烈推荐）

Lightning event 文件中应出现：

- `train/moe_aux_loss`
- `train/moe_load_balance_loss`
- `train/moe_router_z_loss`
- `train/moe_usage_fraction_e0...`

> 如果看到这些 key，说明 MoE forward 输出被 loss 消费并被 logger 记录，MoE 正则链路已打通。

#### 8.2 快速 sanity check（不依赖数据集）

脚本：`MOEDrive/scripts/sanity_check_moe_decoder.py`

验证：

- 输出 shape 正确；
- aux loss 非零；
- usage fraction 不塌缩。

#### 8.3 logging smoke test（不依赖数据集）

脚本：`MOEDrive/scripts/smoke_test_moe_logging.py`

验证：

- `transfuser_loss` 返回 `loss_dict` 且均为标量；
- 包含 `moe_*` 与 `moe_usage_fraction_e*`。

---

### 9. 评测（PDM score / PDMS）为什么终端不一定打印

`run_pdm_score.py` 的最终汇总输出通过 logger 写出，通常会写到：

- `exp/<experiment_name>/<timestamp>/log.txt`

最终 csv 会写到：

- `exp/<experiment_name>/<timestamp>/<timestamp>.csv`

并且 `token=average` 那一行包含最终平均 `score`。

因此：终端不回显 summary ≠ 没有评测结果；优先查 `log.txt` 和 csv 最后一行。

---

### 10. 常见问题（FAQ）

#### Q1：为什么评测时看不到 `moe_*`？

`moe_*` 是训练期/日志期指标，不属于 PDMS 评测脚本必须输出内容；PDMS 输出在 `log.txt` 与 csv 的 `score` 列（`average` 行）。

#### Q2：为什么需要把 `moe_usage_fraction` 展开成 e0,e1...？

Lightning 的 `self.log` 更适合标量；向量可能无法直接记录或显示不友好。展开后更易监控是否塌缩。

---

### 11. 变更清单（便于 code review）

- 新增：
  - `MOEDrive/navsim/agents/moe_transformer_decoder.py`
  - `MOEDrive/scripts/sanity_check_moe_decoder.py`
  - `MOEDrive/scripts/smoke_test_moe_logging.py`
- 修改：
  - `MOEDrive/navsim/agents/transfuser/transfuser_model.py`
  - `MOEDrive/navsim/agents/diffusiondrive/transfuser_model_v2.py`
  - `MOEDrive/navsim/agents/transfuser/transfuser_loss.py`
  - `MOEDrive/navsim/agents/diffusiondrive/transfuser_loss.py`
  - `MOEDrive/navsim/agents/transfuser/transfuser_config.py`
  - `MOEDrive/navsim/agents/diffusiondrive/transfuser_config.py`


