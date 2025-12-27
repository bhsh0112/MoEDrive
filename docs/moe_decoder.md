### MoE Decoder 设计与使用说明（`moe_transformer_decoder.py`）

本文档描述当前已实现的 **MoE Transformer Decoder**（用于替换 Transfuser 的 `_tf_decoder`）的设计目标、模块结构、接口契约、路由与损失项，以及集成与排错建议。

---

### 1. 背景与目标

在 `TransfuserModel` / `V2TransfuserModel` 中，`_tf_decoder` 的核心职责是：

- 将 **query tokens**（1 个轨迹 token + N 个 agent tokens）与 **memory tokens**（BEV 网格 token + status token）进行 cross-attention；
- 输出 `query_out`，并被下游的 `TrajectoryHead` 与 `AgentHead` 消费。

本 MoE Decoder 的目标是：

- **保持与 `torch.nn.TransformerDecoder` 一致的输入/输出张量形状契约**（`batch_first=True`）；
- **仅将 DecoderLayer 中的 FFN 替换为 MoE FFN**（保留 self-attn/cross-attn 的结构与语义）；
- 为训练稳定性提供必要的 **MoE 辅助损失（aux loss）与专家使用统计**，用于避免专家塌缩并便于调试。

当前实现位置：

- 代码文件：`DiffusionDrive/navsim/agents/moe_transformer_decoder.py`

> 重要：截至当前阶段，该模块是“独立实现”，尚未自动替换任何 transfuser 模型；需要后续集成步骤才能启用。

---

### 2. 模块组成

该文件实现了 4 个核心实体：

- **`MoEConfig`**
  - 用于配置专家数、top-k、负载均衡损失系数、router z-loss 系数、温度等。

- **`MoEFeedForward`**
  - Top‑k routed Mixture-of-Experts FFN。
  - 输入/输出保持 `(B, T, D)` 不变，并返回 `aux`（字典）用于训练监控与正则化。

- **`MoETransformerDecoderLayer`**
  - 一个 DecoderLayer：self-attn + cross-attn + MoE FFN。
  - 结构上遵循 Transformer 的“残差 + LayerNorm”范式，FFN 被替换为 MoE FFN。

- **`MoETransformerDecoder`**
  - 多层堆叠的 decoder。
  - 返回 `(output, aux)`，其中 `aux` 汇总了所有层的 MoE 辅助损失与使用统计。

---

### 3. 输入/输出接口契约（最关键）

#### 3.1 张量形状

所有模块均假设 `batch_first=True`，因此形状契约为：

- `tgt`（query / decoder 输入）形状：`(B, Q, D)`
- `memory`（key/value / encoder 输出）形状：`(B, S, D)`
- 输出 `output` 形状：`(B, Q, D)`

其中：

- `B`：batch size
- `Q`：query token 数（Transfuser 常为 `1 + num_bounding_boxes`）
- `S`：memory token 数（Transfuser 常为 `8*8 + 1`）
- `D`：`d_model`（例如 `config.tf_d_model`）

#### 3.2 `MoETransformerDecoder.forward` 返回值

与 `torch.nn.TransformerDecoder` 不同，这里返回两个对象：

1. `output: torch.Tensor`，形状 `(B, Q, D)`
2. `aux: Dict[str, torch.Tensor]`，包含：
   - `moe_aux_loss`: 标量（所有层的 MoE 辅助损失之和）
   - `moe_router_z_loss`: 标量（所有层 router z-loss 之和）
   - `moe_load_balance_loss`: 标量（所有层负载均衡损失之和）
   - `moe_usage_counts`: `(E,)` 每个 expert 的累计被选中次数（跨所有层）
   - `moe_usage_fraction`: `(E,)` 上述 counts 的归一化占比

> 若你希望“完全兼容原调用方式”（即只返回 tensor），集成时可选择：只取第一个返回值 `output`，aux loss 通过其它方式汇出（例如模型输出 dict 增加字段）。

---

### 4. MoE 路由逻辑（Top‑k routing）

MoE FFN 的路由粒度是 **token-level**（对 `x` 的每个 `(b, t)` token 独立路由）：

1. Router：`router_logits = Linear(D -> E)(x)`，形状 `(B, T, E)`
2. Top‑k：对最后一维取 `k` 个最大 logit
3. 权重：对 top‑k logits 做 softmax 得到 `topk_w`（归一化权重）
4. 聚合：对每个 expert，仅对路由到该 expert 的 token 子集做前向，按权重累加到输出

该策略的收益：

- Trajectory token 与 agent tokens 可以自然分流到不同专家；
- 不需要引入额外的场景标签即可学习“专门化”。

---

### 5. 专家（Experts）结构

每个 expert 是一个独立参数的 FFN，结构为：

`Linear(D -> D_ffn) -> Activation -> Dropout -> Linear(D_ffn -> D)`

其中：

- `D` = `d_model`
- `D_ffn` = `dim_feedforward`（对应原 `TransformerDecoderLayer.dim_feedforward`）
- Activation 当前支持 `relu`（默认）或 `gelu`

---

### 6. MoE 辅助损失与统计（避免塌缩/便于调试）

#### 6.1 负载均衡损失（Load balance / importance loss）

目的：避免所有 token 都路由到同一个 expert（专家塌缩）。

实现（Switch 常见形式）：

- `probs = softmax(router_logits)`，形状 `(B, T, E)`
- `importance = mean_{b,t}(probs)`，形状 `(E,)`
- `loss = E * sum_e importance[e]^2`

最终写入：

- `moe_load_balance_loss = loss * load_balance_coef`

#### 6.2 Router z-loss（稳定 router logits）

目的：抑制 router logits 过大导致训练不稳定。

实现：

- `z = logsumexp(router_logits)`，形状 `(B, T)`
- `loss = mean(z^2)`

最终写入：

- `moe_router_z_loss = loss * router_z_loss_coef`

#### 6.3 专家使用统计

统计方式：对 `topk_idx` 做计数（bincount），得到每个 expert 被选择的次数：

- `moe_usage_counts: (E,)`
- `moe_usage_fraction: (E,)`

这些指标用于判断是否出现塌缩（例如某个 expert 长期占比 > 0.8）。

---

### 7. 推荐的默认超参（建议从这里起步）

由于 Transfuser 的 query token 数较小，建议更偏向“稳”：

- `num_experts = 4`
- `top_k = 2`
- `load_balance_coef = 1e-2`（可在 `1e-3 ~ 1e-1` 搜索）
- `router_z_loss_coef = 1e-3`（可在 `0 ~ 1e-2` 搜索）
- `router_temperature = 1.0`（可在前期用 `>1` 更平滑，后期退火到 1）

---

### 8. 集成到 Transfuser 的建议方式（后续步骤）

你后续要做的最小集成方式是：

1. 在 `TransfuserConfig` 增加 MoE 开关与 MoE 参数；
2. 在模型构造函数中：
   - 若 `moe_enable=False`：使用原 `nn.TransformerDecoder`
   - 若 `moe_enable=True`：构造 `MoETransformerDecoder` 并替换 `_tf_decoder`
3. 在 `forward`：
   - 取 `query_out, aux = self._tf_decoder(query, keyval)`
   - `query_out` 走原有 split 与 heads
   - 将 `aux['moe_aux_loss']`、`aux['moe_usage_fraction']` 等加入输出 dict 以便训练记录

---

### 9. 常见风险与排查指南（强烈建议按这几项监控）

- **专家塌缩**（只用一个 expert）
  - 现象：`moe_usage_fraction` 长期单一 expert 占比极高
  - 处理：提高 `load_balance_coef`，使用 top‑2 而非 top‑1，增加温度/噪声（若后续加入）

- **训练不稳定 / NaN**
  - 现象：loss 波动大或突然 NaN，router logits 变极端
  - 处理：提高 `router_z_loss_coef`，对 router 参数单独更小 LR/clip

- **收益不明显**
  - 现象：替换后指标无提升
  - 处理：确认路由统计不塌缩；再考虑让 gate 输入加入更丰富的场景上下文（例如 pooled memory）

---

### 10. 当前实现的限制（需要你知情）

- 目前 `MoETransformerDecoder` 的 layer 复制方式是“从第一层推断超参并重新构造后续层”，目的是保持模块自包含与简洁；如你后续需要更严格/更可控的 layer 构造（例如每层独立 MoEConfig、不同 expert 数），应改为显式传参构造每层。
- 当前实现没有加入 capacity factor / token dropping 等大规模 MoE 机制（在 Transfuser 的小 token 场景一般不需要）。


