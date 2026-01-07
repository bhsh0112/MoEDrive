# MoE规划器方案概述

## 📖 什么是MoE（Mixture of Experts）？

想象一下，你有一个规划任务，不同的驾驶场景（比如高速、城市、转弯）可能需要不同的策略。传统的神经网络对所有场景使用同一套参数，而MoE（专家混合）的思路是：

- **训练多个"专家"网络**，每个专家可能擅长处理不同类型的场景
- **有一个"路由器"（Router）**，根据当前输入决定调用哪些专家
- **每次只激活部分专家**（例如8个专家中选2个），这样既能增加模型容量，又不会让计算量爆炸

这就好比医院有不同科室的专家，根据病人症状由分诊台（路由器）决定看哪个专家。

---

## 🎯 为什么采用MoE方案？

### 核心动机

我们选择用MoE替换DiffusionDrive原有的Transformer Decoder，主要基于以下考虑：

1. **增加模型容量而不显著增加计算成本**：
   - 传统方案：增加模型参数会线性增加训练和推理的计算量
   - MoE方案：通过只激活部分专家（例如8个专家中选2个），可以用更少的计算量获得更大的模型容量
   - **实际效果**：8个专家提供了8倍的理论容量，但每次只激活2个，计算量只增加约2倍

2. **实现专家专门化**：
   - 不同的驾驶场景（高速、城市、转弯、避障等）可能需要不同的规划策略
   - MoE允许不同的专家学习处理不同类型的场景，实现"术业有专攻"
   - 路由器可以自动学习根据当前场景选择合适的专家组合

3. **缓解梯度冲突**（参考ScaleZero论文的思路）：
   - 在多任务学习中，不同任务可能对同一参数有不同的梯度方向，导致梯度冲突
   - MoE通过将不同的任务分配给不同的专家，可以缓解这种冲突
   - 在我们的场景中，不同的驾驶场景可以视为"隐式的多任务"

4. **灵活的模型扩展**：
   - 可以通过增加专家数量（`moe_num_experts`）来提升模型容量
   - 可以通过调整激活的专家数量（`moe_top_k`）来平衡性能和效率
   - 不需要重新设计整个架构

### 为什么选择Layer-wise MoE而不是FFN-MoE？

我们最终选择了**Layer-wise MoE**（对整个Decoder层进行MoE化），而不是更常见的**FFN-MoE**（只对FFN层进行MoE化），原因如下：

1. **更激进的容量提升**：
   - FFN-MoE：只增加FFN部分的容量（通常占参数的2/3左右）
   - Layer-wise MoE：增加整个Decoder层的容量（包括self-attention、cross-attention和FFN）
   - **效果**：Layer-wise MoE可以提供更大的容量提升，理论上更有利于复杂场景的学习

2. **更适合规划任务的特点**：
   - 规划任务需要综合处理视觉、激光雷达、状态等多种模态信息
   - Layer-wise MoE允许不同的专家学习不同的注意力模式和信息融合策略
   - 相比只MoE化FFN，Layer-wise MoE提供了更灵活的表达能力

3. **实验验证**：
   - 虽然Layer-wise MoE在实现上更复杂，但我们的实验表明这种方式可以带来更好的性能
   - 通过合理的路由策略（使用轨迹token作为路由输入），可以实现任务相关的专家选择

### 潜在挑战与解决方案

采用MoE方案也带来了一些挑战，我们的实现已经考虑了这些：

1. **专家使用不均衡（Expert Collapse）**：
   - **挑战**：某些专家可能被过度使用，而其他专家被忽略
   - **解决方案**：通过负载均衡损失（`moe_load_balance_coef`）鼓励专家均匀使用

2. **训练不稳定**：
   - **挑战**：路由器的logits可能过大，导致训练不稳定
   - **解决方案**：通过Router Z-loss（`moe_router_z_loss_coef`）稳定训练

3. **分布式训练兼容性**：
   - **挑战**：MoE的top-k路由会导致某些专家在某个batch中不被使用，DDP会报错
   - **解决方案**：使用`ddp_find_unused_parameters_true`策略，允许未使用的参数

4. **路由策略选择**：
   - **挑战**：如何设计有效的路由策略，让路由器选择最合适的专家
   - **解决方案**：使用轨迹token（query的第一个token）作为路由输入，让路由更直接关联规划任务

---

## 🎯 当前实现方案：Layer-wise MoE Transformer Decoder

### 方案概述

我们将DiffusionDrive原本的Transformer Decoder替换成了**Layer-wise MoE Transformer Decoder**。这是比简单的FFN-MoE更激进的方案：

- **专家是完整的Decoder层**：每个专家包含self-attention、cross-attention和FFN的全部组件
- **样本级路由**：路由器根据每个样本（batch中的每个样本）的特征，选择最合适的专家组合
- **Top-k选择**：每个样本选择`top_k`个专家（默认2个），然后加权融合它们的输出

### 架构示意图

```
输入 (query, key-value) 
    ↓
Layer 1:
  ├─ Router → 选择 top-k 专家 (例如：专家3和专家7)
  ├─ 专家3 处理 → 输出A (权重0.6)
  ├─ 专家7 处理 → 输出B (权重0.4)
  └─ 加权融合 → Layer 1输出
    ↓
Layer 2:
  ├─ Router → 选择 top-k 专家
  └─ ... (类似过程)
    ↓
输出 (增强的query特征)
```

### 关键技术点

1. **路由输入使用轨迹token**：路由器使用query序列中的第一个token（轨迹token），而不是所有token的平均，这样路由更直接关联规划任务。

2. **辅助损失（Auxiliary Loss）**：
   - **负载均衡损失**：鼓励所有专家被均匀使用，避免某些专家被"冷落"
   - **Router Z-loss**：稳定路由器的logits，防止训练不稳定

3. **DDP训练支持**：由于MoE的top-k路由会导致某些专家在某个batch中不被使用，需要启用`find_unused_parameters=True`来支持分布式训练。

### 专家分工机制：隐式学习 vs 显式设计

**重要说明**：8个专家的"分工"**不是预先设计的**，而是通过**训练过程中的自组织学习**实现的。

#### 初始状态

- 所有专家在初始化时**完全相同**（使用PyTorch的默认初始化）
- 每个专家都是完整的Decoder层（self-attention + cross-attention + FFN）
- 路由器也是随机初始化的

#### 学习过程

专家之间的分工是通过以下机制自动学习出来的：

1. **路由器学习选择策略**：
   - 路由器根据轨迹token（query的第一个token）的特征，学习为不同类型的输入选择不同的专家
   - 例如：高速场景的轨迹token可能路由到专家0和专家3，城市场景可能路由到专家1和专家5

2. **专家通过梯度更新分化**：
   - 不同的样本被路由到不同的专家
   - 通过反向传播，每个专家接收到不同样本的梯度
   - 随着训练进行，不同专家逐渐学习到不同的处理模式

3. **负载均衡损失促进分化**：
   - 负载均衡损失确保所有专家都被使用
   - 这避免了"专家塌缩"（所有样本都路由到同一个专家）
   - 促进了专家之间的自然分化

#### 可能的专家分工（隐式学习）

虽然分工是隐式的，但训练得当的话，不同的专家可能会学习到：

- **场景类型**：
  - 专家0、3：擅长处理高速场景
  - 专家1、5：擅长处理城市道路
  - 专家2、6：擅长处理转弯场景
  - 专家4、7：擅长处理避障场景

- **注意力模式**：
  - 某些专家可能更关注前方的车辆
  - 某些专家可能更关注侧方的障碍物
  - 某些专家可能更关注道路边界

- **特征组合方式**：
  - 不同的专家可能以不同的方式融合视觉和激光雷达特征
  - 不同的专家可能对不同的语义特征有不同的权重

#### 如何观察专家的"专长"？

虽然分工是隐式的，但我们可以通过以下方式观察：

1. **专家使用率统计**：
   - 通过TensorBoard查看`train/moe_usage_fraction_e{i}_step`
   - 如果使用率相对均匀（每个约12.5%），说明分工较均衡

2. **场景级别的专家激活分析**（需要额外实现）：
   - 在推理时记录不同场景下哪些专家被激活
   - 分析高速/城市/转弯场景的专家激活模式
   - 可以揭示专家与场景类型的关联

3. **专家输出的特征分析**（需要额外实现）：
   - 对比不同专家对相同输入的输出特征
   - 使用可视化技术（如t-SNE）分析专家输出的分布
   - 可以揭示专家处理模式的差异

**注意**：当前的实现中，专家分工是**完全隐式的**，没有强制性的专业化约束。这意味着：
- ✅ 优点是灵活性高，专家可以自由学习最优分工
- ⚠️ 缺点是我们无法直接"读懂"每个专家的专长，需要额外的分析工具

---

## ⚙️ 主要配置参数

在`transfuser_config.py`中可以配置以下MoE相关参数：

### 核心参数

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `moe_num_experts` | 8 | 专家数量。更多专家=更大容量，但训练和推理成本更高 |
| `moe_top_k` | 2 | 每个样本选择的专家数量。必须 ≤ `moe_num_experts` |

### 路由参数

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `moe_router_temperature` | 1.0 | 路由器温度。>1.0使路由更平滑，<1.0使路由更尖锐（更确定选择） |

### 正则化参数（辅助损失系数）

| 参数名 | 默认值 | 说明 |
|--------|--------|------|
| `moe_load_balance_coef` | 5e-3 | 负载均衡损失系数。越大，越强制专家被均匀使用 |
| `moe_router_z_loss_coef` | 1e-3 | Router Z-loss系数。用于稳定训练 |
| `moe_aux_loss_weight` | 0.5 | MoE辅助损失的全局权重（添加到总训练损失中） |

### 参数调优建议

- **如果专家使用率严重不均**（某些专家几乎不用）：适度增加`moe_load_balance_coef`（例如1e-2）
- **如果路由过于集中**（总是选同样的专家）：可以尝试降低`moe_aux_loss_weight`（例如0.2-0.3），让路由器更自由地学习
- **如果训练不稳定**：检查`moe_router_z_loss_coef`是否足够大

---

## 🚀 如何使用

### 1. 训练

使用提供的训练脚本：

```bash
# 修改 scripts/train_moe.sh 中的GPU配置和超参数
CUDA_VISIBLE_DEVICES=4,5,6,7 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  agent=diffusiondrive_agent \
  experiment_name=your_experiment_name \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=100 \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  +agent.config.moe_aux_loss_weight=0.2  # 覆盖默认配置
```

**注意**：
- 必须使用`ddp_find_unused_parameters_true`策略（MoE动态路由需要）
- 可以通过`+agent.config.参数名=值`的方式覆盖任何MoE配置参数

### 2. 监控训练

训练过程中，TensorBoard会记录以下MoE相关指标：

- `train/moe_aux_loss_step`：MoE辅助损失（负载均衡+router z-loss）
- `train/moe_load_balance_loss_step`：负载均衡损失
- `train/moe_router_z_loss_step`：Router Z-loss
- `train/moe_usage_fraction_e{i}_step`：每个专家（i=0到7）的使用率

**健康指标判断**：
- **专家使用率**：理想情况下，8个专家的使用率应该相对均匀（每个约12.5%）。如果某个专家使用率>50%或<5%，说明路由可能有问题
- **Perplexity**：专家使用分布的熵的指数。8专家均匀分布时，perplexity≈8。如果perplexity<4，说明路由过于集中
- **moe_aux_loss/loss比例**：通常应该在0.1%-1%之间。如果>5%，说明辅助损失可能过强，干扰主任务学习

### 3. 导出TensorBoard指标

使用提供的脚本导出训练指标：

```bash
python scripts/export_tb_scalars.py \
  --event_file <训练目录>/lightning_logs/version_0/events.out.tfevents.* \
  --out_csv /tmp/tb_scalars_export.csv
```

---

## 📊 性能优化经验

基于之前的实验，以下是一些经验总结：

### ✅ 有效的配置

- **去除路由器噪声**：噪声会干扰路由学习
- **使用轨迹token作为路由输入**：相比平均池化，直接使用轨迹token让路由更任务相关
- **适度的负载均衡**：`moe_load_balance_coef=5e-3`（默认值）通常效果较好
- **适度的辅助损失权重**：`moe_aux_loss_weight=0.2-0.5`之间

### ❌ 需要注意的问题

- **过度增加负载均衡系数**：虽然理论上应该让专家使用更均匀，但实验发现增加到`2e-2`反而让路由更集中（可能因为梯度冲突）
- **路由器噪声**：实验显示噪声没有带来正向收益
- **过于激进的配置**：`top_k=1`、极高的辅助损失权重等配置可能导致性能下降

### 调优策略

1. **从默认配置开始**：先用默认参数训练，观察专家使用率和性能
2. **如果路由过于集中**：
   - 尝试降低`moe_aux_loss_weight`（例如0.2）
   - 或尝试增加`moe_top_k`（例如3）
3. **如果路由过于分散但性能不佳**：
   - 适度增加`moe_load_balance_coef`（但要谨慎，建议每次增加50%）
4. **记录每次实验**：记录配置、专家使用率、评测分数，建立自己的调优数据库

---

## 🔧 技术细节

### 文件结构

- **核心实现**：`navsim/agents/moe_transformer_decoder.py`
  - `MoELayerwiseTransformerDecoder`：Layer-wise MoE解码器
  - `MoEConfig`：MoE配置数据类
  - `MoEFullDecoderLayer`：完整的Decoder层（用作专家）

- **模型集成**：
  - `navsim/agents/diffusiondrive/transfuser_model_v2.py`
  - `navsim/agents/transfuser/transfuser_model.py`

- **配置**：
  - `navsim/agents/diffusiondrive/transfuser_config.py`
  - `navsim/agents/transfuser/transfuser_config.py`

- **损失函数**：
  - `navsim/agents/diffusiondrive/transfuser_loss.py`
  - `navsim/agents/transfuser/transfuser_loss.py`

### 与标准TransformerDecoder的区别

| 特性 | 标准TransformerDecoder | MoELayerwiseTransformerDecoder |
|------|----------------------|-------------------------------|
| 每一层 | 固定的一组参数 | E个专家（可选的参数组） |
| 路由 | 无 | 每层有路由器选择top-k专家 |
| 输出 | 直接输出 | 加权融合多个专家的输出 |
| 辅助损失 | 无 | 负载均衡损失 + Router Z-loss |
| 计算量 | 固定 | 可动态调整（通过调整top_k和E） |

---

## 📝 总结

当前的MoE方案通过**Layer-wise专家路由**，在保持计算效率的同时增加了模型容量。关键是通过**轨迹token路由**和**适度的正则化**，让路由器学习到任务相关的专家选择策略。

**下一步优化方向**：
- 继续探索超参数组合（特别是`moe_aux_loss_weight`和`moe_load_balance_coef`的平衡）
- 分析不同场景下哪些专家被激活，理解专家的"专长"
- 考虑更高级的路由策略（例如条件路由、层级路由等）

---

## 📚 参考资料与实现来源

### 实现来源说明

**重要澄清**：当前的Layer-wise MoE Transformer Decoder实现**不是直接借鉴自LightZero**，而是：

1. **参考了ScaleZero论文的思想**：
   - ScaleZero论文（arXiv:2509.07945）提出了多任务MoE世界模型，使用MoE缓解梯度冲突
   - 我们借鉴了其"用MoE实现专家专门化"的核心思想

2. **基于通用的MoE架构设计**：
   - 实现采用标准的top-k routing机制（类似Switch Transformer、GShard等）
   - 使用负载均衡损失和router z-loss（MoE领域的常见技术）
   - 这是一个**自包含的实现**（self-contained implementation），没有直接复制任何代码库

3. **LightZero是什么**：
   - LightZero（https://github.com/opendilab/LightZero）是一个**MCTS+RL框架**，主要用于游戏AI训练
   - 它不是MoE架构，而是一个强化学习工具库
   - ScaleZero论文中提到LightZero作为实验框架，但LightZero本身并不包含MoE Transformer的实现

### 参考资料

- 技术实现细节：`docs/moe_tf_decoder_migration.md`
- 完整替换计划：`docs/tf_decoder_full_replacement_plan.md`
- ScaleZero论文（MoE思想参考）：https://arxiv.org/abs/2509.07945
- 通用MoE架构参考：
  - Switch Transformer（Google, 2021）
  - GShard（Google, 2020）
  - 其他top-k routing MoE实现

