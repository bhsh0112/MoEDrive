# MOE-based Trajectory Head 设计方案

## 概述

本文档描述如何用 Mixture-of-Experts (MOE) 替换 diffusion-based trajectory head，以解决多模态轨迹预测问题。

## 核心思想

### Diffusion 方法的问题
- 需要多步去噪迭代，推理速度慢
- 训练和推理过程不一致（训练时添加噪声，推理时去噪）
- 去噪过程复杂，难以直接控制多模态输出

### MOE 方法的优势
- **直接生成**：每个 expert 直接生成一种轨迹模式，无需去噪过程
- **多模态自然**：多个 experts 自然对应多个轨迹模式
- **高效推理**：只需一次前向传播，推理速度快
- **可解释性**：Router 选择机制清晰，易于理解每个 expert 的作用

## 架构设计

### 1. MoETrajectoryExpert（单个 Expert）

每个 expert 负责生成一种轨迹模式，包含：

```
输入特征 (traj_feature)
    ↓
Cross-Attention with BEV features
    ↓
Cross-Attention with agent queries
    ↓
Cross-Attention with ego query
    ↓
Feed-Forward Network
    ↓
Trajectory Regression Head → (B, num_modes, num_poses, 3)
Trajectory Classification Head → (B, num_modes)  # 模式置信度
```

**关键组件**：
- `GridSampleCrossBEVAttention`: 从 BEV 特征中提取空间信息
- `MultiheadAttention`: 与 agent 和 ego query 交互
- `FFN`: 特征变换
- `trajectory_reg_head`: 回归轨迹坐标 (x, y, heading)
- `trajectory_cls_head`: 分类模式置信度

### 2. MoETrajectoryHead（MOE Head）

整体架构：

```
Plan Anchors (初始化轨迹模式)
    ↓
Plan Anchor Encoder → traj_feature
    ↓
Router (基于 ego_query) → 选择 top-k experts
    ↓
Expert 1 ──┐
Expert 2 ──┤
Expert 3 ──┼──→ 加权组合 → 最终轨迹预测
Expert 4 ──┘
```

**工作流程**：

1. **初始化**：从 plan anchors 生成初始轨迹特征
2. **路由**：Router 根据 ego query 选择 top-k experts
3. **生成**：每个选中的 expert 生成轨迹和置信度
4. **组合**：根据 router 权重组合 expert 输出
5. **选择**：根据置信度选择最佳轨迹模式

### 3. 与 Diffusion 方法的对比

| 特性 | Diffusion 方法 | MOE 方法 |
|------|----------------|----------|
| 生成方式 | 多步去噪迭代 | 直接生成 |
| 推理速度 | 慢（需要多步） | 快（单步） |
| 多模态 | 通过采样噪声实现 | 通过多个 experts 实现 |
| 训练/推理一致性 | 不一致 | 一致 |
| 可解释性 | 低 | 高（router 选择机制） |

## 实现细节

### 训练流程

```python
# 1. 初始化轨迹特征
plan_anchor = self.plan_anchor  # (num_modes, num_poses, 2)
traj_feature = encode(plan_anchor)  # (B, num_modes, d_model)

# 2. Router 选择 experts
router_logits = router(ego_query)  # (B, num_experts)
topk_idx, topk_w = select_top_k(router_logits, k=2)

# 3. 每个 expert 生成轨迹
for expert_id in topk_idx:
    poses_reg, poses_cls = experts[expert_id](
        traj_feature, plan_anchor, bev_feature, ...
    )

# 4. 加权组合
poses_reg = sum(weight * expert_output for expert, weight in zip(experts, weights))
poses_cls = sum(weight * expert_cls for expert, weight in zip(experts, weights))

# 5. 计算损失
loss = loss_computer(poses_reg, poses_cls, targets, plan_anchor)
load_balance_loss = compute_load_balance(router_logits)
total_loss = loss + load_balance_loss
```

### 推理流程

```python
# 1-4. 与训练相同
# 5. 选择最佳模式
mode_idx = poses_cls.argmax(dim=-1)
best_trajectory = poses_reg[batch_idx, mode_idx]
```

### 关键参数

- `num_experts`: MOE experts 数量（默认 4）
- `top_k`: 每个样本使用的 experts 数量（默认 2）
- `num_modes`: 轨迹模式数量（默认 20）
- `load_balance_coef`: 负载均衡损失系数（默认 0.01）

## 配置方式

在 `TransfuserConfig` 中添加：

```python
use_moe_trajectory: bool = True  # 是否使用 MOE trajectory head
moe_trajectory_num_experts: int = 4
moe_trajectory_top_k: int = 2
moe_trajectory_num_modes: int = 20
```

## 优势总结

1. **性能**：推理速度快，无需多步迭代
2. **多模态**：多个 experts 自然支持多模态预测
3. **可解释性**：Router 机制清晰，易于理解
4. **训练稳定性**：训练和推理一致，更稳定
5. **灵活性**：可以轻松调整 experts 数量和组合方式

## 迁移指南

### 从 Diffusion 迁移到 MOE

1. 确保 `use_moe_trajectory=True` 在配置中
2. 移除 diffusion scheduler 相关代码（如果不再需要）
3. 调整超参数（experts 数量、top_k 等）
4. 重新训练模型

### 兼容性

代码中保留了 `TrajectoryHead`（diffusion-based）作为 fallback，可以通过配置切换：
- `use_moe_trajectory=True`: 使用 MOE head
- `use_moe_trajectory=False`: 使用 diffusion head

## 未来改进方向

1. **动态 experts 数量**：根据场景复杂度动态调整
2. **Expert 专业化**：让不同 experts 专注于不同场景（如左转、右转、直行）
3. **更好的路由机制**：使用更复杂的 router（如 attention-based router）
4. **多级 MOE**：在不同层次使用 MOE（如 trajectory level 和 pose level）


