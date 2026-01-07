# MOE 替换 Diffusion Trajectory Head - 实施总结

## 已完成的工作

### 1. 创建了 MOE-based Trajectory Head
- **文件**: `navsim/agents/diffusiondrive/modules/moe_trajectory_head.py`
- **核心组件**:
  - `MoETrajectoryExpert`: 单个 expert，负责生成一种轨迹模式
  - `MoETrajectoryHead`: MOE head，管理多个 experts 和 router

### 2. 修改了主模型
- **文件**: `navsim/agents/diffusiondrive/transfuser_model_v2.py`
- **改动**:
  - 添加了 `use_moe_trajectory` 配置选项
  - 支持在 MOE 和 Diffusion 之间切换
  - 默认使用 MOE trajectory head

### 3. 设计文档
- **文件**: `docs/moe_trajectory_head_design.md`
- 包含详细的架构设计、实现细节和迁移指南

## 核心设计理念

### MOE vs Diffusion

| 方面 | Diffusion | MOE |
|------|-----------|-----|
| **生成方式** | 多步去噪迭代 | 直接生成 |
| **推理速度** | 慢（需要多步） | 快（单步） |
| **多模态** | 通过采样噪声 | 通过多个 experts |
| **训练/推理一致性** | 不一致 | 一致 |
| **可解释性** | 低 | 高 |

### MOE 架构

```
Plan Anchors (初始化)
    ↓
Plan Anchor Encoder
    ↓
Router (基于 ego_query) → 选择 top-k experts
    ↓
Expert 1 ──┐
Expert 2 ──┤
Expert 3 ──┼──→ 加权组合 → 最终轨迹
Expert 4 ──┘
```

## 使用方法

### 配置

在 `TransfuserConfig` 中添加以下参数：

```python
use_moe_trajectory: bool = True  # 启用 MOE trajectory head
moe_trajectory_num_experts: int = 4  # Expert 数量
moe_trajectory_top_k: int = 2  # 每个样本使用的 experts 数量
moe_trajectory_num_modes: int = 20  # 轨迹模式数量
```

### 训练

训练脚本会自动使用 MOE trajectory head（如果 `use_moe_trajectory=True`）。

### 推理

推理时会自动使用 MOE head，无需额外配置。

## 关键特性

1. **直接生成**: 无需去噪过程，推理速度快
2. **多模态支持**: 多个 experts 自然支持多模态预测
3. **可解释性**: Router 机制清晰，易于理解
4. **向后兼容**: 保留了 diffusion head 作为 fallback

## 下一步建议

1. **超参数调优**:
   - 调整 `num_experts`（建议 4-8）
   - 调整 `top_k`（建议 2-3）
   - 调整 `load_balance_coef`（建议 0.01-0.1）

2. **实验验证**:
   - 对比 MOE 和 Diffusion 的性能
   - 验证多模态预测质量
   - 检查推理速度提升

3. **进一步优化**:
   - Expert 专业化（让不同 experts 专注不同场景）
   - 更复杂的 router 机制
   - 动态 experts 数量

## 注意事项

1. **Plan Anchors**: MOE head 仍然使用 plan anchors 作为初始化，这与 diffusion 方法相同
2. **Loss Function**: 使用相同的 `LossComputer`，确保训练目标一致
3. **兼容性**: 代码中保留了 diffusion head，可以通过配置切换

## 文件清单

- ✅ `navsim/agents/diffusiondrive/modules/moe_trajectory_head.py` - MOE head 实现
- ✅ `navsim/agents/diffusiondrive/transfuser_model_v2.py` - 主模型修改
- ✅ `docs/moe_trajectory_head_design.md` - 详细设计文档
- ✅ `docs/moe_trajectory_replacement_summary.md` - 本总结文档

## 测试建议

1. **单元测试**: 测试 MOE head 的前向传播
2. **集成测试**: 测试与主模型的集成
3. **性能测试**: 对比推理速度和内存使用
4. **质量测试**: 验证轨迹预测质量


