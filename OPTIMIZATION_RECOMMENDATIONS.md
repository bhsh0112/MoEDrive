# 多模态MoE模型训练优化建议

基于训练结果（100 epochs）的分析和优化建议。

## 当前配置分析

### 当前训练配置
- **MoE专家数量**: 20
- **轨迹模式数量**: 20
- **MoE top_k**: 20（使用所有专家）
- **损失权重**:
  - `trajectory_weight`: 10.0
  - `trajectory_mode_weight`: 1.0
  - `moe_aux_loss_weight`: 0.5
  - `trajectory_position_weight`: 1.0
  - `trajectory_heading_weight`: 1.0
- **MoE正则化**:
  - `moe_load_balance_coef`: 5e-3
  - `moe_router_z_loss_coef`: 1e-3
- **训练策略**: DDP with `find_unused_parameters=True`

## 优化建议

### 1. 学习率调度优化

**当前问题**: 可能使用固定学习率，导致后期训练不稳定

**建议**:
- 使用余弦退火学习率调度（CosineAnnealingLR）
- 或使用带预热的余弦退火（CosineAnnealingWarmRestarts）
- 添加学习率预热（Warmup）阶段

**优化配置**:
```yaml
trainer:
  params:
    max_epochs: 150  # 增加训练轮数
    accumulate_grad_batches: 1  # 梯度累积
    gradient_clip_val: 1.0  # 梯度裁剪
    gradient_clip_algorithm: norm
```

### 2. 损失权重平衡优化

**问题分析**:
- `trajectory_mode_weight` (1.0) 可能过低，导致模式分类学习不充分
- `moe_aux_loss_weight` (0.5) 可能影响专家负载均衡

**建议**:
- 提高 `trajectory_mode_weight` 到 2.0-3.0，加强模式分类
- 调整 `moe_aux_loss_weight` 到 0.3-0.4，避免过度正则化
- 根据训练曲线动态调整权重

**优化配置**:
```python
trajectory_mode_weight: 2.0  # 从1.0增加到2.0
moe_aux_loss_weight: 0.3  # 从0.5降低到0.3
```

### 3. MoE专家负载均衡优化

**问题**:
- 20个专家可能太多，导致某些专家学习不充分
- `moe_load_balance_coef` 可能需要调整

**建议**:
- 考虑减少专家数量到12-16个
- 增加 `moe_load_balance_coef` 到 1e-2，加强负载均衡
- 监控专家使用分布，确保所有专家都被使用

**优化配置**:
```python
moe_num_experts: 16  # 从20减少到16
num_trajectory_modes: 16  # 匹配专家数量
moe_load_balance_coef: 1e-2  # 从5e-3增加到1e-2
moe_top_k: 16  # 匹配专家数量（多模态模式）
```

### 4. 多模态轨迹损失优化

**问题**:
- 位置和航向权重可能不平衡
- 可能需要考虑不同时间步的重要性

**建议**:
- 调整 `trajectory_heading_weight` 到 1.5-2.0（航向通常更难学习）
- 考虑使用加权L1损失（对后期时间步给予更高权重）

### 5. 训练策略优化

**建议**:
- 使用混合精度训练（FP16/BF16）加速训练
- 增加验证频率，更及时监控过拟合
- 使用早停（Early Stopping）机制
- 添加模型检查点回调，保存最佳模型

**优化配置**:
```python
trainer:
  params:
    precision: 16  # 混合精度训练
    val_check_interval: 0.25  # 每25% epoch验证一次
    callbacks:
      - EarlyStopping:
          monitor: val_loss
          patience: 10
          mode: min
      - ModelCheckpoint:
          monitor: val_loss
          save_top_k: 3
          mode: min
```

### 6. 数据增强和正则化

**建议**:
- 增加Dropout（如果当前为0）
- 考虑添加轨迹平滑正则化
- 使用更丰富的数据增强

### 7. 评估指标优化

**建议**:
- 添加多模态评估指标（如MinADE, MinFDE）
- 监控专家使用分布
- 可视化不同模式的轨迹分布

## 实施优先级

1. **高优先级** (立即实施):
   - 损失权重调整（trajectory_mode_weight, moe_aux_loss_weight）
   - 学习率调度（余弦退火）
   - 梯度裁剪

2. **中优先级** (根据训练结果调整):
   - 专家数量调整（如果负载不均衡）
   - MoE正则化系数调整
   - 混合精度训练

3. **低优先级** (长期优化):
   - 数据增强
   - 评估指标扩展
   - 架构调整

## 监控指标

训练过程中应监控：
1. **损失指标**:
   - `loss` (总损失)
   - `trajectory_loss` (轨迹回归损失)
   - `trajectory_mode_loss` (模式分类损失)
   - `moe_aux_loss` (MoE辅助损失)
   - `moe_load_balance_loss` (负载均衡损失)

2. **MoE指标**:
   - `moe_usage_fraction_e{i}` (每个专家的使用率)
   - 专家负载均衡度

3. **验证指标**:
   - `val_loss`
   - 轨迹预测精度
   - 多模态预测覆盖率



