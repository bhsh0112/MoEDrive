# 训练中断和恢复指南

## 概述

PyTorch Lightning支持从checkpoint恢复训练。训练过程中会自动保存checkpoint，可以随时中断并继续训练。

## 当前训练状态

### 自动保存的Checkpoint

训练过程中，PyTorch Lightning会在以下位置自动保存checkpoint：

```
exp/training_transfuser_moe_multimodal_optimized/2026.01.12.01.01.54/lightning_logs/version_0/checkpoints/
```

### Checkpoint内容

每个checkpoint包含：
- ✅ 模型权重（model state_dict）
- ✅ 优化器状态（optimizer state）
- ✅ 学习率调度器状态（scheduler state，如果有）
- ✅ 当前epoch和step
- ✅ 训练状态信息

## 如何中断训练

### 方法1: 使用Ctrl+C（推荐）

在训练终端按 `Ctrl+C`：
- Lightning会优雅地保存当前状态
- 会保存一个 `last.ckpt` checkpoint（如果配置了）
- 可以安全地中断训练

### 方法2: 直接kill进程

如果必须强制中断：
```bash
# 查找训练进程
ps aux | grep run_training.py

# 终止进程（替换PID）
kill -SIGTERM <PID>
```

**注意**: 强制kill可能导致checkpoint不完整，建议使用Ctrl+C

## 如何继续训练

### 方法1: 使用resume_from_checkpoint参数（旧版本）

如果使用PyTorch Lightning < 2.0：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python navsim/planning/script/run_training.py \
  agent=transfuser_agent \
  experiment_name=training_transfuser_moe_multimodal_optimized \
  trainer.params.resume_from_checkpoint=exp/training_transfuser_moe_multimodal_optimized/2026.01.12.01.01.54/lightning_logs/version_0/checkpoints/epoch=XX-step=XXXX.ckpt \
  # ... 其他参数保持不变
```

### 方法2: 使用ckpt_path参数（新版本，推荐）

如果使用PyTorch Lightning >= 2.0：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python navsim/planning/script/run_training.py \
  agent=transfuser_agent \
  experiment_name=training_transfuser_moe_multimodal_optimized \
  trainer.params.ckpt_path=exp/training_transfuser_moe_multimodal_optimized/2026.01.12.01.01.54/lightning_logs/version_0/checkpoints/epoch=XX-step=XXXX.ckpt \
  # ... 其他参数保持不变
```

### 方法3: 使用专门的恢复脚本（推荐）

我已经创建了恢复训练脚本，见下文。

## 重要注意事项

### ✅ 必须保持一致的配置

恢复训练时，**必须使用与原始训练完全相同的配置**：
- ✅ `moe_num_experts`: 必须相同
- ✅ `num_trajectory_modes`: 必须相同
- ✅ `multimodal_trajectory`: 必须相同
- ✅ 所有模型架构参数必须相同
- ⚠️ 可以修改的参数：
  - `max_epochs`: 可以增加（继续训练更多epoch）
  - 学习率相关参数：可以调整（但需谨慎）

### ⚠️ 不能修改的参数

- 模型架构（层数、维度等）
- MoE专家数量
- 轨迹模式数量
- 数据加载配置

### ✅ 可以修改的参数

- `max_epochs`: 可以增加，继续训练更多轮
- 学习率（如果使用学习率调度器，会自动恢复）
- 损失权重（不推荐，但技术上可行）

## 检查Checkpoint

在恢复训练前，建议检查checkpoint：

```python
import torch

ckpt_path = "exp/training_transfuser_moe_multimodal_optimized/2026.01.12.01.01.54/lightning_logs/version_0/checkpoints/epoch=149-step=25050.ckpt"
ckpt = torch.load(ckpt_path, map_location='cpu')

print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"Global step: {ckpt.get('global_step', 'N/A')}")
print(f"Has optimizer: {'optimizer_states' in ckpt}")
print(f"Has scheduler: {'lr_schedulers' in ckpt}")
```

## 常见问题

### Q1: 训练中断后，从哪个checkpoint恢复？

**A**: 通常从最新的checkpoint恢复：
- 如果配置了`ModelCheckpoint`，使用最佳模型
- 否则使用最后一个epoch的checkpoint
- 检查checkpoint文件名中的epoch和step

### Q2: 可以改变训练配置吗？

**A**: 
- ✅ 可以增加`max_epochs`继续训练
- ❌ 不能改变模型架构参数
- ⚠️ 改变损失权重等可能影响训练，需谨慎

### Q3: 恢复训练后，训练日志会继续吗？

**A**: 
- 如果使用相同的`experiment_name`，日志会继续
- 如果使用新的`experiment_name`，会创建新的日志目录
- 建议使用相同的`experiment_name`以保持连续性

### Q4: DDP训练如何恢复？

**A**: 
- DDP训练恢复与单GPU训练相同
- 确保使用相同的GPU数量和配置
- Checkpoint会自动处理分布式状态

## 最佳实践

1. **定期检查checkpoint**: 训练过程中定期检查checkpoint是否正常保存
2. **保存最佳模型**: 配置`ModelCheckpoint`保存最佳模型
3. **记录配置**: 保存训练配置，方便恢复时参考
4. **测试恢复**: 在重要训练前，先测试恢复功能是否正常


