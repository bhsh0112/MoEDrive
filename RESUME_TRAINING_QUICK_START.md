# 训练恢复快速指南

## ✅ 是的，训练可以中断并继续！

PyTorch Lightning会自动保存checkpoint，支持随时中断和恢复训练。

## 快速开始

### 1. 找到最新的Checkpoint

```bash
# 查看最新的checkpoint
ls -lht exp/training_transfuser_moe_multimodal_optimized/*/lightning_logs/version_0/checkpoints/*.ckpt | head -1
```

### 2. 使用恢复脚本（推荐）

```bash
# 方法1: 使用ckpt_path（PyTorch Lightning 2.0+）
bash scripts/train_moe_multimodal_resume.sh \
  exp/training_transfuser_moe_multimodal_optimized/2026.01.12.01.01.54/lightning_logs/version_0/checkpoints/epoch=149-step=25050.ckpt \
  200

# 方法2: 如果方法1不工作，使用resume_from_checkpoint（旧版本）
bash scripts/train_moe_multimodal_resume_alt.sh \
  exp/training_transfuser_moe_multimodal_optimized/2026.01.12.01.01.54/lightning_logs/version_0/checkpoints/epoch=149-step=25050.ckpt \
  200
```

### 3. 手动恢复（如果脚本不工作）

在训练脚本中添加checkpoint路径：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python navsim/planning/script/run_training.py \
  agent=transfuser_agent \
  experiment_name=training_transfuser_moe_multimodal_optimized \
  trainer.params.ckpt_path="exp/.../checkpoints/epoch=149-step=25050.ckpt" \
  trainer.params.max_epochs=200 \
  # ... 其他参数保持不变
```

## 重要注意事项

### ✅ 必须保持一致

- **模型配置**: 所有模型参数必须与原始训练相同
  - `moe_num_experts=20`
  - `num_trajectory_modes=20`
  - `multimodal_trajectory=True`
  - 所有架构参数

### ✅ 可以修改

- `max_epochs`: 可以增加（如从150增加到200）
- `experiment_name`: 建议保持相同以继续日志

### ❌ 不能修改

- 模型架构参数
- MoE专家数量
- 数据加载配置

## 检查Checkpoint

恢复前检查checkpoint是否完整：

```python
import torch

ckpt = torch.load("path/to/checkpoint.ckpt", map_location='cpu')
print(f"Epoch: {ckpt.get('epoch')}")
print(f"Step: {ckpt.get('global_step')}")
print(f"Has optimizer: {'optimizer_states' in ckpt}")
```

## 常见问题

**Q: 训练中断后，从哪个checkpoint恢复？**  
A: 从最新的checkpoint恢复（epoch和step最大的）

**Q: 可以改变训练配置吗？**  
A: 可以增加max_epochs，但不能改变模型架构参数

**Q: 恢复后训练会从哪里继续？**  
A: 从checkpoint保存的epoch和step继续，不会重复训练

**Q: 如果checkpoint损坏怎么办？**  
A: 尝试使用更早的checkpoint，或重新开始训练

## 示例

### 从epoch 149继续训练到200

```bash
bash scripts/train_moe_multimodal_resume.sh \
  exp/training_transfuser_moe_multimodal_optimized/2026.01.12.01.01.54/lightning_logs/version_0/checkpoints/epoch=149-step=25050.ckpt \
  200
```

训练会从epoch 149继续，直到完成200个epoch。



