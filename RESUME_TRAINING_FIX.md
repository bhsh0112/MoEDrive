# 训练恢复脚本修复说明

## 问题

Checkpoint路径中包含等号（如 `epoch=18-step=3170.ckpt`），Hydra将等号解析为配置覆盖语法，导致错误：
```
mismatched input '=' expecting <EOF>
```

## 解决方案

我提供了**3种方法**来解决这个问题，按推荐顺序：

### 方法1: 使用符号链接脚本（最推荐）⭐

**完全避免等号问题**，通过创建符号链接：

```bash
bash scripts/train_moe_multimodal_resume_symlink.sh \
  exp/training_transfuser_moe_multimodal_stage2/2026.01.12.17.49.12/lightning_logs/version_0/checkpoints/epoch=18-step=3170.ckpt \
  150
```

**优点**:
- ✅ 完全避免Hydra解析问题
- ✅ 不需要转义
- ✅ 最可靠

### 方法2: 使用转义脚本（已修复）

```bash
bash scripts/train_moe_multimodal_resume.sh \
  exp/training_transfuser_moe_multimodal_stage2/2026.01.12.17.49.12/lightning_logs/version_0/checkpoints/epoch=18-step=3170.ckpt \
  150
```

**如果仍然报错**，请使用方法1（符号链接）。

### 方法3: 手动创建符号链接

如果脚本都不工作，可以手动创建符号链接：

```bash
# 1. 创建符号链接
ln -s "$(realpath exp/.../checkpoints/epoch=18-step=3170.ckpt)" /tmp/checkpoint_resume.ckpt

# 2. 使用符号链接路径
bash scripts/train_moe_multimodal_resume.sh /tmp/checkpoint_resume.ckpt 150

# 3. 训练完成后删除符号链接
rm /tmp/checkpoint_resume.ckpt
```

## 快速修复

**立即使用（推荐）**:

```bash
bash scripts/train_moe_multimodal_resume_symlink.sh \
  exp/training_transfuser_moe_multimodal_stage2/2026.01.12.17.49.12/lightning_logs/version_0/checkpoints/epoch=18-step=3170.ckpt \
  150
```

这个脚本会：
1. 自动创建符号链接（避免等号问题）
2. 运行训练
3. 训练完成后自动清理符号链接

## 为什么会出现这个问题？

- Hydra使用 `=` 作为配置覆盖的语法（如 `param=value`）
- Checkpoint文件名包含等号（`epoch=18-step=3170.ckpt`）
- Hydra尝试解析等号，导致语法错误

## 其他注意事项

1. **确保配置一致**: 恢复训练时，所有模型配置必须与原始训练相同
2. **可以增加epochs**: `max_epochs` 可以增加（如从150增加到200）
3. **检查checkpoint**: 确保checkpoint文件完整且可读


