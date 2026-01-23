# 训练配置修复说明

## 问题描述

运行 `train_moe_multimodal.sh` 时遇到 Hydra 配置错误：

```
Could not override 'agent.config.multimodal_trajectory'.
Key 'multimodal_trajectory' is not in struct
```

## 原因分析

1. **Hydra Struct 模式**：配置文件中使用了 `_convert_: 'all'`，这会将 dataclass 转换为 struct 模式
2. **严格类型检查**：Struct 模式不允许覆盖或添加不存在的配置项
3. **新参数未定义**：多模态相关的参数是新添加的，在原始配置文件中不存在

## 解决方案

### 方案1：在配置文件中添加参数（已实施）✅

已在 `transfuser_agent.yaml` 中添加了所有需要的参数：

```yaml
# MoE decoder configuration
use_moe_decoder: False
moe_num_experts: 8
moe_top_k: 2
...

# Multi-modal trajectory prediction
multimodal_trajectory: False
num_trajectory_modes: 20
trajectory_mode_weight: 1.0
trajectory_position_weight: 1.0
trajectory_heading_weight: 1.0
```

**优点**：
- 配置清晰，所有参数都有默认值
- 可以在配置文件中统一管理
- 命令行可以直接覆盖（不需要 `+` 前缀）

**缺点**：
- 需要修改配置文件
- 如果参数很多，配置文件会变得很长

### 方案2：使用 `+` 前缀添加新配置（备选）

如果方案1不工作，可以使用 `+` 前缀：

```bash
+agent.config.multimodal_trajectory=True \
+agent.config.num_trajectory_modes=20 \
+agent.config.trajectory_mode_weight=1.0 \
```

**优点**：
- 不需要修改配置文件
- 只在需要时添加参数

**缺点**：
- 命令行会变得很长
- 参数分散，不易管理

### 方案3：创建单独的配置文件（推荐用于实验）

创建一个新的配置文件 `transfuser_agent_multimodal.yaml`：

```yaml
# @package _global_
defaults:
  - /agent/transfuser_agent
  - _self_

agent:
  config:
    use_moe_decoder: True
    multimodal_trajectory: True
    moe_num_experts: 20
    moe_top_k: 20
    num_trajectory_modes: 20
```

然后在训练脚本中使用：
```bash
agent=transfuser_agent_multimodal
```

## 验证修复

运行训练脚本：
```bash
bash scripts/train_moe_multimodal.sh
```

如果仍然报错，尝试：

1. **检查 dataclass 定义**：
   - 确保 `TransfuserConfig` 中包含了所有参数
   - 确保参数名称与配置文件中的完全一致

2. **使用 `+` 前缀**：
   ```bash
   +agent.config.multimodal_trajectory=True
   ```

3. **检查 Hydra 版本**：
   ```bash
   pip show hydra-core
   ```

4. **启用完整错误信息**：
   ```bash
   export HYDRA_FULL_ERROR=1
   bash scripts/train_moe_multimodal.sh
   ```

## 当前状态

- ✅ 配置文件已更新，包含所有必要参数
- ✅ 训练脚本已更新
- ⚠️ 需要验证是否能正常运行

## 如果仍然失败

如果方案1不工作，请尝试以下步骤：

1. **完全使用 `+` 前缀版本**：
   
   修改 `train_moe_multimodal.sh`：
   ```bash
   +agent.config.use_moe_decoder=True \
   +agent.config.multimodal_trajectory=True \
   +agent.config.moe_num_experts=20 \
   +agent.config.moe_top_k=20 \
   +agent.config.num_trajectory_modes=20 \
   +agent.config.trajectory_mode_weight=1.0 \
   +agent.config.trajectory_position_weight=1.0 \
   +agent.config.trajectory_heading_weight=1.0
   ```

2. **或者创建一个新的配置文件**（方案3）

3. **检查 TransfuserConfig 的字段定义**：
   ```python
   from navsim.agents.transfuser.transfuser_config import TransfuserConfig
   import inspect
   print(inspect.signature(TransfuserConfig))
   ```

## 相关文件

- 配置文件：`navsim/planning/script/config/common/agent/transfuser_agent.yaml`
- 训练脚本：`scripts/train_moe_multimodal.sh`
- Dataclass 定义：`navsim/agents/transfuser/transfuser_config.py`



