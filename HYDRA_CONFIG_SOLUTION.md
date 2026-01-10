# Hydra 配置问题解决方案

## 问题分析

错误信息表明：
1. **第一次错误**：配置项不在struct中，建议使用 `+` 前缀
2. **第二次错误**：配置项已经存在，不能使用 `+`，建议使用 `++` 或移除 `+`

这说明配置文件中已经定义了这些参数，但由于Hydra的struct模式限制，无法直接覆盖。

## 解决方案

### 已实施的修复

1. **清理配置文件** (`transfuser_agent.yaml`)
   - 移除了所有MoE和多模态相关的参数定义
   - 只保留 `use_moe_decoder: False` 作为基础配置
   - 这样避免了配置冲突

2. **更新训练脚本** (`train_moe_multimodal.sh`)
   - 使用 `+` 前缀添加所有需要的参数
   - 包括MoE配置和多模态配置
   - 这样Hydra会将这些参数添加到配置中，而不是尝试覆盖

### 修改后的配置结构

**配置文件** (`transfuser_agent.yaml`):
```yaml
config:
  use_moe_decoder: False
  # 其他基础配置...
```

**训练脚本** (`train_moe_multimodal.sh`):
```bash
agent.config.use_moe_decoder=True \
+agent.config.multimodal_trajectory=True \
+agent.config.moe_num_experts=20 \
+agent.config.moe_top_k=20 \
...
```

## 工作原理

- `+` 前缀：告诉Hydra这是一个新配置项，应该添加到配置中
- 无前缀：覆盖配置文件中的现有值
- `++` 前缀：强制覆盖，即使配置项已存在

由于我们移除了配置文件中的这些参数，使用 `+` 前缀可以安全地添加它们。

## 验证步骤

1. **运行训练脚本**：
   ```bash
   bash scripts/train_moe_multimodal.sh
   ```

2. **如果仍然报错，检查**：
   - Hydra版本：`pip show hydra-core`
   - 完整错误信息：`export HYDRA_FULL_ERROR=1 && bash scripts/train_moe_multimodal.sh`
   - 配置是否正确加载：检查Hydra的输出日志

## 备选方案

如果 `+` 前缀仍然不工作，可以尝试：

### 方案A：使用 `++` 强制覆盖

```bash
++agent.config.multimodal_trajectory=True \
++agent.config.num_trajectory_modes=20 \
```

### 方案B：创建单独的配置文件

创建 `transfuser_agent_multimodal.yaml`：
```yaml
# @package _global_
defaults:
  - /agent/transfuser_agent
  - _self_

agent:
  config:
    use_moe_decoder: True
    multimodal_trajectory: True
    ...
```

然后在训练脚本中使用：
```bash
agent=transfuser_agent_multimodal
```

### 方案C：禁用struct模式（不推荐）

修改配置文件中的 `_convert_: 'all'` 为 `_convert_: 'partial'`，但这可能影响其他配置。

## 当前状态

- ✅ 配置文件已清理
- ✅ 训练脚本已更新使用 `+` 前缀
- ⚠️ 需要验证是否能正常运行

## 如果问题仍然存在

1. **检查dataclass定义**：
   ```python
   from navsim.agents.transfuser.transfuser_config import TransfuserConfig
   import dataclasses
   print([f.name for f in dataclasses.fields(TransfuserConfig)])
   ```

2. **检查Hydra版本兼容性**：
   - Hydra 1.0+: 支持 `+` 和 `++` 前缀
   - 如果版本较旧，可能需要升级

3. **尝试最简单的测试**：
   ```bash
   python -c "from hydra import compose, initialize; initialize(config_path='navsim/planning/script/config'); cfg = compose(config_name='common/agent/transfuser_agent'); print(cfg)"
   ```

## 相关文件

- 配置文件：`navsim/planning/script/config/common/agent/transfuser_agent.yaml`
- 训练脚本：`scripts/train_moe_multimodal.sh`
- Dataclass：`navsim/agents/transfuser/transfuser_config.py`

