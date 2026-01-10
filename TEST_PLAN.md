# 多模态轨迹预测实现 - 测试方案

## 测试目标

验证基于MoE的_tf_decoder + 基于transfuser的trajectory_head能否正确输出类似DiffusionDrive的多模态轨迹预测。

## 测试策略

### 阶段1: 单元测试 (已完成)

#### 1.1 组件级测试
- [x] MoE Decoder 多模态输出
- [x] TrajectoryHead 多模态输入/输出
- [x] 损失函数多模态支持

#### 1.2 集成测试
- [x] TransfuserModel 完整前向传播
- [x] 输出形状验证
- [x] 梯度流检查

### 阶段2: 功能测试 (当前阶段)

#### 2.1 基本功能验证
运行 `test_multimodal_quick.py`:
```bash
python test_multimodal_quick.py
```

**验证点:**
- ✓ 模型能否初始化
- ✓ 前向传播是否正常
- ✓ 输出形状是否正确
- ✓ 是否产生多模态输出

#### 2.2 完整测试套件
运行 `test_multimodal_trajectory.py`:
```bash
python test_multimodal_trajectory.py
```

**验证点:**
- ✓ 单模态/多模态模式切换
- ✓ 输出形状一致性
- ✓ 损失函数正确性
- ✓ 梯度有效性
- ✓ 数值稳定性
- ✓ 模式多样性

### 阶段3: 对比测试 (推荐)

#### 3.1 与DiffusionDrive对比

创建对比脚本 `compare_with_diffusiondrive.py`:

**对比维度:**
1. **输出结构**
   - DiffusionDrive: `(B, num_modes, num_poses, 3)`
   - 我们的实现: `(B, num_modes, num_poses, 3)` ✓

2. **模式数量**
   - DiffusionDrive: 20 modes (使用 plan anchors)
   - 我们的实现: 可配置 (基于 MoE experts)

3. **模式选择机制**
   - DiffusionDrive: 基于 anchor distance + classification head
   - 我们的实现: 基于 expert routing + classification head

#### 3.2 定性评估

**可视化对比:**
```python
# 可视化示例
import matplotlib.pyplot as plt

def visualize_trajectories(gt, pred_modes, pred_best):
    """可视化ground truth和预测的多模态轨迹"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 显示所有模式
    axes[0].plot(gt[:, 0], gt[:, 1], 'k-', label='GT', linewidth=2)
    for i, mode in enumerate(pred_modes):
        axes[0].plot(mode[:, 0], mode[:, 1], '--', alpha=0.5, label=f'Mode {i}')
    axes[0].plot(pred_best[:, 0], pred_best[:, 1], 'r-', label='Best', linewidth=2)
    axes[0].legend()
    axes[0].set_title('All Modes')
    
    # 显示最佳轨迹对比
    axes[1].plot(gt[:, 0], gt[:, 1], 'k-', label='GT', linewidth=2)
    axes[1].plot(pred_best[:, 0], pred_best[:, 1], 'r--', label='Pred Best', linewidth=2)
    axes[1].legend()
    axes[1].set_title('Best Trajectory Comparison')
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png')
```

### 阶段4: 性能测试

#### 4.1 推理速度测试

```python
import time
import torch

def benchmark_inference(model, features, num_runs=100):
    """基准测试推理速度"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        _ = model(features)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(features)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    avg_time = (end - start) / num_runs
    fps = 1.0 / avg_time
    
    print(f"Average inference time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    
    return avg_time, fps
```

**预期性能:**
- CPU: ~100-200ms per sample
- GPU: ~10-20ms per sample (batch_size=1)

#### 4.2 内存占用测试

```python
def measure_memory_usage(model, features):
    """测量内存占用"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    with torch.no_grad():
        _ = model(features)
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_used = mem_after - mem_before
    
    print(f"Memory usage: {mem_used:.2f} MB")
    return mem_used
```

### 阶段5: 训练测试 (可选)

#### 5.1 小规模训练测试

在小型数据集上进行1-2个epoch的训练，验证：
- ✓ 损失是否下降
- ✓ 模式是否多样化
- ✓ 训练是否稳定

#### 5.2 检查点保存/加载

```python
# 保存检查点
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
}, 'checkpoint_multimodal.pth')

# 加载检查点
checkpoint = torch.load('checkpoint_multimodal.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 测试检查清单

### 基础功能
- [ ] 模型可以初始化
- [ ] 前向传播无错误
- [ ] 输出形状正确
- [ ] 损失函数可计算
- [ ] 梯度可以反向传播

### 多模态功能
- [ ] 输出包含多个轨迹模式
- [ ] 模式数量正确
- [ ] 模式之间有差异（多样性）
- [ ] 模式分类头正常工作
- [ ] 最佳轨迹选择正确

### 兼容性
- [ ] 单模态模式仍然工作
- [ ] 输出格式向后兼容
- [ ] 可以与现有训练脚本集成

### 稳定性
- [ ] 无NaN/Inf
- [ ] 数值稳定
- [ ] 内存使用合理
- [ ] 推理速度可接受

## 预期结果

### 成功标准

1. **功能完整性**
   - ✓ 所有测试通过
   - ✓ 输出形状正确
   - ✓ 损失函数正常

2. **多模态输出**
   - ✓ 能够产生多个不同的轨迹模式
   - ✓ 模式之间有明显的多样性
   - ✓ 最佳轨迹选择合理

3. **性能要求**
   - ✓ 推理时间 < 200ms (CPU) 或 < 20ms (GPU)
   - ✓ 内存使用 < 4GB (batch_size=2)
   - ✓ 训练稳定，损失正常下降

### 失败标准

以下情况视为失败：
- ✗ 模型无法初始化
- ✗ 前向传播报错
- ✗ 输出形状不匹配
- ✗ 损失包含NaN/Inf
- ✗ 所有模式完全相同（无多样性）

## 下一步行动

### 如果测试通过

1. **集成到训练流程**
   - 修改训练脚本使用新的多模态配置
   - 添加多模态轨迹的可视化
   - 监控模式多样性和选择准确性

2. **超参数调优**
   - 调整损失权重 (`trajectory_mode_weight`)
   - 优化MoE路由参数 (`moe_top_k`, `moe_router_temperature`)
   - 调整位置和朝向权重

3. **实验评估**
   - 在验证集上评估性能
   - 对比单模态和多模态的指标
   - 分析模式选择的准确性

### 如果测试失败

1. **定位问题**
   - 查看具体错误信息
   - 检查形状不匹配的位置
   - 验证配置参数

2. **修复问题**
   - 修复代码错误
   - 调整配置参数
   - 处理边界情况

3. **重新测试**
   - 运行快速测试验证修复
   - 运行完整测试套件
   - 确保所有测试通过

## 测试数据

### 虚拟数据生成

测试脚本使用虚拟数据，包含：
- 随机图像特征 (camera_feature)
- 随机LiDAR特征 (lidar_feature)
- 随机状态特征 (status_feature)
- 随机轨迹目标 (trajectory)

### 真实数据测试 (推荐)

在真实数据上测试时，确保：
- 数据加载器正常工作
- 特征提取正确
- 目标格式匹配

## 联系方式

如遇问题，请检查：
1. `TESTING_GUIDE.md` - 详细测试指南
2. 代码注释 - 实现细节说明
3. 错误日志 - 具体错误信息

