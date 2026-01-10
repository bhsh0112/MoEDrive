#!/usr/bin/env python3
"""
快速测试脚本：验证多模态轨迹预测的基本功能

用法:
    python test_multimodal_quick.py
"""

import sys
import torch
sys.path.insert(0, '/data2/file_swap/sh_space/MOEDrive')

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_model import TransfuserModel


def quick_test():
    """快速测试基本功能"""
    print("开始快速测试...")
    
    # 创建多模态配置
    config = TransfuserConfig(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        use_moe_decoder=True,
        moe_num_experts=4,  # 小模型用于快速测试
        moe_top_k=4,
        multimodal_trajectory=True,
        num_trajectory_modes=4,
        tf_d_model=64,
        tf_d_ffn=128,
        tf_num_layers=1,
        tf_num_head=2,
        num_bounding_boxes=3,
    )
    
    print(f"配置:")
    print(f"  - multimodal_trajectory: {config.multimodal_trajectory}")
    print(f"  - moe_num_experts: {config.moe_num_experts}")
    print(f"  - num_trajectory_modes: {config.num_trajectory_modes}")
    
    # 创建模型
    print("\n创建模型...")
    model = TransfuserModel(config)
    model.eval()
    print("✓ 模型创建成功")
    
    # 创建虚拟输入
    # 注意：根据config，lidar_feature的通道数应该是 lidar_seq_len (默认1)
    batch_size = 1
    lidar_channels = config.lidar_seq_len if hasattr(config, 'lidar_seq_len') else 1
    features = {
        "camera_feature": torch.randn(batch_size, 3, 256, 1024),
        "lidar_feature": torch.randn(batch_size, lidar_channels, 256, 256),
        "status_feature": torch.randn(batch_size, 8),
    }
    
    # 前向传播
    print("\n执行前向传播...")
    with torch.no_grad():
        outputs = model(features)
    
    print("✓ 前向传播成功")
    
    # 检查输出
    print("\n输出检查:")
    if "trajectory" in outputs:
        print(f"  - trajectory shape: {outputs['trajectory'].shape}")
    if "trajectory_modes" in outputs:
        print(f"  - trajectory_modes shape: {outputs['trajectory_modes'].shape}")
    if "trajectory_mode_scores" in outputs:
        print(f"  - trajectory_mode_scores shape: {outputs['trajectory_mode_scores'].shape}")
        print(f"  - mode scores: {outputs['trajectory_mode_scores']}")
    
    # 检查数值有效性
    trajectory = outputs["trajectory"]
    assert not torch.isnan(trajectory).any(), "轨迹包含 NaN"
    assert not torch.isinf(trajectory).any(), "轨迹包含 Inf"
    print("✓ 数值有效性检查通过")
    
    print("\n✓ 快速测试通过！")
    return True


if __name__ == "__main__":
    try:
        quick_test()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

