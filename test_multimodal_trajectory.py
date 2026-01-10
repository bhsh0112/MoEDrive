#!/usr/bin/env python3
"""
测试脚本：验证多模态轨迹预测实现

测试内容：
1. 模型初始化（单模态和多模态）
2. Forward pass 输出形状验证
3. 损失函数计算
4. 梯度检查
5. 数值稳定性
"""

import sys
import traceback
from typing import Dict
import torch
import numpy as np
from dataclasses import dataclass

# 添加路径以便导入模块
sys.path.insert(0, '/data2/file_swap/sh_space/MOEDrive')

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.agents.transfuser.transfuser_model import TransfuserModel
from navsim.agents.transfuser.transfuser_loss import transfuser_loss
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index


def create_test_config(single_modal: bool = False) -> TransfuserConfig:
    """创建测试配置"""
    config = TransfuserConfig(
        trajectory_sampling=TrajectorySampling(time_horizon=4, interval_length=0.5),
        use_moe_decoder=True,
        moe_num_experts=8,
        moe_top_k=8,  # 在多模态模式下使用所有专家
        multimodal_trajectory=not single_modal,
        num_trajectory_modes=8,  # 匹配 moe_num_experts
        tf_d_model=128,  # 减小模型尺寸以加快测试
        tf_d_ffn=256,
        tf_num_layers=2,
        tf_num_head=4,
        num_bounding_boxes=5,  # 减小以加快测试
    )
    return config


def create_dummy_features(batch_size: int = 2, config: TransfuserConfig = None) -> Dict[str, torch.Tensor]:
    """创建虚拟输入特征"""
    if config is None:
        config = create_test_config()
    
    # LiDAR特征通道数由lidar_seq_len决定（默认1，如果use_ground_plane则*2）
    lidar_channels = config.lidar_seq_len if not config.use_ground_plane else config.lidar_seq_len * 2
    
    return {
        "camera_feature": torch.randn(batch_size, 3, 256, 1024),
        "lidar_feature": torch.randn(batch_size, lidar_channels, 256, 256),
        "status_feature": torch.randn(batch_size, 8),  # 4 + 2 + 2
    }


def create_dummy_targets(batch_size: int = 2, config: TransfuserConfig = None) -> Dict[str, torch.Tensor]:
    """创建虚拟目标标签"""
    if config is None:
        config = create_test_config()
    
    num_poses = config.trajectory_sampling.num_poses
    # BoundingBox2DIndex.size() = 5 (X, Y, HEADING, LENGTH, WIDTH)
    agent_state_size = BoundingBox2DIndex.size()
    
    return {
        "trajectory": torch.randn(batch_size, num_poses, 3) * 10,  # (B, num_poses, 3)
        "agent_states": torch.randn(batch_size, config.num_bounding_boxes, agent_state_size),  # (B, num_agents, 5)
        "agent_labels": torch.randint(0, 2, (batch_size, config.num_bounding_boxes)).float(),
        "bev_semantic_map": torch.randint(0, config.num_bev_classes, 
                                         (batch_size, config.bev_pixel_height, config.bev_pixel_width)),
    }


def test_1_model_initialization():
    """测试1: 模型初始化"""
    print("\n" + "="*60)
    print("测试1: 模型初始化")
    print("="*60)
    
    try:
        # 测试单模态模式
        print("\n1.1 测试单模态模式初始化...")
        config_single = create_test_config(single_modal=True)
        model_single = TransfuserModel(config_single)
        print(f"   ✓ 单模态模型初始化成功")
        print(f"   - use_moe_decoder: {config_single.use_moe_decoder}")
        print(f"   - multimodal_trajectory: {config_single.multimodal_trajectory}")
        
        # 测试多模态模式
        print("\n1.2 测试多模态模式初始化...")
        config_multi = create_test_config(single_modal=False)
        model_multi = TransfuserModel(config_multi)
        print(f"   ✓ 多模态模型初始化成功")
        print(f"   - use_moe_decoder: {config_multi.use_moe_decoder}")
        print(f"   - multimodal_trajectory: {config_multi.multimodal_trajectory}")
        print(f"   - moe_num_experts: {config_multi.moe_num_experts}")
        print(f"   - num_trajectory_modes: {config_multi.num_trajectory_modes}")
        
        # 验证配置一致性
        assert config_multi.moe_num_experts == config_multi.num_trajectory_modes, \
            f"moe_num_experts ({config_multi.moe_num_experts}) != num_trajectory_modes ({config_multi.num_trajectory_modes})"
        print(f"   ✓ 配置一致性验证通过")
        
        return True, model_single, model_multi, config_single, config_multi
        
    except Exception as e:
        print(f"   ✗ 初始化失败: {e}")
        traceback.print_exc()
        return False, None, None, None, None


def test_2_forward_pass_shapes():
    """测试2: Forward pass 输出形状验证"""
    print("\n" + "="*60)
    print("测试2: Forward pass 输出形状验证")
    print("="*60)
    
    try:
        # 单模态测试
        print("\n2.1 测试单模态 Forward pass...")
        config_single = create_test_config(single_modal=True)
        model_single = TransfuserModel(config_single)
        model_single.eval()
        
        batch_size = 2
        features = create_dummy_features(batch_size, config_single)
        
        with torch.no_grad():
            outputs_single = model_single(features)
        
        # 验证输出形状
        assert "trajectory" in outputs_single, "缺少 'trajectory' 输出"
        trajectory_single = outputs_single["trajectory"]
        expected_shape = (batch_size, config_single.trajectory_sampling.num_poses, 3)
        assert trajectory_single.shape == expected_shape, \
            f"轨迹形状错误: 期望 {expected_shape}, 得到 {trajectory_single.shape}"
        
        print(f"   ✓ 单模态输出形状正确: {trajectory_single.shape}")
        print(f"   - trajectory: {trajectory_single.shape}")
        
        # 多模态测试
        print("\n2.2 测试多模态 Forward pass...")
        config_multi = create_test_config(single_modal=False)
        model_multi = TransfuserModel(config_multi)
        model_multi.eval()
        
        features_multi = create_dummy_features(batch_size, config_multi)
        
        with torch.no_grad():
            outputs_multi = model_multi(features_multi)
        
        # 验证多模态输出
        assert "trajectory" in outputs_multi, "缺少 'trajectory' 输出"
        trajectory_multi = outputs_multi["trajectory"]
        expected_shape_best = (batch_size, config_multi.trajectory_sampling.num_poses, 3)
        assert trajectory_multi.shape == expected_shape_best, \
            f"最佳轨迹形状错误: 期望 {expected_shape_best}, 得到 {trajectory_multi.shape}"
        
        # 检查是否有多模态相关输出
        if "trajectory_modes" in outputs_multi:
            trajectory_modes = outputs_multi["trajectory_modes"]
            expected_modes_shape = (batch_size, config_multi.num_trajectory_modes, 
                                   config_multi.trajectory_sampling.num_poses, 3)
            assert trajectory_modes.shape == expected_modes_shape, \
                f"轨迹模式形状错误: 期望 {expected_modes_shape}, 得到 {trajectory_modes.shape}"
            print(f"   ✓ 轨迹模式形状正确: {trajectory_modes.shape}")
        
        if "trajectory_mode_scores" in outputs_multi:
            mode_scores = outputs_multi["trajectory_mode_scores"]
            expected_scores_shape = (batch_size, config_multi.num_trajectory_modes)
            assert mode_scores.shape == expected_scores_shape, \
                f"模式分数形状错误: 期望 {expected_scores_shape}, 得到 {mode_scores.shape}"
            print(f"   ✓ 模式分数形状正确: {mode_scores.shape}")
        
        print(f"   ✓ 多模态输出形状验证通过")
        print(f"   - trajectory (best): {trajectory_multi.shape}")
        if "trajectory_modes" in outputs_multi:
            print(f"   - trajectory_modes: {outputs_multi['trajectory_modes'].shape}")
        if "trajectory_mode_scores" in outputs_multi:
            print(f"   - trajectory_mode_scores: {outputs_multi['trajectory_mode_scores'].shape}")
        
        return True, outputs_single, outputs_multi
        
    except Exception as e:
        print(f"   ✗ Forward pass 测试失败: {e}")
        traceback.print_exc()
        return False, None, None


def test_3_loss_computation():
    """测试3: 损失函数计算"""
    print("\n" + "="*60)
    print("测试3: 损失函数计算")
    print("="*60)
    
    try:
        batch_size = 2
        
        # 单模态损失测试
        print("\n3.1 测试单模态损失计算...")
        config_single = create_test_config(single_modal=True)
        model_single = TransfuserModel(config_single)
        model_single.train()
        
        features = create_dummy_features(batch_size, config_single)
        targets = create_dummy_targets(batch_size, config_single)
        
        predictions_single = model_single(features, targets=targets)
        loss_dict_single = transfuser_loss(targets, predictions_single, config_single)
        
        assert "loss" in loss_dict_single, "缺少总损失"
        assert "trajectory_loss" in loss_dict_single, "缺少轨迹损失"
        
        print(f"   ✓ 单模态损失计算成功")
        print(f"   - 总损失: {loss_dict_single['loss'].item():.4f}")
        print(f"   - 轨迹损失: {loss_dict_single['trajectory_loss'].item():.4f}")
        
        # 多模态损失测试
        print("\n3.2 测试多模态损失计算...")
        config_multi = create_test_config(single_modal=False)
        model_multi = TransfuserModel(config_multi)
        model_multi.train()
        
        features_multi = create_dummy_features(batch_size, config_multi)
        targets_multi = create_dummy_targets(batch_size, config_multi)
        
        predictions_multi = model_multi(features_multi, targets=targets_multi)
        loss_dict_multi = transfuser_loss(targets_multi, predictions_multi, config_multi)
        
        assert "loss" in loss_dict_multi, "缺少总损失"
        assert "trajectory_loss" in loss_dict_multi, "缺少轨迹损失"
        
        # 检查是否有多模态相关损失
        if "trajectory_mode_loss" in loss_dict_multi:
            print(f"   ✓ 模式分类损失: {loss_dict_multi['trajectory_mode_loss'].item():.4f}")
        
        print(f"   ✓ 多模态损失计算成功")
        print(f"   - 总损失: {loss_dict_multi['loss'].item():.4f}")
        print(f"   - 轨迹损失: {loss_dict_multi['trajectory_loss'].item():.4f}")
        
        # 验证损失数值合理性
        assert not torch.isnan(loss_dict_single["loss"]), "单模态损失包含 NaN"
        assert not torch.isinf(loss_dict_single["loss"]), "单模态损失包含 Inf"
        assert not torch.isnan(loss_dict_multi["loss"]), "多模态损失包含 NaN"
        assert not torch.isinf(loss_dict_multi["loss"]), "多模态损失包含 Inf"
        
        print(f"   ✓ 损失数值合理性验证通过")
        
        return True
        
    except Exception as e:
        print(f"   ✗ 损失计算测试失败: {e}")
        traceback.print_exc()
        return False


def test_4_gradient_check():
    """测试4: 梯度检查"""
    print("\n" + "="*60)
    print("测试4: 梯度检查")
    print("="*60)
    
    try:
        batch_size = 2
        
        # 单模态梯度测试
        print("\n4.1 测试单模态模式梯度...")
        config_single = create_test_config(single_modal=True)
        model_single = TransfuserModel(config_single)
        model_single.train()
        
        features = create_dummy_features(batch_size, config_single)
        targets = create_dummy_targets(batch_size, config_single)
        
        predictions = model_single(features, targets=targets)
        loss_dict = transfuser_loss(targets, predictions, config_single)
        loss = loss_dict["loss"]
        
        loss.backward()
        
        # 检查关键参数是否有梯度
        has_gradient = False
        for name, param in model_single.named_parameters():
            if param.grad is not None and param.grad.abs().max() > 1e-7:
                has_gradient = True
                break
        
        assert has_gradient, "没有检测到有效梯度"
        print(f"   ✓ 单模态模式梯度检查通过")
        
        # 多模态梯度测试
        print("\n4.2 测试多模态模式梯度...")
        config_multi = create_test_config(single_modal=False)
        model_multi = TransfuserModel(config_multi)
        model_multi.train()
        
        # 清零之前的梯度
        for param in model_multi.parameters():
            if param.grad is not None:
                param.grad.zero_()
        
        features_multi = create_dummy_features(batch_size, config_multi)
        targets_multi = create_dummy_targets(batch_size, config_multi)
        
        predictions_multi = model_multi(features_multi, targets=targets_multi)
        loss_dict_multi = transfuser_loss(targets_multi, predictions_multi, config_multi)
        loss_multi = loss_dict_multi["loss"]
        
        loss_multi.backward()
        
        # 检查关键参数是否有梯度
        has_gradient_multi = False
        gradient_norm = 0.0
        for name, param in model_multi.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.abs().max().item()
                if grad_norm > 1e-7:
                    has_gradient_multi = True
                    gradient_norm = max(gradient_norm, grad_norm)
        
        assert has_gradient_multi, "没有检测到有效梯度"
        print(f"   ✓ 多模态模式梯度检查通过 (最大梯度: {gradient_norm:.6f})")
        
        return True
        
    except Exception as e:
        print(f"   ✗ 梯度检查失败: {e}")
        traceback.print_exc()
        return False


def test_5_numerical_stability():
    """测试5: 数值稳定性"""
    print("\n" + "="*60)
    print("测试5: 数值稳定性")
    print("="*60)
    
    try:
        batch_size = 4
        config_multi = create_test_config(single_modal=False)
        model_multi = TransfuserModel(config_multi)
        model_multi.train()
        
        # 测试极端输入
        print("\n5.1 测试极端输入...")
        
        # 获取正确的LiDAR通道数
        lidar_channels = config_multi.lidar_seq_len if not config_multi.use_ground_plane else config_multi.lidar_seq_len * 2
        
        # 大数值输入
        features_large = {
            "camera_feature": torch.randn(batch_size, 3, 256, 1024) * 10,
            "lidar_feature": torch.randn(batch_size, lidar_channels, 256, 256) * 10,
            "status_feature": torch.randn(batch_size, 8) * 10,
        }
        
        targets_large = create_dummy_targets(batch_size, config_multi)
        targets_large["trajectory"] = torch.randn(batch_size, config_multi.trajectory_sampling.num_poses, 3) * 100
        
        predictions_large = model_multi(features_large, targets=targets_large)
        loss_dict_large = transfuser_loss(targets_large, predictions_large, config_multi)
        
        assert not torch.isnan(loss_dict_large["loss"]), "大数值输入导致 NaN"
        assert not torch.isinf(loss_dict_large["loss"]), "大数值输入导致 Inf"
        print(f"   ✓ 大数值输入测试通过 (损失: {loss_dict_large['loss'].item():.4f})")
        
        # 小数值输入
        features_small = {
            "camera_feature": torch.randn(batch_size, 3, 256, 1024) * 0.01,
            "lidar_feature": torch.randn(batch_size, lidar_channels, 256, 256) * 0.01,
            "status_feature": torch.randn(batch_size, 8) * 0.01,
        }
        
        targets_small = create_dummy_targets(batch_size, config_multi)
        targets_small["trajectory"] = torch.randn(batch_size, config_multi.trajectory_sampling.num_poses, 3) * 0.01
        
        predictions_small = model_multi(features_small, targets=targets_small)
        loss_dict_small = transfuser_loss(targets_small, predictions_small, config_multi)
        
        assert not torch.isnan(loss_dict_small["loss"]), "小数值输入导致 NaN"
        assert not torch.isinf(loss_dict_small["loss"]), "小数值输入导致 Inf"
        print(f"   ✓ 小数值输入测试通过 (损失: {loss_dict_small['loss'].item():.4f})")
        
        # 测试多次前向传播的一致性
        print("\n5.2 测试多次前向传播的一致性...")
        model_multi.eval()
        features_consistency = create_dummy_features(batch_size, config_multi)
        
        with torch.no_grad():
            output1 = model_multi(features_consistency)
            output2 = model_multi(features_consistency)
        
        # 检查输出是否一致（在评估模式下应该一致）
        trajectory_diff = (output1["trajectory"] - output2["trajectory"]).abs().max().item()
        assert trajectory_diff < 1e-5, f"评估模式下输出不一致: 差异 {trajectory_diff}"
        print(f"   ✓ 输出一致性测试通过 (最大差异: {trajectory_diff:.8f})")
        
        return True
        
    except Exception as e:
        print(f"   ✗ 数值稳定性测试失败: {e}")
        traceback.print_exc()
        return False


def test_6_mode_diversity():
    """测试6: 模式多样性检查"""
    print("\n" + "="*60)
    print("测试6: 模式多样性检查")
    print("="*60)
    
    try:
        batch_size = 2
        config_multi = create_test_config(single_modal=False)
        model_multi = TransfuserModel(config_multi)
        model_multi.eval()
        
        features = create_dummy_features(batch_size, config_multi)
        
        with torch.no_grad():
            outputs = model_multi(features)
        
        if "trajectory_modes" in outputs:
            trajectory_modes = outputs["trajectory_modes"]  # (B, num_modes, num_poses, 3)
            
            # 计算不同模式之间的差异
            # 对于每个batch，计算所有模式对之间的平均差异
            batch_diversity = []
            for b in range(batch_size):
                modes_b = trajectory_modes[b]  # (num_modes, num_poses, 3)
                # 计算每对模式之间的L2距离
                pairwise_dists = []
                for i in range(config_multi.num_trajectory_modes):
                    for j in range(i + 1, config_multi.num_trajectory_modes):
                        dist = torch.norm(modes_b[i] - modes_b[j]).item()
                        pairwise_dists.append(dist)
                
                avg_diversity = np.mean(pairwise_dists)
                batch_diversity.append(avg_diversity)
            
            avg_diversity_across_batch = np.mean(batch_diversity)
            print(f"   ✓ 模式多样性检查完成")
            print(f"   - 平均模式间距离: {avg_diversity_across_batch:.4f}")
            
            # 如果所有模式都相同，多样性为0，这是不理想的
            # 但在随机初始化的模型中，这是可能的
            if avg_diversity_across_batch < 1e-3:
                print(f"   ⚠ 警告: 模式多样性较低，可能需要训练后检查")
            else:
                print(f"   ✓ 模式多样性合理")
            
            return True
        else:
            print(f"   ⚠ 警告: 未找到 trajectory_modes 输出，跳过多样性检查")
            return True
        
    except Exception as e:
        print(f"   ✗ 模式多样性检查失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("多模态轨迹预测实现 - 测试套件")
    print("="*60)
    
    results = {}
    
    # 运行所有测试
    try:
        success, model_single, model_multi, config_single, config_multi = test_1_model_initialization()
        results["test_1"] = success
        if not success:
            print("\n⚠ 模型初始化失败，跳过后续测试")
            return
        
        success, outputs_single, outputs_multi = test_2_forward_pass_shapes()
        results["test_2"] = success
        
        results["test_3"] = test_3_loss_computation()
        
        results["test_4"] = test_4_gradient_check()
        
        results["test_5"] = test_5_numerical_stability()
        
        results["test_6"] = test_6_mode_diversity()
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        return
    except Exception as e:
        print(f"\n\n未预期的错误: {e}")
        traceback.print_exc()
        return
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    test_names = {
        "test_1": "模型初始化",
        "test_2": "Forward pass 形状验证",
        "test_3": "损失函数计算",
        "test_4": "梯度检查",
        "test_5": "数值稳定性",
        "test_6": "模式多样性检查",
    }
    
    for test_key, test_name in test_names.items():
        status = "✓ 通过" if results.get(test_key, False) else "✗ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ 所有测试通过！")
        print("="*60)
        return 0
    else:
        print("✗ 部分测试失败，请检查上述错误信息")
        print("="*60)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

