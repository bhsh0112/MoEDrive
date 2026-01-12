#!/bin/bash
# Source environment variables
source scripts/SetPath.sh

# 优化后的训练脚本：使用MoE Decoder + 多模态轨迹预测
#
# 主要优化：
# 1. 损失权重调整：trajectory_mode_weight从1.0增加到2.0
# 2. MoE辅助损失权重降低：moe_aux_loss_weight从0.5降低到0.3
# 3. MoE负载均衡系数增加：moe_load_balance_coef从5e-3增加到1e-2
# 4. 航向权重增加：trajectory_heading_weight从1.0增加到1.5
# 5. 添加梯度裁剪：gradient_clip_val=1.0
# 6. 增加训练轮数：max_epochs=150
# 7. 添加学习率调度（需要在trainer配置中添加）
#
# 注意：如果训练不稳定，可以考虑：
# - 减少专家数量（moe_num_experts=16）
# - 降低学习率
# - 增加梯度裁剪阈值

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  agent=transfuser_agent \
  experiment_name=training_transfuser_moe_multimodal_optimized \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=150 \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  trainer.params.gradient_clip_val=1.0 \
  trainer.params.gradient_clip_algorithm=norm \
  cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  agent.config.use_moe_decoder=True \
  +agent.config.multimodal_trajectory=True \
  +agent.config.moe_num_experts=20 \
  +agent.config.moe_top_k=20 \
  +agent.config.moe_router_temperature=1.0 \
  +agent.config.moe_load_balance_coef=1e-2 \
  +agent.config.moe_router_z_loss_coef=1e-3 \
  +agent.config.moe_aux_loss_weight=0.3 \
  +agent.config.num_trajectory_modes=20 \
  +agent.config.trajectory_mode_weight=2.0 \
  +agent.config.trajectory_position_weight=1.0 \
  +agent.config.trajectory_heading_weight=1.5

