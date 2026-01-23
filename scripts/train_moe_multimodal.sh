#!/bin/bash
# Source environment variables
source scripts/SetPath.sh

# 训练脚本：使用MoE Decoder + 多模态轨迹预测
#
# 配置说明：
# - use_moe_decoder=True: 启用MoE decoder
# - multimodal_trajectory=True: 启用多模态轨迹预测
# - moe_num_experts=20: MoE专家数量（也是轨迹模式数量）
# - moe_top_k=20: 在多模态模式下，使用所有专家
# - num_trajectory_modes=20: 轨迹模式数量（应与moe_num_experts匹配）
#
# DDP策略配置：
# - MoE模型中，每个训练步骤只激活部分专家，导致其他专家参数未被使用
# - 需要设置 strategy=ddp_find_unused_parameters_true 来允许未使用的参数

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  agent=transfuser_agent \
  experiment_name=training_transfuser_moe_multimodal \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=100 \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  agent.config.use_moe_decoder=True \
  +agent.config.multimodal_trajectory=True \
  +agent.config.moe_num_experts=20 \
  +agent.config.moe_top_k=20 \
  +agent.config.moe_router_temperature=1.0 \
  +agent.config.moe_load_balance_coef=5e-3 \
  +agent.config.moe_router_z_loss_coef=1e-3 \
  +agent.config.moe_aux_loss_weight=0.5 \
  +agent.config.num_trajectory_modes=20 \
  +agent.config.trajectory_mode_weight=1.0 \
  +agent.config.trajectory_position_weight=1.0 \
  +agent.config.trajectory_heading_weight=1.0

