#!/bin/bash
# Source environment variables
source scripts/SetPath.sh

# 训练脚本：使用MoE Decoder + 多模态轨迹预测（使用+前缀版本）
#
# 如果主脚本不工作，使用这个备选版本
# 使用 + 前缀来添加新配置项，避免 Hydra struct 模式的问题

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
  agent.config.moe_num_experts=20 \
  agent.config.moe_top_k=20 \
  +agent.config.num_trajectory_modes=20 \
  +agent.config.trajectory_mode_weight=1.0 \
  +agent.config.trajectory_position_weight=1.0 \
  +agent.config.trajectory_heading_weight=1.0



