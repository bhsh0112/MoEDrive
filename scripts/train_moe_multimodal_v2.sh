#!/bin/bash
# Source environment variables
source scripts/SetPath.sh

# 优化版本2：减少专家数量的训练脚本
#
# 主要变化：
# - 专家数量从20减少到16（如果20个专家导致负载不均衡）
# - 其他优化与optimized版本相同
#
# 使用场景：当20个专家导致某些专家使用率过低时使用

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  agent=transfuser_agent \
  experiment_name=training_transfuser_moe_multimodal_v2 \
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
  +agent.config.moe_num_experts=16 \
  +agent.config.moe_top_k=16 \
  +agent.config.moe_router_temperature=1.0 \
  +agent.config.moe_load_balance_coef=1e-2 \
  +agent.config.moe_router_z_loss_coef=1e-3 \
  +agent.config.moe_aux_loss_weight=0.3 \
  +agent.config.num_trajectory_modes=16 \
  +agent.config.trajectory_mode_weight=2.0 \
  +agent.config.trajectory_position_weight=1.0 \
  +agent.config.trajectory_heading_weight=1.5



