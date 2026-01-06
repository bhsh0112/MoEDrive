#!/bin/bash
# MOE Trajectory Head 训练脚本 - 多个变体配置

# ============================================
# 变体 1: 基础配置 (4 experts, top-2)
# ============================================
echo "Training with MOE Trajectory Head - Variant 1: Basic (4 experts, top-2)"
CUDA_VISIBLE_DEVICES=4,5,6,7 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  agent=diffusiondrive_agent \
  experiment_name=training_diffusiondrive_agent_moe_traj_basic \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=100 \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  +agent.config.use_moe_trajectory=True \
  +agent.config.moe_trajectory_num_experts=4 \
  +agent.config.moe_trajectory_top_k=2 \
  +agent.config.moe_trajectory_num_modes=20 \
  +agent.config.moe_aux_loss_weight=0.2

# ============================================
# 变体 2: 更多 experts (8 experts, top-3)
# ============================================
# echo "Training with MOE Trajectory Head - Variant 2: More Experts (8 experts, top-3)"
# CUDA_VISIBLE_DEVICES=4,5,6,7 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
#   agent=diffusiondrive_agent \
#   experiment_name=training_diffusiondrive_agent_moe_traj_8exp_top3 \
#   train_test_split=navtrain \
#   split=trainval \
#   trainer.params.max_epochs=100 \
#   trainer.params.strategy=ddp_find_unused_parameters_true \
#   cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
#   use_cache_without_dataset=True \
#   force_cache_computation=False \
#   +agent.config.use_moe_trajectory=True \
#   +agent.config.moe_trajectory_num_experts=8 \
#   +agent.config.moe_trajectory_top_k=3 \
#   +agent.config.moe_trajectory_num_modes=20 \
#   +agent.config.moe_aux_loss_weight=0.2

# ============================================
# 变体 3: 更少 experts (2 experts, top-1)
# ============================================
# echo "Training with MOE Trajectory Head - Variant 3: Fewer Experts (2 experts, top-1)"
# CUDA_VISIBLE_DEVICES=4,5,6,7 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
#   agent=diffusiondrive_agent \
#   experiment_name=training_diffusiondrive_agent_moe_traj_2exp_top1 \
#   train_test_split=navtrain \
#   split=trainval \
#   trainer.params.max_epochs=100 \
#   trainer.params.strategy=ddp_find_unused_parameters_true \
#   cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
#   use_cache_without_dataset=True \
#   force_cache_computation=False \
#   +agent.config.use_moe_trajectory=True \
#   +agent.config.moe_trajectory_num_experts=2 \
#   +agent.config.moe_trajectory_top_k=1 \
#   +agent.config.moe_trajectory_num_modes=20 \
#   +agent.config.moe_aux_loss_weight=0.2

# ============================================
# 变体 4: 更多模式 (4 experts, top-2, 40 modes)
# ============================================
# echo "Training with MOE Trajectory Head - Variant 4: More Modes (4 experts, 40 modes)"
# CUDA_VISIBLE_DEVICES=4,5,6,7 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
#   agent=diffusiondrive_agent \
#   experiment_name=training_diffusiondrive_agent_moe_traj_40modes \
#   train_test_split=navtrain \
#   split=trainval \
#   trainer.params.max_epochs=100 \
#   trainer.params.strategy=ddp_find_unused_parameters_true \
#   cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
#   use_cache_without_dataset=True \
#   force_cache_computation=False \
#   +agent.config.use_moe_trajectory=True \
#   +agent.config.moe_trajectory_num_experts=4 \
#   +agent.config.moe_trajectory_top_k=2 \
#   +agent.config.moe_trajectory_num_modes=40 \
#   +agent.config.moe_aux_loss_weight=0.2

