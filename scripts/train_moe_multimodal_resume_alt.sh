#!/bin/bash
# Source environment variables
source scripts/SetPath.sh

# 恢复训练脚本（备用版本）：使用resume_from_checkpoint参数
#
# 如果 train_moe_multimodal_resume.sh 中的 ckpt_path 不工作，
# 请使用此脚本（适用于PyTorch Lightning < 2.0）
#
# 使用方法：
# bash scripts/train_moe_multimodal_resume_alt.sh <checkpoint_path> [max_epochs]

# 检查参数
if [ -z "$1" ]; then
    echo "错误: 必须提供checkpoint路径"
    echo "用法: bash scripts/train_moe_multimodal_resume_alt.sh <checkpoint_path> [max_epochs]"
    exit 1
fi

CHECKPOINT_PATH="$1"
NEW_MAX_EPOCHS="${2:-200}"

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: Checkpoint文件不存在: $CHECKPOINT_PATH"
    exit 1
fi

echo "使用 resume_from_checkpoint 参数恢复训练"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "新的最大轮数: $NEW_MAX_EPOCHS"
echo ""

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  agent=transfuser_agent \
  experiment_name=training_transfuser_moe_multimodal_optimized \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=${NEW_MAX_EPOCHS} \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  trainer.params.gradient_clip_val=1.0 \
  trainer.params.gradient_clip_algorithm=norm \
  +trainer.params.resume_from_checkpoint="${CHECKPOINT_PATH//=/\\=}" \
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

