#!/bin/bash
# 确保在项目根目录运行，并设置环境变量
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# cd "$PROJECT_ROOT" || exit 1

# Source environment variables
source scripts/SetPath.sh

# 阶段2优化训练脚本：添加学习率调度、混合精度、早停等
#
# 新增优化：
# 1. 混合精度训练：precision=16 (FP16)
# 2. 学习率调度：使用余弦退火（需要在Lightning配置中设置）
# 3. 早停机制：monitor=val_loss, patience=15
# 4. 最佳模型保存：save_top_k=3, monitor=val_loss
# 5. 验证频率：val_check_interval=0.25 (每25% epoch验证一次)
#
# 注意：
# - 混合精度训练需要GPU支持Tensor Core（V100/A100/RTX系列）
# - 如果遇到数值不稳定，可以降低precision或使用bf16
# - 早停patience可以根据训练情况调整

# 确保 Python 能找到工程内模块
# export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=4,5,6,7 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  agent=transfuser_agent \
  experiment_name=training_transfuser_moe_multimodal_stage2 \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=150 \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  trainer.params.gradient_clip_val=1.0 \
  trainer.params.gradient_clip_algorithm=norm \
  trainer.params.precision=16 \
  trainer.params.val_check_interval=0.25 \
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

# 注意：早停和模型检查点回调需要在Lightning配置中添加
# 如果训练脚本不支持，可能需要修改训练代码或使用配置文件

