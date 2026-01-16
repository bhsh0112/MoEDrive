#!/bin/bash
set -euo pipefail

# Source environment variables (NAVSIM_DEVKIT_ROOT, NAVSIM_EXP_ROOT, OPENSCENE_DATA_ROOT, ...)
source scripts/SetPath.sh

# 分阶段训练脚本：MoE Decoder + 多模态轨迹预测 + 3-stage staged training（按 epoch 动态更新）
#
# 关键点：
# - moe_staged_training_enabled=True：启用 MoEStagedTrainingCallback（自动更新温度/top_k/负载均衡/损失权重）
# - trainer.params.strategy=ddp_find_unused_parameters_true：MoE top-k 会导致部分专家参数在某 step 未被使用
# - max_epochs=150：对应三阶段 (0-30/30-100/100-150)
#
# 你可以通过修改下面这些变量来定制：
# - CUDA_VISIBLE_DEVICES：使用的 GPU
# - experiment_name：实验名（决定输出目录）
# - train_test_split：数据划分（navtrain/navmini/...）

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-training_transfuser_moe_multimodal_staged}
TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtrain}

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training.py" \
  agent=transfuser_agent \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  split=trainval \
  trainer.params.max_epochs=150 \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  agent.config.use_moe_decoder=True \
  +agent.config.multimodal_trajectory=True \
  +agent.config.moe_num_experts=20 \
  +agent.config.num_trajectory_modes=20 \
  +agent.config.trajectory_topk_regression_k=5 \
  +agent.config.lr_schedule_enabled=True \
  +agent.config.lr_warmup_epochs=5 \
  +agent.config.lr_min_ratio=0.1 \
  +agent.config.lr_stage3_fixed_enabled=True \
  +agent.config.lr_stage3_start_epoch=100 \
  +agent.config.moe_staged_training_enabled=True \
  +agent.config.moe_staged_training_adaptive=True


