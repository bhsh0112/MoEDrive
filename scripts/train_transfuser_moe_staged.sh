#!/bin/bash
set -euo pipefail

# 训练启动脚本（推荐）：Transfuser + MoE + 多模态 + 分阶段训练（staged）
#
# 默认使用 Hydra 配置：
#   navsim/planning/script/config/training/default_training_transfuser_moe_staged.yaml
#
# 可通过环境变量覆盖：
#   CUDA_VISIBLE_DEVICES   (默认: 0,1,2,3,4,5,6,7)
#   EXPERIMENT_NAME        (默认: training_transfuser_moe_staged)
#   TRAIN_TEST_SPLIT       (默认: navtrain)
#   USE_CACHE_ONLY         (默认: 1; 1=CacheOnlyDataset, 0=正常构建数据集)
#   FORCE_CACHE_REBUILD    (默认: 0; 1=强制重建缓存)
#
# 示例：
#   CUDA_VISIBLE_DEVICES=0 EXPERIMENT_NAME=debug TRAIN_TEST_SPLIT=navmini ./scripts/train_transfuser_moe_staged.sh

source scripts/SetPath.sh

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-training_transfuser_moe_staged}
TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtrain}
USE_CACHE_ONLY=${USE_CACHE_ONLY:-1}
FORCE_CACHE_REBUILD=${FORCE_CACHE_REBUILD:-0}

USE_CACHE_WITHOUT_DATASET=true
FORCE_CACHE_COMPUTATION=false
if [[ "${USE_CACHE_ONLY}" == "0" ]]; then
  USE_CACHE_WITHOUT_DATASET=false
fi
if [[ "${FORCE_CACHE_REBUILD}" == "1" ]]; then
  FORCE_CACHE_COMPUTATION=true
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_training.py" \
  --config-path=config/training \
  --config-name=default_training_transfuser_moe_staged \
  experiment_name="${EXPERIMENT_NAME}" \
  train_test_split="${TRAIN_TEST_SPLIT}" \
  cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
  use_cache_without_dataset="${USE_CACHE_WITHOUT_DATASET}" \
  force_cache_computation="${FORCE_CACHE_COMPUTATION}"



