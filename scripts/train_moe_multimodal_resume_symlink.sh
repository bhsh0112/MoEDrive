#!/bin/bash
# Source environment variables
# 确保在正确的目录下运行
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

source scripts/SetPath.sh

# 恢复训练脚本（使用符号链接方法）：从checkpoint继续训练
#
# 这个方法通过创建符号链接来避免checkpoint路径中的等号问题
# 这是最可靠的方法，完全避免Hydra解析问题
#
# 使用方法：
# bash scripts/train_moe_multimodal_resume_symlink.sh <checkpoint_path> [max_epochs]

# 检查参数
if [ -z "$1" ]; then
    echo "错误: 必须提供checkpoint路径"
    echo "用法: bash scripts/train_moe_multimodal_resume_symlink.sh <checkpoint_path> [max_epochs]"
    exit 1
fi

CHECKPOINT_PATH="$1"
NEW_MAX_EPOCHS="${2:-200}"

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: Checkpoint文件不存在: $CHECKPOINT_PATH"
    exit 1
fi

# 创建符号链接，避免路径中的等号问题
SYMLINK_PATH="/tmp/checkpoint_resume_$$.ckpt"
ABS_CHECKPOINT_PATH=$(realpath "$CHECKPOINT_PATH")
ln -sf "$ABS_CHECKPOINT_PATH" "$SYMLINK_PATH"

echo "=========================================="
echo "恢复训练配置（使用符号链接方法）"
echo "=========================================="
echo "原始Checkpoint: $CHECKPOINT_PATH"
echo "符号链接: $SYMLINK_PATH"
echo "新的最大轮数: $NEW_MAX_EPOCHS"
echo "=========================================="
echo ""

# 清理函数：训练结束后删除符号链接
cleanup() {
    if [ -L "$SYMLINK_PATH" ]; then
        rm -f "$SYMLINK_PATH"
        echo "已清理符号链接: $SYMLINK_PATH"
    fi
}
trap cleanup EXIT

# 恢复训练
# 确保 Python 可以找到模块：将当前目录添加到 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  agent=transfuser_agent \
  experiment_name=training_transfuser_moe_multimodal_optimized \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=${NEW_MAX_EPOCHS} \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  trainer.params.gradient_clip_val=1.0 \
  trainer.params.gradient_clip_algorithm=norm \
  +trainer.params.ckpt_path="${SYMLINK_PATH}" \
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

