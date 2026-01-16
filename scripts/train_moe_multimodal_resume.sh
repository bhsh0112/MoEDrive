#!/bin/bash
# Source environment variables
# 确保在正确的目录下运行
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

source scripts/SetPath.sh

# 恢复训练脚本：从checkpoint继续训练
#
# 使用方法：
# 1. 修改 CHECKPOINT_PATH 变量为要恢复的checkpoint路径
# 2. 可以修改 max_epochs 增加训练轮数
# 3. 其他配置必须与原始训练保持一致
#
# 示例：
# bash scripts/train_moe_multimodal_resume.sh \
#   exp/training_transfuser_moe_multimodal_optimized/2026.01.12.01.01.54/lightning_logs/version_0/checkpoints/epoch=149-step=25050.ckpt \
#   200
#
# 参数说明：
# $1: checkpoint路径（必需）
# $2: 新的max_epochs（可选，默认200）

# 检查参数
if [ -z "$1" ]; then
    echo "错误: 必须提供checkpoint路径"
    echo "用法: bash scripts/train_moe_multimodal_resume.sh <checkpoint_path> [max_epochs]"
    echo "示例: bash scripts/train_moe_multimodal_resume.sh exp/.../checkpoints/epoch=149-step=25050.ckpt 200"
    exit 1
fi

CHECKPOINT_PATH="$1"
NEW_MAX_EPOCHS="${2:-200}"

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: Checkpoint文件不存在: $CHECKPOINT_PATH"
    exit 1
fi

echo "=========================================="
echo "恢复训练配置"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "新的最大轮数: $NEW_MAX_EPOCHS"
echo "实验名称: training_transfuser_moe_multimodal_optimized"
echo ""
echo "注意: 所有模型配置必须与原始训练保持一致"
echo "=========================================="
echo ""

# 方法1: 尝试转义等号（如果不行，使用符号链接方法）
# Hydra使用 \= 来转义等号，在bash中需要 \\= 才能得到 \=
ESCAPED_CHECKPOINT_PATH="${CHECKPOINT_PATH//=/\\=}"

echo "原始checkpoint路径: $CHECKPOINT_PATH"
echo "转义后的路径: $ESCAPED_CHECKPOINT_PATH"
echo ""
echo "如果转义方法不工作，请使用: bash scripts/train_moe_multimodal_resume_symlink.sh"
echo ""

# 恢复训练
# 注意：如果这个脚本仍然报错，请使用 train_moe_multimodal_resume_symlink.sh（使用符号链接方法）
# 确保 Python 可以找到模块：将当前目录添加到 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES=4,5,6,7 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  agent=transfuser_agent \
  experiment_name=training_transfuser_moe_multimodal_optimized \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=${NEW_MAX_EPOCHS} \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  trainer.params.gradient_clip_val=1.0 \
  trainer.params.gradient_clip_algorithm=norm \
  +trainer.params.ckpt_path="${ESCAPED_CHECKPOINT_PATH}" \
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

# 如果上面的ckpt_path报错（可能是旧版本Lightning），请尝试：
# 1. 将 trainer.params.ckpt_path 改为 trainer.params.resume_from_checkpoint
# 2. 或者检查PyTorch Lightning版本：pip show pytorch-lightning

