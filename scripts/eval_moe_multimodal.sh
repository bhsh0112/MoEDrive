#!/bin/bash
# 评估脚本：验证基于MoE的_tf_decoder + 基于transfuser的trajectory_head的多模态模型
#
# 使用说明：
#   bash scripts/eval_moe_multimodal.sh [checkpoint_path] [experiment_name]
#
# 参数说明：
# - agent=transfuser_agent: 使用transfuser agent（支持MoE decoder和多模态trajectory_head）
# - agent.config.use_moe_decoder=True: 启用MoE decoder（必须）
# - agent.config.multimodal_trajectory=True: 启用多模态轨迹预测（必须）
# - agent.checkpoint_path: 指定训练好的多模态模型checkpoint路径
# - experiment_name: 实验名称，用于结果输出
#
# 配置参数（需要与训练时保持一致）：
# - moe_num_experts: MoE专家数量（必须与num_trajectory_modes匹配）
# - moe_top_k: 在多模态模式下，建议使用所有专家（等于moe_num_experts）
# - num_trajectory_modes: 轨迹模式数量（必须与moe_num_experts匹配）

# 配置变量
CHECKPOINT_PATH="${1:-ckpts/moe_multimodal_model.ckpt}"
EXPERIMENT_NAME="${2:-transfuser_moe_multimodal_eval}"

# 检查checkpoint路径
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "警告: Checkpoint文件不存在: $CHECKPOINT_PATH"
    echo "请修改脚本中的CHECKPOINT_PATH或作为第一个参数传入"
    echo "用法: bash scripts/eval_moe_multimodal.sh [checkpoint_path] [experiment_name]"
    echo "  示例: bash scripts/eval_moe_multimodal.sh ckpts/moe_multimodal_model.ckpt my_multimodal_experiment"
    echo ""
fi

echo "=== 评估模式: 多模态MoE模型 ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Experiment: $EXPERIMENT_NAME"
echo ""

# 多模态评估命令
# 注意：moe_num_experts 和 num_trajectory_modes 必须匹配
python navsim/planning/script/run_pdm_score.py \
  train_test_split=navtest \
  agent=transfuser_agent \
  worker=ray_distributed \
  agent.checkpoint_path="${CHECKPOINT_PATH}" \
  experiment_name="${EXPERIMENT_NAME}" \
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

