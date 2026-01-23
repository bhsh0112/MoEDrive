#!/bin/bash
# 评估脚本：验证基于MoE的_tf_decoder + 基于transfuser的trajectory_head的模型
#
# 使用说明：
# 1. 单模态MoE模型（默认）：
#    bash scripts/eval_moe.sh [checkpoint_path] [experiment_name]
#
# 2. 多模态MoE模型：
#    bash scripts/eval_moe.sh [checkpoint_path] [experiment_name] true
#    或使用单独的脚本: bash scripts/eval_moe_multimodal.sh
#
# 参数说明：
# - agent=transfuser_agent: 使用transfuser agent（支持MoE decoder和trajectory_head）
# - agent.config.use_moe_decoder=True: 启用MoE decoder（必须）
# - agent.checkpoint_path: 指定训练好的模型checkpoint路径
# - experiment_name: 实验名称，用于结果输出
#
# 配置参数（需要与训练时保持一致）：
# - moe_num_experts: MoE专家数量（必须与训练时一致，默认20）
# - moe_top_k: 每个样本使用的专家数量（单模态默认2，多模态建议使用所有专家）
# - multimodal_trajectory: 是否启用多模态轨迹预测（默认False）

# 配置变量
CHECKPOINT_PATH="${1:-ckpts/moe_test_v1.2.ckpt}"
# 注意：此脚本默认配置为20个专家，如果checkpoint使用不同数量的专家，请修改moe_num_experts参数
EXPERIMENT_NAME="${2:-transfuser_moe_eval}"
MULTIMODAL_MODE="${3:-false}"  # 第三个参数控制是否启用多模态（true/false）

# 检查checkpoint路径
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "警告: Checkpoint文件不存在: $CHECKPOINT_PATH"
    echo "请修改脚本中的CHECKPOINT_PATH或作为第一个参数传入"
    echo "用法: bash scripts/eval_moe.sh [checkpoint_path] [experiment_name] [multimodal_mode]"
    echo "  示例: bash scripts/eval_moe.sh ckpts/moe_model.ckpt my_experiment false"
    echo ""
fi

# 输出模式信息
if [ "$MULTIMODAL_MODE" = "true" ]; then
    echo "=== 评估模式: 多模态MoE模型 ==="
    echo "Checkpoint: $CHECKPOINT_PATH"
    echo "Experiment: $EXPERIMENT_NAME"
    echo ""
    
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
else
    echo "=== 评估模式: 单模态MoE模型 ==="
    echo "Checkpoint: $CHECKPOINT_PATH"
    echo "Experiment: $EXPERIMENT_NAME"
    echo ""
    
    python navsim/planning/script/run_pdm_score.py \
      train_test_split=navtest \
      agent=transfuser_agent \
       worker=ray_distributed \
      agent.checkpoint_path="${CHECKPOINT_PATH}" \
      experiment_name="${EXPERIMENT_NAME}" \
  agent.config.use_moe_decoder=True \
  +agent.config.moe_num_experts=20 \
  +agent.config.moe_top_k=2 \
  +agent.config.moe_router_temperature=1.0 \
  +agent.config.moe_load_balance_coef=5e-3 \
  +agent.config.moe_router_z_loss_coef=1e-3 \
  +agent.config.moe_aux_loss_weight=0.5
fi
