#!/bin/bash
# 训练使用 MOE Trajectory Head 的模型
# 此脚本使用 MOE 替换了 diffusion-based trajectory head

# 指定使用的GPU序号（例如：0,1,2,3 表示使用GPU 0、1、2、3）
# 修改下面的 CUDA_VISIBLE_DEVICES 值来指定你想要的GPU
CUDA_VISIBLE_DEVICES=4,5,6,7 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  agent=diffusiondrive_agent \
  experiment_name=training_diffusiondrive_agent_moe_trajectory \
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

