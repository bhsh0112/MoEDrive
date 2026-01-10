#!/bin/bash
# Source environment variables
source scripts/SetPath.sh

# 指定使用的GPU序号（例如：4,5,6,7 表示使用GPU 4、5、6、7）
# 修改下面的 CUDA_VISIBLE_DEVICES 值来指定你想要的GPU
#
# DDP策略配置：
# - transfuser_model.py 使用了 MoELayerwiseTransformerDecoder（Layer-wise MoE）
# - MoE 模型中，每个训练步骤只激活部分专家，导致其他专家参数未被使用
# - 需要设置 strategy=ddp_find_unused_parameters_true 来允许未使用的参数

CUDA_VISIBLE_DEVICES=4,5,6,7 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  agent=transfuser_agent \
  experiment_name=training_transfuser_moe_decoder \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=100 \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  agent.config.use_moe_decoder=True 