#!/bin/bash
# Source environment variables
source scripts/SetPath.sh

# 指定使用的GPU序号（例如：0,1,2,3 表示使用GPU 0、1、2、3）
# 修改下面的 CUDA_VISIBLE_DEVICES 值来指定你想要的GPU

# 错误原因分析：
# - use_cache_without_dataset=True 时，使用 CacheOnlyDataset 只读取已有缓存，不会重新计算
# - force_cache_computation=True 需要重新计算缓存
# - 这两个选项冲突，不能同时使用
#
# 由于缓存文件损坏（zlib解压错误），需要重新生成缓存：
# - 设置 use_cache_without_dataset=False 来正常构建数据集
# - 设置 force_cache_computation=True 来强制重新生成缓存
#
# 如果缓存正常，可以改为：
#   use_cache_without_dataset=True \
#   force_cache_computation=False

CUDA_VISIBLE_DEVICES=0,1,2,3 python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  agent=transfuser_agent \
  experiment_name=training_transfuser_vanilla_head \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=100 \
  cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
  use_cache_without_dataset=True \
  force_cache_computation=False
