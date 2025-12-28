python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  --config-name default_training_moe \
  agent=diffusiondrive_agent \
  experiment_name=training_diffusiondrive_agent_layerwise_moe \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=100 \
  cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
  use_cache_without_dataset=True \
  force_cache_computation=False