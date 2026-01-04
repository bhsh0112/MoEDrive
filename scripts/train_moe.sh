python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  agent=diffusiondrive_agent \
  experiment_name=training_diffusiondrive_agent_layerwise_moe_no_noise_traj_router \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=100 \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
  use_cache_without_dataset=True \
  force_cache_computation=False