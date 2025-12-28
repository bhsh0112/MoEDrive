python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  --config-name default_training_moe \
  agent=diffusiondrive_agent \
  experiment_name=training_diffusiondrive_agent_layerwise_moe \
  train_test_split=navtrain \
  split=trainval \
  trainer.params.max_epochs=100 \
  cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
  use_cache_without_dataset=True \
  force_cache_computation=False \
  trainer.params.strategy=ddp_find_unused_parameters_true \
  +agent.config.moe_num_experts=8 \
  +agent.config.moe_top_k=1 \
  +agent.config.moe_router_temperature=0.7 \
  +agent.config.moe_router_noise_std=0.1 \
  +agent.config.moe_load_balance_coef=5e-2 \
  +agent.config.moe_router_z_loss_coef=5e-3 \
  +agent.config.moe_aux_loss_weight=2.0