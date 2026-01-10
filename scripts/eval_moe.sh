python navsim/planning/script/run_pdm_score.py train_test_split=navtest \
       agent=diffusiondrive_agent \
       worker=ray_distributed \
       agent.checkpoint_path=ckpts/base_transfuser.ckpt \
        experiment_name=diffusiondrive_agent_eval