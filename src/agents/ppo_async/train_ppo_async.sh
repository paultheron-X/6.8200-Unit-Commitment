export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# This traines a PPO agent for 500k timesteps

python src/ts4uc/agents/ppo_async/train.py \
       --save_dir results/ppo_async/ \
       --workers 4 \
       --num_gen 5 \
       --timesteps 500000 \
       --env_fn src/rl4uc/data/envs/5gen.json \
       --env_name \
       --epochs 5000 \
       --entropy_coef 0.05 \
	   --update_epochs 10 \
	   --clip_ratio 0.1 \
	   --ac_learning_rate 0.00003 \
	   --cr_learning_rate 0.0003 \
