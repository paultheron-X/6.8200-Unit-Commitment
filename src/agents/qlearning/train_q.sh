export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# This traines a PPO agent for 500k timesteps

python src/agents/qlearning/train.py \
       --save_dir results/agents/qlearning \
       --env_name 5gen \
       --env_fn src/rl4uc/data/envs/5gen.json \
       --nb_epochs 1000 \
