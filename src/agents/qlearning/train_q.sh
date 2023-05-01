export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# This traines a PPO agent for 500k timesteps

python src/ts4uc/agents/qlearning/train.py \
       --save_dir results/qlearning/tmp \
       --env_name 5gen \
       --nb_epochs 500 \
