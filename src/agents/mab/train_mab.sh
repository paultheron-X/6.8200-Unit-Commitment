export PYTHONPATH=$PYTHONPATH:$(pwd)/src

python src/agents/mab/train.py \
       --save_dir results/agents/mab/$1 \
       --env_name $1 \
       --env_fn src/rl4uc/data/envs/$1
