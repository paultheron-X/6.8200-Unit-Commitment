export PYTHONPATH=$PYTHONPATH:$(pwd)/src

python src/agents/qlearning/train.py \
       --save_dir results/agents/qlearning/$1 \
       --env_name $1 \
       --env_fn src/rl4uc/data/envs/$1.json \
       --config $2 \
