export PYTHONPATH=$PYTHONPATH:$(pwd)/src

python src/agents/sac/train.py \
    --save_dir results/agents/sac/$1 \
    --env_name $1 \
    --buffer_size 2000 \
    --workers 4 \
    --timesteps 1000000 \
    --steps_per_epoch 1000 \
