export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# This traines a PPO agent for 500k timesteps

if [ -z "$1" ]
then
    echo $0: no argument for num gen, defaulting to 5
    num_gen=5
else    
    num_gen=$1
fi

python src/agents/ppo/train.py \
       --save_dir results/agents/test/ppo/${num_gen}gen \
       --num_gen ${num_gen} \
       --timesteps 500000 \
       --workers 4 \
       --steps_per_epoch 1000 \
       --entropy_coef 0.05 \
	   --update_epochs 10 \
	   --clip_ratio 0.1 \
	   --ac_learning_rate 0.00003 \
	   --cr_learning_rate 0.0003 \
