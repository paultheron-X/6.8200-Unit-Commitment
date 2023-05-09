#/bin/bash

# set today's date to use as save directory
date=$(date +"%d-%m-%Y--%H:%M")

if [ -z "$1" ]
then
    echo $0: no argument for num gen, defaulting to 5
    num_gen=5
else    
    num_gen=$1
fi

if [ -z "$2" ]
then
    echo $0: no argument for algo, defaulting to ppo_async. Values are supposed to be in {ppo, ppo_async, q_learning}
    agent=ppo_async
else    
    agent=$2
fi


save_dir=results/guided_tree_search/${agent}/test_${date}/
params_filename=results/agents/${agent}/${num_gen}gen/params.json #results/tmp/params.json 
env_params_filename=results/agents/${agent}/${num_gen}gen/env_params.json #src/ts4uc_scripts/data/day_ahead/${num_gen}gen/30min/env_params.json
policy_filename=results/agents/${agent}/${num_gen}gen/ac_final.pt #results/tmp/ac_final.pt
horizon=2
branching_threshold=0.05
tree_search_func_name=uniform_cost_search_robust
testfile=data/day_ahead/${num_gen}gen/30min/profile_2019-11-09.csv
heuristic_method=${9:-none} 

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

python src/tree_search_utils/day_ahead_forest.py --save_dir $save_dir \
                                            --policy_params_fn $params_filename \
                                            --env_params_fn $env_params_filename \
                                            --test_data $testfile \
                                            --branching_threshold $branching_threshold \
                                            --horizon $horizon \
                                            --num_trees 100 \
                                            --tree_search_func_name $tree_search_func_name \
                                            --heuristic_method $heuristic_method \
                                            --seed 1 \
                                            --policy_filename $policy_filename \
                                            --obs_corrupter auto \
                                            --action_method max_min

