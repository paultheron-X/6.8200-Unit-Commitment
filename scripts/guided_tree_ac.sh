#/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

save_dir=results/guided_tree_search/test_${date}/
params_filename=results/agents/ppo_async/params.json #results/tmp/params.json 
env_params_filename=results/agents/ppo_async/env_params.json #src/ts4uc_scripts/data/day_ahead/${num_g}gen/30min/env_params.json
policy_filename=results/agents/ppo_async/ac_final.pt #results/tmp/ac_final.pt
horizon=2
branching_threshold=0.05
tree_search_func_name=uniform_cost_search
testfile=data/day_ahead/5gen/30min/profile_2019-11-09.csv
heuristic_method=${9:-none} 

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

python src/tree_search_utils/day_ahead.py --save_dir $save_dir \
                                            --policy_params_fn $params_filename \
                                            --env_params_fn $env_params_filename \
                                            --test_data $testfile \
                                            --branching_threshold $branching_threshold \
                                            --horizon $horizon \
                                            --num_scenarios 100 \
                                            --tree_search_func_name $tree_search_func_name \
                                            --heuristic_method $heuristic_method \
                                            --seed 1 \
                                            --policy_filename $policy_filename 

