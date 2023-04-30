#/bin/bash

# set today's date to use as save directory
date=$(date +"%y-%m-%d")

num_g=5
save_dir=result/guided_tree_search/test_g$num_g/
params_filename=src/ts4uc_scripts/data/dummy_policies/g5/params.json #results/tmp/params.json
env_params_filename=src/ts4uc_scripts/data/dummy_policies/g5/env_params.json #src/ts4uc_scripts/data/day_ahead/${num_g}gen/30min/env_params.json
policy_filename=src/ts4uc_scripts/data/dummy_policies/g5/ac_final.pt #results/tmp/ac_final.pt
horizon=2
branching_threshold=0.05
tree_search_func_name=uniform_cost_search
testfile=src/ts4uc_scripts/data/day_ahead/5gen/30min/profile_2019-11-09.csv
heuristic_method=${9:-none} 


bash src/ts4uc_scripts/hpc_scripts/submit_tree_search.sh $save_dir \
                            $params_filename \
                            $env_params_filename \
                            $policy_filename \
                            $horizon \
                            $branching_threshold \
                            $tree_search_func_name \
                            $testfile

