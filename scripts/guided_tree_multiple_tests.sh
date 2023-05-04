#/bin/bash

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

save_dir=results/guided_tree_search/test_${date}/
params_filename=results/agents/${agent}/${num_gen}gen/params.json #results/tmp/params.json 
env_params_filename=results/agents/${agent}/${num_gen}gen/env_params.json #src/ts4uc_scripts/data/day_ahead/${num_gen}gen/30min/env_params.json
policy_filename=results/agents/${agent}/${num_gen}gen/ac_final.pt #results/tmp/ac_final.pt
horizon=2
branching_threshold=0.05
tree_search_func_name=uniform_cost_search
testfile=data/day_ahead/${num_gen}gen/30min/profile_2019-11-09.csv
heuristic_method=${9:-none} 

# loop through all test files
for testfile in data/day_ahead/5gen/30min/*.csv
do
    echo "Submitting test file: $testfile"
    bash scripts/tree_search/submit_tree_search.sh $save_dir \
                            $params_filename \
                            $env_params_filename \
                            $policy_filename \
                            $horizon \
                            $branching_threshold \
                            $tree_search_func_name \
                            $testfile
done

