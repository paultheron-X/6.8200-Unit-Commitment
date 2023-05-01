#!/bin/bash -l
#$ -wd /home/uclqpde/Scratch/output
#$ -M patrick.demars.14@ucl.ac.uk
#$ -l mem=8G

# number of array jobs: corresponds to number of test profiles.
#$ -t 1-20

save_dir=$1
params_filename=$2
env_params_filename=$3
policy_filename=$4
horizon=$5
branching_threshold=$6
tree_search_func_name=$7
testfile=$8
heuristic_method=${9:-none}

number=$SGE_TASK_ID
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
												  --policy_filename $policy_filename \

