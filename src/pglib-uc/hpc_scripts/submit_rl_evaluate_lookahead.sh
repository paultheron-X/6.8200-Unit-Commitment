#!/bin/bash -l
#$ -wd /home/uclqpde/Scratch/output
#$ -l mem=4G
#$ -l h_rt=12:00:00

# Load python 
module load python3/3.7

policy_dir=$1

echo $policy_dir

cd $TMPDIR

python $HOME/rl-convex-opt/evaluate_rl_lookahead.py ${policy_dir} $HOME/rl-convex-opt/test_data_windy