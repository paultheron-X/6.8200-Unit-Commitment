#!/bin/bash -l
#$ -wd /home/uclqpde/Scratch/output
#$ -l mem=4G
#$ -l h_rt=13:00:00

# number of array jobs: corresponds to number of test profiles.
#$ -t 1-20

# Load python and Gurobi 
module load python3/3.7
module load gurobi/9.1.2

save_dir=$1
num_gen=$2
quantiles=$3

echo $quantiles

paramfile=$HOME/rl-convex-opt/params.txt

mkdir -p $HOME/Scratch/results/${save_dir}

number=$SGE_TASK_ID
index="`sed -n ${number}p $paramfile | awk '{print $1}'`"
test_data="`sed -n ${number}p $paramfile | awk '{print $2}'`"

cat $test_data

cd $TMPDIR

python $HOME/rl-convex-opt/solve_and_test.py --save_dir $HOME/Scratch/results/${save_dir} --num_samples 5000 --env_params_fn $HOME/rl-convex-opt/envs/${num_gen}gen.json --test_data $test_data --reserve_sigma 4 --quantiles_str $quantiles --tee
