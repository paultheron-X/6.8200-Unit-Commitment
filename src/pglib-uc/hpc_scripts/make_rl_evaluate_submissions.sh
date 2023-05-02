#!/bin/bash 

# {10: 32364, 20: 5033, 30: 8835, 40: 11176, 50: 29476, 60: 2416}

qsub submit_evaluate_rl.sh $HOME/rl-convex-opt/results/22-08-26_policies/g10_32364
qsub submit_evaluate_rl.sh $HOME/rl-convex-opt/results/22-08-26_policies/g20_5033
qsub submit_evaluate_rl.sh $HOME/rl-convex-opt/results/22-08-26_policies/g30_8835
qsub submit_evaluate_rl.sh $HOME/rl-convex-opt/results/22-08-26_policies/g40_11176
qsub submit_evaluate_rl.sh $HOME/rl-convex-opt/results/22-08-26_policies/g50_29476