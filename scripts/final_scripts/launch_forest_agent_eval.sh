export PYTHONPATH=$PYTHONPATH:$(pwd)/src

for agent in a3c #a3c sac a3c random
do
    for num_gen in 5 10 
    do
        for action_method in max_min max_avg max_min_central
        do
            for obs_corrupter in auto box
            do
                for num_trees in 10 50 100
                do
                    echo ----------------------- Running Forest agent eval ${agent} with ${num_gen} generators, ${action_method} action method, ${obs_corrupter} obs corrupter, ${num_trees} trees
                    bash scripts/final_scripts/callable/launch_forest_per_agent.sh $num_gen $agent $action_method $obs_corrupter $num_trees
                done
            done
        done
    done
done