export PYTHONPATH=$PYTHONPATH:$(pwd)/src




for agent in ppo_async a3c # ppo sac a3c random
do
    for num_gen in 5 10 
    do
        echo "----------------------- Running Tree agent eval ${agent} with ${num_gen} generators"
        bash scripts/final_scripts/callable/launch_tree_eval_guided_per_agent.sh $num_gen $agent
    done
done


for num_gen in 5 10
do
    echo "----------------------- Running Unguided Tree eval with ${num_gen} generators"
    bash scripts/final_scripts/callable/launch_tree_eval_unguided.sh $num_gen
done