
export PYTHONPATH=$PYTHONPATH:$(pwd)/src



for agent in a3c #ppo sac a3c random
do
    for num_gen in 5 10 
    do
        echo "----------------------- Running raw agent eval ${agent} with ${num_gen} generators"
        bash scripts/final_scripts/callable/launch_raw_agent_eval_per_agent.sh $num_gen $agent
    done
done
