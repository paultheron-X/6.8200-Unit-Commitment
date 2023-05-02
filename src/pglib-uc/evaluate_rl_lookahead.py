#!/usr/bin/env python

import json
import numpy as np
import pandas as pd
import time
import os
import torch

from rl4uc.environment import make_env
from ts4uc.helpers import test_schedule
from ts4uc.agents.ppo_async.ac_agent import ACAgent
from ts4uc.tree_search.day_ahead import solve_day_ahead
from ts4uc.tree_search.scenarios import get_scenarios
from ts4uc.tree_search.algos import a_star


def evaluate_policy_with_lookahead(
    policy_dir,
    test_dir="./test_data_windy",
    n_days=None,
    evaluate=True,
    num_samples=5000,
    test_sample_seed=995,
    num_scenarios=100,
):
    """
    Evaluate a policy with a 1-step lookahead strategy

    Returns:

    - all_results: a pd.DataFrame of costs, load shedding etc. for each scenario
    - schedules: dictionary of schedules
    """

    params = json.load(open(os.path.join(policy_dir, "params.json")))
    env_params = json.load(open(os.path.join(policy_dir, "env_params.json")))

    env = make_env(num_gen=env_params["num_gen"])
    policy = ACAgent(env, **params)
    policy.load_state_dict(torch.load(os.path.join(policy_dir, "ac_final.pt")))
    policy.test_seed = 123
    policy.eval()

    # Make a directory for storing the results to unseen problems
    os.makedirs(os.path.join(policy_dir, "test_solutions_lookahead"), exist_ok=True)

    all_results = []
    schedules = {}
    times = {}

    fs = [x for x in os.listdir(test_dir) if ".csv" in x]
    fs.sort()

    if n_days is not None:
        fs = fs[:n_days]

    for f in fs:

        # Retrieve the date
        date = f.split("_")[1].split(".")[0]

        print(f"Running {date}...")

        # Solve the UC problem
        profile_df = pd.read_csv(os.path.join(test_dir, f))
        profile_df["wind"] = profile_df.wind * env.num_gen / 10.0
        profile_df["demand"] = profile_df.demand * env.num_gen / 10.0

        demand_scenarios, wind_scenarios = get_scenarios(profile_df, env, num_scenarios)

        ts_params = {
            "horizon": 1,
            "branching_threshold": 0.05,
            "heuristic_method": None,
        }
        env = make_env(mode="test", profiles_df=profile_df, **env_params)

        s = time.time()
        schedule, period_times, breadths = solve_day_ahead(
            env=env,
            demand_scenarios=demand_scenarios,
            wind_scenarios=wind_scenarios,
            global_outage_scenarios=None,
            tree_search_func=a_star,
            policy=policy,
            **ts_params,
        )
        times[date] = time.time() - s
        schedules[date] = schedule

        if evaluate:
            # Evaluate the schedule over realisations of uncertainty
            results = test_schedule(env, schedule, test_sample_seed, num_samples)
            results["date"] = date
            all_results.append(results)

    if evaluate:
        # Concat and save the main results (costs, LOLP etc)
        all_results = pd.concat(all_results)
        all_results.to_csv(
            os.path.join(policy_dir, "test_solutions_lookahead", "results.csv")
        )

    # Save the solutions
    for key, schedule in schedules.items():
        pd.DataFrame(schedule, dtype=int).to_csv(
            os.path.join(
                policy_dir, "test_solutions_lookahead", f"{key}_RL_lookahead.csv"
            ),
            index=False,
        )

    # Save the time taken
    pd.DataFrame.from_dict(times, orient="index").to_csv(
        os.path.join(policy_dir, "test_solutions_lookahead", "times.csv")
    )

    return all_results, schedules


if __name__ == "__main__":

    import sys

    policy_dir = sys.argv[1]
    test_dir = sys.argv[2]
    print("*********************************************************************")
    print(policy_dir)
    print("*********************************************************************")
    all_results, schedules = evaluate_policy_with_lookahead(policy_dir, test_dir)
