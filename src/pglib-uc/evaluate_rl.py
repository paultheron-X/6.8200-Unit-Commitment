#!/usr/bin/env python

import json
import numpy as np
import pandas as pd
import os
import time
import torch

from rl4uc.environment import make_env
from ts4uc.helpers import test_schedule
from ts4uc.agents.ppo_async.ac_agent import ACAgent


def solve_model_free(env, policy, deterministic=True):
    """
    Solve a UC problem using trained policy (no tree search)
    """
    obs = env.reset()
    final_schedule = np.zeros((env.episode_length, env.num_gen))
    for t in range(env.episode_length):
        a, sub_obs, sub_acts, log_probs = policy.generate_action(
            env, obs, argmax=deterministic
        )
        obs, reward, done = env.step(a, deterministic=True)
        final_schedule[t, :] = a
    return final_schedule


def evaluate_policy(
    policy_dir,
    test_dir="./test_data_windy",
    n_days=None,
    deterministic=True,
    evaluate=True,
    num_samples=5000,
    test_sample_seed=995,
):
    """
    Evaluate a policy (saved in policy_dir) on a batch of test problems (test_dir)

    Returns:

    - all_results: a pd.DataFrame of costs, load shedding etc. for each scenario
    - schedules: dictionary of schedules
    """

    params = json.load(open(os.path.join(policy_dir, "params.json")))
    env_params = json.load(open(os.path.join(policy_dir, "env_params.json")))

    env = make_env(num_gen=env_params["num_gen"])
    policy = ACAgent(env, **params)
    policy.load_state_dict(torch.load(os.path.join(policy_dir, "ac_final.pt")))
    policy.eval()

    # Make a directory for storing the results to unseen problems
    os.makedirs(os.path.join(policy_dir, "test_solutions"), exist_ok=True)

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

        env = make_env(mode="test", profiles_df=profile_df, **env_params)
        s = time.time()
        schedule = solve_model_free(env, policy, deterministic=deterministic)
        schedules[date] = schedule
        times[date] = time.time() - s

        if evaluate:
            # Evaluate the schedule over realisations of uncertainty
            results = test_schedule(env, schedule, test_sample_seed, num_samples)
            results["date"] = date
            all_results.append(results)

    if evaluate:
        # Concat and save the main results (costs, LOLP etc)
        all_results = pd.concat(all_results)
        all_results.to_csv(os.path.join(policy_dir, "test_solutions", "results.csv"))

    # Save the solutions
    for key, schedule in schedules.items():
        pd.DataFrame(schedule, dtype=int).to_csv(
            os.path.join(policy_dir, "test_solutions", f"{key}_RL.csv"), index=False
        )

    # Save the time taken
    pd.DataFrame.from_dict(times, orient="index").to_csv(
        os.path.join(policy_dir, "test_solutions", "times.csv")
    )

    return all_results, schedules


if __name__ == "__main__":

    import sys

    policy_dir = sys.argv[1]
    test_dir = sys.argv[2]
    print("*********************************************************************")
    print(policy_dir)
    print("*********************************************************************")
    all_results, schedules = evaluate_policy(
        policy_dir, test_dir=test_dir, evaluate=True
    )
