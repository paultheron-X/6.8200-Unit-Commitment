#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# make sure to run from root directory of repo
import sys
sys.path.append('src')


from rl4uc.environment import make_env

from tree_search_utils.scenarios import get_net_demand_scenarios, get_scenarios
from tree_search_utils.day_ahead import solve_day_ahead
from tree_search_utils.algos import uniform_cost_search
from ts4uc.agents.ppo_async.ac_agent import ACAgent
import agents.helpers as helpers

import numpy as np 
import pandas as pd 
import torch
import json

POLICY_FILENAME = 'src/ts4uc_scripts/data/dummy_policies/g5/ac_final.pt' 
POLICY_PARAMS_FN = 'src/ts4uc_scripts/data/dummy_policies/g5/params.json'
ENV_PARAMS_FN = 'src/ts4uc_scripts/data/dummy_policies/g5/env_params.json'
TEST_DATA_FN = 'src/ts4uc_scripts/data/day_ahead/5gen/30min/profile_2019-11-09.csv'
HORIZON = 2
BRANCHING_THRESHOLD = 0.05
TREE_SEARCH_FUNC_NAME = 'uniform_cost_search'
SEED = 1 
NUM_SCENARIOS = 100
TEST_SAMPLE_SEED = 999
TIME_PERIODS = 4
NUM_SAMPLES = 1000

def test_uniform_cost_search():

        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # Load parameters
        env_params = json.load(open(ENV_PARAMS_FN))
        policy_params = json.load(open(POLICY_PARAMS_FN))

        # Load profile 
        profile_df = pd.read_csv(TEST_DATA_FN)[:TIME_PERIODS]

        params = {'horizon': HORIZON,
                          'branching_threshold': BRANCHING_THRESHOLD}

        # Init env
        env = make_env(mode='test', profiles_df=profile_df, **env_params)

        # Load policy
        policy = ACAgent(env, test_seed=SEED, **policy_params)
        policy.load_state_dict(torch.load(POLICY_FILENAME))
        policy.eval()

        # Generate scenarios for demand and wind errors
        # scenarios = get_net_demand_scenarios(profile_df, env, NUM_SCENARIOS)
        demand_scenarios, wind_scenarios = get_scenarios(profile_df, env, NUM_SCENARIOS)

        solve_returns = solve_day_ahead(env=env, 
                                          demand_scenarios=demand_scenarios, 
                                          wind_scenarios=wind_scenarios,
                                          global_outage_scenarios=None,
                                          tree_search_func=uniform_cost_search,
                                          policy=policy,
                                          **params)
        schedule_result = solve_returns[0]

        # Get distribution of costs for solution by running multiple times through environment
        results = helpers.test_schedule(env, schedule_result, TEST_SAMPLE_SEED, NUM_SAMPLES)
        mean_cost = np.mean(results['total_cost'])

        assert np.isclose(mean_cost, 22608.283119377622), "Costs were: {}".format(mean_cost)

if __name__ == '__main__':
        test_uniform_cost_search()