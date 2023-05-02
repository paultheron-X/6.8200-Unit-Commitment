#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 10:18:16 2020

@author: patrickdemars
"""

import numpy as np
from sys import exit

from rl4uc.environment import make_env
#from ts4uc import helpers as ts4uc_helpers


def calculate_piecewise_production(gen_info, idx, n_hrs, N=4):
    mws = np.linspace(gen_info.min_output.values[idx],
                      gen_info.max_output.values[idx], N)
    costs = n_hrs*((gen_info.a.values[idx]*(mws**2) +
                    gen_info.b.values[idx]*mws +
                    gen_info.c.values[idx]))
    pairs = []
    for mw, cost in zip(mws, costs):
        pairs.append({"mw": float(mw), "cost": float(cost)})
    return pairs

def create_problem_dict(demand, wind, env, 
                        reserve_pct=None, reserve_mw=None,
                        n_minus_one=False, prof_name = None): 
                        
    """
    Create a dictionary defining the problem for input to the pglib model.

    Args:
        - demand (array): demand profile
        - params (dict): parameters file, gives number of generators, dispatch frequency.

        
    """
    if wind is not None:
        net_demand = demand - wind
    gen_info = env.gen_info

    if reserve_pct is not None:
        reserves = np.array([a*reserve_pct/100 for a in net_demand]) # Reserve margin is % of net demand
    elif reserve_mw is not None:
        if (reserve_mw.size==1):
            reserves = reserve_mw * np.ones(len(net_demand))
        else:
            reserves = reserve_mw
    else:
        raise ValueError('Must set reserve_pct or reserve_mw')

    # N-1 criterion: must always have at least reserve equal to the largest generator's capacity
    min_reserve = np.max(env.max_output) * np.ones(len(net_demand)) if n_minus_one else 0 * np.ones(len(net_demand))
    #print(n_minus_one)
    reserves = np.maximum(reserves, min_reserve)


    # Reserve shouldn't push beyond max_demand (sum of max_output)
    max_reserves = env.max_demand - net_demand
    #print('max',max_reserves)
    #print('reserves', reserves)
    reserves = np.minimum(reserves, max_reserves)


    reserves = list(reserves)

    #print(reserves)

    # max_reserves = np.ones(net_demand.size)*env.max_demand - np.array(net_demand)
    # reserves = np.clip()
    # reserves = list(np.min(np.array([reserves, max_reserves]), axis=0))

    dispatch_freq = env.dispatch_freq_mins/60
    num_periods = len(net_demand)
    


    all_gens = {}
    for g in range(env.num_gen):
        GEN_NAME = 'GEN'+str(g)
        foo = {"must_run": 0,
               "power_output_minimum": float(gen_info.min_output.values[g]),
               "power_output_maximum": float(gen_info.max_output.values[g]),
               "ramp_up_limit": 10000.,
               "ramp_down_limit": 10000.,
               "ramp_startup_limit": 10000.,
               "ramp_shutdown_limit": 10000.,
               "time_up_minimum": int(gen_info.t_min_up.values[g]),
               "time_down_minimum": int(gen_info.t_min_down.values[g]),
               "power_output_t0": 0.0,
               "unit_on_t0": int(1 if gen_info.status.values[g] > 0 else 0),
               "time_up_t0": int(gen_info.status.values[g] if gen_info.status.values[g] > 0 else 0),
               "time_down_t0": int(abs(gen_info.status.values[g]) if gen_info.status.values[g] < 0 else 0),
               "startup": [{"lag": 1, "cost": float(gen_info.hot_cost.values[g])}],
               "piecewise_production": calculate_piecewise_production(gen_info, g, dispatch_freq),
               "name": GEN_NAME}
        all_gens[GEN_NAME] = foo
        
    all_json = {"time_periods":num_periods,
                "prof_name":prof_name,
                "gen_number": env.num_gen,
                "demand":list(net_demand),
                "reserves":reserves,
                "thermal_generators":all_gens,
                "renewable_generators": {}}
    
    return all_json


