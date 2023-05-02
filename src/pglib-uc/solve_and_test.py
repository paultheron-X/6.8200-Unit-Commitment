#!/usr/bin/env python

import numpy as np
import pandas as pd
import argparse
import json
import os 
import time

from rl4uc.environment import make_env
from ts4uc import helpers as ts4uc_helpers
from ts4uc.tree_search.scenarios import get_net_demand_scenarios
#from helpers import get_scenarios
from create_milp_dict import create_problem_dict
from uc_model import solve_milp, solution_to_schedule
import Scenarios as scen

SEED=999


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='MILP solutions to UC problem (and testing with stochastic environment)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--env_params_fn', type=str, required=True,
                        help='Filename for environment parameters.')
    parser.add_argument('--test_data_dir', type=str, required=False, default=None,
                        help='Directory containing UC problems to solve (in .csv files)')
    parser.add_argument('--test_data', type=str, required=False, default=None,
                        help='Path to a single UC problem to solve (.csv)')
    parser.add_argument('--num_samples', type=int, required=False, default=1000,
                        help='Number of demand realisation to compute costs with.')
    parser.add_argument('--test_sample_seed', type=int, required=False, default=995,
                        help='Seed used to generate scenarios for evaluating solutions')                    
    parser.add_argument('--quantiles', nargs='+', help='<Required> Set flag', required=False)
    parser.add_argument('--quantiles_str', type=str, required=False,
                        help='Input the quantiles as a string separated by commans: e.g. "0.5" or "0.1,0.5,0.9"')                       
    parser.add_argument('--reserve_pct', type=int, required=False, default=None,
                        help='Reserve margin as percent of forecast net demand')
    parser.add_argument('--reserve_sigma', type=float, required=False, default=None,
                        help='Number of sigma to consider for reserve constraint')
    parser.add_argument('--perfect_forecast', action='store_true',
                        help='Boolean for perfect forecast')
    parser.add_argument('--n_minus_one', action='store_true',
                        help='Boolean for including n-1 criterion')
    parser.add_argument('--non_uniform_reserve_req', action='store_false',
                        help='When true (default), deterministic opts will have increasing reserve throughout the day to reflect the increasing uncertainty. Otherwise the av reserve requirement is uniformly applied across the day.')
    
    #Solver Options
    parser.add_argument('--tee', action='store_true',
                        help='Boolean for printing live solver data.')
    parser.add_argument('--MIP_GAP', type=float, required=False, default=0.0001,
                        help='Mixed Integer Programming gap for the optimisation.')
    parser.add_argument('--Warm_Start_dir', type=str, required=False, default=None,
                        help='Directory containing commitment results for warm starting optimisation.')
    parser.add_argument('--warm_start_path', type=str, required=False, default=None,
                        help='Path to commitment results for specific problem for warm starting optimisation.')
    
    
    
    args = parser.parse_args()

    # If using perfect forecast, set reserve margin to 0
    if args.perfect_forecast: args.reserve_pct=0

    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Update params
    params = vars(args)

    res = 'perfect' if args.perfect_forecast else args.reserve_pct
    params.update({'milp': 'true',
                   'reserve': res})
    # Read the parameters
    env_params = json.load(open(args.env_params_fn))

    # Save params file to save_dir 
    with open(os.path.join(args.save_dir, 'params.json'), 'w') as fp:
        fp.write(json.dumps(params, sort_keys=True, indent=4))

    # Save env params to save_dir
    with open(os.path.join(args.save_dir, 'env_params.json'), 'w') as fp:
        fp.write(json.dumps(env_params, sort_keys=True, indent=4))
        
    
    # If using sigma for reserve constraint, determine reserve constraint here:
    if(not args.non_uniform_reserve_req):  #old version which assumes a constant reserve requirement across timesteps
        if args.reserve_sigma is not None:
            np.random.seed(SEED)
            env = make_env(mode='train', **env_params)
            scenarios = ts4uc_helpers.get_scenarios(env, 1000)
            sigma = np.std(scenarios)
            reserve_mw = args.reserve_sigma * sigma
        else:
            reserve_mw = None
    else:
        if args.reserve_sigma is not None:   #new version which increases reserve requirement throughout day as uncertainty increases
            np.random.seed(SEED)
            env = make_env(mode='train', **env_params)
            
            empirical_number = 1000
            demand, wind = ts4uc_helpers.get_scenarios(env, empirical_number) #forecast errors
            demand = np.transpose(np.asarray(demand))
            wind = np.transpose(np.asarray(wind))
 
            ndfe = demand-wind #net demand forecast error
            ndfe_std = np.zeros([48])
            for i in range(48):
                ndfe_std[i] = np.std(ndfe[i,:])
            
            reserve_mw = args.reserve_sigma * ndfe_std
        else:
            reserve_mw = None

    # get list of test profiles
    if args.test_data_dir is not None:
        test_profiles = [os.path.join(args.test_data_dir, f) for f in os.listdir(args.test_data_dir) if '.csv' in f]
        test_profiles.sort()
    elif args.test_data is not None:
        test_profiles = [args.test_data]
    else:
        raise ValueError("Must pass either test_data_dir or test_data")
        
    all_test_costs = {}
    all_times = []

    for f in test_profiles:
        
        prof_name = f.split('/')[-1].split('.')[0]
        print(f"*** solving {prof_name} ***")

    # Formulate the problem dictionary (with wind)
        profile_df = pd.read_csv(f)
        #adjust wind /demand profiles to correct size (ARMA properties done within enviroment)
        profile_df['demand'] = profile_df.demand.values * (env.num_gen / 10)
        profile_df['wind'] = profile_df.wind.values * (env.num_gen / 10)
        demand = profile_df.demand.values 
        wind = profile_df.wind.values
        
        #Create the Scenario Tree
        np.random.seed(SEED)
        env = make_env(mode='train', **env_params)            
        print(f"Wind ARMA sigma = {env.arma_wind.sigma}")
        empirical_number = 5000
        
        if args.quantiles_str is not None:
            quantiles = [float(q) for q in args.quantiles_str.split(',')]
        elif args.quantiles is not None:
            quantiles = [float(q) for q in args.quantiles]
        else:
            raise ValueError("Must pass either quantiles or quantiles_str")
        
#         args.quantiles = [float(q) for q in args.quantiles_str.split(',')]

        if(len(quantiles)==1):
            print("Using deterministic formulation")
            tree_type = 'Determ'
        else:
            print("Using stochastic formulation")
            tree_type = 'Stoch_Root_Node'
            
        # Get scenarios for net demand forecast errors
#         net_demand = get_net_demand_scenarios(profile_df, env, empirical_number)
#         net_demand_forecast_errors = net_demand - (profile_df.demand.values - profile_df.wind.values)
            
        # Create the scenario tree
        Tree = scen.Tree(tree_type, profile_df, env, quantiles, empirical_number)

        problem_dict = create_problem_dict(demand, wind, env_params=env_params, reserve_pct=args.reserve_pct, reserve_mw=reserve_mw, n_minus_one=args.n_minus_one, prof_name = prof_name)
        

        fn = prof_name + '.json'
        with open(os.path.join(args.save_dir, fn), 'w') as fp:
            json.dump(problem_dict, fp)

    # Solve the MILP        
        solution, time_taken = solve_milp(problem_dict,Tree, explicit_reserves=(tree_type == 'Determ'), 
                                          VOLL=env.voll, wind_shed_cost = env.wind_shed_cost_per_mwh, tee=args.tee, MIP_GAP=args.MIP_GAP, 
                                          Warm_Start_dir=args.Warm_Start_dir, warm_start_path=args.warm_start_path)
        all_times.append(time_taken)

        # convert solution to binary schedule
        schedule = solution_to_schedule(solution, problem_dict)
        
        # Save the binary schedule as a .csv file
        columns = ['schedule_' + str(i) for i in range(env_params.get('num_gen'))]
        df = pd.DataFrame(schedule, columns=columns)
        df.to_csv(os.path.join(args.save_dir, '{}_solution.csv'.format(prof_name)), index=False)

        # initialise environment for sample operating costs
        env = make_env(mode='test', profiles_df=profile_df, **env_params)

        if args.perfect_forecast:
            args.num_samples = 1
        results = ts4uc_helpers.test_schedule(env, schedule, args.test_sample_seed, args.num_samples, args.perfect_forecast)
        ts4uc_helpers.save_results(prof_name=prof_name, 
                                 save_dir=args.save_dir, 
                                 env=env, 
                                 schedule=schedule,
                                 test_costs=results['total_cost'].values, 
                                 test_kgco2=results['kgco2'].values,
                                 lost_loads=results['lost_load_events'].values,
                                 results_df=results,
                                 time_taken=time_taken)


        print("Done")
        print()
        print("Mean costs: ${:.2f}".format(np.mean(results['total_cost'])))
        print("Mean CO2: {:.2f}kg".format(np.mean(results['kgco2'])))
        print("Lost load prob: {:.3f}%".format(100*np.sum(results['lost_load_events'])/(args.num_samples * env.episode_length)))
        print("Time taken: {:.2f}s".format(time_taken))
        print() 

