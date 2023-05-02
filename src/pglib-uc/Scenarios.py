# -*- coding: utf-8 -*-
'''
Created by Cormac O'Malley. This contains the classes and functions to generate scenario trees and their intrinsic nodes.
Refer to 'Sturt (2012)' for details of efficient scenario tree that branches at the root node only.
'''
   
from rl4uc.environment import make_env
from ts4uc import helpers as ts4uc_helpers
from ts4uc.tree_search.scenarios import sample_errors
import numpy as np
import json
import matplotlib.pyplot as plt
from sys import exit
from pathlib import Path
import os


def sample_net_demand_forecast_error_scenarios(profile_df, env, num_scenarios, horizon=48, relative_path_to_scenarios='scenarios_npy'):
    """
    Return scenarios for net demand forecast errors. 
    
    Forecast errors are passed through the same sequence of min/max functions as in rl4uc 
    to prevent: 
    
    - demand or wind falling below 0 
    - net demand falling outside the interval (env.min_demand, env.max_demand)
    """
    # Sample errors from ARMA
    scenarios_path = Path(__file__).parent / relative_path_to_scenarios
    demand_errors = np.load(os.path.join(scenarios_path, f'g{env.num_gen}_demand.npy'))
    wind_errors = np.load(os.path.join(scenarios_path, f'g{env.num_gen}_wind.npy'))
    # demand_errors, wind_errors = sample_errors(env, num_scenarios, horizon)

    # Clip the demand and wind scenarios 
    demand_scenarios = np.maximum(0, profile_df.demand.values + demand_errors)
    wind_scenarios = np.maximum(0, profile_df.wind.values + wind_errors)
    
    # Clip the net demand scenarios to [env.min_demand, env.max_demand]
    net_demand_scenarios = np.clip(demand_scenarios - wind_scenarios, env.min_demand, env.max_demand)
    
    # Calculate the forecast errors
    net_demand_forecast_error_scenarios = net_demand_scenarios - (profile_df.demand.values - profile_df.wind.values)

    return net_demand_forecast_error_scenarios

class Node:
    def __init__(self, node_n, timestep, pi, ndfe):
        '''
        Object to describe a node in the scenario tree.
        
        Parameters
        ----------
        node_n : (int)
            Node number (starting at 1)
        timestep : (int)
            timestep of node (starting at 1 whcih corresponds to 00:00, then the next half hour time is at 2 (00:30))
            The commitment decisions are being made at 23:30 for the next day.
        pi : (float)
            Unconditional probability of reaching this node.
        ndfe : (float)
            Net demand forecast Error at this node (MW)
        Returns
        -------
        None.

        ''' 
        self.node_n = node_n
        self.timestep = timestep
        self.pi = pi
        self.ndfe = ndfe
 
     
class Tree:
    def __init__(self, opt_type, profile_df, env, quantiles = [0.5], empirical_number=5000):
        '''
        Class to set up scenario tree of nodes.
        
        Parameters
        ----------
        opt_type : (string)
            Type of scenario tree to be formed. Options: 'Stoch_Root_Node' ; 'Determ' .
        horizon : (int)
            Number of half hour time periods over which the tree is formed.
        wind : (array <float>)
            Output from ts4uc_helpers.get_scenarios(env), samples of wind realisations over the time horizon, wind forecast error! (MW)
        demand : (array <float>)
            Output from ts4uc_helpers.get_scenarios(env), samples of net demand realisations over the time horizon, demand forecast error! (MW)
        quantiles : (array <float>)
            array used to construct scenario tree
        episode_forecast : forecasted demand for each of the 48 timesteps!!
        
        Returns
        -------
        None.

        '''
        if(opt_type == 'Stoch_Root_Node' or opt_type == 'Determ'):
            self.opt_type = opt_type
        else:
            print('#### Error, Opt_Type for Scenario Tree not recognized. #####')
            exit()
        
        # Number of periods
        self.horizon = profile_df.shape[0]
        
        # Sample net demand forecast errors
        ndfe = 0
        
        # Transpose 
        ndfe = np.transpose(ndfe) 
        
        if(self.opt_type == 'Stoch_Root_Node' or self.opt_type == 'Determ'):
            
            self.Nnodes = len(quantiles) * self.horizon #number of nodes in tree 
            #a scenario tree that use predefined quantiles that branch at the root node only
        
        #Create array of quantiles    
            ndfe_quantiles = np.zeros([self.horizon,len(quantiles)])
            for t in range(self.horizon):
                ndfe_quantiles[t,:] = 0
            
            print(profile_df.demand.values - profile_df.wind.values + ndfe_quantiles[:,-1])
        
        #Find Branch Probabilities
            if self.opt_type == 'Stoch_Root_Node':
                qp = Find_Branch_Probability(quantiles)
            elif self.opt_type == 'Determ':
                qp = [1]
                if (len(quantiles) != 1):
                    print('#### Error, For deterministic Scenario Tree only one quantile can be specified. #####')
                    exit()
                
        
        
        #Create nodes and Assign Weighting + Net Demand
            self.nodes = [] 
            n = 0
            
            for t in range(self.horizon):
                self.nodes.append([])
                for q in range(len(quantiles)-1,-1,-1):  #the nodes with high net demand are labelled first 
                    self.nodes[t].append(Node(n,t,qp[q],ndfe_quantiles[t,q]))
                    
                    n = n + 1                   
                

                

        
def Find_Branch_Probability(quantiles):
        '''
        Description
        ----------    
        Uses the trapezium rule to work out cost nodal weighting. As listed in Sturt (2012) Section IV B.   
        
        Parameters
        ----------
        Quantile Probability
        
        Returns
        -------
        qp

        '''
        
        num = len(quantiles)
        if num < 4:
            print('\\\ Error, must be at least 4 quantiles to work out probabilities for Stochastic Scenario tree.///')
            exit()
        
        # using sort() to 
        # check sorted list 
        test_list1 = quantiles[:]
        test_list1.sort()
        if (test_list1 != quantiles):
            print('/// Error, the list of quantiles must be sorted. ///')            

        
        qp = np.zeros([num])
        for x in range(num):
            if x == 0:
                qp[x] = 0.5 * (quantiles[x+1]**2)/(quantiles[x+1]-quantiles[x])
                
            elif x == 1:
                qp[x] = 0.5 * (quantiles[2] - quantiles[0] - quantiles[0]**2/(quantiles[1] - quantiles[0]))
            
            elif x == num - 2:
                qp[x] = 0.5 * (quantiles[num-1]-quantiles[num-3] - (1 - quantiles[num-1])**2/(quantiles[num-1] - quantiles[num-2]))
            
            elif x == num - 1:
                qp[x] = 0.5 * ((1-quantiles[num-2])**2/(quantiles[num-1] - quantiles[num-2]))
            
            else:
                qp[x] = 0.5 * (quantiles[x+1] - quantiles[x-1])
                
            
        return qp
                
        

        
        
        
