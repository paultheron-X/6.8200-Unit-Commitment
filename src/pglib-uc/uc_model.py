#!/usr/bin/env python
import pyomo.environ
from pyomo.environ import *
import csv 
import json
import sys
import numpy as np
import pandas as pd 
from pyomo.opt import SolverFactory
import time
from rl4uc.environment import make_env
import os

SOLVER_NAME = 'gurobi'   #GUROBI is highly recommended, failing that mosek

def solve_milp(data, Tree = None, explicit_reserves = True, VOLL = 10000.0, wind_shed_cost = 200, tee = False, MIP_GAP = 0.0001, threads = 1, Warm_Start_dir = None, warm_start_path=None):
    '''
    Forms and solves UC optimisation.
    
    Parameters
    ----------       
    data : (milp_dict)
        created in the create_milp_dict.py to specify optimiser inputs
    Tree : Scenario Tree, optional
        A Scenario Tree Generated using the scen.Tree function It passes nodal probabilities and net demand realisations to the opt. The default is None.
    explicit_reserves: (bool)
        When True, reserve constraints at each node are active, otherwise reserve is chosen implicitly
    VOLL: (float)
        The value of lost load for loadshed. ATM it is not applied to windshedding which is considered free.
    tee: (bool)
        When True, solver will print ongoing solve status and MIP GAP info
    MIP_GAP: (float)
        Minimum admissable gap between lower and upper bounds of MIP solution. Default is 0.01% denoted by MIP_GAP = 0.0001
    Warm_Start_dir: (str)
        When specified, the program will look in this folder for warm starts of the same name as the current day/env. The only variables currently warm started are the online state binaries i.e. ug.
        These need to be .csv files with gen number as column heading, and 48 entries vertically for each timestep.
    '''
    
    
    thermal_gens = data['thermal_generators']
    renewable_gens = data['renewable_generators']

    time_periods = {t+1 : t for t in range(data['time_periods'])}
    gen_startup_categories = {g : list(range(0, len(gen['startup']))) for (g, gen) in thermal_gens.items()}
    gen_pwl_points = {g : list(range(0, len(gen['piecewise_production']))) for (g, gen) in thermal_gens.items()}

    N =  len(Tree.nodes[0]) #this is the number of nodes per timestep
    NodeIndex = range(N) #0 refers to the quantile with the largest net demand error
    


    print('building model')
    m = ConcreteModel()

    m.cg = Var(thermal_gens.keys(), time_periods.keys(), NodeIndex)
    m.pg = Var(thermal_gens.keys(), time_periods.keys(), NodeIndex, within=NonNegativeReals)  
    m.rg = Var(thermal_gens.keys(), time_periods.keys(), NodeIndex, within=NonNegativeReals) 
    m.P_ls = Var(time_periods.keys(), NodeIndex, within=NonNegativeReals) #Load shed (MW)
    m.P_wc = Var(time_periods.keys(), NodeIndex, within=NonNegativeReals) #Wind Shed (MW)
    
    c_LS = VOLL / 2 #loadshed cost (£/MW), need proper value soon, division by 2 converting from MWh to MW
    c_WS = wind_shed_cost / 2 # windshed cost (£ / MW), division by 2 converting from MWh to MW

    print()
    print("c_LS: ${} / MW".format(c_LS))
    print("c_WS: ${} / MW".format(c_WS))
    print()
    
    m.pw = Var(renewable_gens.keys(), time_periods.keys(), within=NonNegativeReals)
    m.ug = Var(thermal_gens.keys(), time_periods.keys(), within=Binary) 
    
    
    m.vg = Var(thermal_gens.keys(), time_periods.keys(), within=Binary) 
    m.wg = Var(thermal_gens.keys(), time_periods.keys(), within=Binary) 

    m.dg = Var(((g,s,t) for g in thermal_gens for s in gen_startup_categories[g] for t in time_periods), within=Binary) ##
    m.lg = Var(((g,l,t,n) for g in thermal_gens for l in gen_pwl_points[g] for t in time_periods for n in NodeIndex), within=UnitInterval) ##
    #m.lg = Var(((g,l,t) for g in thermal_gens for l in gen_pwl_points[g] for t in time_periods), within=UnitInterval) ##
    
    
    m.obj = Objective(expr=sum(
                            sum(
                                gen['piecewise_production'][0]['cost']*m.ug[g,t]
                                + sum( gen_startup['cost']*m.dg[g,s,t] for (s, gen_startup) in enumerate(gen['startup']))
                            for g, gen in thermal_gens.items() ) 
                            +
                            sum(
                                Tree.nodes[t-1][n].pi * ( sum( m.cg[g,t,n] for g, gen in thermal_gens.items()) + c_LS * (m.P_ls[t,n]) + c_WS * (m.P_wc[t,n]) )# + m.P_wc[t,n])                                  
                            for n in NodeIndex)
                            
                        for t in time_periods)
                    ) #(1)

    m.demand = Constraint(time_periods.keys(),NodeIndex)
    m.reserves = Constraint(time_periods.keys(),NodeIndex)
    for t,t_idx in time_periods.items():
        for n in NodeIndex:
            m.demand[t,n] = m.P_ls[t,n] - m.P_wc[t,n] + sum( m.pg[g,t,n]+gen['power_output_minimum']*m.ug[g,t] for (g, gen) in thermal_gens.items() ) == data['demand'][t_idx] # + Tree.nodes[t-1][n].ndfe #(2)
            
            if (explicit_reserves):
                m.reserves[t,n] = sum( m.rg[g,t,n] for g in thermal_gens ) >= data['reserves'][t_idx] #(3)
            else:
                m.reserves[t,n] = sum( m.rg[g,t,n] for g in thermal_gens ) == 0.0

    m.uptimet0 = Constraint(thermal_gens.keys())
    m.downtimet0 = Constraint(thermal_gens.keys())
    m.logicalt0 = Constraint(thermal_gens.keys())
    m.startupt0 = Constraint(thermal_gens.keys())

    m.rampupt0 = Constraint(thermal_gens.keys())
    m.rampdownt0 = Constraint(thermal_gens.keys())
    m.shutdownt0 = Constraint(thermal_gens.keys())

    for g, gen in thermal_gens.items():
        if gen['unit_on_t0'] == 1:
            if gen['time_up_minimum'] - gen['time_up_t0'] >= 1:
                m.uptimet0[g] = sum( (m.ug[g,t] - 1) for t in range(1, min(gen['time_up_minimum'] - gen['time_up_t0'], data['time_periods'])+1)) == 0 #(4)
        elif gen['unit_on_t0'] == 0:
            if gen['time_down_minimum'] - gen['time_down_t0'] >= 1:
                m.downtimet0[g] = sum( m.ug[g,t] for t in range(1, min(gen['time_down_minimum'] - gen['time_down_t0'], data['time_periods'])+1)) == 0 #(5)
        else:
            raise Exception('Invalid unit_on_t0 for generator {}, unit_on_t0={}'.format(g, gen['unit_on_t0']))

        m.logicalt0[g] = m.ug[g,1] - gen['unit_on_t0'] == m.vg[g,1] - m.wg[g,1] #(6)

        startup_expr = sum( 
                            sum( m.dg[g,s,t] 
                                    for t in range(
                                                    max(1,gen['startup'][s+1]['lag']-gen['time_down_t0']+1),
                                                    min(gen['startup'][s+1]['lag']-1,data['time_periods'])+1
                                                  )
                                ) 
                           for s,_ in enumerate(gen['startup'][:-1])) ## all but last
        if isinstance(startup_expr, int):
            pass
        else:
            m.startupt0[g] = startup_expr == 0 #(7)


        #m.rampupt0[g] = m.pg[g,1] + m.rg[g,1] - gen['unit_on_t0']*(gen['power_output_t0']-gen['power_output_minimum']) <= gen['ramp_up_limit'] #(8)

        #m.rampdownt0[g] = gen['unit_on_t0']*(gen['power_output_t0']-gen['power_output_minimum']) - m.pg[g,1] <= gen['ramp_down_limit'] #(9)


        shutdown_constr = gen['unit_on_t0']*(gen['power_output_t0']-gen['power_output_minimum']) <= gen['unit_on_t0']*(gen['power_output_maximum'] - gen['power_output_minimum']) - max((gen['power_output_maximum'] - gen['ramp_shutdown_limit']),0)*m.wg[g,1] #(10)

        if isinstance(shutdown_constr, bool):
            pass
        else:
            m.shutdownt0[g] = shutdown_constr

    m.mustrun = Constraint(thermal_gens.keys(), time_periods.keys())
    m.logical = Constraint(thermal_gens.keys(), time_periods.keys())
    m.uptime = Constraint(thermal_gens.keys(), time_periods.keys())
    m.downtime = Constraint(thermal_gens.keys(), time_periods.keys())
    m.startup_select = Constraint(thermal_gens.keys(), time_periods.keys())
    m.gen_limit1 = Constraint(thermal_gens.keys(), time_periods.keys(),NodeIndex)
    m.gen_limit2 = Constraint(thermal_gens.keys(), time_periods.keys(),NodeIndex)
    m.ramp_up = Constraint(thermal_gens.keys(), time_periods.keys())
    m.ramp_down = Constraint(thermal_gens.keys(), time_periods.keys())
    m.power_select = Constraint(thermal_gens.keys(), time_periods.keys(), NodeIndex)
    m.cost_select = Constraint(thermal_gens.keys(), time_periods.keys(), NodeIndex)
    m.on_select = Constraint(thermal_gens.keys(), time_periods.keys())

    for g, gen in thermal_gens.items():
        for t in time_periods:
            m.mustrun[g,t] = m.ug[g,t] >= gen['must_run'] #(11)

            if t > 1:
                m.logical[g,t] = m.ug[g,t] - m.ug[g,t-1] == m.vg[g,t] - m.wg[g,t] #(12)

            UT = min(gen['time_up_minimum'],data['time_periods'])
            if t >= UT:
                m.uptime[g,t] = sum(m.vg[g,t] for t in range(t-UT+1, t+1)) <= m.ug[g,t] #(13)
            DT = min(gen['time_down_minimum'],data['time_periods'])
            if t >= DT:
                m.downtime[g,t] = sum(m.wg[g,t] for t in range(t-DT+1, t+1)) <= 1-m.ug[g,t] #(14)
            m.startup_select[g,t] = m.vg[g,t] == sum(m.dg[g,s,t] for s,_ in enumerate(gen['startup'])) #(16)

            for n in NodeIndex:
                m.gen_limit1[g,t,n] = m.pg[g,t,n]+m.rg[g,t,n] <= (gen['power_output_maximum'] - gen['power_output_minimum'])*m.ug[g,t] - max((gen['power_output_maximum'] - gen['ramp_startup_limit']),0)*m.vg[g,t] #(17)
    
                if t < len(time_periods): 
                    m.gen_limit2[g,t,n] = m.pg[g,t,n]+m.rg[g,t,n] <= (gen['power_output_maximum'] - gen['power_output_minimum'])*m.ug[g,t] - max((gen['power_output_maximum'] - gen['ramp_shutdown_limit']),0)*m.wg[g,t+1] #(18)

            # if t > 1:
            #     m.ramp_up[g,t] = m.pg[g,t]+m.rg[g,t] - m.pg[g,t-1] <= gen['ramp_up_limit'] #(19)
            #     m.ramp_down[g,t] = m.pg[g,t-1] - m.pg[g,t] <= gen['ramp_down_limit'] #(20)

            piece_mw1 = gen['piecewise_production'][0]['mw']
            piece_cost1 = gen['piecewise_production'][0]['cost']
            for n in NodeIndex:
                m.power_select[g,t,n] = m.pg[g,t,n] == sum( (piece['mw'] - piece_mw1)*m.lg[g,l,t,n] for l,piece in enumerate(gen['piecewise_production'])) #(21)
                m.cost_select[g,t,n] = m.cg[g,t,n] == sum( (piece['cost'] - piece_cost1)*m.lg[g,l,t,n] for l,piece in enumerate(gen['piecewise_production'])) #(22)
                m.on_select[g,t] = m.ug[g,t] == sum(m.lg[g,l,t,n] for l,_ in enumerate(gen['piecewise_production'])) #(23)

    m.startup_allowed = Constraint(m.dg_index)
    for g, gen in thermal_gens.items():
        for s,_ in enumerate(gen['startup'][:-1]): ## all but last
            for t in time_periods:
                if t >= gen['startup'][s+1]['lag']:
                    m.startup_allowed[g,s,t] = m.dg[g,s,t] <= sum(m.wg[g,t-i] for i in range(gen['startup'][s]['lag'], gen['startup'][s+1]['lag'])) #(15)

    
    # for w, gen in renewable_gens.items():
    #     for t, t_idx in time_periods.items():
    #         m.pw[w,t].setlb(gen['power_output_minimum'][t_idx]) #(24)
    #         m.pw[w,t].setub(gen['power_output_maximum'][t_idx]) #(24)

    print("model setup complete")
    
    if(SOLVER_NAME == 'mosek'):
        solver = SolverFactory(SOLVER_NAME)
    elif(SOLVER_NAME == 'gurobi'):        
        solver = SolverFactory(SOLVER_NAME)
    elif(SOLVER_NAME == 'cplex'):
        solver = SolverFactory(SOLVER_NAME,executable="/Applications/CPLEX_Studio221/cplex/bin/x86-64_osx/cplex")
    else:
        solver = SolverFactory(SOLVER_NAME)
        
    # Setting solver options 
    solver.options["mipgap"] = MIP_GAP
    solver.options["threads"] = threads
        
    print("solving")
    start = time.time()
    
    
    m = m.create_instance()

    # Warm starting    
    warm_start = False
    if warm_start_path is not None: 
        warm_start = True
        ws_csv = warm_start_path
    
    elif(Warm_Start_dir != None):   
        warm_start=True
    #Read in the warm_start variables
        ws_csv = Warm_Start_dir + data['prof_name'] + '_solution_' + str(data['gen_number']) + 'gen.csv'
    
    if warm_start: 
        print(f"***Warm starting using {ws_csv}***")
        if (os.path.isfile(ws_csv)):
            warm_var = pd.read_csv(ws_csv).values
        else:
            raise Exception("No warm start file: " + ws_csv + 'exists. Please specify.')
            
    #Check they are of the correct dimensions
        if(warm_var.shape != (48,data['gen_number'])):
            raise Exception(ws_csv + ' Is not the correct dimensions of 48 x '+ str(data['gen_number']))

    #Initialise the decision variables
        for g, gen in thermal_gens.items():
            for t in time_periods:            
                m.ug[g,t] = warm_var[t-1, int(g[3:])]
    
    # Set time limit: 12 hours
    solver.options['timelimit'] = 43200

    results = solver.solve(m, warmstart = warm_start, tee = tee)
    end = time.time()
    time_taken = end - start
    print('Solved after: ', int(time_taken),'s')

    #Print outputs for bug checking
    # print('Total Load Shed:',sum(value(m.P_ls[:,:])))
    # print('Total Wind Shed:',sum(value(m.P_wc[:,:])))
    loadshed = np.zeros([48,N])
    windshed = np.zeros([48,N])
    for t in range(1,49):
        for n in NodeIndex:
            loadshed[t-1,n] = value(m.P_ls[t,n])
            windshed[t-1,n] = value(m.P_wc[t,n])
            
    # np.savetxt("loadshed.csv", loadshed, delimiter=",")
    # np.savetxt("windshed.csv", windshed, delimiter=",")
    #print('Total Load Shed Cost:',c_LS * sum( sum(value(m1.P_ls[t,n]) for t in range(1,49)) for n in NodeIndex   ) )
    #print('Total gen energy cost: ', sum(value(m.cg[:,:,:])))
    # print('Objective Value: ', value(m.obj))
    
    return m, time_taken

def solution_to_schedule(m, data):
    """
    Retrieve the binary schedule from a MILP solution object m, 
    with problem data (dict) in data.
    """
    num_gen = len(data['thermal_generators'])
    time_periods = data['time_periods']
    schedule = [[0]*time_periods for i in range(num_gen)]

    schedule = np.zeros((time_periods, num_gen))
    dispatch = np.zeros((time_periods, num_gen))

    for v in m.component_data_objects(Var):
        if 'ug' in str(v):
            split_v = str(v).split(',')
            gen = int("".join([s for s in split_v[0] if s.isdigit()]))
            time_period = int("".join([s for s in split_v[1] if s.isdigit()]))
            schedule[time_period-1, gen] = int(round(float(v.value)))

    return schedule

if __name__=="__main__":

    ## Grab instance file from first command line argument
    data_file = sys.argv[1]

    print('loading data')
    data = json.load(open(data_file, 'r'))

    # Solve the MILP
    start_time = time.time()
    m = solve_milp(data)
    time_taken = time.time()-start_time

    num_gen = len(data['thermal_generators'])
    time_periods = data['time_periods']
    schedule = [[0]*time_periods for i in range(num_gen)]

    schedule = np.zeros((time_periods, num_gen))
    dispatch = np.zeros((time_periods, num_gen))

    for v in m.component_data_objects(Var):
        if 'ug' in str(v):
            split_v = str(v).split(',')
            gen = int("".join([s for s in split_v[0] if s.isdigit()]))
            time_period = int("".join([s for s in split_v[1] if s.isdigit()]))
            schedule[int(time_period-1), int(gen)] = int(v.value)

    sched_cols = ['schedule_'+str(i) for i in range(num_gen)]
    schedule = pd.DataFrame(schedule, columns=sched_cols)
    print(schedule)

    save_fn = data_file.split('.json')[0] + '_solution.csv'
    schedule.to_csv(save_fn, index=False)

