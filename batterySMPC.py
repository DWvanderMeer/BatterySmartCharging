'''
Author: Dennis van der Meer
E-mail: denniswillemvandermeer[at]gmail.com

This script contains the smart charging algorithm based on stochastic model
predictive control. The function features an argument that lets you run the
SMPC algorithm in the traditional way (average the outcomes of the S
optimization problems) or the proposed method where a constraint is added
that forces the control to be equal across all scenarios during the first
prediction step.
'''

import cvxpy as cp
import numpy as np
import cplex as cx

def smpc(NL,SoC,eq,lambdas):
    '''
    Arguments:
    - NL is the (net load) multivariate forecast and should be numpy array (T,S),
      where T is the forecast horizon (96 in the article) and S the number of
      scenarios of the forecast.
    - SoC (p.u.) is the initial battery state of charge, i.e., the initial
      state x_0 (kWh).
    - eq is True or False. True means that the (dis-)charging power is equal
      across the scenarios in the first time step (which is our contribution).
    - lambdas: NumPy array (T,2) containing "buy" and "sell" prices in SEK/kWh.
    Returns:
    - The function outputs charge and discharge power as well as power to
      and from the grid.
    '''
    T = NL.shape[0] # Prediction horizon
    # Create a 3D k-by-m-by-n variable.
    u = {} # Control inputs
    x = {} # States
    r = {} # Binary variables
    S = NL.shape[1] # Number of scenarios
    m = 4 # Number of control inputs
    n = 1 # Number of states
    A = 1 # State gain (scalar because n=1)
    B = np.array([[0.25*0.96, -0.25/0.96, 0, 0]]) # Control gain with delta t and efficiency
    C = 0 # Disturbance gain (scalar because d=1)
    x_0 = 7.2*SoC # Initial battery capacity in kWh

    # The dictionary is necessary to create 3D variables. Now, the variables are
    # organized as x[s][0,t] for battery energy and u[s][0,t] for charging power.
    for i in range(S):
        u[i] = cp.Variable((m,T))
        x[i] = cp.Variable((n,T+1))
        #r[i] = cp.Variable((4, T), boolean=True) # Binary variable
    beta = cp.Variable((2,1)) # Equalize (dis-)charging power at first time step
    d = cp.Parameter(shape=(NL.shape), value=NL) # Disturbance (net load forecast)
    l = cp.Parameter(shape=(lambdas.shape), value=lambdas) # el prices
    '''
    Herein, control vector u is the following:
    u[0,t] = Pch
    u[1,t] = Pdis
    u[2,t] = PfrGrid
    u[3,t] = PtoGrid
    '''
    # Add constraints and objective function "cost"
    cost = 0
    constr = []
    for s in range(S):
        for t in range(T):
            cost += cp.sum(l[t,0]*u[s][2,t] - l[t,1]*u[s][3,t]) # objective function
            constr += [x[s][:,t+1] == A*x[s][:,t] + B@u[s][:,t], # system dynamics
                     d[t,s] == u[s][1,t] - u[s][0,t] + u[s][2,t] - u[s][3,t], # power balance
                     u[s][:,t] >= 0.0,
                     u[s][0,t] <= 5,
                     u[s][1,t] <= 5,
                     u[s][2,t] <= 5,
                     u[s][3,t] <= 5,
                     x[s][:,t+1] >= 1.44, x[s][:,t+1] <= 7.0]
            constr += [x[s][:,0] == x_0]
        if eq == True:
            constr += [u[s][0:2,0] == beta[:,0]] # Equalize (dis-)charging power at first time step
    problem = cp.Problem(cp.Minimize(cost), constr)
    #problem.solve(solver=cp.GLPK_MI)
    problem.solve(solver=cp.CPLEX) # Free for academics, otherwise select GLPK_MI
    ###### CHECK WHETHER FEASIBLE ######
    if problem.status in ["infeasible", "unbounded"]:
        print("Optimal value: %s" % problem.value)
        for variable in problem.variables():
            print("Variable %s: value %s" % (variable.name(), variable.value))
        for constraint in problem.constraints:
            if constraint.violation().any():
                print("Constraint %s: violation %s" % (constraint.name(), constraint.violation()))
    ###### CHECK WHETHER FEASIBLE ######
    # Return results:
    if eq == True:
        # Doesn't matter which scenario I choose, they should be the same:
        return(np.transpose(np.around(u[0].value[:,0], decimals=3)))
    else:
        # Calculate the average u across all scenarios for the first time step:
        avg = np.around(np.mean([u[k].value[:,0] for k in range(S)], axis=0), decimals=3)
        # Added the following on 2020-07-06 because Pch and Pdis sometimes both
        # have values in this case study, probably due to the averaging in the
        # previous step. Therefore the following: (basically, whichever one is higher "wins")
        if avg[0] and avg[1] > 0:
            if avg[0] > avg[1]:
                avg[1] = 0
            elif avg[0] <= avg[1]:
                avg[0] = 0
        return(avg)
