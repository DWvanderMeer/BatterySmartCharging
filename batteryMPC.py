'''
Author: Dennis van der Meer
E-mail: denniswillemvandermeer[at]gmail.com

This script contains the smart charging algorithm based on model predictive
control. 
'''

import cvxpy as cp
import numpy as np

def mpc(NL,SoC):
    '''
    - NL is the net load forecast and should be numpy array
    - SoC is the initial battery state of charge, i.e., the initial state
    - The function outputs charge and discharge power as well as power to
      and from the grid.
    '''
    T = len(NL)
    n = 1 # States
    m = 4 # Controls
    d = 1 # Disturbances (net load in this case)
    # The following gains are only used in the dynamic equation
    A = 1 # State gain (scalar because n=1)
    B = np.array([[0.25, -0.25, 0, 0]]) # Control gain with delta t
    C = 0 # Disturbance gain (scalar because d=1)
    x_0 = 7.2*SoC # Initial battery capacity

    # Set up the optimization problem:
    x = cp.Variable((n, T+1)) # State
    u = cp.Variable((m, T)) # Controls
    r = cp.Variable((2, T), boolean=True) # Binary variable
    d = cp.Parameter(shape=(NL.shape), value=NL) # Disturbance
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
    for t in range(T):
        cost += cp.sum(u[2:4,t])
        constr += [x[:,t+1] == A*x[:,t] + B@u[:,t] + C*d[t],
                   d[t] == u[1,t] - u[0,t] + u[2,t] - u[3,t],
                   u[:,t] >= 0.0,
                   u[0,t] <= 5*r[0,t],
                   u[1,t] <= 5*r[1,t],
                   r[0,t] + r[1,t] <= 1,
                   u[3,t] <= 5*r[0,t],
                   x[:,t] >= 1.44, x[:,t] <= 7.0]
    constr += [x[:,0] == x_0]
    # Problem definition:
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve(solver=cp.GLPK_MI)
    # Return results:
    return(np.transpose(np.around(u.value[:,0], decimals=2)))
