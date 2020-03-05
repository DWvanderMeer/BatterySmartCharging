'''
Author: Dennis van der Meer
E-mail: denniswillemvandermeer[at]gmail.com

This script contains all the functions that are necessary to run execute.py
Not all data is publicly available but the scripts should run regardless of
the source of the data.
'''

import numpy as np
import pandas as pd
import os
from copulas.multivariate import GaussianMultivariate
import cvxpy as cp
import cplex as cx
import time
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn import ensemble
import joblib as joblib
import properscoring as ps
from tqdm import tqdm

#dir0 = r"C:\Users\denva787\Documents\dennis\RISE" # Windows
#os.chdir(r"C:\Users\denva787\Documents\dennis\RISE") # Windows
dir0 = "/Users/Dennis/Desktop/Drive/PhD-Thesis/Projects/RISE/" # macOS
os.chdir('/Users/Dennis/Desktop/Drive/PhD-Thesis/Projects/RISE') # macOS

################################################################################
# Function to create a DataFrame of lagged time observations based
# on an input vector.
# Adapted from: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
################################################################################
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        #names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        names += ["{}{}".format("yt",-i) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += ["{}{}".format("yt",-i) for j in range(n_vars)]
            #names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

################################################################################
# Function for the persistence ensemble forecast
################################################################################

def peen_forecast(netLoad, horizon, quantileLevels, i):
    '''
    Arguments:
    - netLoad: a pandas time series.
    - horizon: the forecast horizon (scalar).
    - quantileLevels: a vector with the nominal probabilities expressed
      as quantiles.
    - i: current time step.
    Returns:
    - Numpy array (horizon x length(quantileLevels))
    '''
    timeLags = series_to_supervised(netLoad.to_numpy().tolist(), len(quantileLevels), 0, False)
    timeLags = timeLags.set_index(netLoad.index)
    agg = pd.merge(netLoad.rename('y'), timeLags, left_index=True, right_index=True, sort=True)
    agg.dropna(axis=0,inplace=True)
    # Forecast:
    fc = agg.iloc[i-horizon:i,1:]
    fc_sort = fc.values
    fc_sort.sort(axis=1)
    return(fc_sort)

################################################################################
# Function for the copula scenario generator
################################################################################

def copula(netLoad, horizon, num_samples, i):
    '''
    Arguments:
    - netLoad: a pandas Series.
    - horizon: the forecast horizon (scalar).
    - num_samples: number of samples to draw from copula.
    - i: current time step (at the moment not used).
    Returns:
    - Numpy array (num_samples x horizon)
    '''
    timeLags = series_to_supervised(netLoad.to_numpy().tolist(), horizon, 0, False)
    timeLags = timeLags.set_index(netLoad.index)
    timeLags = timeLags.iloc[:, ::-1]
    timeLags.dropna(axis=0,inplace=True)
    # Define and fit copula:
    copula = GaussianMultivariate(random_seed=i)
    copula.fit(timeLags)
    # Sample
    samples = copula.sample(num_samples)
    return(samples.values)

################################################################################
# Function that creates and solves the optimization problem
################################################################################

def smpc_run(NL,SoC,eq,lambdas):
    '''
    Arguments:
    - NL is the net load forecast and should be numpy array (T,S)
    - SoC is the initial battery state of charge, i.e., the initial state x_0
    - eq is True or False. True means that the (dis-)charging power is equal
      across the scenarios in the first time step.
    - lambdas: NumPy array (96,2) containing "buy" and "sell" prices in SEK/kWh.
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
    x_0 = 7.2*SoC # Initial battery capacity

    # The dictionary is necessary to create 3D variables. Now, the variables are
    # organized as x[s][0,t] for battery energy and u[s][0,t] for charging power.
    for i in range(S):
        u[i] = cp.Variable((m,T))
        x[i] = cp.Variable((n,T+1))
        r[i] = cp.Variable((2, T), boolean=True) # Binary variable
    beta = cp.Variable((2,1)) # Equalize (dis-)charging power at first time step
    d = cp.Parameter(shape=(NL.shape), value=NL) # Disturbance
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
            #cost += cp.sum(u[s][2:4,t])
            cost += cp.sum(l[t,0]*u[s][2,t] - l[t,1]*u[s][3,t])
            constr += [x[s][:,t+1] == A*x[s][:,t] + B@u[s][:,t], # + C*d[t]
                     d[t,s] == u[s][1,t] - u[s][0,t] + u[s][2,t] - u[s][3,t],
                     u[s][:,t] >= 0.0,
                     u[s][0,t] <= 5*r[s][0,t],
                     u[s][1,t] <= 5*r[s][1,t],
                     r[s][0,t] + r[s][1,t] <= 1,
                     u[s][3,t] <= 5*r[s][0,t],
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
        return(avg)

################################################################################
# Function to run the entire problem so that the two case studies (mean
# of charging power and equalizing across scenarios) can be run in parallel.
################################################################################

def run(days,horizon,SoC,NL,eq,num_samples,quantileLevels,inpEndo,inpExo,tar,lambdas,perfectFC):
    '''
    Arguments:
    - days is the number of days to run the mpc over (scalar)
    - horizon is the control horizon (scalar)
    - E_o is the initial SoC
    - NL is the net load time series and should be pandas Series (T,1) (T=days*resolution)
    - eq is True or False. True means that the (dis-)charging power is equal
      across the scenarios in the first time step.
    - num_samples: number of scenarios to generate from the copula
    - quantileLevels: a vector with the nominal probabilities expressed
      as quantiles.
    - inpEndo: pandas DataFrame containing endogeneous inputs.
    - inpExo: pandas DataFrame containing exogenous inputs.
    - tar: pandas DataFrame containing the targets.
    - lambdas: NumPy array (2880,2) containing the electricity "buy" and "sell" prices in SEK/kWh.
    - perfectFC: NumPy array (T,1) containing the net load time series representing perfect fc.
    Result:
    - Two text files containing the results for the two case studies, one
      where the charging power in the first time step is equal across
      scenarios (DF_True.txt) and one where the charging power in the first time
      step are equalized (DF_False.txt).
    '''
    data = [] # Store all results
    E_opt = [] # Store the optimal energy results
    # Load the forecasts in order to speed up the computation. Otherwise, uncomment
    # the code inside the for-loop to produce forecasts on the fly.
    fcs = [np.loadtxt("{}_{}.{}".format("Forecasts/gbrt",h,"txt")) for h in range(1,97,1)]

    for i in tqdm(np.arange(0,days*horizon)):
        E_opt.append(SoC)
        #IF FORECASTS DO
        # Prepare forecast:
        #fc = peen_forecast(NL, horizon, quantileLevels, i) # PeEn
        #fc = qr_forecast(horizon,inpEndo,inpExo,tar,quantileLevels,i) # qr
        #fc = gbrt_forecast(horizon,inpEndo,inpExo,tar,quantileLevels,i) # gbrt
        fc = np.vstack([fcs[h][i,:] for h in range(horizon)]) # Read fc instead of producing it
        # Prepare copula samples:
        samples = copula(NL,horizon,num_samples,0)
        # Generate scenarios from the above two:
        scenarios = np.zeros((num_samples,horizon))
        for h in range(horizon):
            scenarios[:,h] = np.interp(samples[:,h], quantileLevels, fc[h,:], np.amin(fc[h,:]), np.amax(fc[h,:]))
        scenarios = np.transpose(scenarios)
        scenarios[0,:] = perfectFC[i] # At the zero-th prediction step, net load is observed so insert this.
        #ELSE DO PERFECT FORECAST
        #scenarios = perfectFC[i:i+horizon] # perfect forecast
        start_time = time.time()
        res = smpc_run(scenarios, SoC, eq, lambdas[i:i+horizon,:])
        end_time = time.time() - start_time
        SoC += 0.25*(0.96*res[0]-res[1]/0.96)/7.2 # Update SoC  with latest result.
        res = np.append(res,lambdas[i,:])
        res = np.append(res,end_time)
        res = np.append(res,SoC)
        res = np.append(res,NL[NL.index.month == 4][i])
        res = np.reshape(res,(1,9))
        res_df = pd.DataFrame(res, columns=['Pch','Pdis','PfrGrid','PtoGrid','buyPrice','sellPrice','runTime','Energy','netLoad'])
        res_df.index = NL[NL.index.month == 4].index[[i]]
        res_df.to_csv(dir0 + "{}_{}.{}".format("\Results\DF", eq, "txt"), index=True, header=False, sep='\t', mode="a")

################################################################################
# Function to train 1..K qr forecast models
################################################################################

def qrTraining(horizon,inpEndo,inpExo,tar):
    '''
    Arguments:
    - horizon: The forecast horizon for which a model should be learned.
    - inpEndo: A pandas DataFrame containing the endogenous inputs.
    - inpExo: A pandas DataFrame containing the endogenous inputs.
    - tar: A pandas DataFrame containting the targets.
    Returns:
    - Stored .pickle files containing the QR parameters, one for each horizon
      and nominal probability.
    '''
    taus = np.arange(0.1,0.91,0.1)
    # Training months
    tr_m = [2,3,5,6] # Removed months 1, 7 and 8 to speed it up but also because they're less relevant.
    # Test month April
    te_m = 4

    cols = ["{}_{}".format("t",horizon)]

    train = inpEndo[inpEndo.index.month.isin(tr_m)]
    train = train.join(inpExo[inpExo.index.month.isin(tr_m)], how="inner")
    train = train.join(tar[tar.index.month.isin(tr_m)], how="inner")

    test = inpEndo[inpEndo.index.month == te_m]
    test = test.join(inpExo[inpEndo.index.month == te_m], how="inner")
    test = test.join(tar[tar.index.month == te_m], how="inner")

    feature_cols = inpEndo.filter(regex='y').columns.tolist()
    feature_cols_endo = inpEndo.filter(regex='y').columns.tolist()
    feature_cols.extend(["Temperature_{}".format(horizon),"TotalCloudCover_{}".format(horizon)]) # ,"WindUMS_{}".format(horizon),"WindVMS_{}".format(horizon)

    train = train[cols + feature_cols].dropna(how="any")
    test  = test[cols + feature_cols].dropna(how="any")

    train_X = train[feature_cols].values
    test_X  = test[feature_cols].values

    scaler = preprocessing.StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)

    train_y = train[cols].values
    test_y = test[cols].values

    quantreg = sm.QuantReg(train_y, train_X)
    tau = 1
    for q in taus:
        res = quantreg.fit(q=q, max_iter=10000)
        res.save("{}_{}_{}_{}.{}".format("ForecastModels\qr",horizon,"tau",tau,"pickle"))
        tau += 1

################################################################################
# Function to train and predict 1..K qr forecast models
################################################################################

def qr_forecast(horizon,inpEndo,inpExo,tar,quantileLevels,i):
    '''
    This function should read .pickle files and predict for all
    quantileLevels for "horizon" at the current time step i.

    Arguments:
    - horizon: the forecast horizon.
    - inpEndo: A pandas DataFrame containing the endogenous inputs.
    - inpExo: A pandas DataFrame containing the endogenous inputs.
    - tar: A pandas DataFrame containting the targets.
    - quantileLevels: a vector with the nominal probabilities expressed
      as quantiles.
    - i: current time step.
    Returns:
    - Numpy array (horizon x length(quantileLevels))
    '''
    # Training months
    tr_m = [2,3,5,6] # Removed months 1, 7 and 8 to speed it up but also because they're less relevant.
    # Test month April
    te_m = 4

    preds = []
    for h in range(1,horizon+1,1):
        train = inpEndo[inpEndo.index.month.isin(tr_m)]
        train = train.join(inpExo[inpExo.index.month.isin(tr_m)], how="inner")
        train = train.join(tar[tar.index.month.isin(tr_m)], how="inner")

        test = inpEndo[inpEndo.index.month == te_m]
        test = test.join(inpExo[inpEndo.index.month == te_m], how="inner")
        test = test.join(tar[tar.index.month == te_m], how="inner")

        cols = ["{}_{}".format("t",h)]
        feature_cols = inpEndo.filter(regex='y').columns.tolist()
        feature_cols_endo = inpEndo.filter(regex='y').columns.tolist()
        feature_cols.extend(["Temperature_{}".format(h),"TotalCloudCover_{}".format(h)]) # ,"WindUMS_{}".format(h),"WindVMS_{}".format(h)

        train = train[cols + feature_cols].dropna(how="any")
        test  = test[cols + feature_cols].dropna(how="any")

        train_X = train[feature_cols].values
        test_X  = test[feature_cols].values

        scaler = preprocessing.StandardScaler().fit(train_X)
        test_X = scaler.transform(test_X)

        test_y = test[cols].values

        tau = 1
        test_pred = []
        for q in quantileLevels:
            model = sm.load("{}_{}_{}_{}.{}".format("ForecastModels\qr",h,"tau",tau,"pickle"))
            test_pred.append(model.predict(test_X[i,:]))
            tau+=1
        tmp = np.vstack(test_pred).T # List to NumPy array
        preds.append(tmp)
    test_pred = np.vstack(preds)
    test_pred.sort(axis=1)
    print(test_pred,test_y)
    return(test_pred)

################################################################################
# Function to train 1..K gbrt forecast models
################################################################################

def gbrtTraining(horizon,inpEndo,inpExo,tar,params):
    '''
    Arguments:
    - horizon: The forecast horizon for which a model should be learned.
    - inpEndo: A pandas DataFrame containing the endogenous inputs.
    - inpExo: A pandas DataFrame containing the endogenous inputs.
    - tar: A pandas DataFrame containing the targets.
    - params: pandas Series containing the hyperparameters.
    Returns:
    - Stored .sav files containing the GBRT parameters, one for each horizon
      and nominal probability. Approximately 1.25 GB of storage required.
    '''
    taus = np.arange(0.1,0.91,0.1)
    # Training months
    tr_m = [2,3,5,6] # Removed months 1, 7 and 8 to speed it up but also because they're less relevant.
    # Test month April
    te_m = 4

    cols = ["{}_{}".format("t",horizon)]

    train = inpEndo[inpEndo.index.month.isin(tr_m)]
    train = train.join(inpExo[inpExo.index.month.isin(tr_m)], how="inner")
    train = train.join(tar[tar.index.month.isin(tr_m)], how="inner")

    test = inpEndo[inpEndo.index.month == te_m]
    test = test.join(inpExo[inpEndo.index.month == te_m], how="inner")
    test = test.join(tar[tar.index.month == te_m], how="inner")

    feature_cols = inpEndo.filter(regex='y').columns.tolist()
    feature_cols_endo = inpEndo.filter(regex='y').columns.tolist()
    feature_cols.extend(["Temperature_{}".format(horizon),"TotalCloudCover_{}".format(horizon)]) # ,"WindUMS_{}".format(horizon),"WindVMS_{}".format(horizon)

    train = train[cols + feature_cols].dropna(how="any")
    test  = test[cols + feature_cols].dropna(how="any")

    train_X = train[feature_cols].values
    test_X  = test[feature_cols].values

    scaler = preprocessing.StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)

    train_y = train[cols].values
    test_y = test[cols].values

    train_y = np.ravel(train_y)
    tau = 1
    learning_rate, min_samples_split, min_samples_leaf, max_depth, subsample, n_estimators = params
    n_estimators = int(n_estimators)
    max_depth = int(max_depth)
    min_samples_leaf = int(min_samples_leaf)
    for q in taus:
        gbrt = ensemble.GradientBoostingRegressor(loss='quantile',alpha=q,
                                                  n_estimators=n_estimators,
                                                  max_depth=max_depth,
                                                  learning_rate=learning_rate,
                                                  min_samples_leaf=min_samples_leaf,
                                                  min_samples_split=min_samples_split,
                                                  subsample=subsample)
        gbrt.fit(train_X,train_y)
        joblib.dump(gbrt, "{}_{}_{}_{}.{}".format("ForecastModels\gbrt",horizon,"tau",tau,"sav"))
        tau += 1
################################################################################
# Function to predict 1..K gbrt forecast models
################################################################################

def gbrt_forecast(horizon,inpEndo,inpExo,tar,quantileLevels,i):
    '''
    This function should read .sav files and predict for all
    quantileLevels for "horizon" at the current time step i.

    Arguments:
    - horizon: the forecast horizon.
    - inpEndo: A pandas DataFrame containing the endogenous inputs.
    - inpExo: A pandas DataFrame containing the endogenous inputs.
    - tar: A pandas DataFrame containting the targets.
    - quantileLevels: a vector with the nominal probabilities expressed
      as quantiles.
    - i: current time step.
    Returns:
    - Numpy array (horizon x length(quantileLevels))
    '''
    # Training months
    tr_m = [2,3,5,6] # Removed months 1, 7 and 8 to speed it up but also because they're less relevant.
    # Test month April
    te_m = 4

    preds = []
    for h in range(1,horizon+1,1):
        train = inpEndo[inpEndo.index.month.isin(tr_m)]
        train = train.join(inpExo[inpExo.index.month.isin(tr_m)], how="inner")
        train = train.join(tar[tar.index.month.isin(tr_m)], how="inner")

        test = inpEndo[inpEndo.index.month == te_m]
        test = test.join(inpExo[inpEndo.index.month == te_m], how="inner")
        test = test.join(tar[tar.index.month == te_m], how="inner")

        cols = ["{}_{}".format("t",h)]
        feature_cols = inpEndo.filter(regex='y').columns.tolist()
        feature_cols_endo = inpEndo.filter(regex='y').columns.tolist()
        feature_cols.extend(["Temperature_{}".format(h),"TotalCloudCover_{}".format(h)]) # ,"WindUMS_{}".format(h),"WindVMS_{}".format(h)

        train = train[cols + feature_cols].dropna(how="any")
        test  = test[cols + feature_cols].dropna(how="any")

        train_X = train[feature_cols].values
        test_X  = test[feature_cols].values

        scaler = preprocessing.StandardScaler().fit(train_X)
        test_X = scaler.transform(test_X)

        test_y = test[cols].values

        tau = 1
        test_pred = []
        for q in quantileLevels:
            model = joblib.load("{}_{}_{}_{}.{}".format("ForecastModels\gbrt",h,"tau",tau,"sav"))
            test_pred.append(model.predict(test_X[i,:].reshape(1,-1)))
            tau+=1
        tmp = np.vstack(test_pred).T # List to NumPy array
        preds.append(tmp)
    test_pred = np.vstack(preds)
    test_pred.sort(axis=1)
    return(test_pred)
