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
import glob
#os.environ['PYTHONHOME'] = r"C:\Users\denva787\Documents\dennis\RISE\Code\env\Lib"
#os.environ['PYTHONPATH'] = r"C:\Users\denva787\Documents\dennis\RISE\Code\env\Lib\site-packages"
import time
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn import ensemble
import joblib as joblib
import properscoring as ps
from tqdm import tqdm
from scipy import stats
import warnings
warnings.filterwarnings("ignore", message="Error while trying to convert the column ")

# DIRECTORIES ON SERVER
dir0 = r"C:\Users\denva787\Documents\dennis\RISE" # Windows
os.chdir(r"C:\Users\denva787\Documents\dennis\RISE") # Windows
MULTIVARIATE_RESULTS = r"C:\Users\denva787\Documents\dennis\RISE\Results\multivariate" # Windows
FORECASTS = r"C:\Users\denva787\Documents\dennis\RISE\Forecasts" # Windows
OBSERVATIONS = r"C:\Users\denva787\Documents\dennis\RISE\Observations" # Windows
FORECASTMODELS = r"C:\Users\denva787\Documents\dennis\RISE\ForecastModels" # Windows
RESULTS = r"C:\Users\denva787\Documents\dennis\RISE\Results" # Windows
# DIRECTORIES ON LOCAL MACHINE
#dir0 = "/Users/Dennis/Desktop/Drive/PhD-Thesis/Projects/RISE/" # macOS
#os.chdir('/Users/Dennis/Desktop/Drive/PhD-Thesis/Projects/RISE/') # macOS
#MULTIVARIATE_RESULTS = "/Users/Dennis/Dropbox/BatteryRise/multivariate/" # macOS
#FORECASTS = "/Users/Dennis/Dropbox/BatteryRise/Forecasts/" # macOS
#FORECASTMODELS = "/Users/Dennis/Desktop/ForecastModels/" # Drive/PhD-Thesis/Projects/RISE/
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

    #scaler = preprocessing.StandardScaler().fit(train_X)
    #train_X = scaler.transform(train_X)

    train_y = train[cols].values
    test_y = test[cols].values

    # Perhaps add some jitter:
    #Xtra_jitter = np.random.normal(1*Xtra,0.01) # Add some random noise to avoid singular matrix

    quantreg = sm.QuantReg(train_y, train_X)
    tau = 1
    for q in taus:
        res = quantreg.fit(q=q, max_iter=10000)
        #res.save("{}_{}_{}_{}.{}".format("ForecastModels\qr",horizon,"tau",tau,"pickle"))
        res.save(os.path.join(FORECASTMODELS,"{}_{}_{}_{}.{}".format("qr",horizon,"tau",tau,"pickle")))
        tau += 1

################################################################################
# Function to train and predict 1..K qr forecast models
################################################################################

def qr_train_forecast(horizon):
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
    taus = np.arange(0.05,0.96,0.05)
    #taus = np.linspace(0.001,0.999,num=21) # quantreg does not accept 0
    # Training month April
    tr_m = [5] # Train the QR on the uncalibrated GBRT forecasts
    # Test month May
    te_m = [4] # Test the QR on the uncalibrated GBRT forecasts

    preds = []
    gbrt_fc_str = os.path.join(FORECASTS, "{}_{}.{}".format("gbrt",horizon,"txt")) # glob.glob(os.path.join(FORECASTS, "g*.txt")) # Select gbrt forecasts
    gbrt_fc = pd.read_csv(gbrt_fc_str,sep="\t",parse_dates=True)#,header=True,index_col="DateTime",parse_dates=True
    gbrt_fc['DateTime'] = pd.to_datetime(gbrt_fc['DateTime'])
    gbrt_fc = gbrt_fc.set_index(pd.DatetimeIndex(gbrt_fc['DateTime']))
    gbrt_fc = gbrt_fc.drop(['DateTime'], axis=1)

    gbrt_ob_str = os.path.join(OBSERVATIONS, "{}_{}.{}".format("obs",horizon,"txt")) # glob.glob(os.path.join(FORECASTS, "g*.txt")) # Select gbrt forecasts
    gbrt_ob = pd.read_csv(gbrt_ob_str,sep="\t",parse_dates=True)#,header=True,index_col="DateTime",parse_dates=True
    gbrt_ob['DateTime'] = pd.to_datetime(gbrt_ob['DateTime'])
    gbrt_ob = gbrt_ob.set_index(pd.DatetimeIndex(gbrt_ob['DateTime']))
    gbrt_ob = gbrt_ob.drop(['DateTime'], axis=1)

    train = gbrt_fc[gbrt_fc.index.month.isin(tr_m)]
    train = train.join(gbrt_ob[gbrt_ob.index.month.isin(tr_m)], how="inner")
    test = gbrt_fc[gbrt_fc.index.month.isin(te_m)]
    test = test.join(gbrt_ob[gbrt_ob.index.month.isin(te_m)], how="inner")

    cols = ["{}_{}".format("t",horizon)]
    feature_cols = gbrt_fc.columns.tolist() # Take the fc columns as feature names

    train = train[cols + feature_cols].dropna(how="any")
    test  = test[cols + feature_cols].dropna(how="any")

    train_X = train[feature_cols].values
    test_X  = test[feature_cols].values
    train_y = train[cols].values
    test_y = test[cols] # To store as pandas series

    tau = 1
    test_pred = []
    quantreg = sm.QuantReg(train_y, train_X)
    for q in taus:
        model = quantreg.fit(q=q, max_iter=10000)
        test_pred.append(model.predict(test_X))
        tau+=1

    tmp = np.vstack(test_pred).T # List to NumPy array
    fc_df = pd.DataFrame(data=tmp,index=test.index) # To store as pandas DataFrame
    fc_df.to_csv(os.path.join(FORECASTS,"{}_{}.{}".format("qr",horizon,"txt")),sep="\t")
    test_y.to_csv(os.path.join(OBSERVATIONS,"{}_{}.{}".format("qr_obs",horizon,"txt")),sep="\t")

################################################################################
# Function to train 1..K gbrt forecast models
################################################################################

def gbrt_training(horizon,inpEndo,inpExo,tar,params):
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
    tr_m = [3,5,6,7,8,9]
    # Testing months
    te_m = [4]
    # Training months
    tr_m = [2,3,5,6]
    # Test month April
    te_m = [4]

    cols = ["{}_{}".format("t",horizon)]

    train = inpEndo[inpEndo.index.month.isin(tr_m)]
    train = train.join(inpExo[inpExo.index.month.isin(tr_m)], how="inner")
    train = train.join(tar[tar.index.month.isin(tr_m)], how="inner")

    test = inpEndo[inpEndo.index.month.isin(te_m)]
    test = test.join(inpExo[inpEndo.index.month.isin(te_m)], how="inner")
    test = test.join(tar[tar.index.month.isin(te_m)], how="inner")

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
        #joblib.dump(gbrt, "{}_{}_{}_{}.{}".format("ForecastModels\gbrt",horizon,"tau",tau,"sav"))
        joblib.dump(gbrt,os.path.join(FORECASTMODELS,"{}_{}_{}_{}.{}".format("gbrt",horizon,"tau",tau,"sav")))
        tau += 1

################################################################################
# Function to predict 1..K gbrt forecast models
################################################################################

def gbrt_forecast(horizon,inpEndo,inpExo,tar):
    '''
    This function should read .save files and predict for all
    quantileLevels for "horizon". Unlike the forecast model used
    during the operational simulation of this project, this function
    forecasts the entire month in one go and is therefore not
    dependent on i.

    Arguments:
    - horizon: the forecast horizon.
    - inpEndo: A pandas DataFrame containing the endogenous inputs.
    - inpExo: A pandas DataFrame containing the endogenous inputs.
    - tar: A pandas DataFrame containting the targets.
    Returns:
    - K pd DataFrames (test set length x length(taus)) with forecasts
    - K pd Series (test set length) with observations
    '''
    taus = np.arange(0.1,0.91,0.1)
    # Training months
    tr_m = [2,3,5,6]
    # Test month April
    te_m = [4]

    preds = []
    #for h in range(1,horizon+1,1):
    train = inpEndo[inpEndo.index.month.isin(tr_m)]
    train = train.join(inpExo[inpExo.index.month.isin(tr_m)], how="inner")
    train = train.join(tar[tar.index.month.isin(tr_m)], how="inner")

    test = inpEndo[inpEndo.index.month.isin(te_m)]
    test = test.join(inpExo[inpEndo.index.month.isin(te_m)], how="inner")
    test = test.join(tar[tar.index.month.isin(te_m)], how="inner")

    cols = ["{}_{}".format("t",horizon)]
    feature_cols = inpEndo.filter(regex='y').columns.tolist()
    feature_cols_endo = inpEndo.filter(regex='y').columns.tolist()
    feature_cols.extend(["Temperature_{}".format(horizon),"TotalCloudCover_{}".format(horizon)]) # ,"WindUMS_{}".format(h),"WindVMS_{}".format(h)

    train = train[cols + feature_cols].dropna(how="any")
    test  = test[cols + feature_cols].dropna(how="any")
    #test = test.loc['2019-04-02 00:00:00':'2019-04-23 23:45:00'] # Because of NaNs
    #test1 = test[test.isna().any(axis=1)] # Check NaNs, appears first on 24 april

    train_X = train[feature_cols].values
    test_X  = test[feature_cols].values

    scaler = preprocessing.StandardScaler().fit(train_X)
    test_X = scaler.transform(test_X)

    #test_y = test[cols].values
    test_y = test[cols] # To store as pandas series

    tau = 1
    test_pred = []
    for q in taus:
        #model = joblib.load("{}_{}_{}_{}.{}".format("ForecastModels\gbrt",horizon,"tau",tau,"sav")
        model = joblib.load(os.path.join(FORECASTMODELS,"{}_{}_{}_{}.{}".format("gbrt",horizon,"tau",tau,"sav")))
        test_pred.append(model.predict(test_X))
        tau+=1
    tmp = np.vstack(test_pred).T # List to NumPy array
    fc_df = pd.DataFrame(data=tmp,index=test.index) # To store as pandas DataFrame
    #np.savetxt("{}_{}.{}".format("Forecasts/gbrt",horizon,"txt"),tmp,delimiter="\t")
    #np.savetxt("{}_{}.{}".format("Observations/obs",horizon,"txt"),test_y,delimiter="\t")
    fc_df.to_csv(os.path.join(FORECASTS,"{}_{}.{}".format("gbrt",horizon,"txt")),sep="\t")
    test_y.to_csv(os.path.join(OBSERVATIONS,"{}_{}.{}".format("obs",horizon,"txt")),sep="\t")
