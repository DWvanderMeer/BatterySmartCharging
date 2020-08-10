'''
Author: Dennis van der Meer
E-mail: denniswillemvandermeer[at]gmail.com

This script executes the stochastic model predictive controller defined
in functions.py. The scripts relies on several data sources, namely:
1) Electricity usage and PV power production (from a research villa in this case),
2) Exogenous inputs for the forecast models (temperature and cloud cover forecasts),
3) Electricity prices from Nord Pool.

The data are or have already been formatted into matrix format such that each row
represents a time while the columns represent the forecast horizon. It should be
noted that the exogenous inputs (in 2 above) have to be downloaded manually. An
alternative would be to train the forecast model on endogenous data only or to use
the persistence ensemble in functions.py (peen_forecast).

The script can be run in parallel for the two case studies (finding the joint optimal
solution or averaging over the independent optimal solutions) or a single case study
that uses a perfect forecast. The latter represents the upper limit of what can be
achieved.
'''

import os
import multiprocessing as mp
import functions as fn
import run as run
import forecast_functions as fc_fn
import pandas as pd
import numpy as np
import warnings
import time
import glob
start_time = time.time()
# To suppress RuntimeWarning: overflow encountered in long_scalars
# self.num_scalar_data += const.size
# No other warnings at this time.
warnings.filterwarnings("ignore")

days = 20
horizon = 96
SoC = 0.5
num_samples = 49

# DIRECTORIES ON SERVER
dir0 = r"C:\Users\denva787\Documents\dennis\RISE" # Windows
os.chdir(r"C:\Users\denva787\Documents\dennis\RISE") # Windows
MULTIVARIATE_RESULTS = r"C:\Users\denva787\Documents\dennis\RISE\Results\multivariate" # Windows
FORECASTS = r"C:\Users\denva787\Documents\dennis\RISE\Forecasts" # Windows
FORECASTMODELS = r"C:\Users\denva787\Documents\dennis\RISE\ForecastModels" # Windows
RESULTS = r"C:\Users\denva787\Documents\dennis\RISE\Results" # Windows

# Read the various data sets
tar = pd.read_csv(
    os.path.join("Data", "Forskningsvillan_2018_10_01_to_2019_10_01_15min_data.csv"),
    delimiter=";",
    parse_dates=True,
    index_col=0,
    header=0
)
tar = tar[['Solar production','Consumption']].rename(columns={'Solar production':'Production'})
tar_sub = tar['2019-01-01':'2019-10-01 00:15:00'].resample('15min').mean()
NL = (tar_sub.Consumption - tar_sub.Production)

inpEndo = fn.series_to_supervised(NL.values.tolist(),96,0,False)
inpEndo = inpEndo.iloc[:, ::-1]
inpEndo.index = NL.index

tar = pd.read_csv(
    os.path.join("Data", "targets.csv"),
    delimiter=",",
    parse_dates=True,
    index_col=0,
    header=0
)

inpExo = pd.read_csv(
    os.path.join("Data", "exo_nwp.csv"),
    delimiter=",",
    parse_dates=True,
    index_col=0,
    header=0
)

elPrice = pd.read_csv(
    os.path.join("Data", "elPriceApril.csv"),
    delimiter=",",
    parse_dates=True,
    index_col=0,
    header=0
).set_index('datetime')
lambdas = elPrice.loc['2016-04-02 03:15:00':'2016-04-30 23:45:00'].to_numpy() # Select only buy and sell
perfectFC = NL['2019-04-07 03:00:00':'2019-04-30 23:45:00'].to_numpy() # 03:00 because resampling uses ceiling instead of R's floor
perfectFC = np.reshape(perfectFC, (perfectFC.shape[0],1))
#print(NL[NL.index.month==4].head())


# RUNNING STOCHASTIC CASE STUDIES WITH PROBABILISTIC FORECASTS
if __name__ == '__main__':
    pool = mp.Pool(processes=2)
    bo = [True,False]
    taus = np.arange(0.1,0.91,0.1)
    my_res = [pool.apply_async(run.run, args=(days,horizon,SoC,NL,b,num_samples,taus,inpEndo,inpExo,tar,lambdas,perfectFC)) for b in bo]
    my_res = [p.get() for p in my_res]
    pool.close()
    print("--- %s seconds ---" % (time.time() - start_time))

# Run just one process during testing
#taus = np.arange(0.1,0.91,0.1)
#fn.run(days,horizon,SoC,NL,False,num_samples,taus,inpEndo,inpExo,tar,lambdas,perfectFC)

# Run the uncontrolled charging
#fn.uncontrolledCharging(perfectFC,SoC,lambdas)

'''
# RUNNING DETERMINISTIC WITH PERFECT FORECASTS
# NOTE THAT THE FUNCTION RUN SHOULD BE MANUALLY ADJUSTED FOR THIS SPECIAL CASE
#num_samples = 1
taus = np.arange(0.1,0.91,0.1)
fn.run(days,horizon,SoC,NL,False,num_samples,taus,inpEndo,inpExo,tar,lambdas,perfectFC)
'''

'''
# Train the GBRT models
# Use the same hyperparameters for all forecast horizons.
all_files = glob.glob(os.path.join("CrossValidation", "*.txt"))
params = pd.concat((pd.read_csv(f, sep="\t") for f in all_files), ignore_index=True).sort_values(by=['CRPS']).iloc[0,0:6]
if __name__ == '__main__':
    pool = mp.Pool(processes=64)
    horizon = np.arange(1,97,1)
    my_res = [pool.apply_async(fc_fn.gbrt_training, args=(h,inpEndo,inpExo,tar,params)) for h in horizon]
    my_res = [p.get() for p in my_res]
    pool.close()
'''
'''
# Test the GBRT models
if __name__ == '__main__':
    pool = mp.Pool(processes=64)
    horizon = np.arange(1,97,1)
    my_res = [pool.apply_async(fc_fn.gbrt_forecast, args=(h,inpEndo,inpExo,tar)) for h in horizon]
    my_res = [p.get() for p in my_res]
    pool.close()
'''
'''
# Train and test the QR models on GBRT forecasts
if __name__ == '__main__':
    pool = mp.Pool(processes=64)
    horizon = np.arange(1,97,1)
    my_res = [pool.apply_async(fc_fn.qr_train_forecast, args=[h]) for h in horizon]
    my_res = [p.get() for p in my_res]
    pool.close()
'''
