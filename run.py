'''
Author: Dennis van der Meer
E-mail: denniswillemvandermeer[at]gmail.com

This script contains the function that iterates over the time steps and calls
the stochastic model predictive control function every time. Currently, the
input to the smpc algorithm (the scenarios) are prestored and loaded during run
time. However, code is prepared to issue forecasts and generate scenarios on the
fly.
'''

import numpy as np
import pandas as pd
import os
#from copulas.multivariate import GaussianMultivariate
#from copulae import GaussianCopula
import cvxpy as cp
import cplex as cx
import time
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn import ensemble
import joblib as joblib
from tqdm import tqdm
from scipy import stats
import glob
import functions as fn
import batterySMPC as smpc

# For rpy2
#os.environ['PYTHONHOME'] = r"C:\Users\denva787\Documents\dennis\RISE\Code\env\Lib"
#os.environ['PYTHONPATH'] = r"C:\Users\denva787\Documents\dennis\RISE\Code\env\Lib\site-packages"
#os.environ['R_HOME'] = r"C:\Program Files\R\R-3.6.2"
#os.environ['R_USER'] = r"C:\Users\denva787\Documents\dennis\RISE\Code\env\Lib\site-packages\rpy2"

#import rpy2.robjects as ro
#from rpy2.robjects import pandas2ri
#from rpy2.robjects.conversion import localconverter
#import rpy2.robjects.packages as rpackages
#copula = rpackages.importr('copula') # Load copula library
import warnings
warnings.filterwarnings("ignore", message="Error while trying to convert the column ")

# DIRECTORIES ON SERVER
dir0 = r"C:\Users\denva787\Documents\dennis\RISE" # Windows
os.chdir(r"C:\Users\denva787\Documents\dennis\RISE") # Windows
#MULTIVARIATE_RESULTS = r"C:\Users\denva787\Documents\dennis\RISE\Results\multivariate" # Windows
MULTIVARIATE_RESULTS = r"C:\Users\denva787\Dropbox\BatteryRise\multivariate" # Windows
FORECASTS = r"C:\Users\denva787\Documents\dennis\RISE\Forecasts" # Windows
RESULTS = r"C:\Users\denva787\Documents\dennis\RISE\Results" # Windows
RESULTS = r"C:\Users\denva787\Dropbox\BatteryRise" # Windows
# DIRECTORIES ON LOCAL MACHINE
#dir0 = "/Users/Dennis/Desktop/Drive/PhD-Thesis/Projects/RISE/" # macOS
#os.chdir('/Users/Dennis/Desktop/Drive/PhD-Thesis/Projects/RISE/') # macOS
#MULTIVARIATE_RESULTS = "/Users/Dennis/Dropbox/BatteryRise/multivariate/" # macOS
#FORECASTS = "/Users/Dennis/Dropbox/BatteryRise/Forecasts/" # macOS
#FORECASTMODELS = "/Users/Dennis/Desktop/ForecastModels/" # Drive/PhD-Thesis/Projects/RISE/

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
    times = [] # Store the time vector
    # Create a matrix with time lags in the columns to train the copula.
    timeLags = fn.series_to_supervised(NL.to_numpy().tolist(), horizon, 0, False) # [NL.index.month==4]
    timeLags = timeLags.set_index(NL.index)
    timeLags = timeLags.iloc[:, ::-1]
    timeLags.dropna(axis=0,inplace=True)

    # Load the forecasts in order to speed up the computation. Otherwise, uncomment
    # the code inside the for-loop to produce forecasts on the fly.
    #fcs = [np.loadtxt(os.path.join(FORECASTS,"{}_{}.{}".format("gbrt",h,"txt"))) for h in range(1,97,1)]
    for i in tqdm(np.arange(0,5)): # days*horizon 575
        E_opt.append(SoC)
        '''
        Originally, a forecast is issued at each i-th iteration as well as a set
        of autocorrelated random numbers. These are then transformed to the original
        domain to serve as input to the stochastic model predictive control
        algorithm. However, now we simply read the stored scenarios instead. The
        old code can be found directly below.

        # Prepare forecast:
        #fc = gbrt_forecast(horizon,inpEndo,inpExo,tar,quantileLevels,i) # gbrt
        #fc = np.vstack([fcs[h][i,:] for h in range(horizon)]) # Read fc instead of producing it

        # Prepare copula samples:
        samples = copula.rCopula(num_samples, cop_model) # Sample from the copula
        samples = np.array(samples)
        # Generate scenarios from the above two:
        scenarios = np.zeros((num_samples,horizon))
        for h in range(horizon):
            scenarios[:,h] = np.interp(samples[:,h], quantileLevels, fc[h,:], np.amin(fc[h,:]), np.amax(fc[h,:]))
        scenarios = np.transpose(scenarios)

        # In case of a perfect forecast (the ideal situation):
        scenarios = perfectFC[i:i+horizon]
        '''
        scenario_file = os.path.join(MULTIVARIATE_RESULTS,"{}_{}.{}".format("B",i+1,"txt"))
        df=pd.read_csv(scenario_file, header=None, sep="\t", index_col=0, parse_dates=True, infer_datetime_format=True)
        # In df (K x 2+S), the first column is "ValidTime", the second column observations and rest scenarios
        scenarios = df.iloc[:,1:df.shape[1]].values # Select only the scenarios

        start_time = time.time()
        res = smpc.smpc(scenarios, SoC, eq, lambdas[i:i+horizon,:])
        end_time = time.time() - start_time
        '''
        Remember, control vector u is the following:
        u[0,t] = Pch
        u[1,t] = Pdis
        u[2,t] = PfrGrid
        u[3,t] = PtoGrid
        '''
        ## Compensate for erroneous forecasts using the grid (2020-08-10) ##
        PB = res[0] - res[1] + df.iloc[0,0] # The last is the observed net load
        # If PB > 0 --> surplus of power, else shortage of power.
        if PB > 0.0: # Lower power from grid or increase power to grid
            res[2] = np.abs(PB)
            res[3] = 0
        elif PB < 0.0: # Lower power to grid or increase power from grid
            res[3] = np.abs(PB)
            res[2] =0
        ## Compensate for erroneous forecasts using the grid (2020-08-10) ##
        SoC += 0.25*(0.96*res[0]-res[1]/0.96)/7.2 # Update SoC  with latest result.
        res = np.append(res,lambdas[i,:])
        res = np.append(res,end_time)
        res = np.append(res,SoC)
        res = np.append(res,df.iloc[0,0]) # Add the observation
        res = np.reshape(res,(1,9))
        res_df = pd.DataFrame(res, columns=['Pch','Pdis','PfrGrid','PtoGrid','buyPrice','sellPrice','runTime','Energy','netLoad'])
        res_df.index = NL[NL.index.month == 4].index[[i+horizon]] # Because the forecasts start from the second
        data.append(res)
        times.append(df.index[0])
    tmp_df = pd.DataFrame(np.concatenate(data), columns=['Pch','Pdis','PfrGrid','PtoGrid','buyPrice','sellPrice','runTime','Energy','netLoad'], index=times)
    tmp_df.to_csv(os.path.join(RESULTS,"{}_{}.{}".format("mpc_test_with_compensation", eq, "txt")), header=True, sep='\t') # without_obs_without_bin_with_compensation_updated
