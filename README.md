# BatterySmartCharging

These scripts are part of a project that aims to optimally charge a battery considering electricity prices and probabilistic forecasts of the net load (electricity usage minus solar power production). The stochastic model predictive controller (MPC) optimizes the objective over a control horizon subject to constraints for a given number of forecasted net load scenarios. Typically, the expectation of the optimal solutions constitutes the optimal charging strategy to implement in the current time step, which itself need not be an optimal charging strategy for any of the scenarios. Therefore, we propose an alternative in which we find a joint charging strategy that is optimal across all scenarios.

The findings of this study are accepted for publication in Applied Energy. You can find the article here: https://www.sciencedirect.com/science/article/pii/S0306261920316767.

Script functions.py contains all the necessary functions while execute.py is used to load the necessary data and run the stochastic MPC for a certain number of days.
Updated 2020-08-10: functions.py is divided into run.py, forecast_functions.py and batterySMPC.py to improve the organization. Simultaneously, the functions have been updated. Note that batterySMPC.py contains the actual MPC algorithm, which is the main contribution of the article.
