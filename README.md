# BatterySmartCharging

These scripts are part of a project that aims to optimally charge a battery considering electricity prices and probabilistic forecasts of the net load (electricity usage minus solar power production). The stochastic model predictive controller (MPC) optimizes the objective over a control horizon subject to constraints for a given number of forecasted net load scenarios. Typically, the expectation of the optimal solutions constitutes the optimal charging strategy to implement in the current time step, which itself need not be an optimal charging strategy for any of the scenarios. Therefore, we propose an alternative in which we find a joint charging strategy that is optimal across all scenarios.

The findings of this study are submitted to the Probabilistic Methods Applied to Power Systems 2020 conference, which takes place 18-21 August 2020.

Script functions.py contains all the necessary functions while execute.py is used to load the necessary data and run the stochastic MPC for a certain number of days.