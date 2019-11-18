using JuMP
using GLPK

function mpc_run(NL, battInit)
    # NL represents the 24h net load forecast
    # T is the horizon (96 in case of 15 min data)
    # battInit is the battery SoC at the time when this function is called

    T = length(NL) # size(NL)[1]
    #model = Model(solver=GLPKSolverMIP())
    model = Model(with_optimizer(GLPK.Optimizer))

    # Declare variables
    @variable(model, u[t = 1:T], Bin)
    @variable(model, v[t = 1:T], Bin)
    @variable(model, 0 <= Pch[t = 1:T] <= 5 ) # Charging power of battery
    @variable(model, 0 <= Pdis[t = 1:T] <= 5 ) # Disharging power of battery
    @variable(model, 0 <= PfrGrid[t = 1:T] <= 5 ) # Power from grid
    @variable(model, 0 <= PtoGrid[t = 1:T] <= 5 ) # Power to grid
    @variable(model, 0.5 <= Energy[t = 1:T] <= 7.5) # start is a guess, not fixed

    # Declare constraints
    @constraint(model, powerBalance[t = 1:T],
                    NL[t] == PfrGrid[t] - PtoGrid[t] - Pch[t] + Pdis[t])
    @constraint(model, batteryEnergy[t = 2:T],
                    Energy[t] == Energy[t-1] + 0.25*Pch[t-1] - 0.25*Pdis[t-1])
    @constraint(model, batteryInitial[t = 1], Energy[1] == battInit)
    @constraint(model, binVars[t = 1:T], u[t] + v[t] == 1)      # Binary variables for constraints below
    @constraint(model, binCH[t = 1:T], Pch[t] <= u[t] * 5)      # Avoid arbitrage and discharging
    @constraint(model, binDIS[t = 1:T], Pdis[t] <= v[t] * 5)    # battery energy to the grid.
    @constraint(model, bin2GR[t = 1:T], PtoGrid[t] <= u[t] * 5)

    @objective(model, Min, sum(PfrGrid + PtoGrid))
    #solve(model)
    optimize!(model)
    return round(JuMP.value(Pch[1]),digits=2),
           round(JuMP.value(Pdis[1]),digits=2),
           round(JuMP.value(PfrGrid[1]),digits=2),
           round(JuMP.value(PtoGrid[1]),digits=2)
           #round(JuMP.value(Energy[1]),digits=2),
           #round(getobjectivevalue(model),digits=2)
end
