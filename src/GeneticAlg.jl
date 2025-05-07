using Evolutionary, Random, Distances, DataFrames, Plots

#Todo: Probably remove due to lack of usefulness.

# Example: 2-objective on (n×d) design vector
function nsga2_objective(x; n=10, d=2, formula=@formula(y~1+x1+x2))
    X = reshape(x, n, d)
    df = DataFrame(y=ones(n), x1=X[:,1], x2=X[:,2])
    f1 = -calculate_d_optimality(df, formula)
    f2 = max_min_distance(df)
    return [f1, f2]
end

# Box constraints helper
function run_nsga2(; n=10, d=2, popsize=100, iters=2000)
    lb = fill(-1.0, n*d)
    ub = fill( 1.0, n*d)
    cn = BoxConstraints(lb,ub)
    algo = NSGA2(populationSize=popsize)
    opts = Evolutionary.Options(iterations=iters, parallelization=:thread, show_trace=true)
    return Evolutionary.optimize(nsga2_objective, cn, algo, opts)
end

# Plotting Pareto front
res = run_nsga2()
objs = [nsga2_objective(sol) for sol in res.minimizer]
f1 = getindex.(objs,1)
f2 = getindex.(objs,2)
scatter(f1, f2; xlabel="–D‐eff", ylabel="max–min dist", legend=false)