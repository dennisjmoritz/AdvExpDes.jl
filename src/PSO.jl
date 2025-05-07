using Pkg
#Pkg.add("StatsKit")
#Pkg.add("LinearAlgebra")
#Pkg.add("Plots")
using StatsKit
using LinearAlgebra
using Plots
using Base.Threads
Design_df = DataFrame(
  y = repeat([2], 18),
  x1 = repeat([-1,0,1], inner = 6), 
  x2 = repeat(repeat([-1, 0, 1], inner = 2), 3),
  x3 = repeat([-1, 0, 1], 6)
)
form = @formula(y ~ 1 + x1 + x2 + x3 + x1*x2 + x2*x3 + x3*x1 + x1^2 + x2^2 + x3^2)
form_simp = @formula(y ~ 1 + x1 + x2 + x3)

function modelmatrix_df(design_df::DataFrame, formula)
  sch = schema(formula, design_df)
  resolved_formula = apply_schema(formula, sch)
  X_parts = modelcols(resolved_formula, design_df)
  Xmat = reduce(hcat, X_parts)[:, 2:end]
  colnames = coefnames(resolved_formula)[2]  # predictor names only
  return DataFrame(Xmat, Symbol.(collect(colnames)))
end


function modelmatrix(designmatrix::Matrix, formula)
    design_df = DataFrame(designmatrix, :auto)
    design_df.y = ones(nrow(design_df))  
  sch = schema(formula, design_df)
  resolved_formula = apply_schema(formula, sch)
    modelcols(resolved_formula.rhs, design_df)
end

designmat = rand(Float64, (10,4))
modelmatrix(designmat, form_simp)

modelmatrix_df(Design_df, form)
names(modelmatrix_df(Design_df, form))

function calculate_d_optimality(input, formula; is_model_matrix = false)

    if is_model_matrix
        model_matrix_df = input
    else
        model_matrix_df = modelmatrix_df(input, formula)
    end
    model_matrix = Matrix(model_matrix_df)
    # Calculate the determinant of the transpose of the model matrix multiplied by itself
    det_XtX = det(model_matrix' * model_matrix)
    
    # Get the number of observations (rows) in the model matrix
    n = size(model_matrix, 2)
    # Get the number of terms (cols) in the model matrix
    p = size(model_matrix, 1)
    
    # Calculate the D-efficiency
  d_optimality = 100 * (det_XtX ^ (1/p)) / n
    
  return d_optimality
end


calculate_d_optimality(Design_df, form_simp)
function max_min_distance(design_matrix;
                           space_bounds = (-1.0, 1.0),
                           grid_density = 50)

    design_matrix = Matrix(design_matrix)[:,2:end]
    d = size(design_matrix, 2)  # number of variables
    grids = [range(space_bounds[1], space_bounds[2], length=grid_density) for _ in 1:d]
    
    max_min_dist = 0.0
    
    for p in Iterators.product(grids...)
        point = collect(p)
        dists = [norm(row .- point) for row in eachrow(design_matrix)]
        min_dist = minimum(dists)
        if min_dist > max_min_dist
            max_min_dist = min_dist
        end
    end
    
    return max_min_dist
end
#Design_df_neg = DataFrame(
#  y = repeat([2], 12),
#  x1 = repeat([-1,0,-.5], inner = 4), 
#  x2 = repeat(repeat([-1, -.5], inner = 2), 3),
#  x3 = repeat([-.5,0], 6)
#)
max_min_distance(Design_df, space_bounds = (-1.0, 1.0), grid_density = 50)
max_min_distance(Design_df)
#max_min_distance(Matrix(Design_df_neg)[:,2:end], space_bounds = (-1.0, 1.0), grid_density = 50)

#Evaluation function
# Goal: For a collection of design matrices, evaluate various design criteria functions for the matrices

function evaluate_designs(design_matrices,
                          objectives;
                          formulas)

    n = length(design_matrices)
    results = Vector{NamedTuple}(undef, n)

    @threads for i in 1:n
        X = design_matrices[i]
        rowdata = Dict{Symbol, Float64}()

        for (obj_name, obj_fn, needs_formula) in objectives
            for (j, f) in enumerate(formulas)
                colname = needs_formula && length(formulas) > 1 ?
                          Symbol("$(obj_name)_f$(j)") :
                          Symbol(obj_name)
                val = needs_formula ? obj_fn(X, f) : obj_fn(X)
                rowdata[colname] = val
                needs_formula || break
            end
        end

        results[i] = NamedTuple(rowdata)
    end

    return DataFrame(results)
end

# Usage example

designs = [DataFrame(y = ones(10),
                     x1 = rand(10) * 2 .- 1,
                     x2 = rand(10) * 2 .- 1,
                     x3 = rand(10) * 2 .- 1
                     ) for _ in 1:100]

formulas = [
    @formula(y ~ 1 + x1 + x2 + x3),
    @formula(y ~ 1 + x1 + x2 + x3 + x1*x2 + x2*x3 + x3*x1 + x1^2 + x2^2 + x3^2)
]

objectives = [
    ("d_opt", calculate_d_optimality, true),
    ("maxmin", X -> max_min_distance(X, grid_density=50), false)
]

evaluate_designs(designs, objectives; formulas=formulas)



using Evolutionary, Distances, LinearAlgebra, Random, DataFrames

# problem dimensions
n, d = 10, 2
form_2d_simp = @formula(y ~ 1 + x1 + x2)
# your two-objective function
function objective(x)
    X = reshape(x, n, d)
    df = DataFrame(y = ones(n),
                   x1 = X[:,1],
                   x2 = X[:,2])
    f1 = -calculate_d_optimality(df, form_2d_simp)
    f2 = max_min_distance(df)
    return [f1, f2]
end

# set box constraints 
lb   = fill(-1.0, n*d)
ub   = fill( 1.0, n*d)
cnst = BoxConstraints(lb, ub)

# run NSGA-II with population size 100 for 200 generations
res_simp = Evolutionary.optimize(
    objective,
    cnst,                                 
    NSGA2(populationSize = 100),            
    Evolutionary.Options(iterations = 2000,
        parallelization = :thread,
        show_trace = true )  
)


using Plots
objs        = [objective(sol) for sol in res_simp.minimizer]
f1 = getindex.(objs, 1)    # -D‐optimality
f2 = getindex.(objs, 2)    # max–min distance

scatter(
    f1, f2;
    xlabel = "D‐optimality",
    ylabel = "max–min distance",
    legend = false,
    markersize = 6,
    markerstrokewidth = 0
)
res_simp

using Evolutionary, Random, DataFrames, Statistics, Plots, StatsModels

# === Problem Setup ===
n, d = 10, 2
form_2d_simp = @formula(y ~ 1 + x1 + x2)

function objective(x)
    X = reshape(x, n, d)
    df = DataFrame(y = ones(n),
                   x1 = X[:,1],
                   x2 = X[:,2])
    f1 = -calculate_d_optimality(df, form_2d_simp)
    f2 = max_min_distance(df)
    return [f1, f2]
end

lb = fill(-1.0, n * d)
ub = fill( 1.0, n * d)
bounds = BoxConstraints(lb, ub)

# === NSGA2 Runner ===
function run_nsga2(params; seed=0)
    Random.seed!(seed)

    algo = NSGA2(populationSize = params[:popsize])
    opts = Evolutionary.Options(
        iterations = params[:iterations],
        parallelization = :thread,
        show_trace = true
    )

    return Evolutionary.optimize(objective, bounds, algo, opts)
end

# === Evaluation Function ===
function evaluate_result(res)
    fronts = res.minimizer
    scores = [objective(x) for x in fronts]
    f1s = first.(scores)
    f2s = last.(scores)
    return (
        n_points = length(scores),
        f1_range = maximum(f1s) - minimum(f1s),
        f2_range = maximum(f2s) - minimum(f2s),
    )
end

# === Parameter Grid and Run ===
param_grid = [
    Dict(:popsize => p, :iterations => i)
    for p in [50, 100, 200], i in [5000, 10000, 20000]
]
seeds = 1:3
results = []

for params in param_grid, seed in seeds
    res = run_nsga2(params; seed=seed)
    metrics = evaluate_result(res)
    push!(results, merge(params, Dict(pairs(metrics)), Dict(:seed => seed)))
end

# === Convert to DataFrame and Plot ===
df_results = DataFrame(results)
@show df_results

scatter(df_results.f1_range, df_results.f2_range;
    group=df_results.popsize,
    xlabel="f1 range", ylabel="f2 range",
    title="Diversity of Pareto Front vs Population Size",
    legend=:topright)

scatter(df_results.f1_range, df_results.f2_range;
    group=df_results.iterations,
    xlabel="f1 range", ylabel="f2 range",
    title="Diversity of Pareto Front vs Iterations Size",
    legend=:topright)

using Random

function pso(f, lb::Vector{T}, ub::Vector{T};
             swarm_size::Int=30, iterations::Int=100,
             w::T=0.7, c1::T=1.5, c2::T=1.5) where {T<:AbstractFloat}
    d = length(lb)
    # Initialize particles
    X = [lb .+ rand(d) .* (ub .- lb) for _ in 1:swarm_size]
    V = [zeros(d) for _ in 1:swarm_size]
    pbest = copy(X)
    pbest_val = [f(x) for x in X]
    # Global best
    gbest_idx = argmin(pbest_val)
    gbest, gbest_val = pbest[gbest_idx], pbest_val[gbest_idx]

    for iter in 1:iterations
        @threads for i in 1:swarm_size
            r1, r2 = rand(d), rand(d)
            # Velocity and position updates
            V[i] .= w .* V[i] .+
                    c1 .* r1 .* (pbest[i] .- X[i]) .+ # cognative
                    c2 .* r2 .* (gbest   .- X[i]) # social
            X[i] .= clamp.(X[i] .+ V[i], lb, ub) #Keeps the particles in the desired area
            @info A_particle = X[1]
            # Evaluate and update personal/global bests
            val = f(X[i])
            if val < pbest_val[i]
                pbest[i], pbest_val[i] = X[i], val
                if val < gbest_val
                    gbest, gbest_val = X[i], val
                end
            end
        end
        @info "PSO iteration" iteration=iter global_best=gbest_val
    end

    return gbest, gbest_val
end

# ── Example: minimize the 2D sphere function ──
f(x) = sum(x .^ 2)
lb = fill(-5.0, 2)
ub = fill( 5.0, 2)

best, best_val = pso(f, lb, ub;
                     swarm_size=50, iterations=200,
                     w=0.8, c1=1.7, c2=1.7)
@show best, best_val

n, d = 50, 3
form = @formula(y ~ 1 + x1 + x2 + x3 + x1*x2 + x2*x3 + x3*x1 + x1^2 + x2^2 + x3^2)
# your two-objective function
function objective(x)
    X = reshape(x, n, d)
    df = DataFrame(y = ones(n),
                   x1 = X[:,1], x2 = X[:,2], x3 = X[:,3])
    # catch any DomainError (Happens when D-effs are incredibly bad)
    f1 = try
        -calculate_d_optimality(df, form)
    catch e
        isa(e, DomainError) ? 1e6 : rethrow(e)
    end
    return f1
end

lb = fill(-1.0, 150)
ub = fill( 1.0, 150)

best, best_val = pso(objective, lb, ub;
                     swarm_size=100, iterations=2000,
                     w=0.8, c1=1.7, c2=1.7)
@show best, best_val


using PlotlyJS


# fully‐qualified trace + plot
trace = PlotlyJS.scatter3d(; x=reshape(best, 50, 3)[:,1], y=reshape(best, 50, 3)[:,2], z=reshape(best, 50, 3)[:,3], mode="markers")
plt   = PlotlyJS.Plot(trace)

# fully‐qualified save
PlotlyJS.savefig(plt, "plotly3d.html")
# on Linux/macOS you can then
run(`xdg-open plotly3d.html`)

using Random, LinearAlgebra, Base.Threads, Logging

# ——— Helper Functions ————————————————————————————————————————————————————

"""
    dominates(f1, f2)

Return true iff objective vector f1 dominates f2 (≤ in all, < in at least one).
"""
function dominates(f1::Vector{T}, f2::Vector{T}) where {T}
    all(f1 .<= f2) && any(f1 .< f2)
end

"""
    update_archive!(archX, archF, x, fx)

Insert (x,fx) into the archive of non‐dominated solutions, removing any
that it dominates and skipping insertion if it's dominated.
"""
function update_archive!(archX::Vector{Vector{Float64}},
                         archF::Vector{Vector{Float64}},
                         x::Vector{Float64}, fx::Vector{Float64})
    # 1) remove any archived points dominated by fx
    dominated = [i for i in eachindex(archF) if dominates(fx, archF[i])]
    for idx in reverse(dominated)
        deleteat!(archX, idx)
        deleteat!(archF, idx)
    end

    # 2) insert x if it isn't dominated by any remaining member
    if all(!dominates(archF[i], fx) for i in eachindex(archF))
        push!(archX, copy(x))
        push!(archF, copy(fx))
    end
    return
end

"""
    crowding_distance(vals)

Compute the crowding distance of each point in vals (list of M-vectors).
Returns a vector of distances (larger = more isolated).
"""
function crowding_distance(vals::Vector{Vector{Float64}})
    N = length(vals)
    M = length(vals[1])
    d = zeros(N)
    for m in 1:M
        idx = sortperm(vals, by = v->v[m])
        d[idx[1]] = Inf
        d[idx[end]] = Inf
        fmin, fmax = vals[idx[1]][m], vals[idx[end]][m]
        for k in 2:N-1
            d[idx[k]] += (vals[idx[k+1]][m] - vals[idx[k-1]][m]) / (fmax - fmin + eps())
        end
    end
    return d
end

"""
    select_leader(archX, archF)

Pick one archive member by binary tournament on crowding distance.
"""
function select_leader(archX::Vector{Vector{Float64}},
                       archF::Vector{Vector{Float64}})
    cd = crowding_distance(archF)
    i, j = rand(1:length(cd)), rand(1:length(cd))
    return cd[i] > cd[j] ? archX[i] : archX[j]
end


# ——— Threaded MOPSO ——————————————————————————————————————————————————————

"""
    mopso_threaded(f, lb, ub; swarm_size, iterations, w, c1, c2, max_archive)

Multi‐objective PSO with threads.  
- `f(x)` returns a Vector of objectives.  
- `lb, ub` ∈ ℝᵈ give bounds.  
Returns archive of (positions, objective vectors).
"""
function mopso_threaded(f, D; 
                        lowerbound::Float64=-1.0, 
                        upperbound::Float64=1.0,
                        swarm_size::Int=50, 
                        iterations::Int=200,
                        w::Float64=0.4, 
                        c1::Float64=1.5, 
                        c2::Float64=1.5,
                        max_archive::Int=100)

    lb = fill(lowerbound, D)
    ub = fill(upperbound, D)

    # initialize swarm
    X  = [lb .+ rand(D).*(ub .- lb) for _ in 1:swarm_size]
    V  = [zeros(D) for _ in 1:swarm_size]
    pX = deepcopy(X)
    pF = [f(x) for x in X]

    # initial archive
    archX, archF = Vector{Vector{Float64}}(), Vector{Vector{Float64}}()
    for i in 1:swarm_size
        update_archive!(archX, archF, X[i], pF[i])
    end

    # history containers
    X_hist = Vector{Vector{Vector{Float64}}}(undef, iterations)
    F_hist = Vector{Vector{Vector{Float64}}}(undef, iterations)

    curF    = Vector{Vector{Float64}}(undef, swarm_size)
    leaders = Vector{Vector{Float64}}(undef, swarm_size)

    for iter in 1:iterations
        # 1) choose leaders
        for i in 1:swarm_size
            leaders[i] = select_leader(archX, archF)
        end

        # 2) update & eval in parallel
        @threads for i in 1:swarm_size
            r1, r2 = rand(D), rand(D)
            V[i] .= w .* V[i] .+
                    c1 .* r1 .* (pX[i] .- X[i]) .+
                    c2 .* r2 .* (leaders[i] .- X[i])
            X[i] .= clamp.(X[i] .+ V[i], lb, ub)
            curF[i] = f(X[i])
            # personal‐best update…
            if dominates(curF[i], pF[i])
                pX[i], pF[i] = copy(X[i]), curF[i]
            elseif !dominates(pF[i], curF[i]) && rand() < 0.5
                pX[i], pF[i] = copy(X[i]), curF[i]
            end
        end

        # 3) update & prune archive…
        for i in 1:swarm_size
            update_archive!(archX, archF, X[i], curF[i])
        end
        if length(archX) > max_archive
            cd = crowding_distance(archF)
            while length(archX) > max_archive
                idx = argmin(cd)
                deleteat!(archX, idx); deleteat!(archF, idx); deleteat!(cd, idx)
            end
        end

        # 4) record swarm state
        X_hist[iter] = deepcopy(X)
        F_hist[iter] = deepcopy(curF)

        @info "MOPSO iter" iteration=iter archive_size=length(archX)
    end

    return archX, archF, X_hist, F_hist
end

# ——— Example Usage —————————————————————————————————————————————————————
# Two-objective demo: sphere@origin & sphere@[2,1]
f_demo(x) = [sum(x.^2), sum((x .- [2.0,1.0]).^2)]


archive_pos, archive_vals, X_hist, F_hist = mopso_threaded(f_demo, 2;
                                          swarm_size=60, iterations=300,
                                          w=0.5, c1=1.7, c2=1.7,
                                          max_archive=80)

F_hist

@show length(archive_pos), length(archive_vals)
for (x, fv) in zip(archive_pos, archive_vals)
    @show x, fv
end

# problem dimensions
n, d = 50, 3
form = @formula(y ~ 1 + x1 + x2 + x3 + x1*x2 + x2*x3 + x3*x1 + x1^2 + x2^2 + x3^2)
# your two-objective function
function objective(x)
    X = reshape(x, n, d)
    df = DataFrame(y = ones(n),
                   x1 = X[:,1], x2 = X[:,2], x3 = X[:,3])
    # catch any DomainError
    f1 = try
        -calculate_d_optimality(df, form)
    catch e
        isa(e, DomainError) ? 1e6 : rethrow(e)
    end
    f2 = max_min_distance(df)
    return [f1, f2]
end
lb = fill(-1.0, n*d)
ub = fill( 1.0, n*d)

archive_pos, archive_vals = mopso_threaded(objective, lb, ub;
                                          swarm_size=100, iterations=1000,
                                          w=0.5, c1=1.7, c2=1.7,
                                          max_archive=80)

@show length(archive_pos), length(archive_vals)
for (x, fv) in zip(archive_pos, archive_vals)
    @show x, fv
end

using Plots

# extract objectives
f1 = [v[1] for v in archive_vals]
f2 = [v[2] for v in archive_vals]
minimum(f1)
# simple scatter of the front
scatter(f1, f2;
    xlabel="f₁(x)",
    ylabel="f₂(x)",
    title="Approximate Pareto Front",
    marker=:circle,
    legend=false)

using HDF5, JSON, Dates

"""
    save_mopso_run(
        X_hist::Vector{Vector{Vector{Float64}}},
        F_hist::Vector{Vector{Vector{Float64}}},
        archX::Vector{Vector{Float64}},
        archF::Vector{Vector{Float64}};
        algorithm::String = "mopso_threaded",
        swarm_size::Int,
        iterations::Int,
        lowerbound::Float64 = -1.0,
        upperbound::Float64 = 1.0,
        w::Float64 = 0.4,
        c1::Float64 = 1.5,
        c2::Float64 = 1.5,
        max_archive::Int = 100,
        formula::AbstractString
    ) -> String

Save a complete MOPSO run to an HDF5 file with an auto-generated filename
that encodes key parameters and timestamp. Writes:

- `/X_hist` (D × N × T): full swarm positions history
- `/F_hist` (M × N × T): full swarm objective history
- `/archive_X` (D × A): final Pareto front decision-vectors
- `/archive_F` (M × A): final Pareto front objective-vectors
- root attribute `params_json`: JSON metadata

# Arguments
- `X_hist[t][i]`: particle i’s D-vector at iteration t
- `F_hist[t][i]`: particle i’s M-vector at iteration t
- `archX[j]`: j-th Pareto-optimal decision-vector
- `archF[j]`: j-th Pareto-optimal objective-vector
- Keywords: algorithm, swarm_size, iterations, lower/upper bounds,
  PSO coefficients (`w`, `c1`, `c2`), `max_archive`, and `formula`.

# Returns
- filename: name of the saved HDF5 file
"""
function save_mopso_run(
    X_hist::Vector{Vector{Vector{Float64}}},
    F_hist::Vector{Vector{Vector{Float64}}},
    archX::Vector{Vector{Float64}},
    archF::Vector{Vector{Float64}};
    algorithm::String = "mopso_threaded",
    swarm_size::Int,
    iterations::Int,
    lowerbound::Float64 = -1.0,
    upperbound::Float64 = 1.0,
    w::Float64 = 0.4,
    c1::Float64 = 1.5,
    c2::Float64 = 1.5,
    max_archive::Int = 100,
    formula::AbstractString
) :: String
    # dimensions
    T = length(X_hist)
    N = length(X_hist[1])
    D = length(X_hist[1][1])
    M = length(F_hist[1][1])
    A = length(archX)

    # timestamp & filename
    now = Dates.now()
    ts = Dates.format(now, "yyyy-mm-dd_HHMMSS")
    filename = "$(algorithm)_N$(swarm_size)_iter$(iterations)_D$(D)_M$(M)_A$(A)_$(ts).h5"

    # pack histories
    X_arr = Array{Float64}(undef, D, N, T)
    F_arr = Array{Float64}(undef, M, N, T)
    for t in 1:T, i in 1:N
        X_arr[:, i, t] = X_hist[t][i]
        F_arr[:, i, t] = F_hist[t][i]
    end
    # pack archive
    archX_arr = Array{Float64}(undef, D, A)
    archF_arr = Array{Float64}(undef, M, A)
    for j in 1:A
        archX_arr[:, j] = archX[j]
        archF_arr[:, j] = archF[j]
    end

    # metadata
    params = Dict(
        "algorithm"      => algorithm,
        "swarm_size"     => swarm_size,
        "iterations"     => iterations,
        "problem_dim"    => D,
        "num_objectives" => M,
        "archive_size"   => A,
        "lowerbound"     => lowerbound,
        "upperbound"     => upperbound,
        "w"              => w,
        "c1"             => c1,
        "c2"             => c2,
        "max_archive"    => max_archive,
        "formula"        => formula,
        "timestamp"      => string(now)
    )

    # write HDF5
    h5open(filename, "w") do fid
        write(fid, "X_hist", X_arr)
        write(fid, "F_hist", F_arr)
        write(fid, "archive_X", archX_arr)
        write(fid, "archive_F", archF_arr)
        # set attribute via qualified call
        HDF5.attributes(fid)["params_json"] = JSON.json(params)
    end

    return filename
end


# example
# fn = save_mopso_run(X_hist, F_hist, archX, archF;
#                     swarm_size=60, iterations=300,
#                     lowerbound=-1.0, upperbound=1.0,
#                     w=0.5, c1=1.7, c2=1.7,
#                     max_archive=80,
#                     formula=string(form))
# @info "Saved to" fn

# ── Example Usage ──
 filename = save_mopso_run(
     X_hist, F_hist,
     archive_pos, archive_vals;
     swarm_size=60,
     iterations=300,
     lowerbound=-1.0,
     upperbound=1.0,
     w=0.5,
     c1=1.7,
     c2=1.7,
     max_archive=80,
     formula = string(form)
)
# @info "Saved MOPSO run to" filename

using HDF5, JSON, Dates

"""
    save_mopso_run(
        X_hist::Vector{Vector{Vector{Float64}}},
        F_hist::Vector{Vector{Vector{Float64}}},
        archX::Vector{Vector{Float64}},
        archF::Vector{Vector{Float64}};
        algorithm::String = "mopso_threaded",
        swarm_size::Int,
        iterations::Int,
        lowerbound::Float64 = -1.0,
        upperbound::Float64 = 1.0,
        w::Float64 = 0.4,
        c1::Float64 = 1.5,
        c2::Float64 = 1.5,
        max_archive::Int = 100,
        formula::AbstractString
    ) -> String

Save a complete MOPSO run to an HDF5 file with an auto-generated filename
that encodes key parameters and timestamp. Writes:

- `/X_hist` (D × N × T): full swarm positions history
- `/F_hist` (M × N × T): full swarm objective history
- `/archive_X` (D × A): final Pareto front decision-vectors
- `/archive_F` (M × A): final Pareto front objective-vectors
- root attribute `params_json`: JSON metadata

# Arguments
- `X_hist[t][i]`: particle i’s D-vector at iteration t
- `F_hist[t][i]`: particle i’s M-vector at iteration t
- `archX[j]`: j-th Pareto-optimal decision-vector
- `archF[j]`: j-th Pareto-optimal objective-vector
- Keywords: algorithm, swarm_size, iterations, lower/upper bounds,
  PSO coefficients (`w`, `c1`, `c2`), `max_archive`, and `formula`.

# Returns
- filename: name of the saved HDF5 file
"""
function save_mopso_run(
    X_hist::Vector{Vector{Vector{Float64}}},
    F_hist::Vector{Vector{Vector{Float64}}},
    archX::Vector{Vector{Float64}},
    archF::Vector{Vector{Float64}};
    algorithm::String = "mopso_threaded",
    swarm_size::Int,
    iterations::Int,
    lowerbound::Float64 = -1.0,
    upperbound::Float64 = 1.0,
    w::Float64 = 0.4,
    c1::Float64 = 1.5,
    c2::Float64 = 1.5,
    max_archive::Int = 100,
    formula::AbstractString
) :: String
    # dimensions
    T = length(X_hist)
    N = length(X_hist[1])
    D = length(X_hist[1][1])
    M = length(F_hist[1][1])
    A = length(archX)

    # timestamp & filename
    now = Dates.now()
    ts = Dates.format(now, "yyyy-mm-dd_HHMMSS")
    filename = "$(algorithm)_N$(swarm_size)_iter$(iterations)_D$(D)_M$(M)_A$(A)_$(ts).h5"

    # pack histories
    X_arr = Array{Float64}(undef, D, N, T)
    F_arr = Array{Float64}(undef, M, N, T)
    for t in 1:T, i in 1:N
        X_arr[:, i, t] = X_hist[t][i]
        F_arr[:, i, t] = F_hist[t][i]
    end
    # pack archive
    archX_arr = Array{Float64}(undef, D, A)
    archF_arr = Array{Float64}(undef, M, A)
    for j in 1:A
        archX_arr[:, j] = archX[j]
        archF_arr[:, j] = archF[j]
    end

    # metadata
    params = Dict(
        "algorithm"      => algorithm,
        "swarm_size"     => swarm_size,
        "iterations"     => iterations,
        "problem_dim"    => D,
        "num_objectives" => M,
        "archive_size"   => A,
        "lowerbound"     => lowerbound,
        "upperbound"     => upperbound,
        "w"              => w,
        "c1"             => c1,
        "c2"             => c2,
        "max_archive"    => max_archive,
        "formula"        => formula,
        "timestamp"      => string(now)
    )

    # write HDF5
    h5open(filename, "w") do fid
        write(fid, "X_hist", X_arr)
        write(fid, "F_hist", F_arr)
        write(fid, "archive_X", archX_arr)
        write(fid, "archive_F", archF_arr)
        HDF5.attributes(fid)["params_json"] = JSON.json(params)
    end

    return filename
end

"""
    run_mopso_experiments(experiments::Vector{<:NamedTuple}) -> Vector{NamedTuple}

Run multiple MOPSO experiments with varying settings, save each run, and
return a summary of filenames.

# Arguments
- `experiments`: a vector of NamedTuples, each containing:
  - `:f`            => objective function `f(x)::Vector{Float64}`
  - `:D`            => dimensionality of decision vector
  - `:lowerbound`, `:upperbound` => search bounds
  - `:swarm_size`, `:iterations`, `:w`, `:c1`, `:c2`, `:max_archive`
  - `:formula`      => formula string for metadata
  - `:algorithm`    => (optional) algorithm name

# Returns
- A vector of NamedTuples with fields:
  - `:params` => the NamedTuple of experiment settings
  - `:filename` => the HDF5 file saved for that run
"""
function run_mopso_experiments(experiments::Vector{<:NamedTuple})
    results = NamedTuple[]
    for exp in experiments
        # Extract settings
        f         = exp.f
        D         = exp.D
        lb        = get(exp, :lowerbound, -1.0)
        ub        = get(exp, :upperbound,  1.0)
        swarm    = exp.swarm_size
        iters    = exp.iterations
        w         = exp.w
        c1        = exp.c1
        c2        = exp.c2
        max_arc   = exp.max_archive
        formula   = exp.formula
        algo      = get(exp, :algorithm, "mopso_threaded")

        # Run MOPSO
        archX, archF, X_hist, F_hist =
            mopso_threaded(f, D;
                lowerbound=lb, upperbound=ub,
                swarm_size=swarm, iterations=iters,
                w=w, c1=c1, c2=c2, max_archive=max_arc
            )

        # Save results
        filename = save_mopso_run(
            X_hist, F_hist, archX, archF;
            algorithm=algo, swarm_size=swarm,
            iterations=iters, lowerbound=lb,
            upperbound=ub, w=w, c1=c1, c2=c2,
            max_archive=max_arc, formula=formula
        )

        push!(results, (params=exp, filename=filename))
    end
    return results
end

# ── Standalone Usage Example ──
# This script demonstrates how to define objectives, set up experiments,
# run multiple MOPSO runs, and report saved file names.

# 1) Define two simple bi-objective functions
#    - obj1: Sphere at origin and shifted sphere
#    - obj2: Sphere at x=1 and inverted sphere
function obj1(x::Vector{Float64})::Vector{Float64}
    # f1: distance squared to origin, f2: distance squared to [1,1,...]
    f1 = sum(x .^ 2)
    f2 = sum((x .- 1.0) .^ 2)
    return [f1, f2]
end

function obj2(x::Vector{Float64})::Vector{Float64}
    # f1: distance squared to [0.5,...], f2: distance squared to [-0.5,...]
    f1 = sum((x .- 0.5) .^ 2)
    f2 = sum((x .+ 0.5) .^ 2)
    return [f1, f2]
end

# 2) Construct a list of experiments
exps = [
    (
        f = obj1,
        D = 10,
        lowerbound = -1.0,
        upperbound = 1.0,
        swarm_size = 50,
        iterations = 200,
        w = 0.6,
        c1 = 1.4,
        c2 = 1.6,
        max_archive = 80,
        formula = "y ~ 1 + x1 + x2 + x3",
        algorithm = "mopso_threaded"
    ),
    (
        f = obj2,
        D = 5,
        lowerbound = -2.0,
        upperbound = 2.0,
        swarm_size = 30,
        iterations = 150,
        w = 0.5,
        c1 = 1.2,
        c2 = 1.8,
        max_archive = 60,
        formula = "y ~ x1^2 + x2",
        algorithm = "mopso_threaded"
    )
]

# 3) Run all experiments and save results
@info "Starting MOPSO experiments with $(length(exps)) configurations"
results = run_mopso_experiments(exps)

# 4) Report outcomes
for (i, res) in enumerate(results)
    params = res.params
    @info "Experiment $i" params.algorithm * 
          " (D=$(params.D), N=$(params.swarm_size), iter=$(params.iterations)) " *
          "saved to" res.filename
end

using HDF5, JSON, Dates
using Base.Iterators: product

"""
    generate_experiments(opts::NamedTuple) -> Vector{NamedTuple}

Generate a full-factorial collection of experiment settings given a NamedTuple
of parameter vectors. Each field in `opts` should be a vector of possible
values for that parameter.

# Arguments
- `opts`: NamedTuple whose fields are parameter names and values are vectors
  of possible settings, for example:

  # example
  opts = (
      D = [5, 10],
      swarm_size = [30, 50],
      iterations = [100, 200],
      w = [0.4, 0.6],
      c1 = [1.5, 2.0],
      c2 = [1.5, 2.0],
      max_archive = [50, 100],
      lowerbound = [-1.0],
      upperbound = [1.0],
      formula = ["y~x1+x2", "y~x1^2+x2"],
      f = [obj1, obj2],
      algorithm = ["mopso_threaded"]
  )

# Returns
- A vector of NamedTuples, each combining one element from each input vector
"""
function generate_experiments(opts::NamedTuple)
    ks = fieldnames(typeof(opts))                    # parameter names
    vs = [opts[k] for k in ks]                        # parameter value vectors

    # NamedTuple type with keys ks (types default to Any)
    NTType = NamedTuple{ks}                           
    exps = Vector{NTType}()                           

    # Cartesian product over all parameter value vectors
    for combo in product(vs...)
        # combo is a tuple of chosen values; build NamedTuple
        push!(exps, NTType(combo))
    end
    return exps
end

# # ── Standalone Usage Example ──
# # 1) Define simple objectives
# # (re-use obj1, obj2 defined earlier)

# # 2) Define parameter space
# opts = (
#     f = [obj1, obj2],
#     D = [5, 10],
#     lowerbound = [-1.0],
#     upperbound = [1.0],
#     swarm_size = [30, 50],
#     iterations = [100, 200],
#     w = [0.4, 0.6],
#     c1 = [1.5, 2.0],
#     c2 = [1.5, 2.0],
#     max_archive = [50, 100],
#     formula = ["y~x1+x2", "y~x1^2+x2"],
#     algorithm = ["mopso_threaded"]
# )

# # 3) Generate experiments via full factorial
# exps = generate_experiments(opts)
# @info "Generated $(length(exps)) experiments"

# # 4) Run and save
# results = run_mopso_experiments(exps)
# # Logs appear for each experiment

