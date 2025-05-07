using Random, LinearAlgebra, Base.Threads, Logging
using HDF5, JSON, Dates
using Base.Iterators: product
using PlotlyJS
using Plots

# ——— Single‐Objective PSO ————————————————————————————————————————————

"""
    pso(f, lb::Vector{T}, ub::Vector{T};
        swarm_size::Int=30, iterations::Int=100,
        w::T=0.7, c1::T=1.5, c2::T=1.5) where {T<:AbstractFloat}

Minimize scalar f:ℝᵈ→ℝ with PSO.
Returns (gbest_position, gbest_value).
"""
function pso(f, lb::Vector{T}, ub::Vector{T};
             swarm_size::Int=30, iterations::Int=100,
             w::T=0.7, c1::T=1.5, c2::T=1.5) where {T<:AbstractFloat}

    d = length(lb)
    # Initialize
    X = [lb .+ rand(d).*(ub .- lb) for _ in 1:swarm_size]
    V = [zeros(d) for _ in 1:swarm_size]
    pbest = deepcopy(X)
    pval  = [f(x) for x in X]
    # Global best
    gidx = argmin(pval)
    gbest, gbest_val = pbest[gidx], pval[gidx]

    # Iterations
    for iter in 1:iterations
        @threads for i in 1:swarm_size
            r1, r2 = rand(d), rand(d)
            V[i] .= w .* V[i] .+
                    c1 .* r1 .* (pbest[i] .- X[i]) .+
                    c2 .* r2 .* (gbest   .- X[i])
            X[i] .= clamp.(X[i] .+ V[i], lb, ub)
            val = f(X[i])
            if val < pval[i]
                pbest[i], pval[i] = X[i], val
                if val < gbest_val
                    gbest, gbest_val = X[i], val
                end
            end
        end
        @info "PSO iter=$iter gbest_val=$gbest_val"
    end

    return gbest, gbest_val
end


# ─── Example: minimize 2D sphere ────────────────────────────────────────────────
f_sphere(x) = sum(x .^ 2)
lb2 = fill(-5.0, 2); ub2 = fill(5.0, 2)
best, best_val = pso(f_sphere, lb2, ub2; swarm_size=50, iterations=200)
@show best, best_val


# ——— Helper Functions for MOPSO —————————————————————————————————————————

"""
    dominates(f1, f2)
Return true if vector f1 dominates f2 (≤ in all, < in at least one).
"""
function dominates(f1::Vector{T}, f2::Vector{T}) where {T}
    all(f1 .<= f2) && any(f1 .< f2)
end

"""
    update_archive!(archX, archF, x, fx)
Maintain non‐dominated archive: remove any archF dominated by fx, and insert (x,fx) if not itself dominated.
"""
function update_archive!(archX::Vector{Vector{Float64}},
                         archF::Vector{Vector{Float64}},
                         x::Vector{Float64}, fx::Vector{Float64})
    # remove dominated
    to_rm = [i for i in eachindex(archF) if dominates(fx, archF[i])]
    for i in reverse(to_rm)
        deleteat!(archX, i); deleteat!(archF, i)
    end
    # insert if non‐dominated
    if all(!dominates(archF[i], fx) for i in eachindex(archF))
        push!(archX, copy(x)); push!(archF, copy(fx))
    end
end

"""
    crowding_distance(vals)
Compute crowding distance for each M‐vector in vals.
"""
function crowding_distance(vals::Vector{Vector{Float64}})
    N = length(vals); M = length(vals[1])
    d = zeros(N)
    for m in 1:M
        idx = sortperm(vals, by=v->v[m])
        d[idx[1]] = Inf; d[idx[end]] = Inf
        fmin, fmax = vals[idx[1]][m], vals[idx[end]][m]
        for k in 2:N-1
            d[idx[k]] += (vals[idx[k+1]][m] - vals[idx[k-1]][m])/(fmax-fmin+eps())
        end
    end
    return d
end

"""
    select_leader(archX, archF)
Binary‐tournament selection on crowding distance to pick a leader.
"""
function select_leader(archX::Vector{Vector{Float64}},
                       archF::Vector{Vector{Float64}})
    cd = crowding_distance(archF)
    i, j = rand(1:length(cd)), rand(1:length(cd))
    return cd[i] > cd[j] ? archX[i] : archX[j]
end


# ——— Threaded Multi‐Objective PSO —————————————————————————————————————

"""
    mopso_threaded(f, D;
        lowerbound=-1.0, upperbound=1.0,
        swarm_size=50, iterations=200,
        w=0.4, c1=1.5, c2=1.5,
        max_archive=100)
Returns (archX, archF, X_hist, F_hist).
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

    lb = fill(lowerbound, D); ub = fill(upperbound, D)
    # init particles
    X = [lb .+ rand(D).*(ub .- lb) for _ in 1:swarm_size]
    V = [zeros(D) for _ in 1:swarm_size]
    pX = deepcopy(X); pF = [f(x) for x in X]
    archX, archF = Vector{Vector{Float64}}(), Vector{Vector{Float64}}()
    for i in 1:swarm_size
        update_archive!(archX, archF, X[i], pF[i])
    end

    X_hist = Vector{typeof(X)}(undef, iterations)
    F_hist = Vector{typeof(pF)}(undef, iterations)

    for iter in 1:iterations
        # choose leaders
        leaders = [select_leader(archX, archF) for _ in 1:swarm_size]

        # move & evaluate
        @threads for i in 1:swarm_size
            r1, r2 = rand(D), rand(D)
            V[i] .= w*V[i] .+ c1*r1.*(pX[i].-X[i]) .+ c2*r2.*(leaders[i].-X[i])
            X[i] .= clamp.(X[i].+V[i], lb, ub)
            curF = f(X[i])
            # personal best update
            if dominates(curF, pF[i])
                pX[i], pF[i] = copy(X[i]), curF
            elseif !dominates(pF[i], curF) && rand()<0.5
                pX[i], pF[i] = copy(X[i]), curF
            end
        end

        # update archive
        for i in 1:swarm_size
            update_archive!(archX, archF, X[i], pF[i])
        end
        # prune if too large
        if length(archX) > max_archive
            cd = crowding_distance(archF)
            while length(archX) > max_archive
                idx = argmin(cd)
                deleteat!(archX, idx); deleteat!(archF, idx); deleteat!(cd, idx)
            end
        end

        X_hist[iter] = deepcopy(X)
        F_hist[iter] = deepcopy(pF)
        @info "MOPSO iter=$iter archive_size=$(length(archX))"
    end

    return archX, archF, X_hist, F_hist
end

# ─── Two‐Objective Demo ───────────────────────────────────────────────────────
f_demo(x) = [sum(x .^ 2), sum((x .- [2.0,1.0]).^2)]
archive_pos, archive_vals, X_hist, F_hist =
    mopso_threaded(f_demo, 2; swarm_size=60, iterations=300, w=0.5, c1=1.7, c2=1.7, max_archive=80)

@show length(archive_pos), length(archive_vals)
for (x,fv) in zip(archive_pos, archive_vals)
    @show x, fv
end

# scatter Pareto front
f1 = [v[1] for v in archive_vals]; f2 = [v[2] for v in archive_vals]
scatter(f1, f2; xlabel="f₁", ylabel="f₂", title="Approximate Pareto Front", legend=false)


# ——— Save & Experiment Utilities —————————————————————————————————————

"""
    save_mopso_run(X_hist, F_hist, archX, archF; kwargs...) -> filename
Writes an HDF5 file encoding the full run and metadata.
"""
function save_mopso_run(
    X_hist::Vector{Vector{Vector{Float64}}},
    F_hist::Vector{Vector{Vector{Float64}}},
    archX::Vector{Vector{Float64}},
    archF::Vector{Vector{Float64}};
    algorithm::String="mopso_threaded",
    swarm_size::Int,
    iterations::Int,
    lowerbound::Float64=-1.0,
    upperbound::Float64=1.0,
    w::Float64=0.4,
    c1::Float64=1.5,
    c2::Float64=1.5,
    max_archive::Int=100,
    formula::AbstractString=""
)::String

    # meta + arrays
    now = Dates.now(); ts = Dates.format(now,"yyyy-mm-dd_HHMMSS")
    D = length(X_hist[1][1]); N = length(X_hist[1]); T = length(X_hist)
    M = length(F_hist[1][1]); A = length(archX)
    fname = "$(algorithm)_N$(swarm_size)_iter$(iterations)_D$(D)_M$(M)_A$(A)_$(ts).h5"

    Xarr = Array{Float64}(undef,D,N,T)
    Farr = Array{Float64}(undef,M,N,T)
    for t in 1:T, i in 1:N
        Xarr[:,i,t] = X_hist[t][i]
        Farr[:,i,t] = F_hist[t][i]
    end
    archX_arr = hcat(archX...)     # D×A
    archF_arr = hcat(archF...)     # M×A

    h5open(fname,"w") do fid
        write(fid,"X_hist",Xarr); write(fid,"F_hist",Farr)
        write(fid,"archive_X",archX_arr); write(fid,"archive_F",archF_arr)
        HDF5.attributes(fid)["params_json"] =
            JSON.json(Dict(
                "algorithm"=>algorithm, "swarm_size"=>swarm_size,
                "iterations"=>iterations, "D"=>D, "M"=>M, "A"=>A,
                "bounds"=>[lowerbound,upperbound], "w"=>w, "c1"=>c1,
                "c2"=>c2, "max_archive"=>max_archive, "formula"=>formula,
                "timestamp"=>string(now)
            ))
    end
    return fname
end

# Example save
filename = save_mopso_run(X_hist, F_hist, archive_pos, archive_vals;
    swarm_size=60, iterations=300, formula=string("y~x1+x2"), max_archive=80)
@info "Saved MOPSO run to $filename"


"""
    run_mopso_experiments(experiments::Vector{<:NamedTuple})
Runs each experiment (with fields f,D,swarm_size,iterations,w,c1,c2,max_archive,formula)
and saves its output. Returns Vector of NamedTuples (params, filename).
"""
function run_mopso_experiments(experiments::Vector{<:NamedTuple})
    results = NamedTuple[]
    for exp in experiments
        archX, archF, Xh, Fh = mopso_threaded(
            exp.f, exp.D;
            lowerbound=get(exp,:lowerbound,-1.0),
            upperbound=get(exp,:upperbound,1.0),
            swarm_size=exp.swarm_size,
            iterations=exp.iterations,
            w=exp.w, c1=exp.c1, c2=exp.c2,
            max_archive=exp.max_archive
        )
        fn = save_mopso_run(
            Xh, Fh, archX, archF;
            algorithm=get(exp,:algorithm,"mopso_threaded"),
            swarm_size=exp.swarm_size,
            iterations=exp.iterations,
            lowerbound=get(exp,:lowerbound,-1.0),
            upperbound=get(exp,:upperbound,1.0),
            w=exp.w, c1=exp.c1, c2=exp.c2,
            max_archive=exp.max_archive,
            formula=exp.formula
        )
        push!(results, (params=exp, filename=fn))
    end
    return results
end


"""
    generate_experiments(opts::NamedTuple) -> Vector{NamedTuple}
Full‐factorial grid over each field’s vector in `opts`.
"""
function generate_experiments(opts::NamedTuple)
    ks = fieldnames(typeof(opts))
    vs = [opts[k] for k in ks]
    NT  = NamedTuple{ks}
    exps = NT[]
    for combo in product(vs...)
        push!(exps, NT(combo))
    end
    return exps
end

# ─── Example: generate & run ──────────────────────────────────────────────────
opts = (
    f = [f_demo],
    D = [2],
    lowerbound=[-1.0], upperbound=[1.0],
    swarm_size=[50], iterations=[200],
    w=[0.6], c1=[1.4], c2=[1.6],
    max_archive=[80],
    formula=["y~1+x1+x2"],
    algorithm=["mopso_threaded"]
)
exps = generate_experiments(opts)
results = run_mopso_experiments(exps)
@info "Completed $(length(results)) experiments"
