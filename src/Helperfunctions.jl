using Pkg
using StatsKit
using LinearAlgebra
using DataFrames
using Base.Threads

# — Design → Model Matrix Conversions

# 1a) From a DataFrame + formula
function modelmatrix_df(design_df::DataFrame, formula)
    sch = schema(formula, design_df)
    resolved = apply_schema(formula, sch)
    parts = modelcols(resolved, design_df)
    Xmat = reduce(hcat, parts)[:, 2:end]                # drop intercept
    names = coefnames(resolved)[2]                      # predictor names
    return DataFrame(Xmat, Symbol.(collect(names)))
end

# 1b) From a raw matrix + formula (assumes y=1 column prepended)
function modelmatrix(designmatrix::Matrix, formula)
    df = DataFrame(designmatrix, :auto)
    df.y = ones(nrow(df))
    sch = schema(formula, df)
    resolved = apply_schema(formula, sch)
    return modelcols(resolved.rhs, df)
end


# Todo: Do I need this?
function creategrid(n::Integer, d::Integer)

    @assert d >= 1 ("d (number of dimensions)")
    @assert n >= 2 ("n (number of points)")

    r = range(-1, 1, length = n)

    iter = Iterators.product((r for _ in 1:d)...)

    return vec([collect(i) for i in iter])
end
