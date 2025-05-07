using ExperimentalDesign
using LinearAlgebra
using Statistics
using StatsModels
using DataFrames
using StatsBase
using IterTools

module AdvExpDes




# Move to helper functions?
# Can be improved?
function creategrid(n::Integer, d::Integer)

    @assert d >= 1 ("d (number of dimensions)")
    @assert n >= 2 ("n (number of points)")

    r = range(-1, 1, length = n)

    iter = Iterators.product((r for _ in 1:d)...)

    return vec([collect(i) for i in iter])
end
creategrid(10, 3)

r = range(-1, 1, length = 10)
Iterators.product((r for _ in 1:3)...)

pred_var(x, inv_info) = x' * inv_info * x
#pred_var.(creategrid(10, 2), [1 0; 0 1])
#pred_var([[0 1], [1 0], [1 0]], [1 0; 0 1])

end # module AdvExpDes
