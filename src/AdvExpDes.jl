using ExperimentalDesign
module AdvExpDes
function generate_glp_set(n::Int, s::Int, h::Vector{Int})
    """
    Generate a Good Lattice Point (GLP) set.

    Parameters:
        n (Int): Number of points in the set.
        s (Int): Dimension of the design space.
        h (Vector{Int}): Generating vector satisfying gcd(n, h[i]) == 1 for all i.

    Returns:
        Matrix{Float64}: A matrix where each row corresponds to a GLP point.
    """
    @assert length(h) == s "Length of the generating vector h must equal the dimension s."
    @assert all(gcd(n, hi) == 1 for hi in h) "Each element of h must be coprime with n."

    # Initialize the GLP set matrix
    glp_set = zeros(Float64, n, s)

    # Generate the points
    for k in 1:n
        for i in 1:s
            qki = (k * h[i]) % n
            glp_set[k, i] = (2 * qki - 1) / (2n)
        end
    end

    return glp_set
end


using StatsBase  # For gcd
using IterTools

# Function to calculate discrepancy (simple placeholder, replace with actual metric)
function calculate_discrepancy(design::Matrix{Float64})
    return maximum(abs.(design .- 0.5))  # Example metric
end

# Generate all coprime vectors
function generate_coprime_vectors(n::Int, s::Int)
    vectors = []
    # Use IterTools.product with correct splatting
    for h in IterTools.product((1:n-1 for _ in 1:s-1)...)
        h = [1; collect(h)...]  # Always include 1 as the first element
        if all(gcd(n, hi) == 1 for hi in h)
            push!(vectors, h)
        end
    end
    return vectors
end


# Find the best GLP design
function best_glp_design(n::Int, s::Int)
    best_discrepancy = Inf
    best_design = nothing
    best_vector = nothing

    for h in generate_coprime_vectors(n, s)
        design = generate_glp_set(n, s, h)
        discrepancy = calculate_discrepancy(design)
        
        if discrepancy < best_discrepancy
            best_discrepancy = discrepancy
            best_design = design
            best_vector = h
        end
    end
    
    return best_design, best_vector, best_discrepancy
end

# TODO: Verify with Fangwang Example
# Example usage
n = 7  # Number of points
s = 3  # Dimension of the design space

best_design, best_vector, best_discrepancy = best_glp_design(n, s)
println("Best Vector: $best_vector")
println("Best Discrepancy: $best_discrepancy")
println("Best Design:")
best_design


using LinearAlgebra
using Statistics
using StatsModels
using DataFrames

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

# Two factor design from Small Exact Response Surface Designs (jpss03)
# D = 42.3123 A = 23.7678 G = 53.6542 I = 0.203900

design6_1 = [
  1 1
  -1 1
  -1 -1
  1 -.394449
  .394449 -1
  -.131483 .131483
]

design6_1_df = DataFrame(design6_1, :auto)

design6_1_modmat = modelmatrix(@formula(y ~ 1 + x1 + x2 + x1*x2 + 
  x1^2 + x2^2), design6_1_df)


function calc_optim(model_matrix; alpha = ["d" "a" "g"])
    # The number of observations (rows) in the model matrix
    n = size(model_matrix, 2)
    # The number of terms (cols) in the model matrix
    p = size(model_matrix, 1)
    info = model_matrix' * model_matrix

  # Initialize output
  out = Vector{Vector{Any}}()
  if "d" in alpha
    # Calculate the of the transpose of the model matrix multiplied by itself

    # Calculate the D-efficiency
    d_optimality = 100 * (det(info) ^ (1/p)) / n
    
    push!(out,  ["D Optimality", d_optimality])
  end

  inv_info = inv(info)

  if "a" in alpha 
  a_optimality = 100 * p  / tr(n * inv_info)
    push!(out, ["A Optimality", a_optimality])
  end
  if "g" in alpha 
  x_array = 
      collect(Iterators.product(1:5, 1:5, 1:5))
  pred_var_array = 
  x_spv = Vector{Float64}()
    for i in 1:n
    xi_spv = (n * model_matrix[i,:]' * inv_info * model_matrix[i,:])
    push!(x_spv, xi_spv)
    end
    print(x_spv)
    g_optimality = 100 * p  / maximum(x_spv)
    push!(out, ["G Optimality", g_optimality])
  end
  #if "i" in alpha 
  #  i_optimality = mean(spv_array)
  #  push!(out, ["I Optimality", i_optimality])
  #end

  return(out)
end

calc_optim(design6_1_modmat; alpha = ["a" "d" "g" "i"])


function calc_optim_d(model_matrix)
    # The number of observations (rows) in the model matrix
    n = size(model_matrix, 2)
    # The number of terms (cols) in the model matrix
    p = size(model_matrix, 1)
    info = model_matrix' * model_matrix
    # Calculate the D-efficiency
    return(100 * (det(info) ^ (1/p)) / n)
end

function calc_optim_a(model_matrix; alpha = ["d" "a" "g"])
    # The number of observations (rows) in the model matrix
    n = size(model_matrix, 2)
    # The number of terms (cols) in the model matrix
    p = size(model_matrix, 1)
    inv_info = inv(model_matrix' * model_matrix)
    return(100 * p  / tr(n * inv_info))
end

function calc_optim_g(model_matrix; resolution = .01)
    # The number of observations (rows) in the model matrix
    n = size(model_matrix, 2)
    # The number of terms (cols) in the model matrix
    p = size(model_matrix, 1)
    inv_info = inv(model_matrix' * model_matrix)

  x_array = 
      collect(Iterators.product(1:5, 1:5, 1:5))
  pred_var_array = 
  x_spv = Vector{Float64}()
    for i in 1:n
    xi_spv = (n * model_matrix[i,:]' * inv_info * model_matrix[i,:])
    push!(x_spv, xi_spv)
    end
    print(x_spv)
  return(100 * p  / maximum(x_spv))
  end


# Creates a grid of values in n dimensions with step size equal to the resolution
# Remember this grows exponentially with n
function create_grid(num_exp, resolution)
    if num_exp == 1
        return [(x,) for x in -1:resolution:1]
    else
        subgrid = create_nres_grid_range(num_exp - 1, resolution)
        return [(x, y...) for x in -1:resolution:1, y in subgrid]
    end
end

function calc_optim_g(model_matrix, potent_vals_mm)
    # The number of observations (rows) in the model matrix
    n = size(model_matrix, 1)
    # The number of terms (cols) in the model matrix
    p = size(model_matrix, 2)
  inv_info = inv(model_matrix' * model_matrix)
  max_spv = 0
  n_rows =  size(potent_vals_mm, 1)
  for row in eachrow(potent_vals_mm)
    spv = row' * inv_info * row
    max_spv = spv > max_spv ? spv : max_spv
  end
  return(100 * p / (n * max_spv))
end


# Generate a matrix of Uniform random variables between -1 and 1
gen_potent_vals = function(n_exp, n_pts)
  return(2 * rand(n_pts, n_exp) .- 1)
end 
potent_vals_dm = gen_potent_vals(3, 1000000)

potent_vals_mm = modelmatrix(@formula(y ~ 1 + x1 + x2 + x1*x2 + 
  x1^2 + x2^2), DataFrame(potent_vals_dm, :auto))

model_matrix = design6_1_modmat
calc_optim_g(model_matrix, potent_vals_mm)

function calc_optim_i(model_matrix, potent_vals_mm)
    # The number of observations (rows) in the model matrix
    n = size(model_matrix, 1)
    # The number of terms (cols) in the model matrix
    p = size(model_matrix, 2)
  inv_info = inv(model_matrix' * model_matrix)
  sum_spv = 0
  n_rows =  size(potent_vals_mm, 1)
  for row in eachrow(potent_vals_mm)
    sum_spv += row' * inv_info * row
  end
  return(1 / (n * (sum_spv/n_rows)))
end
calc_optim_i(model_matrix, potent_vals_mm)

using LinearAlgebra
using Statistics
using StatsModels
using DataFrames

## To Do:
# Verify functions are correct
# Revise overall function
# Clean up scratch work
# Write test function(s)?

# Test Values
C = [ 0 0 1;
     -1 0 1;
     -1 0 1;
     -1 0 1;
     -1 0 1;
      0 1 -1]

# Model matrix from https://www.itl.nist.gov/div898/handbook/pri/section5/pri521.html
# The D-optimal design (D=0.6825575, A=2.2, G=1, I=4.6625) using 12 runs

ModMat = [-1  -1  -1  
          -1  -1  +1  
          -1  +1  -1  
          -1  +1  +1  
           0  -1  -1  
           0  -1  +1  
           0  +1  -1  
           0  +1  +1  
          +1  -1  -1  
          +1  -1  +1  
          +1  +1  -1  
          +1  +1  +1   ]

ModMat_df = DataFrame(
  x1 = repeat([-1,0,1], inner = 4), 
  x2 = repeat(repeat([-1, 1], inner = 2), 3),
  x3 = repeat([-1,1], 6)
)

ModMat_modmat = modelmatrix(@formula(y ~ 1 + x1 + x2 + x3 + x1*x2 + x2*x3 + x3*x1 + 
  protect(x1^2) + protect(x2^2) + protect(x3^2)), ModMat_df)

# Two factor design from Small Exact Response Surface Designs (jpss03)
# D = 42.3123 A = 23.7678 G = 53.6542 I = 0.203900

design6_1 = [
  1 1
  -1 1
  -1 -1
  1 -.394449
  .394449 -1
  -.131483 .131483
]

design6_1_df = DataFrame(design6_1, :auto)


# Design from skpr jss article page 6
#
skpr_design = [
-1 -1 -1 1 1 1
-1 -1 -1 1 1 1
-1 1 1 -1 -1 1
-1 1 1 -1 -1 1
1 -1 1 -1 1 -1
1 -1 1 -1 1 -1
1 1 -1 1 -1 -1
1 1 -1 1 -1 -1
]


function calculate_d_optimality(model_matrix, contrast_matrix)
    # Calculate the determinant of the transpose of the model matrix multiplied by itself
    det_XtX = det(model_matrix' * model_matrix)
    cont_mat = contrast_matrix
    
    # Get the number of observations (rows) in the model matrix
    n = size(model_matrix, 2)
    # Get the number of terms (cols) in the model matrix
    p = size(model_matrix, 1)
    
    # Calculate the D-efficiency
  d_optimality = 100 * (det_XtX ^ (1/p)) / n
    
  return ["D Optimality", d_optimality]
end


#calculate_d_optimality(ModMat_modmat, C)
calculate_d_optimality(design6_1_modmat, C)

function calculate_a_optimality(model_matrix, contrast_matrix)
    # Get the number of observations (rows) in the model matrix
    n = size(model_matrix, 2)
    # Get the number of terms (cols) in the model matrix
    p = size(model_matrix, 1)
    # Calculate the determinant of the transpose of the model matrix multiplied by itself
  a_optimality = 100 * p  / tr(n * inv(transpose(model_matrix) * model_matrix))
    
  return ["A Optimality" a_optimality]
end

calculate_a_optimality(design6_1_modmat, C)

function calculate_e_optimality(model_matrix, contrast_matrix)
    # Calculate the determinant of the transpose of the model matrix multiplied by itself
    det_XtX = det(transpose(model_matrix) * model_matrix)
    
    # Get the number of observations (rows) in the model matrix
    n = size(model_matrix, 1)
    
    # Calculate the E-efficiency
    e_optimality = sqrt(1 / (det_XtX^(1/n)))
    
  return ["E Optimality" e_optimality]
end


function calculate_c_optimality(model_matrix, contrast_matrix)
    XtX_inv = inv(transpose(model_matrix) * model_matrix)
    cefficiency = 1 / det(XtX_inv)
  return ["C Optimality" cefficiency]
end

function calculate_s_optimality(model_matrix, contrast_matrix)
    # Calculate the information matrix
    info_matrix = transpose(model_matrix) * model_matrix
    
    # Calculate the linear combination of parameters specified by the contrast matrix
    linear_combination = contrast_matrix' * model_matrix'
    
    # Calculate the variance of the linear combination
    variance = linear_combination * inv(info_matrix) * linear_combination'
    
    # Calculate S-optimality as the reciprocal of the variance
    s_optimality = 1 / variance[1]
    
  return ["S Optimality" s_optimality]
end

function calculate_t_optimality(model_matrix, contrast_matrix)
    # Calculate the information matrix
    info_matrix = transpose(model_matrix) * model_matrix
    
    # Calculate the linear combination of parameters specified by the contrast matrix
    linear_combination = contrast_matrix' * model_matrix'
    
    # Calculate the variance-covariance matrix of the linear combination
    var_cov_matrix = linear_combination * inv(info_matrix) * linear_combination'
    
    # Calculate T-optimality as the tr of the variance-covariance matrix
    t_optimality = tr(var_cov_matrix)
    
  return ["T Optimality" t_optimality]
end


function calculate_g_optimality(model_matrix, contrast_matrix)
    # Calculate the information matrix
  cont_mat = contrast_matrix # Included as a hack for evaluating multiple op. crits.
  hat_matrix = model_matrix * inv(model_matrix' * model_matrix) * model_matrix'
  p = size(model_matrix, 2)
  g_optimality = p \ maximum(diag(hat_matrix)) 
    
  return ["G Optimality" g_optimality]
end

calculate_g_optimality(design6_1_modmat, C)

function g_criterion(model_matrix)
    M = model_matrix' * model_matrix 
    M⁻¹ = inv(M)
    H   = model_matrix * M⁻¹ * model_matrix'
    size(model_matrix, 2)/maximum(diag(H))
end

g_criterion(design6_1_modmat)


function calculate_v_optimality(model_matrix, contrast_matrix)
    # Calculate the information matrix
  cont_mat = contrast_matrix # Included as a hack for evaluating multiple op. crits.
  hat_matrix = model_matrix * inv(transpose(model_matrix) * model_matrix) * transpose(model_matrix)
  v_optimality = mean(diag(hat_matrix)) 
    
  return ["V Optimality" v_optimality]
end

function calculate_i_optimality(model_matrix, contrast_matrix)
    # Calculate the information matrix
    info_matrix = transpose(model_matrix) * model_matrix
    
    # Calculate the variance-covariance matrix of parameter estimates
    var_cov_matrix = inv(info_matrix)
    
    # Calculate I-optimality as the average variance
    num_params = size(model_matrix, 1)
    i_optimality = sum(var_cov_matrix[i, i] for i in 1:num_params) / num_params
    
  return ["I Optimality" i_optimality]
end

function calc_effs(model_matrix, contrast_matrix, 
  fun_list = [calculate_d_optimality, calculate_a_optimality, calculate_e_optimality,
              calculate_c_optimality, calculate_s_optimality, calculate_t_optimality,
              calculate_g_optimality, calculate_v_optimality, calculate_i_optimality])
  out = Vector{}()
  for eff_fun in fun_list
      push!(out, eff_fun(model_matrix, contrast_matrix))
  end
  return(out)
end

calc_effs(design6_1_modmat, C)


end # module AdvExpDes
