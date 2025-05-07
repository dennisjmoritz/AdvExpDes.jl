using StatsKit

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
  y = repeat([2], 12),
  x1 = repeat([-1,0,1], inner = 4), 
  x2 = repeat(repeat([-1, 1], inner = 2), 3),
  x3 = repeat([-1,1], 6)
)
form = @formula(y ~ 1 + x1 + x2 + x3 + x1*x2 + x2*x3 + x3*x1 + x1^2 + x2^2 + x3^2)


function modelmatrix_df(design_df::DataFrame, formula)
  sch = schema(formula, design_df)
  resolved_formula = apply_schema(formula, sch)
  X_parts = modelcols(resolved_formula, design_df)
  Xmat = reduce(hcat, X_parts)[:, 2:end]
  colnames = coefnames(resolved_formula)[2]  # predictor names only
  return DataFrame(Xmat, Symbol.(collect(colnames)))
end

Matrix(modelmatrix_df(ModMat_df, form))

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
design6_1_modmat = modelmatrix(@formula(y ~ 1 + x1 + x2 + x3 + x1*x2 + x2*x3 + x3*x1 + x1^2 + x2^2 + x3^2))


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


calculate_d_optimality(ModMat_modmat, C)
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
