using StatsKit

## Space Filling Metrics

function max_min_distance(design;
                        space_bounds = (-1.0,1.0),
                        grid_density = 50)
  X = Matrix(design)[:,2:end]  # drop intercept col if present
  d = size(X,2)
  grids = [range(space_bounds[1], space_bounds[2], length=grid_density) for _ in 1:d]
  maxmin = 0.0
  for pt in Iterators.product(grids...)
      p = collect(pt)
      dists = [norm(row .- p) for row in eachrow(X)]
      maxmin = max(maxmin, minimum(dists))
  end
  return maxmin
end

## Alphabet criteria

# D-optimality (maximize det (X′X)):
# Jack Kiefer. “Optimum Experimental Designs,” J. R. Stat. Soc. B, 21(2), 272–319 (1959).


# A-optimality (minimize trace [(X′X)⁻¹]):
# Jack Kiefer. “Optimum Experimental Designs,” J. R. Stat. Soc. B, 21(2), 272–319 (1959).

# E-optimality (maximize smallest eigenvalue of X′X):
# Jack Kiefer. “Optimum Experimental Designs,” J. R. Stat. Soc. B, 21(2), 272–319 (1959).


# C-optimality (minimize variance of a specified contrast):
# Gustav Elfving. “A geometric characterization of c-optimal designs,” Ann. Math. Statist., 23(2), 255–262 (1952).


# G-optimality (minimize max diag [X (X′X)⁻¹ X′]):
# Jack Kiefer. “Optimum Experimental Designs,” J. R. Stat. Soc. B, 21(2), 272–319 (1959).


# T-optimality (maximal discrimination between two models):
# A. C. Atkinson & V. V. Fedorov. “The design of experiments for discriminating between two rival models,” Biometrika, 62(1), 57–70 (1975).


# S-optimality (maximize a measure of column-orthogonality + det(X′X)):
# Y. Shin & D. Xiu. “Nonadaptive quasi-optimal points selection for least squares linear regression,” SIAM J. Sci. Comput., 38(1), A385–A411 (2016).


# I-optimality (minimize average prediction variance over the region):
# Jack Kiefer. “Optimum Experimental Designs,” J. R. Stat. Soc. B, 21(2), 272–319 (1959).


# V-optimality (minimize average prediction variance at specified points):
# Jack Kiefer. “Optimum Experimental Designs,” J. R. Stat. Soc. B, 21(2), 272–319 (1959). 

## Multiple Criteria

# Model‐Oriented Design of Experiments
# Valerii V. Fedorov & Peter Hackl (1997). Springer-Verlag.
# A comprehensive monograph covering D-, A-, E-, G-, I-, V-, C-, S-, and T-optimality in both linear and nonlinear settings, with detailed theory and numerous examples
# ACM Digital Library

# Optimal Design of Experiments
# Friedrich Pukelsheim (2006). SIAM.
# A mathematically rigorous survey of all the standard information‐based criteria (D, A, E, G, I, V, …), the Kiefer–Wolfowitz equivalence theorem, and the convex‐analysis underpinnings that unite them
# SIAM Ebooks

# Optimum Experimental Designs, with SAS
# A. C. Atkinson, A. N. Donev & R. D. Tobias (2007). Oxford Univ. Press.
# A practitioner‐focused treatment showing how to implement D-, A-, E-, G-, I-, V-, S- and T-optimal designs (and more) in SAS, with both algorithmic and theoretical commentary



function calculate_d_optimality(X::AbstractMatrix)
    info = transpose(X) * X
    p, n = size(info,1), size(X,1)
    return 100 * (det(info)^(1/p)) / n
end


function calculate_a_optimality(X::AbstractMatrix)
    info = transpose(X) * X
    p, n = size(info,1), size(X,1)
    return 100 * p / tr(n * inv(info))
end


function calculate_c_optimality(X::AbstractMatrix)
    info_inv = inv(transpose(X) * X)
    return 1 / det(info_inv)
end

function calculate_e_optimality(X::AbstractMatrix)
    info = transpose(X) * X
    n = size(X,1)
    return sqrt(1 / (det(info)^(1/n)))
end


function calculate_g_optimality(X::AbstractMatrix)
    H = X * inv(transpose(X) * X) * transpose(X)
    p = size(X,2)
    return p / maximum(diag(H))
end


function calculate_v_optimality(X::AbstractMatrix)
    H = X * inv(transpose(X) * X) * transpose(X)
    return mean(diag(H))
end


function calculate_i_optimality(X::AbstractMatrix)
    info_inv = inv(transpose(X) * X)
    p = size(info_inv,1)
    return sum(info_inv[i,i] for i in 1:p) / p
end


function calculate_s_optimality(X::AbstractMatrix, C::AbstractMatrix)
    info_inv = inv(transpose(X) * X)
    c = C[:,1]
    return 1 / (c' * info_inv * c)
end


function calculate_t_optimality(X::AbstractMatrix, C::AbstractMatrix)
    info_inv = inv(transpose(X) * X)
    V = C' * info_inv * C
    return tr(V)
end


