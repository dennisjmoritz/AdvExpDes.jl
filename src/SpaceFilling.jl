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
