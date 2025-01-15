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
# TODO: Update this test to one from FangWang
# TODO: Move to test.jl
# Example usage
n = 7            # Number of points
s = 3            # Dimension of the space
h = [1, 3, 6]    # Generating vector

# Generate the GLP set
glp_set = generate_glp_set(n, s, h)
println("GLP Set:")
println(glp_set)

end # module AdvExpDes
