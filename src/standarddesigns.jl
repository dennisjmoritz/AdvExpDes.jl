function generate_box_behnken_design(n)
    if n < 3
        throw(ArgumentError("A Box-Behnken design requires at least 3 factors."))
    end
    
    # Calculate the number of center points (0 levels)
    num_center_points = 2^n - n - 1
    
    # Initialize the design matrix with center points (0 levels)
    design_matrix = fill(0, num_center_points, n)
    
    # Generate the vertex points (+1 and -1 levels)
    vertex_points = vcat([1 for _ in 1:n], [-1 for _ in 1:n])
    
    # Assign the vertex points to the design matrix
    for i in 1:num_center_points
        design_matrix = vcat(design_matrix, [vertex_points[i, :]])
    end
    
    return design_matrix
end

function generate_central_composite_design(n)
    if n < 2
        throw(ArgumentError("A Central Composite Design requires at least 2 factors."))
    end
    
    # Calculate the number of axial points (α)
    num_axial_points = 2 * n
    
    # Initialize the design matrix with central points (0 levels)
    design_matrix = fill(0, 2^n, n)
    
    # Generate the axial points (+α and -α levels)
    axial_points = [1, -1]
    
    # Generate the factorial points (+1 and -1 levels)
    factorial_points = [1, -1]
    
    # Assign the axial and factorial points to the design matrix
    for i in 1:num_axial_points
        design_matrix = vcat(design_matrix, [axial_points[i] for _ in 1:n])
  end
    
    # Assign the factorial points to the design matrix
    for i in 1:n
        design_matrix = vcat(design_matrix, [factorial_points for _ in 1:n]...)
    end
    
    return design_matrix
end

