import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point , Polygon

def diff_nozzle_profile(z):
    '''This is the profile of the large nozzle that represents the nozzle of the LAE'''
    h, k = 0.0, 0.0  # center of the circle that defines the nozzle profile
    R = 0.760744339  # radius of the circle that defines the nozzle profile
    arc_radius_offset = 0.1420  # offset from the edge of the circle to the centerline of the nozzle
    if z >= 0.0 and z <= 0.4290:
        x_arc = h + np.sqrt(R**2 - (z-k)**2) - R + arc_radius_offset
    else:
        x_arc = 0.0095

    z_transition_start = 0.429
    transition_width = 0.020
    t = (z - z_transition_start) / transition_width
    # Sigmoid based blend to make the nozzle profile differentiable
    blend_factor = 1 / (1 + np.exp(-10 * t))
    nozzle_profile_radius = blend_factor * 0.0095 + (1 - blend_factor) * x_arc
    return nozzle_profile_radius

def dist_to_nozzle_wall(peg_pos):
    '''This function uses the nozzle profile to determine the distance of the peg to the nozzle wall'''
    z = peg_pos[2]
    if z >= 0:
        nozzle_radius = diff_nozzle_profile(z)
        x, y = peg_pos[0], peg_pos[1]
        dist_to_nozzle = np.sqrt(x**2 + y**2) - nozzle_radius
    else:
        dist_to_nozzle = -np.inf
    return dist_to_nozzle

def conic_plane_nozzle_profile(z):
    # Constants
    num_planes = 8
    nozzle_wall_width = 0.005  
    entry_outer_diameter = 0.284
    entry_radius = entry_outer_diameter * 0.5

    # Slope angle from XACRO
    slope_angle = 0.285672797  # In radians
    cone_slope = np.tan(slope_angle)

        # Check if z is greater than 0.429
    if z > 0.429:
        r_outer = .016  # Fixed radius for z > 0.429
    else:
        r_outer = entry_radius - cone_slope * z  # Original calculation

    r_inner = r_outer - nozzle_wall_width

    # Offet angle to aligned the plane with the x-axis
    angle_offset = np.pi / num_planes  # Half of central angle
    theta = np.linspace(0, 2 * np.pi, num_planes, endpoint=False) + angle_offset
    radius = r_inner / np.cos(angle_offset)

    # Compute x and y coordinates of the inner octagon vertices
    x_vertices = radius * np.cos(theta)
    y_vertices = radius * np.sin(theta)
    x_vertices = np.append(x_vertices, x_vertices[0])
    y_vertices = np.append(y_vertices, y_vertices[0])

    return x_vertices, y_vertices

def dist_to_octagon_wall(peg_pos, z):
    '''Computes the minimum distance from the peg to the closest octagonal plane'''
    
    x_values, y_values = conic_plane_nozzle_profile(z)
    min_dist = np.inf
    x_p, y_p = peg_pos[0], peg_pos[1]

    # Compute distance from point to each segment of the octagon
    for i in range(len(x_values) - 1):
        x1, y1 = x_values[i], y_values[i]
        x2, y2 = x_values[i + 1], y_values[i + 1]
        edge_vector = np.array([x2 - x1, y2 - y1])
        point_vector = np.array([x_p - x1, y_p - y1])
        t = np.dot(point_vector, edge_vector) / np.dot(edge_vector, edge_vector)
        t = np.clip(t, 0, 1)  # Clamp t to [0, 1] to stay within the edge's endpoints
        # Closest point on the edge to the point
        closest_point = np.array([x1, y1]) + t * edge_vector

        # Distance from the point to the closest point on the edge
        dist = np.linalg.norm(np.array([x_p, y_p]) - closest_point)

        # Track the minimum distance
        if dist < min_dist:
            min_dist = dist
    
    # Check if the point is inside the octagon
    polygon = Polygon(zip(x_values, y_values))
    if polygon.contains(Point(x_p, y_p)):
        min_dist = -min_dist
        return min_dist
    else:
        min_dist = min_dist
        return min_dist
    


def plot_nozzle_slice_with_point(r_of_z, point, epsilon=1e-6):
    """
    Plots a 2D slice (circular cross-section and octagon) of the nozzle at the given point's z-value.
    
    Args:
    - r_of_z: A function that gives the radius of the nozzle at a given height z.
    - point: The (x, y, z) coordinates of the point to plot.
    - epsilon: The tolerance for checking proximity to the nozzle surface.
    """
    # Unpack the point
    x_p, y_p, z_p = point
    
    # Get the radius of the nozzle at the point's z-value
    nozzle_radius = r_of_z(z_p)
    
    # Generate points for the circular cross-section at z = z_p
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = nozzle_radius * np.cos(theta)
    y_circle = nozzle_radius * np.sin(theta)
    
    # Get the octagonal approximation
    x_octagon, y_octagon = conic_plane_nozzle_profile(z_p)
    # Close the octagon
    x_octagon = np.append(x_octagon, x_octagon[0])
    y_octagon = np.append(y_octagon, y_octagon[0])
    
    # Create the plot
    plt.figure(figsize=(12,12))
    
    # Plot the circular profile
    plt.plot(x_circle, y_circle, label=f'Circular Nozzle slice at z = {z_p}', color='blue')

    # Plot the octagonal profile
    plt.plot(x_octagon, y_octagon, label=f'Octagonal Nozzle slice at z = {z_p}', color='green')

    # Plot the point
    plt.scatter(x_p, y_p, color='red', s=100, label=f'Point ({x_p:.2f}, {y_p:.2f}, {z_p:.2f})')
    
    # Set axis limits and labels
    plt.xlim(-0.15, 0.15)
    plt.ylim(-0.15, 0.15)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Calculate the distance to the nozzle wall (circle and octagon)
    distance_to_circle = dist_to_nozzle_wall(point)
    distance_to_octagon = dist_to_octagon_wall(point, z_p)
    
    # Add labels and title
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title(f'Nozzle Slice at z = {z_p}\nDistance to circular wall: {distance_to_circle:.5f}, '
              f'Distance to octagon wall: {distance_to_octagon:.5f}')
    
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

def likelihood(dt):
    k = 200 
    k_n = 70000 
    if dt < 0:
       
        return np.exp(k*dt)
    else:
        
        return 1 / (1 + k_n*dt**2)

point = (0.0, 0.0, 0.430) 
plot_nozzle_slice_with_point(diff_nozzle_profile, point)
# Show the point on the likely hood function
distance_octagon = dist_to_octagon_wall(point, point[2])
distance_circle = dist_to_nozzle_wall(point)

# Plot the likelihood function
dt_values = np.linspace(-0.05, 0.05, 100)
likelihood_values = [likelihood(dt) for dt in dt_values]

plt.figure(figsize=(12, 6))
plt.plot(dt_values, likelihood_values, label='Likelihood Function', color='blue')
plt.axvline(distance_octagon, color='green', linestyle='--', label='Distance to Octagon Wall')
plt.axvline(distance_circle, color='red', linestyle='--', label='Distance to Circular Wall')
plt.xlabel('Distance to Wall')
plt.ylabel('Likelihood')
plt.title('Likelihood Function for Distance to Wall')
plt.legend()
plt.grid(True)
plt.show()
