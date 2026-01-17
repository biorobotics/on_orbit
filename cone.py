import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_cone_approximation(cone_slope, num_planes=8, cone_vertex_pos=None):
    """
    Visualizes an n-sided pyramid approximation of a cone using matplotlib
    with a pure white background and no axes.

    Args:
        cone_slope (float): The slope of the cone (determines its width).
                            A positive value means the cone opens along the positive Z-axis.
        num_planes (int): The number of planes used to approximate the cone.
        cone_vertex_pos (np.array, optional): 3D position of the cone vertex (x, y, z).
                                              Defaults to [0, 0, 0].
    """
    if cone_vertex_pos is None:
        cone_vertex_pos = np.array([0.0, 0.0, 0.0])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- Set background to pure white ---
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # --- Define the cone ---
    cone_height_viz = 2.0
    base_radius_at_height = cone_height_viz / cone_slope

    # Create the vertices of the base polygon for the approximated cone
    base_points = []
    for i in range(num_planes):
        theta = i * 2 * np.pi / num_planes
        x = base_radius_at_height * np.cos(theta)
        y = base_radius_at_height * np.sin(theta)
        base_points.append([x, y, cone_height_viz])
    base_points = np.array(base_points)

    # Transform base points to the actual cone_vertex_pos
    transformed_base_points = base_points + cone_vertex_pos

    # Plot the planes (triangles connecting vertex to base edges)
    for i in range(num_planes):
        p1 = transformed_base_points[i]
        p2 = transformed_base_points[(i + 1) % num_planes] # Next point, wrapping around
        triangle_vertices = np.array([cone_vertex_pos, p1, p2])
        ax.plot_trisurf(triangle_vertices[:, 0], triangle_vertices[:, 1], triangle_vertices[:, 2],
                        color='skyblue', alpha=0.9, edgecolor='blue', linewidth=0.5)

    # Plot the cone vertex
    ax.scatter(cone_vertex_pos[0], cone_vertex_pos[1], cone_vertex_pos[2],
               color='red', s=100, label='Cone Vertex', depthshade=False)

    # --- Hide the axes and grid ---
    ax.axis('off')

    # Set equal aspect ratio for proper scaling
    max_range = np.array([transformed_base_points[:,0].max()-transformed_base_points[:,0].min(),
                          transformed_base_points[:,1].max()-transformed_base_points[:,1].min(),
                          transformed_base_points[:,2].max()-transformed_base_points[:,2].min()]).max() / 2.0

    mid_x = (transformed_base_points[:,0].max()+transformed_base_points[:,0].min()) * 0.5
    mid_y = (transformed_base_points[:,1].max()+transformed_base_points[:,1].min()) * 0.5
    mid_z = (transformed_base_points[:,2].max()+transformed_base_points[:,2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z)

    plt.show()

# --- Example Usage ---

# Visualize the cone
visualize_cone_approximation(cone_slope=1.0, num_planes=8)