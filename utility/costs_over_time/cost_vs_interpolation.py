import os
import re
import numpy as np
import matplotlib.pyplot as plt

def calculate_cost(u, dt, R_torque=np.eye(7) * 0.0001, R_force=np.eye(3) * 0.0001):
    """Calculate total cost from torque and force, adjusted by dt."""
    torque_cost = 0.5 * u[:7].T @ R_torque @ u[:7] * dt
    force_cost = 0.5 * u[-3:].T @ R_force @ u[-3:] * dt
    return torque_cost + force_cost

def load_npy_data(file_path):
    """Load .npy file from specified file path."""
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True)
    print(f"File not found: {file_path}")
    return None

def cumulative_costs(u_array, dt):
    """Calculate cumulative costs over time."""
    costs = np.array([calculate_cost(u, dt) for u in u_array])
    cumulative_costs = np.cumsum(costs)
    return cumulative_costs, costs

def get_interpolation_distance(directory_name):
    """Extract the interpolation distance from the directory name using regex."""
    match = re.search(r'pos__(-?\d+\.\d+)', directory_name)
    return float(match.group(1)) if match else None

def gather_costs_for_subdirectory(subdirectory, dt_real=0.01):
    """Gather cumulative costs for the real trajectory in a subdirectory."""
    real_forces = load_npy_data(os.path.join(subdirectory, 'mrv_peg_force_trj.npy'))
    real_torques = load_npy_data(os.path.join(subdirectory, 'pybullet_torques.npy'))
    
    if real_forces is not None and real_torques is not None:
        min_steps = min(len(real_forces), len(real_torques))
        real_u = np.concatenate((real_torques[:min_steps], real_forces[:min_steps]), axis=1)
        cumulative_real_costs, _ = cumulative_costs(real_u, dt_real)
        return cumulative_real_costs
    else:
        print(f"Data missing in {subdirectory}, skipping this run.")
        return None

def gather_planned_cost(planned_directory, dt_planned=0.01):
    """Load planned cumulative costs from the specified directory."""
    planned_u = load_npy_data(os.path.join(planned_directory, 'ref_u_trj.npy'))
    if planned_u is not None:
        cumulative_planned_costs, _ = cumulative_costs(planned_u, dt_planned)
        return cumulative_planned_costs
    return None

def plot_interpolated_costs(root_directory, dt_real=0.01, dt_planned=0.01, max_interp_distance=None, plot_to_shortest_length=False):
    """Plot planned costs and interpolated real costs from all subdirectories."""
    subdirectories = [os.path.join(root_directory, d) for d in os.listdir(root_directory)
                      if os.path.isdir(os.path.join(root_directory, d))]
    
    # Get cumulative planned costs from the first subdirectory
    planned_costs = None
    for subdirectory in subdirectories:
        planned_costs = gather_planned_cost(subdirectory, dt_planned)
        if planned_costs is not None:
            break
    
    if planned_costs is None:
        print("No planned data found.")
        return
    
    # Collect real costs with their interpolation distances
    real_costs_data = []
    for subdirectory in subdirectories:
        interp_distance = get_interpolation_distance(subdirectory)
        
        # Check if the interpolation distance is within the max limit (if set)
        if interp_distance is not None and (max_interp_distance is None or interp_distance <= max_interp_distance):
            real_costs = gather_costs_for_subdirectory(subdirectory, dt_real)
            if real_costs is not None:
                real_costs_data.append((interp_distance, real_costs))
    
    # Sort the real costs data by interpolation distance for organized legend
    real_costs_data.sort(key=lambda x: x[0], reverse=True)
    
    # Determine the minimum length if `plot_to_shortest_length` is set
    min_length = len(planned_costs) if plot_to_shortest_length else None
    if plot_to_shortest_length:
        for _, real_costs in real_costs_data:
            min_length = min(min_length, len(real_costs))
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    time_axis_planned = np.arange(0, len(planned_costs)) * dt_planned
    if min_length:
        time_axis_planned = time_axis_planned[:min_length]
        planned_costs = planned_costs[:min_length]
    plt.plot(time_axis_planned, planned_costs, label='Planned Cost', linestyle='--', color='orange', linewidth=3)
    
    # Plot each real cost trajectory with its interpolation distance
    for interp_distance, real_costs in real_costs_data:
        time_axis_real = np.arange(0, len(real_costs)) * dt_real
        if min_length:
            time_axis_real = time_axis_real[:min_length]
            real_costs = real_costs[:min_length]
        plt.plot(time_axis_real, real_costs, label=f'Real Cost (Interp {interp_distance})', linestyle='-')
    
    # Finalize the plot
    plt.xlabel('Time (s)')
    plt.ylabel('Accumulated Cost')
    plt.title('Accumulated Total Cost Over Time')
    plt.legend()
    plt.grid(False)
    plt.show()

# Run the function with the root directory
root_dir = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_25_24/cost_comparisons_no_noise_interpolation/'
plot_interpolated_costs(root_dir, max_interp_distance=0.07, plot_to_shortest_length=True)
