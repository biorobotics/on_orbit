import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def calculate_cost(u, dt, R_torque=np.eye(7) * 0.0001, R_force=np.eye(3) * 0.0001):
    """Calculate torque, force, and total costs from torque and force, adjusted by dt."""
    torque = u[:7]
    force = u[-3:]
    torque_cost = 0.5 * torque.T @ R_torque @ torque * dt
    force_cost = 0.5 * force.T @ R_force @ force * dt
    total_cost = torque_cost + force_cost
    return torque_cost, force_cost, total_cost

def load_npy_data(file_path):
    """Load .npy file from specified file path."""
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True)
    print(f"File not found: {file_path}")
    return None

def cumulative_costs(u_array, dt):
    """Calculate cumulative torque, force, and total costs over time."""
    torque_costs = []
    force_costs = []
    total_costs = []
    for u in u_array:
        torque_cost, force_cost, total_cost = calculate_cost(u, dt)
        torque_costs.append(torque_cost)
        force_costs.append(force_cost)
        total_costs.append(total_cost)
    cumulative_torque_costs = np.cumsum(torque_costs)
    cumulative_force_costs = np.cumsum(force_costs)
    cumulative_total_costs = np.cumsum(total_costs)
    return cumulative_torque_costs, cumulative_force_costs, cumulative_total_costs

def gather_costs(directory, dt=0.01):
    """Gather cumulative costs for real and planned trajectories, averaging over multiple runs."""
    cumulative_real_torque_costs_list = []
    cumulative_real_force_costs_list = []
    cumulative_real_total_costs_list = []
    subdirs = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    # If no subdirectories, use the directory itself
    if not subdirs:
        subdirs = [directory]

    max_length_real = 0

    for run_dir in subdirs:
        real_forces = load_npy_data(os.path.join(run_dir, 'mrv_peg_force_trj.npy'))
        real_torques = load_npy_data(os.path.join(run_dir, 'pybullet_torques.npy'))

        if real_forces is not None and real_torques is not None:
            min_steps = min(len(real_forces), len(real_torques))
            real_u = np.concatenate((real_torques[:min_steps], real_forces[:min_steps]), axis=1)
            cumulative_torque_costs, cumulative_force_costs, cumulative_total_costs = cumulative_costs(real_u, dt)
            cumulative_real_torque_costs_list.append(cumulative_torque_costs)
            cumulative_real_force_costs_list.append(cumulative_force_costs)
            cumulative_real_total_costs_list.append(cumulative_total_costs)
            max_length_real = max(max_length_real, len(cumulative_total_costs))
        else:
            print(f"Data missing in {run_dir}, skipping this run.")
            continue

    # Pad shorter arrays with NaNs and compute the average using np.nanmean
    if cumulative_real_total_costs_list:
        padded_torque_costs_list = []
        padded_force_costs_list = []
        padded_total_costs_list = []
        for torque_costs, force_costs, total_costs in zip(cumulative_real_torque_costs_list, cumulative_real_force_costs_list, cumulative_real_total_costs_list):
            padding_length = max_length_real - len(total_costs)
            padded_torque_costs = np.pad(torque_costs, (0, padding_length), constant_values=np.nan)
            padded_force_costs = np.pad(force_costs, (0, padding_length), constant_values=np.nan)
            padded_total_costs = np.pad(total_costs, (0, padding_length), constant_values=np.nan)
            padded_torque_costs_list.append(padded_torque_costs)
            padded_force_costs_list.append(padded_force_costs)
            padded_total_costs_list.append(padded_total_costs)
        cumulative_real_torque_costs_array = np.array(padded_torque_costs_list)
        cumulative_real_force_costs_array = np.array(padded_force_costs_list)
        cumulative_real_total_costs_array = np.array(padded_total_costs_list)
        average_cumulative_real_torque_costs = np.nanmean(cumulative_real_torque_costs_array, axis=0)
        average_cumulative_real_force_costs = np.nanmean(cumulative_real_force_costs_array, axis=0)
        average_cumulative_real_total_costs = np.nanmean(cumulative_real_total_costs_array, axis=0)
    else:
        average_cumulative_real_torque_costs = np.array([])
        average_cumulative_real_force_costs = np.array([])
        average_cumulative_real_total_costs = np.array([])

    return (average_cumulative_real_torque_costs, average_cumulative_real_force_costs, average_cumulative_real_total_costs)

def gather_planned_costs(directory, dt=0.01):
    """Gather cumulative costs for planned trajectory."""
    planned_u = load_npy_data(os.path.join(directory, 'ref_u_trj.npy'))
    if planned_u is not None:
        cumulative_planned_torque_costs, cumulative_planned_force_costs, cumulative_planned_total_costs = cumulative_costs(planned_u, dt)
        total_time = len(cumulative_planned_total_costs) * dt
    else:
        cumulative_planned_torque_costs = np.array([])
        cumulative_planned_force_costs = np.array([])
        cumulative_planned_total_costs = np.array([])
        total_time = 0
    return (cumulative_planned_torque_costs, cumulative_planned_force_costs, cumulative_planned_total_costs, total_time)

def plot_costs(cumulative_mpc_total=None, cumulative_planned_total=None, cumulative_real_total=None,
               cumulative_mpc_torque=None, cumulative_planned_torque=None, cumulative_real_torque=None,
               cumulative_mpc_force=None, cumulative_planned_force=None, cumulative_real_force=None,
               total_time=None, truncate_to_shortest=False, plot_separate_costs=False, dt=0.01):
    """Plot accumulated costs over the same time period for the trajectories."""
    datasets = [
        ('ITAC Simulated', cumulative_real_total, cumulative_real_torque, cumulative_real_force, 'blue', '-'),
        ('MPC Simulated', cumulative_mpc_total, cumulative_mpc_torque, cumulative_mpc_force, 'green' , '-'),
        ('Planned', cumulative_planned_total, cumulative_planned_torque, cumulative_planned_force, 'orange', '--')
        
    ]

    # Filter out None datasets
    datasets = [(label, total, torque, force, color, line_style) for label, total, torque, force, color, line_style in datasets if total is not None and len(total) > 0]


    if not datasets:
        print("No data available to plot.")
        return

    if truncate_to_shortest:
        min_length = min(len(total) for _, total, _, _, _,_ in datasets)
        time_axis = np.arange(0, min_length) * dt
        for i in range(len(datasets)):
            label, total, torque, force, color, line_style = datasets[i]
            datasets[i] = (label, total[:min_length], torque[:min_length], force[:min_length], color, line_style)
    else:
        max_length = max(len(total) for _, total, _, _, _, _ in datasets)
        time_axis = np.arange(0, max_length) * dt

    plt.figure(figsize=(10, 6))

    if plot_separate_costs:
        for label, total, torque, force, color, line_style in datasets:
            # Plot torque costs
            plt.plot(time_axis[:len(torque)], torque, label=f'{label} Torque Cost', linestyle='-', color=color)
            # Plot force costs
            plt.plot(time_axis[:len(force)], force, label=f'{label} Force Cost', linestyle='--', color=color)
        plt.title('Accumulated Torque and Force Costs Over Time')
    else:
        for label, total, _, _, color , line_style in datasets:
            plt.plot(time_axis[:len(total)], total, label=f'{label} Accumulated Cost', linestyle= line_style, color=color)
        plt.title('Accumulated Total Cost Over Time')

    plt.xlabel('Time (s)')
    plt.ylabel('Accumulated Cost')
    plt.legend()
    plt.show()

def main(mpc_directory=None, planned_directory=None, real_directory=None,
         truncate_to_shortest=False, plot_separate_costs=False):
    dt = 0.01  # Assuming a default time step

    # Gather costs for MPC data
    if mpc_directory:
        (cumulative_mpc_torque, cumulative_mpc_force, cumulative_mpc_total) = gather_costs(mpc_directory, dt)
    else:
        cumulative_mpc_torque = None
        cumulative_mpc_force = None
        cumulative_mpc_total = None

    # Gather costs for planned data
    if planned_directory:
        (cumulative_planned_torque, cumulative_planned_force, cumulative_planned_total, total_time) = gather_planned_costs(planned_directory, dt)
    else:
        cumulative_planned_torque = None
        cumulative_planned_force = None
        cumulative_planned_total = None
        total_time = 0

    # Gather costs for non-MPC sim data (real data)
    if real_directory:
        (cumulative_real_torque, cumulative_real_force, cumulative_real_total) = gather_costs(real_directory, dt)
    else:
        cumulative_real_torque = None
        cumulative_real_force = None
        cumulative_real_total = None

    # Determine total time for plotting
    if total_time == 0:
        total_time = max(
            len(cumulative) * dt for cumulative in [cumulative_mpc_total, cumulative_planned_total, cumulative_real_total] if cumulative is not None
        )

    if total_time == 0:
        print("No data available to determine total time.")
        return

    # Plot the costs
    plot_costs(
        cumulative_mpc_total=cumulative_mpc_total,
        cumulative_planned_total=cumulative_planned_total,
        cumulative_real_total=cumulative_real_total,
        cumulative_mpc_torque=cumulative_mpc_torque,
        cumulative_planned_torque=cumulative_planned_torque,
        cumulative_real_torque=cumulative_real_torque,
        cumulative_mpc_force=cumulative_mpc_force,
        cumulative_planned_force=cumulative_planned_force,
        cumulative_real_force=cumulative_real_force,
        total_time=total_time,
        truncate_to_shortest=truncate_to_shortest,
        plot_separate_costs=plot_separate_costs,
        dt=dt
    )

# Example usage:
if __name__ == "__main__":
    # Directories containing the data
    mpc_directory = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/11_18_24/mpc_torque_inputs'
    planned_directory = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/11_19_24/ITAC_ekf_converge/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241119-152340'
    real_directory = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/11_19_24/ITAC_ekf_converge/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241119-152340'

    # Call the main function with desired options
    main(
        mpc_directory=mpc_directory,
        planned_directory=planned_directory,
        real_directory=real_directory,
        truncate_to_shortest=True,
        plot_separate_costs=False  # Set to True to plot torque and force costs separately
    )

