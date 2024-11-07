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

    # Load planned data
    planned_u = load_npy_data(os.path.join(directory, 'ref_u_trj.npy'))
    if planned_u is not None:
        cumulative_planned_torque_costs, cumulative_planned_force_costs, cumulative_planned_total_costs = cumulative_costs(planned_u, dt)
    else:
        cumulative_planned_torque_costs = np.array([])
        cumulative_planned_force_costs = np.array([])
        cumulative_planned_total_costs = np.array([])

    total_time = max(len(average_cumulative_real_total_costs), len(cumulative_planned_total_costs)) * dt

    # Return all cumulative costs
    return (average_cumulative_real_torque_costs, average_cumulative_real_force_costs, average_cumulative_real_total_costs,
            cumulative_planned_torque_costs, cumulative_planned_force_costs, cumulative_planned_total_costs, total_time)

def plot_costs(cumulative_real_total, cumulative_planned_total, cumulative_real_torque, cumulative_planned_torque,
               cumulative_real_force, cumulative_planned_force, total_time, truncate_to_shortest=False, plot_separate_costs=False, dt=0.01):
    """Plot accumulated costs over the same time period for both trajectories."""
    if truncate_to_shortest:
        min_length = min(len(cumulative_real_total), len(cumulative_planned_total))
        cumulative_real_total = cumulative_real_total[:min_length]
        cumulative_planned_total = cumulative_planned_total[:min_length]
        cumulative_real_torque = cumulative_real_torque[:min_length]
        cumulative_planned_torque = cumulative_planned_torque[:min_length]
        cumulative_real_force = cumulative_real_force[:min_length]
        cumulative_planned_force = cumulative_planned_force[:min_length]
        time_axis = np.arange(0, min_length) * dt
    else:
        max_length = int(total_time / dt)
        time_axis = np.arange(0, max_length) * dt

    plt.figure(figsize=(10, 6))

    if plot_separate_costs:
        # Plot torque costs
        plt.plot(time_axis[:len(cumulative_real_torque)], cumulative_real_torque, label='Average Simulated Torque Cost', color='blue')
        plt.plot(time_axis[:len(cumulative_planned_torque)], cumulative_planned_torque, label='Planned Torque Cost', linestyle='--', color='cyan')

        # Plot force costs
        plt.plot(time_axis[:len(cumulative_real_force)], cumulative_real_force, label='Average Simulated Force Cost', color='red')
        plt.plot(time_axis[:len(cumulative_planned_force)], cumulative_planned_force, label='Planned Force Cost', linestyle='--', color='orange')

        plt.title('Accumulated Torque and Force Costs Over Time')
    else:
        # Plot total costs
        plt.plot(time_axis[:len(cumulative_real_total)], cumulative_real_total, label='Average Simulated Accumulated Cost', color='blue')
        plt.plot(time_axis[:len(cumulative_planned_total)], cumulative_planned_total, label='Planned Accumulated Cost', linestyle='--', color='orange')
        plt.title('Accumulated Total Cost Over Time')

    plt.xlabel('Time (s)')
    plt.ylabel('Accumulated Cost')
    plt.legend()
    plt.show()

def create_cost_progression_video(cumulative_real, cumulative_planned, total_time, output_filename='cost_progression_no_noise.mp4', dt=0.01):
    """Create a video showing cost progression over time."""
    time_axis = np.arange(0, int(total_time / dt) + 1) * dt
    time_axis_real = time_axis[:len(cumulative_real)]
    time_axis_planned = time_axis[:len(cumulative_planned)]

    cumulative_real_interp = np.interp(time_axis, time_axis_real, cumulative_real, left=np.nan, right=np.nan)
    cumulative_planned_interp = np.interp(time_axis, time_axis_planned, cumulative_planned, left=np.nan, right=np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, total_time)
    max_cost = np.nanmax([cumulative_real_interp, cumulative_planned_interp])
    ax.set_ylim(0, max_cost * 1.1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Accumulated Cost')
    ax.set_title('Accumulated Total Cost Over Time')
    line_real, = ax.plot([], [], label='Simulated Accumulated Cost', color='blue')
    line_planned, = ax.plot([], [], label='Planned Accumulated Cost', linestyle='--', color='orange')
    ax.legend()

    def init():
        line_real.set_data([], [])
        line_planned.set_data([], [])
        return line_real, line_planned

    def animate(i):
        x_data = time_axis[:i+1]
        y_real = cumulative_real_interp[:i+1]
        y_planned = cumulative_planned_interp[:i+1]
        line_real.set_data(x_data, y_real)
        line_planned.set_data(x_data, y_planned)
        return line_real, line_planned

    fps = int(1 / dt)
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(time_axis),
                                  interval=dt * 1000, blit=True)
    ani.save(output_filename, writer='ffmpeg', fps=fps)
    plt.close(fig)
    print(f"Video saved as {output_filename}")

def main(directory, truncate_to_shortest=False, create_video=False, plot_separate_costs=False):
    (cumulative_real_torque, cumulative_real_force, cumulative_real_total,
     cumulative_planned_torque, cumulative_planned_force, cumulative_planned_total,
     total_time) = gather_costs(directory)

    if len(cumulative_real_total) == 0 and len(cumulative_planned_total) == 0:
        print("Insufficient data to plot.")
        return
    elif len(cumulative_real_total) == 0:
        print("No real data to plot.")
        num_points = int(total_time / 0.01)
        cumulative_real_total = np.array([np.nan] * num_points)
        cumulative_real_torque = cumulative_real_total.copy()
        cumulative_real_force = cumulative_real_total.copy()
    elif len(cumulative_planned_total) == 0:
        print("No planned data to plot.")
        num_points = int(total_time / 0.01)
        cumulative_planned_total = np.array([np.nan] * num_points)
        cumulative_planned_torque = cumulative_planned_total.copy()
        cumulative_planned_force = cumulative_planned_total.copy()

    plot_costs(cumulative_real_total, cumulative_planned_total, cumulative_real_torque, cumulative_planned_torque,
               cumulative_real_force, cumulative_planned_force, total_time, truncate_to_shortest, plot_separate_costs)

    if create_video:
        create_cost_progression_video(cumulative_real_total, cumulative_planned_total, total_time)

# Run the main function with the directory, truncation, and video creation options
directory = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/11_1_24/iterp_fixed_mpc'
main(directory, truncate_to_shortest=False, create_video=False, plot_separate_costs=False)
