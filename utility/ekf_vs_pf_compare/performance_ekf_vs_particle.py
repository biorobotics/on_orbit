import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def load_runtime(folder, ekf=True):
    '''Load runtime from run_time.npy.'''
    try:
        runtime = np.load(os.path.join(folder, 'run_time.npy'))[()]  # Single value
        return runtime
    except FileNotFoundError:
        print(f"Runtime file not found for {'EKF' if ekf else 'PF'}.")
        return None

def load_data(folder, file_name):
    '''Load data from a given folder and file.'''
    file_path = os.path.join(folder, file_name)
    try:
        return np.load(file_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"File {file_name} not found in {folder}.")
        return None
    


def quat_to_rotvec(quaternions):
    rotation = R.from_quat(quaternions)
    return rotation.as_rotvec()  

def trim_data_to_match_length(*arrays):
    '''Trim all input arrays to match the length of the shortest array.'''
    min_length = min(len(arr) for arr in arrays)
    return [arr[:min_length] for arr in arrays]

def plot_6_variables(sim_ts, noisy_pos, filtered_pos, gt_pos, noisy_ori, filtered_ori, gt_ori, title, filter_label='Filtered'):
    '''Plot 6 variables: X, Y, Z positions and Roll, Pitch, Yaw orientations, each in its own plot'''

    # Create a figure with 2 rows and 3 columns of subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title)

    # Flatten the axs array for easy indexing
    axs = axs.flatten()

    # Position labels and indices
    pos_labels = ['X Position', 'Y Position', 'Z Position']
    pos_indices = [0, 1, 2]

    # Plot positions
    for i in range(3):
        axs[i].plot(sim_ts, noisy_pos[:, pos_indices[i]], 'r--', label='Noisy')
        axs[i].plot(sim_ts, filtered_pos[:, pos_indices[i]], 'g-', label=filter_label)
        axs[i].plot(sim_ts, gt_pos[:, pos_indices[i]], 'y-', label='Ground Truth')
        axs[i].set_title(pos_labels[i])
        axs[i].set_xlabel('Simulation Time')
        axs[i].set_ylabel('Position (m)')
        axs[i].legend()
        axs[i].grid(True)

    # Orientation labels and indices
    ori_labels = ['Roll', 'Pitch', 'Yaw']
    ori_indices = [0, 1, 2]

    for i in range(3):
        # Align initial values before unwrapping
        initial_diff = noisy_ori[0, ori_indices[i]] - gt_ori[0, ori_indices[i]]

        if initial_diff > np.pi:
            noisy_ori[:, ori_indices[i]] -= 2 * np.pi
        elif initial_diff < -np.pi:
            noisy_ori[:, ori_indices[i]] += 2 * np.pi

        # Repeat for filtered data
        initial_diff_filtered = filtered_ori[0, ori_indices[i]] - gt_ori[0, ori_indices[i]]

        if initial_diff_filtered > np.pi:
            filtered_ori[:, ori_indices[i]] -= 2 * np.pi
        elif initial_diff_filtered < -np.pi:
            filtered_ori[:, ori_indices[i]] += 2 * np.pi

        # Now apply unwrapping consistently
        axs[i+3].plot(sim_ts, np.unwrap(noisy_ori[:, ori_indices[i]]), 'r--', label='Noisy')
        axs[i+3].plot(sim_ts, np.unwrap(filtered_ori[:, ori_indices[i]]), 'g-', label=filter_label)
        axs[i+3].plot(sim_ts, np.unwrap(gt_ori[:, ori_indices[i]]), 'y-', label='Ground Truth')
        axs[i+3].set_title(ori_labels[i])
        axs[i+3].set_xlabel('Simulation Time')
        axs[i+3].set_ylabel('Rotation Vector')
        axs[i+3].legend()
        axs[i+3].grid(True)

    plt.tight_layout()
    plt.show()

def plot_ekf_pf_vs_gt(root_folder, ekf=True, num_particles=800):
    '''Main function to load data and plot EKF or PF vs GT.'''
    noisy_pos = load_data(root_folder, 'noisy_position.npy')
    noisy_ori = load_data(root_folder, 'noisy_orientation.npy')
    gt_pos = load_data(root_folder, 'gt_position.npy')
    gt_ori = load_data(root_folder, 'gt_orientation.npy')
    filtered_pos = load_data(root_folder, 'ekf_position.npy' if ekf else 'pf_position_ft.npy')
    filtered_ori = load_data(root_folder, 'ekf_orientation.npy' if ekf else 'pf_orientation_ft.npy')
    sim_ts = load_data(root_folder, 'sim_ts.npy')
    run_time = load_runtime(root_folder, ekf=ekf)

    # Convert quaternions to Rodriguez parameters
    noisy_ori = quat_to_rotvec(noisy_ori)
    filtered_ori = quat_to_rotvec(filtered_ori)
    gt_ori = quat_to_rotvec(gt_ori)

    # Trim all arrays to the shortest length
    sim_ts, noisy_pos, filtered_pos, gt_pos, noisy_ori, filtered_ori, gt_ori = trim_data_to_match_length(
        sim_ts, noisy_pos, filtered_pos, gt_pos, noisy_ori, filtered_ori, gt_ori
    )

    if run_time is not None:
        title = f"EKF (Runtime: {run_time:.2f}s)" if ekf else f"PF (Particles: {num_particles}, Runtime: {run_time:.2f}s)"
    else:
        title = "EKF" if ekf else f"PF (Particles: {num_particles})"

    filter_label = 'EKF' if ekf else 'PF'

    # Plot the 6 variables with noisy, filtered, and GT
    plot_6_variables(sim_ts, noisy_pos, filtered_pos, gt_pos, noisy_ori, filtered_ori, gt_ori, title, filter_label=filter_label)

def main():
    root_dir = "/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_22_24/ft_integrated/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.003490658503988659_0.020241022-153855" 

    # Call the function to plot for EKF or PF
    plot_ekf_pf_vs_gt(root_dir, ekf=False, num_particles=800)  # Toggle `ekf` to False for PF

if __name__ == "__main__":
    main()
