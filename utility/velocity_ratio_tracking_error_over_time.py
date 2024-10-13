import os
import numpy as np
import matplotlib.pyplot as plt

def find_subfolders_with_files(root_dirs, file_name):
    '''Find subdirectories in root_dirs that contain a specific file.'''
    subfolders = []
    for root_dir in root_dirs:
        for subdir, _, files in os.walk(root_dir):
            if file_name in files:
                subfolders.append(subdir)
    return subfolders

def plot_position_errors_and_velocity_ratio_over_trials(root_dirs, threshold=4e-3):
    '''Process multiple subdirectories, each containing peg position error and velocity files, and plot the peg position errors and velocity ratio over time.'''
    trial_dirs = find_subfolders_with_files(root_dirs, 'mrv_arm_joint_vel_cmds_trj.npy')

    if not trial_dirs:
        print("No valid trials found.")
        return

    max_error = 0  # For scaling max peg error
    max_velocity_ratio = 0  # For scaling max velocity ratio

    trial_data = []

    for idx, trial_dir in enumerate(trial_dirs):
        peg_pos_file_path = os.path.join(trial_dir, 'hw_peg_pos_error_trj.npy')
        mrv_vel_file_path = os.path.join(trial_dir, 'mrv_arm_joint_vel_cmds_trj.npy')
        ee_vel_file_path = os.path.join(trial_dir, 'ref_ee_v_trj.npy')
        sim_ts_file_path = os.path.join(trial_dir, 'sim_ts.npy')

        if not (os.path.exists(peg_pos_file_path) and os.path.exists(mrv_vel_file_path)
                and os.path.exists(ee_vel_file_path) and os.path.exists(sim_ts_file_path)):
            print(f"Skipping {trial_dir}: Missing required files.")
            continue

        # Load data
        peg_pos_error_data = np.load(peg_pos_file_path, allow_pickle=True)
        mrv_vel_data = np.load(mrv_vel_file_path, allow_pickle=True)
        ee_vel_data = np.load(ee_vel_file_path, allow_pickle=True)
        sim_ts = np.load(sim_ts_file_path, allow_pickle=True)

        # Ensure all arrays are the same length
        min_length = min(len(sim_ts), len(peg_pos_error_data), len(mrv_vel_data), len(ee_vel_data))
        sim_ts = sim_ts[:min_length]
        peg_pos_error_data = peg_pos_error_data[:min_length]
        mrv_vel_data = mrv_vel_data[:min_length]
        ee_vel_data = ee_vel_data[:min_length]

        # Compute the norm of peg position errors
        peg_pos_error_norm = np.linalg.norm(peg_pos_error_data, axis=1)

        # Compute the norm of MRV joint velocities and end-effector velocities
        mrv_vel_norm = np.linalg.norm(mrv_vel_data, axis=1)
        ee_vel_norm = np.linalg.norm(ee_vel_data, axis=1)

        # Compute the velocity ratio, handling cases where end-effector velocity is very small
        ratio = np.divide(mrv_vel_norm, ee_vel_norm, out=np.ones_like(mrv_vel_norm), where=ee_vel_norm > threshold)

        # Track max values for scaling
        max_error = max(max_error, np.max(peg_pos_error_norm))
        max_velocity_ratio = max(max_velocity_ratio, np.max(ratio))

        # Store the data for later plotting
        trial_data.append((sim_ts, peg_pos_error_norm, ratio))

    # Plot the data with consistent scaling
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Peg Position Error (m)')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Velocity Ratio (Joint/EE)')

    # Plot position errors and velocity ratio for each trial
    colors = ['r','b', 'g', 'c', 'm', 'y', 'k']
    for idx, (sim_ts, peg_error, ratio) in enumerate(trial_data):
        ax1.plot(sim_ts, peg_error, label=f'Configuration-{idx+1} Peg Error', color=colors[idx])
        ax2.plot(sim_ts, ratio, label=f'Configuration-{idx+1} Velocity Ratio', color=colors[idx], linestyle='--')

    # Set consistent limits for all trials
    ax1.set_ylim(0, max_error * 1.1)
    ax2.set_ylim(0, max_velocity_ratio * 1.1)

    # Legends and layout
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Peg Position Errors and Velocity Ratio Over Time Across Multiple Trials')
    plt.tight_layout()
    plt.show()

def main():
    # Specify multiple root directories containing trial subfolders
    root_dirs = [
        "/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_09_24/bad/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.0_20241009-145848",
        #"/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_09_24/good/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.0_20241009-145208"
    ]

    # Call the function to plot position errors and velocity ratio over trials
    plot_position_errors_and_velocity_ratio_over_trials(root_dirs)

if __name__ == "__main__":
    main()
