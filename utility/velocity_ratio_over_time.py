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

def plot_joint_vel_vs_ee_vel_ratio(root_dirs, threshold=4e-3):
    '''Process multiple subdirectories, each containing joint velocity and end-effector velocity files, and plot the ratio over time.'''
    # Find subfolders that contain the required velocity command files
    trial_dirs = find_subfolders_with_files(root_dirs, 'mrv_arm_joint_vel_cmds_trj.npy')

    if not trial_dirs:
        print("No valid trials found.")
        return

    plt.figure(figsize=(10, 8))

    for idx, trial_dir in enumerate(trial_dirs):
        # Load necessary files
        mrv_file_path = os.path.join(trial_dir, 'mrv_arm_joint_vel_cmds_trj.npy')
        ee_v_file_path = os.path.join(trial_dir, 'ref_ee_v_trj.npy')
        sim_ts_file_path = os.path.join(trial_dir, 'sim_ts.npy')

        if not (os.path.exists(mrv_file_path) and os.path.exists(ee_v_file_path) and os.path.exists(sim_ts_file_path)):
            print(f"Skipping {trial_dir}: Missing required files.")
            continue

        # Load the data
        mrv_vel_data = np.load(mrv_file_path, allow_pickle=True)
        ee_vel_data = np.load(ee_v_file_path, allow_pickle=True)
        sim_ts = np.load(sim_ts_file_path, allow_pickle=True)

        # Ensure all arrays have the same length
        min_length = min(len(sim_ts), len(mrv_vel_data), len(ee_vel_data))
        sim_ts = sim_ts[:min_length]
        mrv_vel_data = mrv_vel_data[:min_length]
        ee_vel_data = ee_vel_data[:min_length]

        # Compute the norm of joint velocities and end-effector velocities
        mrv_vel_norm = np.linalg.norm(mrv_vel_data, axis=1)
        ee_vel_norm = np.linalg.norm(ee_vel_data, axis=1)

        # Handle very small end-effector velocities: assume a 1:1 ratio when the norm of ee velocity is below the threshold
        ee_vel_norm[ee_vel_norm < threshold] = 1.0

        # Compute the ratio between joint velocity norm and end-effector velocity norm
        ratio = mrv_vel_norm / ee_vel_norm

        # Plot the ratio over time
        plt.plot(sim_ts, ratio, label=f'Trial {idx+1}')

    plt.xlabel('Time')
    plt.ylabel('Joint Velocity / EE Velocity Ratio')
    if len(trial_dirs) > 1:
        plt.legend()
    plt.grid(False)
    plt.title('Ratio of Joint Velocity Norm to End-Effector Velocity Norm')
    plt.tight_layout()
    plt.show()

def main():
    # Specify multiple root directories to search for trials
    root_dirs = [
        "/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_09_24/bad/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.0_20241009-145848"
    ]

    # Call the function to plot the joint velocity to end-effector velocity ratio
    plot_joint_vel_vs_ee_vel_ratio(root_dirs)

if __name__ == "__main__":
    main()
