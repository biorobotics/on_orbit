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

def plot_max_velocities_over_trials(root_dirs):
    '''Process multiple subdirectories, each containing client and MRV velocity command files, and plot the maximum velocities.'''
    # Find subfolders that contain the required velocity command files
    trial_dirs = find_subfolders_with_files(root_dirs, 'client_arm_joint_vel_cmds_trj.npy')

    if not trial_dirs:
        print("No valid trials found.")
        return

    plt.figure(figsize=(10, 8))

    # Plot for MRV Arm
    plt.subplot(2, 1, 1)
    plt.title('MRV Arm Maximum Commanded Velocities Over Time')
    for idx, trial_dir in enumerate(trial_dirs):
        mrv_file_path = os.path.join(trial_dir, 'mrv_arm_joint_vel_cmds_trj.npy')
        if not os.path.exists(mrv_file_path):
            print(f"Skipping {trial_dir}: Missing MRV velocity file.")
            continue

        mrv_vel_data = np.load(mrv_file_path, allow_pickle=True)

        # Compute the maximum velocity at each time step across all 6 joints
        mrv_max_velocities = np.max(mrv_vel_data, axis=1)

        # Plot maximum commanded velocity over time for MRV arm
        time_steps = np.arange(len(mrv_max_velocities))
        plt.plot(time_steps, mrv_max_velocities, label=f'MRV Trial {idx+1}')

    plt.xlabel('Time Step')
    plt.ylabel('Max Velocity (rad/s)')
    plt.legend()
    plt.grid(True)

    # Plot for Client Arm
    plt.subplot(2, 1, 2)
    plt.title('Client Arm Maximum Commanded Velocities Over Time')
    for idx, trial_dir in enumerate(trial_dirs):
        client_file_path = os.path.join(trial_dir, 'client_arm_joint_vel_cmds_trj.npy')
        if not os.path.exists(client_file_path):
            print(f"Skipping {trial_dir}: Missing Client velocity file.")
            continue

        client_vel_data = np.load(client_file_path, allow_pickle=True)

        # Compute the maximum velocity at each time step across all 6 joints
        client_max_velocities = np.max(client_vel_data, axis=1)

        # Plot maximum commanded velocity over time for Client arm
        time_steps = np.arange(len(client_max_velocities))
        plt.plot(time_steps, client_max_velocities, label=f'Client Trial {idx+1}')

    plt.xlabel('Time Step')
    plt.ylabel('Max Velocity (rad/s)')
    plt.legend()
    plt.grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()

def main():
    # Pass multiple root directories containing trial subfolders
    root_dirs = [
        "/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_08_24/2cm_noise_bad_orient/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.0_20241008-174032"
    ]

    # Call the function to plot velocity comparison across trials
    plot_max_velocities_over_trials(root_dirs)

if __name__ == "__main__":
    main()
