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

def plot_position_errors_over_trials(root_dirs):
    '''Process multiple subdirectories, each containing peg and nozzle position error files, and plot the errors over time.'''
    # Find subfolders that contain the required position error files
    trial_dirs = find_subfolders_with_files(root_dirs, 'hw_peg_pos_error_trj.npy')

    if not trial_dirs:
        print("No valid trials found.")
        return

    plt.figure(figsize=(10, 8))

    max_error = 0  # For keeping track of max error across trials

    # Plot for Peg Position Errors
    plt.subplot(2, 1, 1)
    plt.title('Peg Position Errors Over Time')
    for idx, trial_dir in enumerate(trial_dirs):
        peg_pos_file_path = os.path.join(trial_dir, 'hw_peg_pos_error_trj.npy')
        sim_ts_file_path = os.path.join(trial_dir, 'sim_ts.npy')

        if not (os.path.exists(peg_pos_file_path) and os.path.exists(sim_ts_file_path)):
            print(f"Skipping {trial_dir}: Missing peg position error or sim_ts file.")
            continue

        peg_pos_error_data = np.load(peg_pos_file_path, allow_pickle=True)
        sim_ts = np.load(sim_ts_file_path, allow_pickle=True)

        # Ensure the lengths match
        min_length = min(len(peg_pos_error_data), len(sim_ts))
        peg_pos_error_data = peg_pos_error_data[:min_length]
        sim_ts = sim_ts[:min_length]

        # Compute the norm of the position error at each time step (magnitude of the 3D error vector)
        peg_pos_error_norm = np.linalg.norm(peg_pos_error_data, axis=1)

        # Track max error for consistent scaling
        max_error = max(max_error, np.max(peg_pos_error_norm))

        # Plot peg position error over time
        plt.plot(sim_ts, peg_pos_error_norm, label=f'Trial {idx+1} Peg')

    plt.xlabel('Time')
    plt.ylabel('Position Error (m)')

    if len(trial_dirs) > 1:
        plt.legend()
    plt.grid(False)

    # Plot for Nozzle Position Errors
    plt.subplot(2, 1, 2)
    plt.title('Nozzle Position Errors Over Time')
    for idx, trial_dir in enumerate(trial_dirs):
        nozzle_pos_file_path = os.path.join(trial_dir, 'hw_nozzle_pos_error_trj.npy')
        sim_ts_file_path = os.path.join(trial_dir, 'sim_ts.npy')

        if not (os.path.exists(nozzle_pos_file_path) and os.path.exists(sim_ts_file_path)):
            print(f"Skipping {trial_dir}: Missing nozzle position error or sim_ts file.")
            continue

        nozzle_pos_error_data = np.load(nozzle_pos_file_path, allow_pickle=True)
        sim_ts = np.load(sim_ts_file_path, allow_pickle=True)

        # Ensure the lengths match
        min_length = min(len(nozzle_pos_error_data), len(sim_ts))
        nozzle_pos_error_data = nozzle_pos_error_data[:min_length]
        sim_ts = sim_ts[:min_length]

        # Compute the norm of the position error at each time step (magnitude of the 3D error vector)
        nozzle_pos_error_norm = np.linalg.norm(nozzle_pos_error_data, axis=1)

        # Track max error for consistent scaling
        max_error = max(max_error, np.max(nozzle_pos_error_norm))

        # Plot nozzle position error over time
        plt.plot(sim_ts, nozzle_pos_error_norm, label=f'Trial {idx+1} Nozzle')

    plt.xlabel('Time')
    plt.ylabel('Position Error (m)')

    if len(trial_dirs) > 1:
        plt.legend()
    plt.grid(False)

    # Show the plots
    plt.tight_layout()
    plt.show()

def main():
    # Specify multiple root directories containing trial subfolders
    root_dirs = [
        "/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_09_24/good/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.0_20241009-145208"
    ]

    # Call the function to plot position errors over trials
    plot_position_errors_over_trials(root_dirs)

if __name__ == "__main__":
    main()
