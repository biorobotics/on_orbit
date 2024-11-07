import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def load_runtime(folder, filter_type='EKF'):
    '''Load runtime from run_time.npy.'''
    try:
        runtime = np.load(os.path.join(folder, 'run_time.npy'), allow_pickle=True)[()]
        return runtime
    except FileNotFoundError:
        print(f"Runtime file not found for {filter_type}.")
        return None

def load_data(folder, file_name):
    '''Load data from a given folder and file.'''
    file_path = os.path.join(folder, file_name)
    try:
        return np.load(file_path, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_name} not found in {folder}.")

def trim_data_to_match_length(*arrays):
    '''Trim all input arrays to match the length of the shortest array.'''
    min_length = min(len(arr) for arr in arrays if arr is not None)
    return [arr[:min_length] for arr in arrays]

def quat_to_rotvec(quats):
    '''
    Convert quaternions to rotation vectors.
    Quaternions should be in the format [w, x, y, z].
    '''
    return R.from_quat(quats).as_rotvec()

def compute_errors(filtered_data, gt_data):
    '''Compute the error between filtered estimates and ground truth.'''
    return np.abs(filtered_data - gt_data)

def compute_rotvec_error(filtered_quat, gt_quat):
    # what we want to do insted is to take the difference between the two quaternions, and then convert that to a twist

    # Compute the difference between the quaternions
    quat_diff = R.from_quat(filtered_quat).inv() * R.from_quat(gt_quat)
    return np.abs(quat_diff.as_rotvec())

def plot_errors(sim_ts, ekf_pos_error, pf_pos_error, ekf_rotvec_error, pf_rotvec_error):
    '''
    Plot errors for EKF and PF on a 2x3 grid.
    
    Parameters:
    - sim_ts: Simulation timestamps.
    - ekf_pos_error: EKF position error array [N, 3].
    - pf_pos_error: PF position error array [N, 3].
    - ekf_rotvec_error: EKF rotation vector error array [N, 3].
    - pf_rotvec_error: PF rotation vector error array [N, 3].
    '''
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("EKF vs PF Errors", fontsize=16)

    # Position labels
    position_labels = ['X Position', 'Y Position', 'Z Position']
    for i in range(3):
        axs[0, i].plot(sim_ts, ekf_pos_error[:, i], 'g-', label='EKF Error')
        axs[0, i].plot(sim_ts, pf_pos_error[:, i], 'm--', label='PF Error')
        axs[0, i].set_title(position_labels[i])
        axs[0, i].set_xlabel('Simulation Time')
        axs[0, i].set_ylabel('Position Error (m)')
        axs[0, i].legend()
        axs[0, i].grid(True)

    # Rotation vector labels
    rotvec_labels = ['RotVec X', 'RotVec Y', 'RotVec Z']
    for i in range(3):
        axs[1, i].plot(sim_ts, ekf_rotvec_error[:, i], 'g-', label='EKF Error')
        axs[1, i].plot(sim_ts, pf_rotvec_error[:, i], 'm--', label='PF Error')
        axs[1, i].set_title(rotvec_labels[i])
        axs[1, i].set_xlabel('Simulation Time')
        axs[1, i].set_ylabel('RotVec Error')
        axs[1, i].legend()
        axs[1, i].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_ekf_pf_errors(root_folder):
    '''Main function to load data, compute errors, and plot EKF vs PF errors.'''
    # Load Ground Truth Data (quaternions format)
    gt_pos = load_data(root_folder, 'gt_position.npy')  # Shape: [N, 3]
    gt_ori = load_data(root_folder, 'gt_orientation.npy')  # Shape: [N, 4] quaternions [w, x, y, z]

    # Load EKF Data
    ekf_pos = load_data(root_folder, 'ekf_position.npy')  # Shape: [N, 3]
    ekf_ori = load_data(root_folder, 'ekf_orientation.npy')  # Shape: [N, 4] quaternions [w, x, y, z]
    ekf_runtime = load_runtime(root_folder, filter_type='EKF')

    # Load PF Data
    pf_pos = load_data(root_folder, 'pf_position.npy')  # Shape: [N, 3]
    pf_ori = load_data(root_folder, 'pf_orientation.npy')  # Shape: [N, 4] quaternions [w, x, y, z]
    pf_runtime = load_runtime(root_folder, filter_type='PF')

    # Load Simulation Time
    sim_ts = load_data(root_folder, 'sim_ts.npy')  # Shape: [N,]

    # Trim all data arrays to match the shortest length
    sim_ts, ekf_pos, pf_pos, gt_pos, ekf_ori, pf_ori, gt_ori = trim_data_to_match_length(
        sim_ts, ekf_pos, pf_pos, gt_pos, ekf_ori, pf_ori, gt_ori
    )

    # Convert quaternions to rotation vectors
    gt_rotvec = quat_to_rotvec(gt_ori)        # Ground truth rotation vectors [N, 3]
    ekf_rotvec = quat_to_rotvec(ekf_ori)      # EKF rotation vectors [N, 3]
    pf_rotvec = quat_to_rotvec(pf_ori)        # PF rotation vectors [N, 3]

    # Compute Position Errors
    ekf_pos_error = compute_errors(ekf_pos, gt_pos)  # Shape: [N, 3]
    pf_pos_error = compute_errors(pf_pos, gt_pos)    # Shape: [N, 3]

    # Compute Rotation Vector Errors
    ekf_rotvec_error = compute_rotvec_error(ekf_ori, gt_ori)  # Shape: [N, 3]
    pf_rotvec_error = compute_rotvec_error(pf_ori, gt_ori)    # Shape: [N, 3]


    # Plot all errors on a 2x3 grid
    plot_errors(sim_ts, ekf_pos_error, pf_pos_error, ekf_rotvec_error, pf_rotvec_error)

def main():
    root_dir = "/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_14_24/comparison_case/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241015-124858"

    # Ensure all necessary files are present
    required_files = [
        'gt_position.npy', 'gt_orientation.npy', 'sim_ts.npy',
        'ekf_position.npy', 'ekf_orientation.npy', 'run_time.npy',
        'pf_position.npy', 'pf_orientation.npy'
    ]

    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(root_dir, f))]
    if missing_files:
        print(f"Missing files in {root_dir}: {missing_files}")
        return

    # Plot EKF vs PF Errors
    plot_ekf_pf_errors(root_dir)

if __name__ == "__main__":
    main()
