import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def load_runtime(folder, filter_type='PF'):
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
    '''Convert quaternions to rotation vectors.'''
    return R.from_quat(quats).as_rotvec()

def compute_errors(filtered_data, gt_data):
    '''Compute the error between filtered estimates and ground truth.'''
    return np.abs(filtered_data - gt_data)

def compute_rotvec_error(filtered_quat, gt_quat):
    '''Compute rotational error as a rotation vector between two quaternions.'''
    quat_diff = R.from_quat(filtered_quat).inv() * R.from_quat(gt_quat)
    return np.abs(quat_diff.as_rotvec())

def plot_errors(sim_ts, pos_error_ft, pos_error_no_ft, rotvec_error_ft, rotvec_error_no_ft):
    '''
    Plot errors for PF with FT and PF without FT on a 2x3 grid.
    
    Parameters:
    - sim_ts: Simulation timestamps.
    - pos_error_ft: Position error array for PF with FT [N, 3].
    - pos_error_no_ft: Position error array for PF without FT [N, 3].
    - rotvec_error_ft: Rotation vector error array for PF with FT [N, 3].
    - rotvec_error_no_ft: Rotation vector error array for PF without FT [N, 3].
    '''
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Particle Filter with FT vs No FT Errors", fontsize=16)

    # Position labels
    position_labels = ['X Position', 'Y Position', 'Z Position']
    for i in range(3):
        axs[0, i].plot(sim_ts, pos_error_ft[:, i], 'b-', label='PF Error (FT)')
        axs[0, i].plot(sim_ts, pos_error_no_ft[:, i], 'r--', label='PF Error (No FT)')
        axs[0, i].set_title(position_labels[i])
        axs[0, i].set_xlabel('Simulation Time')
        axs[0, i].set_ylabel('Position Error (m)')
        axs[0, i].legend()
        axs[0, i].grid(True)

    # Rotation vector labels
    rotvec_labels = ['RotVec X', 'RotVec Y', 'RotVec Z']
    for i in range(3):
        axs[1, i].plot(sim_ts, rotvec_error_ft[:, i], 'b-', label='PF Error (FT)')
        axs[1, i].plot(sim_ts, rotvec_error_no_ft[:, i], 'r--', label='PF Error (No FT)')
        axs[1, i].set_title(rotvec_labels[i])
        axs[1, i].set_xlabel('Simulation Time')
        axs[1, i].set_ylabel('RotVec Error')
        axs[1, i].legend()
        axs[1, i].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_pf_ft_no_ft_errors(root_folder):
    '''Main function to load data, compute errors, and plot PF with FT vs PF without FT errors.'''
    # Load Ground Truth Data
    gt_pos = load_data(root_folder, 'gt_position.npy')  # Shape: [N, 3]
    gt_ori = load_data(root_folder, 'gt_orientation.npy')  # Shape: [N, 4] quaternions [w, x, y, z]

    # Load PF with FT Data
    pf_pos_ft = load_data(root_folder, 'pf_position_ft.npy')  # Shape: [N, 3]
    pf_ori_ft = load_data(root_folder, 'pf_orientation_ft.npy')  # Shape: [N, 4] quaternions [w, x, y, z]
    pf_runtime_ft = load_runtime(root_folder, filter_type='PF with FT')

    # Load PF without FT Data
    pf_pos_no_ft = load_data(root_folder, 'pf_position_no_ft.npy')  # Shape: [N, 3]
    pf_ori_no_ft = load_data(root_folder, 'pf_orientation_no_ft.npy')  # Shape: [N, 4] quaternions [w, x, y, z]
    pf_runtime_no_ft = load_runtime(root_folder, filter_type='PF without FT')

    # Load Simulation Time
    sim_ts = load_data(root_folder, 'sim_ts.npy')  # Shape: [N,]

    # Trim all data arrays to match the shortest length
    sim_ts, pf_pos_ft, pf_pos_no_ft, gt_pos, pf_ori_ft, pf_ori_no_ft, gt_ori = trim_data_to_match_length(
        sim_ts, pf_pos_ft, pf_pos_no_ft, gt_pos, pf_ori_ft, pf_ori_no_ft, gt_ori
    )

    # Convert quaternions to rotation vectors
    gt_rotvec = quat_to_rotvec(gt_ori)             # Ground truth rotation vectors [N, 3]
    pf_rotvec_ft = quat_to_rotvec(pf_ori_ft)       # PF with FT rotation vectors [N, 3]
    pf_rotvec_no_ft = quat_to_rotvec(pf_ori_no_ft) # PF without FT rotation vectors [N, 3]

    # Compute Position Errors
    pos_error_ft = compute_errors(pf_pos_ft, gt_pos)    # Shape: [N, 3]
    pos_error_no_ft = compute_errors(pf_pos_no_ft, gt_pos)  # Shape: [N, 3]

    # Compute Rotation Vector Errors
    rotvec_error_ft = compute_rotvec_error(pf_ori_ft, gt_ori)   # Shape: [N, 3]
    rotvec_error_no_ft = compute_rotvec_error(pf_ori_no_ft, gt_ori)  # Shape: [N, 3]

    # Plot all errors on a 2x3 grid
    plot_errors(sim_ts, pos_error_ft, pos_error_no_ft, rotvec_error_ft, rotvec_error_no_ft)

def main():
    root_dir = "/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_22_24/two_particle_filters/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.006981317007977318_0.020241022-150613"

    # Ensure all necessary files are present
    required_files = [
        'gt_position.npy', 'gt_orientation.npy', 'sim_ts.npy',
        'pf_position_ft.npy', 'pf_orientation_ft.npy', 'pf_position_no_ft.npy', 'pf_orientation_no_ft.npy'
    ]

    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(root_dir, f))]
    if missing_files:
        print(f"Missing files in {root_dir}: {missing_files}")
        return

    # Plot PF with FT vs No FT Errors
    plot_pf_ft_no_ft_errors(root_dir)

if __name__ == "__main__":
    main()
