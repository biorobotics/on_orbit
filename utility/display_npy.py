import os
import numpy as np
import matplotlib.pyplot as plt

def load_runtime(folder, ekf=True):
    '''Load runtime from run_time.npy.'''
    try:
        runtime = np.load(os.path.join(folder, 'run_time.npy'))[()]  # Single value
        return runtime
    except FileNotFoundError:
        print(f"Runtime file not found for {'EKF' if ekf else 'PF'}.")
        return None

def load_position_data(folder, ekf=True):
    '''Load position and ground truth data.'''
    pos_file = 'ekf_position.npy' if ekf else 'pf_position.npy'
    gt_pos_file = 'gt_position.npy'
    ts_file = 'sim_ts.npy'

    try:
        pos_data = np.load(os.path.join(folder, pos_file), allow_pickle=True)
        gt_data = np.load(os.path.join(folder, gt_pos_file), allow_pickle=True)
        sim_ts = np.load(os.path.join(folder, ts_file), allow_pickle=True)
        return pos_data, gt_data, sim_ts
    except FileNotFoundError as e:
        print(f"Missing data file: {e}")
        return None, None, None

def load_orientation_data(folder, ekf=True):
    '''Load orientation and ground truth orientation data.'''
    ori_file = 'ekf_orientation.npy' if ekf else 'pf_orientation.npy'
    gt_ori_file = 'gt_orientation.npy'
    ts_file = 'sim_ts.npy'

    try:
        ori_data = np.load(os.path.join(folder, ori_file), allow_pickle=True)
        gt_data = np.load(os.path.join(folder, gt_ori_file), allow_pickle=True)
        sim_ts = np.load(os.path.join(folder, ts_file), allow_pickle=True)
        return ori_data, gt_data, sim_ts
    except FileNotFoundError as e:
        print(f"Missing data file: {e}")
        return None, None, None

def plot_total_error(sim_ts, est_data, gt_data, label, axis_label):
    '''Plot total error (combined x, y, z or roll, pitch, yaw).'''
    error = np.linalg.norm(est_data - gt_data, axis=1)
    plt.plot(sim_ts, error, label=label)
    plt.xlabel('Simulation Time')
    plt.ylabel(axis_label)
    plt.legend()
    plt.grid(True)

def plot_individual_errors(sim_ts, est_data, gt_data, labels, axis_label):
    '''Plot individual errors (x, y, z or roll, pitch, yaw).'''
    for i, label in enumerate(labels):
        plt.figure()
        plt.plot(sim_ts, est_data[:, i], label=f'{label} Estimated')
        plt.plot(sim_ts, gt_data[:, i], '--', label=f'{label} Ground Truth')
        plt.xlabel('Simulation Time')
        plt.ylabel(f'{label} {axis_label}')
        plt.legend()
        plt.grid(True)

def plot_ekf_pf_vs_gt(root_dir, ekf=True, num_particles=0, plot_orientation=True, plot_position=True):
    '''Plot total and individual position/orientation errors for EKF or PF.'''
    # Load runtime
    runtime = load_runtime(root_dir, ekf)

    # Plot Position Errors
    if plot_position:
        pos_data, gt_pos_data, sim_ts = load_position_data(root_dir, ekf)
        if pos_data is not None and gt_pos_data is not None:
            plt.figure()
            plot_total_error(sim_ts, pos_data, gt_pos_data, 'Total Position Error', 'Position Error (m)')
            plt.title(f'Total Position Error - {"EKF" if ekf else f"PF (Particles: {num_particles})"}')
            if runtime:
                plt.title(f'Total Position Error - {"EKF" if ekf else f"PF (Particles: {num_particles})"}, Runtime: {runtime:.2f} s')

            # Plot individual errors for x, y, z
            plot_individual_errors(sim_ts, pos_data, gt_pos_data, ['X', 'Y', 'Z'], 'Position (m)')

    # Plot Orientation Errors
    if plot_orientation:
        ori_data, gt_ori_data, sim_ts = load_orientation_data(root_dir, ekf)
        if ori_data is not None and gt_ori_data is not None:
            plt.figure()
            plot_total_error(sim_ts, ori_data, gt_ori_data, 'Total Orientation Error', 'Orientation Error (degrees)')
            plt.title(f'Total Orientation Error - {"EKF" if ekf else f"PF (Particles: {num_particles})"}')
            if runtime:
                plt.title(f'Total Orientation Error - {"EKF" if ekf else f"PF (Particles: {num_particles})"}, Runtime: {runtime:.2f} s')

            # Plot individual errors for roll, pitch, yaw
            plot_individual_errors(sim_ts, ori_data, gt_ori_data, ['Roll', 'Pitch', 'Yaw'], 'Orientation (degrees)')

    plt.show()

def main():
    root_dir = "/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_14_24/pf_test/800_particles/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241014-193156"

    
    plot_ekf_pf_vs_gt(root_dir, ekf=False, num_particles=800, plot_orientation=True, plot_position=True)

if __name__ == "__main__":
    main()
