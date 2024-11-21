import os
import numpy as np
import matplotlib.pyplot as plt

def load_data(folder, file_name):
    '''Load data from a given folder and file.'''
    file_path = os.path.join(folder, file_name)
    try:
        return np.load(file_path, allow_pickle=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_name} not found in {folder}.")

def report_and_plot_std(sim_ts, pf_pos_std_ft, pf_pos_std_no_ft, pf_rot_std_ft, pf_rot_std_no_ft):
    '''
    Plot standard deviations (position and rotation) and report their averages.
    
    Parameters:
    - sim_ts: Simulation timestamps.
    - pf_pos_std_ft: Standard deviations for position (with FT).
    - pf_pos_std_no_ft: Standard deviations for position (without FT).
    - pf_rot_std_ft: Standard deviations for rotation (with FT).
    - pf_rot_std_no_ft: Standard deviations for rotation (without FT).
    '''
    # Compute overall standard deviation (Euclidean norm) for position and rotation
    overall_pos_std_ft = np.linalg.norm(pf_pos_std_ft, axis=1)
    overall_pos_std_no_ft = np.linalg.norm(pf_pos_std_no_ft, axis=1)
    overall_rot_std_ft = np.linalg.norm(np.unwrap(pf_rot_std_ft), axis=1)
    overall_rot_std_no_ft = np.linalg.norm(np.unwrap(pf_rot_std_no_ft), axis=1)


    # Compute averages
    avg_pos_std_ft = np.mean(overall_pos_std_ft)
    avg_pos_std_no_ft = np.mean(overall_pos_std_no_ft)
    avg_rot_std_ft = np.mean(overall_rot_std_ft)
    avg_rot_std_no_ft = np.mean(overall_rot_std_no_ft)

    # Report the averages
    print(f"Average Position Std (FT): {avg_pos_std_ft:.6f} m")
    print(f"Average Position Std (No FT): {avg_pos_std_no_ft:.6f} m")
    print(f"Average Rotation Std (FT): {avg_rot_std_ft:.4f} deg")
    print(f"Average Rotation Std (No FT): {avg_rot_std_no_ft:.4f} deg")

    # Plot the overall standard deviations
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot overall position standard deviations
    axs[0].plot(sim_ts, overall_pos_std_ft, label='Position Std FT', color='b')
    axs[0].plot(sim_ts, overall_pos_std_no_ft, label='Position Std No FT', color='r')
    axs[0].set_title('Overall Position Standard Deviation')
    axs[0].set_xlabel('Simulation Time')
    axs[0].set_ylabel('Standard Deviation (m)')
    axs[0].legend()
    axs[0].grid(True)

    # Plot overall rotation standard deviations
    axs[1].plot(sim_ts, overall_rot_std_ft, label='Rotation Std FT', color='b')
    axs[1].plot(sim_ts, overall_rot_std_no_ft, label='Rotation Std No FT', color='r')
    axs[1].set_title('Overall Rotation Standard Deviation')
    axs[1].set_xlabel('Simulation Time')
    axs[1].set_ylabel('Standard Deviation (deg)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_std_devs_from_root(root_dir):
    '''Main function to load, plot, and report overall standard deviations from a root directory.'''
    # Load Simulation Time
    sim_ts = load_data(root_dir, 'sim_ts.npy')

    # Load Standard Deviations (Position and Rotation) for both PF with and without FT
    pf_pos_std_ft = load_data(root_dir, 'pf_cov_pos_ft.npy')  # Shape: [N, 3]
    pf_pos_std_no_ft = load_data(root_dir, 'pf_cov_pos_no_ft.npy')  # Shape: [N, 3]
    pf_rot_std_ft = load_data(root_dir, 'pf_cov_rot_ft.npy')  # Shape: [N, 3]
    pf_rot_std_no_ft = load_data(root_dir, 'pf_cov_rot_no_ft.npy')  # Shape: [N, 3]

    # Ensure that the loaded arrays are trimmed to the same length as sim_ts
    sim_ts, pf_pos_std_ft, pf_pos_std_no_ft, pf_rot_std_ft, pf_rot_std_no_ft = trim_data_to_match_length(
        sim_ts, pf_pos_std_ft, pf_pos_std_no_ft, pf_rot_std_ft, pf_rot_std_no_ft
    )

    # Plot and report Standard Deviations
    report_and_plot_std(sim_ts, pf_pos_std_ft, pf_pos_std_no_ft, pf_rot_std_ft, pf_rot_std_no_ft)

def trim_data_to_match_length(*arrays):
    '''Trim all input arrays to match the length of the shortest array.'''
    min_length = min(len(arr) for arr in arrays if arr is not None)
    return [arr[:min_length] for arr in arrays]

def normalize_angles(angles):
    '''Normalize angles to be in the range [-π, π].'''
    return (angles + np.pi) % (2 * np.pi) - np.pi

def complex_representation(angles):
    '''Convert angles in radians to complex numbers for continuity.'''
    return np.exp(1j * angles)

def report_and_plot_std(sim_ts, pf_pos_std_ft, pf_pos_std_no_ft, pf_rot_std_ft, pf_rot_std_no_ft):
    '''
    Plot standard deviations (position and rotation) and report their averages.
    
    Parameters:
    - sim_ts: Simulation timestamps.
    - pf_pos_std_ft: Standard deviations for position (with FT).
    - pf_pos_std_no_ft: Standard deviations for position (without FT).
    - pf_rot_std_ft: Standard deviations for rotation (with FT).
    - pf_rot_std_no_ft: Standard deviations for rotation (without FT).
    '''
    # Compute overall standard deviation (Euclidean norm) for position
    overall_pos_std_ft = np.linalg.norm(pf_pos_std_ft, axis=1)
    overall_pos_std_no_ft = np.linalg.norm(pf_pos_std_no_ft, axis=1)
    
    # Convert angles to complex representation
    pf_rot_std_ft_complex = complex_representation(pf_rot_std_ft)
    pf_rot_std_no_ft_complex = complex_representation(pf_rot_std_no_ft)
    
    # Calculate the standard deviation by taking the angle of the mean complex vector
    overall_rot_std_ft = np.abs(np.angle(np.mean(pf_rot_std_ft_complex, axis=1)))
    overall_rot_std_no_ft = np.abs(np.angle(np.mean(pf_rot_std_no_ft_complex, axis=1)))

    # Compute averages
    avg_pos_std_ft = np.mean(overall_pos_std_ft)
    avg_pos_std_no_ft = np.mean(overall_pos_std_no_ft)
    avg_rot_std_ft = np.mean(overall_rot_std_ft)
    avg_rot_std_no_ft = np.mean(overall_rot_std_no_ft)

    # Report the averages
    print(f"Average Position Std (FT): {avg_pos_std_ft:.6f} m")
    print(f"Average Position Std (No FT): {avg_pos_std_no_ft:.6f} m")
    print(f"Average Rotation Std (FT): {avg_rot_std_ft:.4f} rad")
    print(f"Average Rotation Std (No FT): {avg_rot_std_no_ft:.4f} rad")

    # Plot the overall standard deviations
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot overall position standard deviations
    axs[0].plot(sim_ts, overall_pos_std_ft, label='Position Std FT', color='b')
    axs[0].plot(sim_ts, overall_pos_std_no_ft, label='Position Std No FT', color='r')
    axs[0].set_title('Overall Position Standard Deviation')
    axs[0].set_xlabel('Simulation Time')
    axs[0].set_ylabel('Standard Deviation (m)')
    axs[0].legend()
    axs[0].grid(True)

    # Plot overall rotation standard deviations
    axs[1].plot(sim_ts, overall_rot_std_ft, label='Rotation Std FT', color='b')
    axs[1].plot(sim_ts, overall_rot_std_no_ft, label='Rotation Std No FT', color='r')
    axs[1].set_title('Overall Rotation Standard Deviation')
    axs[1].set_xlabel('Simulation Time')
    axs[1].set_ylabel('Standard Deviation (rad)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    root_dir = "/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_22_24/covariance_compare/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.004363323129985824_0.020241022-172155"
    plot_std_devs_from_root(root_dir)
