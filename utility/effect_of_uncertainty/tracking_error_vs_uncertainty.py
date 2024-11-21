import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def find_subfolders_with_files(root_dir, ekf_file, gt_file):
    '''Find subdirectories in root_dir that contain both the EKF and GT position files.'''
    subfolders = []
    for subdir, _, files in os.walk(root_dir):
        if ekf_file in files and gt_file in files:
            subfolders.append(subdir)
    return subfolders

def calculate_tracking_error_stats(ekf_file_path, gt_file_path):
    '''Calculate the tracking error between EKF and GT positions.'''
    ekf_position = np.load(ekf_file_path, allow_pickle=True)
    gt_position = np.load(gt_file_path, allow_pickle=True)

    # Ensure the lengths match
    min_length = min(len(ekf_position), len(gt_position))
    ekf_position = ekf_position[:min_length]
    gt_position = gt_position[:min_length]

    # Calculate the position error as the Euclidean norm of the difference
    position_error = np.linalg.norm(ekf_position - gt_position, axis=1)
    
    max_error = np.max(position_error)
    avg_error = np.mean(position_error)
    
    return max_error, avg_error

def load_noise_value(directory):
    '''Load noise value from a file named 'noise_value.npy' in the directory.'''
    noise_file_path = os.path.join(directory, 'noise_value.npy')
    
    if os.path.exists(noise_file_path):
        return np.load(noise_file_path)
    else:
        return None

from collections import defaultdict

from collections import defaultdict

def collect_error_stats_and_noise(root_dir, ekf_file='ekf_position.npy', gt_file='gt_position.npy'):
    '''Collect and average max and average tracking errors across trials with the same noise values.'''
    trial_dirs = find_subfolders_with_files(root_dir, ekf_file, gt_file)

    if not trial_dirs:
        print("No valid trials found.")
        return None, None, None

    noise_error_dict = defaultdict(list)

    for trial_dir in trial_dirs:
        ekf_file_path = os.path.join(trial_dir, ekf_file)
        gt_file_path = os.path.join(trial_dir, gt_file)

        # Load the noise value
        noise_value = load_noise_value(trial_dir)
        if noise_value is None:
            print(f"Skipping {trial_dir}: Missing 'noise_value.npy' file.")
            continue

        # Calculate max and average tracking errors between EKF and GT
        max_error, avg_error = calculate_tracking_error_stats(ekf_file_path, gt_file_path)

        # Append the errors to the list associated with this noise value
        noise_error_dict[float(noise_value)].append((max_error, avg_error))

    if not noise_error_dict:
        return None, None, None

    noise_values = []
    avg_max_errors = []
    avg_avg_errors = []

    # Average the errors for each noise value
    for noise_value, errors in noise_error_dict.items():
        max_errors, avg_errors = zip(*errors)  # Separate max and avg errors
        noise_values.append(noise_value)
        avg_max_errors.append(np.mean(max_errors))
        avg_avg_errors.append(np.mean(avg_errors))

    return np.array(avg_max_errors), np.array(avg_avg_errors), np.array(noise_values)



def plot_error_stats(noise_values, max_errors, avg_errors):
    '''Plot the max and average tracking errors over 3-sigma position noise with a 2nd order polynomial best fit.'''
    plt.figure(figsize=(10, 6))

    # Sort values by noise for continuous plotting
    sorted_indices = np.argsort(noise_values)
    noise_values_sorted = noise_values[sorted_indices] * 100  # Convert to cm
    max_errors_sorted = max_errors[sorted_indices] * 100  # Convert to cm
    avg_errors_sorted = avg_errors[sorted_indices] * 100  # Convert to cm

    # Plot max errors with best-fit line
    plt.plot(noise_values_sorted, max_errors_sorted, label='Max Error', marker='o', linestyle='', color='blue')

    # Fit and plot 2nd order polynomial for max errors
    p_max = np.polyfit(noise_values_sorted, max_errors_sorted, 2)
    max_fit = np.polyval(p_max, noise_values_sorted)
    plt.plot(noise_values_sorted, max_fit, 'b--', label='Best Fit (Max Error)', linestyle='--')

    # Plot avg errors with best-fit line
    plt.plot(noise_values_sorted, avg_errors_sorted, label='Avg Error', marker='o', linestyle='', color='red')

    # Fit and plot 2nd order polynomial for avg errors
    p_avg = np.polyfit(noise_values_sorted, avg_errors_sorted, 2)
    avg_fit = np.polyval(p_avg, noise_values_sorted)
    plt.plot(noise_values_sorted, avg_fit, 'r--', label='Best Fit (Avg Error)', linestyle='--')

    # Label the axes
    plt.xlabel('3-Sigma Position Noise (cm)')
    plt.ylabel('Tracking Error (cm)')
    plt.title('Max and Average Tracking Errors (EKF vs GT) vs 3-Sigma Position Noise')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()



def main():
    # Specify the root directory containing trial subfolders
    root_dir = "/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_14_24/effect_of_uncertainty"

    # Collect tracking error stats and corresponding noise values
    max_errors, avg_errors, noise_values = collect_error_stats_and_noise(root_dir)

    if max_errors is None or avg_errors is None:
        print("No errors to plot.")
        return

    # Plot the results
    plot_error_stats(noise_values, max_errors, avg_errors)

if __name__ == "__main__":
    main()
