import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
from collections import defaultdict

def calculate_force_stats(file_path):
    '''Calculate max and average force from the .npy file.'''
    # Load the .npy file
    data = np.load(file_path)
    
    # Check if the data has a valid shape (we expect a 2D array with at least 3 columns)
    if len(data.shape) < 2 or data.shape[1] < 3:
        return None, None
    
    # Extract forces (first 3 columns)
    forces = data[:, 0:3]
    
    # Calculate force magnitude (Euclidean norm)
    force_magnitude = np.linalg.norm(forces, axis=1)
    
    # Calculate max and average force
    max_force = np.max(force_magnitude)
    avg_force = np.mean(force_magnitude)
    
    return max_force, avg_force

def find_npy_files(directory):
    '''Recursively find all 'mrv_peg_force_trj.npy' files in subfolders.'''
    npy_files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename == 'mrv_peg_force_trj.npy':  # Adjust if file naming changes
                npy_files.append(os.path.join(dirpath, filename))
    return npy_files

def load_noise_value(directory):
    '''Load noise value from a file named 'noise_value.npy' in the directory.'''
    noise_file_path = os.path.join(directory, 'noise_value.npy')
    
    if os.path.exists(noise_file_path):
        return np.load(noise_file_path)
    else:
        return None

from collections import defaultdict

def collect_force_stats_and_noise(root_directory):
    '''Collect and average max and average force stats across trials with the same noise values.'''
    max_forces_dict = defaultdict(list)
    avg_forces_dict = defaultdict(list)

    # Search for .npy files and 'noise_value.npy' inside the root directory and subdirectories
    npy_files = find_npy_files(root_directory)
    
    for npy_file in npy_files:
        # Extract the directory where the .npy file is located
        file_dir = os.path.dirname(npy_file)
        
        # Load the noise value from 'noise_value.npy'
        noise_value = load_noise_value(file_dir)
        if noise_value is None:
            print(f"Skipping {file_dir}: Missing 'noise_value.npy' file.")
            continue

        # Calculate the max and average forces
        max_force, avg_force = calculate_force_stats(npy_file)
        if max_force is None or avg_force is None:
            continue

        # Group the forces by noise value
        max_forces_dict[float(noise_value)].append(max_force)
        avg_forces_dict[float(noise_value)].append(avg_force)

    # Calculate the average max and average forces for each noise value
    noise_values = []
    avg_max_forces = []
    avg_avg_forces = []

    for noise_value in sorted(max_forces_dict.keys()):
        max_forces = max_forces_dict[noise_value]
        avg_forces = avg_forces_dict[noise_value]

        noise_values.append(noise_value)
        avg_max_forces.append(np.mean(max_forces))
        avg_avg_forces.append(np.mean(avg_forces))

    return np.array(avg_max_forces), np.array(avg_avg_forces), np.array(noise_values)


def plot_force_stats(noise_values, max_forces, avg_forces):
    '''Plot max and average forces over 3-sigma position noise with best-fit lines.'''
    plt.figure(figsize=(12, 10))

    # Sort values by noise for continuous plotting
    sorted_indices = np.argsort(noise_values)
    noise_values_sorted = noise_values[sorted_indices]*100  # Convert to cm
    max_forces_sorted = max_forces[sorted_indices]
    avg_forces_sorted = avg_forces[sorted_indices]
    
    # Subplot 1: Max Forces
    plt.subplot(2, 1, 1)
    plt.plot(noise_values_sorted, max_forces_sorted, label='Max Force', marker='o', linestyle='', color='blue')

    # Fit a best-fit line to max forces
    p_max = Polynomial.fit(noise_values_sorted, max_forces_sorted, deg=2)
    plt.plot(noise_values_sorted, p_max(noise_values_sorted), 'b--', label='Best Fit (Max Force)')

    # Label axes
    plt.xlabel('3-Sigma Position Noise (cm)')
    plt.ylabel('Force (N)')
    plt.title('Max Force vs 3-Sigma Position Noise')
    plt.grid(True)
    plt.legend()

    # Subplot 2: Avg Forces
    plt.subplot(2, 1, 2)
    plt.plot(noise_values_sorted, avg_forces_sorted, label='Avg Force', marker='o', linestyle='', color='red')

    # Fit a best-fit line to avg forces
    p_avg = Polynomial.fit(noise_values_sorted, avg_forces_sorted, deg=2)
    plt.plot(noise_values_sorted, p_avg(noise_values_sorted), 'r--', label='Best Fit (Avg Force)')

    # Label axes
    plt.xlabel('3-Sigma Position Noise (cm)')
    plt.ylabel('Force (N)')
    plt.title('Avg Force vs 3-Sigma Position Noise')
    plt.grid(True)
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

def main():
    # Specify the root directory containing subdirectories with force data and noise values
    root_directory = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_14_24/effect_of_uncertainty'

    # Collect force stats and corresponding noise values
    max_forces, avg_forces, noise_values = collect_force_stats_and_noise(root_directory)

    if len(noise_values) == 0:
        print("No valid data to plot.")
        return

    # Plot the results
    plot_force_stats(noise_values, max_forces, avg_forces)

if __name__ == "__main__":
    main()