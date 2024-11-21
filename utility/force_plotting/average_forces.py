import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_force_stats(file_path):
    # Load the .npy file
    data = np.load(file_path)
    
    # Check if the data has a valid shape (we expect 2D array with at least 3 columns)
    if len(data.shape) < 2 or data.shape[1] < 3:
        return None, None, None
    
    # Extract forces (first 3 columns)
    forces = data[:, 0:3]
    
    # Calculate force magnitude (Euclidean norm)
    force_magnitude = np.linalg.norm(forces, axis=1)
    
    # Calculate stats
    max_force = np.max(force_magnitude)
    min_force = np.min(force_magnitude)
    avg_force = np.mean(force_magnitude)
    
    return max_force, min_force, avg_force

def find_npy_files(root_directory):
    # Recursively find all 'ft_compensated_trj.npy' files in subfolders
    npy_files = []
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename == 'ft_compensated_trj.npy':
                npy_files.append(os.path.join(dirpath, filename))
    return npy_files

def combine_force_stats(directories):
    combined_max_forces = []
    combined_min_forces = []
    combined_avg_forces = []
    
    # Process each directory
    for directory in directories:
        npy_files = find_npy_files(directory)
        
        max_forces = []
        min_forces = []
        avg_forces = []
        
        # Process each .npy file
        for npy_file in npy_files:
            max_force, min_force, avg_force = calculate_force_stats(npy_file)
            
            if max_force is None or min_force is None or avg_force is None:
                continue
            
            max_forces.append(max_force)
            min_forces.append(min_force)
            avg_forces.append(avg_force)
        
        if max_forces and min_forces and avg_forces:
            combined_max_forces.append(np.max(max_forces))
            combined_min_forces.append(np.min(min_forces))
            combined_avg_forces.append(np.mean(avg_forces))
    
    return combined_max_forces, combined_min_forces, combined_avg_forces

def plot_combined_force_stats(directories, custom_labels):
    # Get combined stats for all directories
    max_forces, min_forces, avg_forces = combine_force_stats(directories)
    
    # Plot the combined data
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set x-axis position for each directory
    x_positions = range(len(custom_labels))
    
    # Plot vertical lines and average circle for each directory
    for i in x_positions:
        ax.plot([i, i], [min_forces[i], max_forces[i]], color='blue', lw=2)  # Vertical line for min-max
        ax.plot(i, avg_forces[i], marker='o', markersize=8, color='red', label='Average Force' if i == 0 else "")  # Red circle for average
        
        # Add the value of average force next to the red circle, adjusted slightly to the right and above
        ax.text(i + 0.01, avg_forces[i] + 0.01, f'{avg_forces[i]:.2f} N', fontsize=10, color='black', ha='left', va='bottom')

    # Customize plot
    ax.set_xticks(x_positions)
    ax.set_xticklabels(custom_labels, rotation=45)
    ax.set_ylabel('Force (N)')
    ax.set_title('Force Comparison')
    ax.grid(True)
    
    # Add legend for average marker
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

# Example usage: Specify multiple directories containing subfolders with 'ft_compensated_trj.npy' files
directories = [
    '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_07_24/2cm_noise/success/',
    '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_07_24/no_noise/',
    # Add more directories here
]

# Manually input the labels for each directory (corresponding to the order of the 'directories' list)
custom_labels = [
    '2cm_noise',  # Label for the first directory
    'no_noise',  # Label for the second directory
]

# Plot force statistics with custom labels
plot_combined_force_stats(directories, custom_labels)
