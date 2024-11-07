import numpy as np
import os
import matplotlib.pyplot as plt

def calculate_force_stats(file_path):
    """Calculate max, min, and average force from the .npy file."""
    try:
        data = np.load(file_path)
        
        # Ensure the data is at least 2D with force values
        if data.ndim < 2 or data.shape[1] < 3:
            return None, None, None
        
        # Extract forces (first 3 columns assumed to be x, y, z forces)
        forces = data[:, 0:3]
        
        # Calculate force magnitude (Euclidean norm of the force vectors)
        force_magnitude = np.linalg.norm(forces, axis=1)
        
        # Calculate the statistics: max, min, and mean force
        max_force = np.max(force_magnitude)
        min_force = np.min(force_magnitude)
        avg_force = np.mean(force_magnitude)
        
        return max_force, min_force, avg_force
    
    except Exception as e:
        # Return None for any file that raises an error (invalid data)
        return None, None, None

def find_npy_files(root_directory):
    """Find all .npy files named 'ft_compensated_trj.npy' in subdirectories."""
    npy_files = []
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename == 'ft_compensated_trj.npy':
                npy_files.append(os.path.join(dirpath, filename))
    return npy_files

def plot_valid_trials(directory):
    """Plot force stats for valid trials only."""
    npy_files = find_npy_files(directory)
    
    # Lists to hold valid force stats and corresponding trial labels
    max_forces = []
    min_forces = []
    avg_forces = []
    trial_labels = []
    
    # Process and filter only valid data
    for trial_index, npy_file in enumerate(npy_files, start=1):
        max_force, min_force, avg_force = calculate_force_stats(npy_file)
        
        # Only keep valid trials (where force stats are not None)
        if max_force is not None and min_force is not None and avg_force is not None:
            max_forces.append(max_force)
            min_forces.append(min_force)
            avg_forces.append(avg_force)
            trial_labels.append(f'Trial {trial_index}')  # Label only valid trials
    
    # If no valid trials, exit
    if not trial_labels:
        print("No valid trials found.")
        return
    
    # Plot the force statistics for valid trials only
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # X-axis positions for each valid trial
    x_positions = range(len(trial_labels))
    
    # Plot vertical lines for min-max forces and red circles for average force
    for i in x_positions:
        ax.plot([i, i], [min_forces[i], max_forces[i]], color='blue', lw=2)  # Vertical line for min-max
        ax.plot(i, avg_forces[i], marker='o', markersize=8, color='red')  # Red circle for average
        
        # Annotate the average force value next to the red circle
        ax.text(i + 0.01, avg_forces[i] + 0.01, f'{avg_forces[i]:.2f} N', fontsize=10, color='black', ha='left', va='bottom')
    
    # Set the x-axis labels and ticks
    ax.set_xticks(x_positions)
    ax.set_xticklabels(trial_labels, rotation=45)
    
    # Set the y-axis label and plot title
    ax.set_ylabel('Force (N)')
    ax.set_title('Max, Min, and Average Force per Valid Trial')
    ax.grid(True)
    
    # Tight layout to ensure the labels fit well
    plt.tight_layout()
    plt.show()

# Example usage: Specify the directory containing subfolders with 'ft_compensated_trj.npy' files
directory = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_07_24/2cm_noise/success/'

# Plot force statistics for valid trials in the specified directory
plot_valid_trials(directory)
