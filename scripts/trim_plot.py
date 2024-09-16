import os
import shutil
import matplotlib.pyplot as plt
import argparse
import re

# Set this to True to delete failed runs, False to keep them
delete_failed = False

def analyze_run_directory(run_directory):
    # Look for the control/<digits>/ directory and count files there
    control_dir = os.path.join(run_directory, 'control')
    if os.path.exists(control_dir):
        # Look for subdirectories within 'control' (e.g., control/12908/)
        subdirs = [os.path.join(control_dir, d) for d in os.listdir(control_dir) if os.path.isdir(os.path.join(control_dir, d))]
        for subdir in subdirs:
            num_files = len(os.listdir(subdir))
            if num_files >= 9:  # Successful if 9 or more files
                return True
    return False

def parse_folder_name(folder_name):
    """Extract wx and wy from folder names using regular expressions."""
    try:
        # Find the part of the string that contains 'client_w' and the following values
        match = re.search(r'client_w_(-?\d+\.\d+)_(-?\d+\.\d+)_(-?\d+\.\d+)', folder_name)
        if match:
            wx = float(match.group(1))
            wy = float(match.group(2))
            return wx, wy
        else:
            raise ValueError("client_w not found")
    except (ValueError, IndexError) as e:
        # If parsing fails, log the error and skip this folder
        print(f"Skipping folder {folder_name}: {str(e)}")
        return None, None

def main(experiment_path, xlim, ylim):
    success_data = []
    fail_data = []

    for run_folder in os.listdir(experiment_path):
        run_directory = os.path.join(experiment_path, run_folder)
        if os.path.isdir(run_directory):
            wx, wy = parse_folder_name(run_folder)
            if wx is not None and wy is not None:  # Ensure valid wx, wy values
                if analyze_run_directory(run_directory):
                    success_data.append((wx, wy))
                else:
                    fail_data.append((wx, wy))
                    # Conditionally delete the failed run
                    if delete_failed:
                        shutil.rmtree(run_directory)
                        print(f"Deleted failed run: {run_directory}")
    
    # Plot the scatter plot
    fig, ax = plt.subplots()
    if success_data:
        success_wx, success_wy = zip(*success_data) if success_data else ([], [])
        ax.scatter(success_wx, success_wy, color='green', s=75, label='Success')  # Set success points to light green
    if fail_data:
        fail_wx, fail_wy = zip(*fail_data) if fail_data else ([], [])
        ax.scatter(fail_wx, fail_wy, color='lightgrey', s=75, label='Failure')  # Set fail points to orange
    
    # Set x and y limits if provided
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Set axis labels with larger font size and include "degrees per second" with omega (ω)
    ax.set_xlabel(r'$\omega_x$ (°/s)', fontsize=21)
    ax.set_ylabel(r'$\omega_y$ (°/s)', fontsize=21)

    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.set_aspect('equal', adjustable='box')

    # Set title and legend
    ax.set_title('Trajectory Convergence', fontsize=24)
    ax.legend(fontsize=10, loc='upper right')
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze runs and delete failures.")
    parser.add_argument("experiment_path", type=str, help="Path to the experiment logs directory")
    parser.add_argument("--xlim", nargs=2, type=float, help="Set x-axis limits as [min, max]", default=None)
    parser.add_argument("--ylim", nargs=2, type=float, help="Set y-axis limits as [min, max]", default=None)
    args = parser.parse_args()

    main(args.experiment_path, args.xlim, args.ylim)
