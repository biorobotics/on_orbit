import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Load data
load_path1 = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/12_2_24/tvlqr/no_orientation_cost/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241202-154712'
load_path1 = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/12_3_24/tvlqr/no_orientation_cost/pos__0.1_0.1_0.0_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.020241203-081155'

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def plot_total_magnitudes(root_folder, trim_to=None):
    """
    Plots total rotation magnitude (theta) for all subdirectories
    in the root folder on a single graph, with an optional trim length.

    Parameters:
    - root_folder: str, path to the root directory.
    - trim_to: int, optional, number of entries to trim each dataset to.
    """
    linestyle_options = ['-']  # Line styles for different folders
    current_style = 0

    # Initialize the plot
    plt.figure(figsize=(12, 6))

    # Loop through all subdirectories
    for subdir, _, files in os.walk(root_folder):
        if 'sw_x_trj.npy' in files and 'sim_ts.npy' in files:
            print(f"Processing: {subdir}")
            
            # Load data
            sw_x_trj = np.load(os.path.join(subdir, 'sw_x_trj.npy'), allow_pickle=True)
            sim_ts = np.load(os.path.join(subdir, 'sim_ts.npy'), allow_pickle=True)
            
            # Trim to the specified length
            if trim_to is not None:
                sw_x_trj = sw_x_trj[:trim_to]
                sim_ts = sim_ts[:trim_to]
            
            # Extract quaternions and convert to rotation vectors
            sw_mrv_quat = sw_x_trj[:, 3:7]
            sw_mrv_rot = R.from_quat(sw_mrv_quat).as_rotvec()
            sw_mrv_rot_deg = np.degrees(sw_mrv_rot)
            
            # Compute total rotation magnitude (theta) in degrees
            rotation_magnitude_deg = np.linalg.norm(np.degrees(sw_mrv_rot), axis=1)

            max_x = np.max(np.abs(sw_mrv_rot_deg[:, 0]))
            max_y = np.max(np.abs(sw_mrv_rot_deg[:, 1]))
            max_z = np.max(np.abs(sw_mrv_rot_deg[:, 2]))
            print(f"  Max X: {max_x:.2f} degrees, Max Y: {max_y:.2f} degrees, Max Z: {max_z:.2f} degrees")
            
            # Plot the total rotation magnitude
            plt.plot(
                sim_ts,
                rotation_magnitude_deg,
                label=os.path.basename(subdir),
                linestyle=linestyle_options[current_style % len(linestyle_options)],
            )

            # Increment style index for variety
            current_style += 1

    # Finalize the plot
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Rotation Magnitude (degrees)')
    plt.title('Total Rotation Magnitudes (Theta) Across All Folders')
    # plt.legend()
    #plt.grid()
    plt.tight_layout()
    plt.show()



root_folder = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/12_3_24/tvlqr/no_orientation_cost' 
plot_total_magnitudes(root_folder, trim_to=None)
