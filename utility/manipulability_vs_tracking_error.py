import os
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import rospkg
import matplotlib.pyplot as plt

# Initialize URDF model
rospack = rospkg.RosPack()
urdf_file = rospack.get_path('on_orbit') + '/urdf/single_arm.urdf'

def compute_manipulability(pin_model, pin_data, q):
    '''Compute manipulability based on the current joint configuration q.'''
    peg_fid = pin_model.getFrameId('ur_1_tool0')
    
    pin.computeJointJacobians(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)

    J = pin.getFrameJacobian(pin_model, pin_data, peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    # Manipulability measure (determinant of the product of the Jacobian and its transpose)
    manipulability = np.sqrt(np.linalg.det(J @ J.T))
    return manipulability

def find_subfolders_with_files(root_dirs, file_name):
    '''Find subdirectories in root_dirs that contain a specific file.'''
    subfolders = []
    for root_dir in root_dirs:
        for subdir, _, files in os.walk(root_dir):
            if file_name in files:
                subfolders.append(subdir)
    return subfolders

def plot_tracking_error_and_manipulability(root_dirs, urdf_file):
    '''Process multiple subdirectories, each containing peg/nozzle position error and joint angles, and plot tracking error and manipulability.'''

    # Build the robot model from the URDF file
    pin_model = RobotWrapper.BuildFromURDF(urdf_file).model
    pin_model.gravity.setZero()
    pin_data = pin.Data(pin_model)

    # Find subfolders that contain the required files
    trial_dirs = find_subfolders_with_files(root_dirs, 'hw_peg_pos_error_trj.npy')

    if not trial_dirs:
        print("No valid trials found.")
        return

    # Initialize variables to store maximum values
    max_manipulability = 0
    max_tracking_error = 0

    trial_data = []

    for idx, trial_dir in enumerate(trial_dirs):
        # File paths
        peg_pos_error_path = os.path.join(trial_dir, 'hw_peg_pos_error_trj.npy')
        nozzle_pos_error_path = os.path.join(trial_dir, 'hw_nozzle_pos_error_trj.npy')
        mrv_angle_file_path = os.path.join(trial_dir, 'mrv_arm_joint_angles_trj.npy')
        sim_ts_file_path = os.path.join(trial_dir, 'sim_ts.npy')

        if not (os.path.exists(peg_pos_error_path) and os.path.exists(nozzle_pos_error_path) and os.path.exists(mrv_angle_file_path) and os.path.exists(sim_ts_file_path)):
            print(f"Skipping {trial_dir}: Missing required files.")
            continue

        # Load data
        peg_pos_error_data = np.load(peg_pos_error_path, allow_pickle=True)
        nozzle_pos_error_data = np.load(nozzle_pos_error_path, allow_pickle=True)
        mrv_joint_angles_data = np.load(mrv_angle_file_path, allow_pickle=True)
        sim_ts = np.load(sim_ts_file_path, allow_pickle=True)

        # Ensure all arrays are the same length
        min_length = min(len(sim_ts), len(peg_pos_error_data), len(nozzle_pos_error_data), len(mrv_joint_angles_data))
        if len(sim_ts) != min_length or len(peg_pos_error_data) != min_length or len(nozzle_pos_error_data) != min_length or len(mrv_joint_angles_data) != min_length:
            print(f"Warning: Truncating data in {trial_dir} due to length mismatch.")
        
        sim_ts = sim_ts[:min_length]
        peg_pos_error_data = peg_pos_error_data[:min_length]
        nozzle_pos_error_data = nozzle_pos_error_data[:min_length]
        mrv_joint_angles_data = mrv_joint_angles_data[:min_length]

        # Compute manipulability for each time step
        mrv_manipulabilities = [compute_manipulability(pin_model, pin_data, q_mrv) for q_mrv in mrv_joint_angles_data]

        # Compute tracking error as the norm of peg position error and nozzle position error
        peg_tracking_error = np.linalg.norm(peg_pos_error_data, axis=1)
        nozzle_tracking_error = np.linalg.norm(nozzle_pos_error_data, axis=1)

        # Combine peg and nozzle tracking error for total tracking error
        total_tracking_error = peg_tracking_error + nozzle_tracking_error

        # Store the data for later plotting
        trial_data.append((sim_ts, total_tracking_error, mrv_manipulabilities))

        # Update max values for scaling
        max_manipulability = max(max_manipulability, np.max(mrv_manipulabilities))
        max_tracking_error = max(max_tracking_error, np.max(total_tracking_error))

    # Plot the data with consistent scaling
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Tracking Error')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Manipulability')

    # Plot tracking error and manipulability for each trial
    colors = plt.cm.viridis(np.linspace(0, 1, len(trial_data)))
    for idx, (sim_ts, tracking_error, manipulabilities) in enumerate(trial_data):
        if len(tracking_error) > 0 and len(manipulabilities) > 0:
            ax1.plot(sim_ts, tracking_error, label=f'Trial {idx+1} Tracking Error', color=colors[idx])
            ax2.plot(sim_ts, manipulabilities, label=f'Trial {idx+1} Manipulability', color=colors[idx], linestyle='--')

    # Check for max values and avoid identical limits issue
    if max_tracking_error > 0:
        ax1.set_ylim(0, max_tracking_error * 1.1)
    else:
        ax1.set_ylim(0, 1)

    if max_manipulability > 0:
        ax2.set_ylim(0, max_manipulability * 1.1)
    else:
        ax2.set_ylim(0, 1)

    # Legends and layout
    if len(trial_data) > 0:
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    plt.title('Tracking Error and Manipulability Over Time')
    plt.tight_layout()
    plt.show()

def main():
    # Specify any number of root directories to search for trials
    root_dirs = [

        # "/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_09_24/2cm_noise/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.0_20241009-121353",
        # "/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_09_24/2cm_noise_very_bad_orient/pos__0.0_0.0_-0.05_rot_0.0_0.0_0.0_delta_v_0.0_0.0_0.0_mrv_w_0.0_0.0_0.0_client_w_0.0_0.0_0.0_20241009-132802"
        "/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_09_24/2cm_noise"
    ]

    # Call the function to plot tracking error and manipulability comparison across multiple trials
    plot_tracking_error_and_manipulability(root_dirs, urdf_file)

if __name__ == "__main__":
    main()
