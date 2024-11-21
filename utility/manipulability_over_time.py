import os
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import rospkg
import matplotlib.pyplot as plt

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

def plot_manipulability_over_trials(root_dirs, urdf_file):
    '''Process multiple subdirectories, each containing client and MRV joint angles files, and plot manipulability.'''
    # Build the robot model from the URDF file (shared for both arms)
    pin_model = RobotWrapper.BuildFromURDF(urdf_file).model
    pin_model.gravity.setZero()
    pin_data = pin.Data(pin_model)

    # Find subfolders that contain the required joint angle files
    trial_dirs = find_subfolders_with_files(root_dirs, 'client_arm_joint_angles_trj.npy')

    if not trial_dirs:
        print("No valid trials found.")
        return

    plt.figure(figsize=(10, 8))

    # Plot for MRV Arm
    plt.subplot(2, 1, 1)
    plt.title('MRV Arm Manipulability Over Time')
    for idx, trial_dir in enumerate(trial_dirs):
        mrv_file_path = os.path.join(trial_dir, 'mrv_arm_joint_angles_trj.npy')
        sim_ts_path = os.path.join(trial_dir, 'sim_ts.npy')
        if not os.path.exists(mrv_file_path) or not os.path.exists(sim_ts_path):
            print(f"Skipping {trial_dir}: Missing MRV file or sim_ts.")
            continue

        mrv_joint_angles_data = np.load(mrv_file_path, allow_pickle=True)
        sim_ts = np.load(sim_ts_path, allow_pickle=True)

        # Ensure sim_ts length matches joint angles length
        if len(sim_ts) != len(mrv_joint_angles_data):
            raise ValueError(f"Length mismatch in {trial_dir}: sim_ts({len(sim_ts)}) and joint angles({len(mrv_joint_angles_data)}).")

        # Compute manipulability for each time step for MRV arm
        mrv_manipulabilities = [compute_manipulability(pin_model, pin_data, q_mrv) for q_mrv in mrv_joint_angles_data]

        # Plot manipulability over actual simulation time for MRV arm
        plt.plot(sim_ts, mrv_manipulabilities, label=f'MRV trial {idx+1}')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Manipulability')
    plt.legend()
    plt.grid(True)

    # Plot for Client Arm
    plt.subplot(2, 1, 2)
    plt.title('Client Arm Manipulability Over Time')
    for idx, trial_dir in enumerate(trial_dirs):
        client_file_path = os.path.join(trial_dir, 'client_arm_joint_angles_trj.npy')
        sim_ts_path = os.path.join(trial_dir, 'sim_ts.npy')
        if not os.path.exists(client_file_path) or not os.path.exists(sim_ts_path):
            print(f"Skipping {trial_dir}: Missing Client file or sim_ts.")
            continue

        client_joint_angles_data = np.load(client_file_path, allow_pickle=True)
        sim_ts = np.load(sim_ts_path, allow_pickle=True)

        # Ensure sim_ts length matches joint angles length
        if len(sim_ts) != len(client_joint_angles_data):
            raise ValueError(f"Length mismatch in {trial_dir}: sim_ts({len(sim_ts)}) and joint angles({len(client_joint_angles_data)}).")

        # Compute manipulability for each time step for Client arm
        client_manipulabilities = [compute_manipulability(pin_model, pin_data, q_client) for q_client in client_joint_angles_data]

        # Plot manipulability over actual simulation time for Client arm
        plt.plot(sim_ts, client_manipulabilities, label=f'Client trial {idx+1}')

    plt.xlabel('Simulation Time (seconds)')
    plt.ylabel('Manipulability')
    plt.legend()
    plt.grid(True)

    # Show the plots
    plt.tight_layout()
    plt.show()

def main():
    # Pass multiple root directories containing trial subfolders
    root_dirs = [
        # Works
        # "/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_08_24/2cm_noise"
        # Does not work
        "/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_logs/10_08_24/2cm_noise_bad_orient"
    ]

    # Call the function to plot manipulability comparison across trials
    plot_manipulability_over_trials(root_dirs, urdf_file)

if __name__ == "__main__":
    main()
