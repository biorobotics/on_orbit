import os
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import copy
import rospkg


rospack = rospkg.RosPack()
rospath = rospack.get_path('on_orbit')


root_dir = "/home/medusar/experiment_logs/07_15_24/constant_gains_scheduled_plunge/"  

# Get the URDF file path
mrv_cv_urdf_file = rospath + '/urdf/robot_cv_detached.urdf'

def compute_manipulability(pin_model, pin_data, x):
    
    peg_fid = pin_model.getFrameId('ee_tip')
    q = x[:pin_model.nq] 
    

    full_config = copy.deepcopy(x[:pin_model.nq])

    pin.computeJointJacobians(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)

    J = pin.getFrameJacobian(pin_model, pin_data, peg_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)

    Jm = J[:, 6:13]

    full_config = copy.deepcopy(x[:pin_model.nq])
    Ag = pin.computeCentroidalMap(pin_model, pin_data, full_config)

    Ab = Ag[:,:6]
    Ab_inv = np.linalg.inv(Ab)
    Atheta = Ag[:,6:13]
    Jb = J[:,:6] #Jacobian for the base frame
    Jstar = Jm - Jb@Ab_inv@Atheta
    
    # Manipulability
    manipulability = np.sqrt(np.linalg.det(Jstar @ Jstar.T))
    return manipulability

def process_directory(root_dir, mrv_cv_urdf_file):
    '''Process each subdirectory to compute mean and min manipulability at the last time step.'''
    # Build the robot model from the URDF file
    pin_model = RobotWrapper.BuildFromURDF(mrv_cv_urdf_file).model
    pin_model.gravity.setZero()
    pin_data = pin.Data(pin_model)

    manipulabilities = []

    # Walk through each subdirectory
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == 'sw_x_trj.npy':
                file_path = os.path.join(subdir, file)
                data = np.load(file_path, allow_pickle=True)
                x = data[-1]  # Get the last time step configuration

                manipulability = compute_manipulability(pin_model, pin_data, x)
                manipulabilities.append(manipulability)

    if manipulabilities:
        mean_manipulability = np.mean(manipulabilities)
        min_manipulability = np.min(manipulabilities)

        print(f"Mean Manipulability: {mean_manipulability}")
        print(f"Minimum Manipulability: {min_manipulability}")
    else:
        print("No valid data found.")


process_directory(root_dir, mrv_cv_urdf_file)
