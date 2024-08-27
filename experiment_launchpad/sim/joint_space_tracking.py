import numpy as np
from ref_traj_point import RefTrajPoint
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

class JointSpaceTracking(object): 

  def __init__(self,sw_sim): 

    self.use_admittance = True

    # Admittance control parameters 
    self.last_des_joint_angles = np.zeros(sw_sim.num_rotary)
    self.last_des_joint_vels = np.zeros(sw_sim.num_rotary)

    joint_adm_mass = 300
    joint_adm_damping = 400
    joint_adm_stiffness = 500
    self.Mq_inv = np.diag(1/joint_adm_mass*np.ones(sw_sim.num_rotary))
    self.Bq = np.diag(joint_adm_damping*np.ones(sw_sim.num_rotary))
    self.Kq = np.diag(joint_adm_stiffness*np.ones(sw_sim.num_rotary))

    self.kp = 10
    self.kd = 2*np.sqrt(self.kp)
    
  def reset_admittance_traj(self,sw_sim):
    '''Reset the admittance controller history

    Inputs:
      sw_sim: MRVClientSim object
    '''
    theta_mrv = sw_sim.get_mrv_config()
    self.last_des_joint_angles = theta_mrv[7:]
    self.last_des_joint_vels = np.zeros(sw_sim.num_rotary)

  def compute_control(self, ref_traj_point, wrist_wrench, dt, config, config_dot, Jwrist):
    '''
    Inputs:
      
    ref_traj_point: RefTrajPoint object representing a single point in the reference trajectory for the current time step
    sw_sim: MRVClientSim object 
    wrsit_wrench: wrench applied to the MRV wrist, expressed in the wrist frame
    dt: time step
    config: MRVClientSim.get_mrv_config()
    config_dot: MRVClientSim.get_mrv_config_dot()
    Jwrist: provided by MRVClientSim.get_wrist_jacobian()'''
    
    if self.use_admittance:
      adm_joint_torque_err = Jwrist[:, 6:].transpose()@(wrist_wrench - ref_traj_point.wrench) # Note: we're doing this in frame at wrist, but this is identical to writing it with Jacobian/wrench at tip
      des_joint_accs = ref_traj_point.theta_ddot + self.Mq_inv@(adm_joint_torque_err - self.Bq@(self.last_des_joint_vels - ref_traj_point.theta_dot) - self.Kq@(self.last_des_joint_angles - ref_traj_point.theta))
    else:
      des_joint_accs = ref_traj_point.theta_ddot

    # Update the desired joint vels and angles
    des_joint_vels = self.last_des_joint_vels + des_joint_accs*dt
    des_joint_angles = self.last_des_joint_angles + des_joint_vels*dt
    self.last_des_joint_vels = des_joint_vels
    self.last_des_joint_angles = des_joint_angles

    joint_acc_cmd = des_joint_accs - self.kp*(config[7:] - des_joint_angles) - self.kd*(config_dot[6:] - des_joint_vels) 

    return joint_acc_cmd
