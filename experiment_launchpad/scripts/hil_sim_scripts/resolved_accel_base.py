import numpy as np
#from ref_traj_point import RefTrajPoint
import pinocchio as pin
from scipy.spatial.transform import Rotation as R
import scipy 

'''A class for resolved accelleration control without any admittance'''

class ResolvedAccelBase(object): 

  def __init__(self): 
    self.use_min_norm = False
    self.account_for_momentum = True #if False, we assume total system momentum is constant and zero and so can use the "simple" generalized Jacobian. This was the controller for the original TFR submission in 2025 (note that Jstar_pinv@twist_d really should have been theta_dot, though)

    # Set these to zero to see how open-loop acceleration controls drifts
    scale = 10
    self.pos_vel_gain = scale*1
    self.ang_vel_gain = scale*1
    self.pos_gain = scale*1
    self.ang_gain = scale*1
    self.set_gains(self.pos_gain,self.ang_gain,self.pos_vel_gain,self.ang_vel_gain)

    self.joints_kp = 30
    self.joints_kd = 2*np.sqrt(self.joints_kp)

  def set_gains(self,pos_gain,ang_gain,pos_vel_gain,ang_vel_gain):
    self.pos_vel_gain = pos_vel_gain
    self.ang_vel_gain = ang_vel_gain
    self.pos_gain = pos_gain
    self.ang_gain = ang_gain
    self.twist_gains = np.diag([pos_vel_gain,pos_vel_gain,pos_vel_gain,ang_vel_gain,ang_vel_gain,ang_vel_gain])
    self.pose_gains = np.diag([pos_gain,pos_gain,pos_gain,ang_gain,ang_gain,ang_gain])
          
  def compute_control(self, ref_traj_point, mrv_client_sim, mrv_config, mrv_config_dot):
      '''
      Inputs:
      
      ref_traj_point: RefTrajPoint object representing a single point in the reference trajectory for the current time step
      mrv_client_sim: MRVClientSim object 
      wrench_peg_peg: wrench applied to the probe, expressed in the probe frame
      mrv_config: MRVClientSim.get_mrv_config()
      mrv_config_dot: MRVClientSim.get_mrv_config_dot()
      dt: time step'''

      p_d = ref_traj_point.p
      rmat_d = ref_traj_point.rmat
      v_d = ref_traj_point.v
      omega_d = ref_traj_point.omega
      vdot_d = ref_traj_point.vdot
      omega_dot_d = ref_traj_point.omega_dot

      twist_d = np.concatenate((v_d,omega_d),axis=0)
      twist_dot_d = np.concatenate((vdot_d,omega_dot_d),axis=0)
      
      # Perfect sensor measurements
      #tip_pos = mrv_client_sim.pin_data.oMf[mrv_client_sim.tip_fid].translation
      #tip_rmat = mrv_client_sim.pin_data.oMf[mrv_client_sim.tip_fid].rotation
      #tip_twist_est = pin.getFrameVelocity(mrv_client_sim.pin_model, mrv_client_sim.pin_data, mrv_client_sim.tip_fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED).vector

      # Noisy sensor measurements
      tip_pos_est, tip_rmat_est, tip_twist_est, _,_,nozzle_twist_est = mrv_client_sim.get_peg_and_nozzle_info_est()

      tip_v = tip_twist_est[:3]
      tip_w = tip_twist_est[3:]
      tip_vel = np.concatenate((tip_v,tip_w),axis=0)

      twist_err = twist_d - tip_vel

      tip_pos_err = p_d - tip_pos_est

      angle_err_mat = np.transpose(tip_rmat_est)@rmat_d
      angle_err =  R.as_rotvec(R.from_matrix(angle_err_mat)) #pin.log3
      pos_err = np.concatenate((tip_pos_err,angle_err),axis=0) #ca.vcat
      

      
      twist_base, twist_dot_base = mrv_client_sim.get_mrv_base_twist()
      if not self.account_for_momentum:
        Jstar, Jstar_dot = mrv_client_sim.get_mrv_generalized_jacobian()
        Jstar_pinv = np.linalg.pinv(Jstar) #ca.pinv
        joint_acc_cmd_min_norm = Jstar_pinv@(  (twist_dot_d - twist_dot_base)  + self.twist_gains@twist_err + self.pose_gains@pos_err - Jstar_dot@Jstar_pinv@(twist_d - twist_base))
      else:
        Jm, Jm_dot, Jb, Jb_dot = mrv_client_sim.get_mrv_kinematic_jacobians()
        Jm_pinv = np.linalg.pinv(Jm)
        joint_acc_cmd_min_norm = Jm_pinv@(     (twist_dot_d-Jb@twist_dot_base) + self.twist_gains@twist_err + self.pose_gains@pos_err - Jm_dot@theta_dot-Jb_dot@twist_base

      if self.use_min_norm: 
        joint_acc_cmd = joint_acc_cmd_min_norm
      else:
        theta_ddot_PD = ref_traj_point.theta_ddot - self.joints_kp*(mrv_config[7:] - ref_traj_point.theta) - self.joints_kd*(mrv_config_dot[6:] - ref_traj_point.theta_dot) 
        if not self.account_for_momentum:
          null_proj = (np.identity(7) - Jstar_pinv@Jstar)
        else:
          null_proj = (np.identity(7) - Jm_pinv@Jm)
        joint_acc_cmd = joint_acc_cmd_min_norm + null_proj@theta_ddot_PD
      return joint_acc_cmd

