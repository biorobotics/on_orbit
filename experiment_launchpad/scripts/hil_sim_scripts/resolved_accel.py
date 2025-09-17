import numpy as np
#from ref_traj_point import RefTrajPoint
import pinocchio as pin
from scipy.spatial.transform import Rotation as R
from resolved_accel_base import ResolvedAccelBase
import scipy 

class ResolvedAccel(object): 

  def __init__(self,use_scheduled_gains): 
    '''
    Args:
        use_scheduled_gains: set to True to use different gains in the throat than within the nozzle
    '''

    n = 1
    # For planar admittance tests with low stiffness
    #self.Mp_inv = np.diag(1/np.array([n*99999,n*100,n*400]))
    #self.Bp = np.diag(np.array([n*99999,n*100,n*400]))
    #self.Kp = np.diag(np.array([n*99999,n*1,n*400]))]

    self.Mp_inv_stiff = np.diag(1/np.array([n*9999,n*9999,n*9999]))
    self.Bp_stiff = np.diag(np.array([n*9999,n*9999,n*9999]))
    self.Kp_stiff = np.diag(np.array([n*9999,n*9999,n*9999]))

    # Gains used before switching to damping only implementation
    #self.Mp_inv = np.diag(1/np.array([n*2500,n*2500,n*1000]))
    #self.Bp = np.diag(np.array([n*3500,n*3500,n*1000]))
    #self.Kp = np.diag(np.array([n*1000,n*1000,n*300]))

    self.Mp_inv_throat = np.diag(1/np.array([n*6000,n*6000,n*2000]))
    self.Bp_throat = np.diag(np.array([n*5000,n*5000,n*2000]))
    self.Kp_throat = np.diag(np.array([n*1200,n*1200,n*300]))

    n = 1.75
    self.Mp_inv_nozzle = np.diag(1/np.array([n*200,n*200,n*400]))
    self.Bp_nozzle = np.diag(np.array([n*200,n*200,n*400]))
    self.Kp_nozzle = np.diag(np.array([n*0,n*0,n*400]))

    '''Orientation compliance not being used'''
    #self.Mtheta_inv = np.diag(1/np.array([600,600,60000]))
    #self.Btheta = np.diag(np.array([600,600,60000]))
    #self.Ktheta = np.diag(np.array([300,300,30000]))

    self.last_v_d = np.zeros(3)
    self.last_pos_d = np.zeros(3)
    self.last_omega_d = np.zeros(3)
    self.last_rmat_d = np.identity(3)

    self.use_admittance = True

    # Warning: Orientation admittance causes tracking to fail right now
    self.use_orientation_admittance = False

    self.use_scheduled_gains = use_scheduled_gains
    
    self.resolved_accel_base = ResolvedAccelBase()

  def get_gains(self,in_throat,past_nozzle): 
    '''Inputs:
        in_throat: True if peg is in the throat of the nozzle
        past_nozzle: True if peg is past the nozzle opening'''

    if self.use_scheduled_gains: 
      if in_throat:
        Mp_inv = self.Mp_inv_throat
        Bp = self.Bp_throat
        Kp = self.Kp_throat
      elif past_nozzle: 
        Mp_inv = self.Mp_inv_nozzle
        Bp = self.Bp_nozzle
        Kp = self.Kp_nozzle
      else: 
        Mp_inv = self.Mp_inv_stiff
        Bp = self.Bp_stiff
        Kp = self.Kp_stiff
    else: 
        if past_nozzle:
          Mp_inv = self.Mp_inv_nozzle
          Bp = self.Bp_nozzle
          Kp = self.Kp_nozzle
        else:
          Mp_inv = self.Mp_inv_stiff
          Bp = self.Bp_stiff
          Kp = self.Kp_stiff

    return Mp_inv, Bp, Kp
  
  def get_cond_number(self,mrv_client_sim):
    Jstar, _= mrv_client_sim.get_mrv_generalized_jacobian()
    cond = np.linalg.cond(Jstar)
    return cond
    
  def reset_admittance_traj(self,mrv_client_sim,init_v_d,init_omega_d):
    '''Reset the admittance controller history

    Inputs:
      mrv_client_sim: MRVClientSim object
    '''
    self.last_v_d = init_v_d
    self.last_pos_d = mrv_client_sim.get_mrv_tip_pos()
    self.last_omega_d = init_omega_d
    self.last_rmat_d = mrv_client_sim.get_mrv_tip_rmat()

  def compute_control(self, ref_traj_point, mrv_client_sim, wrench_peg_peg, dt, mrv_config, mrv_config_dot):
      '''
      Inputs:
      
      ref_traj_point: RefTrajPoint object representing a single point in the reference trajectory for the current time step
      mrv_client_sim: MRVClientSim object 
      wrench_peg_peg: wrench applied to the probe, expressed in the probe frame
      mrv_config: MRVClientSim.get_mrv_config()
      mrv_config_dot: MRVClientSim.get_mrv_config_dot()
      dt: time step'''

      # Get the wrench at the probe tip, expressed in a frame collocated with the probe tip and aligned with the world frame
      #wrench_peg_world = transform_wrench(wrench_peg_peg,np.zeros(3),np.eye(3,3),np.zeros(3),np.transpose(mrv_client_sim.get_mrv_tip_rmat()))

      # This line is suspect - what direction do we assume the wrenches are? Are they on the robot or on the nozzle?
      wrench_error = wrench_peg_peg - ref_traj_point.wrench 

      new_ref_traj_point = ref_traj_point

      if self.use_admittance:
        '''Comply on translation'''
        force_err = wrench_error[:3]
        Mp_inv, Bp, Kp = self.get_gains(mrv_client_sim.is_peg_in_throat(depth_offset = 0.010),mrv_client_sim.is_peg_past_nozzle_opening())
        vdot_d = ref_traj_point.vdot + Mp_inv@(force_err - Bp@(self.last_v_d - ref_traj_point.v) - Kp@(self.last_pos_d - ref_traj_point.p))
        v_d = self.last_v_d + vdot_d*dt
        p_d = self.last_pos_d + v_d*dt

        self.last_v_d = v_d
        self.last_pos_d = p_d

        new_ref_traj_point.p = p_d
        new_ref_traj_point.v = v_d
        new_ref_traj_point.vdot = vdot_d
   
        '''Comply on orientation'''
        if self.use_orientation_admittance: 
          raise Exception("Orientation compliance does not work and needs to be debugged.")
          moment_err = wrench_error[3:]
          orientation_disturbance = vee3(scipy.linalg.logm(np.transpose(self.last_rmat_d)@ref_traj_point.rmat))
          omega_dot_d = ref_traj_point.omega_dot + self.Mtheta_inv@(moment_err - self.Btheta@(self.last_omega_d - ref_traj_point.omega) - self.Ktheta@orientation_disturbance)
          omega_d = self.last_omega_d + omega_dot_d*dt
          rmat_d = self.last_rmat_d@scipy.linalg.expm(hat3(omega_d*dt))
          self.last_omega_d = omega_d
          self.last_rmat_d = rmat_d

      joint_acc_cmd = self.resolved_accel_base.compute_control(new_ref_traj_point, mrv_client_sim, wrench_peg_peg,mrv_config, mrv_config_dot)
      return joint_acc_cmd
