import numpy as np
#from ref_traj_point import RefTrajPoint
#from scipy.spatial.transform import Rotation as R
#from kin_util import transform_wrench, vee3, hat3
#import scipy 

from resolved_accel import ResolvedAccel
from ref_traj_point import RefTrajPoint
from planar_impedance import PlanarImpedance

'''
This controller does not rely on the same type of reference trajectory as other controllers.'''

class WithinNozzleAdmittance(object): 

  def __init__(self, probe_z_axis_plunge_velocity, use_variable_plunge_speed, use_scheduled_gains): 
    '''
    Args:
      probe_z_axis_plunge_velocity: nominal plunge velocity along the probe's z-axis (m/s) '''

    self.goal_z_offset = 0.150 # extra offset added to goal_pos along z-axis of goal frame, added so that we ensure probe goes all the way into the hole
    self.throat_length = 0.0475
    self.nozzle_length = 0.1785 - self.throat_length
    self.nozzle_opening_from_goal = 0.4785 + self.goal_z_offset

    self.resolved_accel = ResolvedAccel(use_scheduled_gains)
    self.resolved_accel.resolved_accel_base.use_min_norm = True

    '''Parameters of the curve to be followed
    See Andrew's (currently handwritten) notes'''
    self.l_0 = self.throat_length + self.goal_z_offset
    self.l_1 = self.nozzle_length

    self.probe_z_axis_plunge_velocity = probe_z_axis_plunge_velocity # magnitude of tangent velocity along curve, m/s

    self.use_variable_plunge_speed = use_variable_plunge_speed

    if self.use_variable_plunge_speed:
      self.throat_speed = 5*self.probe_z_axis_plunge_velocity
    else: 
      self.throat_speed = self.probe_z_axis_plunge_velocity

    self.last_s = self.l_0 + self.l_1

    self.planar_impedance = PlanarImpedance()

    self.admittance_traj_reset = False

    self.set_initial_client_xy_pos = True
    self.initial_client_x_client = None
    self.initial_client_y_client = None

  def update_s(self,dt,nom_speed):
    self.last_s -= nom_speed*dt

    if self.last_s < 0:
      self.last_s = 0
    elif self.last_s > self.l_0 + self.l_1:
      self.last_s = self.l_0 + self.l_1 
  
  def get_curve_pos(self,s):
    '''Return position along curve'''
    if s <=self.l_0:
      p_x = 0
      p_y = -s
    else:
      p_x = 0
      p_y = - self.throat_length - self.goal_z_offset - (s-self.l_0)

    return np.array([p_x,p_y])
  
  def reset_admittance_traj(self,mrv_client_sim):
    peg_pos, peg_rmat, peg_twist, nozzle_pos, nozzle_rmat, nozzle_twist = mrv_client_sim.get_peg_and_nozzle_info()
    angular_vel = nozzle_twist[3]
    client_pos = mrv_client_sim.get_client_pos() 
    client_to_peg = peg_pos - client_pos
    linear_vel = np.cross(nozzle_twist[3:6],client_to_peg)

    self.resolved_accel.reset_admittance_traj(mrv_client_sim,linear_vel,nozzle_twist[3:6])
    self.admittance_traj_reset = True
  
  def compute_control(self, ref_traj_point, mrv_client_sim, wrench_peg_peg, dt, mrv_config, mrv_config_dot):
      '''
      Inputs:
      
      ref_traj_point: RefTrajPoint object representing a single point in the reference trajectory for the current time step
      sw_sim: MRVClientSim object 
      wrench_peg_peg: wrench applied to the probe, expressed in the probe frame
      mrv_config: MRVClientSim.get_mrv_config()
      mrv_config_dot: MRVClientSim.get_mrv_config_dot()
      dt: time step'''
      
      peg_pos_est, peg_rmat_est, peg_twist_est, nozzle_pos_est, nozzle_rmat_est, nozzle_twist_est = mrv_client_sim.get_peg_and_nozzle_info_est()
      peg_pos, peg_rmat, peg_twist, nozzle_pos, nozzle_rmat, nozzle_twist = mrv_client_sim.get_peg_and_nozzle_info()

      if self.set_initial_client_xy_pos:
        peg_pos_client = np.transpose(nozzle_rmat)@peg_pos_est
        self.initial_client_x_client = peg_pos_client[0]
        self.initial_client_y_client = peg_pos_client[1]
        self.set_initial_client_xy_pos = False

      # Use noisy measurements of nozzle
      nozzle_pos = nozzle_pos_est
      nozzle_rmat= nozzle_rmat_est
      nozzle_twist = nozzle_twist_est
      nozzle_twist = nozzle_twist_est

      '''Damping and stiffness'''
      client_imp_force = self.planar_impedance.compute_force(nozzle_pos, nozzle_twist[0:3], mrv_client_sim.initial_nozzle_pos, np.zeros(3))
      wrench_err_peg_peg = np.copy(wrench_peg_peg)
      wrench_err_peg_peg[0:3] = wrench_err_peg_peg[0:3] - client_imp_force[0:3]

      dist_to_throat = mrv_client_sim.dist_to_throat_opening()
      if dist_to_throat < 0.01:
        nom_speed = self.throat_speed
      else:
        nom_speed = self.probe_z_axis_plunge_velocity

      
      #print(f'nom_speed: {nom_speed}')
      p2d = self.get_curve_pos(self.last_s)

      '''Use full 3D position of client to determine goal position'''
      #goal_pos = nozzle_pos + nozzle_rmat@np.array([0,0,self.nozzle_opening_from_goal])

      '''Ignore changes in x/y position since it's noisy anyway'''
      nozzle_pos_client = np.transpose(nozzle_rmat)@nozzle_pos
      goal_pos_client = np.array([self.initial_client_x_client,self.initial_client_y_client,nozzle_pos_client[2]]) + np.array([0,0,self.nozzle_opening_from_goal]) #ignore x/y position because it's noisy anyway
      goal_pos = nozzle_rmat@goal_pos_client
      #goal_pos = goal_pos + nozzle_rmat@np.array([0,0,0.015]) #modified experiment.yaml to put probe deeper into the nozzle
      
      p = goal_pos + nozzle_rmat@np.array([0,p2d[0],p2d[1]]) #Note p2d[1] is typically negative

      nozzle_vel_client_frame = np.transpose(nozzle_rmat)@nozzle_twist[0:3] 
      v = nozzle_rmat@(np.array([0,0,nom_speed + nozzle_vel_client_frame[2]])) 

      traj_point = RefTrajPoint()
      traj_point.p = p
      traj_point.v = v
      traj_point.vdot = np.array([0,0,0])

      traj_point.rmat = peg_rmat
      traj_point.omega = np.zeros(3)
      traj_point.omega_dot = np.zeros(3)
      traj_point.wrench = np.zeros(6)

      self.update_s(dt,nom_speed)

      joint_acc_cmd = self.resolved_accel.compute_control(traj_point, mrv_client_sim, wrench_err_peg_peg, dt, mrv_config, mrv_config_dot)

      return joint_acc_cmd

