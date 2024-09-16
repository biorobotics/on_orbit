import numpy as np
#from ref_traj_point import RefTrajPoint
#from scipy.spatial.transform import Rotation as R
#from kin_util import transform_wrench, vee3, hat3
#import scipy 

from resolved_accel_base import ResolvedAccelBase
from resolved_accel import ResolvedAccel
from ref_traj_point import RefTrajPoint
from planar_impedance import PlanarImpedance

'''
This controller does not rely on the same type of reference trajectory as other controllers.'''

class PlanarAdmittance(object): 

  def __init__(self,use_scheduled_gains): 

    self.goal_z_offset = 0.150 # extra offset added to goal_pos along z-axis of goal frame, added so that we ensure probe goes all the way into the hole
    self.throat_length = 0.0475
    self.nozzle_length = 0.115 - self.throat_length
    self.nozzle_opening_from_goal = 0.115 + self.goal_z_offset
    self.max_ax_deviation = 3/1000

    self.resolved_accel_base = ResolvedAccelBase()
    self.resolved_accel_base.use_min_norm = True

    scale = 10
    pos_gain = scale*1.0
    ang_gain = scale*1
    pos_vel_gain = scale*1
    ang_vel_gain = scale*1
    self.resolved_accel_base.set_gains(pos_gain,ang_gain,pos_vel_gain,ang_vel_gain)

    self.resolved_accel = ResolvedAccel(use_scheduled_gains)
    self.resolved_accel.resolved_accel_base.use_min_norm = True
    self.resolved_accel.resolved_accel_base.set_gains(pos_gain,ang_gain,pos_vel_gain,ang_vel_gain)
 
    '''Parameters for base of the curve'''
    self.last_ax = 0
    self.last_ax_dot = 0
    self.B_ax_inv = 1/0.008
    self.K_ax = 0.03

    self.alpha_step_vel = 3
    self.ax_step_vel = 0.1

    '''Parameters of the curve to be followed
    See Andrew's (currently handwritten) notes'''
    self.pb_x = 0
    self.pb_y = self.throat_length + self.goal_z_offset
    self.l_0 = self.throat_length + self.goal_z_offset
    self.l_1 = self.nozzle_length

    '''Parameters for angle of curve'''
    self.alpha_nom = np.pi
    self.last_alpha = self.alpha_nom
    self.last_alpha_dot = 0
    self.B_alpha_inv = 1/0.05
    self.K_alpha = 0.5

    self.nom_speed = 0.005 # magnitude of tangent velocity along curve, m/s
    self.throat_speed = 0.015
    self.last_s = self.l_0 + self.l_1

    self.planar_impedance = PlanarImpedance()

    self.use_shape_admittance = False
    self.use_angle_admittance = False
    self.use_base_admittance = False


  def update_s(self,dt,nom_speed):
    self.last_s -= nom_speed*dt

    if self.last_s < 0:
      self.last_s = 0
    elif self.last_s > self.l_0 + self.l_1:
      self.last_s = self.l_0 + self.l_1 

  def get_torque_from_force(self,s,force): 
    '''Force is expressed in curve frame'''
    if s <= self.l_0:
      tau = 0
    else:
      beta = self.get_beta(self.last_alpha)
      normal_vector = np.array([np.cos(beta), np.sin(beta)])

      # Tau as a torque about the throat entrace
      tau = np.dot(force,normal_vector) * (s - self.l_0)
      #tau = force[0]*(s - self.l_0)*np.cos(beta) + force[1]*(s - self.l_0)*np.sin(beta)

      # Tau as a torque about the edge of the nozzle opening
      tau = np.dot(force,normal_vector) * (self.l_0 + self.l_1 - s)
      # tau = force[0]*(self.l_0 + self.l_1 - s)*np.cos(beta) + force[1]*(self.l_0 + self.l_1 - s)*np.sin(beta)

      # Tau as the norm of the projection of the force perpindicular to the curve 
      #tau = np.dot(force,normal_vector)

    return tau
  
  def get_beta(self,alpha):
    return np.pi - alpha

  def get_curve_pos(self,s,alpha):
    '''Return position along curve'''
    if s <=self.l_0:
      p_x = self.last_ax
      p_y = -s
    else:
      p_x = self.last_ax + self.pb_x + (s-self.l_0)*np.sin(self.get_beta(alpha))
      p_y = -self.pb_y - (s-self.l_0)*np.cos(self.get_beta(alpha))

    return np.array([p_x,p_y])
  
  def reset_admittance_traj(self,mrv_client_sim):
    peg_pos, peg_rmat, peg_twist, nozzle_pos, nozzle_rmat, nozzle_twist = mrv_client_sim.get_peg_and_nozzle_info()
    angular_vel = nozzle_twist[3]
    client_pos = mrv_client_sim.get_client_pos()
    client_to_peg = peg_pos - client_pos
    linear_vel = np.cross(nozzle_twist[3:6],client_to_peg)

    self.resolved_accel.reset_admittance_traj(mrv_client_sim,linear_vel,nozzle_twist[3:6])
  
  def get_curve_vel(self,s,alpha,nom_speed):
    '''Return the velocity the robot should execute, tangent to the curve, assuming peg was exactly on the curve at s.'''
    if s <= self.l_0:
      dir_x = 0
      dir_y = 1
    else:
      dir_x = -np.sin(self.get_beta(alpha))
      dir_y = np.cos(self.get_beta(alpha))

    dir = np.array([dir_x,dir_y])
    dir = dir/np.linalg.norm(dir)

    return nom_speed*(dir)

  #def get_s(self,peg_pos,nozzle_pos,nozzle_rmat):
  #  raise Exception("This is obsolete and needs to be checked before using.")
  #  pos_goal = nozzle_pos + nozzle_rmat[:, 2]*self.nozzle_opening_from_goal
  #  rel_pos = nozzle_rmat.transpose()@(peg_pos - pos_goal)
  #  return rel_pos[2]

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

      # Add noisy measurements in y-z plane of nozzle only, for purposes of running a planar experiment
      nozzle_pos[1:3] = nozzle_pos_est[1:3]
      nozzle_rmat[:,1:3] = nozzle_rmat_est[:,1:3]
      nozzle_twist[1:3] = nozzle_twist_est[1:3]
      nozzle_twist[4:6] = nozzle_twist_est[4:6]

      '''Damping and stiffness'''

      force_client_client = -np.transpose(nozzle_rmat)@peg_rmat@wrench_peg_peg[0:3]
      force_err = np.array([force_client_client[1],force_client_client[2]])
      print(f'force_client_client: {force_client_client}')
      
      client_imp_force = self.planar_impedance.compute_force(nozzle_pos, nozzle_twist[0:3], mrv_client_sim.initial_nozzle_pos, np.zeros(3))
      #print(f'client_imp_force: {client_imp_force}')
      wrench_err_peg_peg = np.copy(wrench_peg_peg)
      wrench_err_peg_peg[0:3] = wrench_err_peg_peg[0:3] - client_imp_force[0:3]
      #print(f'wrench_err_peg_peg[0:3] {wrench_err_peg_peg[0:3] }')
      #print("force_client_client")
      #print(force_client_client)

      if mrv_client_sim.is_peg_in_throat(0.00):
        nom_speed = self.throat_speed
      else:
        nom_speed = self.nom_speed

      if self.use_shape_admittance or self.use_angle_admittance or self.use_base_admittance:
        alpha_dot = 0
        ax_dot = 0

        if self.use_shape_admittance:
            if self.use_angle_admittance:
              alpha_dot = self.B_alpha_inv*(self.get_torque_from_force(self.last_s,force_err) - self.K_alpha*(self.last_alpha - self.alpha_nom))

            if self.use_base_admittance: 
              ax_dot = self.B_ax_inv*(force_err[0] - self.K_ax*self.last_ax)  
        else: 
            if np.linalg.norm(force_err) > 1e-4:
              if self.use_angle_admittance:
                  alpha_dot = self.alpha_step_vel*(self.get_torque_from_force(self.last_s,force_err))
              if self.use_base_admittance:
                  ax_dot = self.ax_step_vel*(-force_err[0])

        alpha = self.last_alpha + alpha_dot*dt    

        if alpha < np.pi/2: 
          alpha = np.pi/2

        if alpha > 3*np.pi/2:
          alpha = 3*np.pi/2

        self.last_alpha_dot = alpha_dot
        self.last_alpha = alpha

        ax = self.last_ax + ax_dot*dt

        if ax < -self.max_ax_deviation:
          ax = -self.max_ax_deviation

        if ax > self.max_ax_deviation:
          ax = self.max_ax_deviation

        self.last_ax_dot = ax_dot
        self.last_ax = ax

        #print("alpha")
        #print(self.last_alpha*180/np.pi)
        #print("torque")
        #print(self.get_torque_from_force(self.last_s,force_err))
      
        p2d = self.get_curve_pos(self.last_s,self.last_alpha)
        v2d = self.get_curve_vel(self.last_s,self.last_alpha,nom_speed)
      else:
        p2d = self.get_curve_pos(self.last_s,np.pi)
        v2d = self.get_curve_vel(self.last_s,np.pi,nom_speed)

      '''INITIAL TESTING: Add an offset to the nozzle to simulate error in pose'''
      #offset = np.array([0,0.0,0]) 
      #nozzle_pos = nozzle_pos + offset
      #nozzle_pos_goal = mrv_client_sim.initial_nozzle_pos + offset
      nozzle_pos_for_goal = nozzle_pos

      goal_pos = nozzle_pos_for_goal + nozzle_rmat@np.array([0,0,self.nozzle_opening_from_goal])
      goal_pos = goal_pos + nozzle_rmat@np.array([0,0,0.015]) #modified experiment.yaml to put probe deeper into the nozzle
      
      p = goal_pos + nozzle_rmat@np.array([0,p2d[0],p2d[1]]) #Note p2d[1] is typically negative

      nozzle_vel_client_frame = np.transpose(nozzle_rmat)@nozzle_twist[0:3] 
      v = nozzle_rmat@(np.array([0,v2d[0],v2d[1]]) + np.array([0,0,nozzle_vel_client_frame[2]])) 

      #print("p")
      #print(p)
      #print("v")
      #print(v)
      #print("peg")
      #print(peg_pos)

      #print("p")
      #print(p)
      #print("alpha")
      #print(self.last_alpha)
      #print("ax")
      #print(self.last_ax)
      #print("peg_pos")
      #print(peg_pos)
      #print("v")
      #print(v)
      
      traj_point = RefTrajPoint()
      traj_point.p = p
      traj_point.v = v
      traj_point.vdot = np.array([0,0,0])

      traj_point.rmat = nozzle_rmat
      traj_point.omega = nozzle_twist[3:]
      traj_point.omega_dot = np.zeros(3)
      traj_point.wrench = np.zeros(6)

      self.update_s(dt,nom_speed)

      #joint_acc_cmd = self.resolved_accel_base.compute_control(traj_point, mrv_client_sim, mrv_config, mrv_config_dot)
      joint_acc_cmd = self.resolved_accel.compute_control(traj_point, mrv_client_sim, wrench_err_peg_peg, dt, mrv_config, mrv_config_dot)

      return joint_acc_cmd

