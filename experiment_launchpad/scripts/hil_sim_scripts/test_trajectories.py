from ref_traj_point import RefTrajPoint
import numpy as np

class TestTrajectories(object): 

  def __init__(self,traj_type, init_mrv_tip_pos, init_mrv_joint_angles):
   '''Inputs:
   
   traj_type: 
   0 - sinusoid
   1 - step displacement in position
   '''
   self.traj_type = traj_type

   self.init_mrv_tip_pos = init_mrv_tip_pos
   self.init_mrv_joint_angles = init_mrv_joint_angles

  def get_traj_point(self, time):
      ref_traj_point = RefTrajPoint()

      if self.traj_type == 0:
        amp = 0.0
        freq = 0.05
        omega = freq*2*np.pi
        ref_traj_point.p = self.init_mrv_tip_pos + [amp*np.sin(omega*time),0,0]
        ref_traj_point.v = [amp*omega*np.cos(omega*time),0,0]
        ref_traj_point.vdot = [-amp*omega*omega*np.sin(omega*time),0,0]

      elif self.traj_type == 1:

        dx = 0.0
        dy = 0.2
        dz = 0
        ref_traj_point.p = self.init_mrv_tip_pos + [dx,dy,dz]
        ref_traj_point.v = [0,0,0]
        ref_traj_point.vdot = [0,0,0]

      ref_traj_point.rmat = np.eye(3)
      ref_traj_point.omega = [0,0,0]
      ref_traj_point.omega_dot = [0,0,0]

      ref_traj_point.theta = self.init_mrv_joint_angles
      ref_traj_point.theta_dot = 0.0*self.init_mrv_joint_angles
      ref_traj_point.theta_ddot = 0.0*self.init_mrv_joint_angles
    
      '''Choose the desired wrench'''
      ref_traj_point.wrench = [0,0,0,0,0,0]
      return ref_traj_point

