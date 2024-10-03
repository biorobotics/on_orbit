#!/usr/bin/env python3

from ipopt_contact_planner import IpoptContactPlanner          # For slide-to-hole
from ipopt_contact_free_planner import IpoptContactFreePlanner # For contact-free
from ipopt_nozzle_align_planner import IpoptNozzleAlignPlanner  # For aligning with nozzle frame
from ipopt_contact_align_planner import IpoptContactAlignPlanner  # For slide-and-align planning

import numpy as np
import time
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R

import rospkg
import os

import sys
sys.path.append('../')
sys.path.append('/home/medusar/bspin/on_orbit/catkin_ws/devel/lib/python3/dist-packages/')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'hil_sim_scripts')))
from hil_sim_scripts.mrv_controller import MrvController


class IpoptTrajopt():
  
  def __init__(self,save_path_prefix,do_save):

    self.do_save = do_save

    np.set_printoptions(linewidth=np.inf)
    np.set_printoptions(suppress=True)

    self.dist_centering_waypoint_from_goal = 0.481024428 
    self.dist_capture_box_from_nozzle_opening = 0.0

    self.save_path_prefix = save_path_prefix

    rospack = rospkg.RosPack()
    self.rospath = rospack.get_path('on_orbit')
    
    # Confirm this is equal to tan(pi/2 - slope_angle) where slope angle is found in sim_nozzle_geom
    # We narrow the cone by 1 degree so that it avoids getting too close to the edge of the nozzle
    self.cone_slope = np.tan(np.pi/2. - 1.0*0.3073)

    robustness_factor = 0.75
    #mrv_joint_angle_lower_limits = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    #mrv_joint_angle_upper_limits = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    self.mrv_joint_angle_lower_limits= np.array([-2.792526803, -1.745329252, -2.967059728, -2.967059728, -4.36332313, -1.919862177, -4.310963252])
    self.mrv_joint_angle_upper_limits =  np.array([2.792526803, 1.745329252, 2.565634, 2.617993878, 1.221730476, 1.919862177, 1.169370599])

    # self.mrv_joint_vel_limits = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    #self.mrv_joint_acc_limits = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    self.mrv_joint_vel_limits = robustness_factor*np.array([0.069813, 0.069813, 0.10472, 0.10472, 0.12217, 0.12217, 0.12217])
    self.mrv_joint_acc_limits = robustness_factor*np.array([0.12217, 0.12217, 0.22689, 0.22689, 0.3316, 0.33161, 0.33161])


    #self.mrv_joint_torque_limits = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
    self.mrv_joint_torque_limits = np.array([150, 150, 150, 150, 150, 150, 150])

    self.cw_a = 6793137
    self.cw_mu = 3.986e+14
    self.cw_orbit_dir = 'x'

    self.peg_rad = 0.008
    self.nozzle_opening_rad = 0.147

    self.use_cw = True

  def plan(self,delta_pos,delta_rot,initial_client_w,use_contact,nozzle_align=False):
    '''
    Inputs:
      delta_pos: length 3 numpy array for initial position of probe tip from front/center of capture box. 
      delta_rot: length 3 numpy array for initial rotation of probe tip relative to capture box'''
  
    save_path = None
    success = False

    delta_v = np.array([0., 0., 0.])
    initial_mrv_w = np.array([0.0, 0.0, 0.0])*np.pi/180

    prop_time = 0.
    stop_after_iter = -1

    timestr = time.strftime("%Y%m%d-%H%M%S")

    save_path_str = '/' + '_'.join(['pos', str(delta_pos[0]), str(delta_pos[1]), str(delta_pos[2]), 
                                    'rot', str(delta_rot[0]), str(delta_rot[1]), str(delta_rot[2]), 
                                    'delta_v', str(delta_v[0]), str(delta_v[1]), str(delta_v[2]), 
                                    'client_w', str(initial_client_w[0]*180/np.pi), str(initial_client_w[1]*180/np.pi), str(initial_client_w[2]*180/np.pi)])
    
    save_path = self.rospath + '/' + self.save_path_prefix + save_path_str + '/control/' + timestr + '/'

    # The pybullet sim is only used to get the iniitial client state. These parameters are otherwise not important.
    use_pose_noise = True
    time_steps_between_measurements = 10
    do_noisy_state_estimation = False
    client_velocity_noise_ang_amp = 0
    rng = np.random.default_rng(np.random.SeedSequence(12345).spawn(1 + 1)[-1])
    clip_joint_commands = False

    time_limit = 500
    debug_with_test_traj= False
    test_traj_id = 0
    lock_client = False
    lock_mrv = False
    
    mrv_controller = MrvController(self.rospath + '/urdf/robot_cv_detached.urdf', 
                                       self.rospath + '/urdf/robot.urdf', 
                                       self.rospath + '/urdf/robot.urdf', 
                                       self.rospath + '/urdf/cv.urdf', 
                                       self.mrv_joint_angle_lower_limits, self.mrv_joint_angle_upper_limits, 
                                       self.mrv_joint_vel_limits, self.mrv_joint_acc_limits, 
                                       self.mrv_joint_torque_limits, 0.01, 
                                       self.cone_slope, clip_joint_commands,
                                       time_steps_between_measurements, self.cw_a, self.cw_mu, self.cw_orbit_dir, do_noisy_state_estimation, 
                                       self.nozzle_opening_rad, self.peg_rad, client_velocity_noise_ang_amp, time_limit, 
                                       debug_with_test_traj, test_traj_id, lock_client, lock_mrv,
                                       probe_z_axis_plunge_velocity=0.005, use_variable_plunge_speed=True, 
                                       use_scheduled_gains=True, use_cw=self.use_cw)
    
    mrv_controller.reset_wrt_capture_box(None, None, delta_pos, delta_rot, delta_v, initial_client_w, initial_mrv_w, rng, self.dist_centering_waypoint_from_goal)

    if self.do_save:
        os.makedirs(save_path)
    else:
        save_path = None

    x0 = np.copy(mrv_controller.mrv_client_sim.x)

    ecm_bezier_sim = None 

    initial_client_rmat = R.from_quat(mrv_controller.mrv_client_sim.x[mrv_controller.cv_qidx + 3:mrv_controller.cv_qidx + 7]).as_matrix()

    control_cost_weight = 0.0001

    if use_contact and not nozzle_align:

      phase_lengths_sec = np.array([5,5,7,7])
      dt = 0.2

      planner = IpoptContactPlanner(self.rospath + '/urdf/robot_cv_detached.urdf', \
                                    self.rospath + '/urdf/robot.urdf', \
                                    dt, \
                                    self.mrv_joint_angle_lower_limits, \
                                    self.mrv_joint_angle_upper_limits, \
                                    self.mrv_joint_torque_limits, \
                                    self.mrv_joint_vel_limits, \
                                    self.mrv_joint_acc_limits, \
                                    control_cost_weight, phase_lengths_sec, \
                                    self.cw_a, self.cw_mu, self.cw_orbit_dir, initial_client_rmat, \
                                    self.cone_slope, self.use_cw, self.rospath + '/meshes/')
      
      xs, us, dts, phase_starts, success = planner.plan(x0, 
                                                      ecm_bezier_sim, 
                                                      [delta_pos, delta_rot, delta_v, initial_client_w, initial_mrv_w, None], 
                                                      save_path=save_path, 
                                                      count_flop_per_iter=False, 
                                                      count_total_flop=False, 
                                                      stop_after_iter=stop_after_iter, 
                                                      prop_time=prop_time,
                                                      max_iter = 2000)
    
    elif use_contact and nozzle_align:
      
      phase_lengths_sec = np.array([5,5,7,4,0.4])
      dt = 0.2

      planner = IpoptContactAlignPlanner(self.rospath + '/urdf/robot_cv_detached.urdf', \
                                    self.rospath + '/urdf/robot.urdf', \
                                    dt, \
                                    self.mrv_joint_angle_lower_limits, \
                                    self.mrv_joint_angle_upper_limits, \
                                    self.mrv_joint_torque_limits, \
                                    self.mrv_joint_vel_limits, \
                                    self.mrv_joint_acc_limits, \
                                    control_cost_weight, phase_lengths_sec, \
                                    self.cw_a, self.cw_mu, self.cw_orbit_dir, initial_client_rmat, \
                                    self.cone_slope, self.use_cw, self.rospath + '/meshes/')
      
      xs, us, dts, phase_starts, success = planner.plan(x0,
                                                      ecm_bezier_sim,
                                                      [delta_pos, delta_rot, delta_v, initial_client_w, initial_mrv_w, None],
                                                      save_path=save_path,
                                                      count_flop_per_iter=False,
                                                      count_total_flop=False,
                                                      stop_after_iter=stop_after_iter,
                                                      prop_time=prop_time,
                                                      max_iter = 1000)
                                    
    elif nozzle_align:

      phase_scaling = 1
      num_nodes = 200
      phase1_sec = 4
      phase2_sec = 11
      phase3_sec = 0.5

      total_sec = (phase1_sec + phase2_sec + phase3_sec)*phase_scaling
      dt = total_sec/num_nodes #try to keep number of time steps to around 150-200
      # dt = 0.01
      print("dt: ", dt)

      if dt > 0.3: 
        print("Warning: dt is greater than 0.3 due to a large trajectory time.")

      # Try to keep number of time steps to around 150-200 
      phase_lengths_sec = np.array([phase_scaling*phase1_sec, phase_scaling*phase2_sec, phase_scaling*phase3_sec, dt])

      planner = IpoptNozzleAlignPlanner(self.rospath + '/urdf/robot_cv_detached.urdf', \
                                      self.rospath + '/urdf/robot.urdf', \
                                      dt, \
                                      self.mrv_joint_angle_lower_limits, \
                                      self.mrv_joint_angle_upper_limits, \
                                      self.mrv_joint_torque_limits, \
                                      self.mrv_joint_vel_limits, \
                                      self.mrv_joint_acc_limits, \
                                      control_cost_weight, phase_lengths_sec, \
                                      self.cw_a, self.cw_mu, self.cw_orbit_dir, initial_client_rmat, \
                                      self.cone_slope, self.use_cw, self.rospath + '/meshes/')

      xs, us, dts, phase_starts, success = planner.plan(x0, 
                                                      ecm_bezier_sim, 
                                                      [delta_pos, delta_rot, delta_v, initial_client_w, initial_mrv_w, None], 
                                                      save_path=save_path, 
                                                      count_flop_per_iter=False, 
                                                      count_total_flop=False,  
                                                      stop_after_iter=stop_after_iter, 
                                                      prop_time=prop_time,
                                                      max_iter = 1500)

    
    else:
      # Phases:
      # 0 - stay below nozzle opening plane
      # 1 - stay in nozzle cone 
      # 2 - stay in hole
      # 3 - insert
      phase_scaling = 1
      num_nodes = 150
      phase1_sec = 4
      phase2_sec = 15
      phase3_sec = 4
      total_sec = (phase1_sec + phase2_sec + phase3_sec)*phase_scaling
      dt = total_sec/num_nodes #try to keep number of time steps to around 150-200
      print("dt: ", dt)

      if dt > 0.3: 
        print("Warning: dt is greater than 0.3 due to a large trajectory time.")

      if np.linalg.norm(initial_client_w) < 1e-6:
        pause_in_nozzle = False
      else:
        pause_in_nozzle = False

      # Try to keep number of time steps to around 150-200 
      phase_lengths_sec = np.array([phase_scaling*phase1_sec,phase_scaling*phase2_sec,phase_scaling*phase3_sec,dt])

      planner = IpoptContactFreePlanner(self.rospath + '/urdf/robot_cv_detached.urdf', \
                                      self.rospath + '/urdf/robot.urdf', \
                                      dt, \
                                      self.mrv_joint_angle_lower_limits, \
                                      self.mrv_joint_angle_upper_limits, \
                                      self.mrv_joint_torque_limits, \
                                      self.mrv_joint_vel_limits, \
                                      self.mrv_joint_acc_limits, \
                                      control_cost_weight, phase_lengths_sec, pause_in_nozzle, \
                                      self.cw_a, self.cw_mu, self.cw_orbit_dir, initial_client_rmat, \
                                      self.cone_slope, self.use_cw, self.rospath + '/meshes/')

      xs, us, dts, phase_starts, success = planner.plan(x0, 
                                                      ecm_bezier_sim, 
                                                      [delta_pos, delta_rot, delta_v, initial_client_w, initial_mrv_w, None], 
                                                      save_path=save_path, 
                                                      count_flop_per_iter=False, 
                                                      count_total_flop=False,  
                                                      stop_after_iter=stop_after_iter, 
                                                      prop_time=prop_time,
                                                      max_iter = 2500)

    if not success:
      fail_reason = 'Planner failed'
      return False
    else:
      fail_reason = 'None'

      if self.do_save:
        np.save(save_path + '/xs.npy', xs)
        np.save(save_path + '/us.npy', us)
        np.save(save_path + '/dts.npy', dts)
        np.save(save_path + '/phase_starts.npy', phase_starts)
        np.save(save_path + 'success.npy', success)
        np.save(save_path + 'fail_reason.npy', fail_reason)
        np.save(save_path + 'phase_lengths_sec.npy', phase_lengths_sec)

    return True
