#!/usr/bin/env python3
import numpy as np
import time

from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose, PoseArray

import pickle
from scipy.spatial.transform import Rotation as R
from get_grid import get_grid
from trajlib_util import get_trajlib_load_paths, interp_trajectories_on_initial_client_w, interp_trajectories_on_delta_pos, interp_trajectories_on_init_client_state

import rospy
import rospkg

from sim_ros_vis_publisher import SimROSVisPublisher

import os
import gc

from mrv_controller import MrvController

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

class experiments:
  def __init__(self):
    self.dist_centering_waypoint_from_goal = rospy.get_param('dist_centering_waypoint_from_goal')
    self.verify_trajectory_visually = rospy.get_param('verify_trajectory_visually')
    self.dt = 0.01

    # self.save_path = rospy.get_param('save_path')
    self.rospack = rospkg.RosPack()
    self.rospath = self.rospack.get_path('on_orbit')
    self.rate = rospy.Rate(1/self.dt)
    self.mrv_hil_home_angles = np.array(rospy.get_param('mrv_hil_home_angles')) #home angles for the mrv
    self.client_hil_home_angles = np.array(rospy.get_param('client_hil_home_angles'))
    self.sim_vis_publisher = SimROSVisPublisher(self.rospath)
    self.cone_slope = rospy.get_param('cone_slope')
    self.mrv_joint_angle_lower_limits = np.array(rospy.get_param('joint_angle_lower_limits'))
    self.mrv_joint_angle_upper_limits = np.array(rospy.get_param('joint_angle_upper_limits'))
    self.mrv_joint_vel_limits = np.array(rospy.get_param('joint_vel_limits'))
    self.mrv_joint_acc_limits = np.array(rospy.get_param('joint_acc_limits'))
    self.mrv_joint_torque_limits = np.array(rospy.get_param('joint_torque_limits'))
    self.apply_wrench_only_when_close = rospy.get_param('apply_wrench_only_when_close') #what is this?
    self.client_velocity_noise_ang_amp = rospy.get_param('client_velocity_noise_ang_amp')*np.pi/180.
    self.cw_a = rospy.get_param('cw_a')
    self.cw_mu = rospy.get_param('cw_mu')
    self.cw_orbit_dir = rospy.get_param('cw_orbit_dir')
    self.use_cw = False
    # self.use_cw = rospy.get_param('use_cw')

    self.do_noisy_state_estimation = rospy.get_param('do_noisy_state_estimation')
    self.clip_joint_commands = rospy.get_param('clip_joint_commands') #enforce joint limits
    self.time_limit = rospy.get_param('time_limit')
    self.debug_with_test_traj= rospy.get_param('debug_with_test_traj') #what is this?
    self.test_traj_id = rospy.get_param('test_traj_id')
    self.lock_client = rospy.get_param('lock_client')
    self.lock_mrv = rospy.get_param('lock_mrv')
    self.probe_z_axis_plunge_velocity = rospy.get_param('probe_z_axis_plunge_velocity')
    self.use_variable_plunge_speed = rospy.get_param('use_variable_plunge_speed')
    self.use_scheduled_gains = rospy.get_param('use_scheduled_gains')

    #write two publihsers for geometry msgs pose
    self.noz_pose_pub = rospy.Publisher('/nozzle', PoseArray, queue_size=0)
    self.peg_pose_pub = rospy.Publisher('/peg', PoseArray, queue_size=0)


    self.load_paths, self.ws, self.dps = get_trajlib_load_paths(self.rospath)

    self.base_seed = 12345

    #un tested
    # self.rng_sequences = np.random.SeedSequence(self.base_seed).spawn(self.num_trials)
    self.rng_sequences = np.random.SeedSequence(self.base_seed).spawn(100000)
    rospy.on_shutdown(self.on_shutdown)

    self.count = 0

    self.peg_pos = PoseArray()
    self.peg_pos.header.frame_id = 'world'
    self.peg_pos.poses = []
    self.noz_pos = PoseArray()
    self.noz_pos.header.frame_id = 'world'
    self.noz_pos.poses = []

    self.peg_rad = 0.01505
    self.nozzle_opening_rad = 0.142

    #my stuff
    self.poses = []
    self.rerun = False

    self.poses_sub = rospy.Subscriber('/poses', Float32MultiArray, self.poses_callback)


    #to be deleted
    self.grid_type = rospy.get_param('grid_type')
    self.ic_grid = get_grid(self.grid_type)  #get initial conditions grid (includes position, velocity, and angular velocity of the client and mrv)

    self.initial_grid_idx = rospy.get_param('initial_grid_idx')
    self.use_grid = rospy.get_param('use_grid')
    self.num_grid_idx = len(self.ic_grid)  #number of initial conditions
    self.final_grid_idx = self.num_grid_idx
    self.num_trials = rospy.get_param('num_hardware_trials')
    print("Initializations done!!!!!!!!!")

    self.poses=[0,0,0,0,0,0,0,0,0,0,0,0]
    self.run_experiment()

  def poses_callback(self, msg):
    print("Received poses")
    print(msg.data)
    self.poses = msg.data
    rospy.sleep(1)
    self.run_experiment()
    rospy.sleep(1)
  
  def on_shutdown(self):
    gc.enable()
    gc.collect()

  def run_experiment(self):
    if (len(self.poses)!=12):
      print("Please provide 12 poses")
      return
    rng = np.random.default_rng(self.rng_sequences[0])


    # self.hil_runner.calibrate_ft_bias()

    # if self.rerun:
    #   self.holo_control.ur_idle_mode('client')
    #   self.holo_control.ur_idle_mode('mrv')  #idle mode for the mrv
    #   self.holo_control.ur_velocity_mode('mrv') #velocity mode for the mrv to move out of the hole
    #   self.holo_control.ur_velocity_mode('client')
    #   self.hil_runner.move_peg_out_of_hole(visualize_before_moving=self.verify_trajectory_visually) #move the peg out of the hole
    #   self.holo_control.ur_idle_mode('mrv')  #idle mode for the mrv
    #   self.holo_control.ur_idle_mode('client')

    # self.holo_control.ur_idle_mode('mrv')
    # self.holo_control.ur_idle_mode('client')
    # self.hil_runner.reset_to_home_angles(check_for_continue=self.verify_trajectory_visually , seed_used = 0)
    # self.holo_control.ur_idle_mode('mrv')
    # self.holo_control.ur_idle_mode('client')  

    # self.hil_runner.calibrate_ft_bias()

    delta_pos = self.poses[:3]
    delta_rot = np.array(rospy.get_param('delta_rot'))*np.pi/180  #why is this taken from the param server and not the grid??
    
    print('Desired position difference (m) in nozzle frame: ', delta_pos)
    print('Desired rotation difference (deg) in nozzle frame: ', delta_rot)
    delta_v = np.array(rospy.get_param('delta_v'))
    initial_mrv_w = np.array(rospy.get_param('initial_mrv_w'))*np.pi/180
    print(np.array([5.5,0,0])*np.pi/180)
    # delta_v = np.copy(self.poses[3:6])
    # initial_mrv_w = np.copy(self.poses[6:9])*np.pi/180

    initial_client_w = np.copy(self.poses[9:12])*np.pi/180

    print('Delta v: ', delta_v)
    print('initial_mrv_w: ', initial_mrv_w)
    print('initial_client_w: ', initial_client_w)
    print('initial_client_w (deg/s): ', initial_client_w*180/np.pi)
    # rospy.spin()

    mrv_controller = MrvController(self.rospath + '/urdf/robot_cv_detached.urdf', 
                                        self.rospath + '/urdf/robot.urdf', 
                                        self.rospath + '/urdf/robot.urdf', 
                                        self.rospath + '/urdf/cv.urdf', 
                                        self.mrv_joint_angle_lower_limits, 
                                        self.mrv_joint_angle_upper_limits, 
                                        self.mrv_joint_vel_limits, 
                                        self.mrv_joint_acc_limits, 
                                        self.mrv_joint_torque_limits, 
                                        self.dt, 
                                        self.cone_slope, 
                                        self.clip_joint_commands, 
                                        10, 
                                        self.cw_a, 
                                        self.cw_mu, 
                                        self.cw_orbit_dir, 
                                        self.do_noisy_state_estimation, 
                                        self.nozzle_opening_rad, 
                                        self.peg_rad, 
                                        self.client_velocity_noise_ang_amp, 
                                        self.time_limit, 
                                        self.debug_with_test_traj, 
                                        self.test_traj_id, 
                                        self.lock_client, 
                                        self.lock_mrv, 
                                        self.probe_z_axis_plunge_velocity, 
                                        self.use_variable_plunge_speed, 
                                        self.use_scheduled_gains, 
                                        use_cw=self.use_cw)
    print("!!!!!!!!!!!!!!!!!!!!!!!!",self.rospath)
    mrv_controller.initialize_collision_world(self.rospath)
    load_paths_for_interpolation,weights = interp_trajectories_on_init_client_state(self.load_paths,initial_client_w,self.ws,delta_pos,self.dps)
    mrv_controller.reset_wrt_capture_box(load_paths_for_interpolation, 
                                        weights, delta_pos, 
                                        delta_rot, 
                                        delta_v, 
                                        initial_client_w, 
                                        initial_mrv_w, 
                                        rng, 
                                        self.dist_centering_waypoint_from_goal) #reset the mrv controller to the required initial conditions
    sw_peg_pos, sw_peg_rmat, sw_peg_twist, sw_nozzle_pos, sw_nozzle_rmat, sw_nozzle_twist = mrv_controller.get_peg_and_nozzle_info()
    sw_base_pos, sw_base_rmat, sw_joint_angles, sw_base_v, sw_base_w, sw_joint_vels, sw_client_pos, sw_client_rmat, sw_client_v, sw_client_w = mrv_controller.get_state_in_pieces()
    # self.hil_runner.create_data_for_saving()

    success = False
    fail_reason = 'None'
    visualize_initial_sim_steps = True
    ##########################################
    while fail_reason == 'None' and not rospy.is_shutdown() and not self.debug_with_test_traj: 

      # # Check if we're close enough to the nozzle to run hardware emulation
      sw_peg_pos_wrt_nozzle = sw_nozzle_rmat.transpose()@(sw_peg_pos - sw_nozzle_pos)
      if True:
      # if np.abs(sw_peg_pos_wrt_nozzle[0]) <= 0.11 and np.abs(sw_peg_pos_wrt_nozzle[1]) <= 0.11 and sw_peg_pos_wrt_nozzle[2] >= -0.11 and sw_peg_pos_wrt_nozzle[2] <= 1.1*mrv_controller.dist_nozzle_opening_from_goal:
        break
    if rospy.is_shutdown():
      quit()

    if fail_reason != 'None':
      print("something went wrong please restart")

      self.on_shutdown()
      print()
      return
    if rospy.is_shutdown():
      quit()

    rate = rospy.Rate(1/self.dt)

    gc.disable()
    noz = Pose()
    peg = Pose() 
    some= 0
    while not rospy.is_shutdown(): 
      # hw_status, wrench_peg_peg = self.hil_runner.emulate(sw_peg_pos, \
      #                                                     sw_peg_rmat, \
      #                                                     sw_peg_twist, \
      #                                                     sw_nozzle_pos, \
      #                                                     sw_nozzle_rmat, \
      #                                                     sw_nozzle_twist, \
      #                                                     mrv_controller.dist_nozzle_opening_from_goal)
      some += 1
      # if (some >500):
      #   break
      hw_status = 'nothing'
      wrench_peg_peg = np.zeros(6)
      # Step the SW simulation      
      sw_status = mrv_controller.step(wrench_peg_peg, self.apply_wrench_only_when_close)
      
      b,a = mrv_controller.mrv_client_sim.get_peg_nad_noz_pose()
      # print("Peg pose: ", b)

      noz.position.x = a[0]
      noz.position.y = a[1]
      noz.position.z = a[2]
      peg.position.x = b[0]
      peg.position.y = b[1]
      peg.position.z = b[2]
      self.peg_pos.poses.append(peg)
      self.noz_pos.poses.append(noz)
      self.noz_pose_pub.publish(self.noz_pos)
      self.peg_pose_pub.publish(self.peg_pos)
      self.noz_pos.poses = []
      self.peg_pos.poses = []
      sw_peg_pos, sw_peg_rmat, sw_peg_twist, sw_nozzle_pos, sw_nozzle_rmat, sw_nozzle_twist = mrv_controller.get_peg_and_nozzle_info()
      sw_base_pos, sw_base_rmat, sw_joint_angles, sw_base_v, sw_base_w, sw_joint_vels, sw_client_pos, sw_client_rmat, sw_client_v, sw_client_w = mrv_controller.get_state_in_pieces()
      traj_pos, traj_rmat = mrv_controller.get_traj()

      self.sim_vis_publisher.publish(sw_joint_angles, sw_base_pos, sw_base_rmat, sw_client_pos, sw_client_rmat, traj_pos, traj_rmat)      

      if hw_status != 'nothing':
        fail_reason = hw_status
        print('Hardware emulation failed ', hw_status)
        break
      elif sw_status == 'success':
        print('SW simulation succeeded')

        
        
        
        print("Success")
        success = True
        break
      elif sw_status != 'nothing':
        print('SW simulation failed ', sw_status)
        fail_reason = sw_status
        break

      rate.sleep()
    
    self.on_shutdown()
    if (rospy.is_shutdown()):
      quit()

    print("Experiment done")
    print("sw_joint_angles: ", sw_joint_angles) 
    print("sw_base_pos: ", sw_base_pos)
    print("sw_base_rmat: ", sw_base_rmat)
    print("sw_client_pos: ", sw_client_pos)
    print("sw_client_rmat: ", sw_client_rmat)
    print("traj_pos: ", traj_pos)
    print("traj_rmat: ", traj_rmat)

    print("moving peg out of hole")
    # self.hil_runner.move_peg_out_of_hole(visualize_before_moving=self.verify_trajectory_visually)


    mrv_pose_list = []
    client_pose_list = []
    del_angles_mrv = []
    del_angles_client = []
    mrv_controller.mrv_client_sim.combine_and_simulate_for_w()
    # return  
    rospy.sleep(5)
    sw_base_pos, sw_base_rmat, sw_client_pos, sw_client_rmat = mrv_controller.mrv_client_sim.forward()
    self.sim_vis_publisher.publish(sw_joint_angles, sw_base_pos, sw_base_rmat, sw_client_pos, sw_client_rmat, traj_pos, traj_rmat)      

    mrv_R_initial = R.from_matrix(sw_base_rmat)
    client_R_initial = R.from_matrix(sw_client_rmat)
    save_data={} 
    save_data['delta_pos'] = delta_pos
    save_data['delta_rot'] = delta_rot

    save_data['delta_v'] = delta_v
    save_data['initial_mrv_w'] = initial_mrv_w
    save_data['initial_client_w'] = initial_client_w


    return
    for i in range(30000):
      print(i/100)
      sw_base_pos, sw_base_rmat, sw_client_pos, sw_client_rmat = mrv_controller.mrv_client_sim.forward()
      self.sim_vis_publisher.publish(sw_joint_angles, sw_base_pos, sw_base_rmat, sw_client_pos, sw_client_rmat, traj_pos, traj_rmat)
      mrv_R = R.from_matrix(sw_base_rmat)
      client_R = R.from_matrix(sw_client_rmat)
      mrv_pose_list.append(sw_base_pos)
      client_pose_list.append(sw_client_pos)
      del_angles_mrv.append(np.array(mrv_R.as_euler('xyz', degrees=True)) - np.array(mrv_R_initial.as_euler('xyz', degrees=True)))
      del_angles_client.append(np.array(client_R.as_euler('xyz', degrees=True)) - np.array(client_R_initial.as_euler('xyz', degrees=True)))
      print("Difference mrv", np.array(mrv_R.as_euler('xyz', degrees=True)) - np.array(mrv_R_initial.as_euler('xyz', degrees=True)))
      print("Difference _cv", np.array(client_R.as_euler('xyz', degrees=True)) - np.array(client_R_initial.as_euler('xyz', degrees=True)))
      # rate.sleep()
      rospy.sleep(0.002)

    save_data['mrv_pose_list'] = mrv_pose_list
    save_data['client_pose_list'] = client_pose_list
    save_data['del_angles_mrv'] = del_angles_mrv
    save_data['del_angles_client'] = del_angles_client
    save_data['mrv_R_initial'] = mrv_R_initial.as_euler('xyz', degrees=True)
    save_data['client_R_initial'] = client_R_initial.as_euler('xyz', degrees=True)

    # with open('/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_launchpad/scripts/dataclean.pkl', 'wb') as f:
    #   pickle.dump(save_data, f)




    print("done moving peg out of hole")
    # save_path = '/home/medusar/bspin/on_orbit/catkin_ws/src/on_orbit/experiment_launchpad/scripts/hil_sim_scripts'
    # # Ensure the directory exists
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # np.save(os.path.join(save_path, 'nozzle_poses.npy'), nozzle_poses)
    # np.save(os.path.join(save_path, 'nozzle_twists.npy'), nozzle_twists)
    # np.save(os.path.join(save_path, 'peg_force.npy'), peg_force)
    # np.save(os.path.join(save_path, 'peg_is_close.npy'), peg_is_close)
    # shitty stuff
    # self.holo_control.ur_idle_mode('mrv')
    # self.holo_control.ur_idle_mode('client')
    rospy.sleep(1)
    # mrv_controller.mrv_client_sim.combine_and_simulate_for_w()
    
    self.rerun = True
    self.poses = []
    self.count += 1

  def run(self):
    r= rospy.Rate(10)
    while not rospy.is_shutdown():
      r.sleep()



if __name__ == '__main__':
  rospy.init_node('experiment_node')
  exp = experiments()
  exp.run()

























