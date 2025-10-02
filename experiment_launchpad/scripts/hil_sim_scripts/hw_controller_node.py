#!/usr/bin/env python3
import numpy as np
import time

from scipy.spatial.transform import Rotation as R
from get_grid import get_grid
from trajlib_util import get_trajlib_load_paths, interp_trajectories_on_initial_client_w, interp_trajectories_on_delta_pos, interp_trajectories_on_init_client_state

import rospy
import rospkg
from geometry_msgs.msg import Pose, Point, Quaternion
from hil_runner import HILRunner
from holodeck_interface import HolodeckInterface
from sim_ros_vis_publisher import SimROSVisPublisher

import os
import gc

from mrv_controller import MrvController

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

grid_type = rospy.get_param('grid_type')
ic_grid= get_grid(grid_type)

initial_grid_idx = rospy.get_param('initial_grid_idx')
use_grid = rospy.get_param('use_grid')

dist_centering_waypoint_from_goal = rospy.get_param('dist_centering_waypoint_from_goal')
verify_trajectory_visually = rospy.get_param('verify_trajectory_visually')

dt = 0.01

rospy.init_node('hw_control_node')

rospack = rospkg.RosPack()
rospath = rospack.get_path('on_orbit')

rate = rospy.Rate(1/dt)

mrv_hil_home_angles = np.array(rospy.get_param('mrv_hil_home_angles'))
client_hil_home_angles = np.array(rospy.get_param('client_hil_home_angles'))

poseerrorpub = rospy.Publisher('/pose_error', Pose, queue_size=10)
poseerror = Pose()
poseerror.position = Point(0,0,0)
poseerror.orientation = Quaternion(0,0,0,1)
hil_runner = HILRunner(rospath, mrv_hil_home_angles, client_hil_home_angles, dt=dt)
sim_vis_publisher = SimROSVisPublisher(rospath)

cone_slope = rospy.get_param('cone_slope')

mrv_joint_angle_lower_limits = np.array(rospy.get_param('joint_angle_lower_limits'))
mrv_joint_angle_upper_limits = np.array(rospy.get_param('joint_angle_upper_limits'))
mrv_joint_vel_limits = np.array(rospy.get_param('joint_vel_limits'))
mrv_joint_acc_limits = np.array(rospy.get_param('joint_acc_limits'))
mrv_joint_torque_limits = np.array(rospy.get_param('joint_torque_limits'))
apply_wrench_only_when_close = rospy.get_param('apply_wrench_only_when_close')

client_velocity_noise_ang_amp = rospy.get_param('client_velocity_noise_ang_amp')*np.pi/180.

cw_a = rospy.get_param('cw_a')
cw_mu = rospy.get_param('cw_mu')
cw_orbit_dir = rospy.get_param('cw_orbit_dir')

peg_rad = 0.01505
nozzle_opening_rad = 0.142

use_cw = rospy.get_param('use_cw')

do_save = rospy.get_param('do_save')

holo_control = HolodeckInterface()

def save_data(): 
  print('Saving data')
  hil_runner.save(save_path)
  mrv_controller.save(save_path)
  np.save(save_path + 'success.npy', success)
  np.save(save_path + 'fail_reason.npy', fail_reason)
  print('Saved data')

def on_shutdown():
  gc.enable()
  gc.collect()

  if not do_save:
    return
  
  global hil_runner
  global mrv_controller

  global success
  global fail_reason

  global save_path

  save_data()

prefixes = [rospath + rospy.get_param('traj_library_prefix')]

load_paths, ws, dps = get_trajlib_load_paths(rospath)

rospy.on_shutdown(on_shutdown)

num_grid_idx = len(ic_grid)
final_grid_idx = num_grid_idx

print("Grid of initial conditions")
print(ic_grid)

num_trials = rospy.get_param('num_hardware_trials')

# Set a base seed for reproducibility
base_seed = 12345
# Spawn a new seed for each trial
rng_sequences = np.random.SeedSequence(base_seed).spawn(num_trials)

for grid_idx in range(initial_grid_idx, final_grid_idx):
  print('Starting grid index %d/%d' %(grid_idx,num_grid_idx))

  for trial_idx in range(num_trials):
    print('Starting trial %d' %(trial_idx))
    save_path = None

    # Use the seed for the trial corresponding to the trial index
    rng = np.random.default_rng(rng_sequences[trial_idx])
    
    hil_runner.calibrate_ft_bias()
    # if grid_idx > initial_grid_idx or (grid_idx == initial_grid_idx and trial_idx > 0):
    #   holo_control.ur_idle_mode('mrv')
    #   holo_control.ur_idle_mode('client')
    #   holo_control.ur_velocity_mode('mrv')
    #   holo_control.ur_velocity_mode('client')
    #   hil_runner.move_peg_out_of_hole(visualize_before_moving=verify_trajectory_visually)
    #   holo_control.ur_idle_mode('mrv')
    #   holo_control.ur_idle_mode('client')

    holo_control.ur_idle_mode('mrv')
    holo_control.ur_idle_mode('client')
    hil_runner.reset_to_home_angles(check_for_continue=verify_trajectory_visually , seed_used = trial_idx)
    holo_control.ur_idle_mode('mrv')
    holo_control.ur_idle_mode('client')
    

    hil_runner.calibrate_ft_bias() 

    if use_grid:
      delta_pos = ic_grid[grid_idx, :3]
    else:
      delta_pos = np.array(rospy.get_param('delta_pos'))

    delta_rot = np.array(rospy.get_param('delta_rot'))*np.pi/180

    print('Desired position difference (m) in nozzle frame: ', delta_pos)
    print('Desired rotation difference (deg) in nozzle frame: ', delta_rot)

    if use_grid:
      delta_v = np.copy(ic_grid[grid_idx, 3:6])
      initial_mrv_w = ic_grid[grid_idx, 6:9]*np.pi/180
    else:
      delta_v = np.array(rospy.get_param('delta_v'))
      initial_mrv_w = np.array(rospy.get_param('initial_mrv_w'))*np.pi/180

    if use_grid:
      initial_client_w = ic_grid[grid_idx, 9:12]*np.pi/180
    else:
      initial_client_w = np.array(rospy.get_param('initial_client_w'))*np.pi/180

    print('Delta v: ', delta_v)
    print('initial_mrv_w: ', initial_mrv_w)
    print('initial_client_w: ', initial_client_w)
    print('initial_client_w (deg/s): ', initial_client_w*180/np.pi)

    do_noisy_state_estimation = rospy.get_param('do_noisy_state_estimation')
    clip_joint_commands = rospy.get_param('clip_joint_commands')
    time_limit = rospy.get_param('time_limit')
    debug_with_test_traj= rospy.get_param('debug_with_test_traj')
    test_traj_id = rospy.get_param('test_traj_id')
    lock_client = rospy.get_param('lock_client')
    lock_mrv = rospy.get_param('lock_mrv')
    probe_z_axis_plunge_velocity = rospy.get_param('probe_z_axis_plunge_velocity')
    use_variable_plunge_speed = rospy.get_param('use_variable_plunge_speed')
    use_scheduled_gains = rospy.get_param('use_scheduled_gains')

    
    mrv_controller = MrvController(rospath + '/urdf/robot_cv_detached.urdf', 
                                        rospath + '/urdf/robot.urdf', 
                                        rospath + '/urdf/robot.urdf', 
                                        rospath + '/urdf/cv.urdf', mrv_joint_angle_lower_limits, 
                                        mrv_joint_angle_upper_limits, mrv_joint_vel_limits, mrv_joint_acc_limits, 
                                        mrv_joint_torque_limits, dt, cone_slope, clip_joint_commands, 10, cw_a, cw_mu, cw_orbit_dir, 
                                        do_noisy_state_estimation, nozzle_opening_rad, peg_rad, client_velocity_noise_ang_amp, time_limit, 
                                        debug_with_test_traj, test_traj_id, lock_client, lock_mrv, probe_z_axis_plunge_velocity, use_variable_plunge_speed, 
                                        use_scheduled_gains, use_cw=use_cw, collision_thresh = 0.01)



    print("!!!!!!!!!!!!!!!!!!!!!!!!",rospath)
    mrv_controller.initialize_collision_world(rospath)
    load_paths_for_interpolation,weights = interp_trajectories_on_init_client_state(load_paths,initial_client_w,ws,delta_pos,dps)

    mrv_controller.reset_wrt_capture_box(load_paths_for_interpolation, weights, delta_pos, delta_rot, delta_v, initial_client_w, initial_mrv_w, rng, dist_centering_waypoint_from_goal)

    sw_peg_pos, sw_peg_rmat, sw_peg_twist, sw_nozzle_pos, sw_nozzle_rmat, sw_nozzle_twist = mrv_controller.get_peg_and_nozzle_info()
    sw_base_pos, sw_base_rmat, sw_joint_angles, sw_base_v, sw_base_w, sw_joint_vels, sw_client_pos, sw_client_rmat, sw_client_v, sw_client_w = mrv_controller.get_state_in_pieces()

    desired_waypoint = mrv_controller.mrv_client_sim.aheadthroutgoal()
    desired_rot = np.array([[1,0,0],[0,1,0],[0,0,1]]) 

    timestr = time.strftime("%Y%m%d-%H%M%S")

    save_path_str = rospy.get_param('save_folder') + '/' + '_'.join(['pos_', str(delta_pos[0]), str(delta_pos[1]), str(delta_pos[2]), 'rot', str(delta_rot[0]), str(delta_rot[1]), str(delta_rot[2]), 'delta_v', str(delta_v[0]), str(delta_v[1]), str(delta_v[2]), 'mrv_w', str(initial_mrv_w[0]), str(initial_mrv_w[1]), str(initial_mrv_w[2]), 'client_w', str(initial_client_w[0]), str(initial_client_w[1]), str(initial_client_w[2])])
    save_path = rospath + '/' + save_path_str + '_' + timestr + '/'

    if do_save:
      os.makedirs(save_path)

    hil_runner.create_data_for_saving()

    success = False
    fail_reason = 'None'

    # Run the sim until the peg and nozzle are close enough to start the hardware emulation
    visualize_initial_sim_steps = True

    #The trajectories in TestTrajectories may not take us to the nozzle, so we skip this if debugging with TestTrajectories
    while fail_reason == 'None' and not rospy.is_shutdown() and not debug_with_test_traj: 

      # # Check if we're close enough to the nozzle to run hardware emulation
      sw_peg_pos_wrt_nozzle = sw_nozzle_rmat.transpose()@(sw_peg_pos - sw_nozzle_pos)
      if True:
      # if np.abs(sw_peg_pos_wrt_nozzle[0]) <= 0.11 and np.abs(sw_peg_pos_wrt_nozzle[1]) <= 0.11 and sw_peg_pos_wrt_nozzle[2] >= -0.11 and sw_peg_pos_wrt_nozzle[2] <= 1.1*mrv_controller.dist_nozzle_opening_from_goal:
        break

      status = mrv_controller.step(np.zeros(6), apply_wrench_only_when_close=False) # This is pure sim, so always apply the wrench
      
      sw_peg_pos, sw_peg_rmat, sw_peg_twist, sw_nozzle_pos, sw_nozzle_rmat, sw_nozzle_twist = mrv_controller.get_peg_and_nozzle_info()
      sw_base_pos, sw_base_rmat, sw_joint_angles, sw_base_v, sw_base_w, sw_joint_vels, sw_client_pos, sw_client_rmat, sw_client_v, sw_client_w = mrv_controller.get_state_in_pieces()
      traj_pos, traj_rmat = mrv_controller.get_traj()
      
      if visualize_initial_sim_steps:
        sim_vis_publisher.publish(sw_joint_angles, sw_base_pos, sw_base_rmat, sw_client_pos, sw_client_rmat, traj_pos, traj_rmat)

      if status == 'success':
        print('Error: success declared before peg was even close to nozzle')
        quit()
      elif status != 'nothing':
        print('Failure before peg is close enough to nozzle to start hardware emulation')
        fail_reason = status
        break

    if rospy.is_shutdown():
      quit()

    if fail_reason != 'None':
      on_shutdown()
      print()
      continue

    hil_runner.initialize_arms_to_ee_poses(desired_waypoint, \
                                          desired_rot, \
                                          sw_peg_twist, \
                                          sw_nozzle_pos, \
                                          sw_nozzle_rmat, \
                                          sw_nozzle_twist, \
                                          verify_trajectory_visually)
  
    if rospy.is_shutdown():
      quit()

    rate = rospy.Rate(1/dt)

    gc.disable()

    holo_control.ur_idle_mode('mrv')
    holo_control.ur_idle_mode('client')
    holo_control.ur_velocity_mode('mrv')
    holo_control.ur_velocity_mode('client')
    while not rospy.is_shutdown():
      hw_status, wrench_peg_peg = hil_runner.emulate(sw_peg_pos, \
                                                      sw_peg_rmat, \
                                                      sw_peg_twist, \
                                                      sw_nozzle_pos, \
                                                      sw_nozzle_rmat, \
                                                      sw_nozzle_twist, \
                                                      mrv_controller.dist_nozzle_opening_from_goal)
      a = hil_runner.geterror()
      poseerror.position.x = a[0]
      poseerror.position.y = a[1]
      poseerror.position.z = a[2]
      poseerrorpub.publish(poseerror)

      plunging = mrv_controller.getplunging()
      insidenozzle = not plunging

      # Step the SW simulation      
      sw_status = mrv_controller.step(wrench_peg_peg, apply_wrench_only_when_close)

      sw_peg_pos, sw_peg_rmat, sw_peg_twist, sw_nozzle_pos, sw_nozzle_rmat, sw_nozzle_twist = mrv_controller.get_peg_and_nozzle_info()
      sw_base_pos, sw_base_rmat, sw_joint_angles, sw_base_v, sw_base_w, sw_joint_vels, sw_client_pos, sw_client_rmat, sw_client_v, sw_client_w = mrv_controller.get_state_in_pieces()
      traj_pos, traj_rmat = mrv_controller.get_traj()

      sim_vis_publisher.publish(sw_joint_angles, sw_base_pos, sw_base_rmat, sw_client_pos, sw_client_rmat, traj_pos, traj_rmat)      

      if hw_status != 'nothing':
        fail_reason = hw_status
        break
      elif sw_status == 'success':
        success = True
        break
      elif sw_status != 'nothing':
        fail_reason = sw_status
        break

      rate.sleep()

    on_shutdown()
    holo_control.ur_idle_mode('mrv')
    holo_control.ur_idle_mode('client')
    holo_control.ur_velocity_mode('mrv')
    holo_control.ur_velocity_mode('client')
    hil_runner.move_peg_out_of_hole(visualize_before_moving=verify_trajectory_visually)
    holo_control.ur_idle_mode('mrv')
    holo_control.ur_idle_mode('client')
 
    print("Done")

    

  if rospy.is_shutdown():
    quit()

  print()

  if not use_grid:
    break

  print("last grid idx")
  print(grid_idx)

# hil_runner.move_peg_out_of_hole(visualize_before_moving=verify_trajectory_visually)
# print("Done")

while not rospy.is_shutdown():
  sim_vis_publisher.publish(sw_joint_angles, sw_base_pos, sw_base_rmat, sw_client_pos, sw_client_rmat, traj_pos, traj_rmat)
  rate.sleep()


# TODO: Incorporate readme file into the save folders
# import os
# import git
# import pickle

# def write_README(outfolder,**kwargs):
#     git_repo_path=os.path.join(os.path.dirname(__file__),"..","..")
#     repo=git.Repo(os.path.abspath(git_repo_path))
#     commit_name=repo.head.commit.name_rev
#     with open(os.path.join(outfolder,"README"),"w") as fh:
#         fh.write(f"commit: {commit_name}\n")
#         for key,val in kwargs.items():
#             fh.write(f"{key}: {val}\n")