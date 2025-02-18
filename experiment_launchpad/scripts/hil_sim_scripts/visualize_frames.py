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

#set to initial conditions
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
                                        use_scheduled_gains, use_cw=use_cw)

load_paths, ws, dps = get_trajlib_load_paths(rospath)
delta_pos = np.array(rospy.get_param('delta_pos'))
delta_rot = np.array(rospy.get_param('delta_rot'))*np.pi/180
delta_v = np.array(rospy.get_param('delta_v'))
initial_mrv_w = np.array(rospy.get_param('initial_mrv_w'))*np.pi/180
initial_client_w = np.array(rospy.get_param('initial_client_w'))*np.pi/180

load_paths_for_interpolation,weights = interp_trajectories_on_init_client_state(load_paths,initial_client_w,ws,delta_pos,dps)
rng=np.random.default_rng(0)

mrv_controller.reset_wrt_capture_box(load_paths_for_interpolation, weights, delta_pos, delta_rot, delta_v, initial_client_w, initial_mrv_w, rng, dist_centering_waypoint_from_goal)
sw_status = mrv_controller.step(np.zeros(6), apply_wrench_only_when_close)
sw_peg_pos, sw_peg_rmat, sw_peg_twist, sw_nozzle_pos, sw_nozzle_rmat, sw_nozzle_twist = mrv_controller.get_peg_and_nozzle_info()
sw_base_pos, sw_base_rmat, sw_joint_angles, sw_base_v, sw_base_w, sw_joint_vels, sw_client_pos, sw_client_rmat, sw_client_v, sw_client_w = mrv_controller.get_state_in_pieces()
traj_pos, traj_rmat = mrv_controller.get_traj()

while not rospy.is_shutdown():
    sim_vis_publisher.publish(sw_joint_angles, sw_base_pos, sw_base_rmat, sw_client_pos, sw_client_rmat, traj_pos, traj_rmat)
    hil_runner.publish_current_hw_joints_for_viz()
    rate.sleep()