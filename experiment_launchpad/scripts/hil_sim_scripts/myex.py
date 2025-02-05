#!/usr/bin/env python3
import numpy as np
import time

from scipy.spatial.transform import Rotation as R
from get_grid import get_grid
from trajlib_util import get_trajlib_load_paths, interp_trajectories_on_initial_client_w, interp_trajectories_on_delta_pos, interp_trajectories_on_init_client_state

from std_msgs.msg import Float32
import rospy
import rospkg

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

fxpub = rospy.Publisher('/fx', Float32, queue_size=1)
fypub = rospy.Publisher('/fy', Float32, queue_size=1)
fzpub = rospy.Publisher('/fz', Float32, queue_size=1)


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

hil_runner.calibrate_ft_bias()
print("1")
hil_runner.calibrate_ft_bias()
print("2")
hil_runner.calibrate_ft_bias()
print("3")
hil_runner.calibrate_ft_bias()
print("4")
hil_runner.calibrate_ft_bias()

r=rospy.Rate(1/dt)

while not rospy.is_shutdown():
  a=hil_runner.get_ft_compensated(hil_runner.pin_data, hil_runner.ft_fid)
  apub=Float32()
  apub.data=a[0]
  fxpub.publish(apub)
  apub.data=a[1]
  fypub.publish(apub)
  apub.data=a[2]
  fzpub.publish(apub)
  r.sleep()