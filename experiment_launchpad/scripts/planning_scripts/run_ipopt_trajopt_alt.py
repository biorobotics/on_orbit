#!/usr/bin/env python3

from ipopt_trajopt import IpoptTrajopt
from trajopt_helpers import TrajectoryHelper
import numpy as np
import time

date = '06_14_24'
folder_tag = 'top_w_1_5_cross'
save_path_prefix = 'experiment_logs/' + date + '/' + 'test_lib' + '/' + folder_tag 
do_save = True
nozzle_align = True  # Set to True to align ee tip with nozzle frame
break_on_failure = True # Set to True to break as soon as the planner fails
ipopt_traj_opt = IpoptTrajopt(save_path_prefix, do_save)

max_ang_vel_for_contact_avoidance = 100 #deg/s

'''Poses'''
# Corners of capture box
step_size = 20
x_range = 0.01*np.array([np.arange(-10,10+step_size,step_size)])
y_range = 0.01*np.array([np.arange(-10,10+step_size,step_size)])
z_range = [0.00]


# # One point. Note that (0,0,0) is the front/center of the capture box, and the box is 20cm wide, 20cm high, and 10 cm deep
# # Center of capture box is (0,0,-0.05)
# x_range = np.array([0.1])
# y_range = -np.array([0.1])
# z_range = -np.array([0.1])

vx_range = [0.0]
vy_range = [0.0]
vz_range = [0.0]

traj_helper = TrajectoryHelper(x_range, y_range, z_range, vx_range, vy_range, 
                               vz_range, max_ang_vel_for_contact_avoidance, 
                               ipopt_traj_opt, nozzle_align, break_on_failure)

run_wx = False
run_wy = False
run_cross = True

# Step size
step_size = 10
if run_wx:
    traj_helper.generate_and_run_wx(-30, 70, step_size)

if run_wy:
    traj_helper.generate_and_run_wy(-40, 30, step_size)

step_size = 5

if run_cross:
    cross_velocities = [
        {"wx_max": 15, "wy_max": 15},  # ++
        {"wx_max": -15, "wy_max": -15},  # --
        {"wx_max": -15, "wy_max": 15},  # -+
        {"wx_max": 15, "wy_max": -15}  # +-
    ]
    traj_helper.generate_and_run_cross(cross_velocities, step_size)
