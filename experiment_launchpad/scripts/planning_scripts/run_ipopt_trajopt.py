#!/usr/bin/env python3

from ipopt_trajopt import IpoptTrajopt
import numpy as np
import time

date = '10_28_24'
folder_tag = 'MPC_ipopt_test'
save_path_prefix = 'experiment_logs/' + date + '/' + folder_tag
do_save = False
nozzle_align = False # Set to True to algin ee tip with nozzle frame
test_MPC = True  

ipopt_traj_opt = IpoptTrajopt(save_path_prefix,do_save)

max_ang_vel_for_contact_avoidance = 100#deg/s

'''Poses'''
# Corners of capture box
# step_size = 20
# x_range = 0.01*np.array([np.arange(-10,10+step_size,step_size)])
# y_range = 0.01*np.array([np.arange(-10,10+step_size,step_size)])
# z_range = [0.0,-0.1]
# print("x_range")
# print(x_range)
# print("y_range")
# print(y_range)
# exit()

# # One point. Note that (0,0,0) is the front/center of the capture box, and the box is 20cm wide, 20cm high, and 10 cm deep
# # Center of capture box is (0,0,-0.05)
x_range =  np.array([0.07])
y_range = -np.array([0.00])
z_range = -np.array([0.05])

'''Velocities'''
# One points
wx_range = np.array([0.0])
wy_range = np.array([0.0])
wz_range = np.array([0.0])

# From -1 to 1 deg/s in x direction in steps of 0.2
# step_size = 20
# wx_range = 0.01*np.array([np.arange(-120,120+step_size,step_size)])
# wy_range = np.array([0])
# wz_range = np.array([0])

# Four points along cross axis directions
# ang_vel_mag = 0.15 #deg/s
# max_cross_axis_deg = np.sqrt(ang_vel_mag*ang_vel_mag)
# wx_range = np.array([-max_cross_axis_deg, max_cross_axis_deg])
# wy_range = np.array([-max_cross_axis_deg, max_cross_axis_deg])
# wz_range = np.array([0])

vx_range = [0.0]
vy_range = [0.0]
vz_range = [0.0]

'''Convert from deg to radians'''
wx_range = wx_range*np.pi/180.
wy_range = wy_range*np.pi/180.
wz_range = wz_range*np.pi/180.

x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid(x_range, y_range, z_range, 
                                                                                                                vx_range, vy_range, vz_range,
                                                                                                                wx_range, wy_range, wz_range, 
                                                                                                                indexing='ij')
ic_grid = np.stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), 
                    vx_grid.flatten(), vy_grid.flatten(), vz_grid.flatten(), 
                    client_wx_grid.flatten(), client_wy_grid.flatten(), client_wz_grid.flatten()), 1)

print("Initial conditions grid:", ic_grid)

num_grid_pts = len(ic_grid)
print("Number of grid points:")
print(num_grid_pts)

init_grid_idx = 0
final_grid_idx = num_grid_pts
for grid_idx in range(init_grid_idx, final_grid_idx):
  print('Starting grid index %d/%d' %(grid_idx, num_grid_pts))

  delta_pos = ic_grid[grid_idx, :3]
  delta_rot = np.zeros(3)

  delta_v = np.copy(ic_grid[grid_idx, 3:6])
  initial_client_w = ic_grid[grid_idx, 6:9]

  if np.linalg.norm(initial_client_w) < max_ang_vel_for_contact_avoidance*np.pi/180.:
      #raise Exception("Should not be using contact avoidance in current testing.")
      use_contact = False
  else:
    #   raise Exception("Should not be using contact in current testing.")
      use_contact = True

  success = ipopt_traj_opt.plan(delta_pos,delta_rot,initial_client_w,use_contact,nozzle_align,test_MPC)

  if not success:
     print("Planner failed.")
     print("grid_idx")
     print(grid_idx)
     break
