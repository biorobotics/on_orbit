#!/usr/bin/env python3
import numpy as np
import time

def get_grid(grid_type):

  # Use deg/s on angular velocities

  if grid_type == 1: 
    # wx library (no interpolation)
    wx_range = 0.1*np.flip(np.arange(1, 11))
    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid([0.], [0.], [-0.1], [0.], [0.], [0.], [0.], [0.], [0.], wx_range, [0.], [0.], indexing='ij')
  elif grid_type == 2:
    # wy library (no interpolation)
    wy_range = 0.1*np.flip(np.arange(14))
    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid([0.], [0.], [-0.1], [0.], [0.], [0.], [0.], [0.], [0.], [0.], wy_range, [0.], indexing='ij')
  elif grid_type == 3:
    # wx grid
    wx_range = 0.1*np.flip(np.arange(10)) + 0.05
    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid([0.], [0.], [-0.1], [0.], [0.], [0.], [0.], [0.], [0.], wx_range, [0.], [0.], indexing='ij')
  elif grid_type == 4: 
    # wy grid
    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid([0.], [0.], [-0.1], [0.], [0.], [0.], [0.], [0.], [0.], [0.], 0.1*np.flip(np.arange(13)) + 0.05, [0.], indexing='ij')
  elif grid_type == 5: 
    # wx grid, count by 0.3's
    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid([0.], [0.], [-0.1], [0.], [0.], [0.], [0.], [0.], [0.], 0.1*np.flip(np.arange(0, 7, 3)) + 0.15, [0.], [0.], indexing='ij')
  elif grid_type == 6: 
    # wy grid, count by 0.3's
    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid([0.], [0.], [-0.1], [0.], [0.], [0.], [0.], [0.], [0.], [0.], 0.1*np.flip(np.arange(0, 10, 3)) + 0.15, [0.], indexing='ij')
  elif grid_type == 7:
    # wx grid, count by 0.2's
   x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid([0.], [0.], [-0.1], [0.], [0.], [0.], [0.], [0.], [0.], 0.1*np.flip(np.arange(0, 9, 2)) + 0.1, [0.], [0.], indexing='ij')
  elif grid_type == 8: 
    # wy grid, count by 0.2's
    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid([0.], [0.], [-0.1], [0.], [0.], [0.], [0.], [0.], [0.], [0.], 0.1*np.flip(np.arange(0, 11)) + 0.1, [0.], indexing='ij')
  elif grid_type == 9: 
    # grid used for the full trajectory library
    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid([0.], [0.], [-0.1], [0.], [0.], [0.], [0.], [0.], [0.], 0.1*np.arange(-6, 7, 3), 0.1*np.arange(-6, 7, 3), [0.], indexing='ij')
  elif grid_type == 10: 
    # One test with no angular velocity 
    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid([0.], [0.], [-0.1], [0.], [0.], [0.], [0.], [0.], [0.], [0.0], [0.0], [0.], indexing='ij')
  elif grid_type == 11: 
    # wx with 0.5 deg/s max
    wx_range = 0.1*np.flip(np.arange(1, 5))
    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid([0.], [0.], [-0.1], [0.], [0.], [0.], [0.], [0.], [0.], wx_range, [0.], [0.], indexing='ij')
  elif grid_type == 12: 
    # One test with a specified angular velocity
    wx = 0.5
    delta_pos_z = -0.05
    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid([0.], [0.], [delta_pos_z], [0.], [0.], [0.], [0.], [0.], [0.], [wx], [0.0], [0.], indexing='ij')
  elif grid_type == 13: 
    wx_range = np.array([0.21,0.25,0.29])
    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid([0.], [0.], [-0.1], [0.], [0.], [0.], [0.], [0.], [0.], wx_range, [0.0], [0.], indexing='ij')
  elif grid_type == 14: 
    # Corners of capture box with four angular velocities up to 0.15 deg/s for each point
    delta_pos_x = [-0.1,0.1]
    delta_pos_y = [-0.1,0.1]
    delta_pos_z = [0,-0.1]

    max_ang_deg = 0.15 #deg/s
    #max_cross_axis_deg = np.sqrt(max_ang_deg*max_ang_deg/2.)
    #wx_range = np.array([-max_cross_axis_deg, max_cross_axis_deg])
    #wy_range = np.array([-max_cross_axis_deg, max_cross_axis_deg])

    wx_range = np.array([-max_ang_deg, max_ang_deg])
    wy_range = np.array([-max_ang_deg, max_ang_deg])

    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid(delta_pos_x, delta_pos_y, delta_pos_z, [0.], [0.], [0.], [0.], [0.], [0.], wx_range, wy_range, [0.], indexing='ij')
  elif grid_type == 15:
    # One point
    delta_pos_x = np.array([-0.1])
    delta_pos_y = np.array([-0.1])
    delta_pos_z = np.array([-0.05])
    #delta_pos_z = np.array([0.371024428+0.015]) #use this for experiments with robot starting within the nozzle

    wx_range = np.array([-0.0]) #up to 0.3 is OK
    wy_range = np.array([-0.0])

    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid(delta_pos_x, delta_pos_y, delta_pos_z, [0.], [0.], [0.], [0.], [0.], [0.], wx_range, wy_range, [0.], indexing='ij')
  elif grid_type == 16: 
    # A single point in pose capture box with angular velocities at each cross-axis point 
    delta_pos_x = [0]
    delta_pos_y = [0]
    delta_pos_z = [0]

    max_ang = 0.15 #deg/s
    max_cross_axis_deg = np.sqrt(max_ang*max_ang/2.)
    wx_range = np.array([-max_cross_axis_deg, max_cross_axis_deg])
    wy_range = np.array([-max_cross_axis_deg, max_cross_axis_deg])

    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid(delta_pos_x, delta_pos_y, delta_pos_z, [0.], [0.], [0.], [0.], [0.], [0.], wx_range, wy_range, [0.], indexing='ij')
  elif grid_type == 17: 
    # Corners of capture box with no angular velocities
    delta_pos_x = [-0.1,0.1]
    delta_pos_y = [-0.1,0.1]
    delta_pos_z = [0,-0.1]

    wx_range = np.array([0.])
    wy_range = np.array([0.])

    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid(delta_pos_x, delta_pos_y, delta_pos_z, [0.], [0.], [0.], [0.], [0.], [0.], wx_range, wy_range, [0.], indexing='ij')

  elif grid_type == 18: 
    # Single point with a range of angular velocities along x axis
    # Used this for the planar experiments to test admittance on the nominal curve
    delta_pos_x = np.array([0.0])
    delta_pos_y = np.array([0.00])
    delta_pos_z = np.array([0.371024428])

    wx_range = np.array([-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
    wy_range = np.array([0.])

    x_grid, y_grid, z_grid, vx_grid, vy_grid, vz_grid, mrv_wx_grid, mrv_wy_grid, mrv_wz_grid, client_wx_grid, client_wy_grid, client_wz_grid = np.meshgrid(delta_pos_x, delta_pos_y, delta_pos_z, [0.], [0.], [0.], [0.], [0.], [0.], wx_range, wy_range, [0.], indexing='ij')
  
  else:
    raise ValueError("Invalid grid_type.")
  ic_grid = np.stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten(), vx_grid.flatten(), vy_grid.flatten(), vz_grid.flatten(), mrv_wx_grid.flatten(), mrv_wy_grid.flatten(), mrv_wz_grid.flatten(), client_wx_grid.flatten(), client_wy_grid.flatten(), client_wz_grid.flatten()), 1)

  return ic_grid