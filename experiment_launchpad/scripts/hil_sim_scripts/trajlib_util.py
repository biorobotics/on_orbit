#!/usr/bin/env python3
import numpy as np
import rospy
import os

def get_trajlib_load_paths(rospath):
  deg_per_s_folder_names = True
  ws = []
  dps = []
  load_paths = []

  prefixes = [rospath + rospy.get_param('traj_library_prefix')]

  for prefix in prefixes:
    folders = os.listdir(prefix)
    for folder in sorted(folders):
      load_path = prefix + folder + '/control/'
      if not os.path.isdir(load_path):
        continue
      timestr = sorted(os.listdir(load_path))[-1]
      # timestr = sorted(os.listdir(load_path))[0]
      load_path = load_path + timestr
      if np.load(load_path + '/success.npy').item():
        load_paths.append(load_path)

        '''Get the delta in initial angular velocity for each trajectory'''
        w_str_start_idx = folder.find('client_w') + 9
        w_str_end_idx = len(folder)
        w_str = folder[w_str_start_idx:w_str_end_idx]
        w = np.array([float(f) for f in w_str.split('_')])
        if deg_per_s_folder_names:
          w = w*np.pi/180

        '''Get the delta in initial position for each trajectory'''
        pos_str_start_idx = folder.find('pos') + 4 
        pos_str_end_idx = folder.find('rot') - 1
        pos_str = folder[pos_str_start_idx:pos_str_end_idx]

        pos = []
        for f in pos_str.split('_'):
          pos.append(float(f))

        dps.append(pos)
        ws.append(w)
  return load_paths, np.array(ws), np.array(dps)

def interp_trajectories_on_initial_client_w(load_paths,initial_client_w,ws):
  '''Inputs:
  
  ws - the length 3 angular velocities for each trajectory in the library'''

  distances = np.linalg.norm(ws - initial_client_w, axis=1)
  sort_idx = np.argsort(distances)
  w0 = ws[sort_idx[0]]
  w1 = ws[sort_idx[1]]

  dir_1_0 = w1 - w0
  dist_1_0 = np.linalg.norm(dir_1_0)
  dir_1_0 /= dist_1_0
  dot_product = (initial_client_w - w0)@dir_1_0
  if dot_product < 0:
    weights = [1., 0.]
  elif dot_product <= 0.5:
    alpha = dot_product/dist_1_0
    weights = [1. - alpha, alpha]
  else:
    # If we get to this case, it means w0 was reported as closest by argsort, but during weight computation
    # we found that w1 is closer
    raise Exception('Error computing weights in trajectory interpolation')
  
  load_paths_for_interpolation = [load_paths[sort_idx[0]], load_paths[sort_idx[1]]]
  return load_paths_for_interpolation, weights

def interp_trajectories_on_delta_pos(load_paths,delta_pos,dps):
  distances = []
  for x in range(0,len(dps)):
    dp = dps[x]
    distances.append(np.linalg.norm(dp - delta_pos))

  sort_idx = np.argsort(distances)
  dp0 = dps[sort_idx[0]]
  dp1 = dps[sort_idx[1]]

  dir_1_0 = dp1 - dp0
  dist_1_0 = np.linalg.norm(dir_1_0)
  if dist_1_0 < 1e-6:
    raise Exception('Duplicate values in library for delta_pos')
  dir_1_0 /= dist_1_0
  dot_product = (delta_pos - dp0)@dir_1_0
  if dot_product < 0:
    weights = [1., 0.]
  elif dot_product <= 0.5:
    alpha = dot_product/dist_1_0
    weights = [1. - alpha, alpha]
  else:
    # If we get to this case, it means dp0 was reported as closest by argsort, but during weight computation
    # we found that w1 is closer
    raise Exception('Error computing weights in trajectory interpolation')
  
  load_paths_for_interpolation = [load_paths[sort_idx[0]], load_paths[sort_idx[1]]]
  return load_paths_for_interpolation, weights

def interp_trajectories_on_init_client_state(load_paths,initial_client_w,ws,delta_pos,dps):
  '''Use both linear displacement and angular velocity for deciding which trajectories to interpolate
  
  This is expected to behave identical to interp_trajectories_on_initial_client_w and interp_trajectories_on_delta_pos
  in scenarios where the library only has delta_pos or initial_client_w states.'''

  #Define a distance metric that is weighted sum of linear displacement and angular velocity
  distances = []
  for x in range(0,len(dps)):
    dp = dps[x]
    w = ws[x]
    dp_weight = 1.0
    w_weight = 1.0
    distances.append(dp_weight*np.linalg.norm(dp - delta_pos) + w_weight*np.linalg.norm(w - initial_client_w))

  sort_idx = np.argsort(distances)
  dp0 = dps[sort_idx[0]]
  dp1 = dps[sort_idx[1]]
  w0 = ws[sort_idx[0]]
  w1 = ws[sort_idx[1]]

  vec0 = np.concatenate((dp0,w0))
  vec1 = np.concatenate((dp1,w1))
  vec_d = np.concatenate((delta_pos,initial_client_w))

  print("Actual initial state:")
  print(vec_d)
  print("Interpolating between the following initial states:")
  print(vec0)
  print(vec1)

  dir_1_0 = vec1 - vec0
  dist_1_0 = np.linalg.norm(dir_1_0)
  if dist_1_0 < 1e-6:
    weights = [1., 0.] #The two points are very close, so just pick one
  else:
    dir_1_0 /= dist_1_0
    dot_product = (vec_d - vec0)@dir_1_0
  if dot_product < 0:
    weights = [1., 0.]
  elif dot_product <= 0.5:
    alpha = dot_product/dist_1_0
    weights = [1. - alpha, alpha]
  else:
    # If we get to this case, it means dp0 was reported as closest by argsort, but during weight computation
    # we found that w1 is closer
    raise Exception('Error computing weights in trajectory interpolation')
  
  print("Interpolation weights:")
  print(weights)

  load_paths_for_interpolation = [load_paths[sort_idx[0]], load_paths[sort_idx[1]]]
  return load_paths_for_interpolation, weights