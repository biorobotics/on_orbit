import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from casadi_rp_conversion import *
import numpy as np
from inf_def import inf

def export_stay_in_cone_approx_planes_constraint(cpin_model, x_acados, cone_slope):
  cpin_data = cpin_model.createData()

  tip_fid = cpin_model.getFrameId('ee_tip')
  cone_vertex_fid = cpin_model.getFrameId('cone_vertex')

  num_planes = 8

  q = q_to_pin(x_acados[:cpin_model.nv], cpin_model)

  cpin.forwardKinematics(cpin_model, cpin_data, q)
  cpin.updateFramePlacement(cpin_model, cpin_data, tip_fid)
  cpin.updateFramePlacement(cpin_model, cpin_data, cone_vertex_fid)
  tip_wrt_cone = cpin_data.oMf[cone_vertex_fid].actInv(cpin_data.oMf[tip_fid].translation)

  collision_values = []
  for i in range(num_planes):
    theta = i*2*np.pi/num_planes
    max_height = -cone_slope*(np.cos(theta)*tip_wrt_cone[0] + np.sin(theta)*tip_wrt_cone[1])
    collision_values.append(tip_wrt_cone[2] - max_height)
  collision_values = ca.vertcat(*collision_values)

  size = num_planes
  lb = -inf*np.ones(size)
  ub = np.zeros(size)
  return collision_values, lb, ub
