import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from casadi_rp_conversion import *
import numpy as np
from inf_def import inf

def export_contact_force_direction_in_hole_constraint(cpin_model, x_acados, u):
  cpin_data = cpin_model.createData()

  num_rotary = cpin_model.nv - 12

  tip_fid = cpin_model.getFrameId('ee_tip')

  q = q_to_pin(x_acados[:cpin_model.nv], cpin_model)

  cpin.forwardKinematics(cpin_model, cpin_data, q)
  cpin.updateFramePlacement(cpin_model, cpin_data, tip_fid)

  size = 1
  lb = np.zeros(size)
  ub = np.zeros(size)
  return ca.dot(cpin_data.oMf[tip_fid].rotation[:, 2], u[num_rotary:num_rotary + 3]), lb, ub
