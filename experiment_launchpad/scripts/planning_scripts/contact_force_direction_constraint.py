import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from casadi_rp_conversion import *
import numpy as np

def export_contact_force_direction_constraint(cpin_model, x_acados, u, plane_idx, max_mag):
  cpin_data = cpin_model.createData()

  num_rotary = cpin_model.nv - 12

  nozzle_geom_fid = cpin_model.getFrameId('nozzle_geom' + str(plane_idx))

  q = q_to_pin(x_acados[:cpin_model.nv], cpin_model)

  cpin.forwardKinematics(cpin_model, cpin_data, q)
  cpin.updateFramePlacement(cpin_model, cpin_data, nozzle_geom_fid)

  size = 3
  lb = np.zeros(size)
  lb[0] = -2*max_mag
  ub = np.zeros(size)
  return cpin_data.oMf[nozzle_geom_fid].rotation.T@u[num_rotary:num_rotary + 3], lb, ub
