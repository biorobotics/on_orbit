import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from casadi_rp_conversion import *
import numpy as np
from inf_def import inf

def export_peg_direction_constraint(cpin_model, x_acados, cone_slope):
  cpin_data = cpin_model.createData()

  q = q_to_pin(x_acados[:cpin_model.nv], cpin_model)

  tip_fid = cpin_model.getFrameId('ee_tip')
  nozzle_fid = cpin_model.getFrameId('nozzle')

  cpin.forwardKinematics(cpin_model, cpin_data, q)
  cpin.updateFramePlacement(cpin_model, cpin_data, tip_fid)
  ee_rmat = cpin_data.oMf[tip_fid].rotation

  cpin.updateFramePlacement(cpin_model, cpin_data, nozzle_fid)
  nozzle_pos = cpin_data.oMf[nozzle_fid].translation
  nozzle_rmat = cpin_data.oMf[nozzle_fid].rotation
  tip_rmat_wrt_nozzle = nozzle_rmat.T@ee_rmat

  size = 1
  lb = (np.cos(np.pi/2 - np.arctan(cone_slope)) + 2e-4)*np.ones(size)
  ub = 10*np.ones(size)
  return tip_rmat_wrt_nozzle[2, 2], lb, ub
