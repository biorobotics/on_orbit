import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from casadi_rp_conversion import *
import numpy as np
from inf_def import inf

def export_stay_below_nozzle_constraint(cpin_model, x_acados):
  cpin_data = cpin_model.createData()

  tip_fid = cpin_model.getFrameId('ee_tip')
  nozzle_fid = cpin_model.getFrameId('nozzle')

  q = q_to_pin(x_acados[:cpin_model.nv], cpin_model)

  cpin.forwardKinematics(cpin_model, cpin_data, q)
  cpin.updateFramePlacement(cpin_model, cpin_data, tip_fid)
  cpin.updateFramePlacement(cpin_model, cpin_data, nozzle_fid)
  tip_wrt_nozzle = cpin_data.oMf[nozzle_fid].actInv(cpin_data.oMf[tip_fid].translation)

  size = 1
  lb = -inf*np.ones(size)
  ub = np.zeros(size)
  return tip_wrt_nozzle[2], lb, ub
