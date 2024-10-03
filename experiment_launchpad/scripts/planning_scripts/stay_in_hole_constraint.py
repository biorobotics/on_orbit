import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from casadi_rp_conversion import *
import numpy as np
from inf_def import inf

def export_stay_in_hole_constraint(cpin_model, x_acados, z_lower_bound):
  cpin_data = cpin_model.createData()

  tip_fid = cpin_model.getFrameId('ee_tip')
  hole_fid = cpin_model.getFrameId('hole_end')

  q = q_to_pin(x_acados[:cpin_model.nv], cpin_model)

  cpin.forwardKinematics(cpin_model, cpin_data, q)
  cpin.updateFramePlacement(cpin_model, cpin_data, tip_fid)
  ee_pos = cpin_data.oMf[tip_fid].translation
  ee_rmat = cpin_data.oMf[tip_fid].rotation

  cpin.updateFramePlacement(cpin_model, cpin_data, hole_fid)
  hole_pos = cpin_data.oMf[hole_fid].translation
  hole_rmat = cpin_data.oMf[hole_fid].rotation

  tip_pos_wrt_hole = hole_rmat.T@(ee_pos - hole_pos)
  tip_rmat_wrt_hole = hole_rmat.T@ee_rmat
  rot_err = cpin.unSkew(0.5*(tip_rmat_wrt_hole - tip_rmat_wrt_hole.T))

  size = 5
  lb = np.zeros(size)
  ub = np.zeros(size)

  lb[2] = z_lower_bound
  ub[2] = 0.038 # Make sure this matches insertion_constraint

  return ca.vertcat(tip_pos_wrt_hole, rot_err[:2]), lb, ub
