import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
from casadi_rp_conversion import *
import numpy as np
from inf_def import inf

def export_insertion_constraint(cpin_model, x_acados):
  cpin_data = cpin_model.createData()

  tip_fid = cpin_model.getFrameId('ee_tip')
  goal_fid = cpin_model.getFrameId('hole_end') #Try to hit the back of the hole, past the goal

  q = q_to_pin(x_acados[:cpin_model.nv], cpin_model)

  cpin.forwardKinematics(cpin_model, cpin_data, q)
  cpin.updateFramePlacement(cpin_model, cpin_data, tip_fid)
  cpin.updateFramePlacement(cpin_model, cpin_data, goal_fid)

  tip_pos_wrt_goal = cpin_data.oMf[goal_fid].rotation.T@(cpin_data.oMf[tip_fid].translation - cpin_data.oMf[goal_fid].translation)
  tip_rmat_wrt_goal = cpin_data.oMf[goal_fid].rotation.T@cpin_data.oMf[tip_fid].rotation
  rot_err = cpin.unSkew(0.5*(tip_rmat_wrt_goal - tip_rmat_wrt_goal.T))

  # Tell trajopt to hit hole_end exactly. This will make us go past the goal to ensure we actually hit the goal
  size = 6
  lb = np.zeros(size)
  ub = np.zeros(size)

  # Adding an offset in the z direction because with state estimation noise we sometimes don't hit the back of the wall
  # This should be done in a more intelligent way in the future
  # We also add a range so that this constraint isn't as sensitive to the choice of phase length.
  lb[2] = lb[2] + 0.018
  ub[2] = ub[2] + 0.038
  
  return ca.vertcat(tip_pos_wrt_goal, rot_err), lb, ub
